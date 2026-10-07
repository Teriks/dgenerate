"""
Apply Spectrum to a diffusers pipeline for one call.

Block forwards are replaced in place and restored when the context exits.
A full step runs the original block. A predicted step writes the forecast
into that block's image stream and leaves the surrounding embeddings,
normalization, and output projection on the real timestep.
"""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import math
import typing

import torch
import torch.nn as nn

import dgenerate.messages as _messages
import dgenerate.extras.spectrum.forecaster as _forecaster


class SpectrumUnsupported(Exception):
    """The loaded denoiser has no transformer blocks Spectrum can forecast."""


class SpectrumSettings(typing.NamedTuple):
    weight: float
    order: int
    lam: float
    warmup_steps: int
    window_size: float
    flex_window: float
    fallback_steps: int


_ACTIVE: contextvars.ContextVar[SpectrumSettings | None] = contextvars.ContextVar(
    'dgenerate_spectrum',
    default=None,
)

_BLOCK_LISTS = ('transformer_blocks', 'single_transformer_blocks', 'blocks', 'layers')
_DENOISER_ATTRS = ('transformer', 'transformer_2')


def active_settings() -> SpectrumSettings | None:
    return _ACTIVE.get()


def push(settings: SpectrumSettings | None):
    """Remember ``settings`` until :func:`pop`. ``None`` leaves sampling unchanged."""
    if settings is None:
        return None
    return _ACTIVE.set(settings)


def pop(token) -> None:
    if token is not None:
        _ACTIVE.reset(token)


@contextlib.contextmanager
def activate(settings: SpectrumSettings | None):
    """Enable Spectrum for every pipeline call inside this block."""
    if settings is None:
        yield
        return
    token = _ACTIVE.set(settings)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


class _StepState:
    """Skip schedule and one forecaster per guidance stream."""

    def __init__(self, settings: SpectrumSettings, num_steps: int):
        self.settings = settings
        self.num_steps = max(int(num_steps), 1)
        self.cnt = 0
        self.consecutive = 0
        self.curr_ws = float(settings.window_size)
        self.actual_forward = True
        self.full = 0
        self.skipped = 0
        self.stream = 0
        self._open = False
        self._open_key = None
        self._anon = 0
        self._preds: dict[int, torch.Tensor | None] = {}
        self._forecasters: dict[int, _forecaster.SpectrumPredictor] = {}
        self._shapes: dict[int, torch.Size] = {}
        self.patches: list[_BlockPatch] = []

    def _time(self) -> float:
        return self.cnt / max(self.num_steps - 1, 1)

    def _ready(self) -> bool:
        if not self._forecasters:
            return False
        return all(item.ready() for item in self._forecasters.values())

    def _decide(self) -> None:
        if self.cnt < self.settings.warmup_steps or not self._ready():
            self.actual_forward = True
            return
        window = max(1, math.floor(self.curr_ws))
        self.actual_forward = (self.consecutive + 1) % window == 0
        if self.actual_forward:
            self.curr_ws = round(self.curr_ws + self.settings.flex_window, 3)

    def _advance(self) -> None:
        if self.actual_forward:
            self.full += 1
            self.consecutive = 0
        else:
            self.skipped += 1
            self.consecutive += 1
        self.cnt += 1
        self._open = False
        self._preds.clear()

    def _reset_sample(self) -> None:
        self.cnt = 0
        self.consecutive = 0
        self.curr_ws = float(self.settings.window_size)
        for predictor in self._forecasters.values():
            predictor.clear()
        self._forecasters.clear()
        self._shapes.clear()
        self._preds.clear()

    def begin(self, timestep) -> None:
        key = _timestep_key(timestep)
        if key is None:
            key = ('call', self._anon)
            self._anon += 1
        if self._open and key == self._open_key:
            self.stream += 1
            return
        if self._open:
            self._advance()
        if self.cnt >= self.num_steps:
            self._reset_sample()
        self._open = True
        self._open_key = key
        self.stream = 0
        self._preds.clear()
        self._decide()
        if any(patch.returns_pair is None for patch in self.patches):
            self.actual_forward = True

    def close(self) -> None:
        if self._open:
            self._advance()

    def _predictor(self, stream: int) -> _forecaster.SpectrumPredictor:
        found = self._forecasters.get(stream)
        if found is None:
            chebyshev = _forecaster.ChebyshevForecaster(
                M=self.settings.order,
                K=max(100, self.settings.order + 2),
                lam=self.settings.lam,
            )
            found = _forecaster.SpectrumPredictor(chebyshev, w=self.settings.weight)
            self._forecasters[stream] = found
        return found

    def cache(self, feature: torch.Tensor) -> None:
        feature = feature.detach()
        stream = self.stream
        previous = self._shapes.get(stream)
        if previous is not None and previous != feature.shape:
            self._forecasters.pop(stream, None)
        self._shapes[stream] = feature.shape
        self._predictor(stream).update(self._time(), feature.reshape(-1))

    def predict(self) -> torch.Tensor | None:
        stream = self.stream
        if stream in self._preds:
            return self._preds[stream]
        predictor = self._forecasters.get(stream)
        shape = self._shapes.get(stream)
        if predictor is None or shape is None or not predictor.ready():
            self._preds[stream] = None
            return None
        value = predictor.predict(self._time()).reshape(shape)
        self._preds[stream] = value
        return value


def _timestep_key(timestep):
    if timestep is None:
        return None
    if isinstance(timestep, (list, tuple)):
        if not timestep:
            return None
        return _timestep_key(timestep[0])
    if torch.is_tensor(timestep):
        if timestep.numel() == 0:
            return None
        value = float(timestep.detach().flatten()[0].float().item())
    else:
        try:
            value = float(timestep)
        except (TypeError, ValueError):
            return None
    return round(value, 5)


def _hidden_and_encoder(original, args, kwargs):
    hidden = None
    encoder = None
    try:
        signature = inspect.signature(original)
        bound = signature.bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        signature = None
        bound = None
    if signature is not None and bound is not None:
        names = [name for name in signature.parameters if name != 'self']
        if names and names[0] in bound.arguments and torch.is_tensor(bound.arguments[names[0]]):
            hidden = bound.arguments[names[0]]
        if 'encoder_hidden_states' in bound.arguments:
            encoder = bound.arguments['encoder_hidden_states']
    if hidden is None:
        for value in list(args) + list(kwargs.values()):
            if torch.is_tensor(value):
                hidden = value
                break
    return hidden, encoder


def _feature_from_output(out, hidden: torch.Tensor) -> torch.Tensor | None:
    if torch.is_tensor(out):
        return out
    if isinstance(out, tuple):
        for item in out:
            if torch.is_tensor(item) and item.shape == hidden.shape:
                return item
        for item in reversed(out):
            if torch.is_tensor(item):
                return item
    return None


class _BlockPatch:
    def __init__(self, module: nn.Module, state: _StepState, is_cache_point: bool):
        self.module = module
        self.original = module.forward
        self.returns_pair: bool | None = None
        self.is_cache_point = is_cache_point
        state.patches.append(self)

        def wrapper(*args, **kwargs):
            hidden, encoder = _hidden_and_encoder(self.original, args, kwargs)
            if state.actual_forward or self.returns_pair is None or hidden is None:
                out = self.original(*args, **kwargs)
                self.returns_pair = isinstance(out, tuple)
                if self.is_cache_point and hidden is not None:
                    feature = _feature_from_output(out, hidden)
                    if feature is not None:
                        state.cache(feature)
                return out

            predicted = state.predict()
            if self.returns_pair:
                if predicted is None or predicted.shape != hidden.shape:
                    return encoder, hidden
                return encoder, predicted.to(device=hidden.device, dtype=hidden.dtype)
            if predicted is None or predicted.shape != hidden.shape:
                return hidden
            return predicted.to(device=hidden.device, dtype=hidden.dtype)

        module.forward = wrapper

    def restore(self) -> None:
        self.module.forward = self.original


def _block_modules(denoiser: nn.Module) -> list[nn.Module]:
    found = []
    seen = set()
    for name in _BLOCK_LISTS:
        listing = getattr(denoiser, name, None)
        if not isinstance(listing, (nn.ModuleList, list, tuple)) or len(listing) == 0:
            continue
        for module in listing:
            if not isinstance(module, nn.Module) or id(module) in seen:
                continue
            seen.add(id(module))
            found.append(module)
    return found


def _bind_timestep(original, args, kwargs):
    try:
        signature = inspect.signature(original)
        bound = signature.bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return kwargs.get('timestep', kwargs.get('timesteps'))
    for name in ('timestep', 'timesteps'):
        if name in bound.arguments:
            return bound.arguments[name]
    return None


class _DenoiserPatch:
    def __init__(self, denoiser: nn.Module, state: _StepState):
        self.denoiser = denoiser
        self.state = state
        self.original = denoiser.forward
        blocks = _block_modules(denoiser)
        if not blocks:
            raise SpectrumUnsupported(
                f'{denoiser.__class__.__name__} has no transformer blocks Spectrum can forecast.')
        self.blocks = [
            _BlockPatch(module, state, is_cache_point=(index == len(blocks) - 1))
            for index, module in enumerate(blocks)
        ]

        def wrapper(*args, **kwargs):
            state.begin(_bind_timestep(self.original, args, kwargs))
            return self.original(*args, **kwargs)

        denoiser.forward = wrapper

    def restore(self) -> None:
        self.denoiser.forward = self.original
        for block in self.blocks:
            block.restore()


def _denoisers(pipeline) -> list[nn.Module]:
    found = []
    for name in _DENOISER_ATTRS:
        module = getattr(pipeline, name, None)
        if isinstance(module, nn.Module):
            found.append(module)
    return found


@contextlib.contextmanager
def apply(pipeline, num_inference_steps: int, settings: SpectrumSettings):
    """Forecast ``pipeline``'s denoiser blocks for one sampling call."""
    denoisers = _denoisers(pipeline)
    if not denoisers:
        raise SpectrumUnsupported(
            f'{pipeline.__class__.__name__} has no transformer for Spectrum to patch.')
    patches = []
    try:
        for denoiser in denoisers:
            state = _StepState(settings, num_inference_steps)
            patches.append(_DenoiserPatch(denoiser, state))
        yield
    finally:
        for patch in patches:
            patch.state.close()
            full = patch.state.full
            skipped = patch.state.skipped
            name = patch.denoiser.__class__.__name__
            if full or skipped:
                _messages.log(
                    f'Spectrum on {name}: {full} full passes, {skipped} predicted '
                    f'({full + skipped} denoiser calls).')
            patch.restore()
            patch.state._reset_sample()
