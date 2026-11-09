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
    warmup_steps: int | None
    window_size: float
    flex_window: float | None
    fallback_steps: int


_ACTIVE: contextvars.ContextVar[SpectrumSettings | None] = contextvars.ContextVar(
    'dgenerate_spectrum',
    default=None,
)

_BLOCK_LISTS = (
    'transformer_blocks', 'single_transformer_blocks', 'visual_transformer_blocks', 'blocks', 'layers')
_DENOISER_ATTRS = ('transformer', 'transformer_2', 'unet')
_RESIDUAL_ARGS = (
    'down_block_additional_residuals',
    'mid_block_additional_residual',
    'down_intrablock_additional_residuals',
)


def active_settings() -> SpectrumSettings | None:
    return _ACTIVE.get()


def schedule_for(num_steps: int, cfg_multiplier: float | None) -> tuple[int, float]:
    """Warmup and flex for a resolved step count.

    Fifty steps or more use the paper's flex 0.75 and warmup 5, unless
    guidance multiplies the conditional gap. Shorter schedules hold the skip
    interval fixed. Four steps or fewer stay on real passes. Classifier-free
    guidance above 1 also holds the interval fixed, because it multiplies the
    forecast error. ``cfg_multiplier`` is ``None`` when guidance is an
    embedding inside one forward.
    """
    steps = max(int(num_steps), 1)
    if steps <= 4:
        warmup, flex = steps, 0.0
    elif steps <= 8:
        warmup, flex = 2, 0.0
    elif steps < 50:
        warmup, flex = 4, 0.0
    else:
        warmup, flex = 5, 0.75
    if cfg_multiplier is not None and cfg_multiplier > 1:
        flex = 0.0
    return warmup, flex


def resolve_settings(
        settings: SpectrumSettings,
        num_steps: int,
        cfg_multiplier: float | None = None) -> SpectrumSettings:
    """Fill warmup and flex that were left unset until the step count existed."""
    if settings.warmup_steps is not None and settings.flex_window is not None:
        return settings
    warmup, flex = schedule_for(num_steps, cfg_multiplier)
    return settings._replace(
        warmup_steps=settings.warmup_steps if settings.warmup_steps is not None else warmup,
        flex_window=settings.flex_window if settings.flex_window is not None else flex)


def schedule_counts(num_steps: int, settings: SpectrumSettings) -> tuple[int, int, int]:
    """Return full passes, predicted steps, and the closing run of predicted steps.

    Matches :class:`_StepState`: a forecast needs two real passes, and the skip
    interval grows by the flex window after each full pass.
    """
    steps = max(int(num_steps), 1)
    samples = 0
    consecutive = 0
    window = float(settings.window_size)
    kinds: list[bool] = []
    for index in range(steps):
        if index < settings.warmup_steps or samples < 2:
            actual = True
        else:
            width = max(1, math.floor(window))
            actual = (consecutive + 1) % width == 0
            if actual:
                window = round(window + settings.flex_window, 3)
        kinds.append(actual)
        if actual:
            samples += 1
            consecutive = 0
        else:
            consecutive += 1
    trailing = 0
    for actual in reversed(kinds):
        if actual:
            break
        trailing += 1
    full = samples
    return full, steps - full, trailing


_WARNED: set[tuple] = set()


def _guidance_embeds(pipeline) -> bool:
    transformer = getattr(pipeline, 'transformer', None)
    config = getattr(transformer, 'config', None)
    if config is None:
        return False
    if hasattr(config, 'get'):
        return bool(config.get('guidance_embeds', False))
    return bool(getattr(config, 'guidance_embeds', False))


def _cfg_scale(pipeline, guidance_scale) -> float | None:
    """Scale that multiplies the conditional/unconditional gap, when that gap exists.

    Flux-style guidance is an embedding inside one forward, so it does not
    multiply a pair of forecasts.
    """
    if guidance_scale is None or _guidance_embeds(pipeline):
        return None
    try:
        scale = float(guidance_scale)
    except (TypeError, ValueError):
        return None
    if scale > 1:
        return scale
    return None


def cfg_multiplier(pipeline, call_kwargs: dict | None = None, guidance_scale=None) -> float | None:
    """Classifier-free guidance scale that multiplies a pair of forecasts.

    Flux-style guidance embeddings return ``None``. Qwen reports the multiplier
    as ``true_cfg_scale``.
    """
    if _guidance_embeds(pipeline):
        return None
    scale = guidance_scale
    if call_kwargs:
        if call_kwargs.get('true_cfg_scale') is not None:
            scale = call_kwargs.get('true_cfg_scale')
        elif scale is None:
            scale = call_kwargs.get('guidance_scale')
    if scale is None:
        guider = getattr(pipeline, 'guider', None)
        scale = getattr(guider, 'guidance_scale', None)
    try:
        value = float(scale)
    except (TypeError, ValueError):
        return None
    if value > 1:
        return value
    return None


def warn_schedule(pipeline, num_steps: int, settings: SpectrumSettings, guidance_scale=None,
                  cfg: float | None = None) -> None:
    """Warn when this schedule guesses several steps in a row at the end.

    Guidance still selects the flex window, and that choice is quiet. A normal
    classifier-free scale already holds the interval fixed.
    """
    if cfg is None:
        cfg = _cfg_scale(pipeline, guidance_scale)
    settings = resolve_settings(settings, num_steps, cfg)
    full, skipped, trailing = schedule_counts(num_steps, settings)
    if trailing < 4:
        return
    key = (
        int(num_steps),
        int(settings.warmup_steps),
        float(settings.window_size),
        float(settings.flex_window),
    )
    if key in _WARNED:
        return
    _WARNED.add(key)

    fixes = [f'--spectrum-flex-windows 0 keeps the skip interval at {settings.window_size:g}']
    paper_flex = any(math.isclose(settings.flex_window, paper, abs_tol=1e-6) for paper in (0.75, 3.0))
    if num_steps < 50 and paper_flex:
        fixes.append(
            '--inference-steps 50 with --spectrum-warmup-steps 5 is the schedule '
            'this flex was measured on')
    elif settings.warmup_steps < 5:
        fixes.append('--spectrum-warmup-steps 5 delays the point where the window starts growing')
    _messages.warning(
        f'Spectrum on {num_steps} steps (flex {settings.flex_window:g}, '
        f'warmup {settings.warmup_steps}) runs {full} full passes and guesses {skipped}. '
        f'The last {trailing} steps are guessed in a row, and that is where detail is set. '
        + ' '.join(fixes) + '.')


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
        self.force_blocks = False
        # Audio half of a fused (video, audio) block, forecast beside the video feature.
        self._companion_preds: dict[int, torch.Tensor | None] = {}
        self._companion_forecasters: dict[int, _forecaster.SpectrumPredictor] = {}
        self._companion_shapes: dict[int, torch.Size] = {}

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
        self._companion_preds.clear()

    def _reset_sample(self) -> None:
        self.cnt = 0
        self.consecutive = 0
        self.curr_ws = float(self.settings.window_size)
        for predictor in self._forecasters.values():
            predictor.clear()
        self._forecasters.clear()
        self._shapes.clear()
        self._preds.clear()
        self._companion_forecasters.clear()
        self._companion_shapes.clear()
        self._companion_preds.clear()

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
        self._companion_preds.clear()
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

    def _companion_predictor(self, stream: int) -> _forecaster.SpectrumPredictor:
        found = self._companion_forecasters.get(stream)
        if found is None:
            chebyshev = _forecaster.ChebyshevForecaster(
                M=self.settings.order,
                K=max(100, self.settings.order + 2),
                lam=self.settings.lam,
            )
            found = _forecaster.SpectrumPredictor(chebyshev, w=self.settings.weight)
            self._companion_forecasters[stream] = found
        return found

    def cache_companion(self, feature: torch.Tensor) -> None:
        """Record the cache-point audio stream so a skipped step can forecast it."""
        feature = feature.detach()
        stream = self.stream
        previous = self._companion_shapes.get(stream)
        if previous is not None and previous != feature.shape:
            self._companion_forecasters.pop(stream, None)
        self._companion_shapes[stream] = feature.shape
        self._companion_predictor(stream).update(self._time(), feature.reshape(-1))

    def predict_companion(self) -> torch.Tensor | None:
        stream = self.stream
        if stream in self._companion_preds:
            return self._companion_preds[stream]
        predictor = self._companion_forecasters.get(stream)
        shape = self._companion_shapes.get(stream)
        if predictor is None or shape is None or not predictor.ready():
            self._companion_preds[stream] = None
            return None
        value = predictor.predict(self._time()).reshape(shape)
        self._companion_preds[stream] = value
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
        # 0: the block returns (image, text), as Hunyuan and Motif do.
        # 1: the block returns (text, image), as Flux, SD3, and Qwen-Image do.
        self.image_slot: int | None = None
        # Kandinsky 6 fused blocks return (video, audio). Audio is forecast
        # from the cache-point block, the same way the video feature is.
        self.companion_audio = False
        self.companion_absent = False
        self.is_cache_point = is_cache_point
        state.patches.append(self)

        def wrapper(*args, **kwargs):
            hidden, encoder = _hidden_and_encoder(self.original, args, kwargs)
            if state.actual_forward or self.returns_pair is None or hidden is None:
                out = self.original(*args, **kwargs)
                self._remember(out, hidden, encoder)
                if self.is_cache_point and hidden is not None:
                    feature = _feature_from_output(out, hidden)
                    if feature is not None:
                        state.cache(feature)
                    if (self.companion_audio and isinstance(out, tuple)
                            and len(out) > 1 and torch.is_tensor(out[1])):
                        state.cache_companion(out[1])
                return out

            predicted = state.predict()
            image = hidden
            if predicted is not None and predicted.shape == hidden.shape:
                image = predicted.to(device=hidden.device, dtype=hidden.dtype)
            if self.companion_audio:
                audio = state.predict_companion()
                if audio is None:
                    return self.original(*args, **kwargs)
                return image, audio.to(device=hidden.device, dtype=audio.dtype)
            if self.companion_absent:
                return image, None
            if self.returns_pair:
                return _pair_with_image(image, encoder, self.image_slot)
            return image

        module.forward = wrapper

    def _remember(self, out, hidden: torch.Tensor | None, encoder=None) -> None:
        self.returns_pair = isinstance(out, tuple)
        if not self.returns_pair or hidden is None:
            return
        for index, item in enumerate(out):
            if torch.is_tensor(item) and item.shape == hidden.shape:
                self.image_slot = index
                break
        if (encoder is None and self.image_slot == 0 and isinstance(out, tuple)
                and len(out) > 1):
            audio = out[1]
            if audio is None:
                self.companion_absent = True
            elif torch.is_tensor(audio) and audio.shape != hidden.shape:
                self.companion_audio = True

    def restore(self) -> None:
        self.module.forward = self.original


def _pair_with_image(image, encoder, image_slot: int | None):
    """Put the image stream back in the slot this block uses.

    Slot 0 is Hunyuan and Motif, which unpack ``(image, text)``. Anything else
    is the Flux order, ``(text, image)``.
    """
    if image_slot == 0:
        return image, encoder
    return encoder, image


class _FaceAdapterPatch:
    """A predicted step already includes the face residual in the forecast."""

    def __init__(self, module: nn.Module, state: _StepState):
        self.module = module
        self.original = module.forward

        def wrapper(*args, **kwargs):
            if state.actual_forward:
                return self.original(*args, **kwargs)
            hidden = args[0] if args else kwargs.get('hidden_states')
            if not torch.is_tensor(hidden):
                return self.original(*args, **kwargs)
            return torch.zeros_like(hidden)

        module.forward = wrapper

    def restore(self) -> None:
        self.module.forward = self.original


def _face_adapters(denoiser: nn.Module) -> list[nn.Module]:
    listing = getattr(denoiser, 'face_adapter', None)
    if isinstance(listing, (nn.ModuleList, list, tuple)):
        return [module for module in listing if isinstance(module, nn.Module)]
    if isinstance(listing, nn.Module):
        return [listing]
    return []


def _extend_blocks(found: list, seen: set, listing) -> None:
    if not isinstance(listing, (nn.ModuleList, list, tuple)) or len(listing) == 0:
        return
    for module in listing:
        if not isinstance(module, nn.Module) or id(module) in seen:
            continue
        seen.add(id(module))
        found.append(module)


def _block_modules(denoiser: nn.Module) -> list[nn.Module]:
    found = []
    seen = set()
    # FLUX.3 per-mode blocks run before the joint single blocks. The last
    # single block stays the cache point because it is appended below.
    _extend_blocks(found, seen, getattr(denoiser, 'txt_mode_blocks', None))
    mode_blocks = getattr(denoiser, 'mode_blocks', None)
    if isinstance(mode_blocks, nn.ModuleDict):
        for name in ('image', 'image_cond'):
            _extend_blocks(found, seen, mode_blocks.get(name) if hasattr(mode_blocks, 'get') else None)
    for name in _BLOCK_LISTS:
        listing = getattr(denoiser, name, None)
        _extend_blocks(found, seen, listing)
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
        self.faces = [_FaceAdapterPatch(module, state) for module in _face_adapters(denoiser)]

        def wrapper(*args, **kwargs):
            state.begin(_bind_timestep(self.original, args, kwargs))
            return self.original(*args, **kwargs)

        denoiser.forward = wrapper

    def restore(self) -> None:
        self.denoiser.forward = self.original
        for block in self.blocks:
            block.restore()
        for face in self.faces:
            face.restore()


def _bound_argument(original, args, kwargs, name):
    try:
        signature = inspect.signature(original)
        bound = signature.bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return kwargs.get(name)
    return bound.arguments.get(name)


def _down_residual_count(block: nn.Module) -> int:
    count = len(getattr(block, 'resnets', ()) or ())
    downsamplers = getattr(block, 'downsamplers', None)
    if downsamplers:
        count += len(downsamplers)
    return max(count, 1)


class _UNetBlockPatch:
    """Skip an SDXL UNet block, or cache the last up block."""

    def __init__(self, module: nn.Module, state: _StepState, kind: str, is_cache_point: bool):
        self.module = module
        self.original = module.forward
        self.kind = kind

        def wrapper(*args, **kwargs):
            hidden, _encoder = _hidden_and_encoder(self.original, args, kwargs)
            if state.actual_forward or state.force_blocks or hidden is None:
                out = self.original(*args, **kwargs)
                if is_cache_point and state.actual_forward and hidden is not None:
                    feature = _feature_from_output(out, hidden)
                    if feature is not None:
                        state.cache(feature)
                return out
            if is_cache_point:
                predicted = state.predict()
                if predicted is not None and predicted.shape == hidden.shape:
                    return predicted.to(device=hidden.device, dtype=hidden.dtype)
                return self.original(*args, **kwargs)
            if kind == 'down':
                return hidden, (hidden,) * _down_residual_count(module)
            return hidden

        module.forward = wrapper

    def restore(self) -> None:
        self.module.forward = self.original


class _UNetPatch:
    """Forecast the tensor after the up blocks. ``conv_out`` still runs."""

    def __init__(self, unet: nn.Module, state: _StepState):
        self.denoiser = unet
        self.state = state
        self.original = unet.forward
        down_blocks = list(getattr(unet, 'down_blocks', ()) or ())
        up_blocks = list(getattr(unet, 'up_blocks', ()) or ())
        if not down_blocks or not up_blocks:
            raise SpectrumUnsupported(
                f'{unet.__class__.__name__} has no UNet blocks Spectrum can forecast.')
        self.blocks = [
            _UNetBlockPatch(module, state, 'down', is_cache_point=False)
            for module in down_blocks
        ]
        mid = getattr(unet, 'mid_block', None)
        if isinstance(mid, nn.Module):
            self.blocks.append(_UNetBlockPatch(mid, state, 'mid', is_cache_point=False))
        for index, module in enumerate(up_blocks):
            self.blocks.append(_UNetBlockPatch(
                module, state, 'up', is_cache_point=(index == len(up_blocks) - 1)))

        def wrapper(*args, **kwargs):
            state.begin(_bind_timestep(self.original, args, kwargs))
            state.force_blocks = any(
                _bound_argument(self.original, args, kwargs, name) is not None
                for name in _RESIDUAL_ARGS
            )
            try:
                return self.original(*args, **kwargs)
            finally:
                state.force_blocks = False

        unet.forward = wrapper

    def restore(self) -> None:
        self.denoiser.forward = self.original
        for block in self.blocks:
            block.restore()


def _patch_denoiser(denoiser: nn.Module, state: _StepState):
    down_blocks = getattr(denoiser, 'down_blocks', None)
    up_blocks = getattr(denoiser, 'up_blocks', None)
    if down_blocks and up_blocks:
        return _UNetPatch(denoiser, state)
    return _DenoiserPatch(denoiser, state)


def _denoisers(pipeline) -> list[nn.Module]:
    found = []
    for name in _DENOISER_ATTRS:
        module = getattr(pipeline, name, None)
        if isinstance(module, nn.Module):
            found.append(module)
    return found


@contextlib.contextmanager
def apply(pipeline, num_inference_steps: int, settings: SpectrumSettings,
          cfg_multiplier: float | None = None):
    """Forecast ``pipeline``'s denoiser blocks for one sampling call."""
    settings = resolve_settings(settings, num_inference_steps, cfg_multiplier)
    denoisers = _denoisers(pipeline)
    if not denoisers:
        raise SpectrumUnsupported(
            f'{pipeline.__class__.__name__} has no denoiser for Spectrum to patch.')
    patches = []
    try:
        for denoiser in denoisers:
            state = _StepState(settings, num_inference_steps)
            patches.append(_patch_denoiser(denoiser, state))
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
