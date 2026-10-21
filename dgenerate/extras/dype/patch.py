# Copyright (c) 2025 Guy Yariv
#
# This file adapts the RoPE methods from https://github.com/guyyariv/DyPE.
# See LICENSE for the upstream MIT license.

"""Temporary RoPE adapters for supported Diffusers transformer families."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import math

import diffusers
import torch

from .rope import AXIS_SCALINGS, MSCALE_MODES, dype_rope_1d
from .scheduler import resolution_shift_context


class DypeError(ValueError):
    """The loaded pipeline does not expose a RoPE interface supported by DyPE."""


_CURRENT_TIMESTEP = contextvars.ContextVar('dgenerate_dype_timestep', default=1.0)
_IMAGE_BASE_RESOLUTION = 1024
_QWEN_BASE_RESOLUTION = 1328
_WAN_BASE_RESOLUTION = 832
_IMAGE_TOKEN_STRIDE = 16
_WAN_TOKEN_STRIDE = 16


def _attribute_patch(obj, name: str, value):
    had_instance_value = name in getattr(obj, '__dict__', {})
    old_instance_value = getattr(obj, '__dict__', {}).get(name)
    setattr(obj, name, value)

    def restore():
        if had_instance_value:
            setattr(obj, name, old_instance_value)
        else:
            delattr(obj, name)

    return restore


def _get_timestep(original_forward, args, kwargs) -> float:
    timestep = kwargs.get('timestep')
    if timestep is None:
        try:
            bound = inspect.signature(original_forward).bind_partial(*args, **kwargs)
            timestep = bound.arguments.get('timestep')
        except (TypeError, ValueError):
            timestep = None
    if timestep is None:
        return 1.0
    if isinstance(timestep, torch.Tensor):
        value = float(timestep.detach().float().mean().item())
    else:
        value = float(timestep)
    if not math.isfinite(value):
        return 1.0
    return value


def _normalized_timestep(model_family: str, value: float) -> float:
    if abs(value) > 1.0:
        value /= 1000.0
    value = min(1.0, max(0.0, value))
    if model_family == 'zimage':
        value = 1.0 - value
    return value


def _patch_forward_timestep(transformer, model_family: str):
    original_forward = transformer.forward

    @functools.wraps(original_forward)
    def wrapped(*args, **kwargs):
        value = _normalized_timestep(model_family, _get_timestep(original_forward, args, kwargs))
        token = _CURRENT_TIMESTEP.set(value)
        try:
            return original_forward(*args, **kwargs)
        finally:
            _CURRENT_TIMESTEP.reset(token)

    return _attribute_patch(transformer, 'forward', wrapped)


def _image_base_patches(base_resolution: int | None, *, default: int = _IMAGE_BASE_RESOLUTION) -> float:
    return float(base_resolution or default) / _IMAGE_TOKEN_STRIDE


# Height and width position axes in Flux, Flux.2, and Z-Image position IDs.
_SPATIAL_ID_AXES = (1, 2)


def _shared_scale(axis_scaling: str, lengths, base_length: float) -> float | None:
    """Image-wide extrapolation factor for ``shared`` axis scaling, else ``None``."""
    if axis_scaling != 'shared':
        return None
    return max(1.0, max(float(length) / base_length for length in lengths))


def _patch_flux_pos_embed(transformer,
                          base_resolution: int | None,
                          method: str,
                          mscale: str = 'constant',
                          axis_scaling: str = 'shared',
                          embedder_attribute: str = 'pos_embed'):
    pos_embed = getattr(transformer, embedder_attribute, None)
    if pos_embed is None or not hasattr(pos_embed, 'axes_dim') or not hasattr(pos_embed, 'theta'):
        raise DypeError(
            f'The loaded transformer has no compatible {embedder_attribute} positional embedder.')

    axes_dim = tuple(pos_embed.axes_dim)
    theta = float(pos_embed.theta)
    base_patches = _image_base_patches(base_resolution)
    lengths_by_shape = {}

    def forward(ids: torch.Tensor):
        if ids.shape[-1] < len(axes_dim):
            raise DypeError('Flux position IDs have fewer axes than the positional embedder.')
        cos_out = []
        sin_out = []
        pos = ids.float()
        shape_key = (tuple(ids.shape), ids.device.type, ids.device.index)
        if shape_key not in lengths_by_shape:
            lengths_by_shape[shape_key] = tuple(
                max(1, int(pos[..., axis].max().item()) + 1) if axis > 0 else 1
                for axis in range(len(axes_dim)))
        lengths = lengths_by_shape[shape_key]
        shared_scale = _shared_scale(
            axis_scaling, (lengths[axis] for axis in _SPATIAL_ID_AXES), base_patches)
        for axis, dim in enumerate(axes_dim):
            axis_positions = pos[..., axis].reshape(-1)
            cos, sin = dype_rope_1d(
                axis_positions,
                dim,
                theta=theta,
                base_length=base_patches,
                current_length=lengths[axis],
                timestep=_CURRENT_TIMESTEP.get(),
                method=method,
                mscale=mscale,
                shared_scale=shared_scale if axis in _SPATIAL_ID_AXES else None,
            )
            output_shape = (*pos.shape[:-1], dim)
            cos_out.append(cos.reshape(output_shape))
            sin_out.append(sin.reshape(output_shape))
        return torch.cat(cos_out, dim=-1).to(ids.device), torch.cat(sin_out, dim=-1).to(ids.device)

    return _attribute_patch(pos_embed, 'forward', forward)


def _patch_zimage_rope(transformer,
                       base_resolution: int | None,
                       method: str,
                       mscale: str = 'constant',
                       axis_scaling: str = 'shared'):
    rope = getattr(transformer, 'rope_embedder', None)
    if rope is None or not all(hasattr(rope, name) for name in ('axes_dims', 'theta')):
        raise DypeError('The loaded Z-Image transformer has no compatible callable RoPE embedder.')

    axes_dims = tuple(rope.axes_dims)
    theta = float(rope.theta)
    base_patches = _image_base_patches(base_resolution)
    lengths_by_shape = {}

    def call(ids: torch.Tensor):
        if ids.ndim != 2 or ids.shape[-1] < len(axes_dims):
            raise DypeError('Z-Image position IDs do not match the configured RoPE axes.')
        shape_key = (tuple(ids.shape), ids.device.type, ids.device.index)
        if shape_key not in lengths_by_shape:
            lengths_by_shape[shape_key] = tuple(
                max(1, int(ids[:, axis].max().item()) + 1) if axis > 0 else 1
                for axis in range(len(axes_dims)))
        lengths = lengths_by_shape[shape_key]
        shared_scale = _shared_scale(
            axis_scaling, (lengths[axis] for axis in _SPATIAL_ID_AXES), base_patches)
        result = []
        for axis, dim in enumerate(axes_dims):
            axis_positions = ids[:, axis]
            result.append(dype_rope_1d(
                axis_positions,
                dim,
                theta=theta,
                base_length=base_patches,
                current_length=lengths[axis],
                timestep=_CURRENT_TIMESTEP.get(),
                method=method,
                mscale=mscale,
                shared_scale=shared_scale if axis in _SPATIAL_ID_AXES else None,
                output='complex',
                complex_mscale=True,
            ))
        return torch.cat(result, dim=-1)

    return _attribute_patch(transformer, 'rope_embedder', call)


def _axis_positions(length: int, centered: bool, device: torch.device) -> torch.Tensor:
    if not centered:
        return torch.arange(length, device=device)
    negative_count = length - length // 2
    return torch.cat((torch.arange(-negative_count, 0, device=device),
                      torch.arange(length // 2, device=device)))


def _qwen_video_frequencies(embedder,
                            frame: int,
                            height: int,
                            width: int,
                            *,
                            idx: int = 0,
                            device: torch.device | None = None,
                            condition: bool = False,
                            base_resolution: int | None = None,
                            method: str = 'yarn',
                            mscale: str = 'constant',
                            axis_scaling: str = 'shared'):
    if device is None:
        device = embedder.pos_freqs.device
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    if hasattr(embedder, '_get_device_freqs'):
        pos_freqs, neg_freqs = embedder._get_device_freqs(device)
    else:
        pos_freqs, neg_freqs = embedder.pos_freqs.to(device), embedder.neg_freqs.to(device)

    dims = tuple(embedder.axes_dim)
    pos_axes = pos_freqs.split([dim // 2 for dim in dims], dim=1)
    neg_axes = neg_freqs.split([dim // 2 for dim in dims], dim=1)
    frame_freqs = (neg_axes[0][-1:] if condition else pos_axes[0][idx:idx + frame])
    frame_freqs = frame_freqs.view(frame, 1, 1, -1).expand(frame, height, width, -1)

    output = 'angles' if device.type == 'neuron' else 'complex'
    base_length = _image_base_patches(base_resolution, default=_QWEN_BASE_RESOLUTION)
    timestep = _CURRENT_TIMESTEP.get()
    shared_scale = _shared_scale(axis_scaling, (height, width), base_length)
    height_positions = _axis_positions(height, bool(embedder.scale_rope), device)
    width_positions = _axis_positions(width, bool(embedder.scale_rope), device)
    height_freqs = dype_rope_1d(
        height_positions,
        dims[1],
        theta=float(embedder.theta),
        base_length=base_length,
        current_length=height,
        timestep=timestep,
        method=method,
        mscale=mscale,
        shared_scale=shared_scale,
        output=output,
        complex_mscale=True,
        freqs_dtype=torch.float32,
    ).view(1, height, 1, -1).expand(frame, height, width, -1)
    width_freqs = dype_rope_1d(
        width_positions,
        dims[2],
        theta=float(embedder.theta),
        base_length=base_length,
        current_length=width,
        timestep=timestep,
        method=method,
        mscale=mscale,
        shared_scale=shared_scale,
        output=output,
        complex_mscale=True,
        freqs_dtype=torch.float32,
    ).view(1, 1, width, -1).expand(frame, height, width, -1)
    return torch.cat([frame_freqs, height_freqs, width_freqs], dim=-1).reshape(frame * height * width, -1).contiguous()


def _patch_qwen_rope(transformer,
                     base_resolution: int | None,
                     method: str,
                     mscale: str = 'constant',
                     axis_scaling: str = 'shared'):
    embedder = getattr(transformer, 'pos_embed', None)
    if embedder is None or not all(hasattr(embedder, name) for name in ('_compute_video_freqs', 'axes_dim', 'theta')):
        raise DypeError('The loaded Qwen transformer has no compatible shape-based RoPE embedder.')

    def video_freqs(frame, height, width, idx=0, device=None):
        return _qwen_video_frequencies(
            embedder, frame, height, width, idx=idx, device=device, base_resolution=base_resolution,
            method=method, mscale=mscale, axis_scaling=axis_scaling)

    restorers = [_attribute_patch(embedder, '_compute_video_freqs', video_freqs)]
    try:
        if hasattr(embedder, '_compute_condition_freqs'):
            def condition_freqs(frame, height, width, device=None):
                return _qwen_video_frequencies(
                    embedder, frame, height, width, device=device, condition=True,
                    base_resolution=base_resolution, method=method, mscale=mscale,
                    axis_scaling=axis_scaling)

            restorers.append(_attribute_patch(embedder, '_compute_condition_freqs', condition_freqs))
    except Exception:
        for restore_method in reversed(restorers):
            restore_method()
        raise

    def restore():
        for restore_method in reversed(restorers):
            restore_method()

    return restore


def _wan_theta_from_buffer(spatial_cos: torch.Tensor, dim: int) -> float:
    """Recover the RoPE theta baked into a precomputed Wan frequency buffer."""
    if dim < 4 or spatial_cos.shape[0] < 2:
        return 10000.0
    cos_value = float(spatial_cos[1, 2].detach().float().clamp(-1.0, 1.0).item())
    if cos_value >= 1.0:
        return 10000.0
    angle = math.acos(cos_value)
    if angle <= 0.0:
        return 10000.0
    return math.exp(math.log(angle) * (-dim / 2.0))


def _wan_spatial_axis(rope,
                      axis_cos: torch.Tensor,
                      axis_sin: torch.Tensor,
                      length: int,
                      dim: int,
                      *,
                      base_length: float,
                      method: str,
                      mscale: str,
                      device: torch.device):
    """Return one spatial RoPE axis.

    ``base`` keeps the model's precomputed frequencies, including whatever theta
    was used when the buffer was built. Positions past that buffer, and the
    yarn/ntk schedules, are recomputed from the recovered theta.
    """
    if method == 'base' and length <= axis_cos.shape[0]:
        return axis_cos[:length], axis_sin[:length]

    stored_theta = getattr(rope, 'theta', None)
    if isinstance(stored_theta, (int, float)) and stored_theta > 1.0:
        theta = float(stored_theta)
    else:
        theta = _wan_theta_from_buffer(axis_cos, dim)
    positions = torch.arange(length, device=device)
    cos, sin = dype_rope_1d(
        positions,
        dim,
        theta=theta,
        base_length=base_length,
        current_length=length,
        timestep=_CURRENT_TIMESTEP.get(),
        method=method,
        mscale=mscale,
    )
    return cos.to(dtype=axis_cos.dtype), sin.to(dtype=axis_sin.dtype)


def _patch_wan_rope(transformer, base_resolution: int | None, method: str, mscale: str = 'constant'):
    rope = getattr(transformer, 'rope', None)
    needed = ('patch_size', 't_dim', 'h_dim', 'w_dim', 'freqs_cos', 'freqs_sin')
    if rope is None or not all(hasattr(rope, name) for name in needed):
        raise DypeError('The loaded Wan transformer has no compatible 3D rotary embedding module.')

    base_resolution = base_resolution or _WAN_BASE_RESOLUTION

    def forward(hidden_states: torch.Tensor):
        _, _, num_frames, height, width = hidden_states.shape
        patch_t, patch_h, patch_w = rope.patch_size
        patch_frames = num_frames // patch_t
        patch_height = height // patch_h
        patch_width = width // patch_w
        dimensions = [rope.t_dim, rope.h_dim, rope.w_dim]
        freqs_cos = rope.freqs_cos.to(hidden_states.device).split(dimensions, dim=1)
        freqs_sin = rope.freqs_sin.to(hidden_states.device).split(dimensions, dim=1)

        cos_frames = freqs_cos[0][:patch_frames].view(patch_frames, 1, 1, -1).expand(
            patch_frames, patch_height, patch_width, -1)
        sin_frames = freqs_sin[0][:patch_frames].view(patch_frames, 1, 1, -1).expand(
            patch_frames, patch_height, patch_width, -1)

        ref_long_patches = float(base_resolution) / _WAN_TOKEN_STRIDE
        actual_long_patches = max(patch_height, patch_width, 1)
        base_height = ref_long_patches * patch_height / actual_long_patches
        base_width = ref_long_patches * patch_width / actual_long_patches

        cos_height, sin_height = _wan_spatial_axis(
            rope, freqs_cos[1], freqs_sin[1], patch_height, rope.h_dim,
            base_length=base_height, method=method, mscale=mscale, device=hidden_states.device)
        cos_width, sin_width = _wan_spatial_axis(
            rope, freqs_cos[2], freqs_sin[2], patch_width, rope.w_dim,
            base_length=base_width, method=method, mscale=mscale, device=hidden_states.device)
        cos_height = cos_height.view(1, patch_height, 1, -1).expand(
            patch_frames, patch_height, patch_width, -1)
        sin_height = sin_height.view(1, patch_height, 1, -1).expand(
            patch_frames, patch_height, patch_width, -1)
        cos_width = cos_width.view(1, 1, patch_width, -1).expand(
            patch_frames, patch_height, patch_width, -1)
        sin_width = sin_width.view(1, 1, patch_width, -1).expand(
            patch_frames, patch_height, patch_width, -1)

        cos = torch.cat([cos_frames, cos_height, cos_width], dim=-1).reshape(
            1, patch_frames * patch_height * patch_width, 1, -1)
        sin = torch.cat([sin_frames, sin_height, sin_width], dim=-1).reshape(
            1, patch_frames * patch_height * patch_width, 1, -1)
        return cos, sin

    return _attribute_patch(rope, 'forward', forward)


def _patch_factorized_video_rope(transformer,
                                 model_family: str,
                                 base_resolution: int | None,
                                 method: str,
                                 mscale: str = 'constant',
                                 axis_scaling: str = 'shared'):
    """Apply DyPE to the spatial axes of HunyuanVideo15 and Motif RoPE.

    Both models build factorized [time, height, width] cos/sin tables from
    the latent shape. Keep the model's temporal frequencies intact and replace
    only the height and width frequency tables.
    """
    rope = getattr(transformer, 'rope', None)
    needed = ('patch_size', 'patch_size_t', 'rope_dim', 'theta')
    if rope is None or not all(hasattr(rope, name) for name in needed):
        raise DypeError('The loaded video transformer has no compatible factorized 3D RoPE module.')

    default_resolution = 640 if model_family == 'hunyuan-video-15' else _WAN_BASE_RESOLUTION
    base_resolution = base_resolution or default_resolution
    base_patches = float(base_resolution) / _WAN_TOKEN_STRIDE

    def forward(hidden_states: torch.Tensor):
        if method == 'base':
            return original(hidden_states)

        original_cos, original_sin = original(hidden_states)
        _, _, num_frames, height, width = hidden_states.shape
        frames = num_frames // rope.patch_size_t
        patch_height = height // rope.patch_size
        patch_width = width // rope.patch_size
        dims = tuple(int(dim) for dim in rope.rope_dim)
        expected_seq = frames * patch_height * patch_width
        if len(dims) != 3 or original_cos.shape[0] != expected_seq or original_cos.shape[-1] != sum(dims):
            raise DypeError('Video RoPE output does not match its configured [time, height, width] axes.')

        positions = (
            torch.arange(frames, device=hidden_states.device),
            torch.arange(patch_height, device=hidden_states.device),
            torch.arange(patch_width, device=hidden_states.device),
        )
        shared_scale = _shared_scale(
            axis_scaling, (patch_height, patch_width), base_patches)
        cos_axes = [original_cos[:, :dims[0]]]
        sin_axes = [original_sin[:, :dims[0]]]
        for axis, (length, dim) in enumerate(((patch_height, dims[1]), (patch_width, dims[2])), start=1):
            cos, sin = dype_rope_1d(
                positions[axis], dim,
                theta=float(rope.theta),
                base_length=base_patches,
                current_length=length,
                timestep=_CURRENT_TIMESTEP.get(),
                method=method,
                mscale=mscale,
                shared_scale=shared_scale,
            )
            if axis == 1:
                cos = cos.view(1, patch_height, 1, dim).expand(frames, patch_height, patch_width, dim)
                sin = sin.view(1, patch_height, 1, dim).expand(frames, patch_height, patch_width, dim)
            else:
                cos = cos.view(1, 1, patch_width, dim).expand(frames, patch_height, patch_width, dim)
                sin = sin.view(1, 1, patch_width, dim).expand(frames, patch_height, patch_width, dim)
            cos_axes.append(cos.reshape(expected_seq, dim).to(dtype=original_cos.dtype))
            sin_axes.append(sin.reshape(expected_seq, dim).to(dtype=original_sin.dtype))
        return torch.cat(cos_axes, dim=-1), torch.cat(sin_axes, dim=-1)

    original = rope.forward
    return _attribute_patch(rope, 'forward', forward)


def _patch_transformer(transformer,
                       model_family: str,
                       base_resolution: int | None,
                       method: str,
                       mscale: str,
                       axis_scaling: str):
    if model_family in {'flux1', 'flux2'}:
        restore_rope = _patch_flux_pos_embed(transformer, base_resolution, method, mscale, axis_scaling)
    elif model_family == 'krea2':
        # Diffusers' Krea2RotaryPosEmbed is copied from FluxPosEmbed and uses
        # the same position-ID contract, under transformer.rotary_emb.
        restore_rope = _patch_flux_pos_embed(
            transformer, base_resolution, method, mscale, axis_scaling,
            embedder_attribute='rotary_emb')
    elif model_family == 'qwen':
        restore_rope = _patch_qwen_rope(transformer, base_resolution, method, mscale, axis_scaling)
    elif model_family == 'zimage':
        restore_rope = _patch_zimage_rope(transformer, base_resolution, method, mscale, axis_scaling)
    elif model_family == 'wan':
        # Wan already scales both spatial axes by the long side's ratio.
        restore_rope = _patch_wan_rope(transformer, base_resolution, method, mscale)
    elif model_family in {'hunyuan-video-15', 'motif-video'}:
        restore_rope = _patch_factorized_video_rope(
            transformer, model_family, base_resolution, method, mscale, axis_scaling)
    else:
        raise DypeError(f'No DyPE adapter is registered for model family {model_family!r}.')
    try:
        restore_forward = _patch_forward_timestep(transformer, model_family)
    except Exception:
        restore_rope()
        raise

    def restore():
        restore_forward()
        restore_rope()

    return restore


@contextlib.contextmanager
def dype_context(pipeline,
                 model_family: str,
                 base_resolution: int | None = None,
                 method: str | None = None,
                 height: int | None = None,
                 width: int | None = None,
                 mscale: str | None = None,
                 axis_scaling: str | None = None):
    """Apply DyPE adapters for one pipeline call and restore the model afterward."""
    method = 'yarn' if method is None else method
    mscale = 'constant' if mscale is None else mscale
    axis_scaling = 'shared' if axis_scaling is None else axis_scaling
    if method not in {'base', 'ntk', 'yarn'}:
        raise DypeError(f'Unknown DyPE position encoding method {method!r}.')
    if mscale not in MSCALE_MODES:
        raise DypeError(f'Unknown DyPE mscale mode {mscale!r}.')
    if axis_scaling not in AXIS_SCALINGS:
        raise DypeError(f'Unknown DyPE axis scaling {axis_scaling!r}.')
    if base_resolution is not None and base_resolution < 16:
        raise DypeError('The DyPE base resolution must be at least 16 pixels.')
    transformers = []
    names = ('transformer', 'transformer_2') if model_family == 'wan' else ('transformer',)
    for name in names:
        transformer = getattr(pipeline, name, None)
        if transformer is not None and transformer not in transformers:
            transformers.append(transformer)
    if not transformers:
        raise DypeError(f'The loaded {model_family} pipeline exposes no transformer for DyPE.')

    with contextlib.ExitStack() as stack:
        if model_family == 'flux1' and isinstance(pipeline, diffusers.FluxPipeline):
            stack.enter_context(resolution_shift_context(
                pipeline, height=height, width=width))
        for transformer in transformers:
            stack.callback(_patch_transformer(
                transformer, model_family, base_resolution, method, mscale, axis_scaling))
        yield pipeline
