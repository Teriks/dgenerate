# Copyright (c) 2023, Teriks
#
# dgenerate is distributed under the following BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Helpers shared by the Wan, Wan-Animate, and Wan-Animate-2 call paths."""

from __future__ import annotations

import torch

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines


def _wan_mode_needs_image_encoder(mode: str) -> bool:
    return mode in ('wan-image', 'wan-flf', 'wan-animate', 'wan-animate-2')


def _require_image_encoder(pipe, mode: str):
    if not _wan_mode_needs_image_encoder(mode):
        return
    encoder = getattr(pipe, 'image_encoder', None)
    if encoder is None and 'image_encoder' not in getattr(pipe, 'components', {}):
        raise _pipelines.UnsupportedPipelineConfigError(
            'This repository has no image encoder. '
            'Use an I2V checkpoint for image-to-video or first-last-frame, '
            'or a Wan-Animate checkpoint with --model-type wan-animate or wan-animate-2.')


def _temporal_factor(pipe) -> int:
    return int(getattr(pipe, 'vae_scale_factor_temporal', 4) or 4)


def _spatial_multiple(pipe) -> int:
    spatial = int(getattr(pipe, 'vae_scale_factor_spatial', 8) or 8)
    transformer = getattr(pipe, 'transformer', None) or getattr(pipe, 'transformer_2', None)
    patch = getattr(getattr(transformer, 'config', None), 'patch_size', None)
    if isinstance(patch, (list, tuple)) and len(patch) >= 3:
        return spatial * int(patch[1])
    if isinstance(patch, (list, tuple)) and len(patch) >= 2:
        return spatial * int(patch[-1])
    return spatial * 2


def _trim_wan_clip(frames: list, limit: int | None, temporal: int) -> list:
    count = len(frames) if limit is None else min(len(frames), limit)
    count = (count - 1) // temporal * temporal + 1
    if count < 1:
        raise _pipelines.UnsupportedPipelineConfigError(
            'The Wan output is too short to hold the conditioning clip. '
            'Use a longer --video-lengths.')
    return frames[:count]


def _apply_wan_config(pipe, user_args):
    updates = {}
    if user_args.wan_boundary_ratio is not None:
        updates['boundary_ratio'] = float(user_args.wan_boundary_ratio)
    if user_args.wan_expand_timesteps is not None:
        updates['expand_timesteps'] = bool(user_args.wan_expand_timesteps)
    if updates and hasattr(pipe, 'register_to_config'):
        pipe.register_to_config(**updates)


def _vae_is_quantized(vae) -> bool:
    """True when casting the VAE dtype would break quantized weights."""
    if vae is None:
        return False
    if getattr(vae, 'is_quantized', False):
        return True
    if getattr(vae, 'quantization_config', None) is not None:
        return True
    hf_quantizer = getattr(vae, 'hf_quantizer', None)
    return hf_quantizer is not None


def _set_wan_vae_dtype(pipe, dtype=None):
    """
    Cast the Wan VAE dtype.

    AutoencoderKLWan is fragile in bfloat16. When ``dtype`` is ``None``,
    :data:`dgenerate.pipelinewrapper.constants.DEFAULT_WAN_VAE_DTYPE` (float32)
    is used. Pass an explicit dtype
    only when ``--vae`` did not already load the VAE at the desired precision.
    Quantized VAEs are left alone; ``.to(dtype=...)`` would undo quantization.

    Call this before group offload so the pinned CPU copy is already float32.
    Streamed group offload copies that snapshot back on every forward. A cast
    made after the hooks are installed is written into the snapshot on the
    next onload.
    """
    vae = getattr(pipe, 'vae', None)
    if vae is None:
        return
    if _vae_is_quantized(vae):
        _messages.debug_log('Wan VAE dtype left unchanged (quantized).')
        return
    if dtype is None:
        dtype = _constants.DEFAULT_WAN_VAE_DTYPE
    torch_dtype = _enums.get_torch_dtype(dtype)
    if torch_dtype is None:
        torch_dtype = torch.float32
    vae.to(dtype=torch_dtype)
    _messages.debug_log(f'Wan VAE dtype set to {torch_dtype}.')


def _fit_output_media(user_args, primary: str, companions, width, height,
                      align: int, label: str):
    """
    Fit ``primary`` with the still-image ``--output-size`` rule and resize
    ``companions`` onto that same canvas.

    ``companions`` is a list of ``(attribute, nearest)``.
    """
    from dgenerate.pipelinewrapper import videopipelines as _pipelines_pkg

    frames = getattr(user_args, primary)
    if not frames:
        return width, height
    source = frames[0].size
    target_w, target_h = _pipelines_pkg.conditioning_canvas(
        source, width, height, user_args.aspect_correct, align)
    target = (target_w, target_h)
    if source != target:
        _messages.log(
            f'Resizing {label} from {source[0]}x{source[1]} to {target_w}x{target_h}.')
    setattr(user_args, primary, _pipelines_pkg.resize_media(frames, target))
    for attr, nearest in companions:
        extra = getattr(user_args, attr, None)
        if extra:
            setattr(user_args, attr, _pipelines_pkg.resize_media(
                extra, target, nearest=nearest))
    user_args.width = target_w
    user_args.height = target_h
    return target_w, target_h
