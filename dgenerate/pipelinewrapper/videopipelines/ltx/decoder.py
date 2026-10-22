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

"""The LTX diffusion decoder: loading, tiling, progress, and module parking around it."""

from __future__ import annotations

import contextlib
import math

import torch

import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.pipelinewrapper.util as _putil
import dgenerate.pipelinewrapper.videopipelines as _vp
from .components import _is_missing_ltx_component
from .neighborhood import _ltx_diffusion_natten_processor


# Denoiser weights can leave the GPU before the diffusion decoder runs.
# Audio modules stay until the waveform is decoded.
_LTX_PARK_BEFORE_DIFFUSION_DECODE = (
    'transformer',
    'text_encoder',
    'connectors',
    'duration_head',
    'prompt_enhancer',
)


_LTX_PARK_AFTER_DIFFUSION_DECODE = (
    'audio_vae',
    'vocoder',
)


def _ltx_module_parkable(module) -> bool:
    if module is None or not isinstance(module, torch.nn.Module):
        return False
    if _putil.is_loaded_in_8bit_bnb(module):
        return False
    if hasattr(module, '_DGENERATE_ORIGINAL_TO_DISABLED'):
        return False
    if (_pipelines.is_sequential_cpu_offload_enabled(module)
            or _pipelines.is_model_cpu_offload_enabled(module)
            or _pipelines.is_group_offload_enabled(module)):
        return False
    return True


def _ltx_park_module(module) -> None:
    if not _ltx_module_parkable(module):
        return
    try:
        module.to('cpu')
    except Exception as error:
        _messages.debug_log(f'Could not park {module.__class__.__name__} on CPU: {error}')


def _release_ltx_offload_hooks(pipe) -> None:
    free_hooks = getattr(pipe, 'maybe_free_model_hooks', None)
    if callable(free_hooks):
        free_hooks()


def _park_named_ltx_modules(pipe, names) -> None:
    for name in names:
        _ltx_park_module(getattr(pipe, name, None))


def _park_ltx_for_diffusion_decode(pipe, wrapper) -> None:
    """
    Drop denoiser weights from VRAM before the diffusion decoder runs.

    Offloaded pipelines release GPU-resident hooks first. Full-GPU loads move
    the transformer and text stack to CPU. The audio VAE stays until the
    waveform decode that follows the video decode.
    """
    _release_ltx_offload_hooks(pipe)
    if not _vp._offload_requested(wrapper):
        _park_named_ltx_modules(pipe, _LTX_PARK_BEFORE_DIFFUSION_DECODE)
    _memory.torch_gc()


def _place_ltx_module(module, device) -> None:
    if not _ltx_module_parkable(module):
        return
    try:
        module.to(device)
    except Exception as error:
        _messages.debug_log(f'Could not place {module.__class__.__name__} on {device}: {error}')


def _park_ltx_after_diffusion_decode(pipe, decode_pipe, wrapper) -> None:
    """Return audio modules and the diffusion decoder to CPU after both decodes."""
    _release_ltx_offload_hooks(pipe)
    if decode_pipe is not None:
        _release_ltx_offload_hooks(decode_pipe)
    if _vp._offload_requested(wrapper):
        _memory.torch_gc()
        return
    _park_named_ltx_modules(pipe, _LTX_PARK_AFTER_DIFFUSION_DECODE)
    if decode_pipe is not None:
        _ltx_park_module(getattr(decode_pipe, 'diffusion_decoder', None))
    _memory.torch_gc()


def _load_ltx_diffusion_decoder(wrapper, pipe, uri_string=None):
    decoder = getattr(pipe, 'diffusion_decoder', None)
    if uri_string is None and isinstance(decoder, torch.nn.Module):
        _messages.debug_log('Reusing the main LTX pipeline diffusion_decoder for decode.')
        return decoder
    from diffusers import LTX2VideoDiffusionDecoderModel

    model_path = wrapper.model_path
    subfolder = 'diffusion_decoder'
    revision = None
    dtype = _enums.get_torch_dtype(wrapper._dtype) or torch.bfloat16
    if uri_string:
        uri = _uris.LTXDiffusionDecoderUri.parse(uri_string)
        model_path = uri.model
        revision = uri.revision
        subfolder = uri.subfolder or subfolder
        dtype = _enums.get_torch_dtype(uri.dtype or wrapper._dtype) or torch.bfloat16
    if uri_string is None and _is_missing_ltx_component(
            None, model_path, subfolder, LTX2VideoDiffusionDecoderModel.config_name):
        raise _pipelines.UnsupportedPipelineConfigError(
            f'The LTX checkpoint at "{wrapper.model_path}" does not contain a diffusion decoder. '
            'Use --ltx-video-decoder conv to use the convolutional VAE instead.')

    try:
        return LTX2VideoDiffusionDecoderModel.from_pretrained(
            model_path,
            subfolder=subfolder,
            torch_dtype=dtype,
            token=wrapper._auth_token,
            revision=revision,
            local_files_only=bool(wrapper._local_files_only))
    except Exception as e:
        if uri_string is None and _is_missing_ltx_component(
                e, model_path, subfolder, LTX2VideoDiffusionDecoderModel.config_name):
            raise _pipelines.UnsupportedPipelineConfigError(
                f'The LTX checkpoint at "{wrapper.model_path}" does not contain a diffusion decoder. '
                'Use --ltx-video-decoder conv to use the convolutional VAE instead.') from e
        raise


# Pixel tiles for the missing-NATTEN path. The decoder default is 768x768 and 80
# frames, which is the whole frame at the usual LTX size, so the expensive stages
# never split. 384x384 and 24 frames stay on the 8-pixel / 2-frame grid.
_LTX_DECODE_TILE_HEIGHT = 384


_LTX_DECODE_TILE_WIDTH = 384


_LTX_DECODE_TILE_FRAMES = 24


_LTX_DECODE_TILE_OVERLAP_PX = 64


_LTX_DECODE_TILE_OVERLAP_FRAMES = 8


def _aligned_tile_size(value: int, step: int, minimum: int) -> int:
    size = max(minimum, int(value) // step * step)
    return max(size, minimum)


def _ltx_fallback_decode_tile(override: str | None = None) -> tuple[int, int, int, int, int, int]:
    """Height, width, frames, and the three strides, in decoded pixels and frames."""
    raw = '' if override is None else str(override).strip().lower().replace(' ', '')
    height, width, frames = (
        _LTX_DECODE_TILE_HEIGHT, _LTX_DECODE_TILE_WIDTH, _LTX_DECODE_TILE_FRAMES)
    if raw:
        parts = raw.split('x')
        try:
            if len(parts) == 1:
                frames = int(parts[0], 0)
            elif len(parts) == 2:
                width = height = int(parts[0], 0)
                frames = int(parts[1], 0)
            elif len(parts) == 3:
                width = int(parts[0], 0)
                height = int(parts[1], 0)
                frames = int(parts[2], 0)
            else:
                raise ValueError
        except ValueError:
            raise _pipelines.UnsupportedPipelineConfigError(
                '--ltx-diffusion-decode-tile must look like 384x384x24, 384x24, or 24.')
    # The last upsample cell is 8 pixels and 2 frames. A neighborhood kernel needs
    # at least 48 pixels and 12 frames.
    height = _aligned_tile_size(height, 8, 48)
    width = _aligned_tile_size(width, 8, 48)
    frames = _aligned_tile_size(frames, 2, 12)
    overlap_px = min(_LTX_DECODE_TILE_OVERLAP_PX, height // 2, width // 2)
    overlap_frames = min(_LTX_DECODE_TILE_OVERLAP_FRAMES, frames // 2)
    stride_height = _aligned_tile_size(height - overlap_px, 8, 8)
    stride_width = _aligned_tile_size(width - overlap_px, 8, 8)
    stride_frames = _aligned_tile_size(frames - overlap_frames, 2, 2)
    stride_height = min(stride_height, height - 8)
    stride_width = min(stride_width, width - 8)
    stride_frames = min(stride_frames, frames - 2)
    return height, width, frames, stride_height, stride_width, stride_frames


def _enable_ltx_fallback_tiling(decoder, decode_tile: str | None = None) -> None:
    height, width, frames, stride_height, stride_width, stride_frames = (
        _ltx_fallback_decode_tile(decode_tile))
    decoder.enable_tiling(
        tile_sample_min_height=height,
        tile_sample_min_width=width,
        tile_sample_min_num_frames=frames,
        tile_sample_stride_height=stride_height,
        tile_sample_stride_width=stride_width,
        tile_sample_stride_num_frames=stride_frames)
    _messages.log(
        f'LTX diffusion decoder tiles are {width}x{height}, {frames} frames.')


def _prepare_ltx_diffusion_decoder(decoder, decode_tile: str | None = None):
    processor = _ltx_diffusion_natten_processor()
    decoder._dgenerate_ltx_fallback_tiles = bool(getattr(processor, 'tiles', False))
    if (decoder._dgenerate_ltx_fallback_tiles or decode_tile) and hasattr(decoder, 'enable_tiling'):
        _enable_ltx_fallback_tiling(decoder, decode_tile)
    decoder.set_attn_processor(processor)
    return decoder


def _diffusion_decoder_is_tiled(outer, latents) -> bool:
    if not getattr(outer, 'use_tiling', False) or not torch.is_tensor(latents) or latents.ndim < 5:
        return False
    spatial = int(getattr(outer, 'spatial_compression_ratio', 32) or 32)
    temporal = int(getattr(outer, 'temporal_compression_ratio', 8) or 8)
    return (
        latents.shape[2] > outer.tile_sample_min_num_frames // temporal
        or latents.shape[3] > outer.tile_sample_min_height // spatial
        or latents.shape[4] > outer.tile_sample_min_width // spatial)


def _diffusion_decode_tile_count(outer, features) -> int:
    """How many stage-5 tiles ``tiled_decode`` will denoise for ``features``."""
    from diffusers.models.autoencoders.ltx2_diffusion_decoder import _tile_intervals

    inner = outer.decoder
    upsample_stride = inner.upsamples[-1].stride
    patch_size = inner.patch_size
    scale_t = upsample_stride[0]
    scale_h = upsample_stride[1] * patch_size
    scale_w = upsample_stride[2] * patch_size
    min_sizes = [
        max(kernel_4, -(-kernel_5 // stride))
        for kernel_4, kernel_5, stride in zip(
            outer.config.decoder_stage_kernels[-1],
            outer.config.decoder_stage5_kernel,
            upsample_stride)
    ]
    ghost_frames = inner.trailing_pad_latent_frames * math.prod(
        up.stride[0] for up in inner.upsamples[:-1])
    num_frames = features.shape[1] - ghost_frames
    temporal = _tile_intervals(
        num_frames,
        outer.tile_sample_min_num_frames // scale_t,
        outer.tile_sample_stride_num_frames // scale_t,
        min_sizes[0])
    heights = _tile_intervals(
        features.shape[2],
        outer.tile_sample_min_height // scale_h,
        outer.tile_sample_stride_height // scale_h,
        min_sizes[1])
    widths = _tile_intervals(
        features.shape[3],
        outer.tile_sample_min_width // scale_w,
        outer.tile_sample_stride_width // scale_w,
        min_sizes[2])
    return max(1, len(temporal) * len(heights) * len(widths))


@contextlib.contextmanager
def _diffusion_decode_progress(decode_pipe):
    """
    Show a diffusers progress bar for the diffusion decoder.

    The bar is one step per tile. An untiled decode is one step, and that step
    completes when the single denoise returns.
    """
    outer = getattr(decode_pipe, 'diffusion_decoder', None)
    inner = getattr(outer, 'decoder', None)
    if inner is None or not callable(getattr(inner, 'denoise', None)):
        yield
        return
    if not callable(getattr(inner, 'forward_stages_1_to_3', None)):
        yield
        return

    bar = None
    original_stages = inner.forward_stages_1_to_3
    original_denoise = inner.denoise

    def stages(hidden_states, *args, **kwargs):
        nonlocal bar
        features = original_stages(hidden_states, *args, **kwargs)
        if bar is None:
            total = 1
            if _diffusion_decoder_is_tiled(outer, hidden_states):
                try:
                    total = _diffusion_decode_tile_count(outer, features)
                except Exception as error:
                    _messages.debug_log(
                        f'Could not count LTX diffusion decoder tiles: {error}')
            decode_pipe.set_progress_bar_config(desc='Diffusion decode')
            bar = decode_pipe.progress_bar(total=total)
        return features

    def denoise(*args, **kwargs):
        result = original_denoise(*args, **kwargs)
        if bar is not None:
            bar.update()
        return result

    inner.forward_stages_1_to_3 = stages
    inner.denoise = denoise
    try:
        yield
    finally:
        inner.forward_stages_1_to_3 = original_stages
        inner.denoise = original_denoise
        if bar is not None:
            bar.close()


def _decode_ltx_diffusion(wrapper, pipe, held, latents, generator, decode_tile=None,
                          diffusion_decoder_uri=None):
    decode_pipe = getattr(held, 'diffusion_decoder_pipe', None)
    if (decode_pipe is not None and getattr(
            decode_pipe, '_dgenerate_ltx_diffusion_decoder_uri', None) != diffusion_decoder_uri):
        held.diffusion_decoder_pipe = None
        decode_pipe = None
    if decode_pipe is None:
        from diffusers import LTX2VideoDiffusionDecodePipeline

        decoder = _prepare_ltx_diffusion_decoder(
            _load_ltx_diffusion_decoder(wrapper, pipe, diffusion_decoder_uri), decode_tile)
        decode_pipe = LTX2VideoDiffusionDecodePipeline(
            diffusion_decoder=decoder, scheduler=pipe.scheduler, vae=pipe.vae)
        _vp._apply_video_offload(
            decode_pipe, wrapper.device,
            bool(wrapper.model_cpu_offload), bool(wrapper.model_sequential_offload),
            bool(getattr(wrapper, 'model_group_offload', False)))
        held.diffusion_decoder_pipe = decode_pipe
        decode_pipe._dgenerate_ltx_diffusion_decoder_uri = diffusion_decoder_uri
    else:
        decoder = getattr(decode_pipe, 'diffusion_decoder', None)
        if decoder is not None and (
                getattr(decoder, '_dgenerate_ltx_fallback_tiles', False) or decode_tile):
            _enable_ltx_fallback_tiling(decoder, decode_tile)
    if not _vp._offload_requested(wrapper):
        _place_ltx_module(getattr(decode_pipe, 'diffusion_decoder', None), wrapper.device)
        _place_ltx_module(getattr(decode_pipe, 'vae', None), wrapper.device)
    with _diffusion_decode_progress(decode_pipe):
        frames = decode_pipe(
            latents=latents,
            denormalize=False,
            generator=generator,
            output_type='pil',
            return_dict=False)[0]
    return _vp.frames_from_output(frames)
