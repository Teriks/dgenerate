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

"""Wan-Animate-2: segment fitting, loading, compilation, and placement."""

from __future__ import annotations

import sys

import torch

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.types as _types
import dgenerate.pipelinewrapper.videopipelines as _vp
from .common import (
    _fit_output_media,
    _require_image_encoder,
    _spatial_multiple,
    _temporal_factor,
)


def _animate2_resampled_count(num_frames: int,
                              source_fps: float | None,
                              target_fps: float) -> int:
    """
    Frame count Wan-Animate-2 will denoise after resampling to ``target_fps``.

    Matches ``get_frame_indices``: the list length is
    ``int(num_frames / source_fps * target_fps)``. Equal rates, and a missing
    source rate, keep every frame.
    """
    if num_frames <= 0:
        return 0
    if source_fps is None or float(source_fps) == float(target_fps):
        return int(num_frames)
    return int(int(num_frames) / float(source_fps) * float(target_fps))


def _animate2_pad_fills(frame_count: int, segment: int, prev: int) -> bool:
    """
    True when the pipeline's zigzag pad makes every segment exactly ``segment`` frames.

    The pad copies a tail slice and flips it. A slice shorter than the requested
    pad leaves the segment short, and the transformer then rejects the token count.
    """
    if frame_count <= 0 or segment <= prev:
        return False
    effective = segment - prev
    if frame_count > prev:
        leftover = (frame_count - prev) % effective
    else:
        leftover = 0
    num_padding = effective - leftover if leftover > 0 else 0
    if num_padding > frame_count:
        return False
    padded = frame_count + num_padding
    if padded < segment:
        return False
    num_segments = (padded - prev + effective - 1) // effective
    if num_segments < 1:
        return False
    last = (num_segments - 1) * effective
    return last + segment <= padded


def _fit_wan_animate_2_segment(frame_count: int,
                               segment: int,
                               prev: int,
                               temporal: int = 4) -> int:
    """
    Largest ``temporal * k + 1`` segment, at most ``segment``, that this clip can fill.

    Returns ``segment`` unchanged when it already fits, and also when no shorter
    candidate fits.
    """
    temporal = max(1, int(temporal))
    segment = int(segment)
    prev = int(prev)
    if _animate2_pad_fills(frame_count, segment, prev):
        return segment
    k = (segment - 1) // temporal
    while k >= 0:
        candidate = temporal * k + 1
        if candidate < segment and candidate > prev \
                and _animate2_pad_fills(frame_count, candidate, prev):
            return candidate
        k -= 1
    return segment


def _call_wan_animate_2(wrapper, user_args):

    if user_args.video_length is not None:
        raise _pipelines.UnsupportedPipelineConfigError(
            '--video-lengths cannot be used with --model-type wan-animate-2. '
            'The output length follows the driving clip.')
    if not user_args.images:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Wan-Animate-2 needs a character image in --image-seeds.')
    if not user_args.wan_driving_video_frames:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Wan-Animate-2 needs wan-driving= in --image-seeds.')

    pipe, held = _vp._video_pipeline(wrapper, 'wan-animate-2', user_args.scheduler_uri)
    _require_image_encoder(pipe, 'wan-animate-2')
    positive, negative = _vp._prompt_text(user_args)
    width, height = _vp._size(user_args)
    align = _spatial_multiple(pipe)
    width, height = _vp._force_output_size_alignment(
        width, height, align, 'Wan-Animate-2')
    width, height = _fit_output_media(
        user_args, 'wan_driving_video_frames', [],
        width, height, align, 'Wan-Animate-2 driving clip')

    distilled = held.family == 'wan-animate-2-distilled'
    steps = int(_types.default(user_args.inference_steps, _constants.DEFAULT_INFERENCE_STEPS))
    if steps == _constants.DEFAULT_INFERENCE_STEPS:
        steps = _constants.DEFAULT_WAN_ANIMATE_2_DISTILLED_INFERENCE_STEPS if distilled else _constants.DEFAULT_WAN_ANIMATE_2_INFERENCE_STEPS
        user_args.inference_steps = steps
        _messages.debug_log(
            f'Wan-Animate-2 default inference steps '
            f'{_constants.DEFAULT_INFERENCE_STEPS} was replaced with {steps}.')

    fps = float(user_args.video_fps or _constants.DEFAULT_WAN_ANIMATE_2_FPS)
    segment = int(_types.default(
        user_args.wan_segment_frame_length, _constants.DEFAULT_WAN_ANIMATE_2_SEGMENT_FRAME_LENGTH))
    prev = int(_types.default(
        user_args.wan_prev_segment_frames, _constants.DEFAULT_WAN_ANIMATE_2_PREV_SEGMENT_FRAMES))
    source_fps = (
        float(user_args.wan_driving_video_fps)
        if user_args.wan_driving_video_fps else None)
    seen = _animate2_resampled_count(
        len(user_args.wan_driving_video_frames), source_fps, fps)
    fitted = _fit_wan_animate_2_segment(
        seen, segment, prev, _temporal_factor(pipe))
    if fitted != segment:
        if source_fps is not None and float(source_fps) != float(fps):
            clip = (
                f'{len(user_args.wan_driving_video_frames)} driving frames at '
                f'{source_fps:g} fps resample to {seen} frames at {fps:g} fps'
            )
        else:
            clip = f'the driving clip is {seen} frames at {fps:g} fps'
        _messages.warning(
            f'Wan-Animate-2: {clip}, which cannot fill a {segment}-frame segment. '
            f'--wan-segment-frame-lengths was reduced to {fitted}.')
        segment = fitted
    elif not _animate2_pad_fills(seen, segment, prev):
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Wan-Animate-2 driving clip is {seen} frames at {fps:g} fps, '
            f'too short to fill a segment longer than {prev} frames.')
    kwargs = {
        'image': user_args.images[0],
        'driving_video': list(user_args.wan_driving_video_frames),
        'prompt': positive,
        'generator': _vp._generator(wrapper, user_args),
        'output': 'videos',
        'output_type': 'pil',
        'fps': int(round(fps)),
        'num_inference_steps': steps,
        'segment_frame_length': segment,
        'prev_segment_conditioning_frames': prev,
    }
    if negative:
        kwargs['negative_prompt'] = negative
    if user_args.wan_driving_video_fps:
        kwargs['driving_video_fps'] = float(user_args.wan_driving_video_fps)
    if width is not None:
        kwargs['width'] = int(width)
        kwargs['height'] = int(height)
    if user_args.max_sequence_length is not None:
        kwargs['max_sequence_length'] = int(user_args.max_sequence_length)

    _vp._set_video_vae_slicing(pipe, bool(user_args.vae_slicing))
    videos = pipe(**kwargs)
    frames = _vp.frames_from_output(videos)
    return frames, None, None, fps


def load_wan_animate_2_pipeline(pipeline_class, model_path, load_kwargs, injected,
                                model_dtype):
    """
    Load a Wan-Animate-2 modular pipeline and its components.

    The checkpoint class is ``WanAnimate2Pipeline``. This build loads the modular
    pipeline instead. Weights load in ``load_components``. ``injected`` components,
    such as a GGUF transformer or ``--vae``, are registered first so they are not
    loaded again. ``None`` injections (``--text-encoders null``) are omitted from
    the load list; ``load_components`` treats ``None`` as unloaded and would
    otherwise reload them. When the VAE is not injected, it loads in float32.
    """
    from diffusers import WanAnimate2Blocks, WanAnimate2DistilledBlocks

    kwargs = dict(load_kwargs)
    kwargs.pop('torch_dtype', None)
    injected = dict(injected or {})
    for name in injected:
        kwargs.pop(name, None)
    if pipeline_class.__name__.startswith('WanAnimate2Distilled'):
        blocks = WanAnimate2DistilledBlocks()
    else:
        blocks = WanAnimate2Blocks()
    # The published checkpoints name WanAnimate2Pipeline, which this diffusers
    # build does not provide. The modular class is constructed directly and
    # load_components reads the checkpoint index.
    pipe = pipeline_class(
        blocks=blocks,
        pretrained_model_name_or_path=model_path)
    modules = {name: value for name, value in injected.items() if value is not None}
    null_names = frozenset(
        name for name, value in injected.items() if value is None)
    if modules:
        pipe.update_components(**modules)
    model_torch = _enums.get_torch_dtype(model_dtype) or torch.bfloat16
    dtype_map = {'default': model_torch}
    if 'vae' not in injected:
        dtype_map['vae'] = torch.float32
    component_kwargs = {
        'torch_dtype': dtype_map,
    }
    for name in ('revision', 'variant', 'subfolder', 'local_files_only', 'token'):
        if name in kwargs:
            component_kwargs[name] = kwargs[name]
    # Explicit names: skip already-injected modules and null slots. Default
    # load_components(names=None) reloads every getattr(..., None) slot.
    load_names = [
        name for name, spec in pipe._component_specs.items()
        if spec.default_creation_method == 'from_pretrained'
        and spec.pretrained_model_name_or_path is not None
        and getattr(pipe, name, None) is None
        and name not in null_names
    ]
    pipe.load_components(names=load_names, **component_kwargs)
    if null_names:
        pipe.update_components(**{name: None for name in null_names})
    return pipe


def _wan_animate_2_rope_params_real(max_seq_len, dim, theta=10000, offset=0):
    """
    Real-valued replacement for diffusers' Wan-Animate-2 ``rope_params``.

    diffusers builds a complex128 table, which MPS cannot hold. The angles are
    still computed in float64 on the CPU, then stored as a float32 tensor of
    shape ``[max_seq_len, dim // 2, 2]`` holding ``(cos, sin)``.
    """
    assert dim % 2 == 0
    angles = torch.outer(
        torch.arange(max_seq_len, device='cpu') + offset,
        1.0 / torch.pow(theta, torch.arange(0, dim, 2, device='cpu').to(torch.float64).div(dim)),
    )
    return torch.stack([angles.cos(), angles.sin()], dim=-1).float()


def _wan_animate_2_rope_apply_real(complex_rope_apply):
    """
    Build a real-valued replacement for diffusers' Wan-Animate-2 ``rope_apply``.

    The original casts activations to float64 and multiplies by a complex
    table, neither of which MPS supports. This version rotates each
    ``(even, odd)`` pair with ``cos`` / ``sin`` in float32 and returns float32,
    like the original. A complex table, from a pipeline built before the swap,
    is passed to ``complex_rope_apply`` unchanged.
    """

    def rope_apply(x, grid_sizes, freqs, time_stride=1):
        if freqs.is_complex():
            return complex_rope_apply(x, grid_sizes, freqs, time_stride)

        n, c = x.size(2), x.size(3) // 2

        # split freqs, [seq, c, 2] -> three tables along the channel axis
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w

            x_i = x[i, :seq_len].float().reshape(seq_len, n, -1, 2)
            freqs_i = torch.cat(
                [
                    freqs[0][: f * time_stride: time_stride].reshape(f, 1, 1, -1, 2).expand(f, h, w, -1, 2),
                    freqs[1][:h].reshape(1, h, 1, -1, 2).expand(f, h, w, -1, 2),
                    freqs[2][:w].reshape(1, 1, w, -1, 2).expand(f, h, w, -1, 2),
                ],
                dim=-2,
            ).reshape(seq_len, 1, -1, 2)

            cos, sin = freqs_i[..., 0], freqs_i[..., 1]
            real, imag = x_i[..., 0], x_i[..., 1]
            x_i = torch.stack([real * cos - imag * sin, real * sin + imag * cos], dim=-1).flatten(2)
            output.append(torch.cat([x_i, x[i, seq_len:].float()]))
        return torch.stack(output).float()

    return rope_apply


def _make_wan_animate_2_rope_mps_safe(transformer) -> None:
    """
    Swap Wan-Animate-2's float64 / complex128 RoPE helpers for float32 ones.

    diffusers' ``transformer_wan_animate_2`` builds a complex128 frequency
    table and casts query / key to float64 in ``rope_apply``. MPS supports
    neither dtype, so denoising fails there. The helpers are module-level
    functions looked up by name when the transformer runs, so replacing them on
    the module is enough. The swap applies process-wide, and it is safe on
    every device because the replacement ``rope_apply`` still accepts the
    original complex tables.

    :param transformer: loaded Wan-Animate-2 transformer, or ``None``
    """
    if transformer is None:
        return
    module = sys.modules.get(type(transformer).__module__)
    if (module is None
            or not hasattr(module, 'rope_params')
            or not hasattr(module, 'rope_apply')):
        return
    if not getattr(module, '_dgenerate_mps_safe_rope', False):
        module.rope_params = _wan_animate_2_rope_params_real
        module.rope_apply = _wan_animate_2_rope_apply_real(module.rope_apply)
        module._dgenerate_mps_safe_rope = True
        _messages.debug_log(
            'MPS has no float64 or complex128 support, replaced the '
            'Wan-Animate-2 rotary embedding helpers with float32 versions.')
    cache = getattr(transformer, 'rope_freqs_cache', None)
    if cache:
        # Tables built before the swap are complex128.
        cache.clear()


def _flex_attention_can_compile(device) -> bool:
    """Wan-Animate-2 uses the same Inductor check as ``--torch-compile``."""
    return _pipelines.inductor_can_compile(device)


def _compile_wan_animate_2_transformer(transformer, device) -> None:
    """
    Compile each repeated block's ``forward``.

    This runs on every Wan-Animate-2 load. The generation pass uses flex
    attention with a block mask, and the uncompiled call materializes the
    full score matrix. ``--torch-compile`` does not control this path.
    """
    if transformer is None or getattr(transformer, '_dgenerate_compiled', False):
        return
    if _pipelines.torch_compile_disabled():
        _messages.warning(
            'DGENERATE_TORCH_COMPILE=0, so Wan-Animate-2 leaves its transformer '
            'blocks eager. The generation pass allocates the full score matrix '
            'and can run out of memory at video resolution.')
        return
    repeated = getattr(transformer, '_repeated_blocks', None)
    if not repeated or _pipelines.module_skips_group_offload(transformer):
        return
    if not _flex_attention_can_compile(device):
        if torch.device(device).type in ('cuda', 'xpu'):
            detail = 'Triton is not installed, so the blocks stay uncompiled.'
        else:
            detail = 'This device has no compiled flex-attention kernel.'
        _messages.warning(
            'Wan-Animate-2 keeps generation attention block-sparse by compiling '
            f'its transformer blocks. {detail} The generation pass allocates '
            'the full score matrix and can run out of memory at video resolution.')
        return
    _messages.log(
        'Compiling Wan-Animate-2 transformer blocks. The first denoising step '
        'waits on this so attention stays block-sparse instead of allocating '
        'the full score matrix.')
    _pipelines.compile_repeated_forwards(transformer)


def place_wan_animate_2(pipe, device, model_cpu_offload, sequential_cpu_offload,
                        model_group_offload):
    """
    Place Wan-Animate-2 components.

    The modular pipeline has no pipeline-level CPU offload. Offload flags stream
    the transformer in block groups. The text encoder is UMT5-XXL, about 11 GB
    in bf16, and the image encoder is only used before denoising, so both are
    hooked to visit ``device`` for their forward pass and then return to CPU.
    Leaving them resident next to the transformer activations fills a 16 GB
    GPU. The VAE stays on ``device``. Quantized / GGUF transformers stay where
    they were loaded; group offload would move packed weights the same way it
    must not for still pipelines.
    """
    names = (
        'text_encoder', 'image_encoder', 'vae', 'transformer',
        'video_processor', 'image_processor', 'guider',
    )
    # Used once, ahead of the denoising loop. UMT5-XXL alone is ~11 GB.
    encoder_offload = ('text_encoder', 'image_encoder')
    offload = bool(model_cpu_offload or sequential_cpu_offload or model_group_offload)
    if offload:
        # The modular pipeline cannot take the three standard offload paths.
        # Every flag uses the placement below.
        if model_cpu_offload:
            chosen = '--model-cpu-offload'
        elif sequential_cpu_offload:
            chosen = '--model-sequential-offload'
        else:
            chosen = '--model-group-offload'
        onload_kind = torch.device(device).type
        # Diffusers rejects streamed offload unless each group is a single
        # block, and that stream prefetches the next block.
        if onload_kind == 'cuda':
            movement = (
                'moves the transformer one block at a time and prefetches '
                'the next block')
        else:
            movement = 'moves the transformer in groups of 4 blocks'
        _messages.warning(
            'Wan-Animate-2 uses one placement for --model-cpu-offload, '
            '--model-sequential-offload, and --model-group-offload. '
            f'{chosen} {movement}, moves the text encoder and image encoder '
            'to the GPU only for their forward pass, and leaves the VAE on '
            'the GPU.')
    transformer = getattr(pipe, 'transformer', None)
    if str(device).split(':')[0] == 'mps':
        _make_wan_animate_2_rope_mps_safe(transformer)
    # Compile before the offload hooks wrap ``forward``. The compiled function
    # is what those hooks call. The VAE stays resident, and its decoder blocks
    # compile on the same load so the one cached pipeline is always compiled.
    _compile_wan_animate_2_transformer(transformer, device)
    if _pipelines.inductor_can_compile(device):
        vae_compiled = _pipelines.compile_repeated_forwards(
            getattr(pipe, 'vae', None),
            _pipelines.vae_block_names(getattr(pipe, 'vae', None)))
        if vae_compiled:
            _messages.log(
                'Compiling Wan-Animate-2 VAE blocks. The first decode waits on this.')
    if offload and transformer is not None:
        if _pipelines.module_skips_group_offload(transformer):
            # No block streaming: packed BnB/SDNQ/GGUF weights are not ordinary
            # parameters. Still place the module on the run device so
            # CPU-quantized SDNQ and GGUF are not left off the execution device.
            # 8-bit bitsandbytes is left where device_map loaded it.
            _messages.debug_log(
                'Not group offloading Wan-Animate-2 transformer (quantized).')
            _pipelines.place_quantized_module(transformer, device)
        else:
            from diffusers.hooks import apply_group_offloading
            onload = torch.device(device)
            apply_group_offloading(
                transformer,
                onload_device=onload,
                offload_device=torch.device('cpu'),
                offload_type='block_level',
                num_blocks_per_group=1 if onload.type == 'cuda' else 4,
                use_stream=onload.type == 'cuda',
            )
        names = tuple(name for name in names if name != 'transformer')
    if offload:
        import accelerate
        for name in encoder_offload:
            module = getattr(pipe, name, None)
            if module is None or not hasattr(module, 'to'):
                continue
            if _pipelines.module_skips_group_offload(module):
                _pipelines.place_quantized_module(module, device)
                continue
            accelerate.cpu_offload(
                module, device,
                offload_buffers=len(module._parameters) > 0)
        names = tuple(name for name in names if name not in encoder_offload)
    for name in names:
        module = getattr(pipe, name, None)
        if module is not None and hasattr(module, 'to'):
            module.to(device)
