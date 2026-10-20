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

"""LTX-Video and LTX-2 / 2.5 video generation."""

from __future__ import annotations

import collections.abc
import contextlib
import gc
import inspect
import json
import math
import os
import re
import warnings

import numpy
import torch

import dgenerate.eval as _eval
import dgenerate.hfhub as _hfhub
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.pipelinewrapper.util as _putil
import dgenerate.types as _types


class _Shared:
    """Look up helpers on the package so tests can patch ``_invoke`` there."""

    def __getattr__(self, name):
        from dgenerate.pipelinewrapper import videopipelines
        return getattr(videopipelines, name)


_vp = _Shared()

LTX_SCHEDULER_NAMES = frozenset({'FlowMatchEulerDiscreteScheduler'})


def audio_sample_rate_from_pipeline(pipe, audio) -> int | None:
    """
    Sample rate for muxing pipeline audio, or ``None`` when it cannot be muxed.

    :param pipe: loaded LTX pipeline
    :param audio: normalized audio array, or ``None``
    :return: Hz, or ``None``
    """
    if audio is None:
        return None
    vocoder = getattr(pipe, 'vocoder', None)
    config = getattr(vocoder, 'config', None) if vocoder is not None else None
    rate = getattr(config, 'output_sampling_rate', None) if config is not None else None
    if rate is None:
        _messages.warning(
            'LTX returned audio but the pipeline has no vocoder sample rate. '
            'The soundtrack will not be muxed.')
        return None
    return int(rate)

def ltx_family_from_index(index: dict | None) -> str:
    """
    Choose the LTX pipeline family from ``model_index.json``.

    ``LTX2*`` is LTX-2 / 2.5. ``LTX*`` is the earlier T5 pipeline, including
    LTX-Video 0.9.

    :param index: ``model_index.json`` dict
    :return: ``ltx2`` or ``ltx``
    """
    name = str((index or {}).get('_class_name') or '')
    if name.startswith('LTX2'):
        return 'ltx2'
    if name.startswith('LTX'):
        return 'ltx'
    raise _pipelines.UnsupportedPipelineConfigError(
        'This repository is not an LTX pipeline. '
        f'model_index.json _class_name is {name!r}.')


def ltx_num_frames(seconds: float, fps: float) -> int:
    """
    Snap a requested duration to the LTX frame count ``8k+1``.

    :param seconds: requested length in seconds
    :param fps: frame rate
    :return: frame count
    """
    raw = max(1, int(round(float(seconds) * float(fps))))
    k = max(0, int(round((raw - 1) / 8.0)))
    return 8 * k + 1


def _custom_ltx_condition(parsed) -> bool:
    if parsed is None:
        return False
    if parsed.ltx_extra_conditions:
        return True
    if parsed.ltx_condition_index not in (None, 0):
        return True
    return parsed.ltx_condition_strength not in (None, 1, 1.0)


def _classify_ltx(parsed, ic_lora: bool = False) -> str:
    if _vp._control_count(parsed) > 1:
        raise _pipelines.UnsupportedPipelineConfigError(
            'LTX accepts one control= clip, used as the IC-LoRA reference video.')
    if parsed is not None and (parsed.multi_image_mode or _vp._image_count(parsed) > 1):
        raise _pipelines.UnsupportedPipelineConfigError(
            'LTX accepts one conditioning image. Use a single path for the first frame, '
            'or last-frame= for the last frame.')
    _, _, control = _vp.video_seed_slots(parsed, ic_lora)
    if control is not None and not ic_lora:
        raise _pipelines.UnsupportedPipelineConfigError(
            'The image seed control= clip is the IC-LoRA reference video. '
            'Load the IC-LoRA with --ltx-ic-lora.')
    if ic_lora:
        if control is None:
            raise _pipelines.UnsupportedPipelineConfigError(
                '--ltx-ic-lora needs a control clip. Use --image-seeds "control.mp4", '
                'or "first.png;control=control.mp4" to add a first frame.')
        return 'ltx-control'
    if _vp._has_end(parsed) or _custom_ltx_condition(parsed):
        return 'ltx-condition'
    if _vp._image_count(parsed) == 1:
        return 'ltx-image'
    return 'ltx-txt'


def _ltx_pipeline_class(mode: str, family: str = 'ltx2'):
    if family == 'ltx':
        from diffusers import LTXConditionPipeline, LTXImageToVideoPipeline, LTXPipeline
        classes = {
            'ltx-txt': LTXPipeline,
            'ltx-image': LTXImageToVideoPipeline,
            'ltx-condition': LTXConditionPipeline,
        }
        if mode == 'ltx-control':
            raise _pipelines.UnsupportedPipelineConfigError(
                'The image seed control= argument needs an LTX-2 checkpoint. '
                'The earlier LTX-Video pipeline has no IC-LoRA reference conditioning.')
    else:
        from diffusers import (
            LTX2ConditionPipeline,
            LTX2ImageToVideoPipeline,
            LTX2InContextPipeline,
            LTX2Pipeline)
        classes = {
            'ltx-txt': LTX2Pipeline,
            'ltx-image': LTX2ImageToVideoPipeline,
            'ltx-condition': LTX2ConditionPipeline,
            'ltx-control': LTX2InContextPipeline,
        }
    return classes[mode]


def _call_ltx(wrapper, user_args):
    if user_args.reference_video_frames:
        if not getattr(wrapper, 'ltx_ic_lora_uri', None):
            raise _pipelines.UnsupportedPipelineConfigError(
                'An LTX control clip needs an IC-LoRA. Load one with --ltx-ic-lora.')
        mode = 'ltx-control'
    elif user_args.end_images or user_args.video_frames or user_args.end_video_frames:
        mode = 'ltx-condition'
    elif user_args.images and _ltx_image_needs_conditions(user_args):
        mode = 'ltx-condition'
    elif user_args.images:
        mode = 'ltx-image'
    else:
        mode = 'ltx-txt'

    # Inspect the checkpoint/component metadata before _video_pipeline can
    # start loading the main model weights. These checks do not instantiate
    # either optional component.
    _preflight_ltx_components(wrapper, user_args)
    pipe, held = _vp._video_pipeline(wrapper, mode, user_args.scheduler_uri)
    family = held.family
    positive, negative = _vp._prompt_text(user_args)
    width, height = _vp._size(user_args)
    align = _ltx_canvas_align(pipe, held, mode, user_args)
    width, height = _vp._force_output_size_alignment(width, height, align, 'LTX')
    if mode != 'ltx-txt':
        width, height = _fit_ltx_inputs(pipe, held, mode, user_args, width, height)

    fps = float(user_args.video_fps or _constants.DEFAULT_LTX_FPS)
    kwargs = {
        'prompt': positive,
        'frame_rate': fps,
        'generator': _vp._generator(wrapper, user_args),
        'output_type': 'pil'
    }
    if negative:
        kwargs['negative_prompt'] = negative
    if width is not None:
        kwargs['width'] = width
        kwargs['height'] = height

    if user_args.video_length is not None:
        num_frames = ltx_num_frames(user_args.video_length, fps)
        kwargs['num_frames'] = num_frames
        _messages.debug_log(
            f'LTX clip length {user_args.video_length} seconds at {fps} fps '
            f'-> {num_frames} frames.')
    elif mode == 'ltx-control':
        num_frames = len(_trim_ltx_clip(
            user_args.reference_video_frames, None, _ltx_temporal_compression(pipe)))
        kwargs['num_frames'] = num_frames
        _messages.debug_log(f'LTX clip length follows the control clip: {num_frames} frames.')
    elif family == 'ltx2':
        _messages.debug_log('LTX will choose the clip length from the prompt.')
    else:
        _messages.debug_log('LTX-Video will use its default frame count.')

    guidance = float(
        _types.default(user_args.guidance_scale, _constants.DEFAULT_GUIDANCE_SCALE))
    user_audio = user_args.ltx_audio_guidance_scale
    if family == 'ltx':
        if user_audio is not None or user_args.ltx_audio_guidance_rescale is not None:
            raise _pipelines.UnsupportedPipelineConfigError(
                'Audio guidance is only supported by LTX-2. This checkpoint uses the earlier LTX pipeline.')
        kwargs['num_inference_steps'] = int(
            _types.default(user_args.inference_steps, _constants.DEFAULT_INFERENCE_STEPS))
        kwargs['guidance_scale'] = guidance
        _warn_legacy_ltx_canvas(width, height, kwargs.get('num_frames'), fps)
        schedule_size = (
            width if width is not None else 704,
            height if height is not None else 512,
            kwargs.get('num_frames', 161),
            kwargs['num_inference_steps'])
        _require_legacy_ltx_schedule(pipe, *schedule_size)
        if mode == 'ltx-condition':
            timesteps = _fix_legacy_ltx_condition_schedule(pipe, *schedule_size)
            if timesteps is not None:
                kwargs['timesteps'] = timesteps
        if user_args.sigmas is not None:
            raise _pipelines.UnsupportedPipelineConfigError(
                '--sigmas is only applied to LTX-2. This checkpoint uses the earlier LTX pipeline.')
    else:
        distilled = _ltx_is_distilled(pipe)

        if user_args.sigmas is not None:
            if isinstance(user_args.sigmas, str):
                kwargs['sigmas'] = _eval_sigma_expression(
                    user_args.sigmas,
                    _ltx_base_sigmas(pipe, distilled, user_args.inference_steps))
            else:
                kwargs['sigmas'] = [float(value) for value in user_args.sigmas]
            kwargs['guidance_scale'] = guidance
            kwargs['audio_guidance_scale'] = (
                float(user_audio) if user_audio is not None else guidance)
            user_args.inference_steps = len(kwargs['sigmas'])
        elif distilled:
            from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

            kwargs['sigmas'] = list(DISTILLED_SIGMA_VALUES)
            if guidance == _constants.DEFAULT_GUIDANCE_SCALE:
                guidance = 1.0
                user_args.guidance_scale = 1.0
                _messages.debug_log(
                    'LTX distilled checkpoint runs unguided. Default guidance scale 5 was replaced with 1.')
            else:
                _messages.debug_log(
                    f'LTX distilled checkpoint: distilled sigma schedule, guidance {guidance}.')
            kwargs['guidance_scale'] = guidance
            kwargs['audio_guidance_scale'] = (
                float(user_audio) if user_audio is not None else guidance)
            scheduled = len(kwargs['sigmas'])
            ignored = _ltx_ignored_steps_warning(user_args.inference_steps, scheduled)
            if ignored:
                _messages.warning(ignored)
            user_args.inference_steps = scheduled
        else:
            steps = int(_types.default(user_args.inference_steps, _constants.DEFAULT_INFERENCE_STEPS))
            if guidance == _constants.DEFAULT_GUIDANCE_SCALE:
                guidance = 3.0
                user_args.guidance_scale = 3.0
                audio_guidance = 7.0 if user_audio is None else float(user_audio)
                if user_audio is None:
                    user_args.ltx_audio_guidance_scale = audio_guidance
                _messages.warning(
                    f'LTX full checkpoint: guidance scale 5 was replaced with 3, '
                    f'and audio guidance is {audio_guidance}.')
            else:
                audio_guidance = float(user_audio) if user_audio is not None else guidance
                _messages.debug_log(
                    f'LTX full checkpoint: {steps} steps, guidance {guidance}, '
                    f'audio guidance {audio_guidance}.')
            kwargs['num_inference_steps'] = steps
            kwargs['guidance_scale'] = guidance
            kwargs['audio_guidance_scale'] = audio_guidance

    if user_args.guidance_rescale is not None:
        kwargs['guidance_rescale'] = float(user_args.guidance_rescale)
    if family == 'ltx2':
        if user_args.ltx_audio_guidance_rescale is not None:
            kwargs['audio_guidance_rescale'] = float(user_args.ltx_audio_guidance_rescale)
        elif user_args.guidance_rescale is not None:
            kwargs['audio_guidance_rescale'] = float(user_args.guidance_rescale)

    if user_args.max_sequence_length is not None:
        kwargs['max_sequence_length'] = int(user_args.max_sequence_length)

    _vp._set_ltx_vae_slicing(pipe, bool(user_args.vae_slicing))

    if mode == 'ltx-image':
        kwargs['image'] = user_args.images[0]
    elif mode == 'ltx-condition':
        kwargs['conditions'] = _ltx_conditions(pipe, family, user_args, kwargs.get('num_frames'))
    elif mode == 'ltx-control':
        kwargs.update(_ltx_reference_kwargs(
            wrapper, pipe, held, user_args, kwargs['num_frames'], width, height))
        if (user_args.images or user_args.end_images or
                user_args.video_frames or user_args.end_video_frames
                or user_args.ltx_extra_conditions or _custom_ltx_condition_args(user_args)):
            kwargs['conditions'] = _ltx_conditions(pipe, family, user_args, kwargs['num_frames'])

    if family == 'ltx2':
        _apply_ltx2_options(wrapper, pipe, user_args, kwargs)
    else:
        _reject_ltx2_only_options(user_args)

    frames, audio, sample_rate = _complete_ltx_call(wrapper, pipe, held, user_args, kwargs, family)
    if user_args.ltx_latent_upscale and user_args.ltx_latent_upscale_mode == 'temporal':
        fps *= 2 ** int(user_args.ltx_temporal_upscale_rounds)
    return frames, audio, sample_rate, fps


def _custom_ltx_condition_args(user_args) -> bool:
    if user_args.ltx_condition_index not in (None, 0):
        return True
    return user_args.ltx_condition_strength not in (None, 1, 1.0)


def _reject_ltx2_only_options(user_args):
    used = []
    if user_args.ltx_stg_scale is not None or user_args.ltx_stg_blocks:
        used.append('--ltx-stg-scales')
    if user_args.ltx_modality_scale is not None:
        used.append('--ltx-modality-scales')
    if user_args.ltx_latent_upscale:
        used.append('--ltx-latent-upscale')
    if user_args.ltx_video_decoder == 'diffusion':
        used.append('--ltx-video-decoder diffusion')
    if user_args.ltx_diffusion_decoder:
        used.append('--ltx-diffusion-decoder')
    if user_args.ltx_diffusion_decode_tile:
        used.append('--ltx-diffusion-decode-tile')
    if user_args.ltx_prompt_enhancer:
        used.append('--ltx-prompt-enhancer')
    if user_args.ltx_image_crf is not None:
        used.append('--ltx-image-crfs')
    if user_args.ltx_use_cross_timestep is False:
        used.append('--ltx-no-cross-timestep')
    if user_args.ltx_video_min_seconds is not None or user_args.ltx_video_max_seconds is not None:
        used.append('--ltx-video-min-seconds')
    if used:
        raise _pipelines.UnsupportedPipelineConfigError(
            'The earlier LTX pipeline does not support ' + ', '.join(used) + '.')


def _apply_ltx2_options(wrapper, pipe, user_args, kwargs):
    accepted = None

    def require(name, option):
        nonlocal accepted
        if accepted is None:
            accepted = set(inspect.signature(pipe.__call__).parameters)
        if name not in accepted:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'This LTX pipeline does not accept {option}.')

    if user_args.ltx_prompt_enhancer:
        _ensure_prompt_enhancer(wrapper, pipe, user_args.ltx_prompt_enhancer)
        require('enable_prompt_enhancement', '--ltx-prompt-enhancer')
        kwargs['enable_prompt_enhancement'] = True
    if user_args.ltx_system_prompt:
        require('system_prompt', '--ltx-system-prompt')
        kwargs['system_prompt'] = user_args.ltx_system_prompt
    if user_args.ltx_stg_scale is not None:
        require('stg_scale', '--ltx-stg-scales')
        kwargs['stg_scale'] = float(user_args.ltx_stg_scale)
        audio_stg = user_args.ltx_stg_scale if user_args.ltx_audio_stg_scale is None else user_args.ltx_audio_stg_scale
        kwargs['audio_stg_scale'] = float(audio_stg)
        if float(user_args.ltx_stg_scale) > 0 or float(audio_stg) > 0:
            blocks = list(user_args.ltx_stg_blocks or [28])
            require('spatio_temporal_guidance_blocks', '--ltx-stg-blocks')
            kwargs['spatio_temporal_guidance_blocks'] = [int(block) for block in blocks]
            if user_args.ltx_stg_blocks is None:
                _messages.debug_log('LTX spatio-temporal guidance is using transformer block 28.')
    elif user_args.ltx_stg_blocks:
        raise _pipelines.UnsupportedPipelineConfigError(
            '--ltx-stg-blocks needs --ltx-stg-scales greater than 0.')
    if user_args.ltx_modality_scale is not None:
        require('modality_scale', '--ltx-modality-scales')
        kwargs['modality_scale'] = float(user_args.ltx_modality_scale)
        audio_modality = (user_args.ltx_modality_scale if user_args.ltx_audio_modality_scale is None
                          else user_args.ltx_audio_modality_scale)
        kwargs['audio_modality_scale'] = float(audio_modality)
    if user_args.ltx_use_cross_timestep is False:
        require('use_cross_timestep', '--ltx-no-cross-timestep')
        kwargs['use_cross_timestep'] = False
    if user_args.ltx_image_crf is not None:
        require('image_crf', '--ltx-image-crfs')
        kwargs['image_crf'] = int(user_args.ltx_image_crf)
    if user_args.video_length is None:
        if user_args.ltx_video_min_seconds is not None:
            require('min_seconds', '--ltx-video-min-seconds')
            kwargs['min_seconds'] = float(user_args.ltx_video_min_seconds)
        if user_args.ltx_video_max_seconds is not None:
            require('max_seconds', '--ltx-video-max-seconds')
            kwargs['max_seconds'] = float(user_args.ltx_video_max_seconds)
    if user_args.ltx_decode_timestep is not None:
        require('decode_timestep', '--ltx-decode-timesteps')
        kwargs['decode_timestep'] = float(user_args.ltx_decode_timestep)
    if user_args.ltx_decode_noise_scale is not None:
        require('decode_noise_scale', '--ltx-decode-noise-scales')
        kwargs['decode_noise_scale'] = float(user_args.ltx_decode_noise_scale)


def _ensure_prompt_enhancer(wrapper, pipe, uri_string):
    if (getattr(pipe, '_dgenerate_prompt_enhancer', None) == uri_string
            and getattr(pipe, 'prompt_enhancer', None) is not None):
        return
    from transformers import AutoModelForImageTextToText, AutoProcessor

    uri = _uris.LTXPromptEnhancerUri.parse(uri_string)
    enhancer_dtype = uri.dtype or wrapper._dtype
    if _enums.get_torch_dtype(enhancer_dtype) is None:
        enhancer_dtype = _enums.DataType.BFLOAT16
    dtype = _enums.get_torch_dtype(enhancer_dtype)
    load = {
        'token': wrapper._auth_token,
        'local_files_only': bool(wrapper._local_files_only),
    }
    if uri.revision is not None:
        load['revision'] = uri.revision
    if uri.subfolder is not None:
        load['subfolder'] = uri.subfolder
    _messages.debug_log(f'Loading LTX prompt enhancer "{uri.model}".')
    if uri.quantizer:
        pipe.prompt_enhancer = _vp._load_quantized_module(
            AutoModelForImageTextToText, uri.model, uri.subfolder, uri.revision,
            None, enhancer_dtype, uri.quantizer, wrapper._auth_token,
            wrapper._local_files_only,
            _vp._quantized_device_map(wrapper.device, False, uri.quantizer))
    else:
        pipe.prompt_enhancer = AutoModelForImageTextToText.from_pretrained(
            uri.model, torch_dtype=dtype, **load)
    pipe.processor = AutoProcessor.from_pretrained(uri.model, **load)
    device = 'cpu' if _vp._offload_requested(wrapper) else wrapper.device
    if not _putil.is_loaded_in_8bit_bnb(pipe.prompt_enhancer):
        pipe.prompt_enhancer.to(device)
    pipe._dgenerate_prompt_enhancer = uri_string


def _complete_ltx_call(wrapper, pipe, held, user_args, kwargs, family):
    # Run enhancement before entering Diffusers' denoising call so its model can
    # be returned to CPU (or released for BNB 8-bit) before video inference.
    enhancement_requested = (
        kwargs.get('enable_prompt_enhancement')
        and kwargs.get('prompt') is not None
        and getattr(pipe, 'prompt_enhancer', None) is not None)
    _enhance_ltx_prompt_once(wrapper, pipe, kwargs)
    enhancer = _detach_prompt_enhancer(pipe) if enhancement_requested else None
    try:
        diffusion = family == 'ltx2' and user_args.ltx_video_decoder == 'diffusion'
        if user_args.ltx_latent_upscale:
            if user_args.ltx_latent_upscale_mode == 'temporal':
                frames, audio, sample_rate = _ltx_temporal_upscale(
                    wrapper, pipe, held, user_args, kwargs, diffusion)
                return frames, audio, sample_rate
            return _ltx_two_stage(wrapper, pipe, held, user_args, kwargs, diffusion)
        if diffusion:
            kwargs = dict(kwargs)
            kwargs['output_type'] = 'latent'
        output = _vp._invoke(wrapper, pipe, kwargs, user_args=user_args)
        return _ltx_pixels_from_output(
            wrapper, pipe, held, output, diffusion, kwargs.get('generator'),
            getattr(user_args, 'ltx_diffusion_decode_tile', None),
            getattr(user_args, 'ltx_diffusion_decoder', None))
    finally:
        if enhancer is not None:
            pipe.prompt_enhancer = enhancer


def _enhance_ltx_prompt_once(wrapper, pipe, kwargs):
    """
    Rewrite the prompt once, then turn enhancement off for every later pipeline call.

    ``enhance_prompt`` moves Gemma onto the compute device. Return it to CPU
    immediately afterward so it does not occupy accelerator memory during
    denoising, including when the main pipeline has no offload mode enabled.
    """
    if not kwargs.get('enable_prompt_enhancement') or kwargs.get('prompt') is None:
        return
    if getattr(pipe, 'prompt_enhancer', None) is None:
        return
    system_prompt = kwargs.get('system_prompt')
    image = kwargs.get('image')
    conditions = kwargs.get('conditions') or ()
    if image is None:
        for condition in conditions:
            image = getattr(condition, 'frames', getattr(condition, 'image', None))
            if isinstance(image, (list, tuple)):
                image = image[0] if image else None
            if image is not None:
                break
    if system_prompt is None:
        from diffusers.pipelines.ltx2.utils import (
            LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT,
            LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT)
        if image is not None or conditions:
            system_prompt = LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT
        else:
            system_prompt = LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT
    enhanced = pipe.enhance_prompt(
        prompt=kwargs['prompt'],
        system_prompt=system_prompt,
        generator=kwargs.get('generator'),
        device=wrapper.device,
        image=image)
    if isinstance(enhanced, (list, tuple)):
        enhanced = enhanced[0]
    kwargs['prompt'] = enhanced
    kwargs['enable_prompt_enhancement'] = False
    enhancer = pipe.prompt_enhancer
    if _putil.is_loaded_in_8bit_bnb(enhancer):
        # BitsAndBytes 8-bit modules cannot be moved back to CPU with .to().
        # Drop the pipeline reference instead so the accelerator memory can be
        # reclaimed; _ensure_prompt_enhancer reloads it for the next generation.
        pipe.prompt_enhancer = None
        del enhancer
        gc.collect()
    else:
        enhancer.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _detach_prompt_enhancer(pipe):
    """
    Drop the prompt enhancer from the pipeline while a stage LoRA is loaded.

    Loading LoRA removes offload hooks and calls ``enable_sequential_cpu_offload``
    again. That call is dgenerate's method, and it still hooks every component
    present at that moment.
    """
    enhancer = getattr(pipe, 'prompt_enhancer', None)
    if enhancer is None:
        return None
    pipe.prompt_enhancer = None
    return enhancer


def _ltx_two_stage(wrapper, pipe, held, user_args, kwargs, diffusion):
    width = kwargs.get('width')
    height = kwargs.get('height')
    if width is None or height is None:
        raise _pipelines.UnsupportedPipelineConfigError(
            '--ltx-latent-upscale needs an --output-size. '
            'That size is the finished clip.')
    _enhance_ltx_prompt_once(wrapper, pipe, kwargs)
    stage1 = dict(kwargs)
    stage1['width'] = int(width) // 2
    stage1['height'] = int(height) // 2
    stage1['output_type'] = 'latent'
    _messages.log(
        f'LTX stage 1 at {stage1["width"]}x{stage1["height"]}, '
        f'then a latent upscale to {width}x{height}.')
    first = _vp._invoke(wrapper, pipe, stage1, user_args=user_args)
    video_latent = _video_latent_tensor(first.frames)
    audio_latent = getattr(first, 'audio', None)

    upscaled = _upsample_ltx_latents(wrapper, pipe, video_latent, user_args)
    stage2 = {key: value for key, value in kwargs.items() if key not in ('image', 'conditions')}
    stage2['width'] = int(width)
    stage2['height'] = int(height)
    stage2['latents'] = upscaled
    stage2['audio_latents'] = audio_latent
    if stage2.get('num_frames') is None:
        temporal = _ltx_temporal_compression(pipe)
        stage2['num_frames'] = (int(video_latent.shape[2]) - 1) * temporal + 1
    sigmas = _stage_sigmas(user_args)
    stage2['sigmas'] = sigmas
    stage2.pop('num_inference_steps', None)
    stage2['noise_scale'] = (
        float(sigmas[0]) if user_args.ltx_noise_scale is None else float(user_args.ltx_noise_scale))
    stage_guidance = 1.0 if user_args.ltx_stage_guidance_scale is None else float(user_args.ltx_stage_guidance_scale)
    stage_audio = (stage_guidance if user_args.ltx_stage_audio_guidance_scale is None
                   else float(user_args.ltx_stage_audio_guidance_scale))
    stage2['guidance_scale'] = stage_guidance
    stage2['audio_guidance_scale'] = stage_audio
    for key in (
            'ltx_stg_scale', 'ltx_audio_stg_scale', 'ltx_modality_scale', 'ltx_audio_modality_scale',
            'spatio_temporal_guidance_blocks'):
        stage2.pop(key, None)
    stage2['output_type'] = 'latent' if diffusion else 'pil'

    original = pipe.scheduler
    pipe.scheduler = original.__class__.from_config(
        dict(original.config), use_dynamic_shifting=False, shift_terminal=None)
    loaded_stage_lora = False
    enhancer = _detach_prompt_enhancer(pipe)
    try:
        if user_args.ltx_stage_lora_uris:
            _uris.LoRAUri.load_on_pipeline(
                pipeline=pipe,
                uris=user_args.ltx_stage_lora_uris,
                fuse_scale=1.0 if wrapper.lora_fuse_scale is None else wrapper.lora_fuse_scale,
                use_auth_token=wrapper._auth_token,
                local_files_only=bool(wrapper._local_files_only),
                fuse=False)
            loaded_stage_lora = True
        _messages.log('LTX stage 2 refine.')
        second = _vp._invoke(wrapper, pipe, stage2, user_args=user_args)
    finally:
        pipe.scheduler = original
        if loaded_stage_lora and hasattr(pipe, 'unload_lora_weights'):
            pipe.unload_lora_weights()
        if enhancer is not None:
            pipe.prompt_enhancer = enhancer
    return _ltx_pixels_from_output(
        wrapper, pipe, held, second, diffusion, kwargs.get('generator'),
        getattr(user_args, 'ltx_diffusion_decode_tile', None),
        getattr(user_args, 'ltx_diffusion_decoder', None))


def _ltx_temporal_upscale(wrapper, pipe, held, user_args, kwargs, diffusion):
    """Compose the Diffusers 0.41 LTX-2.5 DFR spatial and temporal passes."""
    from diffusers import LTX2DFRPipeline, LTX2DFRTemporalRefinePipeline
    from diffusers.pipelines.ltx2.utils import (
        DISTILLED_SIGMA_VALUES, TEMPORAL_ROUND_DISTILLED_SIGMA_VALUES)
    from diffusers.schedulers import LTXEulerAncestralRFScheduler

    if not getattr(pipe.transformer.config, 'use_keyframes_abs_pos_embedding', False):
        raise _pipelines.UnsupportedPipelineConfigError(
            'Temporal latent upscaling requires an LTX-2.5 transformer with '
            'use_keyframes_abs_pos_embedding enabled.')
    if user_args.reference_video_frames:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Temporal DFR does not preserve the IC-LoRA control clip through its refine rounds. '
            'Use --ltx-latent-upscale-mode spatial for control= generations.')
    if (kwargs.get('guidance_scale', 1.0) != 1.0
            or kwargs.get('audio_guidance_scale', 1.0) != 1.0
            or kwargs.get('guidance_rescale') is not None
            or user_args.ltx_stg_scale not in (None, 0, 0.0)
            or user_args.ltx_audio_stg_scale not in (None, 0, 0.0)
            or user_args.ltx_modality_scale not in (None, 1, 1.0)
            or user_args.ltx_audio_modality_scale not in (None, 1, 1.0)
            or user_args.ltx_audio_guidance_scale not in (None, 1, 1.0)
            or user_args.ltx_audio_guidance_rescale is not None
            or user_args.ltx_stage_guidance_scale not in (None, 1, 1.0)
            or user_args.ltx_stage_audio_guidance_scale not in (None, 1, 1.0)
            or user_args.ltx_stage_lora_uris):
        raise _pipelines.UnsupportedPipelineConfigError(
            'Temporal DFR uses its distilled schedule without CFG, STG, or modality guidance. '
            'Set video/audio guidance to 1, omit guidance rescale, and omit '
            '--ltx-stg-scales and --ltx-modality-scales.')

    width, height = int(kwargs['width']), int(kwargs['height'])
    requested_frames = kwargs.get('num_frames')
    num_frames = int(requested_frames) if requested_frames is not None else None
    fps = float(kwargs['frame_rate'])
    dtype = _enums.get_torch_dtype(wrapper._dtype) or torch.bfloat16
    dfr = LTX2DFRPipeline.from_pipe(pipe, dtype=dtype)
    accepted = set(inspect.signature(dfr.__call__).parameters)

    def dfr_kwargs(values):
        return {key: value for key, value in values.items() if key in accepted}

    _enhance_ltx_prompt_once(wrapper, dfr, kwargs)
    kwargs['enable_prompt_enhancement'] = False
    first_kwargs = dfr_kwargs(kwargs)
    first_kwargs.update(
        width=width // 2,
        height=height // 2,
        output_type='latent',
        return_dict=True,
        generate_slots=True,
        sigmas=(kwargs.get('sigmas') or list(DISTILLED_SIGMA_VALUES)))
    first_kwargs.pop('guidance_scale', None)
    first_kwargs.pop('audio_guidance_scale', None)
    _messages.log(
        f'LTX DFR spatial pass 1 at {width // 2}x{height // 2}, '
        f'pass 2 at {width}x{height}, then {user_args.ltx_temporal_upscale_rounds} '
        'temporal refinement round(s).')
    first = _vp._invoke(wrapper, dfr, first_kwargs, user_args=user_args)
    if num_frames is None:
        num_frames = ((int(first.frames.shape[2]) - 1) *
                      _ltx_temporal_compression(pipe) + 1)

    upscaled = _upsample_ltx_latents(
        wrapper, pipe, first.frames, user_args, use_custom_uri=False)
    second_kwargs = {
        'prompt': kwargs.get('prompt'),
        'conditions': kwargs.get('conditions'),
        'height': height,
        'width': width,
        'num_frames': num_frames,
        'frame_rate': fps,
        'latents': upscaled,
        'audio_latents': first.audio,
        'keyframes_latents': first.keyframes,
        'keyframe_positions': first.keyframe_positions,
        'reference_latents': first.frames,
        'reference_downscale_factor': 2,
        'sigmas': _stage_sigmas(user_args),
        'noise_scale': user_args.ltx_noise_scale,
        'generator': kwargs.get('generator'),
        'use_cross_timestep': kwargs.get('use_cross_timestep', True),
        'output_type': 'latent',
        'return_dict': True,
        'max_sequence_length': kwargs.get('max_sequence_length', 1024),
    }
    second_kwargs = dfr_kwargs(second_kwargs)
    if second_kwargs.get('noise_scale') is None:
        second_kwargs.pop('noise_scale')
    second = _vp._invoke(wrapper, dfr, second_kwargs, user_args=user_args)

    temporal_upsampler = _load_ltx_latent_upsampler(wrapper, user_args)
    temporal = LTX2DFRTemporalRefinePipeline.from_pipe(
        dfr,
        temporal_latent_upsampler=temporal_upsampler,
        scheduler=LTXEulerAncestralRFScheduler(eta=0.5),
        dtype=dtype)
    if (getattr(temporal_upsampler.config, 'spatial_upsample', None) is not False):
        raise _pipelines.UnsupportedPipelineConfigError(
            'Temporal mode needs a temporal-only latent upsampler with '
            'spatial_upsample=false; select subfolder=temporal_latent_upsampler.')
    temporal_upsampler.to(device=wrapper.device, dtype=dtype)
    output = second
    canvas_frames = (int(second.frames.shape[2]) - 1) * _ltx_temporal_compression(pipe) + 1
    source_seconds = num_frames / fps
    try:
        for round_index in range(1, int(user_args.ltx_temporal_upscale_rounds) + 1):
            refine_kwargs = {
                'prompt': kwargs.get('prompt'),
                'conditions': kwargs.get('conditions'),
                'latents': output.frames,
                'keyframes_latents': output.keyframes,
                'keyframe_positions': output.keyframe_positions,
                'audio_latents': output.audio,
                'height': height,
                'width': width,
                'num_frames': canvas_frames,
                'frame_rate': fps,
                'source_seconds': source_seconds,
                'condition_num_frames': num_frames,
                'round_index': round_index,
                'sigmas': list(TEMPORAL_ROUND_DISTILLED_SIGMA_VALUES),
                'generator': kwargs.get('generator'),
                'use_cross_timestep': kwargs.get('use_cross_timestep', True),
                'output_type': 'latent',
                'return_dict': True,
                'max_sequence_length': kwargs.get('max_sequence_length', 1024),
            }
            refine_kwargs = {
                key: value for key, value in refine_kwargs.items()
                if key in inspect.signature(temporal.__call__).parameters}
            _messages.log(f'LTX temporal DFR round {round_index}.')
            output = _vp._invoke(wrapper, temporal, refine_kwargs, user_args=user_args)
            canvas_frames = 2 * (canvas_frames - 1) + 1
            fps *= 2
    finally:
        if _vp._offload_requested(wrapper):
            temporal_upsampler.to('cpu')

    if diffusion:
        result = _ltx_pixels_from_output(
            wrapper, pipe, held, output, True, kwargs.get('generator'),
            getattr(user_args, 'ltx_diffusion_decode_tile', None),
            getattr(user_args, 'ltx_diffusion_decoder', None))
    else:
        latent = output.frames
        normalized = pipe._normalize_latents(
            latent.float(), pipe.vae.latents_mean, pipe.vae.latents_std,
            pipe.vae.config.scaling_factor)
        timestep = None
        if pipe.vae.config.timestep_conditioning:
            timestep = torch.full((latent.shape[0],), float(user_args.ltx_decode_timestep or 0.0),
                                  device=latent.device, dtype=latent.dtype)
        decoded = pipe.vae.decode(
            normalized.to(pipe.vae.dtype), timestep, return_dict=False)[0]
        frames = pipe.video_processor.postprocess_video(decoded, output_type='pil')
        audio = _waveform_from_audio_latents(pipe, output.audio)
        result = (frames, audio, audio_sample_rate_from_pipeline(pipe, audio))
    return result


def _video_latent_tensor(frames):
    if isinstance(frames, (list, tuple)):
        frames = frames[0]
    if not torch.is_tensor(frames):
        raise _pipelines.UnsupportedPipelineConfigError(
            'LTX stage 1 did not return video latents.')
    return frames


def _stage_sigmas(user_args):
    from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

    base = [float(value) for value in STAGE_2_DISTILLED_SIGMA_VALUES]
    if user_args.ltx_stage_sigmas is None:
        return base
    if isinstance(user_args.ltx_stage_sigmas, str):
        return _eval_sigma_expression(user_args.ltx_stage_sigmas, base)
    return [float(value) for value in user_args.ltx_stage_sigmas]


def _ltx_repo_component_files(wrapper, repo_id: str, subfolder: str,
                              revision: str | None,
                              variant: str | None = None) -> set[str]:
    """List component filenames from a local HF cache or Hub metadata only."""
    import huggingface_hub

    local_files_only = bool(wrapper._local_files_only) or _hfhub.is_offline_mode()
    config_name = os.path.join(subfolder, 'config.json').replace('\\', '/')
    cached_config = huggingface_hub.try_to_load_from_cache(
        repo_id, config_name, revision=revision)
    if isinstance(cached_config, str):
        component_dir = os.path.dirname(cached_config)
        cached_names = set(os.listdir(component_dir))
        if local_files_only or _ltx_component_has_weights(cached_names, variant):
            return cached_names

    if local_files_only:
        return set()

    # list_repo_files only queries repository metadata; it does not download
    # model weights. Strip the component prefix to make local and Hub results
    # use the same filename checks below.
    filenames = huggingface_hub.HfApi(token=wrapper._auth_token).list_repo_files(
        repo_id, revision=revision)
    prefix = subfolder.strip('/\\')
    if not prefix:
        return {os.path.basename(name) for name in filenames}
    prefix = prefix.replace('\\', '/') + '/'
    return {
        name[len(prefix):]
        for name in filenames
        if name.replace('\\', '/').startswith(prefix)
    }


def _ltx_component_has_weights(filenames: set[str], variant: str | None) -> bool:
    """Check for Diffusers single-file or complete indexed component weights."""
    base_names = {
        'diffusion_pytorch_model.safetensors',
        'diffusion_pytorch_model.bin',
    }
    if variant:
        base_names = {
            'diffusion_pytorch_model.' + variant + '.safetensors',
            'diffusion_pytorch_model.' + variant + '.bin',
        }

    available = set()
    for filename in filenames:
        name = os.path.basename(filename)
        if name in base_names:
            return True
        if not _putil.variant_match(filename, variant):
            continue
        if name.endswith(('.safetensors', '.bin')) and name.startswith('diffusion_pytorch_model'):
            available.add(name)

    # Sharded Diffusers models have an index plus numbered weight files.
    index_names = {
        'diffusion_pytorch_model.safetensors.index.json',
        'diffusion_pytorch_model.bin.index.json',
    }
    if variant:
        index_names |= {
            'diffusion_pytorch_model.safetensors.index.' + variant + '.json',
            'diffusion_pytorch_model.bin.index.' + variant + '.json',
            'diffusion_pytorch_model.safetensors.' + variant + '.index.json',
            'diffusion_pytorch_model.bin.' + variant + '.index.json',
        }
    indexes = [name for name in filenames if os.path.basename(name) in index_names]
    shards = [name for name in available if '-of-' in name]
    if not indexes or not shards:
        return False

    groups = {}
    for shard in shards:
        match = re.search(r'-(\d+)-of-(\d+)\.(?:safetensors|bin)$', shard)
        if match:
            index, total = map(int, match.groups())
            groups.setdefault((total, os.path.splitext(shard)[1]), set()).add(index)
    return any(len(indices) == total and indices == set(range(1, total + 1))
               for (total, _extension), indices in groups.items())


def _ltx_component_files(wrapper, model_path: str, subfolder: str,
                         revision: str | None, variant: str | None,
                         allow_single_file: bool = False):
    """Return (filenames, is_single_file) without loading model weights."""
    if os.path.isdir(model_path):
        component_dir = os.path.join(model_path, subfolder)
        if not os.path.isdir(component_dir):
            return set(), False
        return set(os.listdir(component_dir)), False

    if os.path.isfile(model_path):
        return ({os.path.basename(model_path)}, True) if allow_single_file else (set(), False)

    blob = _hfhub.HFBlobLink.parse(model_path)
    if blob is not None:
        if not allow_single_file:
            return set(), False
        import huggingface_hub
        filename = os.path.join(blob.subfolder or '', blob.weight_name).replace('\\', '/')
        cached_file = huggingface_hub.try_to_load_from_cache(
            blob.repo_id, filename, revision=blob.revision)
        if isinstance(cached_file, str):
            return {blob.weight_name}, True
        if bool(wrapper._local_files_only) or _hfhub.is_offline_mode():
            return set(), True
        filenames = huggingface_hub.HfApi(token=wrapper._auth_token).list_repo_files(
            blob.repo_id, revision=blob.revision)
        return ({blob.weight_name} if filename in filenames else set()), True

    if model_path.startswith(('http://', 'https://')):
        if not allow_single_file:
            return set(), False
        # Probe a direct URL with HEAD only. Some hosts do not support HEAD;
        # in that case leave the decision to the normal loader.
        if bool(wrapper._local_files_only) or _hfhub.is_offline_mode():
            return set(), True
        import urllib.error
        import urllib.request
        try:
            request = urllib.request.Request(model_path, method='HEAD')
            with urllib.request.urlopen(request, timeout=15) as response:
                return ({os.path.basename(model_path)} if 200 <= response.status < 400 else set()), True
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return set(), True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        return None

    if _hfhub.is_single_file_model_load(model_path):
        return (set(), True) if allow_single_file else (set(), False)

    return _ltx_repo_component_files(wrapper, model_path, subfolder, revision, variant), False


def _check_ltx_component_available(wrapper, model_path: str, component: str,
                                   display_name: str, subfolder: str,
                                   revision: str | None = None,
                                   variant: str | None = None,
                                   allow_single_file: bool = False,
                                   config_name: str = 'config.json') -> None:
    files = _ltx_component_files(
        wrapper, model_path, subfolder, revision, variant, allow_single_file)
    if files is None:
        return
    filenames, single_file = files
    available = bool(filenames) if single_file else (
        config_name in filenames and _ltx_component_has_weights(filenames, variant))
    if available:
        return

    if model_path == wrapper.model_path:
        if bool(wrapper._local_files_only) or _hfhub.is_offline_mode():
            message = (f'The {display_name} in "{model_path}/{subfolder}" is not available '
                       'in the local cache.')
        else:
            message = (f'The LTX checkpoint at "{model_path}" does not contain an available '
                       f'{display_name} in "{subfolder}".')
    else:
        message = (f'The {display_name} model "{model_path}" is missing its config or '
                   'Diffusers weight files.')
    if component == 'diffusion_decoder':
        message += ' Use --ltx-video-decoder conv to use the convolutional VAE instead.'
    elif component == 'latent_upsampler':
        message += ' Ensure the selected model is an LTX latent upsampler.'
    raise _pipelines.UnsupportedPipelineConfigError(message)


def _check_ltx_latent_upsampler_available(wrapper, user_args, temporal: bool = False) -> None:
    model_path = wrapper.model_path
    subfolder = 'temporal_latent_upsampler' if temporal else 'latent_upsampler'
    revision = None
    variant = None
    if isinstance(getattr(user_args, 'ltx_latent_upscale', None), str):
        uri = _uris.LTXLatentUpsamplerUri.parse(user_args.ltx_latent_upscale)
        model_path = uri.model
        revision = uri.revision
        variant = uri.variant
        if uri.subfolder:
            subfolder = uri.subfolder
        if _hfhub.is_single_file_model_load(model_path):
            subfolder = ''

    _check_ltx_component_available(
        wrapper, model_path, 'latent_upsampler', 'latent upsampler',
        subfolder, revision, variant, _hfhub.is_single_file_model_load(model_path))

    config = _ltx_latent_upsampler_config(wrapper, model_path, subfolder, revision)
    if config is not None and config.get('spatial_upsample') is not (not temporal):
        actual = 'spatial' if config.get('spatial_upsample') else 'temporal-only'
        raise _pipelines.UnsupportedPipelineConfigError(
            f'The latent upsampler in "{model_path}/{subfolder}" is {actual}. '
            f'--ltx-latent-upscale-mode {"temporal" if temporal else "spatial"} requires a '
            f'component with spatial_upsample={not temporal}.')


def _check_ltx_diffusion_decoder_available(wrapper, user_args) -> None:
    model_path = wrapper.model_path
    revision = None
    subfolder = 'diffusion_decoder'
    if user_args.ltx_diffusion_decoder:
        uri = _uris.LTXDiffusionDecoderUri.parse(user_args.ltx_diffusion_decoder)
        model_path = uri.model
        revision = uri.revision
        if uri.subfolder:
            subfolder = uri.subfolder
    _check_ltx_component_available(
        wrapper, model_path, 'diffusion_decoder', 'diffusion decoder',
        subfolder, revision=revision, config_name='config.json')


def _ltx_latent_upsampler_config(wrapper, model_path: str, subfolder: str,
                                  revision: str | None) -> dict | None:
    """Read a latent upsampler config without loading its weights."""
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, subfolder, 'config.json')
        if not os.path.isfile(config_path):
            return None
    else:
        # Single-file checkpoints may not have a separate Diffusers config.
        if os.path.isfile(model_path) or _hfhub.HFBlobLink.parse(model_path) is not None \
                or _hfhub.is_single_file_model_load(model_path):
            return None

        import huggingface_hub

        filename = os.path.join(subfolder, 'config.json').replace('\\', '/')
        config_path = huggingface_hub.try_to_load_from_cache(
            model_path, filename, revision=revision)
        local_files_only = bool(wrapper._local_files_only) or _hfhub.is_offline_mode()
        if not isinstance(config_path, str):
            if local_files_only:
                return None
            config_path = huggingface_hub.hf_hub_download(
                model_path,
                filename=filename,
                revision=revision,
                token=wrapper._auth_token,
                local_files_only=False)

    with open(config_path, encoding='utf-8') as config_file:
        return json.load(config_file)


def _preflight_ltx_components(wrapper, user_args) -> None:
    """Check requested optional LTX components using repository metadata only."""
    if not user_args.ltx_latent_upscale and user_args.ltx_video_decoder != 'diffusion':
        return

    if user_args.ltx_latent_upscale:
        width, height = _vp._size(user_args)
        if width is None or height is None:
            raise _pipelines.UnsupportedPipelineConfigError(
                '--ltx-latent-upscale needs an --output-size. '
                'That size is the finished clip.')

    index = _putil.fetch_model_index_dict(
        wrapper.model_path,
        subfolder=wrapper._subfolder,
        revision=wrapper._revision,
        use_auth_token=wrapper._auth_token,
        local_files_only=bool(wrapper._local_files_only))
    family = ltx_family_from_index(index)

    # Report unsupported LTX-2-only options before probing for components that
    # cannot be used by an earlier LTX checkpoint. The normal call path checks
    # these again after loading the pipeline for options that do not require
    # preflight.
    if family != 'ltx2':
        _reject_ltx2_only_options(user_args)

    if user_args.ltx_latent_upscale:
        temporal = user_args.ltx_latent_upscale_mode == 'temporal'
        _check_ltx_latent_upsampler_available(wrapper, user_args, temporal=temporal)
        if temporal:
            _check_ltx_component_available(
                wrapper, wrapper.model_path, 'latent_upsampler', 'spatial latent upsampler',
                subfolder='latent_upsampler')
    if family == 'ltx2' and user_args.ltx_video_decoder == 'diffusion':
        _check_ltx_diffusion_decoder_available(wrapper, user_args)


def _is_missing_ltx_component(error, model_path: str, component: str, config_name: str) -> bool:
    """Return whether Diffusers could not find the component's config file."""
    if os.path.isdir(model_path):
        # Diffusers falls back to the root config when a local subfolder config
        # is missing. Check the requested component directly to avoid treating
        # the main pipeline config as the component config.
        return not os.path.isfile(os.path.join(model_path, component, config_name))

    if error is None:
        return False

    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))

        message = str(current).lower()
        missing_config = config_name.lower()
        if (f'does not appear to have a file named {missing_config}' in message
                or f'error no file named {missing_config} found in directory' in message):
            return True

        pending.extend((current.__cause__, current.__context__))
    return False


def _load_ltx_latent_upsampler(wrapper, user_args, use_custom_uri=True):
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

    dtype = _enums.get_torch_dtype(wrapper._dtype) or torch.bfloat16
    model_path = wrapper.model_path
    load_kwargs = {
        'subfolder': ('temporal_latent_upsampler'
                      if getattr(user_args, 'ltx_latent_upscale_mode', 'spatial') == 'temporal'
                      else 'latent_upsampler'),
        'torch_dtype': dtype,
        'token': wrapper._auth_token,
        'local_files_only': bool(wrapper._local_files_only),
    }

    using_custom_upsampler = (use_custom_uri and
                              isinstance(getattr(user_args, 'ltx_latent_upscale', None), str))
    if using_custom_upsampler:
        uri = _uris.LTXLatentUpsamplerUri.parse(user_args.ltx_latent_upscale)
        model_path = uri.model
        if uri.revision is not None:
            load_kwargs['revision'] = uri.revision
        if uri.variant is not None:
            load_kwargs['variant'] = uri.variant
        if uri.subfolder:
            load_kwargs['subfolder'] = uri.subfolder
        if _hfhub.is_single_file_model_load(model_path):
            load_kwargs.pop('subfolder', None)

    if (not using_custom_upsampler
            and _is_missing_ltx_component(None, model_path, 'latent_upsampler',
                                          LTX2LatentUpsamplerModel.config_name)):
        raise _pipelines.UnsupportedPipelineConfigError(
            f'The LTX checkpoint at "{wrapper.model_path}" does not contain a latent upsampler. '
            'Ensure you are using an LTX-2 checkpoint that supports latent upscaling.')

    try:
        if _hfhub.is_single_file_model_load(model_path):
            from_single_file = getattr(LTX2LatentUpsamplerModel, 'from_single_file', None)
            if from_single_file is None:
                from diffusers.loaders.single_file_model import FromOriginalModelMixin
                from_single_file = FromOriginalModelMixin.from_single_file.__get__(
                    LTX2LatentUpsamplerModel, LTX2LatentUpsamplerModel)
            return from_single_file(model_path, **load_kwargs)
        return LTX2LatentUpsamplerModel.from_pretrained(model_path, **load_kwargs)
    except Exception as e:
        # Preserve the loader's original error for a separately selected model;
        # it may indicate authentication, revision, or weight format problems.
        if (not using_custom_upsampler
                and _is_missing_ltx_component(
                    e, model_path, 'latent_upsampler', LTX2LatentUpsamplerModel.config_name)):
            raise _pipelines.UnsupportedPipelineConfigError(
                f'The LTX checkpoint at "{wrapper.model_path}" does not contain a latent upsampler. '
                'Ensure you are using an LTX-2 checkpoint that supports latent upscaling.') from e
        raise


def _upsample_ltx_latents(wrapper, pipe, video_latent, user_args, use_custom_uri=True):
    from diffusers.pipelines.ltx2.pipeline_ltx2_latent_upsample import LTX2LatentUpsamplePipeline

    # Keep model weights out of memory until stage 1 is complete.
    upsampler = _load_ltx_latent_upsampler(wrapper, user_args, use_custom_uri=use_custom_uri)
    dtype = _enums.get_torch_dtype(wrapper._dtype) or torch.bfloat16
    up_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=upsampler)
    # Sequential offload stores this model's weights as meta tensors. The
    # upsample forward then cannot copy them onto the GPU. The upsampler is
    # small next to the transformer, so it stays on the compute device for
    # this call and is moved off again afterward. The shared VAE is left on
    # the main pipeline's offload hooks; this call does not decode with it.
    device = wrapper.device
    upsampler.to(device=device, dtype=dtype)
    if torch.is_tensor(video_latent):
        video_latent = video_latent.to(device=device)
    offload = _vp._offload_requested(wrapper)
    try:
        upscaled = up_pipe(
            latents=video_latent,
            latents_normalized=False,
            output_type='latent',
            return_dict=False)[0]
    finally:
        if offload:
            upsampler.to('cpu')
    return upscaled


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


def _ltx_pixels_from_output(wrapper, pipe, held, output, diffusion, generator,
                            decode_tile=None, diffusion_decoder_uri=None):
    video = output.frames
    audio_raw = getattr(output, 'audio', None)
    if diffusion:
        if isinstance(video, (list, tuple)):
            video = video[0]
        _park_ltx_for_diffusion_decode(pipe, wrapper)
        try:
            frames = _decode_ltx_diffusion(
                wrapper, pipe, held, video, generator, decode_tile, diffusion_decoder_uri)
            audio = _waveform_from_audio_latents(pipe, audio_raw)
        finally:
            _park_ltx_after_diffusion_decode(
                pipe, getattr(held, 'diffusion_decoder_pipe', None), wrapper)
    else:
        frames = _vp.frames_from_output(video)
        audio = _vp.audio_to_numpy(audio_raw)
    return frames, audio, audio_sample_rate_from_pipeline(pipe, audio)


def _waveform_from_audio_latents(pipe, audio_latents):
    if audio_latents is None:
        return None
    if not torch.is_tensor(audio_latents):
        return _vp.audio_to_numpy(audio_latents)
    mel = pipe.audio_vae.decode(
        audio_latents.to(dtype=pipe.audio_vae.dtype), return_dict=False)[0]
    return _vp.audio_to_numpy(pipe.vocoder(mel))


# Bytes of gathered keys or values kept live for one neighborhood chunk.
# Stage 5 attends to an 11x11x11 cube, so the chunk stays well under a gigabyte.
_NEIGHBORHOOD_BUDGET_BYTES = 256 * 1024 * 1024
_TRITON_NEIGHBORHOOD_KERNEL = None
_triton_neighborhood_failed = False


def _neighborhood_budget_bytes() -> int:
    return _NEIGHBORHOOD_BUDGET_BYTES


def _neighborhood_query_chunk(tokens, heads, dim, element_size, window_volume):
    bytes_per_query = max(window_volume * heads * dim * element_size, dim * element_size)
    return max(1, min(tokens, _neighborhood_budget_bytes() // max(bytes_per_query, 1)))


def _triton_neighborhood_ready() -> bool:
    """True when this process can compile a CUDA Triton neighborhood kernel."""
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _neighborhood_kernel():
    """Compile the Triton neighborhood kernel once."""
    global _TRITON_NEIGHBORHOOD_KERNEL
    if _TRITON_NEIGHBORHOOD_KERNEL is not None:
        return _TRITON_NEIGHBORHOOD_KERNEL

    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
            query_ptr, key_ptr, value_ptr, output_ptr,
            frames, height, width, dim,
            kernel_t, kernel_h, kernel_w,
            token_offset, chunk_tokens,
            stride_bh, stride_token,
            BLOCK_D: tl.constexpr,
            OUT_DTYPE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        plane = height * width
        local_token = pid % chunk_tokens
        token = token_offset + local_token
        group = pid // chunk_tokens
        frame = token // plane
        remainder = token - frame * plane
        row = remainder // width
        column = remainder - row * width
        start_frame = tl.minimum(tl.maximum(frame - (kernel_t // 2), 0), frames - kernel_t)
        start_row = tl.minimum(tl.maximum(row - (kernel_h // 2), 0), height - kernel_h)
        start_column = tl.minimum(tl.maximum(column - (kernel_w // 2), 0), width - kernel_w)

        offsets = tl.arange(0, BLOCK_D)
        mask = offsets < dim
        query = tl.load(
            query_ptr + group * stride_bh + token * stride_token + offsets,
            mask=mask, other=0.0).to(tl.float32)
        maximum = tl.full([1], -1.0e30, dtype=tl.float32)
        normalizer = tl.zeros([1], dtype=tl.float32)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)

        for frame_offset in range(kernel_t):
            for row_offset in range(kernel_h):
                for column_offset in range(kernel_w):
                    neighbor = (
                        (start_frame + frame_offset) * plane
                        + (start_row + row_offset) * width
                        + (start_column + column_offset))
                    base = group * stride_bh + neighbor * stride_token
                    key = tl.load(key_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
                    value = tl.load(value_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
                    score = tl.sum(query * key, axis=0)
                    next_max = tl.maximum(maximum, score)
                    alpha = tl.exp(maximum - next_max)
                    weight = tl.exp(score - next_max)
                    normalizer = normalizer * alpha + weight
                    accumulator = accumulator * alpha + value * weight
                    maximum = next_max

        tl.store(
            output_ptr + group * stride_bh + token * stride_token + offsets,
            (accumulator / normalizer).to(OUT_DTYPE),
            mask=mask)

    _TRITON_NEIGHBORHOOD_KERNEL = kernel
    return kernel


def _triton_dtype(dtype):
    import triton.language as tl
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    return tl.float32


def _grouped_kv(key, value, batch, heads, tokens, dim):
    grouped_key = key.permute(0, 4, 1, 2, 3, 5).contiguous().view(batch * heads, tokens, dim)
    grouped_value = value.permute(0, 4, 1, 2, 3, 5).contiguous().view(batch * heads, tokens, dim)
    return grouped_key, grouped_value


def _neighborhood_attention_triton(query, key, value, kernel_t, kernel_h, kernel_w):
    """One program per query. The window matches NATTEN's inward shift."""
    import triton

    batch, frames, height, width, heads, dim = query.shape
    tokens = frames * height * width
    volume = kernel_t * kernel_h * kernel_w
    grouped_query = query.permute(0, 4, 1, 2, 3, 5).contiguous().view(batch * heads, tokens, dim)
    grouped_key, grouped_value = _grouped_kv(key, value, batch, heads, tokens, dim)
    output = torch.empty_like(grouped_query)
    chunk = _neighborhood_query_chunk(
        tokens, heads, dim, query.element_size(), volume)
    kernel = _neighborhood_kernel()
    block_d = triton.next_power_of_2(dim)
    out_dtype = _triton_dtype(query.dtype)
    stride_bh = grouped_query.stride(0)
    stride_token = grouped_query.stride(1)
    for start in range(0, tokens, chunk):
        stop = min(start + chunk, tokens)
        chunk_tokens = stop - start
        kernel[(batch * heads * chunk_tokens,)](
            grouped_query, grouped_key, grouped_value, output,
            frames, height, width, dim,
            kernel_t, kernel_h, kernel_w,
            start, chunk_tokens,
            stride_bh, stride_token,
            BLOCK_D=block_d,
            OUT_DTYPE=out_dtype,
            num_warps=1,
        )
    return output.view(batch, heads, frames, height, width, dim).permute(0, 2, 3, 4, 1, 5).contiguous()


def _neighborhood_attention_gather(query, key, value, kernel_t, kernel_h, kernel_w):
    """PyTorch gather used where Triton cannot run."""
    batch, frames, height, width, heads, dim = query.shape
    tokens = frames * height * width
    plane = height * width
    query_flat = query.reshape(batch, tokens, heads, dim)
    key_flat = key.reshape(batch, tokens, heads, dim)
    value_flat = value.reshape(batch, tokens, heads, dim)
    output = torch.empty_like(query_flat)

    volume = kernel_t * kernel_h * kernel_w
    chunk = _neighborhood_query_chunk(
        tokens, heads, dim, query.element_size(), volume)
    device = query.device
    offset_t = torch.arange(kernel_t, device=device)
    offset_h = torch.arange(kernel_h, device=device)
    offset_w = torch.arange(kernel_w, device=device)

    for start in range(0, tokens, chunk):
        stop = min(start + chunk, tokens)
        index = torch.arange(start, stop, device=device)
        query_t = index // plane
        remainder = index - query_t * plane
        query_h = remainder // width
        query_w = remainder - query_h * width
        start_t = torch.clamp(query_t - (kernel_t // 2), min=0, max=frames - kernel_t)
        start_h = torch.clamp(query_h - (kernel_h // 2), min=0, max=height - kernel_h)
        start_w = torch.clamp(query_w - (kernel_w // 2), min=0, max=width - kernel_w)
        neighbor_t = start_t[:, None, None, None] + offset_t[None, :, None, None]
        neighbor_h = start_h[:, None, None, None] + offset_h[None, None, :, None]
        neighbor_w = start_w[:, None, None, None] + offset_w[None, None, None, :]
        linear = (neighbor_t * plane + neighbor_h * width + neighbor_w).reshape(stop - start, volume)

        gathered_key = key_flat[:, linear]
        gathered_value = value_flat[:, linear]
        query_chunk = query_flat[:, start:stop]
        scores = torch.einsum('bnhd,bnkhd->bnhk', query_chunk, gathered_key)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=gathered_value.dtype)
        output[:, start:stop] = torch.einsum('bnhk,bnkhd->bnhd', weights, gathered_value)

    return output.reshape(batch, frames, height, width, heads, dim)


def _neighborhood_attention(query, key, value, kernel_size):
    """
    Neighborhood attention with NATTEN's inward-shifted window.

    ``query``, ``key``, and ``value`` are ``(batch, frames, height, width, heads, dim)``.
    Each query attends to a cube of ``kernel_size`` that slides inward at the border
    so the cube always stays full. CUDA uses a Triton kernel. Other devices gather
    the cube in PyTorch. Neither path builds the FlexAttention mask.
    """
    global _triton_neighborhood_failed
    if query.ndim != 6:
        raise ValueError(
            'Neighborhood attention expected query shaped '
            f'(batch, frames, height, width, heads, dim), got {tuple(query.shape)}.')
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    batch, frames, height, width, heads, dim = query.shape
    kernel_t, kernel_h, kernel_w = (int(part) for part in kernel_size)
    kernel_t = min(max(kernel_t, 1), frames)
    kernel_h = min(max(kernel_h, 1), height)
    kernel_w = min(max(kernel_w, 1), width)
    if (query.is_cuda and dim <= 128 and heads > 0
            and not _triton_neighborhood_failed and _triton_neighborhood_ready()):
        try:
            return _neighborhood_attention_triton(
                query, key, value, kernel_t, kernel_h, kernel_w)
        except Exception as error:
            _triton_neighborhood_failed = True
            _messages.log(
                'Triton neighborhood attention failed '
                f'({error}). Gathering the window in PyTorch.')
    return _neighborhood_attention_gather(
        query, key, value, kernel_t, kernel_h, kernel_w)


class _LTXWindowNeighborhoodProcessor:
    """
    LTX diffusion-decoder attention when the NATTEN Hub kernel cannot load.

    The call shape matches ``LTX2VideoVaeNeighborhoodNattenProcessor``. CUDA
    runs the window with Triton. The FlexAttention processor is not used:
    its mask does not fit in GPU memory.
    """

    tiles = True

    def __call__(self, attn, hidden_states, block_mask=None):
        batch_size, num_frames, height, width, channels = hidden_states.shape
        query, key, value = attn.project_qkv(hidden_states)
        attended = _neighborhood_attention(query, key, value, attn.kernel_size)
        attended = attended.reshape(batch_size, num_frames, height, width, channels)
        return attn.to_out[0](attended)


def _ltx_diffusion_natten_processor():
    """
    Attention processor for the LTX diffusion decoder.

    Linux uses the ``shi-labs/natten`` kernel from the Hub. Windows has no
    build of that kernel. The same window then runs with Triton, or is
    gathered in PyTorch when Triton is not installed.
    """
    try:
        from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
            LTX2VideoVaeNeighborhoodNattenProcessor)
        return LTX2VideoVaeNeighborhoodNattenProcessor()
    except Exception as error:
        if _triton_neighborhood_ready():
            path = 'Running that window with Triton.'
        else:
            path = 'Gathering that window in PyTorch.'
        _messages.log(
            f'LTX diffusion decoder has no NATTEN build for this system. {path}')
        _messages.debug_log(error)
        return _LTXWindowNeighborhoodProcessor()


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
        _vp._offload_ltx(
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


def _ltx_temporal_compression(pipe) -> int:
    return int(getattr(pipe, 'vae_temporal_compression_ratio', 8) or 8)


def _trim_ltx_clip(frames: list, limit: int | None, temporal: int) -> list:
    count = len(frames) if limit is None else min(len(frames), limit)
    count = (count - 1) // temporal * temporal + 1
    if count < 1:
        raise _pipelines.UnsupportedPipelineConfigError(
            'The LTX output is too short to hold the conditioning clips. '
            'Use a longer --video-lengths.')
    return frames[:count]


def _ltx_conditions(pipe, family: str, user_args, num_frames: int | None) -> list:
    """
    Build the condition list for the LTX condition pipeline.

    Leading media starts at frame 0 and trailing media ends on the last frame.
    Clips are cut to ``8k+1`` frames, and the leading clip is shortened so the
    two never overlap.
    """
    temporal = _ltx_temporal_compression(pipe)
    start_image = user_args.images[0] if user_args.images else None
    end_image = user_args.end_images[0] if user_args.end_images else None
    start_video = user_args.video_frames or None
    end_video = user_args.end_video_frames or None
    has_start = start_image is not None or start_video is not None

    if num_frames is None:
        if family == 'ltx':
            num_frames = 161
        elif getattr(pipe, 'duration_head', None) is None:
            num_frames = 121
        elif end_video is not None:
            raise _pipelines.UnsupportedPipelineConfigError(
                'An LTX end clip needs a fixed output length. This checkpoint picks its '
                'length from the prompt, so set --video-lengths.')

    end_clip = None
    end_count = 0
    if end_video is not None:
        end_clip = _trim_ltx_clip(
            end_video, num_frames - (1 if has_start else 0), temporal)
        end_count = len(end_clip)
    elif end_image is not None:
        end_count = 1

    start_clip = None
    if start_video is not None:
        start_clip = _trim_ltx_clip(
            start_video, None if num_frames is None else num_frames - end_count, temporal)

    for name, clip in (('start', start_clip), ('end', end_clip)):
        if clip is not None:
            _messages.debug_log(f'LTX {name} conditioning clip: {len(clip)} frames.')

    start_index = 0 if user_args.ltx_condition_index is None else int(user_args.ltx_condition_index)
    start_strength = _ltx_resolved_strength(user_args.ltx_condition_strength, user_args)

    conditions = []
    if family == 'ltx':
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

        if start_clip is not None:
            conditions.append(LTXVideoCondition(
                video=start_clip, frame_index=start_index, strength=start_strength))
        elif start_image is not None:
            conditions.append(LTXVideoCondition(
                image=start_image, frame_index=start_index, strength=start_strength))
        if end_clip is not None:
            conditions.append(LTXVideoCondition(
                video=end_clip, frame_index=num_frames - end_count, strength=1.0))
        elif end_image is not None:
            conditions.append(LTXVideoCondition(
                image=end_image, frame_index=num_frames - 1, strength=1.0))
        _append_extra_ltx_conditions(
            conditions, family, user_args.ltx_extra_conditions, LTXVideoCondition, user_args)
        return conditions

    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

    if start_clip is not None:
        conditions.append(LTX2VideoCondition(
            frames=start_clip, index=start_index, strength=start_strength))
    elif start_image is not None:
        conditions.append(LTX2VideoCondition(
            frames=start_image, index=start_index, strength=start_strength))
    if end_clip is not None:
        # LTX-2 indices are latent frames, and each latent after the first covers ``temporal`` pixels.
        latent_frames = (num_frames - 1) // temporal + 1
        clip_latents = (end_count - 1) // temporal + 1
        conditions.append(LTX2VideoCondition(
            frames=end_clip, index=latent_frames - clip_latents, strength=1.0))
    elif end_image is not None:
        conditions.append(LTX2VideoCondition(frames=end_image, index=-1, strength=1.0))
    _append_extra_ltx_conditions(
        conditions, family, user_args.ltx_extra_conditions, LTX2VideoCondition, user_args)
    return conditions


def _ltx_image_needs_conditions(user_args) -> bool:
    if user_args.ltx_condition_index not in (None, 0):
        return True
    if user_args.ltx_extra_conditions:
        return True
    return _ltx_resolved_strength(user_args.ltx_condition_strength, user_args) != 1.0


def _ltx_resolved_strength(explicit, user_args) -> float:
    """
    Condition weight for one LTX image seed group.

    The image-seed keyword wins. ``--image-seed-strengths`` fills a group
    that omitted it. Otherwise the frame stays at full strength.
    """
    if explicit is not None:
        return float(explicit)
    if user_args.image_seed_strength is not None:
        return float(user_args.image_seed_strength)
    return 1.0


def _append_extra_ltx_conditions(conditions, family, extras, condition_cls, user_args):
    for frames, index, strength in extras or []:
        index = int(index)
        strength = _ltx_resolved_strength(strength, user_args)
        if family == 'ltx':
            if isinstance(frames, list):
                conditions.append(condition_cls(video=frames, frame_index=index, strength=strength))
            else:
                conditions.append(condition_cls(image=frames, frame_index=index, strength=strength))
        else:
            conditions.append(condition_cls(frames=frames, index=index, strength=strength))


def _ltx_reference_kwargs(wrapper, pipe, held, user_args, num_frames: int,
                          width: int | None, height: int | None) -> dict:
    """
    Build the IC-LoRA reference arguments for :py:class:`diffusers.LTX2InContextPipeline`.

    The reference is the image seed control clip. Its downscale factor comes from
    the ``--ltx-ic-lora`` URI, or else from the IC-LoRA file metadata.

    Frames are fit to the output canvas with the still-image ``--output-size``
    rule first. The pipeline then center-crops to the LoRA reference size, and a
    matching aspect means that crop only scales.
    """
    from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import LTX2ReferenceCondition

    factor = held.reference_downscale_factor or 1
    ic_lora = _vp._parsed_ic_lora(wrapper)
    attention = ic_lora.attention if ic_lora is not None else 1.0
    spatial = int(getattr(pipe, 'vae_spatial_compression_ratio', 32) or 32)

    clip = _trim_ltx_clip(
        user_args.reference_video_frames, num_frames, _ltx_temporal_compression(pipe))
    clip = _resize_ltx_control_clip(
        clip, width, height, user_args.aspect_correct, spatial * max(1, int(factor)))
    _messages.debug_log(
        f'LTX IC-LoRA reference clip: {len(clip)} frames, downscale factor {factor}, '
        f'attention {attention}.')
    return {
        'reference_conditions': [LTX2ReferenceCondition(frames=clip, strength=1.0)],
        'reference_downscale_factor': factor,
        'conditioning_attention_strength': attention,
    }


def _ltx_canvas_align(pipe, held, mode: str, user_args) -> int:
    spatial = int(getattr(pipe, 'vae_spatial_compression_ratio', 32) or 32)
    align = spatial
    if mode == 'ltx-control':
        factor = int(getattr(held, 'reference_downscale_factor', None) or 1)
        align = spatial * max(1, factor)
    if user_args.ltx_latent_upscale:
        # stage 1 runs at half size, which must still be spatially aligned
        align = math.lcm(align, spatial * 2)
    return align


def _fit_ltx_inputs(pipe, held, mode, user_args, width, height):
    """
    Fit every LTX conditioning image and clip to one canvas.

    The first available control clip, opening media, or end media chooses
    the canvas. The other slots are resized onto it.
    """
    primary = None
    for attr in ('reference_video_frames', 'video_frames', 'images',
                 'end_video_frames', 'end_images'):
        frames = getattr(user_args, attr, None)
        if frames:
            primary = frames[0].size
            break
    if primary is None:
        return width, height
    align = _ltx_canvas_align(pipe, held, mode, user_args)
    target_w, target_h = _vp.conditioning_canvas(
        primary, width, height, user_args.aspect_correct, align)
    target = (target_w, target_h)
    if primary != target:
        _messages.log(
            f'Resizing LTX conditioning media from {primary[0]}x{primary[1]} '
            f'to {target_w}x{target_h}.')
    for attr in ('reference_video_frames', 'video_frames', 'images',
                 'end_video_frames', 'end_images'):
        frames = getattr(user_args, attr, None)
        if frames:
            setattr(user_args, attr, _vp.resize_media(frames, target))
    extras = []
    for frames, index, strength in user_args.ltx_extra_conditions or []:
        if isinstance(frames, list):
            frames = _vp.resize_media(frames, target)
        else:
            frames = _vp.resize_media([frames], target)[0]
        extras.append((frames, index, strength))
    if extras:
        user_args.ltx_extra_conditions = extras
    user_args.width = target_w
    user_args.height = target_h
    return target_w, target_h


def _resize_ltx_control_clip(frames, width, height, aspect_correct, align: int):
    """Fit an IC-LoRA control clip with the still-image ``--output-size`` rule."""
    source = frames[0].size
    target_w, target_h = _vp.conditioning_canvas(
        source, width, height, aspect_correct, align)
    target = (target_w, target_h)
    if source == target:
        return frames
    _messages.log(
        f'Resizing LTX IC-LoRA control clip from {source[0]}x{source[1]} '
        f'to {target[0]}x{target[1]}.')
    return _vp.resize_media(frames, target)


def _ic_lora_downscale_factor(lora_uri: str, override: int | None,
                              auth_token, local_files_only) -> int:
    """
    Return the ``--ltx-ic-lora`` ``downscale`` value, or else ``reference_downscale_factor``
    from the IC-LoRA safetensors metadata. IC-LoRAs trained on reduced-size references
    store it there.
    """
    if override is not None:
        return override
    try:
        metadata = _lora_file_metadata(lora_uri, auth_token, local_files_only)
    except Exception as e:
        _messages.warning(
            f'Could not read the metadata of IC-LoRA "{lora_uri}", assuming reference '
            f'downscale factor 1. Set weight-name if the repository has several files, '
            f'or set downscale in the --ltx-ic-lora URI. Error: {e}')
        return 1
    value = metadata.get('reference_downscale_factor') if metadata else None
    if value is None:
        return 1
    try:
        factor = max(1, int(float(value)))
    except ValueError:
        _messages.warning(
            f'IC-LoRA "{lora_uri}" has an invalid reference_downscale_factor {value!r}, assuming 1.')
        return 1
    _messages.debug_log(f'IC-LoRA "{lora_uri}" sets reference downscale factor {factor}.')
    return factor


def _lora_file_metadata(uri, auth_token, local_files_only) -> dict | None:
    """
    Return the safetensors header metadata of a ``--loras`` URI, resolving the
    weight file the same way ``load_lora_weights`` does. ``None`` for non-safetensors files.
    """
    import safetensors
    from diffusers.loaders.lora_base import _best_guess_weight_name
    from diffusers.utils import _get_model_file

    lora = uri if isinstance(uri, _uris.LoRAUri) else _uris.LoRAUri.parse(uri)
    path = _hfhub.download_non_hf_slug_model(lora.model)

    if os.path.isfile(path):
        model_file = path
    else:
        weight_name = lora.weight_name
        if weight_name is None:
            search = path
            if os.path.isdir(path) and lora.subfolder:
                search = os.path.join(path, lora.subfolder)
            weight_name = _best_guess_weight_name(
                search, file_extension='.safetensors', local_files_only=local_files_only)
        if not weight_name:
            return None
        model_file = _get_model_file(
            path,
            weights_name=weight_name,
            subfolder=lora.subfolder,
            revision=lora.revision,
            token=auth_token,
            local_files_only=local_files_only)

    if not str(model_file).endswith('.safetensors'):
        return None
    with safetensors.safe_open(model_file, framework='pt') as f:
        return f.metadata()


def _legacy_ltx_schedule_finite(scheduler, width, height, num_frames, steps,
                                spatial=32, temporal=8) -> bool:
    """
    The earlier LTX pipeline shifts timesteps from the latent token count.
    Past a point that shift makes the terminal sigma 1 and the schedule NaN,
    which later crashes the sampler with an empty timestep index.
    """
    return bool(torch.isfinite(
        _legacy_ltx_timesteps(scheduler, width, height, num_frames, steps, spatial, temporal)).all())


def _legacy_ltx_timesteps(scheduler, width, height, num_frames, steps,
                          spatial=32, temporal=8) -> torch.Tensor:
    """
    The timesteps the earlier LTX text and image pipelines build for this clip.
    """
    cfg = scheduler.config
    latent_frames = (int(num_frames) - 1) // int(temporal) + 1
    seq = latent_frames * (int(height) // int(spatial)) * (int(width) // int(spatial))
    base_seq = cfg.get('base_image_seq_len', 256)
    max_seq = cfg.get('max_image_seq_len', 4096)
    base_shift = cfg.get('base_shift', 0.5)
    max_shift = cfg.get('max_shift', 1.15)
    slope = (max_shift - base_shift) / (max_seq - base_seq)
    mu = seq * slope + (base_shift - slope * base_seq)
    probe = scheduler.__class__.from_config(cfg)
    sigmas = numpy.linspace(1.0, 1.0 / int(steps), int(steps))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        probe.set_timesteps(int(steps), device='cpu', sigmas=sigmas, mu=float(mu))
    return probe.timesteps


def _legacy_ltx_compression(pipe) -> tuple[int, int]:
    spatial = int(getattr(pipe, 'vae_spatial_compression_ratio', 32) or 32)
    temporal = int(getattr(pipe, 'vae_temporal_compression_ratio', 8) or 8)
    return spatial, temporal


def _fix_legacy_ltx_condition_schedule(pipe, width, height, num_frames, steps) -> list[float] | None:
    """
    ``LTXConditionPipeline`` sets its own timesteps and never passes ``mu``, so a
    scheduler with dynamic shifting (the LTX-Video 0.9.0 / 0.9.1 checkpoints) raises.

    Resolve the checkpoint's shifted schedule here, then give ``pipe`` a copy of the
    scheduler that applies no further shift, so the timesteps are used as they are.
    ``pipe`` must be the per-call condition pipeline, not the cached one.

    :return: timesteps to pass, or ``None`` when the scheduler needs no change
    """
    scheduler = pipe.scheduler
    if not scheduler.config.get('use_dynamic_shifting', False):
        return None
    timesteps = _legacy_ltx_timesteps(
        scheduler, width, height, num_frames, steps, *_legacy_ltx_compression(pipe))
    pipe.scheduler = scheduler.__class__.from_config({
        **scheduler.config,
        'use_dynamic_shifting': False,
        'shift': 1.0,
        'shift_terminal': None,
        'use_karras_sigmas': False,
        'use_exponential_sigmas': False,
        'use_beta_sigmas': False,
    })
    return [float(value) for value in timesteps]


def _require_legacy_ltx_schedule(pipe, width, height, num_frames, steps):
    if _legacy_ltx_schedule_finite(
            pipe.scheduler, width, height, num_frames, steps, *_legacy_ltx_compression(pipe)):
        return
    raise _pipelines.UnsupportedPipelineConfigError(
        f'LTX-Video cannot build a noise schedule for {width}x{height} and {num_frames} frames. '
        f'The resolution-dependent timestep shift overflows at this size. '
        f'Use a smaller output size or a shorter clip.')


def _warn_legacy_ltx_canvas(width, height, num_frames, fps):
    short = num_frames is not None and int(num_frames) < 121
    small = (
        width is not None and height is not None
        and (int(width) < 704 or int(height) < 480)
    )
    if not short and not small:
        return
    _messages.warning(
        'LTX-Video follows the prompt at about 768x512, 25 fps, and 121 frames. '
        f'This clip is {width}x{height}, {num_frames} frames, {fps} fps.')


def _ltx_ignored_steps_warning(requested, scheduled: int) -> str | None:
    if requested is None:
        requested = _constants.DEFAULT_INFERENCE_STEPS
    requested = int(requested)
    if requested == scheduled:
        return None
    if requested == _constants.DEFAULT_INFERENCE_STEPS:
        passed = f'{requested} (the default)'
    else:
        passed = str(requested)
    return (
        f'The LTX distilled checkpoint uses an {scheduled}-value sigma schedule '
        f'and ignores --inference-steps {passed}.'
    )


def _ltx_base_sigmas(pipe, distilled: bool, inference_steps):
    if distilled:
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
        return list(DISTILLED_SIGMA_VALUES)
    steps = int(_types.default(inference_steps, _constants.DEFAULT_INFERENCE_STEPS))
    scheduler = pipe.scheduler
    old_timesteps = getattr(scheduler, 'timesteps', None)
    old_sigmas = getattr(scheduler, 'sigmas', None)
    old_num = getattr(scheduler, 'num_inference_steps', None)
    try:
        scheduler.set_timesteps(steps)
        return list(scheduler.sigmas)
    except Exception as e:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Could not read scheduler sigmas for an LTX sigma expression. '
            'Pass a CSV list with --sigmas instead.'
        ) from e
    finally:
        if old_timesteps is not None:
            scheduler.timesteps = old_timesteps
        if old_sigmas is not None:
            scheduler.sigmas = old_sigmas
        if old_num is not None:
            scheduler.num_inference_steps = old_num


def _eval_sigma_expression(expression: str, base_sigmas):
    interpreter = _eval.standard_interpreter(
        symtable=_eval.safe_builtins()
    )
    interpreter.symtable['np'] = numpy
    interpreter.symtable['sigmas'] = numpy.array(base_sigmas, dtype=float)
    try:
        value = interpreter.eval(expression, show_errors=False, raise_errors=True)
    except Exception as e:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Error interpreting sigmas expression "{expression}":\n{e}'
        ) from e
    if not isinstance(value, collections.abc.Iterable) or isinstance(value, (str, bytes)):
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Sigmas expression did not evaluate to an array, got: {value}'
        )
    return [float(item) for item in value]


def _ltx_is_distilled(pipe) -> bool:
    scheduler = getattr(pipe, 'scheduler', None)
    config = getattr(scheduler, 'config', None)
    if config is None:
        return False
    if hasattr(config, 'get'):
        return config.get('use_dynamic_shifting', True) is False
    return getattr(config, 'use_dynamic_shifting', True) is False
