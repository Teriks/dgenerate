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
import inspect
import os
import warnings

import numpy
import torch

import dgenerate.eval as _eval
import dgenerate.hfhub as _hfhub
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.types as _types


class _Shared:
    """Look up helpers on the package so tests can patch ``_invoke`` there."""

    def __getattr__(self, name):
        from dgenerate.pipelinewrapper import videopipelines
        return getattr(videopipelines, name)


_vp = _Shared()

_LTX_DEFAULT_FPS = 24.0

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

    pipe, held = _vp._video_pipeline(wrapper, mode, user_args.scheduler_uri)
    family = held.family
    positive, negative = _vp._prompt_text(user_args)
    width, height = _vp._size(user_args)
    if width is not None:
        _vp._require_multiple_of_32(width, height, 'LTX')
    if mode != 'ltx-txt':
        width, height = _fit_ltx_inputs(pipe, held, mode, user_args, width, height)

    fps = float(user_args.video_fps or _LTX_DEFAULT_FPS)
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


def _ensure_prompt_enhancer(wrapper, pipe, repo):
    if getattr(pipe, '_dgenerate_prompt_enhancer', None) == repo:
        return
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = _enums.get_torch_dtype(wrapper._dtype) or torch.bfloat16
    load = {
        'token': wrapper._auth_token,
        'local_files_only': bool(wrapper._local_files_only),
    }
    _messages.debug_log(f'Loading LTX prompt enhancer "{repo}".')
    pipe.prompt_enhancer = AutoModelForImageTextToText.from_pretrained(
        repo, torch_dtype=dtype, **load)
    pipe.processor = AutoProcessor.from_pretrained(repo, **load)
    device = 'cpu' if _vp._offload_requested(wrapper) else wrapper.device
    pipe.prompt_enhancer.to(device)
    pipe._dgenerate_prompt_enhancer = repo


def _complete_ltx_call(wrapper, pipe, held, user_args, kwargs, family):
    diffusion = family == 'ltx2' and user_args.ltx_video_decoder == 'diffusion'
    if user_args.ltx_latent_upscale:
        return _ltx_two_stage(wrapper, pipe, held, user_args, kwargs, diffusion)
    if diffusion:
        kwargs = dict(kwargs)
        kwargs['output_type'] = 'latent'
    output = _vp._invoke(wrapper, pipe, kwargs)
    return _ltx_pixels_from_output(wrapper, pipe, held, output, diffusion, kwargs.get('generator'))


def _enhance_ltx_prompt_once(wrapper, pipe, kwargs):
    """
    Rewrite the prompt once, then turn enhancement off for every later pipeline call.

    ``enhance_prompt`` moves Gemma onto the compute device and leaves it there.
    A second call, which the two-stage refine would otherwise make, also meets
    the sequential-offload hooks that loading a stage LoRA puts back on every
    pipeline module. Gemma indexes ``embed_tokens.weight`` directly, and that
    weight is a meta tensor under those hooks.
    """
    if not kwargs.get('enable_prompt_enhancement') or kwargs.get('prompt') is None:
        return
    if getattr(pipe, 'prompt_enhancer', None) is None:
        return
    system_prompt = kwargs.get('system_prompt')
    if system_prompt is None:
        from diffusers.pipelines.ltx2.utils import LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT
        system_prompt = LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT
    enhanced = pipe.enhance_prompt(
        prompt=kwargs['prompt'],
        system_prompt=system_prompt,
        generator=kwargs.get('generator'),
        device=wrapper.device)
    if isinstance(enhanced, (list, tuple)):
        enhanced = enhanced[0]
    kwargs['prompt'] = enhanced
    kwargs['enable_prompt_enhancement'] = False
    if _vp._offload_requested(wrapper):
        pipe.prompt_enhancer.to('cpu')
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
    if width is None or height is None or int(width) % 64 or int(height) % 64:
        raise _pipelines.UnsupportedPipelineConfigError(
            '--ltx-latent-upscale needs an --output-size divisible by 64. '
            'That size is the finished clip.')
    _enhance_ltx_prompt_once(wrapper, pipe, kwargs)
    stage1 = dict(kwargs)
    stage1['width'] = int(width) // 2
    stage1['height'] = int(height) // 2
    stage1['output_type'] = 'latent'
    _messages.log(
        f'LTX stage 1 at {stage1["width"]}x{stage1["height"]}, '
        f'then a latent upscale to {width}x{height}.')
    first = _vp._invoke(wrapper, pipe, stage1)
    video_latent = _video_latent_tensor(first.frames)
    audio_latent = getattr(first, 'audio', None)

    upscaled = _upsample_ltx_latents(wrapper, pipe, video_latent)
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
        second = _vp._invoke(wrapper, pipe, stage2)
    finally:
        pipe.scheduler = original
        if loaded_stage_lora and hasattr(pipe, 'unload_lora_weights'):
            pipe.unload_lora_weights()
        if enhancer is not None:
            pipe.prompt_enhancer = enhancer
    return _ltx_pixels_from_output(
        wrapper, pipe, held, second, diffusion, kwargs.get('generator'))


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


def _upsample_ltx_latents(wrapper, pipe, video_latent):
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    from diffusers.pipelines.ltx2.pipeline_ltx2_latent_upsample import LTX2LatentUpsamplePipeline

    dtype = _enums.get_torch_dtype(wrapper._dtype) or torch.bfloat16
    upsampler = LTX2LatentUpsamplerModel.from_pretrained(
        wrapper.model_path,
        subfolder='latent_upsampler',
        torch_dtype=dtype,
        token=wrapper._auth_token,
        local_files_only=bool(wrapper._local_files_only))
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


def _ltx_pixels_from_output(wrapper, pipe, held, output, diffusion, generator):
    video = output.frames
    audio_raw = getattr(output, 'audio', None)
    if diffusion:
        if isinstance(video, (list, tuple)):
            video = video[0]
        frames = _decode_ltx_diffusion(wrapper, pipe, held, video, generator)
        audio = _waveform_from_audio_latents(pipe, audio_raw)
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


def _ltx_diffusion_natten_processor():
    """
    NATTEN processor for the LTX diffusion decoder.

    The FlexAttention processor builds a dense neighborhood mask. At a real
    video grid that mask is larger than the GPU, so this does not fall back to it.
    """
    try:
        from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
            LTX2VideoVaeNeighborhoodNattenProcessor)
        return LTX2VideoVaeNeighborhoodNattenProcessor()
    except Exception as error:
        raise _pipelines.UnsupportedPipelineConfigError(
            'The LTX diffusion decoder needs NATTEN. Its FlexAttention path builds a '
            'neighborhood mask that does not fit in GPU memory at video resolution. '
            'The kernels package is a dgenerate dependency; reinstall dgenerate or '
            'run `pip install kernels`, and leave DIFFUSERS_DISABLE_REMOTE_CODE unset. '
            f'{error}'
        ) from error


def _decode_ltx_diffusion(wrapper, pipe, held, latents, generator):
    decode_pipe = getattr(held, 'diffusion_decoder_pipe', None)
    if decode_pipe is None:
        from diffusers import LTX2VideoDiffusionDecodePipeline, LTX2VideoDiffusionDecoderModel

        dtype = _enums.get_torch_dtype(wrapper._dtype) or torch.bfloat16
        decoder = LTX2VideoDiffusionDecoderModel.from_pretrained(
            wrapper.model_path,
            subfolder='diffusion_decoder',
            torch_dtype=dtype,
            token=wrapper._auth_token,
            local_files_only=bool(wrapper._local_files_only))
        decoder.set_attn_processor(_ltx_diffusion_natten_processor())
        decode_pipe = LTX2VideoDiffusionDecodePipeline(
            diffusion_decoder=decoder, scheduler=pipe.scheduler, vae=pipe.vae)
        _vp._offload_ltx(
            decode_pipe, wrapper.device,
            bool(wrapper.model_cpu_offload), bool(wrapper.model_sequential_offload),
            bool(getattr(wrapper, 'model_group_offload', False)))
        held.diffusion_decoder_pipe = decode_pipe
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
    if factor > 1 and width is not None and height is not None:
        width_bad = (int(width) // factor) % spatial
        height_bad = (int(height) // factor) % spatial
        if width_bad or (height_bad and not user_args.aspect_correct):
            raise _pipelines.UnsupportedPipelineConfigError(
                f'The IC-LoRA reads the control clip at 1/{factor} of the output size, '
                f'so the output width and height must be divisible by {spatial * factor}. '
                f'Got {int(width)}x{int(height)}.')

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


def _ltx_canvas_align(pipe, held, mode: str) -> int:
    spatial = int(getattr(pipe, 'vae_spatial_compression_ratio', 32) or 32)
    if mode != 'ltx-control':
        return spatial
    factor = int(getattr(held, 'reference_downscale_factor', None) or 1)
    return spatial * max(1, factor)


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
    align = _ltx_canvas_align(pipe, held, mode)
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
