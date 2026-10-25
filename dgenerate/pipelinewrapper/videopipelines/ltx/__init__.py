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

import contextlib
import gc
import inspect

import numpy
import torch

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.pipelinewrapper.util as _putil
import dgenerate.types as _types
import dgenerate.pipelinewrapper.videopipelines as _vp
from ..schedules import (
    _eval_sigma_expression,
    _eval_video_timesteps_expression,
    _forced_scheduler_timesteps,
    _preserved_scheduler_state,
)
from .components import (
    _check_ltx_component_available,
    _check_ltx_diffusion_decoder_available,
    _check_ltx_latent_upsampler_available,
    _load_ltx_latent_upsampler,
    _upsample_ltx_latents,
)
from .decoder import (
    _decode_ltx_diffusion,
    _park_ltx_after_diffusion_decode,
    _park_ltx_for_diffusion_decode,
)
from .conditions import (
    _fit_ltx_inputs,
    _ltx_canvas_align,
    _ltx_conditions,
    _ltx_image_needs_conditions,
    _ltx_reference_kwargs,
    _ltx_temporal_compression,
    _trim_ltx_clip,
)
from .legacy import (
    _fix_legacy_ltx_condition_schedule,
    _require_legacy_ltx_schedule,
    _warn_legacy_ltx_canvas,
)


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


def ltx_family_for_model_type(model_type, index: dict | None) -> str:
    """
    Choose the LTX pipeline family from ``--model-type``, and check that the
    repository's ``model_index.json`` agrees with it.

    :param model_type: ``ltx2`` or ``ltx-video``
    :param index: ``model_index.json`` dict
    :raise UnsupportedPipelineConfigError: if the repository is the other LTX family
    :return: ``ltx2`` or ``ltx``
    """
    model_type = _enums.get_model_type_enum(model_type)
    expected = 'ltx2' if model_type == _enums.ModelType.LTX2 else 'ltx'
    if index is not None:
        found = ltx_family_from_index(index)
        if found != expected:
            wanted = _enums.get_model_type_string(model_type)
            actual = 'ltx2' if found == 'ltx2' else 'ltx-video'
            raise _pipelines.UnsupportedPipelineConfigError(
                f'--model-type {wanted} was given, but this repository is an '
                f'{actual} pipeline ({(index or {}).get("_class_name")}). '
                f'Use --model-type {actual}.')
    return expected


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
        if user_args.timesteps is not None:
            raise _pipelines.UnsupportedPipelineConfigError(
                '--timesteps is only applied to LTX-2. This checkpoint uses the earlier LTX pipeline.')
    else:
        distilled = _ltx_is_distilled(pipe)

        if user_args.timesteps is not None:
            if isinstance(user_args.timesteps, str):
                base_timesteps = _ltx_default_timesteps(
                    pipe, distilled, user_args.inference_steps)
                kwargs['timesteps'] = _eval_video_timesteps_expression(
                    user_args.timesteps, base_timesteps)
            else:
                kwargs['timesteps'] = [int(value) for value in user_args.timesteps]
            # _complete_ltx_call moves these onto the scheduler.
            kwargs['num_inference_steps'] = len(kwargs['timesteps'])
            if distilled:
                if guidance == _constants.DEFAULT_GUIDANCE_SCALE:
                    guidance = 1.0
                    user_args.guidance_scale = 1.0
                    _messages.debug_log(
                        'LTX distilled checkpoint runs unguided. Default guidance scale 5 was replaced with 1.')
                else:
                    _messages.debug_log(
                        f'LTX distilled checkpoint: distilled timestep schedule, guidance {guidance}.')
            else:
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
                        f'LTX full checkpoint: {len(kwargs["timesteps"])} steps, guidance {guidance}, '
                        f'audio guidance {audio_guidance}.')
            if distilled:
                audio_guidance = float(user_audio) if user_audio is not None else guidance
            kwargs['guidance_scale'] = guidance
            kwargs['audio_guidance_scale'] = audio_guidance
            user_args.inference_steps = len(kwargs['timesteps'])
        elif user_args.sigmas is not None:
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

    _vp._set_video_vae_slicing(pipe, bool(user_args.vae_slicing))

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
    if user_args.ltx_audio_stg_scale is not None:
        used.append('--ltx-audio-stg-scales')
    if user_args.ltx_audio_modality_scale is not None:
        used.append('--ltx-audio-modality-scales')
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
    # LTX2Pipeline always passes sigmas to retrieve_timesteps, which rejects
    # timesteps alongside them, so a --timesteps schedule is applied at the
    # scheduler. The earlier LTX condition pipeline takes its timesteps directly.
    custom_timesteps = None
    if family == 'ltx2' and kwargs.get('timesteps') is not None:
        kwargs = dict(kwargs)
        custom_timesteps = kwargs.pop('timesteps')
        if user_args.ltx_latent_upscale and user_args.ltx_latent_upscale_mode == 'temporal':
            raise _pipelines.UnsupportedPipelineConfigError(
                '--timesteps cannot be used with temporal latent upscaling, '
                'its passes use fixed distilled schedules.')
    forced_timesteps = (_forced_scheduler_timesteps(pipe.scheduler, custom_timesteps)
                        if custom_timesteps is not None else contextlib.nullcontext())

    _enhance_ltx_prompt_once(wrapper, pipe, kwargs)
    enhancer = _detach_prompt_enhancer(pipe) if enhancement_requested else None
    try:
        diffusion = family == 'ltx2' and user_args.ltx_video_decoder == 'diffusion'
        if user_args.ltx_latent_upscale:
            if user_args.ltx_latent_upscale_mode == 'temporal':
                frames, audio, sample_rate = _ltx_temporal_upscale(
                    wrapper, pipe, held, user_args, kwargs, diffusion)
                return frames, audio, sample_rate
            # Stage 2 swaps in a fresh scheduler with its own sigmas, so only
            # stage 1 follows the custom schedule.
            with forced_timesteps:
                return _ltx_two_stage(wrapper, pipe, held, user_args, kwargs, diffusion)
        if diffusion:
            kwargs = dict(kwargs)
            kwargs['output_type'] = 'latent'
        with forced_timesteps:
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
                auth_token=wrapper._auth_token,
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
        auth_token=wrapper._auth_token,
        local_files_only=bool(wrapper._local_files_only))
    family = ltx_family_for_model_type(wrapper.model_type, index)

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


def _ltx_default_timesteps(pipe, distilled: bool, inference_steps):
    """
    The schedule LTX2Pipeline hands to ``set_timesteps``, expressed as timesteps.

    LTX2Pipeline passes ``linspace(1, 1/steps, steps)`` (or the distilled sigma
    values) to the scheduler, which applies the resolution-dependent ``mu``
    shift. Custom timesteps take the same route, so an expression sees the
    unshifted values rather than the shifted ``scheduler.timesteps``.
    """
    if distilled:
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
        sigmas = numpy.asarray(DISTILLED_SIGMA_VALUES, dtype=float)
    else:
        steps = int(_types.default(inference_steps, _constants.DEFAULT_INFERENCE_STEPS))
        sigmas = numpy.linspace(1.0, 1.0 / steps, steps)
    num_train_timesteps = pipe.scheduler.config.get('num_train_timesteps', 1000)
    return numpy.rint(sigmas * num_train_timesteps).astype(numpy.int64).tolist()


def _ltx_base_sigmas(pipe, distilled: bool, inference_steps):
    if distilled:
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
        return list(DISTILLED_SIGMA_VALUES)
    steps = int(_types.default(inference_steps, _constants.DEFAULT_INFERENCE_STEPS))
    try:
        with _preserved_scheduler_state(pipe.scheduler) as scheduler:
            scheduler.set_timesteps(steps)
            return list(scheduler.sigmas)
    except Exception as e:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Could not read scheduler sigmas for an LTX sigma expression. '
            'Pass a CSV list with --sigmas instead.'
        ) from e


def _ltx_is_distilled(pipe) -> bool:
    scheduler = getattr(pipe, 'scheduler', None)
    config = getattr(scheduler, 'config', None)
    if config is None:
        return False
    if hasattr(config, 'get'):
        return config.get('use_dynamic_shifting', True) is False
    return getattr(config, 'use_dynamic_shifting', True) is False
