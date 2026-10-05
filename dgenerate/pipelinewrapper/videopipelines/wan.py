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

"""Wan 2.1 / 2.2, Wan-Animate, and Wan-Animate-2 video generation."""

from __future__ import annotations

import hashlib
import os

import PIL.Image
import torch

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.types as _types

WAN_SCHEDULER_NAMES = frozenset({'FlowMatchEulerDiscreteScheduler'})
WAN_ANIMATE_SCHEDULER_NAMES = frozenset({'UniPCMultistepScheduler'})
WAN_ANIMATE_2_SCHEDULER_NAMES = frozenset({
    'FlowMatchEulerDiscreteScheduler',
    'EulerDiscreteScheduler',
    'UniPCMultistepScheduler',
    'DPMSolverMultistepScheduler',
})

_WAN_DEFAULT_FPS = 16.0
_WAN_ANIMATE_DEFAULT_FPS = 30.0
_WAN_ANIMATE_2_DEFAULT_FPS = 24.0
_WAN_DEFAULT_VAE_DTYPE = _enums.DataType.FLOAT32
_WAN_ANIMATE_DEFAULT_STEPS = 20
_WAN_ANIMATE_DEFAULT_GUIDANCE = 1.0
_WAN_ANIMATE_DEFAULT_SEGMENT = 77
_WAN_ANIMATE_DEFAULT_PREV_SEGMENT = 1
_WAN_ANIMATE_2_DEFAULT_STEPS = 40
_WAN_ANIMATE_2_DISTILLED_STEPS = 10
_WAN_ANIMATE_2_DEFAULT_SEGMENT = 81


def wan_family_from_index(index: dict | None) -> str:
    """
    Choose the Wan pipeline family from ``model_index.json``.

    :param index: ``model_index.json`` dict
    :raise UnsupportedPipelineConfigError: if the repo is Wan-Animate or not Wan
    :return: ``wan-vace``, ``wan-i2v``, or ``wan-t2v``
    """
    name = str((index or {}).get('_class_name') or '')
    if name.startswith('WanVACE'):
        return 'wan-vace'
    if name.startswith('WanImageToVideo'):
        return 'wan-i2v'
    if name.startswith('WanAnimate2'):
        raise _pipelines.UnsupportedPipelineConfigError(
            'This repository is a Wan-Animate-2 checkpoint. '
            'Use --model-type wan-animate-2.')
    if name.startswith('WanAnimate'):
        raise _pipelines.UnsupportedPipelineConfigError(
            'This repository is a Wan-Animate checkpoint. '
            'Use --model-type wan-animate.')
    if name.startswith('Wan'):
        return 'wan-t2v'
    raise _pipelines.UnsupportedPipelineConfigError(
        'This repository is not a Wan pipeline. '
        f'model_index.json _class_name is {name!r}.')


def wan_animate_family_from_index(index: dict | None) -> str:
    """
    Confirm a Wan-Animate ``model_index.json``.

    :param index: ``model_index.json`` dict
    :raise UnsupportedPipelineConfigError: if the repo is not Wan-Animate
    :return: ``wan-animate``
    """
    name = str((index or {}).get('_class_name') or '')
    if name.startswith('WanAnimate2'):
        raise _pipelines.UnsupportedPipelineConfigError(
            'This repository is a Wan-Animate-2 checkpoint. '
            'Use --model-type wan-animate-2.')
    if name.startswith('WanAnimate'):
        return 'wan-animate'
    raise _pipelines.UnsupportedPipelineConfigError(
        'This repository is not a Wan-Animate checkpoint. '
        f'model_index.json _class_name is {name!r}. '
        'Use --model-type wan for T2V, I2V, and VACE repos.')


def wan_animate_2_family_from_index(index: dict | None) -> str:
    """
    Choose the Wan-Animate-2 family from ``model_index.json``.

    :param index: ``model_index.json`` dict
    :raise UnsupportedPipelineConfigError: if the repo is not Wan-Animate-2
    :return: ``wan-animate-2`` or ``wan-animate-2-distilled``
    """
    name = str((index or {}).get('_class_name') or '')
    if not name.startswith('WanAnimate2'):
        raise _pipelines.UnsupportedPipelineConfigError(
            'This repository is not a Wan-Animate-2 checkpoint. '
            f'model_index.json _class_name is {name!r}. '
            'Use --model-type wan-animate for Wan-Animate, or --model-type wan '
            'for T2V, I2V, and VACE repos.')
    if name.startswith('WanAnimate2Distilled'):
        return 'wan-animate-2-distilled'
    scheduler = (index or {}).get('scheduler')
    scheduler_name = scheduler[-1] if isinstance(scheduler, (list, tuple)) and scheduler else ''
    if scheduler_name == 'FlowMatchEulerDiscreteScheduler':
        return 'wan-animate-2-distilled'
    return 'wan-animate-2'


def wan_num_frames(seconds: float, fps: float, temporal: int = 4) -> int:
    """
    Snap a requested duration to the Wan frame count ``temporal*k + 1``.

    :param seconds: requested length in seconds
    :param fps: frame rate
    :param temporal: ``pipe.vae_scale_factor_temporal``, 4 on Wan 2.1
    :return: frame count
    """
    temporal = max(1, int(temporal))
    raw = max(1, int(round(float(seconds) * float(fps))))
    k = max(0, int(round((raw - 1) / float(temporal))))
    return temporal * k + 1


def index_is_moe(index: dict | None) -> bool:
    """True when ``model_index.json`` lists a second MoE transformer."""
    spec = (index or {}).get('transformer_2')
    return isinstance(spec, (list, tuple)) and len(spec) == 2


def _classify_wan(parsed) -> str:
    _reject_animate_keywords(parsed, 'wan')
    if _has_path(parsed, 'control_images') or _has_path(parsed, 'mask_images') \
            or _has_path(parsed, 'reference_images'):
        return 'wan-vace'
    if parsed is not None and parsed.end_image:
        return 'wan-flf'
    if parsed is not None and parsed.images:
        if parsed.multi_image_mode or len(parsed.images) > 1:
            raise _pipelines.UnsupportedPipelineConfigError(
                'Wan accepts one conditioning path. Use a still for image-to-video, '
                'a video for video-to-video, or last-frame= for first-last-frame.')
        return 'wan-image'
    return 'wan-txt'


def _classify_wan_animate(parsed) -> str:
    if parsed is None or not parsed.images:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Wan-Animate needs a character image in --image-seeds.')
    if parsed.multi_image_mode or len(parsed.images) > 1:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Wan-Animate accepts one character image as the seed path.')
    if parsed.end_image:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Wan-Animate does not accept last-frame=. Use wan-pose= and wan-face=, '
            'or wan-driving= with --wan-animate-preprocess.')
    if parsed.control_images:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Wan-Animate does not accept control=. Use wan-pose= and wan-face=, '
            'or wan-driving= with --wan-animate-preprocess.')
    if parsed.reference_images:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Wan-Animate does not accept reference=. That keyword is for Wan VACE.')
    if parsed.latents or getattr(parsed, 'adapter_images', None) or parsed.floyd_image:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Wan-Animate does not accept latents, IP adapter images, or a floyd image.')
    driving = getattr(parsed, 'wan_driving_video', None)
    pose = getattr(parsed, 'wan_pose_video', None)
    face = getattr(parsed, 'wan_face_video', None)
    if driving:
        return 'wan-animate'
    if not pose or not face:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Wan-Animate needs wan-pose= and wan-face= clips, or wan-driving= with '
            '--wan-animate-preprocess or --wan-pose-image-processors and '
            '--wan-face-image-processors to derive them.')
    return 'wan-animate'


def _classify_wan_animate_2(parsed) -> str:
    if parsed is None or not parsed.images:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Wan-Animate-2 needs a character image in --image-seeds.')
    if parsed.multi_image_mode or len(parsed.images) > 1:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Wan-Animate-2 accepts one character image as the seed path.')
    if parsed.end_image or parsed.control_images or parsed.reference_images \
            or parsed.mask_images or parsed.latents \
            or getattr(parsed, 'adapter_images', None) or parsed.floyd_image:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Wan-Animate-2 accepts a character image and wan-driving=. '
            'It does not accept last-frame=, control=, mask=, reference=, '
            'latents, IP adapter images, or a floyd image.')
    if getattr(parsed, 'wan_pose_video', None) or getattr(parsed, 'wan_face_video', None) \
            or getattr(parsed, 'wan_background_video', None):
        raise _pipelines.UnsupportedPipelineConfigError(
            'Wan-Animate-2 does not accept wan-pose=, wan-face=, or wan-background=. '
            'Those keywords are for --model-type wan-animate. '
            'Pass the motion clip as wan-driving=.')
    if not getattr(parsed, 'wan_driving_video', None):
        raise _pipelines.UnsupportedPipelineConfigError(
            'Wan-Animate-2 needs wan-driving= in --image-seeds. '
            'The driving clip is the motion source.')
    return 'wan-animate-2'


def _reject_animate_keywords(parsed, model_name: str):
    if parsed is None:
        return
    used = []
    if getattr(parsed, 'wan_pose_video', None):
        used.append('wan-pose=')
    if getattr(parsed, 'wan_face_video', None):
        used.append('wan-face=')
    if getattr(parsed, 'wan_background_video', None):
        used.append('wan-background=')
    if getattr(parsed, 'wan_driving_video', None):
        used.append('wan-driving=')
    if used:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'{model_name} does not accept {", ".join(used)}. '
            'Those image seed keywords are for --model-type wan-animate.')


def _has_path(parsed, name: str) -> bool:
    if parsed is None:
        return False
    value = getattr(parsed, name, None)
    return bool(value)


def _resolve_wan_load(model_type, index):
    model_type = _enums.get_model_type_enum(model_type)
    if model_type == _enums.ModelType.WAN_ANIMATE:
        family = wan_animate_family_from_index(index)
        return family, _wan_pipeline_class('wan-animate', family), False
    if model_type == _enums.ModelType.WAN_ANIMATE_2:
        family = wan_animate_2_family_from_index(index)
        return family, _wan_pipeline_class('wan-animate-2', family), False
    family = wan_family_from_index(index)
    mode = {
        'wan-vace': 'wan-vace',
        'wan-i2v': 'wan-image',
        'wan-t2v': 'wan-txt',
    }[family]
    return family, _wan_pipeline_class(mode, family), index_is_moe(index)


def _wan_pipeline_class(mode: str, family: str = 'wan-t2v'):
    from diffusers import (
        WanAnimatePipeline,
        WanImageToVideoPipeline,
        WanPipeline,
        WanVACEPipeline,
        WanVideoToVideoPipeline)

    if mode == 'wan-animate-2' or str(family).startswith('wan-animate-2'):
        from diffusers import (
            WanAnimate2DistilledModularPipeline,
            WanAnimate2ModularPipeline)
        if family == 'wan-animate-2-distilled':
            return WanAnimate2DistilledModularPipeline
        return WanAnimate2ModularPipeline
    if mode == 'wan-animate' or family == 'wan-animate':
        return WanAnimatePipeline
    if mode == 'wan-vace' or family == 'wan-vace':
        return WanVACEPipeline
    if mode == 'wan-video':
        return WanVideoToVideoPipeline
    if mode in ('wan-image', 'wan-flf'):
        return WanImageToVideoPipeline
    if mode == 'wan-txt':
        return WanPipeline
    raise _pipelines.UnsupportedPipelineConfigError(
        f'Unknown Wan video mode {mode!r}.')


def _wan_transformer_class(model_type, family: str = 'wan-t2v'):
    model_type = _enums.get_model_type_enum(model_type)
    if model_type == _enums.ModelType.WAN_ANIMATE_2 or str(family).startswith('wan-animate-2'):
        from diffusers import WanAnimate2Transformer3DModel
        return WanAnimate2Transformer3DModel
    if model_type == _enums.ModelType.WAN_ANIMATE or family == 'wan-animate':
        from diffusers import WanAnimateTransformer3DModel
        return WanAnimateTransformer3DModel
    if family == 'wan-vace':
        from diffusers import WanVACETransformer3DModel
        return WanVACETransformer3DModel
    from diffusers import WanTransformer3DModel
    return WanTransformer3DModel


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


def _wan_flf_clip_kind(transformer) -> str:
    """
    How a Wan transformer accepts CLIP image embeddings.

    ``flf`` folds the first and last frame through ``pos_embed_seq_len``.
    ``first`` is a single-frame I2V embedder. ``none`` takes no image embeddings.
    """
    config = getattr(transformer, 'config', None)
    if config is None or getattr(config, 'image_dim', None) is None:
        return 'none'
    if getattr(config, 'pos_embed_seq_len', None):
        return 'flf'
    embedder = getattr(getattr(transformer, 'condition_embedder', None), 'image_embedder', None)
    if getattr(embedder, 'pos_embed', None) is not None:
        return 'flf'
    return 'first'


def _require_wan_flf_clip(pipe):
    transformer = getattr(pipe, 'transformer', None)
    if transformer is None or _wan_flf_clip_kind(transformer) != 'first':
        return
    raise _pipelines.UnsupportedPipelineConfigError(
        'last-frame= needs a first-last-frame checkpoint such as '
        'Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers. '
        'This transformer only embeds the first frame, so the first and last '
        'image embeddings cannot be combined.')


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


def _require_wan_size(width: int | None, height: int | None, pipe, model_name: str):
    if width is None or height is None:
        return
    align = _spatial_multiple(pipe)
    if int(width) % align or int(height) % align:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'{model_name} requires --output-size dimensions divisible by {align}. '
            f'Got {width}x{height}.')


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
    :data:`_WAN_DEFAULT_VAE_DTYPE` (float32) is used. Pass an explicit dtype
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
        dtype = _WAN_DEFAULT_VAE_DTYPE
    torch_dtype = _enums.get_torch_dtype(dtype)
    if torch_dtype is None:
        torch_dtype = torch.float32
    vae.to(dtype=torch_dtype)
    _messages.debug_log(f'Wan VAE dtype set to {torch_dtype}.')


def _call_wan(wrapper, user_args):
    from dgenerate.pipelinewrapper import videopipelines as _vp

    family = None
    if user_args.vace_video_frames or user_args.vace_mask_frames \
            or user_args.vace_reference_images:
        mode = 'wan-vace'
    elif user_args.end_images:
        mode = 'wan-flf'
    elif user_args.video_frames:
        mode = 'wan-video'
    elif user_args.images:
        mode = 'wan-image'
    else:
        mode = 'wan-txt'

    pipe, held = _vp._video_pipeline(wrapper, mode, user_args.scheduler_uri)
    family = held.family
    if family == 'wan-vace' and mode != 'wan-vace':
        mode = 'wan-vace'
        pipe, held = _vp._video_pipeline(wrapper, mode, user_args.scheduler_uri)
    elif mode == 'wan-vace' and family != 'wan-vace':
        raise _pipelines.UnsupportedPipelineConfigError(
            'control=, mask=, and reference= are Wan VACE inputs. '
            'Load a VACE checkpoint such as Wan-AI/Wan2.1-VACE-1.3B-diffusers.')
    if mode == 'wan-video' and user_args.video_length is not None:
        raise _pipelines.UnsupportedPipelineConfigError(
            '--video-lengths cannot be used with video-to-video. '
            'The output length follows the input clip. Trim with frame-start= and frame-end=.')
    if mode in ('wan-image', 'wan-flf') and family == 'wan-t2v':
        _require_image_encoder(pipe, mode)
    if mode == 'wan-flf' and family != 'wan-i2v':
        _require_image_encoder(pipe, mode)

    _apply_wan_config(pipe, user_args)
    positive, negative = _vp._prompt_text(user_args)
    width, height = _vp._size(user_args)
    _require_wan_size(width, height, pipe, 'Wan')

    fps = float(user_args.video_fps or _WAN_DEFAULT_FPS)
    temporal = _temporal_factor(pipe)
    kwargs = {
        'prompt': positive,
        'generator': _vp._generator(wrapper, user_args),
        'output_type': 'pil'
    }
    if negative:
        kwargs['negative_prompt'] = negative
    if width is not None:
        kwargs['width'] = width
        kwargs['height'] = height

    if mode == 'wan-video':
        video = _trim_wan_clip(user_args.video_frames, None, temporal)
        kwargs['video'] = video
        _messages.debug_log(f'Wan video-to-video: {len(video)} input frames.')
    elif user_args.video_length is not None:
        num_frames = wan_num_frames(user_args.video_length, fps, temporal)
        kwargs['num_frames'] = num_frames
        _messages.debug_log(
            f'Wan clip length {user_args.video_length} seconds at {fps} fps '
            f'-> {num_frames} frames.')
    else:
        _messages.debug_log('Wan will use its default frame count.')

    guidance = float(
        _types.default(user_args.guidance_scale, _constants.DEFAULT_GUIDANCE_SCALE))
    steps = int(_types.default(user_args.inference_steps, _constants.DEFAULT_INFERENCE_STEPS))
    kwargs['guidance_scale'] = guidance
    kwargs['num_inference_steps'] = steps
    if user_args.wan_low_noise_guidance_scale is not None:
        kwargs['guidance_scale_2'] = float(user_args.wan_low_noise_guidance_scale)
    if user_args.max_sequence_length is not None:
        kwargs['max_sequence_length'] = int(user_args.max_sequence_length)

    _vp._set_ltx_vae_slicing(pipe, bool(user_args.vae_slicing))

    if mode == 'wan-image':
        kwargs['image'] = user_args.images[0]
    elif mode == 'wan-flf':
        _require_wan_flf_clip(pipe)
        kwargs['image'] = user_args.images[0]
        kwargs['last_image'] = user_args.end_images[0]
    elif mode == 'wan-video':
        if user_args.image_seed_strength is not None:
            kwargs['strength'] = float(user_args.image_seed_strength)
        if user_args.wan_timesteps is not None:
            kwargs['timesteps'] = [int(value) for value in user_args.wan_timesteps]
    elif mode == 'wan-vace':
        _fill_vace_kwargs(kwargs, user_args, temporal)

    if mode != 'wan-video' and user_args.wan_timesteps is not None:
        raise _pipelines.UnsupportedPipelineConfigError(
            '--wan-timesteps is only used for Wan video-to-video.')
    if mode != 'wan-vace' and user_args.wan_conditioning_scale is not None:
        raise _pipelines.UnsupportedPipelineConfigError(
            '--wan-conditioning-scales is only used with a Wan VACE checkpoint.')

    output = _vp._invoke(wrapper, pipe, kwargs)
    frames = _vp.frames_from_output(getattr(output, 'frames', output))
    return frames, None, None, fps


def _fill_vace_kwargs(kwargs, user_args, temporal: int):
    video = user_args.vace_video_frames or user_args.video_frames
    if video:
        limit = kwargs.get('num_frames')
        kwargs['video'] = _trim_wan_clip(video, limit, temporal)
        _messages.debug_log(f'Wan VACE control clip: {len(kwargs["video"])} frames.')
    if user_args.vace_mask_frames:
        kwargs['mask'] = _trim_wan_clip(
            user_args.vace_mask_frames, kwargs.get('num_frames'), temporal)
    if user_args.vace_reference_images:
        kwargs['reference_images'] = list(user_args.vace_reference_images)
    if user_args.wan_conditioning_scale is not None:
        scale = user_args.wan_conditioning_scale
        if isinstance(scale, (list, tuple)):
            kwargs['conditioning_scale'] = [float(value) for value in scale]
        else:
            kwargs['conditioning_scale'] = float(scale)


def _call_wan_animate(wrapper, user_args):
    from dgenerate.pipelinewrapper import videopipelines as _vp

    if user_args.video_length is not None:
        raise _pipelines.UnsupportedPipelineConfigError(
            '--video-lengths cannot be used with --model-type wan-animate. '
            'The output length follows the pose clip.')
    if not user_args.images:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Wan-Animate needs a character image in --image-seeds.')

    pose = user_args.wan_pose_video_frames
    face = user_args.wan_face_video_frames
    if user_args.wan_animate_preprocess and user_args.wan_driving_video_frames:
        derived_pose, derived_face = preprocess_wan_animate_driving(
            user_args.wan_driving_video_frames,
            user_args.wan_driving_video_path,
            user_args.wan_animate_cache_dir,
            motion_encoder_size=_animate_motion_size(wrapper),
            device=wrapper.device)
        pose = pose or derived_pose
        face = face or derived_face
    if not pose or not face:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Wan-Animate needs wan-pose= and wan-face= clips, or wan-driving= with '
            '--wan-animate-preprocess or --wan-pose-image-processors and '
            '--wan-face-image-processors.')

    pipe, held = _vp._video_pipeline(wrapper, 'wan-animate', user_args.scheduler_uri)
    _require_image_encoder(pipe, 'wan-animate')
    _apply_wan_config(pipe, user_args)
    positive, negative = _vp._prompt_text(user_args)
    width, height = _vp._size(user_args)
    _require_wan_size(width, height, pipe, 'Wan-Animate')

    fps = float(user_args.video_fps or _WAN_ANIMATE_DEFAULT_FPS)
    mode = user_args.wan_animate_mode or 'animate'
    if mode not in ('animate', 'replace'):
        raise _pipelines.UnsupportedPipelineConfigError(
            '--wan-animate-mode must be animate or replace.')
    if mode == 'replace':
        if not user_args.wan_background_video_frames or not user_args.mask_video_frames:
            raise _pipelines.UnsupportedPipelineConfigError(
                'Wan-Animate replace mode needs wan-background= and mask= clips.')

    guidance = float(
        _types.default(user_args.guidance_scale, _constants.DEFAULT_GUIDANCE_SCALE))
    steps = int(_types.default(user_args.inference_steps, _constants.DEFAULT_INFERENCE_STEPS))
    if guidance == _constants.DEFAULT_GUIDANCE_SCALE:
        guidance = _WAN_ANIMATE_DEFAULT_GUIDANCE
        user_args.guidance_scale = guidance
        _messages.debug_log(
            'Wan-Animate runs unguided. Default guidance scale 5 was replaced with 1.')
    if steps == _constants.DEFAULT_INFERENCE_STEPS:
        steps = _WAN_ANIMATE_DEFAULT_STEPS
        user_args.inference_steps = steps
        _messages.debug_log(
            'Wan-Animate default inference steps 30 was replaced with 20.')

    temporal = _temporal_factor(pipe)
    pose = _trim_wan_clip(pose, None, temporal)
    face = face[:len(pose)]
    kwargs = {
        'image': user_args.images[0],
        'pose_video': pose,
        'face_video': face,
        'prompt': positive,
        'generator': _vp._generator(wrapper, user_args),
        'output_type': 'pil',
        'mode': mode,
        'guidance_scale': guidance,
        'num_inference_steps': steps,
        'segment_frame_length': int(
            _types.default(user_args.wan_segment_frame_length, _WAN_ANIMATE_DEFAULT_SEGMENT)),
        'prev_segment_conditioning_frames': int(
            _types.default(user_args.wan_prev_segment_frames, _WAN_ANIMATE_DEFAULT_PREV_SEGMENT)),
    }
    if negative:
        kwargs['negative_prompt'] = negative
    if width is not None:
        kwargs['width'] = width
        kwargs['height'] = height
    if user_args.wan_motion_encode_batch_size is not None:
        kwargs['motion_encode_batch_size'] = int(user_args.wan_motion_encode_batch_size)
    if user_args.max_sequence_length is not None:
        kwargs['max_sequence_length'] = int(user_args.max_sequence_length)
    if mode == 'replace':
        kwargs['background_video'] = user_args.wan_background_video_frames[:len(pose)]
        kwargs['mask_video'] = user_args.mask_video_frames[:len(pose)]

    _vp._set_ltx_vae_slicing(pipe, bool(user_args.vae_slicing))
    output = _vp._invoke(wrapper, pipe, kwargs)
    frames = _vp.frames_from_output(getattr(output, 'frames', output))
    return frames[:len(pose)], None, None, fps


def _call_wan_animate_2(wrapper, user_args):
    from dgenerate.pipelinewrapper import videopipelines as _vp

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
    _require_wan_size(width, height, pipe, 'Wan-Animate-2')

    distilled = held.family == 'wan-animate-2-distilled'
    steps = int(_types.default(user_args.inference_steps, _constants.DEFAULT_INFERENCE_STEPS))
    if steps == _constants.DEFAULT_INFERENCE_STEPS:
        steps = _WAN_ANIMATE_2_DISTILLED_STEPS if distilled else _WAN_ANIMATE_2_DEFAULT_STEPS
        user_args.inference_steps = steps
        _messages.debug_log(
            f'Wan-Animate-2 default inference steps '
            f'{_constants.DEFAULT_INFERENCE_STEPS} was replaced with {steps}.')

    fps = float(user_args.video_fps or _WAN_ANIMATE_2_DEFAULT_FPS)
    kwargs = {
        'image': user_args.images[0],
        'driving_video': list(user_args.wan_driving_video_frames),
        'prompt': positive,
        'generator': _vp._generator(wrapper, user_args),
        'output': 'videos',
        'output_type': 'pil',
        'fps': int(round(fps)),
        'num_inference_steps': steps,
    }
    if negative:
        kwargs['negative_prompt'] = negative
    if user_args.wan_driving_video_fps:
        kwargs['driving_video_fps'] = float(user_args.wan_driving_video_fps)
    if width is not None:
        kwargs['width'] = int(width)
        kwargs['height'] = int(height)
    if user_args.wan_segment_frame_length is not None:
        kwargs['segment_frame_length'] = int(user_args.wan_segment_frame_length)
    if user_args.wan_prev_segment_frames is not None:
        kwargs['prev_segment_conditioning_frames'] = int(user_args.wan_prev_segment_frames)
    if user_args.max_sequence_length is not None:
        kwargs['max_sequence_length'] = int(user_args.max_sequence_length)

    _vp._set_ltx_vae_slicing(pipe, bool(user_args.vae_slicing))
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


def place_wan_animate_2(pipe, device, model_cpu_offload, sequential_cpu_offload,
                        model_group_offload):
    """
    Place Wan-Animate-2 components.

    The modular pipeline has no pipeline-level CPU offload. Offload flags stream
    the transformer in block groups and keep the encoders and VAE on ``device``.
    Quantized / GGUF transformers stay where they were loaded; group offload
    would move packed weights the same way it must not for still pipelines.
    """
    names = (
        'text_encoder', 'image_encoder', 'vae', 'transformer',
        'video_processor', 'image_processor', 'guider',
    )
    offload = bool(model_cpu_offload or sequential_cpu_offload or model_group_offload)
    transformer = getattr(pipe, 'transformer', None)
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
                num_blocks_per_group=4,
                use_stream=onload.type == 'cuda',
            )
        names = tuple(name for name in names if name != 'transformer')
    for name in names:
        module = getattr(pipe, name, None)
        if module is not None and hasattr(module, 'to'):
            module.to(device)


def _animate_motion_size(wrapper) -> int:
    return 512


def preprocess_wan_animate_driving(frames: list[PIL.Image.Image],
                                   source_path: str | None,
                                   cache_dir: str | None,
                                   motion_encoder_size: int = 512,
                                   device: str = 'cpu'
                                   ) -> tuple[list[PIL.Image.Image], list[PIL.Image.Image]]:
    """
    Derive pose and face clips from a driving video.

    Pose is ``openpose``. Face is a square crop around the first
    ``yolo`` face box. Both processors run on ``device`` for the clip
    and return to CPU when the clip is finished. Cached files are
    written next to the driving clip when ``source_path`` is a local
    file, otherwise under ``cache_dir``.
    """
    pose_path, face_path = _animate_cache_paths(source_path, cache_dir)
    if _cache_fresh(source_path, pose_path, face_path):
        from dgenerate.pipelinewrapper import videopipelines as _vp
        pose, _ = _vp.load_rgb_frames(pose_path)
        face, _ = _vp.load_rgb_frames(face_path)
        _messages.log(f'Reusing cached Wan-Animate pose and face clips.')
        return pose, face

    pose_processor, face_processor = _animate_processors(motion_encoder_size, device)
    try:
        pose = [pose_processor.process(frame.copy()) for frame in frames]
        face = [face_processor.process(frame.copy()) for frame in frames]
    finally:
        pose_processor.to('cpu')
        if hasattr(face_processor, 'to'):
            face_processor.to('cpu')
    _write_clip_cache(pose, pose_path)
    _write_clip_cache(face, face_path)
    return pose, face


def _animate_cache_paths(source_path: str | None, cache_dir: str | None) -> tuple[str, str]:
    if source_path and os.path.isfile(source_path):
        root, _ = os.path.splitext(source_path)
        return f'{root}.wan-pose.mp4', f'{root}.wan-face.mp4'
    directory = cache_dir or os.getcwd()
    digest = hashlib.sha1((source_path or 'driving').encode('utf-8')).hexdigest()[:12]
    return (os.path.join(directory, f'wan-animate-{digest}-pose.mp4'),
            os.path.join(directory, f'wan-animate-{digest}-face.mp4'))


def _cache_fresh(source_path: str | None, pose_path: str, face_path: str) -> bool:
    if not (os.path.isfile(pose_path) and os.path.isfile(face_path)):
        return False
    if not source_path or not os.path.isfile(source_path):
        return True
    source_mtime = os.path.getmtime(source_path)
    return os.path.getmtime(pose_path) >= source_mtime and os.path.getmtime(face_path) >= source_mtime


def _write_clip_cache(frames: list[PIL.Image.Image], path: str):
    try:
        import dgenerate.mediaoutput as _mediaoutput
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with _mediaoutput.VideoWriter(path, fps=16) as writer:
            for frame in frames:
                writer.write(frame)
        _messages.debug_log(f'Wrote Wan-Animate cache clip "{path}".')
    except Exception as error:
        _messages.debug_log(f'Could not cache Wan-Animate clip "{path}": {error}')


def _animate_processors(motion_encoder_size: int = 512, device: str = 'cpu'):
    import dgenerate.imageprocessors as _imgp
    pose = _imgp.create_image_processor(
        'openpose;include-hand=true', device=device)
    face = _imgp.create_image_processor(
        'yolo;model=Bingsu/adetailer;weight-name=face_yolov8n.pt;'
        f'crops=True;crop-square=True;crop-scale=1.4;'
        f'crop-size={int(motion_encoder_size)}',
        device=device)
    return pose, face
