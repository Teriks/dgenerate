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

import inspect

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.types as _types
import dgenerate.pipelinewrapper.videopipelines as _vp
from ..schedules import (
    _default_video_timesteps,
    _eval_video_timesteps_expression,
)
from .common import (
    _apply_wan_config,
    _fit_output_media,
    _require_image_encoder,
    _spatial_multiple,
    _temporal_factor,
    _trim_wan_clip,
)

# Entry points the videopipelines package calls through ``wan``.
from .animate import _call_wan_animate
from .animate2 import (
    _call_wan_animate_2,
    load_wan_animate_2_pipeline,
    place_wan_animate_2,
)
from .common import _set_wan_vae_dtype


WAN_SCHEDULER_NAMES = frozenset({'FlowMatchEulerDiscreteScheduler'})
WAN_ANIMATE_SCHEDULER_NAMES = frozenset({'UniPCMultistepScheduler'})
WAN_ANIMATE_2_SCHEDULER_NAMES = frozenset({
    'FlowMatchEulerDiscreteScheduler',
    'EulerDiscreteScheduler',
    'UniPCMultistepScheduler',
    'DPMSolverMultistepScheduler',
})


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


def _call_wan(wrapper, user_args):

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
    align = _spatial_multiple(pipe)
    width, height = _vp._force_output_size_alignment(width, height, align, 'Wan')
    if mode == 'wan-image':
        width, height = _fit_output_media(
            user_args, 'images', [], width, height, align, 'Wan image')
    elif mode == 'wan-flf':
        width, height = _fit_output_media(
            user_args, 'images', [('end_images', False)], width, height, align,
            'Wan first frame')
    elif mode == 'wan-video':
        width, height = _fit_output_media(
            user_args, 'video_frames', [], width, height, align, 'Wan video')
    elif mode == 'wan-vace':
        width, height = _fit_vace_canvas(user_args, width, height, align)

    fps = float(user_args.video_fps or _constants.DEFAULT_WAN_FPS)
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

    _vp._set_video_vae_slicing(pipe, bool(user_args.vae_slicing))

    custom_timesteps = None
    if user_args.timesteps is not None:
        if mode != 'wan-video':
            raise _pipelines.UnsupportedPipelineConfigError(
                '--timesteps is only used for Wan video-to-video.')
        if 'timesteps' not in inspect.signature(pipe.scheduler.set_timesteps).parameters:
            raise _pipelines.UnsupportedPipelineConfigError(
                'The loaded Wan scheduler does not support custom timesteps.')
        if isinstance(user_args.timesteps, str):
            base_timesteps = _default_video_timesteps(pipe, 'wan', steps)
            custom_timesteps = _eval_video_timesteps_expression(
                user_args.timesteps, base_timesteps)
        else:
            custom_timesteps = [int(value) for value in user_args.timesteps]
        if not custom_timesteps:
            raise _pipelines.UnsupportedPipelineConfigError(
                'Wan timesteps cannot be empty.')
        kwargs['num_inference_steps'] = len(custom_timesteps)
        user_args.inference_steps = len(custom_timesteps)

    if mode == 'wan-image':
        kwargs['image'] = user_args.images[0]
    elif mode == 'wan-flf':
        _require_wan_flf_clip(pipe)
        kwargs['image'] = user_args.images[0]
        kwargs['last_image'] = user_args.end_images[0]
    elif mode == 'wan-video':
        if user_args.image_seed_strength is not None:
            kwargs['strength'] = float(user_args.image_seed_strength)
        if custom_timesteps is not None:
            kwargs['timesteps'] = custom_timesteps
    elif mode == 'wan-vace':
        _fill_vace_kwargs(kwargs, user_args, temporal)
    if mode != 'wan-vace' and user_args.wan_conditioning_scale is not None:
        raise _pipelines.UnsupportedPipelineConfigError(
            '--wan-conditioning-scales is only used with a Wan VACE checkpoint.')

    # WanVideoToVideoPipeline forwards ``timesteps`` to retrieve_timesteps itself.
    output = _vp._invoke(wrapper, pipe, kwargs, user_args=user_args)
    frames = _vp.frames_from_output(getattr(output, 'frames', output))
    return frames, None, None, fps


def _fit_vace_canvas(user_args, width, height, align: int):
    """
    Resize the VACE control clip and mask onto one canvas.

    The clip chooses the canvas with the still-image ``--output-size`` rule.
    The mask is resized to that same canvas so it stays registered.
    """
    from dgenerate.pipelinewrapper import videopipelines as _pipelines_pkg

    video_attr = 'vace_video_frames' if user_args.vace_video_frames else 'video_frames'
    video = getattr(user_args, video_attr)
    if not video:
        return width, height
    source = video[0].size
    width, height = _fit_output_media(
        user_args, video_attr, [], width, height, align, 'Wan VACE control clip')
    mask = user_args.vace_mask_frames
    target = (width, height)
    if mask and mask[0].size != target:
        if mask[0].size != source:
            _messages.warning(
                f'Wan VACE mask is {mask[0].size[0]}x{mask[0].size[1]} while the '
                f'control clip is {source[0]}x{source[1]}. The mask is resized '
                f'to the control clip canvas.')
        user_args.vace_mask_frames = _pipelines_pkg.resize_media(
            mask, target, nearest=True)
    return width, height


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
