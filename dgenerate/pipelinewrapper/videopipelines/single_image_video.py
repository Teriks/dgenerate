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

"""
Hunyuan Video 1.5, Motif-Video, and Kandinsky 5 video.

These families share one call path: text-to-video, or image-to-video from a
single still image.
"""

from __future__ import annotations

import contextlib
import inspect

import numpy

import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.types as _types
import dgenerate.pipelinewrapper.videopipelines as _vp
from .schedules import (
    _default_video_timesteps,
    _eval_sigma_expression,
    _eval_video_timesteps_expression,
    _forced_scheduler_timesteps,
    _integer_timesteps,
    _preserved_scheduler_state,
)

HUNYUAN_VIDEO_15_SCHEDULER_NAMES = frozenset({'FlowMatchEulerDiscreteScheduler'})
MOTIF_VIDEO_SCHEDULER_NAMES = frozenset({
    'FlowMatchEulerDiscreteScheduler', 'DPMSolverMultistepScheduler'})
KANDINSKY5_VIDEO_SCHEDULER_NAMES = frozenset({'FlowMatchEulerDiscreteScheduler'})

# family: (text-to-video pipeline, image-to-video pipeline, transformer) Diffusers class names
_FAMILY_CLASSES = {
    'hunyuan-video-15': (
        'HunyuanVideo15Pipeline', 'HunyuanVideo15ImageToVideoPipeline',
        'HunyuanVideo15Transformer3DModel'),
    'motif-video': (
        'MotifVideoPipeline', 'MotifVideoImage2VideoPipeline',
        'MotifVideoTransformer3DModel'),
    'kandinsky5-video': (
        'Kandinsky5T2VPipeline', 'Kandinsky5I2VPipeline',
        'Kandinsky5Transformer3DModel'),
}

SINGLE_IMAGE_VIDEO_FAMILIES = frozenset(_FAMILY_CLASSES)
"""Video families served by this module."""

SINGLE_IMAGE_VIDEO_MODEL_TYPES = frozenset({
    _enums.ModelType.HUNYUAN_VIDEO_15,
    _enums.ModelType.MOTIF_VIDEO,
    _enums.ModelType.KANDINSKY5_VIDEO})
"""``--model-type`` values served by this module."""


def _family_classes(family):
    classes = _FAMILY_CLASSES.get(family)
    if classes is None:
        raise _pipelines.UnsupportedPipelineConfigError(f'Unknown video family {family!r}.')
    return classes


def _single_image_video_pipeline_class(family, mode):
    import diffusers
    text_to_video, image_to_video, _ = _family_classes(family)
    return getattr(diffusers, image_to_video if mode == 'image-to-video' else text_to_video)


def _single_image_video_transformer_class(family):
    import diffusers
    return getattr(diffusers, _family_classes(family)[2])


def _classify_single_image_video_seed(model_type, parsed):
    if parsed is None or not parsed.images:
        return 'text-to-video'
    if parsed.multi_image_mode or len(parsed.images) != 1:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'{_enums.get_model_type_string(model_type)} accepts one still condition image.')
    if parsed.end_image or parsed.reference_images or parsed.control_images:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'{_enums.get_model_type_string(model_type)} accepts only a single image condition.')
    return 'image-to-video'


def _resolve_single_image_video_load(model_type, index):
    model_type = _enums.get_model_type_enum(model_type)
    family, prefix = {
        _enums.ModelType.HUNYUAN_VIDEO_15: ('hunyuan-video-15', 'HunyuanVideo15'),
        _enums.ModelType.MOTIF_VIDEO: ('motif-video', 'MotifVideo'),
        _enums.ModelType.KANDINSKY5_VIDEO: ('kandinsky5-video', 'Kandinsky5'),
    }[model_type]
    name = str((index or {}).get('_class_name') or '')
    if name.startswith(prefix):
        image_to_video_markers = ('I2V', 'ImageToVideo', 'Image2Video')
        mode = ('image-to-video'
                if any(marker in name for marker in image_to_video_markers)
                else 'text-to-video')
        return family, _single_image_video_pipeline_class(family, mode), False
    raise _pipelines.UnsupportedPipelineConfigError(
        f'This repository is not a {prefix} pipeline '
        f'(model_index.json _class_name is {name!r}).')


def _motif_default_timesteps(pipe, steps: int, call_kwargs: dict, call_parameters: dict):
    """Read Motif's normal timestep schedule without leaving scheduler state changed."""
    scheduler = pipe.scheduler
    set_timesteps = scheduler.set_timesteps
    set_parameters = inspect.signature(set_timesteps).parameters
    if 'timesteps' not in set_parameters:
        raise _pipelines.UnsupportedPipelineConfigError(
            'The loaded Motif scheduler does not accept custom timesteps.')

    def call_default(name, fallback):
        parameter = call_parameters.get(name)
        if parameter is None or parameter.default is inspect.Parameter.empty:
            return fallback
        return parameter.default

    height = int(call_kwargs.get('height', call_default('height', 736)))
    width = int(call_kwargs.get('width', call_default('width', 1280)))
    num_frames = int(call_kwargs.get('num_frames', call_default('num_frames', 121)))
    vae_spatial = int(getattr(pipe, 'vae_scale_factor_spatial', 8) or 8)
    vae_temporal = int(getattr(pipe, 'vae_scale_factor_temporal', 4) or 4)
    transformer = getattr(pipe, 'transformer', None)
    transformer_config = getattr(transformer, 'config', None)
    spatial_patch = int(getattr(
        pipe, 'transformer_spatial_patch_size',
        getattr(transformer_config, 'patch_size', 2)) or 2)
    temporal_patch = int(getattr(pipe, 'transformer_temporal_patch_size', 1) or 1)
    latent_height = height // vae_spatial
    latent_width = width // vae_spatial
    latent_frames = (num_frames - 1) // vae_temporal + 1
    sequence_length = (
        (latent_frames // temporal_patch) *
        (latent_height // spatial_patch) *
        (latent_width // spatial_patch))

    config = scheduler.config
    calculate_shift = getattr(pipe.__class__.__call__, '__globals__', {}).get('calculate_shift')
    if callable(calculate_shift):
        mu = calculate_shift(
            sequence_length,
            config.get('base_image_seq_len', 256),
            config.get('max_image_seq_len', 4096),
            config.get('base_shift', 0.5),
            config.get('max_shift', 1.15))
    else:
        base_seq_len = config.get('base_image_seq_len', 256)
        max_seq_len = config.get('max_image_seq_len', 4096)
        base_shift = config.get('base_shift', 0.5)
        max_shift = config.get('max_shift', 1.15)
        mu = base_shift + (sequence_length - base_seq_len) * (
            (max_shift - base_shift) / (max_seq_len - base_seq_len))

    kwargs = {}
    if 'device' in set_parameters:
        kwargs['device'] = 'cpu'
    if 'mu' in set_parameters:
        kwargs['mu'] = mu
    is_multistep = scheduler.__class__.__name__ in {
        'DPMSolverMultistepScheduler', 'UniPCMultistepScheduler', 'FlowUniPCMultistepScheduler'}
    if not is_multistep and 'sigmas' in set_parameters:
        kwargs['sigmas'] = numpy.linspace(1.0, 1.0 / steps, steps)

    try:
        with _preserved_scheduler_state(scheduler):
            set_timesteps(steps, **kwargs)
            return _integer_timesteps(getattr(scheduler, 'timesteps', None))
    except Exception as e:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Could not read Motif-Video scheduler timesteps for a --timesteps expression.') from e


@contextlib.contextmanager
def _vae_encode_in_vae_dtype(vae):
    """Run ``vae.encode`` in the VAE's dtype and return latents in the caller's.

    Motif-Video uses the Wan VAE, which dgenerate keeps in float32. Its
    image-to-video pipeline casts the conditioning frame to the transformer
    dtype (usually bfloat16) and encodes it directly, which a float32 VAE
    rejects. The Wan pipelines cast to ``vae.dtype`` for the encode and back
    to the transformer dtype afterward; this does the same for Motif.
    """
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from diffusers.models.modeling_outputs import AutoencoderKLOutput

    shadowed = vae.__dict__.get('encode')
    original = vae.encode

    def encode(x, *args, **kwargs):
        output = original(x.to(vae.dtype), *args, **kwargs)
        dist = output.latent_dist if isinstance(output, AutoencoderKLOutput) else output[0]
        dist = DiagonalGaussianDistribution(
            dist.parameters.to(x.dtype), deterministic=dist.deterministic)
        if isinstance(output, AutoencoderKLOutput):
            return AutoencoderKLOutput(latent_dist=dist)
        return (dist,) + tuple(output[1:])

    vae.encode = encode
    try:
        yield
    finally:
        if shadowed is None:
            del vae.encode
        else:
            vae.encode = shadowed


def _call_single_image_video(wrapper, user_args):
    mode ='image-to-video' if user_args.images else 'text-to-video'
    pipe, held = _vp._video_pipeline(wrapper, mode, user_args.scheduler_uri)
    family = held.family
    call_parameters = inspect.signature(pipe.__call__).parameters
    positive, negative = _vp._prompt_text(user_args)
    width, height = _vp._size(user_args)
    default_fps = 15.0 if family == 'hunyuan-video-15' else 24.0
    fps = float(user_args.video_fps or default_fps)
    kwargs = {
        'prompt': positive,
        'generator': _vp._generator(wrapper, user_args),
        'output_type': 'np' if family == 'hunyuan-video-15' else 'pil',
    }
    if negative:
        kwargs['negative_prompt'] = negative
    if width is not None:
        width, height = _vp._force_output_size_alignment(width, height, 16, family)
    if mode == 'image-to-video':
        source = user_args.images[0]
        width, height = _vp.conditioning_canvas(
            source.size, width, height, user_args.aspect_correct, 16)
        user_args.images[0] = _vp.resize_media([source], (width, height))[0]
        kwargs['image'] = user_args.images[0]
    if width is not None and {'width', 'height'} <= call_parameters.keys():
        kwargs.update(width=width, height=height)
    if user_args.video_length is not None:
        temporal = int(getattr(pipe, 'vae_scale_factor_temporal', 4) or 4)
        kwargs['num_frames'] = _vp.wan_num_frames(user_args.video_length, fps, temporal)

    guidance_default = 6.0 if family == 'hunyuan-video-15' else 5.0
    guidance = float(_types.default(user_args.guidance_scale, guidance_default))
    if family == 'kandinsky5-video':
        kwargs['guidance_scale'] = guidance
    else:
        guider = getattr(pipe, 'guider', None)
        if guider is None or not hasattr(guider, 'new'):
            raise _pipelines.UnsupportedPipelineConfigError(
                f'{family} pipeline does not expose a configurable guider.')
        pipe.guider = guider.new(guidance_scale=guidance)

    steps = int(_types.default(user_args.inference_steps, 50))
    kwargs['num_inference_steps'] = steps
    custom_timesteps = None
    if user_args.timesteps is not None:
        if 'timesteps' not in inspect.signature(pipe.scheduler.set_timesteps).parameters:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'The loaded {family} scheduler does not support custom timesteps.')
        if isinstance(user_args.timesteps, str):
            if family == 'motif-video':
                base_timesteps = _motif_default_timesteps(pipe, steps, kwargs, call_parameters)
            else:
                base_timesteps = _default_video_timesteps(pipe, family, steps)
            custom_timesteps = _eval_video_timesteps_expression(
                user_args.timesteps, base_timesteps)
        else:
            custom_timesteps = [int(value) for value in user_args.timesteps]
        kwargs['num_inference_steps'] = len(custom_timesteps)
        user_args.inference_steps = len(custom_timesteps)
    if user_args.max_sequence_length is not None:
        if 'max_sequence_length' not in call_parameters:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'--max-sequence-length is not supported by {family}.')
        kwargs['max_sequence_length'] = int(user_args.max_sequence_length)
    if user_args.sigmas is not None:
        if 'sigmas' not in call_parameters:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'--sigmas is not supported by {family}; its pipeline accepts a timestep schedule instead.')
        if isinstance(user_args.sigmas, str):
            with _preserved_scheduler_state(pipe.scheduler) as scheduler:
                scheduler.set_timesteps(kwargs['num_inference_steps'])
                kwargs['sigmas'] = _eval_sigma_expression(user_args.sigmas, scheduler.sigmas)
        else:
            kwargs['sigmas'] = [float(value) for value in user_args.sigmas]
        user_args.inference_steps = len(kwargs['sigmas'])

    _vp._set_video_vae_slicing(pipe, bool(user_args.vae_slicing))
    with contextlib.ExitStack() as stack:
        if custom_timesteps is not None:
            stack.enter_context(_forced_scheduler_timesteps(pipe.scheduler, custom_timesteps))
        if family == 'motif-video' and mode == 'image-to-video':
            stack.enter_context(_vae_encode_in_vae_dtype(pipe.vae))
        output = _vp._invoke(wrapper, pipe, kwargs, user_args=user_args)
    frames = _vp.frames_from_output(getattr(output, 'frames', output))
    audio = _vp.audio_to_numpy(getattr(output, 'audio', None))
    sample_rate = getattr(pipe, 'audio_sample_rate', None) if audio is not None else None
    return frames, audio, int(sample_rate) if sample_rate else None, fps
