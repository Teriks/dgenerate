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
Video generation for model types that return a whole clip from one pipeline call.

These models are not run once per input frame. One combination of prompt, seed,
guidance, steps, and the model family's length and fps options produces one clip.
Conditioning still comes from ``--image-seeds``, and the meaning of each slot
depends on ``--model-type``.
"""

import PIL.Image
import numpy
import torch

import dgenerate.exceptions as _d_exceptions
import dgenerate.mediainput as _mediainput
import dgenerate.memory as _memory
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.schedulers as _schedulers
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.types as _types


def apply_video_arg_rewrites(source, dest) -> None:
    """
    Copy LTX rewrite fields from the pipeline call args onto the original args.

    :param source: :py:class:`dgenerate.pipelinewrapper.DiffusionArguments` after generate
    :param dest: the original arguments object passed into the wrapper
    """
    dest.guidance_scale = source.guidance_scale
    dest.inference_steps = source.inference_steps
    dest.ltx_audio_guidance_scale = source.ltx_audio_guidance_scale
    dest.ltx_audio_guidance_rescale = source.ltx_audio_guidance_rescale


def classify_video_seed(model_type: _enums.ModelType | str,
                        parsed: _mediainput.ImageSeedParseResult | None,
                        ic_lora: bool = False) -> str:
    """
    Decide which video conditioning mode an image seed selects.

    :param model_type: video ``--model-type``
    :param parsed: parsed ``--image-seeds`` value, or ``None`` when there is no image seed
    :param ic_lora: ``--ltx-ic-lora`` was given, which makes a plain seed path the control clip
    :raise UnsupportedPipelineConfigError: if the seed does not fit the model
    :return: a mode name used by the loader
    """
    model_type = _enums.get_model_type_enum(model_type)
    _reject_still_extras(
        parsed,
        _enums.get_model_type_string(model_type),
        allow_mask=_enums.model_type_is_wan_family(model_type))
    if model_type == _enums.ModelType.WAN_ANIMATE_2:
        from . import wan
        return wan._classify_wan_animate_2(parsed)
    if model_type != _enums.ModelType.WAN_ANIMATE:
        from . import wan
        wan._reject_animate_keywords(parsed, _enums.get_model_type_string(model_type))

    if _enums.model_type_is_ltx_family(model_type):
        from . import ltx
        return ltx._classify_ltx(parsed, ic_lora)
    if model_type == _enums.ModelType.WAN:
        from . import wan
        return wan._classify_wan(parsed)
    if model_type == _enums.ModelType.WAN_ANIMATE:
        from . import wan
        return wan._classify_wan_animate(parsed)
    if model_type in SINGLE_IMAGE_VIDEO_MODEL_TYPES:
        if ic_lora:
            raise _pipelines.UnsupportedPipelineConfigError(
                '--ltx-ic-lora is only supported with --model-type ltx2.')
        return _classify_single_image_video_seed(model_type, parsed)

    raise _pipelines.UnsupportedPipelineConfigError(
        f'{_enums.get_model_type_string(model_type)} is not a video model type.')


def video_seed_slots(parsed: _mediainput.ImageSeedParseResult | None,
                     ic_lora: bool = False) -> tuple[str | None, str | None, str | None]:
    """
    Split an image seed into its opening, end, and control paths.

    With ``--ltx-ic-lora``, a plain seed path is the control clip, the same way a
    plain path is the control image when ``--control-nets`` is given.

    :param parsed: parsed ``--image-seeds`` value, or ``None``
    :param ic_lora: ``--ltx-ic-lora`` was given
    :return: ``(opening, end, control)``, each a path or ``None``
    """
    if parsed is None:
        return None, None, None
    opening = parsed.images[0] if parsed.images else None
    control = parsed.control_images[0] if parsed.control_images else None
    if ic_lora and parsed.is_single_spec and not parsed.multi_image_mode:
        opening, control = None, opening
    return opening, parsed.end_image, control


def generate(wrapper, user_args) -> tuple[list[PIL.Image.Image], numpy.ndarray | None, int | None, float]:
    """
    Run the video pipeline selected by ``wrapper.model_type``.

    :param wrapper: :py:class:`dgenerate.pipelinewrapper.DiffusionPipelineWrapper`
    :param user_args: :py:class:`dgenerate.pipelinewrapper.DiffusionArguments`
    :return: ``(frames, audio, sample_rate, fps)``. ``audio`` and ``sample_rate`` may be ``None``.
    """
    model_type = wrapper.model_type
    try:
        if _enums.model_type_is_ltx_family(model_type):
            from . import ltx
            return ltx._call_ltx(wrapper, user_args)
        if model_type == _enums.ModelType.WAN:
            from . import wan
            return wan._call_wan(wrapper, user_args)
        if model_type == _enums.ModelType.WAN_ANIMATE:
            from . import wan
            return wan._call_wan_animate(wrapper, user_args)
        if model_type == _enums.ModelType.WAN_ANIMATE_2:
            from . import wan
            return wan._call_wan_animate_2(wrapper, user_args)
        if model_type in {
                _enums.ModelType.HUNYUAN_VIDEO_15,
                _enums.ModelType.MOTIF_VIDEO,
                _enums.ModelType.KANDINSKY5_VIDEO}:
            return _call_single_image_video(wrapper, user_args)
    except _d_exceptions.TORCH_CUDA_OOM_EXCEPTIONS as e:
        _d_exceptions.raise_if_not_cuda_oom(e)
        _memory.torch_gc()
        raise _d_exceptions.OutOfMemoryError(e) from e

    raise _pipelines.UnsupportedPipelineConfigError(
        f'{_enums.get_model_type_string(model_type)} is not a video model type.')


def _reject_still_extras(parsed, model_name: str, allow_mask: bool = False):
    if parsed is None:
        return
    if parsed.mask_images and not allow_mask:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'{model_name} does not accept inpaint masks in --image-seeds.')
    if parsed.latents:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'{model_name} does not accept latents in --image-seeds.')
    if parsed.adapter_images:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'{model_name} does not accept IP adapter images.')
    if parsed.floyd_image:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'{model_name} does not accept a floyd image.')


def _image_count(parsed) -> int:
    if parsed is None or not parsed.images:
        return 0
    return len(parsed.images)


def _control_count(parsed) -> int:
    if parsed is None or not parsed.control_images:
        return 0
    return len(parsed.control_images)


def _has_end(parsed) -> bool:
    return bool(parsed is not None and parsed.end_image)


def _prompt_text(user_args) -> tuple[str, str | None]:
    prompt = user_args.prompt
    if prompt is None or not getattr(prompt, 'positive', None):
        raise _pipelines.UnsupportedPipelineConfigError('Video models require a prompt.')
    negative = prompt.negative if prompt.negative else None
    return prompt.positive, negative


def _size(user_args) -> tuple[int | None, int | None]:
    if user_args.width is None or user_args.height is None:
        return None, None
    return int(user_args.width), int(user_args.height)


def _seed(user_args) -> int:
    if user_args.seed is None:
        return _constants.DEFAULT_SEED
    return int(user_args.seed)


def _offload_requested(wrapper) -> bool:
    return bool(
        getattr(wrapper, 'model_cpu_offload', False)
        or getattr(wrapper, 'model_sequential_offload', False)
        or getattr(wrapper, 'model_group_offload', False))


def _generator(wrapper, user_args) -> torch.Generator:
    device = 'cpu' if _offload_requested(wrapper) else wrapper.device
    return torch.Generator(device=device).manual_seed(_seed(user_args))


def _set_video_vae_slicing(pipe, enabled: bool):
    for name in ('vae', 'audio_vae'):
        vae = getattr(pipe, name, None)
        if vae is None:
            continue
        if enabled and hasattr(vae, 'enable_slicing'):
            vae.enable_slicing()
        elif not enabled and hasattr(vae, 'disable_slicing'):
            vae.disable_slicing()


def _video_pipeline(wrapper, mode: str, scheduler_uri=None):
    kwargs = _cache_kwargs(wrapper)
    kwargs['transformer_uri'] = wrapper.transformer_uri
    kwargs['lora_uris'] = tuple(wrapper.lora_uris) if wrapper.lora_uris else None
    kwargs['lora_fuse_scale'] = wrapper.lora_fuse_scale
    ic_lora = _parsed_ic_lora(wrapper)
    if ic_lora is not None:
        kwargs['ltx_ic_lora_uri'] = ic_lora.lora_uri()
        kwargs['ic_lora_downscale'] = ic_lora.downscale
    second = getattr(wrapper, 'wan_second_transformer_uri', None)
    if second:
        kwargs['wan_second_transformer_uri'] = second
    vae_uri = getattr(wrapper, 'vae_uri', None)
    if vae_uri:
        kwargs['vae_uri'] = vae_uri
    text_encoder_uris = getattr(wrapper, 'text_encoder_uris', None)
    if text_encoder_uris:
        kwargs['text_encoder_uris'] = tuple(text_encoder_uris)
    held = _create_cached_video_pipeline(**kwargs)
    model_mode = getattr(held, 'model_mode', None)
    if model_mode is not None and mode != model_mode:
        expected = 'an image seed' if model_mode == 'image-to-video' else 'no image seed'
        raise _pipelines.UnsupportedPipelineConfigError(
            f'This Kandinsky 5.0 video checkpoint requires {expected}; '
            f'model_index.json selected {type(held.pipeline).__name__}.')
    # Same as still pipelines: scheduler is not a cache key.
    # Overlay the URI on the cached object, then wrap for mode.
    _schedulers.load_scheduler(held.pipeline, scheduler_uri)
    return pipeline_for_mode(held.pipeline, mode, held.family), held


def _parsed_ic_lora(wrapper) -> _uris.ICLoRAUri | None:
    uri = getattr(wrapper, 'ltx_ic_lora_uri', None)
    return _uris.ICLoRAUri.parse(uri) if uri else None


def _invoke(wrapper, pipe, kwargs, user_args):
    return wrapper._call_pipeline_with_cache_dit(
        pipeline=pipe,
        user_args=user_args,
        device=wrapper.device,
        **kwargs)


# Loading, media, family, and schedule implementations live in the
# submodules. Re-export the names tests and callers already import from this
# package.
from .wan import (
    WAN_ANIMATE_2_SCHEDULER_NAMES,
    WAN_ANIMATE_SCHEDULER_NAMES,
    WAN_SCHEDULER_NAMES,
    _call_wan,
    _call_wan_animate,
    wan_animate_2_family_from_index,
    wan_animate_family_from_index,
    wan_family_from_index,
    wan_num_frames,
    _wan_flf_clip_kind,
    _require_wan_flf_clip,
)
from .ltx import (
    LTX_SCHEDULER_NAMES,
    audio_sample_rate_from_pipeline,
    ltx_family_from_index,
    ltx_num_frames,
    _custom_ltx_condition,
    _classify_ltx,
    _ltx_pipeline_class,
    _call_ltx,
    _custom_ltx_condition_args,
    _reject_ltx2_only_options,
    _apply_ltx2_options,
    _ensure_prompt_enhancer,
    _complete_ltx_call,
    _enhance_ltx_prompt_once,
    _detach_prompt_enhancer,
    _ltx_two_stage,
    _video_latent_tensor,
    _stage_sigmas,
    _ltx_pixels_from_output,
    _waveform_from_audio_latents,
    _ltx_ignored_steps_warning,
    _ltx_base_sigmas,
    _ltx_is_distilled,
)
from .ltx.components import _upsample_ltx_latents
from .ltx.conditions import (
    _ltx_temporal_compression,
    _trim_ltx_clip,
    _ltx_conditions,
    _ltx_image_needs_conditions,
    _ltx_resolved_strength,
    _append_extra_ltx_conditions,
    _ltx_reference_kwargs,
    _ic_lora_downscale_factor,
    _lora_file_metadata,
)
from .ltx.decoder import _decode_ltx_diffusion
from .ltx.legacy import (
    _legacy_ltx_schedule_finite,
    _legacy_ltx_timesteps,
    _legacy_ltx_compression,
    _fix_legacy_ltx_condition_schedule,
    _require_legacy_ltx_schedule,
    _warn_legacy_ltx_canvas,
)
from .single_image_video import (
    HUNYUAN_VIDEO_15_SCHEDULER_NAMES,
    KANDINSKY5_VIDEO_SCHEDULER_NAMES,
    MOTIF_VIDEO_SCHEDULER_NAMES,
    SINGLE_IMAGE_VIDEO_FAMILIES,
    SINGLE_IMAGE_VIDEO_MODEL_TYPES,
    _call_single_image_video,
    _classify_single_image_video_seed,
    _motif_default_timesteps,
    _resolve_single_image_video_load,
    _single_image_video_pipeline_class,
    _single_image_video_transformer_class,
)
from .schedules import (
    _default_video_timesteps,
    _eval_sigma_expression,
    _eval_video_timesteps_expression,
    _forced_scheduler_timesteps,
)

from .media import (
    _array_frame_to_image,
    audio_to_numpy,
    conditioning_canvas,
    _force_output_size_alignment,
    _frames_from_array,
    frames_from_output,
    load_rgb_frames,
    resize_media,
    _to_rgb_image,
)
from .quantization import (
    _DEFAULT_QUANT_NAMES,
    _load_quantized_module,
    _LTX_SKIP_OPTIONAL_MODULES,
    _pin_video_8bit_modules,
    _quantize_component,
    _quantized_classic_modules,
    _quantized_device_map,
    _quantizer_config,
    _resolve_index_class,
)
from .lora import (
    _apply_video_loras,
    _load_fused_video_component,
    _preflight_quantized_video_loras,
    _preflight_quantized_video_loras_locked,
    _video_lora_component_quantizers,
    _video_lora_component_uri_cache_key,
)
from .loading import (
    _apply_video_offload,
    _cache_kwargs,
    _create_cached_video_pipeline,
    _enable_vae_tiling,
    extra_weight_directories_from_index,
    _inject_video_text_encoders,
    _load_replacement_transformer,
    _ltx_extra_weight_directories,
    _LTX_FALLBACK_EXTRA_DIRS,
    pipeline_for_mode,
    _pretrained_kwargs,
    _resolve_video_load,
    _text_encoder_slots,
    _torch_compile_for_cache,
    _VIDEO_CORE_INDEX_NAMES,
    _video_on_create,
    _video_on_hit,
    _video_transformer_class,
    _VideoPipeline,
)

__all__ = _types.module_all()
