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

import collections.abc
import gc
import importlib
import inspect
import os
import warnings

import PIL.Image
import numpy
import torch

import dgenerate.eval as _eval
import dgenerate.exceptions as _d_exceptions
import dgenerate.hfhub as _hfhub
import dgenerate.mediainput as _mediainput
import dgenerate.memoize as _d_memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.schedulers as _schedulers
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.pipelinewrapper.util as _util
import dgenerate.types as _types
from dgenerate.memoize import memoize as _memoize


_LTX_FALLBACK_EXTRA_DIRS = ('audio_vae', 'vocoder')

_LTX_CORE_INDEX_NAMES = frozenset({
    'transformer',
    'transformer_2',
    'unet',
    'vae',
    'text_encoder',
    'text_encoder_2',
    'text_encoder_3',
    'image_encoder',
    'scheduler',
    'tokenizer',
    'tokenizer_2',
    'tokenizer_3',
    'feature_extractor',
    'safety_checker',
})


def extra_weight_directories_from_index(index: dict | None) -> list[str]:
    """
    Repo folders that belong on the LTX pipeline cache estimate besides
    transformer, VAE, and text encoders.

    :param index: ``model_index.json`` dict, or ``None``
    :return: folder names such as ``audio_vae`` and ``vocoder``
    """
    if not index:
        return list(_LTX_FALLBACK_EXTRA_DIRS)
    extras = []
    for name, spec in index.items():
        if name.startswith('_') or name in _LTX_CORE_INDEX_NAMES:
            continue
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            extras.append(name)
    return extras


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


class _VideoPipeline:
    """A video pipeline kept in the shared diffusion pipeline cache."""

    def __init__(self, pipeline, family: str, moe: bool = False):
        self.pipeline = pipeline
        self.family = family
        self.moe = moe
        self.reference_downscale_factor = None


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

    if model_type == _enums.ModelType.LTX:
        from . import ltx
        return ltx._classify_ltx(parsed, ic_lora)
    if model_type == _enums.ModelType.WAN:
        from . import wan
        return wan._classify_wan(parsed)
    if model_type == _enums.ModelType.WAN_ANIMATE:
        from . import wan
        return wan._classify_wan_animate(parsed)

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


def load_rgb_frames(path: str,
                    local_files_only: bool = False,
                    resize_resolution: _types.OptionalSize = None,
                    aspect_correct: bool = True,
                    align: int = 1,
                    frame_start: int = 0,
                    frame_end: _types.OptionalInteger = None,
                    max_frames: _types.OptionalInteger = None) -> tuple[list[PIL.Image.Image], float | None]:
    """
    Open a local path or URL as a list of RGB frames.

    A still image gives one frame. Videos and animated images are sliced by
    ``frame_start`` and ``frame_end`` (inclusive), then cut to ``max_frames``.

    :param path: file path or URL
    :param local_files_only: refuse to download
    :param resize_resolution: optional resize
    :param aspect_correct: preserve aspect ratio when resizing
    :param align: pixel alignment, ``1`` disables it
    :param frame_start: first frame index to keep
    :param frame_end: last frame index to keep, ``None`` for the end of the file
    :param max_frames: stop after this many frames, ``None`` for no limit
    :raise UnsupportedPipelineConfigError: if the slice selects no frames
    :return: ``(frames, fps)``. ``fps`` is ``None`` for a still image.
    """
    mime_type, stream = _mediainput.fetch_media_data_stream(
        path, local_files_only=local_files_only)
    try:
        if _mediainput.mimetype_is_static_image(mime_type):
            return [_mediainput.create_image(
                stream,
                file_source=path,
                resize_resolution=resize_resolution,
                aspect_correct=aspect_correct,
                align=align)], None

        if not (_mediainput.mimetype_is_video(mime_type) or
                _mediainput.mimetype_is_animated_image(mime_type)):
            raise _mediainput.UnknownMimetypeError(
                f'Expected an image, animated image, or video for "{path}", '
                f'got mimetype "{mime_type}".')

        frames = []
        try:
            with _mediainput.create_animation_reader(
                    mime_type,
                    file_source=path,
                    file=stream,
                    resize_resolution=resize_resolution,
                    aspect_correct=aspect_correct,
                    align=align) as reader:
                fps = float(reader.fps)
                for index, frame in enumerate(reader):
                    if frame_end is not None and index > frame_end:
                        frame.close()
                        break
                    if index < frame_start:
                        frame.close()
                        continue
                    frames.append(_to_rgb_image(frame))
                    if max_frames is not None and len(frames) >= max_frames:
                        break
        except Exception:
            for frame in frames:
                frame.close()
            raise

        if not frames:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'No frames were read from "{path}" between frame {frame_start} '
                f'and frame {"end" if frame_end is None else frame_end}.')
        return frames, fps
    finally:
        stream.close()


def frames_from_output(value) -> list[PIL.Image.Image]:
    """
    Normalize a pipeline video output to a list of RGB images.

    Accepts a list of images, a batch of those lists, or an array/tensor in
    frame-major channels-last or channels-first layout. A batch uses the first clip.

    :param value: pipeline video output
    :return: RGB frames
    """
    if value is None:
        raise _pipelines.UnsupportedPipelineConfigError(
            'The video pipeline did not return frames.')

    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()

    if isinstance(value, numpy.ndarray):
        return _frames_from_array(value)

    if isinstance(value, list):
        if not value:
            raise _pipelines.UnsupportedPipelineConfigError(
                'The video pipeline returned an empty frame list.')
        head = value[0]
        if isinstance(head, PIL.Image.Image):
            return [_to_rgb_image(frame) for frame in value]
        if isinstance(head, list):
            return frames_from_output(head)
        if torch.is_tensor(head) or isinstance(head, numpy.ndarray):
            array = head.detach().cpu().numpy() if torch.is_tensor(head) else numpy.asarray(head)
            if array.ndim >= 4:
                return frames_from_output(head)
            return [_array_frame_to_image(frame) for frame in value]
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Cannot read video frames from a list of {type(head).__name__}.')

    raise _pipelines.UnsupportedPipelineConfigError(
        f'Cannot read video frames from {type(value).__name__}.')


def audio_to_numpy(audio) -> numpy.ndarray | None:
    """
    Normalize pipeline audio to a float32 array of shape ``(channels, samples)``.

    At most two channels are kept. ``None`` and empty audio return ``None``.

    :param audio: tensor or array, or ``None``
    :return: planar audio, or ``None``
    """
    if audio is None:
        return None

    if torch.is_tensor(audio):
        audio = audio.detach().float().cpu().numpy()

    audio = numpy.asarray(audio, dtype=numpy.float32)
    if audio.size == 0:
        return None
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    if audio.ndim != 2:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Cannot mux audio of shape {audio.shape}.')

    if audio.shape[0] > 2 and audio.shape[1] <= 2:
        audio = numpy.transpose(audio, (1, 0))
    if audio.shape[0] > 2:
        audio = audio[:2]
    return numpy.ascontiguousarray(audio, dtype=numpy.float32)


def generate(wrapper, user_args) -> tuple[list[PIL.Image.Image], numpy.ndarray | None, int | None, float]:
    """
    Run the video pipeline selected by ``wrapper.model_type``.

    :param wrapper: :py:class:`dgenerate.pipelinewrapper.DiffusionPipelineWrapper`
    :param user_args: :py:class:`dgenerate.pipelinewrapper.DiffusionArguments`
    :return: ``(frames, audio, sample_rate, fps)``. ``audio`` and ``sample_rate`` may be ``None``.
    """
    model_type = wrapper.model_type
    try:
        if model_type == _enums.ModelType.LTX:
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


def _to_rgb_image(image: PIL.Image.Image) -> PIL.Image.Image:
    if image.mode == 'RGB':
        return image
    converted = image.convert('RGB')
    if converted is not image:
        image.close()
    return converted


def _array_frame_to_image(frame) -> PIL.Image.Image:
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    frame = numpy.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = numpy.transpose(frame, (1, 2, 0))
    if frame.dtype != numpy.uint8:
        frame = frame.astype(numpy.float32)
        finite = frame[numpy.isfinite(frame)]
        if finite.size and float(finite.max()) <= 1.0:
            frame = frame * 255.0
        frame = numpy.clip(frame, 0, 255).astype(numpy.uint8)
    if frame.ndim == 2:
        return PIL.Image.fromarray(frame, mode='L').convert('RGB')
    if frame.shape[-1] == 1:
        return PIL.Image.fromarray(frame.squeeze(-1), mode='L').convert('RGB')
    if frame.shape[-1] == 4:
        return PIL.Image.fromarray(frame, mode='RGBA').convert('RGB')
    return PIL.Image.fromarray(frame[..., :3], mode='RGB')


def _frames_from_array(array: numpy.ndarray) -> list[PIL.Image.Image]:
    array = numpy.asarray(array)
    if array.ndim == 5:
        array = array[0]
    if array.ndim == 3:
        return [_array_frame_to_image(array)]
    if array.ndim != 4:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Cannot read video frames from an array of shape {array.shape}.')

    if array.shape[-1] in (1, 3, 4):
        frames = (array[index] for index in range(array.shape[0]))
    elif array.shape[1] in (1, 3, 4):
        frames = (numpy.transpose(array[index], (1, 2, 0)) for index in range(array.shape[0]))
    else:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Cannot read video frames from an array of shape {array.shape}.')
    return [_array_frame_to_image(frame) for frame in frames]


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


def _require_multiple_of_32(width: int, height: int, model_name: str):
    if width % 32 or height % 32:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'{model_name} requires --output-size dimensions divisible by 32. '
            f'Got {width}x{height}.')


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


def _pretrained_kwargs(revision, variant, subfolder, local_files_only, auth_token, dtype, include_dtype):
    kwargs = {'local_files_only': local_files_only}
    if revision:
        kwargs['revision'] = revision
    if variant:
        kwargs['variant'] = variant
    if subfolder:
        kwargs['subfolder'] = subfolder
    if auth_token:
        kwargs['token'] = auth_token
    if include_dtype:
        torch_dtype = _enums.get_torch_dtype(dtype)
        if torch_dtype is not None:
            kwargs['torch_dtype'] = torch_dtype
    return kwargs


def _enable_vae_tiling(pipe):
    vae = getattr(pipe, 'vae', None)
    if vae is not None and hasattr(vae, 'enable_tiling'):
        vae.enable_tiling()


def _set_ltx_vae_slicing(pipe, enabled: bool):
    for name in ('vae', 'audio_vae'):
        vae = getattr(pipe, name, None)
        if vae is None:
            continue
        if enabled and hasattr(vae, 'enable_slicing'):
            vae.enable_slicing()
        elif not enabled and hasattr(vae, 'disable_slicing'):
            vae.disable_slicing()


def _offload_ltx(pipe, device, model_cpu_offload, sequential_cpu_offload, model_group_offload=False):
    if sequential_cpu_offload:
        _pipelines.enable_sequential_cpu_offload(pipe, device)
    elif model_cpu_offload:
        _pipelines.enable_model_cpu_offload(pipe, device)
    elif model_group_offload:
        _pipelines.enable_group_offload(pipe, device)


def _video_transformer_class(model_type, family: str = 'ltx2'):
    model_type = _enums.get_model_type_enum(model_type)
    if model_type == _enums.ModelType.LTX:
        if family == 'ltx':
            from diffusers import LTXVideoTransformer3DModel
            return LTXVideoTransformer3DModel
        from diffusers import LTX2VideoTransformer3DModel
        return LTX2VideoTransformer3DModel
    if _enums.model_type_is_wan_family(model_type):
        from . import wan
        return wan._wan_transformer_class(model_type, family)
    raise _pipelines.UnsupportedPipelineConfigError(
        f'{_enums.get_model_type_string(model_type)} does not take --transformer.')


def _load_replacement_transformer(model_type, transformer_uri, dtype, variant,
                                  auth_token, local_files_only, quantizer_uri=None,
                                  quantizer_map=None, device=None, offload=False,
                                  component_name='transformer', config_repo=None,
                                  family: str = 'ltx2'):
    if not isinstance(dtype, _enums.DataType):
        dtype = _enums.DataType.AUTO
    parsed = _uris.TransformerUri.parse(transformer_uri)
    if (not _hfhub.is_gguf_model(parsed.model)
            and quantizer_uri and not parsed.quantizer
            and _quantize_component(component_name, quantizer_uri, quantizer_map)):
        parsed.quantizer = quantizer_uri
    _messages.debug_log(f'Loading replacement video transformer "{transformer_uri}".')
    return parsed.load(
        variant_fallback=variant,
        dtype_fallback=dtype,
        use_auth_token=auth_token,
        local_files_only=local_files_only,
        device_map=_quantized_device_map(
            device, offload, parsed.quantizer) if parsed.quantizer else None,
        config=config_repo,
        config_subfolder=component_name,
        transformer_class=_video_transformer_class(model_type, family))


def _apply_video_loras(pipe, model_type, lora_uris, fuse_scale, auth_token, local_files_only):
    if not lora_uris:
        return
    del model_type
    _uris.LoRAUri.load_on_pipeline(
        pipeline=pipe,
        uris=lora_uris,
        fuse_scale=1.0 if fuse_scale is None else fuse_scale,
        use_auth_token=auth_token,
        local_files_only=local_files_only,
        fuse=True)


_DEFAULT_QUANT_NAMES = {
    'transformer',
    'transformer_2',
    'text_encoder',
    'text_encoder_2',
    'text_encoder_3',
    'image_encoder',
    'connectors',
}

# Loaded as None so from_pretrained does not pull the unused Gemma
# prompt enhancer (about 9.5 GiB) onto the GPU.
_LTX_SKIP_OPTIONAL_MODULES = {
    'prompt_enhancer': None,
}


def _quantize_component(name: str, quantizer_uri, quantizer_map) -> bool:
    if not quantizer_uri:
        return False
    if quantizer_map is None:
        return name in _DEFAULT_QUANT_NAMES
    return name in quantizer_map


def _quantizer_config(quantizer_uri, dtype, transformers_module: bool):
    parsed = _uris.get_quantizer_uri_class(quantizer_uri).parse(quantizer_uri)
    torch_dtype = _enums.get_torch_dtype(dtype) if isinstance(dtype, _enums.DataType) else None
    if transformers_module and hasattr(parsed, 'to_transformers_config'):
        return parsed.to_transformers_config(torch_dtype)
    return parsed.to_config(torch_dtype)


def _resolve_index_class(library: str, class_name: str):
    import diffusers.pipelines as pipelines

    if hasattr(pipelines, library):
        module = importlib.import_module(f'diffusers.pipelines.{library}')
    else:
        module = importlib.import_module(library)
    return getattr(module, class_name)


def _load_quantized_module(component_class, model_path, subfolder, revision, variant,
                           dtype, quantizer_uri, auth_token, local_files_only, device_map,
                           offload=False):
    module_name = getattr(component_class, '__module__', '')
    transformers_module = module_name.startswith('transformers')
    config = _quantizer_config(quantizer_uri, dtype, transformers_module)
    from dgenerate.pipelinewrapper.quant_skips import apply_architecture_quant_skips
    apply_architecture_quant_skips(config, component_class)
    torch_dtype = _enums.get_torch_dtype(dtype) if isinstance(dtype, _enums.DataType) else None
    _messages.debug_log(
        f'Quantizing {component_class.__name__} from "{model_path}" '
        f'subfolder "{subfolder}" with "{quantizer_uri}".')
    load_kwargs = {
        'subfolder': subfolder or '',
        'revision': revision,
        'variant': variant,
        'token': auth_token,
        'local_files_only': local_files_only,
        'quantization_config': config,
        'device_map': device_map,
        'low_cpu_mem_usage': True,
    }
    if torch_dtype is not None:
        if transformers_module:
            load_kwargs['dtype'] = torch_dtype
        else:
            load_kwargs['torch_dtype'] = torch_dtype
    with _hfhub.with_hf_errors_as_model_not_found():
        module = component_class.from_pretrained(model_path, **load_kwargs)
    # 8-bit BnB weights cannot leave the GPU. Pin .to() like still pipelines.
    if _util.is_loaded_in_8bit_bnb(module):
        _pipelines._disable_to(module)
    elif offload:
        module.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return module


def _pin_video_8bit_modules(pipe):
    """Disable ``.to()`` on 8-bit BnB modules after the video pipeline is built."""
    for module in _pipelines.get_pipeline_modules(pipe).values():
        if _util.is_loaded_in_8bit_bnb(module):
            _pipelines._disable_to(module)


def _quantized_device_map(device, offload: bool, quantizer_uri=None):
    # bitsandbytes rejects a CPU or disk device_map, so it has to be
    # created on the GPU. SDNQ weights can be quantized on the CPU and
    # streamed later, which is what keeps a 16GB card from filling up.
    del offload
    if not quantizer_uri or not str(quantizer_uri).split(';', 1)[0] == 'bnb':
        return None
    if device is None or str(device) == 'cpu':
        return None
    return {'': str(device)}


def _quantized_classic_modules(model_path, revision, variant, subfolder, dtype,
                               quantizer_uri, quantizer_map, auth_token,
                               local_files_only, device, offload, skip_names=()):
    if not quantizer_uri:
        return {}
    index = _util.fetch_model_index_dict(
        model_path,
        subfolder=subfolder,
        revision=revision,
        use_auth_token=auth_token,
        local_files_only=local_files_only)
    device_map = _quantized_device_map(device, offload, quantizer_uri)
    modules = {}
    for name, spec in index.items():
        if name.startswith('_') or name in skip_names:
            continue
        if not isinstance(spec, (list, tuple)) or len(spec) != 2:
            continue
        if not _quantize_component(name, quantizer_uri, quantizer_map):
            continue
        component_class = _resolve_index_class(spec[0], spec[1])
        folder = f'{subfolder}/{name}' if subfolder else name
        modules[name] = _load_quantized_module(
            component_class, model_path, folder, revision, variant, dtype,
            quantizer_uri, auth_token, local_files_only, device_map, offload=offload)
    return modules


def pipeline_for_mode(pipe, mode: str, family: str = 'ltx2'):
    """
    Return a video pipeline of the requested class that shares ``pipe.components``.

    ``from_pipe`` is not used: it calls ``.to(dtype=...)``, which raises on
    quantized modules (SDNQ / bitsandbytes).

    :param pipe: cached video pipeline
    :param mode: ``ltx-*`` or ``wan-*`` mode name
    :return: pipeline of the class for ``mode``
    """
    if str(mode).startswith('wan'):
        from . import wan
        if str(family).startswith('wan-animate-2'):
            wan._require_image_encoder(pipe, 'wan-animate-2')
            return pipe
        cls = wan._wan_pipeline_class(mode, family)
        wan._require_image_encoder(pipe, mode)
    else:
        cls = _ltx_pipeline_class(mode, family)
    if type(pipe) is cls:
        return pipe
    accepted = set(inspect.signature(cls.__init__).parameters)
    accepted.discard('self')
    kwargs = {name: value for name, value in pipe.components.items() if name in accepted}
    return cls(**kwargs)


def _ltx_extra_weight_directories(model_path, revision, subfolder, auth_token, local_files_only):
    try:
        index = _util.fetch_model_index_dict(
            model_path,
            subfolder=subfolder,
            revision=revision,
            use_auth_token=auth_token,
            local_files_only=local_files_only)
    except Exception:
        return list(_LTX_FALLBACK_EXTRA_DIRS)
    return extra_weight_directories_from_index(index)


def _cache_kwargs(wrapper) -> dict:
    quantizer_map = wrapper.quantizer_map
    return {
        'model_path': wrapper.model_path,
        'model_type': wrapper.model_type,
        'revision': wrapper._revision,
        'variant': wrapper._variant,
        'subfolder': wrapper._subfolder,
        'dtype': wrapper._dtype,
        'device': wrapper.device,
        'model_cpu_offload': bool(wrapper.model_cpu_offload),
        'sequential_cpu_offload': bool(wrapper.model_sequential_offload),
        'model_group_offload': bool(getattr(wrapper, 'model_group_offload', False)),
        'local_files_only': bool(wrapper._local_files_only),
        'auth_token': wrapper._auth_token,
        'quantizer_uri': wrapper.quantizer_uri,
        'quantizer_map': tuple(quantizer_map) if quantizer_map else None,
    }


def _video_on_hit(key, hit):
    _d_memoize.simple_cache_hit_debug('Torch Video Pipeline', key, hit.pipeline)


def _video_on_create(key, new):
    _d_memoize.simple_cache_miss_debug('Torch Video Pipeline', key, new.pipeline)


def _resolve_video_load(model_type, index, ltx_ic_lora_uri):
    model_type = _enums.get_model_type_enum(model_type)
    if _enums.model_type_is_wan_family(model_type):
        from . import wan
        return wan._resolve_wan_load(model_type, index)
    from . import ltx
    family = ltx.ltx_family_from_index(index)
    mode = 'ltx-control' if ltx_ic_lora_uri else 'ltx-txt'
    return family, _ltx_pipeline_class(mode, family), False


def _text_encoder_slots(index: dict | None) -> list[str]:
    """Ordered ``text_encoder*`` component names from ``model_index.json``."""
    if not index:
        return []
    return [
        name for name, spec in sorted(index.items())
        if name.startswith('text_encoder')
        and isinstance(spec, (list, tuple))
        and len(spec) >= 2
        and spec[0] is not None
    ]


def _inject_video_text_encoders(index,
                                text_encoder_uris,
                                dtype,
                                auth_token,
                                local_files_only,
                                device,
                                offload):
    """
    Load ``--text-encoders`` replacements for a video pipeline.

    ``+`` keeps the checkpoint default for that slot. ``null`` skips loading
    that encoder. Trailing omitted slots keep their defaults.
    """
    if not text_encoder_uris:
        return {}
    slots = _text_encoder_slots(index)
    if not slots:
        raise _pipelines.UnsupportedPipelineConfigError(
            '--text-encoders cannot be used with this video model; '
            'the checkpoint has no text encoder.')
    uris = list(text_encoder_uris)
    if len(uris) > len(slots):
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Too many --text-encoders values for this video model '
            f'(got {len(uris)}, max {len(slots)}).')
    injected = {}
    for name, uri in zip(slots, uris):
        if _pipelines._text_encoder_default(uri):
            continue
        if _pipelines._text_encoder_null(uri):
            injected[name] = None
            _messages.debug_log(f'Video pipeline skipping "{name}" (--text-encoders null).')
            continue
        parsed = _uris.TextEncoderUri.parse(uri)
        # Global --quantizer only applies when the slot is left as +.
        # Put quantizer= on the URI to quantize a replacement encoder.
        device_map = (
            _quantized_device_map(device, offload, parsed.quantizer)
            if parsed.quantizer else None)
        injected[name] = parsed.load(
            dtype_fallback=dtype,
            use_auth_token=auth_token,
            local_files_only=local_files_only,
            device_map=device_map)
        _messages.debug_log(f'Video pipeline using --text-encoders for "{name}": "{uri}".')
    return injected


@_memoize(_pipelines._pipeline_cache,
          exceptions={'local_files_only'},
          hasher=_d_memoize.args_cache_key,
          extra_identities=[lambda held: held.pipeline],
          on_hit=_video_on_hit,
          on_create=_video_on_create)
def _create_cached_video_pipeline(model_path,
                                  model_type,
                                  revision,
                                  variant,
                                  subfolder,
                                  dtype,
                                  device,
                                  model_cpu_offload,
                                  sequential_cpu_offload,
                                  model_group_offload,
                                  local_files_only,
                                  auth_token,
                                  transformer_uri=None,
                                  lora_uris=None,
                                  lora_fuse_scale=None,
                                  quantizer_uri=None,
                                  quantizer_map=None,
                                  ltx_ic_lora_uri=None,
                                  ic_lora_downscale=None,
                                  wan_second_transformer_uri=None,
                                  vae_uri=None,
                                  text_encoder_uris=None):
    all_lora_uris = list(lora_uris or ()) + ([ltx_ic_lora_uri] if ltx_ic_lora_uri else [])
    index = _util.fetch_model_index_dict(
        model_path,
        subfolder=subfolder,
        revision=revision,
        use_auth_token=auth_token,
        local_files_only=local_files_only)
    extra_dirs = extra_weight_directories_from_index(index)
    family, pipeline_class, moe = _resolve_video_load(model_type, index, ltx_ic_lora_uri)
    if transformer_uri and wan_second_transformer_uri is None and index.get('transformer_2'):
        extra_dirs = list(extra_dirs) + ['transformer_2']
    if dtype is _enums.DataType.AUTO:
        detected_dtype = _util.auto_dtype(
            model_path,
            revision=revision,
            subfolder=subfolder,
            use_auth_token=auth_token,
            local_files_only=local_files_only,
            model_index=index)
        if detected_dtype is not None:
            _messages.debug_log(f'--dtype auto selected: {_enums.get_data_type_string(detected_dtype)}')
            dtype = detected_dtype
    te_slots = _text_encoder_slots(index)
    te_uris = list(text_encoder_uris or ())
    all_text_encoders_replaced = bool(
        te_slots
        and len(te_uris) >= len(te_slots)
        and all(
            not _pipelines._text_encoder_default(uri)
            for uri in te_uris[:len(te_slots)]))
    estimate = _pipelines.estimate_pipeline_cache_footprint(
        model_path=model_path,
        model_type=model_type,
        revision=revision or 'main',
        variant=variant,
        subfolder=subfolder,
        include_unet_or_transformer=not transformer_uri,
        include_vae=not vae_uri,
        include_text_encoders=not all_text_encoders_replaced,
        lora_uris=all_lora_uris,
        include_directories=extra_dirs,
        auth_token=auth_token,
        local_files_only=local_files_only)
    _pipelines._enforce_pipeline_cache_size(estimate)

    load_kwargs = _pretrained_kwargs(
        revision, variant, subfolder, local_files_only, auth_token, dtype,
        include_dtype=True)
    if family == 'ltx2':
        load_kwargs.update(_LTX_SKIP_OPTIONAL_MODULES)

    offload = bool(model_cpu_offload or sequential_cpu_offload or model_group_offload)
    injected = {}
    if transformer_uri:
        injected['transformer'] = _load_replacement_transformer(
            model_type, transformer_uri, dtype, variant, auth_token, local_files_only,
            quantizer_uri=quantizer_uri, quantizer_map=quantizer_map, device=device,
            offload=offload, config_repo=model_path, family=family)
        load_kwargs['transformer'] = injected['transformer']
    if wan_second_transformer_uri:
        injected['transformer_2'] = _load_replacement_transformer(
            model_type, wan_second_transformer_uri, dtype, variant, auth_token, local_files_only,
            quantizer_uri=quantizer_uri, quantizer_map=quantizer_map, device=device,
            offload=offload, component_name='transformer_2', config_repo=model_path,
            family=family)
        load_kwargs['transformer_2'] = injected['transformer_2']
    if vae_uri:
        from . import wan as _wan_mod
        parsed_vae = _uris.VAEUri.parse(vae_uri)
        # Wan VAE defaults to float32 even when replaced via --vae, unless the
        # URI sets dtype= explicitly.
        vae_dtype_fallback = dtype
        if (_enums.model_type_is_wan_family(model_type)
                and parsed_vae.dtype is None):
            vae_dtype_fallback = _wan_mod._WAN_DEFAULT_VAE_DTYPE
        injected['vae'] = parsed_vae.load(
            dtype_fallback=vae_dtype_fallback,
            use_auth_token=auth_token,
            local_files_only=local_files_only)
        load_kwargs['vae'] = injected['vae']
        _messages.debug_log(f'Video pipeline using --vae "{vae_uri}".')
    injected.update(_inject_video_text_encoders(
        index, te_uris, dtype, auth_token, local_files_only, device, offload))
    load_kwargs.update(injected)
    injected.update(_quantized_classic_modules(
        model_path, revision, variant, subfolder, dtype, quantizer_uri, quantizer_map,
        auth_token, local_files_only, device, offload,
        skip_names=frozenset(load_kwargs)))
    load_kwargs.update(injected)
    _messages.debug_log(f'Loading {pipeline_class.__name__} from "{model_path}".')
    with _hfhub.with_hf_errors_as_model_not_found():
        if str(family).startswith('wan-animate-2'):
            from . import wan
            pipe = wan.load_wan_animate_2_pipeline(
                pipeline_class, model_path, load_kwargs, injected, dtype)
        else:
            pipe = pipeline_class.from_pretrained(model_path, **load_kwargs)
    _apply_video_loras(
        pipe, model_type, all_lora_uris, lora_fuse_scale, auth_token, local_files_only)
    # Wan VAE defaults to float32. Cast before offload so streamed group
    # offload snapshots float32 weights. Skip when --vae already chose a dtype
    # (including the float32 fallback above), or when the VAE is quantized.
    if _enums.model_type_is_wan_family(model_type) and not vae_uri:
        from . import wan
        wan._set_wan_vae_dtype(pipe)
    if str(family).startswith('wan-animate-2'):
        from . import wan
        wan.place_wan_animate_2(
            pipe, device, model_cpu_offload, sequential_cpu_offload, model_group_offload)
    else:
        _offload_ltx(pipe, device, model_cpu_offload, sequential_cpu_offload, model_group_offload)
    _pin_video_8bit_modules(pipe)
    _enable_vae_tiling(pipe)
    held = _VideoPipeline(pipe, family, moe=moe)
    if ltx_ic_lora_uri:
        held.reference_downscale_factor = _ic_lora_downscale_factor(
            ltx_ic_lora_uri, ic_lora_downscale, auth_token, local_files_only)

    return held, _d_memoize.CachedObjectMetadata(size=estimate)


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
    # Same as still pipelines: scheduler is not a cache key.
    # Overlay the URI on the cached object, then wrap for mode.
    _schedulers.load_scheduler(held.pipeline, scheduler_uri)
    return pipeline_for_mode(held.pipeline, mode, held.family), held


def _parsed_ic_lora(wrapper) -> _uris.ICLoRAUri | None:
    uri = getattr(wrapper, 'ltx_ic_lora_uri', None)
    return _uris.ICLoRAUri.parse(uri) if uri else None


def _invoke(wrapper, pipe, kwargs):
    return _pipelines.call_pipeline(pipe, device=wrapper.device, **kwargs)


# LTX implementation lives in ltx.py. Re-export the names tests and
# callers already import from this package.
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
    _LTX_DEFAULT_FPS,
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
    _upsample_ltx_latents,
    _ltx_pixels_from_output,
    _waveform_from_audio_latents,
    _decode_ltx_diffusion,
    _ltx_temporal_compression,
    _trim_ltx_clip,
    _ltx_conditions,
    _ltx_image_needs_conditions,
    _ltx_resolved_strength,
    _append_extra_ltx_conditions,
    _ltx_reference_kwargs,
    _ic_lora_downscale_factor,
    _lora_file_metadata,
    _legacy_ltx_schedule_finite,
    _legacy_ltx_timesteps,
    _legacy_ltx_compression,
    _fix_legacy_ltx_condition_schedule,
    _require_legacy_ltx_schedule,
    _warn_legacy_ltx_canvas,
    _ltx_ignored_steps_warning,
    _ltx_base_sigmas,
    _eval_sigma_expression,
    _ltx_is_distilled,
)

__all__ = _types.module_all()
