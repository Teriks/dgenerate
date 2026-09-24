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
guidance, steps, ``--video-lengths`` and ``--video-fps`` produces one clip.
Conditioning still comes from ``--image-seeds``, and the meaning of each slot
depends on ``--model-type``.
"""

import collections.abc
import gc
import importlib
import inspect
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
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.pipelinewrapper.util as _util
import dgenerate.types as _types
from dgenerate.memoize import memoize as _memoize

_LTX_DEFAULT_FPS = 24.0

LTX_SCHEDULER_NAMES = frozenset({'FlowMatchEulerDiscreteScheduler'})

_LTX_FALLBACK_EXTRA_DIRS = ('audio_vae', 'vocoder')

_LTX_CORE_INDEX_NAMES = frozenset({
    'transformer',
    'unet',
    'vae',
    'text_encoder',
    'text_encoder_2',
    'text_encoder_3',
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
    dest.audio_guidance_scale = source.audio_guidance_scale
    dest.audio_guidance_rescale = source.audio_guidance_rescale


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


class _VideoPipeline:
    """A video pipeline kept in the shared diffusion pipeline cache."""

    def __init__(self, pipeline, family: str):
        self.pipeline = pipeline
        self.family = family


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


def classify_video_seed(model_type: _enums.ModelType | str,
                        parsed: _mediainput.ImageSeedParseResult | None) -> str:
    """
    Decide which video conditioning mode an image seed selects.

    :param model_type: video ``--model-type``
    :param parsed: parsed ``--image-seeds`` value, or ``None`` when there is no image seed
    :raise UnsupportedPipelineConfigError: if the seed does not fit the model
    :return: a mode name used by the loader
    """
    model_type = _enums.get_model_type_enum(model_type)
    _reject_still_extras(parsed, _enums.get_model_type_string(model_type))

    if model_type == _enums.ModelType.LTX:
        return _classify_ltx(parsed)

    raise _pipelines.UnsupportedPipelineConfigError(
        f'{_enums.get_model_type_string(model_type)} is not a video model type.')


def load_rgb_image(path: str,
                   local_files_only: bool = False,
                   resize_resolution: _types.OptionalSize = None,
                   aspect_correct: bool = True,
                   align: int = 1) -> PIL.Image.Image:
    """
    Open a local path or URL as an RGB image.

    :param path: file path or URL
    :param local_files_only: refuse to download
    :param resize_resolution: optional resize
    :param aspect_correct: preserve aspect ratio when resizing
    :param align: pixel alignment, ``1`` disables it
    :return: RGB image
    """
    mime_type, stream = _mediainput.fetch_media_data_stream(
        path, local_files_only=local_files_only)
    try:
        if not _mediainput.mimetype_is_static_image(mime_type):
            raise _mediainput.UnknownMimetypeError(
                f'Expected an image for "{path}", got mimetype "{mime_type}".')
        return _mediainput.create_image(
            stream,
            file_source=path,
            resize_resolution=resize_resolution,
            aspect_correct=aspect_correct,
            align=align)
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
            return _call_ltx(wrapper, user_args)
    except _d_exceptions.TORCH_CUDA_OOM_EXCEPTIONS as e:
        _d_exceptions.raise_if_not_cuda_oom(e)
        _memory.torch_gc()
        raise _d_exceptions.OutOfMemoryError(e) from e

    raise _pipelines.UnsupportedPipelineConfigError(
        f'{_enums.get_model_type_string(model_type)} is not a video model type.')


def _reject_still_extras(parsed, model_name: str):
    if parsed is None:
        return
    if parsed.mask_images:
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


def _classify_ltx(parsed) -> str:
    if _control_count(parsed):
        raise _pipelines.UnsupportedPipelineConfigError(
            'LTX does not accept control images. Use one image as the first frame, '
            'and end= as the last frame.')
    if parsed is not None and (parsed.multi_image_mode or _image_count(parsed) > 1):
        raise _pipelines.UnsupportedPipelineConfigError(
            'LTX accepts one conditioning image. Use a single path for the first frame, '
            'or end= for the last frame.')
    if _has_end(parsed):
        return 'ltx-condition'
    if _image_count(parsed) == 1:
        return 'ltx-image'
    return 'ltx-txt'


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


def _generator(wrapper, user_args) -> torch.Generator:
    offload = bool(wrapper.model_cpu_offload or wrapper.model_sequential_offload)
    device = 'cpu' if offload else wrapper.device
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


def _offload_ltx(pipe, device, model_cpu_offload, sequential_cpu_offload):
    if sequential_cpu_offload:
        _pipelines.enable_sequential_cpu_offload(pipe, device)
    elif model_cpu_offload:
        _pipelines.enable_model_cpu_offload(pipe, device)


def _video_transformer_class(model_type, family: str = 'ltx2'):
    model_type = _enums.get_model_type_enum(model_type)
    if model_type == _enums.ModelType.LTX:
        if family == 'ltx':
            from diffusers import LTXVideoTransformer3DModel
            return LTXVideoTransformer3DModel
        from diffusers import LTX2VideoTransformer3DModel
        return LTX2VideoTransformer3DModel
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
    'text_encoder',
    'text_encoder_2',
    'text_encoder_3',
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


def _module_is_quantized(module) -> bool:
    if module is None:
        return False
    if getattr(module, 'hf_quantizer', None) is not None:
        return True
    config = getattr(module, 'quantization_config', None)
    if config is None:
        config = getattr(getattr(module, 'config', None), 'quantization_config', None)
    if config is not None:
        return True
    quantized, _, _ = _util.check_bnb_status(module)
    if quantized:
        return True
    if not isinstance(module, torch.nn.Module):
        return False
    for child in module.modules():
        name = type(child).__name__.lower()
        if 'sdnq' in name or 'linear8bit' in name or 'linear4bit' in name:
            return True
    return False


def _confirm_injected_modules(pipe, injected: dict):
    for name, module in injected.items():
        if module is None:
            continue
        current = getattr(pipe, name, None)
        if current is not module:
            _messages.warning(
                f'Pipeline replaced the quantized {name} with a newly loaded copy; '
                f'that module may still be full precision.')
        elif not _module_is_quantized(module):
            _messages.warning(
                f'--quantizer did not quantize {name} ({type(module).__name__}); '
                f'that module is still full precision.')


def _load_quantized_module(component_class, model_path, subfolder, revision, variant,
                           dtype, quantizer_uri, auth_token, local_files_only, device_map,
                           offload=False):
    module_name = getattr(component_class, '__module__', '')
    transformers_module = module_name.startswith('transformers')
    config = _quantizer_config(quantizer_uri, dtype, transformers_module)
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
    if offload:
        module.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return module


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


def _ltx_pipeline_class(mode: str, family: str = 'ltx2'):
    if family == 'ltx':
        from diffusers import LTXConditionPipeline, LTXImageToVideoPipeline, LTXPipeline
        classes = {
            'ltx-txt': LTXPipeline,
            'ltx-image': LTXImageToVideoPipeline,
            'ltx-condition': LTXConditionPipeline,
        }
    else:
        from diffusers import LTX2ConditionPipeline, LTX2ImageToVideoPipeline, LTX2Pipeline
        classes = {
            'ltx-txt': LTX2Pipeline,
            'ltx-image': LTX2ImageToVideoPipeline,
            'ltx-condition': LTX2ConditionPipeline,
        }
    return classes[mode]


def pipeline_for_mode(pipe, mode: str, family: str = 'ltx2'):
    """
    Return an LTX pipeline of the requested class that shares ``pipe.components``.

    ``from_pipe`` is not used: it calls ``.to(dtype=...)``, which raises on
    quantized modules (SDNQ / bitsandbytes).

    :param pipe: cached :py:class:`diffusers.LTX2Pipeline`
    :param mode: ``ltx-txt``, ``ltx-image``, or ``ltx-condition``
    :return: pipeline of the class for ``mode``
    """
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
        'local_files_only': bool(wrapper._local_files_only),
        'auth_token': wrapper._auth_token,
        'quantizer_uri': wrapper.quantizer_uri,
        'quantizer_map': tuple(quantizer_map) if quantizer_map else None,
    }


def _video_on_hit(key, hit):
    _d_memoize.simple_cache_hit_debug('Torch Video Pipeline', key, hit.pipeline)


def _video_on_create(key, new):
    _d_memoize.simple_cache_miss_debug('Torch Video Pipeline', key, new.pipeline)


@_memoize(_pipelines._pipeline_cache,
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
                                  local_files_only,
                                  auth_token,
                                  transformer_uri=None,
                                  lora_uris=None,
                                  lora_fuse_scale=None,
                                  quantizer_uri=None,
                                  quantizer_map=None):
    index = _util.fetch_model_index_dict(
        model_path,
        subfolder=subfolder,
        revision=revision,
        use_auth_token=auth_token,
        local_files_only=local_files_only)
    extra_dirs = extra_weight_directories_from_index(index)
    family = ltx_family_from_index(index)
    estimate = _pipelines.estimate_pipeline_cache_footprint(
        model_path=model_path,
        model_type=model_type,
        revision=revision or 'main',
        variant=variant,
        subfolder=subfolder,
        include_unet_or_transformer=not transformer_uri,
        include_vae=True,
        include_text_encoders=True,
        lora_uris=lora_uris,
        include_directories=extra_dirs,
        auth_token=auth_token,
        local_files_only=local_files_only)
    _pipelines._enforce_pipeline_cache_size(estimate)

    load_kwargs = _pretrained_kwargs(
        revision, variant, subfolder, local_files_only, auth_token, dtype,
        include_dtype=True)
    if family == 'ltx2':
        load_kwargs.update(_LTX_SKIP_OPTIONAL_MODULES)

    if model_type != _enums.ModelType.LTX:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'{_enums.get_model_type_string(model_type)} is not a video model type.')

    pipeline_class = _ltx_pipeline_class('ltx-txt', family)
    offload = bool(model_cpu_offload or sequential_cpu_offload)
    injected = {}
    if transformer_uri:
        injected['transformer'] = _load_replacement_transformer(
            model_type, transformer_uri, dtype, variant, auth_token, local_files_only,
            quantizer_uri=quantizer_uri, quantizer_map=quantizer_map, device=device,
            offload=offload, config_repo=model_path, family=family)
        load_kwargs['transformer'] = injected['transformer']
    injected.update(_quantized_classic_modules(
        model_path, revision, variant, subfolder, dtype, quantizer_uri, quantizer_map,
        auth_token, local_files_only, device, offload,
        skip_names=frozenset(load_kwargs)))
    load_kwargs.update(injected)
    _messages.debug_log(f'Loading {pipeline_class.__name__} from "{model_path}".')
    with _hfhub.with_hf_errors_as_model_not_found():
        pipe = pipeline_class.from_pretrained(model_path, **load_kwargs)
    _confirm_injected_modules(pipe, injected)
    _apply_video_loras(
        pipe, model_type, lora_uris, lora_fuse_scale, auth_token, local_files_only)
    _offload_ltx(pipe, device, model_cpu_offload, sequential_cpu_offload)
    _enable_vae_tiling(pipe)
    held = _VideoPipeline(pipe, family)

    return held, _d_memoize.CachedObjectMetadata(size=estimate)


def _video_pipeline(wrapper, mode: str):
    kwargs = _cache_kwargs(wrapper)
    kwargs['transformer_uri'] = wrapper.transformer_uri
    kwargs['lora_uris'] = tuple(wrapper.lora_uris) if wrapper.lora_uris else None
    kwargs['lora_fuse_scale'] = wrapper.lora_fuse_scale
    held = _create_cached_video_pipeline(**kwargs)
    return pipeline_for_mode(held.pipeline, mode, held.family), held.family


def _invoke(wrapper, pipe, kwargs):
    return _pipelines.call_pipeline(pipe, device=wrapper.device, **kwargs)


def _call_ltx(wrapper, user_args):
    if user_args.end_images:
        mode = 'ltx-condition'
    elif user_args.images:
        mode = 'ltx-image'
    else:
        mode = 'ltx-txt'

    pipe, family = _video_pipeline(wrapper, mode)
    positive, negative = _prompt_text(user_args)
    width, height = _size(user_args)
    if width is not None:
        _require_multiple_of_32(width, height, 'LTX')

    fps = float(user_args.video_fps or _LTX_DEFAULT_FPS)
    kwargs = {
        'prompt': positive,
        'frame_rate': fps,
        'generator': _generator(wrapper, user_args),
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
    elif family == 'ltx2':
        _messages.debug_log('LTX will choose the clip length from the prompt.')
    else:
        _messages.debug_log('LTX-Video will use its default frame count.')

    guidance = float(
        _types.default(user_args.guidance_scale, _constants.DEFAULT_GUIDANCE_SCALE))
    user_audio = user_args.audio_guidance_scale
    if family == 'ltx':
        if user_audio is not None or user_args.audio_guidance_rescale is not None:
            raise _pipelines.UnsupportedPipelineConfigError(
                'Audio guidance is only supported by LTX-2. This checkpoint uses the earlier LTX pipeline.')
        kwargs['num_inference_steps'] = int(
            _types.default(user_args.inference_steps, _constants.DEFAULT_INFERENCE_STEPS))
        kwargs['guidance_scale'] = guidance
        _warn_legacy_ltx_canvas(width, height, kwargs.get('num_frames'), fps)
        _require_legacy_ltx_schedule(
            pipe,
            width if width is not None else 704,
            height if height is not None else 512,
            kwargs.get('num_frames', 161),
            kwargs['num_inference_steps'])
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
                    user_args.audio_guidance_scale = audio_guidance
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
        if user_args.audio_guidance_rescale is not None:
            kwargs['audio_guidance_rescale'] = float(user_args.audio_guidance_rescale)
        elif user_args.guidance_rescale is not None:
            kwargs['audio_guidance_rescale'] = float(user_args.guidance_rescale)

    if user_args.max_sequence_length is not None:
        kwargs['max_sequence_length'] = int(user_args.max_sequence_length)

    _set_ltx_vae_slicing(pipe, bool(user_args.vae_slicing))

    if mode == 'ltx-image':
        kwargs['image'] = user_args.images[0]
    elif mode == 'ltx-condition' and family == 'ltx':
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

        last_index = int(kwargs.get('num_frames', 161)) - 1
        conditions = []
        if user_args.images:
            conditions.append(LTXVideoCondition(image=user_args.images[0], frame_index=0, strength=1.0))
        conditions.append(LTXVideoCondition(
            image=user_args.end_images[0], frame_index=last_index, strength=1.0))
        kwargs['conditions'] = conditions
    elif mode == 'ltx-condition':
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

        conditions = []
        if user_args.images:
            conditions.append(LTX2VideoCondition(frames=user_args.images[0], index=0, strength=1.0))
        conditions.append(LTX2VideoCondition(frames=user_args.end_images[0], index=-1, strength=1.0))
        kwargs['conditions'] = conditions

    output = _invoke(wrapper, pipe, kwargs)
    frames = frames_from_output(output.frames)
    audio = audio_to_numpy(getattr(output, 'audio', None))
    sample_rate = audio_sample_rate_from_pipeline(pipe, audio)
    return frames, audio, sample_rate, fps


def _legacy_ltx_schedule_finite(scheduler, width, height, num_frames, steps,
                                spatial=32, temporal=8) -> bool:
    """
    The earlier LTX pipeline shifts timesteps from the latent token count.
    Past a point that shift makes the terminal sigma 1 and the schedule NaN,
    which later crashes the sampler with an empty timestep index.
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
    return bool(torch.isfinite(probe.timesteps).all())


def _require_legacy_ltx_schedule(pipe, width, height, num_frames, steps):
    spatial = int(getattr(pipe, 'vae_spatial_compression_ratio', 32) or 32)
    temporal = int(getattr(pipe, 'vae_temporal_compression_ratio', 8) or 8)
    if _legacy_ltx_schedule_finite(
            pipe.scheduler, width, height, num_frames, steps, spatial, temporal):
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


__all__ = _types.module_all()
