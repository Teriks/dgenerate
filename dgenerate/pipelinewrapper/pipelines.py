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
import collections.abc
import functools
import gc
import hashlib
import inspect
import os.path
import pathlib
import random
import re
import typing

import accelerate
import diffusers
import diffusers.loaders
import diffusers.loaders.single_file_utils
import diffusers.quantizers.quantization_config
import torch
import torch.nn
import transformers

import dgenerate.devicecache as _devicecache
import dgenerate.exceptions as _d_exceptions
import dgenerate.extras.kolors as _kolors
import dgenerate.extras.ultraedit
import dgenerate.filecache as _filecache
import dgenerate.hfhub as _hfhub
import dgenerate.memoize as _d_memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.schedulers as _schedulers
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.pipelinewrapper.util as _util
import dgenerate.promptweighters as _promptweighters
import dgenerate.textprocessing as _textprocessing
import dgenerate.torchutil as _torchutil
import dgenerate.types as _types
from dgenerate.memoize import memoize as _memoize
from dgenerate.pipelinewrapper import constants as _constants
import dgenerate.pipelinewrapper.models as _models


class UnsupportedPipelineConfigError(Exception):
    """
    Occurs when a diffusers pipeline is requested to be
    configured in a way that is unsupported by that pipeline.
    """
    pass


class InvalidModelFileError(Exception):
    """
    Raised when a file is loaded from disk that is an invalid diffusers model format.

    This indicates that was a problem loading the primary diffusion model,
    This could also refer to an SDXL refiner model or Stable Cascade decoder
    model which are considered primary models.
    """
    pass


_pipeline_cache = _d_memoize.create_object_cache(
    'pipeline', cache_type=_memory.SizedConstrainedObjectCache
)


def _disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False] * num_images
    else:
        return images, False


def _floyd_disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False] * num_images, False
    else:
        return images, False, False


def _set_sd_safety_checker(pipeline: diffusers.DiffusionPipeline, safety_checker: bool):
    if not safety_checker:
        if hasattr(pipeline, 'safety_checker') and pipeline.safety_checker is not None:
            # If it's already None for some reason you'll get a call
            # to an unassigned feature_extractor by assigning it a value

            # The attribute will not exist for SDXL pipelines currently

            pipeline.safety_checker = _disabled_safety_checker


def _set_floyd_safety_checker(pipeline: diffusers.DiffusionPipeline, safety_checker: bool):
    if not safety_checker:
        if hasattr(pipeline, 'safety_checker') and pipeline.safety_checker is not None:
            pipeline.safety_checker = _floyd_disabled_safety_checker


def estimate_pipeline_cache_footprint(
        model_path: str,
        model_type: _enums.ModelType,
        revision: _types.Name = 'main',
        variant: _types.OptionalName = None,
        subfolder: _types.OptionalPath = None,
        include_unet_or_transformer: bool = False,
        include_vae: bool = False,
        include_text_encoders: bool = False,
        lora_uris: _types.OptionalUris = None,
        image_encoder_uri: _types.OptionalUri = None,
        ip_adapter_uris: _types.OptionalUris = None,
        textual_inversion_uris: _types.OptionalUris = None,
        safety_checker: bool = False,
        auth_token: str | None = None,
        extra_args: dict[str, typing.Any] | None = None,
        local_files_only: bool = False) -> int:
    """
    Estimate the CPU side cache memory use of a pipeline.

    This does not include the UNet / Transformer, VAE, or Text Encoders
    as those have their own individual caches.

    :param model_path: huggingface slug, blob link, path to folder on disk, path to model file.
    :param model_type: :py:class:`dgenerate.pipelinewrapper.ModelType`
    :param revision: huggingface repo revision if using a huggingface slug
    :param variant: model file variant desired, for example "fp16"
    :param subfolder: huggingface repo subfolder if using a huggingface slug
        this is currently only supported for Stable Diffusion 3 and Flux models.
    :param include_unet_or_transformer: Include the unet / transformer?
        Under most conditions this is loaded separately and put into
        a cache of its own.
    :param include_text_encoders: Include text encoders?
        Under most conditions these are loaded separately and put into
        a cache of their own.
    :param include_vae: Include VAE? Under most conditions this is
        loaded separately and put into a cache its own.
    :param lora_uris: optional user specified ``--loras`` URIs that will be loaded on to the pipeline
    :param image_encoder_uri: optional user specified ``--image-encoder`` URI that will be loaded on to the pipeline
    :param ip_adapter_uris: optional user specified ``--ip-adapters`` URIs that will be loaded on to the pipeline
    :param textual_inversion_uris: optional user specified ``--textual-inversion`` URIs that will be loaded on to the pipeline
    :param safety_checker: consider the safety checker? dgenerate usually loads the safety checker and then retroactively
        disables it if needed, so it usually considers the size of the safety checker model.
    :param auth_token: optional huggingface auth token to access restricted repositories that your account has access to.
    :param extra_args: ``extra_args`` as to be passed to :py:func:`.create_diffusion_pipeline`
    :param local_files_only: Only ever attempt to look in the local huggingface cache? if ``False`` the huggingface
        API will be contacted when necessary.
    :return: size estimate in bytes.
    """

    if extra_args is None:
        extra_args = dict()

    usage = _util.estimate_model_memory_use(
        repo_id=_hfhub.download_non_hf_slug_model(model_path),
        revision=revision,
        variant=variant,
        subfolder=subfolder,
        include_unet_or_transformer=include_unet_or_transformer,
        include_vae=include_vae,
        safety_checker=safety_checker and 'safety_checker' not in extra_args,
        include_text_encoder=include_text_encoders,
        include_text_encoder_2=include_text_encoders,
        include_text_encoder_3=include_text_encoders,
        use_auth_token=auth_token,
        local_files_only=local_files_only,
        sentencepiece=_enums.model_type_is_floyd(model_type)
    )

    if image_encoder_uri:
        parsed = _uris.ImageEncoderUri.parse(image_encoder_uri)
        usage += _util.estimate_model_memory_use(
            repo_id=_hfhub.download_non_hf_slug_model(parsed.model),
            revision=parsed.revision,
            subfolder=parsed.subfolder,
            use_auth_token=auth_token,
            local_files_only=local_files_only
        )

    if lora_uris:
        for lora_uri in lora_uris:
            parsed = _uris.LoRAUri.parse(lora_uri)

            usage += _util.estimate_model_memory_use(
                repo_id=_hfhub.download_non_hf_slug_model(parsed.model),
                revision=parsed.revision,
                subfolder=parsed.subfolder,
                weight_name=parsed.weight_name,
                use_auth_token=auth_token,
                local_files_only=local_files_only
            )

    if ip_adapter_uris:
        for ip_adapter_uri in ip_adapter_uris:
            parsed = _uris.IPAdapterUri.parse(ip_adapter_uri)

            usage += _util.estimate_model_memory_use(
                repo_id=_hfhub.download_non_hf_slug_model(parsed.model),
                revision=parsed.revision,
                subfolder=parsed.subfolder,
                weight_name=parsed.weight_name,
                use_auth_token=auth_token,
                local_files_only=local_files_only
            )

    if textual_inversion_uris:
        for textual_inversion_uri in textual_inversion_uris:
            parsed = _uris.TextualInversionUri.parse(textual_inversion_uri)

            usage += _util.estimate_model_memory_use(
                repo_id=_hfhub.download_non_hf_slug_model(parsed.model),
                revision=parsed.revision,
                subfolder=parsed.subfolder,
                weight_name=parsed.weight_name,
                use_auth_token=auth_token,
                local_files_only=local_files_only
            )

    return usage


def set_vae_tiling_and_slicing(
        pipeline: diffusers.DiffusionPipeline,
        tiling: bool,
        slicing: bool
):
    """
    Set the ``vae_slicing`` and ``vae_tiling`` status on a diffusers pipeline.

    :raises UnsupportedPipelineConfigError: if the pipeline does not support one or both
        of the provided values for ``vae_tiling`` and ``vae_slicing``

    :param pipeline: pipeline object
    :param tiling: tiling status
    :param slicing: slicing status
    """

    has_vae = hasattr(pipeline, 'vae') and pipeline.vae is not None
    pipeline_class = pipeline.__class__

    if tiling:
        if has_vae:
            if hasattr(pipeline.vae, 'enable_tiling'):
                _messages.debug_log(f'Enabling VAE tiling on Pipeline: "{pipeline_class.__name__}",',
                                    f'VAE: "{pipeline.vae.__class__.__name__}"')
                pipeline.vae.enable_tiling()
            else:
                raise UnsupportedPipelineConfigError(
                    '--vae-tiling not supported as loaded VAE does not support it.'
                )
        else:
            raise UnsupportedPipelineConfigError(
                '--vae-tiling not supported as no VAE is present for the specified model.')
    elif has_vae:
        if hasattr(pipeline.vae, 'disable_tiling'):
            _messages.debug_log(f'Disabling VAE tiling on Pipeline: "{pipeline_class.__name__}",',
                                f'VAE: "{pipeline.vae.__class__.__name__}"')
            pipeline.vae.disable_tiling()

    if slicing:
        if has_vae:
            if hasattr(pipeline.vae, 'enable_slicing'):
                _messages.debug_log(f'Enabling VAE slicing on Pipeline: "{pipeline_class.__name__}",',
                                    f'VAE: "{pipeline.vae.__class__.__name__}"')
                pipeline.vae.enable_slicing()
            else:
                raise UnsupportedPipelineConfigError(
                    '--vae-slicing not supported as loaded VAE does not support it.'
                )
        else:
            raise UnsupportedPipelineConfigError(
                '--vae-slicing not supported as no VAE is present for the specified model.')
    elif has_vae:
        if hasattr(pipeline.vae, 'disable_slicing'):
            _messages.debug_log(f'Disabling VAE slicing on Pipeline: "{pipeline_class.__name__}",',
                                f'VAE: "{pipeline.vae.__class__.__name__}"')
            pipeline.vae.disable_slicing()


def get_pipeline_modules(pipeline: diffusers.DiffusionPipeline):
    """
    Get all component modules of a torch diffusers pipeline.

    :param pipeline: the pipeline
    :return: dictionary of modules by name
    """
    return {k: v for k, v in pipeline.components.items() if isinstance(v, torch.nn.Module)}


def _set_sequential_cpu_offload_flag(module: diffusers.DiffusionPipeline | torch.nn.Module, value: bool):
    module.DGENERATE_SEQUENTIAL_CPU_OFFLOAD = bool(value)

    _messages.debug_log(
        f'setting DGENERATE_SEQUENTIAL_CPU_OFFLOAD={value} on module "{module.__class__.__name__}"')


def _set_cpu_offload_flag(module: diffusers.DiffusionPipeline | torch.nn.Module, value: bool):
    module.DGENERATE_MODEL_CPU_OFFLOAD = bool(value)

    _messages.debug_log(
        f'setting DGENERATE_MODEL_CPU_OFFLOAD={value} on module "{module.__class__.__name__}"')


def is_sequential_cpu_offload_enabled(module: diffusers.DiffusionPipeline | torch.nn.Module):
    """
    Test if a pipeline or torch neural net module created by dgenerate has sequential offload enabled.

    :param module: the module object
    :return: ``True`` or ``False``
    """
    return hasattr(module, 'DGENERATE_SEQUENTIAL_CPU_OFFLOAD') and bool(module.DGENERATE_SEQUENTIAL_CPU_OFFLOAD)


def is_model_cpu_offload_enabled(module: diffusers.DiffusionPipeline | torch.nn.Module):
    """
    Test if a pipeline or torch neural net module created by dgenerate has model cpu offload enabled.

    :param module: the module object
    :return: ``True`` or ``False``
    """
    return hasattr(module, 'DGENERATE_MODEL_CPU_OFFLOAD') and bool(module.DGENERATE_MODEL_CPU_OFFLOAD)


def _disable_to(module, vae=False):
    if hasattr(module, '_DGENERATE_ORIGINAL_TO_DISABLED'):
        return  # Already patched

    module._DGENERATE_ORIGINAL_TO_DISABLED = module.to

    def dummy(*args, **kwargs):
        if vae and module.config.force_upcast and \
                (len(args) == 1 and isinstance(args[0], torch.dtype)) or \
                (len(kwargs) == 1 and 'dtype' in kwargs):

            # basically, is this a VAE that the pipeline needs to upcast
            # this has to happen even if it is described as 'meta'

            module._DGENERATE_ORIGINAL_TO_DISABLED(*args, **kwargs)
        else:
            pass

    module.to = dummy
    _messages.debug_log(
        f'Disabled .to() on module: {_types.fullname(module)}')


def enable_sequential_cpu_offload(pipeline: diffusers.DiffusionPipeline,
                                  device: torch.device | str = _torchutil.default_device()):
    """
    Enable sequential offloading on a torch pipeline, in a way dgenerate can keep track of.

    :param pipeline: the pipeline
    :param device: the device
    """
    torch_device = torch.device(device)
    
    # Check if the requested device type is actually available
    # If not, fall back to the system's default device
    if torch_device.type == 'cuda' and not _torchutil.is_cuda_available():
        fallback_device = _torchutil.default_device()
        _messages.debug_log(
            f'enable_sequential_cpu_offload: CUDA requested but not available, using {fallback_device} for execution device')
        torch_device = torch.device(fallback_device)
    elif torch_device.type == 'mps' and not _torchutil.is_mps_available():
        fallback_device = _torchutil.default_device()
        _messages.debug_log(
            f'enable_sequential_cpu_offload: MPS requested but not available, using {fallback_device} for execution device')
        torch_device = torch.device(fallback_device)
    elif torch_device.type == 'xpu' and not _torchutil.is_xpu_available():
        fallback_device = _torchutil.default_device()
        _messages.debug_log(
            f'enable_sequential_cpu_offload: XPU requested but not available, using {fallback_device} for execution device')
        torch_device = torch.device(fallback_device)
    
    pipeline.remove_all_hooks()

    _set_sequential_cpu_offload_flag(pipeline, True)
    for name, model in get_pipeline_modules(pipeline).items():
        quant, _, _ = _util.check_bnb_status(model)

        if name in pipeline._exclude_from_cpu_offload or quant:
            continue

        elif not is_sequential_cpu_offload_enabled(model):
            _set_sequential_cpu_offload_flag(model, True)
            accelerate.cpu_offload(model, torch_device, offload_buffers=len(model._parameters) > 0)
            _disable_to(
                model,
                name == 'vae'
            )


def enable_model_cpu_offload(pipeline: diffusers.DiffusionPipeline,
                             device: torch.device | str = _torchutil.default_device()):
    """
    Enable sequential model cpu offload on a torch pipeline, in a way dgenerate can keep track of.

    :param pipeline: the pipeline
    :param device: the device
    """

    if pipeline.model_cpu_offload_seq is None:
        raise ValueError(
            "Model CPU offload cannot be enabled because no `model_cpu_offload_seq` class attribute is set."
        )

    torch_device = torch.device(device)

    pipeline.remove_all_hooks()

    pipeline._offload_gpu_id = torch_device.index or getattr(pipeline, "_offload_gpu_id", 0)

    device_type = torch_device.type
    
    # Check if the requested device type is actually available
    # If not, fall back to the system's default device
    if device_type == 'cuda' and not _torchutil.is_cuda_available():
        fallback_device = _torchutil.default_device()
        _messages.debug_log(
            f'enable_model_cpu_offload: CUDA requested but not available, using {fallback_device} for execution device')
        device = torch.device(fallback_device)
    elif device_type == 'mps' and not _torchutil.is_mps_available():
        fallback_device = _torchutil.default_device()
        _messages.debug_log(
            f'enable_model_cpu_offload: MPS requested but not available, using {fallback_device} for execution device')
        device = torch.device(fallback_device)
    elif device_type == 'xpu' and not _torchutil.is_xpu_available():
        fallback_device = _torchutil.default_device()
        _messages.debug_log(
            f'enable_model_cpu_offload: XPU requested but not available, using {fallback_device} for execution device')
        device = torch.device(fallback_device)
    else:
        device = torch.device(f"{device_type}:{pipeline._offload_gpu_id}")

    if pipeline.device.type != "cpu":
        pipeline.to("cpu", silence_dtype_warnings=True)
        device_mod = getattr(torch, pipeline.device.type, None)
        if device_mod is not None and hasattr(device_mod, "empty_cache") and hasattr(device_mod, "is_available") and device_mod.is_available():
            device_mod.empty_cache()

    _set_cpu_offload_flag(pipeline, True)

    all_model_components = {k: v for k, v in pipeline.components.items() if isinstance(v, torch.nn.Module)}

    hook = None
    pipeline._all_hooks = []
    for model_str in pipeline.model_cpu_offload_seq.split("->"):
        model = all_model_components.pop(model_str, None)

        if not isinstance(model, torch.nn.Module):
            continue

        _, _, is_loaded_in_8bit_bnb = _util.check_bnb_status(model)

        if is_loaded_in_8bit_bnb:
            _messages.debug_log(
                f'Not cpu offloading pipeline module: {model_str}, due to bitsandbytes 8 bit quantization.')
            continue

        _, hook = accelerate.cpu_offload_with_hook(model, device, prev_module_hook=hook)
        _set_cpu_offload_flag(model, True)
        pipeline._all_hooks.append(hook)

    for name, model in all_model_components.items():
        if not isinstance(model, torch.nn.Module):
            continue

        if name in pipeline._exclude_from_cpu_offload:
            model.to(device)
        else:
            _, hook = accelerate.cpu_offload_with_hook(model, device)
            _set_cpu_offload_flag(model, True)
            pipeline._all_hooks.append(hook)


def get_torch_device(component: diffusers.DiffusionPipeline | torch.nn.Module) -> torch.device:
    """
    Get the device that a pipeline or pipeline component exists on.

    :param component: pipeline or pipeline component.
    :return: :py:class:`torch.device`
    """
    if hasattr(component, 'device'):
        return component.device
    elif hasattr(component, 'get_device'):
        return component.get_device()

    raise ValueError(f'component type {component.__class__} did not have a '
                     f'device attribute or the function get_device()')


def get_torch_device_string(component: diffusers.DiffusionPipeline | torch.nn.Module) -> str:
    """
    Get the device string that a pipeline or pipeline component exists on.

    :param component: pipeline or pipeline component.
    :return: device string
    """
    return str(get_torch_device(component))


def _pipeline_to(pipeline, device: torch.device | str | None):
    if device is None:
        _messages.debug_log(
            f'pipeline_to() Not moving pipeline "{pipeline.__class__.__name__}" '
            f'as specified device was None.')
        return

    if not hasattr(pipeline, 'to'):
        _messages.debug_log(
            f'pipeline_to() Not moving pipeline "{pipeline.__class__.__name__}" to '
            f'"{device}" as it has no to() method.')
        return

    device = torch.device(device)

    pipeline_device = get_torch_device(pipeline)

    all_modules_on_device = all(
        _torchutil.devices_equal(device, get_torch_device(m)) for m in get_pipeline_modules(pipeline).values())

    pipeline_on_device = _torchutil.devices_equal(get_torch_device(pipeline), pipeline_device)

    if pipeline_on_device and all_modules_on_device:
        _messages.debug_log(
            f'pipeline_to() Not moving pipeline "{pipeline.__class__.__name__}" to '
            f'"{device}" as it is already on that device.')
        return

    if pipeline_on_device != all_modules_on_device:
        # really the most likely way for this to occur is if
        # an OOM happened moving a pipeline to the GPU, which
        # is something we want to be able to recover from hence
        # the fall through above
        #
        # This also happens when the pipeline has cpu offload
        # enabled, we can fall through that harmlessly as its
        # modules can never be moved to anything but the CPU
        # and that is accounted for below
        _messages.debug_log(
            f'pipeline_to() Moving pipeline "{pipeline.__class__.__name__}" to "{device}", '
            f'pipeline_on_device={pipeline_on_device}, all_modules_on_device={all_modules_on_device}.')

    if not _torchutil.devices_equal(pipeline_device, device):
        try:
            cache_metadata = _pipeline_cache.get_metadata(pipeline)

            if device.type != 'cpu':
                _pipeline_cache.size -= cache_metadata.size

                _messages.debug_log(
                    f'Cached {_types.class_and_id_string(pipeline)} Size = '
                    f'{cache_metadata.size} Bytes '
                    f'({_memory.bytes_best_human_unit(cache_metadata.size)}) '
                    f'is leaving CPU side memory, '
                    f'cache size is now '
                    f'{cache_metadata.size} Bytes '
                    f'({_memory.bytes_best_human_unit(cache_metadata.size)})')

            else:
                _messages.debug_log(
                    f'Cached {_types.class_and_id_string(pipeline)} Size = '
                    f'{cache_metadata.size} Bytes '
                    f'({_memory.bytes_best_human_unit(cache_metadata.size)}) '
                    f'is entering CPU side memory, '
                    f'cache size is now '
                    f'{cache_metadata.size} Bytes '
                    f'({_memory.bytes_best_human_unit(cache_metadata.size)})')

                _pipeline_cache.size += cache_metadata.size
        except _d_memoize.ObjectCacheKeyError:
            # does not exist in the cache
            pass

    for name, value in get_pipeline_modules(pipeline).items():

        is_loaded_in_8bit_bnb = _util.is_loaded_in_8bit_bnb(value)

        if is_loaded_in_8bit_bnb:
            _messages.debug_log(
                f'pipeline_to() Not moving module "{name} = {value.__class__.__name__}" to "{device}" '
                f'as it is loaded in 8bit mode via bitsandbytes.')
            _disable_to(value)
            continue

        current_device = get_torch_device(value)

        if current_device.type == 'meta':
            _messages.debug_log(
                f'pipeline_to() Not moving module "{name} = {value.__class__.__name__}" to "{device}" '
                f'as its device value is "meta".')
            _disable_to(
                value,
                name == 'vae'
            )
            continue

        if _torchutil.devices_equal(current_device, device):
            _messages.debug_log(
                f'pipeline_to() Not moving module "{name} = {value.__class__.__name__}" to "{device}" '
                f'as it is already on that device.')
            continue

        if is_model_cpu_offload_enabled(value) and device.type != 'cpu':
            _messages.debug_log(
                f'pipeline_to() Not moving module "{name} = {value.__class__.__name__}" to "{device}" '
                f'as it has cpu offload enabled and can only move to cpu.')
            continue

        _messages.debug_log(
            f'pipeline_to() Moving module "{name}" of pipeline {_types.fullname(pipeline)} '
            f'from device "{current_device}" to device "{device}"')

        value.to(device)

    if device.type == 'cpu':
        _memory.torch_gc()


def pipeline_to(pipeline, device: torch.device | str | None):
    """
    Move a diffusers pipeline to a device if possible, in a way that dgenerate can keep track of.

    This calls methods associated with updating the cache statistics such as
    :py:func:`dgenerate.pipelinewrapper.pipeline_off_cpu_update_cache_info` and
    :py:func:`dgenerate.pipelinewrapper.pipeline_to_cpu_update_cache_info` for you,
    as well as the associated cache update functions for the pipelines individual
    components as needed.

    If ``device==None`` this is a no-op.

    Modules which are meta tensors will not be moved (sequentially offloaded modules)

    Modules which have model cpu offload enabled will not be moved unless they are moving to "cpu"

    :raise dgenerate.OutOfMemoryError: if there is not enough memory on the specified device

    :param pipeline: the pipeline
    :param device: the device

    :return: the moved pipeline
    """

    try:
        _pipeline_to(pipeline=pipeline, device=device)
    except _d_exceptions.TORCH_CUDA_OOM_EXCEPTIONS as e:
        _d_exceptions.raise_if_not_cuda_oom(e)
        # attempt to recover VRAM before rethrowing
        # move any modules back to cpu which have entered VRAM

        _pipeline_to(pipeline=pipeline, device='cpu')
        _memory.torch_gc()
        gc.collect()

        raise _d_exceptions.OutOfMemoryError(e) from e
    except MemoryError as e:
        # probably out of RAM on a back
        # to CPU move not much we can do
        gc.collect()
        raise _d_exceptions.OutOfMemoryError('cpu (system memory)') from e


def _call_args_debug_transformer(key, value):
    if isinstance(value, torch.Generator):
        return f'torch.Generator(seed={value.initial_seed()})'
    if isinstance(value, torch.Tensor):
        return f'torch.Tensor({value.shape})'
    return value


def _warn_prompt_lengths(pipeline, **kwargs):
    prompts = [
        ('Primary positive prompt', kwargs.get('prompt'), 'tokenizer'),
        ('Primary negative prompt', kwargs.get('negative_prompt'), 'tokenizer'),
        ('Secondary positive prompt', kwargs.get('prompt_2'), 'tokenizer_2'),
        ('Secondary negative prompt', kwargs.get('negative_prompt_2'), 'tokenizer_2'),
        ('Tertiary positive prompt', kwargs.get('prompt_3'), 'tokenizer_3'),
        ('Tertiary negative prompt', kwargs.get('negative_prompt_3'), 'tokenizer_3')
    ]

    warned_prompts = {}

    for label, prompt, tokenizer_attr in prompts:
        if prompt and not isinstance(prompt, list):
            prompt = [prompt]

        if prompt:
            tokenizer = getattr(pipeline, tokenizer_attr, None)

            if tokenizer:
                if tokenizer_attr == 'tokenizer_3' and pipeline.__class__.__name__.startswith('StableDiffusion3'):
                    max_length = min(kwargs.get('max_sequence_length', 256), tokenizer.model_max_length)
                elif tokenizer_attr == 'tokenizer_2' and pipeline.__class__.__name__.startswith('Flux'):
                    max_length = min(kwargs.get('max_sequence_length', 512), tokenizer.model_max_length)
                else:
                    max_length = tokenizer.model_max_length

                for p in prompt:
                    if len(tokenizer.tokenize(p)) > max_length:
                        key = f'{label}{tokenizer_attr}{p}'
                        if key not in warned_prompts:
                            _messages.warning(
                                f'{label} exceeds max token length '
                                f'of {max_length} for the model\'s tokenizer '
                                f'and will be truncated: "{p}"'
                            )
                            warned_prompts[key] = True


_LAST_CALLED_PIPELINE = None


def get_last_called_pipeline() -> diffusers.DiffusionPipeline | None:
    """
    Get a reference to the globally cached pipeline last called with :py:func:`call_pipeline`.

    This value may be ``None`` if a pipeline was never called.

    :return: diffusion pipeline object
    """
    return _LAST_CALLED_PIPELINE


def destroy_last_called_pipeline(collect=True):
    """
    Move to CPU and dereference the globally cached pipeline last called with :py:func:`call_pipeline`.

    This is a no-op if a pipeline has never been called with :py:func:`call_pipeline`

    :param collect: call ``gc.collect`` and :py:func:`dgenerate.memory.torch_gc` if
        there is a pipeline to dereference?
    """
    global _LAST_CALLED_PIPELINE

    if _LAST_CALLED_PIPELINE is not None:

        pipeline_to(_LAST_CALLED_PIPELINE, 'cpu')

        _LAST_CALLED_PIPELINE = None

        if collect:
            gc.collect()
            _memory.torch_gc()


def _evict_last_pipeline(device: torch.device | None):
    active_pipe = get_last_called_pipeline()

    if active_pipe is None:
        return

    if device is None or _torchutil.devices_equal(get_torch_device(active_pipe), device):
        # get rid of this reference immediately
        # noinspection PyUnusedLocal
        active_pipe = None

        _messages.debug_log(
            f'{_types.fullname(_devicecache.clear_device_cache)} is attempting to evacuate any previously '
            f'called diffusion pipeline in the VRAM of device: {device}.')

        # potentially free up VRAM on the GPU we are
        # about to move to
        destroy_last_called_pipeline()


def _evict_8bit_bnb_pipelines(device: torch.device | None):
    # clear out any 8bit bnb pipelines in the cache, they are
    # on the GPU and cannot be moved
    for cached_pipeline in _pipeline_cache.values():

        bit8 = any(_util.is_loaded_in_8bit_bnb(module) for
                   module in get_pipeline_modules(cached_pipeline.pipeline).values())

        if bit8 and (device is None or _torchutil.devices_equal(device, get_torch_device(cached_pipeline.pipeline))):
            _messages.debug_log(
                f'Clearing out cached 8bit pipeline from cache: {cached_pipeline.pipeline.__class__.__name__}')
            _pipeline_cache.un_cache(cached_pipeline)
            del cached_pipeline


_devicecache.register_eviction_method(_evict_last_pipeline)
_devicecache.register_eviction_method(_evict_8bit_bnb_pipelines)


# noinspection PyCallingNonCallable
@torch.inference_mode()
def call_pipeline(pipeline: diffusers.DiffusionPipeline,
                  device: torch.device | str | None = _torchutil.default_device(),
                  prompt_weighter: _promptweighters.PromptWeighter = None,
                  **kwargs):
    """
    Call a diffusers pipeline, offload the last called pipeline to CPU before
    doing so if the last pipeline is not being called in succession


    :param pipeline: The pipeline

    :param device: The device to move the pipeline to before calling, it will be
        moved to this device if it is not already on the device. If the pipeline
        does not support moving to specific device, such as with sequentially offloaded
        pipelines which cannot move at all, or cpu offloaded pipelines which can
        only move to CPU, this argument is ignored.

    :param kwargs: diffusers pipeline keyword arguments

    :param prompt_weighter: Optional prompt weighter for weighted prompt syntaxes

    :raises dgenerate.OutOfMemoryError: if there is not enough memory on the specified device

    :raises UnsupportedPipelineConfiguration:
        If the pipeline is missing certain required modules, such as text encoders.

    :return: the result of calling the diffusers pipeline
    """

    global _LAST_CALLED_PIPELINE

    if prompt_weighter is not None:
        if not _torchutil.devices_equal(device, prompt_weighter.device):
            raise UnsupportedPipelineConfigError(
                'dgenerate.pipelinewrapper.call_pipeline: prompt_weighter '
                'must specify the same compute device that pipeline is to be called on. '
                f'Got prompt_weighter={prompt_weighter.device}, and pipeline={device}')

    _messages.debug_log(
        f'Calling Pipeline: "{pipeline.__class__.__name__}",',
        f'Device: "{device}",',
        'Args:',
        lambda: _textprocessing.debug_format_args(
            kwargs, value_transformer=_call_args_debug_transformer))

    enable_retry_pipe = True

    def _cleanup_prompt_weighter():
        try:
            _messages.debug_log(
                f'Executing prompt weighter cleanup for "{prompt_weighter.__class__.__name__}"')
            prompt_weighter.cleanup()
        except Exception as e:
            _messages.debug_log(
                f'Ignoring prompt weighter cleanup '
                f'exception in "{prompt_weighter.__class__.__name__}.cleanup()": {e}')
            pass

    def _call_prompt_weighter():
        nonlocal enable_retry_pipe

        try:
            translated = prompt_weighter.translate_to_embeds(pipeline, kwargs)
        except _d_exceptions.TORCH_CUDA_OOM_EXCEPTIONS as e:
            _d_exceptions.raise_if_not_cuda_oom(e)
            _cleanup_prompt_weighter()
            _memory.torch_gc()
            gc.collect()
            raise _d_exceptions.OutOfMemoryError(e) from e
        except MemoryError as e:
            _cleanup_prompt_weighter()
            gc.collect()
            raise _d_exceptions.OutOfMemoryError('cpu (system memory)') from e
        except Exception:
            _cleanup_prompt_weighter()
            _memory.torch_gc()
            gc.collect()
            raise

        def _debug_string_func():
            return f'{prompt_weighter.__class__.__name__} translated pipeline call args to: ' + \
                _textprocessing.debug_format_args(
                    translated,
                    value_transformer=_call_args_debug_transformer)

        _messages.debug_log(_debug_string_func)

        return translated

    prompt_warning_issued = False

    def _call_pipeline_raw():
        nonlocal prompt_warning_issued
        try:
            if prompt_weighter is None:
                if not prompt_warning_issued:
                    _warn_prompt_lengths(pipeline, **kwargs)
                    prompt_warning_issued = True
                pipeline_to(pipeline, device)
                pipe_result = pipeline(**kwargs)
            else:
                args = _call_prompt_weighter()
                pipeline_to(pipeline, device)
                pipe_result = pipeline(**args)
                prompt_weighter.cleanup()
            return pipe_result
        except TypeError as e:
            null_call_name = _types.get_null_call_name(e)
            if null_call_name:
                raise UnsupportedPipelineConfigError(
                    'Missing pipeline module?, cannot call: ' + null_call_name)
            raise
        except AttributeError as e:
            null_attr_name = _types.get_null_attr_name(e)
            if null_attr_name:
                raise UnsupportedPipelineConfigError(
                    'Missing pipeline module?, cannot access: ' + null_attr_name)
            raise

    def _torch_oom_handler():
        global _LAST_CALLED_PIPELINE

        if pipeline is _LAST_CALLED_PIPELINE:
            _LAST_CALLED_PIPELINE = None

        # move the torch pipeline back to the CPU
        pipeline_to(pipeline, 'cpu')

        # empty the CUDA cache
        _memory.torch_gc()

        # force garbage collection
        gc.collect()

    def _call_pipeline():
        nonlocal enable_retry_pipe
        old_execution_device_property = None
        try:
            if hasattr(pipeline, '_execution_device'):
                # HACK
                # The device this returns is sometimes wrong and causes issues
                # with a randomly generated tensor (complaining about) being
                # generated on the wrong device as compared to the torch.Generator
                # object being used to generate it, this is a diffusers problem in
                # the code of this private property
                old_execution_device_property = pipeline.__class__._execution_device
                pipeline.__class__._execution_device = property(lambda s: torch.device(device))

            return _call_pipeline_raw()
        except _d_exceptions.TORCH_CUDA_OOM_EXCEPTIONS as e:
            _d_exceptions.raise_if_not_cuda_oom(e)
            _torch_oom_handler()
            raise _d_exceptions.OutOfMemoryError(e) from e
        except MemoryError as e:
            gc.collect()
            raise _d_exceptions.OutOfMemoryError('cpu (system memory)') from e
        except Exception:
            # same cleanup
            _torch_oom_handler()
            raise
        finally:
            if old_execution_device_property is not None:
                pipeline.__class__._execution_device = old_execution_device_property

    for cached_pipeline in _pipeline_cache.values():
        # hack to clear out cached 8bit pipelines, only one should ever
        # be in the object cache for sequential calls, this is necessary
        # for efficient main pipeline recall in DiffusionPipelineWrapper
        # which is used for the adetailer post processor
        if cached_pipeline.pipeline is not pipeline:
            bit8 = any(_util.is_loaded_in_8bit_bnb(module) for
                       module in get_pipeline_modules(cached_pipeline.pipeline).values())
            if bit8 and _torchutil.devices_equal(device, get_torch_device(cached_pipeline.pipeline)):
                _messages.debug_log(
                    f'Clearing out cached 8bit pipeline from cache: {cached_pipeline.pipeline.__class__.__name__}')
                _pipeline_cache.un_cache(cached_pipeline)
                del cached_pipeline
                gc.collect()
                _memory.torch_gc()

    if pipeline is _LAST_CALLED_PIPELINE:
        try:
            return _call_pipeline()
        except _d_exceptions.OutOfMemoryError:
            if not enable_retry_pipe:
                raise

            _messages.debug_log(
                f'Attempting to call pipeline '
                f'"{pipeline.__class__.__name__}" again after out '
                f'of memory condition and cleanup.')

            # retry after memory cleanup
            result = _call_pipeline()
            _LAST_CALLED_PIPELINE = pipeline
            return result

    if _LAST_CALLED_PIPELINE is not None and hasattr(_LAST_CALLED_PIPELINE, 'to'):
        _messages.debug_log(
            f'Moving previously called pipeline '
            f'"{_LAST_CALLED_PIPELINE.__class__.__name__}", back to the CPU.')

        pipeline_to(_LAST_CALLED_PIPELINE, 'cpu')

    try:
        result = _call_pipeline()
    except _d_exceptions.OutOfMemoryError:
        if not enable_retry_pipe:
            raise
        _messages.debug_log(
            f'Attempting to call pipeline '
            f'"{pipeline.__class__.__name__}" again after out '
            f'of memory condition and cleanup.')
        # allow for memory cleanup and try again
        # might be able to run now
        result = _call_pipeline()

    _LAST_CALLED_PIPELINE = pipeline
    return result


class PipelineCreationResult:

    model_path: _types.OptionalPath
    """
    Path the model was loaded from.
    """

    parsed_unet_uri: _uris.UNetUri | None
    """
    Parsed UNet URI if one was present
    """

    parsed_vae_uri: _uris.VAEUri | None
    """
    Parsed VAE URI if one was present
    """

    parsed_lora_uris: collections.abc.Sequence[_uris.LoRAUri]
    """
    Parsed LoRA URIs if any were present
    """

    parsed_ip_adapter_uris: collections.abc.Sequence[_uris.IPAdapterUri]
    """
    Parsed IP Adapter URIs if any were present
    """

    parsed_textual_inversion_uris: collections.abc.Sequence[_uris.TextualInversionUri]
    """
    Parsed Textual Inversion URIs if any were present
    """

    parsed_controlnet_uris: collections.abc.Sequence[_uris.ControlNetUri]
    """
    Parsed ControlNet URIs if any were present
    """

    parsed_t2i_adapter_uris: collections.abc.Sequence[_uris.T2IAdapterUri]
    """
    Parsed T2IAdapter URIs if any were present
    """

    parsed_image_encoder_uri: _uris.ImageEncoderUri | None
    """
    Parsed ImageEncoder URI if one was present
    """

    parsed_transformer_uri: _uris.TransformerUri | None
    """
    Parsed Transformer URI if one was present
    """

    def load_scheduler(self, scheduler_uri: _types.Uri | None):
        """
        Load a scheduler onto the pipeline using a URI specification.

        Passing ``None`` to the URI reloads the original scheduler that the model was loaded
        with, if no new scheduler has been set since then, this is a no-op.

        :param scheduler_uri: The scheduler URI
        """
        _schedulers.load_scheduler(self.pipeline, scheduler_uri)

    def set_vae_tiling_and_slicing(self, vae_tiling: bool, vae_slicing: bool):
        """
        Set the VAE tiling and slicing status of the pipeline.

        :param vae_tiling: vae tiling?
        :param vae_slicing: vae slicing?
        """
        set_vae_tiling_and_slicing(self.pipeline, tiling=vae_tiling, slicing=vae_slicing)

    def __init__(self,
                 model_path: _types.Path,
                 pipeline: diffusers.DiffusionPipeline,
                 parsed_unet_uri: _uris.UNetUri | None,
                 parsed_transformer_uri: _uris.TransformerUri | None,
                 parsed_vae_uri: _uris.VAEUri | None,
                 parsed_image_encoder_uri: _uris.ImageEncoderUri | None,
                 parsed_lora_uris: collections.abc.Sequence[_uris.LoRAUri],
                 parsed_ip_adapter_uris: collections.abc.Sequence[_uris.IPAdapterUri],
                 parsed_textual_inversion_uris: collections.abc.Sequence[_uris.TextualInversionUri],
                 parsed_controlnet_uris: collections.abc.Sequence[_uris.ControlNetUri],
                 parsed_t2i_adapter_uris: collections.abc.Sequence[_uris.T2IAdapterUri]):
        self.model_path = model_path
        self.parsed_unet_uri = parsed_unet_uri
        self.parsed_vae_uri = parsed_vae_uri
        self.parsed_lora_uris = parsed_lora_uris
        self.parsed_textual_inversion_uris = parsed_textual_inversion_uris
        self.parsed_controlnet_uris = parsed_controlnet_uris
        self.parsed_t2i_adapter_uris = parsed_t2i_adapter_uris
        self.parsed_ip_adapter_uris = parsed_ip_adapter_uris
        self.parsed_image_encoder_uri = parsed_image_encoder_uri
        self.parsed_transformer_uri = parsed_transformer_uri
        self._pipeline = pipeline

    @property
    def pipeline(self):
        return self._pipeline

    def get_pipeline_modules(self, names: collections.abc.Iterable[str]):
        """
        Get associated pipeline module such as ``vae`` etc, in
        a dictionary mapped from name to module value.

        Possible Module Names:

            * ``unet``
            * ``vae``
            * ``transformer``
            * ``text_encoder``
            * ``text_encoder_2``
            * ``text_encoder_3``
            * ``tokenizer``
            * ``tokenizer_2``
            * ``tokenizer_3``
            * ``safety_checker``
            * ``feature_extractor``
            * ``image_encoder``
            * ``adapter``
            * ``controlnet``
            * ``scheduler``


        If the module is not present or a recognized name, a :py:exc:`ValueError`
        will be thrown describing the module that is not part of the pipeline.

        :raise ValueError:

        :param names: module names, such as ``vae``, ``text_encoder``
        :return: dictionary
        """

        module_values = dict()

        acceptable_lookups = {
            'unet',
            'vae',
            'transformer',
            'text_encoder',
            'text_encoder_2',
            'text_encoder_3',
            'tokenizer',
            'tokenizer_2',
            'tokenizer_3',
            'safety_checker',
            'feature_extractor',
            'image_encoder',
            'adapter',
            'controlnet',
            'scheduler'
        }

        for name in names:
            if name not in acceptable_lookups:
                raise ValueError(f'"{name}" is not a recognized pipeline module name.')
            if not hasattr(self.pipeline, name):
                raise ValueError(f'Created pipeline does not possess a module named: "{name}".')
            module_values[name] = getattr(self.pipeline, name)

        return module_values

    def call(self,
             device: torch.device | str | None = _torchutil.default_device(),
             prompt_weighter: _promptweighters.PromptWeighter | None = None,
             **kwargs) -> diffusers.utils.BaseOutput:
        """
        Call **pipeline**, see: :py:func:`.call_pipeline`

        :param device: move the pipeline to this device before calling
        :param prompt_weighter: Optional prompt weighter for weighted prompt syntaxes
        :param kwargs: forward kwargs to pipeline
        :return: A subclass of :py:class:`diffusers.utils.BaseOutput`
        """
        return call_pipeline(self.pipeline,
                             device,
                             prompt_weighter,
                             **kwargs)


def create_diffusion_pipeline(
        model_path: str,
        model_type: _enums.ModelType = _enums.ModelType.SD,
        pipeline_type: _enums.PipelineType = _enums.PipelineType.TXT2IMG,
        revision: _types.OptionalString = None,
        variant: _types.OptionalString = None,
        subfolder: _types.OptionalString = None,
        dtype: _enums.DataType = _enums.DataType.AUTO,
        unet_uri: _types.OptionalUri = None,
        transformer_uri: _types.OptionalUri = None,
        vae_uri: _types.OptionalUri = None,
        lora_uris: _types.OptionalUris = None,
        lora_fuse_scale: _types.OptionalFloat = None,
        image_encoder_uri: _types.OptionalUri = None,
        ip_adapter_uris: _types.OptionalUris = None,
        textual_inversion_uris: _types.OptionalUris = None,
        text_encoder_uris: _types.OptionalUris = None,
        controlnet_uris: _types.OptionalUris = None,
        t2i_adapter_uris: _types.OptionalUris = None,
        quantizer_uri: _types.OptionalUri = None,
        quantizer_map: _types.OptionalStrings = None,
        pag: bool = False,
        safety_checker: bool = False,
        original_config: _types.OptionalString = None,
        auth_token: _types.OptionalString = None,
        device: str = _torchutil.default_device(),
        extra_modules: dict[str, typing.Any] | None = None,
        model_cpu_offload: bool = False,
        sequential_cpu_offload: bool = False,
        local_files_only: bool = False,
        missing_submodules_ok: bool = False
) -> PipelineCreationResult:
    """
    Create a :py:class:`diffusers.DiffusionPipeline` in dgenerate's in memory cacheing system.

    :param model_type:  :py:class:`dgenerate.pipelinewrapper.ModelType` enum value
    :param model_path: huggingface slug, huggingface blob link, path to folder on disk, path to file on disk
    :param pipeline_type: :py:class:`dgenerate.pipelinewrapper.PipelineType` enum value
    :param revision: huggingface repo revision (branch)
    :param variant: model weights name variant, for example 'fp16'
    :param subfolder: huggingface repo subfolder if applicable
    :param dtype: Optional :py:class:`dgenerate.pipelinewrapper.DataType` enum value
    :param unet_uri: Optional ``--unet`` URI string for specifying a specific UNet
    :param transformer_uri: Optional ``--transformer`` URI string for specifying a specific Transformer,
        currently this is only supported for Stable Diffusion 3 and Flux models.
    :param vae_uri: Optional ``--vae`` URI string for specifying a specific VAE
    :param lora_uris: Optional ``--loras`` URI strings for specifying LoRA weights
    :param lora_fuse_scale: Optional ``--lora-fuse-scale`` global LoRA fuse scale value.
        Once all LoRAs are merged with their individual scales, the merged weights will be fused
        into the pipeline at this scale. The default value is 1.0.
    :param image_encoder_uri: Optional ``--image-encoder`` URI for use with IP Adapter weights or Stable Cascade
    :param ip_adapter_uris: Optional ``--ip-adapters`` URI strings for specifying IP Adapter weights
    :param textual_inversion_uris: Optional ``--textual-inversions`` URI strings for specifying Textual Inversion weights
    :param text_encoder_uris: Optional user specified ``--text-encoders`` URIs that will be loaded on to the
        pipeline in order. A uri value of ``+`` or ``None`` indicates use default, a string value of ``null``
        indicates to explicitly not load any encoder all
    :param controlnet_uris: Optional ``--control-nets`` URI strings for specifying ControlNet models
    :param t2i_adapter_uris: Optional ``--t2i-adapters`` URI strings for specifying T2IAdapter models
    :param quantizer_uri: Optional ``--quantizer`` URI value
    :param quantizer_map: Collection of pipeline submodule names to which quantization should be applied when
        ``quantizer_uri`` is provided. Valid values include: ``unet``, ``transformer``, ``text_encoder``,
        ``text_encoder_2``, ``text_encoder_3``, and ``controlnet``. If ``None``, all supported modules will be quantized,
        except for ``controlnet``.
    :param pag: Use perturbed attention guidance?
    :param safety_checker: Safety checker enabled? default is ``False``
    :param original_config: Optional original training config .yaml file path when loading a single file checkpoint.
    :param auth_token: Optional huggingface API token for accessing repositories that are restricted to your account
    :param device: Optional ``--device`` string, defaults to "cuda"
    :param extra_modules: Extra module arguments to pass directly into
        :py:meth:`diffusers.DiffusionPipeline.from_single_file` or :py:meth:`diffusers.DiffusionPipeline.from_pretrained`
    :param model_cpu_offload: This pipeline has model_cpu_offloading enabled?
    :param sequential_cpu_offload: This pipeline has sequential_cpu_offloading enabled?
    :param local_files_only: Only look in the huggingface cache and do not connect to download models?
    :param missing_submodules_ok: It is okay if Text Encoders or VAE is missing from the checkpoint?

    :raises InvalidModelFileError:
    :raises InvalidModelUriError:
    :raises InvalidSchedulerNameError:
    :raises UnsupportedPipelineConfigError:
    :raises dgenerate.ModelNotFoundError:
    :raises dgenerate.ConfigNotFoundError:
    :raises dgenerate.NonHFModelDownloadError:
    :raises dgenerate.NonHFConfigDownloadError:
    :raises dgenerate.WebFileCacheOfflineModeException:

    :return: :py:class:`.TorchPipelineCreationResult`
    """
    __locals = locals()

    for name, value in __locals.items():
        if name.endswith('_uris') and isinstance(value, str):
            __locals[name] = [value]

    with _hfhub.with_hf_errors_as_model_not_found():
        return _create_diffusion_pipeline(**__locals)

class PipelineFactory:
    """
    Turns :py:func:`.create_diffusion_pipeline` into a factory that can
    repeatedly create a pipeline with the same arguments, possibly from cache.
    """

    def __init__(self,
                 model_path: str,
                 model_type: _enums.ModelType = _enums.ModelType.SD,
                 pipeline_type: _enums.PipelineType = _enums.PipelineType.TXT2IMG,
                 revision: _types.OptionalString = None,
                 variant: _types.OptionalString = None,
                 subfolder: _types.OptionalString = None,
                 dtype: _enums.DataType = _enums.DataType.AUTO,
                 unet_uri: _types.OptionalUri = None,
                 transformer_uri: _types.OptionalUri = None,
                 vae_uri: _types.OptionalUri = None,
                 lora_uris: _types.OptionalUris = None,
                 lora_fuse_scale: _types.OptionalFloat = None,
                 image_encoder_uri: _types.OptionalUri = None,
                 ip_adapter_uris: _types.OptionalUris = None,
                 textual_inversion_uris: _types.OptionalUris = None,
                 controlnet_uris: _types.OptionalUris = None,
                 t2i_adapter_uris: _types.OptionalUris = None,
                 text_encoder_uris: _types.OptionalUris = None,
                 quantizer_uri: _types.OptionalUri = None,
                 quantizer_map: _types.OptionalStrings = None,
                 pag: bool = False,
                 safety_checker: bool = False,
                 original_config: _types.OptionalString = None,
                 auth_token: _types.OptionalString = None,
                 device: str = _torchutil.default_device(),
                 extra_modules: dict[str, typing.Any] | None = None,
                 model_cpu_offload: bool = False,
                 sequential_cpu_offload: bool = False,
                 local_files_only: bool = False):
        self._args = {k: v for k, v in
                      _types.partial_deep_copy_container(locals()).items()
                      if k not in {'self'}}

    def __call__(self) -> PipelineCreationResult:
        """
        :raises InvalidModelFileError:
        :raises ModelNotFoundError:
        :raises InvalidModelUriError:
        :raises InvalidSchedulerNameError:
        :raises UnsupportedPipelineConfigError:
        :raises dgenerate.NonHFModelDownloadError:
        :raises dgenerate.NonHFConfigDownloadError:

        :return: :py:class:`.TorchPipelineCreationResult`
        """
        return create_diffusion_pipeline(**self._args)


def _format_pipeline_creation_debug_arg(arg_name, v):
    if isinstance(v, torch.dtype):
        return str(v)

    if isinstance(v, str):
        return f'"{v}"'

    if v.__class__.__module__ != 'builtins':
        return _types.class_and_id_string(v)

    if isinstance(v, list):
        return '[' + ', '.join(_format_pipeline_creation_debug_arg(None, v) for v in v) + ']'

    if isinstance(v, (set, frozenset)):
        return '{' + ', '.join(_format_pipeline_creation_debug_arg(None, v) for v in v) + '}'

    if isinstance(v, dict):
        return '{' + ', '.join(f'"{k}"={_format_pipeline_creation_debug_arg(None, v)}' for k, v in v.items()) + '}'

    return str(v)


def _pipeline_creation_args_debug(backend, cls, method, model, **kwargs):
    _messages.debug_log(
        lambda:
        f'{backend} Pipeline Creation Call: {cls.__name__}.{method.__name__}("{model}", ' +
        _textprocessing.debug_format_args(kwargs, _format_pipeline_creation_debug_arg, as_kwargs=True) + ')')

    return method(model, **kwargs)


def _text_encoder_default(uri):
    return uri is None or uri.strip() == '+'


def _text_encoder_null(uri):
    return uri and uri.strip().lower() == 'null'


def _pipeline_args_hasher(args):
    def text_encoder_uri_parse(uri):
        if _text_encoder_default(uri):
            return None

        if uri.strip() == 'help':
            return 'help'

        if uri.strip() == 'null':
            return 'null'

        return _uris.TextEncoderUri.parse(uri)

    quantizer_uri = args['quantizer_uri']

    custom_hashes = {
        'unet_uri': _uris.uri_hash_with_parser(_uris.UNetUri.parse),
        'transformer_uri': _uris.uri_hash_with_parser(_uris.TransformerUri),
        'vae_uri': _uris.uri_hash_with_parser(_uris.VAEUri.parse),
        'image_encoder_uri': _uris.uri_hash_with_parser(_uris.ImageEncoderUri),
        'lora_uris': _uris.uri_list_hash_with_parser(_uris.LoRAUri.parse),
        'ip_adapter_uris': _uris.uri_list_hash_with_parser(_uris.IPAdapterUri),
        'textual_inversion_uris': _uris.uri_list_hash_with_parser(_uris.TextualInversionUri.parse),
        'text_encoder_uris': _uris.uri_list_hash_with_parser(text_encoder_uri_parse),
        'controlnet_uris': _uris.uri_list_hash_with_parser(
            lambda s: _uris.ControlNetUri.parse(s, model_type=args['model_type']),
            exclude={'scale', 'start', 'end'}),
        't2i_adapter_uris': _uris.uri_list_hash_with_parser(_uris.T2IAdapterUri.parse,
                                                            exclude={'scale'}),
        'quantizer_uri':
            _uris.uri_hash_with_parser(
                _uris.get_quantizer_uri_class(quantizer_uri).parse)
            if quantizer_uri else lambda x: None,
        'quantizer_map': lambda x: hash(tuple(sorted(x))) if x else None
    }
    return _d_memoize.args_cache_key(args, custom_hashes=custom_hashes)


def _pipeline_on_hit(key, hit):
    _d_memoize.simple_cache_hit_debug("Torch Pipeline", key, hit.pipeline)


def _pipeline_on_create(key, new):
    _d_memoize.simple_cache_miss_debug('Torch Pipeline', key, new.pipeline)


def pipeline_class_supports_textual_inversion(cls: typing.Type[diffusers.DiffusionPipeline]):
    """
    Does a pipeline class support Textual Inversions?

    :param cls: ``diffusers`` pipeline class
    :return: ``True`` or ``False``
    """
    return any('TextualInversionLoaderMixin' in x.__name__ for x in cls.__bases__)


def pipeline_class_supports_lora(cls: typing.Type[diffusers.DiffusionPipeline]):
    """
    Does a pipeline class support LoRAs?

    :param cls: ``diffusers`` pipeline class
    :return: ``True`` or ``False``
    """
    return any('LoraLoaderMixin' in x.__name__ for x in cls.__bases__)


def pipeline_class_supports_ip_adapter(cls: typing.Type[diffusers.DiffusionPipeline]):
    """
    Does a pipeline class support IP Adapters?

    :param cls: ``diffusers`` pipeline class
    :return: ``True`` or ``False``
    """
    return any('IPAdapterMixin' in x.__name__ for x in cls.__bases__)


def get_pipeline_class(
        model_type: _enums.ModelType = _enums.ModelType.SD,
        pipeline_type: _enums.PipelineType = _enums.PipelineType.TXT2IMG,
        unet_uri: _types.OptionalUri = None,
        transformer_uri: _types.OptionalUri = None,
        vae_uri: _types.OptionalUri = None,
        lora_uris: _types.OptionalUris = None,
        image_encoder_uri: _types.OptionalUri = None,
        ip_adapter_uris: _types.OptionalUris = None,
        textual_inversion_uris: _types.OptionalUris = None,
        controlnet_uris: _types.OptionalUris = None,
        t2i_adapter_uris: _types.OptionalUris = None,
        pag: bool = False,
        help_mode: bool = False
) -> typing.Type[diffusers.DiffusionPipeline]:
    """
    Get an appropriate ``diffusers`` pipeline class for the provided arguments.

    :param model_type:  :py:class:`dgenerate.pipelinewrapper.ModelType` enum value
    :param pipeline_type: :py:class:`dgenerate.pipelinewrapper.PipelineType` enum value
    :param unet_uri: Optional ``--unet`` URI string for specifying a specific UNet
    :param transformer_uri: Optional ``--transformer`` URI string for specifying a specific Transformer,
        currently this is only supported for Stable Diffusion 3 and Flux models.
    :param vae_uri: Optional ``--vae`` URI string for specifying a specific VAE
    :param lora_uris: Optional ``--loras`` URI strings for specifying LoRA weights
    :param image_encoder_uri: Optional ``--image-encoder`` URI for use with IP Adapter weights or Stable Cascade
    :param ip_adapter_uris: Optional ``--ip-adapters`` URI strings for specifying IP Adapter weights
    :param textual_inversion_uris: Optional ``--textual-inversions`` URI strings for specifying Textual Inversion weights
    :param controlnet_uris: Optional ``--control-nets`` URI strings for specifying ControlNet models
    :param t2i_adapter_uris: Optional ``--t2i-adapters`` URI strings for specifying T2IAdapter models
    :param pag: Use perturbed attention guidance?
    :param help_mode: Return the class even if it does not support the selected ``pipeline_type``

    :raises UnsupportedPipelineConfigError:
    """

    # PAG check
    if pag:
        if not (model_type == _enums.ModelType.SD or
                model_type == _enums.ModelType.SDXL or
                model_type == _enums.ModelType.SD3 or
                model_type == _enums.ModelType.KOLORS):
            raise UnsupportedPipelineConfigError(
                'Perturbed attention guidance (--pag*) is only supported with '
                '--model-type sd, sdxl, kolors (txt2img), and sd3.')

        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                'Perturbed attention guidance (--pag*) is is not supported '
                'with --t2i-adapters.')

    # Flux model restrictions
    if _enums.model_type_is_flux(model_type):
        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                'Flux --model-type values are not compatible with --t2i-adapters.')
        if ip_adapter_uris and not image_encoder_uri:
            raise UnsupportedPipelineConfigError(
                'Must specify --image-encoder when using --ip-adapters with Flux.')
        if ip_adapter_uris and len(ip_adapter_uris) > 1:
            raise UnsupportedPipelineConfigError(
                'Flux --model-type values do not support multiple --ip-adapters.')

    if model_type == _enums.ModelType.FLUX_FILL:
        if pipeline_type != _enums.PipelineType.INPAINT:
            raise UnsupportedPipelineConfigError(
                'Flux fill --model-type value does not support anything but inpaint mode.'
            )

    # Deep Floyd model restrictions
    if _enums.model_type_is_floyd(model_type):
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                'Deep Floyd --model-type values are not compatible with --control-nets.')
        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                'Deep Floyd --model-type values are not compatible with --t2i-adapters.')
        if vae_uri:
            raise UnsupportedPipelineConfigError(
                'Deep Floyd --model-type values are not compatible with --vae.')
        if image_encoder_uri:
            raise UnsupportedPipelineConfigError(
                'Deep Floyd --model-type values are not compatible with --image-encoder.')

    # Stable Cascade model restrictions
    if _enums.model_type_is_s_cascade(model_type):
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                'Stable Cascade --model-type values are not compatible with --control-nets.')
        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                'Stable Cascade --model-type values are not compatible with --t2i-adapters.')
        if vae_uri:
            raise UnsupportedPipelineConfigError(
                'Stable Cascade --model-type values are not compatible with --vae.')

    # Torch SD3 restrictions
    if _enums.model_type_is_sd3(model_type):
        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                '--model-type sd3 is not compatible with --t2i-adapters.')
        if unet_uri:
            raise UnsupportedPipelineConfigError(
                '--model-type sd3 is not compatible with --unet.')

    # Torch Kolors restrictions
    if _enums.model_type_is_sd3(model_type):
        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                '--model-type kolors is not compatible with --t2i-adapters.')

    if transformer_uri:
        if not _enums.model_type_is_sd3(model_type) and not _enums.model_type_is_flux(model_type):
            raise UnsupportedPipelineConfigError(
                '--transformer is only supported for --model-type sd3 and flux.')

    # Incompatible combinations
    if controlnet_uris and t2i_adapter_uris:
        raise UnsupportedPipelineConfigError(
            '--control-nets and --t2i-adapters cannot be used together.')

    if image_encoder_uri and not ip_adapter_uris and model_type != _enums.ModelType.S_CASCADE:
        raise UnsupportedPipelineConfigError(
            '--image-encoder cannot be specified without --ip-adapters if --model-type is not s-cascade.')

    # Pix2Pix model restrictions
    is_pix2pix = _enums.model_type_is_pix2pix(model_type)

    if is_pix2pix:
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                'Pix2Pix --model-type values are not compatible with --control-nets.')
        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                'Pix2Pix --model-type values are not compatible with --t2i-adapters.')
        if image_encoder_uri and model_type != _enums.ModelType.PIX2PIX:
            raise UnsupportedPipelineConfigError(
                'Only Pix2Pix --model-type pix2pix is compatible '
                'with --image-encoder. Pix2Pix SDXL is not supported.')

    is_sdxl = _enums.model_type_is_sdxl(model_type)
    is_sd3 = _enums.model_type_is_sd3(model_type)

    sdxl_controlnet_union = False
    parsed_control_net_uris = None

    try:
        if controlnet_uris and is_sdxl:
            parsed_control_net_uris = [_uris.ControlNetUri.parse(s, model_type) for s in controlnet_uris]
            sdxl_controlnet_union = controlnet_uris and is_sdxl and any(
                s.mode is not None for s in parsed_control_net_uris)
    except _uris.InvalidControlNetUriError as e:
        raise UnsupportedPipelineConfigError(str(e)) from e

    def eq_cn_uri(
            uri1: _uris.ControlNetUri,
            uri2: _uris.ControlNetUri):
        equals = True

        for name, val in _types.get_public_attributes(uri1).items():
            if name not in {'scale', 'mode', 'start', 'end'}:
                equals = (equals and val == getattr(uri2, name))

        return equals

    if sdxl_controlnet_union and \
            any(not eq_cn_uri(parsed_control_net_uris[0], u)
                for u in parsed_control_net_uris):
        raise UnsupportedPipelineConfigError(
            'SDXL ControlNet Union mode requires all ControlNet '
            'model URIs to be identical with the exception of the '
            '"scale", "mode", "start", and "end" arguments.'
        )

    # Pipeline class selection
    if _enums.model_type_is_upscaler(model_type):
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                'Upscaler models are not compatible with --control-nets.')
        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                'Upscaler models are not compatible with --t2i-adapters.')
        if image_encoder_uri:
            raise UnsupportedPipelineConfigError(
                'Upscaler models are not compatible with --image-encoder.')
        if pipeline_type != _enums.PipelineType.IMG2IMG and not help_mode:
            raise UnsupportedPipelineConfigError(
                'Upscaler models only work with img2img generation, IE: --image-seeds (with no image masks).')

        pipeline_class = (
            diffusers.StableDiffusionUpscalePipeline
            if model_type == _enums.ModelType.UPSCALER_X4
            else diffusers.StableDiffusionLatentUpscalePipeline
        )
    else:
        if pipeline_type == _enums.PipelineType.TXT2IMG:
            if is_pix2pix:
                if not help_mode:
                    raise UnsupportedPipelineConfigError(
                        'Pix2Pix models only work in img2img / inpaint mode and cannot work without --image-seeds.')
                else:
                    if is_sdxl:
                        # noinspection PyUnusedLocal
                        pipeline_class = diffusers.StableDiffusionXLInstructPix2PixPipeline
                    elif is_sd3:
                        # noinspection PyUnusedLocal
                        pipeline_class = dgenerate.extras.ultraedit.StableDiffusion3InstructPix2PixPipeline
                    else:
                        # noinspection PyUnusedLocal
                        pipeline_class = diffusers.StableDiffusionInstructPix2PixPipeline

            if model_type == _enums.ModelType.FLUX_KONTEXT:
                raise UnsupportedPipelineConfigError(
                    'Flux Kontext models only work in img2img / inpaint mode and cannot work without --image-seeds.'
                )

            if model_type == _enums.ModelType.FLUX_FILL:
                raise UnsupportedPipelineConfigError(
                    'Flux Fill models only work in inpaint mode and cannot work without --image-seeds.'
                )

            if model_type == _enums.ModelType.IF:
                pipeline_class = diffusers.IFPipeline
            elif model_type == _enums.ModelType.IFS:
                if not help_mode:
                    raise UnsupportedPipelineConfigError(
                        'Deep Floyd IF super-resolution (IFS) only works in '
                        'img2img mode and cannot work without --image-seeds.')
                else:
                    pipeline_class = diffusers.IFSuperResolutionPipeline
            elif model_type == _enums.ModelType.S_CASCADE:
                pipeline_class = diffusers.StableCascadePriorPipeline
            elif model_type == _enums.ModelType.S_CASCADE_DECODER:
                pipeline_class = diffusers.StableCascadeDecoderPipeline
            elif model_type == _enums.ModelType.FLUX:
                if controlnet_uris:
                    pipeline_class = diffusers.FluxControlNetPipeline
                else:
                    pipeline_class = diffusers.FluxPipeline
            elif model_type == _enums.ModelType.SD3:
                if pag:
                    pipeline_class = diffusers.StableDiffusion3PAGPipeline
                elif controlnet_uris:
                    if pag:
                        raise UnsupportedPipelineConfigError(
                            'Stable Diffusion 3 does not support --pag with controlnets.')

                    pipeline_class = diffusers.StableDiffusion3ControlNetPipeline
                else:
                    pipeline_class = diffusers.StableDiffusion3Pipeline
            elif model_type == _enums.ModelType.KOLORS:
                if controlnet_uris:
                    if pag:
                        raise UnsupportedPipelineConfigError(
                            'Kolors ControlNet mode does not support PAG')
                    else:
                        pipeline_class = _kolors.KolorsControlNetPipeline
                else:
                    if pag:
                        pipeline_class = diffusers.KolorsPAGPipeline
                    else:
                        pipeline_class = diffusers.KolorsPipeline
            elif t2i_adapter_uris:
                # The custom type is a hack to support from_single_file for SD1.5 - 2
                # models with the associated pipeline class which does not inherit
                # the correct mixin to do so but can use the mixin just fine
                pipeline_class = (
                    diffusers.StableDiffusionXLAdapterPipeline
                    if is_sdxl
                    else diffusers.StableDiffusionAdapterPipeline
                )
            elif controlnet_uris:
                if is_sdxl:
                    if pag:
                        if sdxl_controlnet_union:
                            raise UnsupportedPipelineConfigError(
                                'SDXL ControlNet Union mode does not support PAG')
                        pipeline_class = diffusers.StableDiffusionXLControlNetPAGPipeline
                    else:
                        if sdxl_controlnet_union:
                            pipeline_class = \
                                diffusers.StableDiffusionXLControlNetUnionPipeline
                        else:
                            pipeline_class = diffusers.StableDiffusionXLControlNetPipeline
                else:
                    if pag:
                        pipeline_class = diffusers.StableDiffusionControlNetPAGPipeline
                    else:
                        pipeline_class = diffusers.StableDiffusionControlNetPipeline
            else:
                if is_sdxl:
                    if pag:
                        pipeline_class = diffusers.StableDiffusionXLPAGPipeline
                    else:
                        pipeline_class = diffusers.StableDiffusionXLPipeline
                else:
                    if pag:
                        pipeline_class = diffusers.StableDiffusionPAGPipeline
                    else:
                        pipeline_class = diffusers.StableDiffusionPipeline

        elif pipeline_type == _enums.PipelineType.IMG2IMG:
            if controlnet_uris:
                if is_pix2pix:
                    raise UnsupportedPipelineConfigError(
                        'Pix2Pix models are not compatible with --control-nets.')

            if model_type == _enums.ModelType.FLUX_FILL:
                raise UnsupportedPipelineConfigError(
                    'Flux Fill models only work in inpaint mode.'
                )

            if is_pix2pix:
                if is_sdxl:
                    # noinspection PyUnusedLocal
                    pipeline_class = diffusers.StableDiffusionXLInstructPix2PixPipeline
                elif is_sd3:
                    # noinspection PyUnusedLocal
                    pipeline_class = dgenerate.extras.ultraedit.StableDiffusion3InstructPix2PixPipeline
                else:
                    # noinspection PyUnusedLocal
                    pipeline_class = diffusers.StableDiffusionInstructPix2PixPipeline
            elif model_type == _enums.ModelType.IF:
                pipeline_class = diffusers.IFImg2ImgPipeline
            elif model_type == _enums.ModelType.IFS:
                pipeline_class = diffusers.IFSuperResolutionPipeline
            elif model_type == _enums.ModelType.IFS_IMG2IMG:
                pipeline_class = diffusers.IFImg2ImgSuperResolutionPipeline
            elif model_type == _enums.ModelType.S_CASCADE:
                pipeline_class = diffusers.StableCascadePriorPipeline
            elif model_type == _enums.ModelType.S_CASCADE_DECODER:
                raise UnsupportedPipelineConfigError(
                    'Stable Cascade decoder models do not support img2img.')
            elif model_type == _enums.ModelType.FLUX:
                if controlnet_uris:
                    pipeline_class = diffusers.FluxControlNetImg2ImgPipeline
                else:
                    pipeline_class = diffusers.FluxImg2ImgPipeline
            elif model_type == _enums.ModelType.FLUX_KONTEXT:
                if controlnet_uris:
                    raise UnsupportedPipelineConfigError(
                        '--model-type flux-kontext does not support ControlNet models.'
                    )
                pipeline_class = diffusers.FluxKontextPipeline
            elif model_type == _enums.ModelType.SD3:
                if controlnet_uris:
                    raise UnsupportedPipelineConfigError(
                        '--model-type sd3 does not support img2img mode with ControlNet models.')
                if pag:
                    pipeline_class = diffusers.StableDiffusion3PAGImg2ImgPipeline
                else:
                    pipeline_class = diffusers.StableDiffusion3Img2ImgPipeline

            elif model_type == _enums.ModelType.KOLORS:
                if controlnet_uris:
                    if pag:
                        raise UnsupportedPipelineConfigError(
                            'Kolors ControlNet does not support PAG in img2img mode'
                        )
                    pipeline_class = _kolors.KolorsControlNetImg2ImgPipeline
                else:
                    if pag:
                        raise UnsupportedPipelineConfigError(
                            'Kolors does not support PAG in img2img mode'
                        )
                    pipeline_class = diffusers.KolorsImg2ImgPipeline
            elif t2i_adapter_uris:
                raise UnsupportedPipelineConfigError(
                    'img2img mode is not supported with --t2i-adapters.')
            elif controlnet_uris:
                if is_sdxl:
                    if pag:
                        if sdxl_controlnet_union:
                            raise UnsupportedPipelineConfigError(
                                'SDXL ControlNet Union mode does not support PAG')
                        pipeline_class = diffusers.StableDiffusionXLControlNetPAGImg2ImgPipeline
                    else:
                        if sdxl_controlnet_union:
                            pipeline_class = \
                                diffusers.StableDiffusionXLControlNetUnionImg2ImgPipeline
                        else:
                            pipeline_class = diffusers.StableDiffusionXLControlNetImg2ImgPipeline
                else:
                    if pag:
                        raise UnsupportedPipelineConfigError(
                            '--model-type sd (Stable Diffusion 1.5 - 2.*) '
                            'does not support --pag in img2img mode with ControlNet models.')
                    else:
                        pipeline_class = diffusers.StableDiffusionControlNetImg2ImgPipeline
            else:
                if is_sdxl:
                    if pag:
                        pipeline_class = diffusers.StableDiffusionXLPAGImg2ImgPipeline
                    else:
                        pipeline_class = diffusers.StableDiffusionXLImg2ImgPipeline
                else:
                    if pag:
                        raise UnsupportedPipelineConfigError(
                            '--model-type sd (Stable Diffusion 1.5 - 2.*) '
                            'does not support --pag in img2img mode.')
                    else:
                        pipeline_class = diffusers.StableDiffusionImg2ImgPipeline

        elif pipeline_type == _enums.PipelineType.INPAINT:

            if _enums.model_type_is_s_cascade(model_type):
                raise UnsupportedPipelineConfigError(
                    'Stable Cascade model types do not support inpainting.')
            if _enums.model_type_is_upscaler(model_type):
                raise UnsupportedPipelineConfigError(
                    'Stable Diffusion upscaler model types do not support inpainting.')

            if is_pix2pix:
                if is_sdxl:
                    # noinspection PyUnusedLocal
                    pipeline_class = dgenerate.extras.ultraedit.StableDiffusionXLInstructPix2PixPipeline
                elif is_sd3:
                    # noinspection PyUnusedLocal
                    pipeline_class = dgenerate.extras.ultraedit.StableDiffusion3InstructPix2PixPipeline
                else:
                    # noinspection PyUnusedLocal
                    pipeline_class = dgenerate.extras.ultraedit.StableDiffusionInstructPix2PixPipeline
            elif model_type == _enums.ModelType.FLUX:
                if controlnet_uris:
                    pipeline_class = diffusers.FluxControlNetInpaintPipeline
                else:
                    pipeline_class = diffusers.FluxInpaintPipeline
            elif model_type == _enums.ModelType.FLUX_FILL:
                if controlnet_uris:
                    raise UnsupportedPipelineConfigError(
                        '--model-type flux-fill does not support ControlNet models.')
                pipeline_class = diffusers.FluxFillPipeline
            elif model_type == _enums.ModelType.FLUX_KONTEXT:
                if controlnet_uris:
                    raise UnsupportedPipelineConfigError(
                        '--model-type flux-kontext does not support ControlNet models.')
                pipeline_class = diffusers.FluxKontextInpaintPipeline
            elif model_type == _enums.ModelType.IF:
                pipeline_class = diffusers.IFInpaintingPipeline
            elif model_type == _enums.ModelType.IFS:
                pipeline_class = diffusers.IFInpaintingSuperResolutionPipeline
            elif model_type == _enums.ModelType.SD3:
                if controlnet_uris:
                    return diffusers.StableDiffusion3ControlNetInpaintingPipeline
                if pag:
                    raise UnsupportedPipelineConfigError(
                        '--model-type sd3 does not support --pag in inpaint mode.'
                    )

                pipeline_class = diffusers.StableDiffusion3InpaintPipeline
            elif model_type == _enums.ModelType.KOLORS:
                if controlnet_uris:
                    if pag:
                        raise UnsupportedPipelineConfigError(
                            'Kolors ControlNet does not support PAG in inpaint mode'
                        )
                    pipeline_class = _kolors.KolorsControlNetInpaintPipeline
                else:
                    if pag:
                        raise UnsupportedPipelineConfigError(
                            'Kolors does not support PAG in inpaint mode'
                        )
                    pipeline_class = _kolors.KolorsInpaintPipeline
            elif t2i_adapter_uris:
                raise UnsupportedPipelineConfigError(
                    'inpaint mode is not supported with --t2i-adapters.')
            elif controlnet_uris:
                if is_sdxl:
                    if pag:
                        raise UnsupportedPipelineConfigError(
                            '--model-type sdxl does not support --pag '
                            'in inpaint mode with ControlNet models.'
                        )
                    else:
                        if sdxl_controlnet_union:
                            pipeline_class = \
                                diffusers.StableDiffusionXLControlNetUnionInpaintPipeline
                        else:
                            pipeline_class = diffusers.StableDiffusionXLControlNetInpaintPipeline
                else:
                    if pag:
                        pipeline_class = diffusers.StableDiffusionControlNetPAGInpaintPipeline
                    else:
                        pipeline_class = diffusers.StableDiffusionControlNetInpaintPipeline
            else:
                if is_sdxl:
                    if pag:
                        pipeline_class = diffusers.StableDiffusionXLPAGInpaintPipeline
                    else:
                        pipeline_class = diffusers.StableDiffusionXLInpaintPipeline
                else:
                    if pag:
                        raise UnsupportedPipelineConfigError(
                            '--model-type sd (Stable Diffusion 1.5 - 2.*) '
                            'does not support --pag in inpaint mode.')
                    else:
                        pipeline_class = diffusers.StableDiffusionInpaintPipeline
        else:
            # Should be impossible
            raise UnsupportedPipelineConfigError('Pipeline type not implemented.')

    if lora_uris and not pipeline_class_supports_lora(pipeline_class):
        raise UnsupportedPipelineConfigError(
            f'Given current arguments, '
            f'--model-type {_enums.get_model_type_string(model_type)} '
            f'(pipeline: {pipeline_class.__name__}) does not support LoRAs.')

    if textual_inversion_uris and not pipeline_class_supports_textual_inversion(pipeline_class):
        raise UnsupportedPipelineConfigError(
            f'Given current arguments, '
            f'--model-type {_enums.get_model_type_string(model_type)} '
            f'(pipeline: {pipeline_class.__name__}) does not support Textual Inversions.')

    if ip_adapter_uris and not pipeline_class_supports_ip_adapter(pipeline_class):
        raise UnsupportedPipelineConfigError(
            f'Given current arguments, '
            f'--model-type {_enums.get_model_type_string(model_type)} '
            f'(pipeline: {pipeline_class.__name__}) does not support IP Adapters.')

    return pipeline_class


def _enforce_pipeline_cache_size(new_pipeline_size):
    _pipeline_cache.enforce_cpu_mem_constraints(
        _constants.PIPELINE_CACHE_MEMORY_CONSTRAINTS,
        size_var='pipeline_size',
        new_object_size=new_pipeline_size)


@_memoize(_pipeline_cache,
          exceptions={'local_files_only', 'missing_submodules_ok'},
          hasher=_pipeline_args_hasher,
          extra_identities=[lambda m: m.pipeline],
          on_hit=_pipeline_on_hit,
          on_create=_pipeline_on_create)
def _create_diffusion_pipeline(
        model_path: str,
        model_type: _enums.ModelType = _enums.ModelType.SD,
        pipeline_type: _enums.PipelineType = _enums.PipelineType.TXT2IMG,
        revision: _types.OptionalString = None,
        variant: _types.OptionalString = None,
        subfolder: _types.OptionalString = None,
        dtype: _enums.DataType = _enums.DataType.AUTO,
        unet_uri: _types.OptionalUri = None,
        transformer_uri: _types.OptionalUri = None,
        vae_uri: _types.OptionalUri = None,
        lora_uris: _types.OptionalUris = None,
        lora_fuse_scale: _types.OptionalFloat = None,
        image_encoder_uri: _types.OptionalUri = None,
        ip_adapter_uris: _types.OptionalUris = None,
        textual_inversion_uris: _types.OptionalUris = None,
        text_encoder_uris: _types.OptionalUris = None,
        controlnet_uris: _types.OptionalUris = None,
        t2i_adapter_uris: _types.OptionalUris = None,
        quantizer_uri: _types.OptionalUri = None,
        quantizer_map: _types.OptionalStrings = None,
        pag: bool = False,
        safety_checker: bool = False,
        original_config: _types.OptionalString = None,
        auth_token: _types.OptionalString = None,
        device: str = _torchutil.default_device(),
        extra_modules: dict[str, typing.Any] | None = None,
        model_cpu_offload: bool = False,
        sequential_cpu_offload: bool = False,
        local_files_only: bool = False,
        missing_submodules_ok: bool = False
) -> PipelineCreationResult:
    # Ensure model path is specified
    if not model_path:
        raise ValueError('model_path must be specified.')

    # Offload checks
    if model_cpu_offload and sequential_cpu_offload:
        raise UnsupportedPipelineConfigError(
            'model_cpu_offload and sequential_cpu_offload may not be enabled simultaneously.')

    # Device check
    if not _torchutil.is_valid_device_string(device):
        raise UnsupportedPipelineConfigError(
            f'Invalid device argument, {_torchutil.invalid_device_message(device, cap=False)}')

    # Quantizer map check
    quantizer_map_vals = [
        'unet',
        'transformer',
        'text_encoder',
        'text_encoder_2',
        'text_encoder_3',
        'controlnet'
    ]

    if quantizer_map is not None:
        for map_value in quantizer_map:
            if map_value not in quantizer_map_vals:
                raise UnsupportedPipelineConfigError(
                    f'Unknown quantizer_map value: {map_value}, '
                    f'must be one of: {_textprocessing.oxford_comma(quantizer_map_vals, "or")}'
                )

    if not _hfhub.is_single_file_model_load(model_path) and original_config:
        raise UnsupportedPipelineConfigError(
            'Loading original config .yaml file is not supported '
            'when loading from a Hugging Face repo.'
        )

    if quantizer_uri and quantizer_uri.split(';')[0].strip() in {'bnb', 'bitsandbytes'}:
        if dtype is _enums.DataType.AUTO:
            # Default to all modules to float32 if no dtype is specified when using bitsandbytes
            dtype = _enums.DataType.FLOAT32

    if original_config:
        # only instance where it really makes sense to raise config not found
        # everything else is indicative of a missing model
        with _hfhub.with_hf_errors_as_config_not_found():
            original_config = _hfhub.download_non_hf_slug_config(original_config)

    model_path = _hfhub.download_non_hf_slug_model(model_path)

    if (_hfhub.is_single_file_model_load(model_path)
        and _enums.model_type_is_kolors(model_type)
    ):
        raise UnsupportedPipelineConfigError(
            'Kolors models cannot be loaded from a single file.'
        )

    model_index = _util.fetch_model_index_dict(
        model_path,
        subfolder=subfolder,
        revision=revision,
        use_auth_token=auth_token,
        local_files_only=local_files_only
    )

    if '_class_name' in model_index:
        model_class_name = model_index['_class_name']

        model_checks = [
            (_enums.model_type_is_flux, ('^Flux.*', 'Flux')),
            (_enums.model_type_is_sd3, ('^StableDiffusion3.*', 'Stable Diffusion 3')),
            (_enums.model_type_is_sdxl, ('^StableDiffusionXL.*', 'Stable Diffusion XL')),
            (_enums.model_type_is_sd15, ('^StableDiffusion[^X3].*', 'Stable Diffusion')),
            (_enums.model_type_is_sd2, ('^StableDiffusion[^X3].*', 'Stable Diffusion')),
            (_enums.model_type_is_s_cascade, ('^StableCascade.*', 'Stable Cascade')),
            (_enums.model_type_is_kolors, ('^Kolors.*', 'Kolors')),
            (_enums.model_type_is_floyd, ('^IF.*', 'Deep Floyd')),
        ]

        # exceptions to the rules above
        # where left and right evaluate True
        model_fallback_checks = [
            (lambda x: x == _enums.ModelType.SD, '^LatentConsistency.*')
        ]

        for check_func, (pattern, title) in model_checks:
            if check_func(model_type) and re.match(pattern, model_class_name) is None:
                # are there any exceptions to this such as legacy configs?
                if not any(
                        check(model_type) and
                        re.match(pattern, model_class_name) is not None
                        for check, pattern in model_fallback_checks
                ):
                    raise UnsupportedPipelineConfigError(
                        f'{model_path} is not a {title} model, '
                        f'incorrect --model-type value: {_enums.get_model_type_string(model_type)}'
                    )

    pipeline_class = get_pipeline_class(
        model_type=model_type,
        pipeline_type=pipeline_type,
        unet_uri=unet_uri,
        transformer_uri=transformer_uri,
        vae_uri=vae_uri,
        lora_uris=lora_uris,
        image_encoder_uri=image_encoder_uri,
        ip_adapter_uris=ip_adapter_uris,
        textual_inversion_uris=textual_inversion_uris,
        controlnet_uris=controlnet_uris,
        t2i_adapter_uris=t2i_adapter_uris,
        pag=pag
    )

    text_encoder_count = len(
        [a for a in inspect.getfullargspec(pipeline_class.__init__).args if a.startswith('text_encoder')])

    if not text_encoder_uris:
        text_encoder_uris = []

    if len(text_encoder_uris) > text_encoder_count:
        raise UnsupportedPipelineConfigError('To many text encoder URIs specified.')

    if extra_modules is not None:
        _messages.debug_log('Checking extra_modules for meta tensors...')
        for module in extra_modules.items():
            if module[1] is None:
                continue
            _messages.debug_log(f'Checking extra module {module[0]} = {module[1].__class__}...')
            try:
                if get_torch_device(module[1]).type == 'meta':
                    _messages.debug_log(f'"{module[0]}" has meta tensors.')
                    _disable_to(
                        module[1],
                        module[0] == 'vae'
                    )
            except ValueError:
                _messages.debug_log(
                    f'Unable to get device of {module[0]} = {module[1].__class__}')
        extra_modules = extra_modules.copy()
    else:
        extra_modules = dict()

    unet_override = 'unet' in extra_modules
    vae_override = 'vae' in extra_modules
    controlnet_override = 'controlnet' in extra_modules
    adapter_override = 'adapter' in extra_modules
    image_encoder_override = 'image_encoder' in extra_modules
    safety_checker_override = 'safety_checker' in extra_modules
    transformer_override = 'transformer' in extra_modules

    if 'text_encoder' in extra_modules and text_encoder_count == 0:
        raise UnsupportedPipelineConfigError('To many text encoders specified.')

    if 'text_encoder_2' in extra_modules and text_encoder_count < 2:
        raise UnsupportedPipelineConfigError('To many text encoders specified.')

    if 'text_encoder_3' in extra_modules and text_encoder_count < 3:
        raise UnsupportedPipelineConfigError('To many text encoders specified.')

    # noinspection PyTypeChecker
    text_encoders: list[str] = list(text_encoder_uris)

    if len(text_encoders) > 0 and _text_encoder_null(text_encoders[0]):
        extra_modules['text_encoder'] = None
    if len(text_encoders) > 1 and _text_encoder_null(text_encoders[1]):
        extra_modules['text_encoder_2'] = None
    if len(text_encoders) > 2 and _text_encoder_null(text_encoders[2]):
        extra_modules['text_encoder_3'] = None

    text_encoder_override = 'text_encoder' in extra_modules
    text_encoder_2_override = 'text_encoder_2' in extra_modules
    text_encoder_3_override = 'text_encoder_3' in extra_modules

    if len(text_encoders) > 0 and text_encoder_override:
        text_encoders[0] = None
    if len(text_encoders) > 1 and text_encoder_2_override:
        text_encoders[1] = None
    if len(text_encoders) > 2 and text_encoder_3_override:
        text_encoders[2] = None

    # If we are not auto caching UNet/Transformer, VAE, and text encoders
    # into other caches, we should estimate a cache size for the pipeline
    # that includes them

    pipeline_cached_with_submodules = \
        model_cpu_offload or sequential_cpu_offload or quantizer_uri

    estimated_memory_usage = estimate_pipeline_cache_footprint(
        model_type=model_type,
        model_path=model_path,
        revision=revision,
        variant=variant,
        subfolder=subfolder,
        include_unet_or_transformer=pipeline_cached_with_submodules or bool(lora_uris),
        include_vae=pipeline_cached_with_submodules,
        include_text_encoders=pipeline_cached_with_submodules or bool(lora_uris),
        lora_uris=lora_uris,
        image_encoder_uri=image_encoder_uri,
        ip_adapter_uris=ip_adapter_uris,
        textual_inversion_uris=textual_inversion_uris,
        safety_checker=safety_checker and not safety_checker_override,
        auth_token=auth_token,
        extra_args=extra_modules,
        local_files_only=local_files_only
    )

    _messages.debug_log(
        f'Creating Torch Pipeline: "{pipeline_class.__name__}", '
        f'Estimated CPU Side Memory Use: {_memory.bytes_best_human_unit(estimated_memory_usage)}')

    # Helper function to determine if quantization should be applied to a module
    def should_apply_quantizer(module_name):
        if not quantizer_uri:
            return False
        if quantizer_map is None:
            return True
        return module_name in quantizer_map

    # Helper function to determine device_map - always use the selected device currently
    def get_device_map_for_quantizer(quantizer_uri):
        return device if quantizer_uri else None

    # we need to manually emulate bitsandbytes 'compute_dtype'
    # casting for sdnq as it has trouble dequanting to anything
    # but float16 on forward which messes with diffusers
    # in various ways when a model is loaded in float32
    sdnq_cast_hack = False

    if quantizer_uri and (
        _uris.get_quantizer_uri_class(quantizer_uri) is
        _uris.SDNQQuantizerUri
    ):
        sdnq_cast_hack = True

    uri_quant_check = []
    manual_quantizer_components = set()

    # Check text encoder URIs
    for idx, encoder_uri in enumerate(text_encoder_uris):
        if encoder_uri and encoder_uri.lower() not in {'+', 'help', 'null'}:
            parsed_uri = _uris.TextEncoderUri.parse(encoder_uri)
            uri_quant_check.append(parsed_uri)
            if parsed_uri.quantizer:
                encoder_name = f'text_encoder{"_2" if idx == 1 else "_3" if idx == 2 else ""}'
                manual_quantizer_components.add(encoder_name)
                quantizer_class = _uris.get_quantizer_uri_class(parsed_uri.quantizer)
                if quantizer_class is _uris.SDNQQuantizerUri:
                    sdnq_cast_hack = True


    # Check transformer URI
    if transformer_uri:
        parsed_uri = _uris.TransformerUri.parse(transformer_uri)
        uri_quant_check.append(parsed_uri)
        if parsed_uri.quantizer:
            manual_quantizer_components.add('transformer')
            quantizer_class = _uris.get_quantizer_uri_class(parsed_uri.quantizer)
            if quantizer_class is _uris.SDNQQuantizerUri:
                sdnq_cast_hack = True

    # Check unet URI
    if unet_uri:
        parsed_uri = _uris.UNetUri.parse(unet_uri)
        uri_quant_check.append(parsed_uri)
        if parsed_uri.quantizer:
            manual_quantizer_components.add('unet')
            quantizer_class = _uris.get_quantizer_uri_class(parsed_uri.quantizer)
            if quantizer_class is _uris.SDNQQuantizerUri:
                sdnq_cast_hack = True

    # Check controlnet URIs
    if controlnet_uris:
        for controlnet_uri in controlnet_uris:
            parsed_uri = _uris.ControlNetUri.parse(controlnet_uri, model_type=model_type)
            uri_quant_check.append(parsed_uri)
            if parsed_uri.quantizer:
                manual_quantizer_components.add('controlnet')
                quantizer_class = _uris.get_quantizer_uri_class(parsed_uri.quantizer)
                if quantizer_class is _uris.SDNQQuantizerUri:
                    sdnq_cast_hack = True

    if quantizer_uri or any(p.quantizer for p in uri_quant_check):
        # for now, just knock out anything cached on the gpu, such as the last pipeline
        # the quantized pipeline modules are likely going to go straight onto the GPU
        # immediately, and they are guaranteed to be of non-trivial size
        _devicecache.clear_device_cache(device)

    # Granular component caching for quantization scenarios
    cached_component_paths = {}
    manual_quantizers = {}
    components_to_cache = []

    # Determine if we need granular caching
    needs_granular_caching = (quantizer_uri and (_hfhub.is_single_file_model_load(model_path) or lora_uris)) or \
                             (manual_quantizer_components and lora_uris)

    if needs_granular_caching:

        # Determine which components to cache
        if lora_uris:
            # For LoRA + quantizer: cache only components affected by LoRA
            components_to_cache = _get_affected_components_for_lora(pipeline_class)

            # Also include any manual components with quantizers that are LoRA-affected
            lora_affected_components = set(components_to_cache)
            for component in manual_quantizer_components:
                if component in lora_affected_components:
                    if component not in components_to_cache:
                        components_to_cache.append(component)
        else:
            # For single file + quantizer: cache all quantizable components
            components_to_cache = _get_quantizable_components(model_type)

        # Filter based on quantizer_map ONLY for components that would get auto-quantized
        # Components with manual quantizers should always be cached when LoRAs are involved
        if quantizer_map is not None:
            # Keep components that either:
            # 1. Have manual quantizers (always cache these when LoRAs are involved)
            # 2. Are in the quantizer_map (for auto-quantization)
            if lora_uris:
                # When LoRAs are involved, always cache components with manual quantizers
                components_to_cache = [c for c in components_to_cache
                                       if c in manual_quantizer_components or c in quantizer_map]
            else:
                # For single file without LoRAs, only respect quantizer_map
                components_to_cache = [c for c in components_to_cache if c in quantizer_map]

        # Collect manual URIs for components that will be cached
        manual_component_uris = {}
        if unet_uri and 'unet' in components_to_cache:
            manual_component_uris['unet'] = unet_uri
        if transformer_uri and 'transformer' in components_to_cache:
            manual_component_uris['transformer'] = transformer_uri

        # Handle text encoder URIs
        if text_encoder_uris:
            for idx, encoder_uri in enumerate(text_encoder_uris):
                if not _text_encoder_default(encoder_uri):
                    encoder_name = f'text_encoder{"_2" if idx == 1 else "_3" if idx == 2 else ""}'
                    if encoder_name in components_to_cache:
                        manual_component_uris[encoder_name] = encoder_uri

        # Perform granular caching
        if components_to_cache:
            cached_component_paths, manual_quantizers = _cache_components_granular(
                model_path=model_path,
                model_type=model_type,
                pipeline_type=pipeline_type,
                components_to_cache=components_to_cache,
                lora_uris=lora_uris,
                lora_fuse_scale=lora_fuse_scale,
                revision=revision,
                variant=variant,
                subfolder=subfolder,
                dtype=dtype,
                original_config=original_config,
                auth_token=auth_token,
                manual_component_uris=manual_component_uris,
                vae_uri=vae_uri
            )

            _messages.debug_log(
                f"Granular caching complete. Cached components: {list(cached_component_paths.keys())}")
            if manual_component_uris:
                _messages.debug_log(
                    f"Manual component URIs processed: {list(manual_component_uris.keys())}")
            if manual_quantizers:
                _messages.debug_log(
                    f"Manual quantizers preserved: {list(manual_quantizers.keys())}")
        else:
            manual_quantizers = {}

    # ControlNet and VAE loading

    # Used during pipeline load
    creation_kwargs = {}

    torch_dtype = _enums.get_torch_dtype(dtype)

    parsed_controlnet_uris = []
    parsed_t2i_adapter_uris = []
    parsed_image_encoder_uri = None
    parsed_unet_uri = None
    parsed_vae_uri = None
    parsed_transformer_uri = None

    pipe_params = inspect.signature(pipeline_class.__init__).parameters

    def load_text_encoder(uri: _uris.TextEncoderUri):
        return uri.load(
            variant_fallback=variant,
            dtype_fallback=dtype,
            original_config=original_config,
            use_auth_token=auth_token,
            local_files_only=local_files_only,
            no_cache=bool(lora_uris) or model_cpu_offload or sequential_cpu_offload,
            missing_ok=missing_submodules_ok,
            device_map=get_device_map_for_quantizer(uri.quantizer)
        )

    def load_vae(uri: _uris.VAEUri):
        vae_model = uri.load(
            dtype_fallback=dtype,
            original_config=original_config,
            use_auth_token=auth_token,
            local_files_only=local_files_only,
            no_cache=model_cpu_offload or sequential_cpu_offload,
            missing_ok=missing_submodules_ok
        )
        if sdnq_cast_hack:
            og_decode = vae_model.decode
            def sdnq_decode(latents, *args, **kwargs):
                if getattr(vae_model.config, 'use_post_quant_conv', False):
                    cur_dtype = vae_model.post_quant_conv.weight.dtype
                else:
                    cur_dtype = _enums.get_torch_dtype(dtype)
                return og_decode(latents.to(
                    dtype=vae_model.dtype if cur_dtype is None else cur_dtype), *args, **kwargs)
            vae_model.decode = sdnq_decode
        return vae_model

    def sdnq_forward(og_forward, model, *args, **kwargs):
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                args[i] = arg.to(dtype=model.dtype)
        for k,v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.to(dtype=model.dtype)
        return og_forward(*args, **kwargs)

    def controlnet_quant_forward(og_forward, model, *args, **kwargs):
        """
        Forward function for quantized controlnets that casts inputs to the model's dtype.
        This is needed because diffusers doesn't handle controlnet quantization state internally.
        
        Note: Preserves specific parameters that are used as indices in embedding layers.
        Specifically, 'controlnet_mode' parameter in FLUX ControlNets is used as indices
        in the controlnet_mode_embedder embedding layer and must remain as integer tensors.
        """
        # Known parameters that should remain as integer types for embedding layer indices
        # controlnet_mode: Used in FLUX ControlNet Union models with nn.Embedding layer
        # Note: SDXL ControlNet Union uses control_type with Timesteps projection (no integer requirement)
        embedding_index_params = {'controlnet_mode'}
        
        # Cast positional arguments to model dtype (no special handling needed)
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                args[i] = arg.to(dtype=model.dtype)
        
        # Handle keyword arguments with special cases for embedding indices
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                # Preserve specific embedding index parameters or any integer tensor
                if (k in embedding_index_params or 
                    v.dtype in (torch.int64, torch.long, torch.int32, torch.int)):
                    continue  # Don't cast embedding index tensors
                kwargs[k] = v.to(dtype=model.dtype)
        return og_forward(*args, **kwargs)

    def load_unet(uri: _uris.UNetUri, unet_class):
        unet_model = uri.load(
            variant_fallback=variant,
            dtype_fallback=dtype,
            original_config=original_config,
            use_auth_token=auth_token,
            local_files_only=local_files_only,
            no_cache=bool(lora_uris) or
                     bool(ip_adapter_uris) or
                     model_cpu_offload or
                     sequential_cpu_offload,
            device_map=get_device_map_for_quantizer(uri.quantizer),
            unet_class=unet_class
        )

        if sdnq_cast_hack:
            unet_model.forward = functools.partial(
                sdnq_forward,
                unet_model.forward,
                unet_model
            )
        return unet_model

    def load_transformer(uri: _uris.TransformerUri, transformer_class):
        transformer_model = uri.load(
            variant_fallback=variant,
            dtype_fallback=dtype,
            original_config=original_config,
            use_auth_token=auth_token,
            local_files_only=local_files_only,
            no_cache=bool(lora_uris) or model_cpu_offload or sequential_cpu_offload,
            device_map=get_device_map_for_quantizer(uri.quantizer),
            transformer_class=transformer_class
        )
        if sdnq_cast_hack:
            transformer_model.forward = functools.partial(
                sdnq_forward,
                transformer_model.forward,
                transformer_model
            )
        return transformer_model

    def load_default_text_encoder(encoder, encoder_name):
        should_quantize = should_apply_quantizer(encoder_name)

        # Check if we have a cached version from granular caching
        if encoder_name in cached_component_paths:
            # Use manual quantizer if available, otherwise use global quantizer
            text_encoder_quantizer = manual_quantizers.get(encoder_name, quantizer_uri if should_quantize else None)

            return load_text_encoder(
                _uris.TextEncoderUri(
                    encoder=encoder,
                    model=cached_component_paths[encoder_name],
                    variant=variant,
                    dtype=dtype,
                    quantizer=text_encoder_quantizer
                )
            )
        else:
            if _hfhub.is_single_file_model_load(model_path) and should_quantize:
                raise UnsupportedPipelineConfigError(
                    'Cannot use global --quantizer URI when attempting to '
                    f'load default text encoder for "{encoder_name}" from a single file checkpoint. '
                    f'You must specify this text encoder manually with --text-encoders, it is likely missing '
                    f'from the model checkpoint.'
                )
            return load_text_encoder(
                _uris.TextEncoderUri(
                    encoder=encoder,
                    model=model_path,
                    variant=variant,
                    revision=revision,
                    subfolder=encoder_subfolder,
                    dtype=dtype,
                    quantizer=quantizer_uri if should_quantize else None
                )
            )

    text_encoder_override_states = [
        text_encoder_override,
        text_encoder_2_override,
        text_encoder_3_override
    ]

    # Load Text Encoders

    for idx, (name, param) in enumerate(
            [n for n in sorted(model_index.items(), key=lambda x: x[0])
             if n[0].startswith('text_encoder') and n[1][0] is not None]):
        if text_encoder_override_states[idx]:
            continue

        if _hfhub.is_single_file_model_load(model_path):
            encoder_subfolder = name
        else:
            encoder_subfolder = os.path.join(subfolder, name) if subfolder else name

        encoder_class = param[1]

        if text_encoder_uris and len(text_encoder_uris) > idx:
            encoder_uri = text_encoder_uris[idx]

            if _text_encoder_default(encoder_uri):
                creation_kwargs[name] = load_default_text_encoder(encoder_class, name)
            elif not (needs_granular_caching and name in components_to_cache):
                creation_kwargs[name] = load_text_encoder(
                    _uris.TextEncoderUri.parse(encoder_uri)
                )
            # If granular caching is needed and this encoder is in components_to_cache,
            # skip direct loading - it will be handled by granular caching
        else:
            creation_kwargs[name] = load_default_text_encoder(encoder_class, name)

    # Load VAE

    if not vae_override:
        if vae_uri:
            parsed_vae_uri = _uris.VAEUri.parse(vae_uri)

            creation_kwargs['vae'] = load_vae(parsed_vae_uri)

            _messages.debug_log(lambda:
                                f'Added Torch VAE: "{vae_uri}" to pipeline: "{pipeline_class.__name__}"')
        elif 'vae' in pipe_params:
            if _hfhub.is_single_file_model_load(model_path):
                vae_subfolder = 'vae'
            else:
                vae_subfolder = os.path.join(subfolder, 'vae') if subfolder else 'vae'

            vae_param = pipe_params['vae'].annotation

            vae_encoder_name = vae_param.__name__

            if _types.is_union(vae_param):
                try:
                    vae_encoder_name = model_index['vae'][1]
                except (KeyError, IndexError):
                    _messages.debug_log(
                        'Skipping auto VAE caching due to model '
                        'configuration not having a VAE key.')
                    vae_encoder_name = None
                except FileNotFoundError as e:
                    raise UnsupportedPipelineConfigError(
                        'Could not find VAE configuration data.') from e

                if vae_encoder_name not in _uris.VAEUri.supported_encoder_names():
                    raise UnsupportedPipelineConfigError(
                        f'Unsupported VAE encoder type: {vae_encoder_name}'
                    )

            if vae_encoder_name is not None:
                vae_extract_from_checkpoint = _hfhub.is_single_file_model_load(model_path)
                try:
                    creation_kwargs['vae'] = \
                        load_vae(_uris.VAEUri(
                            encoder=vae_encoder_name,
                            model=model_path,
                            variant=variant,
                            revision=revision,
                            subfolder=vae_subfolder,
                            extract=vae_extract_from_checkpoint,
                            dtype=dtype
                        ))
                except _d_exceptions.ModelNotFoundError:
                    if vae_extract_from_checkpoint:
                        raise
                    creation_kwargs['vae'] = \
                        load_vae(_uris.VAEUri(
                            encoder=vae_encoder_name,
                            model=model_path,
                            revision=revision,
                            subfolder=vae_subfolder,
                            dtype=dtype
                        ))

    # Load UNet
    if not unet_override:
        unet_parameter = 'unet'
        if model_type == _enums.ModelType.S_CASCADE:
            unet_parameter = 'prior'
        elif model_type == _enums.ModelType.S_CASCADE_DECODER:
            unet_parameter = 'decoder'

        unet_class = diffusers.UNet2DConditionModel if unet_parameter == 'unet' \
            else diffusers.models.unets.StableCascadeUNet

        if unet_uri is not None and not (needs_granular_caching and 'unet' in components_to_cache):
            parsed_unet_uri = _uris.UNetUri.parse(unet_uri)

            if _enums.model_type_is_kolors(model_type) and parsed_unet_uri.quantizer:
                raise UnsupportedPipelineConfigError(
                    f'--model-type {_enums.get_model_type_string(model_type)} does not support '
                    f'loading a --unet with quantization applied.'
                )

            creation_kwargs[unet_parameter] = load_unet(
                parsed_unet_uri, unet_class=unet_class
            )

            _messages.debug_log(lambda:
                                f'Added Torch UNet: "{unet_uri}" to pipeline: "{pipeline_class.__name__}"')
        elif 'unet' in pipe_params:

            if _hfhub.is_single_file_model_load(model_path):
                unet_subfolder = unet_parameter
            else:
                unet_subfolder = os.path.join(subfolder, unet_parameter) if subfolder else unet_parameter

            # Check if we have a cached version from granular caching
            if 'unet' in cached_component_paths:
                unet_model_path = cached_component_paths['unet']
                unet_subfolder = None  # No subfolder for cached component
                unet_revision = None  # No revision for cached component

                # Use manual quantizer if available, otherwise use global quantizer
                unet_quantizer = manual_quantizers.get('unet',
                                                       quantizer_uri if should_apply_quantizer("unet") else None)
            else:
                unet_model_path = model_path
                unet_revision = revision
                unet_quantizer = quantizer_uri if should_apply_quantizer("unet") else None

            creation_kwargs['unet'] = \
                load_unet(
                    _uris.UNetUri(
                        model=unet_model_path,
                        variant=variant,
                        revision=unet_revision,
                        subfolder=unet_subfolder,
                        dtype=dtype,
                        quantizer=unet_quantizer
                    ), unet_class=unet_class)

    # Load Transformer

    if _enums.model_type_is_sd3(model_type):
        transformer_class = diffusers.SD3Transformer2DModel
    elif _enums.model_type_is_flux(model_type):
        transformer_class = diffusers.FluxTransformer2DModel
    else:
        transformer_class = None

    if not transformer_override:
        if transformer_uri is not None and not (
                needs_granular_caching and transformer_class is not None and 'transformer' in components_to_cache):
            assert transformer_class is not None

            parsed_transformer_uri = _uris.TransformerUri.parse(transformer_uri)

            creation_kwargs['transformer'] = load_transformer(
                parsed_transformer_uri,
                transformer_class=transformer_class
            )

            _messages.debug_log(lambda:
                                f'Added Torch Transformer: "{transformer_uri}" to '
                                f'pipeline: "{pipeline_class.__name__}"')
        elif 'transformer' in pipe_params:
            assert transformer_class is not None

            if _hfhub.is_single_file_model_load(model_path):
                transformer_subfolder = 'transformer'
            else:
                transformer_subfolder = os.path.join(subfolder, 'transformer') if subfolder else 'transformer'

            # Check if we have a cached version from granular caching
            if 'transformer' in cached_component_paths:
                transformer_model_path = cached_component_paths['transformer']
                transformer_subfolder = None  # No subfolder for cached component
                transformer_revision = None  # No revision for cached component

                # Use manual quantizer if available, otherwise use global quantizer
                transformer_quantizer = manual_quantizers.get('transformer', quantizer_uri if should_apply_quantizer(
                    "transformer") else None)
            else:
                transformer_model_path = model_path
                transformer_revision = revision
                transformer_quantizer = quantizer_uri if should_apply_quantizer("transformer") else None

            creation_kwargs['transformer'] = load_transformer(
                _uris.TransformerUri(
                    model=transformer_model_path,
                    variant=variant,
                    revision=transformer_revision,
                    subfolder=transformer_subfolder,
                    dtype=dtype,
                    quantizer=transformer_quantizer
                ), transformer_class=transformer_class)

    # load image encoder

    if image_encoder_uri is not None and not image_encoder_override:
        parsed_image_encoder_uri = _uris.ImageEncoderUri.parse(image_encoder_uri)

        if _enums.model_type_is_sd3(model_type):
            # image encoder does not participate in offloading for SD3
            no_cache_image_encoder = model_cpu_offload
        else:
            no_cache_image_encoder = model_cpu_offload or sequential_cpu_offload

        loaded_image_encoder = parsed_image_encoder_uri.load(
            dtype_fallback=dtype,
            use_auth_token=auth_token,
            local_files_only=local_files_only,
            no_cache=no_cache_image_encoder,
            image_encoder_class=
            _models.SiglipImageEncoder
            if _enums.model_type_is_sd3(model_type) else
            transformers.CLIPVisionModelWithProjection
        )

        if isinstance(loaded_image_encoder, _models.SiglipImageEncoder):
            creation_kwargs['image_encoder'] = loaded_image_encoder.image_encoder
            creation_kwargs['feature_extractor'] = loaded_image_encoder.feature_extractor
        else:
            creation_kwargs['image_encoder'] = loaded_image_encoder

        _messages.debug_log(lambda:
                            f'Added Torch Image Encoder: "{image_encoder_uri}" to '
                            f'pipeline: "{pipeline_class.__name__}"')

    # Load T2I Adapters

    if t2i_adapter_uris and not adapter_override:
        t2i_adapters = None

        for t2i_adapter_uri in t2i_adapter_uris:
            parsed_t2i_adapter_uri = _uris.T2IAdapterUri.parse(t2i_adapter_uri)
            parsed_t2i_adapter_uris.append(parsed_t2i_adapter_uri)

            new_adapter = parsed_t2i_adapter_uri.load(
                use_auth_token=auth_token,
                dtype_fallback=dtype,
                local_files_only=local_files_only,
                no_cache=model_cpu_offload or sequential_cpu_offload
            )

            _messages.debug_log(lambda:
                                f'Added Torch T2IAdapter: "{t2i_adapter_uri}" '
                                f'to pipeline: "{pipeline_class.__name__}"')

            if t2i_adapters is not None:
                if not isinstance(t2i_adapters, list):
                    t2i_adapters = [t2i_adapters, new_adapter]
                else:
                    t2i_adapters.append(new_adapter)
            else:
                t2i_adapters = new_adapter

        if isinstance(t2i_adapters, list):
            creation_kwargs['adapter'] = diffusers.MultiAdapter(t2i_adapters)
        else:
            creation_kwargs['adapter'] = t2i_adapters

    # Load ControlNets

    if controlnet_uris and not controlnet_override:
        controlnets = None
        sdxl_cn_union = None

        for controlnet_uri in controlnet_uris:
            parsed_controlnet_uri = _uris.ControlNetUri.parse(
                uri=controlnet_uri,
                model_type=model_type
            )

            parsed_controlnet_uris.append(parsed_controlnet_uri)

            # Apply global quantizer if controlnet doesn't have
            # its own quantizer and should be quantized
            controlnet_uri_to_load = parsed_controlnet_uri
            if not parsed_controlnet_uri.quantizer and should_apply_quantizer('controlnet'):
                # Create a new URI with the global quantizer
                controlnet_uri_to_load = _uris.ControlNetUri(
                    model=parsed_controlnet_uri.model,
                    revision=parsed_controlnet_uri.revision,
                    variant=parsed_controlnet_uri.variant,
                    subfolder=parsed_controlnet_uri.subfolder,
                    dtype=parsed_controlnet_uri.dtype,
                    scale=parsed_controlnet_uri.scale,
                    start=parsed_controlnet_uri.start,
                    end=parsed_controlnet_uri.end,
                    mode=parsed_controlnet_uri.mode,
                    quantizer=quantizer_uri,
                    model_type=parsed_controlnet_uri.model_type
                )

            new_net = controlnet_uri_to_load.load(
                use_auth_token=auth_token,
                dtype_fallback=dtype,
                local_files_only=local_files_only,
                no_cache=model_cpu_offload or sequential_cpu_offload,
                device_map=get_device_map_for_quantizer(controlnet_uri_to_load.quantizer)
            )

            # Apply casting hack for quantized controlnets
            if controlnet_uri_to_load.quantizer:
                new_net.forward = functools.partial(
                    controlnet_quant_forward,
                    new_net.forward,
                    new_net
                )

            _messages.debug_log(lambda:
                                f'Added Torch ControlNet: "{controlnet_uri}" '
                                f'to pipeline: "{pipeline_class.__name__}"')

            if sdxl_cn_union is not None:
                continue
            if isinstance(new_net, diffusers.ControlNetUnionModel):
                # first model determines controlnet model,
                # the rest of the specifications just provide the mode
                sdxl_cn_union = new_net
                continue

            if controlnets is not None:
                if not isinstance(controlnets, list):
                    controlnets = [controlnets, new_net]
                else:
                    controlnets.append(new_net)
            else:
                controlnets = new_net

        if sdxl_cn_union is not None:
            controlnets = sdxl_cn_union

        if isinstance(controlnets, list):
            # not handled internally for whatever reason like the other pipelines
            if _enums.model_type_is_sd3(model_type):
                creation_kwargs['controlnet'] = diffusers.SD3MultiControlNetModel(controlnets)
            elif _enums.model_type_is_flux(model_type):
                creation_kwargs['controlnet'] = diffusers.FluxMultiControlNetModel(controlnets)
            else:
                creation_kwargs['controlnet'] = controlnets
        else:
            creation_kwargs['controlnet'] = controlnets

    if _enums.model_type_is_floyd(model_type):
        creation_kwargs['watermarker'] = None

    if not safety_checker and \
            (_enums.model_type_is_sd15(model_type) or
             _enums.model_type_is_floyd(model_type)) and not safety_checker_override:
        creation_kwargs['safety_checker'] = None

    creation_kwargs.update(extra_modules)

    def _handle_generic_pipeline_load_failure(e):
        exc_msg = str(e)

        _messages.debug_log(
            f'Failed to load primary pipeline model: "{model_path}", reason: {exc_msg}')

        if model_path in exc_msg:
            if 'restricted' in exc_msg:
                # the gated repo message is far more useful to the user
                raise InvalidModelFileError(exc_msg) from e
            else:
                raise InvalidModelFileError(f'invalid model file or repo slug: {model_path}') from e

        raise InvalidModelFileError(e) from e

    if _hfhub.is_single_file_model_load(model_path):
        if subfolder is not None:
            raise UnsupportedPipelineConfigError(
                'Single file model loads do not support the subfolder option.')
        try:
            _enforce_pipeline_cache_size(estimated_memory_usage)

            pipeline = _pipeline_creation_args_debug(
                backend='Torch',
                cls=pipeline_class,
                method=pipeline_class.from_single_file,
                original_config=original_config,
                model=model_path,
                token=auth_token,
                revision=revision,
                variant=variant,
                torch_dtype=torch_dtype,
                use_safe_tensors=model_path.endswith('.safetensors'),
                local_files_only=local_files_only,
                **creation_kwargs)

        except diffusers.loaders.single_file.SingleFileComponentError as e:
            _handle_single_file_component_error(e)
        except (ValueError, TypeError, NameError, OSError) as e:
            _handle_generic_pipeline_load_failure(e)
    else:
        try:
            pipeline = _pipeline_creation_args_debug(
                backend='Torch',
                cls=pipeline_class,
                method=pipeline_class.from_pretrained,
                model=model_path,
                token=auth_token,
                revision=revision,
                variant=variant,
                torch_dtype=torch_dtype,
                subfolder=subfolder,
                local_files_only=local_files_only,
                **creation_kwargs)
        except (ValueError, TypeError, NameError, OSError) as e:
            _handle_generic_pipeline_load_failure(e)

    if hasattr(pipeline, 'vae') and \
            _enums.model_type_is_sd3(model_type):
        # patch to enable tiling at all resolutions
        if pipeline.vae.quant_conv is None:
            pipeline.vae.quant_conv = lambda x: x
        if pipeline.vae.post_quant_conv is None:
            pipeline.vae.post_quant_conv = lambda x: x

    # Textual Inversions, LoRAs, IP Adapters

    parsed_textual_inversion_uris = []
    parsed_lora_uris = []
    parsed_ip_adapter_uris = []

    if textual_inversion_uris:
        for inversion_uri in textual_inversion_uris:
            parsed = _uris.TextualInversionUri.parse(inversion_uri)
            parsed_textual_inversion_uris.append(parsed)

        _uris.TextualInversionUri.load_on_pipeline(
            pipeline=pipeline,
            uris=parsed_textual_inversion_uris,
            use_auth_token=auth_token,
            local_files_only=local_files_only)

    if lora_uris and not quantizer_uri:
        # LoRAs are fused into the model and cached
        # to disk in the case that quantization is requested

        for lora_uri in lora_uris:
            parsed = _uris.LoRAUri.parse(lora_uri)
            parsed_lora_uris.append(parsed)

        _uris.LoRAUri.load_on_pipeline(
            pipeline=pipeline,
            uris=parsed_lora_uris,
            fuse_scale=lora_fuse_scale if lora_fuse_scale is not None else 1.0,
            use_auth_token=auth_token,
            local_files_only=local_files_only)

    if ip_adapter_uris:
        for ip_adapter_uri in ip_adapter_uris:
            parsed = _uris.IPAdapterUri.parse(ip_adapter_uri)
            parsed_ip_adapter_uris.append(parsed)

        _uris.IPAdapterUri.load_on_pipeline(
            pipeline=pipeline,
            uris=parsed_ip_adapter_uris,
            use_auth_token=auth_token,
            local_files_only=local_files_only)

    if ip_adapter_uris and (not hasattr(pipeline, 'image_encoder') or pipeline.image_encoder is None):
        raise UnsupportedPipelineConfigError(
            'Using --ip-adapters but missing required --image-encoder specification, '
            'your --ip-adapters specification did not include an image encoder model and '
            'you must specify one manually.')

    # Safety Checker

    if not safety_checker_override:
        if _enums.model_type_is_floyd(model_type):
            _set_floyd_safety_checker(pipeline, safety_checker)
        else:
            _set_sd_safety_checker(pipeline, safety_checker)

    # Model Offloading

    # SD3 image_encoder needs to be excluded to avoid meta tensor errors.

    if _enums.model_type_is_sd3(model_type) and sequential_cpu_offload and 'image_encoder' in creation_kwargs:
        pipeline._exclude_from_cpu_offload.append("image_encoder")

    if not device.startswith('cpu'):
        if sequential_cpu_offload:
            enable_sequential_cpu_offload(pipeline, device)
        elif model_cpu_offload:
            enable_model_cpu_offload(pipeline, device)

    _messages.debug_log(f'Finished Creating Torch Pipeline: "{pipeline_class.__name__}"')

    # modules quantized in 8 bit by bitsandbytes cannot be moved off the GPU
    # there is a sequence in call_pipeline that garbage collects them out of the
    # cache before calling other pipelines, there is also a dgenerate.devicecache
    # hook to clear them out when requested
    for module in get_pipeline_modules(pipeline).values():
        if _util.is_loaded_in_8bit_bnb(module):
            _disable_to(module)

    # noinspection PyTypeChecker
    return PipelineCreationResult(
        model_path=model_path,
        pipeline=pipeline,
        parsed_unet_uri=parsed_unet_uri,
        parsed_transformer_uri=parsed_transformer_uri,
        parsed_vae_uri=parsed_vae_uri,
        parsed_lora_uris=parsed_lora_uris,
        parsed_image_encoder_uri=parsed_image_encoder_uri,
        parsed_ip_adapter_uris=parsed_ip_adapter_uris,
        parsed_textual_inversion_uris=parsed_textual_inversion_uris,
        parsed_controlnet_uris=parsed_controlnet_uris,
        parsed_t2i_adapter_uris=parsed_t2i_adapter_uris
    ), _d_memoize.CachedObjectMetadata(size=estimated_memory_usage)


def _handle_single_file_component_error(e):
    msg = str(e)
    if 'text_encoder' in msg:
        raise UnsupportedPipelineConfigError(
            f'Single file load error, missing --text-encoders / --second-model-text-encoders:\n{e}') from e
    elif 'vae =' in msg:
        raise UnsupportedPipelineConfigError(
            f'Single file load error, missing --vae:\n{e}') from e
    else:
        raise UnsupportedPipelineConfigError(
            f'Single file load error, missing component:\n{e}') from e


def get_converted_checkpoint_cache_dir():
    """
    Get cache directory where dgenerate stores any checkpoints that needed to be converted into
    diffusers directory format on disk to function, this process is used to
    support quantization on single file checkpoints.

    Or the value of the environmental variable ``DGENERATE_CACHE`` joined with ``diffusers_converted``.

    :return: string (directory path)
    """
    user_cache_path = os.environ.get('DGENERATE_CACHE')

    if user_cache_path is not None:
        path = os.path.join(user_cache_path, 'web')
    else:
        path = os.path.expanduser(os.path.join('~', '.cache', 'dgenerate', 'diffusers_converted'))

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    return path


_to_diffusers_cache_dir = get_converted_checkpoint_cache_dir()


# Component-specific caches for granular caching
def _get_component_cache_store(component_name: str):
    component_dir = os.path.join(_to_diffusers_cache_dir, component_name)
    pathlib.Path(component_dir).mkdir(parents=True, exist_ok=True)
    return _filecache.KeyValueStore(os.path.join(component_dir, 'cache.db'))


_component_cache_stores = {
    'text_encoder': _get_component_cache_store('text_encoder'),
    'text_encoder_2': _get_component_cache_store('text_encoder_2'),
    'text_encoder_3': _get_component_cache_store('text_encoder_3'),
    'transformer': _get_component_cache_store('transformer'),
    'unet': _get_component_cache_store('unet'),
}


def _get_component_cache_key(
        component_name: str,
        model_path: str,
        model_type: _enums.ModelType,
        lora_uris: _types.OptionalUris,
        lora_fuse_scale: float | None,
        revision: str | None,
        variant: str | None,
        subfolder: str | None,
        dtype: _enums.DataType,
        original_config: str | None,
        manual_component_uris: dict[str, str] | None = None
) -> str:
    manual_uri_part = ""
    if manual_component_uris and component_name in manual_component_uris:
        # Strip quantizer from manual URI for cache key since components are saved in original precision
        try:
            if component_name == 'unet':
                parsed_uri = _uris.UNetUri.parse(manual_component_uris[component_name])
                uri_without_quantizer = _uris.UNetUri(
                    model=parsed_uri.model,
                    revision=parsed_uri.revision,
                    variant=parsed_uri.variant,
                    subfolder=parsed_uri.subfolder,
                    dtype=parsed_uri.dtype,
                    quantizer=None
                )
                manual_uri_part = f"|manual:{str(uri_without_quantizer)}"
            elif component_name == 'transformer':
                parsed_uri = _uris.TransformerUri.parse(manual_component_uris[component_name])
                uri_without_quantizer = _uris.TransformerUri(
                    model=parsed_uri.model,
                    revision=parsed_uri.revision,
                    variant=parsed_uri.variant,
                    subfolder=parsed_uri.subfolder,
                    dtype=parsed_uri.dtype,
                    quantizer=None
                )
                manual_uri_part = f"|manual:{str(uri_without_quantizer)}"
            elif component_name.startswith('text_encoder'):
                parsed_uri = _uris.TextEncoderUri.parse(manual_component_uris[component_name])
                uri_without_quantizer = _uris.TextEncoderUri(
                    encoder=parsed_uri.encoder,
                    model=parsed_uri.model,
                    revision=parsed_uri.revision,
                    variant=parsed_uri.variant,
                    subfolder=parsed_uri.subfolder,
                    dtype=parsed_uri.dtype,
                    quantizer=None
                )
                manual_uri_part = f"|manual:{str(uri_without_quantizer)}"
            else:
                # Fallback for unknown component types
                manual_uri_part = f"|manual:{manual_component_uris[component_name]}"
        except Exception:
            # If parsing fails, use the original URI
            manual_uri_part = f"|manual:{manual_component_uris[component_name]}"

    # LoRA order consideration: LoRA fusion order can affect the final result since
    # LoRA operations are not necessarily commutative. We preserve the original order
    # to ensure mathematical correctness and respect user intent.
    # 
    # If you want to optimize cache efficiency by normalizing LoRA order (at the cost
    # of potentially different results), you could sort the URIs like this:
    # lora_uris_for_cache = sorted(lora_uris) if lora_uris else lora_uris
    lora_uris_for_cache = lora_uris

    # Note: quantizer_uri is NOT included in the cache key because components are saved
    # in original precision. The quantizer is only applied when loading from cache.
    return f"{component_name}|{model_path}|{str(lora_uris_for_cache)}|{str(lora_fuse_scale)}|" \
           f"{str(model_type)}|{str(revision)}|{str(variant)}|{str(subfolder)}|" \
           f"{str(dtype)}|{str(original_config)}{manual_uri_part}"


def _get_affected_components_for_lora(pipeline_class) -> list[str]:
    if hasattr(pipeline_class, '_lora_loadable_modules'):
        return list(pipeline_class._lora_loadable_modules)
    return []


def _get_quantizable_components(model_type: _enums.ModelType) -> list[str]:
    components = []

    # Add UNet or Transformer based on model type
    if _enums.model_type_is_sd3(model_type) or _enums.model_type_is_flux(model_type):
        # For SD3 and Flux, single file checkpoints typically only contain the transformer
        # Text encoders are usually separate standard models that don't need conversion
        components.append('transformer')
    else:
        # For older models (SD1.5, SD2, SDXL), single files often contain UNet and text encoders
        components.append('unet')
        if _enums.model_type_is_sdxl(model_type):
            components.extend(['text_encoder', 'text_encoder_2'])
        else:
            components.append('text_encoder')

    return components


def _create_minimal_pipeline_for_component_extraction(
        model_path: str,
        model_type: _enums.ModelType,
        pipeline_type: _enums.PipelineType,
        components_needed: list[str],
        revision: str | None,
        variant: str | None,
        subfolder: str | None,
        dtype: _enums.DataType,
        original_config: str | None,
        auth_token: str | None,
        lora_uris: _types.OptionalUris,
        lora_fuse_scale: float | None,
        manual_component_uris: dict[str, str] | None = None,
        vae_uri: _types.OptionalUri = None
):
    pipeline_class = get_pipeline_class(
        model_type=model_type,
        pipeline_type=pipeline_type,
        help_mode=True  # Allow creation even if pipeline_type doesn't match
    )

    torch_dtype = _enums.get_torch_dtype(dtype)
    creation_kwargs = {}
    manual_quantizers = {}  # Store quantizer info from manual URIs

    # Get all text encoder parameters that the pipeline expects
    pipe_params = inspect.signature(pipeline_class.__init__).parameters
    text_encoder_params = [name for name in pipe_params.keys() if name.startswith('text_encoder')]

    # Load ALL text encoders that the pipeline expects, even if not being cached
    # This prevents SingleFileComponentError during pipeline creation
    for encoder_param in text_encoder_params:
        # Check if we have a manual URI for this encoder
        if manual_component_uris and encoder_param in manual_component_uris:
            parsed_uri = _uris.TextEncoderUri.parse(manual_component_uris[encoder_param])

            # Save quantizer info and reconstruct URI without quantizer
            if parsed_uri.quantizer:
                manual_quantizers[encoder_param] = parsed_uri.quantizer

            # Reconstruct URI without quantizer
            uri_without_quantizer = _uris.TextEncoderUri(
                encoder=parsed_uri.encoder,
                model=parsed_uri.model,
                revision=parsed_uri.revision,
                variant=parsed_uri.variant,
                subfolder=parsed_uri.subfolder,
                dtype=parsed_uri.dtype,
                quantizer=None  # Explicitly set to None
            )

            creation_kwargs[encoder_param] = uri_without_quantizer.load(
                variant_fallback=variant,
                dtype_fallback=dtype,
                original_config=original_config,
                use_auth_token=auth_token,
                local_files_only=False,
                no_cache=True,
                missing_ok=False
            )
        else:
            # Load default text encoder for this parameter
            # We need to load it even if we're not caching it to prevent errors
            try:
                if _hfhub.is_single_file_model_load(model_path):
                    encoder_subfolder = encoder_param
                else:
                    encoder_subfolder = os.path.join(subfolder, encoder_param) if subfolder else encoder_param

                # Try to determine the encoder class - this is a simplified approach
                # In practice, we might need to look at model_index or infer from model_type
                if encoder_param == 'text_encoder':
                    encoder_name = 'CLIPTextModel'  # Default for most SD models
                elif encoder_param == 'text_encoder_2':
                    if _enums.model_type_is_sdxl(model_type):
                        encoder_name = 'CLIPTextModelWithProjection'
                    elif _enums.model_type_is_sd3(model_type):
                        encoder_name = 'CLIPTextModelWithProjection'
                    elif _enums.model_type_is_flux(model_type):
                        encoder_name = 'CLIPTextModel'
                    else:
                        encoder_name = 'CLIPTextModel'
                elif encoder_param == 'text_encoder_3':
                    if _enums.model_type_is_sd3(model_type) or _enums.model_type_is_flux(model_type):
                        encoder_name = 'T5EncoderModel'
                    else:
                        encoder_name = 'CLIPTextModel'
                else:
                    encoder_name = 'CLIPTextModel'

                default_encoder_uri = _uris.TextEncoderUri(
                    encoder=encoder_name,
                    model=model_path,
                    variant=variant,
                    revision=revision,
                    subfolder=encoder_subfolder,
                    dtype=dtype,
                    quantizer=None
                )

                creation_kwargs[encoder_param] = default_encoder_uri.load(
                    variant_fallback=variant,
                    dtype_fallback=dtype,
                    original_config=original_config,
                    use_auth_token=auth_token,
                    local_files_only=False,
                    no_cache=True,
                    missing_ok=True  # Allow missing for optional encoders
                )
            except Exception as e:
                _messages.debug_log(f"Failed to load default {encoder_param}: {e}")
                # Continue without this encoder - some may be optional

    # Load manual components that are not text encoders
    if manual_component_uris:
        for component in components_needed:
            if component in manual_component_uris and not component.startswith('text_encoder'):
                if component == 'unet':
                    unet_class = diffusers.UNet2DConditionModel
                    parsed_uri = _uris.UNetUri.parse(manual_component_uris[component])

                    # Save quantizer info and reconstruct URI without quantizer
                    if parsed_uri.quantizer:
                        manual_quantizers[component] = parsed_uri.quantizer

                    # Reconstruct URI without quantizer
                    uri_without_quantizer = _uris.UNetUri(
                        model=parsed_uri.model,
                        revision=parsed_uri.revision,
                        variant=parsed_uri.variant,
                        subfolder=parsed_uri.subfolder,
                        dtype=parsed_uri.dtype,
                        quantizer=None  # Explicitly set to None
                    )

                    creation_kwargs['unet'] = uri_without_quantizer.load(
                        variant_fallback=variant,
                        dtype_fallback=dtype,
                        original_config=original_config,
                        use_auth_token=auth_token,
                        local_files_only=False,
                        no_cache=True,
                        unet_class=unet_class
                    )
                elif component == 'transformer':
                    if _enums.model_type_is_sd3(model_type):
                        transformer_class = diffusers.SD3Transformer2DModel
                    elif _enums.model_type_is_flux(model_type):
                        transformer_class = diffusers.FluxTransformer2DModel
                    else:
                        continue  # Skip if no appropriate transformer class

                    parsed_uri = _uris.TransformerUri.parse(manual_component_uris[component])

                    # Save quantizer info and reconstruct URI without quantizer
                    if parsed_uri.quantizer:
                        manual_quantizers[component] = parsed_uri.quantizer

                    # Reconstruct URI without quantizer
                    uri_without_quantizer = _uris.TransformerUri(
                        model=parsed_uri.model,
                        revision=parsed_uri.revision,
                        variant=parsed_uri.variant,
                        subfolder=parsed_uri.subfolder,
                        dtype=parsed_uri.dtype,
                        quantizer=None  # Explicitly set to None
                    )

                    creation_kwargs['transformer'] = uri_without_quantizer.load(
                        variant_fallback=variant,
                        dtype_fallback=dtype,
                        original_config=original_config,
                        use_auth_token=auth_token,
                        local_files_only=False,
                        no_cache=True,
                        transformer_class=transformer_class
                    )

    # Load VAE if specified - this is needed for single file checkpoints that don't contain a VAE
    if vae_uri:
        parsed_vae_uri = _uris.VAEUri.parse(vae_uri)
        creation_kwargs['vae'] = parsed_vae_uri.load(
            dtype_fallback=dtype,
            original_config=original_config,
            use_auth_token=auth_token,
            local_files_only=False,
            no_cache=True,
            missing_ok=False
        )

    # Load the pipeline with all required components
    if _hfhub.is_single_file_model_load(model_path):
        try:
            pipeline = pipeline_class.from_single_file(
                model_path,
                original_config=original_config,
                token=auth_token,
                revision=revision,
                variant=variant,
                torch_dtype=torch_dtype,
                use_safe_tensors=model_path.endswith('.safetensors'),
                **creation_kwargs
            )
        except diffusers.loaders.single_file.SingleFileComponentError as e:
            _handle_single_file_component_error(e)
        except Exception as e:
            _messages.debug_log(f"Failed to create minimal pipeline from single file: {e}")
            raise
    else:
        try:
            pipeline = pipeline_class.from_pretrained(
                model_path,
                token=auth_token,
                revision=revision,
                variant=variant,
                torch_dtype=torch_dtype,
                subfolder=subfolder,
                **creation_kwargs
            )
        except Exception as e:
            _messages.debug_log(f"Failed to create minimal pipeline from pretrained: {e}")
            raise

    # Apply LoRAs if specified
    if lora_uris:
        parsed_lora_uris = [_uris.LoRAUri.parse(uri) for uri in lora_uris]

        _uris.LoRAUri.load_on_pipeline(
            pipeline=pipeline,
            uris=parsed_lora_uris,
            fuse_scale=lora_fuse_scale if lora_fuse_scale is not None else 1.0,
            use_auth_token=auth_token
        )

    return pipeline, manual_quantizers


def _cache_components_granular(
        model_path: str,
        model_type: _enums.ModelType,
        pipeline_type: _enums.PipelineType,
        components_to_cache: list[str],
        lora_uris: _types.OptionalUris,
        lora_fuse_scale: float | None,
        revision: str | None,
        variant: str | None,
        subfolder: str | None,
        dtype: _enums.DataType,
        original_config: str | None,
        auth_token: str | None,
        manual_component_uris: dict[str, str] | None = None,
        vae_uri: _types.OptionalUri = None
) -> tuple[dict[str, str], dict[str, str]]:
    cached_component_paths = {}
    components_to_load = []

    # Check which components are already cached
    for component in components_to_cache:
        cache_key = _get_component_cache_key(
            component, model_path, model_type, lora_uris, lora_fuse_scale,
            revision, variant, subfolder, dtype, original_config,
            manual_component_uris
        )

        cache_store = _component_cache_stores[component]

        with cache_store:
            cached_path = cache_store.get(cache_key)
            if cached_path and os.path.exists(cached_path):
                cached_component_paths[component] = cached_path
                _messages.debug_log(f"Using cached {component} from: {cached_path}")
            else:
                components_to_load.append(component)

    # If all components are cached, we still need to extract manual quantizer info
    manual_quantizers = {}
    if manual_component_uris:
        for component, uri in manual_component_uris.items():
            if component in components_to_cache:
                try:
                    if component == 'unet':
                        parsed_uri = _uris.UNetUri.parse(uri)
                        if parsed_uri.quantizer:
                            manual_quantizers[component] = parsed_uri.quantizer
                    elif component == 'transformer':
                        parsed_uri = _uris.TransformerUri.parse(uri)
                        if parsed_uri.quantizer:
                            manual_quantizers[component] = parsed_uri.quantizer
                    elif component.startswith('text_encoder'):
                        parsed_uri = _uris.TextEncoderUri.parse(uri)
                        if parsed_uri.quantizer:
                            manual_quantizers[component] = parsed_uri.quantizer
                except Exception as e:
                    _messages.debug_log(f"Error extracting quantizer from {component} URI: {e}")

    # If all components are cached, return early
    if not components_to_load:
        return cached_component_paths, manual_quantizers

    # Create pipeline and extract needed components
    if lora_uris:
        _messages.warning(
            f'Model "{model_path}" is having LoRAs '
            f'fused into specific components, that will then be cached on disk '
            f'prior to quantization. This is a one time task per LoRA scale value, '
            f'please be patient...'
        )
    else:
        _messages.warning(
            f'Model "{model_path}" components are being converted to '
            f'diffusers format and cached on disk prior to quantization. '
            f'This is a one time task, please be patient...'
        )

    with _d_memoize.disable_memoization_context():
        pipeline, manual_quantizers = _create_minimal_pipeline_for_component_extraction(
            model_path=model_path,
            model_type=model_type,
            pipeline_type=pipeline_type,
            components_needed=components_to_load,
            revision=revision,
            variant=variant,
            subfolder=subfolder,
            dtype=dtype,
            original_config=original_config,
            auth_token=auth_token,
            lora_uris=lora_uris,
            lora_fuse_scale=lora_fuse_scale,
            manual_component_uris=manual_component_uris,
            vae_uri=vae_uri
        )

    if lora_uris:
        pipeline.unload_lora_weights()

    # Cache each component individually
    for component in components_to_load:
        if not hasattr(pipeline, component):
            _messages.debug_log(
                f"Component {component} not found in pipeline, skipping cache")
            continue

        component_obj = getattr(pipeline, component)
        if component_obj is None:
            _messages.debug_log(
                f"Component {component} is None, skipping cache")
            continue

        # Get the cache store for the component
        cache_store = _component_cache_stores[component]

        # Generate cache key and path
        cache_key = _get_component_cache_key(
            component, model_path, model_type, lora_uris, lora_fuse_scale,
            revision, variant, subfolder, dtype, original_config,
            manual_component_uris
        )

        # Generate cache path with component/uuid structure
        component_dir = os.path.join(_to_diffusers_cache_dir, component)
        pathlib.Path(component_dir).mkdir(parents=True, exist_ok=True)

        with cache_store:
            cache_uuid = hashlib.sha256(cache_key.encode('utf-8')).hexdigest()
            cache_path = os.path.join(component_dir, cache_uuid)

            # Ensure unique path
            while os.path.exists(cache_path):
                cache_uuid = hashlib.sha256((cache_key + str(random.random())).encode('utf-8')).hexdigest()
                cache_path = os.path.join(component_dir, cache_uuid)

        # Save component
        _messages.debug_log(f"Saving cached {component} to: {cache_path}")
        component_obj.save_pretrained(cache_path, variant=variant)

        # Store in cache
        with cache_store:
            cache_store[cache_key] = cache_path

        cached_component_paths[component] = cache_path

    return cached_component_paths, manual_quantizers


__all__ = _types.module_all()
