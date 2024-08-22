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
import ast
import collections.abc
import gc
import inspect
import typing

import accelerate
import diffusers
import diffusers.loaders
import diffusers.loaders.single_file_utils
import huggingface_hub
import torch.nn
import torch.nn

import dgenerate.exceptions as _d_exceptions
import dgenerate.memoize as _d_memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.cache as _cache
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.hfutil as _hfutil
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.promptweighters as _promptweighters
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.memoize import memoize as _memoize
import dgenerate.pipelinewrapper.util as _util
import dgenerate.pipelinewrapper.quanto as _quanto


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


class InvalidSchedulerNameError(Exception):
    """
    Unknown scheduler name used
    """
    pass


class ArgumentHelpException(Exception):
    """
    Not an error, runtime argument help was requested by
    passing "help" or a special value to an argument of
    :py:meth:`.DiffusionPipelineWrapper.__init__` which
    supports a help query.
    """
    pass


class SchedulerHelpException(ArgumentHelpException):
    """
    Not an error, runtime scheduler help was requested by passing "help" to a scheduler name
    argument of :py:meth:`.DiffusionPipelineWrapper.__init__` such as ``scheduler`` or ``sdxl_refiner_scheduler``.
    Upon calling :py:meth:`.DiffusionPipelineWrapper.__call__` info was printed using :py:meth:`dgenerate.messages.log`,
    then this exception raised to get out of the call stack.
    """
    pass


class TextEncodersHelpException(ArgumentHelpException):
    """
    Not an error, runtime text encoder help was requested by passing "help" to a text encoder URI
    argument of :py:meth:`.DiffusionPipelineWrapper.__init__` such as ``text_encoder_uris`` or ``second_text_encoder_uris``.
    Upon calling :py:meth:`.DiffusionPipelineWrapper.__call__` info was printed using :py:meth:`dgenerate.messages.log`,
    then this exception raised to get out of the call stack.
    """
    pass


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


def _set_torch_safety_checker(pipeline: diffusers.DiffusionPipeline, safety_checker: bool):
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


def scheduler_is_help(name: str | None):
    """
    This scheduler name is simply a request for help?, IE: "help"?

    :param name: string to test
    :return: ``True`` or ``False``
    """
    if name is None:
        return False
    lname = name.strip().lower()

    return lname == 'help' or lname == 'helpargs'


def text_encoder_is_help(text_encoder_uris: _types.OptionalUris):
    """
    Text encoder uris specification is simply a request for help?, IE: "help"?

    :param text_encoder_uris: list of text encoder URIs to test
    :return: ``True`` or ``False``
    """
    if text_encoder_uris is None:
        return False
    return any(t == 'help' for t in text_encoder_uris)


def load_scheduler(pipeline: diffusers.DiffusionPipeline,
                   scheduler_name=None, model_path: str | None = None):
    """
    Load a specific compatible scheduler class name onto a huggingface diffusers pipeline object.

    :raises SchedulerHelpException: if "help" is passed as a scheduler name.

    :param pipeline: pipeline object
    :param scheduler_name: compatible scheduler class name, pass "help" to receive a print out to STDOUT
        and raise :py:exc:`.SchedulerHelpException`, this argument can accept a URI in typical dgenerate format,
        for overriding the schedulers constructor parameters.
    :param model_path: Optional model path to be used in the message to STDOUT produced by passing "help"
    :return:
    """

    if scheduler_name is None:
        return

    compatibles = list(pipeline.scheduler.compatibles)

    if isinstance(pipeline, (diffusers.loaders.StableDiffusionLoraLoaderMixin,
                             diffusers.loaders.StableDiffusionXLLoraLoaderMixin)):
        compatibles.append(diffusers.LCMScheduler)

    if isinstance(pipeline, diffusers.StableDiffusionLatentUpscalePipeline):
        # Seems to only work with this scheduler
        compatibles = [c for c in compatibles if c.__name__ == 'EulerDiscreteScheduler']

    if isinstance(pipeline, (diffusers.IFPipeline,
                             diffusers.IFInpaintingPipeline,
                             diffusers.IFImg2ImgPipeline,
                             diffusers.IFSuperResolutionPipeline,
                             diffusers.IFInpaintingSuperResolutionPipeline,
                             diffusers.IFImg2ImgSuperResolutionPipeline)):
        # same here
        compatibles = [c for c in compatibles if c.__name__ == 'DDPMScheduler']

    compatibles = sorted(compatibles, key=lambda c: c.__name__)

    help_name = scheduler_name.strip().lower()
    if help_name == 'help':
        help_string = f'Compatible schedulers for "{model_path}" are:' + '\n\n'
        help_string += '\n'.join((" " * 4) + _textprocessing.quote(i.__name__) for i in compatibles) + '\n'
        _messages.log(help_string)
        raise SchedulerHelpException(help_string)

    if help_name == 'helpargs':
        help_string = f'Compatible schedulers for "{model_path}" are:' + '\n\n'
        help_string += '\n\n'.join((" " * 4) + i.__name__ + (':\n' + ' ' * 8) + ('\n' + ' ' * 8).join(
            _textprocessing.dashup(k[0]) + ('=' + str(k[1]) if len(k) > 1 else '') for k in
            list(_types.get_accepted_args_with_defaults(i.__init__.__wrapped__))[1:]) for i in compatibles) + '\n'
        _messages.log(help_string)
        raise SchedulerHelpException(help_string)

    def _get_value(v):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v

    for i in compatibles:
        if i.__name__.startswith(scheduler_name.split(';')[0]):
            parser = _textprocessing.ConceptUriParser(
                'Scheduler',
                known_args=[_textprocessing.dashup(a) for a in inspect.getfullargspec(i.__init__.__wrapped__).args[1:]])

            try:
                result = parser.parse(scheduler_name)
            except _textprocessing.ConceptUriParseError as e:
                raise InvalidSchedulerNameError(e)

            pipeline.scheduler = i.from_config(
                pipeline.scheduler.config,
                **{_textprocessing.dashdown(k): _get_value(v) for k, v in result.args.items()})

            return

    raise InvalidSchedulerNameError(
        f'Scheduler named "{scheduler_name}" is not a valid compatible scheduler, '
        f'options are:\n\n{chr(10).join(sorted(" " * 4 + _textprocessing.quote(i.__name__.split(".")[-1]) for i in compatibles))}')


def estimate_pipeline_memory_use(
        model_path: str,
        model_type: _enums.ModelType,
        pipeline_type: _enums.PipelineType = _enums.PipelineType.TXT2IMG,
        revision: _types.Name = 'main',
        variant: _types.OptionalName = None,
        subfolder: _types.OptionalPath = None,
        unet_uri: _types.OptionalUri = None,
        transformer_uri: _types.OptionalUri = None,
        vae_uri: _types.OptionalUri = None,
        lora_uris: _types.OptionalUris = None,
        image_encoder_uri: _types.OptionalUri = None,
        ip_adapter_uris: _types.OptionalUris = None,
        textual_inversion_uris: _types.OptionalUris = None,
        text_encoder_uris: _types.OptionalUris = None,
        safety_checker: bool = False,
        auth_token: str | None = None,
        extra_args: dict[str, typing.Any] | None = None,
        local_files_only: bool = False):
    """
    Estimate the CPU side memory use of a pipeline.

    :param model_path: huggingface slug, blob link, path to folder on disk, path to model file.
    :param model_type: :py:class:`dgenerate.pipelinewrapper.ModelType`
    :param pipeline_type: :py:class:`dgenerate.pipelinewrapper.PipelineType`
    :param revision: huggingface repo revision if using a huggingface slug
    :param variant: model file variant desired, for example "fp16"
    :param subfolder: huggingface repo subfolder if using a huggingface slug
    :param unet_uri: optional user specified ``--unet`` URI that will be loaded on to the pipeline
    :param transformer_uri: optional user specified ``--transformer`` URI that will be loaded on to the pipeline,
        this is currently only supported for Stable Diffusion 3 models.
    :param vae_uri: optional user specified ``--vae`` URI that will be loaded on to the pipeline
    :param lora_uris: optional user specified ``--loras`` URIs that will be loaded on to the pipeline
    :param image_encoder_uri: optional user specified ``--image-encoder`` URI that will be loaded on to the pipeline
    :param ip_adapter_uris: optional user specified ``--ip-adapters`` URIs that will be loaded on to the pipeline
    :param textual_inversion_uris: optional user specified ``--textual-inversion`` URIs that will be loaded on to the pipeline
    :param text_encoder_uris: optional user specified ``--text-encoders`` URIs that will be loaded on to the pipeline
    :param safety_checker: consider the safety checker? dgenerate usually loads the safety checker and then retroactively
        disables it if needed, so it usually considers the size of the safety checker model.
    :param auth_token: optional huggingface auth token to access restricted repositories that your account has access to.
    :param extra_args: ``extra_args`` as to be passed to :py:func:`.create_torch_diffusion_pipeline`
    :param local_files_only: Only ever attempt to look in the local huggingface cache? if ``False`` the huggingface
        API will be contacted when necessary.
    :return: size estimate in bytes.
    """

    if extra_args is None:
        extra_args = dict()

    if text_encoder_uris is None:
        text_encoder_uris = []

    include_text_encoder = 'text_encoder' not in extra_args and (
            len(text_encoder_uris) == 0 or not text_encoder_uris[0])
    include_text_encoder_2 = 'text_encoder_2' not in extra_args and (
            len(text_encoder_uris) < 2 or not text_encoder_uris[1])
    include_text_encoder_3 = 'text_encoder_3' not in extra_args and (
            len(text_encoder_uris) < 3 or not text_encoder_uris[2])

    usage = _hfutil.estimate_model_memory_use(
        repo_id=_hfutil.download_non_hf_model(model_path),
        revision=revision,
        variant=variant,
        subfolder=subfolder,
        include_unet=not unet_uri or 'unet' not in extra_args,
        include_vae=not vae_uri or 'vae' not in extra_args,
        safety_checker=safety_checker and 'safety_checker' not in extra_args,
        include_text_encoder=include_text_encoder,
        include_text_encoder_2=include_text_encoder_2,
        include_text_encoder_3=include_text_encoder_3,
        use_auth_token=auth_token,
        local_files_only=local_files_only,
        sentencepiece=_enums.model_type_is_floyd(model_type)
    )

    if transformer_uri:
        parsed = _uris.TransformerUri.parse(transformer_uri)
        usage += _hfutil.estimate_model_memory_use(
            repo_id=_hfutil.download_non_hf_model(parsed.model),
            revision=parsed.revision,
            subfolder=parsed.subfolder,
            use_auth_token=auth_token,
            local_files_only=local_files_only
        )

    if image_encoder_uri:
        parsed = _uris.ImageEncoderUri.parse(image_encoder_uri)
        usage += _hfutil.estimate_model_memory_use(
            repo_id=_hfutil.download_non_hf_model(parsed.model),
            revision=parsed.revision,
            subfolder=parsed.subfolder,
            use_auth_token=auth_token,
            local_files_only=local_files_only
        )

    if lora_uris:
        for lora_uri in lora_uris:
            parsed = _uris.LoRAUri.parse(lora_uri)

            usage += _hfutil.estimate_model_memory_use(
                repo_id=_hfutil.download_non_hf_model(parsed.model),
                revision=parsed.revision,
                subfolder=parsed.subfolder,
                weight_name=parsed.weight_name,
                use_auth_token=auth_token,
                local_files_only=local_files_only
            )

    if ip_adapter_uris:
        for ip_adapter_uri in ip_adapter_uris:
            parsed = _uris.IPAdapterUri.parse(ip_adapter_uri)

            usage += _hfutil.estimate_model_memory_use(
                repo_id=_hfutil.download_non_hf_model(parsed.model),
                revision=parsed.revision,
                subfolder=parsed.subfolder,
                weight_name=parsed.weight_name,
                use_auth_token=auth_token,
                local_files_only=local_files_only
            )

    if textual_inversion_uris:
        for textual_inversion_uri in textual_inversion_uris:
            parsed = _uris.TextualInversionUri.parse(textual_inversion_uri)

            usage += _hfutil.estimate_model_memory_use(
                repo_id=_hfutil.download_non_hf_model(parsed.model),
                revision=parsed.revision,
                subfolder=parsed.subfolder,
                weight_name=parsed.weight_name,
                use_auth_token=auth_token,
                local_files_only=local_files_only
            )

    if text_encoder_uris:
        for text_encoder_uri in text_encoder_uris:
            if not _text_encoder_not_default(text_encoder_uri):
                continue

            parsed = _uris.TextEncoderUri.parse(text_encoder_uri)

            usage += _hfutil.estimate_model_memory_use(
                repo_id=parsed.model,
                revision=parsed.revision,
                subfolder=parsed.subfolder,
                use_auth_token=auth_token,
                local_files_only=local_files_only)

    return usage


def set_vae_slicing_tiling(pipeline: diffusers.DiffusionPipeline,
                           vae_tiling: bool,
                           vae_slicing: bool):
    """
    Set the vae_slicing and vae_tiling status on a created huggingface diffusers pipeline.

    :raises UnsupportedPipelineConfigError: if the pipeline does not support one or both
        of the provided values for ``vae_tiling`` and ``vae_slicing``

    :param pipeline: pipeline object
    :param vae_tiling: tiling status
    :param vae_slicing: slicing status
    """

    has_vae = hasattr(pipeline, 'vae') and pipeline.vae is not None
    pipeline_class = pipeline.__class__

    if vae_tiling:
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

    if vae_slicing:
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


def get_torch_pipeline_modules(pipeline: diffusers.DiffusionPipeline):
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


def _disable_to(module):
    def dummy(*args, **kwargs):
        pass

    module.to = dummy
    _messages.debug_log(
        f'Disabled .to() on module / model containing meta tensors: {_types.fullname(module)}')


def enable_sequential_cpu_offload(pipeline: diffusers.DiffusionPipeline,
                                  device: torch.device | str = "cuda"):
    """
    Enable sequential offloading on a torch pipeline, in a way dgenerate can keep track of.

    :param pipeline: the pipeline
    :param device: the device
    """
    torch_device = torch.device(device)

    _set_sequential_cpu_offload_flag(pipeline, True)
    for name, model in get_torch_pipeline_modules(pipeline).items():
        if name in pipeline._exclude_from_cpu_offload:
            continue
        elif not is_sequential_cpu_offload_enabled(model):
            if _quanto.is_quantized_and_frozen(model):
                _messages.debug_log(
                    f'Skipping sequential offload on optimum '
                    f'quantized & frozen module: {name}={model.__class__.__name__}')
            else:
                _set_sequential_cpu_offload_flag(model, True)
                accelerate.cpu_offload(model, torch_device, offload_buffers=len(model._parameters) > 0)
                _disable_to(model)


def enable_model_cpu_offload(pipeline: diffusers.DiffusionPipeline,
                             device: torch.device | str = "cuda"):
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

    pipeline._offload_gpu_id = torch_device.index or getattr(pipeline, "_offload_gpu_id", 0)

    device_type = torch_device.type
    device = torch.device(f"{device_type}:{pipeline._offload_gpu_id}")

    if pipeline.device.type != "cpu":
        pipeline.to("cpu", silence_dtype_warnings=True)
        device_mod = getattr(torch, pipeline.device.type, None)
        if hasattr(device_mod, "empty_cache") and device_mod.is_available():
            device_mod.empty_cache()

    _set_cpu_offload_flag(pipeline, True)

    all_model_components = {k: v for k, v in pipeline.components.items() if isinstance(v, torch.nn.Module)}

    pipeline._all_hooks = []
    hook = None
    for model_str in pipeline.model_cpu_offload_seq.split("->"):
        model = all_model_components.pop(model_str, None)
        if not isinstance(model, torch.nn.Module):
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

    to_device = torch.device(device)

    pipeline_device = get_torch_device(pipeline)

    all_modules_on_device = all(to_device == get_torch_device(m) for m in get_torch_pipeline_modules(pipeline).values())
    pipeline_on_device = get_torch_device(pipeline) == pipeline_device

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

    if pipeline_device != to_device:
        if to_device.type != 'cpu':
            _cache.pipeline_off_cpu_update_cache_info(pipeline)
        else:
            _cache.pipeline_to_cpu_update_cache_info(pipeline)

    for name, value in get_torch_pipeline_modules(pipeline).items():

        current_device = get_torch_device(value)

        if current_device.type == 'meta':
            _messages.debug_log(
                f'pipeline_to() Not moving module "{name} = {value.__class__.__name__}" to "{device}" '
                f'as its device value is "meta".')
            _disable_to(value)
            continue

        if current_device == to_device:
            _messages.debug_log(
                f'pipeline_to() Not moving module "{name} = {value.__class__.__name__}" to "{device}" '
                f'as it is already on that device.')
            continue

        if is_model_cpu_offload_enabled(value) and to_device.type != 'cpu':
            _messages.debug_log(
                f'pipeline_to() Not moving module "{name} = {value.__class__.__name__}" to "{device}" '
                f'as it has cpu offload enabled and can only move to cpu.')
            continue

        cache_meth = None
        if current_device.type == 'cpu' and to_device.type != 'cpu':
            cache_meth = '_off_cpu_update_cache_info'
        elif current_device.type != 'cpu' and to_device.type == 'cpu':
            cache_meth = '_to_cpu_update_cache_info'

        if cache_meth:
            if name.startswith('text_encoder'):
                getattr(_cache, 'text_encoder' + cache_meth)(value)
            else:
                try:
                    getattr(_cache, name + cache_meth)(value)
                except AttributeError:
                    _messages.debug_log(
                        f'pipeline_to() No cache update method for module "{name}".')

        _messages.debug_log(
            f'pipeline_to() Moving module "{name}" of pipeline {_types.fullname(pipeline)} '
            f'from device "{current_device}" to device "{to_device}"')

        value.to(device)

    if device == 'cpu':
        torch.cuda.empty_cache()


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
        torch.cuda.empty_cache()
        gc.collect()

        raise _d_exceptions.OutOfMemoryError(e)
    except MemoryError:
        # probably out of RAM on a back
        # to CPU move not much we can do
        gc.collect()
        raise _d_exceptions.OutOfMemoryError('cpu (system memory)')


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
                            _messages.log(f'{label} exceeds max token length '
                                          f'of {max_length} for the model\'s tokenizer '
                                          f'and will be truncated: "{p}"', level=_messages.WARNING)
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

    :param collect: call ``gc.collect`` and ``torch.cuda.empty_cache`` if
        there is a pipeline to dereference?
    """
    global _LAST_CALLED_PIPELINE

    if _LAST_CALLED_PIPELINE is not None:

        pipeline_to(_LAST_CALLED_PIPELINE, 'cpu')

        _LAST_CALLED_PIPELINE = None

        if collect:
            gc.collect()
            torch.cuda.empty_cache()


# noinspection PyCallingNonCallable
@torch.inference_mode()
def call_pipeline(pipeline: diffusers.DiffusionPipeline,
                  device: torch.device | str | None = 'cuda',
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
            translated = prompt_weighter.translate_to_embeds(pipeline, device, kwargs)
        except _d_exceptions.TORCH_CUDA_OOM_EXCEPTIONS as e:
            _d_exceptions.raise_if_not_cuda_oom(e)
            _cleanup_prompt_weighter()
            torch.cuda.empty_cache()
            gc.collect()
            raise _d_exceptions.OutOfMemoryError(e)
        except MemoryError:
            _cleanup_prompt_weighter()
            gc.collect()
            raise _d_exceptions.OutOfMemoryError('cpu (system memory)')
        except Exception as e:
            _cleanup_prompt_weighter()
            torch.cuda.empty_cache()
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
                pipe_result = pipeline(**kwargs)
            else:
                pipe_result = pipeline(**_call_prompt_weighter())
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
        torch.cuda.empty_cache()

        # force garbage collection
        gc.collect()

    def _call_pipeline():
        nonlocal enable_retry_pipe
        try:
            return _call_pipeline_raw()
        except _d_exceptions.TORCH_CUDA_OOM_EXCEPTIONS as e:
            _d_exceptions.raise_if_not_cuda_oom(e)
            _torch_oom_handler()
            raise _d_exceptions.OutOfMemoryError(e)
        except MemoryError:
            gc.collect()
            raise _d_exceptions.OutOfMemoryError('cpu (system memory)')
        except Exception as e:
            # same cleanup
            _torch_oom_handler()
            raise

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
            pipeline_to(pipeline, device)
            result = _call_pipeline()
            _LAST_CALLED_PIPELINE = pipeline
            return result

    if _LAST_CALLED_PIPELINE is not None and hasattr(_LAST_CALLED_PIPELINE, 'to'):
        _messages.debug_log(
            f'Moving previously called pipeline '
            f'"{_LAST_CALLED_PIPELINE.__class__.__name__}", back to the CPU.')

        pipeline_to(_LAST_CALLED_PIPELINE, 'cpu')

    try:
        pipeline_to(pipeline, device)
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
        pipeline_to(pipeline, device)
        result = _call_pipeline()

    _LAST_CALLED_PIPELINE = pipeline
    return result


class PipelineCreationResult:
    def __init__(self, pipeline):
        self._pipeline = pipeline

    @property
    def pipeline(self):
        return self._pipeline

    def get_pipeline_modules(self, names=collections.abc.Iterable[str]):
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


class TorchPipelineCreationResult(PipelineCreationResult):
    @property
    def pipeline(self) -> diffusers.DiffusionPipeline:
        """
        A created subclass of :py:class:`diffusers.DiffusionPipeline`
        """
        return super().pipeline

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

    def __init__(self,
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
        super().__init__(pipeline)
        self.parsed_unet_uri = parsed_unet_uri
        self.parsed_vae_uri = parsed_vae_uri
        self.parsed_lora_uris = parsed_lora_uris
        self.parsed_textual_inversion_uris = parsed_textual_inversion_uris
        self.parsed_controlnet_uris = parsed_controlnet_uris
        self.parsed_t2i_adapter_uris = parsed_t2i_adapter_uris
        self.parsed_ip_adapter_uris = parsed_ip_adapter_uris
        self.parsed_image_encoder_uri = parsed_image_encoder_uri
        self.parsed_transformer_uri = parsed_transformer_uri

    def call(self,
             device: str | None = 'cuda',
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


def create_torch_diffusion_pipeline(model_path: str,
                                    model_type: _enums.ModelType = _enums.ModelType.TORCH,
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
                                    scheduler: _types.OptionalString = None,
                                    safety_checker: bool = False,
                                    auth_token: _types.OptionalString = None,
                                    device: str = 'cuda',
                                    extra_modules: dict[str, typing.Any] | None = None,
                                    model_cpu_offload: bool = False,
                                    sequential_cpu_offload: bool = False,
                                    local_files_only: bool = False) -> TorchPipelineCreationResult:
    """
    Create a :py:class:`diffusers.DiffusionPipeline` in dgenerates in memory cacheing system.

    :param model_type:  :py:class:`dgenerate.pipelinewrapper.ModelType` enum value
    :param model_path: huggingface slug, huggingface blob link, path to folder on disk, path to file on disk
    :param pipeline_type: :py:class:`dgenerate.pipelinewrapper.PipelineType` enum value
    :param revision: huggingface repo revision (branch)
    :param variant: model weights name variant, for example 'fp16'
    :param subfolder: huggingface repo subfolder if applicable
    :param dtype: Optional :py:class:`dgenerate.pipelinewrapper.DataType` enum value
    :param unet_uri: Optional ``--unet`` URI string for specifying a specific UNet
    :param transformer_uri: Optional ``--transformer`` URI string for specifying a specific Transformer,
        currently this is only supported for Stable Diffusion 3 models.
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
    :param scheduler: Optional scheduler (sampler) class name, unqualified, or "help" / "helpargs" to print supported values
        to STDOUT and raise :py:exc:`dgenerate.pipelinewrapper.SchedulerHelpException`.  Dgenerate URI syntax is supported
        for overriding the schedulers constructor parameter defaults.
    :param safety_checker: Safety checker enabled? default is ``False``
    :param auth_token: Optional huggingface API token for accessing repositories that are restricted to your account
    :param device: Optional ``--device`` string, defaults to "cuda"
    :param extra_modules: Extra module arguments to pass directly into
        :py:meth:`diffusers.DiffusionPipeline.from_single_file` or :py:meth:`diffusers.DiffusionPipeline.from_pretrained`
    :param model_cpu_offload: This pipeline has model_cpu_offloading enabled?
    :param sequential_cpu_offload: This pipeline has sequential_cpu_offloading enabled?
    :param local_files_only: Only look in the huggingface cache and do not connect to download models?

    :raises InvalidModelFileError:
    :raises ModelNotFoundError:
    :raises InvalidModelUriError:
    :raises InvalidSchedulerNameError:
    :raises UnsupportedPipelineConfigError:

    :return: :py:class:`.TorchPipelineCreationResult`
    """
    __locals = locals()

    for name, value in __locals.items():
        if name.endswith('_uris') and isinstance(value, str):
            __locals[name] = [value]

    try:
        return _create_torch_diffusion_pipeline(**__locals)
    except (huggingface_hub.utils.HFValidationError,
            huggingface_hub.utils.HfHubHTTPError) as e:
        raise _hfutil.ModelNotFoundError(e)


class TorchPipelineFactory:
    """
    Combines :py:func:`.create_torch_diffusion_pipeline` and :py:func:`.set_vae_slicing_tiling` into a factory
    that can recreate the same Torch pipeline over again, possibly from cache.
    """

    def __init__(self,
                 model_path: str,
                 model_type: _enums.ModelType = _enums.ModelType.TORCH,
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
                 scheduler: _types.OptionalString = None,
                 safety_checker: bool = False,
                 auth_token: _types.OptionalString = None,
                 device: str = 'cuda',
                 extra_modules: dict[str, typing.Any] | None = None,
                 model_cpu_offload: bool = False,
                 sequential_cpu_offload: bool = False,
                 local_files_only: bool = False,
                 vae_tiling=False,
                 vae_slicing=False):
        self._args = {k: v for k, v in
                      _types.partial_deep_copy_container(locals()).items()
                      if k not in {'self', 'vae_tiling', 'vae_slicing'}}

        self._vae_tiling = vae_tiling
        self._vae_slicing = vae_slicing

    def __call__(self) -> TorchPipelineCreationResult:
        """
        :raises InvalidModelFileError:
        :raises ModelNotFoundError:
        :raises InvalidModelUriError:
        :raises InvalidSchedulerNameError:
        :raises UnsupportedPipelineConfigError:

        :return: :py:class:`.TorchPipelineCreationResult`
        """
        r = create_torch_diffusion_pipeline(**self._args)
        set_vae_slicing_tiling(r.pipeline,
                               vae_tiling=self._vae_tiling,
                               vae_slicing=self._vae_slicing)
        return r


def _text_encoder_help(pipeline_class):
    _messages.log(
        'Text encoder type help:\n\n' +
        ' ' * 4 + (('\n' + ' ' * 4).join(
            str(idx) + ' = ' + n for idx, n in
            enumerate(v[1].__name__ for v in
                      typing.get_type_hints(pipeline_class.__init__).items()
                      if v[0].startswith('text_encoder')))))
    raise ArgumentHelpException()


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


def _text_encoder_not_default(uri):
    return uri and uri != '+'


def _text_encoder_null(uri):
    return uri and uri.lower() == 'null'


def _torch_args_hasher(args):
    def text_encoder_uri_parse(uri):
        if uri is None or uri.strip() == '+':
            return None

        if uri.strip() == 'help':
            return 'help'

        if uri.strip() == 'null':
            return 'null'

        return _uris.TextEncoderUri.parse(uri)

    custom_hashes = {
        'unet_uri': _cache.uri_hash_with_parser(_uris.UNetUri.parse),
        'transformer_uri': _cache.uri_hash_with_parser(_uris.TransformerUri),
        'vae_uri': _cache.uri_hash_with_parser(_uris.VAEUri.parse),
        'image_encoder_uri': _cache.uri_hash_with_parser(_uris.ImageEncoderUri),
        'lora_uris': _cache.uri_list_hash_with_parser(_uris.LoRAUri.parse),
        'ip_adapter_uris': _cache.uri_list_hash_with_parser(_uris.IPAdapterUri),
        'textual_inversion_uris': _cache.uri_list_hash_with_parser(_uris.TextualInversionUri.parse),
        'text_encoder_uris': _cache.uri_list_hash_with_parser(text_encoder_uri_parse),
        'controlnet_uris': _cache.uri_list_hash_with_parser(_uris.ControlNetUri.parse,
                                                            exclude={'scale', 'start', 'end'}),
        't2i_adapter_uris': _cache.uri_list_hash_with_parser(_uris.T2IAdapterUri.parse,
                                                             exclude={'scale'})
    }
    return _d_memoize.args_cache_key(args, custom_hashes=custom_hashes)


def _torch_on_hit(key, hit):
    _d_memoize.simple_cache_hit_debug("Torch Pipeline", key, hit.pipeline)


def _torch_on_create(key, new):
    _d_memoize.simple_cache_miss_debug('Torch Pipeline', key, new.pipeline)


@_memoize(_cache._PIPELINE_CACHE,
          exceptions={'local_files_only'},
          hasher=_torch_args_hasher,
          on_hit=_torch_on_hit,
          on_create=_torch_on_create)
def _create_torch_diffusion_pipeline(
        model_path: str,
        model_type: _enums.ModelType = _enums.ModelType.TORCH,
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
        scheduler: _types.OptionalString = None,
        safety_checker: bool = False,
        auth_token: _types.OptionalString = None,
        device: str = 'cuda',
        extra_modules: dict[str, typing.Any] | None = None,
        model_cpu_offload: bool = False,
        sequential_cpu_offload: bool = False,
        local_files_only: bool = False
) -> TorchPipelineCreationResult:
    # Ensure model path is specified
    if not model_path:
        raise ValueError('model_path must be specified.')

    # Ensure model type is a Torch ModelType
    if not _enums.model_type_is_torch(model_type):
        raise UnsupportedPipelineConfigError('model_type must be a TORCH ModelType enum value.')

    # Offload checks
    if model_cpu_offload and sequential_cpu_offload:
        raise UnsupportedPipelineConfigError(
            'model_cpu_offload and sequential_cpu_offload may not be enabled simultaneously.')

    # Device check
    if not _util.is_valid_device_string(device):
        raise UnsupportedPipelineConfigError(
            'device" must be "cuda" (optionally with a device ordinal "cuda:N") or "cpu"')

    # Flux model restrictions
    if _enums.model_type_is_flux(model_type):
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                'Flux --model-type values are not compatible with --control-nets.')
        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                'Flux --model-type values are not compatible with --t2i-adapters.')
        if ip_adapter_uris:
            raise UnsupportedPipelineConfigError(
                'Flux --model-type values are not compatible with --ip-adapters.')
        if textual_inversion_uris:
            raise UnsupportedPipelineConfigError(
                'Flux --model-type values are not compatible with --textual-inversions.')
        if image_encoder_uri:
            raise UnsupportedPipelineConfigError(
                'Flux --model-type values are not compatible with --image-encoder.')

    # Deep Floyd model restrictions
    if _enums.model_type_is_floyd(model_type):
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                'Deep Floyd --model-type values are not compatible with --control-nets.')
        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                'Deep Floyd --model-type values are not compatible with --t2i-adapters.')
        if ip_adapter_uris:
            raise UnsupportedPipelineConfigError(
                'Deep Floyd --model-type values are not compatible with --ip-adapters.')
        if textual_inversion_uris:
            raise UnsupportedPipelineConfigError(
                'Deep Floyd --model-type values are not compatible with --textual-inversions.')
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
        if ip_adapter_uris:
            raise UnsupportedPipelineConfigError(
                'Stable Cascade --model-type values are not compatible with --ip-adapters.')
        if textual_inversion_uris:
            raise UnsupportedPipelineConfigError(
                'Stable Cascade --model-type values are not compatible with --textual-inversions.')
        if lora_uris:
            raise UnsupportedPipelineConfigError(
                'Stable Cascade --model-type values are not compatible with --loras.')
        if vae_uri:
            raise UnsupportedPipelineConfigError(
                'Stable Cascade --model-type values are not compatible with --vae.')

    # Torch SD3 restrictions
    if _enums.model_type_is_sd3(model_type):
        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                '--model-type torch-sd3 is not compatible with --t2i-adapters.')
        if ip_adapter_uris:
            raise UnsupportedPipelineConfigError(
                '--model-type torch-sd3 is not compatible with --ip-adapters.')
        if unet_uri:
            raise UnsupportedPipelineConfigError(
                '--model-type torch-sd3 is not compatible with --unet.')
        if textual_inversion_uris:
            raise UnsupportedPipelineConfigError(
                '--model-type torch-sd3 is not compatible with --textual-inversions.')
        if image_encoder_uri:
            raise UnsupportedPipelineConfigError(
                '--model-type torch-sd3 is not compatible with --image-encoder.')

    if transformer_uri:
        if not _enums.model_type_is_sd3(model_type) and not _enums.model_type_is_flux(model_type):
            raise UnsupportedPipelineConfigError(
                '--transformer is only supported for --model-type torch-sd3 and torch-flux.')

    # Incompatible combinations
    if controlnet_uris and t2i_adapter_uris:
        raise UnsupportedPipelineConfigError(
            '--control-nets and --t2i-adapters cannot be used together.')

    if image_encoder_uri and not ip_adapter_uris and model_type != _enums.ModelType.TORCH_S_CASCADE:
        raise UnsupportedPipelineConfigError(
            '--image-encoder cannot be specified without --ip-adapters if --model-type is not torch-s-cascade.')

    # Pix2Pix model restrictions
    is_pix2pix = _enums.model_type_is_pix2pix(model_type)

    if is_pix2pix:
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                'Pix2Pix --model-type values are not compatible with --control-nets.')
        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                'Pix2Pix --model-type values are not compatible with --t2i-adapters.')
        if ip_adapter_uris and model_type != _enums.ModelType.TORCH_PIX2PIX:
            raise UnsupportedPipelineConfigError(
                'Only Pix2Pix --model-type torch-pix2pix is compatible '
                'with --ip-adapters. Pix2Pix SDXL is not supported.')
        if image_encoder_uri and model_type != _enums.ModelType.TORCH_PIX2PIX:
            raise UnsupportedPipelineConfigError(
                'Only Pix2Pix --model-type torch-pix2pix is compatible '
                'with --image-encoder. Pix2Pix SDXL is not supported.')

    is_sdxl = _enums.model_type_is_sdxl(model_type)

    # Pipeline class selection
    if _enums.model_type_is_upscaler(model_type):
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                'Upscaler models are not compatible with --control-nets.')
        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                'Upscaler models are not compatible with --t2i-adapters.')
        if ip_adapter_uris:
            raise UnsupportedPipelineConfigError(
                'Upscaler models are not compatible with --ip-adapters.')
        if image_encoder_uri:
            raise UnsupportedPipelineConfigError(
                'Upscaler models are not compatible with --image-encoder.')
        if pipeline_type != _enums.PipelineType.IMG2IMG and not scheduler_is_help(scheduler):
            raise UnsupportedPipelineConfigError(
                'Upscaler models only work with img2img generation, IE: --image-seeds (with no image masks).')

        if model_type == _enums.ModelType.TORCH_UPSCALER_X2:
            if lora_uris or textual_inversion_uris:
                raise UnsupportedPipelineConfigError(
                    '--model-type torch-upscaler-x2 is not compatible with --loras or --textual-inversions.')

        pipeline_class = (
            diffusers.StableDiffusionUpscalePipeline
            if model_type == _enums.ModelType.TORCH_UPSCALER_X4
            else diffusers.StableDiffusionLatentUpscalePipeline
        )
    else:
        if pipeline_type == _enums.PipelineType.TXT2IMG:
            if is_pix2pix:
                if not (scheduler_is_help(scheduler) or text_encoder_is_help(text_encoder_uris)):
                    raise UnsupportedPipelineConfigError(
                        'Pix2Pix models only work in img2img mode and cannot work without --image-seeds.')
                else:
                    pipeline_class = (
                        diffusers.StableDiffusionXLInstructPix2PixPipeline
                        if is_sdxl
                        else diffusers.StableDiffusionInstructPix2PixPipeline
                    )

            if model_type == _enums.ModelType.TORCH_IF:
                pipeline_class = diffusers.IFPipeline
            elif model_type == _enums.ModelType.TORCH_IFS:
                if not (scheduler_is_help(scheduler) or text_encoder_is_help(text_encoder_uris)):
                    raise UnsupportedPipelineConfigError(
                        'Deep Floyd IF super-resolution (IFS) only works in '
                        'img2img mode and cannot work without --image-seeds.')
                else:
                    pipeline_class = diffusers.IFSuperResolutionPipeline
            elif model_type == _enums.ModelType.TORCH_S_CASCADE:
                pipeline_class = diffusers.StableCascadePriorPipeline
            elif model_type == _enums.ModelType.TORCH_S_CASCADE_DECODER:
                pipeline_class = diffusers.StableCascadeDecoderPipeline
            elif model_type == _enums.ModelType.TORCH_FLUX:
                pipeline_class = type(
                    'FluxPipeline',
                    (diffusers.loaders.FromSingleFileMixin, diffusers.FluxPipeline),
                    {},
                )
            elif model_type == _enums.ModelType.TORCH_SD3:
                pipeline_class = (
                    diffusers.StableDiffusion3Pipeline
                    if not controlnet_uris
                    else diffusers.StableDiffusion3ControlNetPipeline
                )
            elif t2i_adapter_uris:
                # The custom type is a hack to support from_single_file for SD1.5 - 2
                # models with the associated pipeline class which does not inherit
                # the correct mixin to do so but can use the mixin just fine
                pipeline_class = (
                    diffusers.StableDiffusionXLAdapterPipeline
                    if is_sdxl
                    else type(
                        'StableDiffusionAdapterPipeline',
                        (diffusers.loaders.FromSingleFileMixin, diffusers.StableDiffusionAdapterPipeline),
                        {},
                    )
                )
            elif controlnet_uris:
                pipeline_class = (
                    diffusers.StableDiffusionXLControlNetPipeline
                    if is_sdxl
                    else diffusers.StableDiffusionControlNetPipeline
                )
            else:
                pipeline_class = (
                    diffusers.StableDiffusionXLPipeline
                    if is_sdxl
                    else diffusers.StableDiffusionPipeline
                )

        elif pipeline_type == _enums.PipelineType.IMG2IMG:
            if controlnet_uris:
                if is_pix2pix:
                    raise UnsupportedPipelineConfigError(
                        'Pix2Pix models are not compatible with --control-nets.')

            if is_pix2pix:
                pipeline_class = (
                    diffusers.StableDiffusionXLInstructPix2PixPipeline
                    if is_sdxl
                    else diffusers.StableDiffusionInstructPix2PixPipeline
                )
            elif model_type == _enums.ModelType.TORCH_IF:
                pipeline_class = diffusers.IFImg2ImgPipeline
            elif model_type == _enums.ModelType.TORCH_IFS:
                pipeline_class = diffusers.IFSuperResolutionPipeline
            elif model_type == _enums.ModelType.TORCH_IFS_IMG2IMG:
                pipeline_class = diffusers.IFImg2ImgSuperResolutionPipeline
            elif model_type == _enums.ModelType.TORCH_S_CASCADE:
                pipeline_class = diffusers.StableCascadePriorPipeline
            elif model_type == _enums.ModelType.TORCH_S_CASCADE_DECODER:
                raise UnsupportedPipelineConfigError(
                    'Stable Cascade decoder models do not support img2img.')
            elif model_type == _enums.ModelType.TORCH_FLUX:
                raise UnsupportedPipelineConfigError(
                    'Flux model types do not support img2img.')
            elif model_type == _enums.ModelType.TORCH_SD3:
                if controlnet_uris:
                    raise UnsupportedPipelineConfigError(
                        '--model-type torch-sd3 does not support img2img mode with ControlNet models.')
                if lora_uris:
                    raise UnsupportedPipelineConfigError(
                        '--model-type torch-sd3 does not support --loras in img2img mode.')
                pipeline_class = diffusers.StableDiffusion3Img2ImgPipeline
            elif t2i_adapter_uris:
                raise UnsupportedPipelineConfigError(
                    'img2img mode is not supported with --t2i-adapters.')
            elif controlnet_uris:
                pipeline_class = (
                    diffusers.StableDiffusionXLControlNetImg2ImgPipeline
                    if is_sdxl
                    else diffusers.StableDiffusionControlNetImg2ImgPipeline
                )
            else:
                pipeline_class = (
                    diffusers.StableDiffusionXLImg2ImgPipeline
                    if is_sdxl
                    else diffusers.StableDiffusionImg2ImgPipeline
                )

        elif pipeline_type == _enums.PipelineType.INPAINT:
            if is_pix2pix:
                raise UnsupportedPipelineConfigError(
                    'Pix2Pix models only work in img2img mode and cannot work in inpaint mode (with a mask).')
            if _enums.model_type_is_s_cascade(model_type):
                raise UnsupportedPipelineConfigError(
                    'Stable Cascade model types do not support inpainting.')
            if model_type == _enums.ModelType.TORCH_SD3:
                raise UnsupportedPipelineConfigError(
                    'Stable Diffusion 3 model types do not support inpainting.')
            if model_type == _enums.ModelType.TORCH_FLUX:
                raise UnsupportedPipelineConfigError(
                    'Flux model types do not support inpainting.')
            if _enums.model_type_is_upscaler(model_type):
                raise UnsupportedPipelineConfigError(
                    'Stable Diffusion upscaler model types do not support inpainting.')

            if model_type == _enums.ModelType.TORCH_IF:
                pipeline_class = diffusers.IFInpaintingPipeline
            elif model_type == _enums.ModelType.TORCH_IFS:
                pipeline_class = diffusers.IFInpaintingSuperResolutionPipeline
            elif t2i_adapter_uris:
                raise UnsupportedPipelineConfigError(
                    'inpaint mode is not supported with --t2i-adapters.')
            elif controlnet_uris:
                pipeline_class = (
                    diffusers.StableDiffusionXLControlNetInpaintPipeline
                    if is_sdxl
                    else diffusers.StableDiffusionControlNetInpaintPipeline
                )
            else:
                pipeline_class = (
                    diffusers.StableDiffusionXLInpaintPipeline
                    if is_sdxl
                    else diffusers.StableDiffusionInpaintPipeline
                )
        else:
            # Should be impossible
            raise UnsupportedPipelineConfigError('Pipeline type not implemented.')

    text_encoder_count = len(
        [a for a in inspect.getfullargspec(pipeline_class.__init__).args if a.startswith('text_encoder')])

    if not text_encoder_uris:
        text_encoder_uris = []
    elif text_encoder_is_help(text_encoder_uris):
        _text_encoder_help(pipeline_class)

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
                    _disable_to(module[1])
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
    scheduler_override = 'scheduler' in extra_modules
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

    model_path = _hfutil.download_non_hf_model(model_path)

    estimated_memory_usage = estimate_pipeline_memory_use(
        pipeline_type=pipeline_type,
        model_type=model_type,
        model_path=model_path,
        revision=revision,
        variant=variant,
        subfolder=subfolder,
        unet_uri=unet_uri if not unet_override else None,
        transformer_uri=transformer_uri if not transformer_override else None,
        vae_uri=vae_uri if not vae_override else None,
        lora_uris=lora_uris,
        image_encoder_uri=image_encoder_uri,
        ip_adapter_uris=ip_adapter_uris,
        text_encoder_uris=text_encoders,
        textual_inversion_uris=textual_inversion_uris,
        safety_checker=safety_checker and not safety_checker_override,
        auth_token=auth_token,
        extra_args=extra_modules,
        local_files_only=local_files_only
    )

    _messages.debug_log(
        f'Creating Torch Pipeline: "{pipeline_class.__name__}", '
        f'Estimated CPU Side Memory Use: {_memory.bytes_best_human_unit(estimated_memory_usage)}')

    _cache.enforce_pipeline_cache_constraints(
        new_pipeline_size=estimated_memory_usage)

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

    if text_encoder_uris:
        def load_text_encoder(uri):
            return uri.load(
                variant_fallback=variant,
                dtype_fallback=dtype,
                use_auth_token=auth_token,
                local_files_only=local_files_only,
                sequential_cpu_offload_member=sequential_cpu_offload,
                model_cpu_offload_member=model_cpu_offload)

        if not text_encoder_override and (len(text_encoder_uris) > 0) and \
                _text_encoder_not_default(text_encoder_uris[0]):
            creation_kwargs['text_encoder'] = load_text_encoder(
                _uris.TextEncoderUri.parse(text_encoder_uris[0]))
        if not text_encoder_2_override and (len(text_encoder_uris) > 1) and \
                _text_encoder_not_default(text_encoder_uris[1]):
            creation_kwargs['text_encoder_2'] = load_text_encoder(
                _uris.TextEncoderUri.parse(text_encoder_uris[1]))
        if not text_encoder_3_override and (len(text_encoder_uris) > 2) and \
                _text_encoder_not_default(text_encoder_uris[2]):
            creation_kwargs['text_encoder_3'] = load_text_encoder(
                _uris.TextEncoderUri.parse(text_encoder_uris[2]))

    if vae_uri is not None and not vae_override:
        parsed_vae_uri = _uris.VAEUri.parse(vae_uri)

        creation_kwargs['vae'] = \
            parsed_vae_uri.load(
                dtype_fallback=dtype,
                use_auth_token=auth_token,
                local_files_only=local_files_only,
                sequential_cpu_offload_member=sequential_cpu_offload,
                model_cpu_offload_member=model_cpu_offload)

        _messages.debug_log(lambda:
                            f'Added Torch VAE: "{vae_uri}" to pipeline: "{pipeline_class.__name__}"')

    if unet_uri is not None and not unet_override:
        parsed_unet_uri = _uris.UNetUri.parse(unet_uri)

        unet_parameter = 'unet'

        if model_type == _enums.ModelType.TORCH_S_CASCADE:
            unet_parameter = 'prior'
        elif model_type == _enums.ModelType.TORCH_S_CASCADE_DECODER:
            unet_parameter = 'decoder'

        unet_class = diffusers.UNet2DConditionModel if unet_parameter == 'unet' \
            else diffusers.models.unets.StableCascadeUNet

        creation_kwargs[unet_parameter] = \
            parsed_unet_uri.load(
                variant_fallback=variant,
                dtype_fallback=dtype,
                use_auth_token=auth_token,
                local_files_only=local_files_only,
                sequential_cpu_offload_member=sequential_cpu_offload,
                model_cpu_offload_member=model_cpu_offload,
                unet_class=unet_class)

        _messages.debug_log(lambda:
                            f'Added Torch UNet: "{unet_uri}" to pipeline: "{pipeline_class.__name__}"')

    if transformer_uri is not None and not transformer_override:
        parsed_transformer_uri = _uris.TransformerUri.parse(transformer_uri)

        if _enums.model_type_is_sd3(model_type):
            transformer_class = diffusers.SD3Transformer2DModel
        elif _enums.model_type_is_flux(model_type):
            transformer_class = diffusers.FluxTransformer2DModel

        creation_kwargs['transformer'] = \
            parsed_transformer_uri.load(
                variant_fallback=variant,
                dtype_fallback=dtype,
                use_auth_token=auth_token,
                local_files_only=local_files_only,
                sequential_cpu_offload_member=sequential_cpu_offload,
                model_cpu_offload_member=model_cpu_offload,
                transformer_class=transformer_class)

        _messages.debug_log(lambda:
                            f'Added Torch Transformer: "{transformer_uri}" to '
                            f'pipeline: "{pipeline_class.__name__}"')

    if image_encoder_uri is not None and not image_encoder_override:
        parsed_image_encoder_uri = _uris.ImageEncoderUri.parse(image_encoder_uri)

        creation_kwargs['image_encoder'] = parsed_image_encoder_uri.load(
            dtype_fallback=dtype,
            use_auth_token=auth_token,
            local_files_only=local_files_only,
            sequential_cpu_offload_member=sequential_cpu_offload,
            model_cpu_offload_member=model_cpu_offload,
        )

        _messages.debug_log(lambda:
                            f'Added Torch Image Encoder: "{image_encoder_uri}" to '
                            f'pipeline: "{pipeline_class.__name__}"')

    if t2i_adapter_uris and not adapter_override:
        t2i_adapters = None

        for t2i_adapter_uri in t2i_adapter_uris:
            parsed_t2i_adapter_uri = _uris.T2IAdapterUri.parse(t2i_adapter_uri)
            parsed_t2i_adapter_uris.append(parsed_t2i_adapter_uri)

            new_adapter = parsed_t2i_adapter_uri.load(
                use_auth_token=auth_token,
                dtype_fallback=dtype,
                local_files_only=local_files_only,
                sequential_cpu_offload_member=sequential_cpu_offload,
                model_cpu_offload_member=model_cpu_offload
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

    if controlnet_uris and not controlnet_override:

        controlnets = None
        controlnet_model_class = diffusers.ControlNetModel if not \
            _enums.model_type_is_sd3(model_type) else diffusers.SD3ControlNetModel

        for controlnet_uri in controlnet_uris:
            parsed_controlnet_uri = _uris.ControlNetUri.parse(controlnet_uri)

            parsed_controlnet_uris.append(parsed_controlnet_uri)

            new_net = parsed_controlnet_uri.load(
                use_auth_token=auth_token,
                dtype_fallback=dtype,
                local_files_only=local_files_only,
                sequential_cpu_offload_member=sequential_cpu_offload,
                model_cpu_offload_member=model_cpu_offload,
                model_class=controlnet_model_class)

            _messages.debug_log(lambda:
                                f'Added Torch ControlNet: "{controlnet_uri}" '
                                f'to pipeline: "{pipeline_class.__name__}"')

            if controlnets is not None:
                if not isinstance(controlnets, list):
                    controlnets = [controlnets, new_net]
                else:
                    controlnets.append(new_net)
            else:
                controlnets = new_net

        if _enums.model_type_is_sd3(model_type) and isinstance(controlnets, list):
            # not handled internally for whatever reason like the other pipelines
            creation_kwargs['controlnet'] = diffusers.SD3MultiControlNetModel(controlnets)
        else:
            creation_kwargs['controlnet'] = controlnets

    if _enums.model_type_is_floyd(model_type):
        creation_kwargs['watermarker'] = None

    if not safety_checker and not _enums.model_type_is_sdxl(model_type) and not safety_checker_override:
        creation_kwargs['safety_checker'] = None

    creation_kwargs.update(extra_modules)

    if _hfutil.is_single_file_model_load(model_path):
        if subfolder is not None:
            raise UnsupportedPipelineConfigError(
                'Single file model loads do not support the subfolder option.')
        try:
            pipeline = _pipeline_creation_args_debug(
                backend='Torch',
                cls=pipeline_class,
                method=pipeline_class.from_single_file,
                model=model_path,
                token=auth_token,
                revision=revision,
                variant=variant,
                torch_dtype=torch_dtype,
                use_safe_tensors=model_path.endswith('.safetensors'),
                local_files_only=local_files_only,
                **creation_kwargs)

        except diffusers.loaders.single_file.SingleFileComponentError as e:
            msg = str(e)
            if 'text_encoder' in msg:
                raise UnsupportedPipelineConfigError(
                    f'Single file load error, missing --text-encoders / --text-encoders2:\n{e}')
            else:
                raise UnsupportedPipelineConfigError(
                    f'Single file load error, missing component:\n{e}')
        except (ValueError, TypeError, NameError, OSError) as e:
            msg = str(e)
            if model_path in msg:
                raise InvalidModelFileError(f'invalid model file, unable to load: {model_path}')
            raise InvalidModelFileError(e)
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
            msg = str(e)
            if model_path in msg:
                raise InvalidModelFileError(f'invalid model file or repo slug: {model_path}')
            raise InvalidModelFileError(e)

    # Select Scheduler

    if not scheduler_override:
        load_scheduler(pipeline=pipeline,
                       model_path=model_path,
                       scheduler_name=scheduler)

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

    if lora_uris:
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
            _set_torch_safety_checker(pipeline, safety_checker)

    # Model Offloading

    if device.startswith('cuda'):
        if sequential_cpu_offload:
            enable_sequential_cpu_offload(pipeline, device)
        elif model_cpu_offload:
            enable_model_cpu_offload(pipeline, device)

    _cache.pipeline_create_update_cache_info(pipeline=pipeline,
                                             estimated_size=estimated_memory_usage)
    _messages.debug_log(f'Finished Creating Torch Pipeline: "{pipeline_class.__name__}"')

    return TorchPipelineCreationResult(
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
    )


__all__ = _types.module_all()
