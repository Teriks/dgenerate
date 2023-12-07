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
import typing

import accelerate
import diffusers
import diffusers.loaders
import huggingface_hub
import torch.nn
import torch.nn

import dgenerate.memoize as _d_memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.cache as _cache
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.hfutil as _hfutil
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.memoize import memoize as _memoize


class OutOfMemoryError(Exception):
    """
    Raised when a GPU or processing device runs out of memory.
    """

    def __init__(self, message):
        super().__init__(f'Device Out Of Memory: {message}')


class UnsupportedPipelineConfigError(Exception):
    """
    Occurs when the a diffusers pipeline is requested to be
    configured in a way that is unsupported by that pipeline.
    """
    pass


class InvalidSchedulerNameError(Exception):
    """
    Unknown scheduler name used
    """
    pass


class SchedulerHelpException(Exception):
    """
    Not an error, runtime scheduler help was requested by passing "help" to a scheduler name
    argument of :py:meth:`.DiffusionPipelineWrapper.__init__` such as ``scheduler`` or ``sdxl_refiner_scheduler``.
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


def scheduler_is_help(name: typing.Optional[str]):
    """
    This scheduler name is simply a request for help?, IE: "help"?

    :param name: string to test
    :return: true of false
    """
    if name is None:
        return False
    return name.strip().lower() == 'help'


def load_scheduler(pipeline: typing.Union[diffusers.DiffusionPipeline, diffusers.FlaxDiffusionPipeline],
                   scheduler_name=None, model_path: typing.Optional[str] = None):
    """
    Load a specific compatible scheduler class name onto a huggingface diffusers pipeline object.

    :param pipeline: pipeline object
    :param scheduler_name: compatible scheduler class name, pass "help" to receive a print out to STDOUT
        and raise :py:exc:`.SchedulerHelpException`
    :param model_path: Optional model path to be used in the message to STDOUT produced by passing "help"
    :return:
    """

    if scheduler_name is None:
        return

    compatibles = list(pipeline.scheduler.compatibles)

    if isinstance(pipeline, diffusers.loaders.LoraLoaderMixin):
        compatibles.append(diffusers.LCMScheduler)

    if isinstance(pipeline, diffusers.StableDiffusionLatentUpscalePipeline):
        # Seems to only work with this scheduler
        compatibles = [c for c in compatibles if c.__name__ == 'EulerDiscreteScheduler']

    if scheduler_is_help(scheduler_name):
        help_string = f'Compatible schedulers for "{model_path}" are:' + '\n\n'
        help_string += '\n'.join((" " * 4) + _textprocessing.quote(i.__name__) for i in compatibles) + '\n'
        _messages.log(help_string)
        raise SchedulerHelpException(help_string)

    for i in compatibles:
        if i.__name__ == scheduler_name:
            pipeline.scheduler = i.from_config(pipeline.scheduler.config)
            return

    raise InvalidSchedulerNameError(
        f'Scheduler named "{scheduler_name}" is not a valid compatible scheduler, '
        f'options are:\n\n{chr(10).join(sorted(" " * 4 + _textprocessing.quote(i.__name__.split(".")[-1]) for i in compatibles))}')


def estimate_pipeline_memory_use(
        pipeline_type: _enums.PipelineType,
        model_path: str,
        model_type: _enums.ModelType,
        revision: _types.Name = 'main',
        variant: _types.OptionalName = None,
        subfolder: _types.OptionalPath = None,
        unet_uri: _types.OptionalUri = None,
        vae_uri: _types.OptionalUri = None,
        lora_uris: _types.OptionalUris = None,
        textual_inversion_uris: _types.OptionalUris = None,
        safety_checker: bool = False,
        auth_token: str = None,
        extra_args: dict[str, typing.Any] = None,
        local_files_only: bool = False):
    """
    Estimate the CPU side memory use of a model.



    :param pipeline_type: :py:class:`dgenerate.pipelinewrapper.PipelineType`
    :param model_path: huggingface slug, blob link, path to folder on disk, path to model file.
    :param model_type: :py:class:`dgenerate.pipelinewrapper.ModelType`
    :param revision: huggingface repo revision if using a huggingface slug
    :param variant: model file variant desired, for example "fp16"
    :param subfolder: huggingface repo subfolder if using a huggingface slug
    :param unet_uri: optional user specified ``--unet`` URI that will be loaded on to the pipeline
    :param vae_uri: optional user specified ``--vae`` URI that will be loaded on to the pipeline
    :param lora_uris: optional user specified ``--loras`` URIs that will be loaded on to the pipeline
    :param textual_inversion_uris: optional user specified ``--textual-inversion`` URIs that will be loaded on to the pipeline
    :param safety_checker: consider the safety checker? dgenerate usually loads the safety checker and then retroactively
        disables it if needed, so it usually considers the size of the safety checker model.
    :param auth_token: optional huggingface auth token to access restricted repositories that your account has access to.
    :param extra_args: ``extra_args`` as to be passed to :py:func:`.create_torch_diffusion_pipeline`
        or :py:func:`.create_flax_diffusion_pipeline`
    :param local_files_only: Only ever attempt to look in the local huggingface cache? if ``False`` the huggingface
        API will be contacted when necessary.
    :return: size estimate in bytes.
    """

    if extra_args is None:
        extra_args = dict()

    usage = _hfutil.estimate_model_memory_use(
        repo_id=model_path,
        revision=revision,
        variant=variant,
        subfolder=subfolder,
        include_unet=not unet_uri or 'unet' not in extra_args,
        include_vae=not vae_uri or 'vae' not in extra_args,
        safety_checker=safety_checker and 'safety_checker' not in extra_args,
        include_text_encoder='text_encoder' not in extra_args,
        include_text_encoder_2='text_encoder_2' not in extra_args,
        use_auth_token=auth_token,
        local_files_only=local_files_only,
        flax=_enums.model_type_is_flax(model_type),
        sentencepiece=_enums.model_type_is_floyd(model_type)
    )

    if lora_uris:
        for lora_uri in lora_uris:
            parsed = _uris.LoRAUri.parse(lora_uri)

            usage += _hfutil.estimate_model_memory_use(
                repo_id=parsed.model,
                revision=parsed.revision,
                subfolder=parsed.subfolder,
                weight_name=parsed.weight_name,
                use_auth_token=auth_token,
                local_files_only=local_files_only,
                flax=_enums.model_type_is_flax(model_type)
            )

    if textual_inversion_uris:
        for textual_inversion_uri in textual_inversion_uris:
            parsed = _uris.TextualInversionUri.parse(textual_inversion_uri)

            usage += _hfutil.estimate_model_memory_use(
                repo_id=parsed.model,
                revision=parsed.revision,
                subfolder=parsed.subfolder,
                weight_name=parsed.weight_name,
                use_auth_token=auth_token,
                local_files_only=local_files_only,
                flax=_enums.model_type_is_flax(model_type)
            )

    return usage


def set_vae_slicing_tiling(pipeline: typing.Union[diffusers.DiffusionPipeline,
                                                  diffusers.FlaxDiffusionPipeline],
                           vae_tiling: bool,
                           vae_slicing: bool):
    """
    Set the vae_slicing and vae_tiling status on a created huggingface diffusers pipeline.

    :param pipeline: pipeline object
    :param vae_tiling: tiling status
    :param vae_slicing: slicing status
    :return:
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


def _set_sequential_cpu_offload_flag(module: typing.Union[diffusers.DiffusionPipeline, torch.nn.Module], value: bool):
    module.DGENERATE_SEQUENTIAL_CPU_OFFLOAD = bool(value)

    _messages.debug_log(
        f'setting DGENERATE_SEQUENTIAL_CPU_OFFLOAD={value} on module "{module.__class__.__name__}"')


def _set_cpu_offload_flag(module: typing.Union[diffusers.DiffusionPipeline, torch.nn.Module], value: bool):
    module.DGENERATE_MODEL_CPU_OFFLOAD = bool(value)

    _messages.debug_log(
        f'setting DGENERATE_MODEL_CPU_OFFLOAD={value} on module "{module.__class__.__name__}"')


def is_sequential_cpu_offload_enabled(module: typing.Union[diffusers.DiffusionPipeline, torch.nn.Module]):
    """
    Test if a pipeline or torch neural net module created by dgenerate has sequential offload enabled.

    :param module: the module object
    :return: ``True`` or ``False``
    """
    return hasattr(module, 'DGENERATE_SEQUENTIAL_CPU_OFFLOAD') and bool(module.DGENERATE_SEQUENTIAL_CPU_OFFLOAD)


def is_model_cpu_offload_enabled(module: typing.Union[diffusers.DiffusionPipeline, torch.nn.Module]):
    """
    Test if a pipeline or torch neural net module created by dgenerate has model cpu offload enabled.

    :param module: the module object
    :return: ``True`` or ``False``
    """
    return hasattr(module, 'DGENERATE_MODEL_CPU_OFFLOAD') and bool(module.DGENERATE_MODEL_CPU_OFFLOAD)


def enable_sequential_cpu_offload(pipeline: diffusers.DiffusionPipeline,
                                  device: typing.Union[torch.device, str] = "cuda"):
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
            _set_sequential_cpu_offload_flag(model, True)
            accelerate.cpu_offload(model, torch_device, offload_buffers=len(model._parameters) > 0)


def enable_model_cpu_offload(pipeline: diffusers.DiffusionPipeline,
                             device: typing.Union[torch.device, str] = "cuda"):
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


def get_torch_device(component: typing.Union[diffusers.DiffusionPipeline, torch.nn.Module]) -> torch.device:
    """
    Get the device that a pipeline or pipeline component exists on.

    :param component: pipeline or pipeline component.
    :return: :py:class:`torch.device`
    """
    if hasattr(component, 'device'):
        return component.device
    elif hasattr(component, 'get_device'):
        return component.get_device()

    raise ValueError('component did not have a device attribute or the function get_device()')


def get_torch_device_string(component: typing.Union[diffusers.DiffusionPipeline, torch.nn.Module]) -> str:
    """
    Get the device string that a pipeline or pipeline component exists on.

    :param component: pipeline or pipeline component.
    :return: device string
    """
    return str(get_torch_device(component))


def pipeline_to(pipeline, device: typing.Union[torch.device, str, None]):
    """
    Move a diffusers pipeline to a device if possible, in a way that dgenerate can keep track of.

    If the pipeline does not possess the ``.to()`` method (such as with flax pipelines), this is a no-op.

    If ``device==None`` this is a no-op.

    Modules which are meta tensors will not be moved (sequentially offloaded modules)

    Modules which have model cpu offload enabled will not be moved unless they are moving to "cpu"

    :param pipeline: the pipeline
    :param device: the device

    :return: the moved pipeline
    """
    if device is None:
        return

    if not hasattr(pipeline, 'to'):
        return

    to_device = torch.device(device)

    if to_device.type != 'cpu':
        _cache.pipeline_off_cpu_update_cache_info(pipeline)
    else:
        _cache.pipeline_to_cpu_update_cache_info(pipeline)

    for name, value in get_torch_pipeline_modules(pipeline).items():

        current_device = get_torch_device(value)

        if current_device.type == 'meta' or current_device == to_device:
            continue

        if is_model_cpu_offload_enabled(value) and to_device.type != 'cpu':
            continue

        if name == 'unet':
            if to_device.type != 'cpu':
                _cache.unet_off_cpu_update_cache_info(value)
            else:
                _cache.unet_to_cpu_update_cache_info(value)
        elif name == 'vae':
            if to_device.type != 'cpu':
                _cache.vae_off_cpu_update_cache_info(value)
            else:
                _cache.vae_to_cpu_update_cache_info(value)
        elif name == 'controlnet':
            if to_device.type != 'cpu':
                _cache.controlnet_off_cpu_update_cache_info(value)
            else:
                _cache.controlnet_to_cpu_update_cache_info(value)

        _messages.debug_log(
            f'Moving module "{name}" of pipeline {_types.fullname(pipeline)} '
            f'from device "{current_device}" to device "{to_device}"')

        value.to(device)


_LAST_CALLED_PIPELINE = None


# noinspection PyCallingNonCallable
def call_pipeline(pipeline: typing.Union[diffusers.DiffusionPipeline, diffusers.FlaxDiffusionPipeline],
                  device: typing.Union[torch.device, str, None] = 'cuda', *args, **kwargs):
    """
    Call a diffusers pipeline, offload the last called pipeline to CPU before
    doing so if the last pipeline is not being called in succession

    :param pipeline: The pipeline

    :param device: The device to move the pipeline to before calling, it will be
        moved to this device if it is not already on the device. If the pipeline
        does not support moving to a device, such as with flax pipelines,
        this argument is ignored.

    :param args: diffusers pipeline arguments
    :param kwargs: diffusers pipeline keyword arguments
    :return: the result of calling the diffusers pipeline
    """

    global _LAST_CALLED_PIPELINE

    _messages.debug_log(f'Calling Pipeline: "{pipeline.__class__.__name__}",',
                        f'Device: "{device}",',
                        'Args:',
                        lambda: _textprocessing.debug_format_args(kwargs,
                                                                  value_transformer=lambda key, value:
                                                                  f'torch.Generator(seed={value.initial_seed()})'
                                                                  if isinstance(value, torch.Generator) else value))

    if pipeline is _LAST_CALLED_PIPELINE:
        return pipeline(*args, **kwargs)
    else:
        if hasattr(_LAST_CALLED_PIPELINE, 'to'):
            _messages.debug_log(
                f'Moving previously called pipeline "{_LAST_CALLED_PIPELINE.__class__.__name__}", back to the CPU.')

        pipeline_to(_LAST_CALLED_PIPELINE, 'cpu')

    pipeline_to(pipeline, device)

    result = pipeline(*args, **kwargs)

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
            * ``text_encoder``
            * ``text_encoder_2``
            * ``tokenizer``
            * ``tokenizer_2``
            * ``safety_checker``
            * ``feature_extractor``
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
            'text_encoder',
            'text_encoder_2',
            'tokenizer',
            'tokenizer_2',
            'safety_checker',
            'feature_extractor',
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

    parsed_unet_uri: typing.Optional[_uris.TorchUNetUri]
    """
    Parsed UNet URI if one was present
    """

    parsed_vae_uri: typing.Optional[_uris.TorchVAEUri]
    """
    Parsed VAE URI if one was present
    """

    parsed_lora_uris: collections.abc.Sequence[_uris.LoRAUri]
    """
    Parsed LoRA URIs if any were present
    """

    parsed_textual_inversion_uris: collections.abc.Sequence[_uris.TextualInversionUri]
    """
    Parsed Textual Inversion URIs if any were present
    """

    parsed_control_net_uris: collections.abc.Sequence[_uris.TorchControlNetUri]
    """
    Parsed ControlNet URIs if any were present
    """

    def __init__(self,
                 pipeline: diffusers.DiffusionPipeline,
                 parsed_unet_uri: typing.Optional[_uris.TorchUNetUri],
                 parsed_vae_uri: typing.Optional[_uris.TorchVAEUri],
                 parsed_lora_uris: collections.abc.Sequence[_uris.LoRAUri],
                 parsed_textual_inversion_uris: collections.abc.Sequence[_uris.TextualInversionUri],
                 parsed_control_net_uris: collections.abc.Sequence[_uris.TorchControlNetUri]):
        super().__init__(pipeline)
        self.parsed_unet_uri = parsed_unet_uri
        self.parsed_vae_uri = parsed_vae_uri
        self.parsed_lora_uris = parsed_lora_uris
        self.parsed_textual_inversion_uris = parsed_textual_inversion_uris
        self.parsed_control_net_uris = parsed_control_net_uris

    def call(self, device, *args, **kwargs) -> diffusers.utils.BaseOutput:
        """
        Call **pipeline**, see: :py:func:`.call_pipeline`

        :param device: move the pipeline to this device before calling
        :param args: forward args to pipeline
        :param kwargs: forward kwargs to pipeline
        :return: A subclass of :py:class:`diffusers.utils.BaseOutput`
        """
        return call_pipeline(self.pipeline, device, *args, **kwargs)


def create_torch_diffusion_pipeline(pipeline_type: _enums.PipelineType,
                                    model_path: str,
                                    model_type: _enums.ModelType = _enums.ModelType.TORCH,
                                    revision: _types.OptionalString = None,
                                    variant: _types.OptionalString = None,
                                    subfolder: _types.OptionalString = None,
                                    dtype: _enums.DataType = _enums.DataType.AUTO,
                                    unet_uri: _types.OptionalUri = None,
                                    vae_uri: _types.OptionalUri = None,
                                    lora_uris: _types.OptionalUris = None,
                                    textual_inversion_uris: _types.OptionalUris = None,
                                    control_net_uris: _types.OptionalUris = None,
                                    scheduler: _types.OptionalString = None,
                                    safety_checker: bool = False,
                                    auth_token: _types.OptionalString = None,
                                    device: str = 'cuda',
                                    extra_modules: typing.Optional[dict[str, typing.Any]] = None,
                                    model_cpu_offload: bool = False,
                                    sequential_cpu_offload: bool = False,
                                    local_files_only: bool = False) -> TorchPipelineCreationResult:
    """
    Create a :py:class:`diffusers.DiffusionPipeline` in dgenerates in memory cacheing system.


    :param pipeline_type: :py:class:`dgenerate.pipelinewrapper.PipelineType` enum value
    :param model_type:  :py:class:`dgenerate.pipelinewrapper.ModelType` enum value
    :param model_path: huggingface slug, huggingface blob link, path to folder on disk, path to file on disk
    :param revision: huggingface repo revision (branch)
    :param variant: model weights name variant, for example 'fp16'
    :param subfolder: huggingface repo subfolder if applicable
    :param dtype: Optional :py:class:`dgenerate.pipelinewrapper.DataType` enum value
    :param unet_uri: Optional ``--unet`` URI string for specifying a specific UNet
    :param vae_uri: Optional ``--vae`` URI string for specifying a specific VAE
    :param lora_uris: Optional ``--loras`` URI strings for specifying LoRA weights
    :param textual_inversion_uris: Optional ``--textual-inversions`` URI strings for specifying Textual Inversion weights
    :param control_net_uris: Optional ``--control-nets`` URI strings for specifying ControlNet models
    :param scheduler: Optional scheduler (sampler) class name, unqualified, or "help" to print supported values
        to STDOUT and raise :py:exc:`dgenerate.pipelinewrapper.SchedulerHelpException`
    :param safety_checker: Safety checker enabled? default is false
    :param auth_token: Optional huggingface API token for accessing repositories that are restricted to your account
    :param device: Optional ``--device`` string, defaults to "cuda"
    :param extra_modules: Extra module arguments to pass directly into
        :py:meth:`diffusers.DiffusionPipeline.from_single_file` or :py:meth:`diffusers.DiffusionPipeline.from_pretrained`
    :param model_cpu_offload: This pipeline has model_cpu_offloading enabled?
    :param sequential_cpu_offload: This pipeline has sequential_cpu_offloading enabled?
    :param local_files_only: Only look in the huggingface cache and do not connect to download models?

    :raises ModelNotFoundError:
    :raises InvalidModelUriError:
    :raises InvalidSchedulerNameError:
    :raises UnsupportedPipelineConfigError:

    :return: :py:class:`.TorchPipelineCreationResult`
    """
    __locals = locals()
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
                 pipeline_type: _enums.PipelineType,
                 model_path: str,
                 model_type: _enums.ModelType = _enums.ModelType.TORCH,
                 revision: _types.OptionalString = None,
                 variant: _types.OptionalString = None,
                 subfolder: _types.OptionalString = None,
                 dtype: _enums.DataType = _enums.DataType.AUTO,
                 unet_uri: _types.OptionalUri = None,
                 vae_uri: _types.OptionalUri = None,
                 lora_uris: _types.OptionalUris = None,
                 textual_inversion_uris: _types.OptionalUris = None,
                 control_net_uris: _types.OptionalUris = None,
                 scheduler: _types.OptionalString = None,
                 safety_checker: bool = False,
                 auth_token: _types.OptionalString = None,
                 device: str = 'cuda',
                 extra_modules: typing.Optional[dict[str, typing.Any]] = None,
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


def _torch_args_hasher(args):
    custom_hashes = {
        'unet_uri': _cache.uri_hash_with_parser(_uris.TorchUNetUri.parse),
        'vae_uri': _cache.uri_hash_with_parser(_uris.TorchVAEUri.parse),
        'lora_uris': _cache.uri_list_hash_with_parser(_uris.LoRAUri.parse),
        'textual_inversion_uris': _cache.uri_list_hash_with_parser(_uris.TextualInversionUri.parse),
        'control_net_uris': _cache.uri_list_hash_with_parser(_uris.TorchControlNetUri.parse)}
    return _d_memoize.args_cache_key(args, custom_hashes=custom_hashes)


def _torch_on_hit(key, hit):
    _d_memoize.simple_cache_hit_debug("Torch Pipeline", key, hit.pipeline)


def _torch_on_create(key, new):
    _d_memoize.simple_cache_miss_debug('Torch Pipeline', key, new.pipeline)


@_memoize(_cache._TORCH_PIPELINE_CACHE,
          exceptions={'local_files_only'},
          hasher=_torch_args_hasher,
          on_hit=_torch_on_hit,
          on_create=_torch_on_create)
def _create_torch_diffusion_pipeline(pipeline_type: _enums.PipelineType,
                                     model_path: str,
                                     model_type: _enums.ModelType = _enums.ModelType.TORCH,
                                     revision: _types.OptionalString = None,
                                     variant: _types.OptionalString = None,
                                     subfolder: _types.OptionalString = None,
                                     dtype: _enums.DataType = _enums.DataType.AUTO,
                                     unet_uri: _types.OptionalUri = None,
                                     vae_uri: _types.OptionalUri = None,
                                     lora_uris: _types.OptionalUris = None,
                                     textual_inversion_uris: _types.OptionalUris = None,
                                     control_net_uris: _types.OptionalUris = None,
                                     scheduler: _types.OptionalString = None,
                                     safety_checker: bool = False,
                                     auth_token: _types.OptionalString = None,
                                     device: str = 'cuda',
                                     extra_modules: typing.Optional[dict[str, typing.Any]] = None,
                                     model_cpu_offload: bool = False,
                                     sequential_cpu_offload: bool = False,
                                     local_files_only: bool = False) -> TorchPipelineCreationResult:
    if not _enums.model_type_is_torch(model_type):
        raise ValueError('model_type must be a TORCH ModelType enum value.')
    # Pipeline class selection

    if _enums.model_type_is_floyd(model_type):
        if control_net_uris:
            raise UnsupportedPipelineConfigError(
                'Deep Floyd --model-type values are not compatible with --control-nets.')
        if textual_inversion_uris:
            raise UnsupportedPipelineConfigError(
                'Deep Floyd --model-type values are not compatible with --textual-inversions.')

    if _enums.model_type_is_upscaler(model_type):
        if pipeline_type != _enums.PipelineType.IMG2IMG and not scheduler_is_help(scheduler):
            raise UnsupportedPipelineConfigError(
                'Upscaler models only work with img2img generation, IE: --image-seeds (with no image masks).')

        if model_type == _enums.ModelType.TORCH_UPSCALER_X2:
            if lora_uris or textual_inversion_uris:
                raise UnsupportedPipelineConfigError(
                    '--model-type torch-upscaler-x2 is not compatible with --loras or --textual-inversions.')

        pipeline_class = (diffusers.StableDiffusionUpscalePipeline if model_type == _enums.ModelType.TORCH_UPSCALER_X4
                          else diffusers.StableDiffusionLatentUpscalePipeline)
    else:
        sdxl = _enums.model_type_is_sdxl(model_type)
        pix2pix = _enums.model_type_is_pix2pix(model_type)

        if pipeline_type == _enums.PipelineType.TXT2IMG:
            if pix2pix:
                raise UnsupportedPipelineConfigError(
                    'pix2pix models only work in img2img mode and cannot work without --image-seeds.')

            if model_type == _enums.ModelType.TORCH_IF:
                pipeline_class = diffusers.IFPipeline
            elif model_type == _enums.ModelType.TORCH_IFS:
                raise UnsupportedPipelineConfigError(
                    'Deep Floyd IF super resolution (IFS) only works in img2img mode and cannot work without --image-seeds.')
            elif control_net_uris:
                pipeline_class = diffusers.StableDiffusionXLControlNetPipeline if sdxl else diffusers.StableDiffusionControlNetPipeline
            else:
                pipeline_class = diffusers.StableDiffusionXLPipeline if sdxl else diffusers.StableDiffusionPipeline
        elif pipeline_type == _enums.PipelineType.IMG2IMG:
            if pix2pix:
                if control_net_uris:
                    raise UnsupportedPipelineConfigError(
                        'pix2pix models are not compatible with --control-nets.')

                pipeline_class = diffusers.StableDiffusionXLInstructPix2PixPipeline if sdxl else diffusers.StableDiffusionInstructPix2PixPipeline
            elif model_type == _enums.ModelType.TORCH_IF:
                pipeline_class = diffusers.IFImg2ImgPipeline
            elif model_type == _enums.ModelType.TORCH_IFS:
                pipeline_class = diffusers.IFSuperResolutionPipeline
            elif model_type == _enums.ModelType.TORCH_IFS_IMG2IMG:
                pipeline_class = diffusers.IFImg2ImgSuperResolutionPipeline
            elif control_net_uris:
                if sdxl:
                    pipeline_class = diffusers.StableDiffusionXLControlNetImg2ImgPipeline
                else:
                    pipeline_class = diffusers.StableDiffusionControlNetImg2ImgPipeline
            else:
                pipeline_class = diffusers.StableDiffusionXLImg2ImgPipeline if sdxl else diffusers.StableDiffusionImg2ImgPipeline
        elif pipeline_type == _enums.PipelineType.INPAINT:
            if pix2pix:
                raise UnsupportedPipelineConfigError(
                    'pix2pix models only work in img2img mode and cannot work in inpaint mode (with a mask).')
            if model_type == _enums.ModelType.TORCH_IF:
                pipeline_class = diffusers.IFInpaintingPipeline
            elif model_type == _enums.ModelType.TORCH_IFS:
                pipeline_class = diffusers.IFInpaintingSuperResolutionPipeline
            elif control_net_uris:
                if sdxl:
                    pipeline_class = diffusers.StableDiffusionXLControlNetInpaintPipeline
                else:
                    pipeline_class = diffusers.StableDiffusionControlNetInpaintPipeline
            else:
                pipeline_class = diffusers.StableDiffusionXLInpaintPipeline if sdxl else diffusers.StableDiffusionInpaintPipeline
        else:
            # Should be impossible
            raise UnsupportedPipelineConfigError('Pipeline type not implemented.')

    unet_override = extra_modules and 'unet' in extra_modules
    vae_override = extra_modules and 'vae' in extra_modules
    controlnet_override = extra_modules and 'controlnet' in extra_modules
    safety_checker_override = extra_modules and 'safety_checker' in extra_modules
    scheduler_override = extra_modules and 'scheduler' in extra_modules

    estimated_memory_usage = estimate_pipeline_memory_use(
        pipeline_type=pipeline_type,
        model_type=model_type,
        model_path=model_path,
        revision=revision,
        variant=variant,
        subfolder=subfolder,
        unet_uri=unet_uri if not unet_override else None,
        vae_uri=vae_uri if not vae_override else None,
        lora_uris=lora_uris,
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

    # Block invalid Textual Inversion and LoRA usage

    if textual_inversion_uris:
        if model_type == _enums.ModelType.TORCH_UPSCALER_X2:
            raise UnsupportedPipelineConfigError(
                '--model-type torch-upscaler-x2 cannot be used with textual inversion models.')

    if lora_uris:
        if _enums.model_type_is_upscaler(model_type):
            raise UnsupportedPipelineConfigError(
                'LoRA models cannot be used with upscaler models.')

    # ControlNet and VAE loading

    # Used during pipeline load
    creation_kwargs = {}

    torch_dtype = _enums.get_torch_dtype(dtype)

    parsed_control_net_uris = []
    parsed_unet_uri = None
    parsed_vae_uri = None

    if not scheduler_is_help(scheduler):
        # prevent waiting on UNet/VAE load just to get the scheduler
        # help message for the main model

        if vae_uri is not None and not vae_override:
            parsed_vae_uri = _uris.TorchVAEUri.parse(vae_uri)

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
            parsed_unet_uri = _uris.TorchUNetUri.parse(unet_uri)

            creation_kwargs['unet'] = \
                parsed_unet_uri.load(
                    variant_fallback=variant,
                    dtype_fallback=dtype,
                    use_auth_token=auth_token,
                    local_files_only=local_files_only,
                    sequential_cpu_offload_member=sequential_cpu_offload,
                    model_cpu_offload_member=model_cpu_offload)

            _messages.debug_log(lambda:
                                f'Added Torch UNet: "{unet_uri}" to pipeline: "{pipeline_class.__name__}"')

    if control_net_uris and not controlnet_override:
        if _enums.model_type_is_pix2pix(model_type):
            raise UnsupportedPipelineConfigError(
                'Using ControlNets with pix2pix models is not supported.'
            )

        control_nets = None

        for control_net_uri in control_net_uris:
            parsed_control_net_uri = _uris.TorchControlNetUri.parse(control_net_uri)

            parsed_control_net_uris.append(parsed_control_net_uri)

            new_net = parsed_control_net_uri.load(use_auth_token=auth_token,
                                                  dtype_fallback=dtype,
                                                  local_files_only=local_files_only,
                                                  sequential_cpu_offload_member=sequential_cpu_offload,
                                                  model_cpu_offload_member=model_cpu_offload)

            _messages.debug_log(lambda:
                                f'Added Torch ControlNet: "{control_net_uri}" '
                                f'to pipeline: "{pipeline_class.__name__}"')

            if control_nets is not None:
                if not isinstance(control_nets, list):
                    control_nets = [control_nets, new_net]
                else:
                    control_nets.append(new_net)
            else:
                control_nets = new_net

        creation_kwargs['controlnet'] = control_nets

    if _enums.model_type_is_floyd(model_type):
        creation_kwargs['watermarker'] = None

    if not safety_checker and not _enums.model_type_is_sdxl(model_type) and not safety_checker_override:
        creation_kwargs['safety_checker'] = None

    if extra_modules is not None:
        creation_kwargs.update(extra_modules)

    # Create Pipeline

    if _hfutil.is_single_file_model_load(model_path):
        if subfolder is not None:
            raise UnsupportedPipelineConfigError('Single file model loads do not support the subfolder option.')
        pipeline = pipeline_class.from_single_file(model_path,
                                                   revision=revision,
                                                   variant=variant,
                                                   torch_dtype=torch_dtype,
                                                   use_safe_tensors=model_path.endswith('.safetensors'),
                                                   local_files_only=local_files_only,
                                                   **creation_kwargs)
    else:
        pipeline = pipeline_class.from_pretrained(model_path,
                                                  revision=revision,
                                                  variant=variant,
                                                  torch_dtype=torch_dtype,
                                                  subfolder=subfolder,
                                                  use_auth_token=auth_token,
                                                  local_files_only=local_files_only,
                                                  **creation_kwargs)

    # Select Scheduler

    if not scheduler_override:
        load_scheduler(pipeline=pipeline,
                       model_path=model_path,
                       scheduler_name=scheduler)

    # Textual Inversions and LoRAs

    parsed_textual_inversion_uris = []
    parsed_lora_uris = []

    if textual_inversion_uris:
        for inversion_uri in textual_inversion_uris:
            parsed = _uris.TextualInversionUri.parse(inversion_uri)
            parsed_textual_inversion_uris.append(parsed)
            parsed.load_on_pipeline(pipeline,
                                    use_auth_token=auth_token,
                                    local_files_only=local_files_only)

    if lora_uris:
        for lora_uri in lora_uris:
            parsed = _uris.LoRAUri.parse(lora_uri)
            parsed_lora_uris.append(parsed)
            parsed.load_on_pipeline(pipeline,
                                    use_auth_token=auth_token,
                                    local_files_only=local_files_only)

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
        parsed_vae_uri=parsed_vae_uri,
        parsed_lora_uris=parsed_lora_uris,
        parsed_textual_inversion_uris=parsed_textual_inversion_uris,
        parsed_control_net_uris=parsed_control_net_uris
    )


class FlaxPipelineCreationResult(PipelineCreationResult):
    @property
    def pipeline(self) -> diffusers.FlaxDiffusionPipeline:
        """
        A created subclass of :py:class:`diffusers.FlaxDiffusionPipeline`
        """
        return super().pipeline

    flax_params: dict[str, typing.Any]
    """
    Flax specific Pipeline params object
    """

    parsed_unet_uri: typing.Optional[_uris.FlaxUNetUri]
    """
    Parsed UNet URI if one was present
    """

    flax_unet_params: typing.Optional[dict[str, typing.Any]]
    """
    Flax specific UNet params object
    """

    parsed_vae_uri: typing.Optional[_uris.FlaxVAEUri]
    """
    Parsed VAE URI if one was present
    """

    flax_vae_params: typing.Optional[dict[str, typing.Any]]
    """
    Flax specific VAE params object
    """

    parsed_control_net_uris: collections.abc.Sequence[_uris.FlaxControlNetUri]
    """
    Parsed ControlNet URIs if any were present
    """

    flax_control_net_params: typing.Optional[dict[str, typing.Any]]
    """
    Flax specific ControlNet params object
    """

    def __init__(self,
                 pipeline: diffusers.FlaxDiffusionPipeline,
                 flax_params: dict[str, typing.Any],
                 parsed_unet_uri: typing.Optional[_uris.FlaxUNetUri],
                 flax_unet_params: typing.Optional[dict[str, typing.Any]],
                 parsed_vae_uri: typing.Optional[_uris.FlaxVAEUri],
                 flax_vae_params: typing.Optional[dict[str, typing.Any]],
                 parsed_control_net_uris: collections.abc.Sequence[_uris.FlaxControlNetUri],
                 flax_control_net_params: typing.Optional[dict[str, typing.Any]]):
        super().__init__(pipeline)

        self.flax_params = flax_params
        self.parsed_control_net_uris = parsed_control_net_uris
        self.parsed_unet_uri = parsed_unet_uri
        self.flax_unet_params = flax_unet_params
        self.parsed_vae_uri = parsed_vae_uri
        self.flax_vae_params = flax_vae_params
        self.flax_control_net_params = flax_control_net_params

    def call(self, *args, **kwargs) -> diffusers.utils.BaseOutput:
        """
        Call **pipeline**, see: :py:func:`.call_pipeline`

        :param args: forward args to pipeline
        :param kwargs: forward kwargs to pipeline
        :return: A subclass of :py:class:`diffusers.utils.BaseOutput`
        """
        return call_pipeline(self.pipeline, None, *args, **kwargs)


def create_flax_diffusion_pipeline(pipeline_type: _enums.PipelineType,
                                   model_path: str,
                                   model_type: _enums.ModelType = _enums.ModelType.FLAX,
                                   revision: _types.OptionalString = None,
                                   subfolder: _types.OptionalString = None,
                                   dtype: _enums.DataType = _enums.DataType.AUTO,
                                   unet_uri: _types.OptionalUri = None,
                                   vae_uri: _types.OptionalUri = None,
                                   control_net_uris: _types.OptionalUris = None,
                                   scheduler: _types.OptionalString = None,
                                   safety_checker: bool = False,
                                   auth_token: _types.OptionalString = None,
                                   extra_modules: typing.Optional[dict[str, typing.Any]] = None,
                                   local_files_only: bool = False) -> FlaxPipelineCreationResult:
    """
    Create a :py:class:`diffusers.FlaxDiffusionPipeline` in dgenerates in memory cacheing system.


    :param pipeline_type: :py:class:`dgenerate.pipelinewrapper.PipelineType` enum value
    :param model_path: huggingface slug, huggingface blob link, path to folder on disk, path to file on disk
    :param model_type: Currently only accepts :py:attr:`dgenerate.pipelinewrapper.ModelType.FLAX`
    :param revision: huggingface repo revision (branch)
    :param subfolder: huggingface repo subfolder if applicable
    :param dtype: Optional :py:class:`dgenerate.pipelinewrapper.DataType` enum value
    :param unet_uri: Optional Flax specific ``--unet`` URI string for specifying a specific UNet
    :param vae_uri: Optional Flax specific ``--vae`` URI string for specifying a specific VAE
    :param control_net_uris: Optional ``--control-nets`` URI strings for specifying ControlNet models
    :param scheduler: Optional scheduler (sampler) class name, unqualified, or "help" to print supported values
        to STDOUT and raise :py:exc:`dgenerate.pipelinewrapper.SchedulerHelpException`
    :param safety_checker: Safety checker enabled? default is false
    :param auth_token: Optional huggingface API token for accessing repositories that are restricted to your account
    :param extra_modules: Extra module arguments to pass directly into :py:meth:`diffusers.FlaxDiffusionPipeline.from_pretrained`
    :param local_files_only: Only look in the huggingface cache and do not connect to download models?

    :raises ModelNotFoundError:
    :raises InvalidModelUriError:
    :raises InvalidSchedulerNameError:
    :raises UnsupportedPipelineConfigError:

    :return: :py:class:`.FlaxPipelineCreationResult`
    """
    __locals = locals()
    try:
        return _create_flax_diffusion_pipeline(**__locals)
    except (huggingface_hub.utils.HFValidationError,
            huggingface_hub.utils.HfHubHTTPError) as e:
        raise _hfutil.ModelNotFoundError(e)


class FlaxPipelineFactory:
    """
    Turns :py:func:`.create_flax_diffusion_pipeline` into a factory
    that can recreate the same Flax pipeline over again, possibly from cache.
    """

    def __init__(self, pipeline_type: _enums.PipelineType,
                 model_path: str,
                 model_type: _enums.ModelType = _enums.ModelType.FLAX,
                 revision: _types.OptionalString = None,
                 subfolder: _types.OptionalString = None,
                 dtype: _enums.DataType = _enums.DataType.AUTO,
                 unet_uri: _types.OptionalUri = None,
                 vae_uri: _types.OptionalUri = None,
                 control_net_uris: _types.OptionalUris = None,
                 scheduler: _types.OptionalString = None,
                 safety_checker: bool = False,
                 auth_token: _types.OptionalString = None,
                 extra_modules: typing.Optional[dict[str, typing.Any]] = None,
                 local_files_only: bool = False):
        self._args = {k: v for k, v in _types.partial_deep_copy_container(locals()).items() if k not in {'self'}}

    def __call__(self) -> FlaxPipelineCreationResult:
        """
        :raises ModelNotFoundError:
        :raises InvalidModelUriError:
        :raises InvalidSchedulerNameError:
        :raises UnsupportedPipelineConfigError:

        :return: :py:class:`.FlaxPipelineCreationResult`
        """
        return create_flax_diffusion_pipeline(**self._args)


def _flax_args_hasher(args):
    custom_hashes = {'unet_uri': _cache.uri_hash_with_parser(_uris.FlaxUNetUri.parse),
                     'vae_uri': _cache.uri_hash_with_parser(_uris.FlaxVAEUri.parse),
                     'control_net_uris': _cache.uri_list_hash_with_parser(_uris.FlaxControlNetUri.parse)}
    return _d_memoize.args_cache_key(args, custom_hashes=custom_hashes)


def _flax_on_hit(key, hit):
    _d_memoize.simple_cache_hit_debug("Flax Pipeline", key, hit.pipeline)


def _flax_on_create(key, new):
    _d_memoize.simple_cache_miss_debug('Flax Pipeline', key, new.pipeline)


@_memoize(_cache._FLAX_PIPELINE_CACHE,
          exceptions={'local_files_only'},
          hasher=_flax_args_hasher,
          on_hit=_flax_on_hit,
          on_create=_flax_on_create)
def _create_flax_diffusion_pipeline(pipeline_type: _enums.PipelineType,
                                    model_path: str,
                                    model_type: _enums.ModelType = _enums.ModelType.FLAX,
                                    revision: _types.OptionalString = None,
                                    subfolder: _types.OptionalString = None,
                                    dtype: _enums.DataType = _enums.DataType.AUTO,
                                    unet_uri: _types.OptionalUri = None,
                                    vae_uri: _types.OptionalUri = None,
                                    control_net_uris: _types.OptionalUris = None,
                                    scheduler: _types.OptionalString = None,
                                    safety_checker: bool = False,
                                    auth_token: _types.OptionalString = None,
                                    extra_modules: typing.Optional[dict[str, typing.Any]] = None,
                                    local_files_only: bool = False) -> FlaxPipelineCreationResult:
    if not _enums.model_type_is_flax(model_type):
        raise ValueError('model_type must be a FLAX ModelType enum value.')

    has_control_nets = False
    if control_net_uris:
        if len(control_net_uris) > 1:
            raise UnsupportedPipelineConfigError('Flax does not support multiple --control-nets.')
        if len(control_net_uris) == 1:
            has_control_nets = True

    if pipeline_type == _enums.PipelineType.TXT2IMG:
        if has_control_nets:
            pipeline_class = diffusers.FlaxStableDiffusionControlNetPipeline
        else:
            pipeline_class = diffusers.FlaxStableDiffusionPipeline
    elif pipeline_type == _enums.PipelineType.IMG2IMG:
        if has_control_nets:
            raise UnsupportedPipelineConfigError('Flax does not support img2img mode with --control-nets.')
        pipeline_class = diffusers.FlaxStableDiffusionImg2ImgPipeline
    elif pipeline_type == _enums.PipelineType.INPAINT:
        if has_control_nets:
            raise UnsupportedPipelineConfigError('Flax does not support inpaint mode with --control-nets.')
        pipeline_class = diffusers.FlaxStableDiffusionInpaintPipeline
    else:
        raise UnsupportedPipelineConfigError('Pipeline type not implemented.')

    unet_override = extra_modules and 'unet' in extra_modules
    vae_override = extra_modules and 'vae' in extra_modules
    controlnet_override = extra_modules and 'controlnet' in extra_modules
    safety_checker_override = extra_modules and 'safety_checker' in extra_modules
    scheduler_override = extra_modules and 'scheduler' in extra_modules
    feature_extractor_override = extra_modules and 'feature_extractor' in extra_modules

    estimated_memory_usage = estimate_pipeline_memory_use(
        pipeline_type=pipeline_type,
        model_type=model_type,
        model_path=model_path,
        revision=revision,
        subfolder=subfolder,
        unet_uri=unet_uri if not unet_override else None,
        vae_uri=vae_uri if not vae_override else None,
        safety_checker=safety_checker and not safety_checker_override,
        auth_token=auth_token,
        extra_args=extra_modules,
        local_files_only=local_files_only
    )

    _messages.debug_log(
        f'Creating Flax Pipeline: "{pipeline_class.__name__}", '
        f'Estimated CPU Side Memory Use: {_memory.bytes_best_human_unit(estimated_memory_usage)}')

    _cache.enforce_pipeline_cache_constraints(
        new_pipeline_size=estimated_memory_usage)

    creation_kwargs = {}
    unet_params = None
    vae_params = None
    control_net_params = None

    flax_dtype = _enums.get_flax_dtype(dtype)

    parsed_control_net_uris = []
    parsed_flax_vae_uri = None
    parsed_flax_unet_uri = None

    if not scheduler_is_help(scheduler):
        # prevent waiting on UNet/VAE load just get the scheduler
        # help message for the main model

        if vae_uri is not None and not vae_override:
            parsed_flax_vae_uri = _uris.FlaxVAEUri.parse(vae_uri)

            creation_kwargs['vae'], vae_params = parsed_flax_vae_uri.load(
                dtype_fallback=dtype,
                use_auth_token=auth_token,
                local_files_only=local_files_only)
            _messages.debug_log(lambda:
                                f'Added Flax VAE: "{vae_uri}" to pipeline: "{pipeline_class.__name__}"')

        if unet_uri is not None and not unet_override:
            parsed_flax_unet_uri = _uris.FlaxUNetUri.parse(unet_uri)

            creation_kwargs['unet'], unet_params = parsed_flax_unet_uri.load(
                dtype_fallback=dtype,
                use_auth_token=auth_token,
                local_files_only=local_files_only)
            _messages.debug_log(lambda:
                                f'Added Flax UNet: "{unet_uri}" to pipeline: "{pipeline_class.__name__}"')

    if control_net_uris and not controlnet_override:
        control_net_uri = control_net_uris[0]

        parsed_flax_control_net_uri = _uris.FlaxControlNetUri.parse(control_net_uri)

        parsed_control_net_uris.append(parsed_flax_control_net_uri)

        control_net, control_net_params = parsed_flax_control_net_uri \
            .load(use_auth_token=auth_token,
                  dtype_fallback=dtype,
                  local_files_only=local_files_only)

        _messages.debug_log(lambda:
                            f'Added Flax ControlNet: "{control_net_uri}" '
                            f'to pipeline: "{pipeline_class.__name__}"')

        creation_kwargs['controlnet'] = control_net

    if extra_modules is not None:
        creation_kwargs.update(extra_modules)

    if not safety_checker and not safety_checker_override:
        creation_kwargs['safety_checker'] = None

    try:
        pipeline, params = pipeline_class.from_pretrained(model_path,
                                                          revision=revision,
                                                          dtype=flax_dtype,
                                                          subfolder=subfolder,
                                                          use_auth_token=auth_token,
                                                          local_files_only=local_files_only,
                                                          **creation_kwargs)
    except ValueError as e:
        if 'feature_extractor' not in str(e):
            raise e

        # odd diffusers bug

        if not feature_extractor_override:
            creation_kwargs['feature_extractor'] = None

        pipeline, params = pipeline_class.from_pretrained(model_path,
                                                          revision=revision,
                                                          dtype=flax_dtype,
                                                          subfolder=subfolder,
                                                          use_auth_token=auth_token,
                                                          local_files_only=local_files_only,
                                                          **creation_kwargs)
    if unet_params is not None:
        params['unet'] = unet_params

    if vae_params is not None:
        params['vae'] = vae_params

    if control_net_params is not None:
        params['controlnet'] = control_net_params

    if not scheduler_override:
        load_scheduler(pipeline=pipeline,
                       model_path=model_path,
                       scheduler_name=scheduler)

    if not safety_checker and not safety_checker_override:
        pipeline.safety_checker = None

    _cache.pipeline_create_update_cache_info(pipeline=pipeline,
                                             estimated_size=estimated_memory_usage)

    _messages.debug_log(f'Finished Creating Flax Pipeline: "{pipeline_class.__name__}"')

    return FlaxPipelineCreationResult(
        pipeline=pipeline,
        flax_params=params,
        parsed_unet_uri=parsed_flax_unet_uri,
        flax_unet_params=unet_params,
        parsed_vae_uri=parsed_flax_vae_uri,
        flax_vae_params=vae_params,
        parsed_control_net_uris=parsed_control_net_uris,
        flax_control_net_params=control_net_params
    )


__all__ = _types.module_all()
