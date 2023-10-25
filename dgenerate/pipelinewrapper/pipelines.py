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
import typing

import diffusers

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


class InvalidSchedulerName(Exception):
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


def set_torch_safety_checker(pipeline: diffusers.DiffusionPipeline, safety_checker: bool):
    if not safety_checker:
        if hasattr(pipeline, 'safety_checker') and pipeline.safety_checker is not None:
            # If it's already None for some reason you'll get a call
            # to an unassigned feature_extractor by assigning it a value

            # The attribute will not exist for SDXL pipelines currently

            pipeline.safety_checker = _disabled_safety_checker


def set_vae_slicing_tiling(pipeline: typing.Union[diffusers.DiffusionPipeline,
                                                  diffusers.FlaxDiffusionPipeline],
                           vae_tiling: bool,
                           vae_slicing: bool):
    has_vae = hasattr(pipeline, 'vae') and pipeline.vae is not None
    pipeline_class = pipeline.__class__

    if vae_tiling:
        if has_vae:
            if hasattr(pipeline.vae, 'enable_tiling'):
                _messages.debug_log(f'Enabling VAE tiling on Pipeline: "{pipeline_class.__name__}",',
                                    f'VAE: "{pipeline.vae.__class__.__name__}"')
                pipeline.vae.enable_tiling()
            else:
                raise NotImplementedError(
                    '--vae-tiling not supported as loaded VAE does not support it.'
                )
        else:
            raise NotImplementedError(
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
                raise NotImplementedError(
                    '--vae-slicing not supported as loaded VAE does not support it.'
                )
        else:
            raise NotImplementedError(
                '--vae-slicing not supported as no VAE is present for the specified model.')
    elif has_vae:
        if hasattr(pipeline.vae, 'disable_slicing'):
            _messages.debug_log(f'Disabling VAE slicing on Pipeline: "{pipeline_class.__name__}",',
                                f'VAE: "{pipeline.vae.__class__.__name__}"')
            pipeline.vae.disable_slicing()


def scheduler_is_help(name):
    if name is None:
        return False
    return name.strip().lower() == 'help'


def load_scheduler(pipeline: typing.Union[diffusers.DiffusionPipeline, diffusers.FlaxDiffusionPipeline],
                   model_path: str, scheduler_name=None):
    if scheduler_name is None:
        return

    compatibles = pipeline.scheduler.compatibles

    if isinstance(pipeline, diffusers.StableDiffusionLatentUpscalePipeline):
        # Seems to only work with this scheduler
        compatibles = [c for c in compatibles if c.__name__ == 'EulerDiscreteScheduler']

    if scheduler_is_help(scheduler_name):
        help_string = _textprocessing.underline(f'Compatible schedulers for "{model_path}" are:') + '\n\n'
        help_string += '\n'.join((" " * 4) + _textprocessing.quote(i.__name__) for i in compatibles) + '\n'
        _messages.log(help_string, underline=True)
        raise SchedulerHelpException(help_string)

    for i in compatibles:
        if i.__name__.endswith(scheduler_name):
            pipeline.scheduler = i.from_config(pipeline.scheduler.config)
            return

    raise InvalidSchedulerName(
        f'Scheduler named "{scheduler_name}" is not a valid compatible scheduler, '
        f'options are:\n\n{chr(10).join(sorted(" " * 4 + _textprocessing.quote(i.__name__.split(".")[-1]) for i in compatibles))}')


def _disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False] * num_images
    else:
        return images, False


def _estimate_pipeline_memory_use(
        model_path,
        revision,
        variant=None,
        subfolder=None,
        vae_uri=None,
        lora_uris=None,
        textual_inversion_uris=None,
        safety_checker=False,
        auth_token=None,
        extra_args=None,
        local_files_only=False,
        flax=False):
    usage = _hfutil.estimate_model_memory_use(
        repo_id=model_path,
        revision=revision,
        variant=variant,
        subfolder=subfolder,
        include_vae=not vae_uri,
        safety_checker=safety_checker,
        include_text_encoder=not extra_args or 'text_encoder' not in extra_args,
        include_text_encoder_2=not extra_args or 'text_encoder_2' not in extra_args,
        use_auth_token=auth_token,
        local_files_only=local_files_only,
        flax=flax
    )

    if lora_uris:
        if isinstance(lora_uris, str):
            lora_uris = [lora_uris]

        for lora_uri in lora_uris:
            parsed = _uris.parse_lora_uri(lora_uri)

            usage += _hfutil.estimate_model_memory_use(
                repo_id=parsed.model,
                revision=parsed.revision,
                subfolder=parsed.subfolder,
                weight_name=parsed.weight_name,
                use_auth_token=auth_token,
                local_files_only=local_files_only,
                flax=flax
            )

    if textual_inversion_uris:
        if isinstance(textual_inversion_uris, str):
            textual_inversion_uris = [textual_inversion_uris]

        for textual_inversion_uri in textual_inversion_uris:
            parsed = _uris.parse_textual_inversion_uri(textual_inversion_uri)

            usage += _hfutil.estimate_model_memory_use(
                repo_id=parsed.model,
                revision=parsed.revision,
                subfolder=parsed.subfolder,
                weight_name=parsed.weight_name,
                use_auth_token=auth_token,
                local_files_only=local_files_only,
                flax=flax
            )

    return usage


class TorchPipelineCreationResult:
    pipeline: diffusers.DiffusionPipeline
    parsed_vae_uri: typing.Optional[_uris.TorchVAEUri]
    parsed_lora_uris: typing.List[_uris.LoRAUri]
    parsed_textual_inversion_uris: typing.List[_uris.TextualInversionUri]
    parsed_control_net_uris: typing.List[_uris.TorchControlNetUri]

    def __init__(self,
                 pipeline: diffusers.DiffusionPipeline,
                 parsed_vae_uri: typing.Optional[_uris.TorchVAEUri],
                 parsed_lora_uris: typing.List[_uris.LoRAUri],
                 parsed_textual_inversion_uris: typing.List[_uris.TextualInversionUri],
                 parsed_control_net_uris: typing.List[_uris.TorchControlNetUri]):
        self.pipeline = pipeline
        self.parsed_vae_uri = parsed_vae_uri
        self.parsed_lora_uris = parsed_lora_uris
        self.parsed_textual_inversion_uris = parsed_textual_inversion_uris
        self.parsed_control_net_uris = parsed_control_net_uris

    def call(self, *args, **kwargs) -> diffusers.utils.BaseOutput:
        return self.pipeline(*args, **kwargs)


def create_torch_diffusion_pipeline(pipeline_type: _enums.PipelineTypes,
                                    model_type: _enums.ModelTypes,
                                    model_path: str,
                                    revision: _types.OptionalString = None,
                                    variant: _types.OptionalString = None,
                                    subfolder: _types.OptionalString = None,
                                    dtype: _enums.DataTypes = _enums.DataTypes.AUTO,
                                    vae_uri: _types.OptionalUri = None,
                                    lora_uris: _types.OptionalUris = None,
                                    textual_inversion_uris: _types.OptionalUris = None,
                                    control_net_uris: _types.OptionalUris = None,
                                    scheduler: _types.OptionalString = None,
                                    safety_checker: bool = False,
                                    auth_token: _types.OptionalString = None,
                                    device: str = 'cuda',
                                    extra_args: typing.Optional[typing.Dict[str, typing.Any]] = None,
                                    model_cpu_offload: bool = False,
                                    sequential_cpu_offload: bool = False,
                                    local_files_only: bool = False) -> TorchPipelineCreationResult:
    return _create_torch_diffusion_pipeline(**locals())


@_memoize(_cache._TORCH_PIPELINE_CACHE,
          exceptions={'local_files_only'},
          hasher=lambda args: _d_memoize.args_cache_key(args,
                                                        {'vae_uri': _cache.uri_hash_with_parser(
                                                            _uris.parse_torch_vae_uri),
                                                            'lora_uris':
                                                                _cache.uri_list_hash_with_parser(_uris.parse_lora_uri),
                                                            'textual_inversion_uris':
                                                                _cache.uri_list_hash_with_parser(
                                                                    _uris.parse_textual_inversion_uri),
                                                            'control_net_uris':
                                                                _cache.uri_list_hash_with_parser(
                                                                    _uris.parse_torch_control_net_uri)}),
          on_hit=lambda key, hit: _cache.simple_cache_hit_debug("Torch Pipeline", key, hit.pipeline),
          on_create=lambda key, new: _cache.simple_cache_miss_debug('Torch Pipeline', key, new.pipeline))
def _create_torch_diffusion_pipeline(pipeline_type: _enums.PipelineTypes,
                                     model_type: _enums.ModelTypes,
                                     model_path: str,
                                     revision: _types.OptionalString = None,
                                     variant: _types.OptionalString = None,
                                     subfolder: _types.OptionalString = None,
                                     dtype: _enums.DataTypes = _enums.DataTypes.AUTO,
                                     vae_uri: _types.OptionalUri = None,
                                     lora_uris: _types.OptionalUris = None,
                                     textual_inversion_uris: _types.OptionalUris = None,
                                     control_net_uris: _types.OptionalUris = None,
                                     scheduler: _types.OptionalString = None,
                                     safety_checker: bool = False,
                                     auth_token: _types.OptionalString = None,
                                     device: str = 'cuda',
                                     extra_args: typing.Optional[typing.Dict[str, typing.Any]] = None,
                                     model_cpu_offload: bool = False,
                                     sequential_cpu_offload: bool = False,
                                     local_files_only: bool = False) -> TorchPipelineCreationResult:
    # Pipeline class selection

    if _enums.model_type_is_upscaler(model_type):
        if pipeline_type != _enums.PipelineTypes.IMG2IMG and not scheduler_is_help(scheduler):
            raise NotImplementedError(
                'Upscaler models only work with img2img generation, IE: --image-seeds (with no image masks).')

        if model_type == _enums.ModelTypes.TORCH_UPSCALER_X2:
            if lora_uris or textual_inversion_uris:
                raise NotImplementedError(
                    '--model-type torch-upscaler-x2 is not compatible with --lora or --textual-inversions.')

        pipeline_class = (diffusers.StableDiffusionUpscalePipeline if model_type == _enums.ModelTypes.TORCH_UPSCALER_X4
                          else diffusers.StableDiffusionLatentUpscalePipeline)
    else:
        sdxl = _enums.model_type_is_sdxl(model_type)
        pix2pix = _enums.model_type_is_pix2pix(model_type)

        if pipeline_type == _enums.PipelineTypes.TXT2IMG:
            if pix2pix:
                raise NotImplementedError(
                    'pix2pix models only work in img2img mode and cannot work without --image-seeds.')

            if control_net_uris:
                pipeline_class = diffusers.StableDiffusionXLControlNetPipeline if sdxl else diffusers.StableDiffusionControlNetPipeline
            else:
                pipeline_class = diffusers.StableDiffusionXLPipeline if sdxl else diffusers.StableDiffusionPipeline
        elif pipeline_type == _enums.PipelineTypes.IMG2IMG:

            if pix2pix:
                if control_net_uris:
                    raise NotImplementedError('pix2pix models are not compatible with --control-nets.')

                pipeline_class = diffusers.StableDiffusionXLInstructPix2PixPipeline if sdxl else diffusers.StableDiffusionInstructPix2PixPipeline
            else:
                if control_net_uris:
                    if sdxl:
                        pipeline_class = diffusers.StableDiffusionXLControlNetImg2ImgPipeline
                    else:
                        pipeline_class = diffusers.StableDiffusionControlNetImg2ImgPipeline
                else:
                    pipeline_class = diffusers.StableDiffusionXLImg2ImgPipeline if sdxl else diffusers.StableDiffusionImg2ImgPipeline

        elif pipeline_type == _enums.PipelineTypes.INPAINT:
            if pix2pix:
                raise NotImplementedError(
                    'pix2pix models only work in img2img mode and cannot work in inpaint mode (with a mask).')

            if control_net_uris:
                if sdxl:
                    pipeline_class = diffusers.StableDiffusionXLControlNetInpaintPipeline
                else:
                    pipeline_class = diffusers.StableDiffusionControlNetInpaintPipeline
            else:
                pipeline_class = diffusers.StableDiffusionXLInpaintPipeline if sdxl else diffusers.StableDiffusionInpaintPipeline
        else:
            # Should be impossible
            raise NotImplementedError('Pipeline type not implemented.')

    estimated_memory_usage = _estimate_pipeline_memory_use(
        model_path=model_path,
        revision=revision,
        variant=variant,
        subfolder=subfolder,
        vae_uri=vae_uri,
        lora_uris=lora_uris,
        textual_inversion_uris=textual_inversion_uris,
        safety_checker=True,  # it is always going to get loaded, monkey patched out currently
        auth_token=auth_token,
        extra_args=extra_args,
        local_files_only=local_files_only
    )

    _messages.debug_log(
        f'Creating Torch Pipeline: "{pipeline_class.__name__}", '
        f'Estimated CPU Side Memory Use: {_memory.bytes_best_human_unit(estimated_memory_usage)}')

    _cache.enforce_pipeline_cache_constraints(
        new_pipeline_size=estimated_memory_usage)

    # Block invalid Textual Inversion and LoRA usage

    if textual_inversion_uris:
        if model_type == _enums.ModelTypes.TORCH_UPSCALER_X2:
            raise NotImplementedError(
                'Model type torch-upscaler-x2 cannot be used with textual inversion models.')

        if isinstance(textual_inversion_uris, str):
            textual_inversion_uris = [textual_inversion_uris]

    if lora_uris:
        if not isinstance(lora_uris, str):
            raise NotImplementedError('Using multiple LoRA models is currently not supported.')

        if _enums.model_type_is_upscaler(model_type):
            raise NotImplementedError(
                'LoRA models cannot be used with upscaler models.')

        lora_uris = [lora_uris]

    # ControlNet and VAE loading

    # Used during pipeline load
    creation_kwargs = {}

    torch_dtype = _enums.get_torch_dtype(dtype)

    parsed_control_net_uris = []
    parsed_vae_uri = None

    if scheduler is None or not scheduler_is_help(scheduler):
        # prevent waiting on VAE load just to get the scheduler
        # help message for the main model

        if vae_uri is not None:
            parsed_vae_uri = _uris.parse_torch_vae_uri(vae_uri)

            creation_kwargs['vae'] = \
                parsed_vae_uri.load(
                    dtype_fallback=dtype,
                    use_auth_token=auth_token,
                    local_files_only=local_files_only)

            _messages.debug_log(lambda:
                                f'Added Torch VAE: "{vae_uri}" to pipeline: "{pipeline_class.__name__}"')

    if control_net_uris:
        if _enums.model_type_is_pix2pix(model_type):
            raise NotImplementedError(
                'Using ControlNets with pix2pix models is not supported.'
            )

        control_nets = None

        for control_net_uri in control_net_uris:
            parsed_control_net_uri = _uris.parse_torch_control_net_uri(control_net_uri)

            parsed_control_net_uris.append(parsed_control_net_uri)

            new_net = parsed_control_net_uri.load(use_auth_token=auth_token,
                                                  dtype_fallback=dtype,
                                                  local_files_only=local_files_only)

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

    if extra_args is not None:
        creation_kwargs.update(extra_args)

    # Create Pipeline

    if _hfutil.is_single_file_model_load(model_path):
        if subfolder is not None:
            raise NotImplementedError('Single file model loads do not support the subfolder option.')
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

    load_scheduler(pipeline=pipeline,
                   model_path=model_path,
                   scheduler_name=scheduler)

    # Textual Inversions and LoRAs

    parsed_textual_inversion_uris = []
    parsed_lora_uris = []

    if textual_inversion_uris:
        for inversion_uri in textual_inversion_uris:
            parsed = _uris.parse_textual_inversion_uri(inversion_uri)
            parsed_textual_inversion_uris.append(parsed)
            parsed.load_on_pipeline(pipeline,
                                    use_auth_token=auth_token,
                                    local_files_only=local_files_only)

    if lora_uris:
        for lora_uri in lora_uris:
            parsed = _uris.parse_lora_uri(lora_uri)
            parsed_lora_uris.append(parsed)
            parsed.load_on_pipeline(pipeline,
                                    use_auth_token=auth_token,
                                    local_files_only=local_files_only)

    # Safety Checker

    set_torch_safety_checker(pipeline, safety_checker)

    # Model Offloading

    # Tag the pipeline with our own attributes
    pipeline.DGENERATE_SEQUENTIAL_OFFLOAD = sequential_cpu_offload
    pipeline.DGENERATE_CPU_OFFLOAD = model_cpu_offload

    if sequential_cpu_offload and 'cuda' in device:
        pipeline.enable_sequential_cpu_offload(device=device)
    elif model_cpu_offload and 'cuda' in device:
        pipeline.enable_model_cpu_offload(device=device)

    _cache._PIPELINE_CACHE_SIZE += estimated_memory_usage

    # Tag for internal use

    pipeline.DGENERATE_SIZE_ESTIMATE = estimated_memory_usage

    _messages.debug_log(f'Finished Creating Torch Pipeline: "{pipeline_class.__name__}"')

    return TorchPipelineCreationResult(
        pipeline=pipeline,
        parsed_vae_uri=parsed_vae_uri,
        parsed_lora_uris=parsed_lora_uris,
        parsed_textual_inversion_uris=parsed_textual_inversion_uris,
        parsed_control_net_uris=parsed_control_net_uris
    )


class FlaxPipelineCreationResult:
    pipeline: diffusers.FlaxDiffusionPipeline
    flax_params: typing.Dict[str, typing.Any]
    parsed_vae_uri: typing.Optional[_uris.FlaxVAEUri]
    flax_vae_params: typing.Optional[typing.Dict[str, typing.Any]]
    parsed_control_net_uris: typing.List[_uris.FlaxControlNetUri]
    flax_control_net_params: typing.Optional[typing.Dict[str, typing.Any]]

    def __init__(self,
                 pipeline: diffusers.FlaxDiffusionPipeline,
                 flax_params: typing.Dict[str, typing.Any],
                 parsed_vae_uri: typing.Optional[_uris.FlaxVAEUri],
                 flax_vae_params: typing.Optional[typing.Dict[str, typing.Any]],
                 parsed_control_net_uris: typing.List[_uris.FlaxControlNetUri],
                 flax_control_net_params: typing.Optional[typing.Dict[str, typing.Any]]):
        self.pipeline = pipeline
        self.flax_params = flax_params
        self.parsed_control_net_uris = parsed_control_net_uris
        self.parsed_vae_uri = parsed_vae_uri
        self.flax_vae_params = flax_vae_params
        self.flax_control_net_params = flax_control_net_params

    def call(self, *args, **kwargs) -> diffusers.utils.BaseOutput:
        return self.pipeline(*args, **kwargs)


def create_flax_diffusion_pipeline(pipeline_type: _enums.PipelineTypes,
                                   model_path: str,
                                   revision: _types.OptionalString = None,
                                   subfolder: _types.OptionalString = None,
                                   dtype: _enums.DataTypes = _enums.DataTypes.AUTO,
                                   vae_uri: _types.OptionalString = None,
                                   control_net_uris: _types.OptionalString = None,
                                   scheduler: _types.OptionalString = None,
                                   safety_checker: bool = False,
                                   auth_token: _types.OptionalString = None,
                                   extra_args: typing.Optional[typing.Dict[str, typing.Any]] = None,
                                   local_files_only: bool = False) -> FlaxPipelineCreationResult:
    return _create_flax_diffusion_pipeline(**locals())


@_memoize(_cache._FLAX_PIPELINE_CACHE,
          exceptions={'local_files_only'},
          hasher=lambda args: _d_memoize.args_cache_key(args,
                                                        {'vae_uri': _cache.uri_hash_with_parser(
                                                            _uris.parse_flax_vae_uri),
                                                            'control_net_uris':
                                                                _cache.uri_list_hash_with_parser(
                                                                    _uris.parse_flax_control_net_uri)}),
          on_hit=lambda key, hit: _cache.simple_cache_hit_debug("Flax Pipeline", key, hit.pipeline),
          on_create=lambda key, new: _cache.simple_cache_miss_debug('Flax Pipeline', key, new.pipeline))
def _create_flax_diffusion_pipeline(pipeline_type: _enums.PipelineTypes,
                                    model_path: str,
                                    revision: _types.OptionalString = None,
                                    subfolder: _types.OptionalString = None,
                                    dtype: _enums.DataTypes = _enums.DataTypes.AUTO,
                                    vae_uri: _types.OptionalString = None,
                                    control_net_uris: _types.OptionalString = None,
                                    scheduler: _types.OptionalString = None,
                                    safety_checker: bool = False,
                                    auth_token: _types.OptionalString = None,
                                    extra_args: typing.Optional[typing.Dict[str, typing.Any]] = None,
                                    local_files_only: bool = False) -> FlaxPipelineCreationResult:
    has_control_nets = False
    if control_net_uris:
        if len(control_net_uris) > 1:
            raise NotImplementedError('Flax does not support multiple --control-nets.')
        if len(control_net_uris) == 1:
            has_control_nets = True

    if pipeline_type == _enums.PipelineTypes.TXT2IMG:
        if has_control_nets:
            pipeline_class = diffusers.FlaxStableDiffusionControlNetPipeline
        else:
            pipeline_class = diffusers.FlaxStableDiffusionPipeline
    elif pipeline_type == _enums.PipelineTypes.IMG2IMG:
        if has_control_nets:
            raise NotImplementedError('Flax does not support img2img mode with --control-nets.')
        pipeline_class = diffusers.FlaxStableDiffusionImg2ImgPipeline
    elif pipeline_type == _enums.PipelineTypes.INPAINT:
        if has_control_nets:
            raise NotImplementedError('Flax does not support inpaint mode with --control-nets.')
        pipeline_class = diffusers.FlaxStableDiffusionInpaintPipeline
    else:
        raise NotImplementedError('Pipeline type not implemented.')

    estimated_memory_usage = _estimate_pipeline_memory_use(
        model_path=model_path,
        revision=revision,
        subfolder=subfolder,
        vae_uri=vae_uri,
        safety_checker=True,  # it is always going to get loaded, monkey patched out currently
        auth_token=auth_token,
        extra_args=extra_args,
        local_files_only=local_files_only,
        flax=True
    )

    _messages.debug_log(
        f'Creating Flax Pipeline: "{pipeline_class.__name__}", '
        f'Estimated CPU Side Memory Use: {_memory.bytes_best_human_unit(estimated_memory_usage)}')

    _cache.enforce_pipeline_cache_constraints(
        new_pipeline_size=estimated_memory_usage)

    kwargs = {}
    vae_params = None
    control_net_params = None

    flax_dtype = _enums.get_flax_dtype(dtype)

    parsed_control_net_uris = []
    parsed_flax_vae_uri = None

    if scheduler is None or not scheduler_is_help(scheduler):
        # prevent waiting on VAE load just get the scheduler
        # help message for the main model

        if vae_uri is not None:
            parsed_flax_vae_uri = _uris.parse_flax_vae_uri(vae_uri)

            kwargs['vae'], vae_params = parsed_flax_vae_uri.load(
                dtype_fallback=dtype,
                use_auth_token=auth_token,
                local_files_only=local_files_only)
            _messages.debug_log(lambda:
                                f'Added Flax VAE: "{vae_uri}" to pipeline: "{pipeline_class.__name__}"')

    if control_net_uris:
        control_net_uri = control_net_uris[0]

        parsed_flax_control_net_uri = _uris.parse_flax_control_net_uri(control_net_uri)

        parsed_control_net_uris.append(parsed_flax_control_net_uri)

        control_net, control_net_params = parsed_flax_control_net_uri \
            .load(use_auth_token=auth_token,
                  dtype_fallback=dtype,
                  local_files_only=local_files_only)

        _messages.debug_log(lambda:
                            f'Added Flax ControlNet: "{control_net_uri}" '
                            f'to pipeline: "{pipeline_class.__name__}"')

        kwargs['controlnet'] = control_net

    if extra_args is not None:
        kwargs.update(extra_args)

    pipeline, params = pipeline_class.from_pretrained(model_path,
                                                      revision=revision,
                                                      dtype=flax_dtype,
                                                      subfolder=subfolder,
                                                      use_auth_token=auth_token,
                                                      local_files_only=local_files_only,
                                                      **kwargs)

    if vae_params is not None:
        params['vae'] = vae_params

    if control_net_params is not None:
        params['controlnet'] = control_net_params

    load_scheduler(pipeline=pipeline,
                   model_path=model_path,
                   scheduler_name=scheduler)

    if not safety_checker:
        pipeline.safety_checker = None

    _cache._PIPELINE_CACHE_SIZE += estimated_memory_usage

    # Tag for internal use

    pipeline.DGENERATE_SIZE_ESTIMATE = estimated_memory_usage

    _messages.debug_log(f'Finished Creating Flax Pipeline: "{pipeline_class.__name__}"')

    return FlaxPipelineCreationResult(
        pipeline=pipeline,
        flax_params=params,
        parsed_vae_uri=parsed_flax_vae_uri,
        flax_vae_params=vae_params,
        parsed_control_net_uris=parsed_control_net_uris,
        flax_control_net_params=control_net_params
    )
