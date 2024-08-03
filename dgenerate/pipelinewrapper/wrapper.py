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
import decimal
import inspect
import shlex
import typing

import PIL.Image
import diffusers
import torch

import dgenerate.image as _image
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.cache as _cache
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.prompt as _prompt
import dgenerate.promptweighters as _promptweighters
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.pipelinewrapper.arguments import DiffusionArguments

try:
    import jax
    import jaxlib
    import jax.numpy as jnp
    from flax.jax_utils import replicate as _flax_replicate
    from flax.training.common_utils import shard as _flax_shard
except ImportError:
    jaxlib = None
    jnp = None
    _flax_replicate = None
    _flax_shard = None
    jax = None
    flax = None


class PipelineWrapperResult:
    """
    The result of calling :py:class:`.DiffusionPipelineWrapper`
    """
    images: _types.MutableImages | None

    @property
    def image_count(self):
        """
        The number of images produced.

        :return: int
        """
        if self.images is None:
            return 0

        return len(self.images)

    @property
    def image(self):
        """
        The first image in the batch of requested batch size.

        :return: :py:class:`PIL.Image.Image`
        """
        return self.images[0] if self.images else None

    def image_grid(self, cols_rows: _types.Size):
        """
        Render an image grid from the images in this result.

        :raise ValueError: if no images are present on this object.
            This is impossible if this object was produced by :py:class:`.DiffusionPipelineWrapper`.

        :param cols_rows: columns and rows (WxH) desired as a tuple
        :return: :py:class:`PIL.Image.Image`
        """
        if not self.images:
            raise ValueError('No images present.')

        if len(self.images) == 1:
            return self.images[0]

        cols, rows = cols_rows

        w, h = self.images[0].size
        grid = PIL.Image.new('RGB', size=(cols * w, rows * h))
        for i, img in enumerate(self.images):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

    def __init__(self, images: _types.Images | None):
        self.images = images
        self.dgenerate_opts = list()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.images is not None:
            for i in self.images:
                if i is not None:
                    i.close()
                    self.images = None


class DiffusionPipelineWrapper:
    """
    Monolithic diffusion pipelines wrapper.
    """

    def __str__(self):
        return f'{self.__class__.__name__}({str(_types.get_public_attributes(self))})'

    def __repr__(self):
        return str(self)

    def __init__(self,
                 model_path: _types.Path,
                 model_type: _enums.ModelType | str = _enums.ModelType.TORCH,
                 revision: _types.OptionalName = None,
                 variant: _types.OptionalName = None,
                 subfolder: _types.OptionalName = None,
                 dtype: _enums.DataType | str = _enums.DataType.AUTO,
                 unet_uri: _types.OptionalUri = None,
                 second_unet_uri: _types.OptionalUri = None,
                 vae_uri: _types.OptionalUri = None,
                 vae_tiling: bool = False,
                 vae_slicing: bool = False,
                 lora_uris: _types.OptionalUris = None,
                 ip_adapter_uris: _types.OptionalUris = None,
                 textual_inversion_uris: _types.OptionalUris = None,
                 text_encoder_uris: _types.OptionalUris = None,
                 second_text_encoder_uris: _types.OptionalUris = None,
                 control_net_uris: _types.OptionalUris = None,
                 t2i_adapter_uris: _types.OptionalUris = None,
                 scheduler: _types.OptionalUri = None,
                 sdxl_refiner_uri: _types.OptionalUri = None,
                 sdxl_refiner_scheduler: _types.OptionalUri = None,
                 s_cascade_decoder_uri: _types.OptionalUri = None,
                 s_cascade_decoder_scheduler: _types.OptionalUri = None,
                 device: str = 'cuda',
                 safety_checker: bool = False,
                 auth_token: _types.OptionalString = None,
                 local_files_only: bool = False,
                 model_extra_modules: dict[str, typing.Any] = None,
                 second_model_extra_modules: dict[str, typing.Any] = None,
                 model_cpu_offload: bool = False,
                 model_sequential_offload: bool = False,
                 sdxl_refiner_cpu_offload: bool = False,
                 sdxl_refiner_sequential_offload: bool = False,
                 s_cascade_decoder_cpu_offload: bool = False,
                 s_cascade_decoder_sequential_offload: bool = False,
                 prompt_weighter_uri: _types.OptionalUri = None,
                 prompt_weighter_loader: _promptweighters.PromptWeighterLoader | None = None):
        """
        This is a monolithic wrapper around all supported diffusion pipelines which handles
        txt2img, img2img, and inpainting on demand. It spins up the correct pipelines as needed
        in order to handle provided pipeline arguments using lazy initialization.

        Pipelines and user specified sub models are memoized and their lifetimes are managed via
        heuristics based on system memory and available resources.

        All arguments to this constructor should be provided as keyword arguments, using this
        constructor in any other fashion could result in breakage inbetween semver compatible
        versions.

        :param model_path: main model path
        :param model_type: main model type
        :param revision: main model revision
        :param variant: main model variant
        :param subfolder: main model subfolder (huggingface or disk)
        :param dtype: main model dtype
        :param unet_uri: main model UNet URI string
        :param second_unet_uri: secondary model unet uri (SDXL Refiner, Stable Cascade decoder)
        :param vae_uri: main model VAE URI string
        :param vae_tiling: use VAE tiling?
        :param vae_slicing: use VAE slicing?
        :param lora_uris: One or more LoRA URI strings
        :param ip_adapter_uris: One or more IP Adapter URI strings
        :param textual_inversion_uris: One or more Textual Inversion URI strings
        :param text_encoder_uris: One or more Text Encoder URIs
            ("+", or None for default. Or "null" indicating do not load) for the main model
        :param second_text_encoder_uris:  One or more Text Encoder URIs
            ("+", or None for default. Or "null" indicating do not load) for the secondary
            model (SDXL Refiner or Stable Cascade decoder)
        :param control_net_uris: One or more ControlNet URI strings
        :param t2i_adapter_uris: One or more T2IAdapter URI strings
        :param scheduler: Scheduler URI string for the main model
        :param sdxl_refiner_uri: SDXL Refiner model URI string
        :param sdxl_refiner_scheduler: Scheduler URI string for the SDXL Refiner
        :param s_cascade_decoder_uri: Stable Cascade decoder URI string
        :param s_cascade_decoder_scheduler: Scheduler URI string for the Stable Cascade decoder
        :param device: Rendering device string, example: ``cuda:0`` or ``cuda``
        :param safety_checker: Use safety checker model if available? (antiquated, for SD 1/2, Deep Floyd etc.)
        :param auth_token: huggingface authentication token.
        :param local_files_only: Do not attempt to download files from huggingface?
        :param model_extra_modules: Raw extra diffusers modules for the main pipeline
        :param second_model_extra_modules: Raw extra diffusers modules for the secondary pipeline (SDXL Refiner, Stable Cascade decoder)
        :param model_cpu_offload: Use model CPU offloading for the main pipeline via the accelerate module?
        :param model_sequential_offload: Use sequential CPU offloading for the main pipeline via the accelerate module?
        :param sdxl_refiner_cpu_offload: Use CPU offloading for the SDXL Refiner via the accelerate module?
        :param sdxl_refiner_sequential_offload: Use sequential CPU offloading for the SDXL Refiner via the accelerate module?
        :param s_cascade_decoder_cpu_offload: Use CPU offloading for the Stable Cascade decoder via the accelerate module?
        :param s_cascade_decoder_sequential_offload: Use sequential CPU offloading for the Stable Cascade decoder via the accelerate module?
        :param prompt_weighter_uri: Prompt weighter implementation URI, to be loaded from ``prompt_weighter_loader``
        :param prompt_weighter_loader: Plugin loader for prompt weighter implementations, if you pass ``None`` a default instance will be created.
        """

        __locals = locals()

        __locals.pop('self')

        for name, value in __locals.items():
            if name.endswith('_uris') and isinstance(value, str):
                __locals[name] = [value]

        self._init(**__locals)

    def _init(self,
              model_path: _types.Path,
              model_type: _enums.ModelType | str = _enums.ModelType.TORCH,
              revision: _types.OptionalName = None,
              variant: _types.OptionalName = None,
              subfolder: _types.OptionalName = None,
              dtype: _enums.DataType | str = _enums.DataType.AUTO,
              unet_uri: _types.OptionalUri = None,
              second_unet_uri: _types.OptionalUri = None,
              vae_uri: _types.OptionalUri = None,
              vae_tiling: bool = False,
              vae_slicing: bool = False,
              lora_uris: _types.OptionalUris = None,
              ip_adapter_uris: _types.OptionalUris = None,
              textual_inversion_uris: _types.OptionalUris = None,
              text_encoder_uris: _types.OptionalUris = None,
              second_text_encoder_uris: _types.OptionalUris = None,
              control_net_uris: _types.OptionalUris = None,
              t2i_adapter_uris: _types.OptionalUris = None,
              scheduler: _types.OptionalUri = None,
              sdxl_refiner_uri: _types.OptionalUri = None,
              sdxl_refiner_scheduler: _types.OptionalUri = None,
              s_cascade_decoder_uri: _types.OptionalUri = None,
              s_cascade_decoder_scheduler: _types.OptionalUri = None,
              device: str = 'cuda',
              safety_checker: bool = False,
              auth_token: _types.OptionalString = None,
              local_files_only: bool = False,
              model_extra_modules: dict[str, typing.Any] = None,
              second_model_extra_modules: dict[str, typing.Any] = None,
              model_cpu_offload: bool = False,
              model_sequential_offload: bool = False,
              sdxl_refiner_cpu_offload: bool = False,
              sdxl_refiner_sequential_offload: bool = False,
              s_cascade_decoder_cpu_offload: bool = False,
              s_cascade_decoder_sequential_offload: bool = False,
              prompt_weighter_uri: _types.OptionalUri = None,
              prompt_weighter_loader: _promptweighters.PromptWeighterLoader | None = None):

        model_type = _enums.get_model_type_enum(model_type)

        if second_text_encoder_uris and not \
                (_enums.model_type_is_sdxl(model_type) or
                 _enums.model_type_is_s_cascade(model_type)):
            raise _pipelines.UnsupportedPipelineConfigError(
                f'Cannot use "second_text_encoder_uris" with "model_type" '
                f'{_enums.get_model_type_string(model_type)}')

        if _pipelines.scheduler_is_help(sdxl_refiner_scheduler) and not sdxl_refiner_uri:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'Cannot use "sdxl_refiner_scheduler" value "help" / "helpargs" '
                f'if no refiner is specified.')

        if _pipelines.scheduler_is_help(s_cascade_decoder_scheduler) and not s_cascade_decoder_uri:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'Cannot use "s_cascade_decoder_scheduler" value "help" / "helpargs" '
                f'if no decoder is specified.')

        if _enums.model_type_is_sdxl(model_type) and _pipelines.text_encoder_is_help(second_text_encoder_uris) \
                and not sdxl_refiner_uri:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'Cannot use "second_text_encoder_uris" value '
                f'"help" if no refiner is specified.')

        if _enums.model_type_is_s_cascade(model_type) and _pipelines.text_encoder_is_help(second_text_encoder_uris) \
                and not s_cascade_decoder_uri:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'Cannot use "second_text_encoder_uris" value '
                f'"help" if no decoder is specified.')

        if control_net_uris and t2i_adapter_uris:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'Cannot use "control_net_uris" and "t2i_adapter_uris" together.')

        helps_used = [
            _pipelines.scheduler_is_help(scheduler),
            _pipelines.scheduler_is_help(sdxl_refiner_scheduler),
            _pipelines.scheduler_is_help(s_cascade_decoder_scheduler),
            _pipelines.text_encoder_is_help(text_encoder_uris),
            _pipelines.text_encoder_is_help(second_text_encoder_uris)
        ]

        if helps_used.count(True) > 1:
            raise _pipelines.UnsupportedPipelineConfigError(
                'Cannot use the "help" / "helpargs" option value '
                'with multiple arguments simultaneously.')

        if _enums.model_type_is_s_cascade(model_type):
            if textual_inversion_uris:
                raise _pipelines.UnsupportedPipelineConfigError('Textual Inversions not supported for StableCascade.')

            if control_net_uris:
                raise _pipelines.UnsupportedPipelineConfigError('ControlNets not supported for StableCascade.')

            if t2i_adapter_uris:
                raise _pipelines.UnsupportedPipelineConfigError('T2IAdapters not supported for StableCascade.')

        if _enums.model_type_is_floyd(model_type):
            if textual_inversion_uris:
                raise _pipelines.UnsupportedPipelineConfigError('Textual Inversions not supported for Deep Floyd.')

            if control_net_uris:
                raise _pipelines.UnsupportedPipelineConfigError('ControlNets not supported for Deep Floyd.')

            if t2i_adapter_uris:
                raise _pipelines.UnsupportedPipelineConfigError('T2IAdapters not supported for Deep Floyd.')

        if sdxl_refiner_uri is not None:
            if not _enums.model_type_is_sdxl(model_type):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Only Stable Diffusion XL models support refiners, '
                    'please use model_type "torch-sdxl" if you are trying to load an sdxl model.')

        if model_type == _enums.ModelType.FLAX:
            if t2i_adapter_uris:
                raise _pipelines.UnsupportedPipelineConfigError('T2IAdapters not supported for flax.')

            if not _enums.have_jax_flax():
                raise _pipelines.UnsupportedPipelineConfigError('flax and jax are not installed.')

            if textual_inversion_uris:
                raise _pipelines.UnsupportedPipelineConfigError('Textual inversion not supported for flax.')

            if vae_tiling or vae_slicing:
                raise _pipelines.UnsupportedPipelineConfigError('vae_tiling / vae_slicing not supported for flax.')

        if lora_uris:
            if model_type == _enums.ModelType.FLAX:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'LoRA loading is not implemented for flax.')

            if _enums.model_type_is_s_cascade(model_type):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'LoRA loading is not implemented for stable cascade.')

        if ip_adapter_uris:
            if model_type == _enums.ModelType.FLAX:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'IP Adapter loading is not implemented for flax.')

            if _enums.model_type_is_s_cascade(model_type):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'IP Adapter loading is not implemented for stable cascade.')

            if _enums.model_type_is_sd3(model_type):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'IP Adapter loading is not implemented for stable diffusion 3.')

        if textual_inversion_uris:
            if model_type == _enums.ModelType.FLAX:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Textual inversion loading is not implemented for flax.')

            if _enums.model_type_is_s_cascade(model_type):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Textual inversion loading is not implemented for stable cascade.')

            if _enums.model_type_is_sd3(model_type):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Textual inversion loading is not implemented for stable diffusion 3.')

        self._subfolder = subfolder
        self._device = device
        self._model_type = _enums.get_model_type_enum(model_type)
        self._model_path = model_path
        self._pipeline = None
        self._flax_params = None
        self._revision = revision
        self._variant = variant
        self._dtype = _enums.get_data_type_enum(dtype)
        self._device = device
        self._unet_uri = unet_uri
        self._second_unet_uri = second_unet_uri
        self._vae_uri = vae_uri
        self._vae_tiling = vae_tiling
        self._vae_slicing = vae_slicing
        self._safety_checker = safety_checker
        self._scheduler = scheduler
        self._sdxl_refiner_scheduler = sdxl_refiner_scheduler
        self._s_cascade_decoder_scheduler = s_cascade_decoder_scheduler

        self._s_cascade_decoder_cpu_offload = s_cascade_decoder_cpu_offload
        self._s_cascade_decoder_sequential_offload = s_cascade_decoder_sequential_offload

        self._lora_uris = lora_uris
        self._ip_adapter_uris = ip_adapter_uris
        self._textual_inversion_uris = textual_inversion_uris
        self._text_encoder_uris = text_encoder_uris
        self._second_text_encoder_uris = second_text_encoder_uris
        self._control_net_uris = control_net_uris
        self._t2i_adapter_uris = t2i_adapter_uris
        self._parsed_control_net_uris = []
        self._parsed_t2i_adapter_uris = []
        self._sdxl_refiner_pipeline = None
        self._s_cascade_decoder_pipeline = None
        self._auth_token = auth_token
        self._pipeline_type = None
        self._local_files_only = local_files_only
        self._recall_main_pipeline = None
        self._recall_refiner_pipeline = None
        self._model_extra_modules = model_extra_modules
        self._second_model_extra_modules = second_model_extra_modules

        if model_cpu_offload and model_sequential_offload:
            raise _pipelines.UnsupportedPipelineConfigError(
                'model_cpu_offload and model_sequential_offload cannot both be True.')

        self._model_cpu_offload = model_cpu_offload
        self._model_sequential_offload = model_sequential_offload

        if sdxl_refiner_sequential_offload and sdxl_refiner_cpu_offload:
            raise _pipelines.UnsupportedPipelineConfigError(
                'refiner_cpu_offload and refiner_sequential_offload cannot both be True.')

        self._sdxl_refiner_sequential_offload = sdxl_refiner_sequential_offload
        self._sdxl_refiner_cpu_offload = sdxl_refiner_cpu_offload

        self._parsed_sdxl_refiner_uri = None
        self._sdxl_refiner_uri = sdxl_refiner_uri
        if sdxl_refiner_uri is not None:
            # up front validation of this URI is optimal
            self._parsed_sdxl_refiner_uri = _uris.SDXLRefinerUri.parse(sdxl_refiner_uri)

        self._s_cascade_decoder_uri = s_cascade_decoder_uri
        self._parsed_s_cascade_decoder_uri = None
        if s_cascade_decoder_uri is not None:
            # up front validation of this URI is optimal
            self._parsed_s_cascade_decoder_uri = _uris.SCascadeDecoderUri.parse(s_cascade_decoder_uri)

        self._parsed_ip_adapter_uris = None
        if ip_adapter_uris:
            # up front validation of these URIs is optimal
            self._parsed_ip_adapter_uris = []
            for ip_adapter_uri in ip_adapter_uris:
                self._parsed_ip_adapter_uris.append(_uris.IPAdapterUri.parse(ip_adapter_uri))

        self._prompt_weighter_loader = \
            prompt_weighter_loader if prompt_weighter_loader is not None \
                else _promptweighters.PromptWeighterLoader()

        self._prompt_weighter_uri = prompt_weighter_uri
        self._prompt_weighter: _promptweighters.PromptWeighter | None = None

    @property
    def prompt_weighter_loader(self) -> _promptweighters.PromptWeighterLoader:
        """
        Current prompt weighter loader
        """
        return self._prompt_weighter_loader

    @property
    def prompt_weighter_uri(self) -> _types.OptionalUri:
        """
        Current prompt weighter implementation uri
        """
        return self._prompt_weighter_uri

    @property
    def local_files_only(self) -> bool:
        """
        Currently set value for ``local_files_only``
        """
        return self._local_files_only

    @property
    def revision(self) -> _types.OptionalName:
        """
        Currently set ``--revision`` for the main model or ``None``
        """
        return self._revision

    @property
    def safety_checker(self) -> bool:
        """
        Safety checker enabled status
        """
        return self._safety_checker

    @property
    def variant(self) -> _types.OptionalName:
        """
        Currently set ``--variant`` for the main model or ``None``
        """
        return self._variant

    @property
    def dtype(self) -> _enums.DataType:
        """
        Currently set ``--dtype`` enum value for the main model
        """
        return self._dtype

    @property
    def dtype_string(self) -> str:
        """
        Currently set ``--dtype`` string value for the main model
        """
        return _enums.get_data_type_string(self._dtype)

    @property
    def textual_inversion_uris(self) -> _types.OptionalUris:
        """
        List of supplied ``--textual-inversions`` uri strings or an empty list
        """
        return list(self._textual_inversion_uris) if self._textual_inversion_uris else []

    @property
    def control_net_uris(self) -> _types.OptionalUris:
        """
        List of supplied ``--control-nets`` uri strings or an empty list
        """
        return list(self._control_net_uris) if self._control_net_uris else []

    @property
    def t2i_adapter_uris(self) -> _types.OptionalUris:
        """
        List of supplied ``--t2i-adapters`` uri strings or an empty list
        """
        return list(self._t2i_adapter_uris) if self._t2i_adapter_uris else []

    @property
    def text_encoder_uris(self) -> _types.OptionalUris:
        """
        List of supplied ``--text-encoders`` uri strings or an empty list
        """
        return list(self._text_encoder_uris) if self._text_encoder_uris else []

    @property
    def second_text_encoder_uris(self) -> _types.OptionalUris:
        """
        List of supplied ``--text-encoders2`` uri strings or an empty list
        """
        return list(self._second_text_encoder_uris) if self._second_text_encoder_uris else []

    @property
    def device(self) -> _types.Name:
        """
        Currently set ``--device`` string
        """
        return self._device

    @property
    def model_path(self) -> _types.Path:
        """
        Model path for the main model
        """
        return self._model_path

    @property
    def scheduler(self) -> _types.OptionalUri:
        """
        Selected scheduler URI for the main model or ``None``
        """
        return self._scheduler

    @property
    def sdxl_refiner_scheduler(self) -> _types.OptionalUri:
        """
        Selected scheduler URI for the SDXL refiner or ``None``
        """
        return self._sdxl_refiner_scheduler

    @property
    def s_cascade_decoder_scheduler(self) -> _types.OptionalUri:
        """
        Selected scheduler URI for the Stable Cascade decoder or ``None``
        """
        return self._s_cascade_decoder_scheduler

    @property
    def sdxl_refiner_uri(self) -> _types.OptionalUri:
        """
        Model URI for the SDXL refiner or ``None``
        """
        return self._sdxl_refiner_uri

    @property
    def s_cascade_decoder_uri(self) -> _types.OptionalUri:
        """
        Model URI for the Stable Cascade decoder or ``None``
        """
        return self._s_cascade_decoder_uri

    @property
    def model_type(self) -> _enums.ModelType:
        """
        Currently set ``--model-type`` enum value
        """
        return self._model_type

    @property
    def model_type_string(self) -> str:
        """
        Currently set ``--model-type`` string value
        """
        return _enums.get_model_type_string(self._model_type)

    @property
    def subfolder(self) -> _types.OptionalName:
        """
        Selected model ``--subfolder`` for the main model, (remote repo subfolder or local) or ``None``
        """
        return self._subfolder

    @property
    def vae_uri(self) -> _types.OptionalUri:
        """
        Selected ``--vae`` uri for the main model or ``None``
        """
        return self._vae_uri

    @property
    def unet_uri(self) -> _types.OptionalUri:
        """
        Selected ``--unet`` uri for the main model or ``None``
        """
        return self._unet_uri

    @property
    def second_unet_uri(self) -> _types.OptionalUri:
        """
        Selected ``--unet2`` uri for the SDXL refiner or Stable Cascade decoder model or ``None``
        """
        return self._second_unet_uri

    @property
    def vae_tiling(self) -> bool:
        """
        Current ``--vae-tiling`` status
        """
        return self._vae_tiling

    @property
    def vae_slicing(self) -> bool:
        """
        Current ``--vae-slicing`` status
        """
        return self._vae_slicing

    @property
    def lora_uris(self) -> _types.OptionalUris:
        """
        List of supplied ``--loras`` uri strings or an empty list
        """
        return list(self._lora_uris) if self._lora_uris else []

    @property
    def auth_token(self) -> _types.OptionalString:
        """
        Current ``--auth-token`` value or ``None``
        """
        return self._auth_token

    @property
    def model_sequential_offload(self) -> bool:
        """
        Current ``--model-sequential-offload`` value
        """
        return self._model_sequential_offload

    @property
    def model_cpu_offload(self) -> bool:
        """
        Current ``--model-cpu-offload`` value
        """
        return self._model_cpu_offload

    @property
    def sdxl_refiner_sequential_offload(self) -> bool:
        """
        Current ``--sdxl-refiner-sequential-offload`` value
        """
        return self._sdxl_refiner_sequential_offload

    @property
    def sdxl_refiner_cpu_offload(self) -> bool:
        """
        Current ``--sdxl-refiner-cpu-offload`` value
        """
        return self._sdxl_refiner_cpu_offload

    @property
    def s_cascade_decoder_sequential_offload(self) -> bool:
        """
        Current ``--s-cascade-decoder-sequential-offload`` value
        """
        return self._s_cascade_decoder_sequential_offload

    @property
    def s_cascade_decoder_cpu_offload(self) -> bool:
        """
        Current ``--s-cascade-decoder-cpu-offload`` value
        """
        return self._s_cascade_decoder_cpu_offload

    def reconstruct_dgenerate_opts(self,
                                   args: DiffusionArguments | None = None,
                                   extra_opts:
                                   collections.abc.Sequence[
                                       tuple[str] | tuple[str, typing.Any]] | None = None,
                                   omit_device=False,
                                   shell_quote=True,
                                   **kwargs) -> \
            list[tuple[str] | tuple[str, typing.Any]]:
        """
        Reconstruct dgenerates command line arguments from a particular set of pipeline wrapper call arguments.
        
        This does not reproduce ``--image-seeds``, you must include that value in ``extra_opts``, 
        this is because there is not enough information in :py:class:`.DiffusionArguments` to
        accurately reproduce it.

        :param args: :py:class:`.DiffusionArguments` object to take values from

        :param extra_opts: Extra option pairs to be added to the end of reconstructed options,
            this should be a sequence of tuples of length 1 (switch only) or length 2 (switch with args)
            
        :param omit_device: Omit the ``--device`` option? For a shareable configuration it might not
            make sense to include the device specification. And instead simply fallback to whatever 
            the default device is, which is generally ``cuda``

        :param shell_quote: Shell quote and format the argument values? or return them raw.

        :param kwargs: pipeline wrapper keyword arguments, these will override values derived from
            any :py:class:`.DiffusionArguments` object given to the *args* argument. See:
            :py:class:`.DiffusionArguments.get_pipeline_wrapper_kwargs`

        :return: List of tuples of length 1 or 2 representing the option
        """

        copy_args = DiffusionArguments()

        if args is not None:
            copy_args.set_from(args)

        copy_args.set_from(kwargs, missing_value_throws=False)

        args = copy_args

        opts = [(self.model_path,),
                ('--model-type', self.model_type_string)]

        if not omit_device:
            opts.append(('--device', self._device))

        opts.append(('--inference-steps', args.inference_steps))
        opts.append(('--guidance-scales', args.guidance_scale))
        opts.append(('--seeds', args.seed))

        if self.dtype_string != 'auto':
            opts.append(('--dtype', self.dtype_string))

        if args.batch_size is not None and args.batch_size > 1:
            opts.append(('--batch-size', args.batch_size))

        if args.guidance_rescale is not None:
            opts.append(('--guidance-rescales', args.guidance_rescale))

        if args.image_guidance_scale is not None:
            opts.append(('--image-guidance-scales', args.image_guidance_scale))

        if self.prompt_weighter_uri:
            opts.append(('--prompt-weighter', self.prompt_weighter_uri))

        if args.prompt is not None:
            opts.append(('--prompts', args.prompt))

        if args.sd3_max_sequence_length is not None:
            opts.append(('--sd3-max-sequence-length', args.sd3_max_sequence_length))

        if args.sd3_second_prompt is not None:
            opts.append(('--sd3-second-prompts', args.sd3_second_prompt))

        if args.sd3_third_prompt is not None:
            opts.append(('--sd3-third-prompts', args.sd3_third_prompt))

        if args.clip_skip is not None:
            opts.append(('--clip-skips', args.clip_skip))

        if args.sdxl_second_prompt is not None:
            opts.append(('--sdxl-second-prompts', args.sdxl_second_prompt))

        if args.sdxl_refiner_prompt is not None:
            opts.append(('--sdxl-refiner-prompts', args.sdxl_refiner_prompt))

        if args.sdxl_refiner_clip_skip is not None:
            opts.append(('--sdxl-refiner-clip-skips', args.sdxl_refiner_clip_skip))

        if args.sdxl_refiner_second_prompt is not None:
            opts.append(('--sdxl-refiner-second-prompts', args.sdxl_refiner_second_prompt))

        if self._text_encoder_uris:
            opts.append(('--text-encoders', ['+' if x is None else x for x in self._text_encoder_uris]))

        if self._second_text_encoder_uris:
            opts.append(('--text-encoders2', ['+' if x is None else x for x in self._second_text_encoder_uris]))

        if self._s_cascade_decoder_uri is not None:
            opts.append(('--s-cascade-decoder', self._s_cascade_decoder_uri))

        if args.s_cascade_decoder_inference_steps is not None:
            opts.append(('--s-cascade-decoder-inference-steps', args.s_cascade_decoder_inference_steps))

        if args.s_cascade_decoder_guidance_scale is not None:
            opts.append(('--s-cascade-decoder-guidance-scales', args.s_cascade_decoder_guidance_scale))

        if args.s_cascade_decoder_prompt is not None:
            opts.append(('--s-cascade-decoder-prompts', args.s_cascade_decoder_prompt))

        if self._s_cascade_decoder_cpu_offload:
            opts.append(('--s-cascade-decoder-cpu-offload',))

        if self._s_cascade_decoder_sequential_offload:
            opts.append(('--s-cascade-decoder-sequential-offload',))

        if self._s_cascade_decoder_scheduler is not None:
            opts.append(('--s-cascade-decoder-scheduler',
                         self._s_cascade_decoder_scheduler))

        if self._revision is not None and self._revision != 'main':
            opts.append(('--revision', self._revision))

        if self._variant is not None:
            opts.append(('--variant', self._variant))

        if self._subfolder is not None:
            opts.append(('--subfolder', self._subfolder))

        if self._unet_uri is not None:
            opts.append(('--unet', self._unet_uri))

        if self._second_unet_uri is not None:
            opts.append(('--unet2', self._second_unet_uri))

        if self._vae_uri is not None:
            opts.append(('--vae', self._vae_uri))

        if self._vae_tiling:
            opts.append(('--vae-tiling',))

        if self._vae_slicing:
            opts.append(('--vae-slicing',))

        if self._model_cpu_offload:
            opts.append(('--model-cpu-offload',))

        if self._model_sequential_offload:
            opts.append(('--model-sequential-offload',))

        if self._sdxl_refiner_uri is not None:
            opts.append(('--sdxl-refiner', self._sdxl_refiner_uri))

        if self._sdxl_refiner_cpu_offload:
            opts.append(('--sdxl-refiner-cpu-offload',))

        if self._sdxl_refiner_sequential_offload:
            opts.append(('--sdxl-refiner-sequential-offload',))

        if args.sdxl_refiner_edit:
            opts.append(('--sdxl-refiner-edit',))

        if self._lora_uris:
            opts.append(('--loras', self._lora_uris))

        if self._textual_inversion_uris:
            opts.append(('--textual-inversions', self._textual_inversion_uris))

        if self._control_net_uris:
            opts.append(('--control-nets', self._control_net_uris))

        if self._t2i_adapter_uris:
            opts.append(('--t2i-adapters', self._t2i_adapter_uris))

        if self._scheduler is not None:
            opts.append(('--scheduler', self._scheduler))

        if self._sdxl_refiner_scheduler is not None:
            if self._sdxl_refiner_scheduler != self._scheduler:
                opts.append(('--sdxl-refiner-scheduler', self._sdxl_refiner_scheduler))

        if args.sdxl_high_noise_fraction is not None:
            opts.append(('--sdxl-high-noise-fractions', args.sdxl_high_noise_fraction))

        if args.sdxl_refiner_inference_steps is not None:
            opts.append(('--sdxl-refiner-inference-steps', args.sdxl_refiner_inference_steps))

        if args.sdxl_refiner_guidance_scale is not None:
            opts.append(('--sdxl-refiner-guidance-scales', args.sdxl_refiner_guidance_scale))

        if args.sdxl_refiner_guidance_rescale is not None:
            opts.append(('--sdxl-refiner-guidance-rescales', args.sdxl_refiner_guidance_rescale))

        if args.sdxl_aesthetic_score is not None:
            opts.append(('--sdxl-aesthetic-scores', args.sdxl_aesthetic_score))

        if args.sdxl_original_size is not None:
            opts.append(('--sdxl-original-size', args.sdxl_original_size))

        if args.sdxl_target_size is not None:
            opts.append(('--sdxl-target-size', args.sdxl_target_size))

        if args.sdxl_crops_coords_top_left is not None:
            opts.append(('--sdxl-crops-coords-top-left', args.sdxl_crops_coords_top_left))

        if args.sdxl_negative_aesthetic_score is not None:
            opts.append(('--sdxl-negative-aesthetic-scores', args.sdxl_negative_aesthetic_score))

        if args.sdxl_negative_original_size is not None:
            opts.append(('--sdxl-negative-original-sizes', args.sdxl_negative_original_size))

        if args.sdxl_negative_target_size is not None:
            opts.append(('--sdxl-negative-target-sizes', args.sdxl_negative_target_size))

        if args.sdxl_negative_crops_coords_top_left is not None:
            opts.append(('--sdxl-negative-crops-coords-top-left', args.sdxl_negative_crops_coords_top_left))

        if args.sdxl_refiner_aesthetic_score is not None:
            opts.append(('--sdxl-refiner-aesthetic-scores', args.sdxl_refiner_aesthetic_score))

        if args.sdxl_refiner_original_size is not None:
            opts.append(('--sdxl-refiner-original-sizes', args.sdxl_refiner_original_size))

        if args.sdxl_refiner_target_size is not None:
            opts.append(('--sdxl-refiner-target-sizes', args.sdxl_refiner_target_size))

        if args.sdxl_refiner_crops_coords_top_left is not None:
            opts.append(('--sdxl-refiner-crops-coords-top-left', args.sdxl_refiner_crops_coords_top_left))

        if args.sdxl_refiner_negative_aesthetic_score is not None:
            opts.append(('--sdxl-refiner-negative-aesthetic-scores', args.sdxl_refiner_negative_aesthetic_score))

        if args.sdxl_refiner_negative_original_size is not None:
            opts.append(('--sdxl-refiner-negative-original-sizes', args.sdxl_refiner_negative_original_size))

        if args.sdxl_refiner_negative_target_size is not None:
            opts.append(('--sdxl-refiner-negative-target-sizes', args.sdxl_refiner_negative_target_size))

        if args.sdxl_refiner_negative_crops_coords_top_left is not None:
            opts.append(
                ('--sdxl-refiner-negative-crops-coords-top-left', args.sdxl_refiner_negative_crops_coords_top_left))

        if args.width is not None and args.height is not None:
            opts.append(('--output-size', f'{args.width}x{args.height}'))
        elif args.width is not None:
            opts.append(('--output-size', f'{args.width}'))

        if extra_opts is not None:
            for opt in extra_opts:
                opts.append(opt)

        if shell_quote:
            for idx, option in enumerate(opts):
                if len(option) > 1:
                    name, value = option
                    if isinstance(value, (str, _prompt.Prompt)):
                        opts[idx] = (name, shlex.quote(str(value)))
                    elif isinstance(value, tuple):
                        opts[idx] = (name, _textprocessing.format_size(value))
                    else:
                        opts[idx] = (name, str(value))
                else:
                    solo_val = str(option[0])
                    if not solo_val.startswith('-'):
                        # not a solo switch option, some value
                        opts[idx] = (shlex.quote(solo_val),)

        return opts

    @staticmethod
    def _set_opt_value_syntax(val):
        if isinstance(val, tuple):
            return _textprocessing.format_size(val)
        if isinstance(val, str):
            return shlex.quote(str(val))

        try:
            val_iter = iter(val)
        except TypeError:
            return shlex.quote(str(val))

        return ' '.join(DiffusionPipelineWrapper._set_opt_value_syntax(v) for v in val_iter)

    @staticmethod
    def _format_option_pair(val):
        if len(val) > 1:
            opt_name, opt_value = val

            if isinstance(opt_value, _prompt.Prompt):
                header_len = len(opt_name) + 2
                prompt_text = \
                    _textprocessing.wrap(
                        shlex.quote(str(opt_value)),
                        subsequent_indent=' ' * header_len,
                        width=75)

                prompt_text = ' \\\n'.join(prompt_text.split('\n'))

                if '\n' in prompt_text:
                    # need to escape the comment token
                    prompt_text = prompt_text.replace('#', r'\#')

                return f'{opt_name} {prompt_text}'

            return f'{opt_name} {DiffusionPipelineWrapper._set_opt_value_syntax(opt_value)}'

        solo_val = str(val[0])

        if solo_val.startswith('-'):
            return solo_val

        # Not a switch option, some value
        return shlex.quote(solo_val)

    def gen_dgenerate_config(self,
                             args: DiffusionArguments | None = None,
                             extra_opts:
                             collections.abc.Sequence[tuple[str] | tuple[str, typing.Any]] | None = None,
                             extra_comments: collections.abc.Iterable[str] | None = None,
                             omit_device: bool = False,
                             **kwargs):
        """
        Generate a valid dgenerate config file with a single invocation that reproduces the 
        arguments associated with :py:class:`.DiffusionArguments`.
        
        This does not reproduce ``--image-seeds``, you must include that value in ``extra_opts``, 
        this is because there is not enough information in :py:class:`.DiffusionArguments` to
        accurately reproduce it.

        :param args: :py:class:`.DiffusionArguments` object to take values from
        :param extra_opts: Extra option pairs to be added to the end of reconstructed options
            of the dgenerate invocation, this should be a sequence of tuples of length 1 (switch only)
            or length 2 (switch with args)
        :param extra_comments: Extra strings to use as comments after the initial
            version check directive
        :param omit_device: Omit the ``--device`` option? For a shareable configuration it might not
            make sense to include the device specification. And instead simply fallback to whatever 
            the default device is, which is generally ``cuda``
        :param kwargs: pipeline wrapper keyword arguments, these will override values derived from
            any :py:class:`.DiffusionArguments` object given to the *args* argument. See:
            :py:class:`.DiffusionArguments.get_pipeline_wrapper_kwargs`
        :return: The configuration as a string
        """

        from dgenerate import __version__

        config = f'#! /usr/bin/env dgenerate --file\n#! dgenerate {__version__}\n\n'

        if extra_comments:
            wrote_comments = False
            for comment in extra_comments:
                wrote_comments = True
                for part in comment.split('\n'):
                    config += '# ' + part.rstrip()

            if wrote_comments:
                config += '\n\n'

        opts = \
            self.reconstruct_dgenerate_opts(args, **kwargs,
                                            shell_quote=False,
                                            omit_device=omit_device)

        if extra_opts is not None:
            for opt in extra_opts:
                opts.append(opt)

        for opt in opts[:-1]:
            config += f'{self._format_option_pair(opt)} \\\n'

        last = opts[-1]

        return config + self._format_option_pair(last)

    def gen_dgenerate_command(self,
                              args: DiffusionArguments | None = None,
                              extra_opts:
                              collections.abc.Sequence[tuple[str] | tuple[str, typing.Any]] | None = None,
                              omit_device=False,
                              **kwargs):
        """
        Generate a valid dgenerate command line invocation that reproduces the 
        arguments associated with :py:class:`.DiffusionArguments`.
        
        This does not reproduce ``--image-seeds``, you must include that value in ``extra_opts``, 
        this is because there is not enough information in :py:class:`.DiffusionArguments` to
        accurately reproduce it.

        :param args: :py:class:`.DiffusionArguments` object to take values from
        :param extra_opts: Extra option pairs to be added to the end of reconstructed options
            of the dgenerate invocation, this should be a sequence of tuples of length 1 (switch only)
            or length 2 (switch with args)
        :param omit_device: Omit the ``--device`` option? For a shareable configuration it might not
            make sense to include the device specification. And instead simply fallback to whatever 
            the default device is, which is generally ``cuda``
        :param kwargs: pipeline wrapper keyword arguments, these will override values derived from
            any :py:class:`.DiffusionArguments` object given to the *args* argument. See:
            :py:class:`.DiffusionArguments.get_pipeline_wrapper_kwargs`
        :return: A string containing the dgenerate command line needed to reproduce this result.
        """

        opt_string = \
            ' '.join(
                f"{self._format_option_pair(opt)}"
                for opt in self.reconstruct_dgenerate_opts(
                    args, **kwargs,
                    extra_opts=extra_opts,
                    omit_device=omit_device,
                    shell_quote=False))

        return f'dgenerate {opt_string}'

    def _get_pipeline_defaults(self, user_args: DiffusionArguments):
        """
        Get a default arrangement of arguments to be passed to a huggingface
        diffusers pipeline call that are somewhat universal.

        :param user_args: user arguments to the pipeline wrapper
        :return: kwargs dictionary
        """

        args: dict[str, typing.Any] = dict()
        args['guidance_scale'] = float(_types.default(user_args.guidance_scale, _constants.DEFAULT_GUIDANCE_SCALE))
        args['num_inference_steps'] = int(_types.default(user_args.inference_steps, _constants.DEFAULT_INFERENCE_STEPS))

        def set_strength():
            strength = float(_types.default(user_args.image_seed_strength, _constants.DEFAULT_IMAGE_SEED_STRENGTH))

            if (strength * user_args.inference_steps) < 1.0:
                strength = 1.0 / user_args.inference_steps
                _messages.log(
                    f'image-seed-strength * inference-steps '
                    f'was calculated at < 1, image-seed-strength defaulting to (1.0 / inference-steps): {strength}',
                    level=_messages.WARNING)

            args['strength'] = strength

        def set_controlnet_defaults():
            control_images = user_args.control_images

            if not control_images:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Must provide control_images argument when using ControlNet models.')

            control_images_cnt = len(control_images)
            control_net_uris_cnt = len(self._control_net_uris)

            if control_images_cnt != control_net_uris_cnt:
                # User provided a mismatched number of ControlNet models and control_images, behavior is undefined.
                raise _pipelines.UnsupportedPipelineConfigError(
                    f'You specified {control_images_cnt} control guidance images and '
                    f'only {control_net_uris_cnt} ControlNet URIs. The amount of '
                    f'control guidance images must be equal to the amount of ControlNet URIs.')

            first_control_image_size = control_images[0].size

            # Check if all control images have the same size
            for img in control_images[1:]:
                if img.size != first_control_image_size:
                    raise _pipelines.UnsupportedPipelineConfigError(
                        "All control guidance images must have the same dimension.")

            # Set width and height based on control images
            args['width'] = _types.default(user_args.width, control_images[0].width)
            args['height'] = _types.default(user_args.height, control_images[0].height)

            if self._pipeline_type == _enums.PipelineType.TXT2IMG:
                if _enums.model_type_is_sd3(self._model_type):
                    # Handle SD3 model specifics for control images
                    args['control_image'] = self._sd3_force_control_to_a16(args, control_images, user_args)
                else:
                    args['image'] = control_images
            elif self._pipeline_type in {_enums.PipelineType.IMG2IMG, _enums.PipelineType.INPAINT}:
                args['image'] = user_args.image
                args['control_image'] = control_images
                set_strength()

            mask_image = user_args.mask_image
            if mask_image is not None:
                args['mask_image'] = mask_image

        def set_t2iadapter_defaults():
            adapter_control_images = list(user_args.control_images)

            if not adapter_control_images:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Must provide control_images argument when using T2IAdapter models.')

            control_images_cnt = len(adapter_control_images)
            t2i_adapter_uris_cnt = len(self._t2i_adapter_uris)

            if control_images_cnt != t2i_adapter_uris_cnt:
                # User provided a mismatched number of T2IAdapter models and control_images, behavior is undefined.
                raise _pipelines.UnsupportedPipelineConfigError(
                    f'You specified {control_images_cnt} control guidance images and '
                    f'only {t2i_adapter_uris_cnt} T2IAdapter URIs. The amount of '
                    f'control guidance images must be equal to the amount of T2IAdapter URIs.')

            first_control_image_size = adapter_control_images[0].size

            # Check if all control images have the same size
            for img in adapter_control_images[1:]:
                if img.size != first_control_image_size:
                    raise _pipelines.UnsupportedPipelineConfigError(
                        "All control guidance images must have the same dimension.")

            if not _image.is_aligned(first_control_image_size, 16):
                new_size = _image.align_by(first_control_image_size, 16)
                _messages.log(
                    f'T2I Adapter control image(s) of size {first_control_image_size} being forcefully '
                    f'aligned by 16 to {new_size} to prevent errors.', level=_messages.WARNING)

                for idx, img in enumerate(adapter_control_images):
                    adapter_control_images[idx] = img.resize(new_size, PIL.Image.Resampling.LANCZOS)

            if _enums.model_type_is_sdxl(self.model_type) and user_args.sdxl_t2i_adapter_factor is not None:
                args['adapter_conditioning_factor'] = user_args.sdxl_t2i_adapter_factor

            # Set width and height based on control images
            args['width'] = _types.default(user_args.width, adapter_control_images[0].width)
            args['height'] = _types.default(user_args.height, adapter_control_images[0].height)

            if self._pipeline_type == _enums.PipelineType.TXT2IMG:
                args['image'] = adapter_control_images
            else:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'T2IAdapter models only work in txt2img mode.'
                )

        def set_img2img_defaults():
            image = user_args.image

            floyd_og_image_needed = (self._pipeline_type == _enums.PipelineType.INPAINT and
                                     _enums.model_type_is_floyd_ifs(self._model_type)
                                     ) or (self._model_type == _enums.ModelType.TORCH_IFS_IMG2IMG)

            if floyd_og_image_needed:
                if user_args.floyd_image is None:
                    raise _pipelines.UnsupportedPipelineConfigError(
                        'must specify "floyd_image" to disambiguate this operation, '
                        '"floyd_image" being the output of a previous floyd stage.')
                args['original_image'] = image
                args['image'] = user_args.floyd_image
            elif self._model_type == _enums.ModelType.TORCH_S_CASCADE:
                args['images'] = [image]
            else:
                args['image'] = image

            def check_no_image_seed_strength():
                if user_args.image_seed_strength is not None:
                    _messages.log(
                        f'image_seed_strength is not supported by model_type '
                        f'"{_enums.get_model_type_string(self._model_type)}" in '
                        f'mode "{self._pipeline_type.name}" and is being ignored.',
                        level=_messages.WARNING)

            if _enums.model_type_is_upscaler(self._model_type):
                if self._model_type == _enums.ModelType.TORCH_UPSCALER_X4:
                    args['noise_level'] = int(
                        _types.default(
                            user_args.upscaler_noise_level,
                            _constants.DEFAULT_X4_UPSCALER_NOISE_LEVEL
                        )
                    )
                check_no_image_seed_strength()
            elif self._model_type == _enums.ModelType.TORCH_IFS:
                if self._pipeline_type != _enums.PipelineType.INPAINT:
                    args['noise_level'] = int(
                        _types.default(
                            user_args.upscaler_noise_level,
                            _constants.DEFAULT_FLOYD_SUPERRESOLUTION_NOISE_LEVEL
                        )
                    )
                    check_no_image_seed_strength()
                else:
                    args['noise_level'] = int(
                        _types.default(
                            user_args.upscaler_noise_level,
                            _constants.DEFAULT_FLOYD_SUPERRESOLUTION_INPAINT_NOISE_LEVEL
                        )
                    )
                    set_strength()
            elif self._model_type == _enums.ModelType.TORCH_IFS_IMG2IMG:
                args['noise_level'] = int(
                    _types.default(
                        user_args.upscaler_noise_level,
                        _constants.DEFAULT_FLOYD_SUPERRESOLUTION_IMG2IMG_NOISE_LEVEL
                    )
                )
                set_strength()
            elif not _enums.model_type_is_pix2pix(self._model_type) and \
                    self._model_type != _enums.ModelType.TORCH_S_CASCADE:
                set_strength()
            else:
                check_no_image_seed_strength()

            mask_image = user_args.mask_image

            if mask_image is not None:
                args['mask_image'] = mask_image
                if not _enums.model_type_is_floyd(self._model_type):
                    args['width'] = image.size[0]
                    args['height'] = image.size[1]

            if self._model_type == _enums.ModelType.TORCH_SDXL_PIX2PIX:
                args['width'] = image.size[0]
                args['height'] = image.size[1]

            if self._model_type == _enums.ModelType.TORCH_UPSCALER_X2:
                if not _image.is_aligned(image.size, 64):
                    size = _image.align_by(image.size, 64)
                    _messages.log(
                        f'Input image size {image.size} is not aligned by 64. '
                        f'Output dimensions will be forcefully aligned to 64: {size}.',
                        level=_messages.WARNING)
                    args['image'] = image.resize(size, PIL.Image.Resampling.LANCZOS)

            if self._model_type == _enums.ModelType.TORCH_S_CASCADE:
                if not _image.is_aligned(image.size, 128):
                    size = _image.align_by(image.size, 128)
                    _messages.log(
                        f'Input image size {image.size} is not aligned by 128. '
                        f'Output dimensions will be forcefully aligned to 128: {size}.',
                        level=_messages.WARNING)
                else:
                    size = image.size

                if user_args.width and user_args.width > 0:
                    if not (user_args.width % 128) == 0:
                        raise _pipelines.UnsupportedPipelineConfigError(
                            'Stable Cascade requires an output dimension that is aligned by 128.')

                if user_args.height and user_args.height > 0:
                    if not (user_args.height % 128) == 0:
                        raise _pipelines.UnsupportedPipelineConfigError(
                            'Stable Cascade requires an output dimension that is aligned by 128.')

                args['width'] = _types.default(user_args.width, size[0])
                args['height'] = _types.default(user_args.height, size[1])

            if self._model_type == _enums.ModelType.TORCH_SD3:
                if not _image.is_aligned(image.size, 16):
                    size = _image.align_by(image.size, 16)
                    _messages.log(
                        f'Input image size {image.size} is not aligned by 16. '
                        f'Output dimensions will be forcefully aligned to 16: {size}.',
                        level=_messages.WARNING)
                    args['image'] = image.resize(size, PIL.Image.Resampling.LANCZOS)

        def set_txt2img_defaults():
            if _enums.model_type_is_sdxl(self._model_type):
                args['height'] = _types.default(user_args.height, _constants.DEFAULT_SDXL_OUTPUT_HEIGHT)
                args['width'] = _types.default(user_args.width, _constants.DEFAULT_SDXL_OUTPUT_WIDTH)
            elif _enums.model_type_is_floyd_if(self._model_type):
                args['height'] = _types.default(user_args.height, _constants.DEFAULT_FLOYD_IF_OUTPUT_HEIGHT)
                args['width'] = _types.default(user_args.width, _constants.DEFAULT_FLOYD_IF_OUTPUT_WIDTH)
            elif self._model_type == _enums.ModelType.TORCH_S_CASCADE:
                args['height'] = _types.default(user_args.height, _constants.DEFAULT_S_CASCADE_OUTPUT_HEIGHT)
                args['width'] = _types.default(user_args.width, _constants.DEFAULT_S_CASCADE_OUTPUT_WIDTH)

                if not _image.is_aligned((args['width'], args['height']), 128):
                    raise _pipelines.UnsupportedPipelineConfigError(
                        'Stable Cascade requires an output dimension that is aligned by 128.')
            elif self._model_type == _enums.ModelType.TORCH_SD3:
                args['height'] = _types.default(user_args.height, _constants.DEFAULT_SD3_OUTPUT_HEIGHT)
                args['width'] = _types.default(user_args.width, _constants.DEFAULT_SD3_OUTPUT_WIDTH)

                if not _image.is_aligned((args['width'], args['height']), 16):
                    raise _pipelines.UnsupportedPipelineConfigError(
                        'Stable Diffusion 3 requires an output dimension that is aligned by 16.')
            else:
                args['height'] = _types.default(user_args.height, _constants.DEFAULT_OUTPUT_HEIGHT)
                args['width'] = _types.default(user_args.width, _constants.DEFAULT_OUTPUT_WIDTH)

        if self._control_net_uris:
            set_controlnet_defaults()
        elif self._t2i_adapter_uris:
            set_t2iadapter_defaults()
        elif user_args.image is not None:
            set_img2img_defaults()
        else:
            set_txt2img_defaults()

        return args

    @staticmethod
    def _sd3_force_control_to_a16(args, control_images, user_args):
        processed_control_images = list(control_images)
        for idx, img in enumerate(processed_control_images):
            if not _image.is_aligned(img.size, 16):
                size = _image.align_by(img.size, 16)

                if user_args.width:
                    if not (user_args.width % 16) == 0:
                        raise _pipelines.UnsupportedPipelineConfigError(
                            'Stable Diffusion 3 requires an output dimension aligned by 16.')

                if user_args.height:
                    if not (user_args.height % 16) == 0:
                        raise _pipelines.UnsupportedPipelineConfigError(
                            'Stable Diffusion 3 requires an output dimension aligned by 16.')

                args['width'] = _types.default(user_args.width, size[0])
                args['height'] = _types.default(user_args.height, size[1])

                _messages.log(
                    f'Control image size {img.size} is not aligned by 16. '
                    f'Output dimensions will be forcefully aligned by 16: {size}.',
                    level=_messages.WARNING)

                processed_control_images[idx] = img.resize(size, PIL.Image.Resampling.LANCZOS)
        return processed_control_images

    def _get_adapter_conditioning_scale(self):
        if not self._parsed_t2i_adapter_uris:
            return 1.0
        return [p.scale for p in self._parsed_t2i_adapter_uris] if \
            len(self._parsed_t2i_adapter_uris) > 1 else self._parsed_t2i_adapter_uris[0].scale

    def _get_control_net_conditioning_scale(self):
        if not self._parsed_control_net_uris:
            return 1.0
        return [p.scale for p in self._parsed_control_net_uris] if \
            len(self._parsed_control_net_uris) > 1 else self._parsed_control_net_uris[0].scale

    def _get_control_net_guidance_start(self):
        if not self._parsed_control_net_uris:
            return 0.0
        return [p.start for p in self._parsed_control_net_uris] if \
            len(self._parsed_control_net_uris) > 1 else self._parsed_control_net_uris[0].start

    def _get_control_net_guidance_end(self):
        if not self._parsed_control_net_uris:
            return 1.0
        return [p.end for p in self._parsed_control_net_uris] if \
            len(self._parsed_control_net_uris) > 1 else self._parsed_control_net_uris[0].end

    def _check_for_invalid_model_specific_opts(self, user_args: DiffusionArguments):
        if not _enums.model_type_is_sdxl(self.model_type):
            for arg, val in _types.get_public_attributes(user_args).items():
                if arg.startswith('sdxl') and val is not None:
                    raise _pipelines.UnsupportedPipelineConfigError(
                        f'{arg} may only be used with SDXL models.')

        if not _enums.model_type_is_sd3(self.model_type):
            for arg, val in _types.get_public_attributes(user_args).items():
                if arg.startswith('sd3') and val is not None:
                    raise _pipelines.UnsupportedPipelineConfigError(
                        f'{arg} may only be used with Stable Diffusion 3 models.')

        if not _enums.model_type_is_s_cascade(self.model_type):
            for arg, val in _types.get_public_attributes(user_args).items():
                if arg.startswith('s_cascade') and val is not None:
                    raise _pipelines.UnsupportedPipelineConfigError(
                        f'{arg} may only be used with Stable Cascade models.')

    def _call_flax_control_net(self, positive_prompt, negative_prompt, pipeline_args, user_args: DiffusionArguments):
        # Only works with txt2image

        self._check_for_invalid_model_specific_opts(user_args)

        if user_args.clip_skip is not None and user_args.clip_skip > 0:
            raise _pipelines.UnsupportedPipelineConfigError('flax does not support clip skip.')

        device_count = jax.device_count()

        pipe: diffusers.FlaxStableDiffusionControlNetPipeline = self._pipeline

        pipeline_args['prng_seed'] = \
            jax.random.split(
                jax.random.PRNGKey(
                    _types.default(user_args.seed,
                                   _constants.DEFAULT_SEED)),
                device_count)

        prompt_ids = pipe.prepare_text_inputs([positive_prompt] * device_count)

        if negative_prompt is not None:
            negative_prompt_ids = pipe.prepare_text_inputs([negative_prompt] * device_count)
        else:
            negative_prompt_ids = None

        control_net_image = pipeline_args.get('image')
        if isinstance(control_net_image, list):
            control_net_image = control_net_image[0]

        processed_image = pipe.prepare_image_inputs([control_net_image] * device_count)
        pipeline_args.pop('image')

        p_params = _flax_replicate(self._flax_params)
        prompt_ids = _flax_shard(prompt_ids)
        negative_prompt_ids = _flax_shard(negative_prompt_ids)
        processed_image = _flax_shard(processed_image)

        pipeline_args.pop('width', None)
        pipeline_args.pop('height', None)

        images = _pipelines.call_pipeline(
            pipeline=self._pipeline,
            device=None,
            prompt_weighter=self._prompt_weighter,
            prompt_ids=prompt_ids,
            image=processed_image,
            params=p_params,
            neg_prompt_ids=negative_prompt_ids,
            controlnet_conditioning_scale=self._get_control_net_conditioning_scale(),
            jit=True, **pipeline_args)[0]

        return PipelineWrapperResult(
            self._pipeline.numpy_to_pil(images.reshape((images.shape[0],) + images.shape[-3:])))

    def _flax_prepare_text_input(self, text):
        tokenizer = self._pipeline.tokenizer
        text_input = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        return text_input.input_ids

    def _call_flax(self, pipeline_args, user_args: DiffusionArguments):
        self._check_for_invalid_model_specific_opts(user_args)

        if user_args.clip_skip is not None and user_args.clip_skip > 0:
            raise _pipelines.UnsupportedPipelineConfigError('flax does not support clip skip.')

        if user_args.guidance_rescale is not None:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'guidance_rescale is not supported when using flax.')

        prompt: _prompt.Prompt() = _types.default(user_args.prompt, _prompt.Prompt())
        positive_prompt = prompt.positive if prompt.positive else ''
        negative_prompt = prompt.negative

        if hasattr(self._pipeline, 'controlnet'):
            return self._call_flax_control_net(positive_prompt, negative_prompt,
                                               pipeline_args, user_args)

        device_count = jax.device_count()

        pipeline_args['prng_seed'] = \
            jax.random.split(
                jax.random.PRNGKey(
                    _types.default(user_args.seed, _constants.DEFAULT_SEED)),
                device_count)

        if negative_prompt is not None:
            negative_prompt_ids = _flax_shard(
                self._flax_prepare_text_input([negative_prompt] * device_count))
        else:
            negative_prompt_ids = None

        if 'image' in pipeline_args:
            if 'mask_image' in pipeline_args:

                prompt_ids, processed_images, processed_masks = \
                    self._pipeline.prepare_inputs(prompt=[positive_prompt] * device_count,
                                                  image=[pipeline_args['image']] * device_count,
                                                  mask=[pipeline_args['mask_image']] * device_count)

                pipeline_args['masked_image'] = _flax_shard(processed_images)
                pipeline_args['mask'] = _flax_shard(processed_masks)

                # inpainting pipeline does not have a strength argument, simply ignore it
                pipeline_args.pop('strength')

                pipeline_args.pop('image')
                pipeline_args.pop('mask_image')
            else:
                prompt_ids, processed_images = self._pipeline.prepare_inputs(
                    prompt=[positive_prompt] * device_count,
                    image=[pipeline_args['image']] * device_count)
                pipeline_args['image'] = _flax_shard(processed_images)

            pipeline_args['width'] = processed_images[0].shape[2]
            pipeline_args['height'] = processed_images[0].shape[1]
        else:
            prompt_ids = self._pipeline.prepare_inputs([positive_prompt] * device_count)

        images = _pipelines.call_pipeline(
            pipeline=self._pipeline,
            device=None,
            prompt_weighter=self._prompt_weighter,
            prompt_ids=_flax_shard(prompt_ids),
            neg_prompt_ids=negative_prompt_ids,
            params=_flax_replicate(self._flax_params),
            **pipeline_args, jit=True)[0]

        return PipelineWrapperResult(self._pipeline.numpy_to_pil(
            images.reshape((images.shape[0],) + images.shape[-3:])))

    def _set_non_universal_pipeline_arg(self,
                                        pipeline,
                                        pipeline_args,
                                        user_args: DiffusionArguments,
                                        pipeline_arg_name,
                                        user_arg_name,
                                        option_name,
                                        transform=None):
        if pipeline.__call__.__wrapped__ is not None:
            # torch.no_grad()
            func = pipeline.__call__.__wrapped__
        else:
            func = pipeline.__call__

        pipeline_kwargs = user_args.get_pipeline_wrapper_kwargs()

        if pipeline_arg_name in inspect.getfullargspec(func).args:
            if user_arg_name in pipeline_kwargs:
                # Only provide if the user provided the option
                # otherwise, defer to the pipelines default value
                val = getattr(user_args, user_arg_name)
                val = val if not transform else transform(val)
                pipeline_args[pipeline_arg_name] = val
        else:
            val = _types.default(getattr(user_args, user_arg_name), None)
            if val is not None:
                raise _pipelines.UnsupportedPipelineConfigError(
                    f'{option_name} cannot be used with --model-type "{self.model_type_string}" in '
                    f'{_enums.get_pipeline_type_string(self._pipeline_type)} mode with the current '
                    f'combination of arguments and model.')

    def _get_sdxl_conditioning_args(self, pipeline, pipeline_args, user_args: DiffusionArguments, user_prefix=None):
        if user_prefix:
            user_prefix += '_'
            option_prefix = _textprocessing.dashup(user_prefix)
        else:
            user_prefix = ''
            option_prefix = ''

        self._set_non_universal_pipeline_arg(pipeline, pipeline_args, user_args,
                                             'aesthetic_score', f'sdxl_{user_prefix}aesthetic_score',
                                             f'--sdxl-{option_prefix}aesthetic-scores')
        self._set_non_universal_pipeline_arg(pipeline, pipeline_args, user_args,
                                             'original_size', f'sdxl_{user_prefix}original_size',
                                             f'--sdxl-{option_prefix}original-sizes')
        self._set_non_universal_pipeline_arg(pipeline, pipeline_args, user_args,
                                             'target_size', f'sdxl_{user_prefix}target_size',
                                             f'--sdxl-{option_prefix}target-sizes')

        self._set_non_universal_pipeline_arg(pipeline, pipeline_args, user_args,
                                             'crops_coords_top_left',
                                             f'sdxl_{user_prefix}crops_coords_top_left',
                                             f'--sdxl-{option_prefix}crops-coords-top-left')

        self._set_non_universal_pipeline_arg(pipeline, pipeline_args, user_args,
                                             'negative_aesthetic_score',
                                             f'sdxl_{user_prefix}negative_aesthetic_score',
                                             f'--sdxl-{option_prefix}negative-aesthetic-scores')

        self._set_non_universal_pipeline_arg(pipeline, pipeline_args, user_args,
                                             'negative_original_size',
                                             f'sdxl_{user_prefix}negative_original_size',
                                             f'--sdxl-{option_prefix}negative-original-sizes')

        self._set_non_universal_pipeline_arg(pipeline, pipeline_args, user_args,
                                             'negative_target_size',
                                             f'sdxl_{user_prefix}negative_target_size',
                                             f'--sdxl-{option_prefix}negative-target-sizes')

        self._set_non_universal_pipeline_arg(pipeline, pipeline_args, user_args,
                                             'negative_crops_coords_top_left',
                                             f'sdxl_{user_prefix}negative_crops_coords_top_left',
                                             f'--sdxl-{option_prefix}negative-crops-coords-top-left')

    @staticmethod
    def _pop_sdxl_conditioning_args(pipeline_args):
        pipeline_args.pop('aesthetic_score', None)
        pipeline_args.pop('target_size', None)
        pipeline_args.pop('original_size', None)
        pipeline_args.pop('crops_coords_top_left', None)
        pipeline_args.pop('negative_aesthetic_score', None)
        pipeline_args.pop('negative_target_size', None)
        pipeline_args.pop('negative_original_size', None)
        pipeline_args.pop('negative_crops_coords_top_left', None)

    def _call_torch_s_cascade(self, pipeline_args, user_args: DiffusionArguments):
        self._check_for_invalid_model_specific_opts(user_args)

        if user_args.clip_skip is not None and user_args.clip_skip > 0:
            raise _pipelines.UnsupportedPipelineConfigError('Stable Cascade does not support clip skip.')

        prompt: _prompt.Prompt() = _types.default(user_args.prompt, _prompt.Prompt())
        pipeline_args['prompt'] = prompt.positive if prompt.positive else ''
        pipeline_args['negative_prompt'] = prompt.negative

        pipeline_args['num_images_per_prompt'] = _types.default(user_args.batch_size, 1)

        pipeline_args['generator'] = \
            torch.Generator(device=self._device).manual_seed(
                _types.default(user_args.seed, _constants.DEFAULT_SEED))

        prior = _pipelines.call_pipeline(
            pipeline=self._pipeline,
            device=self._device,
            prompt_weighter=self._prompt_weighter,
            **pipeline_args)

        pipeline_args['num_inference_steps'] = user_args.s_cascade_decoder_inference_steps
        pipeline_args['guidance_scale'] = user_args.s_cascade_decoder_guidance_scale
        pipeline_args.pop('height')
        pipeline_args.pop('width')
        pipeline_args.pop('images', None)

        if self._parsed_s_cascade_decoder_uri.dtype is not None:
            image_embeddings = prior.image_embeddings.to(
                _enums.get_torch_dtype(self._parsed_s_cascade_decoder_uri.dtype))
        else:
            image_embeddings = prior.image_embeddings

        if user_args.s_cascade_decoder_prompt:
            prompt: _prompt.Prompt() = user_args.s_cascade_decoder_prompt
            pipeline_args['prompt'] = prompt.positive if prompt.positive else ''
            pipeline_args['negative_prompt'] = prompt.negative

        pipeline_args.pop('num_images_per_prompt')

        return PipelineWrapperResult(_pipelines.call_pipeline(
            image_embeddings=image_embeddings,
            pipeline=self._s_cascade_decoder_pipeline,
            device=self._device,
            prompt_weighter=self._prompt_weighter,
            **pipeline_args).images)

    def _call_torch(self, pipeline_args, user_args: DiffusionArguments):
        self._check_for_invalid_model_specific_opts(user_args)

        prompt: _prompt.Prompt() = _types.default(user_args.prompt, _prompt.Prompt())

        pipeline_args['prompt'] = prompt.positive if prompt.positive else ''
        pipeline_args['negative_prompt'] = prompt.negative

        self._get_sdxl_conditioning_args(self._pipeline, pipeline_args, user_args)

        if _enums.model_type_is_sd3(self.model_type):

            self._set_non_universal_pipeline_arg(self._pipeline,
                                                 pipeline_args, user_args,
                                                 'max_sequence_length', 'sd3_max_sequence_length',
                                                 '--sd3-max-sequence-length')

            self._set_non_universal_pipeline_arg(self._pipeline,
                                                 pipeline_args, user_args,
                                                 'prompt_2', 'sd3_second_prompt',
                                                 '--sd3-second-prompts',
                                                 transform=lambda p: p.positive)

            self._set_non_universal_pipeline_arg(self._pipeline,
                                                 pipeline_args, user_args,
                                                 'prompt_3', 'sd3_third_prompt',
                                                 '--sd3-third-prompts',
                                                 transform=lambda p: p.positive)

            self._set_non_universal_pipeline_arg(self._pipeline,
                                                 pipeline_args, user_args,
                                                 'negative_prompt_2', 'sd3_second_prompt',
                                                 '--sd3-second-prompts',
                                                 transform=lambda p: p.negative)

            self._set_non_universal_pipeline_arg(self._pipeline,
                                                 pipeline_args, user_args,
                                                 'negative_prompt_3', 'sd3_third_prompt',
                                                 '--sd3-third-prompts',
                                                 transform=lambda p: p.negative)

        else:
            self._set_non_universal_pipeline_arg(self._pipeline,
                                                 pipeline_args, user_args,
                                                 'prompt_2', 'sdxl_second_prompt',
                                                 '--sdxl-second-prompts',
                                                 transform=lambda p: p.positive)

            self._set_non_universal_pipeline_arg(self._pipeline,
                                                 pipeline_args, user_args,
                                                 'negative_prompt_2', 'sdxl_second_prompt',
                                                 '--sdxl-second-prompts',
                                                 transform=lambda p: p.negative)

        self._set_non_universal_pipeline_arg(self._pipeline,
                                             pipeline_args, user_args,
                                             'guidance_rescale', 'guidance_rescale',
                                             '--guidance-rescales')

        self._set_non_universal_pipeline_arg(self._pipeline,
                                             pipeline_args, user_args,
                                             'clip_skip', 'clip_skip',
                                             '--clip-skips')

        self._set_non_universal_pipeline_arg(self._pipeline,
                                             pipeline_args, user_args,
                                             'image_guidance_scale', 'image_guidance_scale',
                                             '--image-guidance-scales')

        self._set_non_universal_pipeline_arg(self._pipeline,
                                             pipeline_args, user_args,
                                             'ip_adapter_image', 'ip_adapter_images',
                                             'IP Adapter images')

        if user_args.ip_adapter_images is not None:
            self._pipeline.set_ip_adapter_scale([u.scale for u in self._parsed_ip_adapter_uris])

        batch_size = _types.default(user_args.batch_size, 1)

        mock_batching = False

        if self._model_type != _enums.ModelType.TORCH_UPSCALER_X2:
            # Upscaler does not take this argument, can only produce one image
            pipeline_args['num_images_per_prompt'] = batch_size
        else:
            mock_batching = batch_size > 1

        def generate_images(**kwargs):
            if mock_batching:
                images = []
                for i in range(0, batch_size):
                    images.append(
                        _pipelines.call_pipeline(**kwargs).images[0])
                return images
            else:
                return _pipelines.call_pipeline(**kwargs).images

        pipeline_args['generator'] = \
            torch.Generator(device=self._device).manual_seed(
                _types.default(user_args.seed, _constants.DEFAULT_SEED))

        if isinstance(self._pipeline, diffusers.StableDiffusionInpaintPipelineLegacy):
            # Not necessary, will cause an error
            pipeline_args.pop('width')
            pipeline_args.pop('height')

        has_control_net = hasattr(self._pipeline, 'controlnet')

        has_t2i_adapter = hasattr(self._pipeline, 'adapter') and \
                          isinstance(self._pipeline.adapter,
                                     (diffusers.T2IAdapter, diffusers.MultiAdapter))

        sd_edit = user_args.sdxl_refiner_edit or \
                  has_control_net or has_t2i_adapter or \
                  isinstance(self._pipeline, diffusers.StableDiffusionXLInpaintPipeline)

        if has_control_net:
            pipeline_args['controlnet_conditioning_scale'] = \
                self._get_control_net_conditioning_scale()

            pipeline_args['control_guidance_start'] = \
                self._get_control_net_guidance_start()

            pipeline_args['control_guidance_end'] = \
                self._get_control_net_guidance_end()

        if has_t2i_adapter:
            pipeline_args['adapter_conditioning_scale'] = \
                self._get_adapter_conditioning_scale()

            # T2I adapters require a specific number of input channels
            # in the PIL image, or they will choke, we need to convert
            # it to a 1 channel image if the T2I Adapter model only
            # expects 1 channel

            if isinstance(self._pipeline.adapter, diffusers.T2IAdapter):
                if hasattr(self._pipeline.adapter.config, 'in_channels'):
                    if self._pipeline.adapter.config.in_channels == 1:
                        pipeline_args['image'] = pipeline_args['image'][0].convert('L')

            elif isinstance(self._pipeline.adapter, diffusers.MultiAdapter):
                for idx, adapter in enumerate(self._pipeline.adapter.adapters):
                    if hasattr(adapter.config, 'in_channels'):
                        if adapter.config.in_channels == 1:
                            pipeline_args['image'][idx] = pipeline_args['image'][idx].convert('L')

        if self._sdxl_refiner_pipeline is None:
            return PipelineWrapperResult(generate_images(
                pipeline=self._pipeline,
                prompt_weighter=self._prompt_weighter,
                device=self._device,
                **pipeline_args))

        high_noise_fraction = _types.default(user_args.sdxl_high_noise_fraction,
                                             _constants.DEFAULT_SDXL_HIGH_NOISE_FRACTION)

        if sd_edit:
            i_start = dict()
            i_end = dict()
        else:
            i_start = {'denoising_start': high_noise_fraction}
            i_end = {'denoising_end': high_noise_fraction}

        image = _pipelines.call_pipeline(pipeline=self._pipeline,
                                         device=self._device,
                                         prompt_weighter=self._prompt_weighter,
                                         **pipeline_args,
                                         **i_end,
                                         output_type='latent').images

        pipeline_args['image'] = image

        if not isinstance(self._sdxl_refiner_pipeline, diffusers.StableDiffusionXLInpaintPipeline):
            # Width / Height not necessary for any other refiner
            if not (isinstance(self._pipeline, diffusers.StableDiffusionXLImg2ImgPipeline) and
                    isinstance(self._sdxl_refiner_pipeline, diffusers.StableDiffusionXLImg2ImgPipeline)):
                # Width / Height does not get passed to img2img
                pipeline_args.pop('width')
                pipeline_args.pop('height')

        # refiner does not use LoRA
        pipeline_args.pop('cross_attention_kwargs', None)

        # Or any of these
        self._pop_sdxl_conditioning_args(pipeline_args)
        pipeline_args.pop('ip_adapter_image', None)
        pipeline_args.pop('guidance_rescale', None)
        pipeline_args.pop('controlnet_conditioning_scale', None)
        pipeline_args.pop('control_guidance_start', None)
        pipeline_args.pop('control_guidance_end', None)
        pipeline_args.pop('image_guidance_scale', None)
        pipeline_args.pop('control_image', None)

        # we will handle the strength parameter if it is necessary below
        pipeline_args.pop('strength', None)

        # We do not want to override the refiner secondary prompt
        # with that of --sdxl-second-prompts by default
        pipeline_args.pop('prompt_2', None)
        pipeline_args.pop('negative_prompt_2', None)

        self._set_non_universal_pipeline_arg(self._pipeline,
                                             pipeline_args, user_args,
                                             'prompt', 'sdxl_refiner_prompt',
                                             '--sdxl-refiner-prompts',
                                             transform=lambda p: p.positive)

        self._set_non_universal_pipeline_arg(self._pipeline,
                                             pipeline_args, user_args,
                                             'negative_prompt', 'sdxl_refiner_prompt',
                                             '--sdxl-refiner-prompts',
                                             transform=lambda p: p.negative)

        self._set_non_universal_pipeline_arg(self._pipeline,
                                             pipeline_args, user_args,
                                             'prompt_2', 'sdxl_refiner_second_prompt',
                                             '--sdxl-refiner-second-prompts',
                                             transform=lambda p: p.positive)

        self._set_non_universal_pipeline_arg(self._pipeline,
                                             pipeline_args, user_args,
                                             'negative_prompt_2', 'sdxl_refiner_second_prompt',
                                             '--sdxl-refiner-second-prompts',
                                             transform=lambda p: p.negative)

        self._get_sdxl_conditioning_args(self._sdxl_refiner_pipeline,
                                         pipeline_args, user_args,
                                         user_prefix='refiner')

        self._set_non_universal_pipeline_arg(self._pipeline,
                                             pipeline_args, user_args,
                                             'guidance_rescale', 'sdxl_refiner_guidance_rescale',
                                             '--sdxl-refiner-guidance-rescales')

        if user_args.sdxl_refiner_inference_steps is not None:
            pipeline_args['num_inference_steps'] = user_args.sdxl_refiner_inference_steps

        if user_args.sdxl_refiner_guidance_scale is not None:
            pipeline_args['guidance_scale'] = user_args.sdxl_refiner_guidance_scale

        if user_args.sdxl_refiner_guidance_rescale is not None:
            pipeline_args['guidance_rescale'] = user_args.sdxl_refiner_guidance_rescale

        if user_args.sdxl_refiner_clip_skip is not None:
            pipeline_args['clip_skip'] = user_args.sdxl_refiner_clip_skip

        if sd_edit:
            strength = float(decimal.Decimal('1.0') - decimal.Decimal(str(high_noise_fraction)))

            if strength <= 0.0:
                strength = 0.2
                _messages.log(f'Refiner edit mode image seed strength (1.0 - high-noise-fraction) '
                              f'was calculated at <= 0.0, defaulting to {strength}',
                              level=_messages.WARNING)
            else:
                _messages.log(f'Running refiner in edit mode with '
                              f'refiner image seed strength = {strength}, IE: (1.0 - high-noise-fraction)')

            inference_steps = pipeline_args.get('num_inference_steps')

            if (strength * inference_steps) < 1.0:
                strength = 1.0 / inference_steps
                _messages.log(
                    f'Refiner edit mode image seed strength (1.0 - high-noise-fraction) * inference-steps '
                    f'was calculated at < 1, defaulting to (1.0 / inference-steps): {strength}',
                    level=_messages.WARNING)

            pipeline_args['strength'] = strength

        return PipelineWrapperResult(
            _pipelines.call_pipeline(
                pipeline=self._sdxl_refiner_pipeline,
                device=self._device,
                prompt_weighter=self._prompt_weighter,
                **pipeline_args, **i_start).images)

    def recall_main_pipeline(self) -> _pipelines.PipelineCreationResult:
        """
        Fetch the last used main pipeline creation result, possibly the pipeline
        will be recreated if no longer in the in memory cache. If there is no
        pipeline currently created, which will be the case if an image was
        never generated yet, :py:exc:`RuntimeError` will be raised.

        :raises RuntimeError:

        :return: :py:class:`dgenerate.pipelinewrapper.PipelineCreationResult`
        """

        if self._recall_main_pipeline is None:
            raise RuntimeError('Cannot recall main pipeline as one has not been created.')

        return self._recall_main_pipeline()

    def recall_refiner_pipeline(self) -> _pipelines.PipelineCreationResult:
        """
        Fetch the last used refiner pipeline creation result, possibly the
        pipeline will be recreated if no longer in the in memory cache.
        If there is no refiner pipeline currently created, which will be the
        case if an image was never generated yet or a refiner model was not
        specified, :py:exc:`RuntimeError` will be raised.

        :raises RuntimeError:

        :return: :py:class:`dgenerate.pipelinewrapper.PipelineCreationResult`
        """

        if self._recall_refiner_pipeline is None:
            raise RuntimeError('Cannot recall refiner pipeline as one has not been created.')

        return self._recall_refiner_pipeline()

    def _lazy_init_pipeline(self, pipeline_type):
        if self._pipeline is not None:
            if self._pipeline_type == pipeline_type:
                return False

        self._pipeline_type = pipeline_type

        self._recall_main_pipeline = None
        self._recall_refiner_pipeline = None

        if self._model_type == _enums.ModelType.FLAX:

            if self._pipeline_type != _enums.PipelineType.TXT2IMG and self._control_net_uris:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Inpaint and Img2Img not supported for flax with ControlNet.')

            self._recall_main_pipeline = _pipelines.FlaxPipelineFactory(
                pipeline_type=pipeline_type,
                model_path=self._model_path,
                model_type=self._model_type,
                revision=self._revision,
                dtype=self._dtype,
                unet_uri=self._unet_uri,
                vae_uri=self._vae_uri,
                control_net_uris=self._control_net_uris,
                text_encoder_uris=self._text_encoder_uris,
                scheduler=self._scheduler,
                safety_checker=self._safety_checker,
                auth_token=self._auth_token,
                local_files_only=self._local_files_only,
                extra_modules=self._model_extra_modules)

            creation_result = self._recall_main_pipeline()
            self._pipeline = creation_result.pipeline
            self._flax_params = creation_result.flax_params
            self._parsed_control_net_uris = creation_result.parsed_control_net_uris

        elif self._model_type == _enums.ModelType.TORCH_S_CASCADE:

            if self._s_cascade_decoder_uri is None:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Stable Cascade must be used with a decoder model.')

            if not (_pipelines.scheduler_is_help(self._s_cascade_decoder_scheduler)
                    or _pipelines.text_encoder_is_help(self._second_text_encoder_uris)):
                # Don't load this up if were just going to be getting
                # information about compatible schedulers for the refiner
                self._recall_main_pipeline = _pipelines.TorchPipelineFactory(
                    pipeline_type=pipeline_type,
                    model_path=self._model_path,
                    model_type=self._model_type,
                    subfolder=self._subfolder,
                    revision=self._revision,
                    variant=self._variant,
                    dtype=self._dtype,
                    unet_uri=self._unet_uri,
                    vae_uri=self._vae_uri,
                    lora_uris=self._lora_uris,
                    scheduler=self._scheduler,
                    safety_checker=self._safety_checker,
                    auth_token=self._auth_token,
                    device=self._device,
                    sequential_cpu_offload=self._model_sequential_offload,
                    model_cpu_offload=self._model_cpu_offload,
                    local_files_only=self._local_files_only,
                    extra_modules=self._model_extra_modules,
                    vae_tiling=self._vae_tiling,
                    vae_slicing=self._vae_slicing)
                creation_result = self._recall_main_pipeline()
                self._pipeline = creation_result.pipeline

            self._recall_s_cascade_decoder_pipeline = _pipelines.TorchPipelineFactory(
                pipeline_type=_enums.PipelineType.TXT2IMG,
                model_path=self._parsed_s_cascade_decoder_uri.model,
                model_type=_enums.ModelType.TORCH_S_CASCADE_DECODER,
                subfolder=self._parsed_s_cascade_decoder_uri.subfolder,
                revision=self._parsed_s_cascade_decoder_uri.revision,
                unet_uri=self._second_unet_uri,
                text_encoder_uris=self._second_text_encoder_uris,

                variant=self._parsed_s_cascade_decoder_uri.variant if
                self._parsed_s_cascade_decoder_uri.variant is not None else self._variant,

                dtype=self._parsed_s_cascade_decoder_uri.dtype if
                self._parsed_s_cascade_decoder_uri.dtype is not None else self._dtype,

                scheduler=self._scheduler if
                self._s_cascade_decoder_scheduler is None else self._s_cascade_decoder_scheduler,

                safety_checker=self._safety_checker,
                extra_modules=self._second_model_extra_modules,
                auth_token=self._auth_token,
                local_files_only=self._local_files_only,
                vae_tiling=self._vae_tiling,
                vae_slicing=self._vae_slicing,
                model_cpu_offload=self._s_cascade_decoder_cpu_offload,
                sequential_cpu_offload=self._s_cascade_decoder_sequential_offload)

            creation_result = self._recall_s_cascade_decoder_pipeline()
            self._s_cascade_decoder_pipeline = creation_result.pipeline

        elif self._sdxl_refiner_uri is not None:

            if not (_pipelines.scheduler_is_help(self._sdxl_refiner_scheduler)
                    or _pipelines.text_encoder_is_help(self._second_text_encoder_uris)):
                # Don't load this up if were just going to be getting
                # information about compatible schedulers for the refiner
                self._recall_main_pipeline = _pipelines.TorchPipelineFactory(
                    pipeline_type=pipeline_type,
                    model_path=self._model_path,
                    model_type=self._model_type,
                    subfolder=self._subfolder,
                    revision=self._revision,
                    variant=self._variant,
                    dtype=self._dtype,
                    unet_uri=self._unet_uri,
                    vae_uri=self._vae_uri,
                    lora_uris=self._lora_uris,
                    ip_adapter_uris=self._ip_adapter_uris,
                    textual_inversion_uris=self._textual_inversion_uris,
                    text_encoder_uris=self._text_encoder_uris,
                    control_net_uris=self._control_net_uris,
                    t2i_adapter_uris=self._t2i_adapter_uris,
                    scheduler=self._scheduler,
                    safety_checker=self._safety_checker,
                    auth_token=self._auth_token,
                    device=self._device,
                    local_files_only=self._local_files_only,
                    extra_modules=self._model_extra_modules,
                    vae_tiling=self._vae_tiling,
                    vae_slicing=self._vae_slicing,
                    model_cpu_offload=self._model_cpu_offload,
                    sequential_cpu_offload=self._model_sequential_offload)

                creation_result = self._recall_main_pipeline()
                self._pipeline = creation_result.pipeline
                self._parsed_control_net_uris = creation_result.parsed_control_net_uris
                self._parsed_t2i_adapter_uris = creation_result.parsed_t2i_adapter_uris

            refiner_pipeline_type = _enums.PipelineType.IMG2IMG if pipeline_type is _enums.PipelineType.TXT2IMG else pipeline_type

            if self._pipeline is not None:

                refiner_extra_modules = {'vae': self._pipeline.vae,
                                         'text_encoder_2': self._pipeline.text_encoder_2}

                if self._second_model_extra_modules is not None:
                    refiner_extra_modules.update(self._second_model_extra_modules)

            else:
                refiner_extra_modules = self._second_model_extra_modules

            self._recall_refiner_pipeline = _pipelines.TorchPipelineFactory(
                pipeline_type=refiner_pipeline_type,
                model_path=self._parsed_sdxl_refiner_uri.model,
                model_type=_enums.ModelType.TORCH_SDXL,
                subfolder=self._parsed_sdxl_refiner_uri.subfolder,
                revision=self._parsed_sdxl_refiner_uri.revision,
                unet_uri=self._second_unet_uri,
                text_encoder_uris=self._second_text_encoder_uris,

                variant=self._parsed_sdxl_refiner_uri.variant if
                self._parsed_sdxl_refiner_uri.variant is not None else self._variant,

                dtype=self._parsed_sdxl_refiner_uri.dtype if
                self._parsed_sdxl_refiner_uri.dtype is not None else self._dtype,

                scheduler=self._scheduler if
                self._sdxl_refiner_scheduler is None else self._sdxl_refiner_scheduler,

                safety_checker=self._safety_checker,
                auth_token=self._auth_token,
                extra_modules=refiner_extra_modules,
                local_files_only=self._local_files_only,
                vae_tiling=self._vae_tiling,
                vae_slicing=self._vae_slicing,
                model_cpu_offload=self._sdxl_refiner_cpu_offload,
                sequential_cpu_offload=self._sdxl_refiner_sequential_offload
            )

            self._sdxl_refiner_pipeline = self._recall_refiner_pipeline().pipeline
        else:
            self._recall_main_pipeline = _pipelines.TorchPipelineFactory(
                pipeline_type=pipeline_type,
                model_path=self._model_path,
                model_type=self._model_type,
                subfolder=self._subfolder,
                revision=self._revision,
                variant=self._variant,
                dtype=self._dtype,
                unet_uri=self._unet_uri,
                vae_uri=self._vae_uri,
                lora_uris=self._lora_uris,
                ip_adapter_uris=self._ip_adapter_uris,
                textual_inversion_uris=self._textual_inversion_uris,
                text_encoder_uris=self._text_encoder_uris,
                control_net_uris=self._control_net_uris,
                t2i_adapter_uris=self._t2i_adapter_uris,
                scheduler=self._scheduler,
                safety_checker=self._safety_checker,
                auth_token=self._auth_token,
                device=self._device,
                sequential_cpu_offload=self._model_sequential_offload,
                model_cpu_offload=self._model_cpu_offload,
                local_files_only=self._local_files_only,
                extra_modules=self._model_extra_modules,
                vae_tiling=self._vae_tiling,
                vae_slicing=self._vae_slicing)

            creation_result = self._recall_main_pipeline()
            self._pipeline = creation_result.pipeline
            self._parsed_control_net_uris = creation_result.parsed_control_net_uris
            self._parsed_t2i_adapter_uris = creation_result.parsed_t2i_adapter_uris

        return True

    def __call__(self, args: DiffusionArguments | None = None, **kwargs) -> PipelineWrapperResult:
        """
        Call the pipeline and generate a result.

        :param args: Optional :py:class:`.DiffusionArguments`

        :param kwargs: See :py:meth:`.DiffusionArguments.get_pipeline_wrapper_kwargs`,
            any keyword arguments given here will override values derived from the
            :py:class:`.DiffusionArguments` object given to the *args* parameter.

        :raises InvalidModelFileError:
        :raises UnsupportedPipelineConfigError:
        :raises InvalidModelUriError:
        :raises InvalidSchedulerNameError:
        :raises dgenerate.OutOfMemoryError:


        :return: :py:class:`.PipelineWrapperResult`
        """

        copy_args = DiffusionArguments()

        if args is not None:
            copy_args.set_from(args)

        copy_args.set_from(kwargs, missing_value_throws=False)

        _messages.debug_log(f'Calling Pipeline Wrapper: "{self}"')
        _messages.debug_log(f'Pipeline Wrapper Args: ',
                            lambda: _textprocessing.debug_format_args(
                                copy_args.get_pipeline_wrapper_kwargs()))

        _cache.enforce_cache_constraints()

        pipeline_type = copy_args.determine_pipeline_type()

        if self._prompt_weighter_uri:
            self._prompt_weighter = self._prompt_weighter_loader.load(
                self._prompt_weighter_uri,
                model_type=self.model_type,
                pipeline_type=pipeline_type,
                dtype=self._dtype)

        loaded_new = self._lazy_init_pipeline(pipeline_type)

        if loaded_new:
            _cache.enforce_cache_constraints()

        pipeline_args = \
            self._get_pipeline_defaults(user_args=copy_args)

        if self._model_type == _enums.ModelType.FLAX:
            result = self._call_flax(pipeline_args=pipeline_args,
                                     user_args=copy_args)
        elif self.model_type == _enums.ModelType.TORCH_S_CASCADE:
            result = self._call_torch_s_cascade(
                pipeline_args=pipeline_args,
                user_args=copy_args)
        else:
            result = self._call_torch(pipeline_args=pipeline_args,
                                      user_args=copy_args)

        return result


__all__ = _types.module_all()
