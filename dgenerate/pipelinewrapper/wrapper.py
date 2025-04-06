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
import contextlib
import decimal
import inspect
import math
import typing

import PIL.Image
import diffusers
import torch
import importlib.util
import dgenerate.image as _image
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.prompt as _prompt
import dgenerate.promptweighters as _promptweighters
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.pipelinewrapper.arguments import DiffusionArguments
import dgenerate.pipelinewrapper.util as _util
import dgenerate.extras.asdff.base as _asdff_base
import dgenerate.extras.hidiffusion as _hidiffusion
import dgenerate.extras.teacache.teacache_flux as _teacache_flux
from dgenerate.extras.ras import sd3_ras_context as _sd3_ras_context
from dgenerate.extras.ras import RASArgs as _RASArgs
import dgenerate.pipelinewrapper.help as _help
import dgenerate.pipelinewrapper.schedulers as _schedulers
import dgenerate.memory as _memory
import dgenerate.memoize as _memoize
import dgenerate.torchutil as _torchutil
import DeepCache as _deepcache


@contextlib.contextmanager
def _deep_cache_context(pipeline,
                        cache_interval: int = 5,
                        cache_branch_id: int = 1,
                        skip_mode: str = 'uniform',
                        enabled: bool = False):
    if enabled:
        _messages.debug_log(
            f'Enabling DeepCache on pipeline: {pipeline.__class__.__name__}')
        helper = _deepcache.DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(
            cache_interval=cache_interval,
            cache_branch_id=cache_branch_id,
            skip_mode=skip_mode
        )
        helper.enable()

        try:
            yield
        finally:
            _messages.debug_log(
                f'Disabling DeepCache on pipeline: {pipeline.__class__.__name__}')
            helper.disable()
    else:
        yield


class DiffusionArgumentsHelpException(Exception):
    """
    Thrown when a :py:class:`DiffusionArguments` attribute that supports
    passing a help request value (such as :py:attr:`DiffusionArguments.scheduler_uri`)
    is passed its help value.

    This exception returns the help string to the caller.
    """
    pass


def _enforce_cache_constraints():
    if _memory.memory_constraints(_constants.PIPELINE_WRAPPER_CACHE_GC_CONSTRAINTS):
        _messages.debug_log(f'dgenerate.pipelinewrapper.constants.PIPELINE_WRAPPER_CACHE_GC_CONSTRAINTS '
                            f'{_constants.PIPELINE_WRAPPER_CACHE_GC_CONSTRAINTS} met, '
                            f'calling {_types.fullname(_memoize.clear_object_caches)}.')

        _memoize.clear_object_caches()
        return True

    return False


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


@contextlib.contextmanager
def _hi_diffusion(pipeline, generator, enabled: bool):
    if enabled:
        _messages.debug_log(
            f'Enabling HiDiffusion on pipeline: {pipeline.__class__.__name__}')
        _hidiffusion.apply_hidiffusion(pipeline, generator=generator)
    try:
        yield
    finally:
        if enabled:
            _messages.debug_log(
                f'Disabling HiDiffusion on pipeline: {pipeline.__class__.__name__}')
            _hidiffusion.remove_hidiffusion(pipeline)


class DiffusionPipelineWrapper:
    """
    Monolithic diffusion pipelines wrapper.
    """

    __LAST_CALLED = None

    @staticmethod
    def last_called_wrapper() -> typing.Optional['DiffusionPipelineWrapper']:
        """
        Return a reference to the last :py:class:`DiffusionPipelineWrapper`
        that successfully executed an image generation.

        :return: :py:class:`DiffusionPipelineWrapper`
        """
        return DiffusionPipelineWrapper.__LAST_CALLED

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
                 second_model_unet_uri: _types.OptionalUri = None,
                 transformer_uri: _types.OptionalUri = None,
                 vae_uri: _types.OptionalUri = None,
                 lora_uris: _types.OptionalUris = None,
                 lora_fuse_scale: _types.OptionalFloat = None,
                 image_encoder_uri: _types.OptionalUri = None,
                 ip_adapter_uris: _types.OptionalUris = None,
                 textual_inversion_uris: _types.OptionalUris = None,
                 text_encoder_uris: _types.OptionalUris = None,
                 second_model_text_encoder_uris: _types.OptionalUris = None,
                 controlnet_uris: _types.OptionalUris = None,
                 t2i_adapter_uris: _types.OptionalUris = None,
                 sdxl_refiner_uri: _types.OptionalUri = None,
                 s_cascade_decoder_uri: _types.OptionalUri = None,
                 quantizer_uri: _types.OptionalUri = None,
                 second_model_quantizer_uri: _types.OptionalUri = None,
                 device: str = _torchutil.default_device(),
                 safety_checker: bool = False,
                 original_config: _types.OptionalString = None,
                 second_model_original_config: _types.OptionalString = None,
                 auth_token: _types.OptionalString = None,
                 local_files_only: bool = False,
                 model_extra_modules: dict[str, typing.Any] = None,
                 second_model_extra_modules: dict[str, typing.Any] = None,
                 model_cpu_offload: bool = False,
                 model_sequential_offload: bool = False,
                 second_model_cpu_offload: bool = False,
                 second_model_sequential_offload: bool = False,
                 prompt_weighter_loader: _promptweighters.PromptWeighterLoader | None = None,
                 adetailer_detector_uris: _types.OptionalUris = None,
                 adetailer_crop_control_image: bool = False):
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
        :param second_model_unet_uri: secondary model unet uri (SDXL Refiner, Stable Cascade decoder)
        :param transformer_uri: Optional transformer URI string for specifying a specific Transformer,
            currently this is only supported for Stable Diffusion 3 models.
        :param vae_uri: main model VAE URI string
        :param lora_uris: One or more LoRA URI strings
        :param lora_fuse_scale: Optional global LoRA fuse scale value. Once all LoRAs are merged with
            their individual scales, the merged weights will be fused into the pipeline at this scale.
            The default value is 1.0.
        :param image_encoder_uri: One or more Image Encoder URI strings,
            Image Encoders are used with IP Adapters and Stable Cascade
        :param ip_adapter_uris: One or more IP Adapter URI strings
        :param textual_inversion_uris: One or more Textual Inversion URI strings
        :param text_encoder_uris: One or more Text Encoder URIs
            ("+", or None for default. Or "null" indicating do not load) for the main model
        :param second_model_text_encoder_uris:  One or more Text Encoder URIs
            ("+", or None for default. Or "null" indicating do not load) for the secondary
            model (SDXL Refiner or Stable Cascade decoder)
        :param controlnet_uris: One or more ControlNet URI strings
        :param t2i_adapter_uris: One or more T2IAdapter URI strings
        :param sdxl_refiner_uri: SDXL Refiner model URI string
        :param s_cascade_decoder_uri: Stable Cascade decoder URI string
        :param quantizer_uri: Global --quantizer URI value
        :param second_model_quantizer_uri: Global --second-model-quantizer URI value
        :param device: Rendering device string, example: ``cuda:0`` or ``cuda``
        :param safety_checker: Use safety checker model if available? (antiquated, for SD 1/2, Deep Floyd etc.)
        :param original_config: Optional original LDM config .yaml file path when loading a single file checkpoint.
        :param second_model_original_config: Optional original LDM config .yaml file path when loading a single file checkpoint
            for the secondary model (SDXL Refiner, Stable Cascade Decoder).
        :param auth_token: huggingface authentication token.
        :param local_files_only: Do not attempt to download files from huggingface?
        :param model_extra_modules: Raw extra diffusers modules for the main pipeline
        :param second_model_extra_modules: Raw extra diffusers modules for the secondary pipeline (SDXL Refiner, Stable Cascade decoder)
        :param model_cpu_offload: Use model CPU offloading for the main pipeline via the accelerate module?
        :param model_sequential_offload: Use sequential CPU offloading for the main pipeline via the accelerate module?
        :param second_model_cpu_offload: Use CPU offloading for the SDXL Refiner or Stable Cascade Decoder  via the accelerate module?
        :param second_model_sequential_offload: Use sequential CPU offloading for the SDXL Refiner or Stable Cascade Decoder via the accelerate module?
        :param prompt_weighter_loader: Plugin loader for prompt weighter implementations, if you pass ``None`` a default instance will be created.
        :param adetailer_detector_uris: adetailer subject detection model URIs, specifying this argument indicates ``img2img`` mode implicitly,
            the pipeline wrapper will accept a single image and preform the adetailer inpainting algorithm on it using the provided
            detector URIs.
        :param adetailer_crop_control_image: Should adetailer crop any provided ControlNet control image
            in the same way that it crops the generated mask to the detection area? Otherwise,
            use the full control image resized down to the size of the detection area. If you enable
            this and your control image is not the same size as your input image, a warning will be
            issued and resizing will be used instead of cropping.

        :raises UnsupportedPipelineConfigError:
        :raises InvalidModelUriError:
        """

        __locals = locals()

        __locals.pop('self')

        for name, value in __locals.items():
            if name.endswith('_uris') and isinstance(value, str):
                __locals[name] = [value]

        self._init(**__locals)

    def _init(
            self,
            model_path: _types.Path,
            model_type: _enums.ModelType = _enums.ModelType.TORCH,
            revision: _types.OptionalName = None,
            variant: _types.OptionalName = None,
            subfolder: _types.OptionalName = None,
            dtype: _enums.DataType = _enums.DataType.AUTO,
            unet_uri: _types.OptionalUri = None,
            second_model_unet_uri: _types.OptionalUri = None,
            transformer_uri: _types.OptionalUri = None,
            vae_uri: _types.OptionalUri = None,
            lora_uris: _types.OptionalUris = None,
            lora_fuse_scale: _types.OptionalFloat = None,
            image_encoder_uri: _types.OptionalUri = None,
            ip_adapter_uris: _types.OptionalUris = None,
            textual_inversion_uris: _types.OptionalUris = None,
            text_encoder_uris: _types.OptionalUris = None,
            second_model_text_encoder_uris: _types.OptionalUris = None,
            controlnet_uris: _types.OptionalUris = None,
            t2i_adapter_uris: _types.OptionalUris = None,
            sdxl_refiner_uri: _types.OptionalUri = None,
            s_cascade_decoder_uri: _types.OptionalUri = None,
            quantizer_uri: _types.OptionalUri = None,
            second_model_quantizer_uri: _types.OptionalUri = None,
            device: str = _torchutil.default_device(),
            safety_checker: bool = False,
            original_config: _types.OptionalString = None,
            second_model_original_config: _types.OptionalString = None,
            auth_token: _types.OptionalString = None,
            local_files_only: bool = False,
            model_extra_modules: dict[str, typing.Any] = None,
            second_model_extra_modules: dict[str, typing.Any] = None,
            model_cpu_offload: bool = False,
            model_sequential_offload: bool = False,
            second_model_cpu_offload: bool = False,
            second_model_sequential_offload: bool = False,
            prompt_weighter_loader: _promptweighters.PromptWeighterLoader | None = None,
            adetailer_detector_uris: _types.OptionalUris = None,
            adetailer_crop_control_image: bool = False
    ):
        # Check that model_path is provided
        if model_path is None:
            raise ValueError('model_path must be specified')

        # Check for valid device string
        if not _torchutil.is_valid_device_string(device):
            raise _pipelines.UnsupportedPipelineConfigError(
                'device must be "cuda" (optionally with a device ordinal "cuda:N") or "cpu", '
                'or other device supported by torch.')

        # Offload options should not be enabled simultaneously
        if model_cpu_offload and model_sequential_offload:
            raise _pipelines.UnsupportedPipelineConfigError(
                '"model_cpu_offload" and "model_sequential_offload" may not be enabled simultaneously.'
            )

        if second_model_cpu_offload and second_model_sequential_offload:
            raise _pipelines.UnsupportedPipelineConfigError(
                '"second_model_cpu_offload" and "second_model_sequential_offload" '
                'may not be enabled simultaneously.'
            )

        # Text encoder check
        if not sdxl_refiner_uri and not s_cascade_decoder_uri:
            if second_model_text_encoder_uris:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Cannot use "second_model_text_encoder_uris" if "sdxl_refiner_uri" '
                    'or "s_cascade_decoder_uri" is not specified.'
                )

        # Incompatible combinations
        if controlnet_uris and t2i_adapter_uris:
            raise _pipelines.UnsupportedPipelineConfigError(
                'Cannot use "controlnet_uris" and "t2i_adapter_uris" together.'
            )

        if image_encoder_uri and not ip_adapter_uris and model_type != _enums.ModelType.TORCH_S_CASCADE:
            raise _pipelines.UnsupportedPipelineConfigError(
                'Cannot use "image_encoder_uri" without "ip_adapter_uris" '
                'if "model_type" is not TORCH_S_CASCADE.'
            )

        if not _util.is_single_file_model_load(model_path):
            if original_config:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'You cannot specify "original_config" when the main '
                    'model is not a a single file checkpoint.'
                )

        if second_model_original_config:
            if not sdxl_refiner_uri and not s_cascade_decoder_uri:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'You cannot specify "second_model_original_config" '
                    'without "sdxl_refiner_uri" or "s_cascade_decoder_uri".'
                )

            if sdxl_refiner_uri and \
                    not _util.is_single_file_model_load(
                        _uris.SDXLRefinerUri.parse(sdxl_refiner_uri).model):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'You cannot specify "second_model_original_config" '
                    'when the "sdxl_refiner_uri" model is not a '
                    'single file checkpoint.'
                )
            if s_cascade_decoder_uri and \
                    not _util.is_single_file_model_load(
                        _uris.SCascadeDecoderUri.parse(s_cascade_decoder_uri).model):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'You cannot specify "second_model_original_config" '
                    'when the "s_cascade_decoder_uri" model is not a '
                    'single file checkpoint.'
                )

        if sdxl_refiner_uri is not None:
            if not (_enums.model_type_is_sdxl(model_type) or
                    _enums.model_type_is_kolors(model_type)):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Only Stable Diffusion XL models support refiners, '
                    'please use model_type "torch-sdxl" if you are trying to load an sdxl model.'
                )

        if s_cascade_decoder_uri is not None:
            if not _enums.model_type_is_s_cascade(model_type):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Only Stable Cascade models support decoders, '
                    'please use model_type "torch-s-cascade" if you are trying to load an Stable Cascade model.'
                )

        if transformer_uri:
            if not _enums.model_type_is_sd3(model_type) and not _enums.model_type_is_flux(model_type):
                raise _pipelines.UnsupportedPipelineConfigError(
                    '--transformer is only supported for --model-type torch-sd3 and torch-flux.')

        if adetailer_detector_uris and model_type not in {
            _enums.ModelType.TORCH,
            _enums.ModelType.TORCH_SDXL,
            _enums.ModelType.TORCH_KOLORS,
            _enums.ModelType.TORCH_SD3,
            _enums.ModelType.TORCH_FLUX,
            _enums.ModelType.TORCH_FLUX_FILL
        }:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'--adetailer-detectors is only compatible with '
                f'--model-type torch, torch-sdxl, torch-kolors, torch-sd3, and torch-flux')

        if quantizer_uri is not None:
            try:
                _uris.get_quantizer_uri_class(quantizer_uri)
            except ValueError as e:
                raise _pipelines.UnsupportedPipelineConfigError(str(e)) from e

        if second_model_quantizer_uri is not None:
            try:
                _uris.get_quantizer_uri_class(second_model_quantizer_uri)
            except ValueError as e:
                raise _pipelines.UnsupportedPipelineConfigError(str(e)) from e

        self._quantizer_uri = quantizer_uri
        self._second_model_quantizer_uri = second_model_quantizer_uri
        self._subfolder = subfolder
        self._device = device
        self._model_type = _enums.get_model_type_enum(model_type)
        self._model_path = model_path
        self._pipeline = None
        self._revision = revision
        self._variant = variant
        self._dtype = _enums.get_data_type_enum(dtype)
        self._device = device
        self._unet_uri = unet_uri
        self._second_model_unet_uri = second_model_unet_uri
        self._transformer_uri = transformer_uri
        self._image_encoder_uri = image_encoder_uri
        self._vae_uri = vae_uri
        self._safety_checker = safety_checker

        self._original_config = original_config
        self._second_model_original_config = second_model_original_config

        self._second_model_cpu_offload = second_model_cpu_offload
        self._second_model_sequential_offload = second_model_sequential_offload

        self._lora_uris = lora_uris
        self._lora_fuse_scale = lora_fuse_scale
        self._ip_adapter_uris = ip_adapter_uris
        self._textual_inversion_uris = textual_inversion_uris
        self._text_encoder_uris = text_encoder_uris
        self._second_model_text_encoder_uris = second_model_text_encoder_uris
        self._controlnet_uris = controlnet_uris
        self._t2i_adapter_uris = t2i_adapter_uris
        self._parsed_controlnet_uris = []
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
        self._model_cpu_offload = model_cpu_offload
        self._model_sequential_offload = model_sequential_offload

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

        self._prompt_weighter_cache = dict()

        self._adetailer_detector_uris = adetailer_detector_uris
        self._parsed_adetailer_detector_uris = None

        self._adetailer_crop_control_image = adetailer_crop_control_image

        if adetailer_detector_uris:
            self._parsed_adetailer_detector_uris = []
            for adetailer_detector_uri in adetailer_detector_uris:
                self._parsed_adetailer_detector_uris.append(
                    _uris.AdetailerDetectorUri.parse(adetailer_detector_uri))

    @property
    def prompt_weighter_loader(self) -> _promptweighters.PromptWeighterLoader:
        """
        Current prompt weighter loader.
        """
        return self._prompt_weighter_loader

    @property
    def local_files_only(self) -> bool:
        """
        Currently set value for ``local_files_only``.
        """
        return self._local_files_only

    @property
    def revision(self) -> _types.OptionalName:
        """
        Currently set ``--revision`` for the main model or ``None``.
        """
        return self._revision

    @property
    def safety_checker(self) -> bool:
        """
        Safety checker enabled status.
        """
        return self._safety_checker

    @property
    def variant(self) -> _types.OptionalName:
        """
        Currently set ``--variant`` for the main model or ``None``.
        """
        return self._variant

    @property
    def dtype(self) -> _enums.DataType:
        """
        Currently set ``--dtype`` enum value for the main model.
        """
        return self._dtype

    @property
    def dtype_string(self) -> str:
        """
        Currently set ``--dtype`` string value for the main model.
        """
        return _enums.get_data_type_string(self._dtype)

    @property
    def textual_inversion_uris(self) -> _types.OptionalUris:
        """
        List of supplied ``--textual-inversions`` URI strings or an empty list.
        """
        return list(self._textual_inversion_uris) if self._textual_inversion_uris else []

    @property
    def controlnet_uris(self) -> _types.OptionalUris:
        """
        List of supplied ``--control-nets`` URI strings or an empty list.
        """
        return list(self._controlnet_uris) if self._controlnet_uris else []

    @property
    def t2i_adapter_uris(self) -> _types.OptionalUris:
        """
        List of supplied ``--t2i-adapters`` URI strings or an empty list.
        """
        return list(self._t2i_adapter_uris) if self._t2i_adapter_uris else []

    @property
    def ip_adapter_uris(self) -> _types.OptionalUris:
        """
        List of supplied ``--ip-adapters`` URI strings or an empty list.
        """
        return list(self._ip_adapter_uris) if self._ip_adapter_uris else []

    @property
    def text_encoder_uris(self) -> _types.OptionalUris:
        """
        List of supplied ``--text-encoders`` URI strings or an empty list.
        """
        return list(self._text_encoder_uris) if self._text_encoder_uris else []

    @property
    def second_model_text_encoder_uris(self) -> _types.OptionalUris:
        """
        List of supplied ``--second-model-text-encoders`` URI strings or an empty list.
        """
        return list(self._second_model_text_encoder_uris) if self._second_model_text_encoder_uris else []

    @property
    def adetailer_detector_uris(self) -> _types.OptionalUris:
        """
        List of supplied ``--adetailer-detectors`` URI strings or an empty list.
        """
        return list(self._adetailer_detector_uris) if self._adetailer_detector_uris else []

    @property
    def adetailer_crop_control_image(self) -> bool:
        """
        Should adetailer crop any provided control image in the same way that it crops the
        generated mask to the detection area? Otherwise, use the full control image
        resized down to the size of the detection area.
        """
        return self._adetailer_crop_control_image

    @adetailer_crop_control_image.setter
    def adetailer_crop_control_image(self, value: bool):
        """
        Should adetailer crop any provided control image in the same way that it crops the
        generated mask to the detection area? Otherwise, use the full control image
        resized down to the size of the detection area.
        """
        self._adetailer_crop_control_image = value

    @property
    def device(self) -> _types.Name:
        """
        Currently set ``--device`` string.
        """
        return self._device

    @property
    def model_path(self) -> _types.Path:
        """
        Model path for the main model.
        """
        return self._model_path

    @property
    def sdxl_refiner_uri(self) -> _types.OptionalUri:
        """
        Model URI for the SDXL refiner or ``None``.
        """
        return self._sdxl_refiner_uri

    @property
    def s_cascade_decoder_uri(self) -> _types.OptionalUri:
        """
        Model URI for the Stable Cascade decoder or ``None``.
        """
        return self._s_cascade_decoder_uri

    @property
    def transformer_uri(self) -> _types.OptionalUri:
        """
        Model URI for the SD3 Transformer or ``None``.
        """
        return self._transformer_uri

    @property
    def model_type(self) -> _enums.ModelType:
        """
        Currently set ``--model-type`` enum value.
        """
        return self._model_type

    @property
    def model_type_string(self) -> str:
        """
        Currently set ``--model-type`` string value.
        """
        return _enums.get_model_type_string(self._model_type)

    @property
    def subfolder(self) -> _types.OptionalName:
        """
        Selected model ``--subfolder`` for the main model, (remote repo subfolder or local) or ``None``.
        """
        return self._subfolder

    @property
    def vae_uri(self) -> _types.OptionalUri:
        """
        Selected ``--vae`` uri for the main model or ``None``.
        """
        return self._vae_uri

    @property
    def image_encoder_uri(self) -> _types.OptionalUri:
        """
        Selected ``--image-encoder`` uri for the main model or ``None``.
        """
        return self._image_encoder_uri

    @property
    def unet_uri(self) -> _types.OptionalUri:
        """
        Selected ``--unet`` uri for the main model or ``None``.
        """
        return self._unet_uri

    @property
    def second_model_unet_uri(self) -> _types.OptionalUri:
        """
        Selected ``--second-model-unet`` uri for the SDXL refiner or Stable Cascade decoder model or ``None``.
        """
        return self._second_model_unet_uri

    @property
    def lora_uris(self) -> _types.OptionalUris:
        """
        List of supplied ``--loras`` uri strings or an empty list.
        """
        return list(self._lora_uris) if self._lora_uris else []

    @property
    def lora_fuse_scale(self) -> float:
        """
        Supplied ``--lora-fuse-scale`` value.
        """
        return self._lora_fuse_scale

    @property
    def auth_token(self) -> _types.OptionalString:
        """
        Current ``--auth-token`` value or ``None``.
        """
        return self._auth_token

    @property
    def model_sequential_offload(self) -> bool:
        """
        Current ``--model-sequential-offload`` value.
        """
        return self._model_sequential_offload

    @property
    def model_cpu_offload(self) -> bool:
        """
        Current ``--model-cpu-offload`` value.
        """
        return self._model_cpu_offload

    @property
    def second_model_sequential_offload(self) -> bool:
        """
        Current ``--second-model-sequential-offload`` value.
        """
        return self._second_model_sequential_offload

    @property
    def second_model_cpu_offload(self) -> bool:
        """
        Current ``--second-model-cpu-offload`` value.
        """
        return self._second_model_cpu_offload

    @property
    def quantizer_uri(self) -> _types.OptionalUri:
        """
        Current ``--quantizer`` value.
        """
        return self._quantizer_uri

    @property
    def second_model_quantizer_uri(self) -> _types.OptionalUri:
        """
        Current ``--second-model-quantizer`` value.
        """
        return self._second_model_quantizer_uri

    @property
    def original_config(self) -> _types.OptionalPath:
        """
        Current ``--original-config`` value.
        """
        return self._original_config

    @property
    def second_model_original_config(self) -> _types.OptionalPath:
        """
        Current ``--second-model-original-config`` value.
        """
        return self._second_model_original_config

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
        Reconstruct dgenerate's command line arguments from a particular set of pipeline wrapper call arguments.
        
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

        if self.original_config:
            opts.append(('--original-config', self.original_config))

        if self.second_model_original_config:
            opts.append(('--second-model-original-config', self.second_model_original_config))

        if self.quantizer_uri:
            opts.append(('--quantizer', self.quantizer_uri))

        if self.second_model_quantizer_uri:
            opts.append(('--second-model-quantizer', self.quantizer_uri))

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

        if args.prompt_weighter_uri:
            opts.append(('--prompt-weighter', args.prompt_weighter_uri))

        if args.second_model_prompt_weighter_uri:
            opts.append(('--second-model-prompt-weighter', args.second_model_prompt_weighter_uri))

        if args.prompt is not None:
            opts.append(('--prompts', args.prompt))

        if args.second_prompt is not None:
            opts.append(('--second-prompts', args.second_prompt))

        if args.third_prompt is not None:
            opts.append(('--third-prompts', args.third_prompt))

        if args.second_model_prompt is not None:
            opts.append(('--second-model-prompts', args.second_model_prompt))

        if args.second_model_second_prompt is not None:
            opts.append(('--second-model-second-prompts', args.second_model_second_prompt))

        if args.max_sequence_length is not None:
            opts.append(('--max-sequence-length', args.max_sequence_length))

        if args.clip_skip is not None:
            opts.append(('--clip-skips', args.clip_skip))

        if args.sdxl_refiner_clip_skip is not None:
            opts.append(('--sdxl-refiner-clip-skips', args.sdxl_refiner_clip_skip))

        if self._adetailer_detector_uris:
            opts.append(('--adetailer-detectors', self._adetailer_detector_uris))

        if args.adetailer_index_filter is not None:
            opts.append(('--adetailer-index-filter',
                         ' '.join(str(i) for i in args.adetailer_index_filter)))

        if args.adetailer_mask_shape is not None:
            opts.append(('--adetailer-mask-shapes', args.adetailer_mask_shape))

        if args.adetailer_detector_padding is not None:
            opts.append(('--adetailer-detector-paddings',
                         _textprocessing.format_size(args.adetailer_detector_padding)))

        if args.adetailer_mask_padding is not None:
            opts.append(('--adetailer-mask-paddings',
                         _textprocessing.format_size(args.adetailer_mask_padding)))

        if args.adetailer_mask_blur is not None:
            opts.append(('--adetailer-mask-blurs', args.adetailer_mask_blur))

        if args.adetailer_mask_dilation is not None:
            opts.append(('--adetailer-mask-dilations', args.adetailer_mask_dilation))

        if self._adetailer_crop_control_image:
            opts.append(('--adetailer-crop-control-image',))

        if self._text_encoder_uris:
            opts.append(('--text-encoders', ['+' if x is None else x for x in self._text_encoder_uris]))

        if self._second_model_text_encoder_uris:
            opts.append(('--second-model-text-encoders',
                         ['+' if x is None else x for x in self._second_model_text_encoder_uris]))

        if self._s_cascade_decoder_uri is not None:
            opts.append(('--s-cascade-decoder', self._s_cascade_decoder_uri))

        if self._revision is not None and self._revision != 'main':
            opts.append(('--revision', self._revision))

        if self._variant is not None:
            opts.append(('--variant', self._variant))

        if self._subfolder is not None:
            opts.append(('--subfolder', self._subfolder))

        if self._unet_uri is not None:
            opts.append(('--unet', self._unet_uri))

        if self._second_model_unet_uri is not None:
            opts.append(('--second-model-unet', self._second_model_unet_uri))

        if self._transformer_uri is not None:
            opts.append(('--transformer', self._transformer_uri))

        if self._vae_uri is not None:
            opts.append(('--vae', self._vae_uri))

        if args.vae_tiling:
            opts.append(('--vae-tiling',))

        if args.vae_slicing:
            opts.append(('--vae-slicing',))

        if self._model_cpu_offload:
            opts.append(('--model-cpu-offload',))

        if self._model_sequential_offload:
            opts.append(('--model-sequential-offload',))

        if self._second_model_cpu_offload:
            opts.append(('--second-model-cpu-offload',))

        if self._second_model_sequential_offload:
            opts.append(('--second-model-sequential-offload',))

        if self._sdxl_refiner_uri is not None:
            opts.append(('--sdxl-refiner', self._sdxl_refiner_uri))

        if args.sdxl_refiner_edit:
            opts.append(('--sdxl-refiner-edit',))

        if self._lora_uris:
            opts.append(('--loras', self._lora_uris))

        if self._lora_fuse_scale is not None:
            opts.append(('--lora-fuse-scale', self._lora_fuse_scale))

        if self._image_encoder_uri:
            opts.append(('--image-encoder', self._image_encoder_uri))

        if self._ip_adapter_uris:
            opts.append(('--ip-adapters', self._ip_adapter_uris))

        if self._textual_inversion_uris:
            opts.append(('--textual-inversions', self._textual_inversion_uris))

        if self._controlnet_uris:
            opts.append(('--control-nets', self._controlnet_uris))

        if self._t2i_adapter_uris:
            opts.append(('--t2i-adapters', self._t2i_adapter_uris))

        if args.sdxl_t2i_adapter_factor is not None:
            opts.append(('--sdxl-t2i-adapter-factors', args.sdxl_t2i_adapter_factor))

        if args.scheduler_uri is not None:
            opts.append(('--scheduler', args.scheduler_uri))

        if args.second_model_scheduler_uri is not None:
            if args.second_model_scheduler_uri != args.scheduler_uri:
                opts.append(('--second-model-scheduler', args.second_model_scheduler_uri))

        if args.hi_diffusion:
            opts.append(('--hi-diffusion',))

        if args.sdxl_refiner_hi_diffusion:
            opts.append(('--sdxl-refiner-hi-diffusion',))

        if args.tea_cache:
            opts.append(('--tea-cache',))

        if args.tea_cache_rel_l1_threshold is not None and \
                args.tea_cache_rel_l1_threshold != _constants.DEFAULT_TEA_CACHE_REL_L1_THRESHOLD:
            opts.append(('--tea-cache-rel-l1-thresholds', args.tea_cache_rel_l1_threshold))

        if args.ras:
            opts.append(('--ras',))

        if args.ras_index_fusion:
            opts.append(('--ras-index-fusion',))

        if args.ras_sample_ratio is not None and \
                args.ras_sample_ratio != _constants.DEFAULT_RAS_SAMPLE_RATIO:
            opts.append(('--ras-sample-ratios', args.ras_sample_ratio))

        if args.ras_high_ratio is not None and \
                args.ras_high_ratio != _constants.DEFAULT_RAS_HIGH_RATIO:
            opts.append(('--ras-high-ratios', args.ras_high_ratio))

        if args.ras_starvation_scale is not None \
                and args.ras_starvation_scale != _constants.DEFAULT_RAS_STARVATION_SCALE:
            opts.append(('--ras-starvation-scales', args.ras_starvation_scale))

        if args.ras_error_reset_steps is not None and \
                args.ras_error_reset_steps != _constants.DEFAULT_RAS_ERROR_RESET_STEPS:
            opts.append(('--ras-error-reset-steps', args.ras_error_reset_steps))

        if args.ras_metric is not None and \
                args.ras_metric != _constants.DEFAULT_RAS_METRIC:
            opts.append(('--ras-metrics', args.ras_metric))

        if args.ras_start_step is not None and \
                args.ras_start_step != _constants.DEFAULT_RAS_START_STEP:
            opts.append(('--ras-start-steps', args.ras_start_step))

        if args.ras_end_step is not None and \
                args.ras_end_step != args.inference_steps:
            opts.append(('--ras-end-steps', args.ras_end_step))

        if args.ras_skip_num_step is not None and \
                args.ras_skip_num_step != _constants.DEFAULT_RAS_SKIP_NUM_STEP:
            opts.append(('--ras-skip-num-steps', args.ras_skip_num_step))

        if args.ras_skip_num_step_length is not None and \
                args.ras_skip_num_step_length != _constants.DEFAULT_RAS_SKIP_NUM_STEP_LENGTH:
            opts.append(('--ras-skip-num-step-lengths', args.ras_skip_num_step_length))

        if args.deep_cache:
            opts.append(('--deep-cache',))

        if args.deep_cache_interval is not None and \
                args.deep_cache_interval != _constants.DEFAULT_DEEP_CACHE_INTERVAL:
            opts.append(('--deep-cache-intervals', args.deep_cache_interval))

        if args.deep_cache_branch_id is not None and \
                args.deep_cache_branch_id != _constants.DEFAULT_DEEP_CACHE_BRANCH_ID:
            opts.append(('--deep-cache-branch-ids', args.deep_cache_branch_id))

        if args.second_model_deep_cache:
            opts.append(('--second-model-deep-cache',))

        if args.second_model_deep_cache_interval is not None and \
                args.second_model_deep_cache_interval != _constants.DEFAULT_SDXL_REFINER_DEEP_CACHE_INTERVAL:
            opts.append(('--second-model-deep-cache-intervals', args.second_model_deep_cache_interval))

        if args.second_model_deep_cache_branch_id is not None and \
                args.second_model_deep_cache_branch_id != _constants.DEFAULT_SDXL_REFINER_DEEP_CACHE_BRANCH_ID:
            opts.append(('--second-model-deep-cache-branch-ids', args.second_model_deep_cache_branch_id))

        if args.pag_scale == _constants.DEFAULT_PAG_SCALE \
                and args.pag_adaptive_scale == _constants.DEFAULT_PAG_ADAPTIVE_SCALE:
            opts.append(('--pag',))
        else:
            if args.pag_scale is not None:
                opts.append(('--pag-scales', args.pag_scale))
            if args.pag_adaptive_scale is not None:
                opts.append(('--pag-adaptive-scales', args.pag_adaptive_scale))

        if args.sdxl_refiner_pag_scale == _constants.DEFAULT_SDXL_REFINER_PAG_SCALE and \
                args.sdxl_refiner_pag_adaptive_scale == _constants.DEFAULT_SDXL_REFINER_PAG_ADAPTIVE_SCALE:
            opts.append(('--sdxl-refiner-pag',))
        else:
            if args.sdxl_refiner_pag_scale is not None:
                opts.append(('--sdxl-refiner-pag-scales', args.sdxl_refiner_pag_scale))
            if args.sdxl_refiner_pag_adaptive_scale is not None:
                opts.append(('--sdxl-refiner-pag-adaptive-scales', args.sdxl_refiner_pag_adaptive_scale))

        if args.sdxl_high_noise_fraction is not None:
            opts.append(('--sdxl-high-noise-fractions', args.sdxl_high_noise_fraction))

        if args.second_model_inference_steps is not None:
            opts.append(('--second-model-inference-steps', args.second_model_inference_steps))

        if args.second_model_guidance_scale is not None:
            opts.append(('--second-model-guidance-scales', args.second_model_guidance_scale))

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
                        opts[idx] = (name, _textprocessing.shell_quote(str(value)))
                    elif isinstance(value, tuple):
                        opts[idx] = (name, _textprocessing.format_size(value))
                    else:
                        opts[idx] = (name, str(value))
                else:
                    solo_val = str(option[0])
                    if not solo_val.startswith('-'):
                        # not a solo switch option, some value
                        opts[idx] = (_textprocessing.shell_quote(solo_val),)

        return opts

    @staticmethod
    def _set_opt_value_syntax(val):
        if isinstance(val, tuple):
            return _textprocessing.format_size(val)
        if isinstance(val, str):
            return _textprocessing.shell_quote(str(val))

        try:
            val_iter = iter(val)
        except TypeError:
            return _textprocessing.shell_quote(str(val))

        return ' '.join(DiffusionPipelineWrapper._set_opt_value_syntax(v) for v in val_iter)

    @staticmethod
    def _format_option_pair(val):
        if len(val) > 1:
            opt_name, opt_value = val

            if isinstance(opt_value, _prompt.Prompt):
                header_len = len(opt_name) + 2
                prompt_text = \
                    _textprocessing.wrap(
                        _textprocessing.shell_quote(str(opt_value)),
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
        return _textprocessing.shell_quote(solo_val)

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
            ifs = int(_types.default(user_args.inference_steps, _constants.DEFAULT_INFERENCE_STEPS))
            if (strength * ifs) < 1.0:
                strength = 1.0 / ifs
                _messages.warning(
                    f'image-seed-strength * inference-steps '
                    f'was calculated at < 1, image-seed-strength defaulting to (1.0 / inference-steps): {strength}'
                )

            args['strength'] = strength

        def set_controlnet_defaults():
            control_images = user_args.control_images

            if not control_images:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Must provide control_images argument when using ControlNet models.')

            control_images_cnt = len(control_images)
            controlnet_uris_cnt = len(self._controlnet_uris)

            if control_images_cnt != controlnet_uris_cnt:
                # User provided a mismatched number of ControlNet models and control_images, behavior is undefined.
                raise _pipelines.UnsupportedPipelineConfigError(
                    f'You specified {control_images_cnt} control guidance images and '
                    f'only {controlnet_uris_cnt} ControlNet URIs. The amount of '
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

            sdxl_cn_union = _enums.model_type_is_sdxl(self._model_type) and \
                            any(p.mode is not None for p in self._parsed_controlnet_uris)

            if self._pipeline_type == _enums.PipelineType.TXT2IMG:
                if _enums.model_type_is_sd3(self._model_type):
                    # Handle SD3 model specifics for control images
                    args['control_image'] = self._sd3_force_control_to_a16(args, control_images, user_args)
                elif (_enums.model_type_is_flux(self._model_type) or
                      _enums.model_type_is_kolors(self._model_type)):
                    args['control_image'] = control_images
                elif sdxl_cn_union:
                    # controlnet union pipeline does not use "image"
                    # it also destructively modifies
                    # this input value if it is a list for
                    # whatever reason
                    args['control_image'] = list(control_images)
                else:
                    args['image'] = control_images
            elif self._pipeline_type in {_enums.PipelineType.IMG2IMG, _enums.PipelineType.INPAINT}:
                args['image'] = user_args.images
                args['control_image'] = control_images if not sdxl_cn_union else list(control_images)
                set_strength()

            mask_images = user_args.mask_images
            if mask_images is not None:
                args['mask_image'] = mask_images

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
                _messages.warning(
                    f'T2I Adapter control image(s) of size {first_control_image_size} being forcefully '
                    f'aligned by 16 to {new_size} to prevent errors.'
                )

                for idx, img in enumerate(adapter_control_images):
                    adapter_control_images[idx] = _image.resize_image(img, new_size)

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
            images = user_args.images

            floyd_og_image_needed = (self._pipeline_type == _enums.PipelineType.INPAINT and
                                     _enums.model_type_is_floyd_ifs(self._model_type)
                                     ) or (self._model_type == _enums.ModelType.TORCH_IFS_IMG2IMG)

            if floyd_og_image_needed:
                if user_args.floyd_image is None:
                    raise _pipelines.UnsupportedPipelineConfigError(
                        'must specify "floyd_image" to disambiguate this operation, '
                        '"floyd_image" being the output of a previous floyd stage.')
                args['original_image'] = images
                args['image'] = user_args.floyd_image
            elif self._model_type == _enums.ModelType.TORCH_S_CASCADE:
                args['images'] = images
            else:
                args['image'] = images

            def check_no_image_seed_strength():
                if user_args.image_seed_strength is not None:
                    _messages.warning(
                        f'image_seed_strength is not supported by model_type '
                        f'"{_enums.get_model_type_string(self._model_type)}" in '
                        f'mode "{self._pipeline_type.name}" and is being ignored.'
                    )

            if _enums.model_type_is_upscaler(self._model_type):
                if self._model_type == _enums.ModelType.TORCH_UPSCALER_X4:
                    args['noise_level'] = int(
                        _types.default(
                            user_args.upscaler_noise_level,
                            _constants.DEFAULT_X4_UPSCALER_NOISE_LEVEL
                        )
                    )
                check_no_image_seed_strength()
            elif self._model_type == _enums.ModelType.TORCH_FLUX_FILL:
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

            mask_images = user_args.mask_images

            if mask_images is not None:
                args['mask_image'] = mask_images
                if not (_enums.model_type_is_floyd(self._model_type) or
                        _enums.model_type_is_sd3(self._model_type)):
                    args['width'] = images[0].size[0]
                    args['height'] = images[0].size[1]

            if self._parsed_adetailer_detector_uris:
                # inpainting pipeline, just no mask
                # because it is auto generated
                if not _enums.model_type_is_sd3(self._model_type):
                    args['width'] = images[0].size[0]
                    args['height'] = images[0].size[1]

            if self._model_type == _enums.ModelType.TORCH_SDXL_PIX2PIX:
                args['width'] = images[0].size[0]
                args['height'] = images[0].size[1]

            elif self._model_type == _enums.ModelType.TORCH_UPSCALER_X2:
                images = list(images)
                args['image'] = images

                for idx, image in enumerate(images):
                    if not _image.is_aligned(image.size, 64):
                        size = _image.align_by(image.size, 64)
                        _messages.warning(
                            f'Input image size {image.size} is not aligned by 64. '
                            f'Output dimensions will be forcefully aligned to 64: {size}.'
                        )
                        images[idx] = _image.resize_image(image, size)

            elif self._model_type == _enums.ModelType.TORCH_S_CASCADE:
                if not _image.is_aligned(images[0].size, 128):
                    size = _image.align_by(images[0].size, 128)
                    _messages.warning(
                        f'Input image size {images[0].size} is not aligned by 128. '
                        f'Output dimensions will be forcefully aligned to 128: {size}.'
                    )
                else:
                    size = images[0].size

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

            elif self._model_type == _enums.ModelType.TORCH_SD3:
                images = list(images)
                args['image'] = images

                for idx, image in enumerate(images):
                    if not _image.is_aligned(image.size, 16):
                        size = _image.align_by(image.size, 16)
                        _messages.warning(
                            f'Input image size {image.size} is not aligned by 16. '
                            f'Dimensions will be forcefully aligned to 16: {size}.'
                        )
                        images[idx] = _image.resize_image(image, size)

                if mask_images:
                    mask_images = list(mask_images)
                    args['mask_image'] = mask_images

                    for idx, image in enumerate(mask_images):
                        if not _image.is_aligned(image.size, 16):
                            size = _image.align_by(image.size, 16)
                            _messages.warning(
                                f'Input mask image size {image.size} is not aligned by 16. '
                                f'Dimensions will be forcefully aligned to 16: {size}.'
                            )
                            mask_images[idx] = _image.resize_image(image, size)

                    args['width'] = mask_images[0].size[0]
                    args['height'] = mask_images[0].size[1]

        def set_txt2img_defaults():
            if _enums.model_type_is_sdxl(self._model_type):
                args['height'] = _types.default(user_args.height, _constants.DEFAULT_SDXL_OUTPUT_HEIGHT)
                args['width'] = _types.default(user_args.width, _constants.DEFAULT_SDXL_OUTPUT_WIDTH)
            elif _enums.model_type_is_kolors(self._model_type):
                args['height'] = _types.default(user_args.height, _constants.DEFAULT_KOLORS_OUTPUT_HEIGHT)
                args['width'] = _types.default(user_args.width, _constants.DEFAULT_KOLORS_OUTPUT_WIDTH)
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
            elif self._model_type == _enums.ModelType.TORCH_FLUX:
                args['height'] = _types.default(user_args.height, _constants.DEFAULT_FLUX_OUTPUT_HEIGHT)
                args['width'] = _types.default(user_args.width, _constants.DEFAULT_FLUX_OUTPUT_WIDTH)
            else:
                args['height'] = _types.default(user_args.height, _constants.DEFAULT_OUTPUT_HEIGHT)
                args['width'] = _types.default(user_args.width, _constants.DEFAULT_OUTPUT_WIDTH)

        if self._controlnet_uris:
            set_controlnet_defaults()
        elif self._t2i_adapter_uris:
            set_t2iadapter_defaults()
        elif user_args.images is not None:
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

                _messages.warning(
                    f'Control image size {img.size} is not aligned by 16. '
                    f'Output dimensions will be forcefully aligned by 16: {size}.'
                )

                processed_control_images[idx] = _image.resize_image(img, size)
        return processed_control_images

    def _get_adapter_conditioning_scale(self):
        if not self._parsed_t2i_adapter_uris:
            return 1.0
        return [p.scale for p in self._parsed_t2i_adapter_uris] if \
            len(self._parsed_t2i_adapter_uris) > 1 else self._parsed_t2i_adapter_uris[0].scale

    def _get_controlnet_conditioning_scale(self):
        if not self._parsed_controlnet_uris:
            return 1.0
        return [p.scale for p in self._parsed_controlnet_uris] if \
            len(self._parsed_controlnet_uris) > 1 else self._parsed_controlnet_uris[0].scale

    def _get_controlnet_mode(self):
        if not self._parsed_controlnet_uris:
            return None
        return [p.mode for p in self._parsed_controlnet_uris] if \
            len(self._parsed_controlnet_uris) > 1 else self._parsed_controlnet_uris[0].mode

    def _get_controlnet_guidance_start(self):
        if not self._parsed_controlnet_uris:
            return 0.0
        return [p.start for p in self._parsed_controlnet_uris] if \
            len(self._parsed_controlnet_uris) > 1 else self._parsed_controlnet_uris[0].start

    def _get_controlnet_guidance_end(self):
        if not self._parsed_controlnet_uris:
            return 1.0
        return [p.end for p in self._parsed_controlnet_uris] if \
            len(self._parsed_controlnet_uris) > 1 else self._parsed_controlnet_uris[0].end

    def _check_for_invalid_model_specific_opts(self, user_args: DiffusionArguments):
        if not _enums.model_type_is_flux(self.model_type):
            for arg, val in _types.get_public_attributes(user_args).items():
                if arg.startswith('flux') and val is not None:
                    raise _pipelines.UnsupportedPipelineConfigError(
                        f'{arg} may only be used with Flux models.')

        if not (_enums.model_type_is_sdxl(self.model_type) or
                _enums.model_type_is_kolors(self.model_type)):
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

    @staticmethod
    def _set_prompt_weighter_extra_supported_args(
            pipeline_args: dict,
            prompt_weighter: _promptweighters.PromptWeighter | None,
            diffusion_args: DiffusionArguments,
            second_model: bool,
    ) -> list[str]:
        if prompt_weighter is None:
            return []

        poppable_args = []
        second_prompt_arg = 'second_prompt' if not second_model else 'second_model_second_prompt'

        arg_map = {
            'prompt_2': second_prompt_arg,
            'negative_prompt_2': second_prompt_arg,
            'prompt_3': 'third_prompt',
            'negative_prompt_3': 'third_prompt'
        }

        prompt_weighter_extra_args = prompt_weighter.get_extra_supported_args()

        for arg_name in prompt_weighter_extra_args:

            if arg_name not in arg_map:
                raise RuntimeError(
                    f'Prompt weighter plugin: {prompt_weighter.__class__.__name__}, '
                    f'returned invalid "get_extra_supported_args()" value: {arg_name}.  '
                    f'This is a bug, acceptible values are: {", ".join(arg_map.keys())}')

            source = arg_map[arg_name]
            if 'negative' in arg_name:
                user_value = getattr(diffusion_args, source, None)
                if user_value:
                    pipeline_args[arg_name] = user_value.negative
                    poppable_args.append(arg_name)
            else:
                user_value = getattr(diffusion_args, source, None)
                if user_value:
                    pipeline_args[arg_name] = user_value.positive
                    poppable_args.append(arg_name)

        return poppable_args

    def _set_non_universal_pipeline_arg(self,
                                        pipeline,
                                        pipeline_args: dict,
                                        user_args: DiffusionArguments,
                                        pipeline_arg_name: str,
                                        user_arg_name: str,
                                        option_name: str,
                                        transform: typing.Callable[
                                            [typing.Any], typing.Any] = None):

        pipeline_kwargs = user_args.get_pipeline_wrapper_kwargs()

        if pipeline.__call__.__wrapped__ is not None:
            # torch.no_grad()
            func = pipeline.__call__.__wrapped__
        else:
            func = pipeline.__call__

        if pipeline_arg_name in inspect.getfullargspec(func).args:
            if user_arg_name in pipeline_kwargs:
                # Only provide if the user provided the option
                # otherwise, defer to the pipelines default value
                val = getattr(user_args, user_arg_name)
                val = val if not transform else transform(val)
                pipeline_args[pipeline_arg_name] = val
        else:
            if pipeline_arg_name in pipeline_args:
                # we are forcing it to be allowed.
                return

            val = _types.default(getattr(user_args, user_arg_name), None)
            if val is not None:
                raise _pipelines.UnsupportedPipelineConfigError(
                    f'{option_name} cannot be used with --model-type "{self.model_type_string}" in '
                    f'{_enums.get_pipeline_type_string(self._pipeline_type)} mode with the current '
                    f'combination of arguments and model.')

    def _get_sdxl_conditioning_args(self, pipeline, pipeline_args: dict, user_args: DiffusionArguments,
                                    user_prefix=None):
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

    def _call_torch_flux(self, pipeline_args, user_args: DiffusionArguments):
        self._check_for_invalid_model_specific_opts(user_args)

        if user_args.clip_skip is not None and user_args.clip_skip > 0:
            raise _pipelines.UnsupportedPipelineConfigError('Flux does not support clip skip.')

        prompt: _prompt.Prompt = _types.default(user_args.prompt, _prompt.Prompt())
        prompt_2: _prompt.Prompt = _types.default(user_args.second_prompt, _prompt.Prompt())

        pipeline_args['prompt'] = prompt.positive if prompt.positive else ''
        pipeline_args['prompt_2'] = prompt_2.positive if prompt.positive else ''

        if user_args.max_sequence_length is not None:
            pipeline_args['max_sequence_length'] = user_args.max_sequence_length

        if prompt.negative:
            _messages.warning(
                'Flux is ignoring the provided negative prompt as it '
                'does not support negative prompting.'
            )

        if prompt_2.negative:
            _messages.warning(
                'Flux is ignoring the provided second negative prompt as it '
                'does not support negative prompting.'
            )

        batch_size = _types.default(user_args.batch_size, 1)

        if user_args.images:
            if batch_size % len(user_args.images) != 0:
                batch_size = len(user_args.images)
                if user_args.batch_size is not None:
                    # only warn if the user specified a value
                    _messages.warning(
                        f'Setting --batch-size to {batch_size} because '
                        f'given batch size did not divide evenly with the '
                        f'provided number of input images.'
                    )

        pipeline_args['num_images_per_prompt'] = batch_size

        self._set_non_universal_pipeline_arg(self._pipeline,
                                             pipeline_args, user_args,
                                             'ip_adapter_image', 'ip_adapter_images',
                                             'IP Adapter images')

        pipeline_args['generator'] = \
            torch.Generator(device=self._device).manual_seed(
                _types.default(user_args.seed, _constants.DEFAULT_SEED))

        if hasattr(self._pipeline, 'controlnet'):
            pipeline_args['controlnet_conditioning_scale'] = \
                self._get_controlnet_conditioning_scale()
            pipeline_args['control_guidance_start'] = \
                self._get_controlnet_guidance_start()
            pipeline_args['control_guidance_end'] = \
                self._get_controlnet_guidance_end()
            pipeline_args['control_mode'] = \
                self._get_controlnet_mode()

        prompt_weighter = self._get_prompt_weighter(user_args)

        self._set_prompt_weighter_extra_supported_args(
            pipeline_args=pipeline_args,
            prompt_weighter=prompt_weighter,
            diffusion_args=user_args,
            second_model=False
        )

        with _teacache_flux.teacache_context(
                self._pipeline,
                user_args.inference_steps,
                rel_l1_thresh=_types.default(
                    user_args.tea_cache_rel_l1_threshold,
                    _constants.DEFAULT_TEA_CACHE_REL_L1_THRESHOLD
                ),
                enable=_types.default(user_args.tea_cache, False),
        ):
            if self._parsed_adetailer_detector_uris:
                return self._call_asdff(
                    user_args=user_args,
                    pipeline_args=pipeline_args,
                    batch_size=batch_size,
                    prompt_weighter=prompt_weighter
                )
            else:
                return PipelineWrapperResult(_pipelines.call_pipeline(
                    pipeline=self._pipeline,
                    prompt_weighter=prompt_weighter,
                    device=self._device,
                    **pipeline_args).images)

    def _call_asdff(self,
                    user_args: DiffusionArguments,
                    prompt_weighter: _promptweighters.PromptWeighter,
                    pipeline_args: dict[str, typing.Any],
                    batch_size: int
                    ):
        asdff_pipe = _asdff_base.AdPipelineBase(self._pipeline)

        # use the provided pipe as is, it must be
        # some sort of inpainting pipe
        asdff_pipe.auto_detect_pipe = False

        # should we crop any control image the same way that we crop the mask?
        asdff_pipe.crop_control_image = self._adetailer_crop_control_image

        asdff_output = None
        for detector_uri in self._parsed_adetailer_detector_uris:
            input_images = pipeline_args['image'] if asdff_output is None else asdff_output.images
            input_images *= (batch_size // len(input_images))

            mask_blur = int(_types.default(user_args.adetailer_mask_blur, _constants.DEFAULT_ADETAILER_MASK_BLUR))
            if detector_uri.mask_blur is not None:
                mask_blur = detector_uri.mask_blur
                _messages.log(f'Overriding global adetailer mask-blur '
                              f'value with adetailer detector URI value: {mask_blur}')

            mask_dilation = int(
                _types.default(user_args.adetailer_mask_dilation, _constants.DEFAULT_ADETAILER_MASK_DILATION))
            if detector_uri.mask_dilation is not None:
                mask_dilation = detector_uri.mask_dilation
                _messages.log(f'Overriding global adetailer mask-dilation '
                              f'value with adetailer detector URI value: {mask_dilation}')

            mask_padding = _types.default(user_args.adetailer_mask_padding, _constants.DEFAULT_ADETAILER_MASK_PADDING)
            if detector_uri.mask_padding is not None:
                mask_padding = detector_uri.mask_padding
                _messages.log(f'Overriding global adetailer mask-dilation '
                              f'value with adetailer detector URI value: {mask_dilation}')

            detector_padding = _types.default(user_args.adetailer_detector_padding,
                                              _constants.DEFAULT_ADETAILER_DETECTOR_PADDING)
            if detector_uri.detector_padding is not None:
                detector_padding = detector_uri.detector_padding
                _messages.log(f'Overriding global adetailer detector-padding '
                              f'value with adetailer detector URI value: {detector_padding}')

            mask_shape = str(_types.default(user_args.adetailer_mask_shape, _constants.DEFAULT_ADETAILER_MASK_SHAPE))
            if detector_uri.mask_shape is not None:
                mask_shape = detector_uri.mask_shape
                _messages.log(f'Overriding global adetailer mask-shape '
                              f'value with adetailer detector URI value: {mask_shape}')

            index_filter = _types.default(user_args.adetailer_index_filter, None)
            if detector_uri.index_filter is not None:
                index_filter = detector_uri.index_filter
                _messages.log(f'Overriding global adetailer index-filter '
                              f'value with adetailer detector URI value: {index_filter}')

            if detector_uri.prompt is not None:
                pipeline_args['prompt'] = detector_uri.prompt
                _messages.log(f'Overriding global positive prompt '
                              f'value with adetailer detector URI value: "{detector_uri.prompt}"')

            if detector_uri.negative_prompt is not None:
                pipeline_args['negative_prompt'] = detector_uri.negative_prompt
                _messages.log(f'Overriding global negative prompt '
                              f'value with adetailer detector URI value: "{detector_uri.negative_prompt}"')

            asdff_output = asdff_pipe(
                pipeline_args=pipeline_args,
                model_path=detector_uri.get_model_path(
                    local_files_only=self._local_files_only, use_auth_token=self._auth_token),
                images=input_images,
                device=self._device,
                detector_device=_types.default(detector_uri.device, self._device),
                confidence=detector_uri.confidence,
                prompt_weighter=prompt_weighter,
                index_filter=index_filter,
                mask_blur=mask_blur,
                mask_shape=mask_shape,
                detector_padding=detector_padding,
                mask_padding=mask_padding,
                mask_dilation=mask_dilation
            )

        return PipelineWrapperResult(asdff_output.images)

    def _call_torch_s_cascade(self, pipeline_args, user_args: DiffusionArguments):
        self._check_for_invalid_model_specific_opts(user_args)

        if user_args.clip_skip is not None and user_args.clip_skip > 0:
            raise _pipelines.UnsupportedPipelineConfigError('Stable Cascade does not support clip skip.')

        prompt: _prompt.Prompt = _types.default(user_args.prompt, _prompt.Prompt())
        pipeline_args['prompt'] = prompt.positive if prompt.positive else ''
        pipeline_args['negative_prompt'] = prompt.negative

        pipeline_args['num_images_per_prompt'] = _types.default(user_args.batch_size, 1)

        pipeline_args['generator'] = \
            torch.Generator(device=self._device).manual_seed(
                _types.default(user_args.seed, _constants.DEFAULT_SEED))

        prompt_weighter = self._get_prompt_weighter(user_args)

        self._set_prompt_weighter_extra_supported_args(
            pipeline_args=pipeline_args,
            prompt_weighter=prompt_weighter,
            diffusion_args=user_args,
            second_model=False
        )

        prior = _pipelines.call_pipeline(
            pipeline=self._pipeline,
            device=self._device,
            prompt_weighter=prompt_weighter,
            **pipeline_args)

        pipeline_args['num_inference_steps'] = user_args.second_model_inference_steps
        pipeline_args['guidance_scale'] = user_args.second_model_guidance_scale
        pipeline_args.pop('height')
        pipeline_args.pop('width')
        pipeline_args.pop('images', None)

        if self._parsed_s_cascade_decoder_uri.dtype is not None:
            image_embeddings = prior.image_embeddings.to(
                _enums.get_torch_dtype(self._parsed_s_cascade_decoder_uri.dtype))
        else:
            image_embeddings = prior.image_embeddings

        if user_args.second_model_prompt:
            prompt: _prompt.Prompt = user_args.second_model_prompt
            pipeline_args['prompt'] = prompt.positive if prompt.positive else ''
            pipeline_args['negative_prompt'] = prompt.negative

        pipeline_args.pop('num_images_per_prompt')

        return PipelineWrapperResult(_pipelines.call_pipeline(
            image_embeddings=image_embeddings,
            pipeline=self._s_cascade_decoder_pipeline,
            device=self._device,
            prompt_weighter=self._get_second_model_prompt_weighter(user_args),
            **pipeline_args).images)

    def _call_torch(self, pipeline_args, user_args: DiffusionArguments):
        self._check_for_invalid_model_specific_opts(user_args)

        prompt: _prompt.Prompt = _types.default(user_args.prompt, _prompt.Prompt())

        pipeline_args['prompt'] = prompt.positive if prompt.positive else ''
        pipeline_args['negative_prompt'] = prompt.negative

        self._get_sdxl_conditioning_args(self._pipeline, pipeline_args, user_args)

        prompt_weighter = self._get_prompt_weighter(user_args)

        prompt_weighter_pop_args = self._set_prompt_weighter_extra_supported_args(
            pipeline_args=pipeline_args,
            prompt_weighter=prompt_weighter,
            diffusion_args=user_args,
            second_model=False
        )

        if _enums.model_type_is_sd3(self.model_type):

            self._set_non_universal_pipeline_arg(self._pipeline,
                                                 pipeline_args, user_args,
                                                 'max_sequence_length', 'max_sequence_length',
                                                 '--max-sequence-length')

            self._set_non_universal_pipeline_arg(self._pipeline,
                                                 pipeline_args, user_args,
                                                 'prompt_2', 'second_prompt',
                                                 '--second-prompts',
                                                 transform=lambda p: p.positive)

            self._set_non_universal_pipeline_arg(self._pipeline,
                                                 pipeline_args, user_args,
                                                 'negative_prompt_2', 'second_prompt',
                                                 '--second-prompts',
                                                 transform=lambda p: p.negative)

            self._set_non_universal_pipeline_arg(self._pipeline,
                                                 pipeline_args, user_args,
                                                 'prompt_3', 'third_prompt',
                                                 '--third-prompts',
                                                 transform=lambda p: p.positive)

            self._set_non_universal_pipeline_arg(self._pipeline,
                                                 pipeline_args, user_args,
                                                 'negative_prompt_3', 'third_prompt',
                                                 '--third-prompts',
                                                 transform=lambda p: p.negative)

        else:
            self._set_non_universal_pipeline_arg(self._pipeline,
                                                 pipeline_args, user_args,
                                                 'prompt_2', 'second_prompt',
                                                 '--second-prompts',
                                                 transform=lambda p: p.positive)

            self._set_non_universal_pipeline_arg(self._pipeline,
                                                 pipeline_args, user_args,
                                                 'negative_prompt_2', 'second_prompt',
                                                 '--second-prompts',
                                                 transform=lambda p: p.negative)

        self._set_non_universal_pipeline_arg(self._pipeline,
                                             pipeline_args, user_args,
                                             'pag_scale', 'pag_scale',
                                             '--pag-scale')

        self._set_non_universal_pipeline_arg(self._pipeline,
                                             pipeline_args, user_args,
                                             'pag_adaptive_scale', 'pag_adaptive_scale',
                                             '--pag-adaptive-scale')

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

        if user_args.images:
            if batch_size % len(user_args.images) != 0:
                batch_size = len(user_args.images)
                if user_args.batch_size is not None:
                    # only warn if the user specified a value
                    _messages.warning(
                        f'Setting --batch-size to {batch_size} because '
                        f'given batch size did not divide evenly with the '
                        f'provided number of input images.'
                    )

        if self._model_type != _enums.ModelType.TORCH_UPSCALER_X2:
            pipeline_args['num_images_per_prompt'] = batch_size
        else:
            in_img_cnt = len(pipeline_args['image'])
            if batch_size > in_img_cnt:
                batch_mul = batch_size // in_img_cnt
            else:
                batch_mul = 1

            in_imgs = pipeline_args['image'] * batch_mul
            num_prompts = len(in_imgs)
            pipeline_args['image'] = in_imgs

            pipeline_args['prompt'] = \
                [pipeline_args['prompt']] * num_prompts

            if pipeline_args.get('negative_prompt', None) is not None:
                pipeline_args['negative_prompt'] = \
                    [pipeline_args['negative_prompt']] * num_prompts

        generator = torch.Generator(device=self._device).manual_seed(
            _types.default(user_args.seed, _constants.DEFAULT_SEED))

        pipeline_args['generator'] = generator

        if isinstance(self._pipeline, diffusers.StableDiffusionInpaintPipelineLegacy):
            # Not necessary, will cause an error
            pipeline_args.pop('width')
            pipeline_args.pop('height')

        has_controlnet = hasattr(self._pipeline, 'controlnet')

        has_t2i_adapter = hasattr(self._pipeline, 'adapter') and \
                          isinstance(self._pipeline.adapter,
                                     (diffusers.T2IAdapter, diffusers.MultiAdapter))

        sd_edit = user_args.sdxl_refiner_edit or \
                  has_controlnet or has_t2i_adapter or \
                  self._parsed_adetailer_detector_uris or \
                  isinstance(self._pipeline,
                             (diffusers.StableDiffusionXLInpaintPipeline,
                              diffusers.StableDiffusionXLPAGInpaintPipeline))

        if has_controlnet:
            is_xl_union_model = isinstance(
                self._pipeline.controlnet, diffusers.ControlNetUnionModel) \
                                and len(self.controlnet_uris) > 1

            pipeline_args.update({
                'controlnet_conditioning_scale': self._get_controlnet_conditioning_scale(),
                'control_guidance_start': self._get_controlnet_guidance_start()[0] if
                is_xl_union_model else self._get_controlnet_guidance_start(),
                'control_guidance_end': self._get_controlnet_guidance_end()[0] if
                is_xl_union_model else self._get_controlnet_guidance_end()
            })

            if 'control_mode' in inspect.signature(self._pipeline.__call__).parameters:
                pipeline_args['control_mode'] = self._get_controlnet_mode()

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
                pipeline_args['image'] = list(pipeline_args['image'])
                for idx, adapter in enumerate(self._pipeline.adapter.adapters):
                    if hasattr(adapter.config, 'in_channels'):
                        if adapter.config.in_channels == 1:
                            pipeline_args['image'][idx] = pipeline_args['image'][idx].convert('L')

        def generate_asdff():
            return self._call_asdff(
                user_args=user_args,
                pipeline_args=pipeline_args,
                batch_size=batch_size,
                prompt_weighter=prompt_weighter
            )

        if self._sdxl_refiner_pipeline is None:
            ras_args = self._get_sd3_ras_args(user_args)

            with _hi_diffusion(self._pipeline,generator=generator, enabled=user_args.hi_diffusion), \
                 _sd3_ras_context(self._pipeline, args=ras_args, enabled=user_args.ras), \
                 _deep_cache_context(self._pipeline,
                                     cache_interval=_types.default(
                                         user_args.deep_cache_interval, _constants.DEFAULT_DEEP_CACHE_INTERVAL),
                                     cache_branch_id=_types.default(
                                         user_args.deep_cache_branch_id, _constants.DEFAULT_DEEP_CACHE_BRANCH_ID),
                                     enabled=user_args.deep_cache):
                if self._parsed_adetailer_detector_uris:
                    return generate_asdff()
                else:
                    return PipelineWrapperResult(_pipelines.call_pipeline(
                        pipeline=self._pipeline,
                        prompt_weighter=prompt_weighter,
                        device=self._device,
                        **pipeline_args).images)

        high_noise_fraction = _types.default(user_args.sdxl_high_noise_fraction,
                                             _constants.DEFAULT_SDXL_HIGH_NOISE_FRACTION)

        if sd_edit:
            i_start = dict()
            i_end = dict()
        else:
            i_start = {'denoising_start': high_noise_fraction}
            i_end = {'denoising_end': high_noise_fraction}

        output_type = 'latent'
        if isinstance(self._sdxl_refiner_pipeline,
                      diffusers.StableDiffusionXLPAGInpaintPipeline):
            # cannot handle latent input
            output_type = 'pil'

        with _hi_diffusion(self._pipeline,
                           generator=generator,
                           enabled=user_args.hi_diffusion), \
             _deep_cache_context(self._pipeline,
                                 cache_interval=_types.default(
                                     user_args.deep_cache_interval, _constants.DEFAULT_DEEP_CACHE_INTERVAL),
                                 cache_branch_id=_types.default(
                                     user_args.deep_cache_branch_id, _constants.DEFAULT_DEEP_CACHE_BRANCH_ID),
                                 enabled=user_args.deep_cache):
            if self._parsed_adetailer_detector_uris:
                image = generate_asdff().images
            else:
                image = _pipelines.call_pipeline(
                    pipeline=self._pipeline,
                    device=self._device,
                    prompt_weighter=prompt_weighter,
                    **pipeline_args,
                    **i_end,
                    output_type=output_type).images

        pipeline_args['image'] = image

        if not isinstance(self._sdxl_refiner_pipeline,
                          (diffusers.StableDiffusionXLInpaintPipeline,
                           diffusers.StableDiffusionXLPAGInpaintPipeline)):
            # Width / Height not necessary for any other refiner
            if not (isinstance(self._pipeline,
                               (diffusers.StableDiffusionXLImg2ImgPipeline,
                                diffusers.StableDiffusionXLPAGImg2ImgPipeline,
                                diffusers.KolorsImg2ImgPipeline)) and
                    isinstance(self._sdxl_refiner_pipeline,
                               (diffusers.StableDiffusionXLImg2ImgPipeline,
                                diffusers.StableDiffusionXLPAGImg2ImgPipeline))):
                # Width / Height does not get passed to img2img
                pipeline_args.pop('width')
                pipeline_args.pop('height')

        # Or any of these
        self._pop_sdxl_conditioning_args(pipeline_args)
        pipeline_args.pop('ip_adapter_image', None)
        pipeline_args.pop('guidance_rescale', None)
        pipeline_args.pop('controlnet_conditioning_scale', None)
        pipeline_args.pop('control_guidance_start', None)
        pipeline_args.pop('control_guidance_end', None)
        pipeline_args.pop('image_guidance_scale', None)
        pipeline_args.pop('control_image', None)

        # these are only passed if set for the refiner specifically
        pipeline_args.pop('pag_scale', None)
        pipeline_args.pop('pag_adaptive_scale', None)

        # we will handle the strength parameter if it is necessary below
        pipeline_args.pop('strength', None)

        # We do not want to override the refiner secondary prompt
        # with that of --second-prompts by default
        pipeline_args.pop('prompt_2', None)
        pipeline_args.pop('negative_prompt_2', None)

        if prompt_weighter_pop_args:
            for arg_name in prompt_weighter_pop_args:
                if arg_name in pipeline_args:
                    pipeline_args.pop(arg_name)

        second_model_prompt_weighter = self._get_second_model_prompt_weighter(user_args)

        self._set_prompt_weighter_extra_supported_args(
            pipeline_args=pipeline_args,
            prompt_weighter=second_model_prompt_weighter,
            diffusion_args=user_args,
            second_model=True
        )

        self._set_non_universal_pipeline_arg(self._pipeline,
                                             pipeline_args, user_args,
                                             'prompt', 'second_model_prompt',
                                             '--second-model-prompts',
                                             transform=lambda p: p.positive)

        self._set_non_universal_pipeline_arg(self._pipeline,
                                             pipeline_args, user_args,
                                             'negative_prompt', 'second_model_prompt',
                                             '--second-model-prompts',
                                             transform=lambda p: p.negative)

        self._set_non_universal_pipeline_arg(self._pipeline,
                                             pipeline_args, user_args,
                                             'prompt_2', 'second_model_second_prompt',
                                             '--second-model-second-prompts',
                                             transform=lambda p: p.positive)

        self._set_non_universal_pipeline_arg(self._pipeline,
                                             pipeline_args, user_args,
                                             'negative_prompt_2', 'second_model_second_prompt',
                                             '--second-model-second-prompts',
                                             transform=lambda p: p.negative)

        self._get_sdxl_conditioning_args(self._sdxl_refiner_pipeline,
                                         pipeline_args, user_args,
                                         user_prefix='refiner')

        self._set_non_universal_pipeline_arg(self._pipeline,
                                             pipeline_args, user_args,
                                             'guidance_rescale', 'sdxl_refiner_guidance_rescale',
                                             '--sdxl-refiner-guidance-rescales')

        if user_args.second_model_inference_steps is not None:
            pipeline_args['num_inference_steps'] = user_args.second_model_inference_steps

        if user_args.sdxl_refiner_pag_scale is not None:
            pipeline_args['pag_scale'] = user_args.sdxl_refiner_pag_scale

        if user_args.sdxl_refiner_pag_adaptive_scale is not None:
            pipeline_args['pag_adaptive_scale'] = user_args.sdxl_refiner_pag_adaptive_scale

        if user_args.second_model_guidance_scale is not None:
            pipeline_args['guidance_scale'] = user_args.second_model_guidance_scale

        if user_args.sdxl_refiner_guidance_rescale is not None:
            pipeline_args['guidance_rescale'] = user_args.sdxl_refiner_guidance_rescale

        if user_args.sdxl_refiner_clip_skip is not None:
            pipeline_args['clip_skip'] = user_args.sdxl_refiner_clip_skip

        if sd_edit:
            strength = float(decimal.Decimal('1.0') - decimal.Decimal(str(high_noise_fraction)))

            if strength <= 0.0:
                strength = 0.2
                _messages.warning(
                    f'Refiner edit mode image seed strength (1.0 - high-noise-fraction) '
                    f'was calculated at <= 0.0, defaulting to {strength}'
                )
            else:
                _messages.log(f'Running refiner in edit mode with '
                              f'refiner image seed strength = {strength}, IE: (1.0 - high-noise-fraction)')

            inference_steps = pipeline_args.get('num_inference_steps')

            if (strength * inference_steps) < 1.0:
                strength = 1.0 / inference_steps
                _messages.warning(
                    f'Refiner edit mode image seed strength (1.0 - high-noise-fraction) * inference-steps '
                    f'was calculated at < 1, defaulting to (1.0 / inference-steps): {strength}'
                )

            pipeline_args['strength'] = strength

        if isinstance(self._sdxl_refiner_pipeline.scheduler, diffusers.LCMScheduler):
            # This will error out catastrophically if we let it happen.

            original_steps = self._sdxl_refiner_pipeline.scheduler.config['original_inference_steps']
            inference_steps = pipeline_args.get('num_inference_steps')

            if sd_edit:
                float_limit = strength * original_steps
                limit = int(math.floor(float_limit))
                if limit < inference_steps:
                    _messages.warning(
                        f'Refiner inference-steps is being reduced to {limit} '
                        f'due to LCMScheduler requirements. "LCMScheduler;original-inference-steps={original_steps}" and '
                        f'refiner inference-steps must less than or equal to "strength" (inverse high-noise-fraction) * original-inference-steps. '
                        f'i.e. refiner inference-steps <= ({strength} * {original_steps} = {float_limit}).'
                    )
            else:
                limit = original_steps
                if limit < inference_steps:
                    _messages.warning(
                        f'Refiner inference-steps is being reduced to {limit} '
                        f'due to LCMScheduler requirements. "LCMScheduler;original-inference-steps={original_steps}" and '
                        f'refiner inference-steps must less than or equal to that.'
                    )

            pipeline_args['num_inference_steps'] = limit

        with _hi_diffusion(self._sdxl_refiner_pipeline,
                           generator=generator,
                           enabled=user_args.sdxl_refiner_hi_diffusion), \
             _deep_cache_context(self._sdxl_refiner_pipeline,
                                 cache_interval=_types.default(
                                     user_args.deep_cache_interval,
                                     _constants.DEFAULT_SDXL_REFINER_DEEP_CACHE_INTERVAL),
                                 cache_branch_id=_types.default(
                                     user_args.deep_cache_branch_id,
                                     _constants.DEFAULT_SDXL_REFINER_DEEP_CACHE_BRANCH_ID),
                                 enabled=user_args.second_model_deep_cache):
            return PipelineWrapperResult(
                _pipelines.call_pipeline(
                    pipeline=self._sdxl_refiner_pipeline,
                    device=self._device,
                    prompt_weighter=self._get_second_model_prompt_weighter(user_args),
                    **pipeline_args, **i_start).images)

    def _get_sd3_ras_args(self, user_args) -> _RASArgs | None:
        if user_args.ras:
            ras_args = _RASArgs(
                num_inference_steps=user_args.inference_steps,
                patch_size=self._pipeline.transformer.config.patch_size,
                sample_ratio=_types.default(user_args.ras_sample_ratio, _constants.DEFAULT_RAS_SAMPLE_RATIO),
                high_ratio=_types.default(user_args.ras_high_ratio, _constants.DEFAULT_RAS_HIGH_RATIO),
                starvation_scale=_types.default(user_args.ras_starvation_scale,
                                                _constants.DEFAULT_RAS_STARVATION_SCALE),
                error_reset_steps=_types.default(user_args.ras_error_reset_steps,
                                                 _constants.DEFAULT_RAS_ERROR_RESET_STEPS),
                width=_types.default(user_args.width, _constants.DEFAULT_SD3_OUTPUT_WIDTH),
                height=_types.default(user_args.height, _constants.DEFAULT_SD3_OUTPUT_HEIGHT),
                enable_index_fusion=user_args.ras_index_fusion,
                metric=_types.default(user_args.ras_metric, _constants.DEFAULT_RAS_METRIC),
                scheduler_start_step=_types.default(user_args.ras_start_step, _constants.DEFAULT_RAS_START_STEP),
                scheduler_end_step=_types.default(user_args.ras_end_step, user_args.inference_steps),
                skip_num_step=_types.default(
                    user_args.ras_skip_num_step, _constants.DEFAULT_RAS_SKIP_NUM_STEP),
                skip_num_step_length=_types.default(
                    user_args.ras_skip_num_step_length, _constants.DEFAULT_RAS_SKIP_NUM_STEP_LENGTH),
                replace_with_flash_attn=importlib.util.find_spec('flash-attn') is not None
            )
        else:
            ras_args = None
        return ras_args

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

    def _lazy_init_pipeline(self, args: DiffusionArguments):

        pag = args.pag_scale is not None or args.pag_adaptive_scale is not None
        sdxl_refiner_pag = args.sdxl_refiner_pag_scale is not None or args.sdxl_refiner_pag_adaptive_scale is not None
        pipeline_type = args.determine_pipeline_type()

        if self._pipeline is not None:
            if self._pipeline_type == pipeline_type:
                return False

        if pag:
            if not (self.model_type == _enums.ModelType.TORCH or
                    self.model_type == _enums.ModelType.TORCH_SDXL or
                    self.model_type == _enums.ModelType.TORCH_SD3 or
                    self.model_type == _enums.ModelType.TORCH_KOLORS):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Perturbed attention guidance (pag arguments) are only supported with '
                    '--model-type torch, torch-sdxl, torch-kolors (txt2img), and torch-sd3.')

            if self.t2i_adapter_uris:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Perturbed attention guidance (pag* arguments) are is not supported '
                    'with --t2i-adapters.')

            args.pag_scale = _types.default(
                args.pag_scale, _constants.DEFAULT_PAG_SCALE)
            args.pag_adaptive_scale = _types.default(
                args.pag_adaptive_scale, _constants.DEFAULT_PAG_ADAPTIVE_SCALE)

        if sdxl_refiner_pag:
            if not self._sdxl_refiner_uri:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'sdxl_refiner_pag* arguments are not supported when '
                    'an SDXL refiner is not specified.')

            args.sdxl_refiner_pag_scale = _types.default(
                args.sdxl_refiner_pag_scale, _constants.DEFAULT_SDXL_REFINER_PAG_SCALE)
            args.sdxl_refiner_pag_adaptive_scale = _types.default(
                args.sdxl_refiner_pag_adaptive_scale, _constants.DEFAULT_SDXL_REFINER_PAG_ADAPTIVE_SCALE)

        self._pipeline_type = pipeline_type

        self._recall_main_pipeline = None
        self._recall_refiner_pipeline = None

        if self._parsed_adetailer_detector_uris:
            pipeline_type = _enums.PipelineType.INPAINT

        if self._model_type == _enums.ModelType.TORCH_S_CASCADE:

            if self._s_cascade_decoder_uri is None:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Stable Cascade must be used with a decoder model.')

            self._recall_main_pipeline = _pipelines.TorchPipelineFactory(
                model_path=self._model_path,
                model_type=self._model_type,
                pipeline_type=pipeline_type,
                subfolder=self._subfolder,
                revision=self._revision,
                variant=self._variant,
                dtype=self._dtype,
                original_config=self._original_config,
                unet_uri=self._unet_uri,
                vae_uri=self._vae_uri,
                lora_uris=self._lora_uris,
                lora_fuse_scale=self._lora_fuse_scale,
                quantizer_uri=self._quantizer_uri,
                safety_checker=self._safety_checker,
                auth_token=self._auth_token,
                device=self._device,
                sequential_cpu_offload=self._model_sequential_offload,
                model_cpu_offload=self._model_cpu_offload,
                local_files_only=self._local_files_only,
                extra_modules=self._model_extra_modules
            )
            creation_result = self._recall_main_pipeline()
            self._pipeline = creation_result.pipeline

            self._recall_s_cascade_decoder_pipeline = _pipelines.TorchPipelineFactory(
                model_path=self._parsed_s_cascade_decoder_uri.model,
                model_type=_enums.ModelType.TORCH_S_CASCADE_DECODER,
                pipeline_type=_enums.PipelineType.TXT2IMG,
                subfolder=self._parsed_s_cascade_decoder_uri.subfolder,
                revision=self._parsed_s_cascade_decoder_uri.revision,
                unet_uri=self._second_model_unet_uri,
                text_encoder_uris=self._second_model_text_encoder_uris,
                quantizer_uri=self._second_model_quantizer_uri,

                variant=self._parsed_s_cascade_decoder_uri.variant if
                self._parsed_s_cascade_decoder_uri.variant is not None else self._variant,

                dtype=self._parsed_s_cascade_decoder_uri.dtype if
                self._parsed_s_cascade_decoder_uri.dtype is not None else self._dtype,

                original_config=self._second_model_original_config,
                safety_checker=self._safety_checker,
                extra_modules=self._second_model_extra_modules,
                auth_token=self._auth_token,
                device=self._device,
                local_files_only=self._local_files_only,
                model_cpu_offload=self._second_model_cpu_offload,
                sequential_cpu_offload=self._second_model_sequential_offload)

            creation_result = self._recall_s_cascade_decoder_pipeline()
            self._s_cascade_decoder_pipeline = creation_result.pipeline

        elif self._sdxl_refiner_uri is not None:

            self._recall_main_pipeline = _pipelines.TorchPipelineFactory(
                model_path=self._model_path,
                model_type=self._model_type,
                pipeline_type=pipeline_type,
                subfolder=self._subfolder,
                revision=self._revision,
                variant=self._variant,
                dtype=self._dtype,
                original_config=self._original_config,
                unet_uri=self._unet_uri,
                vae_uri=self._vae_uri,
                lora_uris=self._lora_uris,
                lora_fuse_scale=self._lora_fuse_scale,
                image_encoder_uri=self._image_encoder_uri,
                ip_adapter_uris=self._ip_adapter_uris,
                textual_inversion_uris=self._textual_inversion_uris,
                text_encoder_uris=self._text_encoder_uris,
                controlnet_uris=self._controlnet_uris,
                t2i_adapter_uris=self._t2i_adapter_uris,
                quantizer_uri=self._quantizer_uri,
                pag=pag,
                safety_checker=self._safety_checker,
                auth_token=self._auth_token,
                device=self._device,
                local_files_only=self._local_files_only,
                extra_modules=self._model_extra_modules,
                model_cpu_offload=self._model_cpu_offload,
                sequential_cpu_offload=self._model_sequential_offload)

            creation_result = self._recall_main_pipeline()
            self._pipeline = creation_result.pipeline
            self._parsed_controlnet_uris = creation_result.parsed_controlnet_uris
            self._parsed_t2i_adapter_uris = creation_result.parsed_t2i_adapter_uris

            if pipeline_type is _enums.PipelineType.TXT2IMG or self._parsed_adetailer_detector_uris:
                refiner_pipeline_type = _enums.PipelineType.IMG2IMG
            else:
                refiner_pipeline_type = pipeline_type

            if self._pipeline is not None:

                if _enums.model_type_is_sdxl(self.model_type):
                    refiner_extra_modules = {'vae': self._pipeline.vae,
                                             'text_encoder_2': self._pipeline.text_encoder_2}
                else:
                    refiner_extra_modules = {'vae': self._pipeline.vae}

                if self._second_model_extra_modules is not None:
                    refiner_extra_modules.update(self._second_model_extra_modules)

            else:
                refiner_extra_modules = self._second_model_extra_modules

            self._recall_refiner_pipeline = _pipelines.TorchPipelineFactory(
                model_path=self._parsed_sdxl_refiner_uri.model,
                model_type=_enums.ModelType.TORCH_SDXL,
                pipeline_type=refiner_pipeline_type,
                subfolder=self._parsed_sdxl_refiner_uri.subfolder,
                revision=self._parsed_sdxl_refiner_uri.revision,
                unet_uri=self._second_model_unet_uri,
                text_encoder_uris=self._second_model_text_encoder_uris,
                quantizer_uri=self._second_model_quantizer_uri,

                variant=self._parsed_sdxl_refiner_uri.variant if
                self._parsed_sdxl_refiner_uri.variant is not None else self._variant,

                dtype=self._parsed_sdxl_refiner_uri.dtype if
                self._parsed_sdxl_refiner_uri.dtype is not None else self._dtype,

                original_config=self._second_model_original_config,
                pag=sdxl_refiner_pag,
                safety_checker=self._safety_checker,
                auth_token=self._auth_token,
                device=self._device,
                extra_modules=refiner_extra_modules,
                local_files_only=self._local_files_only,
                model_cpu_offload=self._second_model_cpu_offload,
                sequential_cpu_offload=self._second_model_sequential_offload
            )
            self._sdxl_refiner_pipeline = self._recall_refiner_pipeline().pipeline
        else:
            self._recall_main_pipeline = _pipelines.TorchPipelineFactory(
                model_path=self._model_path,
                model_type=self._model_type,
                pipeline_type=pipeline_type,
                subfolder=self._subfolder,
                revision=self._revision,
                variant=self._variant,
                dtype=self._dtype,
                original_config=self._original_config,
                unet_uri=self._unet_uri,
                transformer_uri=self._transformer_uri,
                vae_uri=self._vae_uri,
                lora_uris=self._lora_uris,
                lora_fuse_scale=self._lora_fuse_scale,
                image_encoder_uri=self._image_encoder_uri,
                ip_adapter_uris=self._ip_adapter_uris,
                textual_inversion_uris=self._textual_inversion_uris,
                text_encoder_uris=self._text_encoder_uris,
                quantizer_uri=self._quantizer_uri,
                controlnet_uris=self._controlnet_uris,
                t2i_adapter_uris=self._t2i_adapter_uris,
                pag=pag,
                safety_checker=self._safety_checker,
                auth_token=self._auth_token,
                device=self._device,
                sequential_cpu_offload=self._model_sequential_offload,
                model_cpu_offload=self._model_cpu_offload,
                local_files_only=self._local_files_only,
                extra_modules=self._model_extra_modules,
            )

            creation_result = self._recall_main_pipeline()
            self._pipeline = creation_result.pipeline
            self._parsed_controlnet_uris = creation_result.parsed_controlnet_uris
            self._parsed_t2i_adapter_uris = creation_result.parsed_t2i_adapter_uris

        return True

    def _load_prompt_weighter(
            self,
            uri: str,
            model_type: _enums.ModelType,
            dtype: _enums.DataType
    ):
        return self._prompt_weighter_loader.load(
            uri,
            model_type=model_type,
            dtype=dtype,
            local_files_only=self.local_files_only
        )

    def _default_prompt_weighter(self, *sources):
        for source in sources:
            if isinstance(source, str):  # Direct URI case
                return self._load_prompt_weighter(source, model_type=self.model_type, dtype=self._dtype)
            elif source is not None and source.weighter:  # Object case with weighter
                return self._load_prompt_weighter(source.weighter, model_type=self.model_type, dtype=self._dtype)
        return None

    def _get_prompt_weighter(self, args: DiffusionArguments):
        # prioritize in descending order
        return self._default_prompt_weighter(
            args.prompt,
            args.prompt_weighter_uri
        )

    def _get_second_model_prompt_weighter(self, args: DiffusionArguments):
        # prioritize in descending order
        return self._default_prompt_weighter(
            args.second_model_prompt,
            args.second_model_prompt_weighter_uri,
            args.prompt,
            args.prompt_weighter_uri
        )

    def _argument_help_check(self, args: DiffusionArguments):
        scheduler_help = _help.scheduler_is_help(args.scheduler_uri)
        second_model_scheduler_help = _help.scheduler_is_help(args.second_model_scheduler_uri)
        text_encoder_help = _help.text_encoder_is_help(self.text_encoder_uris)
        second_model_text_encoder_help = _help.text_encoder_is_help(self.second_model_text_encoder_uris)
        help_text = []
        model_path = self.model_path

        if scheduler_help or second_model_scheduler_help:
            pipe_class = _pipelines.get_torch_pipeline_class(
                model_type=self.model_type,
                pipeline_type=args.determine_pipeline_type(),
                unet_uri=self.unet_uri,
                transformer_uri=self.transformer_uri,
                vae_uri=self.vae_uri,
                lora_uris=self.lora_uris,
                image_encoder_uri=self.image_encoder_uri,
                ip_adapter_uris=self.ip_adapter_uris,
                textual_inversion_uris=self.textual_inversion_uris,
                controlnet_uris=self.controlnet_uris,
                t2i_adapter_uris=self.t2i_adapter_uris,
                pag=args.pag_scale is not None or args.pag_adaptive_scale is not None,
                help_mode=True
            )
            if scheduler_help:
                help_text.append(
                    f'Schedulers compatible with: {model_path}\n\n' +
                    _help.get_scheduler_help(
                        pipe_class,
                        help_args=_help.scheduler_is_help_args(
                            args.scheduler_uri),
                        indent=4
                    ))
            if text_encoder_help:
                help_text.append(
                    f'Text encoders compatible with: {model_path}\n\n' +
                    _help.text_encoder_help(
                        pipe_class,
                        indent=4
                    ))

        if second_model_scheduler_help or second_model_text_encoder_help:
            second_pipe_class = _pipelines.get_torch_pipeline_class(
                model_type=_enums.ModelType.TORCH_SDXL if
                self.sdxl_refiner_uri else _enums.ModelType.TORCH_S_CASCADE_DECODER,
                pipeline_type=_enums.PipelineType.IMG2IMG,
                unet_uri=self.second_model_unet_uri,
                vae_uri=self.vae_uri,
                pag=args.pag_scale is not None or args.pag_adaptive_scale is not None,
                help_mode=True
            )
            second_model_path = self.sdxl_refiner_uri or self.s_cascade_decoder_uri

            if second_model_scheduler_help:
                help_text.append(
                    f'Schedulers compatible with: {second_model_path}\n\n' +
                    _help.get_scheduler_help(
                        second_pipe_class,
                        help_args=_help.scheduler_is_help_args(
                            args.second_model_scheduler_uri),
                        indent=4
                    ))

            if second_model_text_encoder_help:
                help_text.append(
                    f'Text encoders compatible with: {second_model_path}\n\n' +
                    _help.text_encoder_help(
                        second_pipe_class,
                        indent=4
                    ))

        return '\n\n'.join(help_text)

    def _set_scheduler_and_vae_settings(self, args):
        second_model_scheduler_uri = _types.default(
            args.second_model_scheduler_uri,
            args.scheduler_uri
        )
        if args.scheduler_uri:
            _schedulers.load_scheduler(
                pipeline=self._pipeline,
                scheduler_uri=args.scheduler_uri
            )
        if self._sdxl_refiner_pipeline and second_model_scheduler_uri:
            _schedulers.load_scheduler(
                pipeline=self._sdxl_refiner_pipeline,
                scheduler_uri=second_model_scheduler_uri
            )
        if self._s_cascade_decoder_pipeline and second_model_scheduler_uri:
            _schedulers.load_scheduler(
                pipeline=self._s_cascade_decoder_pipeline,
                scheduler_uri=second_model_scheduler_uri
            )
        _pipelines.set_vae_tiling_and_slicing(
            pipeline=self._pipeline,
            tiling=args.vae_tiling,
            slicing=args.vae_slicing
        )

    def _auto_ras_check(self, args: DiffusionArguments):
        for prop in args.__dict__.keys():
            if prop.startswith('ras_'):
                value = getattr(args, prop)
                if value is not None or (isinstance(value, bool) and value is True):
                    args.ras = True
                    break

        if args.ras:
            if not _enums.model_type_is_sd3(self.model_type):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'RAS is only supported for SD3.')

            if self._pipeline.transformer.config.qk_norm == 'rms_norm':
                raise _pipelines.UnsupportedPipelineConfigError(
                    'RAS does not support SD3.5, only SD3.'
                )

            if importlib.util.find_spec('triton') is None:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'RAS is only supported with triton / triton-windows installed.')

            if self.model_cpu_offload:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'RAS does not support model CPU offloading.')

            start_step = _types.default(args.ras_start_step, _constants.DEFAULT_RAS_START_STEP)
            end_step = _types.default(args.ras_end_step, args.inference_steps)
            if start_step > end_step:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'RAS start step must be less than or equal to end step.')

    def _auto_deep_cache_check(self, args: DiffusionArguments):
        # Auto-enable deep_cache if any deep_cache_ parameters are set
        for prop in args.__dict__.keys():
            if prop.startswith('deep_cache_'):
                value = getattr(args, prop)
                if value is not None or (isinstance(value, bool) and value is True):
                    args.deep_cache = True
                    break

        if args.deep_cache:
            if not (_enums.model_type_is_sd15(self.model_type) or
                    _enums.model_type_is_sd2(self.model_type) or
                    _enums.model_type_is_sdxl(self.model_type) or
                    _enums.model_type_is_kolors(self.model_type)):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'DeepCache is only supported for Stable Diffusion, Stable Diffusion XL, and Kolors.'
                )

        for prop in args.__dict__.keys():
            if prop.startswith('second_model_deep_cache_'):
                value = getattr(args, prop)
                if value is not None or (isinstance(value, bool) and value is True):
                    args.second_model_deep_cache = True
                    break

    def _auto_tea_cache_check(self, args: DiffusionArguments):
        for prop in args.__dict__.keys():
            if prop.startswith('tea_cache_'):
                value = getattr(args, prop)
                if value is not None or (isinstance(value, bool) and value is True):
                    args.tea_cache = True
                    break

        if args.tea_cache:
            if not _enums.model_type_is_flux(self.model_type):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'TeaCache is only supported for Flux.')

            if self.model_cpu_offload:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'TeaCache does not support model CPU offloading.')

    def __call__(self, args: DiffusionArguments | None = None, **kwargs) -> PipelineWrapperResult:
        """
        Call the pipeline and generate a result.

        :param args: Optional :py:class:`.DiffusionArguments`

        :param kwargs: See :py:meth:`.DiffusionArguments.get_pipeline_wrapper_kwargs`,
            any keyword arguments given here will override values derived from the
            :py:class:`.DiffusionArguments` object given to the *args* parameter.

        :raises InvalidModelFileError:
        :raises InvalidModelUriError:
        :raises InvalidSchedulerNameError:
        :raises dgenerate.OutOfMemoryError:
        :raises UnsupportedPipelineConfigError:

        :return: :py:class:`.PipelineWrapperResult`
        """

        copy_args = DiffusionArguments()

        if args is not None:
            copy_args.set_from(args)

        copy_args.set_from(kwargs, missing_value_throws=False)

        help_text = self._argument_help_check(copy_args)
        if help_text:
            raise DiffusionArgumentsHelpException(help_text)

        _messages.debug_log(f'Calling Pipeline Wrapper: "{self}"')
        _messages.debug_log(f'Pipeline Wrapper Args: ',
                            lambda: _textprocessing.debug_format_args(
                                copy_args.get_pipeline_wrapper_kwargs()))

        _enforce_cache_constraints()

        loaded_new = self._lazy_init_pipeline(copy_args)

        # this needs to happen even if a cached pipeline
        # was loaded, since the settings for scheduler
        # and vae tiling / slicing may be different
        self._set_scheduler_and_vae_settings(args)

        if loaded_new:
            _enforce_cache_constraints()

        pipeline_args = \
            self._get_pipeline_defaults(user_args=copy_args)

        self._auto_tea_cache_check(copy_args)
        self._auto_ras_check(copy_args)
        self._auto_deep_cache_check(copy_args)

        if self.model_type == _enums.ModelType.TORCH_S_CASCADE:
            if args.hi_diffusion:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'HiDiffusion is not supported for Stable Cascade.'
                )

            result = self._call_torch_s_cascade(
                pipeline_args=pipeline_args,
                user_args=copy_args)
        elif _enums.model_type_is_flux(self.model_type):
            if args.hi_diffusion:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'HiDiffusion is not supported for Flux.'
                )

            result = self._call_torch_flux(pipeline_args=pipeline_args,
                                           user_args=copy_args)
        else:
            if args.hi_diffusion and _enums.model_type_is_sd3(self.model_type):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'HiDiffusion is not supported for SD3.'
                )

            result = self._call_torch(pipeline_args=pipeline_args,
                                      user_args=copy_args)

        DiffusionPipelineWrapper.__LAST_CALLED = self

        return result


__all__ = _types.module_all()
