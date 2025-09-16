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
import functools
import importlib.util
import inspect
import math
import typing

import DeepCache as _deepcache
import PIL.Image
import diffusers
import numpy
import torch

import dgenerate.eval as _eval
import dgenerate.extras.asdff.base as _asdff_base
import dgenerate.extras.hidiffusion as _hidiffusion
import dgenerate.extras.sada.patch as _sada
import dgenerate.extras.teacache.teacache_flux as _teacache_flux
import dgenerate.hfhub as _hfhub
import dgenerate.image as _image
import dgenerate.imageprocessors as _imageprocessors
import dgenerate.latentsprocessors as _latentsprocessors
import dgenerate.mediainput as _mediainput
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.help as _help
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.schedulers as _schedulers
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.pipelinewrapper.util as _util
import dgenerate.prompt as _prompt
import dgenerate.promptweighters as _promptweighters
import dgenerate.textprocessing as _textprocessing
import dgenerate.torchutil as _torchutil
import dgenerate.types as _types
from dgenerate.extras.ras import RASArgs as _RASArgs
from dgenerate.extras.ras import sd3_ras_context as _sd3_ras_context
from dgenerate.pipelinewrapper.arguments import DiffusionArguments
from dgenerate.pipelinewrapper.denoise_range import DenoiseRangeError as _DenoiseRangeError
from dgenerate.pipelinewrapper.denoise_range import denoise_range as _denoise_range
from dgenerate.pipelinewrapper.denoise_range import supports_native_denoising_start as _supports_native_denoising_start


class _InpaintCropInfo:
    """
    Contains state information for inpaint crop processing.
    
    This object stores all necessary information to apply inpaint crop pasting
    after the diffusion process completes.
    """

    def __init__(self,
                 original_images: list[PIL.Image.Image],
                 original_masks: list[PIL.Image.Image] | None,
                 crop_bounds: tuple[int, int, int, int],
                 use_masked: bool = False,
                 feather: int | None = None):
        """
        Initialize inpaint crop information.
        
        :param original_images: List of original uncropped images to paste onto
        :param original_masks: List of original uncropped masks (for masked pasting)
        :param crop_bounds: Crop bounds as (left, top, right, bottom)
        :param use_masked: Whether to use masked pasting
        :param feather: Optional feather value for feathered pasting
        """
        self.original_images = original_images
        self.original_masks = original_masks
        self.crop_bounds = crop_bounds
        self.use_masked = use_masked
        self.feather = feather

    def __repr__(self) -> str:
        return (f"_InpaintCropInfo(original_images={len(self.original_images)}, "
                f"crop_bounds={self.crop_bounds}, use_masked={self.use_masked}, "
                f"feather={self.feather})")


class DiffusionArgumentsHelpException(Exception):
    """
    Thrown when a :py:class:`DiffusionArguments` attribute that supports
    passing a help request value (such as :py:attr:`DiffusionArguments.scheduler_uri`)
    is passed its help value.

    This exception returns the help string to the caller.
    """
    pass


class PipelineWrapperResult:
    """
    The result of calling :py:class:`.DiffusionPipelineWrapper`
    """
    images: _types.MutableImages | None
    latents: _types.MutableTensors | None

    @property
    def image_count(self) -> int:
        """
        The number of images produced.

        :return: int
        """
        if self.images is None:
            return 0

        return len(self.images)

    @property
    def latents_count(self) -> int:
        """
        The number of latents produced.

        :return: int
        """
        if self.latents is None:
            return 0

        return len(self.latents)

    @property
    def output_count(self) -> int:
        """
        The number of outputs produced (images or latents).

        :return: int
        """
        return max(self.image_count, self.latents_count)

    @property
    def image(self) -> PIL.Image.Image | None:
        """
        The first image in the batch of requested batch size.

        :return: :py:class:`PIL.Image.Image`
        """
        return self.images[0] if self.images else None

    @property
    def latent(self) -> torch.Tensor | None:
        """
        The first latent in the batch of requested batch size.

        :return: :py:class:`torch.Tensor`
        """
        return self.latents[0] if self.latents else None

    @property
    def has_images(self) -> bool:
        """
        Whether this result contains images.

        :return: bool
        """
        return self.images is not None and len(self.images) > 0

    @property
    def has_latents(self) -> bool:
        """
        Whether this result contains latents.

        :return: bool
        """
        return self.latents is not None and len(self.latents) > 0

    def image_grid(self, cols_rows: _types.Size):
        """
        Render an image grid from the images in this result.

        :raise ValueError: if no images are present on this object.
            This is impossible if this object was produced by :py:class:`.DiffusionPipelineWrapper`.
        :raise ValueError: if this result contains latents instead of images.
            Image grids can only be created from decoded images, not raw latent tensors.

        :param cols_rows: columns and rows (WxH) desired as a tuple
        :return: :py:class:`PIL.Image.Image`
        """
        if not self.images:
            if self.has_latents:
                raise ValueError(
                    'Cannot create image grid from latent tensors. '
                    'Image grids can only be created from decoded images, not raw latent tensors. '
                    'Use output_latents=False to get decoded images instead.'
                )
            else:
                raise ValueError('No images present.')

        if len(self.images) == 1:
            return self.images[0]

        cols, rows = cols_rows

        w, h = self.images[0].size
        grid = PIL.Image.new('RGB', size=(cols * w, rows * h))
        for i, img in enumerate(self.images):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

    def __init__(self, images: _types.Images | None = None, latents: _types.MutableTensors | None = None):
        if images is None and latents is None:
            raise ValueError("PipelineWrapperResult must have either images or latents, both cannot be None")
        if images is not None and latents is not None:
            raise ValueError("PipelineWrapperResult cannot have both images and latents, only one is allowed")

        self.images = images
        self.latents = latents
        self.dgenerate_opts = list()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.images is not None:
            for i in self.images:
                if i is not None:
                    i.close()
            self.images = None
        # Latents don't need explicit cleanup like PIL images


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


@contextlib.contextmanager
def _hi_diffusion(
        pipeline,
        generator,
        enabled: bool,
        no_raunet: bool | None = None,
        no_window_attn: bool | None = None
):
    if enabled:
        sd15cn = pipeline.__class__.__name__.startswith('StableDiffusionControlNet')

        if no_raunet is None:
            no_raunet = sd15cn
        if no_window_attn is None:
            no_window_attn = sd15cn

        _messages.debug_log(
            f'Enabling HiDiffusion on pipeline: {pipeline.__class__.__name__} '
            f'(no_raunet={no_raunet}, no_window_attn={no_window_attn})')
        _hidiffusion.apply_hidiffusion(
            pipeline,
            generator=generator,
            apply_raunet=not no_raunet,
            apply_window_attn=not no_window_attn
        )
    try:
        yield
    finally:
        if enabled:
            _messages.debug_log(
                f'Disabling HiDiffusion on pipeline: {pipeline.__class__.__name__}')
            _hidiffusion.remove_hidiffusion(pipeline)


@contextlib.contextmanager
def _sada_context(
        pipeline,
        width: int,
        height: int,
        enabled: bool,
        max_downsample: int = 1,
        sx: int = 2,
        sy: int = 2,
        acc_range: tuple = (10, 47),
        lagrange_term: int = 0,
        lagrange_int: int | None = None,
        lagrange_step: int | None = None,
        max_fix: int = 5 * 1024,
        max_interval: int = 4,

):
    """
    Context manager for SADA (Stability-guided Adaptive Diffusion Acceleration).
    """
    try:
        if enabled:
            # Calculate latent size for transformer models (SD3, Flux)
            latent_size = None
            if hasattr(pipeline, 'transformer'):
                # For Flux and other transformer models, calculate latent size based on width/height
                # Based on sada-icml examples: latent_size = (height // 16, width // 16)
                latent_size = (height // 16, width // 16)

            def debug_message(args = locals()):
                args.pop('pipeline')
                return f'Enabling SADA on pipeline: {pipeline.__class__.__name__}, Args: {args}'

            _messages.debug_log(debug_message)

            _sada.apply_patch(
                pipeline,
                max_downsample=max_downsample,
                sx=sx,
                sy=sy,
                latent_size=latent_size,
                acc_range=acc_range,
                lagrange_term=lagrange_term,
                lagrange_int=lagrange_int,
                lagrange_step=lagrange_step,
                max_fix=max_fix,
                max_interval=max_interval
            )

        yield
    except _sada.exceptions.SADAUnsupportedError as e:
        raise _pipelines.UnsupportedPipelineConfigError(str(e)) from e
    finally:
        if enabled:
            _messages.debug_log(
                f'Disabling SADA on pipeline: {pipeline.__class__.__name__}')
            _sada.remove_patch(pipeline)


@contextlib.contextmanager
def _freeu(pipeline, params: tuple[float, float, float, float] | None):
    if params is not None:
        _messages.debug_log(
            f'Enabling FreeU on pipeline: {pipeline.__class__.__name__}')
        pipeline.enable_freeu(*params)
    try:
        yield
    finally:
        if params is not None:
            _messages.debug_log(
                f'Disabling FreeU on pipeline: {pipeline.__class__.__name__}')
            pipeline.disable_freeu()


class DiffusionPipelineWrapper:
    """
    Monolithic diffusion pipelines wrapper.
    """

    __LAST_RECALL_PIPELINE: _pipelines.PipelineFactory = None
    __LAST_RECALL_SECONDARY_PIPELINE: _pipelines.PipelineFactory = None

    @staticmethod
    def _normalize_uris(uris: _types.OptionalUris | str | None) -> _types.OptionalUris:
        """
        Normalize URI arguments - convert single strings to lists.

        :param uris: Single URI string, list of URIs, or None
        :return: List of URIs or None
        """
        if uris is None:
            return None
        if isinstance(uris, str):
            return [uris]
        return uris

    def __init__(self,
                 model_path: _types.Path,
                 model_type: _enums.ModelType | str = _enums.ModelType.SD,
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
                 quantizer_map: _types.OptionalStrings = None,
                 second_model_quantizer_uri: _types.OptionalUri = None,
                 second_model_quantizer_map: _types.OptionalStrings = None,
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
                 latents_processor_loader: _latentsprocessors.LatentsProcessorLoader | None = None,
                 decoded_latents_image_processor_loader: _imageprocessors.ImageProcessorLoader | None = None,
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
        :param quantizer_map: Collection of pipeline submodule names to which quantization should be applied when
            ``quantizer_uri`` is provided. Valid values include: ``unet``, ``transformer``, ``text_encoder``,
            ``text_encoder_2``, ``text_encoder_3``. If ``None``, all supported modules will be quantized.
        :param second_model_quantizer_uri: Global --second-model-quantizer URI value
        :param second_model_quantizer_map: Collection of pipeline submodule names to which quantization should be
            applied when ``second_model_quantizer_uri`` is provided. Valid values include: ``unet``,
            ``transformer``, ``text_encoder``, ``text_encoder_2``, ``text_encoder_3``.
            If ``None``, all supported modules will be quantized.
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
        :param latents_processor_loader: Plugin loader for latents processor implementations, if you pass ``None`` a default instance will be created.
        :param decoded_latents_image_processor_loader: Plugin loader for image processor implementations that process images decoded from incoming latents, if you pass ``None`` a default instance will be created.
        :param adetailer_detector_uris: adetailer subject detection model URIs, specifying this argument indicates ``img2img`` mode implicitly,
            the pipeline wrapper will accept a single image and perform the adetailer inpainting algorithm on it using the provided
            detector URIs.
        :param adetailer_crop_control_image: Should adetailer crop any provided ControlNet control image
            in the same way that it crops the generated mask to the detection area? Otherwise,
            use the full control image resized down to the size of the detection area. If you enable
            this and your control image is not the same size as your input image, a warning will be
            issued and resizing will be used instead of cropping.

        :raises UnsupportedPipelineConfigError:
        :raises InvalidModelUriError:
        """

        # Normalize URI arguments - convert strings to lists where needed
        lora_uris = self._normalize_uris(lora_uris)
        ip_adapter_uris = self._normalize_uris(ip_adapter_uris)
        textual_inversion_uris = self._normalize_uris(textual_inversion_uris)
        text_encoder_uris = self._normalize_uris(text_encoder_uris)
        second_model_text_encoder_uris = self._normalize_uris(second_model_text_encoder_uris)
        controlnet_uris = self._normalize_uris(controlnet_uris)
        t2i_adapter_uris = self._normalize_uris(t2i_adapter_uris)
        adetailer_detector_uris = self._normalize_uris(adetailer_detector_uris)

        # Check that model_path is provided
        if model_path is None:
            raise ValueError('model_path must be specified')

        # Check for valid device string
        if not _torchutil.is_valid_device_string(device):
            raise _pipelines.UnsupportedPipelineConfigError(
                f'Invalid device argument, {_torchutil.invalid_device_message(device, cap=False)}')

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

        if image_encoder_uri and not ip_adapter_uris and model_type != _enums.ModelType.S_CASCADE:
            raise _pipelines.UnsupportedPipelineConfigError(
                'Cannot use "image_encoder_uri" without "ip_adapter_uris" '
                'if "model_type" is not S_CASCADE.'
            )

        if not _hfhub.is_single_file_model_load(model_path):
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
                    not _hfhub.is_single_file_model_load(
                        _uris.SDXLRefinerUri.parse(sdxl_refiner_uri).model):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'You cannot specify "second_model_original_config" '
                    'when the "sdxl_refiner_uri" model is not a '
                    'single file checkpoint.'
                )
            if s_cascade_decoder_uri and \
                    not _hfhub.is_single_file_model_load(
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
                    'please use model_type "sdxl" if you are trying to load an sdxl model.'
                )

        if s_cascade_decoder_uri is not None:
            if not _enums.model_type_is_s_cascade(model_type):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Only Stable Cascade models support decoders, '
                    'please use model_type "s-cascade" if you are trying to load an Stable Cascade model.'
                )

        if transformer_uri:
            if not _enums.model_type_is_sd3(model_type) and not _enums.model_type_is_flux(model_type):
                raise _pipelines.UnsupportedPipelineConfigError(
                    '--transformer is only supported for --model-type sd3 and flux.')

        if adetailer_detector_uris and model_type not in {
            _enums.ModelType.SD,
            _enums.ModelType.SDXL,
            _enums.ModelType.KOLORS,
            _enums.ModelType.SD3,
            _enums.ModelType.FLUX,
            _enums.ModelType.FLUX_FILL
        }:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'--adetailer-detectors is only compatible with '
                f'--model-type sd, sdxl, kolors, sd3, and flux')

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
                    raise _pipelines.UnsupportedPipelineConfigError(
                        f'Unknown quantizer_map value: {map_value}, '
                        f'must be one of: {_textprocessing.oxford_comma(quantizer_map_vals, "or")}'
                    )

        if second_model_quantizer_map is not None:
            for map_value in second_model_quantizer_map:
                if map_value not in quantizer_map_vals:
                    raise _pipelines.UnsupportedPipelineConfigError(
                        f'Unknown second_model_quantizer_map value: {map_value}, '
                        f'must be one of: {_textprocessing.oxford_comma(quantizer_map_vals, "or")}'
                    )

        self._quantizer_uri = quantizer_uri
        self._quantizer_map = quantizer_map
        self._second_model_quantizer_uri = second_model_quantizer_uri
        self._second_model_quantizer_map = second_model_quantizer_map
        self._subfolder = subfolder
        self._device = device
        self._model_type = _enums.get_model_type_enum(model_type)
        self._model_path = model_path
        self._pipeline = None
        self._revision = revision
        self._variant = variant
        self._dtype = _enums.get_data_type_enum(dtype)
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
        self._recall_secondary_pipeline = None
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

        self._latents_processor_loader = \
            latents_processor_loader if latents_processor_loader is not None \
                else _latentsprocessors.LatentsProcessorLoader()

        self._decoded_latents_image_processor_loader = \
            decoded_latents_image_processor_loader if decoded_latents_image_processor_loader is not None \
                else _imageprocessors.ImageProcessorLoader()

        self._adetailer_detector_uris = adetailer_detector_uris
        self._parsed_adetailer_detector_uris = None

        self._adetailer_crop_control_image = adetailer_crop_control_image

        # Initialize inpaint crop info (used internally for crop/paste operations)
        self._inpaint_crop_info = None

        if adetailer_detector_uris:
            self._parsed_adetailer_detector_uris = []
            for adetailer_detector_uri in adetailer_detector_uris:
                self._parsed_adetailer_detector_uris.append(
                    _uris.AdetailerDetectorUri.parse(adetailer_detector_uri))

        # storage for determination of render width/height

        self._inference_width = None
        self._inference_height = None

    @property
    def prompt_weighter_loader(self) -> _promptweighters.PromptWeighterLoader:
        """
        Current prompt weighter loader.
        """
        return self._prompt_weighter_loader

    @property
    def latents_processor_loader(self) -> _latentsprocessors.LatentsProcessorLoader:
        """
        Current latents processor loader.
        """
        return self._latents_processor_loader

    @property
    def decoded_latents_image_processor_loader(self) -> _imageprocessors.ImageProcessorLoader:
        """
        Current decoded latents image processor loader.
        """
        return self._decoded_latents_image_processor_loader

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
    def quantizer_map(self) -> _types.OptionalStrings:
        """
        Current ``--quantizer-map`` value.
        """
        return list(self._quantizer_map) if self._quantizer_map is not None else None

    @property
    def second_model_quantizer_uri(self) -> _types.OptionalUri:
        """
        Current ``--second-model-quantizer`` value.
        """
        return self._second_model_quantizer_uri

    @property
    def second_model_quantizer_map(self) -> _types.OptionalStrings:
        """
        Current ``--second-model-quantizer-map`` value.
        """
        return list(self._second_model_quantizer_map) if self._second_model_quantizer_map is not None else None

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

    @staticmethod
    def recall_last_used_main_pipeline() -> typing.Optional[_pipelines.PipelineCreationResult]:
        """
        Return a reference to the last :py:class:`dgenerate.pipelinewrapper.pipelines.TorchPipelineCreationResult`
        for the pipeline that successfully executed an image generation.

        This may recreate the pipeline if it is not cached.

        If no image generation has occurred, this will return ``None``.

        :return: :py:class:`dgenerate.pipelinewrapper.pipelines.TorchPipelineCreationResult` or ``None``
        """
        if DiffusionPipelineWrapper.__LAST_RECALL_PIPELINE is None:
            return None

        return DiffusionPipelineWrapper.__LAST_RECALL_PIPELINE()

    @staticmethod
    def recall_last_used_secondary_pipeline() -> typing.Optional[_pipelines.PipelineCreationResult]:
        """
        Return a reference to the last :py:class:`dgenerate.pipelinewrapper.pipelines.TorchPipelineCreationResult`
        for the secondary pipeline (refiner / stable cascade decoder) that successfully executed an image generation.

        This may recreate the pipeline if it is not cached.

        If no image generation has occurred or no secondary pipeline has been called, this will return ``None``.

        :return: :py:class:`dgenerate.pipelinewrapper.pipelines.TorchPipelineCreationResult` or ``None``
        """
        if DiffusionPipelineWrapper.__LAST_RECALL_SECONDARY_PIPELINE is None:
            return None

        return DiffusionPipelineWrapper.__LAST_RECALL_SECONDARY_PIPELINE()

    def reconstruct_dgenerate_opts(self,
                                   args: DiffusionArguments | None = None,
                                   extra_opts:
                                   collections.abc.Sequence[
                                       tuple[str] | tuple[str, typing.Any]] | None = None,
                                   omit_device: bool = False,
                                   shell_quote: bool = True,
                                   overrides: dict[str, typing.Any] = None) -> \
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

        :param overrides: pipeline wrapper keyword arguments, these will override values derived from
            any :py:class:`.DiffusionArguments` object given to the *args* argument. See:
            :py:class:`.DiffusionArguments.get_pipeline_wrapper_kwargs`

        :return: List of tuples of length 1 or 2 representing the option
        """
        import dgenerate.pipelinewrapper.argreconstruct as _a

        return _a.reconstruct_dgenerate_opts(
            self, args, extra_opts, omit_device, shell_quote, overrides
        )

    def gen_dgenerate_config(self,
                             args: DiffusionArguments | None = None,
                             extra_opts:
                             collections.abc.Sequence[tuple[str] | tuple[str, typing.Any]] | None = None,
                             extra_comments: collections.abc.Iterable[str] | None = None,
                             omit_device: bool = False,
                             overrides: dict[str, typing.Any] = None):
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
        :param overrides: pipeline wrapper keyword arguments, these will override values derived from
            any :py:class:`.DiffusionArguments` object given to the *args* argument. See:
            :py:class:`.DiffusionArguments.get_pipeline_wrapper_kwargs`
        :return: The configuration as a string
        """
        import dgenerate.pipelinewrapper.argreconstruct as _a

        return _a.gen_dgenerate_config(
            self, args, extra_opts, extra_comments, omit_device, overrides
        )

    def gen_dgenerate_command(self,
                              args: DiffusionArguments | None = None,
                              extra_opts:
                              collections.abc.Sequence[tuple[str] | tuple[str, typing.Any]] | None = None,
                              omit_device: bool = False,
                              overrides: dict[str, typing.Any] = None):
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
        :param overrides: pipeline wrapper keyword arguments, these will override values derived from
            any :py:class:`.DiffusionArguments` object given to the *args* argument. See:
            :py:class:`.DiffusionArguments.get_pipeline_wrapper_kwargs`
        :return: A string containing the dgenerate command line needed to reproduce this result.
        """
        import dgenerate.pipelinewrapper.argreconstruct as _a

        return _a.gen_dgenerate_command(
            self, args, extra_opts, omit_device, overrides
        )

    @staticmethod
    def _separate_images_and_tensors(items: _types.ImagesOrTensors | None) \
            -> tuple[list[PIL.Image.Image] | None, list[torch.Tensor] | None]:
        """
        Separate a sequence of images or tensors into separate sequences.

        Note: The input should be homogeneous (all images or all tensors), but this method
        can handle mixed inputs for validation purposes.

        :param items: Sequence of PIL Images or torch Tensors (should be homogeneous), or None
        :return: Tuple of (images, tensors) where each can be None if no items of that type exist
        """
        if items is None:
            return None, None

        images, tensors = _mediainput.separate_images_and_tensors(items)
        return images if images else None, tensors if tensors else None

    def _validate_latent_channels(self, tensors: _types.Tensors):
        """
        Validate that latent tensors have the correct number of channels for the current model type.

        :param tensors: Sequence of tensors to validate
        :raises UnsupportedPipelineConfigError: If tensors have incorrect number of channels
        """

        if _enums.model_type_is_s_cascade(self.model_type):
            raise _pipelines.UnsupportedPipelineConfigError(
                'Stable Cascade does not support accepting latents as input.'
            )

        # Get expected channels based on model type
        if _enums.model_type_is_flux(self.model_type):
            # Flux uses unpacked format [B, C, H, W] or [C, H, W] for external interface
            # where C is 16 (64/4 from the internal packed format)
            expected_channels = 16  # Flux models expect 16 channels in unpacked format
            for i, tensor in enumerate(tensors):
                if len(tensor.shape) not in (3, 4):
                    raise _pipelines.UnsupportedPipelineConfigError(
                        f'Invalid shape for Flux latents tensor at index {i}. '
                        f'Expected 3D [C, H, W] or 4D [B, C, H, W] tensor in unpacked format, '
                        f'but got shape {tensor.shape}'
                    )
                channels = tensor.shape[1 - (4 - len(tensor.shape))]  # Channel is at index 1 for 4D, 0 for 3D
                if channels != expected_channels:
                    raise _pipelines.UnsupportedPipelineConfigError(
                        f'Invalid number of channels in Flux latents tensor at index {i}. '
                        f'Expected {expected_channels} channels in unpacked format, '
                        f'but got {channels} channels instead. Shape: {tensor.shape}'
                    )
        elif _enums.model_type_is_sd3(self.model_type):
            # SD3 uses 16 channels in latent space
            expected_channels = self._pipeline.transformer.config.in_channels

            for i, tensor in enumerate(tensors):
                if len(tensor.shape) not in (3, 4):  # Must be [C, H, W] or [B, C, H, W]
                    raise _pipelines.UnsupportedPipelineConfigError(
                        f'Invalid shape for SD3 latents tensor at index {i}. '
                        f'Expected 3D [C, H, W] or 4D tensor [B, C, H, W], but got shape {tensor.shape}'
                    )
                channels = tensor.shape[1 - (4 - len(tensor.shape))]
                if channels != expected_channels:
                    raise _pipelines.UnsupportedPipelineConfigError(
                        f'Invalid number of channels in SD3 latents tensor at index {i}. '
                        f'Expected {expected_channels} channels for model type "{self.model_type_string}", '
                        f'but got {channels} channels instead. Shape: {tensor.shape}'
                    )
        else:
            # Standard SD models use channels from VAE config
            expected_channels = 4  # Default if not specified in config
            if hasattr(self._pipeline.vae, 'config'):
                if hasattr(self._pipeline.vae.config, 'latent_channels'):
                    expected_channels = self._pipeline.vae.config.latent_channels
                # Some models use in_channels instead
                elif hasattr(self._pipeline.vae.config, 'in_channels'):
                    expected_channels = self._pipeline.vae.config.in_channels

            for i, tensor in enumerate(tensors):
                if len(tensor.shape) not in (3, 4):  # Must be [C, H, W] or [B, C, H, W]
                    raise _pipelines.UnsupportedPipelineConfigError(
                        f'Invalid shape for latents tensor at index {i}. '
                        f'Expected 3D [C, H, W] or 4D tensor [B, C, H, W], but got shape {tensor.shape}'
                    )
                channels = tensor.shape[1 - (4 - len(tensor.shape))]
                if channels != expected_channels:
                    raise _pipelines.UnsupportedPipelineConfigError(
                        f'Invalid number of channels in latents tensor at index {i}. '
                        f'Expected {expected_channels} channels for model type "{self.model_type_string}", '
                        f'but got {channels} channels instead. Shape: {tensor.shape}')

    @staticmethod
    def _validate_images_all_same_size(title, images):
        first_image_size = images[0].size
        # Check if all control images have the same size
        for img in images[1:]:
            if img.size != first_image_size:
                raise _pipelines.UnsupportedPipelineConfigError(
                    f"All {title} must have the same dimension.")

    def _resize_images_to_user_dimensions(
            self,
            images: _types.Images,
            user_args: DiffusionArguments
    ) -> list[PIL.Image.Image]:
        """
        Resize images to user-specified width and height using dgenerate's image resize utility.

        :param images: List of PIL Images to resize
        :param user_args: DiffusionArguments containing width and height
        :return: List of resized PIL Images
        """
        if not images:
            return []


        target_size = self._calc_image_target_size(images[0], user_args)

        resized_images = []

        for img in images:

            new_size = _image.resize_image_calc(
                old_size=img.size,
                new_size=target_size,
                aspect_correct=user_args.aspect_correct,
                align=8)

            if img.size != new_size:
                img = _image.resize_image(img=img, size=new_size)

            resized_images.append(img.convert('RGB'))

        return resized_images

    @staticmethod
    def _process_ip_adapter_images(images: _types.OptionalImagesSequence):
        """
        Align IP Adapter images by 8
        :param images: sequence of image sequences
        :return: processed array
        """
        if not images:
            return None

        output = []

        for img_s in images:
            processed_images = []

            for img in img_s:

                new_size = _image.resize_image_calc(old_size=img.size, new_size=None, align=8)

                if img.size != new_size:
                    img = _image.resize_image(img=img, size=new_size)

                processed_images.append(img.convert('RGB'))

            output.append(processed_images)

        return output

    @staticmethod
    def _process_floyd_image(image: _types.ImageOrTensor):
        """
        Align floyd image by 8
        :param image: floyd image (maybe tensor)
        :return: processed image, untouched tensor
        """
        if not isinstance(image, torch.Tensor):
            new_size = _image.resize_image_calc(old_size=image.size, new_size=None, align=8)

            if image.size != new_size:
                return _image.resize_image(image, size=None, align=8)

        return image

    def _apply_inpaint_crop(self,
                            images: list[PIL.Image.Image],
                            masks: list[PIL.Image.Image],
                            control_images: list[PIL.Image.Image] | None,
                            padding: int | tuple[int, int] | tuple[int, int, int, int],
                            decoded_latents: bool,
                            user_args: DiffusionArguments) \
            -> tuple[
                list[PIL.Image.Image],
                list[PIL.Image.Image],
                list[PIL.Image.Image] | None,
                tuple[int, int, int, int]
            ]:
        """
        Crop images, masks, and control images to mask bounds with padding.
        
        :param images: List of images to crop
        :param masks: List of masks to crop
        :param control_images: Optional list of control images to crop
        :param padding: Padding around mask bounds (left, top, right, bottom)
        :param decoded_latents: Were ``images`` decoded from latents?
        :param user_args: diffusion arguments for reference
        :return: Tuple of (cropped_images, cropped_masks, cropped_control_images, crop_bounds)
        """
        if not masks:
            raise _pipelines.UnsupportedPipelineConfigError(
                "Cannot apply inpaint crop without masks."
            )

        # Calculate bounds for the single mask
        crop_bounds = _image.find_mask_bounds(masks[0], padding)
        if crop_bounds is None:
            raise _pipelines.UnsupportedPipelineConfigError(
                "No white pixels found in mask for inpaint crop."
            )

        cropped_images = [images[0].crop(crop_bounds)]
        cropped_masks = [masks[0].crop(crop_bounds)]

        cropped_control_images = None
        if control_images:
            cropped_control_images = [control_images[0].crop(crop_bounds)]

        # Process decoded images if processors are configured
        # and they were decoded from latents

        # this is here so we can honor processors pre-resize settings
        # the cropped image is what we want to process

        if decoded_latents and user_args.decoded_latents_image_processor_uris:
            cropped_images = self._process_decoded_latents_images(
                cropped_images,
                user_args.decoded_latents_image_processor_uris,
                user_args,
            )

        # Since we only allow single images with inpaint_crop, 
        # we'll always have exactly one set of bounds

        return cropped_images, cropped_masks, cropped_control_images, crop_bounds

    def _paste_inpaint_result(self,
                              original_images: list[PIL.Image.Image],
                              generated_images: list[PIL.Image.Image],
                              crop_bounds: tuple[int, int, int, int],
                              masks: list[PIL.Image.Image] = None,
                              feather: int = None) -> list[PIL.Image.Image]:
        """
        Paste generated images back onto original images at crop bounds.
        
        :param original_images: List of original uncropped images
        :param generated_images: List of generated images to paste
        :param crop_bounds: Bounds where to paste (left, top, right, bottom)
        :param masks: Optional masks for masked pasting
        :param feather: Optional feather value for feathered pasting
        :return: List of images with generated content pasted back
        """
        result_images = []

        for i, generated in enumerate(generated_images):
            # Since inpaint_crop doesn't support batching, map generated images to the single original
            original = original_images[0]
            background_image = original.copy()

            # Use the single crop bounds for all generated images

            # Resize generated image to fit the crop bounds
            crop_size = (crop_bounds[2] - crop_bounds[0], crop_bounds[3] - crop_bounds[1])

            if generated.size != crop_size:
                _messages.debug_log(
                    f'Inpaint crop paste: Resizing generated image {i} from {generated.size} to {crop_size} for bounds {crop_bounds}')
                resampling = _image.best_pil_resampling(generated.size, crop_size)
                generated = generated.resize(crop_size, resampling)

            if feather is not None:
                # Use feathered pasting
                background_image = _image.paste_with_feather(
                    background=background_image,
                    foreground=generated,
                    location=crop_bounds,
                    feather=feather,
                    shape='rectangle'
                )
            elif masks:
                # Use masked pasting (single mask since we don't support batching)
                mask = masks[0]

                # Crop and resize mask to match generated image size
                cropped_mask = mask.crop(crop_bounds)
                if cropped_mask.size != crop_size:
                    mask_resampling = _image.best_pil_resampling(cropped_mask.size, crop_size)
                    cropped_mask = cropped_mask.resize(crop_size, mask_resampling)

                # Convert to grayscale if needed
                if cropped_mask.mode != 'L':
                    cropped_mask = cropped_mask.convert('L')

                background_image.paste(generated, crop_bounds, cropped_mask)
            else:
                # Simple paste without transparency
                background_image.paste(generated, crop_bounds)

            result_images.append(background_image)

        return result_images

    def _set_pipeline_strength(self, user_args: DiffusionArguments, pipeline_args: dict[str, typing.Any]):
        strength = float(_types.default(user_args.image_seed_strength, _constants.DEFAULT_IMAGE_SEED_STRENGTH))
        ifs = int(_types.default(user_args.inference_steps, _constants.DEFAULT_INFERENCE_STEPS))
        if (strength * ifs) < 1.0:
            strength = 1.0 / ifs
            _messages.warning(
                f'image-seed-strength * inference-steps '
                f'was calculated at < 1, image-seed-strength defaulting to (1.0 / inference-steps): {strength}'
            )

        pipeline_args['strength'] = strength

    def _set_pipeline_controlnet_defaults(self, user_args: DiffusionArguments, pipeline_args: dict[str, typing.Any]):
        control_images = user_args.control_images

        if not control_images:
            raise _pipelines.UnsupportedPipelineConfigError(
                'Must provide control_images argument when using ControlNet models.')

        # sanity check that control images are the same dimension
        self._validate_images_all_same_size(
            "control guidance images", control_images
        )

        # Resize control images to user-specified dimensions first thing
        control_images = self._resize_images_to_user_dimensions(
            control_images, user_args
        )

        image_arg_inputs = self._get_pipeline_img2img_inputs(user_args)

        if image_arg_inputs is not None:
            non_latent_input = not _torchutil.is_tensor(image_arg_inputs[0])

            if non_latent_input:
                if not image_arg_inputs[0].size == control_images[0].size:
                    raise _pipelines.UnsupportedPipelineConfigError(
                        'Img2Img images and ControlNet images must be equal in dimension.'
                    )
            else:
                if not self.get_decoded_latents_size(image_arg_inputs[0]) == control_images[0].size:
                    raise _pipelines.UnsupportedPipelineConfigError(
                        'Img2Img latents must decode to the same dimension as any provided ControlNet images.'
                    )

        control_images_cnt = len(control_images)
        controlnet_uris_cnt = len(self._controlnet_uris)

        if control_images_cnt != controlnet_uris_cnt:
            # User provided a mismatched number of ControlNet models and control_images, behavior is undefined.
            raise _pipelines.UnsupportedPipelineConfigError(
                f'You specified {control_images_cnt} control guidance images and '
                f'only {controlnet_uris_cnt} ControlNet URIs. The amount of '
                f'control guidance images must be equal to the amount of ControlNet URIs.')
        else:
            # set dimensions to match the control image
            self._set_pipe_dimensions(
                None, None,
                control_images[0].width, control_images[0].height,
                pipeline_args
            )

        sdxl_cn_union = _enums.model_type_is_sdxl(self._model_type) and \
                        any(p.mode is not None for p in self._parsed_controlnet_uris)

        if self._pipeline_type == _enums.PipelineType.TXT2IMG:
            if _enums.model_type_is_sd3(self._model_type):
                # Handle SD3 model specifics for control images
                pipeline_args['control_image'] = self._sd3_force_control_to_a16(
                    pipeline_args, control_images, user_args
                )
            elif _enums.model_type_is_flux(self._model_type):
                pipeline_args['control_image'] = control_images
            elif sdxl_cn_union:
                # controlnet union pipeline does not use "image"
                # it also destructively modifies
                # this input value if it is a list for
                # whatever reason
                pipeline_args['control_image'] = list(control_images)
            else:
                pipeline_args['image'] = control_images
        elif self._pipeline_type in {_enums.PipelineType.IMG2IMG, _enums.PipelineType.INPAINT}:
            pipeline_args['image'] = image_arg_inputs
            pipeline_args['control_image'] = control_images if not sdxl_cn_union else list(control_images)
            self._set_pipeline_strength(user_args, pipeline_args)

        mask_images = user_args.mask_images
        if mask_images is not None:
            # Resize mask images to user-specified dimensions (includes RGB conversion)
            self._validate_images_all_same_size("inpaint mask images", mask_images)
            mask_images = self._resize_images_to_user_dimensions(mask_images, user_args)
            pipeline_args['mask_image'] = mask_images

    def _set_pipeline_t2iadapter_defaults(self, user_args: DiffusionArguments, pipeline_args: dict[str, typing.Any]):
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

        # Resize control images to user-specified dimensions first thing
        self._validate_images_all_same_size("T2I adapter control images", adapter_control_images)
        adapter_control_images = self._resize_images_to_user_dimensions(adapter_control_images, user_args)

        if not _image.is_aligned(first_control_image_size, 16):
            # noinspection PyTypeChecker
            new_size: tuple[int, int] = _image.align_by(first_control_image_size, 16)
            _messages.warning(
                f'T2I Adapter control image(s) of size {first_control_image_size} being forcefully '
                f'aligned by 16 to {new_size} to prevent errors.'
            )

            for idx, img in enumerate(adapter_control_images):
                adapter_control_images[idx] = _image.resize_image(img, new_size)

        if _enums.model_type_is_sdxl(self.model_type) and user_args.sdxl_t2i_adapter_factor is not None:
            pipeline_args['adapter_conditioning_factor'] = user_args.sdxl_t2i_adapter_factor

        self._set_pipe_dimensions(
            None, None,
            adapter_control_images[0].width, adapter_control_images[0].height,
            pipeline_args
        )

        if self._pipeline_type == _enums.PipelineType.TXT2IMG:
            pipeline_args['image'] = adapter_control_images
        else:
            raise _pipelines.UnsupportedPipelineConfigError(
                'T2IAdapter models only work in txt2img mode.'
            )

    def _get_pipeline_img2img_inputs(self, user_args: DiffusionArguments):
        # Separate images and tensors but skip validation initially
        images, img2img_latents = self._separate_images_and_tensors(user_args.images)

        # Don't allow mixing images and tensors in the same input
        if images and img2img_latents:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'Cannot mix PIL Images and latents tensors in img2img inputs. '
                f'All inputs must be either images or latents tensors, not both.'
            )

        # Process input tensors
        if img2img_latents:
            img2img_latents = self._process_input_latents(
                "img2img", img2img_latents, user_args.img2img_latents_processors
            )

        # Resize input images to user-specified dimensions first thing
        if images:
            if not _enums.model_type_is_s_cascade(self._model_type):
                self._validate_images_all_same_size('img2img images', images)
                images = self._resize_images_to_user_dimensions(images, user_args)

        if self._model_type != _enums.ModelType.UPSCALER_X2 and \
                hasattr(self._pipeline, 'vae') and self._pipeline.vae is not None:
            # we need to decode the latents into an image using the VAE for
            # the best img2img result, passing already denoised latents
            # in does not make sense to the receiving UNet/Transformer
            # except in the case of the X2 latent upscaler, which can
            # work with the already denoised latents
            if img2img_latents and not (
                    _supports_native_denoising_start(self._pipeline.__class__)
                    and user_args.denoising_start is not None
                    and user_args.denoising_start > 0.0
            ):
                if _enums.model_type_is_flux(self._model_type):
                    img2img_latents = self._repack_flux_latents(self._stack_latents(img2img_latents))

                images = self.decode_latents(img2img_latents)
                # Process decoded images if processors are configured (handles pre-resize, resize, post-resize)
                images = self._process_decoded_latents_images(
                    images, user_args.decoded_latents_image_processor_uris, user_args
                )
                img2img_latents = None

        # Use the final result (tensors or images)
        if img2img_latents:
            inputs = img2img_latents
        else:
            inputs = images

        return inputs

    # noinspection PyMethodMayBeStatic
    def _aligned_8_user_dimensions(self, user_args: DiffusionArguments):
        if user_args.height is not None:
            if user_args.height % 8 != 0:
                user_height = user_args.height - (user_args.height % 8)
            else:
                user_height = user_args.height
        else:
            user_height = None

        if user_args.width is not None:
            if user_args.width % 8 != 0:
                user_width = user_args.width - (user_args.width % 8)
            else:
                user_width = user_args.width
        else:
            user_width = None

        return user_width, user_height

    def _set_pipe_dimensions(
            self,
            user_width: int | None,
            user_height: int | None,
            inference_width: int | None,
            inference_height: int | None,
            pipeline_args: dict | None = None
    ):
        width = user_width if user_width is not None else inference_width
        height = user_height if user_height is not None else inference_height

        self._inference_width = width
        self._inference_height = height

        if pipeline_args is not None:
            pipeline_args['width'] = width
            pipeline_args['height'] = height

    # noinspection PyUnresolvedReferences,PyTypeChecker
    def _set_pipeline_img2img_defaults(self, user_args: DiffusionArguments, pipeline_args: dict[str, typing.Any]):
        user_width, user_height = self._aligned_8_user_dimensions(user_args)
        image_arg_inputs = self._get_pipeline_img2img_inputs(user_args)
        non_latent_input = not _torchutil.is_tensor(image_arg_inputs[0])

        # Calculate dimensions once for reuse
        if not non_latent_input:
            inference_width, inference_height = self.get_decoded_latents_size(image_arg_inputs[0])
        else:
            inference_width, inference_height = image_arg_inputs[0].width, image_arg_inputs[0].height

        # Handle special model type configurations
        floyd_og_image_needed = (self._pipeline_type == _enums.PipelineType.INPAINT and
                                 _enums.model_type_is_floyd_ifs(self._model_type)
                                 ) or (self._model_type == _enums.ModelType.IFS_IMG2IMG)

        if floyd_og_image_needed:
            if user_args.floyd_image is None:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'must specify "floyd_image" to disambiguate this operation, '
                    '"floyd_image" being the output of a previous floyd stage.')
            pipeline_args['original_image'] = image_arg_inputs
            pipeline_args['image'] = user_args.floyd_image
            self._set_pipe_dimensions(
                user_width, user_height,
                inference_width, inference_height
            )

        elif self._model_type == _enums.ModelType.S_CASCADE:
            pipeline_args['images'] = image_arg_inputs
            # Stable cascade output dimension will not be based on the image input for img2img
            self._set_pipe_dimensions(
                user_width, user_height,
                _constants.DEFAULT_S_CASCADE_OUTPUT_WIDTH, _constants.DEFAULT_S_CASCADE_OUTPUT_HEIGHT
            )
        else:
            pipeline_args['image'] = image_arg_inputs
            # Set dimensions for general img2img case - will be used unless overridden later
            self._set_pipe_dimensions(
                user_width, user_height,
                inference_width, inference_height
            )

        # Handle model-specific settings
        def check_no_image_seed_strength():
            if user_args.image_seed_strength is not None:
                _messages.warning(
                    f'image_seed_strength is not supported by model_type '
                    f'"{_enums.get_model_type_string(self._model_type)}" in '
                    f'mode "{self._pipeline_type.name}" and is being ignored.'
                )

        def is_sd3_or_flux():
            return _enums.model_type_is_sd3(self._model_type) or _enums.model_type_is_flux(self._model_type)

        if _enums.model_type_is_upscaler(self._model_type):
            if self._model_type == _enums.ModelType.UPSCALER_X4:
                pipeline_args['noise_level'] = int(
                    _types.default(user_args.upscaler_noise_level, _constants.DEFAULT_X4_UPSCALER_NOISE_LEVEL)
                )
            check_no_image_seed_strength()
        elif self._model_type in [_enums.ModelType.FLUX_FILL, _enums.ModelType.FLUX_KONTEXT]:
            check_no_image_seed_strength()
        elif self._model_type == _enums.ModelType.IFS:
            if self._pipeline_type != _enums.PipelineType.INPAINT:
                pipeline_args['noise_level'] = int(
                    _types.default(user_args.upscaler_noise_level, _constants.DEFAULT_FLOYD_SUPERRESOLUTION_NOISE_LEVEL)
                )
                check_no_image_seed_strength()
            else:
                pipeline_args['noise_level'] = int(
                    _types.default(user_args.upscaler_noise_level,
                                   _constants.DEFAULT_FLOYD_SUPERRESOLUTION_INPAINT_NOISE_LEVEL)
                )
                self._set_pipeline_strength(user_args, pipeline_args)
        elif self._model_type == _enums.ModelType.IFS_IMG2IMG:
            pipeline_args['noise_level'] = int(
                _types.default(user_args.upscaler_noise_level,
                               _constants.DEFAULT_FLOYD_SUPERRESOLUTION_IMG2IMG_NOISE_LEVEL)
            )
            self._set_pipeline_strength(user_args, pipeline_args)
        elif not _enums.model_type_is_pix2pix(self._model_type) and self._model_type != _enums.ModelType.S_CASCADE:
            self._set_pipeline_strength(user_args, pipeline_args)
        else:
            check_no_image_seed_strength()

        # Handle mask images
        mask_images = user_args.mask_images
        if mask_images is not None:
            self._validate_images_all_same_size('inpaint mask images', mask_images)
            mask_images = self._resize_images_to_user_dimensions(mask_images, user_args)

            images_size = (inference_width, inference_height)
            if mask_images[0].size != images_size:
                raise _pipelines.UnsupportedPipelineConfigError(
                    f'Image seed img2img images and inpaint masks must '
                    f'have the same dimension, got: {images_size}, '
                    f'and {mask_images[0].size} respectively.'
                )

            pipeline_args['mask_image'] = mask_images

            if not (_enums.model_type_is_floyd(self._model_type) or is_sd3_or_flux()):
                # Override dimensions for masked models
                self._set_pipe_dimensions(
                    user_width, user_height,
                    inference_width, inference_height,
                    pipeline_args
                )

        # Handle adetailer (auto-generated masks)
        if self._parsed_adetailer_detector_uris:
            if not is_sd3_or_flux():
                # Override dimensions for adetailer
                self._set_pipe_dimensions(
                    user_width, user_height,
                    inference_width, inference_height,
                    pipeline_args
                )

        # Handle specific model types that need special dimension handling
        if self._model_type == _enums.ModelType.SDXL_PIX2PIX:
            self._set_pipe_dimensions(
                user_width, user_height,
                inference_width, inference_height,
                pipeline_args
            )
        elif self._model_type == _enums.ModelType.UPSCALER_X2:
            image_arg_inputs = list(image_arg_inputs)
            pipeline_args['image'] = image_arg_inputs

            if non_latent_input:
                for idx, image in enumerate(image_arg_inputs):
                    if not _image.is_aligned(image.size, 64):
                        size = _image.align_by(image.size, 64)
                        _messages.warning(
                            f'Input image size {image.size} is not aligned by 64. '
                            f'Output dimensions will be forcefully aligned to 64: {size}.'
                        )
                        image_arg_inputs[idx] = _image.resize_image(image, size)

            self._set_pipe_dimensions(
                None, None,
                image_arg_inputs[0].width, image_arg_inputs[0].height
            )

        elif self._model_type == _enums.ModelType.S_CASCADE:
            # Validate output dimensions for stable cascade
            if user_width and user_width > 0 and not (user_width % 128) == 0:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Stable Cascade requires an output dimension that is aligned by 128.')
            if user_height and user_height > 0 and not (user_height % 128) == 0:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Stable Cascade requires an output dimension that is aligned by 128.')

            # Override with cascade defaults and set both pipeline and inference dimensions
            cascade_width = _types.default(user_width, _constants.DEFAULT_S_CASCADE_OUTPUT_WIDTH)
            cascade_height = _types.default(user_height, _constants.DEFAULT_S_CASCADE_OUTPUT_HEIGHT)

            self._set_pipe_dimensions(
                None, None,
                cascade_width, cascade_height,
                pipeline_args
            )

        elif is_sd3_or_flux():
            if _enums.model_type_is_sd3(self._model_type):
                image_arg_inputs = list(image_arg_inputs)
                pipeline_args['image'] = image_arg_inputs

                if non_latent_input:
                    for idx, image in enumerate(image_arg_inputs):
                        if not _image.is_aligned(image.size, 16):
                            size = _image.align_by(image.size, 16)
                            _messages.warning(
                                f'Input image size {image.size} is not aligned by 16. '
                                f'Dimensions will be forcefully aligned to 16: {size}.'
                            )
                            image_arg_inputs[idx] = _image.resize_image(image, size)

                    inference_width = image_arg_inputs[0].width
                    inference_height = image_arg_inputs[0].height

                if mask_images:
                    mask_images = list(mask_images)
                    pipeline_args['mask_image'] = mask_images

                    for idx, image in enumerate(mask_images):
                        if not _image.is_aligned(image.size, 16):
                            size = _image.align_by(image.size, 16)
                            _messages.warning(
                                f'Input mask image size {image.size} is not aligned by 16. '
                                f'Dimensions will be forcefully aligned to 16: {size}.'
                            )
                            mask_images[idx] = _image.resize_image(image, size)

                    inference_width = mask_images[0].width
                    inference_height = mask_images[0].height

            self._set_pipe_dimensions(
                None, None,
                inference_width, inference_height,
                pipeline_args
            )


    def _set_pipeline_txt2img_defaults(self, user_args: DiffusionArguments, pipeline_args: dict[str, typing.Any]):

        width, height = self._aligned_8_user_dimensions(user_args)

        if width != user_args.width:
            _messages.warning('Forcing alignment of txt2img generation argument "width" to 8.')

        if height != user_args.height:
            _messages.warning('Forcing alignment of txt2img generation argument "height" to 8.')

        if _enums.model_type_is_sdxl(self._model_type):
            self._inference_height = _types.default(height, _constants.DEFAULT_SDXL_OUTPUT_HEIGHT)
            self._inference_width = _types.default(width, _constants.DEFAULT_SDXL_OUTPUT_WIDTH)
        elif _enums.model_type_is_kolors(self._model_type):
            self._inference_height = _types.default(height, _constants.DEFAULT_KOLORS_OUTPUT_HEIGHT)
            self._inference_width = _types.default(width, _constants.DEFAULT_KOLORS_OUTPUT_WIDTH)
        elif _enums.model_type_is_floyd_if(self._model_type):
            self._inference_height = _types.default(height, _constants.DEFAULT_FLOYD_IF_OUTPUT_HEIGHT)
            self._inference_width = _types.default(width, _constants.DEFAULT_FLOYD_IF_OUTPUT_WIDTH)
        elif self._model_type == _enums.ModelType.S_CASCADE:
            self._inference_height = _types.default(height, _constants.DEFAULT_S_CASCADE_OUTPUT_HEIGHT)
            self._inference_width = _types.default(width, _constants.DEFAULT_S_CASCADE_OUTPUT_WIDTH)

            if not _image.is_aligned((self._inference_height, self._inference_width), 128):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Stable Cascade requires an output dimension that is aligned by 128.')
        elif self._model_type == _enums.ModelType.SD3:
            self._inference_height = _types.default(height, _constants.DEFAULT_SD3_OUTPUT_HEIGHT)
            self._inference_width = _types.default(width, _constants.DEFAULT_SD3_OUTPUT_WIDTH)

            if not _image.is_aligned((self._inference_height, self._inference_width), 16):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Stable Diffusion 3 requires an output dimension that is aligned by 16.')
        elif self._model_type == _enums.ModelType.FLUX:
            self._inference_height = _types.default(height, _constants.DEFAULT_FLUX_OUTPUT_HEIGHT)
            self._inference_width = _types.default(width, _constants.DEFAULT_FLUX_OUTPUT_WIDTH)
        else:
            self._inference_height = _types.default(height, _constants.DEFAULT_OUTPUT_HEIGHT)
            self._inference_width = _types.default(width, _constants.DEFAULT_OUTPUT_WIDTH)

        self._set_pipe_dimensions(
            None, None,
            self._inference_width, self._inference_height,
            pipeline_args
        )

    def _prepare_inpaint_crop(self, user_args: DiffusionArguments):
        """
        Handle inpaint crop preparation including validation and tensor decoding.
        
        :param user_args: user arguments to validate and potentially modify
        """
        # Automatically enable inpaint crop if padding, feathering, or masking is specified
        if not user_args.inpaint_crop and (
                user_args.inpaint_crop_padding is not None or user_args.inpaint_crop_feather is not None or user_args.inpaint_crop_masked):
            user_args.inpaint_crop = True

        if not user_args.inpaint_crop:
            return

        if not user_args.aspect_correct:
            raise _pipelines.UnsupportedPipelineConfigError(
                'aspect_correct=False is not compatible with inpaint_crop=True.'
            )

        # Validate that inpaint crop has required inputs
        if user_args.images is None or len(user_args.images) == 0:
            raise _pipelines.UnsupportedPipelineConfigError(
                "inpaint_crop requires images to be provided."
            )

        if user_args.mask_images is None or len(user_args.mask_images) == 0:
            raise _pipelines.UnsupportedPipelineConfigError(
                "inpaint_crop requires mask_images to be provided."
            )

        # Check that we're not outputting latents  
        if user_args.output_latents:
            raise _pipelines.UnsupportedPipelineConfigError(
                "inpaint_crop is not supported when outputting latents, only images are supported."
            )

        # Disallow batching multiple different images with inpaint_crop
        # (batch_size > 1 is OK for generating variations of a single crop)
        if len(user_args.images) > 1 or len(user_args.mask_images) > 1:
            raise _pipelines.UnsupportedPipelineConfigError(
                "inpaint_crop cannot be used with multiple input images. "
                "Each image/mask pair should be processed individually for optimal cropping. "
                "Consider processing one image at a time or disable inpaint_crop for batch processing. "
                "Note: batch_size > 1 is supported for generating multiple variations of a single crop."
            )

        decoded_latents = False
        # If images are tensors (latents), decode them with the VAE first
        if not isinstance(user_args.images[0], PIL.Image.Image):
            _messages.debug_log('Inpaint crop: decoding tensor inputs with VAE...')

            # Check that we have a VAE for decoding
            if not hasattr(self._pipeline, 'vae') or self._pipeline.vae is None:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Cannot decode tensor inputs for inpaint_crop as the pipeline does not have a VAE. '
                    'Use images instead.'
                )

            # Handle Flux-specific repacking if needed
            latents = user_args.images
            if _enums.model_type_is_flux(self._model_type):
                latents = self._repack_flux_latents(self._stack_latents(latents))

            # Decode the latents to PIL Images
            user_args.images = self.decode_latents(latents)

            decoded_latents = True

        # Apply the actual inpaint crop if we get here
        _messages.debug_log('Applying inpaint crop...')

        # Store references to original images before cropping
        original_images = list(user_args.images)
        original_masks = list(user_args.mask_images)
        original_control_images = list(user_args.control_images) if user_args.control_images else None

        # Get padding from user args or use default, normalize to (left, top, right, bottom)
        if user_args.inpaint_crop_padding is not None:
            raw_padding = user_args.inpaint_crop_padding
            if isinstance(raw_padding, int):
                # Same padding on all sides
                padding = (raw_padding, raw_padding, raw_padding, raw_padding)
            elif isinstance(raw_padding, tuple) and len(raw_padding) == 2:
                # (horizontal, vertical) padding
                padding = (raw_padding[0], raw_padding[1], raw_padding[0], raw_padding[1])
            elif isinstance(raw_padding, tuple) and len(raw_padding) == 4:
                # (left, top, right, bottom) padding
                padding = raw_padding
            else:
                raise _pipelines.UnsupportedPipelineConfigError(
                    f"Invalid inpaint_crop_padding format: {raw_padding}. "
                    f"Expected int, tuple[int, int], or tuple[int, int, int, int]")
        else:
            # Default padding on all sides
            padding = _constants.DEFAULT_INPAINT_CROP_PADDING

        # Apply crop to get the cropped area
        (cropped_images,
         cropped_masks,
         cropped_control_images,
         crop_bounds) = self._apply_inpaint_crop(
            images=original_images,
            masks=original_masks,
            control_images=original_control_images,
            padding=padding,
            decoded_latents=decoded_latents,
            user_args=user_args
        )

        # Store crop info for later pasting
        crop_info = _InpaintCropInfo(
            original_images=original_images,
            original_masks=original_masks,
            crop_bounds=crop_bounds,
            use_masked=user_args.inpaint_crop_masked,
            feather=user_args.inpaint_crop_feather
        )
        self._inpaint_crop_info = crop_info

        # Replace user_args with cropped images so all downstream logic handles them transparently
        user_args.images = cropped_images
        user_args.mask_images = cropped_masks
        if cropped_control_images is not None:
            user_args.control_images = cropped_control_images

        _messages.debug_log(
            f'Inpaint crop applied: {original_images[0].size if original_images else None} -> crop bounds {crop_bounds}')

    def _get_pipeline_defaults(self, user_args: DiffusionArguments):
        """
        Get a default arrangement of arguments to be passed to a huggingface
        diffusers pipeline call that are somewhat universal.

        :param user_args: user arguments to the pipeline wrapper
        :return: kwargs dictionary
        """

        self._inference_width = None
        self._inference_height = None

        # Apply inpaint crop if enabled and we have the necessary inputs
        # This must happen first so all downstream methods work with cropped images
        self._prepare_inpaint_crop(user_args)

        pipeline_args: dict[str, typing.Any] = dict()

        pipeline_args['guidance_scale'] = float(
            _types.default(user_args.guidance_scale, _constants.DEFAULT_GUIDANCE_SCALE))
        pipeline_args['num_inference_steps'] = int(
            _types.default(user_args.inference_steps, _constants.DEFAULT_INFERENCE_STEPS))

        # Create generator once and reuse it throughout
        pipeline_args['generator'] = torch.Generator(device=self._device).manual_seed(
            _types.default(user_args.seed, _constants.DEFAULT_SEED))

        if self._controlnet_uris:
            self._set_pipeline_controlnet_defaults(user_args, pipeline_args)
        elif self._t2i_adapter_uris:
            self._set_pipeline_t2iadapter_defaults(user_args, pipeline_args)
        elif user_args.images is not None:
            self._set_pipeline_img2img_defaults(user_args, pipeline_args)
        else:
            self._set_pipeline_txt2img_defaults(user_args, pipeline_args)

        if user_args.latents:
            # this uses 'width' and 'height' from pipeline_args as input
            pipeline_args['latents'] = self._process_raw_input_latents(user_args, pipeline_args)

        if user_args.ip_adapter_images:
            user_args.ip_adapter_images = self._process_ip_adapter_images(user_args.ip_adapter_images)

        if user_args.floyd_image is not None:
            user_args.floyd_image = self._process_floyd_image(user_args.floyd_image)

        return pipeline_args

    def _process_raw_input_latents(self, user_args: DiffusionArguments,
                                   pipeline_args: dict[str, typing.Any]) -> torch.Tensor:
        """
        Process and validate incoming raw / noisy latents from ``latents``

        :param user_args: Diffusion arguments
        :return: Batched latents tensor
        """
        latents = self._process_input_latents("raw", user_args.latents, user_args.latents_processors)
        latents = self._stack_latents(latents)
        decoded_latents_size = self.get_decoded_latents_size(latents)

        expected_width = pipeline_args.get('width', None)
        expected_height = pipeline_args.get('height', None)

        if user_args.images:
            if not _torchutil.is_tensor(user_args.images[0]):
                expected_width, expected_height = _image.resize_image_calc(
                    user_args.images[0].size,
                    self._calc_image_target_size(user_args.images[0], user_args),
                    aspect_correct=user_args.aspect_correct,
                    align=8
                )
            else:
                expected_width, expected_height = self.get_decoded_latents_size(user_args.images[0])

        output_size_expected = (expected_width, expected_height)
        if output_size_expected != decoded_latents_size:
            raise _pipelines.UnsupportedPipelineConfigError(
                f"Render width / height not compatible with "
                f"given raw latents, output size: {_textprocessing.format_size(output_size_expected)}, "
                f"latents decoded size: {_textprocessing.format_size(decoded_latents_size)}. This can "
                f"be caused by an explicitly set width / height that is incorrect for the incoming raw "
                f"latents, or a missmatch in the size of incoming img2img images / latents with the "
                f"raw latents."
            )

        # Store dimensions for optimizations
        self._inference_width = expected_width
        self._inference_height = expected_height

        if latents.dtype != self._pipeline.dtype:
            _messages.debug_log(
                f'Casting incoming raw latents from: {latents.dtype}, to: {self._pipeline.dtype}'
            )
            latents = latents.to(self._device, dtype=self._pipeline.dtype)
        else:
            latents = latents.to(self._device)

        if _enums.model_type_is_flux(self._model_type):
            latents = self._repack_flux_latents(latents)
        return latents

    @staticmethod
    def _stack_latents(latents):
        if latents[0].ndim == 4:
            # List of [B, C, H, W] tensors - concatenate along batch dimension
            latents = torch.cat(list(latents), dim=0)  # [total_B, C, H, W]
        elif latents[0].ndim == 3:
            # List of [C, H, W] tensors - stack to create batch dimension
            latents = torch.stack(list(latents), dim=0)  # [num_tensors, C, H, W]
        else:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'Unsupported latents shape: {latents[0].shape}')
        return latents

    @staticmethod
    def _sd3_force_control_to_a16(args, control_images, user_args):
        processed_control_images = list(control_images)
        for idx, img in enumerate(processed_control_images):

            if not _image.is_aligned(img.size, 16):
                # noinspection PyTypeChecker
                size: tuple[int, int] = _image.align_by(img.size, 16)

                if user_args.width:
                    if not (user_args.width % 16) == 0:
                        raise _pipelines.UnsupportedPipelineConfigError(
                            'Stable Diffusion 3 requires an output dimension aligned by 16.')

                if user_args.height:
                    if not (user_args.height % 16) == 0:
                        raise _pipelines.UnsupportedPipelineConfigError(
                            'Stable Diffusion 3 requires an output dimension aligned by 16.')

                args['width'] = size[0]
                args['height'] = size[1]

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
            'negative_prompt_3': 'third_prompt',
            'clip_skip': 'clip_skip'
        }

        prompt_weighter_extra_args = prompt_weighter.get_extra_supported_args()

        for arg_name in prompt_weighter_extra_args:

            if arg_name not in arg_map:
                raise RuntimeError(
                    f'Prompt weighter plugin: {prompt_weighter.__class__.__name__}, '
                    f'returned invalid "get_extra_supported_args()" value: {arg_name}. '
                    f'This is a bug, acceptable values are: {", ".join(arg_map.keys())}')

            source = arg_map[arg_name]
            if 'negative' in arg_name:
                user_value = getattr(diffusion_args, source, None)
                if user_value:
                    pipeline_args[arg_name] = user_value.negative
                    poppable_args.append(arg_name)
            elif 'prompt' in arg_name:
                user_value = getattr(diffusion_args, source, None)
                if user_value:
                    pipeline_args[arg_name] = user_value.positive
                    poppable_args.append(arg_name)
            else:
                user_value = getattr(diffusion_args, source, None)
                if user_value:
                    pipeline_args[arg_name] = user_value
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

        def _hw_swizzle(x):
            return x[1], x[0]

        self._set_non_universal_pipeline_arg(pipeline, pipeline_args, user_args,
                                             'aesthetic_score', f'sdxl_{user_prefix}aesthetic_score',
                                             f'--sdxl-{option_prefix}aesthetic-scores')
        self._set_non_universal_pipeline_arg(pipeline, pipeline_args, user_args,
                                             'original_size', f'sdxl_{user_prefix}original_size',
                                             f'--sdxl-{option_prefix}original-sizes', _hw_swizzle)
        self._set_non_universal_pipeline_arg(pipeline, pipeline_args, user_args,
                                             'target_size', f'sdxl_{user_prefix}target_size',
                                             f'--sdxl-{option_prefix}target-sizes', _hw_swizzle)

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
                                             f'--sdxl-{option_prefix}negative-original-sizes', _hw_swizzle)

        self._set_non_universal_pipeline_arg(pipeline, pipeline_args, user_args,
                                             'negative_target_size',
                                             f'sdxl_{user_prefix}negative_target_size',
                                             f'--sdxl-{option_prefix}negative-target-sizes', _hw_swizzle)

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

    def _unpack_flux_latents(self,
                             latents: torch.Tensor,
                             height: int | None = None,
                             width: int | None = None) -> torch.Tensor:
        """
        Unpack Flux latents from internal packed format [B, L, C] to external unpacked format [B, C, H, W].
        
        This method converts from the packed sequence format that Flux pipelines use internally
        to the standard spatial format used as the external interface.
        
        :param latents: Input latents in packed shape [B, L, C] or [L, C]
        :param height: Optional target height, will use default if not specified
        :param width: Optional target width, will use default if not specified
        :return: Unpacked latents in shape [B, C, H, W]
        """

        # Add batch dimension if needed
        if len(latents.shape) == 2:  # If [L, C] add batch dimension
            latents = latents.unsqueeze(0)

        # Calculate dimensions
        height = height or self._pipeline.default_sample_size * self._pipeline.vae_scale_factor
        width = width or self._pipeline.default_sample_size * self._pipeline.vae_scale_factor

        # VAE applies 8x compression on images, but we must also account for packing which requires
        # latent height and width to be divisible by 2
        height = 2 * (int(height) // (self._pipeline.vae_scale_factor * 2))
        width = 2 * (int(width) // (self._pipeline.vae_scale_factor * 2))

        # Unpack from [B, L, C] to [B, C, H, W]
        batch_size, num_patches, channels = latents.shape
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    @staticmethod
    def _repack_flux_latents(latents: torch.Tensor) -> torch.Tensor:
        """
        Repack Flux latents from external unpacked format [B, C, H, W] to internal packed format [B, L, C].
        
        This method converts from the standard spatial format used as the external interface
        to the packed sequence format that Flux pipelines expect internally.
        This is the inverse operation of _unpack_flux_latents.
        
        :param latents: Input latents in unpacked shape [B, C, H, W]
        :return: Repacked latents in shape [B, L, C]
        """
        batch_size, channels, height, width = latents.shape

        # Repack from [B, C, H, W] to [B, L, C]
        # This reverses the operations in _unpack_flux_latents
        latents = latents.reshape(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)

        return latents

    def _call_torch_flux(self, pipeline_args, user_args: DiffusionArguments):
        self._check_for_invalid_model_specific_opts(user_args)

        if user_args.clip_skip is not None and user_args.clip_skip > 0:
            raise _pipelines.UnsupportedPipelineConfigError('Flux does not support clip skip.')

        prompt: _prompt.Prompt = _types.default(user_args.prompt, _prompt.Prompt())
        prompt_2: _prompt.Prompt = _types.default(user_args.second_prompt, _prompt.Prompt())

        pipeline_args['prompt'] = prompt.positive if prompt.positive else ''
        pipeline_args['prompt_2'] = prompt_2.positive if prompt_2.positive else None


        if inspect.signature(self._pipeline.__call__).parameters.get('negative_prompt') is None:
            if prompt.negative:
                _messages.warning(
                    'Flux is ignoring the provided negative prompt as it '
                    'does not support negative prompting in the current configuration.'
                )

            if prompt_2.negative:
                _messages.warning(
                    'Flux is ignoring the provided second negative prompt as it '
                    'does not support negative prompting in the current configuration.'
                )
        else:
            pipeline_args['negative_prompt'] = prompt.negative if prompt.negative else None
            pipeline_args['negative_prompt_2'] = prompt_2.negative if prompt_2.negative else None

        if user_args.max_sequence_length is not None:
            pipeline_args['max_sequence_length'] = user_args.max_sequence_length

        batch_size = _types.default(user_args.batch_size, 1)

        # Adjust batch size to match raw latents if provided
        if 'latents' in pipeline_args:
            latents_batch_size = pipeline_args['latents'].shape[0]
            if latents_batch_size != batch_size:
                batch_size = latents_batch_size
                if user_args.batch_size is not None:
                    # only warn if the user specified a value
                    _messages.warning(
                        f'Setting --batch-size to {batch_size} because '
                        f'raw latents batch size did not match the specified batch size.'
                    )

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

        self._set_non_universal_pipeline_arg(
            self._pipeline,
            pipeline_args, user_args,
            'sigmas', 'sigmas',
            '--sigmas',
            transform=functools.partial(
                self._sigmas_eval,
                'primary',
                self._pipeline,
                _types.default(
                    user_args.inference_steps,
                    _constants.DEFAULT_INFERENCE_STEPS)
            ))

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
        ), _sada_context(
            self._pipeline,
            width=self._inference_width,
            height=self._inference_height,
            enabled=user_args.sada,
            **self._get_sada_args(user_args)
        ), _denoise_range(
            self._pipeline,
            user_args.denoising_start,
            user_args.denoising_end
        ):
            output_type = 'latent' if user_args.output_latents else 'pil'

            if self._parsed_adetailer_detector_uris:
                return self._call_asdff(
                    user_args=user_args,
                    pipeline_args=pipeline_args,
                    batch_size=batch_size,
                    prompt_weighter=prompt_weighter
                )
            else:
                pipeline_output = _pipelines.call_pipeline(
                    pipeline=self._pipeline,
                    prompt_weighter=prompt_weighter,
                    device=self._device,
                    output_type=output_type,
                    **pipeline_args
                )
                return self._create_pipeline_result(pipeline_output, output_type, user_args, pipeline_args)

    def _call_asdff(self,
                    user_args: DiffusionArguments,
                    prompt_weighter: _promptweighters.PromptWeighter,
                    pipeline_args: dict[str, typing.Any],
                    batch_size: int):
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

            model_masks = _types.default(user_args.adetailer_model_masks, _constants.DEFAULT_ADETAILER_MODEL_MASKS)
            if detector_uri.model_masks is not None:
                model_masks = detector_uri.model_masks
                _messages.log(f'Overriding global adetailer model-masks '
                              f'value with adetailer detector URI value: {model_masks}')

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

            class_filter = _types.default(user_args.adetailer_class_filter, None)
            if detector_uri.class_filter is not None:
                class_filter = detector_uri.class_filter
                _messages.log(f'Overriding global adetailer class-filter '
                              f'value with adetailer detector URI value: {class_filter}')

            if detector_uri.prompt is not None:
                pipeline_args['prompt'] = detector_uri.prompt
                _messages.log(f'Overriding global positive prompt '
                              f'value with adetailer detector URI value: "{detector_uri.prompt}"')

            if detector_uri.negative_prompt is not None:
                pipeline_args['negative_prompt'] = detector_uri.negative_prompt
                _messages.log(f'Overriding global negative prompt '
                              f'value with adetailer detector URI value: "{detector_uri.negative_prompt}"')

            processing_size = user_args.adetailer_size
            
            if detector_uri.size is not None:
                processing_size = detector_uri.size
                _messages.log(f'Overriding global adetailer size '
                              f'value with adetailer detector URI value: {detector_uri.size}')

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
                class_filter=class_filter,
                mask_blur=mask_blur,
                mask_shape=mask_shape,
                detector_padding=detector_padding,
                mask_padding=mask_padding,
                mask_dilation=mask_dilation,
                model_masks=model_masks,
                processing_size=processing_size
            )

        return self._create_pipeline_result(asdff_output, user_args=user_args, pipeline_kwargs=pipeline_args)

    def _call_torch_s_cascade(self, pipeline_args, user_args: DiffusionArguments):
        self._check_for_invalid_model_specific_opts(user_args)

        if user_args.clip_skip is not None and user_args.clip_skip > 0:
            prompt_weighter_name = getattr(user_args, 'prompt_weighter', None)
            if not prompt_weighter_name:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Stable Cascade only supports clip skip through '
                    'prompt weighters (such as compel or sd-embed).')

        if user_args.sigmas is not None:
            raise _pipelines.UnsupportedPipelineConfigError('Stable Cascade does not support sigmas.')

        prompt: _prompt.Prompt = _types.default(user_args.prompt, _prompt.Prompt())
        pipeline_args['prompt'] = prompt.positive if prompt.positive else ''
        pipeline_args['negative_prompt'] = prompt.negative

        pipeline_args['num_images_per_prompt'] = _types.default(user_args.batch_size, 1)

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
        pipeline_args.pop('height', None)
        pipeline_args.pop('width', None)
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

        output_type = 'latent' if user_args.output_latents else 'pil'

        pipeline_output = _pipelines.call_pipeline(
            image_embeddings=image_embeddings,
            pipeline=self._s_cascade_decoder_pipeline,
            device=self._device,
            prompt_weighter=self._get_second_model_prompt_weighter(user_args),
            output_type=output_type,
            **pipeline_args)
        return self._create_pipeline_result(pipeline_output, output_type, user_args, pipeline_args)

    @staticmethod
    def _flux_sigmas_calculate_shift(
            image_seq_len,
            base_seq_len: int = 256,
            max_seq_len: int = 4096,
            base_shift: float = 0.5,
            max_shift: float = 1.15,
    ):
        # mu calculation for use_dynamic_shifting=True with Flux
        # This code comes from the Flux pipelines

        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    @staticmethod
    def _sigmas_eval(model_title: str, pipeline, steps: int, val: str | list):
        accept_sigmas = "sigmas" in set(
            inspect.signature(pipeline.scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'The current {model_title} model scheduler "{pipeline.scheduler.__class__.__name__}" '
                f'does not support custom sigmas schedules. Please ensure that '
                f'you are using a supported scheduler.'
            )

        if not isinstance(val, str):
            return val

        try:
            if pipeline.__class__.__name__.startswith('Flux'):
                # This code comes from the Flux pipelines
                mu = DiffusionPipelineWrapper._flux_sigmas_calculate_shift(
                    pipeline.transformer.config.in_channels // 4,  # latents.shape[1]
                    pipeline.scheduler.config.get("base_image_seq_len", 256),
                    pipeline.scheduler.config.get("max_image_seq_len", 4096),
                    pipeline.scheduler.config.get("base_shift", 0.5),
                    pipeline.scheduler.config.get("max_shift", 1.15),
                )
                pipeline.scheduler.set_timesteps(steps, mu=mu)
            else:
                pipeline.scheduler.set_timesteps(steps)
        except Exception as e:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'Custom sigmas not supported for the {model_title} model and scheduler combination.'
            ) from e

        try:
            sigmas = pipeline.scheduler.sigmas
        except AttributeError as e:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'Selected {model_title} model scheduler '
                f'{pipeline.scheduler.__class__.__name__} did not produce sigmas.'
            ) from e

        interpreter = _eval.standard_interpreter(
            symtable=_eval.safe_builtins()
        )

        interpreter.symtable['np'] = numpy
        interpreter.symtable['sigmas'] = numpy.array(sigmas)

        try:
            val = interpreter.eval(val, show_errors=False, raise_errors=True)
        except Exception as e:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'Error interpreting sigmas expression "{val}":\n{e}'
            )

        if not isinstance(val, collections.abc.Iterable):
            raise _pipelines.UnsupportedPipelineConfigError(
                f'Sigmas expression for the {model_title} model '
                f'did not evaluate to an array, got: {val}'
            )
        else:
            return list(val)

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

        self._set_non_universal_pipeline_arg(
            self._pipeline,
            pipeline_args, user_args,
            'sigmas', 'sigmas',
            '--sigmas',
            transform=functools.partial(
                self._sigmas_eval,
                'primary',
                self._pipeline,
                _types.default(
                    user_args.inference_steps,
                    _constants.DEFAULT_INFERENCE_STEPS)
            ))

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
            if not self._parsed_ip_adapter_uris:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Cannot specify IP Adapter images without loading any IP Adapter models.'
                )

            if _enums.model_type_is_sd3(self.model_type):
                self._pipeline.set_ip_adapter_scale(self._parsed_ip_adapter_uris[0].scale)
            else:
                self._pipeline.set_ip_adapter_scale([u.scale for u in self._parsed_ip_adapter_uris])

        batch_size = _types.default(user_args.batch_size, 1)

        # Adjust batch size to match raw latents if provided
        if 'latents' in pipeline_args:
            latents_batch_size = pipeline_args['latents'].shape[0]
            if latents_batch_size != batch_size:
                batch_size = latents_batch_size
                if user_args.batch_size is not None:
                    # only warn if the user specified a value
                    _messages.warning(
                        f'Setting --batch-size to {batch_size} because '
                        f'raw latents batch size did not match the specified batch size.'
                    )

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

        if self._model_type != _enums.ModelType.UPSCALER_X2:
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

        generator = pipeline_args['generator']

        if isinstance(self._pipeline, diffusers.StableDiffusionInpaintPipelineLegacy):
            # Not necessary, will cause an error
            pipeline_args.pop('width', None)
            pipeline_args.pop('height', None)

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

            with _freeu(self._pipeline, user_args.freeu_params), \
                    _sd3_ras_context(self._pipeline, args=ras_args, enabled=user_args.ras), \
                    _deep_cache_context(self._pipeline,
                                        cache_interval=_types.default(
                                            user_args.deep_cache_interval, _constants.DEFAULT_DEEP_CACHE_INTERVAL),
                                        cache_branch_id=_types.default(
                                            user_args.deep_cache_branch_id, _constants.DEFAULT_DEEP_CACHE_BRANCH_ID),
                                        enabled=user_args.deep_cache), \
                    _hi_diffusion(self._pipeline,
                                  generator=generator,
                                  enabled=user_args.hi_diffusion,
                                  no_raunet=user_args.hi_diffusion_no_raunet,
                                  no_window_attn=user_args.hi_diffusion_no_win_attn), \
                    _sada_context(self._pipeline,
                                  width=self._inference_width,
                                  height=self._inference_height,
                                  enabled=user_args.sada,
                                  **self._get_sada_args(user_args)), \
                    _denoise_range(self._pipeline, user_args.denoising_start, user_args.denoising_end):

                if self._parsed_adetailer_detector_uris:
                    return generate_asdff()
                else:
                    output_type = 'latent' if user_args.output_latents else 'pil'

                    pipeline_output = _pipelines.call_pipeline(
                        pipeline=self._pipeline,
                        prompt_weighter=prompt_weighter,
                        device=self._device,
                        output_type=output_type,
                        **pipeline_args
                    )
                    return self._create_pipeline_result(
                        pipeline_output, output_type, user_args, pipeline_args
                    )

        if user_args.denoising_start is not None or user_args.denoising_end is not None:
            raise _pipelines.UnsupportedPipelineConfigError(
                'denoising_start and denoising_end are not supported when using an SDXL refiner.'
            )

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

        with _freeu(self._pipeline, user_args.freeu_params), \
                _deep_cache_context(self._pipeline,
                                    cache_interval=_types.default(
                                        user_args.deep_cache_interval, _constants.DEFAULT_DEEP_CACHE_INTERVAL),
                                    cache_branch_id=_types.default(
                                        user_args.deep_cache_branch_id, _constants.DEFAULT_DEEP_CACHE_BRANCH_ID),
                                    enabled=user_args.deep_cache), \
                _hi_diffusion(self._pipeline,
                              generator=generator,
                              enabled=user_args.hi_diffusion,
                              no_raunet=user_args.hi_diffusion_no_raunet,
                              no_window_attn=user_args.hi_diffusion_no_win_attn):

            if self._parsed_adetailer_detector_uris:
                image = generate_asdff().images
            else:
                image = _pipelines.call_pipeline(
                    pipeline=self._pipeline,
                    device=self._device,
                    prompt_weighter=prompt_weighter,
                    output_type=output_type,
                    **pipeline_args,
                    **i_end
                ).images

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
                pipeline_args.pop('width', None)
                pipeline_args.pop('height', None)

        # Or any of these
        self._pop_sdxl_conditioning_args(pipeline_args)
        pipeline_args.pop('ip_adapter_image', None)
        pipeline_args.pop('guidance_rescale', None)
        pipeline_args.pop('sigmas', None)
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

        self._set_non_universal_pipeline_arg(
            self._sdxl_refiner_pipeline,
            pipeline_args, user_args,
            'prompt', 'second_model_prompt',
            '--second-model-prompts',
            transform=lambda p: p.positive)

        self._set_non_universal_pipeline_arg(
            self._sdxl_refiner_pipeline,
            pipeline_args, user_args,
            'negative_prompt', 'second_model_prompt',
            '--second-model-prompts',
            transform=lambda p: p.negative)

        self._set_non_universal_pipeline_arg(
            self._sdxl_refiner_pipeline,
            pipeline_args, user_args,
            'prompt_2', 'second_model_second_prompt',
            '--second-model-second-prompts',
            transform=lambda p: p.positive)

        self._set_non_universal_pipeline_arg(
            self._sdxl_refiner_pipeline,
            pipeline_args, user_args,
            'negative_prompt_2', 'second_model_second_prompt',
            '--second-model-second-prompts',
            transform=lambda p: p.negative)

        self._get_sdxl_conditioning_args(
            self._sdxl_refiner_pipeline,
            pipeline_args, user_args,
            user_prefix='refiner')

        self._set_non_universal_pipeline_arg(
            self._sdxl_refiner_pipeline,
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

        self._set_non_universal_pipeline_arg(
            self._sdxl_refiner_pipeline,
            pipeline_args, user_args,
            'sigmas', 'sdxl_refiner_sigmas',
            '--sdxl-refiner-sigmas',
            transform=functools.partial(
                self._sigmas_eval,
                'refiner',
                self._sdxl_refiner_pipeline,
                pipeline_args.get('num_inference_steps', _constants.DEFAULT_INFERENCE_STEPS)
            )
        )

        with _freeu(self._sdxl_refiner_pipeline, user_args.sdxl_refiner_freeu_params), \
                _deep_cache_context(self._sdxl_refiner_pipeline,
                                    cache_interval=_types.default(
                                        user_args.deep_cache_interval,
                                        _constants.DEFAULT_SDXL_REFINER_DEEP_CACHE_INTERVAL),
                                    cache_branch_id=_types.default(
                                        user_args.deep_cache_branch_id,
                                        _constants.DEFAULT_SDXL_REFINER_DEEP_CACHE_BRANCH_ID),
                                    enabled=user_args.sdxl_refiner_deep_cache):

            output_type = 'latent' if user_args.output_latents else 'pil'

            pipeline_output = _pipelines.call_pipeline(
                pipeline=self._sdxl_refiner_pipeline,
                device=self._device,
                prompt_weighter=self._get_second_model_prompt_weighter(user_args),
                output_type=output_type,
                **pipeline_args,
                **i_start
            )

            return self._create_pipeline_result(
                pipeline_output, output_type, user_args, pipeline_args
            )

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
                width=self._inference_width,
                height=self._inference_height,
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

    def _get_sada_args(self, user_args: DiffusionArguments) -> dict:
        model_defaults = _util.get_sada_model_defaults(self.model_type)

        return {
            'max_downsample': _types.default(user_args.sada_max_downsample, model_defaults['max_downsample']),
            'sx': _types.default(user_args.sada_sx, model_defaults['sx']),
            'sy': _types.default(user_args.sada_sy, model_defaults['sy']),
            'acc_range': _types.default(user_args.sada_acc_range, model_defaults['acc_range']),
            'lagrange_term': _types.default(user_args.sada_lagrange_term, model_defaults['lagrange_term']),
            'lagrange_int': user_args.sada_lagrange_int or model_defaults['lagrange_int'],
            'lagrange_step': user_args.sada_lagrange_step or model_defaults['lagrange_step'],
            'max_fix': _types.default(user_args.sada_max_fix, model_defaults['max_fix']),
            'max_interval': _types.default(user_args.sada_max_interval, model_defaults['max_interval']),
        }

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

    def recall_secondary_pipeline(self) -> _pipelines.PipelineCreationResult:
        """
        Fetch the last used refiner / stable cascade decoder pipeline creation result,
        possibly the pipeline will be recreated if no longer in the in memory cache.
        If there is no refiner / decoder pipeline currently created, which will be the
        case if an image was never generated yet or a refiner / decoder model was not
        specified, :py:exc:`RuntimeError` will be raised.

        :raises RuntimeError:

        :return: :py:class:`dgenerate.pipelinewrapper.PipelineCreationResult`
        """

        if self._recall_secondary_pipeline is None:
            raise RuntimeError('Cannot recall refiner pipeline as one has not been created.')

        return self._recall_secondary_pipeline()

    def _lazy_init_pipeline(self, args: DiffusionArguments):

        pag = args.pag_scale is not None or args.pag_adaptive_scale is not None
        sdxl_refiner_pag = args.sdxl_refiner_pag_scale is not None or args.sdxl_refiner_pag_adaptive_scale is not None
        pipeline_type = args.determine_pipeline_type()

        if self._pipeline is not None:
            if self._pipeline_type == pipeline_type:
                return False

        if pag:
            if not (self.model_type == _enums.ModelType.SD or
                    self.model_type == _enums.ModelType.SDXL or
                    self.model_type == _enums.ModelType.SD3 or
                    self.model_type == _enums.ModelType.KOLORS):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Perturbed attention guidance (pag arguments) are only supported with '
                    '--model-type sd, sdxl, kolors (txt2img), and sd3.')

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
        self._recall_secondary_pipeline = None

        if self._parsed_adetailer_detector_uris:
            pipeline_type = _enums.PipelineType.INPAINT

        if self._model_type == _enums.ModelType.S_CASCADE:

            if self._s_cascade_decoder_uri is None:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Stable Cascade must be used with a decoder model.')

            self._recall_main_pipeline = _pipelines.PipelineFactory(
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
                quantizer_map=self._quantizer_map,
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

            self._recall_secondary_pipeline = _pipelines.PipelineFactory(
                model_path=self._parsed_s_cascade_decoder_uri.model,
                model_type=_enums.ModelType.S_CASCADE_DECODER,
                pipeline_type=_enums.PipelineType.TXT2IMG,
                subfolder=self._parsed_s_cascade_decoder_uri.subfolder,
                revision=self._parsed_s_cascade_decoder_uri.revision,
                unet_uri=self._second_model_unet_uri,
                text_encoder_uris=self._second_model_text_encoder_uris,
                quantizer_uri=self._second_model_quantizer_uri,
                quantizer_map=self._second_model_quantizer_map,

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

            creation_result = self._recall_secondary_pipeline()
            self._s_cascade_decoder_pipeline = creation_result.pipeline

        elif self._sdxl_refiner_uri is not None:

            self._recall_main_pipeline = _pipelines.PipelineFactory(
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
                quantizer_map=self._quantizer_map,
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

            self._recall_secondary_pipeline = _pipelines.PipelineFactory(
                model_path=self._parsed_sdxl_refiner_uri.model,
                model_type=_enums.ModelType.SDXL,
                pipeline_type=refiner_pipeline_type,
                subfolder=self._parsed_sdxl_refiner_uri.subfolder,
                revision=self._parsed_sdxl_refiner_uri.revision,
                unet_uri=self._second_model_unet_uri,
                text_encoder_uris=self._second_model_text_encoder_uris,
                quantizer_uri=self._second_model_quantizer_uri,
                quantizer_map=self._second_model_quantizer_map,

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
            self._sdxl_refiner_pipeline = self._recall_secondary_pipeline().pipeline
        else:
            self._recall_main_pipeline = _pipelines.PipelineFactory(
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
                quantizer_map=self._quantizer_map,
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
            dtype: _enums.DataType,
            device: str | None = None
    ):
        return self._prompt_weighter_loader.load(
            uri,
            model_type=model_type,
            dtype=dtype,
            device=device,
            local_files_only=self.local_files_only
        )

    def _default_prompt_weighter(self, *sources):
        for source in sources:
            if isinstance(source, str):  # Direct URI case
                return self._load_prompt_weighter(
                    source,
                    model_type=self.model_type,
                    dtype=self._dtype,
                    device=self._device
                )
            elif source is not None and source.weighter:  # Object case with weighter
                return self._load_prompt_weighter(
                    source.weighter,
                    model_type=self.model_type,
                    dtype=self._dtype,
                    device=self._device
                )
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

    def _load_latents_processors_with_batching(self, processors):
        if not processors:
            return None

        processor_chain = [[]]

        for processor in processors:
            if processor != _constants.LATENTS_PROCESSOR_SEP:
                processor_chain[-1].append(processor)
            else:
                processor_chain.append([])

        return [
            self._latents_processor_loader.load(
                p, device=self._device,
                model_type=self._model_type,
                local_files_only=self._local_files_only) for p in processor_chain
        ]

    def _process_input_latents(self,
                               title: str, latents: _types.Tensors,
                               processor_uris: _types.OptionalUris
                               ) -> list[torch.Tensor]:
        """
        Process input latents using configured latents input processors.

        :param title: Title for logging purposes
        :param latents: Input latents tensor
        :param processor_uris: List of processor URIs to apply
        :return: Processed latents tensor
        """

        if not processor_uris:
            return list(t.unsqueeze(0) if t.dim() == 3 else t for t in latents)

        processors = self._load_latents_processors_with_batching(processor_uris)

        _messages.debug_log(f'Processing {title} input latents with processors: {processor_uris}')
        if processors is not None:

            processed = []

            for idx, t in enumerate(latents):
                processor = processors[idx] if idx < len(processors) else None

                t = t.unsqueeze(0) if t.dim() == 3 else t

                # Process the latents
                if processor is not None:
                    processed.append(processor.process(self._pipeline, t))
                else:
                    processed.append(t)

        else:
            return []

        self._validate_latent_channels(processed)

        return processed

    def _process_output_latents(self, latents: torch.Tensor, processor_uris: _types.OptionalUris) -> torch.Tensor:
        """
        Process output latents using configured latents output processors.
        
        :param latents: Output latents tensor in unpacked format
        :param processor_uris: List of processor URIs to apply
        :return: Processed latents tensor
        """
        if not processor_uris:
            return latents

        processor = self._latents_processor_loader.load(
            processor_uris,
            model_type=self.model_type,
            device=self.device,
            local_files_only=self.local_files_only
        )

        # Ensure proper batch dimension for processing, also always output with a batch dimension

        _messages.debug_log(f'Processing output latents with processors: {processor_uris}')
        if processor is not None:
            return processor.process(self._pipeline, latents.unsqueeze(0) if latents.ndim == 3 else latents)
        else:
            return latents.unsqueeze(0) if latents.ndim == 3 else latents

    def _process_decoded_latents_images(
            self,
            images: _types.Images,
            processor_uris: _types.OptionalUris,
            user_args: DiffusionArguments) -> list[PIL.Image.Image]:
        """
        Process images decoded from latents using configured image processors.
        
        The processor handles the full flow: pre-resize processing, resizing to user dimensions, post-resize processing.
        
        :param images: List of PIL Images decoded from latents
        :param processor_uris: List of processor URIs to apply
        :param user_args: User arguments containing target dimensions
        :return: Processed images
        """
        if not processor_uris:
            # No processors configured, still need to resize to user dimensions
            return self._resize_images_to_user_dimensions(images, user_args)

        processor = self._decoded_latents_image_processor_loader.load(
            processor_uris,
            device=self.device,
            local_files_only=self.local_files_only
        )

        _messages.debug_log(
            f'Processing decoded latents images with processors: {processor_uris}'
        )

        if processor is not None:
            processed_images = []
            for image in images:

                if not _enums.model_type_is_s_cascade(self._model_type):

                    target_size = self._calc_image_target_size(image, user_args)

                    # The processor handles pre-resize, resize, and post-resize steps
                    image = processor.process(
                        image,
                        resize_resolution=target_size if target_size != image.size else None,
                        aspect_correct=user_args.aspect_correct,
                        align=8
                    )

                else:
                    # just align to 8
                    image = processor.process(image, align=8)

                processed_images.append(image.convert('RGB'))  # Ensure images are in RGB format
            return processed_images
        else:
            # Processor loader returned None, fallback to simple resize
            return self._resize_images_to_user_dimensions(images, user_args)

    @staticmethod
    def _calc_image_target_size(image: PIL.Image.Image, user_args: DiffusionArguments):
        if user_args.width is not None and user_args.height is not None:
            target_size = (user_args.width, user_args.height)
        elif user_args.width is not None:
            target_size = (user_args.width, image.height)
        elif user_args.height is not None:
            target_size = (image.width, user_args.height)
        else:
            target_size = image.size
        return target_size

    def _create_pipeline_result(self,
                                pipeline_output,
                                output_type: str = 'pil',
                                user_args: DiffusionArguments = None,
                                pipeline_kwargs: dict = None) -> PipelineWrapperResult:
        """
        Create a PipelineWrapperResult from pipeline output and process output latents if needed.

        :param pipeline_output: The output from a diffusers pipeline call
        :param output_type: The output type that was used ('pil' or 'latent')
        :param user_args: DiffusionArguments to get processor URIs from
        :param pipeline_kwargs: Pipeline keyword arguments, used for dimension prioritization
        :return: PipelineWrapperResult instance with processed latents if applicable
        """
        # Initialize variables for final object creation
        final_images = None
        final_latents = None

        # Process based on output type
        if output_type == 'latent':
            # Extract latents
            if hasattr(pipeline_output, 'images'):
                raw_latents = pipeline_output.images
            else:
                raw_latents = getattr(pipeline_output, 'latents', None)

            # Normalize latents to torch tensors on CPU
            if raw_latents is not None:
                normalized_latents = []
                for latent in raw_latents:
                    if isinstance(latent, numpy.ndarray):
                        latent_tensor = torch.from_numpy(latent).cpu()
                    elif isinstance(latent, torch.Tensor):
                        latent_tensor = latent.cpu()
                    else:
                        raise TypeError(
                            f"Unexpected latent type: {type(latent)}. Expected numpy.ndarray or torch.Tensor"
                        )
                    normalized_latents.append(latent_tensor)
                final_latents = normalized_latents
        else:
            # Extract PIL images
            final_images = getattr(pipeline_output, 'images', None)

        # Process latents if we have them
        if final_latents is not None:
            # For Flux models, unpack latents to external unpacked format
            if _enums.model_type_is_flux(self._model_type):
                # Get dimensions with priority: pipeline_kwargs > user_args > None
                height = None
                width = None
                if pipeline_kwargs:
                    height = pipeline_kwargs.get('height')
                    width = pipeline_kwargs.get('width')
                if height is None and user_args:
                    height = user_args.height
                if width is None and user_args:
                    width = user_args.width

                unpacked_latents = []
                for latent in final_latents:
                    unpacked_latent = self._unpack_flux_latents(latent, height, width)
                    unpacked_latents.append(unpacked_latent)
                final_latents = unpacked_latents

            # Apply post-processors if configured
            if user_args and user_args.latents_post_processors:
                if len(final_latents) == 1:
                    # Single latent, process directly
                    processed_latent = self._process_output_latents(
                        final_latents[0], user_args.latents_post_processors
                    )
                    final_latents = [processed_latent]
                else:
                    # Multiple latents, batch them together for processing
                    # Ensure all tensors have batch dimension before concatenating
                    tensors_with_batch = []
                    for latent in final_latents:
                        if latent.ndim == 3:  # [C, H, W] - add batch dimension
                            tensors_with_batch.append(latent.unsqueeze(0))  # [1, C, H, W]
                        else:  # Already has batch dimension
                            tensors_with_batch.append(latent)

                    batched_latents = torch.cat(tensors_with_batch, dim=0)
                    processed_batched = self._process_output_latents(
                        batched_latents, user_args.latents_post_processors
                    )

                    # Split back into individual tensors matching original shapes
                    processed_latents = []
                    start_idx = 0
                    for original_latent in final_latents:
                        # Determine how many batch items this original tensor contributed
                        batch_size = 1 if original_latent.ndim == 3 else original_latent.shape[0]
                        end_idx = start_idx + batch_size

                        processed_tensor = processed_batched[start_idx:end_idx]

                        # If original was 3D, squeeze back to 3D
                        if original_latent.ndim == 3:
                            processed_tensor = processed_tensor.squeeze(0)

                        processed_latents.append(processed_tensor)
                        start_idx = end_idx

                    final_latents = processed_latents

        # Apply inpaint crop pasting if we cropped earlier
        if final_images is not None and self._inpaint_crop_info is not None:
            crop_info = self._inpaint_crop_info

            # Paste generated images back onto originals
            pasted_images = self._paste_inpaint_result(
                original_images=crop_info.original_images,
                generated_images=final_images,
                crop_bounds=crop_info.crop_bounds,
                masks=crop_info.original_masks if crop_info.use_masked else None,
                feather=crop_info.feather
            )

            final_images = pasted_images

            # Clean up temporary crop info
            self._inpaint_crop_info = None

        # Create and return the result object at the end
        return PipelineWrapperResult(images=final_images, latents=final_latents)

    def _argument_help_check(self, args: DiffusionArguments):
        scheduler_help = _help.scheduler_is_help(args.scheduler_uri)
        second_model_scheduler_help = _help.scheduler_is_help(args.second_model_scheduler_uri)
        text_encoder_help = _help.text_encoder_is_help(self.text_encoder_uris)
        second_model_text_encoder_help = _help.text_encoder_is_help(self.second_model_text_encoder_uris)
        help_text = []
        model_path = self.model_path

        if scheduler_help or second_model_scheduler_help:
            pipe_class = _pipelines.get_pipeline_class(
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
            second_pipe_class = _pipelines.get_pipeline_class(
                model_type=_enums.ModelType.SDXL if
                self.sdxl_refiner_uri else _enums.ModelType.S_CASCADE_DECODER,
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
        _schedulers.load_scheduler(
            pipeline=self._pipeline,
            scheduler_uri=args.scheduler_uri
        )
        if self._sdxl_refiner_pipeline:
            _schedulers.load_scheduler(
                pipeline=self._sdxl_refiner_pipeline,
                scheduler_uri=second_model_scheduler_uri
            )
        if self._s_cascade_decoder_pipeline:
            _schedulers.load_scheduler(
                pipeline=self._s_cascade_decoder_pipeline,
                scheduler_uri=second_model_scheduler_uri
            )
        _pipelines.set_vae_tiling_and_slicing(
            pipeline=self._pipeline,
            tiling=args.vae_tiling,
            slicing=args.vae_slicing
        )

    def _auto_denoise_range_check(self, args: DiffusionArguments):
        if _enums.model_type_is_sdxl(self._model_type):

            have_latent_input = any(
                _torchutil.is_tensor(i) for i in args.images
            ) if args.images else False

            have_image_input = any(
                isinstance(i, PIL.Image.Image) for i in args.images
            ) if args.images else False

            if args.denoising_start is not None and args.denoising_start != 0.0:
                if args.mask_images and have_latent_input:
                    raise _pipelines.UnsupportedPipelineConfigError(
                        'Denoising start parameter is not supported for SDXL models '
                        'with latent input and inpaint mask images defined. In order '
                        'to refine an inpainted image, just pass in the generated image '
                        'and use normal inpainting mode on it.'
                    )

                if not have_latent_input:
                    raise _pipelines.UnsupportedPipelineConfigError(
                        'Denoising start parameter is not supported for SDXL models '
                        'without latents being passed as image inputs.'
                    )

                if have_image_input:
                    raise _pipelines.UnsupportedPipelineConfigError(
                        'Denoising start parameter is not supported for SDXL models '
                        'with image inputs for img2img, it can only accept latents.'
                    )

    def _auto_latents_check(self, args: DiffusionArguments):
        if args.output_latents:
            if self.adetailer_detector_uris:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Adetailer does not support outputting to latents.'
                )

            if _enums.model_type_is_floyd(self.model_type):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Deep Floyd model types do not support outputting to latents.'
                )
        else:
            if args.latents_post_processors:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Cannot use latents post processors when not outputting to latents.'
                )

        if args.images and any(_torchutil.is_tensor(i) for i in args.images):
            # validation that input type is not mixed happens in _get_pipeline_defaults

            if _enums.model_type_is_floyd(self.model_type):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Deep Floyd model types do not support accepting latents as input.'
                )

            if _enums.model_type_is_s_cascade(self.model_type):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Stable Cascade does not support accepting latents as input.'
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

            if args.ras_index_fusion and self._pipeline.transformer.config.qk_norm == 'rms_norm':
                raise _pipelines.UnsupportedPipelineConfigError(
                    'RAS index fusion not supported with SD3.5, only SD3.'
                )

            if args.ras_index_fusion and not importlib.util.find_spec('triton'):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'RAS index fusion is only supported with triton / triton-windows installed.')

            if self.model_cpu_offload:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'RAS does not support model CPU offloading.')

            if args.ras_index_fusion and self.model_sequential_offload:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Index fusion is not supported for RAS when sequential offloading is enabled.')

            if args.ras_index_fusion and (
                    self.quantizer_uri or (self._unet_uri and _uris.UNetUri.parse(self._unet_uri).quantizer)
            ):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Index fusion is not supported for RAS when UNet quantization is enabled, '
                    'quantize the text encoders individually.')

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
            if not (
                    self.model_type == _enums.ModelType.SDXL or
                    self.model_type == _enums.ModelType.SDXL_PIX2PIX or
                    self.model_type == _enums.ModelType.KOLORS or
                    self.model_type == _enums.ModelType.SD or
                    self.model_type == _enums.ModelType.PIX2PIX or
                    self.model_type == _enums.ModelType.UPSCALER_X4):
                raise _pipelines.UnsupportedPipelineConfigError(
                    f'DeepCache is only supported with Stable Diffusion, Stable Diffusion XL, '
                    f'Stable Diffusion Upscaler X4, Kolors, and Pix2Pix variants.'
                )

        for prop in args.__dict__.keys():
            if prop.startswith('sdxl_refiner_deep_cache_'):
                value = getattr(args, prop)
                if value is not None or (isinstance(value, bool) and value is True):
                    args.sdxl_refiner_deep_cache = True
                    break

    def _auto_hi_diffusion_check(self, args: DiffusionArguments):
        if args.hi_diffusion:
            if not (
                    self.model_type == _enums.ModelType.SDXL or
                    self.model_type == _enums.ModelType.KOLORS or
                    self.model_type == _enums.ModelType.SD):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'HiDiffusion is only supported for '
                    '--model-type sd, sdxl, and kolors'
                )

            if self.t2i_adapter_uris:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'HiDiffusion is not supported with T2I Adapters'
                )
        else:
            if args.hi_diffusion_no_raunet is not None:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'HiDiffusion no-raunet option is only supported when HiDiffusion is enabled.'
                )
            if args.hi_diffusion_no_win_attn is not None:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'HiDiffusion no-window-attention option is only supported when HiDiffusion is enabled.'
                )

    def _auto_sada_check(self, args: DiffusionArguments):
        for prop in args.__dict__.keys():
            if prop.startswith('sada_'):
                value = getattr(args, prop)
                if value is not None or (isinstance(value, bool) and value is True):
                    args.sada = True
                    break

        if args.sada:
            # SADA supports SD, SDXL/Kolors, and Flux
            if not (
                    self.model_type == _enums.ModelType.SD or
                    self.model_type == _enums.ModelType.SDXL or
                    self.model_type == _enums.ModelType.KOLORS or
                    _enums.model_type_is_flux(self.model_type)):
                raise _pipelines.UnsupportedPipelineConfigError(
                    'SADA is only supported for '
                    '--model-type sd, sdxl, kolors, and flux*'
                )

            # Check for conflicts with other acceleration methods

            if args.tea_cache:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'SADA cannot be used simultaneously with TeaCache'
                )

            if args.deep_cache:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'SADA cannot be used simultaneously with DeepCache'
                )

            if args.hi_diffusion:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'SADA cannot be used simultaneously with HiDiffusion'
                )

            # Validate Lagrangian interpolation parameters
            sada_args = self._get_sada_args(args)
            if sada_args['lagrange_term'] != 0:
                if (sada_args['lagrange_int'] is None or
                        sada_args['lagrange_step'] is None):
                    raise _pipelines.UnsupportedPipelineConfigError(
                        'When using SADA Lagrangian interpolation (lagrange_term != 0), '
                        'both lagrange_int and lagrange_step must be specified'
                    )
                if sada_args['lagrange_step'] % sada_args['lagrange_int'] != 0:
                    raise _pipelines.UnsupportedPipelineConfigError(
                        'SADA lagrange_step must be divisible by lagrange_int'
                    )

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
                    'TeaCache is only supported for Flux.'
                )

            if self.model_cpu_offload:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'TeaCache does not support model CPU offloading.'
                )

    def _auto_freeu_check(self, args: DiffusionArguments):
        freeu_model_types = {
            _enums.ModelType.SD,
            _enums.ModelType.SDXL,
            _enums.ModelType.KOLORS,
            _enums.ModelType.PIX2PIX,
            _enums.ModelType.SDXL_PIX2PIX,
            _enums.ModelType.UPSCALER_X2,
            _enums.ModelType.UPSCALER_X4
        }

        if args.freeu_params is not None:
            if self._model_type not in freeu_model_types:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'Current primary model does not utilize a UNet, and therefore does not support FreeU parameters.'
                )

        if args.sdxl_refiner_freeu_params is not None:
            if self._sdxl_refiner_uri is None:
                raise _pipelines.UnsupportedPipelineConfigError(
                    'SDXL refiner is not in use, so cannot supply FreeU parameters to it.'
                )

    def get_decoded_latents_size(self, latents: torch.Tensor) -> _types.Size:
        """
        Given a latent tensor return the expected decoded image (width, height) in pixels.

        :param latents: Latent tensor of shape [B, C, H, W] or [C, H, W].
        :return: width, height
        """
        if self._pipeline is None:
            raise _pipelines.UnsupportedPipelineConfigError(
                'Cannot decode latents as a pipeline has not been initialized, you must perform a generation first.'
            )

        if not hasattr(self._pipeline, 'vae') or self._pipeline.vae is None:
            raise _pipelines.UnsupportedPipelineConfigError(
                'Cannot decode latents as the initialized pipeline does not have a VAE.'
            )

        # Get the latent dimensions
        if len(latents.shape) == 4:
            _, _, h, w = latents.shape
        else:
            _, h, w = latents.shape

        # The scale factor is fixed at 8 due to the VAE architecture having 3 downsampling blocks
        # This is true for SD1.5, SDXL, and most other stable diffusion VAEs
        scale_factor = 8

        # Calculate the decoded size
        height = h * scale_factor
        width = w * scale_factor

        return width, height

    @torch.inference_mode()
    def decode_latents(
            self,
            latents: _types.TensorsOrTensor,
    ) -> list[PIL.Image.Image]:
        """
        Decode latents using the main pipeline's VAE.

        A generation must have occurred at least once for this method to be usable.

        You must be using a model type that utilizes a VAE, Stable Cascade and Deep Floyd model types
        are not supported by this method.

        :param latents: Latents to decode, can be a sequence of tensors (batched), or a single tensor.
            A single tensor with a batch dimension [B, C, H, W] will be assumed to be a batch of latents
            and batched if the batch dimension is > 1, [C, H, W] will be assumed to be a single latent tensor.
            For Flux models, latents should be in unpacked format [B, C, H, W] where C=16.

        :raise dgenerate.pipelinewrapper.UnsupportedPipelineConfigError: If the decoding the latents is not supported.
        """

        if self._pipeline is None:
            raise _pipelines.UnsupportedPipelineConfigError(
                'Cannot decode latents as a pipeline has not been initialized, you must perform a generation first.'
            )

        if not hasattr(self._pipeline, 'vae') or self._pipeline.vae is None:
            raise _pipelines.UnsupportedPipelineConfigError(
                'Cannot decode latents as the initialized pipeline does not have a VAE.'
            )

        if isinstance(latents, torch.Tensor):
            if latents.ndim == 3:
                latents = latents.unsqueeze(0)
        elif latents:
            if latents[0].ndim == 4:
                # List of [B, C, H, W] tensors - concatenate along batch dimension
                latents = torch.cat(list(latents), dim=0)  # [total_B, C, H, W]
            elif latents[0].ndim == 3:
                # List of [C, H, W] tensors - stack to create batch dimension
                latents = torch.stack(list(latents), dim=0)  # [num_tensors, C, H, W]

        vae = self._pipeline.vae
        vae_og_dtype = vae.dtype

        needs_upcasting = vae.dtype == torch.float16 and getattr(vae.config, 'force_upcast', False)

        if needs_upcasting:
            try:
                vae.to(self._device, dtype=torch.float32)
            except NotImplementedError:
                vae.to(dtype=torch.float32)

        try:
            if latents.dtype != vae.dtype:
                latents = latents.to(dtype=vae.dtype)

            if latents.device != vae.device:
                latents = latents.to(self._device)

            if _enums.model_type_is_sdxl(self.model_type) or _enums.model_type_is_kolors(self.model_type):
                # SDXL and Kolors
                has_latents_mean = \
                    hasattr(vae.config, "latents_mean") and \
                    vae.config.latents_mean is not None
                has_latents_std = \
                    hasattr(vae.config, "latents_std") and \
                    vae.config.latents_std is not None

                if has_latents_mean and has_latents_std:
                    latents_mean = (
                        torch.tensor(vae.config.latents_mean).reshape(1, 4, 1, 1).expand(latents.shape[0], -1, 1, 1).to(
                            latents.device, latents.dtype
                        )
                    )
                    latents_std = (
                        torch.tensor(vae.config.latents_std).reshape(1, 4, 1, 1).expand(latents.shape[0], -1, 1, 1).to(
                            latents.device, latents.dtype
                        )
                    )
                    latents = latents * latents_std / vae.config.scaling_factor + latents_mean
                else:
                    latents = latents / vae.config.scaling_factor
            elif _enums.model_type_is_sd15(self.model_type) or _enums.model_type_is_sd2(self.model_type):
                # SD15 and SD2
                latents = latents / vae.config.scaling_factor
            elif _enums.model_type_is_sd3(self.model_type):
                # SD3
                latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
            elif _enums.model_type_is_flux(self.model_type):
                # Flux - latents are already in unpacked format [B, C, H, W]
                # Apply VAE scaling and shift
                latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
            else:
                raise _pipelines.UnsupportedPipelineConfigError(
                    f'Unable to decode latents for unsupported model type: {_enums.get_model_type_string(self.model_type)}'
                )

            decoded_images = vae.decode(latents).sample

        finally:
            if needs_upcasting:
                vae.to(dtype=vae_og_dtype)

        return self._pipeline.image_processor.postprocess(decoded_images)

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

        # always reset inpaint crop state per call
        self._inpaint_crop_info = None

        copy_args = DiffusionArguments()

        if args is not None:
            copy_args.set_from(args)

        copy_args.set_from(kwargs, missing_value_throws=False)

        self._auto_freeu_check(copy_args)
        self._auto_tea_cache_check(copy_args)
        self._auto_deep_cache_check(copy_args)
        self._auto_hi_diffusion_check(copy_args)
        self._auto_sada_check(copy_args)
        self._auto_latents_check(copy_args)
        self._auto_denoise_range_check(copy_args)

        help_text = self._argument_help_check(copy_args)
        if help_text:
            raise DiffusionArgumentsHelpException(help_text)

        _messages.debug_log(f'Calling Pipeline Wrapper: "{self}"')
        _messages.debug_log(f'Pipeline Wrapper Args: ',
                            lambda: _textprocessing.debug_format_args(
                                copy_args.get_pipeline_wrapper_kwargs()))

        self._lazy_init_pipeline(copy_args)

        # this needs to happen even if a cached pipeline
        # was loaded, since the settings for scheduler
        # and vae tiling / slicing may be different
        self._set_scheduler_and_vae_settings(copy_args)

        pipeline_args = \
            self._get_pipeline_defaults(user_args=copy_args)

        # needs the pipeline initialized
        self._auto_ras_check(copy_args)

        try:
            if self.model_type == _enums.ModelType.S_CASCADE:
                result = self._call_torch_s_cascade(
                    pipeline_args=pipeline_args,
                    user_args=copy_args)
            elif _enums.model_type_is_flux(self.model_type):
                result = self._call_torch_flux(pipeline_args=pipeline_args,
                                               user_args=copy_args)
            else:
                result = self._call_torch(pipeline_args=pipeline_args,
                                          user_args=copy_args)
        except _DenoiseRangeError as e:
            raise _pipelines.UnsupportedPipelineConfigError(e) from e

        DiffusionPipelineWrapper.__LAST_RECALL_PIPELINE = self._recall_main_pipeline
        DiffusionPipelineWrapper.__LAST_RECALL_SECONDARY_PIPELINE = self._recall_secondary_pipeline

        return result

    def __str__(self):
        return f'{self.__class__.__name__}({str(_types.get_public_attributes(self))})'

    def __repr__(self):
        return str(self)


__all__ = _types.module_all()
