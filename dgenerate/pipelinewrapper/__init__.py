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


import dgenerate.types as _types

# Cache constants
from .cache import (
    CACHE_MEMORY_CONSTRAINTS,
    PIPELINE_CACHE_MEMORY_CONSTRAINTS,
    VAE_CACHE_MEMORY_CONSTRAINTS,
    UNET_CACHE_MEMORY_CONSTRAINTS,
    CONTROLNET_CACHE_MEMORY_CONSTRAINTS,
    TEXT_ENCODER_CACHE_MEMORY_CONSTRAINTS,
    ADAPTER_CACHE_MEMORY_CONSTRAINTS,
    IMAGE_ENCODER_CACHE_MEMORY_CONSTRAINTS,
    TRANSFORMER_CACHE_MEMORY_CONSTRAINTS
)

# Clear Cache Functions
from .cache import (
    clear_vae_cache,
    clear_unet_cache,
    clear_text_encoder_cache,
    clear_model_cache,
    clear_pipeline_cache,
    clear_controlnet_cache,
    clear_adapter_cache,
    clear_image_encoder_cache,
    clear_transformer_cache
)

# Cache Size Functions
from .cache import (
    vae_cache_size,
    unet_cache_size,
    text_encoder_cache_size,
    pipeline_cache_size,
    controlnet_cache_size,
    adapter_cache_size,
    image_encoder_cache_size,
    transformer_cache_size
)

# Enforce Cache Constraints Functions
from .cache import (
    enforce_cache_constraints,
    enforce_pipeline_cache_constraints,
    enforce_vae_cache_constraints,
    enforce_unet_cache_constraints,
    enforce_controlnet_cache_constraints,
    enforce_adapter_cache_constraints,
    enforce_text_encoder_cache_constraints,
    enforce_image_encoder_cache_constraints,
    enforce_transformer_cache_constraints
)

# Cache Info Update Functions
from .cache import (
    pipeline_create_update_cache_info,
    pipeline_off_cpu_update_cache_info,
    pipeline_to_cpu_update_cache_info,
    vae_create_update_cache_info,
    vae_to_cpu_update_cache_info,
    vae_off_cpu_update_cache_info,
    unet_create_update_cache_info,
    unet_to_cpu_update_cache_info,
    unet_off_cpu_update_cache_info,
    controlnet_create_update_cache_info,
    controlnet_to_cpu_update_cache_info,
    controlnet_off_cpu_update_cache_info,
    adapter_create_update_cache_info,
    adapter_to_cpu_update_cache_info,
    adapter_off_cpu_update_cache_info,
    text_encoder_create_update_cache_info,
    text_encoder_to_cpu_update_cache_info,
    text_encoder_off_cpu_update_cache_info,
    image_encoder_create_update_cache_info,
    image_encoder_to_cpu_update_cache_info,
    image_encoder_off_cpu_update_cache_info,
    transformer_create_update_cache_info,
    transformer_to_cpu_update_cache_info,
    transformer_off_cpu_update_cache_info
)

# Utility Functions
from .cache import (
    uri_hash_with_parser,
    uri_list_hash_with_parser
)

# Constants
from .constants import (
    DEFAULT_SDXL_OUTPUT_WIDTH,
    DEFAULT_OUTPUT_HEIGHT,
    DEFAULT_SDXL_OUTPUT_HEIGHT,
    DEFAULT_FLOYD_IF_OUTPUT_HEIGHT,
    DEFAULT_FLOYD_IF_OUTPUT_WIDTH,
    DEFAULT_FLOYD_SUPERRESOLUTION_NOISE_LEVEL,
    DEFAULT_FLOYD_SUPERRESOLUTION_IMG2IMG_NOISE_LEVEL,
    DEFAULT_FLOYD_SUPERRESOLUTION_INPAINT_NOISE_LEVEL,
    DEFAULT_SEED,
    DEFAULT_OUTPUT_WIDTH,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_IMAGE_SEED_STRENGTH,
    DEFAULT_IMAGE_GUIDANCE_SCALE,
    DEFAULT_SDXL_HIGH_NOISE_FRACTION,
    DEFAULT_X4_UPSCALER_NOISE_LEVEL,
    DEFAULT_S_CASCADE_DECODER_GUIDANCE_SCALE,
    DEFAULT_S_CASCADE_DECODER_INFERENCE_STEPS,
    DEFAULT_S_CASCADE_OUTPUT_HEIGHT,
    DEFAULT_S_CASCADE_OUTPUT_WIDTH,
    DEFAULT_SD3_OUTPUT_WIDTH,
    DEFAULT_SD3_OUTPUT_HEIGHT,
    DEFAULT_FLUX_OUTPUT_WIDTH,
    DEFAULT_FLUX_OUTPUT_HEIGHT,
    DEFAULT_ADETAILER_MASK_SHAPE,
    DEFAULT_ADETAILER_MASK_DILATION,
    DEFAULT_ADETAILER_MASK_BLUR,
    DEFAULT_ADETAILER_MASK_PADDING,
    DEFAULT_ADETAILER_DETECTOR_PADDING,
    DEFAULT_PAG_SCALE,
    DEFAULT_PAG_ADAPTIVE_SCALE,
    DEFAULT_SDXL_REFINER_PAG_ADAPTIVE_SCALE,
    DEFAULT_SDXL_REFINER_PAG_SCALE
)

# Enums
from .enums import (
    ModelType,
    DataType,
    model_type_is_sd3,
    model_type_is_sdxl,
    model_type_is_floyd_if,
    model_type_is_pix2pix,
    model_type_is_upscaler,
    model_type_is_floyd_ifs,
    model_type_is_floyd,
    model_type_is_s_cascade,
    PipelineType,
    get_model_type_enum,
    get_torch_dtype,
    get_data_type_enum,
    supported_data_type_enums,
    model_type_is_torch,
    get_data_type_string,
    supported_data_type_strings,
    get_model_type_string,
    supported_model_type_enums,
    get_pipeline_type_string,
    get_pipeline_type_enum,
    supported_model_type_strings,
    model_type_is_flux,
    model_type_is_kolors
)

# HF Utility Errors
from .hfutil import (
    ModelNotFoundError,
    NonHFModelDownloadError
)

# Pipelines
from .pipelines import (
    InvalidModelFileError,
    SchedulerLoadError,
    SchedulerArgumentError,
    InvalidSchedulerNameError,
    TorchPipelineFactory,
    TorchPipelineCreationResult,
    PipelineCreationResult,
    ArgumentHelpException,
    SchedulerHelpException,
    TextEncodersHelpException,
    set_vae_slicing_tiling,
    get_torch_pipeline_class,
    create_torch_diffusion_pipeline,
    estimate_pipeline_memory_use,
    get_scheduler_uri_schema,
    load_scheduler,
    scheduler_is_help,
    text_encoder_is_help,
    UnsupportedPipelineConfigError,
    get_torch_pipeline_modules,
    is_model_cpu_offload_enabled,
    is_sequential_cpu_offload_enabled,
    call_pipeline,
    get_torch_device,
    pipeline_to,
    enable_sequential_cpu_offload,
    enable_model_cpu_offload,
    get_torch_device_string,
    get_last_called_pipeline,
    destroy_last_called_pipeline
)

# Quantization and module flag test
from .quanto import (
    quantize_freeze,
    is_quantized_and_frozen
)

# URI Errors and Types
from .uris import (
    InvalidModelUriError,
    LoRAUri,
    TextualInversionUri,
    SDXLRefinerUri,
    ModelUriLoadError,
    InvalidControlNetUriError,
    ControlNetUriLoadError,
    InvalidLoRAUriError,
    LoRAUriLoadError,
    InvalidSDXLRefinerUriError,
    InvalidTextualInversionUriError,
    TextualInversionUriLoadError,
    InvalidUNetUriError,
    UNetUriLoadError,
    InvalidVaeUriError,
    VAEUriLoadError,
    ControlNetUri,
    VAEUri,
    UNetUri,
    IPAdapterUri,
    T2IAdapterUri,
    T2IAdapterUriLoadError,
    InvalidT2IAdapterUriError,
    TextEncoderUriLoadError,
    TextEncoderUri,
    InvalidSCascadeDecoderUriError,
    InvalidTextEncoderUriError,
    SCascadeDecoderUri
)

# Utility Functions
from .util import (
    InvalidDeviceOrdinalException,
    is_valid_device_string
)

# Wrapper
from .wrapper import (
    PipelineWrapperResult,
    DiffusionPipelineWrapper,
    DiffusionArguments
)

__doc__ = """
huggingface diffusers pipeline wrapper / driver interface.

All functionality needed from the diffusers library is behind this interface.
"""

__all__ = _types.module_all()
