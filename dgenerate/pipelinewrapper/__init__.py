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
    DEFAULT_SDXL_REFINER_PAG_SCALE,
    DEFAULT_KOLORS_OUTPUT_HEIGHT,
    DEFAULT_KOLORS_OUTPUT_WIDTH,
    CONTROLNET_CACHE_MEMORY_CONSTRAINTS,
    TEXT_ENCODER_CACHE_MEMORY_CONSTRAINTS,
    ADAPTER_CACHE_MEMORY_CONSTRAINTS,
    IMAGE_ENCODER_CACHE_MEMORY_CONSTRAINTS,
    TRANSFORMER_CACHE_MEMORY_CONSTRAINTS,
    UNET_CACHE_MEMORY_CONSTRAINTS,
    CACHE_MEMORY_CONSTRAINTS,
    PIPELINE_CACHE_MEMORY_CONSTRAINTS,
    VAE_CACHE_MEMORY_CONSTRAINTS
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
from .schedulers import (
    SchedulerLoadError,
    SchedulerArgumentError,
    InvalidSchedulerNameError,
    get_scheduler_uri_schema,
    load_scheduler
)

# Pipelines
from .pipelines import (
    InvalidModelFileError,
    TorchPipelineFactory,
    TorchPipelineCreationResult,
    PipelineCreationResult,
    set_vae_tiling_and_slicing,
    get_torch_pipeline_class,
    create_torch_diffusion_pipeline,
    estimate_pipeline_cache_footprint,
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

from .help import (
    text_encoder_help,
    text_encoder_is_help,
    scheduler_is_help,
    get_scheduler_help
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

# Utility
from .util import (
    ModelNotFoundError,
    NonHFModelDownloadError,
    NonHFConfigDownloadError,
    InvalidDeviceOrdinalException,
    is_valid_device_string
)
# Wrapper
from .wrapper import (
    PipelineWrapperResult,
    DiffusionPipelineWrapper,
    DiffusionArguments,
    DiffusionArgumentsHelpException
)

__doc__ = """
huggingface diffusers pipeline wrapper / driver interface.

All functionality needed from the diffusers library is behind this interface.
"""

__all__ = _types.module_all()
