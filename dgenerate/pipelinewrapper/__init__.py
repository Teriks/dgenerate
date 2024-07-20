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


import os

import dgenerate.types as _types
from .cache import \
    CACHE_MEMORY_CONSTRAINTS, \
    PIPELINE_CACHE_MEMORY_CONSTRAINTS, \
    VAE_CACHE_MEMORY_CONSTRAINTS, \
    UNET_CACHE_MEMORY_CONSTRAINTS, \
    CONTROL_NET_CACHE_MEMORY_CONSTRAINTS, \
    TEXT_ENCODER_CACHE_MEMORY_CONSTRAINTS, \
    clear_vae_cache, \
    clear_unet_cache, \
    vae_cache_size, \
    clear_text_encoder_cache, \
    text_encoder_cache_size, \
    unet_cache_size, \
    clear_model_cache, \
    clear_pipeline_cache, \
    pipeline_cache_size, \
    uri_hash_with_parser, \
    enforce_cache_constraints, \
    clear_control_net_cache, \
    control_net_cache_size, \
    enforce_pipeline_cache_constraints, \
    enforce_vae_cache_constraints, \
    enforce_unet_cache_constraints, \
    enforce_control_net_cache_constraints, \
    enforce_text_encoder_cache_constraints, \
    pipeline_create_update_cache_info, \
    pipeline_off_cpu_update_cache_info, \
    pipeline_to_cpu_update_cache_info, \
    vae_create_update_cache_info, \
    unet_create_update_cache_info, \
    uri_list_hash_with_parser, \
    controlnet_create_update_cache_info, \
    text_encoder_create_update_cache_info
from .constants import \
    DEFAULT_SDXL_OUTPUT_WIDTH, \
    DEFAULT_OUTPUT_HEIGHT, \
    DEFAULT_SDXL_OUTPUT_HEIGHT, \
    DEFAULT_FLOYD_IF_OUTPUT_HEIGHT, \
    DEFAULT_FLOYD_IF_OUTPUT_WIDTH, \
    DEFAULT_FLOYD_SUPERRESOLUTION_NOISE_LEVEL, \
    DEFAULT_FLOYD_SUPERRESOLUTION_IMG2IMG_NOISE_LEVEL, \
    DEFAULT_FLOYD_SUPERRESOLUTION_INPAINT_NOISE_LEVEL, \
    DEFAULT_SEED, \
    DEFAULT_OUTPUT_WIDTH, \
    DEFAULT_GUIDANCE_SCALE, \
    DEFAULT_INFERENCE_STEPS, \
    DEFAULT_IMAGE_SEED_STRENGTH, \
    DEFAULT_IMAGE_GUIDANCE_SCALE, \
    DEFAULT_SDXL_HIGH_NOISE_FRACTION, \
    DEFAULT_X4_UPSCALER_NOISE_LEVEL, \
    DEFAULT_S_CASCADE_DECODER_GUIDANCE_SCALE, \
    DEFAULT_S_CASCADE_DECODER_INFERENCE_STEPS, \
    DEFAULT_S_CASCADE_OUTPUT_HEIGHT, \
    DEFAULT_S_CASCADE_OUTPUT_WIDTH
from .enums import \
    ModelType, \
    DataType, \
    model_type_is_sd3, \
    model_type_is_sdxl, \
    model_type_is_floyd_if, \
    model_type_is_pix2pix, \
    model_type_is_upscaler, \
    model_type_is_floyd_ifs, \
    model_type_is_floyd, \
    model_type_is_flax, \
    model_type_is_s_cascade, \
    PipelineType, \
    get_model_type_enum, \
    get_flax_dtype, \
    get_torch_dtype, \
    get_data_type_enum, \
    have_jax_flax, \
    supported_data_type_enums, \
    model_type_is_torch, \
    get_data_type_string, \
    supported_data_type_strings, \
    get_model_type_string, \
    supported_model_type_enums, \
    get_pipeline_type_string, \
    get_pipeline_type_enum, \
    supported_model_type_strings
from .hfutil import \
    ModelNotFoundError, \
    NonHFModelDownloadError
from .pipelines import \
    InvalidModelFileError, \
    InvalidSchedulerNameError, \
    TorchPipelineFactory, \
    FlaxPipelineCreationResult, \
    TorchPipelineCreationResult, \
    PipelineCreationResult, \
    FlaxPipelineFactory, \
    ArgumentHelpException, \
    SchedulerHelpException, \
    TextEncodersHelpException, \
    create_flax_diffusion_pipeline, \
    set_vae_slicing_tiling, \
    create_torch_diffusion_pipeline, \
    estimate_pipeline_memory_use, \
    load_scheduler, \
    scheduler_is_help, \
    text_encoder_is_help, \
    UnsupportedPipelineConfigError, \
    get_torch_pipeline_modules, \
    is_model_cpu_offload_enabled, \
    is_sequential_cpu_offload_enabled, \
    call_pipeline, \
    get_torch_device, \
    pipeline_to, \
    enable_sequential_cpu_offload, \
    enable_model_cpu_offload, \
    get_torch_device_string
from .uris import \
    InvalidModelUriError, \
    LoRAUri, \
    TextualInversionUri, \
    FlaxVAEUri, \
    FlaxControlNetUri, \
    SDXLRefinerUri, \
    ModelUriLoadError, \
    InvalidControlNetUriError, \
    ControlNetUriLoadError, \
    InvalidLoRAUriError, \
    LoRAUriLoadError, \
    InvalidSDXLRefinerUriError, \
    InvalidTextualInversionUriError, \
    TextualInversionUriLoadError, \
    InvalidUNetUriError, \
    UNetUriLoadError, \
    InvalidVaeUriError, \
    VAEUriLoadError, \
    TorchControlNetUri, \
    TorchVAEUri, \
    TorchUNetUri, \
    FlaxUNetUri
from .util import \
    InvalidDeviceOrdinalException, \
    is_valid_device_string
from .wrapper import \
    PipelineWrapperResult, \
    DiffusionPipelineWrapper, \
    DiffusionArguments

__doc__ = """
huggingface diffusers pipeline wrapper / driver interface.

All functionality needed from the diffusers library is behind this interface.
"""

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

__all__ = _types.module_all()
