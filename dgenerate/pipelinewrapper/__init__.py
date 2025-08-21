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

__doc__ = """
huggingface diffusers pipeline wrapper / driver interface.

All functionality needed from the diffusers library is behind this interface.
"""

import dgenerate.pipelinewrapper.constants
import dgenerate.types as _types
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
    get_data_type_string,
    supported_data_type_strings,
    get_model_type_string,
    supported_model_type_enums,
    get_pipeline_type_string,
    get_pipeline_type_enum,
    supported_model_type_strings,
    model_type_is_flux,
    model_type_is_kolors,
    model_type_is_sd2,
    model_type_is_sd15
)
from .help import (
    text_encoder_help,
    text_encoder_is_help,
    scheduler_is_help,
    scheduler_is_help_args,
    get_scheduler_help,
)
# Pipelines
from .pipelines import (
    InvalidModelFileError,
    PipelineFactory,
    PipelineCreationResult,
    PipelineCreationResult,
    set_vae_tiling_and_slicing,
    get_pipeline_class,
    create_diffusion_pipeline,
    estimate_pipeline_cache_footprint,
    UnsupportedPipelineConfigError,
    get_pipeline_modules,
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

from .schedulers import (
    SchedulerLoadError,
    SchedulerArgumentError,
    InvalidSchedulerNameError,
    get_compatible_schedulers,
    get_scheduler_uri_schema,
    load_scheduler
)

# URI Errors and Types
from .uris import (
    # General Model URI Errors
    InvalidModelUriError,
    ModelUriLoadError,

    # LoRA
    LoRAUri,
    InvalidLoRAUriError,
    LoRAUriLoadError,

    # Textual Inversion
    TextualInversionUri,
    InvalidTextualInversionUriError,
    TextualInversionUriLoadError,

    # SDXL Refiner
    SDXLRefinerUri,
    InvalidSDXLRefinerUriError,

    # ControlNet
    ControlNetUri,
    InvalidControlNetUriError,
    ControlNetUriLoadError,
    FluxControlNetUnionUriModes,
    SDXLControlNetUnionUriModes,

    # VAE
    VAEUri,
    InvalidVaeUriError,
    VAEUriLoadError,

    # UNet
    UNetUri,
    InvalidUNetUriError,
    UNetUriLoadError,

    # Transformer
    TransformerUri,
    TransformerUriLoadError,
    InvalidTransformerUriError,

    # Image Encoder
    ImageEncoderUri,
    InvalidImageEncoderUriError,
    ImageEncoderUriLoadError,

    # IP Adapter
    IPAdapterUri,
    InvalidIPAdapterUriError,
    IPAdapterUriLoadError,

    # T2I Adapter
    T2IAdapterUri,
    InvalidT2IAdapterUriError,
    T2IAdapterUriLoadError,

    # Text Encoder
    TextEncoderUri,
    InvalidTextEncoderUriError,
    TextEncoderUriLoadError,

    # SCascade Decoder
    SCascadeDecoderUri,
    InvalidSCascadeDecoderUriError,

    # Quantizer URIs
    BNBQuantizerUri,
    InvalidBNBQuantizerUriError,
    SDNQQuantizerUri,
    InvalidSDNQQuantizerUriError,

    # Adetailer
    AdetailerDetectorUri,
    InvalidAdetailerDetectorUriError,
    AdetailerDetectorUriLoadError,

    uri_hash_with_parser,
    uri_list_hash_with_parser,
    get_uri_accepted_args_schema,
    get_quantizer_uri_class,
    get_uri_help,
    get_uri_names,
    quantizer_help,
    UnknownQuantizerName
)

# Wrapper
from .wrapper import (
    PipelineWrapperResult,
    DiffusionPipelineWrapper,
    DiffusionArguments,
    DiffusionArgumentsHelpException
)

__all__ = _types.module_all()
