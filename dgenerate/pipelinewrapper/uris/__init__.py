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
from .adetailerdetectoruri import AdetailerDetectorUri
from .bnbquantizeruri import BNBQuantizerUri
from .sdnqquantizeruri import SDNQQuantizerUri
from .controlneturi import (
    ControlNetUri,
    FluxControlNetUnionUriModes,
    SDXLControlNetUnionUriModes
)
from .exceptions import (
    # Model-related errors
    ModelUriLoadError,
    InvalidModelUriError,

    # ControlNet
    ControlNetUriLoadError,
    InvalidControlNetUriError,

    # Textual Inversion
    TextualInversionUriLoadError,
    InvalidTextualInversionUriError,

    # SDXL Refiner
    InvalidSDXLRefinerUriError,

    # SCascade Decoder
    InvalidSCascadeDecoderUriError,

    # VAE
    VAEUriLoadError,
    InvalidVaeUriError,

    # LoRA
    LoRAUriLoadError,
    InvalidLoRAUriError,

    # UNet
    UNetUriLoadError,
    InvalidUNetUriError,

    # Text Encoder
    TextEncoderUriLoadError,
    InvalidTextEncoderUriError,

    # T2I Adapter
    T2IAdapterUriLoadError,
    InvalidT2IAdapterUriError,

    # Image Encoder
    ImageEncoderUriLoadError,
    InvalidImageEncoderUriError,

    # IP Adapter
    IPAdapterUriLoadError,
    InvalidIPAdapterUriError,

    # Transformer
    TransformerUriLoadError,
    InvalidTransformerUriError,

    # BNB Quantizer
    InvalidBNBQuantizerUriError,

    # SDNQ Quantizer
    InvalidSDNQQuantizerUriError,

    # Adetailer Detector
    AdetailerDetectorUriLoadError,
    InvalidAdetailerDetectorUriError
)
from .imageencoderuri import ImageEncoderUri
from .ipadapteruri import IPAdapterUri
from .lorauri import LoRAUri
from .scascadedecoderuri import SCascadeDecoderUri
from .sdxlrefineruri import SDXLRefinerUri
from .t2iadapteruri import T2IAdapterUri
from .textencoderuri import TextEncoderUri
from .textualinversionuri import TextualInversionUri
from .transformeruri import TransformerUri
from .uneturi import UNetUri
from .util import (
    uri_hash_with_parser,
    uri_list_hash_with_parser,
    get_uri_accepted_args_schema,
    get_quantizer_uri_class,
    get_uri_help,
    get_uri_names,
    quantizer_help,
    UnknownQuantizerName
)
from .vaeuri import VAEUri

__all__ = _types.module_all()
