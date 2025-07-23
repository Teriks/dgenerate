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

class ModelUriLoadError(Exception):
    """
    Thrown when model fails to load from a URI for a
    reason other than not being found, such as being
    unsupported.

    This exception refers to loadable sub models such as
    VAEs, LoRAs, ControlNets, Textual Inversions etc.
    """
    pass


class InvalidModelUriError(Exception):
    """
    Thrown on model path syntax or logical usage error
    """
    pass


class InvalidBNBQuantizerUriError(InvalidModelUriError):
    """
    Error in ``--quantizer`` uri
    """
    pass


class InvalidSDNQQuantizerUriError(InvalidModelUriError):
    """
    Error in ``--quantizer`` uri for SDNQ backend
    """
    pass


class InvalidSDXLRefinerUriError(InvalidModelUriError):
    """
    Error in ``--sdxl-refiner`` uri
    """
    pass


class InvalidSCascadeDecoderUriError(InvalidModelUriError):
    """
    Error in ``--s-cascade-decoder`` uri
    """
    pass


class InvalidVaeUriError(InvalidModelUriError):
    """
    Error in ``--vae`` uri
    """
    pass


class InvalidUNetUriError(InvalidModelUriError):
    """
    Error in ``--unet`` uri
    """
    pass


class InvalidControlNetUriError(InvalidModelUriError):
    """
    Error in ``--control-nets`` uri
    """
    pass


class InvalidT2IAdapterUriError(InvalidModelUriError):
    """
    Error in ``--t2i-adapters`` uri
    """
    pass


class InvalidLoRAUriError(InvalidModelUriError):
    """
    Error in ``--loras`` uri
    """
    pass


class InvalidIPAdapterUriError(InvalidModelUriError):
    """
    Error in ``--ip-adapters`` uri
    """
    pass


class InvalidTextualInversionUriError(InvalidModelUriError):
    """
    Error in ``--textual-inversions`` uri
    """
    pass


class InvalidTextEncoderUriError(InvalidModelUriError):
    """
    Error in ``--text-encoder*`` uri
    """
    pass


class InvalidImageEncoderUriError(InvalidModelUriError):
    """
    Error in ``--image-encoder`` uri
    """
    pass


class InvalidTransformerUriError(InvalidModelUriError):
    """
    Error in ``--transformer`` uri
    """
    pass


class TextEncoderUriLoadError(InvalidModelUriError):
    """
    Error loading ``--text-encoder*`` uri
    """
    pass


class InvalidAdetailerDetectorUriError(InvalidModelUriError):
    """
    Error in ``--adetailer-detectors`` uri
    """
    pass


class ControlNetUriLoadError(ModelUriLoadError):
    """
    Error while loading model file in ``--control-nets`` uri
    """
    pass


class T2IAdapterUriLoadError(ModelUriLoadError):
    """
    Error while loading model file in ``--t2i-adapters`` uri
    """
    pass


class VAEUriLoadError(ModelUriLoadError):
    """
    Error while loading model file in ``--vae`` uri
    """
    pass


class UNetUriLoadError(ModelUriLoadError):
    """
    Error while loading model file in ``--unet / --second-model-unet`` uri
    """
    pass


class LoRAUriLoadError(ModelUriLoadError):
    """
    Error while loading model file in ``--loras`` uri
    """
    pass


class IPAdapterUriLoadError(ModelUriLoadError):
    """
    Error while loading model file in ``--ip-adapters`` uri
    """
    pass


class TextualInversionUriLoadError(ModelUriLoadError):
    """
    Error while loading model file in ``--textual-inversions`` uri
    """
    pass


class ImageEncoderUriLoadError(ModelUriLoadError):
    """
    Error while loading model file in ``--image-encoder`` uri
    """
    pass


class TransformerUriLoadError(ModelUriLoadError):
    """
    Error while loading model file in ``--transformer`` uri
    """
    pass


class AdetailerDetectorUriLoadError(ModelUriLoadError):
    """
    Error while loading model file in ``--adetailer-detectors`` uri
    """
    pass
