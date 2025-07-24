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

import torch

import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.extras.sdnq import SDNQConfig
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions

_sdnq_quantizer_uri_parser = _textprocessing.ConceptUriParser(
    'SDNQ Quantizer',
    [
        'type',
        'group-size',
        'quant-conv',
        'quantized-matmul',
        'quantized-matmul-conv'
    ])


class SDNQQuantizerUri:
    """
    Representation of ``--quantizer`` URI for SDNQ backend.
    """

    _valid_weight_dtypes = [
        "int8", "int7", "int6", "int5", "int4", "int3", "int2",
        "uint8", "uint7", "uint6", "uint5", "uint4", "uint3", "uint2", "uint1", "bool",
        "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"
    ]

    # pipelinewrapper.uris.util.get_uri_accepted_args_schema metadata

    NAMES = ['sdnq']

    @staticmethod
    def help():
        return """
        SD.Next quantization backend configuration.
        
        This backend can be specified as "sdnq" in the URI.
        
        URI Format: sdnq;argument1=value1;argument2=value2
        
        Example: sdnq;type=int4;group-size=8;quant-conv=true
        
        The argument "type" is the target data type for weights after quantization.
        
        NOWRAP!
        Integer types: 
          - int8 (default), 
          - int7 
          - int6 
          - int5 
          - int4 
          - int3 
          - int2
            
        NOWRAP! 
        Unsigned integer types: 
          - uint8
          - uint7
          - uint6
          - uint5
          - uint4
          - uint3
          - uint2
          - uint1
          - bool
            
        NOWRAP!
        Floating point types: 
          - float8_e4m3fn
          - float8_e4m3fnuz
          - float8_e5m2
          - float8_e5m2fnuz
        
        The argument "group-size" is used to decide how many elements of a tensor 
        will share the same quantization group. Must be >= 0. When 0 (default), uses per-tensor 
        quantization. When > 0, groups tensor elements for more granular quantization scaling.
        
        The argument "quant-conv" is enables quantization of convolutional layers in UNet models.
        When True, quantizes Conv2d layers in addition to Linear layers. Only affects UNet architectures.
        
        The argument "quantized-matmul" is enables use of quantized INT8 or FP8 matrix multiplication 
        instead of BF16/FP16. When True, uses optimized quantized matmul operations for improved 
        performance and reduced memory usage.
        
        The argument "quantized-matmul-conv" is enables quantized matrix multiplication for 
        convolutional layers. Same as quantized-matmul but specifically for convolutional 
        layers in UNets like SDXL.
        """

    OPTION_ARGS = {
        'type': _valid_weight_dtypes
    }

    # ===

    def __init__(self,
                 type: str = "int8",
                 group_size: int = 0,
                 quant_conv: bool = False,
                 quantized_matmul: bool = False,
                 quantized_matmul_conv: bool = False):

        if type not in self._valid_weight_dtypes:
            raise _exceptions.InvalidSDNQQuantizerUriError(
                f'SDNQ type must be one of: '
                f'{_textprocessing.oxford_comma(self._valid_weight_dtypes, "or")}.')

        if group_size < 0:
            raise _exceptions.InvalidSDNQQuantizerUriError(
                'SDNQ group-size must be >= 0.')

        self.type = type
        self.group_size = group_size
        self.quant_conv = quant_conv
        self.quantized_matmul = quantized_matmul
        self.quantized_matmul_conv = quantized_matmul_conv

    def to_config(self, compute_dtype: str | torch.dtype | None = None) -> SDNQConfig:
        return SDNQConfig(
            weights_dtype=self.type,
            group_size=self.group_size,
            quant_conv=self.quant_conv,
            use_quantized_matmul=self.quantized_matmul,
            use_quantized_matmul_conv=self.quantized_matmul_conv
        )

    @staticmethod
    def parse(uri: _types.Uri) -> 'SDNQQuantizerUri':
        try:
            r = _sdnq_quantizer_uri_parser.parse(uri)

            if r.concept not in {'sdnq'}:
                raise _exceptions.InvalidSDNQQuantizerUriError(
                    f'Unknown quantization backend: {r.concept}'
                )

            weights_dtype = r.args.get('type', 'int8')
            group_size = int(r.args.get('group-size', 0))
            quant_conv = _types.parse_bool(r.args.get('quant-conv', False))
            quantized_matmul = _types.parse_bool(r.args.get('quantized-matmul', False))
            quantized_matmul_conv = _types.parse_bool(r.args.get('quantized-matmul-conv', False))

            return SDNQQuantizerUri(
                type=weights_dtype,
                group_size=group_size,
                quant_conv=quant_conv,
                quantized_matmul=quantized_matmul,
                quantized_matmul_conv=quantized_matmul_conv
            )

        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidSDNQQuantizerUriError(e) from e 