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
    ['type', 'group-size', 'quant-conv', 'use-quantized-matmul',
     'use-quantized-matmul-conv', 'quantization-device',
     'return-device'])


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

    @staticmethod
    def help():
        import dgenerate.arguments as _a
        return _a.get_raw_help_text('--quantizer')

    OPTION_ARGS = {
        'type': _valid_weight_dtypes
    }

    # ===

    def __init__(self,
                 type: str = "int8",
                 group_size: int = 0,
                 quant_conv: bool = False,
                 use_quantized_matmul: bool = False,
                 use_quantized_matmul_conv: bool = False):

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
        self.use_quantized_matmul = use_quantized_matmul
        self.use_quantized_matmul_conv = use_quantized_matmul_conv

    def to_config(self, compute_dtype: str | torch.dtype | None = None) -> SDNQConfig:
        return SDNQConfig(
            weights_dtype=self.type,
            group_size=self.group_size,
            quant_conv=self.quant_conv,
            use_quantized_matmul=self.use_quantized_matmul,
            use_quantized_matmul_conv=self.use_quantized_matmul_conv
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
            use_quantized_matmul = _types.parse_bool(r.args.get('use-quantized-matmul', False))
            use_quantized_matmul_conv = _types.parse_bool(r.args.get('use-quantized-matmul-conv', False))

            return SDNQQuantizerUri(
                type=weights_dtype,
                group_size=group_size,
                quant_conv=quant_conv,
                use_quantized_matmul=use_quantized_matmul,
                use_quantized_matmul_conv=use_quantized_matmul_conv
            )

        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidSDNQQuantizerUriError(e) from e 