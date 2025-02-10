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

import diffusers

import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions

_bnb_quantizer_uri_parser = _textprocessing.ConceptUriParser(
    'BNB Quantizer',
    ['bits', 'bits4_compute_dtype', 'bits4_quant_type', 'bits4_use_double_quant', 'bits4_quant_storage'])


class BNBQuantizerUri:
    """
    Representation of ``--quantizer`` uri when ``--model-type`` torch*
    """

    def __init__(self,
                 bits: int = 8,
                 bits4_compute_dtype: str = None,
                 bits4_quant_type: str = "fp4",
                 bits4_use_double_quant=False,
                 bits4_quant_storage: str = None):

        if bits not in {4, 8}:
            raise _exceptions.InvalidBNBQuantizerUriError(
                'BNB Quant Config bits must be 4 or 8.')

        if bits4_quant_type not in {'fp4', 'nf4'}:
            raise _exceptions.InvalidBNBQuantizerUriError(
                'BNB Quant Config bits must be fp4 or nf4.')

        self.bits4_quant_storage = self._dtype_check(bits4_quant_storage)
        self.bits4_compute_dtype = self._dtype_check(bits4_compute_dtype)
        self.bits = bits
        self.bits4_quant_type = bits4_quant_type
        self.bits4_use_double_quant = bits4_use_double_quant

    @staticmethod
    def _dtype_check(s):
        if s is None:
            return
        if s not in {"float16", "float32", "int8", "uint8", "float64", "bfloat16"}:
            raise _exceptions.InvalidBNBQuantizerUriError(
                'BNB Quant dtypes must be one of: float16, float32, int8, uint8, float64 or bfloat16.')
        return s

    def to_config(self) -> diffusers.BitsAndBytesConfig:
        return diffusers.BitsAndBytesConfig(
            load_in_4bit=self.bits == 4,
            load_in_8bit=self.bits == 8,
            bnb_4bit_use_double_quant=self.bits4_use_double_quant,
            bnb_4bit_quant_type=self.bits4_quant_type,
            bnb_4bit_quant_storage=self.bits4_quant_storage,
            bnb_4bit_compute_dtype=self.bits4_compute_dtype
        )

    @staticmethod
    def parse(uri: _types.Uri) -> 'BNBQuantizerUri':
        try:
            r = _bnb_quantizer_uri_parser.parse(uri)

            if r.concept not in {'bnb', 'bitsandbytes'}:
                raise _exceptions.InvalidBNBQuantizerUriError(
                    f'Unknown quantization backend: {r.concept}'
                )

            bits = int(r.args.get('bits', 8))
            bits4_compute_dtype = r.args.get('bits4_compute_dtype')
            bits4_quant_type = r.args.get('bits4_quant_type', 'fp4')
            bits4_use_double_quant = _types.parse_bool(r.args.get('bits4_use_double_quant', False))
            bits4_quant_storage = r.args.get('bits4_quant_storage')

            return BNBQuantizerUri(
                bits=bits,
                bits4_compute_dtype=bits4_compute_dtype,
                bits4_quant_type=bits4_quant_type,
                bits4_use_double_quant=bits4_use_double_quant,
                bits4_quant_storage=bits4_quant_storage
            )

        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidBNBQuantizerUriError(e)
