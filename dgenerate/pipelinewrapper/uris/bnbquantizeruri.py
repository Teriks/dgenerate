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
import torch

import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.pipelinewrapper.enums import get_torch_dtype as _get_torch_dtype
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions

_bnb_quantizer_uri_parser = _textprocessing.ConceptUriParser(
    'BNB Quantizer',
    ['bits', 'bits4-compute-dtype', 'bits4-quant-type', 'bits4-use-double-quant', 'bits4-quant-storage'])


class BNBQuantizerUri:
    """
    Representation of ``--quantizer`` URI.
    """

    _valid_dtypes = ["float16", "bfloat16", "float32", "float64", "int8", "uint8"]

    # pipelinewrapper.uris.util.get_uri_accepted_args_schema metadata

    NAMES = ['bnb', 'bitsandbytes']

    @staticmethod
    def help():
        return """
        Bitsandbytes quantization backend configuration.

        This backend can be specified as "bnb" or "bitsandbytes" in the URI.

        URI Format: bnb;argument1=value1;argument2=value2
        
        Example: bnb;bits=4;bits4-quant-type=nf4

        The argument "bits" is Quantization bit width. Must be 4 or 8.
        
        NOWRAP!
          - bits=8: Uses LLM.int8() quantization method
          - bits=4: Uses QLoRA 4-bit quantization method

        The argument "bits4-compute-dtype" is the compute data type for 4-bit quantization.
        Only applies when bits=4. When None, automatically determined. This should generally
        match the dtype that you loaded the model with.

        The argument "bits4-quant-type" is the quantization data type for 4-bit weights.
        Only applies when bits=4.
        
        NOWRAP!
          - "fp4": 4-bit floating point (default)
          - "nf4": Normal Float 4 data type, adapted for weights from normal distribution.

        The argument "bits4-use-double-quant" Enables nested quantization for 4-bit mode.
        Only applies when bits=4. When True, performs a second quantization of already
        quantized weights to save an additional 0.4 bits/parameter with no performance cost.

        The argument "bits4-quant-storage" is the storage data type for 4-bit quantized weights.
        Only applies when bits=4. When None, uses default storage format. Controls memory
        layout of quantized parameters.
        """

    OPTION_ARGS = {
        'bits': [8, 4],
        'bits4-compute-dtype': _valid_dtypes,
        'bits4-quant-type': ["fp4", "nf4"],
        'bits4-quant-storage': _valid_dtypes
    }

    # ===

    def __init__(self,
                 bits: int = 8,
                 bits4_compute_dtype: str | None = None,
                 bits4_quant_type: str = "fp4",
                 bits4_use_double_quant: bool = False,
                 bits4_quant_storage: str | None = None):

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
            return None
        if s not in BNBQuantizerUri._valid_dtypes:
            raise _exceptions.InvalidBNBQuantizerUriError(
                f'BNB Quant dtypes must be one of: '
                f'{_textprocessing.oxford_comma(BNBQuantizerUri._valid_dtypes, "or")}.')
        return s

    def to_config(self, compute_dtype: str | torch.dtype | None = None) -> diffusers.BitsAndBytesConfig:

        compute_dtype = _get_torch_dtype(compute_dtype)

        return diffusers.BitsAndBytesConfig(
            load_in_4bit=self.bits == 4,
            load_in_8bit=self.bits == 8,
            bnb_4bit_use_double_quant=self.bits4_use_double_quant,
            bnb_4bit_quant_type=self.bits4_quant_type,
            bnb_4bit_quant_storage=self.bits4_quant_storage,
            bnb_4bit_compute_dtype=_types.default(self.bits4_compute_dtype, compute_dtype)
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
            bits4_compute_dtype = r.args.get('bits4-compute-dtype', None)
            bits4_quant_type = r.args.get('bits4-quant-type', 'fp4')
            bits4_use_double_quant = _types.parse_bool(r.args.get('bits4-use-double-quant', False))
            bits4_quant_storage = r.args.get('bits4-quant-storage', None)

            return BNBQuantizerUri(
                bits=bits,
                bits4_compute_dtype=bits4_compute_dtype,
                bits4_quant_type=bits4_quant_type,
                bits4_use_double_quant=bits4_use_double_quant,
                bits4_quant_storage=bits4_quant_storage
            )

        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidBNBQuantizerUriError(e) from e
