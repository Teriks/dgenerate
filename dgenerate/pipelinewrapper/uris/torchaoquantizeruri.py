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
    'TorchAO Quantizer', ['type'])


class TorchAOQuantizerUri:
    """
    Representation of ``--quantizer`` uri when ``--model-type`` torch*
    """

    def __init__(self,
                 type: str = 'int8wo'):

        types = sorted(diffusers.TorchAoConfig._get_torchao_quant_type_to_method().keys())

        if type not in types:
            raise _exceptions.InvalidTorchAOQuantizerUriError(
                f'TorchAO quantizer type must be one of: {_textprocessing.oxford_comma(types, "or")}'
            )

        self.type = type

    def to_config(self) -> diffusers.TorchAoConfig:
        return diffusers.TorchAoConfig(
            quant_type=self.type
        )

    @staticmethod
    def parse(uri: _types.Uri) -> 'TorchAOQuantizerUri':
        try:
            r = _bnb_quantizer_uri_parser.parse(uri)

            if r.concept != 'torchao':
                raise _exceptions.InvalidTorchAOQuantizerUriError(
                    f'Unknown quantization backend: {r.concept}'
                )

            return TorchAOQuantizerUri(type=r.args.get('type', 'int8wo'))

        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidTorchAOQuantizerUriError(e) from e
