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
#     from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions


_ltx_diffusion_decoder_uri_parser = _textprocessing.ConceptUriParser(
    'LTX Diffusion Decoder', ['revision', 'subfolder', 'dtype'])


class LTXDiffusionDecoderUri:
    """Representation of an ``--ltx-diffusion-decoder`` URI."""

    # pipelinewrapper.uris.util.get_uri_accepted_args_schema metadata

    NAMES = ['LTX Diffusion Decoder']

    OPTION_ARGS = {
        'dtype': ['float16', 'bfloat16', 'float32']
    }

    @staticmethod
    def help():
        import dgenerate.arguments as _a
        return _a.get_raw_help_text('--ltx-diffusion-decoder')

    FILE_ARGS = {
        'model': {'mode': 'dir'}
    }

    # ===

    @property
    def model(self) -> str:
        """Model path or Hugging Face repository slug."""
        return self._model

    @property
    def revision(self) -> _types.OptionalString:
        """Model repository revision."""
        return self._revision

    @property
    def subfolder(self) -> _types.OptionalPath:
        """Model repository subfolder."""
        return self._subfolder

    @property
    def dtype(self) -> _enums.DataType | None:
        """Model data type."""
        return self._dtype

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None,
                 dtype: _enums.DataType | str | None = None):
        self._model = model
        self._revision = revision
        self._subfolder = subfolder
        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            self._dtype = None
        if self._dtype is _enums.DataType.AUTO or (
                self._dtype is None and dtype is not None):
            raise _exceptions.InvalidLTXDiffusionDecoderUriError(
                'dtype must be float16, bfloat16, or float32.')

    @staticmethod
    def parse(uri: _types.Uri) -> 'LTXDiffusionDecoderUri':
        """Parse an ``--ltx-diffusion-decoder`` URI."""
        try:
            parsed = _ltx_diffusion_decoder_uri_parser.parse(uri)
            return LTXDiffusionDecoderUri(
                model=parsed.concept,
                revision=parsed.args.get('revision'),
                subfolder=parsed.args.get('subfolder'),
                dtype=parsed.args.get('dtype'))
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidLTXDiffusionDecoderUriError(e) from e
