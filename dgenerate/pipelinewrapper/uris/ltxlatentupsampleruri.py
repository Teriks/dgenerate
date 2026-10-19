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

import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions

_ltx_latent_upsampler_uri_parser = _textprocessing.ConceptUriParser(
    'LTX Latent Upsampler', [
        'revision',
        'variant',
        'subfolder'
    ]
)


class LTXLatentUpsamplerUri:
    """
    Representation of ``--ltx-latent-upscale`` URI.
    """

    # pipelinewrapper.uris.util.get_uri_accepted_args_schema metadata

    NAMES = ['LTX Latent Upsampler']

    @staticmethod
    def help():
        import dgenerate.arguments as _a
        return _a.get_raw_help_text('--ltx-latent-upscale')

    FILE_ARGS = {
        'model': {'mode': ['in', 'dir'], 'filetypes': [('Models', ['*.safetensors', '*.pt', '*.pth', '*.bin'])]}
    }

    # ===

    @property
    def model(self) -> str:
        """
        Model path, huggingface slug, file path, or blob link
        """
        return self._model

    @property
    def revision(self) -> _types.OptionalString:
        """
        Model repo revision
        """
        return self._revision

    @property
    def variant(self) -> _types.OptionalString:
        """
        Model repo variant
        """
        return self._variant

    @property
    def subfolder(self) -> _types.OptionalPath:
        """
        Model repo subfolder
        """
        return self._subfolder

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 variant: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None):
        """
        :param model: model path
        :param revision: model revision (branch name)
        :param variant: model variant, for example ``fp16``
        :param subfolder: model subfolder
        """

        self._model = model
        self._revision = revision
        self._variant = variant
        self._subfolder = subfolder

    @staticmethod
    def parse(uri: _types.Uri) -> 'LTXLatentUpsamplerUri':
        """
        Parse an ``--ltx-latent-upscale`` uri and return an object representing its constituents

        :param uri: string with ``--ltx-latent-upscale`` uri syntax

        :raise InvalidLTXLatentUpsamplerUriError:

        :return: :py:class:`.LTXLatentUpsamplerUri`
        """
        try:
            r = _ltx_latent_upsampler_uri_parser.parse(uri)
            return LTXLatentUpsamplerUri(
                model=r.concept,
                revision=r.args.get('revision', None),
                variant=r.args.get('variant', None),
                subfolder=r.args.get('subfolder', None))
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidLTXLatentUpsamplerUriError(e) from e
