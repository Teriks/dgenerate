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
from dgenerate.pipelinewrapper.uris import lorauri as _lorauri

_ic_lora_uri_parser = _textprocessing.ConceptUriParser(
    'IC-LoRA', ['scale', 'attention', 'downscale', 'revision', 'subfolder', 'weight-name'])


class ICLoRAUri:
    """
    Representation of an ``--ic-lora`` uri
    """

    # pipelinewrapper.uris.util.get_uri_accepted_args_schema metadata

    NAMES = ['IC-LoRA']

    @staticmethod
    def help():
        import dgenerate.arguments as _a
        return _a.get_raw_help_text('--ic-lora')

    FILE_ARGS = {
        'model': {'mode': ['in', 'dir'], 'filetypes': [('Models', ['*.safetensors'])]}
    }

    # ===

    @property
    def model(self) -> str:
        """
        Model path, huggingface slug, file path
        """
        return self._model

    @property
    def revision(self) -> _types.OptionalString:
        """
        Model repo revision
        """
        return self._revision

    @property
    def subfolder(self) -> _types.OptionalPath:
        """
        Model repo subfolder
        """
        return self._subfolder

    @property
    def weight_name(self) -> _types.OptionalName:
        """
        Model weight-name
        """
        return self._weight_name

    @property
    def scale(self) -> float:
        """
        LoRA weight scale
        """
        return self._scale

    @property
    def attention(self) -> float:
        """
        Attention strength between the generated video and the reference clip, from 0 to 1
        """
        return self._attention

    @property
    def downscale(self) -> _types.OptionalInteger:
        """
        Reference downscale factor, ``None`` to read it from the LoRA file metadata
        """
        return self._downscale

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None,
                 weight_name: _types.OptionalName = None,
                 scale: float = 1.0,
                 attention: float = 1.0,
                 downscale: _types.OptionalInteger = None):
        self._model = model
        self._revision = revision
        self._subfolder = subfolder
        self._weight_name = weight_name
        self._scale = scale
        self._attention = attention
        self._downscale = downscale

    def lora_uri(self) -> str:
        """
        The ``--loras`` URI that loads these weights.
        """
        parts = [self.model, f'scale={self.scale}']
        if self.revision:
            parts.append(f'revision={self.revision}')
        if self.subfolder:
            parts.append(f'subfolder={self.subfolder}')
        if self.weight_name:
            parts.append(f'weight-name={self.weight_name}')
        return ';'.join(parts)

    def parsed_lora_uri(self) -> _lorauri.LoRAUri:
        """
        The :py:class:`.LoRAUri` that loads these weights.
        """
        return _lorauri.LoRAUri(
            model=self.model,
            revision=self.revision,
            subfolder=self.subfolder,
            weight_name=self.weight_name,
            scale=self.scale)

    def __str__(self):
        return f'{self.__class__.__name__}({str(_types.get_public_attributes(self))})'

    def __repr__(self):
        return str(self)

    @staticmethod
    def parse(uri: _types.Uri) -> 'ICLoRAUri':
        """
        Parse an ``--ic-lora`` uri and return an object representing its constituents

        :param uri: string with ``--ic-lora`` uri syntax

        :raise InvalidLoRAUriError:

        :return: :py:class:`.ICLoRAUri`
        """
        try:
            r = _ic_lora_uri_parser.parse(uri)
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidLoRAUriError(e) from e

        def number(name, default, cast):
            value = r.args.get(name, None)
            if value is None:
                return default
            try:
                return cast(value)
            except ValueError:
                raise _exceptions.InvalidLoRAUriError(
                    f'IC-LoRA "{name}" must be a number, received: {value}')

        attention = number('attention', 1.0, float)
        if not 0.0 <= attention <= 1.0:
            raise _exceptions.InvalidLoRAUriError(
                f'IC-LoRA "attention" must be between 0 and 1, received: {attention}')

        downscale = number('downscale', None, int)
        if downscale is not None and downscale < 1:
            raise _exceptions.InvalidLoRAUriError(
                f'IC-LoRA "downscale" must be 1 or greater, received: {downscale}')

        return ICLoRAUri(model=r.concept,
                         scale=number('scale', 1.0, float),
                         attention=attention,
                         downscale=downscale,
                         weight_name=r.args.get('weight-name', None),
                         revision=r.args.get('revision', None),
                         subfolder=r.args.get('subfolder', None))
