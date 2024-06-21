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

import os.path

import diffusers
import huggingface_hub

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.hfutil as _hfutil
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions

_lora_uri_parser = _textprocessing.ConceptUriParser('LoRA', ['scale', 'revision', 'subfolder', 'weight-name'])


class LoRAUri:
    """
    Representation of a ``--loras`` uri
    """

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
        LoRA scale
        """
        return self._scale

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None,
                 weight_name: _types.OptionalName = None,
                 scale: float = 1.0):
        self._model = model
        self._scale = scale
        self._revision = revision
        self._subfolder = subfolder
        self._weight_name = weight_name

    def __str__(self):
        return f'{self.__class__.__name__}({str(_types.get_public_attributes(self))})'

    def __repr__(self):
        return str(self)

    def load_on_pipeline(self,
                         pipeline: diffusers.DiffusionPipeline,
                         use_auth_token: _types.OptionalString = None,
                         local_files_only: bool = False):
        """
        Load LoRA weights on to a pipeline using this URI

        :param pipeline: :py:class:`diffusers.DiffusionPipeline`
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug

        :raises ModelNotFoundError: If the model could not be found.
        """
        try:
            self._load_on_pipeline(pipeline=pipeline,
                                   use_auth_token=use_auth_token,
                                   local_files_only=local_files_only)
        except (huggingface_hub.utils.HFValidationError,
                huggingface_hub.utils.HfHubHTTPError) as e:
            raise _hfutil.ModelNotFoundError(e)
        except Exception as e:
            raise
            raise _exceptions.LoRAUriLoadError(
                f'error loading lora "{self.model}": {e}')

    def _load_on_pipeline(self,
                          pipeline: diffusers.DiffusionPipeline,
                          use_auth_token: _types.OptionalString = None,
                          local_files_only: bool = False):

        if hasattr(pipeline, 'load_lora_weights'):
            debug_args = {k: v for k, v in locals().items() if k not in {'self', 'pipeline'}}
            _messages.debug_log('pipeline.load_lora_weights('
                                + str(_types.get_public_attributes(self) | debug_args) + ')')

            model_path = _hfutil.download_non_hf_model(self.model)

            if local_files_only and not os.path.exists(model_path):
                # Temporary fix for diffusers bug

                subfolder = self.subfolder if self.subfolder else ''

                probable_path_1 = os.path.join(
                    subfolder, 'pytorch_lora_weights.safetensors' if
                    self.weight_name is None else self.weight_name)

                probable_path_2 = os.path.join(
                    subfolder, 'pytorch_lora_weights.bin')

                file_path = huggingface_hub.try_to_load_from_cache(self.model,
                                                                   filename=probable_path_1,
                                                                   revision=self.revision)

                if not isinstance(file_path, str):
                    file_path = huggingface_hub.try_to_load_from_cache(self.model,
                                                                       filename=probable_path_2,
                                                                       revision=self.revision)

                if not isinstance(file_path, str):
                    raise RuntimeError(
                        f'LoRA model "{self.model}" '
                        'was not available in the local huggingface cache.')

                model_path = os.path.dirname(file_path)

            pipeline.load_lora_weights(model_path,
                                       revision=self.revision,
                                       subfolder=self.subfolder,
                                       weight_name=self.weight_name,
                                       local_files_only=local_files_only,
                                       use_safetensors=True,
                                       token=use_auth_token)

            if hasattr(pipeline, 'fuse_lora'):
                pipeline.fuse_lora(lora_scale=self.scale)
            elif self.scale != 1.0:
                _messages.log('lora scale argument not supported, ignored.',
                              level=_messages.WARNING)

            _messages.debug_log(f'Added LoRA: "{self}" to pipeline: "{pipeline.__class__.__name__}"')
        else:
            raise RuntimeError(f'Pipeline: {pipeline.__class__.__name__} '
                               f'does not support loading LoRAs.')

    @staticmethod
    def parse(uri: _types.Uri) -> 'LoRAUri':
        """
        Parse a ``--loras`` uri and return an object representing its constituents

        :param uri: string with ``--loras`` uri syntax

        :raise InvalidLoRAUriError:

        :return: :py:class:`.LoRAPath`
        """
        try:
            r = _lora_uri_parser.parse(uri)

            return LoRAUri(model=r.concept,
                           scale=float(r.args.get('scale', 1.0)),
                           weight_name=r.args.get('weight-name', None),
                           revision=r.args.get('revision', None),
                           subfolder=r.args.get('subfolder', None))
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidLoRAUriError(e)
