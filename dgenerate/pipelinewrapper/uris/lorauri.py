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
import typing

import diffusers
import huggingface_hub

import dgenerate.hfhub as _hfhub
import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions

_lora_uri_parser = _textprocessing.ConceptUriParser(
    'LoRA', ['scale', 'revision', 'subfolder', 'weight-name'])


class LoRAUri:
    """
    Representation of a ``--loras`` uri
    """

    # pipelinewrapper.uris.util.get_uri_accepted_args_schema metadata

    NAMES = ['LoRA']

    @staticmethod
    def help():
        import dgenerate.arguments as _a
        return _a.get_raw_help_text('--loras')

    FILE_ARGS = {
        'model': {'mode': ['in', 'dir'], 'filetypes': [('Models', ['*.safetensors', '*.pt', '*.pth', '*.cpkt', '*.bin'])]}
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

    @staticmethod
    def load_on_pipeline(pipeline: diffusers.DiffusionPipeline,
                         uris: typing.Iterable[typing.Union["LoRAUri", str]],
                         fuse_scale: float = 1.0,
                         use_auth_token: _types.OptionalString = None,
                         local_files_only: bool = False):
        """
        Load LoRA weights on to a pipeline using this URI


        :param pipeline: :py:class:`diffusers.DiffusionPipeline`
        :param uris: Iterable of :py:class:`LoRAUri` or ``str`` LoRA URIs to load
        :param fuse_scale: Global scale for the fused LoRAs, all LoRAs are fused together
            using their individual scale value, and then fused into the main model using this scale.
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug

        :raises dgenerate.ModelNotFoundError: If the model could not be found.
        :raises dgenerate.pipelinewrapper.uris.exceptions.InvalidLoRAUriError: On URI parsing errors.
        :raises dgenerate.pipelinewrapper.uris.exceptions.LoRAUriLoadError: On loading errors.
        """
        def cache_all(e):
            if isinstance(e, _exceptions.InvalidLoRAUriError):
                raise e
            else:
                raise _exceptions.LoRAUriLoadError(
                    f'error loading LoRAs: {e}') from e

        with _hfhub.with_hf_errors_as_model_not_found(cache_all):
            LoRAUri._load_on_pipeline(pipeline,
                                      uris=uris,
                                      fuse_scale=fuse_scale,
                                      use_auth_token=use_auth_token,
                                      local_files_only=local_files_only)

    @staticmethod
    def _load_on_pipeline(pipeline: diffusers.DiffusionPipeline,
                          uris: typing.Iterable[typing.Union["LoRAUri", str]],
                          fuse_scale: float = 1.0,
                          use_auth_token: _types.OptionalString = None,
                          local_files_only: bool = False):

        if hasattr(pipeline, 'load_lora_weights'):
            adapter_names = []
            adapter_weights = []
            for adapter_index, lora_uri in enumerate(uris):

                if not isinstance(lora_uri, LoRAUri):
                    lora_uri = LoRAUri.parse(lora_uri)

                model_path = _hfhub.download_non_hf_slug_model(lora_uri.model)

                weight_name = lora_uri.weight_name

                if local_files_only and not os.path.exists(model_path):

                    subfolder = lora_uri.subfolder if lora_uri.subfolder else ''

                    probable_path_1 = os.path.join(
                        subfolder, 'pytorch_lora_weights.safetensors' if
                        weight_name is None else weight_name)

                    probable_path_2 = os.path.join(
                        subfolder, 'pytorch_lora_weights.bin')

                    file_path = huggingface_hub.try_to_load_from_cache(lora_uri.model,
                                                                       filename=probable_path_1,
                                                                       revision=lora_uri.revision)

                    if not isinstance(file_path, str):
                        file_path = huggingface_hub.try_to_load_from_cache(lora_uri.model,
                                                                           filename=probable_path_2,
                                                                           revision=lora_uri.revision)
                        if not isinstance(file_path, str):
                            raise RuntimeError(
                                f'LoRA model "{lora_uri.model}" '
                                'was not available in the local huggingface cache.')
                        else:
                            weight_name = 'pytorch_lora_weights.bin'
                    else:
                        if weight_name is None:
                            weight_name = 'pytorch_lora_weights.safetensors'

                    model_path = os.path.dirname(file_path)

                adapter_name = str(adapter_index)
                adapter_names.append(adapter_name)
                adapter_weights.append(lora_uri.scale)

                try:
                    pipeline.load_lora_weights(model_path,
                                               revision=lora_uri.revision,
                                               subfolder=lora_uri.subfolder,
                                               weight_name=weight_name,
                                               local_files_only=local_files_only,
                                               use_safetensors=True,
                                               token=use_auth_token,
                                               adapter_name=adapter_name)
                except EnvironmentError:
                    # brute force, try for .bin files
                    pipeline.load_lora_weights(model_path,
                                               revision=lora_uri.revision,
                                               subfolder=lora_uri.subfolder,
                                               weight_name=weight_name,
                                               local_files_only=local_files_only,
                                               token=use_auth_token,
                                               adapter_name=adapter_name)

                if not pipeline.get_list_adapters():
                    raise RuntimeError(f'LoRA model "{lora_uri.model}" contained no usable weights.')

                _messages.debug_log(f'Added LoRA: "{lora_uri}" to pipeline: "{pipeline.__class__.__name__}"')

            _messages.debug_log(f'Fusing all LoRAs into pipeline with global scale: {fuse_scale}')
            pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
            pipeline.fuse_lora(adapter_names=adapter_names, lora_scale=fuse_scale)
        else:
            raise RuntimeError(f'Pipeline: {pipeline.__class__.__name__} '
                               f'does not support loading LoRAs.')

    @staticmethod
    def parse(uri: _types.Uri) -> 'LoRAUri':
        """
        Parse a ``--loras`` uri and return an object representing its constituents

        :param uri: string with ``--loras`` uri syntax

        :raise InvalidLoRAUriError:

        :return: :py:class:`.LoRAUri`
        """
        try:
            r = _lora_uri_parser.parse(uri)
            
            scale = r.args.get('scale', 1.0)
            try:
                scale = float(scale)
            except ValueError:
                raise _exceptions.InvalidLoRAUriError(
                    f'LoRA "scale" must be a floating point number, received: {scale}')

            return LoRAUri(model=r.concept,
                           scale=scale,
                           weight_name=r.args.get('weight-name', None),
                           revision=r.args.get('revision', None),
                           subfolder=r.args.get('subfolder', None))
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidLoRAUriError(e) from e
