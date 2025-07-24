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
import typing

import diffusers

import dgenerate.hfhub as _hfhub
import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions

_ip_adapter_uri_parser = _textprocessing.ConceptUriParser(
    'IP Adapter', ['scale', 'revision', 'subfolder', 'weight-name'])


class IPAdapterUri:
    """
    Representation of a ``--ip-adapters`` uri
    """

    # pipelinewrapper.uris.util.get_uri_accepted_args_schema metadata

    NAMES = ['IP Adapter']

    @staticmethod
    def help():
        import dgenerate.arguments as _a
        return _a.get_raw_help_text('--ip-adapters')

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
        IP Adapter scale
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
                         uris: typing.Iterable[typing.Union["IPAdapterUri", str]],
                         use_auth_token: _types.OptionalString = None,
                         local_files_only: bool = False):
        """
        Load IP Adapter weights on to a pipeline using this URI

        :param pipeline: :py:class:`diffusers.DiffusionPipeline`
        :param uris: IP Adapter URIs to load on to the pipeline
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug

        :raises ModelNotFoundError: If the model could not be found.
        :raises dgenerate.pipelinewrapper.uris.exceptions.InvalidIPAdapterUriError: On URI parsing errors.
        :raises dgenerate.pipelinewrapper.uris.exceptions.IPAdapterUriLoadError: On loading errors.
        """

        def cache_all(e):
            if isinstance(e, _exceptions.InvalidIPAdapterUriError):
                raise e
            else:
                if "NoneType" in str(e):
                    # Rectify highly useless error caused by bug in diffusers
                    raise _exceptions.IPAdapterUriLoadError(
                        'Cannot find IP Adapter weights in repository, '
                        'you may need to specify a "subfolder" and/or "weight-name" URI value.') from e
            raise _exceptions.IPAdapterUriLoadError(
                f'error loading IP Adapter: {e}') from e

        with _hfhub.with_hf_errors_as_model_not_found(cache_all):
            IPAdapterUri._load_on_pipeline(uris=uris,
                                           pipeline=pipeline,
                                           use_auth_token=use_auth_token,
                                           local_files_only=local_files_only)

    @staticmethod
    def _load_on_pipeline(pipeline: diffusers.DiffusionPipeline,
                          uris: typing.Iterable[typing.Union["IPAdapterUri", str]],
                          use_auth_token: _types.OptionalString = None,
                          local_files_only: bool = False):

        if hasattr(pipeline, 'load_ip_adapter'):

            models = []
            subfolders = []
            weight_names = []
            revisions = set()

            if pipeline.__class__.__name__.startswith('StableDiffusion3'):
                uris = list(uris)
                if len(uris) > 1:
                    raise ValueError(
                        'Stable Diffusion 3 does not support loading'
                        'multiple IP Adapters, you may only specify one IP Adapter model.'
                    )
                if isinstance(uris[0], str):
                    ip_adapter_uri = IPAdapterUri.parse(uris[0])
                else:
                    ip_adapter_uri = uris[0]

                models = ip_adapter_uri.model
                subfolders = ip_adapter_uri.subfolder
                weight_names = ip_adapter_uri.weight_name
                revision = ip_adapter_uri.revision
            else:
                for ip_adapter_uri in uris:
                    if isinstance(ip_adapter_uri, str):
                        ip_adapter_uri = IPAdapterUri.parse(ip_adapter_uri)

                    models.append(_hfhub.download_non_hf_slug_model(ip_adapter_uri.model))
                    if ip_adapter_uri.subfolder is None:
                        subfolders.append('.')
                    else:
                        subfolders.append(ip_adapter_uri.subfolder)
                    weight_names.append(ip_adapter_uri.weight_name)
                    revisions.add(ip_adapter_uri.revision)

                if len(revisions) > 1:
                    raise ValueError(
                        f'All IP Adapter URIs must have matching "revision" URI argument values.')

                revision = revisions.pop()

            try:
                pipeline.load_ip_adapter(models,
                                         subfolder=subfolders,
                                         revision=revision,
                                         weight_name=weight_names,
                                         local_files_only=local_files_only,
                                         use_safetensors=True,
                                         token=use_auth_token)
            except EnvironmentError:
                # brute force, try for .bin files
                pipeline.load_ip_adapter(models,
                                         subfolder=subfolders,
                                         revision=revision,
                                         weight_name=weight_names,
                                         local_files_only=local_files_only,
                                         token=use_auth_token)

            _messages.debug_log(f'Added IP Adapters {list(uris)} to pipeline: "{pipeline.__class__.__name__}"')
        else:
            raise RuntimeError(f'Pipeline: {pipeline.__class__.__name__} '
                               f'does not support loading IP Adapters.')

    @staticmethod
    def parse(uri: _types.Uri) -> 'IPAdapterUri':
        """
        Parse a ``--ip-adapters`` uri and return an object representing its constituents

        :param uri: string with ``--ip-adapters`` uri syntax

        :raise InvalidIPAdapterUriError:

        :return: :py:class:`.IPAdapterUri`
        """
        try:
            r = _ip_adapter_uri_parser.parse(uri)

            scale = r.args.get('scale', 1.0)
            try:
                scale = float(scale)
            except ValueError:
                raise _exceptions.InvalidIPAdapterUriError(
                    f'IP Adapter "scale" must be a floating point number, received: {scale}')

            return IPAdapterUri(model=r.concept,
                                scale=scale,
                                weight_name=r.args.get('weight-name', None),
                                revision=r.args.get('revision', None),
                                subfolder=r.args.get('subfolder', None))
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidIPAdapterUriError(e) from e
