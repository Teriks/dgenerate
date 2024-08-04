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
import huggingface_hub

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.hfutil as _hfutil
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions

_ip_adapter_uri_parser = _textprocessing.ConceptUriParser('IP Adapter',
                                                          ['scale', 'revision', 'subfolder', 'weight-name'])


class IPAdapterUri:
    """
    Representation of a ``--ip-adapters`` uri
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
    def load_on_pipeline(ip_adapter_uris: typing.Iterable["IPAdapterUri"] | typing.Iterable[str],
                         pipeline: diffusers.DiffusionPipeline,
                         use_auth_token: _types.OptionalString = None,
                         local_files_only: bool = False):
        """
        Load IP Adapter weights on to a pipeline using this URI

        :param ip_adapter_uris: IP Adapter URIs to load on to the pipeline
        :param pipeline: :py:class:`diffusers.DiffusionPipeline`
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug

        :raises ModelNotFoundError: If the model could not be found.
        """
        try:
            IPAdapterUri._load_on_pipeline(ip_adapter_uris=ip_adapter_uris,
                                           pipeline=pipeline,
                                           use_auth_token=use_auth_token,
                                           local_files_only=local_files_only)
        except (huggingface_hub.utils.HFValidationError,
                huggingface_hub.utils.HfHubHTTPError) as e:
            raise _hfutil.ModelNotFoundError(e)
        except Exception as e:
            raise _exceptions.IPAdapterUriLoadError(
                f'error loading IP Adapter: {e}')

    @staticmethod
    def _load_on_pipeline(ip_adapter_uris: typing.Iterable["IPAdapterUri"] | typing.Iterable[str],
                          pipeline: diffusers.DiffusionPipeline,
                          use_auth_token: _types.OptionalString = None,
                          local_files_only: bool = False):

        models = []
        subfolders = []
        weight_names = []
        revisions = set()

        for ip_adapter in ip_adapter_uris:
            if isinstance(ip_adapter, str):
                ip_adapter = IPAdapterUri.parse(ip_adapter)

            models.append(_hfutil.download_non_hf_model(ip_adapter.model))
            subfolders.append(ip_adapter.subfolder)
            weight_names.append(ip_adapter.weight_name)
            revisions.add(ip_adapter.revision)

        if len(revisions) > 1:
            raise _exceptions.IPAdapterUriLoadError(
                f'All IP Adapter URIs must have matching "revision" URI argument values.')

        if hasattr(pipeline, 'load_ip_adapter'):
            try:
                pipeline.load_ip_adapter(models,
                                         subfolder=subfolders,
                                         revision=revisions.pop(),
                                         weight_name=weight_names,
                                         local_files_only=local_files_only,
                                         use_safetensors=True,
                                         token=use_auth_token)
            except EnvironmentError:
                # brute force, try for .bin files
                pipeline.load_ip_adapter(models,
                                         subfolder=subfolders,
                                         revision=revisions.pop(),
                                         weight_name=weight_names,
                                         local_files_only=local_files_only,
                                         token=use_auth_token)

            _messages.debug_log(f'Added IP Adapters to pipeline: "{pipeline.__class__.__name__}"')
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

            return IPAdapterUri(model=r.concept,
                                scale=float(r.args.get('scale', 1.0)),
                                weight_name=r.args.get('weight-name', None),
                                revision=r.args.get('revision', None),
                                subfolder=r.args.get('subfolder', None))
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidIPAdapterUriError(e)
