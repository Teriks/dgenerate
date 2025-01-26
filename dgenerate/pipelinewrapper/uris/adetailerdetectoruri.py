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
import dgenerate.pipelinewrapper.hfutil as _hfutil
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions
import huggingface_hub
import dgenerate.pipelinewrapper.util as _util

_lora_uri_parser = _textprocessing.ConceptUriParser(
    'Adetailer Detector', ['revision', 'subfolder', 'weight-name', 'device'])


class AdetailerDetectorUri:
    """
    Representation of a ``--adetailer-detectors`` uri
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
    def device(self) -> _types.OptionalName:
        """
        Model device override
        """
        return self._device

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None,
                 weight_name: _types.OptionalName = None,
                 device: _types.OptionalName = None):
        self._model = model
        self._revision = revision
        self._subfolder = subfolder
        self._weight_name = weight_name
        self._device = device

        if self._device is not None:
            if _util.is_valid_device_string(device):
                self._device = device
            else:
                self._device = None
                raise _exceptions.InvalidAdetailerDetectorUriError(
                    f'invalid adetailer detector device specification: {device}')

    def __str__(self):
        return f'{self.__class__.__name__}({str(_types.get_public_attributes(self))})'

    def __repr__(self):
        return str(self)

    def get_model_path(self,
                       local_files_only: bool = False,
                       use_auth_token: _types.OptionalString = None):
        try:
            if not self.model.startswith('http://') or self.model.startswith('https://'):
                _, ext = os.path.splitext(self.model)
            else:
                ext = ''

            if _hfutil.is_single_file_model_load(self.model) or ext in {'.yaml', '.yml'}:
                if os.path.exists(self.model):
                    return self.model
                else:
                    if local_files_only:
                        raise _exceptions.AdetailerDetectorUriLoadError(f'Could not find adetailer model: {self.model}')
                    return _hfutil.download_non_hf_model(self.model)
            else:
                return huggingface_hub.hf_hub_download(
                    self.model,
                    filename=self.weight_name,
                    subfolder=self.subfolder,
                    token=use_auth_token,
                    revision=self.revision,
                    local_files_only=local_files_only)
        except Exception as e:
            raise _exceptions.AdetailerDetectorUriLoadError(
                f'Error loading adetailer model: {e}')

    @staticmethod
    def parse(uri: _types.Uri) -> 'AdetailerDetectorUri':
        """
        Parse a ``--adetailer-detectors`` uri and return an object representing its constituents

        :param uri: string with ``--adetailer-detectors`` uri syntax

        :raise InvalidAdetailerDetectorUriError:

        :return: :py:class:`.AdetailerDetectorUri`
        """
        try:
            r = _lora_uri_parser.parse(uri)

            return AdetailerDetectorUri(model=r.concept,
                                        weight_name=r.args.get('weight-name', None),
                                        revision=r.args.get('revision', None),
                                        subfolder=r.args.get('subfolder', None),
                                        device=r.args.get('device', None))
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidAdetailerDetectorUriError(e)
