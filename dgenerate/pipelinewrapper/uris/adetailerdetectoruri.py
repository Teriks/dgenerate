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

import ast
import os.path
import typing

import huggingface_hub

import dgenerate.pipelinewrapper.util as _util
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions

_lora_uri_parser = _textprocessing.ConceptUriParser(
    'Adetailer Detector', [
        'revision',
        'subfolder',
        'weight-name',
        'confidence',
        'index-filter',
        'detector-padding',
        'mask-shape',
        'mask-padding',
        'mask-blur',
        'mask-dilation',
        'prompt',
        'negative-prompt',
        'device'
    ], args_raw=['index-filter']
)


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

    @property
    def confidence(self) -> float:
        """
        Confidence value for YOLO detector model.
        """
        return self._confidence

    @property
    def mask_padding(self) -> typing.Optional[int | tuple[int, int] | tuple[int, int, int, int]]:
        """
        Optional mask padding

        Option 1: Uniform padding
        Option 2: (Left/Right, Top/Bottom)
        Option 3: (Left, Top, Right, Bottom)
        """
        return self._mask_padding

    @property
    def detector_padding(self) -> typing.Optional[int | tuple[int, int] | tuple[int, int, int, int]]:
        """
        Optional detector padding

        Option 1: Uniform padding
        Option 2: (Left/Right, Top/Bottom)
        Option 3: (Left, Top, Right, Bottom)
        """
        return self._detector_padding

    @property
    def mask_shape(self) -> _types.OptionalName:
        """
        Optional mask shape override.
        """
        return self._mask_shape

    @property
    def mask_blur(self) -> _types.OptionalInteger:
        """
        Optional mask blur override.
        """
        return self._mask_blur

    @property
    def mask_dilation(self) -> _types.OptionalInteger:
        """
        Optional mask dilation override.
        """
        return self._mask_dilation

    @property
    def index_filter(self) -> _types.OptionalIntegers:
        """
        Process these YOLO detection indices.
        """
        return self._index_filter

    @property
    def index_filter(self) -> _types.OptionalIntegers:
        """
        Process these YOLO detection indices.
        """
        return self._index_filter

    @property
    def prompt(self) -> _types.OptionalString:
        """
        Positive prompt override.
        """
        return self._prompt

    @property
    def negative_prompt(self) -> _types.OptionalString:
        """
        Negative prompt override.
        """
        return self._negative_prompt

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None,
                 weight_name: _types.OptionalName = None,
                 confidence: float = 0.3,
                 detector_padding: typing.Optional[int | tuple[int, int] | tuple[int, int, int, int]] = None,
                 mask_shape: _types.OptionalName = None,
                 mask_padding: typing.Optional[int | tuple[int, int] | tuple[int, int, int, int]] = None,
                 mask_blur: _types.OptionalInteger = None,
                 mask_dilation: _types.OptionalInteger = None,
                 index_filter: _types.OptionalIntegers = None,
                 prompt: _types.OptionalString = None,
                 negative_prompt: _types.OptionalString = None,
                 device: _types.OptionalName = None):

        self._model = model
        self._revision = revision
        self._subfolder = subfolder
        self._weight_name = weight_name
        self._device = device
        self._mask_blur = mask_blur
        self._mask_dilation = mask_dilation
        self._mask_padding = mask_padding
        self._detector_padding = detector_padding
        self._prompt = prompt
        self._negative_prompt = negative_prompt

        if mask_shape is not None:
            mask_shape = mask_shape.lower()
            if mask_shape not in {'rectangle', 'circle'}:
                raise _exceptions.InvalidAdetailerDetectorUriError(
                    'adetailer detector mask-shape must be one of: rectangle or circle.'
                )

        self._mask_shape = mask_shape

        if mask_blur is not None and mask_blur < 0:
            raise _exceptions.InvalidAdetailerDetectorUriError(
                'adetailer detector mask-blur value must be greater than 0.'
            )

        self._mask_blur = mask_blur

        if mask_dilation is not None and mask_dilation < 0:
            raise _exceptions.InvalidAdetailerDetectorUriError(
                'adetailer detector mask-dilation value must be greater than 0.'
            )

        self._mask_dilation = mask_dilation

        if index_filter:
            if any(i < 0 for i in index_filter):
                raise _exceptions.InvalidAdetailerDetectorUriError(
                    'adetailer detector index-filter values must be greater than 0.'
                )

        self._index_filter = index_filter

        if confidence < 0.0:
            raise _exceptions.InvalidAdetailerDetectorUriError(
                'adetailer detector confidence must be greater than 0.'
            )

        self._confidence = confidence

        if self._device is not None:
            device = str(device)

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

            if _util.is_single_file_model_load(self.model) or ext in {'.yaml', '.yml'}:
                if os.path.exists(self.model):
                    return self.model
                else:
                    if local_files_only:
                        raise _exceptions.AdetailerDetectorUriLoadError(f'Could not find adetailer model: {self.model}')
                    return _util.download_non_hf_model(self.model)
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

            confidence = r.args.get('confidence', 0.3)

            try:
                confidence = float(confidence)
            except ValueError:
                raise _exceptions.InvalidAdetailerDetectorUriError(
                    f'could not parse adetailer detector confidence value: {confidence}'
                )

            mask_padding = AdetailerDetectorUri._parse_padding(
                r.args.get('mask-padding', None), 'mask-padding')

            detector_padding = AdetailerDetectorUri._parse_padding(
                r.args.get('detector-padding', None), 'detector-padding')

            mask_blur = r.args.get('mask-blur', None)

            if mask_blur:
                try:
                    mask_blur = int(mask_blur)
                except ValueError:
                    raise _exceptions.InvalidAdetailerDetectorUriError(
                        'adetailer detector mask-blur must be an integer value.')

            mask_dilation = r.args.get('mask-dilation', None)

            if mask_dilation:
                try:
                    mask_dilation = int(mask_dilation)
                except ValueError:
                    raise _exceptions.InvalidAdetailerDetectorUriError(
                        'adetailer detector mask-dilation must be an integer value.')

            index_filter = r.args.get('index-filter', None)

            if index_filter is not None:
                try:
                    val = ast.literal_eval(index_filter)
                    if not isinstance(val, (list, tuple, set)):
                        val = [val]
                    for i in val:
                        int(i)
                except (ValueError, SyntaxError):
                    raise _exceptions.InvalidAdetailerDetectorUriError(
                        'adetailer detector index-filter must be a list of integers.'
                    )
                index_filter = val

            return AdetailerDetectorUri(
                model=r.concept,
                weight_name=r.args.get('weight-name', None),
                revision=r.args.get('revision', None),
                subfolder=r.args.get('subfolder', None),
                confidence=confidence,
                mask_padding=mask_padding,
                detector_padding=detector_padding,
                mask_blur=mask_blur,
                mask_dilation=mask_dilation,
                index_filter=index_filter,
                prompt=r.args.get('prompt', None),
                negative_prompt=r.args.get('negative-prompt', None),
                mask_shape=r.args.get('mask-shape', None),
                device=r.args.get('device', None))
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidAdetailerDetectorUriError(e)

    @staticmethod
    def _parse_padding(padding, name):
        if padding is not None:
            try:
                padding = _textprocessing.parse_dimensions(padding)

                if len(padding) not in {1, 2, 4}:
                    raise ValueError()

            except ValueError:
                raise _exceptions.InvalidAdetailerDetectorUriError(
                    f'adetailer detector {name} must be an '
                    'integer value, WIDTHxHEIGHT, or LEFTxTOPxRIGHTxBOTTOM')

            if len(padding) == 1:
                padding = padding[0]
        return padding
