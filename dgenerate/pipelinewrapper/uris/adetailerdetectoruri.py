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

import huggingface_hub

import dgenerate.hfhub as _hfhub
import dgenerate.pipelinewrapper.constants as _pipelinewrapper_constants
import dgenerate.textprocessing as _textprocessing
import dgenerate.torchutil as _torchutil
import dgenerate.types as _types
import dgenerate.webcache as _webcache
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions

_lora_uri_parser = _textprocessing.ConceptUriParser(
    'Adetailer Detector', [
        'revision',
        'subfolder',
        'weight-name',
        'confidence',
        'class-filter',
        'index-filter',
        'model-masks',
        'mask-shape',
        'detector-padding',
        'mask-padding',
        'mask-blur',
        'mask-dilation',
        'prompt',
        'negative-prompt',
        'device',
        'size'
    ], args_raw=['class-filter', 'index-filter'])


class AdetailerDetectorUri:
    """
    Representation of a ``--adetailer-detectors`` uri
    """

    # pipelinewrapper.uris.util.get_uri_accepted_args_schema metadata

    NAMES = ['Adetailer Detector']

    @staticmethod
    def help():
        import dgenerate.arguments as _a
        return _a.get_raw_help_text('--adetailer-detectors')

    OPTION_ARGS = {
        'mask-shape': ['r', 'rect', 'rectangle', 'c', 'circle', 'ellipse'],
    }

    FILE_ARGS = {
        'model': {'mode': 'in', 'filetypes': [('Models', ['*.safetensors', '*.pt', '*.pth', '*.cpkt', '*.bin'])]}
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
    def mask_padding(self) -> _types.OptionalPadding:
        """
        Optional mask padding

        Option 1: Uniform padding
        Option 2: (Left/Right, Top/Bottom)
        Option 3: (Left, Top, Right, Bottom)
        """
        return self._mask_padding

    @property
    def detector_padding(self) -> _types.OptionalPadding:
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
    def model_masks(self) -> _types.OptionalBoolean:
        """
        Prefer masks generated by the model if available?
        """
        return self._model_masks

    @property
    def index_filter(self) -> _types.OptionalIntegersBag:
        """
        Process these YOLO detection indices.
        """
        return self._index_filter

    @property
    def class_filter(self) -> _types.OptionalIntegersAndStringsBag:
        """
        Process only these YOLO detection classes.
        """
        return self._class_filter

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

    @property
    def size(self) -> _types.OptionalInteger:
        """
        Target size for processing detected areas.
        """
        return self._size

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None,
                 weight_name: _types.OptionalName = None,
                 confidence: float = _pipelinewrapper_constants.DEFAULT_ADETAILER_DETECTOR_CONFIDENCE,
                 detector_padding: _types.OptionalPadding = None,
                 mask_shape: _types.OptionalName = None,
                 mask_padding: _types.OptionalPadding = None,
                 mask_blur: _types.OptionalInteger = None,
                 mask_dilation: _types.OptionalInteger = None,
                 model_masks: _types.OptionalBoolean = None,
                 index_filter: _types.OptionalIntegersBag = None,
                 class_filter: _types.OptionalIntegersAndStringsBag = None,
                 prompt: _types.OptionalString = None,
                 negative_prompt: _types.OptionalString = None,
                 device: _types.OptionalName = None,
                 size: _types.OptionalInteger = None):

        self._model = model
        self._revision = revision
        self._subfolder = subfolder
        self._weight_name = weight_name
        self._device = device
        self._mask_blur = mask_blur
        self._mask_dilation = mask_dilation
        self._mask_padding = mask_padding
        self._model_masks = model_masks
        self._detector_padding = detector_padding
        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._class_filter = class_filter
        self._index_filter = index_filter

        if size is not None and size <= 1:
            raise _exceptions.InvalidAdetailerDetectorUriError(
                'adetailer detector size must be an integer greater than 1.'
            )

        self._size = size

        if mask_shape is not None:
            mask_shape = mask_shape.lower()
            try:
                parsed_shape = _textprocessing.parse_basic_mask_shape(mask_shape)
            except ValueError:
                parsed_shape = None

            if parsed_shape is None or parsed_shape not in {
                _textprocessing.BasicMaskShape.RECTANGLE,
                _textprocessing.BasicMaskShape.ELLIPSE
            }:
                raise _exceptions.InvalidAdetailerDetectorUriError(
                    'adetailer detector mask-shape must be one of: '
                    '"r", "rect", "rectangle" or "c", "circle", "ellipse".'
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
                    'adetailer detector index-filter values must be greater than or equal to 0.'
                )

        if class_filter:
            if any(isinstance(i, int) and i < 0 for i in class_filter):
                raise _exceptions.InvalidAdetailerDetectorUriError(
                    'adetailer detector class-filter ID values must be greater than or equal to 0.'
                )

        if confidence < 0.0:
            raise _exceptions.InvalidAdetailerDetectorUriError(
                'adetailer detector confidence must be greater than 0.'
            )

        self._confidence = confidence

        if self._device is not None:
            device = str(device)

            if _torchutil.is_valid_device_string(device):
                self._device = device
            else:
                self._device = None
                raise _exceptions.InvalidAdetailerDetectorUriError(
                    f'invalid adetailer detector device specification, '
                    f'{_torchutil.invalid_device_message(device, cap=False)}')

    def __str__(self):
        return f'{self.__class__.__name__}({str(_types.get_public_attributes(self))})'

    def __repr__(self):
        return str(self)

    def get_model_path(self,
                       local_files_only: bool = False,
                       use_auth_token: _types.OptionalString = None):
        try:
            if _hfhub.is_single_file_model_load(self.model):
                if os.path.exists(self.model):
                    return self.model
                else:
                    # any mimetype
                    return _hfhub.webcache_or_hf_blob_download(self.model, local_files_only=local_files_only)
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
                f'Error loading adetailer model: {e}') from e

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

            confidence = r.args.get('confidence', _pipelinewrapper_constants.DEFAULT_ADETAILER_DETECTOR_CONFIDENCE)

            try:
                confidence = float(confidence)
            except ValueError:
                raise _exceptions.InvalidAdetailerDetectorUriError(
                    f'adetailer detector confidence must be a float value, received: {confidence}'
                )

            try:
                model_masks = r.args.get('model-masks', None)
                if model_masks is not None:
                    model_masks = _types.parse_bool(model_masks)
            except ValueError:
                raise _exceptions.InvalidAdetailerDetectorUriError(
                    f'adetailer detector model-masks must be a boolean value, received: {confidence}'
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
                        f'adetailer detector mask-blur must be an integer value, received: {mask_blur}')

            mask_dilation = r.args.get('mask-dilation', None)

            if mask_dilation:
                try:
                    mask_dilation = int(mask_dilation)
                except ValueError:
                    raise _exceptions.InvalidAdetailerDetectorUriError(
                        f'adetailer detector mask-dilation must be an integer value, received: {mask_dilation}')

            size = r.args.get('size', None)
            
            if size:
                try:
                    size = int(size)
                except ValueError:
                    raise _exceptions.InvalidAdetailerDetectorUriError(
                        f'adetailer detector size must be an integer value, received: {size}')

            # Process class_filter and index_filter using shared utility function
            class_filter_raw = r.args.get('class-filter', None)
            index_filter_raw = r.args.get('index-filter', None)

            # Convert string representations to Python objects for yolo_filters_parse
            class_filter_parsed = None
            index_filter_parsed = None

            if class_filter_raw is not None:
                try:
                    # First try to parse as a literal Python expression
                    try:
                        val = ast.literal_eval(class_filter_raw)
                        if not isinstance(val, (list, tuple, set)):
                            if not isinstance(val, (str, int)):
                                raise _exceptions.InvalidAdetailerDetectorUriError(
                                    f'adetailer detector class-filter '
                                    f'cannot except parsed literal type: {type(val).__name__}'
                                )
                            val = [val]
                        class_filter_parsed = list(val)
                    except (ValueError, SyntaxError):
                        # If that fails, treat as a string (could be comma-separated)
                        class_filter_parsed = class_filter_raw
                except Exception as e:
                    raise _exceptions.InvalidAdetailerDetectorUriError(
                        f'adetailer detector class-filter: {e}'
                    )

            if index_filter_raw is not None:
                try:
                    val = ast.literal_eval(index_filter_raw)
                    if not isinstance(val, (list, tuple, set)):
                        if not isinstance(val, (str, int)):
                            raise _exceptions.InvalidAdetailerDetectorUriError(
                                f'adetailer detector index-filter '
                                f'cannot except parsed literal type: {type(val).__name__}'
                            )
                        val = [int(val)]
                    index_filter_parsed = list(val)
                except (ValueError, SyntaxError):
                    raise _exceptions.InvalidAdetailerDetectorUriError(
                        f'adetailer detector index-filter must be an integer or list of integers, received: {index_filter_raw}'
                    )

            # Use shared utility function with custom error handler
            # Import locally to avoid circular import
            from dgenerate.imageprocessors.util import yolo_filters_parse
            
            def argument_error(msg):
                raise _exceptions.InvalidAdetailerDetectorUriError(f'adetailer detector: {msg}')

            try:
                class_filter, index_filter = yolo_filters_parse(
                    class_filter_parsed, index_filter_parsed, argument_error
                )
            except Exception as e:
                # Re-raise with adetailer context if not already handled
                if not isinstance(e, _exceptions.InvalidAdetailerDetectorUriError):
                    raise _exceptions.InvalidAdetailerDetectorUriError(f'adetailer detector filter parsing error: {e}') from e
                raise

            result = AdetailerDetectorUri(
                model=r.concept,
                weight_name=r.args.get('weight-name', None),
                revision=r.args.get('revision', None),
                subfolder=r.args.get('subfolder', None),
                confidence=confidence,
                mask_padding=mask_padding,
                detector_padding=detector_padding,
                mask_blur=mask_blur,
                mask_dilation=mask_dilation,
                model_masks=model_masks,
                class_filter=class_filter,
                index_filter=index_filter,
                prompt=r.args.get('prompt', None),
                negative_prompt=r.args.get('negative-prompt', None),
                mask_shape=r.args.get('mask-shape', None),
                device=r.args.get('device', None),
                size=size)

            return result
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidAdetailerDetectorUriError(e) from e

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
