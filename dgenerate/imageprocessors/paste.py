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

import os

import PIL.Image
import PIL.ImageFilter

import dgenerate.image as _image
import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types


class PasteProcessor(_imageprocessor.ImageProcessor):
    """
    Paste an image on top of the incoming image at a specified position.
    
    The "image" argument specifies the path to the image file to paste.
    
    The "position" argument specifies where to paste the image. It can be:

    NOWRAP!
    - "LEFTxTOP" format (e.g., "100x50") to specify the top-left coordinate
    - "LEFTxTOPxRIGHTxBOTTOM" format (e.g., "100x50x300x200") to specify a bounding
      box where the source image will be resized to fit

    The "feather" argument specifies the feathering radius in pixels for softening edges.
    This creates smooth transitions from opaque to transparent. If not specified, no feathering 
    is applied. Cannot be used together with the "mask" parameter.

    The "feather-shape" argument controls the shape of the feathering:

    NOWRAP!
    - "r" or "rect" or "rectangle" (default): Rectangular feathering from edges
    - "c" or "circle" or "ellipse": Elliptical feathering from center

    Only used when "feather" is specified.
    
    The "mask" argument allows you to specify a mask image path that will be used to control
    the transparency of the pasted image. The mask should be a grayscale image where white
    areas represent full opacity and black areas represent full transparency. Cannot be used 
    together with the "feather" parameter.
    
    The "pre-resize" argument determines if the processing occurs before or after dgenerate 
    resizes the image. This defaults to False, meaning the image is processed after dgenerate 
    is done resizing it.
    """

    NAMES = ['paste']

    # hide inherited arguments that are device related
    HIDE_ARGS = ['device', 'model-offload']

    def __init__(self,
                 image: str,
                 position: str = "0x0",
                 feather: int | None = None,
                 feather_shape: str = "rectangle",
                 mask: str | None = None,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param image: path to the image file to paste
        :param position: position specification in "LEFTxTOP" or "LEFTxTOPxRIGHTxBOTTOM" format
        :param feather: feathering radius in pixels for softening edges (cannot be used with mask)
        :param feather_shape: shape of feathering ("rectangle", "rect", "circle", or "ellipse")
        :param mask: path to a mask image file for controlling transparency (cannot be used with feather)
        :param pre_resize: process the image before it is resized, or after? default is False (after)
        :param kwargs: forwarded to base class
        """
        super().__init__(**kwargs)

        if feather is not None and mask is not None:
            raise self.argument_error(
                'Cannot use both feather and mask parameters together. Choose one method for transparency.')

        if feather is not None and feather < 0:
            raise self.argument_error(
                'Feather value must be greater than or equal to 0')

        if feather is not None and feather_shape is not None:

            try:
                parsed_shape = _textprocessing.parse_basic_mask_shape(feather_shape)
            except ValueError:
                parsed_shape = None

            if parsed_shape is None or parsed_shape not in {
                _textprocessing.BasicMaskShape.RECTANGLE,
                _textprocessing.BasicMaskShape.ELLIPSE
            }:
                raise self.argument_error(
                    'Feather shape must be: "r", "rect", "rectangle", or "c", "circle", "ellipse"')

        self._feather = feather
        self._feather_shape = feather_shape
        self._pre_resize = pre_resize

        # Load source image upfront
        if not os.path.exists(image):
            raise self.argument_error(f'Source image file does not exist: {image}')

        try:
            self._source_image = PIL.Image.open(image)
            # Ensure source image is in RGB mode
            if self._source_image.mode != 'RGB':
                self._source_image = self._source_image.convert('RGB')
        except Exception as e:
            raise self.argument_error(f'Failed to load source image: {e}')

        # Load mask image upfront if provided
        self._mask_image = None
        if mask is not None:
            if not os.path.exists(mask):
                raise self.argument_error(f'Mask image file does not exist: {mask}')

            try:
                self._mask_image = PIL.Image.open(mask)
                # Convert to grayscale if needed
                if self._mask_image.mode != 'L':
                    self._mask_image = self._mask_image.convert('L')
            except Exception as e:
                raise self.argument_error(f'Failed to load mask image: {e}')

        # Parse position argument
        self._position = self._parse_position(position)

    def _parse_position(self, position: str) -> tuple:
        """Parse position string into coordinates"""
        try:
            parts = _textprocessing.parse_dimensions(position)
        except ValueError:
            raise self.argument_error(
                f'Invalid position format: {position}. Expected "LEFTxTOP" or "LEFTxTOPxRIGHTxBOTTOM"'
            )

        if len(parts) == 2:
            # LEFTxTOP format
            left, top = parts
            if left < 0 or top < 0:
                raise self.argument_error('Position coordinates must be non-negative')
            return parts

        elif len(parts) == 4:
            # LEFTxTOPxRIGHTxBOTTOM format
            left, top, right, bottom = parts
            if left < 0 or top < 0 or right < 0 or bottom < 0:
                raise self.argument_error('Position coordinates must be non-negative')
            if left >= right or top >= bottom:
                raise self.argument_error(
                    'Invalid bounding box: left must be less than right and top must be less than bottom')
            return parts

        else:
            raise self.argument_error(
                f'Invalid position format: {position}. Expected "LEFTxTOP" or "LEFTxTOPxRIGHTxBOTTOM"')

    def _process(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """Process the image by pasting the source image onto it"""
        # Make a copy of the base image to avoid modifying the original
        result_image = image.copy()

        # Use the preloaded source image
        source_image = self._source_image

        if len(self._position) == 2:
            left, top = self._position
            paste_box = (left, top)  # PIL expects (left, upper)
            paste_size = source_image.size

        elif len(self._position) == 4:
            left, top, right, bottom = self._position
            paste_box = (left, top, right, bottom)  # PIL expects (left, upper, right, lower)
            paste_size = (right - left, bottom - top)

            # Resize source image to fit the bounding box
            resampling = _image.best_pil_resampling(source_image.size, paste_size)
            source_image = source_image.resize(paste_size, resampling)
        else:
            assert False, f"invalid paste point / box with {len(self._position)} dimensions"

        # Handle transparency - either feather or mask, but not both
        if self._feather is not None:
            # Use paste_with_feather for smooth blending
            result_image = _image.paste_with_feather(
                background=result_image,
                foreground=source_image,
                feather=self._feather,
                shape=self._feather_shape,
                location=paste_box
            )
        elif self._mask_image is not None:
            # Apply custom mask
            mask_resampling = _image.best_pil_resampling(self._mask_image.size, paste_size)
            mask = self._mask_image.resize(paste_size, mask_resampling)
            result_image.paste(source_image, paste_box, mask)
        else:
            # Simple paste without transparency
            result_image.paste(source_image, paste_box)

        return result_image

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Process the image before resizing if pre_resize is True.

        :param image: image to process
        :param resize_resolution: purely informational, is unused by this processor
        :return: the processed image
        """
        if self._pre_resize:
            return self._process(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Process the image after resizing if pre_resize is False.

        :param image: image to process
        :return: the processed image
        """
        if not self._pre_resize:
            return self._process(image)
        return image

    def to(self, device) -> "PasteProcessor":
        """
        Does nothing for this processor since it's PIL-based.

        :param device: the device (ignored)
        :return: this processor
        """
        return self
