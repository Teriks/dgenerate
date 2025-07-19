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

import PIL.Image
import cv2
import numpy

import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.imageprocessors import imageprocessor as _imageprocessor

__doc__ = """This module provides some basic image processing capabilities using OpenCV."""


class DilateProcessor(_imageprocessor.ImageProcessor):
    """
    Apply morphological dilation to the input image using OpenCV.
    
    Dilation is a morphological operation that expands white regions (foreground objects) 
    in a binary or grayscale image. It's commonly used to fill small holes inside objects
    or to connect nearby objects.
    
    The "size" argument specifies the size of the structuring element used for dilation.
    It can be either an odd integer (e.g., 3, 5, 7, etc.) representing both width and height,
    or a string specifying different dimensions like "5x3" for width x height.
    All dimensions must be odd positive integers.
    
    The "steps" argument specifies how many times the dilation operation is applied.
    More steps result in more expansion.
    
    The "shape" argument specifies the shape of the structuring element:

    NOWRAP!
    - "r" or "rect" or "rectangle": rectangular kernel (default)
    - "c" or "circle" or "ellipse": elliptical kernel
    - "+" or "cross": cross-shaped kernel
    
    The "pre-resize" argument determines if the processing occurs before or after dgenerate resizes the image.
    This defaults to False, meaning the image is processed after dgenerate is done resizing it.
    """

    NAMES = ['dilate']

    # hide inherited arguments
    # that are device related
    HIDE_ARGS = ['device', 'model-offload']

    OPTION_ARGS = {
        'shape': ['r', 'rect', 'rectangle', 'c', 'circle', 'ellipse', '+', 'cross'],
    }

    def __init__(self,
                 size: int | str = 3,
                 steps: int = 1,
                 shape: str = 'rectangle',
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param size: size of the structuring element (must be odd integer or string like "5x3")
        :param steps: number of times to apply the dilation operation
        :param shape: shape of the structuring element ("r" or "rect" or "rectangle", "c" or "circle" or "ellipse", or "+" or "cross")
        :param pre_resize: process the image before it is resized, or after? default is False (after)
        :param kwargs: forwarded to base class
        """
        super().__init__(**kwargs)

        # Parse kernel_size
        if isinstance(size, str):
            try:
                dimensions = _textprocessing.parse_dimensions(size)
            except ValueError as e:
                raise self.argument_error(f'Could not parse "size" as dimensions: {e}') from e

            if len(dimensions) == 0:
                raise self.argument_error('Argument "size" must specify at least one dimension.')

            if len(dimensions) > 2:
                raise self.argument_error('Argument "size" cannot have more than 2 dimensions.')

            if len(dimensions) == 1:
                kernel_width = kernel_height = dimensions[0]
            else:
                kernel_width, kernel_height = dimensions
        else:
            kernel_width = kernel_height = size

        # Validate dimensions
        if kernel_width < 1 or kernel_height < 1:
            raise self.argument_error('Argument "size" dimensions must be positive integers.')

        if kernel_width % 2 == 0 or kernel_height % 2 == 0:
            raise self.argument_error('Argument "size" dimensions must be odd integers.')

        if steps < 1:
            raise self.argument_error('Argument "steps" must be a positive integer.')

        self._kernel_width = kernel_width
        self._kernel_height = kernel_height
        self._iterations = steps
        self._pre_resize = pre_resize

        # Create the kernel based on shape

        if shape.lower() in {'+', 'cross'}:
            self._kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_width, kernel_height))
        else:
            try:
                parsed_shape = _textprocessing.parse_basic_mask_shape(shape)
            except ValueError:
                parsed_shape = None

            if parsed_shape == _textprocessing.BasicMaskShape.RECTANGLE:
                self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
            elif parsed_shape == _textprocessing.BasicMaskShape.ELLIPSE:
                self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_width, kernel_height))
            else:
                raise self.argument_error(f'Unknown kernel shape: {shape}')

    def _process(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Apply dilation to the image using OpenCV.
        
        :param image: PIL image to process
        :return: processed PIL image
        """
        # Convert PIL image to numpy array
        cv_img = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)

        # Apply dilation
        dilated = cv2.dilate(cv_img, self._kernel, iterations=self._iterations)

        # Convert back to PIL image
        dilated_rgb = cv2.cvtColor(dilated, cv2.COLOR_BGR2RGB)

        return PIL.Image.fromarray(dilated_rgb)

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Pre resize, dilation may or may not occur here depending
        on the boolean value of the processor argument "pre-resize"

        :param image: image to process
        :param resize_resolution: purely informational, is unused by this processor
        :return: possibly a dilated image, or the input image
        """
        if self._pre_resize:
            return self._process(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Post resize, dilation may or may not occur here depending
        on the boolean value of the processor argument "pre-resize"

        :param image: image to process
        :return: possibly a dilated image, or the input image
        """
        if not self._pre_resize:
            return self._process(image)
        return image

    def to(self, device) -> "DilateProcessor":
        """
        Does nothing for this processor.
        :param device: the device
        :return: this processor
        """
        return self


class GaussianBlurProcessor(_imageprocessor.ImageProcessor):
    """
    Apply Gaussian blur to the input image using OpenCV.
    
    Gaussian blur is a widely used effect in image processing that reduces image noise
    and detail by convolving the image with a Gaussian kernel.
    
    The "size" argument specifies the size of the Gaussian kernel.
    It can be either an odd integer (e.g., 3, 5, 7, etc.) representing both width and height,
    or a string specifying different dimensions like "5x3" for width x height.
    All dimensions must be odd positive integers. Larger kernel sizes produce more blur.
    
    The "sigma-x" argument specifies the standard deviation in the X direction.
    If 0, it's calculated from the kernel size.
    
    The "sigma-y" argument specifies the standard deviation in the Y direction.
    If 0, it's set to the same value as sigma-x.
    
    The "pre-resize" argument determines if the processing occurs before or after dgenerate resizes the image.
    This defaults to False, meaning the image is processed after dgenerate is done resizing it.
    """

    NAMES = ['gaussian-blur']

    # hide inherited arguments
    # that are device related
    HIDE_ARGS = ['device', 'model-offload']

    def __init__(self,
                 size: int | str = 5,
                 sigma_x: float = 0.0,
                 sigma_y: float = 0.0,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param size: size of the Gaussian kernel (must be odd integer or string like "5x3")
        :param sigma_x: standard deviation in X direction (0 = calculate from kernel size)
        :param sigma_y: standard deviation in Y direction (0 = same as sigma_x)
        :param pre_resize: process the image before it is resized, or after? default is False (after)
        :param kwargs: forwarded to base class
        """
        super().__init__(**kwargs)

        # Parse kernel_size
        if isinstance(size, str):
            try:
                dimensions = _textprocessing.parse_dimensions(size)
            except ValueError as e:
                raise self.argument_error(f'Could not parse "size" as dimensions: {e}') from e

            if len(dimensions) == 0:
                raise self.argument_error('Argument "size" must specify at least one dimension.')

            if len(dimensions) > 2:
                raise self.argument_error('Argument "size" cannot have more than 2 dimensions.')

            if len(dimensions) == 1:
                kernel_width = kernel_height = dimensions[0]
            else:
                kernel_width, kernel_height = dimensions
        else:
            kernel_width = kernel_height = size

        # Validate dimensions
        if kernel_width < 1 or kernel_height < 1:
            raise self.argument_error('Argument "size" dimensions must be positive integers.')

        if kernel_width % 2 == 0 or kernel_height % 2 == 0:
            raise self.argument_error('Argument "size" dimensions must be odd integers.')

        if sigma_x < 0:
            raise self.argument_error('Argument "sigma-x" must be non-negative.')

        if sigma_y < 0:
            raise self.argument_error('Argument "sigma-y" must be non-negative.')

        self._kernel_width = kernel_width
        self._kernel_height = kernel_height
        self._sigma_x = sigma_x
        self._sigma_y = sigma_y
        self._pre_resize = pre_resize

    def _process(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Apply Gaussian blur to the image using OpenCV.
        
        :param image: PIL image to process
        :return: processed PIL image
        """
        # Convert PIL image to numpy array
        cv_img = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(cv_img, (self._kernel_width, self._kernel_height),
                                   self._sigma_x, sigmaY=self._sigma_y)

        # Convert back to PIL image
        blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

        return PIL.Image.fromarray(blurred_rgb)

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Pre resize, Gaussian blur may or may not occur here depending
        on the boolean value of the processor argument "pre-resize"

        :param image: image to process
        :param resize_resolution: purely informational, is unused by this processor
        :return: possibly a blurred image, or the input image
        """
        if self._pre_resize:
            return self._process(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Post resize, Gaussian blur may or may not occur here depending
        on the boolean value of the processor argument "pre-resize"

        :param image: image to process
        :return: possibly a blurred image, or the input image
        """
        if not self._pre_resize:
            return self._process(image)
        return image

    def to(self, device) -> "GaussianBlurProcessor":
        """
        Does nothing for this processor.
        :param device: the device
        :return: this processor
        """
        return self


__all__ = _types.module_all()
