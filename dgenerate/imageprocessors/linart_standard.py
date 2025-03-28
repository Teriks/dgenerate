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
import controlnet_aux.util as _cna_util
import cv2
import numpy

import dgenerate.image as _image
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.imageprocessors import imageprocessor as _imageprocessor


class LineArtStandardProcessor(_imageprocessor.ImageProcessor):
    """
    Standard lineart detector, generate lineart from an image.

    The "gaussian-sigma" argument is the gaussian filter sigma value.

    The "intensity-threshold" argument is the pixel value intensity threshold.

    The "detect-resolution" argument is the resolution the image is resized to internal to the processor before
    detection is run on it. It should be a single dimension for example: "detect-resolution=512" or the X/Y dimensions
    seperated by an "x" character, like so: "detect-resolution=1024x512". If you do not specify this argument,
    the detector runs on the input image at its full resolution. After processing the image will be resized to
    whatever you have requested dgenerate resize it to via --output-size or --resize/--align in the case of the
    image-process sub-command, if you have not requested any resizing the output will be resized back to the original
    size of the input image.

    The "detect-aspect" argument determines if the image resize requested by "detect-resolution" before
    detection runs is aspect correct, this defaults to true.

    The "detect-align" argument determines the pixel alignment of the image resize requested by
    "detect-resolution", it defaults to 1 indicating no requested alignment.

    The "pre-resize" argument determines if the processing occurs before or after dgenerate resizes the image.
    This defaults to False, meaning the image is processed after dgenerate is done resizing it.
    """

    NAMES = ['lineart-standard']

    # hide inherited arguments
    # that are device related
    HIDE_ARGS = ['device', 'model-offload']

    def __init__(self,
                 gaussian_sigma: float = 6.0,
                 intensity_threshold: int = 8,
                 detect_resolution: str | None = None,
                 detect_aspect: bool = True,
                 detect_align: int = 1,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param gaussian_sigma: gaussian filter sigma value
        :param intensity_threshold: pixel value intensity threshold
        :param detect_resolution: the input image is resized to this dimension before being processed,
            providing ``None`` indicates it is not to be resized.  If there is no resize requested during
            the processing action via ``resize_resolution`` it will be resized back to its original size.
        :param detect_aspect: if the input image is resized by ``detect_resolution`` or ``detect_align``
            before processing, will it be an aspect correct resize?
        :param detect_align: the input image is forcefully aligned to this amount of pixels
            before being processed.
        :param pre_resize: process the image before it is resized, or after? default is ``False`` (after).
        :param kwargs: forwarded to base class
        """

        super().__init__(**kwargs)

        if detect_align < 1:
            raise self.argument_error('Argument "detect-align" may not be less than 1.')

        if gaussian_sigma < 0:
            raise self.argument_error('Argument "gaussian-sigma" may not be less than 0.')

        if intensity_threshold < 0:
            raise self.argument_error('Argument "intensity-threshold" may not be less than 0.')

        if intensity_threshold > 255:
            raise self.argument_error('Argument "intensity-threshold" may not be greater than 255.')

        self._detect_aspect = detect_aspect
        self._detect_align = detect_align
        self._pre_resize = pre_resize
        self._gaussian_sigma = gaussian_sigma
        self._intensity_threshold = intensity_threshold

        if detect_resolution is not None:
            try:
                self._detect_resolution = _textprocessing.parse_image_size(detect_resolution)
            except ValueError as e:
                raise self.argument_error(f'Could not parse the "detect-resolution" argument as an image dimension: {e}')
        else:
            self._detect_resolution = None

    def _process(self, image):
        original_size = image.size

        with image:
            resized = _image.resize_image(
                image,
                self._detect_resolution,
                aspect_correct=self._detect_aspect,
                align=self._detect_align
            )

        input_image = _cna_util.HWC3(numpy.array(resized, dtype=numpy.uint8))

        x = input_image.astype(numpy.float32)
        g = cv2.GaussianBlur(x, (0, 0), self._gaussian_sigma)
        intensity = numpy.min(g - x, axis=2).clip(0, 255)
        # noinspection PyTypeChecker
        intensity /= max(16, numpy.median(intensity[intensity > self._intensity_threshold]))
        intensity *= 127
        detected_map = intensity.clip(0, 255).astype(numpy.uint8)

        detected_map = _cna_util.HWC3(detected_map)

        detected_map = _image.cv2_resize_image(detected_map, original_size)

        return PIL.Image.fromarray(detected_map)

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Pre resize.

        :param image: image to process
        :param resize_resolution: resize resolution
        :return: possibly a lineart image, or the input image
        """

        if self._pre_resize:
            return self._process(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Post resize.

        :param image: image
        :return: possibly a lineart image, or the input image
        """

        if not self._pre_resize:
            return self._process(image)
        return image

    def to(self, device) -> "LineArtStandardProcessor":
        """
        Does nothing for this processor.
        :param device: the device
        :return: this processor
        """
        return self


__all__ = _types.module_all()
