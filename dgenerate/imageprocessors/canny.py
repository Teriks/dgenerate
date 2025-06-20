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

import dgenerate.image as _image
import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.imageprocessors import imageprocessor as _imageprocessor


class CannyEdgeDetectProcessor(_imageprocessor.ImageProcessor):
    """
    Process the input image with the Canny edge detection algorithm.
    
    The "lower" argument indicates the lower threshold value for the algorithm, and the "upper"
    argument indicates the upper threshold. "aperture-size" is the size of Sobel kernel used for find image gradients,
    it must be an odd integer from 3 to 7. "L2-gradient" specifies the equation for finding gradient magnitude,
    if True a more accurate equation is used. See: https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html.

    If "blur" is true, apply a 3x3 gaussian blur before processing. If "gray" is true, convert the image to the cv2
    "GRAY" format before processing, which does not happen automatically unless you are using a "threshold_algo" value,
    OpenCV is capable of edge detection on colored images, however you may find better results by converting to its
    internal grayscale format before processing, or you may not, it depends.

    If "threshold_algo" is one of ("otsu", "triangle", "median") try to calculate the lower and upper threshold
    automatically using cv2.threshold or cv2.median in the case of "median". "sigma" scales the range of the automatic
    threshold calculation done when a value for "threshold_algo" is selected. "pre-resize" is a boolean value determining
    if the processing should take place before or after the image is resized by dgenerate.

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

    NAMES = ['canny']

    # hide inherited arguments
    # that are device related
    HIDE_ARGS = ['device', 'model-offload']

    # noinspection PyPep8Naming
    def __init__(self,
                 lower: int = 50,
                 upper: int = 100,
                 aperture_size: int = 3,
                 L2_gradient: bool = False,
                 blur: bool = False,
                 gray: bool = False,
                 threshold_algo: str | None = None,
                 sigma: float = 0.33,
                 detect_resolution: str | None = None,
                 detect_aspect: bool = True,
                 detect_align: int = 1,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param lower: lower threshold for canny edge detection
        :param upper: upper threshold for canny edge detection
        :param aperture_size: aperture size, an odd integer from 3 to 7
        :param L2_gradient: Use L2_gradient? https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
        :param blur: apply a 3x3 gaussian blur before processing?
        :param gray: convert to cv2.GRAY format before processing?
        :param threshold_algo: optional auto thresholding algorithm. One of "otsu", "triangle", or "median".
            the lower, and upper threshold values are determined automagically from the image content if
            this argument is supplied a value.
        :param sigma: scales the range of the automatic threshold calculation
        :param detect_resolution: the input image is resized to this dimension before being processed,
            providing ``None`` indicates it is not to be resized. If there is no resize requested during
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

        if detect_resolution is not None:
            try:
                self._detect_resolution = _textprocessing.parse_image_size(detect_resolution)
            except ValueError as e:
                raise self.argument_error(
                    f'Could not parse the "detect-resolution" argument as an image dimension: {e}') from e
        else:
            self._detect_resolution = None

        if threshold_algo is not None:
            if threshold_algo not in {'otsu', 'triangle', 'median'}:
                raise self.argument_error(
                    'Argument "threshold-algo" must be undefined or one of: otsu, triangle, manual')

        self._threshold_algo = threshold_algo

        self._sigma = sigma
        self._blur = blur
        self._gray = gray
        self._lower = lower
        self._upper = upper
        self._aperture_size = aperture_size
        self._detect_aspect = detect_aspect
        self._detect_align = detect_align

        if (self._aperture_size % 2 == 0 or
                self._aperture_size < 3 or self._aperture_size > 7):
            raise self.argument_error(
                f'Argument "aperture-size" should be an odd number between 3 and 7, received {self._aperture_size}.')

        self._L2_gradient = L2_gradient
        self._pre_resize = pre_resize

    def _get_range(self, threshold):
        return int(max(0, (1 - self._sigma) * threshold)), int(min(255, (1 + self._sigma) * threshold))

    def _process(self, image):
        original_size = image.size

        with image:
            resized = _image.resize_image(
                image,
                self._detect_resolution,
                aspect_correct=self._detect_aspect,
                align=self._detect_align
            )

        image = resized

        gray = self._threshold_algo is not None or self._gray

        convert_back = cv2.COLOR_BGR2RGB if not gray else cv2.COLOR_GRAY2RGB

        lower = self._lower
        upper = self._upper

        cv_img = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)

        if self._blur:
            cv_img = cv2.GaussianBlur(cv_img, (3, 3), 0)

        if gray:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        if self._threshold_algo:
            if self._threshold_algo == 'otsu':
                lower, upper = self._get_range(cv2.threshold(cv_img, 0, 255, cv2.THRESH_OTSU)[0])
            elif self._threshold_algo == 'triangle':
                lower, upper = self._get_range(cv2.threshold(cv_img, 0, 255, cv2.THRESH_TRIANGLE)[0])
            elif self._threshold_algo == 'median':
                lower, upper = self._get_range(numpy.median(cv_img))

        _messages.debug_log(f'Canny Processing with cv2.Canny: (lower={lower}, upper={upper}, '
                            f'apertureSize={self._aperture_size}, L2gradient={self._L2_gradient})')

        edges = cv2.Canny(cv_img, lower, upper,
                          apertureSize=self._aperture_size,
                          L2gradient=self._L2_gradient)

        detected_map = cv2.cvtColor(edges, convert_back)

        detected_map = _image.cv2_resize_image(detected_map, original_size)

        return PIL.Image.fromarray(detected_map)

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Pre resize, canny edge detection may or may not occur here depending
        on the boolean value of the processor argument "pre-resize"

        :param image: image to process
        :param resize_resolution: purely informational, is unused by this processor
        :return: possibly a canny edge detected image, or the input image
        """
        if self._pre_resize:
            return self._process(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Post resize, canny edge detection may or may not occur here depending
        on the boolean value of the processor argument "pre-resize"

        :param image: image to process
        :return: possibly a canny edge detected image, or the input image
        """
        if not self._pre_resize:
            return self._process(image)
        return image

    def to(self, device) -> "CannyEdgeDetectProcessor":
        """
        Does nothing for this processor.
        :param device: the device
        :return: this processor
        """
        return self


__all__ = _types.module_all()
