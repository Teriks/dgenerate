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

import PIL.Image
import cv2
import numpy
import numpy as np

from .preprocessor import ImagePreprocessor
from .. import messages


class CannyEdgeDetectPreprocess(ImagePreprocessor):
    """
    Process the input image with the Canny edge detection algorithm for use with a ControlNet.
    The "lower" argument indicates the lower threshold value for the algorithm, and the "upper"
    argument indicates the upper threshold. "aperture-size" is the size of Sobel kernel used for find image gradients,
    it must be an odd integer from 3 to 7. "L2-gradient" specifies the equation for finding gradient magnitude,
    if True a more accurate equation is used. If "blur" is true, apply a 3x3 gaussian blur before processing.
    If "threshold_algo" is one of ("otsu", "triangle", "median") try to calculate the lower and upper threshold
    automatically using cv2.threshold. "sigma" scales the range of the automatic threshold calculation done
    when a value for "threshold_algo" is selected. "pre-resize" is a boolean value determining if the processing
    should take place before or after the image is resized by dgenerate.
    See: https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
    """

    NAMES = ['canny']

    def __init__(self,
                 lower=50,
                 upper=100,
                 aperture_size=3,
                 L2_gradient=False,
                 blur=False,
                 threshold_algo=None,
                 sigma=0.33,
                 pre_resize=False, **kwargs):
        super().__init__(**kwargs)

        # The module loader will pass the values from the command line as strings

        if threshold_algo is not None:
            if threshold_algo not in {'otsu', 'triangle', 'median'}:
                self.argument_error(
                    'Argument "threshold_algo" must be undefined or one of: otsu, triangle, manual')

        self._threshold_algo = threshold_algo

        self._sigma = self.get_float_arg('sigma', sigma)
        self._blur = self.get_bool_arg('blur', blur)
        self._lower = self.get_int_arg('lower', lower)
        self._upper = self.get_int_arg('upper', upper)
        self._aperture_size = self.get_int_arg('aperture_size', aperture_size)

        if (self._aperture_size % 2 == 0 or
                self._aperture_size < 3 or self._aperture_size > 7):
            self.argument_error(
                f'Argument "aperture_size" should be an odd number between 3 and 7, received {self._aperture_size}.')

        self._L2_gradient = self.get_bool_arg('L2_gradient', L2_gradient)
        self._pre_resize = self.get_bool_arg('pre_resize', pre_resize)

    def _get_range(self, threshold):
        return int(max(0, (1 - self._sigma) * threshold)), int(min(255, (1 + self._sigma) * threshold))

    def __str__(self):
        args = [
            ('lower', self._lower),
            ('upper', self._upper),
            ('aperture_size', self._aperture_size),
            ('L2_gradient', self._L2_gradient),
            ('blur', self._blur),
            ('threshold_algo', self._pre_resize),
            ('sigma', self._sigma),
            ('pre_resize', self._pre_resize)
        ]
        return f'{self.__class__.__name__}({", ".join(f"{k}={v}" for k, v in args)})'

    def _process(self, image):
        convert_enum = cv2.COLOR_BGR2RGB

        lower = self._lower
        upper = self._upper

        img = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)

        if self._blur:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        if self._threshold_algo:
            img = cv2.cvtColor(numpy.array(img), cv2.COLOR_BGR2GRAY)
            convert_enum = cv2.COLOR_GRAY2RGB

            if self._threshold_algo == 'otsu':
                lower, upper = self._get_range(cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[0])
            elif self._threshold_algo == 'triangle':
                lower, upper = self._get_range(cv2.threshold(img, 0, 255, cv2.THRESH_TRIANGLE)[0])
            elif self._threshold_algo == 'median':
                lower, upper = self._get_range(np.median(img))

        messages.debug_log(f'Canny Processing with: lower {lower}, upper {upper}, '
                           f'aperature-size {self._aperture_size}, L2gradient {self._L2_gradient}')

        edges = cv2.Canny(img, lower, upper,
                          apertureSize=self._aperture_size,
                          L2gradient=self._L2_gradient)

        return PIL.Image.fromarray(cv2.cvtColor(edges, convert_enum))

    def pre_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        if self._pre_resize:
            return self._process(image)
        return image

    def post_resize(self, image: PIL.Image):
        if not self._pre_resize:
            return self._process(image)
        return image
