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

from .preprocessor import ImagePreprocessor


class CannyEdgeDetectPreprocess(ImagePreprocessor):
    NAMES = ['canny']

    def __init__(self,
                 lower=50,
                 upper=100,
                 aperture_size=3,
                 l2_gradient=False,
                 pre_resize=False, **kwargs):
        super().__init__(**kwargs)

        # The module loader will pass the values from the command line as strings

        self._lower = ImagePreprocessor.get_int_arg('lower', lower)
        self._upper = ImagePreprocessor.get_int_arg('upper', upper)
        self._aperture_size = ImagePreprocessor.get_int_arg('aperture_size', aperture_size)

        if (self._aperture_size % 2 == 0 or
                self._aperture_size < 3 or self._aperture_size > 7):
            ImagePreprocessor.argument_error(
                f'Argument "aperture_size" should be an odd number between 3 and 7, received {self._aperture_size}.')

        self._l2_gradient = ImagePreprocessor.get_bool_arg('l2_gradient', l2_gradient)

        self._pre_resize = ImagePreprocessor.get_bool_arg('pre_resize', pre_resize)

    def __str__(self):
        args = [
            ('lower', self._lower),
            ('upper', self._upper),
            ('aperture_size', self._aperture_size),
            ('l2_gradient', self._l2_gradient),
            ('pre_resize', self._pre_resize)
        ]
        return f'{self.__class__.__name__}({", ".join(f"{k}={v}" for k, v in args)})'

    def __repr__(self):
        return str(self)

    def _process(self, image):
        cv_img = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
        edges = cv2.Canny(cv_img, self._lower, self._upper,
                          apertureSize=self._aperture_size,
                          L2gradient=self._l2_gradient)

        return PIL.Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))

    def pre_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        if self._pre_resize:
            return self._process(image)
        return image

    def post_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        if not self._pre_resize:
            return self._process(image)
        return image
