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
import PIL.ImageOps

from .preprocessor import ImagePreprocessor


class FlipPreprocess(ImagePreprocessor):
    NAMES = ['flip']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def pre_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        return PIL.ImageOps.flip(image)

    def post_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        return image


class MirrorPreprocess(ImagePreprocessor):
    NAMES = ['mirror']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def pre_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        return PIL.ImageOps.mirror(image)

    def post_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        return image


class GrayscalePreprocess(ImagePreprocessor):
    NAMES = ['grayscale']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def pre_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        return image

    def post_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        return PIL.ImageOps.grayscale(image)


class InvertPreprocess(ImagePreprocessor):
    NAMES = ['invert']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def pre_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        return image

    def post_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        return PIL.ImageOps.invert(image)


class PosterizePreprocess(ImagePreprocessor):
    NAMES = ['posterize']

    def __init__(self, bits, **kwargs):
        super().__init__(**kwargs)

        self._bits = ImagePreprocessor.get_int_arg('bits', bits)

        if self._bits < 1 or self._bits > 8:
            ImagePreprocessor.argument_error(
                f'Argument "bits" must be an integer value between 1 and 8, received {self._bits}.')

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def pre_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        return PIL.ImageOps.posterize(image, self._bits)

    def post_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        return image


class SolarizePreprocess(ImagePreprocessor):
    NAMES = ['solarize']

    def __init__(self, threshold=128, **kwargs):
        super().__init__(**kwargs)

        self._threshold = ImagePreprocessor.get_int_arg('threshold', threshold)

        if self._threshold < 0 or self._threshold > 255:
            ImagePreprocessor.argument_error(
                f'Argument "threshold" must be an integer value between 0 and 255, received {self._threshold}.')

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def pre_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        return PIL.ImageOps.solarize(image, self._threshold)

    def post_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        return image
