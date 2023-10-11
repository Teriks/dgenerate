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
import PIL.ImageOps

import dgenerate.preprocessors.preprocessor as _preprocessor
import dgenerate.types as _types


class MirrorFlipPreprocess(_preprocessor.ImagePreprocessor):
    NAMES = ['mirror', 'flip']

    @staticmethod
    def help(called_by_name):
        if called_by_name == 'mirror':
            return 'Mirror the input image horizontally.'
        if called_by_name == 'flip':
            return 'Flip the input image vertically.'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.called_by_name == 'flip':
            self._func = PIL.ImageOps.flip
        elif self.called_by_name == 'mirror':
            self._func = PIL.ImageOps.mirror

    def __str__(self):
        return f'{self.__class__.__name__}(function="{self.called_by_name}")'

    def pre_resize(self, image: PIL.Image, resize_resolution: _types.OptionalSize):
        return self._func(image)

    def post_resize(self, image: PIL.Image):
        return image


class SimpleColorPreprocess(_preprocessor.ImagePreprocessor):
    NAMES = ['grayscale', 'invert']

    @staticmethod
    def help(called_by_name):
        if called_by_name == 'grayscale':
            return 'Convert the input image to grayscale.'
        if called_by_name == 'invert':
            return 'Invert the colors of the input image.'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.called_by_name == 'grayscale':
            self._func = PIL.ImageOps.grayscale
        elif self.called_by_name == 'invert':
            self._func = PIL.ImageOps.invert

    def __str__(self):
        return f'{self.__class__.__name__}(function="{self.called_by_name}")'

    def pre_resize(self, image: PIL.Image, resize_resolution: _types.OptionalSize):
        return image

    def post_resize(self, image: PIL.Image):
        return self._func(image)


class PosterizePreprocess(_preprocessor.ImagePreprocessor):
    """
    Posterize the input image with PIL.ImageOps.posterize
    """

    NAMES = ['posterize']

    def __init__(self, bits, **kwargs):
        super().__init__(**kwargs)

        self._bits = self.get_int_arg('bits', bits)

        if self._bits < 1 or self._bits > 8:
            self.argument_error(
                f'Argument "bits" must be an integer value between 1 and 8, received {self._bits}.')

    def pre_resize(self, image: PIL.Image, resize_resolution: _types.OptionalSize):
        return PIL.ImageOps.posterize(image, self._bits)

    def post_resize(self, image: PIL.Image):
        return image


class SolarizePreprocess(_preprocessor.ImagePreprocessor):
    """
    Solarize the input image with PIL.ImageOps.solarize
    """

    NAMES = ['solarize']

    def __init__(self, threshold=128, **kwargs):
        super().__init__(**kwargs)

        self._threshold = self.get_int_arg('threshold', threshold)

        if self._threshold < 0 or self._threshold > 255:
            self.argument_error(
                f'Argument "threshold" must be an integer value between 0 and 255, received {self._threshold}.')

    def pre_resize(self, image: PIL.Image, resize_resolution: _types.OptionalSize):
        return PIL.ImageOps.solarize(image, self._threshold)

    def post_resize(self, image: PIL.Image):
        return image
