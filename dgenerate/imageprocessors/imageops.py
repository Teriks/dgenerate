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

import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.types as _types


class MirrorFlipProcessor(_imageprocessor.ImageProcessor):
    """
    Implements the "mirror" and "flip" PIL.ImageOps operations
    as an image imageprocessor
    """

    NAMES = ['mirror', 'flip']

    @staticmethod
    def help(loaded_by_name):
        if loaded_by_name == 'mirror':
            return 'Mirror the input image horizontally.'
        if loaded_by_name == 'flip':
            return 'Flip the input image vertically.'

    def __init__(self, **kwargs):
        """
        :param kwargs: forwarded to base class
        """
        super().__init__(**kwargs)

        if self.loaded_by_name == 'flip':
            self._func = PIL.ImageOps.flip
        elif self.loaded_by_name == 'mirror':
            self._func = PIL.ImageOps.mirror

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Mirrors or flips the image depending on what name was used to invoke this
        imageprocessor implementation.

        :param image: image to process
        :param resize_resolution: purely informational, is unused by this imageprocessor
        :return: the mirrored or flipped image.
        """
        return self._func(image)

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Post resize, for this imageprocessor nothing happens post-resize.

        :param image: image to process
        :return: the same image
        """
        return image


class SimpleColorProcessor(_imageprocessor.ImageProcessor):
    """
    Implements the "grayscale" and "invert" PIL.ImageOps operations
    as an image imageprocessor.
    """

    NAMES = ['grayscale', 'invert']

    @staticmethod
    def help(loaded_by_name):
        if loaded_by_name == 'grayscale':
            return 'Convert the input image to grayscale.'
        if loaded_by_name == 'invert':
            return 'Invert the colors of the input image.'

    def __init__(self, **kwargs):
        """
        :param kwargs: forwarded to base class
        """
        super().__init__(**kwargs)
        if self.loaded_by_name == 'grayscale':
            self._func = PIL.ImageOps.grayscale
        elif self.loaded_by_name == 'invert':
            self._func = PIL.ImageOps.invert

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Pre resize, for this imageprocessor nothing happens pre-resize.

        :param image: image to process
        :param resize_resolution: purely informational, is unused by this imageprocessor
        :return: the same image
        """
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Invert or grayscale the image depending on which name was used to invoke this imageprocessor.

        :param image: image to process
        :return: the inverted or grayscale image
        """
        return self._func(image)


class PosterizeProcessor(_imageprocessor.ImageProcessor):
    """
    Posterize the input image with PIL.ImageOps.posterize.

    Accepts the argument 'bits', an integer value from 1 to 8.
    """

    NAMES = ['posterize']

    def __init__(self, bits: int, **kwargs):
        """
        :param bits: required argument, integer value from 1 to 8
        :param kwargs: forwarded to base class
        """
        super().__init__(**kwargs)

        self._bits = bits

        if self._bits < 1 or self._bits > 8:
            raise self.argument_error(
                f'Argument "bits" must be an integer value from 1 to 8, received {self._bits}.')

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Posterize operation is preformed by this method.

        :param image: image to process
        :param resize_resolution: purely informational, is unused by this imageprocessor
        :return: the posterized image
        """
        return PIL.ImageOps.posterize(image, self._bits)

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Post resize, for this imageprocessor nothing happens post-resize.

        :param image: image to process
        :return: the same image
        """
        return image


class SolarizeProcessor(_imageprocessor.ImageProcessor):
    """
    Solarize the input image with PIL.ImageOps.solarize.

    Accepts the argument "threshold" which is an integer value from 0 to 255.
    """

    NAMES = ['solarize']

    def __init__(self, threshold: int = 128, **kwargs):
        """
        :param threshold: integer value from 0 to 255, default is 128
        :param kwargs: forwarded to base class
        """
        super().__init__(**kwargs)

        self._threshold = threshold

        if self._threshold < 0 or self._threshold > 255:
            raise self.argument_error(
                f'Argument "threshold" must be an integer value from 0 to 255, received {self._threshold}.')

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Solarize operation is preformed by this method.

        :param image: image to process
        :param resize_resolution: purely informational, is unused by this imageprocessor
        :return: the solarized image
        """
        return PIL.ImageOps.solarize(image, self._threshold)

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Post resize, for this imageprocessor nothing happens post-resize.

        :param image: image to process
        :return: the same image
        """
        return image


__all__ = _types.module_all()
