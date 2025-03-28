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
import re

import PIL.Image
import PIL.ImageOps

import dgenerate.image as _image
import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types


class MirrorFlipProcessor(_imageprocessor.ImageProcessor):
    """
    Implements the "mirror" and "flip" PIL.ImageOps operations
    as an image imageprocessor

    The "pre-resize" argument determines if the processing occurs before or after dgenerate resizes the image.
    This defaults to False, meaning the image is processed after dgenerate is done resizing it.
    """

    NAMES = ['mirror', 'flip']

    # hide inherited arguments
    # that are device related
    HIDE_ARGS = ['device', 'model-offload']

    @staticmethod
    def help(loaded_by_name):
        if loaded_by_name == 'mirror':
            return 'Mirror the input image horizontally.'
        if loaded_by_name == 'flip':
            return 'Flip the input image vertically.'

    def __init__(self, pre_resize: bool = False, **kwargs):
        """
        :param kwargs: forwarded to base class
        """
        super().__init__(**kwargs)

        if self.loaded_by_name == 'flip':
            self._func = PIL.ImageOps.flip
        elif self.loaded_by_name == 'mirror':
            self._func = PIL.ImageOps.mirror

        self._pre_resize = pre_resize

    def __str__(self):
        args = [
            ('pre_resize', self._pre_resize)
        ]
        return f'{self.__class__.__name__}({", ".join(f"{k}={v}" for k, v in args)})'

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Mirrors or flips the image depending on what name was used to invoke this
        imageprocessor implementation.

        Executes if ``pre_resize`` constructor argument was ``True``.

        :param image: image to process
        :param resize_resolution: purely informational, is unused by this imageprocessor
        :return: the mirrored or flipped image.
        """
        if self._pre_resize:
            return self._func(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Mirrors or flips the image depending on what name was used to invoke this
        imageprocessor implementation.

        Executes if ``pre_resize`` constructor argument was ``False``.

        :param image: image to process
        :return: the mirrored or flipped image.
        """
        if not self._pre_resize:
            return self._func(image)
        return image

    def to(self, device) -> "MirrorFlipProcessor":
        """
        Does nothing for this processor.
        :param device: the device
        :return: this processor
        """
        return self


class SimpleColorProcessor(_imageprocessor.ImageProcessor):
    """
    Implements the "grayscale" and "invert" PIL.ImageOps operations
    as an image imageprocessor.

    The "pre-resize" argument determines if the processing occurs before or after dgenerate resizes the image.
    This defaults to False, meaning the image is processed after dgenerate is done resizing it.
    """

    NAMES = ['grayscale', 'invert']

    # hide inherited arguments
    # that are device related
    HIDE_ARGS = ['device', 'model-offload']

    @staticmethod
    def help(loaded_by_name):
        if loaded_by_name == 'grayscale':
            return 'Convert the input image to grayscale.'
        if loaded_by_name == 'invert':
            return 'Invert the colors of the input image.'

    def __init__(self, pre_resize: bool = False, **kwargs):
        """
        :param kwargs: forwarded to base class
        """
        super().__init__(**kwargs)
        if self.loaded_by_name == 'grayscale':
            self._func = PIL.ImageOps.grayscale
        elif self.loaded_by_name == 'invert':
            self._func = PIL.ImageOps.invert

        self._pre_resize = pre_resize

    def __str__(self):
        args = [
            ('pre_resize', self._pre_resize)
        ]
        return f'{self.__class__.__name__}({", ".join(f"{k}={v}" for k, v in args)})'

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Invert or grayscale the image depending on which name was used to invoke this imageprocessor.

        Executes if ``pre_resize`` constructor argument was ``True``.

        :param image: image to process
        :param resize_resolution: purely informational, is unused by this imageprocessor
        :return: the inverted or grayscale image
        """
        if self._pre_resize:
            return self._func(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Invert or grayscale the image depending on which name was used to invoke this imageprocessor.

        Executes if ``pre_resize`` constructor argument was ``False``.

        :param image: image to process
        :return: the inverted or grayscale image
        """

        if not self._pre_resize:
            return self._func(image)
        return image

    def to(self, device) -> "SimpleColorProcessor":
        """
        Does nothing for this processor.
        :param device: the device
        :return: this processor
        """
        return self


class PosterizeProcessor(_imageprocessor.ImageProcessor):
    """
    Posterize the input image with PIL.ImageOps.posterize.

    Accepts the argument 'bits', an integer value from 1 to 8.

    The "pre-resize" argument determines if the processing occurs before or after dgenerate resizes the image.
    This defaults to False, meaning the image is processed after dgenerate is done resizing it.
    """

    NAMES = ['posterize']

    # hide inherited arguments
    # that are device related
    HIDE_ARGS = ['device', 'model-offload']

    def __init__(self, bits: int, pre_resize: bool = False, **kwargs):
        """
        :param bits: required argument, integer value from 1 to 8
        :param kwargs: forwarded to base class
        """
        super().__init__(**kwargs)

        self._bits = bits
        self._pre_resize = pre_resize

        if self._bits < 1 or self._bits > 8:
            raise self.argument_error(
                f'Argument "bits" must be an integer value from 1 to 8, received {self._bits}.')

    def __str__(self):
        args = [
            ('bits', self._bits),
            ('pre_resize', self._pre_resize)
        ]
        return f'{self.__class__.__name__}({", ".join(f"{k}={v}" for k, v in args)})'

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Posterize operation is preformed by this method if ``pre_resize`` constructor argument was ``True``.

        :param image: image to process
        :param resize_resolution: purely informational, is unused by this imageprocessor
        :return: the posterized image
        """
        if self._pre_resize:
            return PIL.ImageOps.posterize(image, self._bits)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Posterize operation is preformed by this method if ``pre_resize`` constructor argument was ``False``.

        :param image: image to process
        :return: the posterized image
        """
        if not self._pre_resize:
            return PIL.ImageOps.posterize(image, self._bits)
        return image

    def to(self, device) -> "PosterizeProcessor":
        """
        Does nothing for this processor.
        :param device: the device
        :return: this processor
        """
        return self


class SolarizeProcessor(_imageprocessor.ImageProcessor):
    """
    Solarize the input image with PIL.ImageOps.solarize.

    Accepts the argument "threshold" which is an integer value from 0 to 255.

    The "pre-resize" argument determines if the processing occurs before or after dgenerate resizes the image.
    This defaults to False, meaning the image is processed after dgenerate is done resizing it.
    """

    NAMES = ['solarize']

    # hide inherited arguments
    # that are device related
    HIDE_ARGS = ['device', 'model-offload']

    def __init__(self, threshold: int = 128, pre_resize: bool = False, **kwargs):
        """
        :param threshold: integer value from 0 to 255, default is 128
        :param kwargs: forwarded to base class
        """
        super().__init__(**kwargs)

        self._threshold = threshold
        self._pre_resize = pre_resize

        if self._threshold < 0 or self._threshold > 255:
            raise self.argument_error(
                f'Argument "threshold" must be an integer value from 0 to 255, received {self._threshold}.')

    def __str__(self):
        args = [
            ('threshold', self._threshold),
            ('pre_resize', self._pre_resize)
        ]
        return f'{self.__class__.__name__}({", ".join(f"{k}={v}" for k, v in args)})'

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Solarize operation is preformed by this method if ``pre_resize`` constructor argument was ``True``.

        :param image: image to process
        :param resize_resolution: purely informational, is unused by this imageprocessor
        :return: the solarized image
        """
        if self._pre_resize:
            return PIL.ImageOps.solarize(image, self._threshold)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Solarize operation is preformed by this method if ``pre_resize`` constructor argument was ``False``.

        :param image: image to process
        :return: the solarized image
        """
        if not self._pre_resize:
            return PIL.ImageOps.solarize(image, self._threshold)
        return image

    def to(self, device) -> "SolarizeProcessor":
        """
        Does nothing for this processor.
        :param device: the device
        :return: this processor
        """
        return self


class ResizeProcessor(_imageprocessor.ImageProcessor):
    """
    Resize an image using basic resampling algorithms.

    The "size" argument is the new image size.

    The "align" argument is the new image alignment.

    The "aspect-correct" argument is a boolean argument that determines if the resize is aspect correct.

    The "algo" argument is the resize filtering algorithm, which can be one of:
    "auto", "nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"

    The "pre-resize" argument determines if the processing occurs before or after dgenerate resizes the image.
    This defaults to False, meaning the image is processed after dgenerate is done resizing it.
    """

    NAMES = ['resize']

    # hide inherited arguments
    # that are device related
    HIDE_ARGS = ['device', 'model-offload']

    def __init__(self,
                 size: str | None = None,
                 align: int | None = None,
                 aspect_correct: bool = True,
                 algo: str = 'auto',
                 pre_resize: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        if size is None and align is None:
            raise self.argument_error('no arguments provided that result in resizing.')

        if algo not in {"auto", "nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"}:
            raise self.argument_error(
                'algo must be one of: "auto", "nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"')

        if size is not None:
            try:
                self._size = _textprocessing.parse_image_size(size)
            except ValueError as e:
                raise self.argument_error(str(e).strip())
        else:
            self._size = None

        if align is not None and align < 1:
            raise self.argument_error('align must not be less than 1.')

        self._align = align
        self._aspect_correct = aspect_correct
        self._algo = algo
        self._pre_resize = pre_resize

    def __str__(self):
        args = [
            ('size', self._size),
            ('align', self._align),
            ('aspect_correct', self._aspect_correct),
            ('algo', self._algo),
            ('pre_resize', self._pre_resize)
        ]
        return f'{self.__class__.__name__}({", ".join(f"{k}={v}" for k, v in args)})'

    def _process(self, image: PIL.Image.Image):
        if self._algo == 'auto':
            algo = None
        else:
            algo = getattr(PIL.Image.Resampling, self._algo.upper())

        return _image.resize_image(img=image,
                                   size=self._size,
                                   aspect_correct=self._aspect_correct,
                                   align=self._align,
                                   algo=algo)

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Does nothing, no-op.

        :param image: the image.
        :param resize_resolution: dimension dgenerate will resize to.
        :return: The same image.
        """
        if self._pre_resize:
            return self._process(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Preform the resize operation.

        :param image: The image after being resized by dgenerate.
        :return: The resized image.
        """

        if not self._pre_resize:
            return self._process(image)
        return image

    def to(self, device) -> "ResizeProcessor":
        """
        Does nothing for this processor.
        :param device: the device
        :return: this processor
        """
        return self


class LetterboxProcessor(_imageprocessor.ImageProcessor):
    """
    Letterbox an image.

    The "box-size" argument is the size of the outer letterbox.

    The "box-is-padding" argument can be used to indicate that "box-size" should be interpreted as padding.

    The "box-color" argument specifies the color to use for the letter box background, the default is black.
    This should be specified as a HEX color code. e.g. #FFFFFF or #FFF

    The "inner-size" argument specifies the size of the inner image.

    The "aspect-correct" argument can be used to determine if the aspect ratio
    of the inner image is maintained or not.

    The "pre-resize" argument determines if the processing occurs before or after dgenerate resizes the image.
    This defaults to False, meaning the image is processed after dgenerate is done resizing it.
    """

    NAMES = ['letterbox']

    # hide inherited arguments
    # that are device related
    HIDE_ARGS = ['device', 'model-offload']

    @staticmethod
    def _match_hex_color(color):
        pattern = r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$'
        if re.match(pattern, color):
            return True
        else:
            return False

    def __init__(self,
                 box_size: str,
                 box_is_padding: bool = False,
                 box_color: str | None = None,
                 inner_size: str | None = None,
                 aspect_correct: bool = True,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param box_size: Size of the outer letterbox
        :param box_is_padding: The ``letterbox_size`` argument should be interpreted as padding?
        :param box_color: What color to use for the letter box background, the default is black.
            This should be specified in as a HEX color code.
        :param inner_size: The size of the inner image
        :param aspect_correct: Should the size of the inner image be aspect correct?
        """
        super().__init__(**kwargs)

        if box_color is not None and not self._match_hex_color(box_color):
            raise self.argument_error('box-color must be a HEX color code, e.g. #FFFFFF or #FFF')

        try:
            self._box_size = _textprocessing.parse_image_size(box_size)
        except ValueError as e:
            raise self.argument_error(f'Could not parse the "box_size" argument as an image dimension: {e}')

        self._box_is_padding = box_is_padding
        self._box_color = box_color

        try:
            if inner_size:
                self._inner_size = _textprocessing.parse_image_size(inner_size)
            else:
                self._inner_size = None
        except ValueError as e:
            raise self.argument_error(f'Could not parse the "inner_size" argument as an image dimension: {e}')

        self._aspect_correct = aspect_correct
        self._pre_resize = pre_resize

    def __str__(self):
        args = [
            ('box_size', self._box_size),
            ('box_is_padding', self._box_is_padding),
            ('box_color', self._box_color),
            ('inner_size', self._inner_size),
            ('aspect_correct', self._aspect_correct),
            ('pre_resize', self._pre_resize)
        ]
        return f'{self.__class__.__name__}({", ".join(f"{k}={v}" for k, v in args)})'

    def _process(self, image: PIL.Image.Image):
        return _image.letterbox_image(image,
                                      box_size=self._box_size,
                                      box_is_padding=self._box_is_padding,
                                      box_color=self._box_color,
                                      inner_size=self._inner_size,
                                      aspect_correct=self._aspect_correct)

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Letterbox operation is preformed by this method if ``pre_resize`` constructor argument was ``True``.

        :param image: image to process
        :param resize_resolution: purely informational, is unused by this imageprocessor
        :return: the letterboxed image
        """
        if self._pre_resize:
            return self._process(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Letterbox operation is preformed by this method if ``pre_resize`` constructor argument was ``False``.

        :param image: image to process
        :return: the letterboxed image
        """
        if not self._pre_resize:
            return self._process(image)
        return image

    def to(self, device) -> "LetterboxProcessor":
        """
        Does nothing for this processor.
        :param device: the device
        :return: this processor
        """
        return self


__all__ = _types.module_all()
