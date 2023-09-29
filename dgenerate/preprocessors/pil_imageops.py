import typing

import PIL.Image
import PIL.ImageOps

from .preprocessor import ImagePreprocessor


class FlipPreprocess(ImagePreprocessor):
    NAMES = {'flip'}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def pre_resize(self, resize_resolution: typing.Union[None, tuple], image: PIL.Image):
        return PIL.ImageOps.flip(image)

    def post_resize(self, resize_resolution: typing.Union[None, tuple], image: PIL.Image):
        return image


class MirrorPreprocess(ImagePreprocessor):
    NAMES = {'mirror'}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def pre_resize(self, resize_resolution: typing.Union[None, tuple], image: PIL.Image):
        return PIL.ImageOps.mirror(image)

    def post_resize(self, resize_resolution: typing.Union[None, tuple], image: PIL.Image):
        return


class GrayscalePreprocess(ImagePreprocessor):
    NAMES = {'grayscale'}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def pre_resize(self, resize_resolution: typing.Union[None, tuple], image: PIL.Image):
        return image

    def post_resize(self, resize_resolution: typing.Union[None, tuple], image: PIL.Image):
        return PIL.ImageOps.grayscale(image)


class InvertPreprocess(ImagePreprocessor):
    NAMES = {'invert'}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def pre_resize(self, resize_resolution: typing.Union[None, tuple], image: PIL.Image):
        return image

    def post_resize(self, resize_resolution: typing.Union[None, tuple], image: PIL.Image):
        return PIL.ImageOps.invert(image)


class PosterizePreprocess(ImagePreprocessor):
    NAMES = {'posterize'}

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

    def pre_resize(self, resize_resolution: typing.Union[None, tuple], image: PIL.Image):
        return PIL.ImageOps.posterize(image, self._bits)

    def post_resize(self, resize_resolution: typing.Union[None, tuple], image: PIL.Image):
        return image


class SolarizePreprocess(ImagePreprocessor):
    NAMES = {'solarize'}

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

    def pre_resize(self, resize_resolution: typing.Union[None, tuple], image: PIL.Image):
        return PIL.ImageOps.solarize(image, self._threshold)

    def post_resize(self, resize_resolution: typing.Union[None, tuple], image: PIL.Image):
        return image
