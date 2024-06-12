import typing

import PIL.Image
import PIL.ImageOps

import dgenerate.imageprocessors


class MyProcessor(dgenerate.imageprocessors.ImageProcessor):
    # A more indepth example for how to implement these
    # can be found in the examples/writing_plugins/image_processor folder
    # This is a very minimal implementation that just inverts the image
    # before it gets resized

    NAMES = ['foo']

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: tuple[int, int] | None):
        return PIL.ImageOps.invert(image)

    def impl_post_resize(self, image: PIL.Image.Image):
        return image
