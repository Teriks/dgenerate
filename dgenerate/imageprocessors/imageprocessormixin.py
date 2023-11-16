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

import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.types as _types


class ImageProcessorMixin:
    """
    Mixin functionality for objects that can do image processing such as
    implementors of :py:class:`dgenerate.mediainput.AnimationReader`

    This object can also be instantiated and used alone.
    """

    image_processor_enabled: bool
    """
    Enable or disable image processing.
    """

    image_processor: typing.Optional[_imageprocessor.ImageProcessor] = None
    """
    Current image processor.
    """

    def __init__(self, image_processor: typing.Optional[_imageprocessor.ImageProcessor] = None, *args, **kwargs):
        """
        :param processor: the processor implementation that will be doing
            the image processing.

        :param args: mixin forwarded args
        :param kwargs: mixin forwarded kwargs
        """
        super().__init__(*args, **kwargs)
        self.image_processor = image_processor
        self.image_processor_enabled: bool = True

    def process_image(self,
                      image: PIL.Image.Image,
                      resize_resolution: _types.OptionalSize = None,
                      aspect_correct: bool = True,
                      align: typing.Optional[int] = 8):
        """
        Preform image processing on an image, including the requested resizing step.

        Invokes the assigned image processor on an image.

        If no processor is assigned or the processor is disabled, this is a no-op.

        The original image will be closed if the processor returns a new image
        instead of modifying it in place, you should not count on the original image
        being open and usable once this function completes with a processor assigned
        and the processor enabled, though it is safe to use the input image in a ``with``
        context, if you need to retain a copy, pass a copy.

        :param image: image to process
        :param resize_resolution: image will be resized to this dimension by this method.
        :param aspect_correct: Should the resize operation be aspect correct?

        :param align: Align by this amount of pixels, if the input image is not aligned
            to this amount of pixels, it will be aligned by resizing. Passing ``None``
            or ``1`` disables alignment.

        :return: the processed image, processed by the
            processor assigned in the constructor.
        """

        if self.image_processor is None or not self.image_processor_enabled:
            return image

        return self.image_processor.process(image,
                                            resize_resolution=resize_resolution,
                                            aspect_correct=aspect_correct,
                                            align=align)


__all__ = _types.module_all()
