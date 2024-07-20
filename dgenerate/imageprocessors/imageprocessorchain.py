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
import collections.abc
import typing

import PIL.Image
import torch

import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.types as _types


class ImageProcessorChain(_imageprocessor.ImageProcessor):
    """
    Implements chainable image processors.

    Chains processing steps together in a sequence.
    """

    HIDDEN = True

    def __init__(self,
                 image_processors: typing.Optional[collections.abc.Iterable[_imageprocessor.ImageProcessor]] = None):
        """
        :param image_processors: optional initial image processors to fill the chain, accepts an iterable
        """
        super().__init__(loaded_by_name='chain')

        if image_processors is None:
            self._image_processors = []
        else:
            self._image_processors = list(image_processors)

    def _imageprocessor_names(self):
        for imageprocessor in self._image_processors:
            yield str(imageprocessor)

    def __str__(self):
        if not self._image_processors:
            return f'{self.__class__.__name__}([])'
        else:
            return f'{self.__class__.__name__}([{", ".join(self._imageprocessor_names())}])'

    def __repr__(self):
        return str(self)

    def add_processor(self, image_processor: _imageprocessor.ImageProcessor):
        """
        Add a imageprocessor implementation to the chain.

        :param image_processor: :py:class:`dgenerate.imageprocessors.imageprocessor.ImageProcessor`
        """
        self._image_processors.append(image_processor)

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Invoke pre_resize on all image processors in this imageprocessor chain in turn.

        Every subsequent invocation receives the last processed image as its argument.

        This method should not be invoked directly, use the class method
        :py:meth:`dgenerate.imageprocessors.imageprocessor.ImageProcessor.pre_resize` to invoke it.

        :param image: initial image to process
        :param resize_resolution: the size which the image will be resized to after this
            step, this is only information for the image processors and the image will not be
            resized by this method. Image processors should never resize images as it is
            the responsibility of dgenerate to do that for the user.

        :return: the processed image, possibly affected by every image processor in the chain
        """

        if self._image_processors:
            p_image = image
            for imageprocessor in self._image_processors:
                p_image = imageprocessor.pre_resize(p_image, resize_resolution)
            return p_image
        else:
            return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Invoke post_resize on all image processors in this image processor chain in turn.

        Every subsequent invocation receives the last processed image as its argument.

        This method should not be invoked directly, use the class method
        :py:meth:`dgenerate.imageprocessors.imageprocessor.ImageProcessor.post_resize` to invoke it.

        :param image: initial image to process
        :return: the processed image, possibly affected by every imageprocessor in the chain
        """

        if self._image_processors:
            p_image = image
            for imageprocessor in self._image_processors:
                p_image = imageprocessor.post_resize(p_image)
            return p_image
        else:
            return image

    def to(self, device: torch.device | str) -> "ImageProcessorChain":
        """
        Move all :py:class:`torch.nn.Module` modules registered
        to this image processor to a specific device.

        :raise dgenerate.OutOfMemoryError: if there is not enough memory on the specified device

        :param device: The device string, or torch device object
        :return: the image processor itself
        """
        for p in self._image_processors:
            p.to(device)

        return self


__all__ = _types.module_all()
