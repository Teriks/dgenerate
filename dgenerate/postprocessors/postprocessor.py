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

import dgenerate.image as _image
import dgenerate.plugin as _plugin
import dgenerate.postprocessors.exceptions as _exceptions
import dgenerate.types as _types


class ImagePostprocessor(_plugin.InvokablePlugin):
    """
    Abstract base class for image postprocessor implementations.
    """

    def __init__(self, called_by_name, device: str = 'cpu', **kwargs):
        super().__init__(called_by_name=called_by_name,
                         argument_error_type=_exceptions.ImagePostprocessorArgumentError,
                         **kwargs)
        self.__device = device

    @property
    def device(self) -> str:
        """
        The rendering device requested for this postprocessor.

        :return: device string, for example "cuda", "cuda:N", or "cpu"
        """
        return self.__device

    def impl_process(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Implementation of process that does nothing. Inheritor must implement.

        :param image: input image
        :return: output image
        """
        return image

    def process(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Invoke a postprocessors :py:meth:`.ImagePostprocessor.impl_process` method.

        This is the only appropriate way to invoke a postprocessor manually.

        The original image will be closed if the implementation returns a new image
        instead of modifying it in place, you should not count on the original image
        being open and usable once this function completes though it is safe to
        use the input image in a ``with`` context, if you need to retain a
        copy, pass a copy.

        :param image: the image to pass

        :return: processed image, may be the same image or a copy.
        """

        img = self.impl_process(image)

        if img is not image:
            image.close()

        img.filename = _image.get_filename(image)

        return img

    def __str__(self):
        return f'{self.__class__.__name__}(called_by_name="{self.called_by_name}")'

    def __repr__(self):
        return str(self)

__all__ = _types.module_all()