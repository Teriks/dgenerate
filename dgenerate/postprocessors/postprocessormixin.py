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
import dgenerate.messages as _messages
import dgenerate.postprocessors.postprocessor as _postprocessor
import dgenerate.types as _types
import typing


class ImagePostprocessorMixin:
    """
    Mixin functionality for objects that do image postprocessing such as :py:class:`dgenerate.renderloop.RenderLoop`

    This object can also be instantiated and used alone.
    """

    image_postprocessor_enabled: bool
    """
    Enable or disable image postprocessing.
    """

    image_postprocessor: typing.Optional[_postprocessor.ImagePostprocessor] = None
    """
    Current image postprocessor.
    """

    def __init__(self, postprocessor: typing.Optional[_postprocessor.ImagePostprocessor] = None, *args, **kwargs):
        """
        :param postprocessor: the postprocessor implementation that will be doing
            the image postprocessing.

        :param args: mixin forwarded args
        :param kwargs: mixin forwarded kwargs
        """
        super().__init__(*args, **kwargs)
        self.image_postprocessor = postprocessor
        self.image_postprocessor_enabled: bool = True

    def _postprocess(self, image: PIL.Image.Image):
        if self.image_postprocessor is not None and self.image_postprocessor_enabled:
            filename = _image.get_filename(image)

            _messages.debug_log('Starting Image Postprocess - '
                                f'{self.image_postprocessor}.pre_resize(image="{filename}")')

            processed = self.image_postprocessor.process(image)

            _messages.debug_log(f'Finished Image Postprocess - {self.image_postprocessor}.process')
            return processed
        return image

    def postprocess_image(self, image: PIL.Image.Image):
        """
        Preform image postprocessing on an image.

        Invokes the assigned image postprocessor.

        If no postprocessor is assigned or the postprocessor is disabled, this is a no-op.

        :param image: image to process

        :return: the processed image, processed by the
            current :py:attr:`.PostProcessorMixin.image_postprocessor`.
        """

        return self._postprocess(image)


__all__ = _types.module_all()