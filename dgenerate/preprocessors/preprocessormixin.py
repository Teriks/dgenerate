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

import dgenerate.image as _d_image
import dgenerate.image as _image
import dgenerate.messages as _messages
import dgenerate.preprocessors.preprocessor as _preprocessor
import dgenerate.types as _types


class ImagePreprocessorMixin:
    """
    Mixin functionality for objects that do image preprocessing such as
    implementors of :py:class:`dgenerate.mediainput.AnimationReader`
    """

    preprocessor_enabled: bool
    """
    Enable or disable image preprocessing.
    """

    def __init__(self, preprocessor: _preprocessor.ImagePreprocessor, *args, **kwargs):
        """
        :param preprocessor: the preprocessor implementation that will be doing
            the image preprocessing.

        :param args: mixin forwarded args
        :param kwargs: mixin forwarded kwargs
        """
        super().__init__(*args, **kwargs)
        self._preprocessor = preprocessor
        self.preprocess_enabled: bool = True

    def _preprocess_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        if self._preprocessor is not None and self.preprocess_enabled:
            filename = _image.get_filename(image)

            _messages.debug_log('Starting Image Preprocess - '
                                f'{self._preprocessor}.pre_resize('
                                f'image="{filename}", resize_resolution={resize_resolution})')

            processed = self._preprocessor.pre_resize(image, resize_resolution)

            _messages.debug_log(f'Finished Image Preprocess - {self._preprocessor}.pre_resize')
            return processed
        return image

    def _preprocess_post_resize(self, image: PIL.Image.Image):
        if self._preprocessor is not None and self.preprocess_enabled:
            filename = _image.get_filename(image)

            _messages.debug_log('Starting Image Preprocess - '
                                f'{self._preprocessor}.post_resize('
                                f'image="{filename}")')

            processed = self._preprocessor.post_resize(image)

            _messages.debug_log(f'Finished Image Preprocess - {self._preprocessor}.post_resize')
            return processed
        return image

    def preprocess_image(self, image: PIL.Image.Image, resize_to: _types.OptionalSize, aspect_correct: bool = True):
        """
        Preform image preprocessing on an image, including the requested resizing step.

        Invokes the assigned image preprocessor pre and post resizing with appropriate
        arguments and correct resource management.


        :param image: image to process
        :param resize_to: image will be resized to this dimension by this method.
        :param aspect_correct: Should the resize operation be aspect correct?

        :return: the processed image, processed by the
            preprocessor assigned in the constructor.
        """

        # This is the actual size it will end
        # up being resized to by resize_image
        calculate_new_size = _d_image.resize_image_calc(old_size=image.size,
                                                        new_size=resize_to,
                                                        aspect_correct=aspect_correct)

        pre_processed = self._preprocess_pre_resize(image,
                                                    calculate_new_size)

        if pre_processed is not image:
            image.close()

        if resize_to is None:
            image = pre_processed
        else:
            image = _d_image.resize_image(img=pre_processed,
                                          size=resize_to,
                                          aspect_correct=aspect_correct)

        if image is not pre_processed:
            pre_processed.close()

        pre_processed = self._preprocess_post_resize(image)
        if pre_processed is not image:
            image.close()

        return pre_processed


__all__ = _types.module_all()
