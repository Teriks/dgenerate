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
from . import ImagePreprocessor
from .. import messages
from ..image import resize_image, resize_image_calc


class ImagePreprocessorMixin:
    def __init__(self, preprocessor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._preprocessor = preprocessor

    def _preprocess_pre_resize(self, image, resize_resolution):
        if self._preprocessor is not None:
            messages.debug_log('Starting Image Preprocess - '
                               f'{self._preprocessor}.pre_resize('
                               f'image="{image.filename}", resize_resolution={resize_resolution})')

            processed = ImagePreprocessor.call_pre_resize(self._preprocessor, image, resize_resolution)

            messages.debug_log(f'Finished Image Preprocess - {self._preprocessor}.pre_resize')
            return processed
        return image

    def _preprocess_post_resize(self, image, resize_resolution):
        if self._preprocessor is not None:
            messages.debug_log('Starting Image Preprocess - '
                               f'{self._preprocessor}.post_resize('
                               f'image="{image.filename}", resize_resolution={resize_resolution})')

            processed = ImagePreprocessor.call_post_resize(self._preprocessor, image)

            messages.debug_log(f'Finished Image Preprocess - {self._preprocessor}.post_resize')
            return processed
        return image

    def preprocess_image(self, image, resize_to):

        pre_processed = self._preprocess_pre_resize(image,
                                                    resize_image_calc(old_size=image.size,
                                                                      new_size=resize_to))

        if pre_processed is not image:
            image.close()

        if resize_to is None:
            image = pre_processed
        else:
            image = resize_image(pre_processed, resize_to)

        if image is not pre_processed:
            pre_processed.close()

        pre_processed = self._preprocess_post_resize(image, resize_to)
        if pre_processed is not image:
            image.close()

        return pre_processed
