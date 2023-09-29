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
from .. import messages


class ImagePreprocessorMixin:
    def __init__(self, preprocessor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._preprocessor = preprocessor

    def preprocess_pre_resize(self, resize_resolution, image):
        if self._preprocessor is not None:
            messages.debug_log('Starting Image Preprocess - '
                               f'{self._preprocessor}.pre_resize('
                               f'resize_resolution={resize_resolution}, image="{image.filename}")')

            processed = self._preprocessor.pre_resize(resize_resolution, image)
            processed.filename = image.filename

            messages.debug_log(f'Finished Image Preprocess - {self._preprocessor}.pre_resize')
            return processed
        return image

    def preprocess_post_resize(self, resize_resolution, image):
        if self._preprocessor is not None:
            messages.debug_log('Starting Image Preprocess - '
                               f'{self._preprocessor}.post_resize('
                               f'resize_resolution={resize_resolution}, image="{image.filename}")')

            processed = self._preprocessor.post_resize(resize_resolution, image)
            processed.filename = image.filename

            messages.debug_log(f'Finished Image Preprocess - {self._preprocessor}.post_resize')
            return processed
        return image
