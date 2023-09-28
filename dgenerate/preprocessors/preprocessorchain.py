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

from .preprocessor import ImagePreprocessor


class ImagePreprocessorChain(ImagePreprocessor):
    HIDDEN = True

    def __init__(self, preprocessors: typing.Union[None, list, tuple] = None, **kwargs):
        super().__init__(**kwargs)

        if preprocessors is None:
            self._preprocessors = []
        else:
            self._preprocessors = preprocessors

    def _preprocessor_names(self):
        for preprocessor in self._preprocessors:
            if hasattr(preprocessor, 'name'):
                preprocessor_name = preprocessor.name
            else:
                preprocessor_name = preprocessor.__class__.__name__

            yield preprocessor_name

    @property
    def name(self):
        if not self._preprocessors:
            return f'{self.__class__.__name__}([])'
        else:
            return f'{self.__class__.__name__}([{", ".join(self._preprocessor_names())}])'

    def add_processor(self, preprocessor):
        self._preprocessors.append(preprocessor)

    def pre_resize(self, resize_resolution: typing.Union[None, tuple], image: PIL.Image):
        if self._preprocessors:
            p_image = image
            for preprocessor in self._preprocessors:
                p_image = preprocessor.pre_resize(resize_resolution, image)
                if p_image is not image:
                    image.close()

            p_image.filename = image.filename
            return p_image
        else:
            return image

    def post_resize(self, resize_resolution: typing.Union[None, tuple], image: PIL.Image):
        if self._preprocessors:
            p_image = image
            for preprocessor in self._preprocessors:
                p_image = preprocessor.post_resize(resize_resolution, image)
                if p_image is not image:
                    image.close()

            p_image.filename = image.filename
            return p_image
        else:
            return image
