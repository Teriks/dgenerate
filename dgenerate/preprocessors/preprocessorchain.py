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

import dgenerate.preprocessors.preprocessor as _preprocessor
import dgenerate.types as _types


class ImagePreprocessorChain(_preprocessor.ImagePreprocessor):
    """
    Implements chainable image preprocessors.

    Chains preprocessing steps together in a sequence.
    """

    HIDDEN = True

    def __init__(self, preprocessors: typing.Optional[typing.Iterable[_preprocessor.ImagePreprocessor]] = None):
        """
        :param preprocessors: optional initial preprocessors to fill the chain, accepts an iterable
        """
        super().__init__()

        if preprocessors is None:
            self._preprocessors = []
        else:
            self._preprocessors = list(preprocessors)

    def _preprocessor_names(self):
        for preprocessor in self._preprocessors:
            yield str(preprocessor)

    def __str__(self):
        if not self._preprocessors:
            return f'{self.__class__.__name__}([])'
        else:
            return f'{self.__class__.__name__}([{", ".join(self._preprocessor_names())}])'

    def __repr__(self):
        return str(self)

    def add_processor(self, preprocessor: _preprocessor.ImagePreprocessor):
        """
        Add a preprocessor implementation to the chain.

        :param preprocessor: :py:class:`dgenerate.preprocessors.preprocessor.ImagePreprocessor`
        """
        self._preprocessors.append(preprocessor)

    def pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Invoke pre_resize on all preprocessors in this preprocessor chain in turn.

        Every subsequent invocation receives the last preprocessed image as its argument.

        This method should not be invoked directly, use the class method
        :py:meth:`dgenerate.preprocessors.preprocessor.ImagePreprocessor.call_pre_resize` to invoke it.

        :param image: initial image to preprocess
        :param resize_resolution: the size which the image will be resized to after this
            step, this is only information for the preprocessors and the image will not be
            resized by this method. Image preprocessors should never resize images as it is
            the responsibility of dgenerate to do that for the user.

        :return: the processed image, possibly affected by every preprocessor in the chain
        """

        if self._preprocessors:
            p_image = image
            for preprocessor in self._preprocessors:
                new_img = _preprocessor.ImagePreprocessor.call_pre_resize(preprocessor, p_image, resize_resolution)
                if new_img is not p_image:
                    p_image.close()
                p_image = new_img
            return p_image
        else:
            return image

    def post_resize(self, image: PIL.Image.Image):
        """
        Invoke post_resize on all preprocessors in this preprocessor chain in turn.

        Every subsequent invocation receives the last preprocessed image as its argument.

        This method should not be invoked directly, use the class method
        :py:meth:`dgenerate.preprocessors.preprocessor.ImagePreprocessor.call_post_resize` to invoke it.

        :param image: initial image to preprocess
        :return: the processed image, possibly affected by every preprocessor in the chain
        """

        if self._preprocessors:
            p_image = image
            for preprocessor in self._preprocessors:
                new_img = _preprocessor.ImagePreprocessor.call_post_resize(preprocessor, p_image)
                if new_img is not p_image:
                    p_image.close()
                p_image = new_img
            return p_image
        else:
            return image
