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

import dgenerate.postprocessors.postprocessor as _postprocessor
import dgenerate.types as _types


class ImagePostprocessorChain(_postprocessor.ImagePostprocessor):
    """
    Implements chainable image postprocessors.

    Chains postprocessing steps together in a sequence.
    """

    HIDDEN = True

    def __init__(self, postprocessors: typing.Optional[typing.Iterable[_postprocessor.ImagePostprocessor]] = None):
        """
        :param postprocessors: optional initial postprocessors to fill the chain, accepts an iterable
        """
        super().__init__(called_by_name='chain')

        if postprocessors is None:
            self._postprocessors = []
        else:
            self._postprocessors = list(postprocessors)

    def _postprocessor_names(self):
        for postprocessor in self._postprocessors:
            yield str(postprocessor)

    def __str__(self):
        if not self._postprocessors:
            return f'{self.__class__.__name__}([])'
        else:
            return f'{self.__class__.__name__}([{", ".join(self._postprocessor_names())}])'

    def __repr__(self):
        return str(self)

    def add_processor(self, postprocessor: _postprocessor.ImagePostprocessor):
        """
        Add a postprocessor implementation to the chain.

        :param postprocessor: :py:class:`dgenerate.postprocessors.postprocessor.ImagePostprocessor`
        """
        self._postprocessors.append(postprocessor)

    def impl_process(self, image: PIL.Image.Image):
        """
        Invoke process on all postprocessors in this postprocessor chain in turn.

        Every subsequent invocation receives the last processed image as its argument.

        This method should not be invoked directly, use the class method
        :py:meth:`dgenerate.postprocessors.postprocessor.ImagePostprocessor.process` to invoke it.

        :param image: initial image to process

        :return: the processed image, possibly affected by every postprocessor in the chain
        """

        if self._postprocessors:
            p_image = image
            for postprocessor in self._postprocessors:
                p_image = postprocessor.process(p_image)
            return p_image
        else:
            return image


__all__ = _types.module_all()
