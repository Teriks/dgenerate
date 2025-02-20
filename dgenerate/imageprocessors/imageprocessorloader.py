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

import dgenerate.imageprocessors.exceptions as _exceptions
import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.imageprocessors.imageprocessorchain as _imageprocessorchain
import dgenerate.plugin as _plugin
import dgenerate.types as _types
from dgenerate.plugin import PluginArg as _Pa


class ImageProcessorLoader(_plugin.PluginLoader):
    """
    Loads :py:class:`dgenerate.imageprocessor.ImageProcessor` plugins.
    """

    def __init__(self):

        # The empty string above disables sphinx inherited doc
        super().__init__(base_class=_imageprocessor.ImageProcessor,
                         description='image processor',
                         reserved_args=[_Pa('output-file', type=str, default=None),
                                        _Pa('output-overwrite', type=bool, default=False),
                                        _Pa('device', type=str, default='cpu'),
                                        _Pa('model-offload', type=bool, default=False),
                                        _Pa('local-files-only', type=bool, default=False)],
                         argument_error_type=_exceptions.ImageProcessorArgumentError,
                         not_found_error_type=_exceptions.ImageProcessorNotFoundError)

        self.add_search_module_string('dgenerate.imageprocessors')

    def load(self,
             uri: _types.Uri | collections.abc.Iterable[_types.Uri],
             device: str = 'cpu',
             local_files_only: bool = False,
             **kwargs) -> _imageprocessor.ImageProcessor | _imageprocessorchain.ImageProcessorChain | None:
        """
        Load an image processor or multiple image processors. They are loaded by URI, which
        is their name and any module arguments, for example: ``canny;lower=50;upper=100``

        Specifying multiple processors with a list will create an image processor chain object.

        :raises RuntimeError: if more than one class was found using the provided name mentioned in the URI.
        :raises ImageProcessorNotFoundError: if the name mentioned in the URI could not be found.
        :raises ImageProcessorArgumentError: if the URI contained invalid arguments.


        :param uri: Processor URI or list of URIs
        :param device: Request a specific rendering device, default is CPU
        :param local_files_only: Should the image processor(s) avoid downloading
            files from Hugging Face hub and only check the cache or local directories?
        :param kwargs: Default argument values, will be overridden by arguments specified in the URI
        :return: :py:class:`dgenerate.imageprocessors.ImageProcessor` or
            :py:class:`dgenerate.imageprocessors.ImageProcessorChain`
        """
        s = super()

        if uri is None:
            raise ValueError('uri must not be None')

        if isinstance(uri, str):
            return typing.cast(
                _imageprocessor.ImageProcessor,
                s.load(uri, device=device, local_files_only=local_files_only, **kwargs))

        paths = list(uri)

        if not paths:
            return None

        if len(paths) == 1:
            return typing.cast(
                _imageprocessor.ImageProcessor,
                s.load(paths[0], device=device, local_files_only=local_files_only, **kwargs))

        return _imageprocessorchain.ImageProcessorChain(
            typing.cast(
                _imageprocessor.ImageProcessor,
                s.load(i, device=device, local_files_only=local_files_only, **kwargs)) for i in paths)


__all__ = _types.module_all()
