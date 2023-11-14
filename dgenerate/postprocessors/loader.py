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

import dgenerate.plugin as _plugin
import dgenerate.postprocessors.exceptions as _exceptions
import dgenerate.postprocessors.postprocessor as _postprocessor
import dgenerate.postprocessors.postprocessorchain as _postprocessorchain
import dgenerate.types as _types


class Loader(_plugin.PluginLoader):
    def __init__(self):
        """"""

        # The empty string above disables sphinx inherited doc

        super().__init__(base_class=_postprocessor.ImagePostprocessor,
                         description='postprocessor',
                         reserved_args=[('device',)],
                         argument_error_type=_exceptions.ImagePostprocessorArgumentError,
                         not_found_error_type=_exceptions.ImagePostprocessorNotFoundError)

        self.add_search_module_string('dgenerate.postprocessors')

    def load(self, uri: typing.Union[_types.Uri, typing.Iterable[_types.Uri]], device: str = 'cpu') -> \
            typing.Union[_postprocessor.ImagePostprocessor, _postprocessorchain.ImagePostprocessorChain, None]:
        """
        Load an image postprocessor or multiple image postprocessors. They are loaded by URI, which
        is their name and any module arguments, for example: ``upscaler;model=esrgan_x4.pth;tile=512``

        Specifying multiple postprocessors with a list will create an image postprocessor chain object.

        :raises RuntimeError: if more than one class was found using the provided name mentioned in the URI.
        :raises ImagePostprocessorNotFoundError: if the name mentioned in the URI could not be found.
        :raises ImagePostprocessorArgumentError: if the URI contained invalid arguments.


        :param uri: Postprocessor URI or list of URIs
        :param device: Request a specific rendering device, default is CPU
        :return: :py:class:`dgenerate.postprocessors.ImagePostprocessor` or
            :py:class:`dgenerate.postprocessors.ImagePostprocessorChain`
        """
        s = super()

        if uri is None:
            raise ValueError('uri must not be None')

        if isinstance(uri, str):
            return typing.cast(_postprocessor.ImagePostprocessor, s.load(uri, device=device))

        paths = list(uri)

        if not paths:
            return None

        if len(paths) == 1:
            return typing.cast(_postprocessor.ImagePostprocessor, s.load(paths[0], device=device))

        return _postprocessorchain.ImagePostprocessorChain(
            typing.cast(_postprocessor.ImagePostprocessor, s.load(i, device=device)) for i in paths)


__all__ = _types.module_all()
