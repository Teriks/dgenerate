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

import dgenerate.latentsprocessors.latentsprocessor as _latentsprocessor
import dgenerate.latentsprocessors.exceptions as _exceptions
import dgenerate.plugin as _plugin
from dgenerate.plugin import PluginArg as _Pa
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.types as _types
import typing


class LatentsProcessorLoader(_plugin.PluginLoader):
    """
    Loads :py:class:`dgenerate.latentsprocessors.LatentsProcessor` plugins.
    """

    def __init__(self):
        super().__init__(base_class=_latentsprocessor.LatentsProcessor,
                         description='latents processor',
                         reserved_args=[_Pa('model-type', type=_enums.ModelType, required=True),
                                        _Pa('device', type=str, default='cpu'),
                                        _Pa('local-files-only', type=bool, default=False)],
                         argument_error_type=_exceptions.LatentsProcessorArgumentError,
                         not_found_error_type=_exceptions.LatentsProcessorNotFoundError)

        self.add_search_module_string('dgenerate.latentsprocessors')

    def load(self,
             uri: _types.Uri,
             device: str = 'cpu',
             local_files_only: bool = False,
             **kwargs) -> _latentsprocessor.LatentsProcessor:
        """
        Load a latents processor by URI.

        :param uri: latents processor URI
        :param device: Request a specific rendering device, default is CPU
        :param local_files_only: Should the latents processor avoid downloading
            files from Hugging Face hub and only check the cache or local directories?
        :param kwargs: Additional plugin arguments
        :return: :py:class:`dgenerate.latentsprocessors.LatentsProcessor`
        """
        return typing.cast(_latentsprocessor.LatentsProcessor, super().load(
            uri, model_type=model_type, device=device, local_files_only=local_files_only, **kwargs))


__all__ = _types.module_all() 