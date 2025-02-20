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

import dgenerate.promptupscalers.promptupscaler as _promptupscaler
import dgenerate.promptupscalers.exceptions as _exceptions
from dgenerate.plugin import PluginArg as _Pa
import dgenerate.plugin as _plugin
import dgenerate.types as _types
import typing


class PromptUpscalerLoader(_plugin.PluginLoader):
    """
    Loads :py:class:`dgenerate.promptupscalers.PromptUpscaler` plugins.
    """

    def __init__(self):

        # The empty string above disables sphinx inherited doc
        super().__init__(base_class=_promptupscaler.PromptUpscaler,
                         description='prompt upscaler',
                         reserved_args=[_Pa('device', type=str, default='cpu'),
                                        _Pa('local-files-only', type=bool, default=False)],
                         argument_error_type=_exceptions.PromptUpscalerArgumentError,
                         not_found_error_type=_exceptions.PromptUpscalerNotFoundError)

        self.add_search_module_string('dgenerate.promptupscalers')

    def load(self, uri: _types.Uri,
             device: str = 'cpu',
             local_files_only: bool = False,
             **kwargs
             ) -> _promptupscaler.PromptUpscaler:
        """
        :param uri: prompt upscaler URI
        :param device: Device used for any text processing models.
        :param local_files_only: Should the prompt upscaler avoid downloading
            files from Hugging Face hub and only check the cache or local directories?
        :param kwargs: Additional plugin arguments
        :return: :py:class:`dgenerate.promptupscalers.PromptUpscaler`
        """
        return typing.cast(_promptupscaler.PromptUpscaler, super().load(
            uri, device=device, local_files_only=local_files_only, **kwargs))
