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

import dgenerate.promptupscalers.exceptions as _exceptions
import dgenerate.plugin as _plugin
import dgenerate.prompt as _prompt


class PromptUpscaler(_plugin.Plugin):
    """
    Abstract base class for prompt upscaler implementations.
    """

    def __init__(self,
                 loaded_by_name: str,
                 device: str,
                 **kwargs):
        """
        :param loaded_by_name: The name the prompt upscaler was loaded by
        :param kwargs: child class forwarded arguments
        """
        super().__init__(loaded_by_name=loaded_by_name,
                         argument_error_type=_exceptions.PromptUpscalerArgumentError,
                         **kwargs)

        self._device = device

    @property
    def device(self):
        """
        Device that will be used for any text processing models.
        """
        return self._device

    def upscale(self, prompt: _prompt.Prompt) -> _prompt.Prompts:
        """
        Upscale a prompt and return it modified
        :param prompt: The incoming prompt
        :return: Modified prompts, you may return multiple prompts to indicate expansion
        """
        return [prompt]
