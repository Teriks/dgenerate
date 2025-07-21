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

import dgenerate.batchprocess.configrunnerplugin as _configrunnerplugin
import dgenerate.subcommands.subcommandloader as _subcommandloader


class ToDiffusersDirective(_configrunnerplugin.ConfigRunnerPlugin):
    def __init__(self, **kwargs):
        """
        :param kwargs: plugin base class arguments
        """

        super().__init__(**kwargs)

        self.register_directive('to_diffusers', self._directive)

    def _directive(self, args: collections.abc.Sequence[str]) -> int:
        """
        Alias for: --sub-command to-diffusers

        Convert single file diffusion model checkpoints from CivitAI and elsewhere into diffusers format (a folder on disk with configuration).

        This can be useful if you want to load a single file checkpoint with quantization.

        You may also save models loaded from Hugging Face repos.

        Examples:

        \\to_diffusers "all_in_one.safetensors" --model-type sd --output model_directory

        \\to_diffusers "https://modelsite.com/all_in_one.safetensors" --model-type sdxl --output model_directory

        See: \\to_diffusers --help

        This does not cause the config to exit.
        """

        subcommand = _subcommandloader.SubCommandLoader().load(
            'to-diffusers',
            args=args,
            program_name='\\to_diffusers',
            local_files_only=self.local_files_only
        )

        return subcommand()
