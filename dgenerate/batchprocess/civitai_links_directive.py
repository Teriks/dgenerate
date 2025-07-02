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
import dgenerate.messages as _messages

class CivitAILinksDirective(_configrunnerplugin.ConfigRunnerPlugin):
    def __init__(self, **kwargs):
        """
        :param kwargs: plugin base class arguments
        """

        super().__init__(**kwargs)

        self.register_directive('civitai_links', self._directive)

    def _directive(self, args: collections.abc.Sequence[str]) -> int:
        """
        Alias for: --sub-command civitai-links

        This allows listing all the hard links to models on a CivitAI model page.

        See: \\civitai_links --help

        This does not cause the config to exit.
        """

        if self.local_files_only:
            _messages.error('The \\civitai_links directive does not support --offline-mode')
            return 1

        subcommand = _subcommandloader.SubCommandLoader().load(
            'civitai-links',
            args=args,
            program_name='\\civitai_links'
        )

        return subcommand()
