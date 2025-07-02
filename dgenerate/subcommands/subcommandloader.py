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
import dgenerate.subcommands.exceptions as _exceptions
import dgenerate.subcommands.subcommand as _subcommand
import dgenerate.types as _types
from dgenerate.plugin import PluginArg as _Pa


class SubCommandLoader(_plugin.PluginLoader):
    """
    Loads :py:class:`dgenerate.batchprocess.ConfigRunnerPlugin` plugins.
    """

    def __init__(self):
        """"""

        # The empty string above disables sphinx inherited doc

        super().__init__(base_class=_subcommand.SubCommand,
                         description='sub-command',
                         reserved_args=[_Pa('args', type=list),
                                        _Pa('plugin-module-paths', type=list, default=None),
                                        _Pa('local-files-only', type=bool, default=False)],
                         argument_error_type=_exceptions.SubCommandArgumentError,
                         not_found_error_type=_exceptions.SubCommandNotFoundError)

        self.add_search_module_string('dgenerate.subcommands')

    def load(self, uri: _types.Uri, local_files_only: bool = False, **kwargs) -> _subcommand.SubCommand:
        """
        Load a sub-command by URI.

        :param uri: sub-command URI
        :param local_files_only: Should the sub-command avoid downloading
            files from the internet and only check the cache or local directories?
        :param kwargs: Additional plugin arguments
        :return: :py:class:`dgenerate.subcommands.SubCommand`
        """
        return typing.cast(_subcommand.SubCommand, super().load(uri, local_files_only=local_files_only, **kwargs))


__all__ = _types.module_all()
