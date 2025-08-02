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
import abc
import collections.abc

import dgenerate.plugin as _plugin
import dgenerate.subcommands.exceptions as _exceptions


class SubCommand(_plugin.Plugin, abc.ABC):
    """
    Abstract base class for sub-command implementations.
    """

    # you cannot specify these via a URI
    HIDE_ARGS = ['plugin-module-paths', 'args', 'local-files-only']

    def __init__(self,
                 loaded_by_name: str,
                 plugin_module_paths: list[str],
                 args: list[str],
                 local_files_only: bool = False,
                 **kwargs):
        """
        :param loaded_by_name: The name the sub-command was loaded by
        :param plugin_module_paths: Plugin module paths to search
        :param args: Command line arguments for the sub-command
        :param local_files_only: if ``True``, the sub-command should never try to download
            models from the internet automatically, and instead only look
            for them in cache / on disk.
        :param kwargs: child class forwarded arguments
        """
        super().__init__(loaded_by_name=loaded_by_name,
                         argument_error_type=_exceptions.SubCommandArgumentError,
                         **kwargs)
        self.__args = list(args)
        self.__plugin_module_paths = frozenset(plugin_module_paths) if plugin_module_paths else frozenset()
        self.__local_files_only = local_files_only

    @property
    def plugin_module_paths(self) -> frozenset[str]:
        return self.__plugin_module_paths

    @property
    def args(self) -> collections.abc.Sequence:
        return self.__args

    @property
    def local_files_only(self) -> bool:
        """
        Is this sub-command only going to look for resources such as models in cache / on disk?
        """
        return self.__local_files_only

    @abc.abstractmethod
    def __call__(self) -> int:
        """
        Inheritor must implement.

        Perform the sub-command functionality and return a return code.

        :return: return code.
        """
        return 0
