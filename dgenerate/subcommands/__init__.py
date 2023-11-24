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

import dgenerate.messages as _messages
import dgenerate.plugin as _plugin
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from .exceptions import SubCommandArgumentError, SubCommandNotFoundError
from .imageprocess import ImageProcessSubCommand
from .subcommand import SubCommand
from .subcommandloader import SubCommandLoader


class SubCommandHelpUsageError(Exception):
    """
    Raised on argument parse errors in :py:func:`.sub_command_help`
    """
    pass


def remove_sub_command_arg(args: list[str]):
    """
    Returns a list of arguments with --scm/--sub-command and its singular argument removed.

    :param args: command line arguments
    :return: modified copy of the args list.
    """
    opt_l = '--sub-command'
    opt_s = '-scm'

    opts = (opt_l, opt_s)

    while any(opt in args for opt in opts):
        for opt in opts:
            if opt in args:
                index = args.index(opt)
                args = args[:index] + args[index + 2:]
    return args


def sub_command_help(names: _types.Names, plugin_module_paths: _types.Paths):
    """
    Implements ``--sub-command-help`` command line option


    :param names: arguments (processor names, or empty list)
    :param plugin_module_paths: plugin module paths to search

    :raises SubCommandHelpUsageError:
    :raises SubCommandNotFoundError:

    :return: return-code, anything other than 0 is failure
    """

    module_loader = SubCommandLoader()

    try:
        module_loader.load_plugin_modules(plugin_module_paths)
    except _plugin.ModuleFileNotFoundError as e:
        raise SubCommandHelpUsageError(e)

    if len(names) == 0:
        available = ('\n' + ' ' * 4).join(_textprocessing.quote(name) for name in module_loader.get_all_names())
        _messages.log(
            f'Available sub-commands:\n\n{" " * 4}{available}')
        return 0

    help_strs = []
    for name in names:
        try:
            help_strs.append(module_loader.get_help(name))
        except SubCommandNotFoundError:
            _messages.log(f'A sub-command with the name of "{name}" could not be found!',
                          level=_messages.ERROR)
            return 1

    for help_str in help_strs:
        _messages.log(help_str + '\n', underline=True)
    return 0


__all__ = _types.module_all()
