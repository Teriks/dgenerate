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
import os
import types
import typing
from importlib.machinery import SourceFileLoader

import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types

LOADED_PLUGIN_MODULES: typing.Dict[str, types.ModuleType] = {}
"""Plugin module in memory cache"""


class InvokablePlugin:

    @classmethod
    def get_names(cls) -> typing.List[str]:
        """
        Get the names that this class can be loaded / invoked by.

        :return:
        """

        if hasattr(cls, 'NAMES'):
            if isinstance(cls.NAMES, str):
                return [cls.NAMES]
            else:
                return cls.NAMES
        else:
            return [cls.__name__]

    @classmethod
    def get_help(cls, called_by_name: str) -> str:
        """
        Get formatted help information about the plugin.

        This includes any implemented help strings and an auto formatted
        description of the plugins accepted arguments.

        :param called_by_name: The name used to invoke the plugin.
            Help may vary depending on how many names the plugin
            implementation handles and what invoking it by a certain
            name does.

        :return: Formatted help string
        """

        help_str = None
        if hasattr(cls, 'help'):
            help_str = cls.help(called_by_name)
            if help_str:
                help_str = _textprocessing.justify_left(help_str).strip()
        elif cls.__doc__:
            help_str = _textprocessing.justify_left(cls.__doc__).strip()

        args_with_defaults = cls.get_accepted_args_with_defaults(called_by_name)
        arg_descriptors = []

        for arg in args_with_defaults:
            if len(arg) == 1:
                arg_descriptors.append(arg[0])
            else:
                default_value = arg[1]
                if isinstance(default_value, str):
                    default_value = _textprocessing.quote(default_value)
                arg_descriptors.append(f'{arg[0]}={default_value}')

        if arg_descriptors:
            args_part = f'\n{" " * 4}arguments:\n{" " * 8}{(chr(10) + " " * 8).join(arg_descriptors)}\n'
        else:
            args_part = ''

        if help_str:
            wrap = \
                _textprocessing.wrap_paragraphs(
                    help_str,
                    initial_indent=' ' * 4,
                    subsequent_indent=' ' * 4,
                    width=_textprocessing.long_text_wrap_width())

            return called_by_name + f':{args_part}\n' + wrap
        else:
            return called_by_name + f':{args_part}'

    @classmethod
    def get_accepted_args(cls, called_by_name: str) -> typing.List[str]:
        """
        Get a list of accepted argument names for a plugin class.

        :param called_by_name: The name used to invoke the plugin.
            Arguments may vary depending on what name was used to invoke the plugin.

        :return: list of argument names
        """
        return [a[0] for a in
                cls.get_accepted_args_with_defaults(called_by_name)]

    @classmethod
    def get_required_args(cls, called_by_name: str) -> typing.List[str]:
        """
        Get a list of required arguments for this plugin class.

        :param called_by_name: The name used to invoke the plugin.
            Required arguments may vary by name used to invoke.

        :return: list of argument names
        """
        return [a[0] for a in
                cls.get_accepted_args_with_defaults(called_by_name) if len(a) == 1]

    @classmethod
    def get_default_args(cls, called_by_name: str) -> typing.List[typing.Tuple[str, typing.Any]]:
        """
        Get the names and values of arguments for this plugin that possess default values.

        :param called_by_name: The name used to invoke the plugin.
            Default arguments may vary by name used to invoke.

        :return: list of arguments with default value: (name, value)
        """
        return [a for a in
                cls.get_accepted_args_with_defaults(called_by_name) if len(a) == 2]

    @classmethod
    def get_accepted_args_with_defaults(cls, called_by_name) -> typing.List[typing.Tuple[str, typing.Any]]:
        """
        Retrieve the argument signature of a plugin implementation. As a list of tuples
        which are: (name,) or (name, default_value) depending on if a default value for the argument
        is present in the signature.

        :param called_by_name: The name used to invoke the plugin.
            Argument signature may vary by name used to invoke.

        :return: mixed list of tuples, of length 1 or 2, depending on if
            the argument has a default value. IE: (name,) or (name, default_value).
            Arguments with defaults are not guaranteed to be positioned at the end
            of this sequence.
        """

        if hasattr(cls, 'ARGS'):
            if isinstance(cls.ARGS, dict):
                if called_by_name not in cls.ARGS:
                    raise RuntimeError(
                        'InvokablePlugin module implementation bug, args for '
                        f'"{called_by_name}" not specified in ARGS dictionary.')
                args_with_defaults = cls.ARGS.get(called_by_name)
            else:
                args_with_defaults = cls.ARGS

            fixed_args = []
            for arg in args_with_defaults:
                if not isinstance(arg, tuple):
                    if not isinstance(arg, str):
                        raise RuntimeError(
                            f'{cls.__name__}.ARGS["{called_by_name}"] '
                            f'contained a non tuple or str value: {arg}')
                    fixed_args.append((_textprocessing.dashup(arg),))
                elif len(arg) == 1:
                    fixed_args.append((_textprocessing.dashup(arg[0]),))
                else:
                    fixed_args.append((_textprocessing.dashup(arg[0]), arg[1]))

            return [] if fixed_args is None else fixed_args

        args_with_defaults = []

        spec = _types.get_accepted_args_with_defaults(cls.__init__)

        for arg in spec:
            if len(arg) == 1:
                args_with_defaults.append((_textprocessing.dashup(arg[0]),))
            else:
                args_with_defaults.append((_textprocessing.dashup(arg[0]), arg[1]))

        return args_with_defaults

    @property
    def called_by_name(self) -> str:
        """
        The name the plugin was invoked by.

        :return: name
        """
        return self.__called_by_name

    def __init__(self, called_by_name: str, **kwargs):
        self.__called_by_name = called_by_name


def load_modules(paths: _types.OptionalPaths) -> typing.List[types.ModuleType]:
    """
    Load python modules from a folder or directly from a .py file.
    Cache them so that repeat requests for loading return an already loaded module.

    :param paths: list of folder/file paths
    :return: list of :py:class:`types.ModuleType`
    """
    r = []
    for plugin_path in paths:
        name, ext = os.path.splitext(os.path.basename(plugin_path))
        if name in LOADED_PLUGIN_MODULES:
            mod = LOADED_PLUGIN_MODULES[name]
        elif ext:
            mod = SourceFileLoader(name, plugin_path).load_module()
        else:
            mod = SourceFileLoader(name,
                                   os.path.join(plugin_path, '__init__.py')).load_module()

        LOADED_PLUGIN_MODULES[name] = mod
        r.append(mod)
    return r
