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
import ast
import collections.abc
import importlib.machinery
import inspect
import itertools
import os
import sys
import types
import typing

import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types

__doc__ = """
URI based plugin loading system base implementations.
"""

LOADED_PLUGIN_MODULES: dict[str, types.ModuleType] = {}
"""Plugin module in memory cache"""


class PluginArg:
    def __init__(self, name: str, type: type = typing.Any, **kwargs):
        self.name = name
        self.have_default = 'default' in kwargs
        self.default = kwargs['default'] if self.have_default else None
        self.options = kwargs.get('options', None)
        self.files = kwargs.get('files', None)
        self.type = type

    @property
    def is_hinted_optional(self):
        return _types.is_optional(self.type)

    @property
    def hinted_optional_type(self):
        return _types.get_type_of_optional(self.type, get_origin=False)

    @property
    def base_type(self) -> type:
        if self.is_hinted_optional:
            return _types.get_type(self.hinted_optional_type)
        else:
            return _types.get_type(self.type)

    def name_dashup(self) -> 'PluginArg':
        r = PluginArg(_textprocessing.dashup(self.name))
        r.have_default = self.have_default
        r.default = self.default
        r.type = r.type
        return r

    def name_dashdown(self) -> 'PluginArg':
        r = PluginArg(_textprocessing.dashdown(self.name))
        r.have_default = self.have_default
        r.default = self.default
        r.type = r.type
        return r

    def type_string(self):
        if not _types.is_typing_hint(self.type):
            return self.type.__name__
        return str(self.type).replace('typing.', '')

    def validate_non_parsed(self, value: typing.Any):
        if (self.is_hinted_optional or (self.have_default and self.default is None)) \
                and value is None:
            return None

        if self.base_type is typing.Any:
            return value

        if self.base_type is float and isinstance(value, int):
            return float(value)

        if isinstance(value, self.base_type):
            return value

        if _types.is_union(self.base_type) and isinstance(value, int):
            union_types = typing.get_args(self.type)

            has_int_type = any(t is int or _types.get_type(t) is int for t in union_types)
            if not has_int_type:
                for t in union_types:
                    if t is float or _types.get_type(t) is float:
                        return float(value)

        raise ValueError(
            f'Literal type "{value.__class__.__name__}" '
            f'does not match plugin argument "{self.name}" type '
            f'hint "{self.type_string()}".'
        )

    def parse_by_type(self, value: str | typing.Any):
        if not isinstance(value, str):
            if isinstance(value, int) and self.base_type is float:
                return float(value)
            return value

        base_type = self.base_type
        try:
            if not _types.is_typing_hint(base_type) or base_type is typing.Any:
                if base_type is bool:
                    return _types.parse_bool(value)
                if any(base_type is t for t in (list, tuple, dict, set, typing.Any)):
                    try:
                        evaled = ast.literal_eval(value)
                    except ValueError:
                        if base_type is typing.Any:
                            return value
                        raise

                    if base_type is not typing.Any and not isinstance(evaled, base_type):
                        if base_type is float and isinstance(evaled, int):
                            return float(evaled)

                        if not self.is_hinted_optional or evaled is not None:
                            raise ValueError(
                                f'Literal type "{evaled.__class__.__name__}" '
                                f'does not match plugin argument "{self.name}" type '
                                f'hint "{self.type_string()}".')
                    return evaled
                return base_type(value)

            if _types.is_union(base_type):
                try:
                    evaled = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    # string
                    evaled = value

                failures = 0
                union_types = typing.get_args(self.type)
                for t in union_types:
                    if _types.is_type(t, type(evaled)):
                        continue
                    else:
                        failures += 1

                if failures == len(union_types) and isinstance(evaled, int):
                    has_int_type = any(t is int or _types.get_type(t) is int for t in union_types)
                    if not has_int_type:
                        for t in union_types:
                            if t is float or _types.get_type(t) is float:
                                evaled = float(evaled)
                                failures = 0
                                break

                if failures == len(union_types):
                    raise ValueError(
                        f'Literal type "{evaled.__class__.__name__}" '
                        f'does not match plugin argument "{self.name}" type '
                        f'hint "{self.type_string()}".')

                return evaled

            return value
        except SyntaxError as e:
            if base_type is typing.Any:
                return value
            offset = e.offset - 1 if e.offset > 0 else 0
            raise ValueError(f'Syntax Error: {e.text[:offset]}[ERROR HERE>]{e.text[offset:]}') from e

    def __str__(self):
        default_part = f', default={repr(self.default)}' if self.have_default else ''
        return f'{self.__class__.__name__}(name="{self.name}", type={self.type}{default_part})'

    def __repr__(self):
        return str(self)


class PluginArgumentError(Exception):
    """
    Raised when a plugin encounters an error in the arguments it is loaded by.

    Or errors in arguments used for execution.
    """
    pass


class Plugin:

    def __init__(self,
                 loaded_by_name: str | None = None,
                 argument_error_type: type[PluginArgumentError] = PluginArgumentError,
                 **kwargs):
        """
        :param loaded_by_name: The name the plugin was loaded by, will be passed by the loader.
            If ``None`` is passed, the first name mentioned by the plugin implementation
            will be used. This can simplify using some plugin classes directly without
            loading them through a loader implementation.
        :param argument_error_type: This exception type will be raised upon argument errors (invalid arguments)
            when loading a plugin using a :py:class:`.PluginLoader` implementation. It should match the
            ``argument_error_type`` given to the :py:class:`.PluginLoader` implementation being used
            to load the inheritor of this class.
        :param kwargs: Additional arguments that may arise when using an ``ARGS`` static signature definition
            with multiple ``NAMES`` in your implementation.
        """

        if loaded_by_name is None:
            loaded_by_name = self.get_names()[0]

        self.__loaded_by_name = loaded_by_name
        self.__argument_error_type = argument_error_type

        # Store the constructor arguments for __str__ method
        self.__constructor_args_strs = {'loaded-by-name': loaded_by_name}

        for arg in self.__class__.get_accepted_args(loaded_by_name):
            snake_name = _textprocessing.dashdown(arg.name)
            if snake_name in kwargs:
                self.__constructor_args_strs[arg.name] = str(kwargs[snake_name])
            elif arg.have_default:
                self.__constructor_args_strs[arg.name] = str(arg.default)

    def argument_error(self, msg: str):
        """
        Return an constructed exception that is suitable for raising
        as an argument error for this plugin.

        Example: ``raise self.argument_error('oops!')``

        :param msg: exception message
        :return: the exception object, you must ``raise`` it.
        """
        return self.__argument_error_type(msg)

    @classmethod
    def get_names(cls) -> list[str]:
        """
        Get the names that this class can be loaded by.

        :return:
        """

        if hasattr(cls, 'NAMES'):
            if isinstance(cls.NAMES, str):
                return [cls.NAMES]
            else:
                return cls.NAMES
        else:
            return [_types.fullname(cls)]

    @classmethod
    def get_help(cls, loaded_by_name: str,
                 wrap_width: int | None = None,
                 include_bases: bool = False) -> str:
        """
        Get formatted help information about the plugin.

        This includes any implemented help strings and an auto formatted
        description of the plugins accepted arguments.

        :param loaded_by_name: The name used to load the plugin.
            Help may vary depending on how many names the plugin
            implementation handles and what loading it by a certain
            name does.

        :param wrap_width: wrap paragraphs to this width.

        :param include_bases: include argument names and inherited help from base classes?

        :return: Formatted help string
        """

        help_str = None
        if hasattr(cls, 'help'):
            help_str = cls.help(loaded_by_name)
            if help_str:
                help_str = inspect.cleandoc(help_str).strip()
        elif cls.__doc__:
            help_str = inspect.cleandoc(cls.__doc__).strip()

        if include_bases:
            base_help = dict()
            for base in cls.get_bases():
                if hasattr(base, 'inheritable_help'):
                    bhelp_val = base.inheritable_help(loaded_by_name)
                    if bhelp_val:
                        base_help.update(bhelp_val)

            if hasattr(cls, 'inheritable_help'):
                bhelp_val = cls.inheritable_help(loaded_by_name)
                if bhelp_val:
                    base_help.update(bhelp_val)

            hidden_args = cls.get_hidden_args(loaded_by_name)

            base_help = '\n\n'.join(
                message for arg, message in base_help.items() if arg not in hidden_args)

            if base_help:
                help_str = help_str + '\n\n' + base_help

        args_with_defaults = cls.get_accepted_args(
            loaded_by_name, include_bases=include_bases)

        arg_descriptors = []

        for arg in args_with_defaults:

            if not arg.have_default:
                arg_descriptors.append(arg.name + ': ' + arg.type_string())
            else:
                default_value = arg.default
                if isinstance(default_value, str):
                    default_value = _textprocessing.quote(default_value)
                arg_descriptors.append(f'{arg.name}: {arg.type_string()} = {default_value}')

        if arg_descriptors:
            args_part = f'\n{" " * 4}arguments:\n{" " * 8}{(chr(10) + " " * 8).join(arg_descriptors)}\n'
        else:
            args_part = '\n'

        if help_str:
            wrap = \
                _textprocessing.wrap_paragraphs(
                    help_str,
                    initial_indent=' ' * 4,
                    subsequent_indent=' ' * 4,
                    width=_textprocessing.long_text_wrap_width()
                    if wrap_width is None else wrap_width)

            return loaded_by_name + f':{args_part}\n' + wrap
        else:
            return loaded_by_name + f':{args_part}'

    @classmethod
    def get_required_args(cls, loaded_by_name: str) -> list[PluginArg]:
        """
        Get a list of required arguments for this plugin class.

        :param loaded_by_name: The name used to load the plugin.
            Required arguments may vary by name used to load.

        :return: list of argument names
        """
        return [a for a in
                cls.get_accepted_args(loaded_by_name) if not a.have_default]

    @classmethod
    def get_default_args(cls, loaded_by_name: str) -> list[PluginArg]:
        """
        Get the names and values of arguments for this plugin that possess default values.

        :param loaded_by_name: The name used to load the plugin.
            Default arguments may vary by name used to load.

        :return: list of arguments with default value: (name, value)
        """
        return [a for a in
                cls.get_accepted_args(loaded_by_name) if a.have_default]

    @classmethod
    def get_bases(cls) -> list[typing.Type['Plugin']]:
        """
        Return a list of base classes, except for :py:class:`Plugin`

        :return: list of class type objects
        """
        return list(c for c in _types.get_all_base_classes(cls) if issubclass(c, Plugin) and c is not Plugin)

    @classmethod
    def get_accepted_args_schema(cls, loaded_by_name: str, include_bases: bool = False):
        """
        Reduce the accepted arguments to a schema dict.

        Keyed by argument ``name``, content keys include:

        ``default`` contains any default value, this key may not exist if the argument has no default value.

        ``types`` contains all accepted types for the argument in string form.

        ``optional`` can the argument accept the value ``None``?

        :param loaded_by_name: Plugin loaded by name
        :param include_bases: Include all base classes except :py:class:`Plugin`?

        :return: dict
        """
        schema = dict()

        for arg in cls.get_accepted_args(loaded_by_name, include_bases=include_bases):
            entry = {}
            schema[arg.name] = entry

            def _type_name(t):
                return (str(t) if t.__module__ != 'builtins' else t.__name__).strip()

            def _resolve_union(t):
                name = _type_name(t)
                if _types.is_union(t):
                    return set(itertools.chain.from_iterable(
                        [_resolve_union(t) for t in arg.type.__args__]))
                return [name]

            type_name = _type_name(arg.type)

            if _types.is_union(arg.type):
                union_args = _resolve_union(arg.type)
                if 'NoneType' in union_args:
                    entry['optional'] = True
                    union_args.remove('NoneType')

                # deterministic order is ideal
                # for schemas in version control
                entry['types'] = list(sorted(union_args))

            else:
                entry['optional'] = False
                entry['types'] = [type_name]

            if arg.have_default:
                schema[arg.name]['default'] = arg.default

            if arg.options:
                schema[arg.name]['options'] = arg.options

            if arg.files:
                schema[arg.name]['files'] = arg.files

        return schema

    @classmethod
    def get_hidden_args(cls, loaded_by_name: str) -> set[str]:
        """
        Get argument names that have been explicitly
        hidden from use or disabled by the plugin for
        URI use. i.e. ``HIDE_ARGS``.

        These may be unsupported arguments inherited
        from a base class, or just arguments the plugin
        does not want you to use via a URI.

        These arguments can still be passed manually from
        code in the interest of maintaining a generic
        interface, but they may be ignored by the plugin
        implementation.

        .. code-block:: python

            HIDE_ARGS = ['hide-me', 'dont-use-me']

            # or

            HIDE_ARGS = {
                "plugin_name": ['hide-me', 'dont-use-me'],
                "alt_plugin_name": ['hide-me-2']
            }

        :param loaded_by_name: The name used to load the plugin.
            Argument signature may vary by name used to load.

        :return: hidden argument names
        """
        args_hidden = set()

        for base in cls.mro():
            if hasattr(base, 'HIDE_ARGS'):
                if isinstance(base.HIDE_ARGS, dict):
                    if loaded_by_name in base.HIDE_ARGS:
                        for arg in base.HIDE_ARGS[loaded_by_name]:
                            args_hidden.add(_textprocessing.dashup(arg))
                else:
                    for arg in base.HIDE_ARGS:
                        args_hidden.add(_textprocessing.dashup(arg))

        return args_hidden

    @classmethod
    def get_option_args(cls, loaded_by_name: str) -> dict[str, list]:
        """
        Get argument names that have an associated
        list of option valid values. i.e. ``OPTION_ARGS``.

        This returns metadata provided by the plugin about
        specific arguments which accept a limited set of
        values, such as shape names like ``circle`` or ``rectangle```
        etc.

        ``OPTION_ARGS`` should be in the form of a dict,
        where the keys are argument names, and the values
        are lists of valid values for that argument.

        .. code-block:: python

            OPTION_ARGS = {
                "shape": ['circle', 'rectangle', 'triangle'],
            }


        If the plugin supports multiple names for loading,
        this can be a dict of dicts, where the outer keys are
        the names used to load the plugin, and the inner
        keys are the argument names.

        .. code-block:: python

            OPTION_ARGS = {
                "plugin_name": {
                    "shape": ['circle', 'rectangle', 'triangle']
                }
            }

        These values can be inherited.

        :param loaded_by_name: The name used to load the plugin.
            Argument signature may vary by name used to load.

        :return: options argument names
        """
        args_option = dict()

        for base in cls.mro():
            if hasattr(base, 'OPTION_ARGS'):
                if isinstance(base.OPTION_ARGS, dict):
                    values = list(base.OPTION_ARGS.values())
                    if values and isinstance(values[0], dict):
                        # segregated by loaded_by_name
                        if loaded_by_name in base.OPTION_ARGS:
                            options = base.OPTION_ARGS[loaded_by_name]
                            args_option.update({_textprocessing.dashup(k): list(v) for k, v in options.items()})
                    else:
                        args_option.update({_textprocessing.dashup(k): list(v) for k, v in base.OPTION_ARGS.items()})
                else:
                    raise RuntimeError(
                        f'Plugin metadata {cls.__name__}.OPTION_ARGS must be a dict.')

        return args_option

    @classmethod
    def get_file_args(cls, loaded_by_name: str) -> dict[str, dict[str, typing.Any]]:
        """
        Get argument names that have an associated
        list of option valid file types. i.e. ``FILE_ARGS``.

        This returns metadata provided by the plugin about
        specific arguments which accept a limited set of
        file types, such as model file types or config file types.

        ``FILE_ARGS`` should be a dictionary where keys are
        argument names, for example:

        .. code-block:: python

            FILE_ARGS = {
                "model": {
                    "mode": "in/out"
                    "filetypes": [('Model', ['*.safetensors', '*.pt'])]
                }
            }

            # only accepts directories

            FILE_ARGS = {
                "model": {
                    "mode": "dir"
                }
            }

            # accepts input files or directories

            FILE_ARGS = {
                "model": {
                    "mode": ["in", "dir"]
                    "filetypes": [('Model', ['*.safetensors', '*.pt'])]
                }
            }

            # output files or directories
            # (mutually exclusive with "in" mode)

            FILE_ARGS = {
                "model": {
                    "mode": ["out", "dir"]
                    "filetypes": [('Model', ['*.safetensors', '*.pt'])]
                }
            }

        If the plugin supports multiple names for loading, this
        can be a dict of dicts, where the outer keys are the names
        used to load the plugin, and the inner keys are the argument
        names.

        .. code-block:: python

            FILE_ARGS = {
                "plugin_name": {
                    "model": {
                        "mode": "in/out"
                        "filetypes": [('Model', ['*.safetensors', '*.pt'])]
                    }
                }
            }

        These values can be inherited.

        :param loaded_by_name: The name used to load the plugin.
            Argument signature may vary by name used to load.

        :return: options argument names
        """
        args_file = dict()

        def _descriptor_format(d: dict[str, typing.Any]):
            if not isinstance(d, dict):
                raise RuntimeError(
                    f'Plugin metadata {cls.__name__}.FILE_ARGS descriptor '
                    f'values must be dicts, not {type(d).__name__}.')


            if 'mode' not in d:
                raise RuntimeError(
                    f'Plugin metadata {cls.__name__}.FILE_ARGS descriptor missing "mode" key.'
                )

            mode = d['mode']

            if isinstance(mode, list):
                if any(m not in {'in', 'out', 'dir'} for m in mode):
                    raise RuntimeError(
                        f'Plugin metadata {cls.__name__}.FILE_ARGS descriptor "mode" '
                        f'values must be "in", "out", or "dir", not: {d["mode"]}.'
                    )
                else:
                    if 'in' in mode and 'out' in mode:
                        raise RuntimeError(
                            f'Plugin metadata {cls.__name__}.FILE_ARGS descriptor "mode" '
                            f'cannot contain both "in" and "out" at the same time: {d["mode"]}.'
                        )
            elif isinstance(mode, str) and mode not in {'in', 'out', 'dir'}:
                raise RuntimeError(
                    f'Plugin metadata {cls.__name__}.FILE_ARGS descriptor "mode" '
                    f'values must be "in", "out", or "dir", not: {d["mode"]}.'
                )

            dir_only = (isinstance(mode, list) and all(m == 'dir' for m in mode)) or mode == 'dir'

            if 'filetypes' not in d and not dir_only:
                raise RuntimeError(
                    f'Plugin metadata {cls.__name__}.FILE_ARGS descriptor missing '
                    f'"filetypes" key and not in directory only "mode".'
                )

            if not dir_only:
                if not isinstance(d['filetypes'], (list, tuple)) or not all(
                        isinstance(ft, (tuple, list)) and len(ft) == 2 and isinstance(ft[0], str)
                        and isinstance(ft[1], (tuple, list))
                        for ft in d['filetypes']
                ):
                    raise RuntimeError(
                        f'Plugin metadata {cls.__name__}.FILE_ARGS descriptor "filetypes" '
                        f'must be a list/tuple of lists/tuples, where each tuple contains '
                        f'a descriptor string and a list of file extensions, not: {d["filetypes"]}.'
                    )

            return d

        for base in cls.mro():
            if hasattr(base, 'FILE_ARGS'):
                if isinstance(base.FILE_ARGS, dict):
                    values = list(base.FILE_ARGS.values())
                    if (values and isinstance(values[0], dict)
                            and isinstance(list(values[0].values())[0], dict)):
                        # segregated by loaded_by_name
                        if loaded_by_name in base.FILE_ARGS:
                            options = base.FILE_ARGS[loaded_by_name]
                            args_file.update(
                                {_textprocessing.dashup(k): _descriptor_format(v) for k, v in options.items()}
                            )
                    else:
                        args_file.update(
                            {_textprocessing.dashup(k): _descriptor_format(v) for k, v in base.FILE_ARGS.items()}
                        )
                else:
                    raise RuntimeError(
                        f'Plugin metadata {cls.__name__}.FILE_ARGS must be a dict.')

        return args_file

    @classmethod
    def get_accepted_args(cls, loaded_by_name: str, include_bases: bool = False):
        """
        Retrieve the argument signature of a plugin implementation.

        :param loaded_by_name: The name used to load the plugin.
            Argument signature may vary by name used to load.

        :param include_bases: Include all base classes except :py:class:`Plugin`?

        :return: List of argument descriptors, :py:class:`.PluginArg`
        """
        if include_bases:
            rest = itertools.chain.from_iterable(
                c._get_accepted_args(loaded_by_name)
                for c in cls.get_bases())
        else:
            rest = []

        arg_name_set = dict()
        hidden_args = cls.get_hidden_args(loaded_by_name)

        for a in itertools.chain(cls._get_accepted_args(loaded_by_name), rest):
            if a.name == 'loaded-by-name':
                continue
            if a.name in hidden_args:
                continue
            if a.name in arg_name_set:
                raise RuntimeError(
                    'Cannot handle base classes with shadowed constructor arguments.')
            else:
                arg_name_set[a.name] = a

        return list(arg_name_set.values())

    @classmethod
    def _get_accepted_args(cls, loaded_by_name: str) -> list[PluginArg]:
        if hasattr(cls, 'ARGS'):
            args_with_defaults = []
            if isinstance(cls.ARGS, dict):
                if loaded_by_name in cls.ARGS:
                    args_with_defaults = cls.ARGS.get(loaded_by_name)
            else:
                args_with_defaults = cls.ARGS

            fixed_args = []
            for arg in args_with_defaults:
                if not isinstance(arg, PluginArg):
                    raise RuntimeError(
                        f'{cls.__name__}.ARGS["{loaded_by_name}"] '
                        f'contained a non PluginArg value: {arg}')
                fixed_args.append(arg.name_dashup())
            return [] if fixed_args is None else fixed_args

        args_with_defaults = []

        option_args = cls.get_option_args(loaded_by_name)
        file_args = cls.get_file_args(loaded_by_name)
        spec = list(_types.get_accepted_args_with_defaults(cls.__init__))[1:]
        hints = typing.get_type_hints(cls.__init__)

        for arg in spec:
            name = arg[0]

            hint = hints.get(name)
            extra = {}

            name = _textprocessing.dashup(name)

            if hint is not None:
                extra['type'] = hint

            if name in option_args:
                extra['options'] = option_args[name]

            if name in file_args:
                extra['files'] = file_args[name]

            if len(arg) == 1:
                args_with_defaults.append(
                    PluginArg(name, **extra))
            else:
                args_with_defaults.append(
                    PluginArg(name, default=arg[1], **extra))

        return args_with_defaults

    @property
    def loaded_by_name(self) -> str:
        """
        The name the plugin was loaded by.

        :return: name
        """
        return self.__loaded_by_name

    def __str__(self):
        """
        Return a string representation of this plugin instance, showing all constructor arguments.

        Format: ClassName(arg1=value1, arg2=value2, ...)

        Child classes can override this method if needed.

        :return: string representation
        """
        return f'{self.__class__.__name__}({", ".join(f"{k}={v}" for k, v in self.__constructor_args_strs.items())})'

    def __repr__(self):
        return str(self)


class ModuleFileNotFoundError(FileNotFoundError):
    """
    Raised by :py:func:`.load_modules` if a module could not be found on disk.
    """
    pass


def load_modules(paths: collections.abc.Iterable[str]) -> list[types.ModuleType]:
    """
    Load python modules from a folder, directly from a .py file, or from a python module
    installed in the environment. Cache them so that repeat requests for loading return
    an already loaded module.

    :raises ModuleFileNotFoundError: If a module path could not be found on disk,
        or when a module could not be loaded from the python environment.

    :param paths: list of folder/file paths, or references to python modules installed
        in the environment

    :return: list of :py:class:`types.ModuleType`
    """
    r = []
    for plugin_path in paths:

        if os.path.exists(plugin_path):
            plugin_path, ext = os.path.splitext(os.path.abspath(plugin_path))

            if not ext:
                plugin_path = os.path.join(plugin_path, '__init__.py')
            else:
                plugin_path += ext

            if plugin_path in LOADED_PLUGIN_MODULES:
                mod = LOADED_PLUGIN_MODULES[plugin_path]
            else:
                try:
                    mod = importlib.machinery.SourceFileLoader(plugin_path, plugin_path).load_module()
                except FileNotFoundError as e:
                    raise ModuleFileNotFoundError(e) from e
                LOADED_PLUGIN_MODULES[plugin_path] = mod

            r.append(mod)
        else:

            try:
                mod = importlib.import_module(plugin_path)
            except Exception as e:
                raise ModuleFileNotFoundError(e) from e
            LOADED_PLUGIN_MODULES[plugin_path] = mod
            r.append(mod)

    return r


PluginArgumentsDef = list[PluginArg] | None


class PluginNotFoundError(Exception):
    """
    Raised when a plugin could not be located by a name.
    """
    pass


PLUGIN_PATHS = set()
"""
Plugin paths that are considered by all :py:class:`PluginLoader` instances.

This should be updated with :py:func:`import_plugins`
"""


def import_plugins(paths: collections.abc.Iterable[str]):
    """
    Set plugin paths that will be considered by all plugin loader instances.

    :param paths: environment modules, python script paths, directory paths
    """
    PLUGIN_PATHS.update(paths)


class PluginLoader:
    def __init__(self,
                 base_class=Plugin,
                 description: str = "plugin",
                 reserved_args: PluginArgumentsDef = None,
                 argument_error_type: type[PluginArgumentError] = PluginArgumentError,
                 not_found_error_type: type[PluginNotFoundError] = PluginNotFoundError):
        """
        :param base_class: Base class of plugins, will be used for searching modules.
        :param description: Short plugin description / name, used in exception messages.
        :param reserved_args: Constructor arguments that are used by the plugin class which
            cannot be redefined by implementors of the plugin class. This should be a
            list of plugin argument descriptors, :py:class:`.PluginArg`
        :param argument_error_type: This exception type will be raised when the plugin is loaded
            with invalid URI arguments.
        :param not_found_error_type: This exception type will be raised when a plugin could
            not be located by a name specified in a loading URI.
        """
        self.__classes = set()
        self.__classes_by_name = dict()
        self.__plugin_module_paths = set()

        self.__reserved_args = reserved_args if reserved_args else []
        self.__argument_error_type = argument_error_type
        self.__not_found_error_type = not_found_error_type
        self.__description = description
        self.__base_class = base_class

        self.load_plugin_modules(PLUGIN_PATHS)

    @property
    def plugin_module_paths(self) -> frozenset[str]:
        """
        Every module path ever seen by :py:meth:`PluginLoader.load_plugin_modules`.

        :return: frozen set
        """
        self.load_plugin_modules(PLUGIN_PATHS)

        return frozenset(self.__plugin_module_paths)

    def add_class(self, cls: type[Plugin]):
        """
        Add an implementation class to this loader.

        :raises RuntimeError: If the added class specifies a name that already exists in this loader.

        :param cls: the class
        """
        if cls in self.__classes or (hasattr(cls, 'HIDDEN') and getattr(cls, 'HIDDEN')):
            # no-op
            return

        for name in cls.get_names():
            if name in self.__classes_by_name:
                raise RuntimeError(
                    f'plugin class using the name {name} already exists.')
            self.__classes_by_name[name] = cls

        self.__classes.add(cls)

    def add_search_module_string(self, string: str) -> list[type[Plugin]]:
        """
        Add a module string (in sys.modules) that will be searched for implementations.

        :param string: the module string
        :return: list of classes that were newly discovered
        """
        classes = self._load_classes([sys.modules[string]])
        for cls in classes:
            self.add_class(cls)
        return classes

    def add_search_module(self, module: types.ModuleType) -> list[type[Plugin]]:
        """
        Directly add a module object that will be searched for implementations.

        :param module: the module object

        :raises ValueError: If ``module`` is not a python module object.

        :return: list of classes that were newly discovered
        """

        if not isinstance(module, types.ModuleType):
            raise ValueError('passed object in not a python module')

        classes = self._load_classes([module])
        for cls in classes:
            self.add_class(cls)
        return classes

    def load_plugin_modules(self, paths: collections.abc.Iterable[str]) -> list[type[Plugin]]:
        """
        Modules that will be loaded from disk, or the python environment, and searched for implementations.

        Either python files, or module directories containing __init__.py, or
        names of python modules installed in the environment.

        It can be a mix of these.

        :raises ModuleFileNotFoundError: If a module path could not be found on disk,
            or when a module could not be loaded from the python environment.

        :param paths: list of folder/file paths, or references to python modules installed
            in the environment

        :return: list of classes that were newly discovered
        """

        paths = set(paths)
        paths.update(PLUGIN_PATHS)

        classes = self._load_classes(load_modules(
            [path for path in paths if path not in self.__plugin_module_paths]))

        self.__plugin_module_paths.update(paths)

        for cls in classes:
            self.add_class(cls)

        return classes

    def _load_classes(self, modules: collections.abc.Iterable[types.ModuleType]):
        found_classes = set()

        for mod in modules:
            def _excluded(cls):
                try:
                    if cls in self.__classes:
                        return True
                except TypeError:
                    # handle un-hashable
                    return True

                if not inspect.isclass(cls):
                    return True

                if cls is self.__base_class:
                    return True

                if not issubclass(cls, self.__base_class):
                    return True

                if hasattr(cls, 'HIDDEN'):
                    return cls.HIDDEN
                else:
                    return False

            found_classes.update([value for value in _types.get_public_members(mod).values() if not _excluded(value)])

        return list(found_classes)

    def get_available_classes(self) -> list[type[Plugin]]:
        """
        Get classes seen by this plugin loader.

        :return: list of classes (types)
        """
        self.load_plugin_modules(PLUGIN_PATHS)

        return list(self.__classes)

    def get_class_by_name(self, plugin_name: _types.Name) -> type[Plugin]:
        """
        Get a plugin class by one of its names.

        IE: one of the names listed in its ``NAMES`` static attribute.

        :param plugin_name: a name associated with a plugin class

        :raises PluginNotFoundError: If the plugin name could not be found.

        :return: class (type)
        """

        self.load_plugin_modules(PLUGIN_PATHS)

        cls = self.__classes_by_name.get(plugin_name)

        if cls is None:
            raise self.__not_found_error_type(
                f'Found no {self.__description} with the name: {plugin_name}')

        return cls

    def get_all_names(self) -> _types.Names:
        """
        Get all plugin names that this loader can see.

        :return: list of names (strings)
        """

        self.load_plugin_modules(PLUGIN_PATHS)

        return list(self.__classes_by_name.keys())

    def get_accepted_args_schema(self, include_bases: bool = False) -> dict[str, dict[str, typing.Any]]:
        """
        Get a :py:meth:`Plugin.get_accepted_args_schema` for every plugin class, keyed by callable plugin name.

        :param include_bases: Include base class arguments? This excludes the base :py:class:`Plugin`
        :return: dict
        """
        schema = dict()

        for name in self.get_all_names():
            schema[name] = self.get_class_by_name(name).get_accepted_args_schema(
                name, include_bases=include_bases)
        return schema

    def get_help(self,
                 plugin_name: _types.Name,
                 wrap_width: int | None = None,
                 include_bases: bool = False) -> str:
        """
        Get a formatted help string for a plugin by one of its loadable names.

        :param plugin_name: a name associated with the plugin class

        :param wrap_width: wrap paragraphs to this width.

        :param include_bases: include argument names and inherited help from base classes?

        :raises PluginNotFoundError: If the plugin name could not be found.

        :return: formatted string
        """

        return self.get_class_by_name(plugin_name).get_help(
            plugin_name, wrap_width=wrap_width, include_bases=include_bases)

    def load(self, uri: _types.Uri, **kwargs) -> Plugin:
        """
        Load an plugin using a URI string containing its name and arguments.

        :param uri: The URI string
        :param kwargs: default argument values, will be override by arguments specified in the URI

        :raises ValueError: If uri is ``None``
        :raises RuntimeError: If a plugin is discovered to be using a reserved argument name upon loading it.
        :raises dgenerate.plugin.PluginArgumentError: If there is an error in the loading arguments for the plugin.
        :raises dgenerate.plugin.PluginNotFoundError: If the plugin name mentioned in the URI could not be found.

        :return: plugin instance
        """

        if uri is None:
            raise ValueError('uri must not be None')

        self.load_plugin_modules(PLUGIN_PATHS)

        loaded_by_name = uri.split(';', 1)[0].strip()

        plugin_class = self.get_class_by_name(loaded_by_name)

        parser_accepted_args = [a.name for a in plugin_class.get_accepted_args(loaded_by_name)]

        parser_raw_args = [a.name for a in plugin_class.get_accepted_args(loaded_by_name)
                           if a.base_type not in (int, str, float, bool)]

        if 'loaded-by-name' in parser_accepted_args:
            # inheritors of base_class can't define this

            raise RuntimeError(f'"loaded-by-name" is a reserved {self.__description} module argument, '
                               'chose another argument name for your module.')

        hidden_args = plugin_class.get_hidden_args(loaded_by_name)

        for module_arg in self.__reserved_args:
            # reserved args always go into **kwargs
            # inheritors of base_class

            if module_arg.name in parser_accepted_args:
                raise RuntimeError(f'"{module_arg}" is a reserved {self.__description} module argument, '
                                   'chose another argument name for your module.')

            if module_arg.name in hidden_args:
                continue

            parser_accepted_args.append(module_arg.name)

        arg_parser = _textprocessing.ConceptUriParser(
            self.__description,
            known_args=parser_accepted_args,
            args_raw=parser_raw_args)

        try:
            parsed_args = arg_parser.parse(uri).args
        except _textprocessing.ConceptUriParseError as e:
            raise self.__argument_error_type(str(e)) from e

        args_dict = {}

        for arg in plugin_class.get_default_args(loaded_by_name):
            # defaults specified by the implementation class
            args_dict[_textprocessing.dashdown(arg.name)] = arg.default

        for reserved_arg in self.__reserved_args:
            # defaults specified by the loader
            snake_case = _textprocessing.dashdown(reserved_arg.name)

            try:
                if reserved_arg.have_default:
                    args_dict[snake_case] = reserved_arg.parse_by_type(
                        parsed_args.get(reserved_arg.name, reserved_arg.default))
                else:
                    if reserved_arg.name in parsed_args:
                        args_dict[snake_case] = reserved_arg.parse_by_type(
                            parsed_args.get(reserved_arg.name))
                    elif snake_case not in kwargs:
                        # Nothing provided this reserved argument value
                        if reserved_arg.is_hinted_optional:
                            args_dict[snake_case] = None
                        else:
                            raise self.__argument_error_type(
                                f'Missing required argument "{reserved_arg.name}" for {self.__description} '
                                f'"{loaded_by_name}".')
            except ValueError as e:
                raise self.__argument_error_type(
                    f'Argument "{reserved_arg.name}" must match type: "{reserved_arg.type_string()}". '
                    f'Failure cause: {str(e).strip()}')

        accepted_args = {_textprocessing.dashup(n.name): n for n in
                         itertools.chain(plugin_class.get_accepted_args(loaded_by_name=loaded_by_name),
                                         self.__reserved_args)}

        # plugin user in code arguments
        for k, v in kwargs.items():
            try:
                arg = accepted_args[_textprocessing.dashup(k)]
            except KeyError:
                raise self.__argument_error_type(
                    f'Unknown plugin argument: "{k}"'
                )
            try:
                args_dict[k] = arg.validate_non_parsed(v)
            except ValueError as e:
                raise self.__argument_error_type(
                    f'Argument "{k}" must match type: "{arg.type_string()}". '
                    f'Failure cause: {str(e).strip()}')

        for k, v in parsed_args.items():
            # URI overrides everything
            arg = accepted_args[k]
            try:
                args_dict[_textprocessing.dashdown(k)] = arg.parse_by_type(v)
            except ValueError as e:
                raise self.__argument_error_type(
                    f'Argument "{k}" must match type: "{arg.type_string()}". '
                    f'Failure cause: {str(e).strip()}')

        # Automagic argument
        args_dict['loaded_by_name'] = loaded_by_name

        for arg_name, plugin_arg in ((k, v) for k, v in accepted_args.items() if not v.have_default):
            snake_case = _textprocessing.dashdown(arg_name)
            if snake_case not in args_dict:
                if plugin_arg.is_hinted_optional:
                    args_dict[snake_case] = None
                else:
                    raise self.__argument_error_type(
                        f'Missing required argument "{arg_name}" for {self.__description} "{loaded_by_name}".')

        try:
            return plugin_class(**args_dict)
        except self.__argument_error_type as e:
            raise self.__argument_error_type(
                f'Invalid argument given to {self.__description} '
                f'"{loaded_by_name}": {str(e).strip()}')

    def loader_help(self,
                    names: _types.Names,
                    plugin_module_paths: _types.OptionalPaths = None,
                    title='plugin',
                    title_plural='plugins',
                    throw=False,
                    log_error=True,
                    include_bases: bool = False):
        """
        Implements ``--sub-command-help`` and ``--image-processor-help``
        command line options for example.


        :param names: arguments (sub-command names, or empty list)
        :param plugin_module_paths: extra plugin module paths to search
        :param title: plugin title, used in messages
        :param title_plural: plural plugin title, used in messages
        :param throw: throw on error?
        :param log_error: log errors to stderr?
        :param include_bases: include argument names from base classes?

        :raises PluginNotFoundError: ``names`` contained an unknown plugin name
        :raises ModuleFileNotFoundError: ``plugin_module_paths`` contained a missing module

        :return: return-code, anything other than 0 is failure
        """

        self.load_plugin_modules(PLUGIN_PATHS)

        if plugin_module_paths is not None:
            try:
                self.load_plugin_modules(plugin_module_paths)
            except ModuleFileNotFoundError as e:
                if log_error:
                    _messages.error(
                        f'Plugin module could not be found: {str(e).strip()}'
                    )
                if throw:
                    raise
                return 1

        if len(names) == 0:
            available = ('\n' + ' ' * 4).join(_textprocessing.quote(name) for name in sorted(self.get_all_names()))
            _messages.log(
                f'Available {title_plural}:\n\n{" " * 4}{available}')
            return 0

        help_strs = []
        for name in names:
            try:
                help_strs.append(self.get_help(name, include_bases=include_bases))
            except PluginNotFoundError:
                if log_error:
                    _messages.error(
                        f'An {title} with the name of "{name}" could not be found.'
                    )
                if throw:
                    raise
                return 1

        for help_str in help_strs:
            _messages.log(help_str + '\n', underline=True)
        return 0
