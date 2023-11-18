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
import inspect
import itertools
import os
import sys
import types
import typing
from importlib.machinery import SourceFileLoader

import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types

LOADED_PLUGIN_MODULES: typing.Dict[str, types.ModuleType] = {}
"""Plugin module in memory cache"""


class Plugin:

    def __init__(self, loaded_by_name: str, argument_error_type: typing.Type[Exception] = ValueError, **kwargs):
        """
        :param loaded_by_name: The name the plugin was loaded by, will be passed by the loader.
        :param argument_error_type: This exception type will be raised by ``get_*_arg`` and friends when
            an argument is of an invalid format (they are passed as strings).  It should match the ``argument_error_type``
            given to the :py:class:`.PluginLoader` implementation being used to load the inheritor of this class.
        :param kwargs: Additional arguments that may arise when using an ``ARGS`` static signature definition
            with multiple ``NAMES`` in your implementation.
        """
        self.__loaded_by_name = loaded_by_name
        self.__argument_error_type = argument_error_type

    def get_int_arg(self, name: str, value: typing.Union[str, int, typing.Dict]) -> int:
        """
        Convert an argument value from a string to an integer.
        Throw ``argument_error_type`` if there is an error parsing the value.

        :param name: the argument name for descriptive purposes,
            and/or for specifying the dictionary key when *value*
            is a dictionary.

        :param value: an integer value as a string, or optionally a
            dictionary to get the value from using the argument *name*.

        :return: int
        """
        if isinstance(value, dict):
            value = value.get(name)
        try:
            return int(value)
        except ValueError:
            raise self.__argument_error_type(f'Argument "{name}" must be an integer value.')

    def get_float_arg(self, name: str, value: typing.Union[str, float, typing.Dict]) -> float:
        """
        Convert an argument value from a string to a float.
        Throw ``argument_error_type`` if there is an error parsing the value.

        :param name: the argument name for descriptive purposes,
            and/or for specifying the dictionary key when *value*
            is a dictionary.

        :param value: a float value as a string, or optionally a
            dictionary to get the value from using the argument *name*.

        :return: float
        """

        if isinstance(value, dict):
            value = value.get(name)
        try:
            return float(value)
        except ValueError:
            raise self.__argument_error_type(f'Argument "{name}" must be a floating point value.')

    def get_bool_arg(self, name: str, value: typing.Union[str, bool, typing.Dict]) -> bool:
        """
        Convert an argument value from a string to a boolean value.
        Throw ``argument_error_type`` if there is an error parsing the value.

        :param name: the argument name for descriptive purposes,
            and/or for specifying the dictionary key when *value*
            is a dictionary.

        :param value: a boolean value as a string, or optionally a
            dictionary to get the value from using the argument *name*.

        :return: bool
        """

        if isinstance(value, dict):
            value = value.get(name)
        try:
            return _types.parse_bool(value)
        except ValueError:
            raise self.__argument_error_type(
                f'Argument "{name}" must be a boolean value.')

    def argument_error(self, msg: str):
        raise self.__argument_error_type(msg)

    @classmethod
    def get_names(cls) -> typing.List[str]:
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
            return [cls.__name__]

    @classmethod
    def get_help(cls, loaded_by_name: str) -> str:
        """
        Get formatted help information about the plugin.

        This includes any implemented help strings and an auto formatted
        description of the plugins accepted arguments.

        :param loaded_by_name: The name used to load the plugin.
            Help may vary depending on how many names the plugin
            implementation handles and what invoking it by a certain
            name does.

        :return: Formatted help string
        """

        help_str = None
        if hasattr(cls, 'help'):
            help_str = cls.help(loaded_by_name)
            if help_str:
                help_str = _textprocessing.justify_left(help_str).strip()
        elif cls.__doc__:
            help_str = _textprocessing.justify_left(cls.__doc__).strip()

        args_with_defaults = cls.get_accepted_args_with_defaults(loaded_by_name)
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
            args_part = '\n'

        if help_str:
            wrap = \
                _textprocessing.wrap_paragraphs(
                    help_str,
                    initial_indent=' ' * 4,
                    subsequent_indent=' ' * 4,
                    width=_textprocessing.long_text_wrap_width())

            return loaded_by_name + f':{args_part}\n' + wrap
        else:
            return loaded_by_name + f':{args_part}'

    @classmethod
    def get_accepted_args(cls, loaded_by_name: str) -> typing.List[str]:
        """
        Get a list of accepted argument names for a plugin class.

        :param loaded_by_name: The name used to load the plugin.
            Arguments may vary depending on what name was used to invoke the plugin.

        :return: list of argument names
        """
        return [a[0] for a in
                cls.get_accepted_args_with_defaults(loaded_by_name)]

    @classmethod
    def get_required_args(cls, loaded_by_name: str) -> typing.List[str]:
        """
        Get a list of required arguments for this plugin class.

        :param loaded_by_name: The name used to load the plugin.
            Required arguments may vary by name used to invoke.

        :return: list of argument names
        """
        return [a[0] for a in
                cls.get_accepted_args_with_defaults(loaded_by_name) if len(a) == 1]

    @classmethod
    def get_default_args(cls, loaded_by_name: str) -> typing.List[typing.Tuple[str, typing.Any]]:
        """
        Get the names and values of arguments for this plugin that possess default values.

        :param loaded_by_name: The name used to load the plugin.
            Default arguments may vary by name used to invoke.

        :return: list of arguments with default value: (name, value)
        """
        return [a for a in
                cls.get_accepted_args_with_defaults(loaded_by_name) if len(a) == 2]

    @classmethod
    def get_accepted_args_with_defaults(cls, loaded_by_name) -> typing.List[typing.Tuple[str, typing.Any]]:
        """
        Retrieve the argument signature of a plugin implementation. As a list of tuples
        which are: (name,) or (name, default_value) depending on if a default value for the argument
        is present in the signature.

        :param loaded_by_name: The name used to load the plugin.
            Argument signature may vary by name used to invoke.

        :return: mixed list of tuples, of length 1 or 2, depending on if
            the argument has a default value. IE: (name,) or (name, default_value).
            Arguments with defaults are not guaranteed to be positioned at the end
            of this sequence.
        """

        if hasattr(cls, 'ARGS'):
            if isinstance(cls.ARGS, dict):
                if loaded_by_name not in cls.ARGS:
                    raise RuntimeError(
                        'Plugin module implementation bug, args for '
                        f'"{loaded_by_name}" not specified in ARGS dictionary.')
                args_with_defaults = cls.ARGS.get(loaded_by_name)
            else:
                args_with_defaults = cls.ARGS

            fixed_args = []
            for arg in args_with_defaults:
                if not isinstance(arg, tuple):
                    if not isinstance(arg, str):
                        raise RuntimeError(
                            f'{cls.__name__}.ARGS["{loaded_by_name}"] '
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
    def loaded_by_name(self) -> str:
        """
        The name the plugin was loaded by.

        :return: name
        """
        return self.__loaded_by_name


class ModuleFileNotFoundError(FileNotFoundError):
    """
    Raised by :py:func:`.load_modules` if a module could not be found on disk.
    """
    pass


def load_modules(paths: _types.OptionalPaths) -> typing.List[types.ModuleType]:
    """
    Load python modules from a folder or directly from a .py file.
    Cache them so that repeat requests for loading return an already loaded module.

    :raises FileNotFoundError: If a module path could not be found on disk.

    :param paths: list of folder/file paths
    :return: list of :py:class:`types.ModuleType`
    """
    r = []
    for plugin_path in paths:
        plugin_path, ext = os.path.splitext(os.path.abspath(plugin_path))

        if not ext:
            plugin_path = os.path.join(plugin_path, '__init__.py')
        else:
            plugin_path += ext

        if plugin_path in LOADED_PLUGIN_MODULES:
            mod = LOADED_PLUGIN_MODULES[plugin_path]
        else:
            try:
                mod = SourceFileLoader(plugin_path, plugin_path).load_module()
            except FileNotFoundError as e:
                raise ModuleFileNotFoundError(e)
            LOADED_PLUGIN_MODULES[plugin_path] = mod

        r.append(mod)
    return r


PluginArgumentsDef = typing.Optional[typing.List[typing.Union[typing.Tuple[str], typing.Tuple[str, typing.Any]]]]


class PluginLoader:
    def __init__(self,
                 base_class=Plugin,
                 description: str = "plugin",
                 reserved_args: PluginArgumentsDef = None,
                 argument_error_type: typing.Type[Exception] = ValueError,
                 not_found_error_type: typing.Type[Exception] = RuntimeError):
        """
        :param base_class: Base class of plugins, will be used for searching modules.
        :param description: Short plugin description / name, used in exception messages.
        :param reserved_args: Constructor arguments that are used by the plugin class which
            cannot be redefined by implementors of the plugin class. This should be a list of tuples,
            (arg-name, default-value), or (arg-name, ) for arguments that do not have a default
            and must be provided.
        :param argument_error_type: This exception type will be raised when the plugin is loaded
            with invalid URI arguments.
        :param not_found_error_type: This exception type will be raised when a plugin could
            not be located by a name specified in a loading URI.
        """
        self._classes = set()
        self._classes_by_name = dict()

        self.__reserved_args = reserved_args if reserved_args else []
        self.__argument_error_type = argument_error_type
        self.__not_found_error_type = not_found_error_type
        self.__description = description
        self.__base_class = base_class

    def add_class(self, cls: typing.Type[Plugin]):
        """p
        Add an implementation class to this loader.

        :param cls: the class
        """
        if cls in self._classes:
            # no-op
            return

        for name in cls.get_names():
            if name in self._classes_by_name:
                raise RuntimeError(
                    f'plugin class using the name {name} already exists.')
            self._classes_by_name[name] = cls

        self._classes.add(cls)

    def add_search_module_string(self, string: str) -> typing.List[typing.Type[Plugin]]:
        """
        Add a module string (in sys.modules) that will be searched for implementations.

        :param string: the module string
        :return: list of classes that were discovered
        """
        classes = self._load_classes([sys.modules[string]])
        for cls in classes:
            self.add_class(cls)
        return classes

    def add_search_module(self, module: types.ModuleType) -> typing.List[typing.Type[Plugin]]:
        """
        Directly add a module object that will be searched for implementations.

        :param module: the module object
        :return: list of classes that were discovered
        """
        classes = self._load_classes([module])
        for cls in classes:
            self.add_class(cls)
        return classes

    def load_plugin_modules(self, paths: _types.Paths) -> typing.List[typing.Type[Plugin]]:
        """
        Modules that will be loaded from disk and searched for implementations.

        Either python files, or module directory containing __init__.py

        It can be a mix of these.

        :raises dgenerate.plugin.ModuleFileNotFoundError: If a module could not be found on disk.

        :param paths: python files or module directories
        :return: list of classes that were discovered
        """
        classes = self._load_classes(load_modules(paths))

        for cls in classes:
            self.add_class(cls)

        return classes

    def _load_classes(self, modules: typing.Sequence[types.ModuleType]):
        found_classes = set()

        for mod in modules:
            def _excluded(cls):
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

    def _load(self, path, **kwargs):
        call_by_name = path.split(';', 1)[0].strip()

        plugin_class = self.get_class_by_name(call_by_name)

        parser_accepted_args = plugin_class.get_accepted_args(call_by_name)

        if 'loaded-by-name' in parser_accepted_args:
            # inheritors of base_class can't define this

            raise RuntimeError(f'"loaded-by-name" is a reserved {self.__description} module argument, '
                               'chose another argument name for your module.')

        for module_arg in self.__reserved_args:
            # reserved args always go into **kwargs
            # inheritors of base_class

            if module_arg[0] in parser_accepted_args:
                raise RuntimeError(f'"{module_arg}" is a reserved {self.__description} module argument, '
                                   'chose another argument name for your module.')

            parser_accepted_args.append(module_arg[0])

        arg_parser = _textprocessing.ConceptUriParser(
            self.__description, parser_accepted_args)

        try:
            parsed_args = arg_parser.parse_concept_uri(path).args
        except _textprocessing.ConceptPathParseError as e:
            raise self.__argument_error_type(str(e))

        args_dict = {}

        for arg in plugin_class.get_default_args(call_by_name):
            # defaults specified by the implementation class
            args_dict[_textprocessing.dashdown(arg[0])] = arg[1]

        for name_default in self.__reserved_args:
            # defaults specified by the loader
            snake_case = _textprocessing.dashdown(name_default[0])
            if len(name_default) == 2:
                name, default = name_default
                args_dict[snake_case] = parsed_args.get(name, default)
            elif len(name_default) == 1:
                name = name_default[0]
                if name in parsed_args:
                    args_dict[snake_case] = parsed_args.get(name)
                elif snake_case not in kwargs:
                    # Nothing provided this reserved argument value
                    raise self.__argument_error_type(
                        f'Missing required argument "{name}" for {self.__description} '
                        f'"{call_by_name}".')
            else:
                raise ValueError('plugin reserved_args must be tuples of length 1 or 2')

        # plugin invoker/load user arguments
        args_dict.update(kwargs)

        for k, v in parsed_args.items():
            # URI overrides everything
            args_dict[_textprocessing.dashdown(k)] = v

        # Automagic argument
        args_dict['loaded_by_name'] = call_by_name

        for arg in itertools.chain(plugin_class.get_required_args(call_by_name),
                                   (i[0] for i in self.__reserved_args if len(i) == 1)):

            if _textprocessing.dashdown(arg) not in args_dict:
                raise self.__argument_error_type(
                    f'Missing required argument "{arg}" for {self.__description} "{call_by_name}".')

        try:
            return plugin_class(**args_dict)
        except TypeError as e:
            msg = str(e)
            if 'required positional argument' in msg:
                raise self.__argument_error_type(msg)
        except self.__argument_error_type as e:
            raise self.__argument_error_type(
                f'Invalid argument given to {self.__description} "{call_by_name}": {e}')

    def get_available_classes(self) -> typing.List[typing.Type[Plugin]]:
        """
        Get classes seen by this plugin loader.

        :return: list of classes (types)
        """

        return list(self._classes)

    def get_class_by_name(self, plugin_name: _types.Name) -> typing.Type[Plugin]:
        """
        Get a plugin class by one of its names.

        IE: one of the names listed in its ``NAMES`` static attribute.

        :param plugin_name: a name associated with a plugin class
        :return: class (type)
        """

        cls = self._classes_by_name.get(plugin_name)

        if cls is None:
            raise self.__not_found_error_type(
                f'Found no {self.__description} with the name: {plugin_name}')

        return cls

    def get_all_names(self) -> _types.Names:
        """
        Get all plugin invokable names that this loader can see.

        :return: list of names (strings)
        """

        return list(self._classes_by_name.keys())

    def get_help(self, plugin_name: _types.Name) -> str:
        """
        Get a formatted help string for a plugin by one of its invokable names.

        :param plugin_name: a name associated with the plugin class
        :return: formatted string
        """

        return self.get_class_by_name(plugin_name).get_help(plugin_name)

    def load(self, uri: _types.Uri, **kwargs) -> Plugin:
        """
        Load an plugin using a URI string containing its name and arguments.

        :param uri: The URI string
        :param kwargs: default argument values, will be override by arguments specified in the URI
        :return: plugin instance
        """

        if uri is None:
            raise ValueError('uri must not be None')
        return self._load(uri, **kwargs)
