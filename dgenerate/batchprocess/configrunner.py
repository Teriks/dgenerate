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
import glob
import shlex
import typing

import dgenerate
import dgenerate.arguments as _arguments
import dgenerate.batchprocess.batchprocessor as _batchprocessor
import dgenerate.batchprocess.configrunnerpluginloader as _configrunnerpluginloader
import dgenerate.messages as _messages
import dgenerate.prompt as _prompt
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types


class ConfigRunner(_batchprocessor.BatchProcessor):
    """
    A :py:class:`.BatchProcessor` that can run dgenerate batch processing configs from a string or file.
    """

    @property
    def plugin_module_paths(self) -> frozenset[str]:
        """
        Set of plugin module paths if they were injected into the config runner by ``--plugin-modules``
        or used in a ``\\import_plugins`` statement in a config.

        :return: a set of paths, may be empty but not ``None``
        """
        return frozenset(self._plugin_module_paths)

    def __init__(self,
                 injected_args: typing.Optional[collections.abc.Sequence[str]] = None,
                 plugin_loader: _configrunnerpluginloader.ConfigRunnerPluginLoader = None,
                 version: typing.Union[_types.Version, str] = dgenerate.__version__):
        """

        :raises dgenerate.plugin.ModuleFileNotFoundError: If a module path parsed from
            ``--plugin-modules`` in ``injected_args`` could not be found on disk.


        :param injected_args: dgenerate command line arguments in the form of a list, see: shlex module, or sys.argv.
            These arguments will be injected at the end of every dgenerate invocation in the config. ``--plugin-modules``
            are parsed from ``injected_args`` and added to ``plugin_loader``. If ``-v/--verbose`` is present in ``injected_args``
            debugging output will be enabled globally while the config runs, and not just for invocations.

        :param plugin_loader: Batch processor plugin loader, if one is not provided one will be created.
        :param version: Config version for ``#! dgenerate x.x.x`` version checks, defaults to ``dgenerate.__version__``
        """

        def invoker(args):
            return 0

        super().__init__(
            invoker=invoker,
            name='dgenerate',
            version=version,
            injected_args=injected_args if injected_args else [])

        def _format_prompt(prompt):
            pos = prompt.positive
            neg = prompt.negative

            if pos is None:
                raise _batchprocessor.BatchProcessError('Attempt to format a prompt with no positive prompt value.')

            if pos and neg:
                return shlex.quote(f"{pos}; {neg}")
            return shlex.quote(pos)

        def format_prompt(string_or_iterable):
            if isinstance(string_or_iterable, _prompt.Prompt):
                return _format_prompt(string_or_iterable)
            return ' '.join(_format_prompt(p) for p in string_or_iterable)

        def quote(string_or_iterable):
            if isinstance(string_or_iterable, str):
                return shlex.quote(str(string_or_iterable))
            return ' '.join(shlex.quote(str(s)) for s in string_or_iterable)

        def unquote(string_or_iterable):
            if isinstance(string_or_iterable, str):
                return shlex.split(str(string_or_iterable))
            return [shlex.split(str(s)) for s in string_or_iterable]

        def last(list_or_iterable):
            if isinstance(list_or_iterable, list):
                return list_or_iterable[-1]
            try:
                *_, last_item = list_or_iterable
            except ValueError:
                raise _batchprocessor.BatchProcessError(
                    'Usage of template function "last" on an empty iterable.')
            return last_item

        def first(iterable):
            try:
                v = next(iter(iterable))
            except StopIteration:
                raise _batchprocessor.BatchProcessError(
                    'Usage of template function "first" on an empty iterable.')
            return v

        self.template_variables = {
            'injected_args': self.injected_args,
            'injected_device': _arguments.parse_device(self.injected_args)[0],
            'injected_verbose': _arguments.parse_verbose(self.injected_args)[0],
            'injected_plugin_modules': _arguments.parse_plugin_modules(self.injected_args)[0],
            'saved_modules': dict(),
            'glob': glob
        }

        self.template_variables = dict()

        self.reserved_template_variables = set(self.template_variables.keys())

        self.template_functions = {
            'unquote': unquote,
            'quote': quote,
            'format_prompt': format_prompt,
            'format_size': _textprocessing.format_size,
            'last': last,
            'first': first
        }

        self.directives = {
            'templates_help': self._templates_help_directive,
            'exit': self._exit_directive
        }

        self.plugin_loader = \
            _configrunnerpluginloader.ConfigRunnerPluginLoader() if \
                plugin_loader is None else plugin_loader

        self._plugin_module_paths = set()

        if injected_args:
            self._plugin_module_paths.update(_arguments.parse_plugin_modules(injected_args)[0])
            self.plugin_loader.load_plugin_modules(self._plugin_module_paths)

        for plugin_class in self.plugin_loader.get_available_classes():
            self.plugin_loader.load(plugin_class.get_names()[0],
                                    config_runner=self)

        self.directives['import_plugins'] = self._import_plugins_directive

    def _import_plugins_directive(self, plugin_paths: collections.abc.Sequence[str]):
        """
        Imports plugins from within a config, this imports config plugins as well as image processor plugins.
        This has an identical effect to the --plugin-modules argument. You may specify multiple plugin
        module directories or python files containing plugin implementations.
        """

        if len(plugin_paths) == 0:
            raise _batchprocessor.BatchProcessError(
                '\\import_plugins must be used with at least one argument.')

        self._plugin_module_paths.update(plugin_paths)
        new_classes = self.plugin_loader.load_plugin_modules(plugin_paths)
        for cls in new_classes:
            self.plugin_loader.load(cls.get_names()[0],
                                    config_runner=self)

        return 0

    def _exit_directive(self, args: collections.abc.Sequence[str]):
        """
        Causes the dgenerate process to exit with a specific return code.
        This directive accepts one argument, the return code, which is optional
        and 0 by default. It must be an integer value.
        """
        if (len(args)) == 0:
            exit(0)

        try:
            return_code = int(args[0])
        except ValueError:
            raise _batchprocessor.BatchProcessError(
                f'\\exit return code must be an integer value, received: {args[0]}')

        exit(return_code)

    def generate_directives_help(self, directive_names: typing.Optional[typing.Collection[str]] = None):
        """
        Generate the help string for ``--directives-help``


        :param directive_names: Display help for specific directives, if ``None`` or ``[]`` is specified, display all.

        :raise ValueError: if given directive names could not be found

        :return: help string
        """

        directives: dict[str, typing.Union[str, typing.Callable]] = self.directives.copy()

        directives.update({
            'set': 'Sets a template variable, accepts two arguments, the variable name and the value. '
                   'Attempting to set a reserved template variable such as those pre-defined by dgenerate '
                   'will result in an error. The second argument is accepted as a raw value, it is not shell '
                   'parsed in any way, only striped of leading and trailing whitespace.',
            'print': 'Prints all content to the right to stdout, no shell parsing of the argument occurs.'
        })

        if len(directive_names) == 0:

            help_string = f'Available config directives:' + '\n\n'
            help_string += '\n'.join((' ' * 4) + _textprocessing.quote('\\' + i) for i in directives.keys())

        else:
            help_string = ''

            directive_names = {n.lstrip('\\') for n in directive_names}

            if directive_names is not None and len(directive_names) > 0:
                found = dict()
                not_found = []
                for n in directive_names:
                    if n not in directives:
                        not_found.append(n)
                        continue
                    found[n] = directives[n]
                if not_found:
                    raise ValueError(
                        f'No directives named: {_textprocessing.oxford_comma(not_found, "or")}')
                directives = found

            def docs():
                for name, impl in directives.items():
                    if isinstance(impl, str):
                        doc = impl
                    else:
                        doc = _textprocessing.justify_left(impl.__doc__).strip() \
                            if impl.__doc__ is not None else 'No documentation provided.'
                    doc = \
                        _textprocessing.wrap_paragraphs(
                            doc,
                            initial_indent=' ' * 4,
                            subsequent_indent=' ' * 4,
                            width=_textprocessing.long_text_wrap_width())
                    yield name + _textprocessing.underline(':\n\n' + doc + '\n')

            help_string += '\n'.join(docs())

        return help_string

    def generate_template_variables_help(self,
                                         variable_names: typing.Optional[typing.Collection[str]] = None,
                                         show_values: bool = True):
        """
        Generate a help string describing available template variables, their types, and values for use in batch processing.

        This is used for ``--templates-help``

        :param variable_names: Display help for specific variables, if ``None`` or ``[]`` is specified, display all.

        :param show_values: Show the value of the template variable or just the name?

        :raise ValueError: if given variable names could not be found

        :return: a human-readable description of all template variables
        """

        values = dict()

        for k, v in self.template_variables.items():
            if k not in values:
                values[k] = (v.__class__, v)

        if variable_names is not None and len(variable_names) > 0:
            found = dict()
            not_found = []
            for n in variable_names:
                if n not in values:
                    not_found.append(n)
                    continue
                found[n] = values[n]
            if not_found:
                raise ValueError(
                    f'No template variables named: {_textprocessing.oxford_comma(not_found, "or")}')
            values = found

        if len(values) > 1:
            header = 'Config template variables are'
        else:
            header = 'Config template variable is'

        help_string = f'{header}:' + '\n\n'

        def wrap(val):
            return _textprocessing.wrap(
                str(val),
                width=_textprocessing.long_text_wrap_width(),
                subsequent_indent=' ' * 17)

        return help_string + '\n'.join(
            ' ' * 4 + f'Name: {_textprocessing.quote(i[0])}\n{" " * 8}'
                      f'Type: {i[1][0]}' + (f'\n{" " * 8}Value: {wrap(i[1][1])}' if show_values else '') for i in
            values.items())

    def _templates_help_directive(self, args: collections.abc.Sequence[str]):
        """
        Prints all template variables in the global scope, with their types and values.

        This does not cause the config to exit.
        """
        _messages.log(self.generate_template_variables_help(args) + '\n')
        return 0


__all__ = _types.module_all()
