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
import glob
import shlex
import types
import typing

import dgenerate
import dgenerate.arguments as _arguments
import dgenerate.batchprocess.batchprocessor as _batchprocessor
import dgenerate.batchprocess.configrunnerpluginloader as _configrunnerpluginloader
import dgenerate.invoker as _invoker
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.prompt as _prompt
import dgenerate.renderloop as _renderloop
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types


class ConfigRunner(_batchprocessor.BatchProcessor):
    """
    A :py:class:`.BatchProcessor` that can run dgenerate batch processing configs from a string or file.
    """

    @property
    def plugin_module_paths(self) -> typing.FrozenSet[str]:
        """
        Set of plugin module paths if they were injected into the config runner by ``-pm/--plugin-modules``
        or used in a ``\\import_plugins`` statement in a config.

        :return: a set of paths, may be empty but not ``None``
        """
        return frozenset(self._plugin_module_paths)

    def __init__(self,
                 injected_args: typing.Optional[typing.Sequence[str]] = None,
                 render_loop: typing.Optional[_renderloop.RenderLoop] = None,
                 plugin_loader: _configrunnerpluginloader.ConfigRunnerPluginLoader = None,
                 version: typing.Union[_types.Version, str] = dgenerate.__version__,
                 throw: bool = False):
        """

        :raises dgenerate.plugin.ModuleFileNotFoundError: If a module path parsed from
            ``-pm/--plugin-modules`` in ``injected_args`` could not be found on disk.


        :param injected_args: dgenerate command line arguments in the form of a list, see: shlex module, or sys.argv.
            These arguments will be injected at the end of every dgenerate invocation in the config. ``-pm/--plugin-modules``
            are parsed from ``injected_args`` and added to ``plugin_loader``. If ``-v/--verbose`` is present in ``injected_args``
            debugging output will be enabled globally while the config runs, and not just for invocations.

        :param render_loop: RenderLoop instance, if ``None`` is provided one will be created.
        :param plugin_loader: Batch processor plugin loader, if one is not provided one will be created.
        :param version: Config version for ``#! dgenerate x.x.x`` version checks, defaults to ``dgenerate.__version__``
        :param throw: Whether to throw exceptions from :py:func:`dgenerate.invoker.invoke_dgenerate` or handle them.
            If you set this to ``True`` exceptions will propagate out of dgenerate invocations instead of a :py:exc:`dgenerate.batchprocess.BatchProcessError`
            being raised by the created :py:class:`dgenerate.batchprocess.BatchProcessor`. A line number where the error
            occurred can be obtained using :py:attr:`dgenerate.batchprocess.BatchProcessor.current_line`.
        """

        def invoker(args):
            try:
                return_code = \
                    _invoker.invoke_dgenerate(args,
                                              render_loop=self.render_loop,
                                              throw=throw)
                if return_code == 0:
                    self.template_variables.update(self._generate_template_variables())
                return return_code
            finally:
                self.render_loop.model_extra_modules = None

        super().__init__(
            invoker=invoker,
            name='dgenerate',
            version=version,
            injected_args=injected_args if injected_args else [])

        if render_loop is None:
            render_loop = _renderloop.RenderLoop()

        self.render_loop = render_loop

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
            'saved_modules': dict(),
            'glob': glob
        }

        self.template_variables = self._generate_template_variables()

        self.template_functions = {
            'unquote': unquote,
            'quote': quote,
            'format_prompt': format_prompt,
            'format_size': _textprocessing.format_size,
            'last': last,
            'first': first
        }

        def return_zero(func):
            def wrap(args):
                func()
                return 0

            return wrap

        self.directives = {
            'templates_help': self._templates_help_directive,
            'clear_model_cache': return_zero(_pipelinewrapper.clear_model_cache),
            'clear_pipeline_cache': return_zero(_pipelinewrapper.clear_pipeline_cache),
            'clear_vae_cache': return_zero(_pipelinewrapper.clear_vae_cache),
            'clear_control_net_cache': return_zero(_pipelinewrapper.clear_control_net_cache),
            'save_modules': self._save_modules_directive,
            'use_modules': self._use_modules_directive,
            'clear_modules': self._clear_modules_directive,
            'gen_seeds': self._gen_seeds_directive,
        }

        self.plugin_loader = \
            _configrunnerpluginloader.ConfigRunnerPluginLoader() if \
                plugin_loader is None else plugin_loader

        self._plugin_module_paths = set()

        if injected_args:
            self._plugin_module_paths.update(_arguments.parse_plugin_modules(injected_args)[0])
            self.plugin_loader.load_plugin_modules(self._plugin_module_paths)
            self.render_loop.image_processor_loader.load_plugin_modules(self._plugin_module_paths)

        for plugin_class in self.plugin_loader.get_available_classes():
            self.plugin_loader.load(plugin_class.get_names()[0],
                                    config_runner=self,
                                    render_loop=self.render_loop)

        self.directives['import_plugins'] = self._import_plugins_directive

    def _import_plugins_directive(self, plugin_paths):
        self._plugin_module_paths.update(plugin_paths)
        self.render_loop.image_processor_loader.load_plugin_modules(plugin_paths)
        new_classes = self.plugin_loader.load_plugin_modules(plugin_paths)
        for cls in new_classes:
            self.plugin_loader.load(cls.get_names()[0],
                                    config_runner=self,
                                    render_loop=self.render_loop)

        return 0

    def _save_modules_directive(self, args):
        saved_modules = self.template_variables.get('saved_modules')

        if len(args) < 2:
            raise _batchprocessor.BatchProcessError(
                '\\save_modules directive must have at least 2 arguments, '
                'a variable name and one or more module names.')

        creation_result = self.render_loop.pipeline_wrapper.recall_main_pipeline()
        saved_modules[args[0]] = creation_result.get_pipeline_modules(args[1:])
        return 0

    def _use_modules_directive(self, args):
        saved_modules = self.template_variables.get('saved_modules')

        if not saved_modules:
            raise _batchprocessor.BatchProcessError(
                'no modules are currently saved that can be referenced.')

        saved_name = args[0]
        self.render_loop.model_extra_modules = saved_modules[saved_name]
        return 0

    def _clear_modules_directive(self, args):
        saved_modules = self.template_variables.get('saved_modules')

        if len(args) > 0:
            for arg in args:
                del saved_modules[arg]
        else:
            saved_modules.clear()
        return 0

    def _gen_seeds_directive(self, args):
        if len(args) == 2:
            try:
                self.template_variables[args[0]] = \
                    [str(s) for s in _renderloop.gen_seeds(int(args[1]))]
            except ValueError:
                raise _batchprocessor.BatchProcessError(
                    'The second argument of \\gen_seeds must be an integer value.')
        else:
            raise _batchprocessor.BatchProcessError(
                '\\gen_seeds directive takes 2 arguments, template variable '
                'name (to store value at), and number of seeds to generate.')
        return 0

    def _config_generate_template_variables_with_types(self, variable_prefix: typing.Optional[str] = None) \
            -> dict[str, tuple[typing.Type, typing.Any]]:

        template_variables = {}

        if variable_prefix is None:
            variable_prefix = ''

        for attr, hint in typing.get_type_hints(self.render_loop.config.__class__).items():
            value = getattr(self.render_loop.config, attr)
            if variable_prefix:
                prefix = variable_prefix if not attr.startswith(variable_prefix) else ''
            else:
                prefix = ''
            gen_name = prefix + attr
            if gen_name not in template_variables:
                if _types.is_type_or_optional(hint, list):
                    t_val = value if value is not None else []
                    template_variables[gen_name] = (hint, t_val)
                else:
                    template_variables[gen_name] = (hint, value)

        return template_variables

    def _generate_template_variables_with_types(self) -> dict[str, tuple[typing.Type, typing.Any]]:
        template_variables = self._config_generate_template_variables_with_types(variable_prefix='last_')

        template_variables.update({
            'last_images': (typing.Iterator[str], self.render_loop.written_images),
            'last_animations': (typing.Iterator[str], self.render_loop.written_animations),
        })

        template_variables['saved_modules'] = (dict[str, dict[str, typing.Any]],
                                               self.template_variables.get('saved_modules'))

        template_variables['glob'] = (types.ModuleType, self.template_variables.get('glob'))

        return template_variables

    def _generate_template_variables(self) -> dict[str, typing.Any]:
        return {k: v[1] for k, v in self._generate_template_variables_with_types().items()}

    def generate_template_variables_help(self,
                                         variable_names: typing.Optional[typing.Collection[str]] = None,
                                         show_values: bool = True):
        """
        Generate a help string describing available template variables, their types, and values for use in batch processing.

        :type variable_names: Display help for specific variables, if ``None`` or ``[]`` is specified, display all.

        :param show_values: Show the value of the template variable or just the name?

        :return: a human-readable description of all template variables
        """

        values = self._generate_template_variables_with_types()
        if variable_names is not None and len(variable_names) > 0:
            values = {k: v for k, v in values.items() if k in variable_names}

        if len(values) > 1:
            header = 'Config template variables are'
        else:
            header = 'Config template variable is'

        help_string = _textprocessing.underline(f'{header}:') + '\n\n'

        def wrap(val):
            return _textprocessing.wrap(
                str(val),
                width=_textprocessing.long_text_wrap_width(),
                subsequent_indent=' ' * 17)

        return help_string + '\n'.join(
            ' ' * 4 + f'Name: {_textprocessing.quote(i[0])}\n{" " * 8}'
                      f'Type: {i[1][0]}' + (f'\n{" " * 8}Value: {wrap(i[1][1])}' if show_values else '') for i in
            values.items())

    def _templates_help_directive(self, args):
        _messages.log(self.generate_template_variables_help(args) + '\n', underline=True)
        return 0


__all__ = _types.module_all()
