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
import dgenerate.batchprocess.batchprocessorpluginloader as _batchprocessorpluginloader
import dgenerate.invoker as _invoker
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.prompt as _prompt
import dgenerate.renderloop as _renderloop
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types


def create_config_runner(injected_args: typing.Optional[typing.Sequence[str]] = None,
                         render_loop: typing.Optional[_renderloop.RenderLoop] = None,
                         plugin_loader: _batchprocessorpluginloader.BatchProcessorPluginLoader = None,
                         version: typing.Union[_types.Version, str] = dgenerate.__version__,
                         throw: bool = False):
    """
    Create a :py:class:`.BatchProcessor` that can run dgenerate batch processing configs from a string or file.


    :param injected_args: dgenerate command line arguments in the form of a list, see: shlex module, or sys.argv.
        These arguments will be injected at the end of every dgenerate invocation in the config.
    :param render_loop: RenderLoop instance, if None is provided one will be created.
    :param plugin_loader: Batch processor plugin loader, if one is not provided one will be created.
    :param version: Config version for ``#! dgenerate x.x.x`` version checks, defaults to ``dgenerate.__version__``
    :param throw: Whether to throw exceptions from :py:func:`dgenerate.invoker.invoke_dgenerate` or handle them,
        if you set this to True exceptions will propagate out of dgenerate invocations instead of a
        :py:exc:`.BatchProcessError` being raised, a line number where the error occurred can be obtained
        using :py:attr:`.BatchProcessor.current_line`.
    :return: integer return-code, anything other than 0 is failure
    """

    if render_loop is None:
        render_loop = _renderloop.RenderLoop()

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
        *_, last_item = list_or_iterable
        return last_item

    def first(iterable):
        return next(iter(iterable))

    template_variables = {
        'saved_modules': dict(),
        'glob': glob
    }

    funcs = {
        'unquote': unquote,
        'quote': quote,
        'format_prompt': format_prompt,
        'format_size': _textprocessing.format_size,
        'last': last,
        'first': first
    }

    def save_modules_directive(args):
        saved_modules = template_variables.get('saved_modules')

        if len(args) < 2:
            raise _batchprocessor.BatchProcessError(
                '\\save_modules directive must have at least 2 arguments, '
                'a variable name and one or more module names.')

        creation_result = render_loop.pipeline_wrapper.recall_main_pipeline()
        saved_modules[args[0]] = creation_result.get_pipeline_modules(args[1:])

    def use_modules_directive(args):
        saved_modules = template_variables.get('saved_modules')

        if not saved_modules:
            raise _batchprocessor.BatchProcessError(
                'no modules are currently saved that can be referenced.')

        saved_name = args[0]
        render_loop.model_extra_modules = saved_modules[saved_name]

    def clear_modules_directive(args):
        saved_modules = template_variables.get('saved_modules')

        if len(args) > 0:
            for arg in args:
                del saved_modules[arg]
        else:
            saved_modules.clear()

    def gen_seeds_directive(args):
        if len(args) == 2:
            try:
                template_variables[args[0]] = \
                    [str(s) for s in _renderloop.gen_seeds(int(args[1]))]
            except ValueError:
                raise _batchprocessor.BatchProcessError(
                    'The second argument of \\gen_seeds must be an integer value.')
        else:
            raise _batchprocessor.BatchProcessError(
                '\\gen_seeds directive takes 2 arguments, template variable '
                'name (to store value at), and number of seeds to generate.')

    def templates_help_directive(args):
        values = render_loop.generate_template_variables_with_types()
        values['saved_modules'] = (typing.Dict[str, typing.Dict[str, typing.Any]],
                                   template_variables.get('saved_modules'))
        values['glob'] = (types.ModuleType, "<module 'glob'>")

        header = None
        if len(args) > 0:
            values = {k: v for k, v in values.items() if k in args}

            if len(values) > 1:
                header = "Template variables are"
            else:
                header = 'Template variable is'

        _messages.log(
            render_loop.generate_template_variables_help(values=values,
                                                         header=header,
                                                         show_values=True) + '\n',
            underline=True)

    directives = {
        'templates_help': templates_help_directive,
        'clear_model_cache': lambda args: _pipelinewrapper.clear_model_cache(),
        'clear_pipeline_cache': lambda args: _pipelinewrapper.clear_pipeline_cache(),
        'clear_vae_cache': lambda args: _pipelinewrapper.clear_vae_cache(),
        'clear_control_net_cache': lambda args: _pipelinewrapper.clear_control_net_cache(),
        'save_modules': save_modules_directive,
        'use_modules': use_modules_directive,
        'clear_modules': clear_modules_directive,
        'gen_seeds': gen_seeds_directive,
    }

    plugin_loader = \
        _batchprocessorpluginloader.BatchProcessorPluginLoader() if \
            plugin_loader is None else plugin_loader

    plugin_module_paths = []

    if injected_args:

        parsed = _arguments.parse_known_args(
            injected_args,
            log_error=False,
            throw=False)

        # The created object will throw a BatchProcessError when we try to
        # run a file if the injected arguments are incorrect, which is a
        # better behavior than throwing here
        if parsed is not None and parsed.plugin_module_paths:
            plugin_module_paths = parsed.plugin_module_paths
            plugin_loader.load_plugin_modules(parsed.plugin_module_paths)

    runner = None

    loaded_plugin_classes = set()
    loaded_plugins = []

    for plugin_class in plugin_loader.get_available_classes():
        loaded_plugin_classes.add(plugin_class)
        loaded_plugins.append(
            plugin_loader.load(plugin_class.get_names()[0],
                               batch_processor=runner,
                               render_loop=render_loop,
                               plugin_module_paths=plugin_module_paths)
        )

    def import_plugins(plugin_paths):
        plugin_loader.load_plugin_modules(plugin_paths)
        for cls in plugin_loader.get_available_classes():
            if cls not in loaded_plugin_classes:
                loaded_plugins.append(
                    plugin_loader.load(cls.get_names()[0],
                                       batch_processor=runner,
                                       render_loop=render_loop,
                                       plugin_module_paths=plugin_module_paths))
                loaded_plugin_classes.add(cls)

    def directive_lookup(name):
        if name == 'import_plugins':
            return import_plugins

        impl = directives.get(name)
        if impl is None:
            implementations = []
            for plugin in loaded_plugins:
                impl = plugin.directive_lookup(name)
                if impl is not None:
                    implementations.append(impl)

            if len(implementations) > 1:
                raise RuntimeError(
                    f'Multiple BatchProcessorPlugins implement the batch processing directive \\{name}.')

            if implementations:
                return implementations[0]

            return None
        else:
            return impl

    def invoker(args):
        try:
            return _invoker.invoke_dgenerate(args, render_loop=render_loop, throw=throw)
        finally:
            render_loop.model_extra_modules = None

    runner = _batchprocessor.BatchProcessor(
        invoker=invoker,
        template_variable_generator=lambda: render_loop.generate_template_variables(),
        name='dgenerate',
        version=version,
        template_variables=template_variables,
        template_functions=funcs,
        injected_args=injected_args if injected_args else [],
        directive_lookup=directive_lookup)

    return runner


__all__ = _types.module_all()
