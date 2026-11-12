# Copyright (c) 2026, Teriks
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

"""Runtime for scripts produced by ``dgenerate --sub-command transpile``.

A generated script creates one Script and calls it. The helpers that used to be
copied into every script live here.
"""

import contextlib

import dgenerate
import dgenerate.arguments
import dgenerate.batchprocess.batchprocessor as batchprocessor
import dgenerate.batchprocess.configrunner as configrunner
import dgenerate.invoker as invoker
import dgenerate.messages as messages
import dgenerate.plugin as plugin
import dgenerate.renderloop

DgenerateArguments = dgenerate.arguments.DgenerateArguments
BatchProcessError = batchprocessor.BatchProcessError
env = batchprocessor._EnvNamespace()


class _Loop:
    def __init__(self, index0, length):
        self.index0 = index0
        self.index = index0 + 1
        self.length = length
        self.first = index0 == 0
        self.last = index0 == max(length - 1, 0)
        self.revindex0 = length - index0 - 1
        self.revindex = length - index0

    def __str__(self):
        return f'<loop {self.index}/{self.length}>'


class _Block:
    def __init__(self, script, start=None, end=None):
        self.script = script
        self.start = start
        self.end = end

    def __enter__(self):
        self.script.begin_block(self.start, self.end)
        return self

    def __exit__(self, typ, exc, tb):
        runner = self.script._runner
        if runner is not None:
            runner._running_template_continuation = False
        self.script._template_span = None
        if self.start is None or self.end is None:
            return False
        if exc is None or isinstance(exc, (BatchProcessError, SystemExit, KeyboardInterrupt)):
            return False
        raise BatchProcessError(
            'Error inside template continuation '
            f'from line {self.start} to line {self.end}:\n'
            f'{" " * 4}Template Render Error: {exc}'
        ) from exc


class _Line:
    """Pin continuation errors to one rendered line, then restore the outer span."""

    def __init__(self, script, line):
        self.script = script
        self.line = line
        self.previous = None
        self.runner_span = None

    def __enter__(self):
        script = self.script
        self.previous = script._template_span
        runner = script._runner
        if runner is None:
            return self
        self.runner_span = (
            runner._template_continuation_start_line,
            runner._template_continuation_end_line,
        )
        script._template_span = (self.line, self.line)
        runner._template_continuation_start_line = self.line
        runner._template_continuation_end_line = self.line
        runner._current_line = 0
        return self

    def __exit__(self, typ, exc, tb):
        script = self.script
        script._template_span = self.previous
        if self.runner_span is not None and script._runner is not None:
            script._runner._template_continuation_start_line = self.runner_span[0]
            script._runner._template_continuation_end_line = self.runner_span[1]
        return False


class _Undefined:
    """Stand-in for a Jinja name that was not set when a block started rendering."""

    def __str__(self):
        return ''

    def __bool__(self):
        return False

    def __iter__(self):
        return iter(())

    def __eq__(self, other):
        return False

    def __getattr__(self, name):
        raise AttributeError(name)


class Script:
    """One transpiled config: its render loop, template variables, and directives."""

    def __init__(self):
        self.render_loop = dgenerate.renderloop.RenderLoop()
        self.injected_args = []
        self._runner = None
        self._eager = None
        self._filters = None
        self._template_span = None
        self._field_types = None

    def arguments(self) -> DgenerateArguments:
        """A fresh arguments object for one static invocation."""
        return DgenerateArguments()

    @contextlib.contextmanager
    def options(self, verbose: bool = False, plugin_module_paths=None):
        """Set the log level and plugin path for one render-loop run.

        ``verbose`` is ``-v``. ``plugin_module_paths`` is ``--plugin-modules``.
        :py:class:`dgenerate.renderloopconfig.RenderLoopConfig` does not carry
        either value, so a relaxed script sets them here. Both are restored
        when the block exits. Modules imported during the block stay imported.

        :param verbose: Enable debug logging when ``True``.
        :param plugin_module_paths: Plugin files, directories, or module names.
        """
        paths = [str(path) for path in plugin_module_paths] if plugin_module_paths else []
        previous_paths = set(plugin.PLUGIN_PATHS)
        if paths:
            plugin.import_plugins(paths)
        level = messages.DEBUG if verbose else messages.INFO
        try:
            with messages.with_level(level):
                yield
        finally:
            plugin.PLUGIN_PATHS.clear()
            plugin.PLUGIN_PATHS.update(previous_paths)

    def run(self, config, line=None):
        """Run one prepared arguments object on this script's render loop."""
        if line is not None and self._runner is not None:
            self._runner._current_line = line
        try:
            self.render_loop.config = config
            plugins = getattr(config, 'plugin_module_paths', None) or None
            verbose = bool(getattr(config, 'verbose', False))
            if plugins or verbose:
                with self.options(verbose=verbose, plugin_module_paths=plugins):
                    self.render_loop.run()
            else:
                self.render_loop.run()
        except BatchProcessError as exc:
            if line is not None:
                self._raise(exc)
            raise
        self._refresh_templates()

    def sync(self):
        """Copy ``last_images`` and the other template values off the render loop."""
        self._refresh_templates()

    def run_argv(self, argv, line=None):
        """Parse an argument vector, run it, and return the invocation's status code."""
        argv = list(argv)
        if any(
            arg == '-h' or arg == '--help' or (str(arg).startswith('--') and str(arg).endswith('-help'))
            for arg in argv
        ):
            code = invoker.invoke_dgenerate(argv, render_loop=self.render_loop)
            if code:
                error = BatchProcessError(f'Invocation error return code: {code}')
                if line is not None:
                    self._raise(error)
                raise error
            return 0
        if line is not None and self._runner is not None:
            self._runner._current_line = line
        try:
            self.render_loop.config = dgenerate.arguments.parse_args(list(argv))
        except Exception as exc:
            messages.error(f'dgenerate: error: {exc}')
            error = BatchProcessError('Invocation error return code: 1')
            if line is not None:
                self._raise(error)
            raise error from exc
        try:
            self.render_loop.run()
        except BatchProcessError as exc:
            if line is not None:
                self._raise(exc)
            raise
        self._refresh_templates()
        return 0

    def version_check(self, line_idx, line=None):
        """Replay a ``#! dgenerate x.y.z`` line with the interpreter's own check.

        A relaxed script passes only the line. A strict script also passes the
        interpreter's 0-based line index, which the warning cites.
        """
        if line is None:
            line = line_idx
            line_idx = None
        probe = batchprocessor.BatchProcessor(
            invoker=lambda *_args: 0,
            name='dgenerate',
            version=dgenerate.__version__,
        )
        probe._look_for_version_mismatch(line_idx, line)

    def defined(self, value):
        """Jinja's ``is defined`` test."""
        return not isinstance(value, _Undefined)

    def filter(self, name, value, *args, **kwargs):
        """Apply one Jinja filter, including to values that were not set."""
        import jinja2

        if isinstance(value, _Undefined):
            value = jinja2.Undefined()
        if self._filters is None:
            self._filters = jinja2.Environment().filters
        try:
            func = self._filters[name]
        except KeyError as exc:
            raise BatchProcessError(f'No filter named {name}.') from exc
        return func(value, *args, **kwargs)

    def live(self, name):
        """Read a template variable as it is when a directive or invocation runs."""
        runner = self._ensure_runner()
        if name in runner.template_variables:
            return runner.template_variables[name]
        if name in getattr(runner, 'template_functions', {}):
            return runner.template_functions[name]
        if name in getattr(runner, 'builtins', {}):
            return runner.builtins[name]
        if name == 'env':
            return batchprocessor._EnvNamespace()
        return _Undefined()

    def shell(self, text):
        """Shell-parse one rendered argument the way an invocation does."""
        import dgenerate.textprocessing as textprocessing

        return textprocessing.shell_parse(
            str(text),
            expand_vars_func=textprocessing.shell_expandvars,
        )

    def only(self, words):
        words = list(words)
        if len(words) != 1:
            raise BatchProcessError(
                f'expected one shell argument, got {words!r}'
            )
        return words[0]

    def model_word(self, words):
        """One positional model token. An empty shell result is a missing model."""
        words = list(words)
        if len(words) == 1:
            return words[0]
        if not words:
            try:
                dgenerate.arguments.parse_args([])
            except Exception as exc:
                messages.error(f'dgenerate: error: {exc}')
            self._raise(BatchProcessError('Invocation error return code: 1'))
        raise BatchProcessError(
            f'expected one shell argument, got {words!r}'
        )

    def line(self, line):
        """Pin a continuation error to one rendered line that starts with ``{``."""
        return _Line(self, line)

    def typed(self, dest, value):
        """Apply the argument parser's converter for one config field."""
        converter = self._field_type(dest)
        if converter is None:
            return value
        return converter(value)

    def typed_list(self, dest, values):
        converter = self._field_type(dest)
        values = list(values)
        if converter is None:
            return values
        return [converter(value) for value in values]

    def _field_type(self, dest):
        if self._field_types is None:
            import dgenerate.arguments as arguments

            parser = arguments._create_parser(add_model=True, add_help=False)[0]
            self._field_types = {}
            for action in parser._actions:
                if action.option_strings:
                    for option in action.option_strings:
                        self._field_types[option] = action.type
                else:
                    self._field_types[action.dest] = action.type
        return self._field_types.get(dest)

    def expand(self, text):
        """Expand ``$VAR`` and ``%VAR%`` the way ``\\setp`` does before evaluation."""
        import dgenerate.textprocessing as textprocessing

        return textprocessing.shell_expandvars(str(text))

    def loop(self, index0, length):
        """The Jinja ``loop`` object for one iteration."""
        return _Loop(index0, length)

    def set_expr(self, line, name=None, fn=None):
        """Evaluate a ``\\setp`` expression in Python and store the result.

        Strict scripts pass ``(line, name, fn)``. Relaxed scripts pass ``(name, fn)``.
        """
        if callable(name) and fn is None:
            fn = name
            name = line
            line = None
        runner = self._ensure_runner()
        if line is not None:
            runner._current_line = line
        try:
            value = fn()
        except BatchProcessError as exc:
            self._raise(exc)
        except Exception as exc:
            self._raise(BatchProcessError(f'\\setp eval error: {type(exc).__name__}: {exc}'))
        self.set_value(line, name, value)

    def set_value(self, line, name, value):
        """Store one ``\\setp`` result without evaluating it again."""
        runner = self._ensure_runner()
        if line is not None:
            runner._current_line = line
        try:
            runner.user_define(str(name), value)
        except BatchProcessError as exc:
            self._raise(exc)

    def directive(self, line, name=None, *args):
        """Run one ``\\name`` directive.

        Strict scripts pass the config line number first. Relaxed scripts pass the
        directive name first.
        """
        if isinstance(line, str):
            if name is not None:
                args = (name,) + args
            name = line
            line = None
        text = '\\' + name
        if args:
            text += ' ' + ' '.join(str(arg) for arg in args)
        self.step(text, current_line=line)

    def run_directive(self, line, name=None, args=None):
        """Call one shell-parsed directive with arguments that are already values.

        Strict scripts pass the config line number first. The arguments are not
        joined and parsed again, so a value with spaces stays one argument.
        """
        if isinstance(line, str):
            args = name
            name = line
            line = None
        runner = self._ensure_runner()
        if line is not None:
            runner._current_line = line
        impl = runner.directives.get(name)
        if impl is None:
            self._raise(BatchProcessError(f'Unknown directive "\\{name}".'))
        try:
            code = impl([str(item) for item in args])
            if code:
                raise BatchProcessError(
                    f'Directive "{name}" error return code: {code}'
                )
        except BatchProcessError as exc:
            self._raise(exc)
        except Exception as exc:
            self._raise(BatchProcessError(exc))

    def block(self, start=None, end=None):
        """Snapshot template variables for one compiled ``{% if %}`` or ``{% for %}``.

        Strict scripts pass the continuation's line range so errors match the
        interpreter. Relaxed scripts omit it.
        """
        return _Block(self, start, end)

    def begin_block(self, start=None, end=None):
        """Remember template variables as they are before a control block renders."""
        runner = self._ensure_runner()
        self._eager = dict(runner.template_variables)
        if start is None or end is None:
            self._template_span = None
            return
        self._template_span = (start, end)
        runner._template_continuation_start_line = start
        runner._template_continuation_end_line = end
        runner._running_template_continuation = True

    def _raise(self, exc):
        current = self._runner.current_line if self._runner is not None else 0
        if not isinstance(exc, BatchProcessError):
            exc = BatchProcessError(str(exc))
        wrapped = BatchProcessError(
            f'Error on line {current}:\n{" " * 4}{str(exc).strip()}'
        )
        span = getattr(self, '_template_span', None)
        if span is None:
            raise wrapped from exc
        start, end = span
        sub = str(wrapped).split(':', 1)[-1].strip()
        raise BatchProcessError(
            'Error inside template continuation '
            f'from line {start} to line {end}:\n'
            f'{" " * 4}Error on line {start + current}:\n'
            f'{" " * 8}{sub}'
        ) from exc

    def eager(self, name):
        """Read a name the way the block's first render would, before its directives run."""
        if self._eager is not None and name in self._eager:
            return self._eager[name]
        runner = self._ensure_runner()
        if name in getattr(runner, 'template_functions', {}):
            return runner.template_functions[name]
        if name in getattr(runner, 'builtins', {}):
            return runner.builtins[name]
        if name == 'env':
            return batchprocessor._EnvNamespace()
        return _Undefined()

    def invoke(self, line, text):
        """Execute one rendered config line. Delayed ``{{ }}`` in it renders again."""
        text = str(text)
        if text.lstrip().startswith('{'):
            previous = self._template_span
            runner = self._ensure_runner()
            self._template_span = (line, line)
            runner._template_continuation_start_line = line
            runner._template_continuation_end_line = line
            try:
                self.step(text, current_line=line, template_start=line, template_end=line)
            finally:
                self._template_span = previous
                if previous is None:
                    runner._running_template_continuation = False
            return
        self.step(text, current_line=line)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        runner = self._ensure_runner()
        if name in runner.template_variables:
            return runner.template_variables[name]
        if name in getattr(runner, 'template_functions', {}):
            return runner.template_functions[name]
        if name in getattr(runner, 'builtins', {}):
            return runner.builtins[name]
        if name == 'env':
            return batchprocessor._EnvNamespace()
        raise AttributeError(name)

    def run_rendered(self, text):
        """Execute config lines that a compiled template block already rendered."""
        rendered = '' if text is None else str(text)
        if not rendered.strip():
            return
        if not rendered.endswith('\n'):
            rendered += '\n'
        runner = self._ensure_runner()
        try:
            runner.run_string(rendered)
        except batchprocessor._TemplateContinuationInternalError as exc:
            raise BatchProcessError(str(exc).strip()) from exc
        except BatchProcessError as exc:
            self._raise(exc)

    def step(self, text, current_line=None, template_start=None, template_end=None):
        """Execute one joined config line with the interpreter's own handlers."""
        text = text.strip()
        if not text:
            return
        runner = self._ensure_runner()
        if current_line is not None:
            runner._current_line = current_line
        if template_start is not None:
            runner._template_continuation_start_line = template_start
        if template_end is not None:
            runner._template_continuation_end_line = template_end
        runner._executing_text = text
        try:
            handled = runner._directive_handlers(text)
            if not handled:
                runner._lex_and_run_invocation(text)
        except batchprocessor._TemplateContinuationInternalError as exc:
            raise BatchProcessError(str(exc).strip()) from exc
        except BatchProcessError as exc:
            self._raise(exc)

    def _ensure_runner(self):
        if self._runner is not None:
            return self._runner
        runner = configrunner.ConfigRunner(
            render_loop=self.render_loop,
            injected_args=self.injected_args,
        )

        def invoke(_command_line, argv):
            argv = list(argv)
            if any(
                arg == '-h' or arg == '--help' or (arg.startswith('--') and arg.endswith('-help'))
                for arg in argv
            ):
                return invoker.invoke_dgenerate(argv, render_loop=self.render_loop)
            try:
                code = self.run_argv(argv)
            except BatchProcessError as exc:
                if str(exc).startswith('Invocation error return code:'):
                    return 1
                raise
            return 0 if code is None else code

        runner.invoker = invoke
        self._runner = runner
        return runner

    def _refresh_templates(self):
        if self._runner is not None:
            self._runner.template_variables.update(self._runner._generate_template_variables())
