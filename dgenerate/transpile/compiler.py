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

# Experimental config-to-API transpiler.
#
# The config language, including its Jinja, is the ANTLR grammar in
# dgenerate.transpile.grammar. Generated scripts import dgenerate.transpile.runtime.
# A generation line is a DgenerateArguments object, including inside if and
# for. {% set %}, {% filter %}, and {% with %} are Python as well. Delayed
# {{ }} is read when that line runs. Directives call script.directive.
# Anything the grammar cannot parse is script.step(), except a {% for %},
# {% if %}, {% set %}, {% filter %}, or {% with %} that does not parse.
# Those are syntax errors, and so is anything the config scanner rejects.
# Translation stops. The script does not contain that error as a raise.
# Config comments are kept by default, in the same places the interpreter
# ignores them. comments=False strips every one. Blank lines separate version
# checks, set directives, other directives, and template blocks.

import argparse
import ast
import builtins
import enum
import os
import pathlib
import re
import sys

_PY_BUILTINS = set(dir(builtins))

from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener

from dgenerate.transpile.grammar import Config, ConfigLexer


class _ParseError(Exception):
    pass


ParseError = _ParseError


class _Errors(ErrorListener):
    def __init__(self):
        self.messages = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.messages.append(f'{line}:{column} {msg}')


def parse_config(text: str):
    lexer = ConfigLexer(InputStream(text))
    errors = _Errors()
    lexer.removeErrorListeners()
    lexer.addErrorListener(errors)
    parser = Config(CommonTokenStream(lexer))
    parser.removeErrorListeners()
    parser.addErrorListener(errors)
    tree = parser.config()
    if errors.messages:
        raise _ParseError('\n'.join(errors.messages))
    return tree


def _nodes(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


class _EnvNamespace:
    def __getattr__(self, name):
        return os.environ.get(name, '')


def logical_events(text: str) -> list[tuple]:
    """Joined lines and version checks, in the order the interpreter runs them.

    Directive lines are ``('exec', current_line, text, template_start, template_end)``.
    Version hashbangs are ``('version', line_idx, text)``. ``\\reset_lineno`` is not an
    event; it is already applied to the next line's ``current_line``.
    """
    import dgenerate.batchprocess.batchprocessor as batchprocessor

    events = []

    class _Collector(batchprocessor.BatchProcessor):
        def _look_for_version_mismatch(self, line_idx, line):
            if '#!' in line:
                events.append(('version', line_idx, line))

        def _directive_handlers(self, line: str) -> bool:
            start = end = None
            if line.lstrip().startswith('{'):
                start = getattr(self, '_template_continuation_start_line', None)
                end = getattr(self, '_template_continuation_end_line', None)
            events.append(('exec', self._current_line, line, start, end))
            return True

    try:
        _Collector(
            invoker=lambda *_args: 0,
            name='dgenerate',
            version='6.0.0',
        ).run_string(text)
    except batchprocessor.BatchProcessError as exc:
        events.append(('raise', str(exc)))
    return events


def logical_lines(text: str) -> list[str]:
    """The same joined lines the config interpreter executes, before Jinja."""
    return [event[2] for event in logical_events(text) if event[0] == 'exec']


def _config_comments(text: str) -> list[tuple[int, str]]:
    """Comments the interpreter ignores, as ``(source line, '# ...')``.

    A ``#!`` line is a version check, not a comment. A ``#`` inside quotes, and
    a ``\\#``, stay part of the value. The same rules as
    ``textprocessing.remove_tail_comments``.
    """
    import dgenerate.textprocessing as textprocessing

    found = []
    for number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith('#!'):
            continue
        if line.lstrip().startswith('#'):
            found.append((number, line.strip()))
            continue
        removed, _kept = textprocessing.remove_tail_comments(line)
        if not removed:
            continue
        comment = _tail_comment(line)
        if comment:
            found.append((number, comment))
    return found


def _tail_comment(line: str) -> str | None:
    """The ``#...`` tail of a line, using the interpreter's quote and escape rules."""
    quote = None
    index = 0
    while index < len(line):
        char = line[index]
        if quote is None and char == '\\' and index + 1 < len(line) and line[index + 1] == '#':
            index += 2
            continue
        if quote is None and char in '"\'':
            quote = char
        elif quote == char:
            quote = None
        elif quote is None and char == '#':
            comment = line[index:].rstrip()
            return comment or None
        index += 1
    return None


def _has_jinja(text: str) -> bool:
    return '{{' in text or '{%' in text or '{#' in text


_BUILTIN_DIRECTIVE_PREFIXES = (
    'unset_env', 'setp', 'sete', 'set', 'unset', 'print', 'echo', 'env',
)


def _handled_by_builtin(name: str) -> bool:
    """True when the interpreter's own directive handlers claim this name."""
    return any(name.startswith(prefix) for prefix in _BUILTIN_DIRECTIVE_PREFIXES)


def _merge_string_literals(bits: list[str]) -> list[str]:
    merged = []
    for bit in bits:
        if merged and _constant_string(merged[-1]) is not None and _constant_string(bit) is not None:
            merged[-1] = repr(_constant_string(merged[-1]) + _constant_string(bit))
        else:
            merged.append(bit)
    return merged


def _parts_source(parts):
    chunks = []
    holes = []
    for kind, value in parts:
        if kind == 'text':
            chunks.append(value)
        elif kind == 'expr':
            holes.append(value)
            chunks.append(f'__hole{len(holes) - 1}__')
        else:
            return None
    return ''.join(chunks), holes


class _SetpRewrite(ast.NodeTransformer):
    def __init__(self, holes, functions, transpiler):
        self.holes = holes
        self.functions = functions
        self.transpiler = transpiler
        self.failed = False
        self.saw_template_call = False

    def visit_Call(self, node):
        self.generic_visit(node)
        func = node.func
        if isinstance(func, ast.Name) and func.id in self.functions:
            self.saw_template_call = True
        return node

    def visit_Name(self, node):
        match = re.fullmatch(r'__hole(\d+)__', node.id)
        if match:
            self.failed = True
            return node
        if node.id in self.functions:
            self.transpiler.used_functions.add(node.id)
            return node
        if node.id in _safe_builtin_names():
            return node
        self.failed = True
        return node

    def visit_Constant(self, node):
        if not isinstance(node.value, str):
            return node
        return _mark_env(_constant_with_holes(node.value, self.holes))

    def visit_Lambda(self, node):
        self.failed = True
        return node

    def visit_NamedExpr(self, node):
        self.failed = True
        return node

    def visit_JoinedStr(self, node):
        self.failed = True
        return node

    def visit_FormattedValue(self, node):
        self.failed = True
        return node

    def visit_Yield(self, node):
        self.failed = True
        return node

    def visit_YieldFrom(self, node):
        self.failed = True
        return node

    def visit_Await(self, node):
        self.failed = True
        return node


def _mark_env(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.search(r'[$%]', node.value):
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='script', ctx=ast.Load()),
                attr='expand',
                ctx=ast.Load(),
            ),
            args=[node],
            keywords=[],
        )
    if isinstance(node, ast.BinOp):
        node.left = _mark_env(node.left)
        node.right = _mark_env(node.right)
    return node


def _constant_with_holes(text, holes):
    pieces = re.split(r'(__hole\d+__)', text)
    if len(pieces) == 1:
        return ast.Constant(text)
    nodes = []
    for piece in pieces:
        match = re.fullmatch(r'__hole(\d+)__', piece)
        if match:
            expr = ast.parse(f'str({holes[int(match.group(1))]})', mode='eval').body
            nodes.append(expr)
        elif piece:
            nodes.append(ast.Constant(piece))
    value = nodes[0]
    for node in nodes[1:]:
        value = ast.BinOp(left=value, op=ast.Add(), right=node)
    return value


def _setp_python(source, holes, transpiler):
    try:
        tree = ast.parse(source, mode='eval')
    except SyntaxError:
        return None
    rewriter = _SetpRewrite(holes, _template_function_names(), transpiler)
    tree = rewriter.visit(tree)
    if rewriter.failed or not rewriter.saw_template_call:
        return None
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _safe_builtin_names():
    import dgenerate.eval as evaluate

    names = getattr(_safe_builtin_names, 'cache', None)
    if names is None:
        names = set(evaluate.safe_builtins())
        _safe_builtin_names.cache = names
    return names


def _template_function_names():
    import dgenerate.batchprocess.configrunnerbuiltins as functions

    names = getattr(_template_function_names, 'cache', None)
    if names is None:
        names = {
            name for name in dir(functions)
            if not name.startswith('_') and callable(getattr(functions, name))
        }
        _template_function_names.cache = names
    return names


def _constant_string(expr: str):
    try:
        value = ast.literal_eval(expr)
    except (SyntaxError, ValueError):
        return None
    if isinstance(value, str):
        return value
    return None


def _roundtrip_words(text: str) -> list[str]:
    words = text.split(' ')
    if words and '' not in words and ' '.join(words) == text:
        return words
    return [text]


def _directive_call(line: str):
    """Split a directive line into ``(name, args)`` that join back to the same line."""
    stripped = line.lstrip()
    if not stripped.startswith('\\'):
        return None
    pieces = stripped.split(None, 1)
    name = pieces[0][1:]
    if not name.isidentifier():
        return None
    rest = pieces[1] if len(pieces) > 1 else None
    if name in {'set', 'setp', 'sete'}:
        parts = stripped.split(None, 2)
        if len(parts) < 3:
            return name, [] if rest is None else [rest]
        import dgenerate.batchprocess.batchprocessor as batchprocessor
        var, value = batchprocessor.BatchProcessor._set_split(None, parts, stripped)
        return name, [var, value]
    if rest is None:
        return name, []
    if name in {'print', 'echo', 'unset'} or _has_jinja(rest):
        return name, [rest]
    return name, _roundtrip_words(rest)


def _is_directive(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith('\\') or stripped.startswith('{')


def _is_dynamic(line: str) -> bool:
    stripped = line.lstrip()
    return (
        stripped.startswith('{%')
        or stripped.startswith('{#')
        or '{{' in line
        or '{%' in line
        or '{#' in line
        or '#' in line
    )


def _owned_opening(line: str) -> str | None:
    """The compiled tag a block starts with, or None for a tag left to the interpreter."""
    stripped = line.lstrip()
    if not stripped.startswith('{%'):
        return None
    rest = stripped[2:].lstrip()
    for name in ('for', 'if', 'elif', 'set', 'filter', 'with'):
        if rest.startswith(name) and (len(rest) == len(name) or rest[len(name)] in ' \t\r\n%'):
            return name
    return None


def _template_syntax_error(exc: _ParseError, line_number) -> ParseError:
    return ParseError(f'Template syntax error on line {line_number}:\n{exc}')


def _template_functions():
    import inspect

    import dgenerate.batchprocess.batchprocessor as batchprocessor
    import dgenerate.batchprocess.configrunnerbuiltins as builtins

    functions = {}
    for name in dir(builtins):
        if name.startswith('_'):
            continue
        func = getattr(builtins, name)
        if callable(func):
            functions[name] = func
    return batchprocessor.BatchProcessor.default_builtins(), functions


def _jinja_environment():
    import inspect

    import jinja2

    environment = jinja2.Environment()
    builtins, functions = _template_functions()
    for name, func in builtins.items():
        if name not in environment.globals:
            environment.globals[name] = func
    for name, func in functions.items():
        environment.globals[name] = func
        try:
            if len(inspect.signature(func).parameters) > 0:
                environment.filters[name] = func
        except (TypeError, ValueError):
            pass
    environment.globals['env'] = _EnvNamespace()
    return environment


_JINJA = None


def _render(text: str, namespace: dict) -> str:
    import typing

    import dgenerate.batchprocess.batchprocessor as batchprocessor

    global _JINJA
    if _JINJA is None:
        _JINJA = _jinja_environment()
    variables = {}
    for key, value in namespace.items():
        if not str(key).isidentifier() or str(key).startswith('_') or callable(value):
            continue
        variables[key] = value
    import glob as glob_module
    variables['glob'] = glob_module
    variables['os'] = os
    variables['env'] = _EnvNamespace()
    config = namespace.get('last_config')
    if config is not None:
        try:
            attrs = typing.get_type_hints(type(config))
        except Exception:
            attrs = {}
        for attr in attrs:
            if attr.startswith('_'):
                continue
            try:
                variables['last_' + attr] = getattr(config, attr)
            except Exception:
                continue
    written_images = namespace.get('written_images')
    written_animations = namespace.get('written_animations')
    variables['last_images'] = list(written_images or [])
    variables['last_animations'] = list(written_animations or [])
    variables['injected_args'] = namespace.get('injected_args', [])
    variables['injected_device'] = namespace.get('injected_device')
    variables['injected_verbose'] = namespace.get('injected_verbose', False)
    variables['injected_plugin_modules'] = namespace.get('injected_plugin_modules', [])
    variables['saved_modules'] = namespace.get('saved_modules', {})
    try:
        return _JINJA.from_string(text).render(**variables)
    except Exception as exc:
        raise batchprocessor.BatchProcessError(
            f'Template Render Error: {str(exc).strip()}'
        ) from exc


def _set_split(line: str):
    import dgenerate.batchprocess.batchprocessor as batchprocessor

    directive_args = line.split(None, 2)
    return batchprocessor.BatchProcessor._set_split(None, directive_args, line)


def _assign(namespace: dict, name: str, value):
    if name.isidentifier():
        namespace[name] = value
        return
    namespace[name] = value


def _eval_setp(expression: str, namespace: dict):
    import dgenerate.batchprocess.batchprocessor as batchprocessor
    import dgenerate.eval as evaluate

    table = {}
    for key, value in namespace.items():
        if str(key).isidentifier() and not str(key).startswith('_') and not callable(value):
            table[key] = value
    table.update({
        'injected_args': namespace.get('injected_args', []),
        'injected_device': namespace.get('injected_device'),
        'injected_verbose': namespace.get('injected_verbose', False),
        'injected_plugin_modules': namespace.get('injected_plugin_modules', []),
        'saved_modules': namespace.get('saved_modules', {}),
        'last_images': list(namespace.get('written_images') or []),
        'last_animations': list(namespace.get('written_animations') or []),
    })
    interpreter = evaluate.standard_interpreter(symtable=table)
    builtins, functions = _template_functions()
    interpreter.symtable.update(builtins)
    interpreter.symtable.update(functions)
    try:
        return interpreter.eval(expression, show_errors=False, raise_errors=True)
    except Exception as exc:
        raise batchprocessor.BatchProcessError(f'\\setp eval error: {exc}') from exc


def _shell(text: str) -> list[str]:
    import dgenerate.textprocessing as textprocessing
    return textprocessing.shell_parse(text, expand_vars_func=textprocessing.shell_expandvars)


def _expand_vars(text: str) -> str:
    import dgenerate.textprocessing as textprocessing
    return textprocessing.shell_expandvars(text)


def version_check(line_idx: int, line: str):
    """Replay ``#! dgenerate x.y.z`` with the interpreter's own check."""
    import dgenerate
    import dgenerate.batchprocess.batchprocessor as batchprocessor

    probe = batchprocessor.BatchProcessor(
        invoker=lambda *_args: 0,
        name='dgenerate',
        version=dgenerate.__version__,
    )
    probe._look_for_version_mismatch(line_idx, line)


def ensure_runner(namespace: dict):
    """The ConfigRunner that owns template variables for this generated script."""
    runner = namespace.get('_runner')
    if runner is not None:
        return runner
    import dgenerate.batchprocess.configrunner as configrunner

    runner = configrunner.ConfigRunner(
        render_loop=namespace.get('render_loop'),
        injected_args=namespace.get('injected_args') or [],
    )

    def invoke(_command_line, argv):
        import dgenerate.batchprocess.batchprocessor as batchprocessor
        import dgenerate.invoker as invoker

        argv = list(argv)
        if any(
            arg == '-h' or arg == '--help' or (arg.startswith('--') and arg.endswith('-help'))
            for arg in argv
        ):
            return invoker.invoke_dgenerate(argv, render_loop=namespace.get('render_loop'))
        try:
            code = namespace['_run_argv'](argv)
        except batchprocessor.BatchProcessError as exc:
            if str(exc).startswith('Invocation error return code:'):
                return 1
            raise
        return 0 if code is None else code

    runner.invoker = invoke
    namespace['_runner'] = runner
    namespace.setdefault('_user_keys', set())
    return runner


def _sync_user_variables(namespace: dict, runner):
    reserved = runner.reserved_template_variables
    previous = set(namespace.get('_user_keys', ()))
    current = set()
    for key, value in runner.template_variables.items():
        if key in reserved or not str(key).isidentifier() or str(key).startswith('_'):
            continue
        namespace[key] = value
        current.add(key)
    for key in previous - current:
        namespace.pop(key, None)
    namespace['_user_keys'] = current
    if 'saved_modules' in runner.template_variables:
        namespace['saved_modules'] = runner.template_variables['saved_modules']


def step(text: str, namespace: dict, current_line=None, template_start=None, template_end=None):
    """Execute one joined config line with the interpreter's own handlers."""
    import dgenerate.batchprocess.batchprocessor as batchprocessor

    runner = ensure_runner(namespace)
    if current_line is not None:
        runner._current_line = current_line
    if template_start is not None:
        runner._template_continuation_start_line = template_start
    if template_end is not None:
        runner._template_continuation_end_line = template_end
    runner._executing_text = text
    try:
        try:
            handled = runner._directive_handlers(text)
            if not handled:
                runner._lex_and_run_invocation(text)
        except batchprocessor._TemplateContinuationInternalError as exc:
            raise batchprocessor.BatchProcessError(str(exc).strip()) from exc
        except batchprocessor.BatchProcessError as exc:
            raise batchprocessor.BatchProcessError(
                f'Error on line {runner.current_line}:\n{" " * 4}{str(exc).strip()}'
            ) from exc
    finally:
        _sync_user_variables(namespace, runner)


def _dispatch(line: str, namespace: dict):
    if not line or line.startswith('#'):
        return
    if line.startswith('{%') or line.startswith('{#'):
        step(line, namespace)
        return
    if line.startswith('\\env'):
        _handle_env(line, namespace)
        return
    if line.startswith('\\unset_env'):
        _handle_unset_env(line, namespace)
        return
    if line.startswith('\\setp'):
        _handle_set(line, namespace, 'setp')
        return
    if line.startswith('\\sete'):
        _handle_set(line, namespace, 'sete')
        return
    if line.startswith('\\set'):
        _handle_set(line, namespace, 'set')
        return
    if line.startswith('\\unset'):
        _handle_unset(line, namespace)
        return
    if line.startswith('\\print'):
        _handle_print(line, namespace)
        return
    if line.startswith('\\echo'):
        _handle_echo(line, namespace)
        return
    if line.startswith('\\'):
        _handle_custom(line, namespace)
        return
    _handle_invocation(line, namespace)


def _handle_set(line: str, namespace: dict, kind: str):
    var, value = _set_split(line)
    name = _expand_vars(_render(var, namespace)).strip()
    if kind == 'set':
        _assign(namespace, name, _expand_vars(_render(value, namespace)))
        return
    if kind == 'sete':
        _assign(namespace, name, _shell(_render(value, namespace)))
        return
    _assign(namespace, name, _eval_setp(_expand_vars(_render(value, namespace)), namespace))


def _handle_unset(line: str, namespace: dict):
    args = line.split(None, 1)
    if len(args) < 2:
        return
    name = _expand_vars(_render(args[1], namespace)).strip()
    namespace.pop(name, None)


def _handle_print(line: str, namespace: dict):
    args = line.split(None, 1)
    printer = namespace.get('print', print)
    if len(args) == 2:
        printer(_expand_vars(_render(args[1], namespace)))
    else:
        printer()


def _handle_echo(line: str, namespace: dict):
    args = line.split(None, 1)
    printer = namespace.get('print', print)
    if len(args) == 2:
        printer(*_shell(_render(args[1], namespace)))
    else:
        printer()


def _handle_env(line: str, namespace: dict):
    args = line.split(None, 1)
    if len(args) == 2:
        rendered = _render(args[1], namespace).split()
        for opt in rendered:
            assignment = opt.split('=', 1)
            value = _expand_vars(_render(assignment[1], namespace)) if len(assignment) > 1 else ''
            os.environ[assignment[0]] = value
        return
    printer = namespace.get('print', print)
    for key, value in os.environ.items():
        printer(f'\\env "{key}={value}"')


def _handle_unset_env(line: str, namespace: dict):
    args = line.split(None, 1)
    if len(args) < 2:
        return
    for name in _shell(_render(args[1].strip(), namespace)):
        os.environ.pop(name, None)


def _handle_custom(line: str, namespace: dict):
    parts = line.split(None, 1)
    name = parts[0].lstrip('\\')
    argv = _shell(_render(parts[1], namespace)) if len(parts) > 1 else []
    directive = namespace.get('_directive')
    if directive is None:
        raise RuntimeError(f'No directive runner for \\{name}')
    directive(name, argv)


def _handle_invocation(line: str, namespace: dict):
    import dgenerate.textprocessing as textprocessing

    rendered = _render(line, namespace)
    argv = _shell(rendered)
    for arg in namespace.get('injected_args', []):
        arg = str(arg).strip()
        if arg:
            argv.append(textprocessing.shell_parse(
                arg, expand_glob=False, expand_vars=False, expand_home=False
            )[0])
    namespace['_run_argv'](argv)


class Transpiler:
    def __init__(self, strict=True, comments=True):
        self.strict = strict
        self.comments = comments
        self._comments = []
        self._comment_at = 0
        self.lines = []
        self.indent = 0
        self.needs_run = False
        self.needs_loop = False
        self.needs_directive = False
        self.needs_shell = False
        self.needs_templates = False
        self.needs_dgenerate = False
        self.needs_runtime = False
        self.control = False
        self.assign_names = False
        self._render_count = 0
        self.expand_delayed = False
        self.loop_depth = 0
        self.bound_names = set()
        self.used_functions = set()
        self.used_modules = set()
        self.line_base = 1
        self.stmt_line = 1
        self.live_names = False

    def emit(self, line=''):
        self.lines.append(('    ' * self.indent + line) if line else '')

    def _flush_comments(self, up_to_line):
        if not self.comments or up_to_line is None:
            return
        while (
            self._comment_at < len(self._comments)
            and self._comments[self._comment_at][0] <= up_to_line
        ):
            self.emit(self._comments[self._comment_at][1])
            self._comment_at += 1

    def _node_line(self, node):
        """File line of a template token. ``line_base`` is the block's first line."""
        if node is None or not self.control:
            return None
        symbol = getattr(node, 'symbol', None)
        if symbol is None:
            return None
        return self.line_base + symbol.line - 1

    def compile(self, text: str) -> str:
        if self.comments:
            self._comments = _config_comments(text)
        for event in logical_events(text):
            if event[0] == 'version':
                _kind, line_idx, line = event
                # line_idx is the 0-based source index. Comments use 1-based lines,
                # so this flushes comments on earlier lines and leaves this line.
                self._flush_comments(line_idx)
                self.needs_runtime = True
                self.needs_dgenerate = True
                self.emit(f'script.version_check({line_idx}, {line.rstrip(chr(13) + chr(10))!r})')
                continue
            if event[0] == 'raise':
                raise _ParseError(event[1])
            _kind, current, line, template_start, template_end = event
            # A template's current line is its last line. Flushing through it
            # would pull the block's own comments out ahead of the block.
            if template_start is not None:
                self._flush_comments(template_start - 1)
            else:
                self._flush_comments(current)
            if line.lstrip().startswith('{%') and self._emit_control(
                line,
                template_start if template_start is not None else current,
                template_end if template_end is not None else current,
            ):
                continue
            if (
                '{{' in line
                and not line.lstrip().startswith('\\')
                and not line.lstrip().startswith('{%')
            ):
                lowered = self._lower_templated_invocation(line)
                if lowered is not None:
                    self._emit_config(lowered)
                    continue
            if line.lstrip().startswith('\\') and template_start is None:
                parsed = _directive_call(line)
                if parsed is not None:
                    self._emit_directive(current, parsed[0], parsed[1])
                    continue
            # '$', '%', '~', and unquoted '*' expand per argument when the
            # script runs. They do not send the rest of the invocation through
            # the interpreter.
            if _is_directive(line) or _is_dynamic(line):
                self._emit_step(line, current, template_start, template_end)
                continue
            if _has_shell_word(line):
                lowered = self._fields_from_holes(line, [], [])
            else:
                lowered = _lower_static_invocation(line)
            if lowered is None:
                self._emit_step(line, current, template_start, template_end)
                continue
            self._emit_config(lowered)
        self._flush_comments(10**9)
        self.lines = _space_statements(self.lines)
        return self.render()

    def _emit_control(self, line, line_base, line_end=None) -> bool:
        try:
            tree = parse_config(line if line.endswith('\n') else line + '\n')
        except _ParseError as exc:
            if _owned_opening(line):
                where = 1 if line_base is None else line_base
                raise _template_syntax_error(exc, where) from exc
            return False
        elements = [element for element in _nodes(tree.element()) if element.stmt() is not None]
        if not elements:
            return False
        rules = []
        for element in elements:
            pieces = self._pieces(element.stmt())
            if len(pieces) != 1:
                return False
            rule = self._rule(pieces[0])
            if rule not in {'jinjaFor', 'jinjaIf', 'jinjaSet', 'jinjaFilter', 'jinjaWith'}:
                return False
            rules.append(rule)
        self.control = True
        self.assign_names = False
        self.line_base = 1 if line_base is None else line_base
        self.bound_names = set()
        end = line_base if line_end is None else line_end
        flow = all(rule in {'jinjaFor', 'jinjaIf'} for rule in rules)
        try:
            if flow:
                self.needs_runtime = True
                self.needs_run = True
                self.needs_dgenerate = True
                if self.strict:
                    self.emit(f'with script.block({line_base}, {end}):')
                else:
                    self.emit('with script.block():')
                self.indent += 1
                try:
                    self.visit_config(tree)
                finally:
                    self.indent -= 1
            else:
                self.visit_config(tree)
        finally:
            self.control = False
            self.assign_names = False
        self._flush_comments(end)
        return True

    def _source_line(self, ctx):
        start = getattr(ctx, 'start', None)
        if start is None:
            return 0
        # Inside a template block the interpreter numbers lines in the rendered
        # text, which is one less than the block-relative source line.
        return start.line - 1

    def _file_line(self, ctx):
        """File line of a statement inside a template block."""
        start = getattr(ctx, 'start', None)
        if start is None or not self.control:
            return 0
        return self.line_base + start.line - 1

    def _emit_control_directive(self, parts):
        name, rest = self._split_directive(parts)
        if not str(name).isidentifier():
            self.needs_runtime = True
            self.needs_run = True
            self.needs_dgenerate = True
            expr = self._string_expr(parts)
            if self.strict:
                self.emit(f'script.invoke({self.stmt_line}, {expr})')
            else:
                self.emit(f'script.step({expr})')
            return
        if name == 'setp' and rest:
            built = _parts_source(rest)
            if built is not None:
                source, holes = built
                head, tail = source.split(None, 1) if len(source.split()) >= 2 else ('', '')
                if head.isidentifier() and tail:
                    python_value = _setp_python(tail, holes, self)
                    if python_value is not None:
                        self.needs_runtime = True
                        self.needs_run = True
                        self.needs_dgenerate = True
                        self._emit_set_expr(self.stmt_line, head, python_value)
                        return
        if rest and not _handled_by_builtin(name):
            words = self._lower_directive_parts(rest, second_pass=True)
            if words is not None:
                self._emit_run_directive(self.stmt_line, name, words)
                return
        expr = self._string_expr(rest) if rest else None
        if expr is None:
            self._emit_directive(self.stmt_line, name, [])
            return
        constant = _constant_string(expr)
        if constant is not None:
            parsed = _directive_call('\\' + name + ((' ' + constant) if constant else ''))
            if parsed is not None:
                self._emit_directive(self.stmt_line, parsed[0], parsed[1])
                return
        self.needs_runtime = True
        self.needs_run = True
        self.needs_dgenerate = True
        if self.strict:
            self.emit(f'script.directive({self.stmt_line}, {name!r}, {expr})')
        else:
            self.emit(f'script.directive({name!r}, {expr})')

    def _emit_control_invoke(self, parts):
        lowered = self._lower_invocation_parts(parts)
        if isinstance(lowered, list):
            line = self.stmt_line if self.strict else None
            built = _invocation_statement(parts)
            if self.strict and built is not None and built[0].lstrip().startswith('{'):
                self.emit(f'with script.line({self.stmt_line}):')
                self.indent += 1
                self._emit_config(lowered, line)
                self.indent -= 1
            else:
                self._emit_config(lowered, line)
            return
        if isinstance(lowered, tuple):
            self._emit_run_argv(repr(lowered[1]))
            return
        self._emit_run_argv(self._shell_expr(parts))

    def _emit_config(self, fields, line=None):
        self.needs_runtime = True
        self.needs_run = True
        self.needs_dgenerate = True
        verbose = None
        plugins = None
        kept = []
        for statement in fields:
            if not self.strict and statement.startswith('config.verbose = '):
                verbose = statement.split('=', 1)[1].strip()
                continue
            if not self.strict and statement.startswith('config.plugin_module_paths = '):
                plugins = statement.split('=', 1)[1].strip()
                continue
            kept.append(statement)
        if self.strict:
            self.emit('config: DgenerateArguments = script.arguments()')
        else:
            self.needs_loop = True
            self.emit('config: RenderLoopConfig = RenderLoopConfig()')
        for statement in kept:
            self.emit(statement)
        if self.strict:
            if line is None:
                self.emit('script.run(config)')
            else:
                self.emit(f'script.run(config, {line})')
        else:
            self.emit('render_loop.config = config')
            if verbose is not None or plugins is not None:
                arguments = []
                if plugins is not None:
                    arguments.append(f'plugin_module_paths={plugins}')
                if verbose is not None:
                    arguments.append(f'verbose={verbose}')
                self.emit(f'with script.options({", ".join(arguments)}):')
                self.indent += 1
                self.emit('render_loop.run()')
                self.indent -= 1
            else:
                self.emit('render_loop.run()')

    def _emit_run_argv(self, argv_expr):
        self.needs_runtime = True
        self.needs_run = True
        self.needs_dgenerate = True
        self.needs_shell = True
        if self.strict:
            self.emit(f'script.run_argv(script.shell({argv_expr}), {self.stmt_line})')
        else:
            self.emit(f'script.run_argv(script.shell({argv_expr}))')

    def _emit_set_expr(self, line, name, python_value):
        self.needs_runtime = True
        self.needs_run = True
        self.needs_dgenerate = True
        if self.strict:
            self.emit(f'script.set_expr({line}, {name!r}, lambda: {python_value})')
        else:
            self.emit(f'script.set_expr({name!r}, lambda: {python_value})')

    def _shell_expr(self, parts):
        previous = self.expand_delayed
        self.expand_delayed = True
        try:
            return self._string_expr(parts)
        finally:
            self.expand_delayed = previous

    def _emit_directive(self, line, name, args):
        self.needs_runtime = True
        self.needs_run = True
        self.needs_dgenerate = True
        if (
            not _handled_by_builtin(name)
            and len(args) == 1
            and _has_jinja(args[0])
        ):
            words = self._lower_directive_parts([('text', args[0])], second_pass=False)
            if words is not None:
                self._emit_run_directive(line, name, words)
                return
        if name == 'setp' and len(args) == 2 and args[0].isidentifier():
            python_value = _setp_python(args[1], [], self)
            if python_value is not None:
                self._emit_set_expr(line, args[0], python_value)
                return
        if self.strict:
            pieces = [repr(line), repr(name), *(repr(arg) for arg in args)]
        else:
            pieces = [repr(name), *(repr(arg) for arg in args)]
        self.emit(f'script.directive({", ".join(pieces)})')

    def _emit_run_directive(self, line, name, words):
        self.needs_runtime = True
        self.needs_run = True
        self.needs_dgenerate = True
        if any('script.shell' in word or 'script.expand' in word for word in words):
            self.needs_shell = True
        body = '[' + ', '.join(words) + ']'
        if self.strict:
            self.emit(f'script.run_directive({line}, {name!r}, {body})')
        else:
            self.emit(f'script.run_directive({name!r}, {body})')

    def _lower_directive_parts(self, parts, second_pass):
        """Split a shell-parsed directive's arguments into Python values."""
        if any(kind not in {'text', 'expr', 'for', 'if'} for kind, _value in parts):
            return None
        if any(kind in {'for', 'if'} for kind, _value in parts):
            return [f'*script.shell({self._shell_expr(parts)})']
        built = _invocation_statement(parts)
        if built is None:
            return None
        statement, py_holes = built
        planted = self._plant_inline_control(statement, len(py_holes))
        if planted is None:
            return None
        statement, extra = planted
        return self._directive_words(statement, py_holes + extra, second_pass)

    def _directive_words(self, statement, py_holes, second_pass):
        replaced, hole_sources, hole_quoted = _extract_holes(statement)
        expanded = _expand_delayed_holes(replaced, hole_sources, hole_quoted, second_pass)
        if expanded is None:
            return None
        replaced, hole_sources, hole_quoted = expanded
        compiled = self._compile_hole_sources(hole_sources, force_live=second_pass)
        if compiled is None:
            return None
        replaced = _attach_py_holes(replaced, compiled, hole_quoted, py_holes)
        if 'DGPY' in replaced:
            return None
        try:
            import dgenerate.textprocessing
            parsed = dgenerate.textprocessing.shell_parse(
                replaced,
                expand_home=False,
                expand_vars=False,
                expand_glob=False,
            )
        except Exception:
            return None
        tokens = _shell_tokens(replaced)
        if tokens is None or [token for token, _quote in tokens] != list(parsed):
            return None
        words = []
        for token, quote in tokens:
            words.extend(_directive_token_items(token, compiled, quote))
        return words

    def _plant_inline_control(self, text, start):
        """Replace inline ``{% for %}`` and ``{% if %}`` with ``DGPY`` holes."""
        if '{%' not in text and '{#' not in text:
            return text, []
        pieces = []
        holes = []
        quote = None
        index = 0
        while index < len(text):
            if text.startswith('{%', index) or text.startswith('{#', index):
                block = _read_jinja_block(text, index)
                if block is None:
                    return None
                source, end = block
                if source.startswith('{#'):
                    index = end
                    continue
                expr = self._compile_inline_block(source)
                if expr is None:
                    return None
                pieces.append(f'DGPY{start + len(holes)}')
                holes.append((expr, quote))
                index = end
                continue
            if text.startswith('{{', index):
                end = _jinja_expr_end(text, index)
                if end is None:
                    return None
                pieces.append(text[index:end])
                index = end
                continue
            char = text[index]
            previous = text[index - 1] if index else ''
            if quote is None and char in {'"', "'"} and (index == 0 or previous.isspace()):
                quote = char
            elif quote == char and previous != '\\':
                quote = None
            pieces.append(char)
            index += 1
        return ''.join(pieces), holes

    def _compile_inline_block(self, source):
        try:
            tree = parse_config(source if source.endswith('\n') else source + '\n')
        except _ParseError as exc:
            if _owned_opening(source):
                raise _template_syntax_error(exc, self.stmt_line) from exc
            return None
        elements = [element for element in _nodes(tree.element()) if element.stmt() is not None]
        if len(elements) != 1:
            return None
        pieces = self._pieces(elements[0].stmt())
        if len(pieces) != 1:
            return None
        previous_live = self.live_names
        previous_control = self.control
        self.live_names = True
        self.control = True
        try:
            if pieces[0].jinjaFor() is not None:
                return self._for_text(pieces[0].jinjaFor())
            if pieces[0].jinjaIf() is not None:
                return self._if_text(pieces[0].jinjaIf())
            if pieces[0].jinjaFilter() is not None:
                return self._filter_expr(pieces[0].jinjaFilter())
            if pieces[0].jinjaWith() is not None:
                return self._with_expr(pieces[0].jinjaWith())
        finally:
            self.live_names = previous_live
            self.control = previous_control
        return None

    def _emit_step(self, line: str, current_line=None, template_start=None, template_end=None):
        self.needs_runtime = True
        self.needs_run = True
        self.needs_shell = True
        self.needs_dgenerate = True
        if not self.strict:
            self.emit(f'script.step({line!r})')
            return
        args = [repr(line)]
        if current_line is not None or template_start is not None:
            args.append(repr(current_line))
        if template_start is not None:
            args.extend((repr(template_start), repr(template_end)))
        self.emit(f'script.step({", ".join(args)})')

    def _lower_templated_invocation(self, statement: str):
        """Turn an invocation with ``{{ }}`` holes into field assignments."""
        py_holes = []
        if '{%' in statement or '{#' in statement:
            planted = self._plant_inline_control(statement, 0)
            if planted is None:
                return None
            statement, py_holes = planted
        replaced, hole_sources, hole_quoted = _extract_holes(statement)
        if not hole_sources and not py_holes:
            return None
        expanded = _expand_delayed_holes(replaced, hole_sources, hole_quoted, False)
        if expanded is None:
            return None
        replaced, hole_sources, hole_quoted = expanded
        if not hole_sources and not py_holes:
            if _has_shell_word(replaced):
                return self._fields_from_holes(replaced, [], [])
            return _lower_static_invocation(replaced)
        compiled = self._compile_hole_sources(hole_sources, force_live=False)
        if compiled is None:
            return None
        replaced = _attach_py_holes(replaced, compiled, hole_quoted, py_holes)
        if 'DGPY' in replaced:
            return None
        return self._fields_from_holes(replaced, compiled, hole_quoted)

    def _lower_invocation_parts(self, parts):
        """Lower one invocation inside a compiled ``if`` or ``for``."""
        built = _invocation_statement(parts)
        if built is None:
            return None
        statement, py_holes = built
        replaced, hole_sources, hole_quoted = _extract_holes(statement)
        expanded = _expand_delayed_holes(replaced, hole_sources, hole_quoted, True)
        if expanded is None:
            return None
        replaced, hole_sources, hole_quoted = expanded
        if not hole_sources and not py_holes:
            return self._fields_from_holes(replaced, [], [])
        compiled = self._compile_hole_sources(hole_sources, force_live=True)
        if compiled is None:
            return None
        replaced = _attach_py_holes(replaced, compiled, hole_quoted, py_holes)
        if 'DGPY' in replaced:
            return None
        return self._fields_from_holes(replaced, compiled, hole_quoted)

    def _compile_hole_sources(self, holes, force_live):
        compiled = []
        previous_live = self.live_names
        previous_control = self.control
        previous_bound = self.bound_names
        self.live_names = True
        self.control = True
        if force_live:
            # The second render looks up template variables, not loop targets.
            self.bound_names = set()
        try:
            for hole in holes:
                try:
                    tree = parse_config('\\print {{ ' + hole + ' }}\n')
                except _ParseError:
                    return None
                expr = _find_jinja_expr(tree)
                if expr is None:
                    return None
                compiled.append(self.py_expr(expr))
        finally:
            self.live_names = previous_live
            self.control = previous_control
            self.bound_names = previous_bound
        return compiled

    def _template_text_python(self, content):
        """Compile the text left behind by a delayed ``{{ '{{ ... }}' }}``."""
        replaced, holes, quoted = _extract_holes(content)
        if not holes:
            return repr(content)
        expanded = _expand_delayed_holes(replaced, holes, quoted, True)
        if expanded is None:
            return None
        replaced, holes, quoted = expanded
        compiled = self._compile_hole_sources(holes, force_live=True)
        if compiled is None:
            return None
        bits = []
        for piece in re.split(r'(DGHOLE\d+)', replaced):
            match = re.fullmatch(r'DGHOLE(\d+)', piece)
            if match:
                bits.append(f'str({compiled[int(match.group(1))]})')
            elif piece:
                bits.append(repr(piece))
        if not bits:
            return "''"
        if len(bits) == 1:
            return bits[0]
        return '(' + ' + '.join(bits) + ')'

    def _fields_from_holes(self, replaced, hole_python, hole_quoted):
        try:
            import dgenerate.arguments
            import dgenerate.textprocessing
            replaced, globs = _mask_shell_words(replaced)
            argv = dgenerate.textprocessing.shell_parse(
                replaced,
                expand_home=False,
                expand_vars=False,
                expand_glob=False,
            )
            parser = dgenerate.arguments._create_parser(add_model=True, add_help=False)[0]
        except Exception:
            return None
        options = {}
        positional = None
        for action in parser._actions:
            if action.option_strings:
                for option in action.option_strings:
                    options[option] = action
            elif action.dest == 'model_path':
                positional = action
        fields = []
        index = 0
        if positional is not None and argv and not str(argv[0]).startswith('-'):
            rendered = _value_expr(argv[0], hole_python, hole_quoted, globs)
            if rendered is None:
                return None
            baked = _bake_value(rendered, positional)
            if baked is not None:
                fields.append(f'config.model_path = {baked}')
            else:
                if self.strict and rendered.startswith('script.only('):
                    rendered = 'script.model_word' + rendered[len('script.only'):]
                fields.append(f'config.model_path = script.typed({positional.dest!r}, {rendered})')
            index = 1
        while index < len(argv):
            token = str(argv[index])
            action = options.get(token)
            if action is None or action.dest == 'help':
                return None
            index += 1
            if action.nargs == 0:
                fields.append(f'config.{action.dest} = True')
                continue
            if action.nargs in (None, 1):
                count = 1
            elif action.nargs in {'+', '*'}:
                count = 0
                values = []
                while index < len(argv) and not str(argv[index]).startswith('-'):
                    values.append(str(argv[index]))
                    index += 1
                if action.nargs == '+' and not values:
                    return None
                rendered = _typed_list_expr(values, hole_python, hole_quoted, globs, action, token)
                if rendered is None:
                    return None
                fields.append(f'config.{action.dest} = {rendered}')
                continue
            elif isinstance(action.nargs, int):
                count = action.nargs
            else:
                return None
            if index + count > len(argv):
                return None
            values = [str(item) for item in argv[index:index + count]]
            index += count
            option = argv[index - count - 1]
            if count == 1:
                rendered = _value_expr(values[0], hole_python, hole_quoted, globs)
                if rendered is None:
                    return None
                baked = _bake_value(rendered, action)
                if baked is not None:
                    fields.append(f'config.{action.dest} = {baked}')
                elif action.type is None:
                    fields.append(f'config.{action.dest} = {rendered}')
                else:
                    fields.append(f'config.{action.dest} = script.typed({option!r}, {rendered})')
            else:
                rendered = _typed_list_expr(values, hole_python, hole_quoted, globs, action, option)
                if rendered is None:
                    return None
                fields.append(f'config.{action.dest} = {rendered}')
        return fields or None

    def render(self) -> str:
        if not self.strict and self.needs_loop and any('script.' in line for line in self.lines):
            rewritten = []
            for line in self.lines:
                rewritten.append(line)
                if line.strip() == 'render_loop.run()':
                    indent = line[:len(line) - len(line.lstrip())]
                    rewritten.append(f'{indent}script.sync()')
            self.lines = rewritten
        body = '\n'.join(self.lines)
        library = []
        if 'dgenerate.' in body:
            library.append('import dgenerate')
        if 'DgenerateArguments' in body:
            library.append('from dgenerate.arguments import DgenerateArguments')
        if 'BatchProcessError' in body:
            library.append('from dgenerate.batchprocess.batchprocessor import BatchProcessError')
        functions = [
            name for name in sorted(self.used_functions)
            if re.search(rf'\b{name}\s*\(', body)
        ]
        if functions:
            library.append(
                'from dgenerate.batchprocess.configrunnerbuiltins import '
                + ', '.join(functions)
            )
        if 'Prompt' in body:
            library.append('from dgenerate.prompt import Prompt')
        if not self.strict and self.needs_loop:
            library.append('from dgenerate.renderloop import RenderLoop')
        if 'RenderLoopConfig' in body:
            library.append('from dgenerate.renderloopconfig import RenderLoopConfig')
        runtime = []
        if 'script.' in body:
            runtime.append('Script')
        if 'env' in self.used_modules:
            runtime.append('env')
        banner = '# Generated by dgenerate --sub-command transpile.'
        if not self.strict:
            banner += ' Relaxed: runs the config without interpreter error text.'
        header = [banner, '']
        header.extend(library)
        if runtime:
            header.append(f'from dgenerate.transpile.runtime import {", ".join(runtime)}')
        if library or runtime:
            header.append('')
        if 'glob' in self.used_modules:
            header += ['import glob', '']
        if 'os' in self.used_modules:
            header += ['import os', '']
        preamble = []
        if 'script.' in body:
            preamble.append('script = Script()')
        if not self.strict and self.needs_loop:
            if 'script.' in body:
                preamble.append('render_loop: RenderLoop = script.render_loop')
            else:
                preamble.append('render_loop: RenderLoop = RenderLoop()')
        if preamble:
            header.extend(preamble)
            header.append('')
        return '\n'.join(header + self.lines).rstrip() + '\n'

    def visit_config(self, ctx):
        for element in _nodes(ctx.element()):
            self.visit_element(element)

    def visit_element(self, element):
        if element.NEWLINE() is not None and element.stmt() is None:
            return
        statement = element.stmt()
        if statement is not None:
            self.visit_stmt(statement)

    def visit_stmt(self, stmt):
        self.stmt_line = self._source_line(stmt)
        self._flush_comments(self._file_line(stmt))
        pieces = self._pieces(stmt)
        if len(pieces) == 1 and self._rule(pieces[0]) == 'jinjaFor':
            self.visit_for(pieces[0].jinjaFor())
            return
        if len(pieces) == 1 and self._rule(pieces[0]) == 'jinjaIf':
            self.visit_if(pieces[0].jinjaIf())
            return
        if len(pieces) == 1 and self._rule(pieces[0]) == 'jinjaSet':
            self.visit_set(pieces[0].jinjaSet())
            return
        if len(pieces) == 1 and self._rule(pieces[0]) == 'jinjaFilter':
            self.visit_filter(pieces[0].jinjaFilter())
            return
        if len(pieces) == 1 and self._rule(pieces[0]) == 'jinjaWith':
            self.visit_with(pieces[0].jinjaWith())
            return
        if len(pieces) == 1 and pieces[0].COMMENT() is not None:
            return
        parts = [part for part in self._parts(stmt) if part[0] != 'comment']
        if not parts:
            return
        if parts[0][0] == 'text' and parts[0][1].lstrip().startswith('\\'):
            self.visit_directive(parts)
            return
        self.visit_invocation(parts)

    def visit_for(self, ctx):
        names = [token.getText() for token in _nodes(ctx.jTargets().ID())]
        targets = ', '.join(names)
        depth = self.loop_depth
        self.loop_depth += 1
        items = f'_items{depth}'
        index = f'_loop{depth}'
        self.emit(f'{items} = list({self.py_expr(ctx.jExpr())})')
        otherwise = ctx.forElse()
        if otherwise is not None:
            self.emit(f'if {items}:')
            self.indent += 1
        target = targets if len(names) == 1 else f'({targets})'
        self.emit(f'for {index}, {target} in enumerate({items}):')
        fresh = [name for name in names + ['loop'] if name not in self.bound_names]
        self.bound_names.update(names)
        self.bound_names.add('loop')
        self.indent += 1
        self.emit(f'loop = script.loop({index}, len({items}))')
        self.stmt_line = self._source_line(ctx)
        previous_assign = self.assign_names
        self.assign_names = True
        try:
            body_start = len(self.lines)
            pieces = list(_nodes(ctx.piece()))
            if pieces:
                self._emit_pieces(pieces)
            statements = [element.stmt() for element in _nodes(ctx.element()) if element.stmt() is not None]
            for statement in statements:
                self.visit_stmt(statement)
            if otherwise is not None:
                cap = self._node_line(otherwise.CLOSER())
            else:
                cap = self._node_line(ctx.CLOSER())
            if cap is not None:
                self._flush_comments(cap - 1)
            if len(self.lines) == body_start:
                self.emit('pass')
            self.indent -= 1
            if otherwise is not None:
                self.indent -= 1
                self.emit('else:')
                self.indent += 1
                end_cap = self._node_line(ctx.CLOSER())
                until = end_cap - 1 if end_cap is not None else None
                self._visit_body(otherwise.element(), until=until)
                self.indent -= 1
        finally:
            self.assign_names = previous_assign
            for name in fresh:
                self.bound_names.discard(name)
            self.loop_depth -= 1

    def _emit_pieces(self, pieces):
        parts = []
        for piece in pieces:
            parts.extend(self._piece_parts(piece))
        parts = [part for part in parts if part[0] != 'comment']
        if not parts:
            return
        if parts[0][0] == 'text' and parts[0][1].lstrip().startswith('\\'):
            self.visit_directive(parts)
            return
        self.visit_invocation(parts)

    def visit_if(self, ctx):
        previous_assign = self.assign_names
        self.assign_names = True
        try:
            stops = [self._node_line(part.ELIF()) for part in _nodes(ctx.elifPart())]
            else_part = ctx.elsePart()
            if else_part is not None:
                stops.append(self._node_line(else_part.ELSE()))
            stops.append(self._node_line(ctx.ENDIF()))
            stops = [line for line in stops if line is not None]

            def cap(index):
                if index < len(stops):
                    return stops[index] - 1
                return None

            self.emit(f'if {self.py_expr(ctx.jExpr())}:')
            self.indent += 1
            self._visit_body(ctx.ifBody().element(), until=cap(0))
            self.indent -= 1
            index = 1
            for part in _nodes(ctx.elifPart()):
                self.emit(f'elif {self.py_expr(part.jExpr())}:')
                self.indent += 1
                self._visit_body(part.element(), until=cap(index))
                self.indent -= 1
                index += 1
            if else_part is not None:
                self.emit('else:')
                self.indent += 1
                self._visit_body(else_part.element(), until=cap(index))
                self.indent -= 1
        finally:
            self.assign_names = previous_assign

    def _visit_body(self, elements, until=None):
        body_start = len(self.lines)
        statements = [element.stmt() for element in _nodes(elements) if element.stmt() is not None]
        for statement in statements:
            self.visit_stmt(statement)
        if until is not None:
            self._flush_comments(until)
        if len(self.lines) == body_start:
            self.emit('pass')

    def visit_directive(self, parts):
        if self.control:
            self._emit_control_directive(parts)
            return
        name, rest = self._split_directive(parts)
        if name in {'set', 'setp', 'sete'}:
            self._assign_directive(name, rest)
            return
        if name == 'unset':
            self._unset(rest)
            return
        if name in {'print', 'echo'}:
            self._emit_control_directive(parts)
            return
        if name == 'env':
            self._env(rest)
            return
        if name == 'unset_env':
            for item in self._literal_words(rest):
                self.emit(f'os.environ.pop({item!r}, None)')
            return
        if name == 'gen_seeds':
            self.needs_templates = True
            words = self._literal_words(rest)
            if len(words) == 2 and words[0].isidentifier():
                self.emit(f'{words[0]} = gen_seeds({words[1]})')
                return
        self.needs_directive = True
        self.needs_dgenerate = True
        self.needs_shell = True
        rendered = self._string_expr(rest)
        if rendered == "''":
            self.emit(f"_directive({name!r}, [])")
        else:
            self.emit(
                f'_directive({name!r}, dgenerate.textprocessing.shell_parse({rendered}))'
            )

    def _assign_directive(self, name, rest):
        target, value = self._assignment(rest)
        if name == 'set':
            self.emit(f'{target} = _expand({self._string_expr(value)})')
            return
        if name == 'sete':
            self.needs_shell = True
            self.needs_dgenerate = True
            self.emit(
                f'{target} = dgenerate.textprocessing.shell_parse({self._string_expr(value)})'
            )
            return
        self._note_template_calls(value)
        if self._only_text(value) and not re.search(r'[$%]', self._text(value)):
            self.emit(f'{target} = {self._text(value).strip()}')
            return
        self.emit(f'{target} = eval(_expand({self._string_expr(value)}))')

    def _unset(self, rest):
        target, _value = self._assignment(rest + [ ('text', '') ])
        if target.isidentifier():
            self.emit(f'del {target}')
        elif target.startswith('globals()[') and target.endswith(']'):
            self.emit(f'globals().pop({target[len("globals()["):-1]}, None)')
        else:
            self.emit(f'# unset {target}')

    def _env(self, rest):
        if not rest or self._string_expr(rest) == "''":
            self.emit("for _key, _value in os.environ.items():")
            self.emit("    print(f'\\\\env \"{_key}={_value}\"')")
            return
        if self._only_text(rest):
            for piece in self._text(rest).split():
                key, sep, value = piece.partition('=')
                if not sep:
                    self.emit(f'os.environ[{key!r}] = ""')
                else:
                    self.emit(f'os.environ[{key!r}] = _expand({value!r})')
            return
        self.needs_directive = True
        self.needs_dgenerate = True
        self.needs_shell = True
        self.emit(
            f"_directive('env', _expand({self._string_expr(rest)}).split())"
        )

    def visit_invocation(self, parts):
        if self.control:
            self._emit_control_invoke(parts)
            return
        self.needs_run = True
        self.needs_dgenerate = True
        dynamic = any(part[0] != 'text' for part in parts)
        source = self._text(parts)
        if not dynamic and _has_shell_word(source):
            lowered = self._fields_from_holes(source, [], [])
            if lowered is not None:
                self._emit_config(lowered)
                return
        if not dynamic:
            lowered = _lower_static_invocation(source)
            if lowered is not None:
                self.emit('config = dgenerate.arguments.DgenerateArguments()')
                for line in lowered:
                    self.emit(line)
                self.emit('_run_config(config)')
                return
        self.needs_shell = True
        rendered = source if not dynamic else None
        if rendered is not None:
            self.emit(f'_run_argv(dgenerate.textprocessing.shell_parse({rendered!r}))')
            return
        self.emit(
            f'_run_argv(dgenerate.textprocessing.shell_parse({self._string_expr(parts)}))'
        )

    def _pieces(self, stmt):
        pieces = list(_nodes(stmt.piece()))
        for continuation in _nodes(stmt.continuation()):
            pieces.extend(_nodes(continuation.piece()))
        return pieces

    def _parts(self, stmt):
        parts = []
        for piece in _nodes(stmt.piece()):
            parts.extend(self._piece_parts(piece))
        for continuation in _nodes(stmt.continuation()):
            dash = continuation.LINE_DASH()
            parts.append(('text', ' ' + (dash.getText() if dash is not None else '-')))
            for piece in _nodes(continuation.piece()):
                parts.extend(self._piece_parts(piece))
        return parts

    def _piece_parts(self, piece):
        if piece.TEXT() is not None:
            return [('text', piece.TEXT().getText())]
        if piece.QUOTE() is not None:
            return [('text', piece.QUOTE().getText())]
        if piece.HASH_CHAR() is not None:
            return [('text', piece.HASH_CHAR().getText())]
        if piece.LINE_DASH() is not None:
            return [('text', piece.LINE_DASH().getText())]
        if piece.COMMENT() is not None:
            return [('comment', piece.COMMENT().getText())]
        if piece.jinjaExpr() is not None:
            return [('expr', self.py_expr(piece.jinjaExpr()))]
        if piece.jinjaFor() is not None:
            return [('for', piece.jinjaFor())]
        if piece.jinjaIf() is not None:
            return [('if', piece.jinjaIf())]
        if piece.jinjaFilter() is not None:
            return [('expr', self._filter_expr(piece.jinjaFilter()))]
        if piece.jinjaWith() is not None:
            return [('expr', self._with_expr(piece.jinjaWith()))]
        if piece.jinjaSet() is not None:
            return [('expr', self._set_inline(piece.jinjaSet()))]
        return [('text', piece.getText())]

    def _split_directive(self, parts):
        text = ''
        index = 0
        while index < len(parts) and parts[index][0] == 'text':
            text += parts[index][1]
            match = re.match(r'\\(\w+)\s*(.*)\Z', text, re.S)
            if match and (index + 1 == len(parts) or parts[index + 1][0] != 'text'):
                rest = [('text', match.group(2))] + parts[index + 1:]
                return match.group(1), self._trim_rest(rest)
            index += 1
        match = re.match(r'\\(\w+)\s*(.*)\Z', text, re.S)
        if not match:
            return text.strip(), parts
        return match.group(1), self._trim_rest([('text', match.group(2))] + parts[index:])

    def _trim_rest(self, parts):
        if parts and parts[0][0] == 'text':
            kind, text = parts[0]
            parts = [(kind, text.lstrip())] + parts[1:]
            if parts[0][1] == '':
                parts = parts[1:]
        return parts

    def _assignment(self, rest):
        if rest and rest[0][0] == 'expr':
            return f'globals()[{rest[0][1]}]', self._trim_rest(rest[1:])
        if rest and rest[0][0] == 'text':
            text = rest[0][1]
            if text.startswith('$'):
                env_name = text.split(None, 1)[0][1:]
                if env_name.startswith('{') and '}' in env_name:
                    env_name = env_name[1:env_name.index('}')]
                    remain = text[text.index('}') + 1:]
                else:
                    remain = text[len(env_name) + 1:]
                target = f'globals()[os.environ[{env_name!r}]]'
                return target, [('text', remain.lstrip())] + rest[1:]
            name, remain = _split_first(text)
            return self._target(name), [('text', remain)] + rest[1:]
        return 'value', rest

    def _target(self, name):
        name = name.strip()
        if name.startswith('{{') and name.endswith('}}'):
            return f'globals()[{name[2:-2].strip()}]'
        if name.startswith('$'):
            env_name = name[1:]
            if env_name.startswith('{') and env_name.endswith('}'):
                env_name = env_name[1:-1]
            return f'globals()[os.environ[{env_name!r}]]'
        if name.isidentifier():
            return name
        return f'globals()[{name!r}]'

    def _string_expr(self, parts):
        bits = []
        for kind, value in parts:
            if kind == 'text':
                if value:
                    bits.append(repr(value))
            elif kind == 'expr':
                constant = _constant_string(value)
                if constant is not None and self.expand_delayed and '{{' in constant:
                    rendered = self._template_text_python(constant)
                    if rendered is not None:
                        bits.append(rendered)
                        continue
                if constant is not None:
                    bits.append(value)
                else:
                    bits.append(f'str({value})')
            elif kind == 'for':
                bits.append(self._for_text(value))
            elif kind == 'if':
                bits.append(self._if_text(value))
        bits = _merge_string_literals(bits)
        if not bits:
            return "''"
        if len(bits) == 1:
            return bits[0]
        return '(' + ' + '.join(bits) + ')'

    def _for_text(self, ctx):
        names = [token.getText() for token in _nodes(ctx.jTargets().ID())]
        targets = ', '.join(names)
        iterator = self.py_expr(ctx.jExpr())
        fresh = [name for name in names if name not in self.bound_names]
        self.bound_names.update(names)
        try:
            bits = []
            pieces = list(_nodes(ctx.piece()))
            if pieces:
                parts = []
                for piece in pieces:
                    parts.extend(self._piece_parts(piece))
                bits.append(self._string_expr(parts))
            element_text = self._body_text(ctx.element())
            if element_text != "''":
                bits.append(element_text)
        finally:
            for name in fresh:
                self.bound_names.discard(name)
        if not bits:
            body = "''"
        elif len(bits) == 1:
            body = bits[0]
        else:
            body = '(' + ' + '.join(bits) + ')'
        return f"''.join({body} for {targets} in {iterator})"

    def _if_text(self, ctx):
        then = self._body_text(ctx.ifBody().element())
        if ctx.elsePart() is not None:
            otherwise = self._body_text(ctx.elsePart().element())
        else:
            otherwise = "''"
        text = f'({then} if {self.py_expr(ctx.jExpr())} else {otherwise})'
        for part in reversed(_nodes(ctx.elifPart())):
            elif_body = self._body_text(part.element())
            text = f'({elif_body} if {self.py_expr(part.jExpr())} else {text})'
        return text

    def _body_text(self, elements):
        bits = []
        for element in _nodes(elements):
            if element.stmt() is None:
                continue
            bits.append(self._string_expr(self._parts(element.stmt())))
        if not bits:
            return "''"
        if len(bits) == 1:
            return bits[0]
        return '(' + ' + '.join(bits) + ')'

    def _only_text(self, parts):
        return all(kind == 'text' for kind, _value in parts)

    def _text(self, parts):
        return ''.join(value for kind, value in parts if kind == 'text')

    def _literal_words(self, parts):
        if not self._only_text(parts):
            return []
        return self._text(parts).split()

    def _note_template_calls(self, parts):
        for _kind, value in parts:
            if re.search(r'\b[A-Za-z_][A-Za-z0-9_]*\s*\(', value if isinstance(value, str) else ''):
                self.needs_templates = True

    def _rule(self, piece):
        if piece.jinjaFor() is not None:
            return 'jinjaFor'
        if piece.jinjaIf() is not None:
            return 'jinjaIf'
        if piece.jinjaSet() is not None:
            return 'jinjaSet'
        if piece.jinjaFilter() is not None:
            return 'jinjaFilter'
        if piece.jinjaWith() is not None:
            return 'jinjaWith'
        return ''

    def py_expr(self, ctx):
        name = type(ctx).__name__
        if name == 'JinjaExprContext':
            return self.py_expr(ctx.jExpr())
        if name == 'JExprContext':
            value = self.py_expr(_nodes(ctx.jOr())[0])
            ors = _nodes(ctx.jOr())
            if ctx.IF() is not None:
                condition = self.py_expr(ors[1])
                otherwise = self.py_expr(ctx.jExpr())
                return f'({value} if {condition} else {otherwise})'
            return value
        if name == 'JOrContext':
            return ' or '.join(self.py_expr(item) for item in _nodes(ctx.jAnd()))
        if name == 'JAndContext':
            return ' and '.join(self.py_expr(item) for item in _nodes(ctx.jNot()))
        if name == 'JNotContext':
            if ctx.NOT() is not None and ctx.jComparison() is None:
                return f'(not {self.py_expr(ctx.jNot())})'
            return self.py_expr(ctx.jComparison())
        if name == 'JComparisonContext':
            values = _nodes(ctx.jConcat())
            ops = _nodes(ctx.jCmpOp())
            text = self.py_expr(values[0])
            for index, op in enumerate(ops):
                rhs = self.py_expr(values[index + 1]) if index + 1 < len(values) else None
                text = self._apply_cmp(text, op, rhs)
            return text
        if name == 'JConcatContext':
            values = [self.py_expr(item) for item in _nodes(ctx.jSum())]
            if len(values) == 1:
                return values[0]
            return '(' + ' + '.join(f'str({item})' for item in values) + ')'
        if name == 'JSumContext':
            return self._binary(ctx, ctx.jTerm(), {'+', '-'})
        if name == 'JTermContext':
            return self._binary(ctx, ctx.jFactor(), {'*', '/', '//', '%'})
        if name == 'JFactorContext':
            if ctx.jPower() is not None and ctx.jFactor() is None:
                return self.py_expr(ctx.jPower())
            op = '+' if ctx.PLUS() is not None else '-'
            return f'({op}{self.py_expr(ctx.jFactor())})'
        if name == 'JPowerContext':
            base = self.py_expr(ctx.jPostfix())
            if ctx.jFactor() is not None:
                return f'({base} ** {self.py_expr(ctx.jFactor())})'
            return base
        if name == 'JPostfixContext':
            text = self.py_expr(ctx.jPrimary())
            for suffix in _nodes(ctx.jSuffix()):
                text = self._suffix(text, suffix)
            return text
        if name == 'JPrimaryContext':
            return self._primary(ctx)
        return ctx.getText()

    def _binary(self, ctx, operands, operators):
        values = [self.py_expr(item) for item in _nodes(operands)]
        ops = [token.getText() for token in ctx.children if getattr(token, 'getText', lambda: '')() in operators]
        # children mix rules and tokens; tokens are TerminalNode
        ops = []
        for child in ctx.children:
            text = child.getText()
            if text in operators:
                ops.append(text)
        text = values[0]
        for op, rhs in zip(ops, values[1:]):
            text = f'({text} {op} {rhs})'
        return text

    def _apply_cmp(self, text, op, rhs):
        if op.IS() is not None:
            name = op.ID().getText() if op.ID() is not None else ''
            negated = op.NOT() is not None
            if name == 'defined':
                check = f'script.defined({text})'
                return f'(not {check})' if negated else check
            if name == 'undefined':
                check = f'script.defined({text})'
                return check if negated else f'(not {check})'
        rendered = self._cmp(op)
        if rhs is None:
            return text
        return f'({text} {rendered} {rhs})'

    def _cmp(self, ctx):
        if ctx.NOT() is not None and ctx.IN() is not None:
            return 'not in'
        if ctx.IN() is not None:
            return 'in'
        if ctx.IS() is not None:
            name = ctx.ID().getText() if ctx.ID() is not None else ''
            if name == 'defined':
                return 'is not None'
            if name == 'undefined':
                return 'is None'
            if ctx.NOT() is not None:
                return f'is not {name}' if name else 'is not'
            return f'is {name}' if name else 'is'
        return ctx.getText()

    def _suffix(self, text, suffix):
        if suffix.DOT() is not None:
            return f'{text}.{suffix.ID().getText()}'
        if suffix.LBRACK() is not None:
            return f'{text}[{self.py_expr(suffix.jExpr())}]'
        if suffix.PIPE() is not None:
            name = suffix.ID().getText()
            args = self._args(suffix.jArgs()) if suffix.jArgs() is not None else []
            if name == 'join':
                sep = args[0] if args else "''"
                return f'{sep}.join(str(item) for item in {text})'
            arguments = ', '.join(args)
            if arguments:
                arguments = ', ' + arguments
            return f'script.filter({name!r}, {text}{arguments})'
        args = self._args(suffix.jArgs()) if suffix.jArgs() is not None else []
        if suffix.ID() is None and not text.endswith(')'):
            # A call on a bare name uses a template function.
            if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', text):
                self.needs_templates = True
        return f'{text}({", ".join(args)})'

    def _args(self, ctx):
        if ctx is None:
            return []
        rendered = []
        for arg in _nodes(ctx.jArg()):
            if arg.ASSIGN() is not None:
                rendered.append(f'{arg.ID().getText()}={self.py_expr(arg.jExpr())}')
            else:
                rendered.append(self.py_expr(arg.jExpr()))
        return rendered

    def _primary(self, ctx):
        if ctx.ID() is not None:
            return self._name(ctx.ID().getText())
        if ctx.NUMBER() is not None:
            return ctx.NUMBER().getText()
        if ctx.STRING() is not None:
            return ctx.STRING().getText()
        if ctx.jList() is not None or (ctx.LBRACK() is not None and ctx.jDict() is None and ctx.LPAREN() is None):
            items = []
            if ctx.jList() is not None:
                items = [self.py_expr(item) for item in _nodes(ctx.jList().jExpr())]
            return '[' + ', '.join(items) + ']'
        if ctx.LBRACE() is not None:
            pairs = []
            if ctx.jDict() is not None:
                for pair in _nodes(ctx.jDict().jDictPair()):
                    key, value = _nodes(pair.jExpr())
                    pairs.append(f'{self.py_expr(key)}: {self.py_expr(value)}')
            return '{' + ', '.join(pairs) + '}'
        if ctx.LPAREN() is not None:
            return self._paren(ctx)
        return ctx.getText()

    def _paren(self, ctx):
        group = ctx.jParenItems()
        if group is None:
            return '()'
        exprs = [self.py_expr(item) for item in _nodes(group.jExpr())]
        if len(exprs) == 1 and not _nodes(group.COMMA()):
            return f'({exprs[0]})'
        body = ', '.join(exprs)
        if len(exprs) == 1:
            body += ','
        return f'({body})'

    def _targets(self, ctx):
        return [token.getText() for token in _nodes(ctx.jTargets().ID())]

    def _span_text(self, ctx):
        """Python expression for the text inside a template block."""
        bits = []
        parts = []
        for piece in _nodes(ctx.piece()):
            parts.extend(self._piece_parts(piece))
        if parts:
            bits.append(self._string_expr(parts))
        for element in _nodes(ctx.element()):
            if element.stmt() is None:
                bits.append(repr('\n'))
                continue
            bits.append(self._string_expr(self._parts(element.stmt())))
            bits.append(repr('\n'))
        bits = [bit for bit in bits if bit != "''"]
        if not bits:
            return "''"
        if len(bits) == 1:
            return bits[0]
        return '(' + ' + '.join(bits) + ')'

    def visit_set(self, ctx):
        names = self._targets(ctx)
        targets = ', '.join(names)
        if ctx.ASSIGN() is not None:
            value = self.py_expr(ctx.jExpr())
        else:
            value = self._span_text(ctx)
        if self.assign_names:
            self.bound_names.update(names)
            self.emit(f'{targets} = {value}')
            return
        if ctx.ASSIGN() is not None and len(names) > 1:
            self.emit(f'(lambda {targets}: None)(*({value}))')
            return
        self.emit(value)

    def _set_inline(self, ctx):
        """Evaluate a ``{% set %}`` that is part of a larger line and keep its text empty."""
        if ctx.ASSIGN() is not None:
            names = self._targets(ctx)
            value = self.py_expr(ctx.jExpr())
            if len(names) == 1:
                return f'({value}, "")[1]'
            targets = ', '.join(names)
            return f'(lambda {targets}: "")(*({value}))'
        return f'({self._span_text(ctx)}, "")[1]'

    def visit_filter(self, ctx):
        self._emit_run_rendered(self._filter_expr(ctx))

    def _filter_expr(self, ctx):
        return self._apply_filters(ctx.jPostfix(), self._span_text(ctx))

    def _apply_filters(self, postfix, value):
        primary = postfix.jPrimary()
        if primary is None or primary.ID() is None:
            return value
        name = primary.ID().getText()
        suffixes = list(_nodes(postfix.jSuffix()))
        index = 0
        args = []
        if suffixes and suffixes[0].LPAREN() is not None and suffixes[0].PIPE() is None:
            args = self._args(suffixes[0].jArgs()) if suffixes[0].jArgs() is not None else []
            index = 1
        text = self._pipe(name, value, args)
        while index < len(suffixes):
            suffix = suffixes[index]
            if suffix.PIPE() is None or suffix.ID() is None:
                break
            name = suffix.ID().getText()
            args = self._args(suffix.jArgs()) if suffix.jArgs() is not None else []
            text = self._pipe(name, text, args)
            index += 1
        return text

    def _pipe(self, name, value, args):
        if name == 'join':
            sep = args[0] if args else "''"
            return f'{sep}.join(str(item) for item in {value})'
        arguments = ', '.join(args)
        if arguments:
            arguments = ', ' + arguments
        return f'script.filter({name!r}, {value}{arguments})'

    def visit_with(self, ctx):
        assigns = self._with_assigns(ctx)
        fresh = []
        for names, _expr in assigns:
            for name in names:
                if name not in self.bound_names:
                    fresh.append(name)
                self.bound_names.add(name)
        try:
            text = self._span_text(ctx)
        finally:
            for name in fresh:
                self.bound_names.discard(name)
        self._render_count += 1
        helper = f'_render_{self._render_count}'
        self.needs_runtime = True
        self.needs_run = True
        self.needs_dgenerate = True
        self.emit(f'def {helper}():')
        self.indent += 1
        for names, expr in assigns:
            self.emit(f'{", ".join(names)} = {expr}')
        self.emit(f'script.run_rendered({text})')
        self.indent -= 1
        self.emit(f'{helper}()')

    def _with_assigns(self, ctx):
        assigns = []
        for item in _nodes(ctx.jAssign()):
            assigns.append((self._targets(item), self.py_expr(item.jExpr())))
        return assigns

    def _with_expr(self, ctx):
        assigns = self._with_assigns(ctx)
        fresh = []
        params = []
        arguments = []
        for names, expr in assigns:
            params.extend(names)
            if len(names) == 1:
                arguments.append(expr)
            else:
                arguments.append(f'*({expr})')
            for name in names:
                if name not in self.bound_names:
                    fresh.append(name)
                self.bound_names.add(name)
        try:
            text = self._span_text(ctx)
        finally:
            for name in fresh:
                self.bound_names.discard(name)
        if not params:
            return f'(lambda: {text})()'
        return f'(lambda {", ".join(params)}: {text})({", ".join(arguments)})'

    def _emit_run_rendered(self, expr):
        self.needs_runtime = True
        self.needs_run = True
        self.needs_dgenerate = True
        self.emit(f'script.run_rendered({expr})')

    def _name(self, name):
        constants = {'true': 'True', 'false': 'False', 'none': 'None'}
        if name in constants:
            return constants[name]
        if (self.control or self.live_names) and name not in self.bound_names:
            if name in _PY_BUILTINS or name in _template_function_names():
                if name in _template_function_names():
                    self.used_functions.add(name)
                return name
            if name in {'env', 'glob', 'os'}:
                self.used_modules.add(name)
                return name
            if self.live_names:
                return f'script.live({name!r})'
            return f'script.eager({name!r})'
        if name == 'last_images':
            self.needs_run = True
            return 'written_images'
        if name == 'last_animations':
            self.needs_run = True
            return 'written_animations'
        if name.startswith('last_'):
            self.needs_run = True
            return f'last_config.{name[len("last_"):]}'
        return name


def _advance_quote(quote, text, last):
    """Track shell quotes. A quote starts only at the beginning of a word."""
    index = 0
    while index < len(text):
        char = text[index]
        previous = text[index - 1] if index else last
        if quote is None and char in {'"', "'"} and (not previous or previous.isspace()):
            quote = char
        elif quote == char and previous != '\\':
            quote = None
        last = char
        index += 1
    return quote, last


def _invocation_statement(parts):
    """Rebuild an invocation, leaving dynamic pieces as ``DGPY`` markers."""
    chunks = []
    py_holes = []
    quote = None
    last = ''
    for kind, value in parts:
        if kind == 'text':
            chunks.append(value)
            quote, last = _advance_quote(quote, value, last)
        elif kind == 'expr':
            constant = _constant_string(value)
            if constant is not None:
                chunks.append(constant)
                quote, last = _advance_quote(quote, constant, last)
            else:
                marker = f'DGPY{len(py_holes)}'
                chunks.append(marker)
                py_holes.append((value, quote))
                last = marker[-1]
        else:
            return None
    return ''.join(chunks), py_holes


def _attach_py_holes(replaced, hole_python, hole_quoted, py_holes):
    def replace(match):
        number = int(match.group(1))
        expression, quoted = py_holes[number]
        hole_python.append(expression)
        hole_quoted.append(quoted)
        return f'DGHOLE{len(hole_python) - 1}'

    return re.sub(r'DGPY(\d+)', replace, replaced)


def _expand_delayed_holes(replaced, holes, quoted, second_pass):
    """Turn ``{{ '{{ expr }}' }}`` into either literal text or a real hole.

    Outside a template block the expression is a string, so the invocation
    sees the characters ``{{ expr }}``. Inside a block those characters are
    rendered again when the line runs.
    """
    pieces = re.split(r'(DGHOLE\d+)', replaced)
    built = []
    new_holes = []
    new_quoted = []
    for piece in pieces:
        match = re.fullmatch(r'DGHOLE(\d+)', piece)
        if not match:
            built.append(piece)
            continue
        index = int(match.group(1))
        hole = holes[index]
        if not _delayed_hole(hole):
            new_holes.append(hole)
            new_quoted.append(quoted[index])
            built.append(f'DGHOLE{len(new_holes) - 1}')
            continue
        content = _constant_string(hole)
        if content is None:
            return None
        if not second_pass or '{{' not in content:
            built.append(content)
            continue
        sub_replaced, sub_holes, sub_quoted = _extract_holes(content)
        if not sub_holes:
            built.append(content)
            continue
        if quoted[index]:
            sub_quoted = [quoted[index]] * len(sub_holes)
        expanded = _expand_delayed_holes(sub_replaced, sub_holes, sub_quoted, True)
        if expanded is None:
            return None
        sub_replaced, sub_holes, sub_quoted = expanded

        def renumber(match, holes=sub_holes, quotes=sub_quoted):
            number = int(match.group(1))
            new_holes.append(holes[number])
            new_quoted.append(quotes[number])
            return f'DGHOLE{len(new_holes) - 1}'

        built.append(re.sub(r'DGHOLE(\d+)', renumber, sub_replaced))
    return ''.join(built), new_holes, new_quoted


def _delayed_hole(hole: str) -> bool:
    text = hole.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1].lstrip().startswith('{{')
    return False


def _jinja_expr_end(text, start):
    """Index just past the ``{{ }}`` that begins at ``start``."""
    depth = 2
    cursor = start + 2
    while cursor < len(text) and depth:
        if text.startswith('{{', cursor):
            depth += 2
            cursor += 2
            continue
        if text.startswith('}}', cursor):
            depth -= 2
            cursor += 2
            continue
        cursor += 1
    if depth:
        return None
    return cursor


def _read_jinja_block(text, start):
    """Return ``(source, end)`` for one balanced inline block, or None."""
    if text.startswith('{#', start):
        end = text.find('#}', start + 2)
        if end < 0:
            return None
        return text[start:end + 2], end + 2
    tag_end = text.find('%}', start)
    if tag_end < 0:
        return None
    name = text[start + 2:tag_end].strip().split(None, 1)[0]
    if name not in {'for', 'if'}:
        return None
    closer = 'endfor' if name == 'for' else 'endif'
    depth = 1
    cursor = tag_end + 2
    while cursor < len(text):
        nxt = text.find('{%', cursor)
        if nxt < 0:
            return None
        end = text.find('%}', nxt)
        if end < 0:
            return None
        inner = text[nxt + 2:end].strip().split(None, 1)[0]
        if inner == name:
            depth += 1
        elif inner == closer:
            depth -= 1
            if depth == 0:
                return text[start:end + 2], end + 2
        cursor = end + 2
    return None


def _shell_tokens(text):
    """Split one shell line without expanding it. Quoted tokens lose their quotes."""
    tokens = []
    buf = []
    quote = None
    started = None
    index = 0
    while index < len(text):
        char = text[index]
        previous = text[index - 1] if index else ''
        if quote is None and char in {'"', "'"} and not buf:
            quote = char
            started = char
            index += 1
            continue
        if quote == char and previous != '\\':
            quote = None
            index += 1
            continue
        if quote is None and char.isspace():
            if buf:
                tokens.append((''.join(buf), started))
                buf = []
                started = None
            index += 1
            continue
        buf.append(char)
        index += 1
    if quote is not None:
        return None
    if buf:
        tokens.append((''.join(buf), started))
    return tokens


def _directive_token_items(token, holes, token_quote):
    """One already-parsed directive argument. Quoted values are not parsed again."""
    if 'DGHOLE' not in token:
        if token_quote is None and re.search(r'[*?$%~]', token):
            return [f'*script.shell({token!r})']
        if token_quote == '"' and ('$' in token or '%' in token):
            return [f'script.expand({token!r})']
        return [repr(token)]
    bits = []
    for part in re.split(r'(DGHOLE\d+)', token):
        hole = re.fullmatch(r'DGHOLE(\d+)', part)
        if hole:
            bits.append(f'str({holes[int(hole.group(1))]})')
        elif part:
            bits.append(repr(part))
    combined = bits[0] if len(bits) == 1 else '(' + ' + '.join(bits) + ')'
    if token_quote is not None:
        return [combined]
    return [f'*script.shell({combined})']


def _extract_holes(text: str):
    pieces = []
    holes = []
    quoted = []
    index = 0
    quote = None
    while index < len(text):
        char = text[index]
        if text.startswith('{{', index):
            depth = 2
            cursor = index + 2
            while cursor < len(text) and depth:
                if text.startswith('{{', cursor):
                    depth += 2
                    cursor += 2
                    continue
                if text.startswith('}}', cursor):
                    depth -= 2
                    cursor += 2
                    continue
                cursor += 1
            if depth:
                return text, [], []
            holes.append(text[index + 2:cursor - 2].strip())
            quoted.append(quote)
            pieces.append(f'DGHOLE{len(holes) - 1}')
            index = cursor
            continue
        previous = text[index - 1] if index else ''
        if quote is None and char in {'"', "'"} and (index == 0 or previous.isspace()):
            quote = char
        elif quote == char and previous != '\\':
            quote = None
        pieces.append(char)
        index += 1
    return ''.join(pieces), holes, quoted


def _find_jinja_expr(ctx):
    if type(ctx).__name__ == 'JinjaExprContext':
        return ctx
    for child in getattr(ctx, 'getChildren', lambda: [])():
        found = _find_jinja_expr(child)
        if found is not None:
            return found
    return None


def _iter_shell_words(text):
    """Yield ``(start, end, quote)`` for each shell word.

    A quote counts only when it is the first character of the word.
    """
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        start = index
        quote = None
        if text[index] in {'"', "'"}:
            quote = text[index]
            index += 1
            while index < length:
                char = text[index]
                previous = text[index - 1]
                if char == quote and previous != '\\':
                    index += 1
                    break
                index += 1
        else:
            while index < length and not text[index].isspace():
                index += 1
        yield start, index, quote


def _word_needs_shell(raw, quote):
    """True when this shell word expands at runtime.

    Single quotes suppress expansion. Double quotes expand ``$`` and ``%``
    only. An unquoted word also expands ``~`` and globs ``*``.
    """
    if 'DGHOLE' in raw:
        return False
    if quote == "'":
        return False
    if quote == '"':
        return '$' in raw or '%' in raw
    return any(char in raw for char in '*$%~')


def _has_shell_word(text):
    for start, end, quote in _iter_shell_words(text):
        if _word_needs_shell(text[start:end], quote):
            return True
    return False


def _mask_shell_words(text):
    """Replace expandable shell words with ``DGSHELL`` markers.

    ``script.shell`` expands each marker when the script runs. Words that
    already contain a Jinja hole keep their own shell expression.
    """
    patterns = []
    pieces = []
    cursor = 0
    for start, end, quote in _iter_shell_words(text):
        pieces.append(text[cursor:start])
        raw = text[start:end]
        if _word_needs_shell(raw, quote):
            patterns.append(raw)
            pieces.append(f'DGSHELL{len(patterns) - 1}')
        else:
            pieces.append(raw)
        cursor = end
    pieces.append(text[cursor:])
    return ''.join(pieces), patterns


def _token_items(token: str, holes: list[str], quoted: list[bool], globs=()):
    glob = re.fullmatch(r'DGSHELL(\d+)', token)
    if glob:
        return [f'*script.shell({globs[int(glob.group(1))]!r})']
    numbers = [int(number) for number in re.findall(r'DGHOLE(\d+)', token)]
    if not numbers:
        return [repr(token)]
    protected = all(quoted[number] for number in numbers)
    match = re.fullmatch(r'DGHOLE(\d+)', token)
    if match and not protected:
        expression = holes[int(match.group(1))]
        # quote() exists so the shell does not split. Parsing its result
        # preserves that, including when the value is several words.
        return [f'*script.shell(str({expression}))']
    if 'DGHOLE' not in token:
        return [repr(token)]
    bits = []
    for part in re.split(r'(DGHOLE\d+)', token):
        hole = re.fullmatch(r'DGHOLE(\d+)', part)
        if hole:
            bits.append(f'str({holes[int(hole.group(1))]})')
        elif part:
            bits.append(repr(part))
    combined = bits[0] if len(bits) == 1 else '(' + ' + '.join(bits) + ')'
    if protected:
        char = quoted[numbers[0]]
        return [f'script.only(script.shell({char!r} + {combined} + {char!r}))']
    return [f'*script.shell({combined})']


def _value_expr(token: str, holes: list[str], quoted: list[bool], globs=()):
    items = _token_items(token, holes, quoted, globs)
    if len(items) == 1 and not items[0].startswith('*'):
        return items[0]
    return f'script.only([{", ".join(items)}])'


# These converters do work that has to happen when the script runs.
_RUNTIME_TYPES = {'_type_gen_seeds', '_type_latents'}


def _bake_value(expr, action):
    """Turn a literal argument into the value the parser would store.

    A shell word or a Jinja hole stays an expression. ``--gen-seeds`` and
    ``--latents`` stay runtime conversions.
    """
    if not isinstance(expr, str) or expr.startswith('*'):
        return None
    try:
        value = ast.literal_eval(expr)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(value, str):
        return None
    converter = getattr(action, 'type', None)
    if getattr(converter, '__name__', '') in _RUNTIME_TYPES:
        return None
    if converter is not None:
        try:
            value = converter(value)
        except Exception:
            return None
    return _py_value(value)


def _typed_list_expr(tokens, holes, quoted, globs, action, option):
    """One list field. Literal items are baked. The rest convert at runtime."""
    pieces = []
    for token in tokens:
        pieces.extend(_token_items(token, holes, quoted, globs))
    items = []
    converter = getattr(action, 'type', None)
    for piece in pieces:
        if piece.startswith('*'):
            inner = piece[1:]
            if converter is None:
                items.append('*' + inner)
            else:
                items.append(f'*script.typed_list({option!r}, {inner})')
            continue
        baked = _bake_value(piece, action)
        if baked is not None:
            items.append(baked)
            continue
        if converter is None:
            items.append(piece)
        else:
            items.append(f'script.typed({option!r}, {piece})')
    return '[' + ', '.join(items) + ']'


def _split_first(text):
    stripped = text.lstrip()
    if not stripped:
        return '', ''
    parts = stripped.split(None, 1)
    if len(parts) == 1:
        return parts[0], ''
    return parts[0], parts[1]


def _option_values(parser, argv):
    """Map each destination to the option token and the raw strings that set it."""
    options = {}
    positional = None
    for action in parser._actions:
        if action.option_strings:
            for option in action.option_strings:
                options[option] = action
        elif action.dest == 'model_path':
            positional = action
    found = {}
    index = 0
    if positional is not None and argv and not str(argv[0]).startswith('-'):
        found[positional.dest] = (positional.dest, [str(argv[0])], positional)
        index = 1
    while index < len(argv):
        token = str(argv[index])
        action = options.get(token)
        if action is None:
            return None
        index += 1
        if action.nargs == 0:
            found[action.dest] = (token, [], action)
            continue
        if action.nargs in (None, 1):
            count = 1
        elif action.nargs in {'+', '*'}:
            values = []
            while index < len(argv) and not str(argv[index]).startswith('-'):
                values.append(str(argv[index]))
                index += 1
            found[action.dest] = (token, values, action)
            continue
        elif isinstance(action.nargs, int):
            count = action.nargs
        else:
            return None
        if index + count > len(argv):
            return None
        values = [str(item) for item in argv[index:index + count]]
        index += count
        found[action.dest] = (token, values, action)
    return found


def _lower_static_invocation(statement: str):
    try:
        import dgenerate.arguments
        import dgenerate.textprocessing
    except Exception:
        return None
    try:
        argv = dgenerate.textprocessing.shell_parse(
            statement,
            expand_home=False,
            expand_vars=False,
            expand_glob=False,
        )
        namespace = dgenerate.arguments.DgenerateArguments()
        names = []
        for cls in reversed(type(namespace).__mro__):
            names.extend(getattr(cls, '__annotations__', {}))
        before = {}
        for name in names:
            try:
                before[name] = getattr(namespace, name)
            except Exception:
                continue
        parser = dgenerate.arguments._create_parser(add_model=True, add_help=False)[0]
        parsed, unknown = parser.parse_known_args(argv, namespace=namespace)
        assignments = _option_values(parser, argv)
    except Exception:
        return None
    if unknown:
        return None
    lines = []
    seen = set()
    for name in names:
        if name in seen or name.startswith('_'):
            continue
        seen.add(name)
        try:
            value = getattr(parsed, name)
        except Exception:
            continue
        if name in before and value == before[name]:
            continue
        assignment = None if assignments is None else assignments.get(name)
        if assignment is not None and getattr(assignment[2].type, '__name__', '') == '_type_gen_seeds':
            option, values, action = assignment
            if len(values) == 1 and action.nargs in (None, 1):
                lines.append(f'config.{name} = script.typed({option!r}, {values[0]!r})')
            else:
                lines.append(f'config.{name} = script.typed_list({option!r}, {values!r})')
            continue
        if assignment is None and any(
            getattr(action.type, '__name__', '') == '_type_gen_seeds'
            for action in parser._actions
            if action.dest == name
        ):
            return None
        rendered = _py_value(value)
        if rendered is None:
            return None
        lines.append(f'config.{name} = {rendered}')
    return lines or None


def _py_value(value):
    if isinstance(value, str):
        return repr(value)
    if value is None or isinstance(value, (int, float, bool)):
        return repr(value)
    if isinstance(value, enum.Enum):
        return f'dgenerate.{type(value).__qualname__}.{value.name}'
    if isinstance(value, pathlib.PurePath):
        return repr(str(value))
    if type(value).__name__ == 'Prompt':
        text = str(value)
        if not text and value.positive is None:
            return 'Prompt()'
        return f'Prompt.parse({text!r})'
    if isinstance(value, list):
        items = [_py_value(item) for item in value]
        if any(item is None for item in items):
            return None
        return '[' + ', '.join(items) + ']'
    if isinstance(value, tuple):
        items = [_py_value(item) for item in value]
        if any(item is None for item in items):
            return None
        inner = ', '.join(items)
        if len(items) == 1:
            inner += ','
        return '(' + inner + ')'
    rendered = repr(value)
    if '<' in rendered:
        return None
    try:
        ast.parse(rendered)
    except SyntaxError:
        return None
    return rendered


class _Pretty:
    """Wrap calls and collections that do not fit on one line."""

    def __init__(self, width):
        self.width = width

    def statement(self, stmt):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            prefix = ast.unparse(stmt.targets[0]) + ' = '
            lines = self.placed(stmt.value, len(prefix), 0)
            lines[0] = prefix + lines[0]
            return lines
        if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            prefix = (
                f'{ast.unparse(stmt.target)}: {ast.unparse(stmt.annotation)} = '
            )
            lines = self.placed(stmt.value, len(prefix), 0)
            lines[0] = prefix + lines[0]
            return lines
        if isinstance(stmt, ast.Expr):
            return self.placed(stmt.value, 0, 0)
        return ast.unparse(stmt).splitlines()

    def placed(self, node, column, level):
        """One line when it fits at ``column``. A break indents under ``level``."""
        single = ast.unparse(node)
        if column + len(single) <= self.width:
            return [single]
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return self.sequence(node, level)
        if isinstance(node, ast.Dict):
            return self.mapping(node, level)
        if isinstance(node, ast.Call):
            return self.call(node, level)
        if isinstance(node, ast.Starred):
            lines = self.placed(node.value, column + 1, level + 1)
            lines[0] = '*' + lines[0]
            return lines
        return [single]

    def sequence(self, node, level):
        if isinstance(node, ast.List):
            open_mark, close_mark, elts = '[', ']', node.elts
        elif isinstance(node, ast.Tuple):
            open_mark, close_mark, elts = '(', ')', node.elts
        else:
            open_mark, close_mark, elts = '{', '}', node.elts
        return self._group(open_mark, close_mark, elts, level, labeled=False)

    def mapping(self, node, level):
        pairs = []
        for key, value in zip(node.keys, node.values):
            if key is None:
                starred = ast.Starred(value=value)
                starred.double = True
                pairs.append(starred)
            else:
                pairs.append((key, value))
        return self._group('{', '}', pairs, level, labeled=True)

    def call(self, node, level):
        items = list(node.args)
        for keyword in node.keywords:
            if keyword.arg is None:
                starred = ast.Starred(value=keyword.value)
                starred.double = True
                items.append(starred)
            else:
                items.append(keyword)
        if not items:
            return [ast.unparse(node)]
        pad = ' ' * (level + 4)
        lines = [ast.unparse(node.func) + '(']
        for item in items:
            if isinstance(item, ast.keyword):
                parts = self.placed(
                    item.value, level + 4 + len(item.arg) + 1, level + 4,
                )
                parts[0] = f'{item.arg}={parts[0]}'
            elif isinstance(item, ast.Starred) and getattr(item, 'double', False):
                parts = self.placed(item.value, level + 6, level + 4)
                parts[0] = '**' + parts[0]
            else:
                parts = self.placed(item, level + 4, level + 4)
            parts[0] = pad + parts[0]
            parts[-1] += ','
            lines.extend(parts)
        lines.append(' ' * level + ')')
        return lines

    def _group(self, open_mark, close_mark, elts, level, labeled):
        if not elts:
            return [open_mark + close_mark]
        pad = ' ' * (level + 4)
        lines = [open_mark]
        for elt in elts:
            if labeled and isinstance(elt, tuple):
                key, value = elt
                key_text = ast.unparse(key)
                parts = self.placed(value, level + 4 + len(key_text) + 2, level + 4)
                parts[0] = f'{key_text}: {parts[0]}'
            elif isinstance(elt, ast.Starred) and getattr(elt, 'double', False):
                parts = self.placed(elt.value, level + 6, level + 4)
                parts[0] = '**' + parts[0]
            else:
                parts = self.placed(elt, level + 4, level + 4)
            parts[0] = pad + parts[0]
            parts[-1] += ','
            lines.extend(parts)
        lines.append(' ' * level + close_mark)
        return lines


def _statement_kind(body: str, previous: str) -> str | None:
    """How a generated line should be spaced, or None when it continues the previous one."""
    if body.startswith('#'):
        return None
    if re.match(r'_items\d+ = ', body) or body.startswith('loop = script.loop('):
        return 'flow'
    if re.fullmatch(r'if _items\d+:', body):
        return None
    if previous and re.match(r'_items\d+ = ', previous) and body.startswith('for '):
        return None
    if body.startswith(('elif ', 'else:')):
        return None
    if body.startswith('for ') or body.startswith('if ') or body.startswith('while '):
        return 'flow'
    if body.startswith('with script.block'):
        return 'block'
    if body.startswith('with script.line('):
        return 'config'
    if body.startswith('with script.options('):
        return None
    if body.startswith('def '):
        return 'block'
    if body.startswith('with '):
        return 'block'
    if re.fullmatch(r'_render_\d+\(\)', body):
        return None
    if body.startswith('script.version_check('):
        return 'version'
    if body.startswith('script.set_expr('):
        return 'set'
    if body.startswith('script.run_rendered('):
        return 'block'
    if (
        body.startswith('script.run(')
        or body.startswith('script.sync(')
        or body.startswith('render_loop.')
        or body.startswith('config.')
        or body == 'pass'
    ):
        return None
    directive = _directive_name(body)
    if directive in {'set', 'setp', 'sete'}:
        return 'set'
    if directive == 'print' or body.startswith('print('):
        return 'print'
    if directive == 'exit':
        return 'exit'
    if directive is not None or body.startswith((
        'script.directive(', 'script.run_directive(', 'script.invoke(',
        'script.step(', 'script.run_argv(', 'os.environ',
    )):
        return 'directive'
    if body.startswith('config:') or body.startswith('config ='):
        return 'config'
    if re.match(r'[A-Za-z_][A-Za-z0-9_]*\s*=', body):
        return 'set'
    if body.startswith('(lambda ') or re.fullmatch(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', body):
        return 'set'
    return None


def _directive_name(body: str) -> str | None:
    match = re.match(
        r"script\.(?:directive|run_directive)\((?:\d+\s*,\s*)?'([A-Za-z_][A-Za-z0-9_]*)'",
        body,
    )
    if match is None:
        return None
    return match.group(1)


def _same_group(prior: str, kind: str) -> bool:
    if prior == kind == 'version':
        return True
    if prior == kind == 'set':
        return True
    if prior == 'print' and kind == 'exit':
        return True
    if prior == 'flow' and kind == 'flow':
        return False
    return False


def _space_statements(lines: list[str]) -> list[str]:
    """Blank line between statements, keeping version checks, sets, and print/exit together."""
    spaced = []
    previous_kind = {}
    for line in lines:
        body = line.strip()
        if not body:
            spaced.append(line)
            continue
        indent = len(line) - len(line.lstrip(' '))
        previous = ''
        for earlier in reversed(spaced):
            if earlier.strip():
                previous = earlier.strip()
                break
        if body.startswith('#'):
            # A comment introduces the next section. Keep a run of comment
            # lines together, and do not split one off from the block it sits in.
            if (
                previous
                and not previous.startswith('#')
                and not previous.endswith(':')
                and spaced[-1].strip()
            ):
                spaced.append('')
            spaced.append(line)
            continue
        kind = _statement_kind(body, previous)
        if kind is None:
            spaced.append(line)
            continue
        for depth in [depth for depth in previous_kind if depth > indent]:
            del previous_kind[depth]
        prior = previous_kind.get(indent)
        if prior is not None and not _same_group(prior, kind):
            if spaced and spaced[-1].strip():
                spaced.append('')
        previous_kind[indent] = kind
        spaced.append(line)
    return spaced


def _format_source(source: str, width: int = 88) -> str:
    """Wrap long calls and collections, and separate invocations."""
    formatted = []
    for line in source.splitlines():
        formatted.extend(_format_line(line, width))
    separated = []
    for line in formatted:
        stripped = line.strip()
        if (
            (stripped.startswith('config:') or stripped.startswith('config ='))
            and separated
            and separated[-1].strip()
            and not separated[-1].strip().endswith(':')
        ):
            separated.append('')
        separated.append(line)
    return '\n'.join(separated).rstrip() + '\n'


def _format_line(line: str, width: int):
    body = line.strip()
    if not body or body.startswith('#') or len(line) <= width:
        return [line]
    indent = line[:len(line) - len(line.lstrip())]
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return [line]
    pretty = _Pretty(max(width - len(indent), 40))
    rows = []
    for stmt in tree.body:
        rows.extend(pretty.statement(stmt))
    return [indent + row if row else row for row in rows]


def transpile(text: str, relaxed: bool = False, format_output: bool = False,
              comments: bool = True) -> str:
    """Translate a config into Python.

    ``relaxed=False`` reproduces the interpreter's line numbers and error text.
    ``relaxed=True`` emits the same actions without that error bookkeeping.
    ``format_output=True`` wraps long calls and lists and leaves a blank line
    between invocations. ``comments=False`` strips config comments. The default
    keeps them, including comments on their own line and comments at the end
    of a line.
    """
    source = Transpiler(strict=not relaxed, comments=comments).compile(text)
    if format_output:
        return _format_source(source)
    return source


def main():
    parser = argparse.ArgumentParser(description='Transpile a dgenerate config using the ANTLR grammar.')
    parser.add_argument('--stdin', action='store_true', help='Read the config from stdin instead of a file.')
    parser.add_argument('config', nargs='?', help='Path to a .dgen config.')
    parser.add_argument('--tree', action='store_true', help='Print the parse tree and exit')
    parser.add_argument('--relaxed', action='store_true', help='Omit interpreter error-line bookkeeping')
    parser.add_argument(
        '--format',
        action='store_true',
        help='Wrap long calls and lists, and leave a blank line between invocations.',
    )
    parser.add_argument(
        '--comments',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Keep config comments in the script. They are left out by default.',
    )
    args = parser.parse_args()
    if args.stdin and args.config:
        print('Pass a config path or --stdin, not both.', file=sys.stderr)
        sys.exit(1)
    if args.stdin:
        text = sys.stdin.read()
    elif args.config:
        text = pathlib.Path(args.config).read_text(encoding='utf-8')
    else:
        print('A config path is required, or use --stdin.', file=sys.stderr)
        sys.exit(1)
    try:
        tree = parse_config(text)
    except _ParseError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    if args.tree:
        print(tree.toStringTree(recog=Config(None)))
        return
    print(transpile(
        text, relaxed=args.relaxed, format_output=args.format, comments=args.comments,
    ), end='')


if __name__ == '__main__':
    main()
