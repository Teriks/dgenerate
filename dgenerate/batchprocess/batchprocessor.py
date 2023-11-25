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

import io
import os
import re
import shlex
import typing

import jinja2

import dgenerate.arguments as _arguments
import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types


class PeekReader:
    """
    Read from a ``TextIO`` file object while peeking at the next line in the file.

    This is an iterable reader wrapper that yields the tuple (current_line, next_line)

    **next_line** will be ``None`` if the next line is the end of the file.
    """

    def __init__(self, file: typing.TextIO):
        """
        :param file: The ``TextIO`` reader to wrap.
        """
        self._file = file
        self._last_next_line = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._last_next_line is not None:
            self._cur_line = self._last_next_line
            self._last_next_line = None
        else:
            self._cur_line = next(self._file)

        try:
            self._next_line = next(self._file)
            self._last_next_line = self._next_line
        except StopIteration:
            self._next_line = None

        return self._cur_line, self._next_line


class BatchProcessError(Exception):
    """
    Thrown on errors encountered within a batch processing script
    including non-zero return codes from the invoker.
    """
    pass


class BatchProcessor:
    """
    Implements dgenerates batch processing scripts in a generified manner.

    This is the bare-bones implementation of the shell with nothing
    implemented for you except the ``\\print`` and ``\\set`` directives.

    If you wish to create this object to run a dgenerate configuration, use
    :py:class:`dgenerate.batchprocess.ConfigRunner`
    """

    invoker: typing.Callable[[list], int]
    """
    Invoker function, responsible for executing lines recognized as shell commands.
    """

    name: _types.Name
    """
    Name of this batch processor, currently used in the hash bang version check directive and messages.
    """

    version: _types.Version
    """
    Version tuple for the version check hash bang directive.
    """

    template_variables: dict[str, typing.Any]
    """
    Live template variables.
    """

    template_functions: dict[str, typing.Callable[[typing.Any], typing.Any]]
    """
    Functions available when templating is occurring.
    """

    directives: dict[str, typing.Optional[typing.Callable[[list], int]]]
    """
    Batch process directives, shell commands starting with a backslash.
    
    Dictionary of callable(list) -> int.
    
    The function should return a return code, 0 for success, anything else for failure.
    """

    injected_args: list[str]
    """
    Shell arguments to inject at the end of every invocation.
    """

    def __init__(self,
                 invoker: typing.Callable[[list], int],
                 name: _types.Name,
                 version: typing.Union[_types.Version, str],
                 template_variables: typing.Optional[dict[str, typing.Any]] = None,
                 template_functions: typing.Optional[
                     dict[str, typing.Callable[[typing.Any], typing.Any]]] = None,
                 directives: dict[str, typing.Optional[typing.Callable[[list], None]]] = None,
                 injected_args: typing.Optional[list[str]] = None):
        """
        :param invoker: A function for invoking lines recognized as shell commands, should return a return code.
        :param name: The name of this batch processor, currently used in the version check directive and messages
        :param version: Version for version check hash bang directive.
        :param template_variables: Live template variables, the initial environment, this dictionary will be
            modified during runtime.
        :param template_functions: Functions available to Jinja2
        :param directives: batch processing directive handlers, for: *\\\\directives*. This is a dictionary
            of names to functions which accept a single parameter, a list of directive arguments, and return
            a return code.
        :param injected_args: Arguments to be injected at the end of user specified arguments for every shell invocation.
            If ``-v/--verbose`` is present in ``injected_args`` debugging output will be enabled globally while the config
            runs, and not just for invocations. Passing ``-v/--verbose`` also disables handling of unhandled non
            :py:exc:`SystemExit` exceptions raised by config directive implementations, a stack trace will be
            printed when these exceptions are encountered.
        """

        self._template_functions = None
        self.invoker = invoker
        self.name = name

        self.template_variables = template_variables if template_variables else dict()
        self.template_functions = template_functions if template_functions else dict()

        self.directives = directives if directives else dict()

        self._directive_exceptions = False

        self.injected_args = injected_args if injected_args else []

        self._current_line = 0

        if isinstance(version, str):
            ver_parts = version.split('.')
            if len(ver_parts) != 3:
                raise ValueError(
                    f'version expected to be a version string in the format major.minor.patch. received: "{version}"')
            self.version: tuple[int, int, int] = \
                (int(ver_parts[0]), int(ver_parts[1]), int(ver_parts[2]))
        else:
            self.version: tuple[int, int, int] = tuple(version)
            if len(self.version) != 3:
                raise ValueError(
                    f'version tuple expected to contain three components: (major, minor, patch). received: {self.version}')

        self.expand_vars = os.path.expandvars

    @property
    def current_line(self) -> int:
        """
        The current line in the file being processed.
        """
        return self._current_line

    def render_template(self, string: str):
        """
        Render a template from a string
        
        :param string: the string containing the Jinja2 template.
        :return: rendered string
        """

        jinja_env = jinja2.Environment()

        for name, func in self.template_functions.items():
            jinja_env.globals[name] = func
            jinja_env.filters[name] = func

        try:
            return self.expand_vars(
                jinja_env.from_string(string).
                render(**self.template_variables))
        except jinja2.TemplateSyntaxError as e:
            raise BatchProcessError(f'Template Syntax Error: {e}')

    def _look_for_version_mismatch(self, line_idx, line):
        versioning = re.match(r'#!\s+' + self.name + r'\s+([0-9]+\.[0-9]+\.[0-9]+)', line)
        if versioning:
            config_file_version = versioning.group(1)
            config_file_version_parts = config_file_version.split('.')

            cur_major_version = self.version[0]
            config_major_version = int(config_file_version_parts[0])
            cur_minor_version = self.version[1]
            config_minor_version = int(config_file_version_parts[1])

            version_str = '.'.join(str(i) for i in self.version)

            if cur_major_version != config_major_version:
                _messages.log(
                    f'Failed version check (major version missmatch) on line {line_idx}, '
                    f'running an incompatible version of {self.name}! You are running version {version_str} '
                    f'and the config file specifies the required version: {config_file_version}'
                    , underline=True, level=_messages.WARNING)
            elif cur_minor_version < config_minor_version:
                _messages.log(
                    f'Failed version check (current minor version less than requested) '
                    f'on line {line_idx}, running an incompatible version of {self.name}! '
                    f'You are running version {version_str} and the config file specifies '
                    f'the required version: {".".join(config_file_version)}'
                    , underline=True, level=_messages.WARNING)

    def _jinja_user_define(self, name, value):
        if name in self.template_functions:
            raise BatchProcessError(
                f'Cannot define template variable "{name}" on line {self.current_line}, '
                f'as that name is taken by a template function.')
        self.template_variables[name] = value

    def _directive_handlers(self, line_idx, line):
        if line.startswith('\\set'):
            directive_args = line.split(' ', 2)
            if len(directive_args) == 3:
                self._jinja_user_define(directive_args[1].strip(), self.render_template(directive_args[2].strip()))
                return True
            else:
                raise BatchProcessError(
                    f'\\set directive received less than 2 arguments, '
                    f'syntax is: \\set name value')
        elif line.startswith('\\print'):
            directive_args = line.split(' ', 1)
            if len(directive_args) == 2:
                _messages.log(self.render_template(directive_args[1].strip()))
                return True
            else:
                raise BatchProcessError(
                    f'\\print directive received no arguments, '
                    f'syntax is: \\print value')
        if line.startswith('{'):
            self.run_string(self.render_template(line))
            return True
        elif line.startswith('\\'):
            directive_args = line.split(' ', 1)

            directive = directive_args[0].lstrip('\\')
            impl = self.directives.get(directive)
            if impl is None:
                raise BatchProcessError(f'Unknown directive "\\{directive}".')
            directive_args = directive_args[1:]
            try:
                if directive_args:
                    return_code = impl(shlex.split(directive_args[0]))
                else:
                    return_code = impl([])
                if return_code != 0:
                    raise BatchProcessError(
                        f'Directive error return code: {return_code}')
            except Exception as e:
                if self._directive_exceptions:
                    raise e
                raise BatchProcessError(e)
            return True
        return False

    def _lex_and_run_invocation(self, line_idx, invocation_string):
        raw_templated_string = self.render_template(invocation_string)

        shell_lexed = shlex.split(raw_templated_string) + self.injected_args

        raw_injected_args = ' '.join(str(a) for a in self.injected_args)

        if raw_injected_args:
            cmd_info = raw_templated_string + ' ' + raw_injected_args
        else:
            cmd_info = raw_templated_string

        header = 'Processing Arguments: '
        args_wrapped = \
            _textprocessing.wrap(
                cmd_info,
                width=_textprocessing.long_text_wrap_width() - len(header),
                subsequent_indent=' ' * len(header))

        _messages.log(header + args_wrapped, underline=True)

        return_code = self.invoker(shell_lexed)

        if return_code != 0:
            raise BatchProcessError(
                f'Invocation error return code: {return_code}')

    def _run_file(self, stream: typing.TextIO):
        continuation = ''
        top_level_template = False

        def run_continuation(cur_line, cur_line_idx):
            nonlocal continuation, top_level_template
           
            if not top_level_template:
                completed_continuation = (continuation + ' ' + cur_line).strip()
            else:
                completed_continuation = (continuation + cur_line).strip()

            top_level_template = False
            continuation = ''

            if self._directive_handlers(cur_line_idx, completed_continuation):
                return

            self._lex_and_run_invocation(cur_line_idx, completed_continuation)

        last_line = None
        line_idx = 0

        for line_idx, line_and_next in enumerate(PeekReader(stream)):
            line, next_line = line_and_next

            line_rstrip = line.rstrip()
            line_lstrip = line.lstrip()
            line_strip = line_rstrip.lstrip()

            last_line_lstrip = None if last_line is None else last_line.lstrip()
            last_line_rstrip = None if last_line is None else last_line.rstrip()
  
            next_line_lstrip = None if next_line is None else next_line.lstrip()

            self._current_line = line_idx

            if line_strip == '':
                if continuation and last_line is not None \
                        and last_line_lstrip.startswith('-') \
                        and not last_line_rstrip.endswith('\\'):
                    run_continuation('', line_idx)
            elif line_lstrip.startswith('#'):
                self._look_for_version_mismatch(line_idx, line)
            elif line_lstrip.startswith('{') and not top_level_template:
                continuation += line
                top_level_template = True
            elif not top_level_template and line_strip.endswith('\\') \
                    or next_line and next_line_lstrip.startswith('-'):
                continuation += ' ' + line_strip.rstrip(' \\')
            elif top_level_template:
                if line_rstrip.endswith('!END'):
                    run_continuation(line_rstrip.removesuffix('!END'), line_idx)
                    top_level_template = False
                else:
                    continuation += line
            else:
                run_continuation(line, line_idx)

            last_line = line

        if continuation:
            run_continuation('', line_idx)

    def run_file(self, stream: typing.TextIO):
        """
        Process a batch processing script from a file stream

        :raise BatchProcessError:

        :param stream: A filestream in text read mode
        """

        try:
            parsed, _ = _arguments.parse_known_args(self.injected_args, log_error=False)
        except _arguments.DgenerateUsageError as e:
            raise BatchProcessError(f'Error parsing injected arguments: {e}')

        directive_exceptions_last = self._directive_exceptions
        if parsed.verbose:
            _messages.push_level(_messages.DEBUG)
            self._directive_exceptions = True

        try:
            self._run_file(stream)
        except BatchProcessError as e:
            raise BatchProcessError(f'Error on line {self.current_line}: {e}')
        finally:
            _messages.pop_level()
            self._directive_exceptions = directive_exceptions_last

    def run_string(self, string: str):
        """
        Process a batch processing script from a string

        :raise BatchProcessError:

        :param string: a string containing the script
        """
        self.run_file(io.StringIO(string))


__all__ = _types.module_all()
