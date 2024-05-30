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
import inspect
import io
import itertools
import os
import re
import typing

import asteval
import jinja2

import dgenerate.arguments as _arguments
import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
import dgenerate.files as _files


class BatchProcessError(Exception):
    """
    Thrown by :py:meth:`.BatchProcessor.run_file` and :py:meth:`.BatchProcessor.run_string`
    when an error in a batch processing script is encountered.
    """
    pass


class BatchProcessor:
    """
    Implements dgenerates batch processing scripts in a generified manner.

    This is the bare-bones implementation of the shell with nothing
    implemented for you except the ``\\print``,  ``\\set``, ``\\setp``,
    and `\\unset`` directives.

    If you wish to create this object to run a dgenerate configuration, use
    :py:class:`dgenerate.batchprocess.ConfigRunner`
    """

    invoker: typing.Callable[[collections.abc.Sequence[str]], int]
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

    reserved_template_variables: set[str]
    """
    These template variables cannot be set with the ``\\set`` or ``\\setp`` 
    directive, or un-defined with the ``\\unset`` directive.
    """

    template_functions: dict[str, typing.Callable[[typing.Any], typing.Any]]
    """
    Functions available when templating is occurring.
    """

    builtins: dict[str, typing.Callable[[typing.Any], typing.Any]]
    """
    Safe python builtins that are always available as template functions and also usable with ``\\setp``
    
    They may be overridden by functions defined in :py:attr:`dgenerate.batchprocess.BatchProcessor.template_functions`
    """

    directives: dict[str, typing.Optional[typing.Callable[[collections.abc.Sequence[str]], int]]]
    """
    Batch process directives, shell commands starting with a backslash.
    
    Dictionary of callable(list) -> int.
    
    The function should return a return code, 0 for success, anything else for failure.
    """

    injected_args: collections.abc.Sequence[str]
    """
    Shell arguments to inject at the end of every invocation.
    """

    expand_vars: typing.Callable[[str], str]
    """
    A function for expanding environmental variables, defaults to :py:func:`os.path.expandvars`
    """

    def __init__(self,
                 invoker: typing.Callable[[collections.abc.Sequence[str]], int],
                 name: _types.Name,
                 version: typing.Union[_types.Version, str],
                 template_variables: typing.Optional[dict[str, typing.Any]] = None,
                 reserved_template_variables: typing.Optional[set[str]] = None,
                 template_functions: typing.Optional[
                     dict[str, typing.Callable[[typing.Any], typing.Any]]] = None,
                 directives: dict[str, typing.Optional[typing.Callable[[list], None]]] = None,
                 builtins: typing.Optional[dict[str, typing.Callable[[typing.Any], typing.Any]]] = None,
                 injected_args: typing.Optional[collections.abc.Sequence[str]] = None):
        """
        :param invoker: A function for invoking lines recognized as shell commands, should return a return code.
        :param name: The name of this batch processor, currently used in the version check directive and messages
        :param version: Version for version check hash bang directive.
        :param template_variables: Live template variables, the initial environment, this dictionary will be
            modified during runtime.
        :param reserved_template_variables: These template variable names cannot be set with the
            ``\\set`` or ``\\setp`` directive, or un-defined with the ``\\unset`` directive.
        :param template_functions: Functions available to Jinja2
        :param directives: batch processing directive handlers, for: ``\\directives``. This is a dictionary
            of names to functions which accept a single parameter, a list of directive arguments, and return
            a return code.
        :param builtins: builtin functions available as template functions and ``\\setp`` functions.
            A safe default collection of functions is used if this is not specified.  Builtins may
            be overridden by functions defined in ``template_functions``
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
        self.reserved_template_variables = reserved_template_variables if reserved_template_variables else set()

        self.template_functions = template_functions if template_functions else dict()

        self.directives = directives if directives else dict()

        self._directive_exceptions = False

        self.injected_args = injected_args if injected_args else []

        self._current_line = 0
        self._executing_text = None
        self._running_template_continuation = False

        if isinstance(version, str):
            self.version = _textprocessing.parse_version(version)
        else:
            self.version: tuple[int, int, int] = tuple(version)
            if len(self.version) != 3:
                raise ValueError(
                    f'version tuple expected to contain three components: (major, minor, patch). received: {self.version}')

        self.expand_vars = os.path.expandvars

        if builtins is None:
            self.builtins = {
                'abs': abs,
                'all': all,
                'any': any,
                'ascii': ascii,
                'bin': bin,
                'bool': bool,
                'bytearray': bytearray,
                'bytes': bytes,
                'callable': callable,
                'chr': chr,
                'complex': complex,
                'dict': dict,
                'divmod': divmod,
                'enumerate': enumerate,
                'filter': filter,
                'float': float,
                'format': format,
                'frozenset': frozenset,
                'getattr': getattr,
                'hasattr': hasattr,
                'hash': hash,
                'hex': hex,
                'int': int,
                'iter': iter,
                'len': len,
                'list': list,
                'map': map,
                'max': max,
                'min': min,
                'next': next,
                'object': object,
                'oct': oct,
                'ord': ord,
                'pow': pow,
                'range': range,
                'repr': repr,
                'reversed': reversed,
                'round': round,
                'set': set,
                'slice': slice,
                'sorted': sorted,
                'str': str,
                'sum': sum,
                'tuple': tuple,
                'type': type,
                'zip': zip,
            }
        else:
            self.builtins = builtins

    @property
    def directives_builtins_help(self):
        """
        Returns a dictionary of help strings for directives that are
        built into the :py:class:`BatchProcessor` base class.
        """

        return {
            'set': 'Sets a template variable, accepts two arguments, the variable name and the value. '
                   'Attempting to set a reserved template variable such as those pre-defined by dgenerate '
                   'will result in an error. The second argument is accepted as a raw value, it is not shell '
                   'parsed in any way, only stripped of leading and trailing whitespace.',
            'sete': 'Sets a template variable to an array of shell arguments using shell parsing and expansion. '
                    'For example, this could be utilized for convenient shell globbing: '
                    '\\setp my_files my_directory1/* my_directory2/*',
            'setp': 'Sets a template variable to a (safely) evaluated Python expression, accepts two arguments, '
                    'the variable name and the value. Attempting to set a reserved template variable such '
                    'as those pre-defined by dgenerate will result in an error. Template variables can be '
                    'referred to by name within a definition, EG: \\setp my_list [1, 2, my_var, 4]. Template '
                    'functions are also available, EG: \\setp working_dir cwd(). Python unary and binary '
                    'expression operators, python list slicing, and comprehensions are supported. '
                    'This functionality is provided by the asteval package.',
            'unset': 'Undefines a template variable previously set with \\set or \\setp, accepts one argument, '
                     'the variable name. Attempting to unset a reserved variable such as those '
                     'pre-defined by dgenerate will result in an error.',
            'print': 'Prints all content to the right to stdout, no shell parsing of the argument occurs.',
            'echo': 'Echo shell arguments with shell parsing and expansion.'
        }

    @property
    def current_line(self) -> int:
        """
        The current line number in the file being processed.
        """
        return self._current_line

    @property
    def executing_text(self) -> typing.Union[None, str]:
        """
        The text / command line that is currently
        being executed, or that was last executed.
        """
        return self._executing_text

    def render_template(self, string: str, stream: bool = False) -> typing.Union[str, typing.Iterator[str]]:
        """
        Render a template from a string

        :param string: the string containing the Jinja2 template.
        :param stream: Stream the results of generating this template line by line?
        :return: rendered string
        """

        jinja_env = jinja2.Environment()

        for name, func in self.builtins.items():
            if name in jinja_env.globals:
                continue
            jinja_env.globals[name] = func

        for name, func in self.template_functions.items():
            jinja_env.globals[name] = func

            if len(inspect.signature(func).parameters) > 0:
                jinja_env.filters[name] = func

        if stream:
            def stream_generator():
                buffer = ''
                try:
                    for piece in jinja_env.from_string(string). \
                            stream(**self.template_variables):
                        buffer += piece
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            yield os.path.expandvars(line)
                    if buffer:
                        yield os.path.expandvars(buffer)
                except Exception as e:
                    raise BatchProcessError(f'Template Render Error: {str(e).strip()}')

            return stream_generator()
        else:
            try:
                return self.expand_vars(
                    jinja_env.from_string(string).
                    render(**self.template_variables))
            except Exception as e:
                raise BatchProcessError(f'Template Render Error: {str(e).strip()}')

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
        if not name.isidentifier():
            raise BatchProcessError(
                f'Cannot define template variable "{name}" on line {self.current_line}, '
                f'invalid identifier/name token, must be a valid python variable name / identifier.')

        if name in self.template_functions:
            raise BatchProcessError(
                f'Cannot define template variable "{name}" on line {self.current_line}, '
                f'as that name is taken by a template function.')
        if name in self.reserved_template_variables:
            raise BatchProcessError(
                f'Cannot define template variable "{name}" on line {self.current_line}, '
                f'as that name is a reserved variable name.')
        if name in self.builtins:
            raise BatchProcessError(
                f'Cannot define template variable "{name}" on line {self.current_line}, '
                f'as that name is the name of a builtin function.')
        self.template_variables[name] = value

    def _jinja_user_undefine(self, name):
        if not name.isidentifier():
            raise BatchProcessError(
                f'Cannot un-define template variable "{name}" on line {self.current_line}, '
                f'invalid identifier/name token, must be a valid python variable name / identifier.')

        if name in self.template_functions:
            raise BatchProcessError(
                f'Cannot un-define template variable "{name}" on line {self.current_line}, '
                f'as that name is taken by a template function.')
        if name in self.reserved_template_variables:
            raise BatchProcessError(
                f'Cannot un-define template variable "{name}" on line {self.current_line}, '
                f'as that name is a reserved variable name.')
        if name in self.builtins:
            raise BatchProcessError(
                f'Cannot un-define template variable "{name}" on line {self.current_line}, '
                f'as that name is the name of a builtin function.')
        try:
            self.template_variables.pop(name)
        except KeyError:
            raise BatchProcessError(
                f'Cannot un-define template variable "{name}" on line {self.current_line}, '
                f'variable does not exist.')

    def _intepret_setp_value(self, value):

        interpreter = asteval.Interpreter(
            minimal=True,
            with_listcomp=True,
            with_dictcomp=True,
            with_setcomp=True,
            symtable=self.template_variables.copy())

        if 'print' in interpreter.symtable:
            del interpreter.symtable['print']

        interpreter.symtable.update(self.builtins)
        interpreter.symtable.update(self.template_functions)

        try:
            return interpreter.eval(value,
                                    show_errors=False,
                                    raise_errors=True)
        except Exception as e:
            raise BatchProcessError(f'\\setp eval error: {e}')

    def _directive_handlers(self, line):
        if line.startswith('\\setp'):
            directive_args = line.split(' ', 2)
            if len(directive_args) == 3:
                self._jinja_user_define(
                    directive_args[1].strip(),
                    self._intepret_setp_value(
                        self.render_template(directive_args[2].strip())))
                return True
            else:
                raise BatchProcessError(
                    f'\\setp directive received less than 2 arguments, '
                    f'syntax is: \\setp name value')
        elif line.startswith('\\sete'):
            directive_args = line.split(' ', 2)
            if len(directive_args) == 3:
                try:
                    self._jinja_user_define(
                        directive_args[1].strip(),
                        _textprocessing.shell_parse(
                            self.render_template(directive_args[2].strip())))
                except _textprocessing.ShellParseSyntaxError as e:
                    raise BatchProcessError(e)
                return True
            else:
                raise BatchProcessError(
                    f'\\sete directive received less than 2 arguments, '
                    f'syntax is: \\sete name args...')
        elif line.startswith('\\set'):
            directive_args = line.split(' ', 2)
            if len(directive_args) == 3:
                self._jinja_user_define(
                    directive_args[1].strip(),
                    self.render_template(directive_args[2].strip()))
                return True
            else:
                raise BatchProcessError(
                    f'\\set directive received less than 2 arguments, '
                    f'syntax is: \\set name value')
        elif line.startswith('\\unset'):
            directive_args = line.split(' ', 1)
            if len(directive_args) == 2:
                self._jinja_user_undefine(directive_args[1].strip())
                return True
            else:
                raise BatchProcessError(
                    f'\\unset directive received less than 1 arguments, '
                    f'syntax is: \\unset name')
        elif line.startswith('\\print'):
            directive_args = line.split(' ', 1)
            if len(directive_args) == 2:
                _messages.log(self.render_template(directive_args[1].strip()))
                return True
            else:
                raise BatchProcessError(
                    f'\\print directive received no arguments, '
                    f'syntax is: \\print value')
        elif line.startswith('\\echo'):
            directive_args = line.split(' ', 1)
            if len(directive_args) == 2:
                try:
                    _messages.log(*_textprocessing.shell_parse(
                        self.render_template(directive_args[1].strip()),
                        expand_vars=False))
                except _textprocessing.ShellParseSyntaxError as e:
                    raise BatchProcessError(e)
                return True
            else:
                raise BatchProcessError(
                    f'\\echo directive received no arguments, '
                    f'syntax is: \\echo args...')
        if line.startswith('{'):
            try:
                self._running_template_continuation = True
                self.run_file(self.render_template(line, stream=True))
            finally:
                self._running_template_continuation = False
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
                    return_code = impl(
                        _textprocessing.shell_parse(
                            self.render_template(directive_args[0].strip()),
                            expand_vars=False))
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

    def _lex_and_run_invocation(self, invocation_string):
        raw_templated_string = self.render_template(invocation_string)

        try:
            shell_lexed = _textprocessing.shell_parse(
                raw_templated_string, expand_vars=False)
        except _textprocessing.ShellParseSyntaxError as e:
            raise BatchProcessError(e)

        for arg in self.injected_args:
            shell_lexed.append(arg)

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

    def _run_file(self, stream: typing.Iterator[str]):
        continuation = ''
        template_continuation = False
        normal_continuation = False

        def run_continuation(cur_line):
            nonlocal continuation, template_continuation, normal_continuation

            if not template_continuation:
                completed_continuation = (continuation + ' ' + cur_line).strip()
            else:
                completed_continuation = (continuation + cur_line).strip()

            template_continuation = False
            normal_continuation = False
            continuation = ''

            self._executing_text = completed_continuation

            if self._directive_handlers(completed_continuation):
                return

            self._lex_and_run_invocation(completed_continuation)

        def remove_tail_comments_unlexable(string) -> str:
            try:
                # find the start of a possible comment
                comment_start = string.index('#')
            except ValueError:
                # no comments, all good
                return string

            if comment_start == 0:
                # it starts the string, ignore
                return string

            segment = string[:comment_start]
            if segment.endswith('\\'):
                # found a comment token, but it was escaped

                next_spaces = ''.join(itertools.takewhile(lambda x: x.isspace(), string[comment_start + 1:]))
                # record all space characters after the #, they may be taken by the lexer if
                # it can make sense of what came after #

                lexed, result = remove_tail_comments(string[comment_start + 1:])
                # recursive decent to the right starting after the # (operator lol) to solve the rest of the string

                if next_spaces and lexed:
                    # the spaces would have been consumed by lexing, add them back to the
                    # left side of the string where the user intended them to be
                    result = next_spaces + result

                # return the left side minus the escape sequence + the comment # token, plus the evaluated right side
                return segment.removesuffix('\\') + '#' + result

            # unescaped comment start, return the segment to the left
            return segment

        def remove_tail_comments(string) -> tuple[bool, str]:
            # this is a more difficult problem than I imagined.
            try:
                parts = _textprocessing.tokenized_split(string, '#',
                                                        escapable_separator=True,
                                                        allow_unterminated_strings=True)
                # attempt to split off the comment

                if not parts:
                    # empty case, effectively un-lexed
                    return False, string

                new_value = parts[0]

                if not new_value.strip():
                    # do not remove if the comment is all that exists on the line
                    # that is handled elsewhere
                    return False, string

                # the left side was lexed and stripped of leading and trailing whitespace
                return True, new_value
            except _textprocessing.TokenizedSplitSyntaxError:
                # could not lex this because of a syntax error, since unterminated
                # strings are understandable by the lexer given our options, this
                # is an uncommon if not near impossible occurrence
                return False, remove_tail_comments_unlexable(string)

        last_line = None

        for line_idx, line_and_next in enumerate(_files.PeekReader(stream)):
            line: str
            next_line: typing.Optional[str]
            line, next_line = line_and_next

            line_strip = remove_tail_comments(line)[1].strip()

            if not self._running_template_continuation:
                self._current_line = line_idx

            if line_strip == '' and not template_continuation:
                if continuation and last_line is not None:
                    if last_line.startswith('-') and \
                            not last_line.endswith('\\'):
                        run_continuation('')
            elif line_strip.startswith('#') and not template_continuation:
                self._look_for_version_mismatch(line_idx, line)
            elif line_strip.startswith('{') and not template_continuation and not normal_continuation:
                continuation += line
                template_continuation = True
            elif not template_continuation and (line_strip.endswith('\\') or next_line
                                                and next_line.lstrip().startswith('-')):
                continuation += ' ' + line_strip.strip().removesuffix('\\').strip()
                normal_continuation = True
            elif template_continuation:
                line_rstrip = remove_tail_comments(line)[1].rstrip()
                if line_rstrip.endswith('!END'):
                    run_continuation(line_rstrip.removesuffix('!END'))
                else:
                    continuation += line
            else:
                run_continuation(line_strip)

            last_line = line_strip

        if continuation:
            run_continuation('')

    def run_file(self, stream: typing.Iterator[str]):
        """
        Process a batch processing script from a file stream.

        Technically, from an iterator over lines of text.

        :raise BatchProcessError:

        :param stream: A filestream in text read mode
        """

        try:
            parsed, _ = _arguments.parse_known_args(
                self.injected_args,
                log_error=False
            )
        except _arguments.DgenerateUsageError as e:
            raise BatchProcessError(f'Error parsing injected arguments: {str(e).strip()}')

        directive_exceptions_last = self._directive_exceptions
        if parsed.verbose:
            _messages.push_level(_messages.DEBUG)
            self._directive_exceptions = True

        try:
            self._run_file(stream)
        except BatchProcessError as e:
            raise BatchProcessError(f'Error on line {self.current_line}: {str(e).strip()}')
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
