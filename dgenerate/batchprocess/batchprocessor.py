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
import os
import re
import traceback
import typing
import warnings


import jinja2

import dgenerate.arguments as _arguments
import dgenerate.batchprocess.jinjabalancechecker as _jinjabalancechecker
import dgenerate.files as _files
import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
import dgenerate.eval as _eval


class _EnvNamespace:
    def __getattr__(self, name):
        return os.environ.get(name, '')


class BatchProcessError(Exception):
    """
    Thrown by :py:meth:`.BatchProcessor.run_file` and :py:meth:`.BatchProcessor.run_string`
    when an error in a batch processing script is encountered.
    """
    pass


class _TemplateContinuationInternalError(Exception):
    pass


InvokerType = typing.Callable[[str, collections.abc.Sequence[str]], int] | typing.Callable[
    [collections.abc.Sequence[str]], int]


class BatchProcessor:
    """
    Implements dgenerate's batch processing scripts in a generified manner.

    This is the bare-bones implementation of the shell with nothing
    implemented for you except for:

        * ``\\env``
        * ``\\set``
        * ``\\sete``
        * ``\\setp``
        * ``\\unset``
        * ``\\unset_env``
        * ``\\print``
        * ``\\echo``
        * ``\\reset_lineno``

    If you wish to create this object to run a dgenerate configuration, use
    :py:class:`dgenerate.batchprocess.ConfigRunner`
    """

    invoker: InvokerType
    """
    Invoker function, responsible for executing lines recognized as shell commands.
    
    This can be a function that just accepts a sequence of arguments, or a function 
    that accepts the raw command line as a string followed by the parsed 
    sequence of arguments.
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
    These template variables cannot be set with the ``\\set``, ``\\sete``, or ``\\setp`` 
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

    directives: dict[str, typing.Callable[[collections.abc.Sequence[str]], int]] | None
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
    A function for expanding environmental variables, defaults to :py:func:`dgenerate.textprocessing.shell_expandvars`
    """

    disable_directives: bool = False
    """
    If ``True``, disables the use of all directives, including the built-in ones.
    
    This also disable template continuations, (lines starting with "{") which are a form of directive.
    """

    def __init__(self,
                 invoker: InvokerType,
                 name: _types.Name,
                 version: _types.Version | str,
                 template_variables: dict[str, typing.Any] | None = None,
                 reserved_template_variables: set[str] | None = None,
                 template_functions: dict[str, typing.Callable[[typing.Any], typing.Any]] | None = None,
                 directives: dict[str, typing.Callable[[list], None]] | None = None,
                 builtins: dict[str, typing.Callable[[typing.Any], typing.Any]] | None = None,
                 injected_args: collections.abc.Sequence[str] | None = None,
                 disable_directives: bool = False
                 ):
        """
        :param invoker: A function for invoking lines recognized as shell commands, should return a return code,
            This can be a function that just accepts a sequence of arguments, or a function that accepts the
            raw command line as a string followed by the parsed sequence of arguments.
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
            A safe default collection of functions is used if this is not specified. Builtins may
            be overridden by functions defined in ``template_functions``
        :param injected_args: Arguments to be injected at the end of user specified arguments for every shell invocation.
            If ``-v/--verbose`` is present in ``injected_args`` debugging output will be enabled globally while the config
            runs, and not just for invocations. Passing ``-v/--verbose`` also enables printing stack traces for all
            unhandled directive exceptions to ``stderr``.
        :param disable_directives: If ``True``, disables the use of all directives, including the built-in ones.
            This also disable template continuations, (lines starting with "{") which are a form of directive.
        """

        self._template_functions = None
        self.invoker = invoker
        self.name = name

        self.template_variables = template_variables if template_variables else dict()
        self.reserved_template_variables = reserved_template_variables if reserved_template_variables else set()

        self.template_functions = template_functions if template_functions else dict()

        self.directives = directives if directives else dict()

        self.disable_directives = disable_directives

        self._directive_error_traces = False

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

        self.expand_vars = _textprocessing.shell_expandvars

        if builtins is None:
            self.builtins = self.default_builtins()
        else:
            self.builtins = builtins

    @staticmethod
    def default_builtins() -> dict[str, typing.Callable[[typing.Any], typing.Any]]:
        """Return the default builtins available as template functions."""

        return _eval.safe_builtins()

    @property
    def directives_builtins_help(self) -> dict[str, str]:
        """
        Returns a dictionary of help strings for directives that are
        built into the :py:class:`BatchProcessor` base class.
        """

        return {
            'env': 'Sets environmental variables, example: \\env VAR_1=value VAR_2=value. '
                   'Using without arguments will print the current environment. Indirect '
                   'expansion is allowed, for example: \\env %VAR_NAME%=value, or '
                   '\\env {{ var_name }}=value',
            'set': 'Sets a template variable, accepts two arguments, the variable name and the value. '
                   'Attempting to set a reserved template variable such as those pre-defined by dgenerate '
                   'will result in an error. The first argument is the variable name, which may be an identifier '
                   'or a template / environmental variable that expands to an identifier. The second argument is '
                   'accepted as a raw value, it is not shell parsed in any way, only stripped of leading and trailing '
                   'whitespace after templating and environmental variable expansion.',
            'sete': 'Sets a template variable to an array of shell arguments using shell parsing and expansion. '
                    'The variable name may be an identifier or an identifier expanded from a '
                    'template / environmental variable. For example, this could be utilized for '
                    'convenient shell globbing: \\sete my_files my_directory1/* my_directory2/*',
            'setp': 'Sets a template variable to a (safely) evaluated Python expression, accepts two arguments, '
                    'the variable name and the value. Attempting to set a reserved template variable such '
                    'as those pre-defined by dgenerate will result in an error. The first argument is the '
                    'variable name, which may be an identifier or a template / environmental variable '
                    'that expands to an identifier. The second argument is a python expression that undergoes '
                    'template and environmental variable expansion. Template variables can be referred to '
                    'by name within a definition, EG: \\setp my_list [1, 2, my_var, 4]. Template functions '
                    'are also available, EG: \\setp working_dir cwd(). Python unary and binary expression '
                    'operators, python list slicing, and comprehensions are supported. '
                    'This functionality is provided by the asteval package.',
            'unset': 'Undefines a template variable previously set with \\set or \\setp, accepts one argument, '
                     'the variable name. The variable name may be an identifier or a template / environmental '
                     'variable that expands to an identifier. Attempting to unset a reserved variable such as those '
                     'pre-defined by dgenerate will result in an error.',
            'unset_env': 'Undefines environmental variables previously set with \\env, accepts multiple arguments, '
                         'each argument is an environmental variable name, or a template / environmental variable '
                         'that expands to the name of an environmental variable. Attempting to unset a variable '
                         'that does not exist is a no-op.',
            'print': 'Prints all content to the right to stdout, no shell parsing of the argument occurs.',
            'echo': 'Echo shell arguments with shell parsing and expansion.',
            'reset_lineno': 'Reset the interpreter line number record to 0, this affects error messages and can '
                            'be issued by the user, or automatically by a connected terminal.'
        }

    @property
    def current_line(self) -> int:
        """
        The current line number in the file being processed.
        """
        return self._current_line

    @property
    def running_template_continuation(self) -> bool:
        """
        Is code that exists inside a template continuation being processed?
        :return: ``True`` or ``False``
        """
        return self._running_template_continuation

    @property
    def template_continuation_start_line(self) -> int:
        """
        Start line of the template continuation being processed.

        Value is only valid if :py:attr:`BatchProcessor.running_template_continuation` is ``True``.

        :return: line number
        """
        if not self.running_template_continuation:
            return -1

        return self._template_continuation_start_line

    @property
    def template_continuation_end_line(self) -> int:
        """
        End line of the template continuation being processed.

        Value is only valid if :py:attr:`BatchProcessor.running_template_continuation` is ``True``.

        :return: line number
        """
        if not self.running_template_continuation:
            return -1

        return self._template_continuation_end_line

    @property
    def executing_text(self) -> None | str:
        """
        The text / command line that is currently
        being executed, or that was last executed.
        """
        return self._executing_text

    def render_template(self, string: str, stream: bool = False) -> str | typing.Iterator[str]:
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

        jinja_env.globals['env'] = _EnvNamespace()

        if stream:
            def stream_generator():
                buffer = ''
                try:
                    for piece in jinja_env.from_string(string). \
                            stream(**self.template_variables):
                        buffer += piece
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            yield line + '\n'
                    if buffer:
                        yield buffer
                except Exception as e:
                    raise BatchProcessError(f'Template Render Error: {str(e).strip()}') from e

            return stream_generator()
        else:
            try:
                return jinja_env.from_string(string).render(**self.template_variables)
            except Exception as e:
                raise BatchProcessError(f'Template Render Error: {str(e).strip()}') from e

    def _look_for_version_mismatch(self, line_idx, line):
        versioning = re.match(r'#!\s+' + re.escape(self.name) + r'\s+([0-9]+\.[0-9]+\.[0-9]+)', line)
        if versioning:
            config_file_version = versioning.group(1)
            config_file_version_parts = [int(part) for part in config_file_version.split('.')]

            cur_major_version = self.version[0]
            config_major_version = config_file_version_parts[0]
            cur_minor_version = self.version[1]
            config_minor_version = config_file_version_parts[1]

            version_str = '.'.join(str(i) for i in self.version)

            if cur_major_version != config_major_version:
                _messages.warning(
                    f'Failed version check (major version mismatch) on line {line_idx}, '
                    f'running an incompatible version of {self.name}! You are running version {version_str} '
                    f'and the config file specifies the required version: {config_file_version}',
                    underline=True
                )
            elif cur_minor_version < config_minor_version:
                _messages.warning(
                    f'Failed version check (current minor version less than requested) '
                    f'on line {line_idx}, running an incompatible version of {self.name}! '
                    f'You are running version {version_str} and the config file specifies '
                    f'the required version: {config_file_version}',
                    underline=True
                )

    def user_define_check(self, name: str):
        """
        Check if a template variable can be defined by the user, raise if not.

        :raises BatchProcessError: if the specified variable name cannot be defined
            by the user due to not being a valid identifier string, being the name of
            a template function, being the name of a reserved template variable, or
            being the name of a builtin function.

        :param name: Variable name
        """
        if not name.isidentifier():
            raise BatchProcessError(
                f'Cannot define template variable "{name}" on line {self.current_line}, '
                f'invalid identifier/name token, must be a valid python variable name / identifier.')

        if name == 'env':
            raise BatchProcessError(
                f'Cannot define template variable "{name}" on line {self.current_line}, '
                f'as that name refers to the special namespace used to access '
                f'environmental variables.')
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

    def user_define(self, name: str, value: typing.Any):
        """
        Define a template variable as if you were the user.

        :raises BatchProcessError: if the specified variable name cannot be defined
            by the user due to not being a valid identifier string, being the name of
            a template function, being the name of a reserved template variable, or
            being the name of a builtin function.

        :param name: Variable name
        :param value: Assigned value
        """
        self.user_define_check(name)
        self.template_variables[name] = value

    def user_undefine_check(self, name: str):
        """
        Check if a template variable can be undefined by the user, raise if not.

        :raises BatchProcessError: if the specified variable name cannot be undefined
            by the user due to not being a valid identifier string, being the name of
            a template function, being the name of a reserved template variable,
            being the name of a builtin function, or a non existing
            template variable.

        :param name: Variable name
        """
        if not name.isidentifier():
            raise BatchProcessError(
                f'Cannot un-define template variable "{name}" on line {self.current_line}, '
                f'invalid identifier/name token, must be a valid python variable name / identifier.')

        if name == 'env':
            raise BatchProcessError(
                f'Cannot un-define template variable "{name}" on line {self.current_line}, '
                f'as that name refers to the special namespace used to access '
                f'environmental variables.')
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

        if name not in self.template_variables:
            raise BatchProcessError(
                f'Cannot un-define template variable "{name}" on line {self.current_line}, '
                f'variable does not exist.')

    def user_undefine(self, name: str):
        """
        Undefine a template variable as if you were the user.

        :raises BatchProcessError: if the specified variable name cannot be undefined
            by the user due to not being a valid identifier string, being the name of
            a template function, being the name of a reserved template variable,
            being the name of a builtin function, or a non existing
            template variable.

        :param name: Variable name
        """
        self.user_undefine_check(name)
        self.template_variables.pop(name)

    def user_set(self, name: str, value: str):
        """
        Set a template variable as if you were using the \\set directive.
        
        This applies template expansion and environmental variable expansion to both
        the name and value, then sets the template variable.

        :raises BatchProcessError: if the specified variable name cannot be defined
            by the user due to validation errors.

        :param name: Variable name (can contain template expressions)
        :param value: Variable value (can contain template expressions and env vars)
        """
        processed_name = self.expand_vars(self.render_template(name))
        processed_value = self.expand_vars(self.render_template(value))
        self.user_define(processed_name, processed_value)

    def user_setp(self, name: str, expression: str):
        """
        Set a template variable to the result of evaluating a Python expression
        as if you were using the \\setp directive.
        
        This applies template expansion and environmental variable expansion to the name,
        then evaluates the expression as Python and sets the template variable to the result.

        :raises BatchProcessError: if the specified variable name cannot be defined
            by the user due to validation errors, or if the expression cannot be evaluated.

        :param name: Variable name (can contain template expressions)
        :param expression: Python expression to evaluate (can contain template expressions and env vars)
        """
        processed_name = self.expand_vars(self.render_template(name))
        processed_expression = self.expand_vars(self.render_template(expression))
        interpreted_value = self._intepret_setp_value(processed_expression)
        self.user_define(processed_name, interpreted_value)

    def _intepret_setp_value(self, value):
        interpreter = _eval.standard_interpreter(
            symtable=self.template_variables.copy()
        )

        interpreter.symtable.update(self.builtins)
        interpreter.symtable.update(self.template_functions)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=SyntaxWarning,
                message=r".*invalid escape sequence '\\%'.*")
            warnings.filterwarnings(
                "ignore",
                category=SyntaxWarning,
                message=r".*invalid escape sequence '\\\$'.*")

            try:
                return interpreter.eval(value,
                                        show_errors=False,
                                        raise_errors=True)
            except Exception as e:
                raise BatchProcessError(
                    f'\\setp eval error: {(chr(10) + "  " * 4).join(str(e).strip().split(chr(10)))}') from e

    def _set_split(self, directive_args, line):
        name_part = directive_args[1]
        if not name_part.startswith('{{'):
            return directive_args[1], directive_args[2]

        # Handle the case where the name_part starts with '{{'
        without_directive = line.split(None, 1)[1]
        t_depth = 0
        var_name = ''
        value_part = ''
        idx = 0
        var_mode = True

        while idx < len(without_directive):
            char = without_directive[idx]
            if var_mode:
                var_name += char
                if char == '{':
                    t_depth += 1
                elif char == '}':
                    t_depth -= 1
                if t_depth == 0 and var_name.endswith('}}'):
                    var_mode = False
            else:
                value_part += char
            idx += 1

        return var_name.strip(), value_part.strip()

    def _directive_handlers(self, line: str) -> bool:
        """Handle shell directives."""
        if self.disable_directives:
            return False

        if line.startswith('\\env'):
            return self._handle_env_directive(line)
        elif line.startswith('\\unset_env'):
            return self._handle_unset_env_directive(line)
        elif line.startswith('\\setp'):
            return self._handle_setp_directive(line)
        elif line.startswith('\\sete'):
            return self._handle_sete_directive(line)
        elif line.startswith('\\set'):
            return self._handle_set_directive(line)
        elif line.startswith('\\unset'):
            return self._handle_unset_directive(line)
        elif line.startswith('\\print'):
            return self._handle_print_directive(line)
        elif line.startswith('\\echo'):
            return self._handle_echo_directive(line)
        elif line.startswith('{'):
            return self._handle_template_continuation(line)
        elif line.startswith('\\'):
            return self._handle_custom_directive(line)
        return False

    def _handle_env_directive(self, line: str) -> bool:
        """Handle the \\env directive."""
        directive_args = line.split(None, 1)
        if len(directive_args) == 2:
            opts = self.render_template(directive_args[1]).split()
            for opt in opts:
                assignment = opt.split('=', 1)
                value = self.expand_vars(self.render_template(assignment[1])) if len(assignment) > 1 else ''
                try:
                    os.environ[assignment[0]] = value
                except ValueError as e:
                    if not assignment[0].strip():
                        raise BatchProcessError(
                            f'Environmental variable name expanded to nothing!') from e
                    raise BatchProcessError(
                        f'Illegal environmental variable name value: {assignment[0]}') from e
        elif line != '\\env':
            raise BatchProcessError(f'Unknown directive "{line}".')
        else:
            for key, value in os.environ.items():
                _messages.log(f'\\env "{key}={value}"')
        return True

    def _handle_unset_env_directive(self, line: str) -> bool:
        """Handle the \\unset_env directive."""
        directive_args = line.split(None, 1)
        if len(directive_args) == 2:
            opts = _textprocessing.shell_parse(
                self.render_template(directive_args[1].strip()),
                expand_home=False,
                expand_glob=False,
                expand_vars_func=self.expand_vars
            )
            for opt in opts:
                os.environ.pop(opt, None)
        elif line != '\\unset_env':
            raise BatchProcessError(f'Unknown directive "{line}".')
        else:
            raise BatchProcessError('\\unset_env was provided no arguments.')
        return True

    def _handle_setp_directive(self, line: str) -> bool:
        """Handle the \\setp directive."""
        directive_args = line.split(None, 2)
        if len(directive_args) == 3:
            var, value = self._set_split(directive_args, line)
            self.user_define(
                self.expand_vars(self.render_template(var)),
                self._intepret_setp_value(
                    self.expand_vars(self.render_template(value))))
            return True
        else:
            raise BatchProcessError(
                f'\\setp directive received less than 2 arguments, '
                f'syntax is: \\setp name value')

    def _handle_sete_directive(self, line: str) -> bool:
        """Handle the \\sete directive."""
        directive_args = line.split(None, 2)
        if len(directive_args) == 3:
            try:
                var, value = self._set_split(directive_args, line)
                self.user_define(
                    self.expand_vars(self.render_template(var)),
                    _textprocessing.shell_parse(
                        self.render_template(value),
                        expand_vars_func=self.expand_vars
                    )
                )
            except _textprocessing.ShellParseSyntaxError as e:
                raise BatchProcessError(e) from e
            return True
        else:
            raise BatchProcessError(
                f'\\sete directive received less than 2 arguments, '
                f'syntax is: \\sete name args...')

    def _handle_set_directive(self, line: str) -> bool:
        """Handle the \\set directive."""
        directive_args = line.split(None, 2)
        if len(directive_args) == 3:
            var, value = self._set_split(directive_args, line)
            self.user_define(
                self.expand_vars(self.render_template(var)),
                self.expand_vars(self.render_template(value))
            )
            return True
        else:
            raise BatchProcessError(
                f'\\set directive received less than 2 arguments, '
                f'syntax is: \\set name value')

    def _handle_unset_directive(self, line: str) -> bool:
        """Handle the \\unset directive."""
        directive_args = line.split(None, 1)
        if len(directive_args) == 2:
            self.user_undefine(
                self.expand_vars(self.render_template(directive_args[1])))
            return True
        elif line != '\\unset':
            raise BatchProcessError(f'Unknown directive "{line}".')
        else:
            raise BatchProcessError(
                f'\\unset directive received less than 1 arguments, '
                f'syntax is: \\unset name')

    def _handle_print_directive(self, line: str) -> bool:
        """Handle the \\print directive."""
        directive_args = line.split(None, 1)
        if len(directive_args) == 2:
            _messages.log(self.expand_vars(self.render_template(directive_args[1])))
        elif line != '\\print':
            raise BatchProcessError(f'Unknown directive "{line}".')
        else:
            _messages.log()
        return True

    def _handle_echo_directive(self, line: str) -> bool:
        """Handle the \\echo directive."""
        directive_args = line.split(None, 1)
        if len(directive_args) == 2:
            try:
                _messages.log(
                    *_textprocessing.shell_parse(
                        self.render_template(directive_args[1]),
                        expand_vars_func=self.expand_vars
                    )
                )
            except _textprocessing.ShellParseSyntaxError as e:
                raise BatchProcessError(e) from e
        elif line != '\\echo':
            raise BatchProcessError(f'Unknown directive "{line}".')
        else:
            _messages.log()
        return True

    def _handle_template_continuation(self, line: str) -> bool:
        """Handle template continuation directive."""
        try:
            self._running_template_continuation = True
            self._current_line = -1
            self.run_file(self.render_template(line, stream=True))
        except BatchProcessError as e:
            sub_err_msg = str(e).split(":", 1)[-1].strip()
            raise _TemplateContinuationInternalError(
                f'Error inside template continuation '
                f'from line {self._template_continuation_start_line} to line '
                f'{self._template_continuation_end_line}:\n{" " * 4}Error on line '
                f'{self._template_continuation_start_line + self.current_line}:'
                f'\n{" " * 8}{sub_err_msg}') from e
        finally:
            self._running_template_continuation = False
            self._current_line = self._template_continuation_end_line
        return True

    def _handle_custom_directive(self, line: str) -> bool:
        """Handle custom directives defined by the user."""
        directive_args = line.split(None, 1)
        directive = directive_args[0].lstrip('\\')
        impl = self.directives.get(directive)
        if impl is None:
            raise BatchProcessError(f'Unknown directive "\\{directive}".')
        directive_args = directive_args[1:]
        try:
            if directive_args:
                return_code = impl(
                    _textprocessing.shell_parse(
                        self.render_template(directive_args[0]),
                        expand_vars_func=self.expand_vars
                    )
                )
            else:
                return_code = impl([])
            if return_code != 0:
                raise BatchProcessError(
                    f'Directive "{directive}" error return code: {return_code}')
        except Exception as e:
            if self._directive_error_traces:
                _messages.error(
                    _textprocessing.underline(f'\\{directive} error trace:') + "\n" +
                    _textprocessing.underline('\n'.join(traceback.format_exception(e)))
                )
            raise BatchProcessError(e) from e
        return True

    def _lex_and_run_invocation(self, invocation_string: str):
        """Run a line of shell code"""
        raw_templated_string = self.render_template(invocation_string)

        try:
            shell_lexed = _textprocessing.shell_parse(
                raw_templated_string,
                expand_vars_func=self.expand_vars
            )
        except _textprocessing.ShellParseSyntaxError as e:
            raise BatchProcessError(e) from e

        for arg in self.injected_args:
            arg = arg.strip()
            if arg:
                shell_lexed.append(
                    _textprocessing.shell_parse(
                        arg,
                        expand_glob=False,
                        expand_vars=False,
                        expand_home=False
                    )[0]
                )

        raw_injected_args = ' '.join(str(a) for a in self.injected_args)

        if raw_injected_args:
            cmd_info = raw_templated_string + ' ' + raw_injected_args
        else:
            cmd_info = raw_templated_string

        if len(inspect.signature(self.invoker).parameters) == 2:
            return_code = self.invoker(cmd_info, shell_lexed)
        else:
            return_code = self.invoker(shell_lexed)

        if return_code != 0:
            raise BatchProcessError(
                f'Invocation error return code: {return_code}')

    def _run_file(self, stream: typing.Iterator[str]):
        # only me and god understand this.

        continuation = ''
        continuation_char = '\\'
        template_continuation = False
        normal_continuation = False

        last_line = None

        jinja_lexer = None

        def run_continuation(cur_line):
            nonlocal continuation, template_continuation, normal_continuation, last_line

            # local
            l_last_line_starts_with_dash = last_line and last_line.startswith('-')
            l_last_line_ends_with_cont = last_line and last_line.endswith(continuation_char)

            if not template_continuation:
                if cur_line.startswith('-') or (l_last_line_starts_with_dash and not l_last_line_ends_with_cont):
                    completed_continuation = (continuation + ' ' + cur_line).strip()
                else:
                    completed_continuation = (continuation + cur_line).strip()
            else:
                completed_continuation = (continuation + cur_line).strip()

            template_continuation = False
            normal_continuation = False
            continuation = ''

            self._executing_text = completed_continuation

            if self._directive_handlers(completed_continuation):
                return

            self._lex_and_run_invocation(completed_continuation)

        for line_idx, line_and_next in enumerate(_files.PeekReader(stream)):
            line: str
            next_line: str | None
            line, next_line = line_and_next

            line_rstrip = _textprocessing.remove_tail_comments(line)[1].rstrip()
            line_strip = line_rstrip.lstrip()

            if line_strip == '\\reset_lineno':
                self._current_line = -1
                continue

            line_strip_end_with_cont = line_strip.endswith(continuation_char)
            next_line_starts_with_dash = next_line and next_line.lstrip().startswith('-')
            last_line_starts_with_dash = last_line and last_line.startswith('-')
            last_line_ends_with_cont = last_line and last_line.endswith(continuation_char)
            line_strip_starts_with_dash = line_strip.startswith('-')

            def start_or_append_normal_continuation():
                nonlocal continuation, normal_continuation

                normal_continuation = True
                if (line_strip_end_with_cont and not line_strip_starts_with_dash) or \
                        (next_line_starts_with_dash and not line_strip_starts_with_dash):
                    continuation += line_strip.removesuffix(continuation_char).lstrip().rstrip("\r\n")
                else:
                    continuation += ' ' + line_strip.removesuffix(continuation_char).lstrip().rstrip("\r\n")

            self._current_line += 1

            if line_strip == '' and not template_continuation:
                if continuation and last_line is not None:
                    if last_line_starts_with_dash and not last_line_ends_with_cont:
                        run_continuation('')
            elif line_strip.startswith('#') and not template_continuation:

                self._look_for_version_mismatch(line_idx, line)

            elif line_strip.startswith('{') and not template_continuation and not normal_continuation:

                jinja_lexer = _jinjabalancechecker.JinjaBalanceChecker(
                    jinja2.Environment()
                )

                self._template_continuation_start_line = self._current_line

                try:
                    jinja_lexer.put_source(line_strip)
                except jinja2.TemplateSyntaxError as e:
                    raise BatchProcessError(f'Template Syntax Error: {e.message}') from e

                if jinja_lexer.is_balanced() or next_line is None:
                    self._template_continuation_end_line = self._current_line

                    if line_strip.startswith('{{') and (line_strip_end_with_cont or next_line_starts_with_dash):
                        template_continuation = False
                        start_or_append_normal_continuation()
                    else:
                        run_continuation(line_rstrip)
                else:
                    continuation += line
                    template_continuation = True

            elif not template_continuation and (line_strip_end_with_cont or next_line_starts_with_dash):
                start_or_append_normal_continuation()
            elif template_continuation:

                try:
                    jinja_lexer.put_source(line_strip)
                except jinja2.TemplateSyntaxError as e:
                    raise BatchProcessError(f'Template Syntax Error: {e.message}') from e

                if jinja_lexer.is_balanced() or next_line is None:
                    self._template_continuation_end_line = self._current_line

                    if continuation.startswith('{{') and (line_strip_end_with_cont or next_line_starts_with_dash):
                        template_continuation = False
                        start_or_append_normal_continuation()
                    else:
                        run_continuation(line_rstrip)
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
            raise BatchProcessError(f'Error parsing injected arguments: {str(e).strip()}') from e

        directive_error_traces = self._directive_error_traces

        if parsed.verbose:
            _messages.push_level(_messages.DEBUG)
            self._directive_error_traces = True

        try:
            self._run_file(stream)
        except _TemplateContinuationInternalError as e:
            raise BatchProcessError(str(e).strip()) from e
        except BatchProcessError as e:
            raise BatchProcessError(
                f'Error on line {self.current_line}:'
                f'\n{" " * 4}{str(e).strip()}') from e
        finally:
            if parsed.verbose:
                _messages.pop_level()
            self._directive_error_traces = directive_error_traces

    def run_string(self, string: str):
        """
        Process a batch processing script from a string

        :raise BatchProcessError:

        :param string: a string containing the script
        """
        self.run_file(io.StringIO(string))


__all__ = _types.module_all()
