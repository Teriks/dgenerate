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

import dgenerate
import dgenerate.diffusionloop as _diffusionloop
import dgenerate.invoker as _invoker
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types


class BatchProcessError(Exception):
    """
    Thrown on errors encountered within a batch processing script
    including non-zero return codes from the invoker.
    """
    pass


class BatchProcessor:
    """
    Implements dgenerates batch processing scripts in a generified manner.
    """

    invoker: typing.Callable[[list], int]
    name: _types.Name
    version: _types.Version
    template_variable_generator: typing.Callable[[], dict]
    template_variables: typing.Dict[str, typing.Any]
    template_functions: typing.Dict[str, typing.Callable[[typing.Any], typing.Any]]
    directives: typing.Dict[str, typing.Callable[[list], None]]
    injected_args: typing.Sequence[str]

    def __init__(self,
                 invoker: typing.Callable[[list], int],
                 name: _types.Name,
                 version: typing.Union[_types.Version, str],
                 template_variable_generator: typing.Optional[typing.Callable[[], dict]] = None,
                 template_variables: typing.Optional[typing.Dict[str, typing.Any]] = None,
                 template_functions: typing.Optional[
                     typing.Dict[str, typing.Callable[[typing.Any], typing.Any]]] = None,
                 directives: typing.Optional[typing.Dict[str, typing.Callable[[list], None]]] = None,
                 injected_args: typing.Optional[typing.Sequence[str]] = None):
        """
        :param invoker: A function for invoking lines recognized as shell commands, should return a return code.
        :param template_variable_generator: A function that generates template variables for templating after an
            invocation, should return a dictionary.
        :param name: The name of this batch processor, currently used in the version check directive and messages
        :param version: Version for version check hash bang directive.
        :param template_variables: Live template variables, the initial environment, this dictionary will be
            modified during runtime.
        :param template_functions: Functions available to Jinja2
        :param directives: batch processing directive handlers, for: *\\\\directives*. This is a dictionary
            of names to functions which accept a single parameter, a list of directive arguments.
        :param injected_args: Arguments to be injected at the end of user specified arguments for every shell invocation
        """

        self.invoker = invoker
        self.name = name

        self.template_variable_generator = \
            template_variable_generator if \
                template_variable_generator else lambda x: dict()

        self.template_variables = template_variables if template_variables else dict()
        self.template_functions = template_functions if template_functions else dict()

        self.directives = directives if directives else dict()

        self.injected_args = injected_args if injected_args else []

        self._current_line = 0

        if isinstance(version, str):
            ver_parts = version.split('.')
            if len(ver_parts) != 3:
                raise ValueError(
                    f'version expected to be a version string in the format major.minor.patch. recieved: "{version}"')
            self.version: typing.Tuple[int, int, int] = \
                (int(ver_parts[0]), int(ver_parts[1]), int(ver_parts[2]))
        else:
            self.version: typing.Tuple[int, int, int] = tuple(version)
            if len(self.version) != 3:
                raise ValueError(
                    f'version tuple expected to contain three components: (major, minor, patch). recieved: {self.version}')

        self.expand_vars = os.path.expandvars
        self._jinja_env = jinja2.Environment()

        for name, func in self.template_functions.items():
            self._jinja_env.globals[name] = func
            self._jinja_env.filters[name] = func

    @property
    def current_line(self) -> int:
        """
        The current line in the file being processed.
        """
        return self._current_line

    def _render_template(self, input_string: str):
        try:
            return self.expand_vars(
                self._jinja_env.from_string(input_string).
                render(**self.template_variables))
        except Exception as e:
            raise BatchProcessError(e)

    def _look_for_version_mismatch(self, line_idx, line):
        versioning = re.match(r'#!\s+' + self.name + r'\s+([0-9]+\.[0-9]+\.[0-9]+)', line)
        if versioning:
            config_file_version = versioning.group(1)
            config_file_version = [int(p) for p in config_file_version.split('.')]

            cur_major_version = self.version[0]
            config_major_version = config_file_version[0]
            cur_minor_version = self.version[1]
            config_minor_version = config_file_version[1]

            if cur_major_version != config_major_version:
                _messages.log(
                    f'Failed version check (major version missmatch) on line {line_idx}, '
                    f'running an incompatible version of {self.name}! You are running version {self.version} '
                    f'and the config file specifies the required version: {config_file_version}'
                    , underline=True, level=_messages.WARNING)
            elif cur_minor_version < config_minor_version:
                _messages.log(
                    f'Failed version check (current minor version less than requested) '
                    f'on line {line_idx}, running an incompatible version of {self.name}! '
                    f'You are running version {self.version} and the config file specifies '
                    f'the required version: {config_file_version}'
                    , underline=True, level=_messages.WARNING)

    def _jinja_user_define(self, name, value):
        if name in self.template_functions:
            raise BatchProcessError(
                f'Cannot define template variable "{name}", reserved variable name.')
        self._jinja_env.globals[name] = value

    def _directive_handlers(self, line_idx, line):
        if line.startswith('\\set'):
            directive_args = line.split(' ', 2)
            if len(directive_args) == 3:
                self._jinja_user_define(directive_args[1].strip(), self._render_template(directive_args[2].strip()))
                return True
            else:
                raise BatchProcessError(
                    '\\set directive received less than 2 arguments, syntax is: \\set name value')
        elif line.startswith('\\print'):
            directive_args = line.split(' ', 1)
            if len(directive_args) == 2:
                _messages.log(self._render_template(directive_args[1].strip()))
                return True
            else:
                raise BatchProcessError(
                    '\\print directive received no arguments, syntax is: \\print value')
        if line.startswith('{'):
            try:
                self.run_string(self._render_template(line.replace('!END', '\n')))
            except Exception as e:
                raise BatchProcessError(
                    f'Error executing template, reason: {e}')
            return True
        elif line.startswith('\\'):
            directive_args = line.split(' ', 1)

            directive = directive_args[0].lstrip('\\')
            impl = self.directives.get(directive)
            if impl is None:
                raise BatchProcessError(f'Unknown directive "\\{directive}" on line {line_idx}.')

            impl(directive_args[1:])
            return True
        return False

    def _lex_and_run_invocation(self, line_idx, invocation_string):
        templated_cmd = self._render_template(invocation_string)

        injected_args = list(self.injected_args)

        shell_lexed = shlex.split(templated_cmd) + injected_args

        injected_args = _textprocessing.quote_spaces(injected_args)

        if injected_args:
            templated_cmd += ' ' + ' '.join(injected_args)

        header = 'Processing Arguments: '
        args_wrapped = \
            _textprocessing.wrap(
                templated_cmd,
                width=_textprocessing.long_text_wrap_width() - len(header),
                subsequent_indent=' ' * len(header))

        _messages.log(header + args_wrapped, underline=True)

        return_code = self.invoker(shell_lexed)

        if return_code != 0:
            raise BatchProcessError(
                f'Invocation error in input config file line: {line_idx}')

        self.template_variables.update(self.template_variable_generator())

    def run_file(self, stream: typing.TextIO):
        """
        Process a batch processing script from a file string

        :raise: :py:class:`.BatchProcessError`

        :param stream: A filestream in text read mode
        """
        continuation = ''

        for line_idx, line in enumerate(stream):
            self._current_line = line_idx
            line = line.strip()
            if line == '':
                continue
            if line.startswith('#'):
                self._look_for_version_mismatch(line_idx, line)
                continue
            if line.endswith('\\'):
                continuation += ' ' + line.rstrip(' \\')
            else:
                completed_continuation = (continuation + ' ' + line).lstrip()

                if self._directive_handlers(line_idx, completed_continuation):
                    continuation = ''
                    continue

                self._lex_and_run_invocation(line_idx, completed_continuation)

                continuation = ''

    def run_string(self, string: str):
        """
        Process a batch processing script from a string

        :raise: :py:class:`.BatchProcessError`

        :param string: a string containing the script
        """
        self.run_file(io.StringIO(string))


def create_config_runner(injected_args: typing.Optional[typing.Sequence[str]] = None,
                         render_loop: typing.Optional[_diffusionloop.DiffusionRenderLoop] = None,
                         version: typing.Union[_types.Version, str] = dgenerate.__version__,
                         throw: bool = False):
    """
    Create a :py:class:`.BatchProcessor` that can run dgenerate batch processing configs from a string or file.

    :param injected_args: dgenerate command line arguments in the form of list, see: shlex module, or sys.argv.
        These arguments will be injected at the end of every dgenerate invocation in the config file.
    :param render_loop: DiffusionRenderLoop instance, if None is provided one will be created.
    :param version: Config version for "#! dgenerate x.x.x" version checks, defaults to dgenerate.__version__
    :param throw: Whether to throw exceptions from :py:meth:`dgenerate.invoker.invoke_dgenerate` or handle them,
        if you set this to True exceptions will propagate out of dgenerate invocations instead of a
        :py:exc:`.BatchProcessError` being raised, a line number where the error occurred can be obtained
        using :py:attr:`.BatchProcessor.current_line`.
    :return: integer return-code, anything other than 0 is failure
    """

    if render_loop is None:
        render_loop = _diffusionloop.DiffusionRenderLoop()

    def _format_prompt(prompt):
        pos = prompt.get('positive')
        neg = prompt.get('negative')

        if pos is None:
            raise BatchProcessError('Attempt to format a prompt with no positive prompt value.')

        if pos and neg:
            return _textprocessing.quote(f"{pos}; {neg}")
        return _textprocessing.quote(pos)

    def format_prompt(prompt_or_list):
        if isinstance(prompt_or_list, dict):
            return _format_prompt(prompt_or_list)
        return ' '.join(_format_prompt(p) for p in prompt_or_list)

    template_variables = {}

    funcs = {
        'unquote': _textprocessing.unquote,
        'quote': _textprocessing.quote,
        'format_prompt': format_prompt,
        'format_size': _textprocessing.format_size,
        'quote_spaces': _textprocessing.quote_spaces,
        'last': lambda a: a[-1] if a else None
    }

    directives = {
        'clear_pipeline_caches': lambda args: _pipelinewrapper.clear_all_cache()
    }

    runner = BatchProcessor(
        invoker=lambda args: _invoker.invoke_dgenerate(args, render_loop=render_loop, throw=throw),
        template_variable_generator=lambda: render_loop.generate_template_variables(),
        name='dgenerate',
        version=version,
        template_variables=template_variables,
        template_functions=funcs,
        injected_args=injected_args if injected_args else [],
        directives=directives)

    return runner
