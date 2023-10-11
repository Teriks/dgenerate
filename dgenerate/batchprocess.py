import io
import os
import re
import shlex
import textwrap
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
    pass


class BatchProcessor:
    def __init__(self,
                 invoker: typing.Callable[[list], int],
                 template_variable_generator: typing.Callable[[], dict],
                 name: _types.Name,
                 version: typing.Union[_types.Version, str],
                 template_variables: typing.Dict[str, typing.Any],
                 template_functions: typing.Dict[str, typing.Callable[[typing.Any], typing.Any]],
                 directives: typing.Dict[str, typing.Callable[[list], None]],
                 injected_args: typing.Sequence[str]):

        self.invoker = invoker
        self.template_variable_generator = template_variable_generator
        self.template_variables = template_variables
        self.template_functions = template_functions
        self.directives = directives
        self.injected_args = injected_args
        self.name = name

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
                    f'WARNING: Failed version check (major version missmatch) on line {line_idx}, '
                    f'running an incompatible version of {self.name}! You are running version {self.version} '
                    f'and the config file specifies the required version: {config_file_version}'
                    , underline=True, level=_messages.WARNING)
            elif cur_minor_version < config_minor_version:
                _messages.log(
                    f'WARNING: Failed version check (current minor version less than requested) '
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
        args_wrapped = textwrap.fill(templated_cmd,
                                     width=_textprocessing.long_text_wrap_width() - len(header),
                                     break_long_words=False,
                                     break_on_hyphens=False,
                                     subsequent_indent=' ' * len(header))

        _messages.log(header + args_wrapped, underline=True)

        return_code = self.invoker(shell_lexed)

        if return_code != 0:
            raise BatchProcessError(
                f'Invocation error in input config file line: {line_idx}')

        self.template_variables.update(self.template_variable_generator())

    def run_file(self, stream: typing.TextIO):
        continuation = ''

        for line_idx, line in enumerate(stream):
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
        self.run_file(io.StringIO(string))


def create_config_runner(injected_args: typing.Sequence[str],
                         render_loop: typing.Optional[_diffusionloop.DiffusionRenderLoop] = None,
                         version: typing.Union[_types.Version, str] = dgenerate.__version__,
                         throw: bool = False):
    """
    Create a :py:class:`BatchProcessor` that can run dgenerate batch processing configs from a string or file.

    :param injected_args: dgenerate command line arguments in the form of list, see: shlex module, or sys.argv.
        These arguments will be injected at the end of every dgenerate invocation in the config file.
    :param render_loop: DiffusionRenderLoop instance, if None is provided one will be created.
    :param version: Config version for "#! dgenerate x.x.x" version checks, defaults to dgenerate.__version__
    :param throw: Whether to throw exceptions or handle them.
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
        'last': lambda a: a[-1] if a else None
    }

    directives = {
        'clear_model_cache': lambda args: _pipelinewrapper.clear_model_cache()
    }

    runner = BatchProcessor(
        invoker=lambda args: _invoker.invoke_dgenerate(args, render_loop=render_loop, throw=throw),
        template_variable_generator=lambda: render_loop.generate_template_variables(),
        name='dgenerate',
        version=version,
        template_variables=template_variables,
        template_functions=funcs,
        injected_args=injected_args,
        directives=directives)

    return runner
