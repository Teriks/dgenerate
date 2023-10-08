import io
import os
import re
import shlex
import textwrap

import jinja2

import dgenerate.diffusionloop as _diffusionloop
import dgenerate.invoker as _invoker
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.textprocessing as _textprocessing


class BatchProcessError(Exception):
    pass


class BatchProcessor:
    def __init__(self,
                 render_loop: _diffusionloop.DiffusionRenderLoop,
                 name: str,
                 version: str,
                 template_variables: dict,
                 template_functions: dict,
                 directives: dict,
                 injected_args: list):
        self.render_loop = render_loop
        self.template_variables = template_variables
        self.template_functions = template_functions
        self.directives = directives
        self.injected_args = injected_args
        self.name = name
        self.version = version
        self.expand_vars = os.path.expandvars
        self._jinja_env = jinja2.Environment()

        for name, func in self.template_functions.items():
            self._jinja_env.globals[name] = func
            self._jinja_env.filters[name] = func

    def render_template(self, input_string):
        return self.expand_vars(
            self._jinja_env.from_string(input_string).
            render(**self.template_variables))

    def _look_for_version_mismatch(self, line_idx, line):
        versioning = re.match(r'#!\s+' + self.name + r'\s+([0-9]+\.[0-9]+\.[0-9]+)', line)
        if versioning:
            config_file_version = versioning.group(1)
            cur_version = [int(p) for p in self.version.split('.')]
            config_file_version = [int(p) for p in config_file_version.split('.')]

            cur_major_version = cur_version[0]
            config_major_version = config_file_version[0]
            cur_minor_version = cur_version[1]
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
                self._jinja_user_define(directive_args[1].strip(), self.render_template(directive_args[2].strip()))
                return True
            else:
                raise BatchProcessError(
                    '\\set directive received less than 2 arguments, syntax is: \\set name value')
        elif line.startswith('\\print'):
            directive_args = line.split(' ', 1)
            if len(directive_args) == 2:
                _messages.log(self.render_template(directive_args[1].strip()))
                return True
            else:
                raise BatchProcessError(
                    '\\print directive received no arguments, syntax is: \\print value')
        if line.startswith('{'):
            try:
                self.process_file(io.StringIO(self.render_template(line.replace('!END', '\n'))))
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

    def _update_template_variables(self):
        def jinja_prompt(prompts):
            if not prompts:
                # Completely undefined
                return [{'positive': None, 'negative': None}]
            else:
                # inside prompt values might be None, don't want that in
                # the jinja2 template because it might be annoying
                # to work with. Also abstract the internal representation
                # of the prompt dictionary to something with friendlier
                # names
                return [{'positive': p.get('prompt', None),
                         'negative': p.get('negative_prompt', None)} for p in prompts]

        def last_or_none(ls):
            if ls:
                val = ls[-1]
                if isinstance(val, str):
                    val = _textprocessing.quote(val)
                return val
            return None

        def quote_string_lists(ls):
            if ls and isinstance(ls[0], str):
                return [_textprocessing.quote(i) for i in ls]
            return ls

        self.template_variables.update({
            'last_images': quote_string_lists(self.render_loop.written_images),
            'last_image': last_or_none(self.render_loop.written_images),
            'last_animations': quote_string_lists(self.render_loop.written_animations),
            'last_animation': last_or_none(self.render_loop.written_animations)
        })

        for k, v in self.render_loop.config.__dict__.items():

            if not (k.startswith('_') or callable(v)):
                prefix = 'last_' if not k.startswith('last_') else ''
                if k.endswith('s') or 'coords' in k:
                    t_val = v if v is not None else []
                    if 'prompt' in k:
                        self.template_variables[prefix + k] = jinja_prompt(t_val)
                        self.template_variables[prefix + k.rstrip('s')] = jinja_prompt(t_val)[-1]
                    else:
                        t_val = v if v is not None else []
                        self.template_variables[prefix + k] = quote_string_lists(t_val)
                        self.template_variables[prefix + k.replace('coords', 'coord').rstrip('s')] = last_or_none(t_val)
                else:
                    self.template_variables[prefix + k] = v if v is not None else None

    def _lex_and_run_invocation(self, line_idx, invocation_string):
        templated_cmd = self.render_template(invocation_string)

        injected_args = self.injected_args.copy()

        shell_lexed = shlex.split(templated_cmd) + injected_args

        for idx, extra_arg in enumerate(injected_args):
            if any(c.isspace() for c in extra_arg):
                injected_args[idx] = _textprocessing.quote(extra_arg)

        header = 'Processing Arguments: '
        args_wrapped = textwrap.fill(templated_cmd + ' ' + ' '.join(injected_args),
                                     width=_textprocessing.long_text_wrap_width() - len(header),
                                     break_long_words=False,
                                     break_on_hyphens=False,
                                     subsequent_indent=' ' * len(header))

        _messages.log(header + args_wrapped, underline=True)

        return_code = _invoker.invoke_dgenerate(self.render_loop, shell_lexed)

        if return_code != 0:
            raise BatchProcessError(
                f'Invocation error in input config file line: {line_idx}')

        self._update_template_variables()

    def process_file(self, stream):
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


def process_config(render_loop: _diffusionloop.DiffusionRenderLoop,
                   version_string,
                   injected_args,
                   file_stream):
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
        'format_prompt': format_prompt
    }

    directives = {
        'clear_model_cache': lambda args: _pipelinewrapper.clear_model_cache()
    }

    runner = BatchProcessor(render_loop=render_loop,
                            name='dgenerate',
                            version=version_string,
                            template_variables=template_variables,
                            template_functions=funcs,
                            injected_args=injected_args,
                            directives=directives)

    runner.process_file(stream=file_stream)
