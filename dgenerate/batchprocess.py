import io
import os
import re
import shlex
import textwrap

import jinja2

from dgenerate.pipelinewrappers import clear_model_cache
from . import messages
from .textprocessing import long_text_wrap_width, quote, unquote


class BatchProcessSyntaxException(Exception):
    pass


def process_config(file_stream, injected_args, version_string, invocation_runner):
    template_args = {
        'last_prompt': {'positive': '', 'negative': ''},
        'last_prompts': [{'positive': '', 'negative': ''}],
        'last_sdxl_second_prompt': {'positive': '', 'negative': ''},
        'last_sdxl_second_prompts': [{'positive': '', 'negative': ''}],
        'last_sdxl_refiner_prompt': {'positive': '', 'negative': ''},
        'last_sdxl_refiner_prompts': [{'positive': '', 'negative': ''}],
        'last_sdxl_refiner_second_prompt': {'positive': '', 'negative': ''},
        'last_sdxl_refiner_second_prompts': [{'positive': '', 'negative': ''}],
        'last_image': '',
        'last_images': [],
        'last_animation': '',
        'last_animations': []
    }

    def _format_prompt(prompt):
        pos = prompt.get('positive')
        neg = prompt.get('negative')
        if pos and neg:
            return quote(f"{pos}; {neg}")
        return quote(pos)

    def format_prompt(prompt_or_list):
        if isinstance(prompt_or_list, dict):
            return _format_prompt(prompt_or_list)
        return ' '.join(_format_prompt(p) for p in prompt_or_list)

    jinja_env = jinja2.Environment()

    jinja_reserved = set()

    def add_jinja_func(name, func):
        jinja_env.globals[name] = func
        jinja_env.filters[name] = func
        jinja_reserved.add(name)

    def jinja_user_define(name, value):
        if name in jinja_reserved:
            raise BatchProcessSyntaxException(
                f'Cannot define template variable "{name}", reserved variable name.')
        jinja_env.globals[name] = value

    add_jinja_func('unquote', unquote)
    add_jinja_func('quote', quote)
    add_jinja_func('format_prompt', format_prompt)

    def look_for_version_mismatch(line_idx, line):
        versioning = re.match(r'#!\s+dgenerate\s+([0-9]+\.[0-9]+\.[0-9]+)', line)
        if versioning:
            config_file_version = versioning.group(1)
            cur_version = [int(p) for p in version_string.split('.')]
            config_file_version = [int(p) for p in config_file_version.split('.')]

            cur_major_version = cur_version[0]
            config_major_version = config_file_version[0]
            cur_minor_version = cur_version[1]
            config_minor_version = config_file_version[1]

            if cur_major_version != config_major_version:
                messages.log(
                    f'WARNING: Failed version check (major version missmatch) on line {line_idx}, '
                    f'running an incompatible version of dgenerate! You are running version {version_string} '
                    f'and the config file specifies the required version: {config_file_version}'
                    , underline=True, level=messages.WARNING)
            elif cur_minor_version < config_minor_version:
                messages.log(
                    f'WARNING: Failed version check (current minor version less than requested) '
                    f'on line {line_idx}, running an incompatible version of dgenerate! '
                    f'You are running version {version_string} and the config file specifies '
                    f'the required version: {config_file_version}'
                    , underline=True, level=messages.WARNING)

    def directive_handlers(line_idx, line):
        # This is post continuation handling

        if line.startswith('\\set'):
            directive_args = line.split(' ', 2)
            if len(directive_args) == 3:
                jinja_user_define(directive_args[1].strip(),
                                  jinja_env.from_string(os.path.expandvars(directive_args[2].strip()))
                                  .render(**template_args))
            else:
                raise BatchProcessSyntaxException(
                    '\\set directive received less than 2 arguments, syntax is: \\set name value')
            return True

        if line.startswith('\\print'):
            directive_args = line.split(' ', 1)
            if len(directive_args) == 2:
                messages.log(jinja_env.from_string(os.path.expandvars(directive_args[1].strip()))
                             .render(**template_args))
            else:
                raise BatchProcessSyntaxException(
                    '\\print directive received no arguments, syntax is: \\print value')
            return True

        if line.startswith('\\clear_model_cache'):
            clear_model_cache()
            return True
        return False

    def lex_and_run_invocation(line_idx, invocation_string):
        templated_cmd = jinja_env. \
            from_string(os.path.expandvars(invocation_string)).render(**template_args)

        shell_lexed = shlex.split(templated_cmd) + injected_args

        for idx, extra_arg in enumerate(injected_args):
            if any(c.isspace() for c in extra_arg):
                injected_args[idx] = quote(extra_arg)

        header = 'Processing Arguments: '
        args_wrapped = textwrap.fill(templated_cmd + ' ' + ' '.join(injected_args),
                                     width=long_text_wrap_width() - len(header),
                                     break_long_words=False,
                                     break_on_hyphens=False,
                                     subsequent_indent=' ' * len(header))

        messages.log(header + args_wrapped, underline=True)

        try:
            template_args.update(invocation_runner(shell_lexed))
        except Exception as e:
            messages.log(f'Invocation error in input config file line: {line_idx}',
                         level=messages.ERROR, underline=True)
            raise e

    def process_file(stream):
        continuation = ''

        for line_idx, line in enumerate(stream):
            line = line.strip()
            if line == '':
                continue
            if line.startswith('#'):
                look_for_version_mismatch(line_idx, line)
                continue
            if line.endswith('\\'):
                continuation += ' ' + line.rstrip(' \\')
            else:
                completed_continuation = (continuation + ' ' + line).lstrip()

                if directive_handlers(line_idx, completed_continuation):
                    continuation = ''
                    continue

                if completed_continuation.startswith('\\execute'):
                    execute_args = completed_continuation.split(' ', 1)
                    if len(execute_args) == 2:
                        config = execute_args[1].replace('!END', '\n')

                        config = jinja_env. \
                            from_string(os.path.expandvars(config)).render(**template_args)

                        process_file(io.StringIO(config))
                    else:
                        raise BatchProcessSyntaxException(
                            '\\execute directive received no arguments, syntax is: \\execute invocation')

                    continuation = ''
                    continue

                lex_and_run_invocation(line_idx, completed_continuation)

                continuation = ''

    # Process
    process_file(stream=file_stream)
