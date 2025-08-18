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
import argparse
import json
import shlex

import toml
import yaml

import dgenerate.arguments as _arguments
import dgenerate.batchprocess.util as _b_util
import dgenerate.messages as _messages
import dgenerate.prompt as _prompt
import dgenerate.promptupscalers as _promptupscalers
import dgenerate.subcommands.subcommand as _subcommand
import dgenerate.textprocessing as _textprocessing
import dgenerate.torchutil as _torchutil


def _type_output_format(s):
    s = s.lower()
    if s in {'text', 'json', 'toml', 'yaml'}:
        return s
    raise ValueError(f'Invalid output format: {s}')


def _quote_style(s):
    s = s.lower()
    if s in {'none', 'shell', 'dgenerate'}:
        return s
    raise ValueError(f'Invalid quote style: {s}')


def _type_prompts(prompt):
    try:
        prompt = _prompt.Prompt.parse(prompt, embedded_arg_names=['upscaler'])
        return prompt
    except (ValueError, _prompt.PromptEmbeddedArgumentError) as e:
        raise argparse.ArgumentTypeError(
            f'Prompt parse error: {str(e).strip()}')


class PromptUpscaleSubCommand(_subcommand.SubCommand):
    """
    Upscale prompts without performing image generation.

    This allows you to run prompt upscalers on prompts and simply output the results.

    See: dgenerate --sub-command prompt-upscale --help
    """

    NAMES = ['prompt-upscale']

    def __init__(self, program_name='prompt-upscale', output_list: list = None, **kwargs):
        super().__init__(**kwargs)

        self._parser = parser = _b_util.DirectiveArgumentParser(
            prog=program_name,
            description='Upscale prompts without performing image generation.')

        self._output_list = output_list

        parser.add_argument(
            '-p', '--prompts',
            help="""Prompts (required), identical to the dgenerate --prompts argument.
                    The embedded prompt argument <upscaler: ...>, is understood. 
                    All other embedded prompt arguments are entirely ignored and 
                    left in the prompt, be aware of this.""", nargs='+',
            type=_type_prompts,
            required=True
        )

        parser.add_argument(
            '-u', '--upscaler', '--upscalers',
            metavar='PROMPT_UPSCALER_URI',
            help="""Global prompt upscaler(s) to use, identical to the
                    dgenerate --prompt-upscaler argument. Providing multiple
                    prompt upscaler plugin URIs indicates chaining.""", nargs='+',
            type=_arguments._type_prompt_upscaler
        )

        default_device = _torchutil.default_device()

        parser.add_argument(
            '-d', '--device',
            help=f"""Acceleration device to use for prompt upscalers that support
                     acceleration. Defaults to: {default_device}""", default=default_device,
            type=_arguments._type_device
        )

        if output_list is None:
            parser.add_argument(
                '-of', '--output-format',
                help='Output format. defaults to "text", can be: "text", "json", "toml", "yaml".',
                default='text',
                type=_type_output_format
            )

            parser.add_argument(
                '-o', '--output',
                help='Output file path. default to printing to stdout.', default=None
            )

            parser.add_argument(
                '-q', '--quote',
                help="""Quoting method when --output-format is "text", defaults to "none".
                        May be one of: none (raw strings), shell (shlex.quote), dgenerate (dgenerate config shell syntax).
                        If you are generating output in text mode, and you intend to do something with the output
                        other than just look at it, --quote "none" will be problematic for multiline prompts.""",
                default='none',
                type=_quote_style
            )

        parser.add_argument(
            '-ofm', '--offline-mode', action='store_true',
            help="""Prevent downloads of resources that do not exist on disk already.""")

    @staticmethod
    def _do_quote(args, s):
        if args.quote == 'none':
            return s
        elif args.quote == 'shell':
            return shlex.quote(s)
        elif args.quote == 'dgenerate':
            return _textprocessing.shell_quote(s)
        else:
            assert False, f'Invalid quote style: {args.quote}'

    def __call__(self) -> int:
        args = self._parser.parse_args(self.args)

        if self._parser.return_code is not None:
            return self._parser.return_code

        def get_upscaled():
            return _promptupscalers.upscale_prompts(
                prompts=args.prompts,
                default_upscaler_uri=args.upscaler,
                device=args.device,
                local_files_only=self.local_files_only or args.offline_mode
            )

        if self._output_list is not None:
            for upscaled in get_upscaled():
                self._output_list.append(str(upscaled))
            return 0

        if args.quote != 'none' and args.output_format != 'text':
            _messages.error(
                'Quote style -q/--quote only applies to text output format.')
            return 1

        if args.output_format == 'text':
            if not args.output:
                for upscaled in get_upscaled():
                    _messages.log(self._do_quote(args, str(upscaled)))
            else:
                with open(args.output, 'wt') as output_file:
                    for upscaled in get_upscaled():
                        output_file.write(self._do_quote(args, str(upscaled)) + '\n')
        else:
            output_object = []
            for upscaled in get_upscaled():
                output_object.append(
                    {
                        'positive': upscaled.positive,
                        'negative': upscaled.negative
                    }
                )

            if args.output:
                with open(args.output, 'w', encoding='utf-8') as output_file:
                    if args.output_format == 'json':
                        json.dump(output_object, output_file, indent=4)
                    elif args.output_format == 'toml':
                        toml.dump({'prompts': output_object}, output_file)
                    elif args.output_format == 'yaml':
                        yaml.dump(output_object, output_file, sort_keys=False, allow_unicode=True)
                    else:
                        assert False, f'Invalid output format: {args.output_format}'
            else:
                if args.output_format == 'json':
                    _messages.log(json.dumps(output_object, indent=4))
                elif args.output_format == 'toml':
                    _messages.log(toml.dumps({'prompts': output_object}))
                elif args.output_format == 'yaml':
                    _messages.log(yaml.dump(output_object, sort_keys=False, allow_unicode=True))
                else:
                    assert False, f'Invalid output format: {args.output_format}'

        return 0
