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

import argparse
import pathlib
import sys

import dgenerate.batchprocess.util as _b_util
import dgenerate.messages as _messages
import dgenerate.subcommands.subcommand as _subcommand


class TranspileSubCommand(_subcommand.SubCommand):
    """
    Translate a dgenerate config into a Python script.

    Static generation lines become ``DgenerateArguments`` objects.
    Directives become ``script.directive`` calls. ``{% if %}``, ``{% for %}``,
    ``{% set %}``, ``{% filter %}``, and ``{% with %}`` become Python.

    The default output reproduces the interpreter's error lines. ``--relaxed``
    emits the same actions without that bookkeeping. ``--format`` wraps long
    calls and lists and leaves a blank line between invocations. Config
    comments are left out. ``--comments`` keeps them. A syntax error in the
    config stops translation.

    Examples:

    dgenerate --sub-command transpile examples/config_syntax/basic-syntax-config.dgen

    dgenerate --sub-command transpile config.dgen -o config.py

    dgenerate --sub-command transpile --stdin --relaxed -o config.py

    See: dgenerate --sub-command transpile --help
    """

    NAMES = ['transpile']

    def __init__(self, program_name='transpile', **kwargs):
        super().__init__(**kwargs)
        self._parser = parser = _b_util.DirectiveArgumentParser(
            prog=program_name,
            description='Transpile a dgenerate config into a Python script.')
        parser.add_argument(
            'config',
            nargs='?',
            help='Path to a .dgen config. Omit this with --stdin to read the config from stdin.')
        parser.add_argument(
            '--stdin',
            action='store_true',
            help='Read the config from stdin instead of a file.')
        parser.add_argument(
            '-o', '--output',
            help='Write the script to this file instead of stdout.')
        parser.add_argument(
            '--tree',
            action='store_true',
            help='Print the ANTLR parse tree and exit.')
        parser.add_argument(
            '--relaxed',
            action='store_true',
            help='Emit a script that runs the config without reproducing '
                 'the interpreter\'s error lines.')
        parser.add_argument(
            '--format',
            action='store_true',
            help='Wrap long calls and lists, and leave a blank line between invocations.')
        parser.add_argument(
            '--comments',
            action=argparse.BooleanOptionalAction,
            default=False,
            help='Keep config comments in the script. They are left out by default.')

    def __call__(self) -> int:
        args = self._parser.parse_args(self.args)
        if self._parser.return_code is not None:
            return self._parser.return_code

        if args.stdin and args.config:
            _messages.error('Pass a config path or --stdin, not both.')
            return 1
        if args.stdin:
            text = sys.stdin.read()
        elif args.config:
            path = pathlib.Path(args.config)
            if not path.is_file():
                _messages.error(f'Config file not found: {path}')
                return 1
            text = path.read_text(encoding='utf-8')
        else:
            _messages.error('A config path is required, or use --stdin.')
            return 1

        import dgenerate.transpile as _transpile

        if args.tree:
            try:
                tree = _transpile.parse_config(text)
            except _transpile.ParseError as exc:
                _messages.error(str(exc))
                return 1
            output = tree.toStringTree(recog=_transpile.Config(None))
        else:
            try:
                output = _transpile.transpile(
                    text,
                    relaxed=args.relaxed,
                    format_output=args.format,
                    comments=args.comments,
                )
            except _transpile.ParseError as exc:
                _messages.error(str(exc))
                return 1

        if args.output:
            pathlib.Path(args.output).write_text(output, encoding='utf-8')
        else:
            sys.stdout.write(output if output.endswith('\n') else output + '\n')
        return 0
