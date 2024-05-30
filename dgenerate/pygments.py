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
import typing

import pygments.lexer as _lexer
import pygments.token as _token

import dgenerate.files as _files


class DgenerateLexer(_lexer.RegexLexer):
    """
    pygments lexer for dgenerate configuration / script
    """

    name = 'DgenerateLexer'
    aliases = ['dgenerate']
    filenames = ['*.dgen', '*.txt']

    DGENERATE_FUNCTIONS = (
        'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
        'callable', 'chr', 'complex', 'cwd', 'dict', 'divmod', 'enumerate',
        'filter', 'first', 'float', 'format', 'format_prompt', 'format_size',
        'frozenset', 'gen_seeds', 'getattr', 'hasattr', 'hash', 'hex', 'int',
        'iter', 'last', 'len', 'list', 'map', 'max', 'min', 'next', 'object',
        'oct', 'ord', 'pow', 'quote', 'range', 'repr', 'reversed', 'round',
        'set', 'slice', 'sorted', 'str', 'sum', 'tuple', 'type', 'unquote', 'zip',
        'glob.glob', 'glob.iglob', 'glob.escape', 'path.abspath', 'path.basename',
        'path.commonpath', 'path.commonprefix',
        'path.dirname', 'path.exists', 'path.expanduser', 'path.expandvars',
        'path.getatime', 'path.getctime', 'path.getmtime', 'path.getsize',
        'path.isabs', 'path.isdir', 'path.isfile', 'path.islink',
        'path.ismount', 'path.join', 'path.lexists', 'path.normcase',
        'path.normpath', 'path.realpath', 'path.relpath', 'path.samefile',
        'path.sameopenfile', 'path.samestat', 'path.split', 'path.splitdrive',
        'path.splitext', 'path.supports_unicode_filenames'
    )

    JINJA2_FUNCTIONS = (
        # Functions
        'range', 'lipsum', 'cycler', 'joiner',

        # Filters
        'safe', 'capitalize', 'lower', 'upper', 'title', 'trim', 'striptags', 'urlencode',
        'wordcount', 'replace', 'format', 'escape', 'tojson', 'join', 'sort', 'reverse',
        'length', 'sum', 'random', 'batch', 'slice', 'first', 'last', 'default', 'groupby'
    )

    # Define Jinja2 keywords
    JINJA2_KEYWORDS = (
        'true', 'false', 'none', 'and', 'or', 'not', 'if', 'else', 'elif', 'for', 'endfor', 'in', 'do', 'async',
        'await',
        'block', 'extends', 'include', 'import', 'from', 'macro', 'call', 'set', 'with', 'without', 'filter',
        'endfilter',
        'capture', 'endcapture', 'spaceless', 'endspaceless', 'flush', 'load', 'url', 'static', 'trans', 'endtrans',
        'as',
        'safe', 'end', 'autoescape', 'endautoescape', 'raw', 'endraw')

    # asteval keywords
    SETP_KEYWORDS = ('for', 'in', 'if', 'else', 'elif', 'and', 'or', 'not', 'is', 'True', 'False')

    # Common patterns
    comment_pattern = (r'(?<!\\)(#.*$)', _token.Comment.Single)
    env_var_pattern = (r'\$(?:\w+|\{\w+\})|%[\w]+%', _token.Name.Constant)
    jinja_block_pattern = (r'(\{%)(\s*)(\w+)',
                           _lexer.bygroups(_token.String.Interpol, _token.Text, _token.Keyword), 'jinja_block')
    jinja_comment_pattern = (r'(\{#)', _token.Comment.Multiline, 'jinja_comment')
    jinja_interpolate_pattern = (r'(\{\{)', _token.String.Interpol, 'jinja_interpolate')
    shell_globs_and_paths_pattern = (r'(~|(\.\.?|~)?/[^=\s\[\]{}()$"\'`\\<&|;]*)', _token.String.Other)
    operators_punctuation_pattern = (r'[\[\]{}()=\\]', _token.Operator.Punctuation)
    operators_pattern = (r'\*\*|<<|>>|[-+*/%^|&<>!]', _token.Operator)
    size_pattern = (r'(?<!\w)\d+[xX]\d+(?!\w)', _token.Number.Hex)
    number_float_pattern = (r'(?<!\w)(-?\d+(\.\d*)?([eE][-+]?\d+)?)(?!\w)', _token.Number.Float)
    decimal_integer_pattern = (r'(?<!\w)-?\d+(?!\w)', _token.Number.Integer)
    binary_integer_pattern = (r'(?<!\w)0[bB][01]+(?!\w)', _token.Number.Binary)
    hexadecimal_integer_pattern = (r'(?<!\w)0[xX][0-9a-fA-F]+(?!\w)', _token.Number.Hex)
    octal_integer_pattern = (r'(?<!\w)0[oO][0-7]+(?!\w)', _token.Number.Octal)
    text_pattern = (r'[^=\s\[\]{}()$"\'`\\<&|;]+', _token.Text)
    double_string_content_pattern = (r'((?:(?!\{\{|{#|\{%)[^"\\\n]|\\[^ \n]|\\[ \t]*(#.*?)?\n)+)', _token.String)
    single_string_content_pattern = (r'((?:(?!\{\{|{#|\{%)[^\'\\\n]|\\[^ \n]|\\[ \t]*(#.*?)?\n)+)', _token.String)
    variable_names_pattern = (r'\b[a-zA-Z_][a-zA-Z0-9_.]*\b', _token.Name.Variable)

    number_patterns = (number_float_pattern,
                       decimal_integer_pattern,
                       binary_integer_pattern,
                       hexadecimal_integer_pattern,
                       octal_integer_pattern)

    tokens = {
        'root': [
            comment_pattern,
            env_var_pattern,
            (r'(\\set[e]?|\\gen_seeds)(\s+)([a-zA-Z_][a-zA-Z0-9_]*)',
             _lexer.bygroups(_token.Name.Builtin, _token.Text.Whitespace, _token.Name.Variable), 'value'),
            (r'(\\setp)(\s+)([a-zA-Z_][a-zA-Z0-9_]*)',
             _lexer.bygroups(_token.Name.Builtin, _token.Text.Whitespace, _token.Name.Variable), 'setp_value'),
            (r'(\\unset|\\save_modules|\\use_modules|\\clear_modules)(\s+)([a-zA-Z_][a-zA-Z0-9_]*)',
             _lexer.bygroups(_token.Name.Builtin, _token.Text.Whitespace, _token.Name.Variable)),
            (r'(\\[a-zA-Z_][a-zA-Z0-9_]*)', _token.Name.Builtin),
            (r'!END', _token.Keyword.Pseudo),
            shell_globs_and_paths_pattern,
            jinja_block_pattern,
            jinja_comment_pattern,
            jinja_interpolate_pattern,
            size_pattern,
            *number_patterns,
            operators_punctuation_pattern,
            operators_pattern,
            (r'"', _token.String.Double, 'double_string'),
            (r"'", _token.String.Single, 'single_string'),
            text_pattern,
            (r'\s+', _token.Whitespace),
        ],
        'value': [
            (r'\n', _token.Whitespace, '#pop'),
            comment_pattern,
            (r'!END', _token.Keyword.Pseudo),
            env_var_pattern,
            shell_globs_and_paths_pattern,
            jinja_block_pattern,
            jinja_comment_pattern,
            jinja_interpolate_pattern,
            size_pattern,
            *number_patterns,
            operators_punctuation_pattern,
            operators_pattern,
            (r'"', _token.String.Double, 'double_string'),
            (r"'", _token.String.Single, 'single_string'),
            text_pattern,
            (r'\s+', _token.Whitespace),
        ],
        'setp_value': [
            (r'\n', _token.Whitespace, '#pop'),
            comment_pattern,
            (r'!END', _token.Keyword.Pseudo),
            env_var_pattern,
            (r'\b(%s)\b' % '|'.join(SETP_KEYWORDS), _token.Keyword),
            (r'\b(%s)\b' % '|'.join(DGENERATE_FUNCTIONS), _token.Name.Function),
            variable_names_pattern,
            jinja_block_pattern,
            jinja_comment_pattern,
            jinja_interpolate_pattern,
            *number_patterns,
            operators_punctuation_pattern,
            operators_pattern,
            (r'"', _token.String.Double, 'double_string'),
            (r"'", _token.String.Single, 'single_string'),
            text_pattern,
            (r'\s+', _token.Whitespace),
        ],
        'double_string': [
            env_var_pattern,
            jinja_block_pattern,
            jinja_comment_pattern,
            jinja_interpolate_pattern,
            double_string_content_pattern,
            (r'"', _token.String.Double, '#pop'),
            (r'\n', _token.String.Error, '#pop'),
            (r'$', _token.String.Error, '#pop'),
        ],
        'single_string': [
            env_var_pattern,
            jinja_block_pattern,
            jinja_comment_pattern,
            jinja_interpolate_pattern,
            single_string_content_pattern,
            (r"'", _token.String.Single, '#pop'),
            (r'\n', _token.String.Error, '#pop'),
            (r'$', _token.String.Error, '#pop'),
        ],
        'jinja_block': [
            # End of Jinja2 block statement
            (r'(\s*)(%\})', _lexer.bygroups(_token.Text, _token.String.Interpol), '#pop'),

            # Function names
            (r'\b(%s)\b' % '|'.join(DGENERATE_FUNCTIONS), _token.Name.Function),

            # Jinja2 Function names
            (r'\b(%s)\b' % '|'.join(JINJA2_FUNCTIONS), _token.Name.Function),

            # Jinja2 Keywords
            (r'\b(%s)\b' % '|'.join(JINJA2_KEYWORDS), _token.Keyword),

            # Variable names
            variable_names_pattern,

            # Strings
            (r'"', _token.String.Double, 'double_string'),
            (r"'", _token.String.Single, 'single_string'),

            # Numbers
            *number_patterns,

            # Operators and punctuation
            operators_punctuation_pattern,
            operators_pattern,

            # Other text
            text_pattern,
            (r'\s+', _token.Whitespace),
        ],
        'jinja_comment': [
            # End of Jinja2 comment
            (r'(#\})', _token.Comment.Multiline, '#pop'),

            # Comment content
            (r'.+?(?=#\})', _token.Comment.Multiline),
        ],
        'jinja_interpolate': [
            # End of Jinja2 template expression
            (r'(\}\})', _token.String.Interpol, '#pop'),

            # Function names
            (r'\b(%s)\b' % '|'.join(DGENERATE_FUNCTIONS), _token.Name.Function),

            # Jinja2 Function names
            (r'\b(%s)\b' % '|'.join(JINJA2_FUNCTIONS), _token.Name.Function),

            # Jinja2 Keywords
            (r'\b(%s)\b' % '|'.join(JINJA2_KEYWORDS), _token.Keyword),

            # Variable names
            variable_names_pattern,

            # Strings
            (r'"', _token.String.Double, 'double_string'),
            (r"'", _token.String.Single, 'single_string'),

            # Numbers
            *number_patterns,

            # Operators and punctuation
            operators_punctuation_pattern,
            operators_pattern,

            # Other text
            text_pattern,
            (r'\s+', _token.Whitespace),
        ],
    }


def format_code(lines: typing.Iterator[str], indentation=' ' * 4) -> typing.Iterator[str]:
    """
    A very rudimentary code formatter for dgenerate configuration / script

    :param lines: iterator over lines
    :param indentation: level of indentation for top level jinja control blocks
    :return: formatted code
    """

    lines = _files.PeekReader(lines)
    indent_level = 0
    in_continuation = False
    continuation_lines = []

    def add_continuation_lines():
        if continuation_lines:
            # Find the minimum indentation level among non-empty lines
            min_indent = min((len(ln) - len(ln.lstrip())) for ln in continuation_lines if ln.strip())
            # Remove the minimum indentation from all continuation lines
            cleaned_lines = [ln[min_indent:] if ln.strip() else ln for ln in continuation_lines]
            yield from [indentation * indent_level + ln for ln in cleaned_lines]
            continuation_lines.clear()

    def is_continuation_ended():
        # Check backward through the continuation lines to see
        # if the last non-comment, non-empty line ended with a backslash
        for line in reversed(continuation_lines):
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith('#'):
                return not stripped_line.endswith('\\')
        return True

    for line, next_line in lines:
        stripped = line.strip()

        # Handle line continuation with backslash or hyphen
        if in_continuation or line.endswith('\\') or (next_line is not None and next_line.strip().startswith('-')):
            continuation_lines.append(line)
            if next_line is None:
                in_continuation = False
                yield from add_continuation_lines()
            elif not line.endswith('\\') and not (next_line.strip().startswith('-')):
                if is_continuation_ended():
                    in_continuation = False
                    yield from add_continuation_lines()
            else:
                in_continuation = True
            continue

        # Handle Jinja2 control structures
        if stripped.startswith('{% end') or stripped.startswith('{% else') or stripped.startswith('{% elif'):
            indent_level = max(indent_level - 1, 0)

        # Process any accumulated continuation lines before adding a new block line
        yield from add_continuation_lines()

        # Add the current line with the appropriate indentation
        yield indentation * indent_level + stripped

        # Increase indent level for starting control structures
        if (stripped.startswith('{% for') or stripped.startswith('{% if') or
                stripped.startswith('{% block') or stripped.startswith('{% macro') or
                stripped.startswith('{% filter') or stripped.startswith('{% with') or
                stripped.startswith('{% else') or stripped.startswith('{% elif')):
            indent_level += 1

        # Set continuation flag if the line ends with a backslash
        if line.endswith('\\'):
            in_continuation = True

    # Handle any remaining continuation lines after the last line
    yield from add_continuation_lines()
