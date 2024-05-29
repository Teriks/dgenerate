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

import pygments.lexer as _lexer
import pygments.token as _token


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
    operators_punctuation_pattern = (r'[\[\]{}()=\\]', _token.Operator)
    operators_pattern = (r'\*\*|<<|>>|[-+*/%^|&<>!]', _token.Operator)
    number_float_pattern = (r'(?<!\w)(-?\d+(\.\d*)?([eE][-+]?\d+)?)(?!\w)', _token.Number.Float)
    size_pattern = (r'\d+x\d+', _token.Number.Hex)
    number_integer_pattern = (r'(?<!\w)\d+(?!\w)', _token.Number.Integer)
    text_pattern = (r'[^=\s\[\]{}()$"\'`\\<&|;]+', _token.Text)
    double_string_content_pattern = (r'((?:(?!\{\{|{#|\{%)[^"\\\n]|\\[^ \n]|\\[ \t]*(#.*?)?\n)+)', _token.String)
    single_string_content_pattern = (r'((?:(?!\{\{|{#|\{%)[^\'\\\n]|\\[^ \n]|\\[ \t]*(#.*?)?\n)+)', _token.String)
    variable_names_pattern = (r'\b[a-zA-Z_][a-zA-Z0-9_.]*\b', _token.Name.Variable)

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
            operators_punctuation_pattern,
            operators_pattern,
            (r'"', _token.String.Double, 'double_string'),
            (r"'", _token.String.Single, 'single_string'),
            (r';', _token.Punctuation),
            (r'\|', _token.Punctuation),
            (r'\s+', _token.Whitespace),
            number_float_pattern,
            size_pattern,
            number_integer_pattern,
            text_pattern,
            (r'<', _token.Text),
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
            operators_punctuation_pattern,
            operators_pattern,
            (r'"', _token.String.Double, 'double_string'),
            (r"'", _token.String.Single, 'single_string'),
            number_float_pattern,
            size_pattern,
            number_integer_pattern,
            text_pattern,
            (r'\s+', _token.Whitespace),
        ],
        'setp_value': [
            (r'\n', _token.Whitespace, '#pop'),
            comment_pattern,
            (r'!END', _token.Keyword.Pseudo),
            env_var_pattern,
            (r'\b(%s)\b' % '|'.join(SETP_KEYWORDS), _token.Keyword),
            shell_globs_and_paths_pattern,
            jinja_block_pattern,
            jinja_comment_pattern,
            jinja_interpolate_pattern,
            operators_punctuation_pattern,
            operators_pattern,
            (r'"', _token.String.Double, 'double_string'),
            (r"'", _token.String.Single, 'single_string'),
            number_float_pattern,
            size_pattern,
            number_integer_pattern,
            (r'\b(%s)\b' % '|'.join(DGENERATE_FUNCTIONS), _token.Name.Function),
            text_pattern,
            (r'\s+', _token.Whitespace),
        ],
        'double_string': [
            env_var_pattern,
            jinja_block_pattern,
            jinja_comment_pattern,
            jinja_interpolate_pattern,
            double_string_content_pattern,
            (r'"', _token.String, '#pop'),
            (r'\n', _token.String.Error, '#pop'),
            (r'$', _token.String.Error, '#pop'),
        ],
        'single_string': [
            env_var_pattern,
            jinja_block_pattern,
            jinja_comment_pattern,
            jinja_interpolate_pattern,
            single_string_content_pattern,
            (r"'", _token.String, '#pop'),
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
            number_float_pattern,
            size_pattern,
            number_integer_pattern,

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
            number_float_pattern,
            size_pattern,
            number_integer_pattern,

            # Operators and punctuation
            operators_punctuation_pattern,
            operators_pattern,

            # Other text
            text_pattern,
            (r'\s+', _token.Whitespace),
        ],
    }
