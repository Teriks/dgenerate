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

__doc__ = """
This module provides a pygments lexer for the dgenerate config / shell language.

This can be used for syntax highlighting.
"""

_DGENERATE_FUNCTIONS = sorted((
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
), key=lambda s: len(s), reverse=True)

_JINJA2_FUNCTIONS = sorted((
    # Functions
    'range', 'lipsum', 'cycler', 'joiner',

    # Filters
    'safe', 'capitalize', 'lower', 'upper', 'title', 'trim', 'striptags', 'urlencode',
    'wordcount', 'replace', 'format', 'escape', 'tojson', 'join', 'sort', 'reverse',
    'length', 'sum', 'random', 'batch', 'slice', 'first', 'last', 'default', 'groupby'
), key=lambda s: len(s), reverse=True)

# Define Jinja2 keywords
_JINJA2_KEYWORDS = sorted((
    'true', 'false', 'none', 'and', 'or', 'not', 'if', 'else', 'elif', 'for', 'endfor', 'in', 'do', 'async', 'await',
    'block', 'extends', 'include', 'import', 'from', 'macro', 'call', 'set', 'with', 'without', 'filter',
    'endfilter', 'capture', 'endcapture', 'spaceless', 'endspaceless', 'flush', 'load', 'url', 'static',
    'trans', 'endtrans', 'as', 'safe', 'end', 'autoescape', 'endautoescape', 'raw', 'endraw'
), key=lambda s: len(s), reverse=True)

# asteval keywords
_SETP_KEYWORDS = sorted((
    'for', 'in', 'if', 'else', 'elif', 'and', 'or', 'not', 'is', 'True', 'False'
), key=lambda s: len(s), reverse=True)

# Common patterns
_comment_pattern = (r'(?<!\\)(#.*$)', _token.Comment.Single)
_env_var_pattern = (r'\$(?:\w+|\{\w+\})|%[\w]+%', _token.Name.Constant)
_jinja_block_pattern = (r'(\{%)(\s*)(\w+)',
                        _lexer.bygroups(_token.String.Interpol, _token.Text, _token.Keyword), 'jinja_block')
_jinja_comment_pattern = (r'(\{#)', _token.Comment.Multiline, 'jinja_comment')
_jinja_interpolate_pattern = (r'(\{\{)', _token.String.Interpol, 'jinja_interpolate')
_operators_punctuation_pattern = (r'[\[\]{}()=\\]', _token.Operator)
_operators_pattern = (r'\*\*|<<|>>|[-+*/%^|&<>!]', _token.Operator)
_size_pattern = (r'(?<!\w)\d+[xX]\d+(?!\w)', _token.Number.Hex)
_number_float_pattern = (r'(?<!\w)(-?\d+(\.\d*)?([eE][-+]?\d+)?)(?!\w)', _token.Number.Float)
_decimal_integer_pattern = (r'(?<!\w)-?\d+(?!\w)', _token.Number.Integer)
_binary_integer_pattern = (r'(?<!\w)0[bB][01]+(?!\w)', _token.Number.Binary)
_hexa_decimal_integer_pattern = (r'(?<!\w)0[xX][0-9a-fA-F]+(?!\w)', _token.Number.Hex)
_octal_integer_pattern = (r'(?<!\w)0[oO][0-7]+(?!\w)', _token.Number.Octal)
_text_pattern = (r'[^=\s\[\]{}()$"\'`\\<&|;:]+', _token.Text)
_variable_names_pattern = (r'\b[a-zA-Z_][a-zA-Z0-9_.]*\b', _token.Name.Variable)

_SCHEDULER_KEYWORDS = sorted((
    "DDIMScheduler",
    "DDPMScheduler",
    "DEISMultistepScheduler",
    "DPMSolverMultistepScheduler",
    "DPMSolverSDEScheduler",
    "DPMSolverSinglestepScheduler",
    "EDMEulerScheduler",
    "EulerAncestralDiscreteScheduler",
    "EulerDiscreteScheduler",
    "HeunDiscreteScheduler",
    "KDPM2AncestralDiscreteScheduler",
    "KDPM2DiscreteScheduler",
    "LCMScheduler",
    "LMSDiscreteScheduler",
    "PNDMScheduler",
    "UniPCMultistepScheduler"
    "DDPMWuerstchenScheduler"),
    key=lambda s: len(s), reverse=True)

_VAE_KEYWORDS = sorted((
    "AutoencoderKL",
    "AsymmetricAutoencoderKL",
    "AutoencoderTiny",
    "ConsistencyDecoderVAE",
    "FlaxAutoencoderKL"
), key=lambda s: len(s), reverse=True)

_MODEL_TYPE_KEYWORDS = sorted((
    'torch',
    'torch-pix2pix',
    'torch-sdxl',
    'torch-if',
    'torch-ifs',
    'torch-ifs-img2img',
    'torch-sdxl-pix2pix',
    'torch-upscaler-x2',
    'torch-upscaler-x4',
    'torch-s-cascade',
    'flax',
    'help',
    'helpargs',
), key=lambda s: len(s), reverse=True)

_DTYPE_KEYWORDS = sorted((
    'float16',
    'bfloat16',
    'float32',
    'float64'
), key=lambda s: len(s), reverse=True)

_number_patterns = (_number_float_pattern,
                    _decimal_integer_pattern,
                    _binary_integer_pattern,
                    _hexa_decimal_integer_pattern,
                    _octal_integer_pattern)


def _create_string_continue(name, char, root_state):
    string_token = _token.String.Double if char == '"' else _token.String.Single
    return {
        name: [
            _env_var_pattern,
            _jinja_block_pattern,
            _jinja_comment_pattern,
            _jinja_interpolate_pattern,
            (r'\\[^\s]', _token.String.Escape),
            (r'\\', _token.Escape, f"{name}_escape"),
            (rf'(?<!\\)(#[^{char}\n]*)(\s*\n\s*)(-)',
             _lexer.bygroups(_token.Comment.Single, _token.Whitespace, string_token), f'{name}_escape'),
            (rf'(?<!\\)(#[^{char}\n]*)(\s*\n)', _lexer.bygroups(_token.Comment.Single, _token.Whitespace), root_state),
            (r'#', string_token),
            (rf'[^{char}\\\n#]+\n\s*-', string_token, f'{name}_escape'),
            (rf'[^{char}\\\n#]+', string_token),
            (char, string_token, '#pop'),
            (r'\n', _token.Whitespace, root_state),
            (r'$', _token.Whitespace, root_state),
        ],
        f'{name}_escape': [
            (r'(#[^\n]*)(\s*\n)', _lexer.bygroups(_token.Comment.Single, _token.Whitespace)),
            (r'(\s|\n)+', _token.Whitespace),
            (r'(?<=[-\n\s.])', _token.Whitespace, f'{name}_continue')
        ],
        f'{name}_continue': [
            _env_var_pattern,
            _jinja_block_pattern,
            _jinja_comment_pattern,
            _jinja_interpolate_pattern,
            (r'\\[^\s]', _token.String.Escape),
            (r'\\', _token.Escape, f"{name}_escape"),
            (rf'(?<!\\)(#[^\n]*)(\s*\n\s*)(-)',
             _lexer.bygroups(_token.Comment.Single, _token.Whitespace, string_token), f'{name}_escape'),
            (r'(?<!\\)(#[^\n]*)(\s*\n)', _lexer.bygroups(_token.Comment.Single, _token.Whitespace), root_state),
            (r'#', string_token),
            (rf'[^{char}\\\n#]+\n\s*-', string_token, f'{name}_escape'),
            (rf'[^{char}\\\n#]+', string_token),
            (char, string_token, root_state),
            (r'\n', _token.Whitespace, root_state),
            (r'$', _token.Whitespace, root_state),
        ]}


class DgenerateLexer(_lexer.RegexLexer):
    """
    pygments lexer for dgenerate configuration / script
    """

    name = 'DgenerateLexer'
    aliases = ['dgenerate']
    filenames = ['*.dgen', '*.txt']

    tokens = {
        'root': [
            _comment_pattern,
            _env_var_pattern,
            (r'(\\set[e]?|\\gen_seeds)(\s+)([a-zA-Z_][a-zA-Z0-9_]*)',
             _lexer.bygroups(_token.Name.Builtin, _token.Text.Whitespace, _token.Name.Variable), 'value'),
            (r'(\\setp)(\s+)([a-zA-Z_][a-zA-Z0-9_]*)',
             _lexer.bygroups(_token.Name.Builtin, _token.Text.Whitespace, _token.Name.Variable), 'setp_value'),
            (r'(\\unset|\\save_modules|\\use_modules|\\clear_modules)(\s+)([a-zA-Z_][a-zA-Z0-9_]*)',
             _lexer.bygroups(_token.Name.Builtin, _token.Text.Whitespace, _token.Name.Variable)),
            (r'(\\[a-zA-Z_][a-zA-Z0-9_]*)', _lexer.bygroups(_token.Name.Builtin)),
            (r'\b(%s)\b' % '|'.join(_SCHEDULER_KEYWORDS), _token.Keyword),
            (r'\b(%s)\b' % '|'.join(_VAE_KEYWORDS), _token.Keyword),
            (r'\b(%s)\b' % '|'.join(_MODEL_TYPE_KEYWORDS), _token.Keyword),
            (r'\b(%s)\b' % '|'.join(_DTYPE_KEYWORDS), _token.Keyword),
            (r'\bauto\b', _token.Keyword),
            (r'\bcpu\b', _token.Keyword),
            (r'\bcuda\b', _token.Keyword),
            (r'[!]END\b', _token.Keyword),
            _jinja_block_pattern,
            _jinja_comment_pattern,
            _jinja_interpolate_pattern,
            _size_pattern,
            *_number_patterns,
            _operators_punctuation_pattern,
            _operators_pattern,
            (r'"', _token.String.Double, 'double_string'),
            (r"'", _token.String.Single, 'single_string'),
            _text_pattern,
            (r'\s+', _token.Whitespace),
        ],
        'value': [
            (r'(\s*?\n\s*?)(-)', _lexer.bygroups(_token.Whitespace, _token.Operator)),
            (r'\s*?\n', _token.Whitespace, 'root'),
            (r'\\[^\s]', _token.Escape),
            (r'\\\s*?\n', _token.Escape, 'value_escape'),
            (r'(\\\s+)(#[^\n]*)', _lexer.bygroups(_token.Whitespace, _token.Comment.Single), 'value_escape'),
            _comment_pattern,
            (r'!END', _token.Keyword.Pseudo),
            _env_var_pattern,
            _jinja_block_pattern,
            _jinja_comment_pattern,
            _jinja_interpolate_pattern,
            _size_pattern,
            *_number_patterns,
            _operators_punctuation_pattern,
            _operators_pattern,
            (r'"', _token.String.Double, 'double_string'),
            (r"'", _token.String.Single, 'single_string'),
            _text_pattern,
            (r'\s+', _token.Whitespace),
        ],
        'setp_value': [
            (r'(\s*?\n\s*?)(-)', _lexer.bygroups(_token.Whitespace, _token.Operator)),
            (r'\s*?\n', _token.Whitespace, 'root'),
            (r'(\n\s*?)(-)', _lexer.bygroups(_token.Whitespace, _token.Number)),
            (r'\\[^\s]', _token.Escape),
            (r'\\\s*?\n', _token.Escape, 'setp_value_escape'),
            (r'(\\\s+)(#[^\n]*)', _lexer.bygroups(_token.Whitespace, _token.Comment.Single), 'setp_value_escape'),
            _comment_pattern,
            (r'!END', _token.Keyword.Pseudo),
            _env_var_pattern,
            (r'\b(%s)\b' % '|'.join(_SETP_KEYWORDS), _token.Keyword),
            (r'\b(%s)\b' % '|'.join(_DGENERATE_FUNCTIONS), _token.Name.Function),
            _variable_names_pattern,
            _jinja_block_pattern,
            _jinja_comment_pattern,
            _jinja_interpolate_pattern,
            *_number_patterns,
            _operators_punctuation_pattern,
            _operators_pattern,
            (r'"', _token.String.Double, 'double_string_setp_value'),
            (r"'", _token.String.Single, 'single_string_setp_value'),
            (r'\s+', _token.Whitespace),
        ],
        f'setp_value_escape': [
            (r'(#[^\n]*)(\s*\n)', _lexer.bygroups(_token.Comment.Single, _token.Whitespace)),
            (r'(\s|\n)+', _token.Whitespace),
            (r'(?<=[-\n\s.])', _token.Whitespace, 'setp_value')
        ],
        f'value_escape': [
            (r'(#[^\n]*)(\s*\n)', _lexer.bygroups(_token.Comment.Single, _token.Whitespace)),
            (r'(\s|\n)+', _token.Whitespace),
            (r'(?<=[-\n\s.])', _token.Whitespace, 'value')
        ],
        **_create_string_continue('double_string', '"', 'root'),
        **_create_string_continue('single_string', "'", 'root'),
        **_create_string_continue('double_string_setp_value', '"', 'setp_value'),
        **_create_string_continue('single_string_setp_value', "'", 'setp_value'),
        'jinja_block': [
            # End of Jinja2 block statement
            (r'(\s*)(%\})', _lexer.bygroups(_token.Text, _token.String.Interpol), '#pop'),

            # Function names
            (r'\b(%s)\b' % '|'.join(_DGENERATE_FUNCTIONS), _token.Name.Function),

            # Jinja2 Function names
            (r'\b(%s)\b' % '|'.join(_JINJA2_FUNCTIONS), _token.Name.Function),

            # Jinja2 Keywords
            (r'\b(%s)\b' % '|'.join(_JINJA2_KEYWORDS), _token.Keyword),

            # Variable names
            _variable_names_pattern,

            # Strings
            (r':?"(\\\\|\\[^\\]|[^"\\])*"', _token.String.Double),
            (r":?'(\\\\|\\[^\\]|[^'\\])*'", _token.String.Single),

            # Numbers
            *_number_patterns,

            # Operators and punctuation
            _operators_punctuation_pattern,
            _operators_pattern,

            # Other text
            _text_pattern,
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
            (r'\b(%s)\b' % '|'.join(_DGENERATE_FUNCTIONS), _token.Name.Function),

            # Jinja2 Function names
            (r'\b(%s)\b' % '|'.join(_JINJA2_FUNCTIONS), _token.Name.Function),

            # Jinja2 Keywords
            (r'\b(%s)\b' % '|'.join(_JINJA2_KEYWORDS), _token.Keyword),

            # Variable names
            _variable_names_pattern,

            # Strings
            (r':?"(\\\\|\\[^\\]|[^"\\])*"', _token.String.Double),
            (r":?'(\\\\|\\[^\\]|[^'\\])*'", _token.String.Single),

            # Numbers
            *_number_patterns,

            # Operators and punctuation
            _operators_punctuation_pattern,
            _operators_pattern,

            # Other text
            _text_pattern,
            (r'\s+', _token.Whitespace),
        ],
    }
