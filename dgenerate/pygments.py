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
_ecos = r'(\\[^\s]|[^/\s\\#=;:{}"\'])'
_comment_pattern = (r'(?<!\\)(#.*$)', _token.Comment.Single)
_env_var_pattern = (r'\$(?:\w+|\{\w+\})|%[\w]+%', _token.Name.Constant)

_jinja_block_pattern = (
    r'(\{%)(\s*)(\w+)',
    _lexer.bygroups(
        _token.String.Interpol,
        _token.Whitespace,
        _token.Keyword
    ), 'jinja_block'
)

_jinja_comment_pattern = (r'(\{#)', _token.Comment.Multiline, 'jinja_comment')
_jinja_interpolate_pattern = (r'(\{\{)', _token.String.Interpol, 'jinja_interpolate')
_operators_punctuation_pattern = (r'[\[\]{}()=\\;,:]', _token.Operator)
_operators_pattern = (r'\*\*|<<|>>|[-+*/%^|&<>!.$]', _token.Operator)
_dimensions_pattern = (r'(?<!\w)-?\d+[xX]-?\d+([xX]-?\d+){0,2}(?!\w)', _token.Number.Hex)
_number_float_pattern = (r'(?<!\w)(-?\d+(\.\d*)?([eE][-+]?\d+)?)(?!\w)', _token.Number.Float)
_decimal_integer_pattern = (r'(?<!\w)-?\d+(?!\w)', _token.Number.Integer)
_binary_integer_pattern = (r'(?<!\w)0[bB][01]+(?!\w)', _token.Number.Binary)
_hexa_decimal_integer_pattern = (r'(?<!\w)0[xX][0-9a-fA-F]+(?!\w)', _token.Number.Hex)
_octal_integer_pattern = (r'(?<!\w)0[oO][0-7]+(?!\w)', _token.Number.Octal)
_text_pattern = (r'[^=\s\[\]{}()$"\'`\\<&|;:,]+', _token.Text)
_variable_names_pattern = (r'(?<!\w)[a-zA-Z_][a-zA-Z0-9_]*(?!\w)', _token.Name.Variable)

_function_call_pattern = (
    r'(?<!\w)([a-zA-Z_][a-zA-Z0-9_]*)(\()',
    _lexer.bygroups(_token.Name.Function,
                    _token.Operator)
)

_http_pattern = (
    r'(?<!\w)(https?://(?:'  # Protocol
    r'(?:[a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,6}|'  # Domain
    r'localhost|'  # Localhost
    r'(?:\d{1,3}\.){3}\d{1,3}|'  # IPv4
    r'\[[a-fA-F0-9:]+\])'  # IPv6
    r'(?::\d+)?'  # Optional port
    r'(?:/(\\[$%\']?|[a-zA-Z0-9\-._~%!$&\'()*+,=:@/])*)?'  # Path (including allowed shell escapes)
    r'(?:\?(\\[$%\']?|[a-zA-Z0-9\-._~%!$&\'()*+,=:@/?])*)?'  # Query parameters (including allowed shell escapes)
    r')', _token.String
)

_path_patterns = (
    _http_pattern,
    (rf'(?<!\w)([a-zA-Z]:(([/]|\\\\){_ecos}*)+)', _token.String),  # Drive letters
    (rf'(?<!\w)(~|..?)?(([/]|\\\\){_ecos}+)(([/]|\\\\){_ecos}*)*', _token.String),
    # absolute paths with relative components
    (rf'(?<!\w)({_ecos}+)(([/]|\\\\){_ecos}*)+', _token.String),  # relative paths with relative components
    (rf'(?<!\w){_ecos}+\.{_ecos}+', _token.String)
)

_MISC_KEYWORDS = sorted((
    'help',
    'helpargs',
    'null',
    'auto',
    'cpu',
    'cuda',
    'mps'
), key=lambda s: len(s), reverse=True)

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
    "UniPCMultistepScheduler",
    "DDPMWuerstchenScheduler",
    "FlowMatchEulerDiscreteScheduler"
), key=lambda s: len(s), reverse=True)

_CLASS_KEYWORDS = sorted((
    "AutoencoderKL",
    "AsymmetricAutoencoderKL",
    "AutoencoderTiny",
    "ConsistencyDecoderVAE",
    "CLIPTextModel",
    "CLIPTextModelWithProjection",
    "T5EncoderModel",
    "ChatGLMModel",
    "DistillT5EncoderModel"
), key=lambda s: len(s), reverse=True)

_MODEL_TYPE_KEYWORDS = sorted((
    'sd',
    'pix2pix',
    'sd3',
    'sd3-pix2pix',
    'flux',
    'flux-fill',
    'flux-kontext',
    'sdxl',
    'kolors',
    'if',
    'ifs',
    'ifs-img2img',
    'sdxl-pix2pix',
    'upscaler-x2',
    'upscaler-x4',
    's-cascade'
), key=lambda s: len(s), reverse=True)

_DTYPE_KEYWORDS = sorted((
    'float16',
    'bfloat16',
    'float32',
    'float64',
    'fp16',
    'bf16'
), key=lambda s: len(s), reverse=True)

_number_patterns = (_number_float_pattern,
                    _decimal_integer_pattern,
                    _binary_integer_pattern,
                    _hexa_decimal_integer_pattern,
                    _octal_integer_pattern)


def _create_string_continue(name, char, root_state):
    string_token = _token.String.Double if char == '"' else _token.String.Single
    env_vars = [_env_var_pattern] if char == '"' else []
    return {
        name: env_vars + [
            _jinja_block_pattern,
            _jinja_comment_pattern,
            _jinja_interpolate_pattern,
            (r'\\[^\s]', _token.String.Escape),
            (r'\\', _token.Operator, f"{name}_escape"),
            (rf'(?<!\\)(#[^{char}\n]*)(\s*\n\s*)(-)',
             _lexer.bygroups(_token.Comment.Single, _token.Whitespace, string_token), f'{name}_escape'),
            (rf'(?<!\\)(#[^{char}\n]*)(\s*\n)', _lexer.bygroups(_token.Comment.Single, _token.Whitespace), root_state),
            (r'#', string_token),
            (rf'[^{char}\\\n#{{]+\n\s*-', string_token, f'{name}_escape'),
            (rf'[^{char}\\\n#{{]+', string_token),
            (r'{[^{#%]', string_token),
            (char, string_token, '#pop'),
            (r'\n', _token.Whitespace, root_state),
            (r'$', _token.Whitespace, root_state),
        ],
        f'{name}_escape': [
            (r'(#[^\n]*)(\s*\n)', _lexer.bygroups(_token.Comment.Single, _token.Whitespace)),
            (r'(\s|\n)+', _token.Whitespace),
            (r'(?<=[-\n\s.])', _token.Whitespace, f'{name}_continue')
        ],
        f'{name}_continue': env_vars + [
            _jinja_block_pattern,
            _jinja_comment_pattern,
            _jinja_interpolate_pattern,
            (r'\\[^\s]', _token.String.Escape),
            (r'\\', _token.Operator, f"{name}_escape"),
            (rf'(?<!\\)(#[^\n]*)(\s*\n\s*)(-)',
             _lexer.bygroups(_token.Comment.Single, _token.Whitespace, string_token), f'{name}_escape'),
            (r'(?<!\\)(#[^\n]*)(\s*\n)', _lexer.bygroups(_token.Comment.Single, _token.Whitespace), root_state),
            (r'#', string_token),
            (rf'[^{char}\\\n#{{]+\n\s*-', string_token, f'{name}_escape'),
            (rf'[^{char}\\\n#{{]+', string_token),
            (r'{[^{#%]', string_token),
            (char, string_token, root_state),
            (r'\n', _token.Whitespace, root_state),
            (r'$', _token.Whitespace, root_state),
        ]}


def _create_wait_for_var(name, next_state):
    return {
        name: [
            ('([a-zA-Z_][a-zA-Z0-9_]*)', _token.Name.Variable, next_state),
            _env_var_pattern + (next_state,),
            (r'(?<=\}\})\s', _token.Whitespace, next_state),
            (r'(?<=%\})\s', _token.Whitespace, next_state),
            (r'(?<=#\})\s', _token.Whitespace, next_state),
            _comment_pattern,
            _jinja_block_pattern,
            _jinja_comment_pattern,
            _jinja_interpolate_pattern,
            _operators_punctuation_pattern,
            _operators_pattern,
            _text_pattern,
            (r'\s+', _token.Whitespace),
        ]
    }

def _keyword_top_level(s):
    return rf'(?<![-_])\b({s})\b(?![-_])'

class DgenerateLexer(_lexer.RegexLexer):
    """
    pygments lexer for dgenerate configuration / script
    """

    name = 'DgenerateLexer'
    aliases = ['dgenerate']
    filenames = ['*.dgen']

    tokens = {
        **_create_wait_for_var('var_then_value', 'value'),
        **_create_wait_for_var('var_then_setp_value', 'setp_value'),
        **_create_wait_for_var('var_then_root', 'root'),
        'root': [
            _comment_pattern,
            _env_var_pattern,
            (r'(?<!\w)(\\set[e]?|\\gen_seeds|\\download)(\s+)',
             _lexer.bygroups(_token.Name.Builtin, _token.Text.Whitespace), 'var_then_value'),
            (r'(?<!\w)(\\setp)(\s+)',
             _lexer.bygroups(_token.Name.Builtin, _token.Text.Whitespace), 'var_then_setp_value'),
            (r'(?<!\w)(\\import_plugins)(\s+)',
             _lexer.bygroups(_token.Name.Builtin, _token.Text.Whitespace), 'import_plugins_value'),
            (r'(?<!\w)(\\import)(\s+)',
             _lexer.bygroups(_token.Name.Builtin, _token.Text.Whitespace), 'import_value'),
            (r'(?<!\w)(\\unset|\\save_modules|\\use_modules|\\clear_modules)(\s+)',
             _lexer.bygroups(_token.Name.Builtin, _token.Text.Whitespace), 'var_then_root'),
            (r'(?<!\w)(\\[a-zA-Z_][a-zA-Z0-9_]*)', _lexer.bygroups(_token.Name.Builtin)),
            (r'\\[^\s]', _token.Escape),
            (_keyword_top_level('|'.join(_MISC_KEYWORDS)), _token.Keyword),
            (_keyword_top_level('|'.join(_SCHEDULER_KEYWORDS)), _token.Keyword),
            (_keyword_top_level('|'.join(_CLASS_KEYWORDS)), _token.Keyword),
            (_keyword_top_level('|'.join(_MODEL_TYPE_KEYWORDS)), _token.Keyword),
            (_keyword_top_level('|'.join(_DTYPE_KEYWORDS)), _token.Keyword),
            (_keyword_top_level('True|true'), _token.Keyword),
            (_keyword_top_level('False|false'), _token.Keyword),
            _jinja_block_pattern,
            _jinja_comment_pattern,
            _jinja_interpolate_pattern,
            _dimensions_pattern,
            *_number_patterns,
            *_path_patterns,
            _operators_punctuation_pattern,
            _operators_pattern,
            _function_call_pattern,
            (r'"', _token.String.Double, 'double_string'),
            (r"'", _token.String.Single, 'single_string'),
            _text_pattern,
            (r'\s+', _token.Whitespace),
        ],
        'value': [
            (r'(\s*?\n\s*?)(-)', _lexer.bygroups(_token.Whitespace, _token.Operator)),
            (r'\s*?\n', _token.Whitespace, 'root'),
            (r'\\[^\s]', _token.Escape),
            (r'(\\)(\s*?\n)', _lexer.bygroups(_token.Operator, _token.Whitespace), 'value_escape'),
            (r'(\\)(\s+)(#[^\n]*)', _lexer.bygroups(_token.Operator, _token.Whitespace, _token.Comment.Single),
             'value_escape'),
            _comment_pattern,
            _env_var_pattern,
            _jinja_block_pattern,
            _jinja_comment_pattern,
            _jinja_interpolate_pattern,
            _dimensions_pattern,
            *_number_patterns,
            *_path_patterns,
            _operators_punctuation_pattern,
            _operators_pattern,
            _function_call_pattern,
            (r'"', _token.String.Double, 'double_string'),
            (r"'", _token.String.Single, 'single_string'),
            _text_pattern,
            (r'\s+', _token.Whitespace),
        ],
        'setp_value': [
            (r'(\s*?\n\s*?)(-)', _lexer.bygroups(_token.Whitespace, _token.Operator)),
            (r'\s*?\n', _token.Whitespace, 'root'),
            (r'(\n\s*?)(-)', _lexer.bygroups(_token.Whitespace, _token.Operator)),
            (r'\\[^\s]', _token.Escape),
            (r'(\\)(\s*?\n)', _lexer.bygroups(_token.Operator, _token.Whitespace), 'setp_value_escape'),
            (r'(\\)(\s+)(#[^\n]*)', _lexer.bygroups(_token.Operator, _token.Whitespace, _token.Comment.Single),
             'setp_value_escape'),
            _comment_pattern,
            _env_var_pattern,
            (r'\b(%s)\b' % '|'.join(_SETP_KEYWORDS), _token.Keyword),
            _function_call_pattern,
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
        'import_plugins_value': [
            (r'(\s*?\n\s*?)(-)', _lexer.bygroups(_token.Whitespace, _token.Operator)),
            (r'\s*?\n', _token.Whitespace, 'root'),
            (r'(\n\s*?)(-)', _lexer.bygroups(_token.Whitespace, _token.Operator)),
            (r'\\[^\s]', _token.Escape),
            (r'(\\)(\s*?\n)', _lexer.bygroups(_token.Operator, _token.Whitespace), 'import_plugins_value_escape'),
            (r'(\\)(\s+)(#[^\n]*)', _lexer.bygroups(_token.Operator, _token.Whitespace, _token.Comment.Single),
             'import_plugins_value_escape'),
            _comment_pattern,
            _env_var_pattern,
            *((p[0], _token.Keyword) for p in _path_patterns[1:4]),  # everything except http and file.ext
            (_variable_names_pattern[0], _token.Keyword),  # plugin names are keywords
            (r'[.]', _token.Operator),  # namespace operator
            _jinja_block_pattern,
            _jinja_comment_pattern,
            _jinja_interpolate_pattern,
            (r'"', _token.String.Double, 'double_string_import_plugins_value'),
            (r"'", _token.String.Single, 'single_string_import_plugins_value'),
            (r'\s+', _token.Whitespace),
        ],
        'import_value': [
            (r'(\s*?\n\s*?)(-)', _lexer.bygroups(_token.Whitespace, _token.Operator)),
            (r'\s*?\n', _token.Whitespace, 'root'),
            (r'(\n\s*?)(-)', _lexer.bygroups(_token.Whitespace, _token.Operator)),
            (r'\\[^\s]', _token.Escape),
            (r'(\\)(\s*?\n)', _lexer.bygroups(_token.Operator, _token.Whitespace), 'import_value_escape'),
            (r'(\\)(\s+)(#[^\n]*)', _lexer.bygroups(_token.Operator, _token.Whitespace, _token.Comment.Single),
             'import_value_escape'),
            _comment_pattern,
            _env_var_pattern,
            *((p[0], _token.Keyword) for p in _path_patterns[1:4]),  # everything except http and file.ext
            (r'(?<!\.)as(?!\.)', _token.Keyword),  # 'as' keyword for imports - only when standalone
            (_variable_names_pattern[0], _token.Name.Namespace),  # python module names are namespace
            (r'[.]', _token.Operator),  # namespace operator
            _jinja_block_pattern,
            _jinja_comment_pattern,
            _jinja_interpolate_pattern,
            (r'"', _token.String.Double, 'double_string_import_value'),
            (r"'", _token.String.Single, 'single_string_import_value'),
            (r'\s+', _token.Whitespace),
        ],
        f'import_value_escape': [
            (r'(#[^\n]*)(\s*\n)', _lexer.bygroups(_token.Comment.Single, _token.Whitespace)),
            (r'(\s|\n)+', _token.Whitespace),
            (r'(?<=[-\n\s.])', _token.Whitespace, 'import_value')
        ],
        f'import_plugins_value_escape': [
            (r'(#[^\n]*)(\s*\n)', _lexer.bygroups(_token.Comment.Single, _token.Whitespace)),
            (r'(\s|\n)+', _token.Whitespace),
            (r'(?<=[-\n\s.])', _token.Whitespace, 'import_plugins_value')
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
        **_create_string_continue('double_string_import_plugins_value', '"', 'import_plugins_value'),
        **_create_string_continue('single_string_import_plugins_value', "'", 'import_plugins_value'),
        **_create_string_continue('double_string_import_value', '"', 'import_value'),
        **_create_string_continue('single_string_import_value', "'", 'import_value'),
        'jinja_block': [
            # End of Jinja2 block statement
            (r'(\s*)(%\})', _lexer.bygroups(_token.Text, _token.String.Interpol), '#pop'),

            # Function calls
            _function_call_pattern,

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

            # Function calls
            _function_call_pattern,

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
