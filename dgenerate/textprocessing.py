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
import collections.abc
import datetime
import enum
import glob
import inspect
import itertools
import math
import os
import pathlib
import re
import shutil
import textwrap
import typing

import packaging.version

import dgenerate.files as _files
import dgenerate.memoize as _memoize
import dgenerate.types as _types

__doc__ = """
Text processing, console text rendering, and parsing utilities. URI parser, and reusable tokenization.
"""


class ConceptUriParseError(Exception):
    """
    Raised by :py:meth:`.ConceptUriParser.parse` on parsing errors.
    """
    pass


class ConceptUri:
    """
    Represents a parsed concept URI.
    """

    concept: str
    """
    The primary concept mentioned in the URI.
    """

    args: dict[str, str | list[str]]
    """
    Provided keyword arguments with their (string) values.
    """

    def __init__(self, concept: str, args: dict[str, str]):
        self.concept = concept
        self.args = args

    def __str__(self):
        return f"{self.concept}: {self.args}"

    def copy(self):
        return ConceptUri(self.concept, self.args.copy())


class TokenizedSplitSyntaxError(Exception):
    """
    Raised by :py:func:`tokenized_split` on syntax errors.
    """
    pass


class ShellParseSyntaxError(Exception):
    """
    Raised by :py:func:`shell_parse` on syntax errors.
    """
    pass


def expand_escape_code(code: str) -> str:
    """
    Expand a single escape code character.

    For example: ``expand_escape_code('n')`` will return ``'\n'``.

    :param code: The escape character code.
    :return: Expanded character.
    """
    if code == '\\':
        return '\\'
    return fr'\{code}'.encode('utf-8').decode('unicode_escape')


def is_escape_code(code: str) -> bool:
    """
    Does a character represent an escape code?

    :param code: The character.
    :return: ``True`` or ``False``.
    """
    return code in {'n', 'r', 't', 'b', 'f', '\\'}


def has_unescaped_quotes(string, double: bool = True, single: bool = True) -> bool:
    """
    Does a string contain unescaped quotes?


    :param string: The string
    :param double: Detect double quotes?
    :param single: Detect single quotes?

    :return: ``True`` or ``False``.
    """
    quote_pattern = ''
    if double:
        quote_pattern += '"'
    if single:
        quote_pattern += "'"

    if not quote_pattern:
        raise ValueError('At least one of double or single quotes must be enabled.')

    return bool(re.search(rf'(?<!\\)(?:\\\\)*[{quote_pattern}]', string))


def tokenized_split(string: str,
                    separator: str | None,
                    remove_quotes: bool = False,
                    strict: bool = False,
                    escapes_in_unquoted: bool = False,
                    escapes_in_unquoted_is_escape: typing.Callable[[str], bool] = is_escape_code,
                    escapes_in_unquoted_handler: typing.Callable[[str], str] = expand_escape_code,
                    escapes_in_quoted: bool = False,
                    escapes_in_quoted_is_escape: typing.Callable[[str], bool] = is_escape_code,
                    escapes_in_quoted_handler: typing.Callable[[str], str] = expand_escape_code,
                    single_quotes_raw: bool = False,
                    double_quotes_raw: bool = False,
                    process_string_token: typing.Callable[[str], str] = None,
                    process_intermixed_token: typing.Callable[[str], str] = None,
                    string_expander: typing.Callable[[str, str], str] = None,
                    text_expander: typing.Callable[[str], str | list[str]] = None,
                    remove_stray_separators: bool = False,
                    escapable_separator: bool = False,
                    allow_unterminated_strings: bool = False,
                    first_string_halts: bool = False) -> list[str]:
    """
    Split a string by a separator and discard whitespace around tokens, avoid
    splitting within single or double-quoted strings. Empty fields may be used.

    Quotes can be always be escaped with a backslash to avoid the creation of a
    string type token. The backslash will remain in the output if ``escapes_in_unquoted``
    or ``escapes_in_quoted`` are ``False`` and the escape occurs in the relevant context.


    :raise TokenizedSplitSyntaxError: on syntax errors.

    :param string: the string
    :param separator: separator
    :param remove_quotes: remove quotes from quoted string tokens?

    :param strict: Text tokens cannot be intermixed with quoted strings? disallow IE: ``"text'string'text"``

    :param escapes_in_unquoted: evaluate escape sequences in text tokens (unquoted strings)?
        The slash is retained by default when escaping quotes, this disables that, and also
        enables handling of the escapes ``n, r, t, b, f, and \\``. IE: given ``separator =";"`` parse
        ``\\"token\\"; "a b"`` -> ``['"token"', 'a b']``, instead of ``\\"token\\"; "a b"``-> ``['\\"token\\"', 'a b']``

    :param escapes_in_unquoted_is_escape: Determine if a character should be expanded with
        ``escapes_in_unquoted_handler`` when ``escapes_in_unquoted`` is ``True``.

    :param escapes_in_unquoted_handler: Escape character handler for unquoted tokens when
        ``escapes_in_unquoted`` is ``True``, defaults to ``expand_escape_code``.

    :param escapes_in_quoted: evaluate escape sequences in quoted string tokens?
        The slash is retained by default when escaping quotes, this disables that, and also
        enables handling of the escapes ``n, r, t, b, f, and \\``. IE given ``separator = ";"`` parse
        ``token; "a \\" b"`` -> ``['token', 'a " b']``, instead of ``token; "a \\" b"``-> ``['token', 'a \\" b']``

    :param escapes_in_quoted_is_escape: Determine if a character should be expanded with
        ``escapes_in_quoted_handler`` when ``escapes_in_quoted`` is ``True``.

    :param escapes_in_quoted_handler: Escape character handler for quoted strings when
        ``escapes_in_quoted`` is ``True``, defaults to ``expand_escape_code``.

    :param single_quotes_raw: Never evaluate escape sequences in single-quoted strings?
    :param double_quotes_raw: Never evaluate escape sequences in double-quoted strings?

    :param process_string_token: post process standalone string tokens.
        The function should take a string and return a string.

    :param process_intermixed_token: This can be used to run a custom process on intermixed tokens, which
        consist of text intermixed with quoted strings. The function should take a string and return a string.

    :param string_expander: User post process string expansion hook ``string_expander(quote_char, string) -> str``
    :param text_expander: User post process text token expansion hook ``text_expander(text_token) -> str | list[str]``.

        should return a list of new tokens (indicates globbing) or a single token string (indicates no globbing).
    :param remove_stray_separators: Remove consecutive seperator characters with no inner content at the end of the string?
        In effect, do not create entries for empty separators at the end of a string.
    :param escapable_separator: The seperator character may be escaped with a backslash where it would otherwise cause a split?

    :param allow_unterminated_strings: Allows the lex to end on an unterminated string without a syntax error being produced.
        It is necessary to perform lookahead N to determine if a seperator is quoted by a string or not, this allows your input
        to end with an unterminated string and still split correctly, complete strings proceeding the unterminated string which
        contain the seperator character will not be split on the seperator because the seperator is considered quoted in a string token.

    :param first_string_halts: The first completed string token halts lexing immediately, this is mainly used by the lexer internally
        for recursion in cases where a lookahead for string termination is required, but may be useful for some external parsing tasks.

    :return: parsed fields
    """

    if remove_quotes and process_string_token:
        raise ValueError(
            'cannot use "remove_quotes" and "process_string_token" together, incompatible.')

    if remove_quotes and process_intermixed_token:
        raise ValueError(
            'cannot use "remove_quotes" and "process_intermixed_token" together, incompatible.')

    if process_string_token and strict:
        raise ValueError(
            'cannot use "process_string_token" and "strict" together, incompatible.')

    if process_intermixed_token and strict:
        raise ValueError(
            'cannot use "process_intermixed_token" and "strict" together, incompatible.')

    if string_expander is None:
        def string_expander(q, s):
            return s

    if text_expander is None:
        def text_expander(s):
            return s

    class _States(enum.Enum):
        AWAIT_TEXT = 0
        TEXT_TOKEN = 1
        TEXT_TOKEN_STRICT = 2
        TEXT_ESCAPE = 3
        STRING = 4
        STRING_ESCAPE = 5
        SEP_REQUIRED = 6
        EOL = 7

    state = _States.AWAIT_TEXT
    last_state = state

    def state_change(new_state):
        nonlocal state, last_state
        last_state = state
        state = new_state

    # tokens out
    parts = []

    # quoted string token accumulator
    cur_string = ''

    # the quote character that initiated
    # the current string token being accumulated
    cur_quote = ''

    # accepted quote characters
    QUOTE_CHARS = {'"', "'"}

    back_expand = 0

    def append_text(text_to_append):
        # append text to the last token
        if back_expand > 0:
            # might be appending to multiple tokens
            # due to globbing, stack has content
            for i in range(back_expand):
                offset = -(i + 1)
                if isinstance(parts[offset], tuple):
                    # custom token processor requested
                    # for the appended text, run it
                    parts[offset] = parts[offset][0] + parts[offset][1](text_to_append)
                else:
                    parts[offset] += text_to_append
        else:
            # normal append, token character
            if parts:
                parts[-1] += text_to_append
            else:
                # nothing on the stack
                parts.append(text_to_append)

    def post_process_token(token):
        if process_string_token and len(token) > 1 and \
                ((token[0] == "'" and token[-1] == "'") or
                 (token[0] == '"' and token[-1] == '"')):
            if has_unescaped_quotes(
                    token[1:-1],
                    double=(token[0] == '"'),
                    single=(token[0] == "'")
            ):
                if process_intermixed_token:
                    return process_intermixed_token(token)
                else:
                    return token
            else:
                return process_string_token(token)
        elif process_intermixed_token and has_unescaped_quotes(token):
            return process_intermixed_token(token)
        return token

    def process_back_expansion_string(token):
        return process_string_token(string_expander(token[0], token))

    def separate_here(cur_idx):
        nonlocal parts
        if last_state != _States.STRING:
            expanded_token = text_expander(parts[-1].rstrip())
            if isinstance(expanded_token, str):
                parts[-1] = expanded_token
            elif len(expanded_token) > 0:
                parts[-1] = expanded_token[0]
                parts += expanded_token[1:]
        else:
            parts[-1] = post_process_token(parts[-1].rstrip())

        if remove_stray_separators and not string[cur_idx:].strip(separator):
            state_change(_States.EOL)
        else:
            parts.append('')
            state_change(_States.AWAIT_TEXT)

    # returned to None upon encountering
    # a string termination condition during normal
    # lexing, when a lookahead is required to determine
    # if the current string token completes in the future
    # the lexer can reuse that information while still
    # in that string token
    string_lookahead_terminated_memoize = None

    def string_lookahead_is_terminated(cur_idx):
        # determine if the string is terminated or not early on using lookahead
        nonlocal string_lookahead_terminated_memoize

        if string_lookahead_terminated_memoize is not None:
            return string_lookahead_terminated_memoize

        segment = cur_quote + string[cur_idx: len(string)]
        # try to test if where we are at currently is
        # inside a terminated string

        try:
            tokenized_split(
                segment,
                separator=separator,
                strict=strict,
                escapes_in_unquoted=escapes_in_unquoted,
                escapes_in_unquoted_is_escape=escapes_in_unquoted_is_escape,
                escapes_in_unquoted_handler=escapes_in_unquoted_handler,
                escapes_in_quoted=escapes_in_quoted,
                escapes_in_quoted_is_escape=escapes_in_quoted_is_escape,
                escapes_in_quoted_handler=escapes_in_quoted_handler,
                escapable_separator=escapable_separator,
                single_quotes_raw=single_quotes_raw,
                double_quotes_raw=double_quotes_raw,
                remove_stray_separators=remove_stray_separators,
                first_string_halts=True
            )
            # syntactically valid string from this point on
            string_lookahead_terminated_memoize = True
            return True
        except TokenizedSplitSyntaxError:
            # syntactically invalid string from this point on
            string_lookahead_terminated_memoize = False
            return False

    def syntax_error(msg, cur_idx):
        # create syntax error
        return TokenizedSplitSyntaxError(f'{msg}: \'{string[:cur_idx]}[ERROR HERE>]{string[cur_idx:]}\'')

    for idx, c in enumerate(string):
        if state == _States.EOL:
            break

        if state == _States.STRING:
            # inside a quoted string

            if c == separator and allow_unterminated_strings:
                # unescaped seperator, need to lookahead N to
                # resolve this the result is memoized for the context
                # of the current string token

                terminated = string_lookahead_is_terminated(idx)

                if not terminated:
                    # the seperator is not quoted by a complete
                    # string and therefore separates
                    append_text(cur_string)
                    separate_here(idx)
                else:
                    cur_string += c
            elif c == '\\':

                # encountered an escape sequence start
                state_change(_States.STRING_ESCAPE)
                if not escapes_in_quoted or \
                        (single_quotes_raw and cur_quote == "'") or \
                        (double_quotes_raw and cur_quote == '"'):
                    cur_string += c
            elif c == cur_quote:
                # encountered the terminator quote
                append_text(
                    string_expander(
                        cur_quote,
                        cur_string + (c if not remove_quotes else '')
                    )
                )
                cur_string = ''

                if idx == len(string) - 1:
                    # safe to stop here entirely
                    state_change(_States.EOL)
                else:
                    # Strict mode requires a separator after a quoted string token
                    state_change(_States.SEP_REQUIRED if strict else _States.TEXT_TOKEN)

                # we finished a string token, any memoized result
                # of lookahead N for string termination is invalid now
                string_lookahead_terminated_memoize = None

                if first_string_halts:
                    break
            else:
                # append to current string
                cur_string += c
        elif state == _States.TEXT_ESCAPE:
            # after encountering an escape sequence start inside of a text token

            if c == separator:
                if not escapable_separator:
                    if escapes_in_unquoted:
                        append_text('\\')
                    separate_here(idx)
                    continue

            state_change(_States.TEXT_TOKEN_STRICT if strict else _States.TEXT_TOKEN)
            # return to the appropriate state

            if c in QUOTE_CHARS:
                # this is an escaped quotation character
                append_text(c)
            elif escapes_in_unquoted_is_escape(c):
                # this is a character that translates into
                # an escape code that we have decided to support
                if escapes_in_unquoted:
                    append_text(escapes_in_unquoted_handler(c))
                else:
                    append_text(c)
            elif c == separator and escapable_separator:
                # unknown escape code case 1
                append_text(c)
            elif escapes_in_unquoted:
                # unknown escape code case 2
                append_text(fr'\{c}')
            else:
                # unknown escape code case 3
                append_text(c)
        elif state == _States.STRING_ESCAPE:
            # after encountering an escape sequence start inside a quoted string
            state_change(_States.STRING)
            # return to string state

            if c in QUOTE_CHARS:
                # this is an escaped quotation character
                cur_string += c
            elif escapes_in_quoted_is_escape(c):
                # this is a character that translates into an
                # escape code that we have decided to support
                if escapes_in_quoted and not \
                        ((single_quotes_raw and cur_quote == "'") or
                         (double_quotes_raw and cur_quote == '"')):
                    cur_string += escapes_in_quoted_handler(c)
                else:
                    cur_string += c

            elif escapes_in_quoted:
                # unknown escape code case 1
                if escapable_separator and allow_unterminated_strings and c == separator:
                    terminated = string_lookahead_is_terminated(idx)
                    if not terminated:
                        cur_string += c
                    else:
                        cur_string += fr'\{c}'
                else:
                    cur_string += fr'\{c}'
            else:
                # unknown escape code case 2
                if escapable_separator and allow_unterminated_strings and c == separator:
                    terminated = string_lookahead_is_terminated(idx)
                    if not terminated:
                        cur_string = cur_string.removesuffix('\\') + c
                    else:
                        cur_string += c
                else:
                    cur_string += c

        elif state == _States.SEP_REQUIRED:
            # This state is only reached in strict mode
            # where separators are required after a string token
            if c == separator:
                state_change(_States.AWAIT_TEXT)
                parts.append('')
            elif not c.isspace():
                raise syntax_error('Missing separator after string', idx)
        elif state == _States.AWAIT_TEXT:
            if c == '\\':
                # started an escape sequence
                state_change(_States.TEXT_ESCAPE)
                if not escapes_in_unquoted:
                    append_text(c)
            elif c in QUOTE_CHARS:
                # started a string token
                cur_quote = c
                cur_string += (c if not remove_quotes else '')
                state_change(_States.STRING)
            elif c.isspace():
                # ignore space until token starts
                pass
            elif c == separator:
                # started the string with a separator
                if not parts:
                    parts += ['', '']
                else:
                    parts.append('')
            else:
                # started a text token
                state_change(_States.TEXT_TOKEN_STRICT if strict else _States.TEXT_TOKEN)
                append_text(c)
        elif state == _States.TEXT_TOKEN:
            # this is the non-strict mode parsing state inside a text token
            if c == '\\':
                # started an escape sequence in a text token

                state_change(_States.TEXT_ESCAPE)

                next_char_sep = (len(string) > idx + 1 and string[idx + 1] == separator)

                if escapable_separator:
                    if not next_char_sep:
                        if not escapes_in_unquoted:
                            append_text(c)
                elif not escapes_in_unquoted:
                    append_text(c)

            elif c in QUOTE_CHARS:
                # started a string token intermixed with a text token
                cur_quote = c
                cur_string += (c if not remove_quotes else '')
                state_change(_States.STRING)

                expanded = text_expander(parts[-1])

                if isinstance(expanded, str):
                    # if a string, just set the token
                    # to it, no glob involved
                    parts[-1] = expanded
                else:
                    back_expand = len(expanded)
                    if back_expand > 0:
                        # this is some kind of glob returning
                        # a non-zero amount of names, run custom
                        # post processes any appended text (back expansion syntax)

                        if remove_quotes:
                            # already removes them
                            parts[-1] = expanded[0]
                            parts += expanded[1:]
                        else:
                            parts[-1] = (expanded[0], process_back_expansion_string)
                            parts += ((e, process_back_expansion_string) for e in expanded[1:])

            elif c == separator:
                # encountered a separator
                # the last element needs to be right stripped
                # because spaces are allowed inside a text token
                # and there is no way to differentiate 'inside' and
                # 'outside' without lookahead, or until there occurs
                # a separator
                back_expand = 0
                separate_here(idx)
            else:
                # append text token character
                append_text(c)
        elif state == _States.TEXT_TOKEN_STRICT:
            # This state is only reached in strict mode
            if c == '\\':
                # encountered an escape sequence in a text token
                state_change(_States.TEXT_ESCAPE)

                next_char_sep = (len(string) > idx + 1 and string[idx + 1] == separator)

                if escapable_separator:
                    if not next_char_sep:
                        if not escapes_in_unquoted:
                            append_text(c)
                elif not escapes_in_unquoted:
                    append_text(c)

            elif c in QUOTE_CHARS:
                # cannot have a string intermixed with a text token in strict mode
                raise syntax_error('Cannot intermix quoted strings and text tokens', idx)
            elif c == separator:
                # encountered a separator
                # the last element needs to be right stripped
                # because spaces are allowed inside a text token
                # and there is no way to differentiate 'inside' and
                # 'outside' without lookahead, or until there occurs
                # a separator
                separate_here(idx)
            else:
                # append text token character
                append_text(c)

    if state == _States.STRING or state == _States.STRING_ESCAPE:
        # state machine ended inside a quoted string
        if not allow_unterminated_strings:
            raise syntax_error(f'un-terminated string: \'{cur_string}\'', len(string))
        else:
            if parts:
                parts[-1] = string_expander(cur_quote, parts[-1] + cur_string)
            else:
                parts = [string_expander(cur_quote, cur_string)]

    if state == _States.TEXT_TOKEN_STRICT or state == _States.TEXT_TOKEN:
        # if we end on a text token, right strip the text token, as it is
        # considered 'outside' the token, and spaces are only allowed 'inside'
        # and this is ambiguous without lookahead
        expanded = text_expander(parts[-1].rstrip())

        if isinstance(expanded, str):
            parts[-1] = expanded
        elif len(expanded) > 0:
            parts[-1] = expanded[0]
            parts += expanded[1:]

    if state == _States.TEXT_ESCAPE and escapes_in_unquoted:
        # incomplete escapes are not allowed in text tokens
        raise syntax_error(f'un-finished escape sequence: \'{cur_string}\'', len(string))

    if state == _States.EOL and back_expand == 0:
        parts[-1] = post_process_token(parts[-1])

    if remove_stray_separators:
        while parts and parts[-1] == "":
            parts.pop()
        return parts

    return parts


class UnquoteSyntaxError(Exception):
    """
    Raised by :py:func:`.unquote` on parsing errors.
    """
    pass


def unquote(string: str,
            escapes_in_unquoted: bool = False,
            escapes_in_unquoted_is_escape: typing.Callable[[str], bool] = is_escape_code,
            escapes_in_unquoted_handler: typing.Callable[[str], str] = expand_escape_code,
            escapes_in_quoted: bool = False,
            escapes_in_quoted_is_escape: typing.Callable[[str], bool] = is_escape_code,
            escapes_in_quoted_handler: typing.Callable[[str], str] = expand_escape_code,
            single_quotes_raw: bool = False,
            double_quotes_raw: bool = False,
            strict=True,
            ) -> str:
    """
    Remove quotes from a string, including single quotes.

    Unquoted strings will have leading and trailing whitespace stripped.

    Quoted strings will have leading and trailing whitespace stripped up to where the quotes were.

    :param escapes_in_unquoted: evaluate escape sequences in text tokens (unquoted strings)?
        The slash is retained by default when escaping quotes, this disables that, and also
        enables handling of the escapes ``n, r, t, b, f, and \\``.

    :param escapes_in_unquoted_is_escape: Determine if a character should be expanded with
        ``escapes_in_unquoted_handler`` when ``escapes_in_unquoted`` is ``True``.

    :param escapes_in_unquoted_handler: Escape character handler for unquoted tokens when
        ``escapes_in_unquoted`` is ``True``, defaults to ``expand_escape_code``.

    :param escapes_in_quoted: evaluate escape sequences in quoted string tokens?
        The slash is retained by default when escaping quotes, this disables that, and also
        enables handling of the escapes ``n, r, t, b, f, and \\``.

    :param escapes_in_quoted_is_escape: Determine if a character should be expanded with
        ``escapes_in_quoted_handler`` when ``escapes_in_quoted`` is ``True``.

    :param escapes_in_quoted_handler: Escape character handler for quoted strings when
        ``escapes_in_quoted`` is ``True``, defaults to ``expand_escape_code``.

    :param strict: Disallow intermixed tokens?

    :param escapes_in_unquoted: Render escape sequences in strings that are unquoted?
    :param escapes_in_quoted: Render escape sequences in strings that are quoted?
    :param single_quotes_raw: Never evaluate escape sequences in single-quoted strings?
    :param double_quotes_raw: Never evaluate escape sequences in double-quoted strings? T
    :param string: the string
    :return: The un-quoted string
    """
    try:
        val = tokenized_split(
            string,
            escapes_in_quoted=escapes_in_quoted,
            escapes_in_quoted_is_escape=escapes_in_quoted_is_escape,
            escapes_in_quoted_handler=escapes_in_quoted_handler,
            escapes_in_unquoted=escapes_in_unquoted,
            escapes_in_unquoted_is_escape=escapes_in_unquoted_is_escape,
            escapes_in_unquoted_handler=escapes_in_unquoted_handler,
            single_quotes_raw=single_quotes_raw,
            double_quotes_raw=double_quotes_raw,
            strict=strict,
            remove_quotes=True,
            separator=None)
        if val:
            return val[0]
        return ''
    except TokenizedSplitSyntaxError as e:
        if 'Missing separator' in str(e):
            raise UnquoteSyntaxError(str(e).replace('Missing separator', 'Extraneous text')) from e
        raise UnquoteSyntaxError(e) from e


def shell_expandvars(string: str) -> str:
    """
    Expand shell variables of form ``$var``, ``${var}``, ``%var%`` in a string.

    Escaped characters ``\\$`` and ``\\%`` are ignored.

    Unknown variables expand to an empty string.

    Supported formats:
    - Unix-style: $VAR or ${VAR}
    - Windows-style: %VAR%

    :param string: Input string containing variables to expand.
    :return: String with expanded variables.
    """

    def replace_var(match):
        var = match.group('unix') or match.group('braced') or match.group('windows')
        return os.environ.get(var, '')

    re_var = re.compile(
        r'(?<!\\)(?:\$(?P<unix>\w+)|\$\{(?P<braced>\w+)\}|%(?P<windows>\w+)%)'
    )
    result = re_var.sub(replace_var, string)

    return result


def shell_quote(
        string: str,
        double: bool = False,
        quotes: bool = True,
        strict: bool = False,
):
    """
    Shell quote a string, compatible with dgenerate config shell syntax.

    :param string: The input string.
    :param double: Intended to be a double-quoted string?
    :param quotes: Add quotes? if ``False`` only add the proper escape sequences and no surrounding quotes.
    :param strict: Setting this to true disallows text from being intermixed next to complete strings,
                   for example: ``test'string'`` would be fully quoted, even though the shell can normally
                   parse this as a single token. URI arguments do not allow intermixed strings,
                   so this is useful for quoting URI argument values.
    :return: The quoted string.
    """
    try:
        result = tokenized_split(
            string, ' ', strict=strict, remove_stray_separators=True
        )
        needs_quoting = len(result) > 1
    except TokenizedSplitSyntaxError:
        needs_quoting = True

    if not needs_quoting:
        return string

    if not double:
        processed = re.sub(r"(?<!\\)'", r"\'", string)
        return f"'{processed}'" if quotes else processed

    else:
        processed = re.sub(r'(?<!\\)"', r'\"', string)
        return f'"{processed}"' if quotes else processed


def shell_parse(
        string,
        expand_home: bool = True,
        expand_vars: bool = True,
        expand_glob: bool = True,
        expand_vars_func: typing.Callable[[str], str] = shell_expandvars,
        glob_hidden: bool = False,
        glob_recursive: bool = False
) -> list[str]:
    """
    Shell command line parsing, implements basic home directory expansion, globbing, and
    environmental variable expansion.

    Globbing and home directory expansion do not occur inside strings.

    The shell syntax implemented is compatible with the dgenerate config shell syntax,
    and is unique to dgenerate.

    Only stand-alone string arguments undergo POSIX like quote removal and escape
    code evaluation. The characters: $, %, \\, ', and ", are always resolved in double-quote
    strings when they are escaped with \\, meaning that the backslash used to escape them will
    be removed. In single quoted strings they are not resolved, i.e. the backslash will remain
    in the string. This process occurs only for string tokens surrounded by white space.

    Strings that are intermixed with other tokens, for example: plugin;argument="test"
    are not processed at all, the intermixed tokens are lexed as into a single token
    with no quote removal or escape code expansion. This allows for internal
    parsing on complex arguments such as URI values to occur without the loss
    of quoting and escaping information. dgenerate URIs implement custom
    escaping and quoting rules.

    .. code-block:: python

        # basic glob
        shell_parse('command *.png')

        # recursive glob
        shell_parse('command dir/**/*.png')

        # home directory
        shell_parse('command ~')

        # home directory of user test
        shell_parse('command ~test')

        # everything under home directory
        shell_parse('command ~/*')

        # append text to every glob result (back expansion)
        # the quotes are always removed from
        # the outside of the appended text
        shell_parse('command *".png"') # -> ['command', 'file.png', 'file2.png', ...]

        # append text to every glob result (back expansion)
        # the quotes are always removed from
        # the outside of the appended text
        shell_parse("command *'.png'") # -> ['command', 'file.png', 'file2.png', ...]

        # environmental variable syntax 1
        shell_parse('command $ENVVAR')

        # environmental variable syntax 2
        shell_parse('command %ENVVAR%')


    :param string: String to parse
    :param expand_home: Expand ``~`` ?
    :param expand_vars: Expand unix style ``$`` and windows style ``%`` environmental variables?
    :param expand_glob: Expand ``*`` glob expressions including recursive globs?
    :param expand_vars_func: This function is used to expand shell variables in a string,
        analogous to `os.path.expandvars`
    :param glob_hidden: Should globs include hidden directories?
    :param glob_recursive: Should globs be recursive?
    :return: shell arguments
    """

    def text_token_processor(token):
        # the characters above simply resolve to their
        # value, i.e. the backslash is just removed

        return tokenized_split(
            token,
            None,
            escapes_in_unquoted=True,
            escapes_in_unquoted_is_escape=is_escape,
            escapes_in_unquoted_handler=lambda x: x
        )[0]

    def text_expander(token):

        if expand_home and '~' in token:
            token = pathlib.Path(os.path.expanduser(token)).as_posix()

        if expand_vars and ('$' in token or '%' in token):
            token = expand_vars_func(token)

        if expand_glob and '*' in token:

            globs = [
                pathlib.Path(p).as_posix() for p in glob.glob(
                    token,
                    include_hidden=glob_hidden,
                    recursive=glob_recursive
                )
            ]

            if len(globs) == 0:
                raise ShellParseSyntaxError(
                    f'glob expression "{token}" returned zero files.')
            return globs

        # if no globs, run standard processing
        return text_token_processor(token) if token else token

    def string_expander(q, s):
        if q == '"':
            if expand_vars and ('$' in s or '%' in s):
                return expand_vars_func(s)
            return s
        return s

    # other than resolving \" and \' automatically
    # we need to resolve the special characters
    # used for env-vars that can be escaped
    def is_escape(s):
        return s in {'\\', '%', '$'}

    def string_token_processor(token):
        # the characters above simply resolve to their
        # value, i.e. the backslash is just removed

        return unquote(
            token,
            escapes_in_quoted=True,
            escapes_in_quoted_is_escape=is_escape,
            escapes_in_quoted_handler=lambda x: x,
        )

    try:
        opts = dict()

        return tokenized_split(
            string, ' ',
            strict=False,
            text_expander=text_expander,
            string_expander=string_expander,
            process_string_token=string_token_processor,
            remove_stray_separators=True,
            **opts
        )

    except TokenizedSplitSyntaxError as e:
        raise ShellParseSyntaxError(e) from e


_concept_uri_parse_cache = dict()


class ConceptUriParser:
    """
    Parser for dgenerate concept paths with arguments, IE: ``concept;arg1="a";arg2="b"``

    Used for ``--vae``, ``--loras`` etc. as well as image processor plugin module arguments.
    """

    concept_name: _types.Name
    """
    Name / title string for this concept. Used in parse error exceptions.
    """

    known_args: set[str]
    """
    Unique recognized keyword arguments
    """

    args_raw: bool | set[str] | None
    """
    ``True`` indicates all argument values are returned without any unquoting or processing into lists.
    
    ``None`` or ``False`` indicates no argument values skip extended processing.
    
    Assigning a set containing argument names indicates only the specified 
    arguments skip extended processing (unquoting or splitting).
    """

    args_lists: bool | set[str] | None
    """
    ``True`` indicates all arguments can accept a comma separated list.
    
    ``None`` or ``False`` indicates no arguments can accept a comma separated list.
    
    Assigning a set containing argument names indicates only the specified 
    arguments can accept a comma separated list.
    
    When an argument is parsed as a comma separated list, its value/type
    in :py:attr:`ConceptUri.args` will be that of a list.
    """

    delimiter: str
    """
    URI argument delimiter, the default is semicolon.
    """

    def __init__(self,
                 concept_name: _types.Name,
                 known_args: collections.abc.Iterable[str],
                 args_lists: bool | collections.abc.Iterable[str] | None = None,
                 args_raw: bool | collections.abc.Iterable[str] | None = None,
                 delimiter: str = ';'):
        """
        :raises ValueError: if duplicate argument names are specified.

        :param concept_name: Concept name, used in error messages
        :param known_args: valid arguments for the parser, must be unique
        """

        check = set()
        for arg in known_args:
            if arg in check:
                raise ValueError(f'duplicate known_args specification {arg}')
            check.add(arg)
        self.known_args = check

        if args_lists is not None and not isinstance(args_lists, bool):
            check = set()
            for arg in args_lists:
                if arg in check:
                    raise ValueError(f'duplicate args_lists specification {arg}')
                check.add(arg)
            self.args_lists = check
        else:
            self.args_lists = None

        if args_raw is not None and not isinstance(args_raw, bool):
            check = set()
            for arg in args_raw:
                if arg in check:
                    raise ValueError(f'duplicate args_raw specification {arg}')
                check.add(arg)
            self.args_raw = check
        else:
            self.args_raw = None

        self.concept_name = concept_name
        self.delimiter = delimiter

    def parse(self, uri: _types.Uri) -> ConceptUri:
        """
        Parse a string.

        :param uri: the string

        :raise ConceptUriParseError: on parsing errors
        :raise ValueError: if uri is ``None``

        :return: :py:class:`.ConceptPath`
        """
        val = self._parse(uri).copy()

        if len(_concept_uri_parse_cache) > 128:
            _concept_uri_parse_cache.clear()

        return val

    @_memoize.memoize(cache=_concept_uri_parse_cache)
    def _parse(self, uri: _types.Uri) -> ConceptUri:

        args = dict()

        if uri is None:
            raise ValueError('uri must not be None')

        if not uri.strip():
            raise ConceptUriParseError(f'Error parsing {self.concept_name} URI, URI was empty.')

        try:
            parts = tokenized_split(uri, self.delimiter)
        except TokenizedSplitSyntaxError as e:
            raise ConceptUriParseError(
                f'Error parsing {self.concept_name} URI "{uri}": {str(e).strip()}')

        parts = iter(parts)
        concept = parts.__next__()

        if is_quoted(concept):
            try:
                concept = unquote(concept, escapes_in_quoted=True)
            except UnquoteSyntaxError as e:
                raise ConceptUriParseError(
                    f'Error parsing {self.concept_name} concept name in URI "{uri}": {str(e).strip()}') from e

        for i in parts:
            vals = i.split('=', 1)
            if not vals or not vals[0]:
                raise ConceptUriParseError(f'Error parsing path arguments for '
                                           f'{self.concept_name} concept "{concept}", Empty argument space, '
                                           f'stray "{self.delimiter}" delimiter?')
            name = vals[0].strip()
            if self.known_args is not None and name not in self.known_args:
                raise ConceptUriParseError(
                    f'Unknown path argument "{name}" for {self.concept_name} concept "{concept}", '
                    f'valid arguments: {", ".join(sorted(self.known_args))}')

            if len(vals) == 1:
                raise ConceptUriParseError(f'Error parsing path arguments for '
                                           f'{self.concept_name} concept "{concept}", missing value '
                                           f'assignment for argument {vals[0]}.')

            if name in args:
                raise ConceptUriParseError(
                    f'Duplicate argument "{name}" provided for {self.concept_name} concept "{concept}".')

            try:
                if self.args_raw is True or (self.args_raw is not None and name in self.args_raw):
                    args[name] = vals[1]
                elif self.args_lists is True or (self.args_lists is not None and name in self.args_lists):
                    vals = tokenized_split(
                        vals[1], ',',
                        remove_quotes=True,
                        strict=True,
                        escapes_in_quoted=True
                    )
                    if len(vals) > 1:
                        args[name] = vals
                    elif len(vals) == 1 and vals[0]:
                        args[name] = vals[0]
                    else:
                        raise ConceptUriParseError(
                            f'Syntax error parsing argument "{name}" for '
                            f'{self.concept_name} concept "{concept}", missing assignment value.')
                else:
                    if not vals[1].strip():
                        raise ConceptUriParseError(
                            f'Syntax error parsing argument "{name}" for '
                            f'{self.concept_name} concept "{concept}", missing assignment value.')

                    args[name] = unquote(
                        vals[1],
                        escapes_in_quoted=True
                    )

            except (TokenizedSplitSyntaxError, UnquoteSyntaxError) as e:
                raise ConceptUriParseError(
                    f'Syntax error parsing argument "{name}" for '
                    f'{self.concept_name} concept "{concept}": {str(e).strip()}')
        return ConceptUri(concept, args)


def oxford_comma(elements: collections.abc.Collection[str], conjunction: str) -> str:
    """
    Join a sequence of strings with commas, end with an oxford comma and conjunction if needed.

    :param elements: strings
    :param conjunction: "and", "or"
    :return: a joined string
    """
    cnt = len(elements)
    elements = (str(i) for i in elements)
    if cnt == 1:
        return next(elements)
    if cnt == 2:
        return next(elements) + f' {conjunction} ' + next(elements)
    output = ''
    for idx, element in enumerate(elements):
        if idx == cnt - 1:
            output += f', {conjunction} {element}'
        elif idx == 0:
            output += element
        else:
            output += f', {element}'
    return output


def long_text_wrap_width() -> int:
    """
    Return the current terminal width or the default value of 150 characters for text-wrapping purposes.

    This can be affected by the environmental variable ``COLUMNS``.

    :raise ValueError: if the environmental variable
        ``COLUMNS`` is not an integer value or is less than 0.

    :return: int
    """
    env_width = os.environ.get('COLUMNS', 150)
    try:
        env_width = int(env_width)
    except ValueError:
        raise ValueError(
            f'Invalid non-integer value "{env_width}" assigned to environmental '
            f'variable COLUMNS.')

    if env_width < 0:
        raise ValueError(
            f'Invalid integer value "{env_width}" assigned to environmental '
            f'variable COLUMNS. Must be greater than or equal to 0.')

    val = min(shutil.get_terminal_size(fallback=(env_width, 150))[0], env_width)
    if val == 0:
        # should not be able to happen, but it has, wonderful
        return env_width
    return val


def underline(string: str, underline_char: str = '=') -> str:
    """
    Underline a string with the selected character.

    :param string: the string
    :param underline_char: the character to underline with
    :return: the underlined string
    """
    return string + '\n' + (underline_char * min(len(max(string.split('\n'), key=len)), long_text_wrap_width()))


def quote(string: str, char='"') -> str:
    """
    Wrap a string with a quote character.

    Double quotes by default.

    This is not equivalent to shell quoting.


    :param string: the string
    :param char: the quote character to use
    :return: the quoted string
    """
    return f'{char}{string}{char}'


def is_quoted(string: str) -> bool:
    """
    Return ``True`` if a string is quoted with an identical starting and end quote.

    :param string: the string
    :return: ``True`` or ``False``
    """
    return (string.startswith('"') and string.endswith('"')) or (string.startswith("'") and string.endswith("'"))


def dashdown(string: str) -> str:
    """
    Replace '-' with '_'

    :param string: the string
    :return: modified string
    """
    return string.replace('-', '_')


def dashup(string: str) -> str:
    """
    Replace '_' with '-'

    :param string: the string
    :return: modified string
    """
    return string.replace('_', '-')


def contains_space(string: str) -> bool:
    """
    Check if a string contains any whitespace characters including newlines

    :param string: the string
    :return: bool
    """
    return any(c.isspace() for c in string)


def quote_spaces(
        value_or_struct: typing.Any | collections.abc.Iterable[typing.Any | list | tuple]) -> list | tuple | typing.Any:
    """
    Quote any ``str`` type values containing spaces, or ``str`` type values containing
    spaces within a list, or list of lists/tuples.

    The entire content of the data structure is stringified by this process.

    This is not equivalent to shell quoting.

    :param value_or_struct: value or (list of values, and or lists/tuples containing values)
    :return: input data structure with strings quoted if needed
    """

    if not isinstance(value_or_struct, (list, tuple)):
        return quote(str(value_or_struct)) if contains_space(str(value_or_struct)) else value_or_struct

    vals = []
    for v in value_or_struct:
        if isinstance(v, list):
            vals.append(quote_spaces(v))
            continue
        if isinstance(v, tuple):
            vals.append(tuple(quote_spaces(v)))
            continue

        vals.append(quote_spaces(v))
    return vals if isinstance(value_or_struct, list) else tuple(vals)


def indent_text(text,
                initial_indent: str | None = None,
                subsequent_indent: str | None = None):
    """
    Indent consecutive lines of text.

    :param text: Text to be indented
    :param initial_indent: String of characters to be used for the initial indentation
    :param subsequent_indent: String of characters to be used for the subsequent indentation
    :return: Indented text
    """

    if initial_indent is None:
        initial_indent = ''

    if subsequent_indent is None:
        subsequent_indent = ''

    lines = text.split('\n')
    indented_lines = [initial_indent + lines[0]] + [subsequent_indent + line for line in lines[1:]]
    return '\n'.join(indented_lines)


def _clean_lines(lines: list):
    l = []
    for line in lines:
        l.append(line.rstrip())
    return l


def wrap_paragraphs(text: str,
                    width: int,
                    break_long_words: bool = False,
                    break_on_hyphens: bool = False,
                    clean_lines: bool = True,
                    **fill_args):
    """
    Wrap text that may contain paragraphs without removing separating whitespace.

    The directive ``NOWRAP!`` can be used to start a paragraph block with no word wrapping,
    which is useful for manually formatting small blocks of text. ``NOWRAP!`` should exist on its
    own line, immediately followed by the block of text which will have wrapping disabled.
    The ``NOWRAP!`` directive line will not exist in the output text.

    :param text: Text containing paragraphs
    :param width: Wrap with in characters
    :param break_long_words: break on long words? default ``False``
    :param break_on_hyphens: break on hyphens? default ``False``
    :param clean_lines: Right strip lines?
    :param fill_args: extra keyword arguments to :py:func:`textwrap.fill` if desired
    :return: text wrapped string
    """
    paragraphs = text.split('\n\n')
    wrapped_text = ''

    for paragraph in paragraphs:
        # Check if the paragraph contains 'NOWRAP!' after leading whitespace
        if paragraph.lstrip().startswith('NOWRAP!'):
            # Add the paragraph to the wrapped text as is
            lines = paragraph.split('\n')
            # Find and remove the line containing NOWRAP!
            lines_without_nowrap = []
            for line in lines:
                if line.lstrip().startswith('NOWRAP!'):
                    continue
                lines_without_nowrap.append(line)
            wrapped_text += indent_text(
                '\n'.join(_clean_lines(lines_without_nowrap) if clean_lines else lines_without_nowrap),
                initial_indent=fill_args.get('initial_indent', None),
                subsequent_indent=fill_args.get('subsequent_indent', None)) + '\n\n'
        else:
            # Wrap the paragraph as before
            wrapped_paragraph = textwrap.fill(
                paragraph if not clean_lines else '\n'.join(_clean_lines(paragraph.split('\n'))), width=width,
                break_long_words=break_long_words,
                break_on_hyphens=break_on_hyphens,
                **fill_args
            )
            wrapped_text += wrapped_paragraph + '\n\n'

    return wrapped_text.rstrip()


def wrap(text: str,
         width: int,
         initial_indent='',
         subsequent_indent='',
         break_long_words=False,
         break_on_hyphens=False,
         **fill_args):
    """
    Wrap text.

    :param text: The prompt text
    :param width: The wrap width
    :param initial_indent: initial indent string
    :param subsequent_indent: subsequent indent string
    :param break_long_words: Break on long words?
    :param break_on_hyphens: Break on hyphens?
    :param fill_args: extra keyword arguments to :py:func:`textwrap.fill` if desired
    :return: text wrapped string
    """
    return textwrap.fill(
        text,
        width=width,
        break_on_hyphens=break_on_hyphens,
        break_long_words=break_long_words,
        initial_indent=initial_indent,
        subsequent_indent=subsequent_indent,
        **fill_args)


def format_size(size: int | collections.abc.Iterable[int]):
    """
    Join together an iterable of integers with the character x

    :param size: the iterable
    :return: formatted string
    """
    if isinstance(size, int):
        return str(size)

    return 'x'.join(str(a) for a in size)


def justify_left(string: str):
    """
    Justify text to the left.

    :param string: string with text
    :return: left justified text
    """
    return '\n'.join(line.strip() if not line.isspace() else line for line in string.split('\n'))


def parse_version(string: str) -> _types.Version:
    """
    Parse a SemVer version string into a tuple of 3 integers

    :param string: the version string

    :return: tuple of three ints
    """

    try:
        version = packaging.version.Version(string)
    except packaging.version.InvalidVersion:
        raise ValueError(
            f'version expected to be a version string in the format '
            f'major.minor.patch. received: "{string}"')

    return version.major, version.minor, version.micro


def parse_dimensions(string):
    """
    Parse a dimensions tuple from a string, integers seperated by the character 'x'

    :param string: the string

    :raises ValueError: On non integer dimension values.

    :return: a tuple representing the dimensions
    """
    try:
        return tuple(int(s.strip()) for s in string.lower().split('x'))
    except ValueError as e:
        raise ValueError('Dimensions must consist of integer values.') from e


def parse_image_size(string):
    """
    Parse an image size tuple from a string, 2 integers seperated by the character 'x', or a single
    integer specifying both dimensions.


    :param string: the string

    :raises ValueError: On non integer dimension values, or if more than 2 dimensions are provided,
        or if the product of the dimensions is 0.

    :return: a tuple representing the dimensions
    """
    dimensions = parse_dimensions(string)

    if len(dimensions) > 2:
        raise ValueError('An image size cannot possess over 2 dimensions.')

    if len(dimensions) == 0:
        raise ValueError('No image size dimension values were present.')

    if math.prod(dimensions) == 0:
        raise ValueError('The product of an image sizes dimensions cannot be 0.')

    if len(dimensions) == 1:
        return dimensions[0], dimensions[0]

    return dimensions


class BasicMaskShape(enum.Enum):
    """Represents a basic mask shape"""
    RECTANGLE = 0
    ELLIPSE = 1


def parse_basic_mask_shape(string: str) -> BasicMaskShape:
    """
    Parse a basic mask shape from a string.

    ``r`` -> ``RECTANGLE``
    ``rect`` -> ``RECTANGLE``
    ``rectangle`` -> ``RECTANGLE``
    ``c`` -> ``ELLIPSE``
    ``circle`` -> ``ELLIPSE``
    ``ellipse`` -> ``ELLIPSE``


    :param string: mask shape
    :return: :py:enum:`BasicMaskShape`
    """
    l_string = string.lower()

    if l_string in {'r', 'rect', 'rectangle'}:
        return BasicMaskShape.RECTANGLE
    elif l_string in {'c', 'circle', 'ellipse'}:
        return BasicMaskShape.ELLIPSE

    raise ValueError(f'Unknown mask shape: {string}')


def debug_format_args(args_dict: dict[str, typing.Any],
                      value_transformer: typing.Callable[[str, typing.Any], str] | None = None,
                      max_value_len: int = 256,
                      as_kwargs: bool = False):
    """
    Format function arguments in a way that can be printed for debug messages.

    :param args_dict: argument dictionary
    :param value_transformer: transform values in the argument dictionary
    :param max_value_len: Max length of a formatted value before it is turned into a class and id string only
    :param as_kwargs: Format the string as python keyword arguments instead of a dictionary
    :return: formatted string
    """

    def _value_transformer(key, value):
        if value_transformer is not None:
            output = value_transformer(key, value)
        else:
            output = value

        if len(str(output)) > max_value_len:
            return _types.class_and_id_string(value)

        return str(output)

    if not as_kwargs:
        return str(
            {k: str(_value_transformer(k, v)) for k, v in args_dict.items()})
    else:
        return ', '.join(f'{k}={_value_transformer(k, v)}' for k, v in args_dict.items())


class TimeDeltaParseError(Exception):
    """
    Raised by :py:func:`.parse_timedelta` on parse errors.
    """
    pass


def parse_timedelta(string: str | None) -> datetime.timedelta:
    """
    Parse a ``datetime.timedelta`` object from an arguments string.

    Passing 'forever', an empty string, or ``None`` will result in this function returning ``datetime.timedelta.max``

    Accepts all named arguments of ``datetime.timedelta``

    .. code-block:: python

        parse_time_delta('days=1; seconds=30')

    :raise TimeDeltaParseError: on parse errors

    :param string: the arguments string
    :return: ``datetime.timedelta``
    """
    parser = ConceptUriParser(
        concept_name='timedelta',
        known_args=['days', 'seconds', 'microseconds', 'milliseconds', 'minutes', 'hours', 'weeks'])

    if string is None or string.lower().strip() == 'forever' or not string.strip():
        return datetime.timedelta.max

    try:
        result = parser.parse('timedelta;' + string)
    except ConceptUriParseError as e:
        raise TimeDeltaParseError(e) from e

    args = dict()
    for k, v in result.args.items():
        try:
            args[k] = float(v)
        except ValueError as e:
            raise TimeDeltaParseError(f'Argument "{k}" must be a floating point value or integer.') from e

    try:
        return datetime.timedelta(**args)
    except Exception as e:
        raise TimeDeltaParseError(e) from e


def remove_terminal_escape_sequences(string):
    """
    Remove any terminal escape sequences from a string.

    :param string: the string
    :return: the clean string
    """
    ansi_escape = re.compile(r'''
        \x1B  # ESC
        (?:   # 7-bit C1 Fe (except CSI) [@-Z\\-_] | 
              # or [ for CSI, followed by a control sequence
            \[ [0-?]* [ -/]* [@-~]   
        )
    ''', re.VERBOSE)

    return ansi_escape.sub('', string)


def _remove_tail_comments_unlexable(string) -> str:
    try:
        # find the start of a possible comment
        comment_start = string.index('#')
    except ValueError:
        # no comments, all good
        return string

    if comment_start == 0:
        # it starts the string, ignore
        return string

    segment = string[:comment_start]
    if segment.endswith('\\'):
        # found a comment token, but it was escaped

        next_spaces = ''.join(itertools.takewhile(lambda x: x.isspace(), string[comment_start + 1:]))
        # record all space characters after the #, they may be taken by the lexer if
        # it can make sense of what came after #

        lexed, result = remove_tail_comments(string[comment_start + 1:])
        # recursive decent to the right starting after the # (operator lol) to solve the rest of the string

        if next_spaces and lexed:
            # the spaces would have been consumed by lexing, add them back to the
            # left side of the string where the user intended them to be
            result = next_spaces + result

        # return the left side minus the escape sequence + the comment # token, plus the evaluated right side
        return segment.removesuffix('\\') + '#' + result

    # unescaped comment start, return the segment to the left
    return segment


def remove_tail_comments(string) -> tuple[bool, str]:
    """
    Remove trailing comments from a dgenerate config line

    Will not remove a comment if it is the only thing on the line.

    Considers strings and comment escape sequences.

    :param string: the string
    :return: ``(removed anything?, stripped string)``
    """
    # this is a more difficult problem than I imagined.
    try:
        parts = tokenized_split(string, '#',
                                escapable_separator=True,
                                allow_unterminated_strings=True)
        # attempt to split off the comment

        if not parts:
            # empty case, effectively un-lexed
            return False, string

        new_value = parts[0]

        if not new_value.strip():
            # do not remove if the comment is all that exists on the line
            # that is handled elsewhere
            return False, string

        # the left side was lexed and stripped of leading and trailing whitespace
        return True, new_value
    except TokenizedSplitSyntaxError:
        # could not lex this because of a syntax error, since unterminated
        # strings are understandable by the lexer given our options, this
        # is an uncommon if not near impossible occurrence
        return False, _remove_tail_comments_unlexable(string)


def format_image_seed_uri(seed_images: str | collections.abc.Iterable[str] | None = None,
                          mask_images: str | collections.abc.Iterable[str] | None = None,
                          control_images: str | collections.abc.Iterable[str] | None = None,
                          adapter_images: str | collections.abc.Iterable[str] | None = None,
                          latents: str | collections.abc.Iterable[str] | None = None,
                          floyd_image: str | None = None,
                          resize: str | tuple[int | str, int | str] | None = None,
                          aspect: bool = True,
                          frame_start: int | None = None,
                          frame_end: int | None = None) -> str:
    """
    Formats a ``--image-seeds`` URI to its shortest possible string form.

    :raise ValueError: if ``inpaint_image`` is specified without ``seed_image``.
                       if keyword arguments are present without ``seed_image`` or ``control_images``.
                       if ``frame_start`` or ``frame_end`` are negative values.
                       if ``frame_start`` is greater than ``frame_end``.
                       if ``adapter_images`` are used with ``floyd_image``.
                       if ``latents`` are used with ``floyd_image``.
                       if both ``control_images`` and ``floyd_image`` are specified.
                       if ``resize`` is specified when only ``latents`` are provided.
                       if ``aspect=False`` is specified when only ``latents`` are provided.
                       if ``frame_start`` or ``frame_end`` is specified when only ``latents`` are provided.
                       if too many mask images are provided.
                       if too few mask images are provided.
                       if no arguments are provided.

    :param seed_images: Seed image path(s)
    :param mask_images: Inpaint image path(s)
    :param control_images: Control image path(s)
    :param adapter_images: Adapter image path(s)
    :param latents: Raw latent tensor path(s) (.pt, .pth, .safetensors files)
    :param floyd_image: Path to a Floyd image
    :param resize: Optional resize dimension (WxH string)
    :param aspect: Preserve aspect ratio?
    :param frame_start: Optional frame start index
    :param frame_end: Optional frame end index
    :return: The generated ``--image-seeds`` URI string
    """

    if all(v is None for v in locals().values() if not isinstance(v, bool)):
        raise ValueError('format_image_seed_uri, no arguments provided.')

    # Handle input collections
    if seed_images is not None and not isinstance(seed_images, str):
        if mask_images is not None and not isinstance(mask_images, str):
            mask_image_cnt = len(list(mask_images))
            seed_image_cnt = len(list(seed_images))

            if mask_image_cnt > seed_image_cnt:
                raise ValueError('too many mask images provided.')
            if mask_image_cnt != 1 and mask_image_cnt < seed_image_cnt:
                raise ValueError('too few mask images provided.')

        seed_images = 'images:' + ', '.join(seed_images)

    if mask_images is not None and not isinstance(mask_images, str):
        mask_images = ', '.join(mask_images)

    if control_images is not None and not isinstance(control_images, str):
        control_images = ', '.join(control_images)

    if latents is not None and not isinstance(latents, str):
        latents = ', '.join(latents)

    if adapter_images is not None and not isinstance(adapter_images, str):
        adapter_images = ' + '.join(adapter_images)

    components = []

    def add_component_if_valid(component, prefix=""):
        if component is not None:
            if prefix:
                components.append(f"{prefix}={component}")
            else:
                components.append(str(component))

    # Validate inputs
    if mask_images and not seed_images and not floyd_image:
        raise ValueError('inpaint_image cannot be specified without seed_image.')

    if adapter_images and floyd_image:
        raise ValueError('adapter_images cannot be specified with floyd_image.')

    if latents and floyd_image:
        raise ValueError('latents cannot be specified with floyd_image.')

    if control_images and floyd_image:
        raise ValueError('control_images cannot be specified with floyd_image.')

    # Handle resize validation
    if resize is not None:
        if isinstance(resize, str):
            resize = resize.strip()
            if resize:
                try:
                    parse_image_size(resize)
                except ValueError as e:
                    raise ValueError(f'invalid resize value: {e}') from e
        elif isinstance(resize, tuple):
            try:
                resize = format_size(tuple(int(i) for i in resize))
            except ValueError as e:
                raise ValueError(f'invalid resize value: {e}') from e
        else:
            raise ValueError('resize argument expects a string or a tuple.')

    # Check if only latents are specified (no images that can be resized)
    if latents and not (seed_images or control_images or adapter_images):
        if resize:
            raise ValueError(
                'resize cannot be specified when only latents are provided (latents are used as-is).')
        if aspect is False:  # Non-default value
            raise ValueError(
                'aspect=False cannot be specified when only latents are '
                'provided (aspect ratio is meaningless for latents).')
        if frame_start is not None or frame_end is not None:
            raise ValueError(
                'frame_start or frame_end cannot be specified when only latents are '
                'provided (latents are used as-is and do not have animation frames).')

    # Special case: adapter images only
    if adapter_images and not any([seed_images, mask_images, control_images, latents, floyd_image]):
        components.append('adapter:' + adapter_images)
        return ";".join(components)

    # Special case: floyd image
    if floyd_image and (seed_images or mask_images):
        components.append(seed_images or mask_images)
        add_component_if_valid(floyd_image, "floyd")
        return ";".join(components)

    # Handle base image (seed, control, or latents)
    if control_images and not seed_images and not mask_images and not latents:
        # Control image alone becomes the seed
        seed_images = control_images
        control_images = None
    elif latents and not seed_images and not mask_images and not control_images:
        # Latents alone becomes the seed
        seed_images = 'latents:' + latents
        latents = None

    # Add base image
    if seed_images:
        components.append(seed_images)

    # Check if we can use legacy format (no keyword arguments)
    use_legacy = (
        # No complex arguments that require keywords
        not adapter_images and
        not latents and
        not floyd_image and
        aspect is True and
        frame_start is None and
        frame_end is None and
        # Either:
        # 1. Seed image with optional mask
        # 2. Control image only
        (
            (seed_images and not control_images) or  # Case 1: seed image (with optional mask)
            (control_images and not seed_images and not mask_images)  # Case 2: control image only
        )
    )

    if use_legacy:
        # Legacy format: img.png;mask.png;512x512 or control.png;512x512
        if mask_images:
            components.append(mask_images)
        if control_images:
            components.append(control_images)
        if resize:
            components.append(resize)
    else:
        # Modern format with keywords
        if mask_images:
            add_component_if_valid(mask_images, "mask")
        if latents:
            add_component_if_valid(latents, "latents")
        if adapter_images:
            add_component_if_valid(adapter_images, "adapter")
        if control_images:
            add_component_if_valid(control_images, "control")
        if resize:
            add_component_if_valid(resize, "resize")
        if aspect is False:
            add_component_if_valid(str(aspect), "aspect")
        if frame_start is not None:
            add_component_if_valid(frame_start, "frame-start")
        if frame_end is not None:
            add_component_if_valid(frame_end, "frame-end")

    return ";".join(components)


def format_dgenerate_config(lines: typing.Iterator[str], indentation=' ' * 4) -> typing.Iterator[str]:
    """
    A very rudimentary code formatter for dgenerate configuration / script.

    Does not handle breaking jinja control blocks on to a new
    line if multiple start blocks exist on the same line.

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

    for line, next_line in lines:
        no_tail_comment_line = remove_tail_comments(line)[1].strip()

        # Handle line continuation with backslash or hyphen
        if in_continuation or no_tail_comment_line.endswith('\\') or (
                next_line is not None and next_line.strip().startswith('-')):
            continuation_lines.append(line)
            if next_line is None:
                in_continuation = False
                yield from add_continuation_lines()
            elif not no_tail_comment_line.endswith('\\') \
                    and not (next_line.strip().startswith('-')) \
                    and not (no_tail_comment_line.startswith('#') or not no_tail_comment_line.strip()):
                in_continuation = False
                yield from add_continuation_lines()
            else:
                in_continuation = True
            continue

        # Handle Jinja2 control structures
        if no_tail_comment_line.startswith('{% end') or no_tail_comment_line.startswith(
                '{% else') or no_tail_comment_line.startswith('{% elif'):
            indent_level = max(indent_level - 1, 0)

        # Process any accumulated continuation lines before adding a new block line
        yield from add_continuation_lines()

        # Add the current line with the appropriate indentation
        yield indentation * indent_level + line.strip()

        # Increase indent level for starting control structures
        if (no_tail_comment_line.startswith('{% for') or no_tail_comment_line.startswith('{% if') or
                no_tail_comment_line.startswith('{% block') or no_tail_comment_line.startswith('{% macro') or
                no_tail_comment_line.startswith('{% filter') or no_tail_comment_line.startswith('{% with') or
                no_tail_comment_line.startswith('{% else') or no_tail_comment_line.startswith('{% elif')):
            indent_level += 1

        # Set continuation flag if the line ends with a backslash
        if line.endswith('\\'):
            in_continuation = True

    # Handle any remaining continuation lines after the last line
    yield from add_continuation_lines()


class ArgparseParagraphFormatter(argparse.HelpFormatter):
    """
    Argparse formatter which preserves paragraphs in help text.

    This formatter also underlines option text for better visual segregation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, max_help_position=8, **kwargs)

    def _fill_text(self, text, width, indent):
        return wrap_paragraphs(
            inspect.cleandoc(text),
            width=width,
            initial_indent=indent,
            subsequent_indent=indent
        )

    def _split_lines(self, text, width):
        wrapped_lines = wrap_paragraphs(
            inspect.cleandoc(text), width=width
        ).split('\n')

        wrapped_lines.append('-' * len(wrapped_lines[-1]))
        return wrapped_lines
