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
import collections.abc
import enum
import re
import shutil
import textwrap
import typing

import dgenerate.types as _types


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

    args: dict[str, typing.Union[str, list[str]]]
    """
    Provided keyword arguments with their (string) values.
    """

    def __init__(self, concept: str, args: dict[str, str]):
        self.concept = concept
        self.args = args

    def __str__(self):
        return f"{self.concept}: {self.args}"


class TokenizedSplitSyntaxError(Exception):
    """
    Raised by :py:func:`tokenized_split` on syntax errors.
    """
    pass


def tokenized_split(string,
                    separator,
                    remove_quotes=False,
                    strict=False,
                    escapes_in_unquoted=False,
                    escapes_in_quoted=False):
    """
    Split a string by a separator and discard whitespace around tokens, avoid
    splitting within single or double quoted strings. Empty fields may be used.

    Quotes can be always be escaped with a backslash to avoid the creation of a
    string type token. The backslash will remain in the output if ``escapes_in_unquoted``
    or ``escapes_in_quoted`` are ``False`` and the escape occurs in the relevant context.

    :raise TokenizedSplitSyntaxError: on syntax errors.

    :param string: the string
    :param separator: separator
    :param remove_quotes: remove quotes from quoted string tokens?
    :param strict: Text tokens cannot be intermixed with quoted strings? disallow IE: ``"text'string'text"``
    :param escapes_in_unquoted: evaluate escape sequences in text tokens (unquoted strings)?
        The slash is retained by default when escaping quotes, this disables that, and also enables handling of the escapes ``n, r, t, b, f, and \\``.
        IE: given ``separator =";"`` parse ``\\"token\\"; "a b"`` -> ``['"token"', 'a b']``, instead of ``\\"token\\"; "a b"``-> ``['\\"token\\"', 'a b']``
    :param escapes_in_quoted: evaluate escape sequences in quoted string tokens?
        The slash is retained by default when escaping quotes, this disables that, and also enables handling of the escapes ``n, r, t, b, f, and \\``.
        IE given ``separator = ";"`` parse ``token; "a \\" b"`` -> ``['token', 'a " b']``, instead of ``token; "a \\" b"``-> ``['token', 'a \\" b']``

    :return: parsed fields
    """

    class _States(enum.Enum):
        AWAIT_TEXT = 0
        TEXT_TOKEN = 1
        TEXT_TOKEN_STRICT = 2
        ESCAPE_TEXT = 3
        STRING = 4
        STRING_ESCAPE = 5
        SEP_REQUIRED = 6

    state = _States.AWAIT_TEXT

    # tokens out
    parts = []

    # quoted string token accumulator
    cur_string = ''

    # the quote character that initiated
    # the current string token being accumulated
    cur_quote = ''

    # accepted quote characters
    QUOTE_CHARS = {'"', "'"}

    # recognized escape codes
    RECOGNIZED_ESCAPE_CODES = {'n', 'r', 't', 'b', 'f', '\\'}

    def append_text(t):
        # append text to the last token
        if parts:
            parts[-1] += t
        else:
            parts.append(t)

    def syntax_error(msg, idx):
        # create syntax error
        return TokenizedSplitSyntaxError(f'{msg}: \'{string[:idx]}[ERROR HERE>]{string[idx:]}\'')

    for idx, c in enumerate(string):
        if state == _States.STRING:
            # inside of a quoted string
            if c == '\\':
                # encountered an escape sequence start
                state = _States.STRING_ESCAPE
                if not escapes_in_quoted:
                    cur_string += c
            elif c == cur_quote:
                # encountered the terminator quote
                append_text(cur_string + (c if not remove_quotes else ''))
                cur_string = ''
                # Strict mode requires a separator after a quoted string token
                state = _States.SEP_REQUIRED if strict else _States.TEXT_TOKEN
            else:
                # append to current string
                cur_string += c
        elif state == _States.ESCAPE_TEXT:
            # after encountering an escape sequence start inside of a text token

            state = _States.TEXT_TOKEN_STRICT if strict else _States.TEXT_TOKEN
            # return to the appropriate state

            if c in QUOTE_CHARS:
                # this is an escaped quotation character
                append_text(c)
            elif c in RECOGNIZED_ESCAPE_CODES:
                # this is a character that translates into utf-8 escape code
                # that we have decided to support
                if escapes_in_unquoted:
                    append_text(fr'\{c}'.encode('utf-8').decode('unicode_escape'))
                else:
                    append_text(c)
            elif escapes_in_unquoted:
                # unknown escape code case 1
                append_text(fr'\{c}')
            else:
                # unknown escape code case 2
                append_text(c)
        elif state == _States.STRING_ESCAPE:
            # after encountering an escape sequence start inside of a quoted string
            state = _States.STRING
            # return to string state
            if c in QUOTE_CHARS:
                # this is an escaped quotation character
                cur_string += c
            elif c in RECOGNIZED_ESCAPE_CODES:
                # this is a character that translates into utf-8 escape code
                # that we have decided to support
                if escapes_in_quoted:
                    cur_string += fr'\{c}'.encode('utf-8').decode('unicode_escape')
                else:
                    cur_string += c
            elif escapes_in_quoted:
                # unknown escape code case 1
                cur_string += fr'\{c}'
            else:
                # unknown escape code case 2
                cur_string += c
        elif state == _States.SEP_REQUIRED:
            # This state is only reached in strict mode
            # where separators are required after a string token
            if c == separator:
                state = _States.AWAIT_TEXT
                parts.append('')
            elif not c.isspace():
                raise syntax_error('Missing separator after string', idx)
        elif state == _States.AWAIT_TEXT:
            if c == '\\':
                # started an escape sequence
                state = _States.ESCAPE_TEXT
                if not escapes_in_unquoted:
                    append_text(c)
            elif c in QUOTE_CHARS:
                # started a string token
                cur_quote = c
                cur_string += (c if not remove_quotes else '')
                state = _States.STRING
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
                state = _States.TEXT_TOKEN_STRICT if strict else _States.TEXT_TOKEN
                append_text(c)
        elif state == _States.TEXT_TOKEN:
            # this is the non strict mode parsing state inside a text token
            if c == '\\':
                # started an escape sequence in a text token
                state = _States.ESCAPE_TEXT
                if not escapes_in_unquoted:
                    append_text(c)
            elif c in QUOTE_CHARS:
                # started a string token intermixed with a text token
                cur_quote = c
                cur_string += (c if not remove_quotes else '')
                state = _States.STRING
            elif c == separator:
                # encountered a separator
                # the last element needs to be right stripped
                # because spaces are allowed inside a text token
                # and there is no way to differentiate 'inside' and
                # 'outside' without lookahead, or until there occurs
                # a separator
                parts[-1] = parts[-1].rstrip()
                parts.append('')
                state = _States.AWAIT_TEXT
            else:
                # append text token character
                append_text(c)
        elif state == _States.TEXT_TOKEN_STRICT:
            # This state is only reached in strict mode
            if c == '\\':
                # encountered an escape sequence in a text token
                state = _States.ESCAPE_TEXT
                if not escapes_in_unquoted:
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
                parts[-1] = parts[-1].rstrip()
                parts.append('')
                state = _States.AWAIT_TEXT
            else:
                # append text token character
                append_text(c)

    if state == _States.STRING:
        # state machine ended inside a quoted string
        raise syntax_error(f'un-terminated string: \'{cur_string}\'', len(string))

    if state == _States.TEXT_TOKEN_STRICT or state == _States.TEXT_TOKEN:
        # if we end on a text token, right strip the text token, as it is
        # considered 'outside' the token, and spaces are only allowed 'inside'
        # and this is ambiguous without lookahead
        parts[-1] = parts[-1].rstrip()

    return parts


class UnquoteSyntaxError(Exception):
    """
    Raised by :py:func:`.unquote` on parsing errors.
    """
    pass


def unquote(string: str, escapes_in_quoted=True, escapes_in_unquoted=False) -> str:
    """
    Remove quotes from a string, including single quotes.

    Unquoted strings will have leading an trailing whitespace stripped.

    Quoted strings will have leading and trailing whitespace stripped up to where the quotes were.

    :param escapes_in_unquoted: Render escape sequences in strings that are unquoted?
    :param escapes_in_quoted: Render escape sequences in strings that are quoted?
    :param string: the string
    :return: The un-quoted string
    """
    try:
        val = tokenized_split(string,
                              escapes_in_quoted=escapes_in_quoted,
                              escapes_in_unquoted=escapes_in_unquoted,
                              strict=True,
                              remove_quotes=True,
                              separator=None)
        if val:
            return val[0]
        return ''
    except TokenizedSplitSyntaxError as e:
        if 'Missing separator' in str(e):
            raise UnquoteSyntaxError(str(e).replace('Missing separator', 'Extraneous text'))
        raise UnquoteSyntaxError(e)


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

    args_raw: typing.Union[None, bool, set[str]]
    """
    ``True`` indicates all argument values are returned without any unquoting or processing into lists.
    
    ``None`` or ``False`` indicates no argument values skip extended processing.
    
    Assigning a set containing argument names indicates only the specified 
    arguments skip extended processing (unquoting or splitting).
    """

    args_lists: typing.Union[None, bool, set[str]]
    """
    ``True`` indicates all arguments can accept a comma separated list.
    
    ``None`` or ``False`` indicates no arguments can accept a comma separated list.
    
    Assigning a set containing argument names indicates only the specified 
    arguments can accept a comma separated list.
    
    When an argument is parsed as a comma separated list, its value/type
    in :py:attr:`ConceptUri.args` will be that of a list.
    """

    def __init__(self,
                 concept_name: _types.Name,
                 known_args: collections.abc.Iterable[str],
                 args_lists: typing.Union[None, bool, collections.abc.Iterable[str]] = None,
                 args_raw: typing.Union[None, bool, collections.abc.Iterable[str]] = None):
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

    def parse(self, uri: _types.Uri) -> ConceptUri:
        """
        Parse a string.

        :param uri: the string

        :raise ConceptUriParseError: on parsing errors
        :raise ValueError: if uri is ``None``

        :return: :py:class:`.ConceptPath`
        """
        args = dict()

        if uri is None:
            raise ValueError('uri must not be None')

        if not uri.strip():
            raise ConceptUriParseError(f'Error parsing {self.concept_name} URI, URI was empty.')

        try:
            parts = tokenized_split(uri, ';')
        except TokenizedSplitSyntaxError as e:
            raise ConceptUriParseError(f'Error parsing {self.concept_name} URI "{uri}": {e}')

        parts = iter(parts)
        concept = parts.__next__()
        for i in parts:
            vals = i.split('=', 1)
            if not vals or not vals[0]:
                raise ConceptUriParseError(f'Error parsing path arguments for '
                                           f'{self.concept_name} concept "{concept}", Empty argument space, '
                                           f'stray semicolon?')
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
                    vals = tokenized_split(vals[1], ',',
                                           remove_quotes=True,
                                           strict=True,
                                           escapes_in_quoted=True)

                    args[name] = vals if len(vals) > 1 else vals[0]
                else:
                    args[name] = unquote(vals[1])

            except (TokenizedSplitSyntaxError, UnquoteSyntaxError) as e:
                raise ConceptUriParseError(f'Syntax error parsing argument "{name}" for '
                                            f'{self.concept_name} concept "{concept}": {e}')
        return ConceptUri(concept, args)


def oxford_comma(elements: collections.abc.Sequence[str], conjunction: str) -> str:
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

    :return: int
    """
    return min(shutil.get_terminal_size(fallback=(150, 0))[0], 150)


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
    Return True if a string is quoted with an identical starting and end quote.

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
        value_or_struct: typing.Union[typing.Any, collections.abc.Iterable[typing.Union[typing.Any, list, tuple]]]) -> \
        typing.Union[list, tuple, typing.Any]:
    """
    Quote any str(value) containing spaces, or str(value)s containing spaces within a list, or list of lists/tuples.

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


def wrap_paragraphs(text: str,
                    width: int,
                    break_long_words=False,
                    break_on_hyphens=False,
                    **fill_args):
    """
    Wrap text that may contain paragraphs without removing separating whitespace.

    :param text: Text containing paragraphs
    :param width: Wrap with in characters
    :param break_long_words: break on long words? default False
    :param break_on_hyphens: break on hyphens? default False
    :param fill_args: extra keyword arguments to :py:func:`textwrap.fill` if desired
    :return: text wrapped string
    """
    paragraphs = []
    in_spacing = False
    for line in text.splitlines():
        if not line and not in_spacing:
            in_spacing = True
            paragraphs.append([])
        elif line:
            in_spacing = False
            if len(paragraphs) == 0:
                paragraphs.append([])

            paragraphs[-1].append(line)

    return '\n\n'.join(
        textwrap.fill(
            text,
            width=width,
            break_long_words=break_long_words,
            break_on_hyphens=break_on_hyphens,
            **fill_args) for
        text in (' '.join(paragraph) for paragraph in paragraphs))


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


def format_size(size: collections.abc.Iterable[int]):
    """
    Join together an iterable of integers with the character x

    :param size: the iterable
    :return: formatted string
    """
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
    ver_parts = string.split('.')
    minor_re = re.compile(r'\d+')

    if len(ver_parts) != 3 or not minor_re.match(ver_parts[2]):
        raise ValueError(
            f'version expected to be a version string in the format major.minor.patch. received: "{string}"')

    return int(ver_parts[0]), int(ver_parts[1]), int(minor_re.match(ver_parts[2])[0])


def debug_format_args(args_dict: dict[str, typing.Any],
                      value_transformer: typing.Optional[typing.Callable[[str, typing.Any], str]] = None,
                      max_value_len: int = 256):
    """
    Format function arguments in a way that can be printed for debug messages.

    :param args_dict: argument dictionary
    :param value_transformer: transform values in the argument dictionary
    :param max_value_len: Max length of a formatted value before it is turned into a class and id string only
    :return: formatted string
    """

    def _value_transformer(key, value):
        if value_transformer is not None:
            return value_transformer(key, value)
        return value

    return str(
        {k: str(_value_transformer(k, v)) if
        len(str(_value_transformer(k, v))) < max_value_len
        else _types.class_and_id_string(v) for k, v in args_dict.items()})
