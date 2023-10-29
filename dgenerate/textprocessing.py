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
import ast
import shutil
import textwrap
import typing

import dgenerate.types as _types


class ConceptPathParseError(Exception):
    pass


class ConceptUri:
    """
    Represents a parsed concept path.
    """

    concept: str
    """
    The primary concept mentioned in the URI.
    """

    args: typing.Dict[str, str]
    """
    Provided keyword arguments with their (string) values.
    """

    def __init__(self, concept: str, args: typing.Dict[str, str]):
        self.concept = concept
        self.args = args

    def __str__(self):
        return f"{self.concept}: {self.args}"


class ConceptUriParser:
    """
    Parser for dgenerate concept paths with arguments, IE: concept;arg1="a";arg2="b"

    Used for ``--vae``, ``--lora`` etc. as well as image preprocessor plugin module arguments.
    """

    concept_name: _types.Name
    """
    Name / title string for this concept. Used in parse error exceptions.
    """

    known_args: typing.Set[str]
    """
    Unique recognized keyword arguments
    """

    def __init__(self, concept_name: _types.Name, known_args: typing.Sequence[str] = None):
        """
        :raises: :py:exc:`ValueError` if duplicate argument names are specified.

        :param concept_name: Concept name, used in error messages
        :param known_args: valid arguments for the parser, must be unique
        """

        check = set()
        for arg in known_args:
            if arg in check:
                raise ValueError(f'duplicate argument specification {arg}')
            check.add(arg)

        self.known_args = check
        self.concept_name = concept_name

    def parse_concept_uri(self, uri: _types.Uri):
        """
        Parse a string.

        :param uri: the string

        :raise: :py:class:`.ConceptPathParseError`

        :return: :py:class:`.ConceptPath`
        """
        args = dict()
        parts = uri.split(';')
        parts = iter(parts)
        concept = parts.__next__()
        for i in parts:
            vals = i.split('=', 1)
            if not vals:
                raise ConceptPathParseError(f'Error parsing path arguments for '
                                            f'{self.concept_name} concept "{concept}", Empty argument space, '
                                            f'stray semicolon?')
            if len(vals) == 1:
                raise ConceptPathParseError(f'Error parsing path arguments for '
                                            f'{self.concept_name} concept "{concept}", missing value '
                                            f'assignment for argument {vals[0]}.')
            name = vals[0].strip().lower()
            if self.known_args is not None and name not in self.known_args:
                raise ConceptPathParseError(
                    f'Unknown path argument "{name}" for {self.concept_name} concept "{concept}", '
                    f'valid arguments: {", ".join(sorted(self.known_args))}')

            if name in args:
                raise ConceptPathParseError(
                    f'Duplicate argument "{name}" provided for {self.concept_name} concept "{concept}".')

            try:
                args[name] = unquote(vals[1])
            except SyntaxError as e:
                raise ConceptPathParseError(f'Syntax Error parsing argument {name} for '
                                            f'{self.concept_name} concept "{concept}": {e}')
        return ConceptUri(concept, args)


def oxford_comma(elements: typing.Sequence[str], conjunction: str) -> str:
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


def quote(string: str) -> str:
    """
    Wrap a string in double quotes.

    This is not equivalent to shell quoting.

    :param string: the string
    :return: The quoted string
    """
    return f'"{string}"'


def is_quoted(string: str) -> bool:
    """
    Return True if a string is quoted with an identical starting and end quote.

    :param string: the string
    :return: ``True`` or ``False``
    """
    return (string.startswith('"') and string.endswith('"')) or (string.startswith("'") and string.endswith("'"))


def unquote(string: str) -> str:
    """
    Remove quotes from a string, including single quotes.

    :param string: the string
    :return: The un-quoted string
    """
    string = string.strip(' ')
    if string.startswith('"'):
        if not string.endswith('"'):
            raise ValueError('Missing ending ["] quote.')
        return str(ast.literal_eval('r' + string))
    if string.startswith("'"):
        if not string.endswith("'"):
            raise ValueError('Missing ending [\'] quote.')
        return str(ast.literal_eval('r' + string))
    else:
        # Is an unquoted string
        return str(string.strip(' '))


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


def quote_spaces(value_or_struct: typing.Union[typing.Any, typing.Sequence[typing.Union[typing.Any, list, tuple]]]) -> \
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
    :param fill_args: extra keyword arguments to :py:meth:`textwrap.fill` if desired
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
    :param fill_args: extra keyword arguments to :py:meth:`textwrap.fill` if desired
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


def format_size(size: typing.Iterable[int]):
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


def debug_format_args(args_dict: typing.Dict[str, typing.Any],
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
