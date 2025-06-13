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

__doc__ = """
Prompt representation object / prompt parsing.
"""

import ast
import collections.abc
import functools
import re
import typing

import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types


class PromptEmbeddedArgumentError(Exception):
    """
    Error involving a prompt embedded argument other than ``weighter``
    """
    pass


class Prompt:
    """
    Represents a combined positive and optional negative prompt split by a delimiter character.
    """

    def __init__(self,
                 positive: str | None = None,
                 negative: str | None = None,
                 delimiter: str = ';',
                 weighter: _types.OptionalUri = None,
                 upscaler: _types.OptionalUriOrUris = None,
                 embedded_args: dict[str, str] | None = None):
        """
        :param positive: positive prompt component.
        :param negative: negative prompt component.
        :param delimiter: delimiter for stringification.
        :param weighter: ``--prompt-weighter`` plugin URI.
        :param upscaler: ``--prompt-upscaler`` plugin URI.
        :param embedded_args: embedded prompt arguments parsed from ``<argument: value_text>``.
        """

        import dgenerate.promptupscalers as _promptupscalers
        import dgenerate.promptweighters as _promptweighters

        if weighter is not None and not _promptweighters.prompt_weighter_exists(weighter):
            raise PromptEmbeddedArgumentError(
                f'Unknown prompt "weighter" implementation: {_promptweighters.prompt_weighter_name_from_uri(weighter)}, '
                f'must be one of: {_textprocessing.oxford_comma(_promptweighters.prompt_weighter_names(), "or")}')

        if upscaler is not None:
            if isinstance(upscaler, str):
                check_upscalers = [upscaler]
            else:
                check_upscalers = upscaler

            for u in check_upscalers:
                if not _promptupscalers.prompt_upscaler_exists(u):
                    raise PromptEmbeddedArgumentError(
                        f'Unknown prompt "upscaler" implementation: {_promptupscalers.prompt_upscaler_name_from_uri(u)}, '
                        f'must be one of: {_textprocessing.oxford_comma(_promptupscalers.prompt_upscaler_names(), "or")}'
                    )

        self._positive = positive
        self._negative = negative
        self._delimiter = delimiter
        self._weighter = weighter
        self._upscaler = upscaler

        if embedded_args is not None:
            self._embedded_args = embedded_args
        else:
            self._embedded_args = dict()

    def __str__(self):
        if self._positive and self._negative:
            return f'{self._positive}{self._delimiter} {self._negative}'
        elif self._positive:
            return self._positive
        elif self._negative:
            return self._delimiter + self._negative
        else:
            return ''

    @property
    def delimiter(self) -> str:
        """
        Positive / Negative delimiter for this prompt, for example ";"
        """
        return self._delimiter

    @property
    def positive(self) -> str | None:
        """
        Positive prompt value.
        """
        return self._positive

    @positive.setter
    def positive(self, value):
        self._positive = value

    @property
    def negative(self) -> str | None:
        """
        Negative prompt value.
        """
        return self._negative

    @negative.setter
    def negative(self, value):
        self._negative = value

    @property
    def weighter(self) -> _types.OptionalUri:
        """
        Embedded prompt weighter URI argument for this prompt if any.
        """
        return self._weighter

    @property
    def upscaler(self) -> _types.OptionalUriOrUris:
        """
        Embedded prompt upscaler URI argument for this prompt if any.
        """
        return self._upscaler

    @property
    def embedded_args(self) -> list[tuple[str, str]]:
        """
        Other embedded arguments parsed out of the prompt.
        """
        return list(self._embedded_args.items())

    def __repr__(self):
        return f"'{str(self)}'"

    def copy_embedded_args_from(self, prompt: 'Prompt'):
        self._weighter = prompt._weighter
        self._upscaler = prompt._upscaler
        self._embedded_args = prompt._embedded_args.copy()

    def set_embedded_args_on(
            self,
            on_object: typing.Any,
            forbidden_checker: typing.Optional[typing.Callable[[str, typing.Any], bool]] = None,
            validate_only: bool = False
    ):
        """
        Set the other embedded arguments parsed out of a prompt on to an object.

        The object should be type hinted using types from :py:mod:`dgenerate.types`

        Specifically, any of:

        * :py:class:`dgenerate.types.Size`
        * :py:class:`dgenerate.types.Sizes`
        * :py:class:`dgenerate.types.OptionalSize`
        * :py:class:`dgenerate.types.OptionalSizes`
        * :py:class:`dgenerate.types.Padding`
        * :py:class:`dgenerate.types.Paddings`
        * :py:class:`dgenerate.types.OptionalPadding`
        * :py:class:`dgenerate.types.OptionalPaddings`
        * :py:class:`dgenerate.types.Boolean`
        * :py:class:`dgenerate.types.OptionalBoolean`
        * :py:class:`dgenerate.types.Float`
        * :py:class:`dgenerate.types.Floats`
        * :py:class:`dgenerate.types.OptionalFloat`
        * :py:class:`dgenerate.types.OptionalFloats`
        * :py:class:`dgenerate.types.Integer`
        * :py:class:`dgenerate.types.Integers`
        * :py:class:`dgenerate.types.OptionalInteger`
        * :py:class:`dgenerate.types.OptionalIntegers`
        * :py:class:`dgenerate.types.String`
        * :py:class:`dgenerate.types.Strings`
        * :py:class:`dgenerate.types.OptionalString`
        * :py:class:`dgenerate.types.OptionalStrings`
        * :py:class:`dgenerate.types.Name`
        * :py:class:`dgenerate.types.Names`
        * :py:class:`dgenerate.types.OptionalName`
        * :py:class:`dgenerate.types.OptionalNames`
        * :py:class:`dgenerate.types.Uri`
        * :py:class:`dgenerate.types.Uris`
        * :py:class:`dgenerate.types.OptionalUri`
        * :py:class:`dgenerate.types.OptionalUris`


        :raise PromptEmbeddedArgumentError: If there was a problem applying the embedded arguments to the object.

        :param on_object: The object to set values on.
        :param forbidden_checker: This is a function that should return ``True`` if an argument name / value is forbidden to use.
        :param validate_only: Only run validation and do not set any values?
        """
        if forbidden_checker and not callable(forbidden_checker):
            raise ValueError('forbidden_checker must be a callable function')

        hints = typing.get_type_hints(on_object)

        def is_forbidden(name, value):
            return forbidden_checker(name, value) if forbidden_checker else False

        def parse_padding(value):
            parsed_value = _textprocessing.parse_dimensions(value)

            length = len(parsed_value)

            if length > 4:
                raise ValueError('too many padding values.')

            if length == 3:
                raise ValueError('3 values is invalid for padding specification.')

            return parsed_value if length > 1 else parsed_value[0]

        def list_of(the_type, v):
            try:
                values = ast.literal_eval(v)
                if not isinstance(values, collections.abc.Iterable):
                    raise ValueError()
                values = list(values)
                for idx, v in enumerate(values):
                    values[idx] = the_type(v)
                return values
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Invalid value, expected iterable and got: {v}") from e

        def optional(the_type, v):
            return None if v == 'None' else the_type(v)

        type_parsers = {
            # dimensions
            _types.Size: _textprocessing.parse_image_size,
            _types.Sizes: functools.partial(
                list_of, _textprocessing.parse_image_size),
            _types.OptionalSize: functools.partial(
                optional, _textprocessing.parse_image_size),
            _types.OptionalSizes: functools.partial(
                optional, functools.partial(list_of, _textprocessing.parse_image_size)),

            # paddings
            _types.Padding: parse_padding,
            _types.Paddings: functools.partial(list_of, parse_padding),
            _types.OptionalPadding: functools.partial(optional, parse_padding),
            _types.OptionalPaddings: functools.partial(optional, functools.partial(list_of, parse_padding)),

            # bool
            _types.Boolean: _types.parse_bool,
            _types.OptionalBoolean: functools.partial(optional, _types.parse_bool),

            # float
            _types.Float: float,
            _types.Floats: functools.partial(list_of, float),
            _types.OptionalFloat: functools.partial(optional, float),
            _types.OptionalFloats: functools.partial(optional, functools.partial(list_of, float)),

            # int
            _types.Integer: int,
            _types.Integers: functools.partial(list_of, int),
            _types.OptionalInteger: functools.partial(optional, int),
            _types.OptionalIntegers: functools.partial(optional, functools.partial(list_of, int)),

            # string
            _types.String: str,
            _types.Strings: functools.partial(list_of, str),
            _types.OptionalString: functools.partial(optional, str),
            _types.OptionalStrings: functools.partial(optional, functools.partial(list_of, str)),

            # name
            _types.Name: str,
            _types.Names: functools.partial(list_of, str),
            _types.OptionalName: functools.partial(optional, str),
            _types.OptionalNames: functools.partial(optional, functools.partial(list_of, str)),

            # uri
            _types.Uri: str,
            _types.Uris: functools.partial(optional, functools.partial(list_of, str)),
            _types.OptionalUri: functools.partial(optional, str),
            _types.OptionalUris: functools.partial(optional, functools.partial(list_of, str)),
        }

        for name, value in self._embedded_args.items():
            name = _textprocessing.dashdown(name)

            if is_forbidden(name, value):
                raise PromptEmbeddedArgumentError(
                    f'Setting diffusion argument "{name}" from the prompt is forbidden.')

            if not hasattr(on_object, name):
                raise PromptEmbeddedArgumentError(
                    f'Unknown embedded prompt argument: {name}')

            hint = hints.get(name, None)

            try:
                if hint in type_parsers:
                    value = type_parsers[hint](value)
                else:
                    raise PromptEmbeddedArgumentError(
                        f'Setting diffusion argument "{name}" from the prompt is forbidden.')
            except (ValueError, SyntaxError):
                raise PromptEmbeddedArgumentError(
                    f'Could not parse embedded prompt argument: {name}, value: {value}, into type: {hint}')

            if not validate_only:
                setattr(on_object, name, value)

    @staticmethod
    def _get_embeded_args(value: str, embedded_arg_names: list[str] | None):
        embedded_args = {}

        def find_arg(match):
            nonlocal embedded_args
            name = match.group(1)
            arg_value = match.group(2)

            if name == 'upscaler':
                # If upscaler is already set, convert it to a list
                if name in embedded_args:
                    if isinstance(embedded_args[name], list):
                        embedded_args[name].append(arg_value)
                    else:
                        embedded_args[name] = [embedded_args[name], arg_value]
                else:
                    embedded_args[name] = arg_value
            else:
                embedded_args[name] = arg_value

            return ' '

        if embedded_arg_names:
            cleaned_value = value
            for arg_name in embedded_arg_names:
                cleaned_value = re.sub(rf"\s*<\s*{arg_name}\s*:\s*(.*?)\s*>\s*", find_arg, cleaned_value)
        else:
            cleaned_value = re.sub(r"\s*<\s*([a-zA-Z_-][a-zA-Z0-9_-]+)\s*:\s*(.*?)\s*>\s*", find_arg, value)

        return cleaned_value.strip(), embedded_args

    @staticmethod
    def copy(prompt: 'Prompt'):
        """
        Return a copy of another prompt.

        :param prompt: The prompt to copy.
        :return: A copy of the provided prompt.
        """
        new_prompt = Prompt(
            positive=prompt.positive,
            negative=prompt.negative,
            delimiter=prompt.delimiter
        )

        new_prompt.copy_embedded_args_from(prompt)
        return new_prompt

    @staticmethod
    def parse(
            value: str,
            delimiter=';',
            parse_embedded_args: bool = True,
            embedded_arg_names: list[str] | None = None,
    ) -> 'Prompt':
        """
        Parse the positive and negative prompt from a string and return a prompt object.

        :param value: the string
        :param delimiter: The prompt delimiter character
        :param parse_embedded_args: parse embedded args? ``< arg: value >``
        :param embedded_arg_names: list of embedded argument names to parse, if ``None``, all are parsed.

        :raise ValueError: if value is ``None``

        :return: :py:class:`.Prompt` (returns self)
        """
        if value is None:
            raise ValueError('Input string may not be None.')

        weighter = None
        upscaler = None
        embedded_args = None

        if parse_embedded_args:
            value, embedded_args = Prompt._get_embeded_args(
                value, embedded_arg_names
            )
            weighter = embedded_args.get('weighter', None)
            if weighter is not None:
                embedded_args.pop('weighter')
            upscaler = embedded_args.get('upscaler', None)
            if upscaler is not None:
                embedded_args.pop('upscaler')

        parse = value.split(delimiter, 1)
        if len(parse) == 1:
            positive = parse[0].strip()
            negative = None
        elif len(parse) == 2:
            positive = parse[0].strip()
            negative = parse[1].strip()
        else:
            positive = None
            negative = None

        return Prompt(positive=positive,
                      negative=negative,
                      delimiter=delimiter,
                      weighter=weighter,
                      upscaler=upscaler,
                      embedded_args=embedded_args)


OptionalPrompt = typing.Optional[Prompt]
Prompts = collections.abc.Sequence[Prompt]
PromptOrPrompts = typing.Union[Prompt, collections.abc.Sequence[Prompt]]
OptionalPrompts = typing.Optional[Prompts]
OptionalPromptOrPrompts = typing.Optional[PromptOrPrompts]

__all__ = _types.module_all()
