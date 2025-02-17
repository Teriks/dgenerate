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
import re
import typing
import dgenerate.types as _types
import dgenerate.textprocessing as _textprocessing


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
        else:
            return ''

    @property
    def positive(self) -> str:
        return self._positive

    @property
    def negative(self) -> str:
        return self._negative

    @property
    def weighter(self) -> _types.OptionalUri:
        return self._weighter

    @property
    def upscaler(self) -> _types.OptionalUriOrUris:
        return self._upscaler

    @property
    def embedded_args(self) -> list[tuple[str, str]]:
        return list(self._embedded_args.items())

    def __repr__(self):
        return f"'{str(self)}'"

    def copy_embedded_args_from(self, prompt: 'Prompt'):
        self._embedded_args = prompt._embedded_args.copy()

    def set_embedded_args_on(
            self,
            on_object: typing.Any,
            forbidden_checker: typing.Callable[[str, typing.Any], bool] = None,
            validate_only: bool = False
    ):
        hints = typing.get_type_hints(on_object)

        def verboten(name, value):
            if forbidden_checker is not None:
                return forbidden_checker(name, value)
            return False

        for name, value in self._embedded_args.items():
            name = _textprocessing.dashdown(name)

            if verboten(name, value):
                raise PromptEmbeddedArgumentError(
                    f'Setting diffusion argument "{name}" '
                    'from the prompt is forbidden.'
                )

            if hasattr(on_object, name):
                hint = hints[name]

                try:
                    if hint is _types.OptionalSize:
                        value = _textprocessing.parse_dimensions(value)
                    elif hint is _types.OptionalPadding:
                        value = _textprocessing.parse_dimensions(value)
                        if len(value) == 1:
                            value = value
                    elif any(hint is h for h in {_types.Integer, _types.OptionalFloat, _types.OptionalFloats}):
                        value = ast.literal_eval(value)
                        if isinstance(value, typing.Iterable):
                            value = list(value)
                            for idx, v in value:
                                value[idx] = float(v)
                        else:
                            value = float(value)
                    elif any(hint is h for h in {_types.Integer, _types.OptionalInteger, _types.OptionalIntegers}):
                        value = ast.literal_eval(value)
                        if isinstance(value, typing.Iterable):
                            value = list(value)
                            for idx, v in enumerate(value):
                                value[idx] = int(v)
                        else:
                            value = int(value)
                    elif any(hint is h for h in {_types.OptionalUri, _types.OptionalUris}):
                        try:
                            value = ast.literal_eval(value)
                            if isinstance(value, typing.Iterable):
                                value = list(value)
                                for idx, v in enumerate(value):
                                    value[idx] = str(v)
                        except SyntaxError:
                            value = value
                    else:
                        raise PromptEmbeddedArgumentError(
                            f'Setting diffusion argument "{name}" '
                            'from the prompt is forbidden.'
                        )
                except (ValueError, SyntaxError):
                    raise PromptEmbeddedArgumentError(
                        f'Could not parse embedded prompt '
                        f'argument: {name}, value: {value}'
                    )

                if not validate_only:
                    setattr(on_object, name, value)
            else:
                raise PromptEmbeddedArgumentError(
                    f'Unknown embedded prompt argument: {name}'
                )

    @staticmethod
    def _get_embeded_args(value):
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

        cleaned_value = re.sub(r"\s*<\s*([a-zA-Z_-][a-zA-Z0-9_-]+)\s*:\s*(.*?)\s*>\s*", find_arg, value)

        return cleaned_value.strip(), embedded_args

    @staticmethod
    def parse(value: str, delimiter=';', parse_embedded_args: bool = True) -> 'Prompt':
        """
        Parse the positive and negative prompt from a string and return a prompt object.

        :param value: the string
        :param delimiter: The prompt delimiter character
        :param parse_embedded_args: parse embedded args? ``< arg: value >``

        :raise ValueError: if value is ``None``

        :return: :py:class:`.Prompt` (returns self)
        """
        if value is None:
            raise ValueError('Input string may not be None.')

        weighter = None
        upscaler = None
        embedded_args = None

        if parse_embedded_args:
            value, embedded_args = Prompt._get_embeded_args(value)
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
OptionalPrompts = typing.Optional[Prompts]

__all__ = _types.module_all()
