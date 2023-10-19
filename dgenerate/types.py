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

import dgenerate.prompt as _prompt

Uri = str
Path = str
Name = str

Size = typing.Tuple[int, int]
Sizes = typing.List[Size]

OptionalSize = typing.Optional[Size]
OptionalSizes = typing.Optional[Sizes]

Coordinate = typing.Tuple[int, int]
OptionalCoordinate = typing.Optional[Coordinate]

CoordinateList = typing.List[Coordinate]
OptionalCoordinateList = typing.Optional[CoordinateList]

Paths = typing.List[str]
OptionalPaths = typing.Optional[Paths]

Uris = typing.List[str]
OptionalUris = typing.Optional[Uris]

Names = typing.List[Name]
OptionalNames = typing.Optional[Names]

OptionalUri = typing.Optional[Uri]
OptionalPath = typing.Optional[Path]
OptionalName = typing.Optional[Name]

Integer = int
Integers = typing.List[int]

OptionalInteger = typing.Optional[int]
OptionalIntegers = typing.Optional[Integers]

Float = float
Floats = typing.List[float]

OptionalFloat = typing.Optional[float]
OptionalFloats = typing.Optional[Floats]

Version = typing.Tuple[int, int, int]

OptionalPrompt = typing.Optional[_prompt.Prompt]
Prompts = typing.List[_prompt.Prompt]

OptionalPrompts = typing.Optional[Prompts]
OptionalString = typing.Optional[str]


def class_and_id_string(obj) -> str:
    """
    Return a string formated with an objects class name next to its memory ID.

    IE: `<ClassName: id_integer>`

    :param obj: the object
    :return: formated string
    """
    return f'<{obj.__class__.__name__}: {str(id(obj))}>'


def get_public_attributes(obj) -> typing.Dict[str, typing.Any]:
    """
    Get the public attributes (excluding functions) and their values from an object.

    :param obj: the object
    :return: dict of attribute names to values
    """
    return {k: getattr(obj, k) for k in dir(obj)
            if not k.startswith("_") and not callable(getattr(obj, k))}


def get_public_members(obj) -> typing.Dict[str, typing.Any]:
    """
    Get the public members (including functions) and their values from an object.

    :param obj: the object
    :return: dict of attribute names to values
    """
    return {k: getattr(obj, k) for k in dir(obj)
            if not k.startswith("_")}


def is_type(hinted_type, comparison_type):
    """
    Check if a hinted type is equal to a comparison type.

    :param hinted_type: The hinted type
    :param comparison_type: The type to check for
    :return: bool
    """

    if hinted_type is comparison_type:
        return True

    if typing.get_origin(hinted_type) is comparison_type:
        return True

    return False


def is_type_or_optional(hinted_type, comparison_type):
    """
    Check if a hinted type is equal to a comparison type, even if the type hint is
    optional (compare the inside if necessary).

    :param hinted_type: The hinted type
    :param comparison_type: The type to check for
    :return: bool
    """
    if hinted_type == comparison_type:
        return True

    origin = typing.get_origin(hinted_type)

    if origin is comparison_type:
        return True
    if origin is typing.Union:
        union_args = typing.get_args(hinted_type)
        if len(union_args) == 2:
            for a in union_args:
                if is_type(a, comparison_type):
                    return True
    return False


def get_type(hinted_type):
    """
    Get the basic type of hinted type

    :param hinted_type: the type hint
    :return: bool
    """

    o = typing.get_origin(hinted_type)
    if o is None:
        return hinted_type
    return o


def is_optional(hinted_type):
    """
    Check if a hinted type is optional

    :param hinted_type: The hinted type
    :param comparison_type: The type to check for
    :return: bool
    """

    origin = typing.get_origin(hinted_type)

    if origin is typing.Union:
        union_args = typing.get_args(hinted_type)
        if len(union_args) == 2:
            for a in union_args:
                if is_type(a, None):
                    return True
    return False


def get_type_of_optional(hinted_type):
    """
    Get the first possible type for an optional type hint

    :param hinted_type: The hinted type to extract from
    :return: the type, or None
    """

    origin = typing.get_origin(hinted_type)
    if origin is typing.Union:
        union_args = typing.get_args(hinted_type)
        if len(union_args) == 2:
            for a in union_args:
                o = typing.get_origin(a)
                if o is None:
                    return a
                return o
    return None


def fullname(obj):
    """
    Get the fully qualified name of an object.
    :param obj: The object
    :return: Fully qualified name
    """
    module = obj.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return obj.__class__.__name__
    return module + '.' + obj.__class__.__name__
