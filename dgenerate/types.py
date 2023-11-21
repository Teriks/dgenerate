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
import inspect
import types
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
OptionalUriOrUris = typing.Union[str, Uris, None]

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


OptionalBoolean = typing.Optional[bool]


def class_and_id_string(obj) -> str:
    """
    Return a string formatted with an objects class name next to its memory ID.

    IE: `<ClassName: id_integer>`

    :param obj: the object
    :return: formatted string
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

def is_typing_hint(obj):
    if obj is None:
        return False
    return obj.__module__ == 'typing'

def parse_bool(string_or_bool: typing.Union[str, bool]):
    """
    Parse a case insensitive boolean value from a string, for example "true" or "false"

    Additionally, values that are already bool are passed through.

    :raises ValueError: on parse failure.

    :param string_or_bool: the string, or a bool value
    :return: python boolean type equivalent
    """
    if isinstance(string_or_bool, bool):
        return string_or_bool

    try:
        return {'true': True, 'false': False}[string_or_bool.strip().lower()]
    except KeyError:
        raise ValueError(
            f'"{string_or_bool}" is not a boolean value')


def fullname(obj):
    """
    Get the fully qualified name of an object or function

    :param obj: The object
    :return: Fully qualified name
    """
    if inspect.isfunction(obj):
        mod = inspect.getmodule(obj)
        if mod is not None:
            return mod.__name__ + '.' + obj.__qualname__
        else:
            return obj.__qualname__

    module = obj.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return obj.__class__.__qualname__
    return module + '.' + obj.__class__.__qualname__


class SetFromMixin:
    """
    Allows an object ot have its attributes set from a dictionary
    or attributes taken from another object with an overlapping set
    of attribute names.
    """

    def set_from(self,
                 obj: typing.Union[typing.Any, dict],
                 missing_value_throws: bool = True):
        """
        Set the attributes in this configuration object from a dictionary or another object
        possessing keys / attributes of the same name.

        :param obj: The object, or dictionary
        :param missing_value_throws: whether to throw :py:class:`ValueError` if obj is missing
            an attribute that exist in this object
        :return: self
        """

        if isinstance(obj, dict):
            source = obj
        else:
            source = get_public_attributes(obj)

        for k, v in get_public_attributes(self).items():
            if not callable(v):
                if k not in source:
                    if missing_value_throws:
                        raise ValueError(f'Source object does not define: "{k}"')
                else:
                    setattr(self, k, source.get(k))
        return self


def get_accepted_args_with_defaults(func) -> \
        typing.Iterator[typing.Union[typing.Tuple[str], typing.Tuple[str, typing.Any]]]:
    """
    Get the argument signature of a simple function with any default values present.

    :param func: the function
    :return: an iterator over tuples of length 1 or 2,
        length 2 indicates a default argument value is present. (name,) or (name, value)
    """

    spec = inspect.getfullargspec(func)
    sig_args = spec.args[1:]
    defaults_cnt = len(spec.defaults) if spec.defaults else 0
    no_defaults_before = len(sig_args) - defaults_cnt

    default_idx = 0
    for idx, arg in enumerate(sig_args):
        if idx < no_defaults_before:
            yield arg,
        else:
            yield arg, spec.defaults[default_idx]

            default_idx += 1


def get_default_args(func) -> typing.Iterator[typing.Tuple[str, typing.Any]]:
    """
    Get a list of default arguments from a simple function with their default values.

    :param func: the function
    :return: iterator over tuples of length 2 (name, value)
    """
    for arg in get_accepted_args_with_defaults(func):
        if len(arg) > 1:
            yield arg


def default(value, default_value):
    """
    Return value if value is not None, otherwise default
    :param value:
    :param default_value:
    :return: bool
    """
    return value if value is not None else default_value


def module_all():
    """
    Return the name of all public non-module type global objects inside the current module.

    Can be used for __all__

    :return: list of names
    """
    all_names = []
    local_stack = inspect.stack()[1][0]
    for name, value in local_stack.f_globals.items():
        if not name.startswith('_') and not isinstance(value, types.ModuleType):
            all_names.append(name)
    return all_names
