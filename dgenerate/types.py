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
import inspect
import itertools
import types
import typing

import PIL.Image

import dgenerate.prompt as _prompt
import dgenerate.textprocessing

__doc__ = """
Commonly used static type definitions and utilities for introspecting on objects, functions, types, etc.
"""

Uri = str
Path = str
Name = str

Size = tuple[int, int]
Sizes = collections.abc.Sequence[Size]

OptionalSize = typing.Optional[Size]
OptionalSizes = typing.Optional[Sizes]

Coordinate = tuple[int, int]
OptionalCoordinate = typing.Optional[Coordinate]

Coordinates = collections.abc.Sequence[Coordinate]
OptionalCoordinates = typing.Optional[Coordinates]

Paths = collections.abc.Sequence[str]
OptionalPaths = typing.Optional[Paths]

Uris = collections.abc.Sequence[str]
OptionalUris = typing.Optional[Uris]

Names = collections.abc.Sequence[Name]
OptionalNames = typing.Optional[Names]

OptionalUri = typing.Optional[Uri]
OptionalPath = typing.Optional[Path]
OptionalName = typing.Optional[Name]

Integer = int
Integers = collections.abc.Sequence[int]

OptionalInteger = typing.Optional[int]
OptionalIntegers = typing.Optional[Integers]

Float = float
Floats = collections.abc.Sequence[float]

OptionalFloat = typing.Optional[float]
OptionalFloats = typing.Optional[Floats]

Version = tuple[int, int, int]

OptionalPrompt = typing.Optional[_prompt.Prompt]
Prompts = collections.abc.Sequence[_prompt.Prompt]

OptionalPrompts = typing.Optional[Prompts]
OptionalString = typing.Optional[str]

OptionalBoolean = typing.Optional[bool]

Images = collections.abc.Sequence[PIL.Image.Image]
MutableImages = collections.abc.MutableSequence[PIL.Image.Image]


def iterate_attribute_combinations(
        attribute_defs: collections.abc.Iterable[tuple[str, collections.abc.Iterable]],
        return_type: type) -> collections.abc.Iterator:
    """
    Iterate over every combination of attributes in a given class using a list of tuples mapping
    attribute names to a list of possible values.

    :param attribute_defs: sequence of tuple (attribute_name, [sequence of values])
    :param return_type: Construct this type and assign attribute values to it
    :return: an iterator over instances of the type mentioned in the my_class argument
    """

    def assign(ctx, dir_attr, name, val):
        if val is not None:
            if name in dir_attr:
                setattr(ctx, name, val)
            else:
                raise RuntimeError(f'{ctx.__class__.__name__} missing attribute "{name}"')

    for combination in itertools.product(*[d[1] for d in attribute_defs]):
        ctx_out = return_type()
        dir_attributes = set(get_public_attributes(ctx_out).keys())
        for idx, d in enumerate(attribute_defs):
            attr = d[0]
            if len(d) == 2:
                assign(ctx_out, dir_attributes, attr, combination[idx])
            else:
                assign(ctx_out, dir_attributes, attr, d[2](ctx_out, attr, combination[idx]))
        yield ctx_out


def class_and_id_string(obj) -> str:
    """
    Return a string formatted with an objects class name next to its memory ID.

    IE: `<ClassName: id_integer>`

    :param obj: the obj
    :return: formatted string
    """
    return f'<{obj.__class__.__name__}: {str(id(obj))}>'


def get_public_attributes(obj) -> dict[str, typing.Any]:
    """
    Get the public attributes (excluding functions) and their values from an obj.

    :param obj: the obj
    :return: dict of attribute names to values
    """
    return {k: getattr(obj, k) for k in dir(obj)
            if not k.startswith("_") and not callable(getattr(obj, k))}


def get_public_members(obj) -> dict[str, typing.Any]:
    """
    Get the public members (including functions) and their values from an obj.

    :param obj: the obj
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

    origin = typing.get_origin(hinted_type)

    if origin is comparison_type:
        return True

    return False


def is_type_or_optional(hinted_type, comparison_type):
    """
    Check if a hinted type is equal to a comparison type, even if the type hint is
    ``typing.Optional[]`` (compare the inside if necessary).

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
    Check if a hinted type is ``typing.Optional[]``

    :param hinted_type: The hinted type
    :return: bool
    """

    origin = typing.get_origin(hinted_type)

    if origin is typing.Union:
        union_args = typing.get_args(hinted_type)
        if len(union_args) == 2:
            for a in union_args:
                if is_type(a, type(None)):
                    return True
    return False


def get_type_of_optional(hinted_type: type, get_origin=True):
    """
    Get the first possible type for an ``typing.Optional[]`` type hint

    :param get_origin: Should the returned type be the origin type?
    :param hinted_type: The hinted type to extract from
    :return: the type, or ``None``
    """

    origin = typing.get_origin(hinted_type)
    if origin is typing.Union:
        union_args = typing.get_args(hinted_type)
        if len(union_args) == 2:
            for a in union_args:
                o = typing.get_origin(a)
                if o is None:
                    return a
                return o if get_origin else a
    return None


def is_typing_hint(obj):
    """
    Does a type hint object originate from the the builtin ``typing`` module?

    :param obj: type hint
    :return: ``True`` or ``False``
    """
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
    Get the fully qualified name of an obj or function or type/class

    :param obj: The obj
    :return: Fully qualified name
    """
    if inspect.isfunction(obj) or inspect.isclass(obj):
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
    Allows an obj ot have its attributes set from a dictionary
    or attributes taken from another obj with an overlapping set
    of attribute names.
    """

    def set_from(self,
                 obj: typing.Union[typing.Any, dict],
                 missing_value_throws: bool = True):
        """
        Set the attributes in this configuration obj from a dictionary or another obj
        possessing keys / attributes of the same name.

        :param obj: The obj, or dictionary
        :param missing_value_throws: whether to throw :py:class:`ValueError` if obj is missing
            an attribute that exist in this obj
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
                        raise ValueError(f'Source obj does not define: "{k}"')
                else:
                    setattr(self, k, source.get(k))
        return self


def get_accepted_args_with_defaults(func) -> \
        collections.abc.Iterator[typing.Union[tuple[str], tuple[str, typing.Any]]]:
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


def get_default_args(func) -> collections.abc.Iterator[tuple[str, typing.Any]]:
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
    :return: ``value`` or ``default_value``
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


def partial_deep_copy_container(container: typing.Union[list, dict, set]):
    """
    Partially copy nested containers, handles lists, dicts, and sets.

    :param container: top level container

    :return: structure with all containers copied
    """
    if isinstance(container, list):
        return [partial_deep_copy_container(v) for v in container]
    elif isinstance(container, dict):
        return {k: partial_deep_copy_container(v) for k, v in container.items()}
    elif isinstance(container, set):
        return {partial_deep_copy_container(v) for v in container}
    else:
        return container


def type_check_struct(obj,
                      attribute_namer: typing.Optional[typing.Callable[[str], str]] = None):
    """
    Preform some basic type checks on a struct like objects attributes using their type hints.

    :raise ValueError: on type checking failure

    :param obj: the object
    :param attribute_namer: function which names attributes an alternate name
    """

    def a_namer(attr_name):
        if attribute_namer:
            return attribute_namer(attr_name)
        return f'{obj.__class__.__name__}.{attr_name}'

    def _is_optional_tuple(name, value, type_hint):
        optional_type = [i for i in typing.get_args(type_hint) if is_type(i, tuple)][0]
        arg_len = len(typing.get_args(optional_type))
        if arg_len > 0:
            if value is not None and not (isinstance(value, tuple) and len(value) == arg_len):
                raise ValueError(
                    f'{a_namer(name)} must be None or a tuple of length {arg_len}, value was: {value}')

    def _is_tuple(name, value, type_hint):
        arg_len = len(typing.get_args(type_hint))
        if arg_len > 0:
            if not (isinstance(value, tuple) and len(value) == arg_len):
                raise ValueError(
                    f'{a_namer(name)} must be a tuple of length {arg_len}, value was: {value}')

    def _is_optional(value_type, name, value):
        if value is not None and not isinstance(value, value_type):
            raise ValueError(
                f'{a_namer(name)} must be None or type {fullname(value_type)}, value was: {value}')

    def _is_optional_literal(name, value, type_hint):
        optional_type = [i for i in typing.get_args(type_hint) if is_type(i, typing.Literal)][0]
        args = typing.get_args(optional_type)
        if value is not None and value not in args:
            raise ValueError(
                f'{a_namer(name)} must be None or one of these literal values: '
                f'{dgenerate.textprocessing.oxford_comma(args, "or")}')

    def _is_literal(name, value, type_hint):
        args = typing.get_args(type_hint)
        if value not in args:
            raise ValueError(
                f'{a_namer(name)} must be one of these literal values: '
                f'{dgenerate.textprocessing.oxford_comma(args, "or")}')

    def _is(value_type, name, value):
        if not isinstance(value, value_type):
            raise ValueError(
                f'{a_namer(name)} must be type {fullname(value_type)}, value was: {value}')

    # Detect some incorrect types
    for attr, hint in typing.get_type_hints(obj).items():
        v = getattr(obj, attr)
        if is_optional(hint):
            if is_type_or_optional(hint, tuple):
                _is_optional_tuple(attr, v, hint)
            elif is_type_or_optional(hint, typing.Literal):
                _is_optional_literal(attr, v, hint)
            else:
                _is_optional(get_type_of_optional(hint), attr, v)
        else:
            if is_type(hint, tuple):
                _is_tuple(attr, v, hint)
            elif is_type(hint, typing.Literal):
                _is_literal(attr, v, hint)
            else:
                _is(get_type(hint), attr, v)
