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
import enum
import inspect
import numbers
import typing

import dgenerate.messages as _messages
import dgenerate.types as _types

__doc__ = """
Function memoization wrapper and associated hashing tools.
"""


def args_cache_key(args_dict: dict[str, typing.Any],
                   custom_hashes: dict[str, typing.Callable[[typing.Any], str]] = None):
    """
    Generate a cache key for a functions arguments to use for memoization.

    :param args_dict: The args dictionary of the function
    :param custom_hashes: Custom hash functions for specific argument names if needed
    :return: string
    """

    def value_hash(obj):
        if isinstance(obj, dict):
            return '{' + args_cache_key(obj) + '}'
        elif isinstance(obj, list):
            return f'[{",".join(args_cache_key(o) if o is isinstance(o, (dict, list)) else value_hash(o) for o in obj)}]'
        elif obj is None or isinstance(obj, (str, numbers.Number, enum.Enum)):
            return str(obj)
        else:
            return _types.class_and_id_string(obj)

    if custom_hashes:
        # Only for the top level, let user control recursion
        return ','.join(f'{k}={value_hash(v) if k not in custom_hashes else custom_hashes[k](v)}' for k, v in
                        sorted(args_dict.items()))
    else:
        return ','.join(f'{k}={value_hash(v)}' for k, v in sorted(args_dict.items()))


def memoize(cache: dict[str, typing.Any],
            exceptions: set[str] = None,
            hasher: typing.Callable[[dict[str, typing.Any]], str] = args_cache_key,
            on_hit: typing.Callable[[str, typing.Any], None] = None,
            on_create: typing.Callable[[str, typing.Any], None] = None):
    """
    Decorator used to Memoize a function using a dictionary as a value cache.

    :param cache: The dictionary to serve as a cache
    :param exceptions: Function arguments to ignore
    :param hasher: Responsible for hashing arguments and argument values
    :param on_hit: Called on cache hit for the wrapped function
    :param on_create: Called on cache miss for the wrapped function
    :return: decorator
    """
    if exceptions is None:
        exceptions = set()

    def _on_hit(key, hit):
        if on_hit is not None:
            on_hit(key, hit)

    def _on_create(key, new):
        if on_create is not None:
            on_create(key, new)

    def decorate(func):
        def wrapper(*args, **kwargs):
            spec = inspect.getfullargspec(func)
            args_len = len(args)

            # Set Default Arguments

            args_after_positionals = spec.args[args_len:]

            unprovided_args = [arg for arg in args_after_positionals if arg not in kwargs]

            defaults = {arg[0]: arg[1] for arg in _types.get_default_args(func)}

            if defaults:
                try:
                    kwargs.update({k: defaults[k] for k in unprovided_args})
                except KeyError as e:
                    raise ValueError(f'Missing positional argument: {str(e).strip()}.')
            else:
                if unprovided_args:
                    raise ValueError(
                        f'Missing arguments: {", ".join(unprovided_args)}')

            # provided_arguments
            provided_arguments = spec.args[:args_len]

            named_provided_arguments = {k: args[idx] for idx, k in enumerate(provided_arguments)}

            # Add keyword arguments and defaults
            named_provided_arguments.update(kwargs)

            # Cache key for all arguments except those excluded
            cache_args = {k: v for k, v in named_provided_arguments.items() if k not in exceptions}
            cache_key = hasher(cache_args)

            cache_hit = cache.get(cache_key, None)
            if cache_hit is not None:
                _on_hit(cache_key, cache_hit)
                return cache_hit

            val = func(**named_provided_arguments)
            cache[cache_key] = val

            _on_create(cache_key, val)
            return val

        return wrapper

    return decorate


def simple_cache_hit_debug(title: str, cache_key: str, cache_hit: typing.Any):
    """
    Basic cache hit debug message for :py:func:`.memoize` decorator **on_hit** parameter.

    Messages are printed using :py:func:`dgenerate.messages.debug_log`

    Example:
        ``on_hit=lambda key, hit: simple_cache_hit_debug("My Object", key, hit)``


    Debug Prints:
        ``Cache Hit, Loaded My Object: (fully qualified name of hit object), Cache Key: (key)``


    :param title: Object Title
    :param cache_key: cache key
    :param cache_hit: cached object
    """
    _messages.debug_log(f'Cache Hit, Loaded {title}: "{_types.fullname(cache_hit)}",',
                        f'Cache Key: "{cache_key}"')


def simple_cache_miss_debug(title: str, cache_key: str, new: typing.Any):
    """
    Basic cache hit debug message for :py:func:`.memoize` decorator **on_create** parameter.

    Messages are printed using :py:func:`dgenerate.messages.debug_log`

    Example:
        ``on_create=lambda key, hit: simple_cache_miss_debug("My Object", key, hit)``


    Debug Prints:
        ``Cache Miss, Created My Object: (fully qualified name of new object), Cache Key: (key)``


    :param title: Object Title
    :param cache_key: cache key
    :param new: newly created object
    """
    _messages.debug_log(f'Cache Miss, Created {title}: "{_types.fullname(new)}",',
                        f'Cache Key: "{cache_key}"')


def struct_hasher(obj: typing.Any,
                  custom_hashes: dict[str, typing.Callable[[typing.Any], str]] = None) -> str:
    """
    Create a hash string from a simple objects public attributes.

    :param obj: the object
    :param custom_hashes: Custom hash functions for specific attribute names if needed
    :return: string
    """
    return '{' + args_cache_key(args_dict=_types.get_public_attributes(obj), custom_hashes=custom_hashes) + '}'
