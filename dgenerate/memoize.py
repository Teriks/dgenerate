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
import contextlib
import enum
import gc
import inspect
import numbers
import typing

import dgenerate.messages as _messages
import dgenerate.types as _types

__doc__ = """
Function memoization wrapper and associated hashing tools.
"""


class CachedObjectMetadata:
    """
    Represents optional metadata for an object cached in :py:class:`ObjectCache`

    This object is a namespace, you can access the metadata attributes using the dot operator.
    """

    def __init__(self, skip=False, **kwargs):
        self._metadata = kwargs
        self.skip = skip

    def __str__(self):
        return str(self._metadata)

    def __repr__(self):
        return repr(self._metadata)

    def __getattr__(self, item):
        try:
            return self._metadata[item]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        if key == "_metadata":
            # Handle setting _metadata directly
            super().__setattr__(key, value)
        else:
            self._metadata[key] = value


class ObjectCacheKeyError(KeyError):
    """
    Raised when an object cannot be found in the object cache.

    Or upon adding an object to the cache that already exists in the cache.
    """
    pass


class ObjectCache:
    """
    A cache for objects with unique memory IDs.
    """

    def __init__(self, name):
        self.__cache = dict()
        self.__id_to_metadata: dict[int, CachedObjectMetadata] = dict()
        self.__id_to_key: dict[int, str] = dict()
        self.__id_to_extra_ids: dict[int, list[int]] = dict()
        self.name = name
        self.__on_un_cache = []
        self.__on_cache = []
        self.__on_clear = []

    def register_on_clear(self, action: typing.Callable[['ObjectCache'], None]):
        """
        Register a callback for when the cache is cleared.

        :param action: callback action, accepts :py:class:`ObjectClass` as the only parameter
        """
        self.__on_clear.append(action)

    def register_on_cache(self, action: typing.Callable[['ObjectCache', typing.Any], None]):
        """
        Register a callback for when an object enters the cache.

        :param action: callback action, accepts :py:class:`ObjectClass`, and the object entering cache.
        """
        self.__on_cache.append(action)

    def register_on_un_cache(self, action: typing.Callable[['ObjectCache', typing.Any], None]):
        """
        Register a callback for when an object exits the cache.

        :param action: callback action, accepts :py:class:`ObjectClass`, and the object exiting cache.
        """
        self.__on_un_cache.append(action)

    def values(self):
        """
        Return all cached objects in a list.
        """
        return list(self.__cache.values())

    def __len__(self):
        return len(self.__cache)

    def clear(self, collect=True):
        """
        Clear the cache and trigger callbacks.

        :param collect: call ``gc.collect()`` ?
        """
        for action in self.__on_un_cache:
            for cached_object in self.__cache.values():
                action(self, cached_object)

        self.__cache.clear()
        self.__id_to_metadata.clear()
        self.__id_to_key.clear()
        self.__id_to_extra_ids.clear()

        for action in self.__on_clear:
            action(self)

        if collect:
            gc.collect()

    def get(self, key: str, default: typing.Any | None = None):
        """
        Get an object by its cache key.

        :param key: the key
        :param default: default value if non-existant
        :return: the object, or default
        """
        return self.__cache.get(key, default)

    def get_hash_key(self, value) -> str:
        """
        Get the cache key used for an object that exists in the cache.

        :param value: The object

        :raise ObjectCacheKeyError: if the object does not exist in the cache.

        :return: hash key
        """
        identity = id(value)
        if identity in self.__id_to_key:
            return self.__id_to_key[identity]

        raise ObjectCacheKeyError(
            f'Object: {value}, not a member of cache: {self.name}')

    def cache(self,
              key,
              value,
              metadata: typing.Optional[CachedObjectMetadata] = None,
              extra_identities: list[typing.Callable[[typing.Any], typing.Any]] = None):
        """
        Add an object to the cache. It must not already exist in the cache.

        This method triggers callbacks.

        :param key: Cache key
        :param value: Cached object
        :param metadata: Object metadata
        :param extra_identities: Functions which return members of the cached object,
            these members can be used to identify the object in cache.

        :raise ObjectCacheKeyError: If the object already exists in the cache.
        """
        identity = id(value)
        if identity in self.__id_to_key:
            raise ObjectCacheKeyError(f'Object: {value}, is already in object cache: {self.name}')

        if metadata is not None and metadata.skip:
            return

        _messages.debug_log(f'Entering object cache "{self.name}": {_types.fullname(value)}')

        self.__cache[key] = value
        self.__id_to_key[identity] = key

        extra_identities = [id(i(value)) for i in extra_identities] if extra_identities is not None else []

        if extra_identities:
            self.__id_to_extra_ids[identity] = extra_identities

        for extra_identity in extra_identities:
            self.__id_to_key[extra_identity] = key

        if metadata is not None:

            self.__id_to_metadata[identity] = metadata

            for extra_identity in extra_identities:
                self.__id_to_metadata[extra_identity] = metadata

        for action in self.__on_cache:
            action(self, value)

    def un_cache(self, value):
        """
        Remove an object from the cache by its reference.

        This method triggers callbacks.

        :param value: Object
        :raise ObjectCacheKeyError: If the object is not a member of the cache.
        """
        identity = id(value)
        if identity in self.__id_to_key:
            for action in self.__on_un_cache:
                action(self, value)

            key = self.__id_to_key[identity]
            del self.__cache[key]
            del self.__id_to_key[identity]

            if identity in self.__id_to_extra_ids:
                for extra_id in self.__id_to_extra_ids[identity]:
                    del self.__id_to_key[extra_id]
                    del self.__id_to_metadata[extra_id]
                del self.__id_to_extra_ids[identity]

            if identity in self.__id_to_metadata:
                del self.__id_to_metadata[identity]
        else:
            raise ObjectCacheKeyError(
                f'Object: {value}, not a member of cache: {self.name}')

    def get_metadata(self, value) -> CachedObjectMetadata | None:
        """
        Get any metadata that exists for an object in the cache, or ``None``
        if no metadata exists for the object.

        :param value: The object
        :return: :py:class:`CachedObjectMetadata`
        """
        identity = id(value)

        if identity not in self.__id_to_key:
            raise ObjectCacheKeyError(
                f'Object: {value}, not a member of cache: {self.name}')

        if identity in self.__id_to_metadata:
            return self.__id_to_metadata[identity]

        return None


_CACHES: dict[str, ObjectCache] = dict()


def clear_object_caches(collect: bool = True):
    """
    Call :py:meth:`ObjectCache.clear` on every object cache.

    :param collect: call ``gc.collect()`` ?
    """
    for cache in _CACHES.values():
        cache.clear(collect=False)

    if collect:
        gc.collect()


def get_object_cache_names() -> list[str]:
    """
    Return a list of active object cache names.

    :return: list of names
    """
    return list(_CACHES.keys())


ObjectCacheType = typing.TypeVar('ObjectCacheType', bound=ObjectCache)


def create_object_cache(cache_name, cache_type: type[ObjectCacheType] = ObjectCache) -> ObjectCacheType:
    """
    Create a new object cache.

    :param cache_name: Cache name.
    :param cache_type: :py:class:`ObjectCache` implementation.

    :raise RuntimeError: If the cache name is taken.

    :return: :py:class:`ObjectCache`
    """
    if cache_name in _CACHES:
        raise RuntimeError(f'Object Cache: {cache_name} already exists.')
    new_cache = cache_type(cache_name)
    _CACHES[cache_name] = new_cache
    return new_cache


def get_object_cache(cache_name) -> ObjectCache:
    """
    Return an object cache by name.

    :param cache_name: the cache name.

    :raise RuntimeError: if the cache name does not exist.

    :return: :py:class:`ObjectCache`
    """
    if cache_name not in _CACHES:
        raise RuntimeError(f'Object Cache: {cache_name} does not exist.')
    return _CACHES[cache_name]


def _pop_tuple_item(tpl, index):
    item = tpl[index]
    new_tpl = tpl[:index] + tpl[index + 1:]
    return new_tpl, item


def _is_user_defined_class_instance(obj):
    # Check if the class is a user-defined class (not from builtins or standard library)
    return inspect.isclass(obj.__class__) and obj.__class__.__module__ != "builtins"


def _assert_memory_id(obj, func):
    if not _is_user_defined_class_instance(obj):
        raise RuntimeError(f'Memoized function: {func} expected to return '
                           f'user defined object with unique memory address, '
                           f'returned: {obj.__class__}')


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


_MEMOIZE_DISABLED = False


@contextlib.contextmanager
def disable_memoization_context(disabled=True):
    """
    Context manager which allows you to temporarily disable memoization on
    functions decorated with :py:func:`.memoize`

    The default action is to disable memoization.

    :param disabled: You can use this parameter to allow user configuration of
    the memoization state without writing separate code outside of this context
    block for that. Setting `disable=False` leaves memoization enabled in the context.
    """
    global _MEMOIZE_DISABLED

    _MEMOIZE_DISABLED = disabled
    try:
        yield
    finally:
        _MEMOIZE_DISABLED = False


def memoize(cache: dict[str, typing.Any] | ObjectCache,
            exceptions: set[str] = None,
            skip_check: typing.Callable[[typing.Any], bool] = None,
            hasher: typing.Callable[[dict[str, typing.Any]], str] = args_cache_key,
            extra_identities: list[typing.Callable[[typing.Any], typing.Any]] = None,
            on_hit: typing.Callable[[str, typing.Any], None] = None,
            on_create: typing.Callable[[str, typing.Any], None] = None):
    """
    Decorator used to Memoize a function using a dictionary as a value cache.

    :param cache: The dictionary or :py:class:`ObjectCache` to serve as a cache
    :param exceptions: Function arguments to ignore
    :param skip_check: Check the created object itself to determine if caching should proceed,
        should return ``True`` if you want to skip caching of the object.
    :param hasher: Responsible for hashing arguments and argument values
    :param extra_identities: List of functions which return member objects of
        the cached object, that can be used as extra identifiers for the object
        in cache, the returned objects can be used to retrieve cache metadata or the
        hash key for the parent object in the cache via the methods on :py:class:`ObjectCache`.
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

    def _skip_check(new):
        if skip_check is not None:
            return skip_check(new)
        return False

    def decorate(func):
        def wrapper(*args, **kwargs):
            global _MEMOIZE_DISABLED
            if _MEMOIZE_DISABLED:
                val = func(*args, **kwargs)
                if isinstance(val, tuple):
                    meta_index = [idx for idx, o in enumerate(val) if isinstance(o, CachedObjectMetadata)]
                    if len(meta_index) > 1:
                        raise RuntimeError(
                            f'Memoized function: {func} returned too many metadata objects.')
                    if len(meta_index) > 0:
                        val, _ = _pop_tuple_item(val, meta_index[0])
                        val = val if len(val) > 1 else val[0]
                return val

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
                    raise ValueError(f'Missing positional argument: {str(e).strip()}.') from e
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

            if isinstance(cache, ObjectCache):
                if isinstance(val, tuple):
                    meta_index = [idx for idx, o in enumerate(val) if isinstance(o, CachedObjectMetadata)]
                    if len(meta_index) > 1:
                        raise RuntimeError(
                            f'Memoized function: {func} returned too many metadata objects.')
                    if len(meta_index) > 0:
                        val, metadata = _pop_tuple_item(val, meta_index[0])
                        val = val if len(val) > 1 else val[0]
                        _assert_memory_id(val, func)
                        if _skip_check(val):
                            return val

                        if metadata.skip:
                            return val

                        cache.cache(cache_key, val, metadata, extra_identities)
                    else:
                        _assert_memory_id(val, func)
                        if _skip_check(val):
                            return val
                        cache.cache(cache_key, val, extra_identities)
                else:
                    _assert_memory_id(val, func)
                    if _skip_check(val):
                        return val
                    cache.cache(cache_key, val, extra_identities)
            else:
                if _skip_check(val):
                    return val
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
                  custom_hashes: dict[str, typing.Callable[[typing.Any], str]] = None,
                  exclude: set[str] | None = None,
                  properties_only: bool = False) -> str:
    """
    Create a hash string from a simple objects public attributes.

    :param obj: the object
    :param custom_hashes: Custom hash functions for specific attribute names if needed
    :param exclude: Exclude attributes by name
    :param properties_only: Only include public properties, not methods or other attributes
    :return: string
    """

    if exclude is None:
        exclude = set()

    reflect_method = \
        _types.get_public_properties if \
        properties_only else \
        _types.get_public_attributes

    return '{' + args_cache_key(
        args_dict={k: v for k, v in reflect_method(obj).items()
                   if k not in exclude},
        custom_hashes=custom_hashes) + '}'

def property_hasher(obj: typing.Any,
                    custom_hashes: dict[str, typing.Callable[[typing.Any], str]] = None,
                    exclude: set[str] | None = None):
    """
    Create a hash string from an objects public decorated properties.

    :param obj: the object
    :param custom_hashes: Custom hash functions for specific property names if needed
    :param exclude: Exclude property by name
    :return: string
    """
    return struct_hasher(obj, custom_hashes, exclude, True)
