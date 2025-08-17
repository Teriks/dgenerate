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
import inspect
import itertools
import typing

import diffusers
import torch

import dgenerate.memoize as _memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.uris.bnbquantizeruri as _bnbquantizeruri
import dgenerate.pipelinewrapper.uris.sdnqquantizeruri as _sdnqquantizeruri
import dgenerate.pipelinewrapper.util as _pipelinewrapper_util
import dgenerate.textprocessing as _textprocessing
import dgenerate.torchutil as _torchutil
import dgenerate.types as _types


def uri_hash_with_parser(parser, exclude: set[str] | None = None):
    """
    Create a hash function from a particular URI parser function that hashes a URI string.

    The URI is parsed and then the object that results from parsing is hashed with
    :py:func:`dgenerate.memoize.property_hasher`.

    If the parser returns a string, it is regarded as the hash value instead of being
    passed to :py:func:`dgenerate.memoize.property_hasher`.

    :param parser: The URI parser function
    :param exclude: URI argument names to exclude from hashing
    :return: a hash function compatible with :py:func:`dgenerate.memoize.memoize`
    """

    def hasher(path):
        if not path:
            # None gets hashed to 'None' string
            return str(path)

        parser_result = parser(path)

        if isinstance(parser_result, str):
            # user returned string, override the hash value
            return parser_result

        return _memoize.property_hasher(
            parser_result,
            exclude=exclude
        )

    return hasher


def uri_list_hash_with_parser(parser, exclude: set[str] | None = None):
    """
    Create a hash function from a particular URI parser function that hashes a list of URIs.

    :param parser: The URI parser function
    :param exclude: URI argument names to exclude from hashing
    :return: a hash function compatible with :py:func:`dgenerate.memoize.memoize`

    """

    def hasher(paths):
        if not paths:
            return '[]'

        return '[' + ','.join(uri_hash_with_parser(parser, exclude=exclude)(path) for path in paths) + ']'

    return hasher


def get_uri_names(uri_cls: type) -> list:
    """
    Return human names / loading names for a URI type.

    :param uri_cls: the URI class
    :return: list of names, guaranteed to be `len() > 0`
    """

    result = getattr(uri_cls, 'NAMES', [uri_cls.__name__])
    if not isinstance(result, (list, tuple, set)):
        result = [result]
    else:
        result = list(result)
    return result


def get_uri_help(uri_cls: type, wrap_width: int | None = None) -> str | None:
    """
    Get the help text for a URI class.

    This function introspects the ``help`` method of the URI class, if it exists.

    :return: str
    """

    if not hasattr(uri_cls, '__init__'):
        raise RuntimeError(f'URI class: {uri_cls.__name__} has no __init__ method.')

    if hasattr(uri_cls, 'help') and callable(uri_cls.help):

        arg_descriptors = []

        for arg, properties in get_uri_accepted_args_schema(uri_cls).items():

            if 'types' in properties:
                type_string = ': ' + ' | '.join(properties['types'])
                if 'optional' in properties and properties['optional']:
                    type_string += ' | None'
            else:
                type_string = ''

            if not 'default' in properties:
                arg_descriptors.append(arg + type_string)
            else:
                default_value = properties['default']
                if isinstance(default_value, str):
                    default_value = _textprocessing.quote(default_value)
                arg_descriptors.append(f'{arg}{type_string} = {default_value}')

        if arg_descriptors:
            args_part = f'\n{" " * 4}arguments:\n{" " * 8}{(chr(10) + " " * 8).join(arg_descriptors)}\n'
        else:
            args_part = '\n'

        help_str: str = uri_cls.help()
        name = getattr(uri_cls, 'NAMES', uri_cls.__name__)
        if isinstance(name, (list, tuple, set)):
            name = ' | '.join(name)

        if help_str:
            wrap = \
                _textprocessing.wrap_paragraphs(
                    inspect.cleandoc(help_str),
                    initial_indent=' ' * 4,
                    subsequent_indent=' ' * 4,
                    width=_textprocessing.long_text_wrap_width()
                    if wrap_width is None else wrap_width)

            return name + f':{args_part}\n' + wrap
        else:
            return name + f':{args_part}'

    return None


def _validate_file_arg_metadata(cls, d: dict[str, typing.Any]):
    if not isinstance(d, dict):
        raise RuntimeError(
            f'URI metadata {cls.__name__}.FILE_ARGS descriptor '
            f'values must be dicts, not {type(d).__name__}.')


    if 'mode' not in d:
        raise RuntimeError(
            f'URI metadata {cls.__name__}.FILE_ARGS descriptor missing "mode" key.'
        )

    mode = d['mode']

    if isinstance(mode, list):
        if any(m not in {'in', 'out', 'dir'} for m in mode):
            raise RuntimeError(
                f'URI metadata {cls.__name__}.FILE_ARGS descriptor "mode" '
                f'values must be "in", "out", or "dir", not: {d["mode"]}.'
            )
        else:
            if 'in' in mode and 'out' in mode:
                raise RuntimeError(
                    f'URI metadata {cls.__name__}.FILE_ARGS descriptor "mode" '
                    f'cannot contain both "in" and "out" at the same time: {d["mode"]}.'
                )
    elif isinstance(mode, str) and mode not in {'in', 'out', 'dir'}:
        raise RuntimeError(
            f'URI metadata {cls.__name__}.FILE_ARGS descriptor "mode" '
            f'values must be "in", "out", or "dir", not: {d["mode"]}.'
        )

    dir_only = (isinstance(mode, list) and all(m == 'dir' for m in mode)) or mode == 'dir'

    if 'filetypes' not in d and not dir_only:
        raise RuntimeError(
            f'URI metadata {cls.__name__}.FILE_ARGS descriptor missing '
            f'"filetypes" key and not in directory only "mode".'
        )

    if not dir_only:
        if not isinstance(d['filetypes'], (list, tuple)) or not all(
                isinstance(ft, (tuple, list)) and len(ft) == 2 and isinstance(ft[0], str)
                and isinstance(ft[1], (tuple, list))
                for ft in d['filetypes']
        ):
            raise RuntimeError(
                f'URI metadata {cls.__name__}.FILE_ARGS descriptor "filetypes" '
                f'must be a list/tuple of lists/tuples, where each tuple contains '
                f'a descriptor string and a list of file extensions, not: {d["filetypes"]}.'
            )

    return d

def get_uri_accepted_args_schema(uri_cls: type) -> dict:
    """
    Get the accepted arguments as a schema dict for a URI class.

    This function introspects the ``__init__`` method of the URI class and returns an argument schema.

    The ``HIDE_ARGS`` static class attribute (list or set) can be used to hide arguments from the schema,
    in the same fashion as the dgenerate plugin API. This is useful for arguments that are not part of the
    parseable URI, but are used internally as supplemental arguments when parsing the URI. Arguments
    such as ``model_type`` (indicating parent model type for sub-models) etc. are examples
    of such arguments.

    The ``OPTION_ARGS`` static class attribute (dict) can be used to specify a that an arguments
    values consists of a set of valid options, such as specific strings.

    The ``FILE_ARGS`` static class attribute (dict) can be used to specify that an argument
    accepts a file or directory, and can contain metadata about the filetypes.

    The static metadata attributes for URI classes are defined identically to plugins,
    aside from the absent ``loaded_by_name`` API. The schema output is identical to
    dgenerate plugin schema output.

    Keyed by argument ``name``, content keys include:

    ``default`` contains any default value, this key may not exist if the argument has no default value.

    ``types`` contains all accepted types for the argument in string form.

    ``optional`` can the argument accept the value ``None``?

    ``options`` contains a list of valid options for the argument, if the argument is an option argument
    annotated with the ``OPTION_ARGS`` static class attribute. This list may contain ``None`` if the argument
    can accept the value ``None``, i.e. it is optional.

    ``files`` contains a dict with metadata about what sort of file types (and/or directory)
    the argument accepts if applicable.

    :param uri_cls: The URI class to introspect, must have an ``__init__`` method.

    :return: dict
    """

    args_hidden = getattr(uri_cls, 'HIDE_ARGS', set())
    args_options = getattr(uri_cls, 'OPTION_ARGS', dict())
    args_files = getattr(uri_cls, 'FILE_ARGS', dict())

    if not isinstance(args_hidden, collections.abc.Collection):
        raise RuntimeError(
            f'URI class: {uri_cls.__name__} HIDE_ARGS must be a collection, got: {type(args_hidden).__name__}'
        )

    if not isinstance(args_options, collections.abc.Mapping):
        raise RuntimeError(
            f'URI class: {uri_cls.__name__} OPTION_ARGS must be a dict / map, got: {type(args_hidden).__name__}'
        )

    if not isinstance(args_files, collections.abc.Mapping):
        raise RuntimeError(
            f'URI class: {uri_cls.__name__} FILE_ARGS must be a dict / map, got: {type(args_files).__name__}'
        )

    args_hidden = {_textprocessing.dashup(a) for a in args_hidden}
    args_options = {_textprocessing.dashup(k): list(v) for k, v in args_options.items()}
    args_files = {_textprocessing.dashup(k): _validate_file_arg_metadata(uri_cls, v) for k, v in args_files.items()}

    schema = dict()
    for name, arg in inspect.signature(uri_cls.__init__).parameters.items():
        if name == 'self':
            continue

        name = _textprocessing.dashup(name)

        if name in args_hidden:
            # skip hidden arguments
            continue

        entry = {}
        schema[name] = entry

        def _type_name(t):
            if getattr(t,'__module__','') == 'collections.abc':
                n = getattr(t, '__name__', '')
                if n in {'Collection', 'Sequence', 'MutableSequence', 'Iterable'}:
                    # simplified compatible type name for collections
                    return 'list'
                if n in {'Mapping', 'MutableMapping', 'Dict'}:
                    # simplified compatible type name for mappings
                    return 'dict'
                if n in {'Set', 'MutableSet'}:
                    # simplified compatible type name for sets
                    return 'set'


            return (str(t) if t.__module__ != 'builtins' else t.__name__).strip()

        # all arguments that accept enum in a union should also accept a string
        def _not_enum(t):
            return not (inspect.isclass(t) and issubclass(t, enum.Enum))

        def _resolve_union(t):
            n = _type_name(t)
            if _types.is_union(t):
                return set(itertools.chain.from_iterable(
                    [_resolve_union(t) for t in arg.annotation.__args__ if _not_enum(t)]))
            return [n]

        type_name = _type_name(arg.annotation)

        if _types.is_union(arg.annotation):
            union_args = _resolve_union(arg.annotation)
            if 'NoneType' in union_args:
                entry['optional'] = True
                union_args.remove('NoneType')

            if union_args:
                entry['types'] = list(sorted(union_args))
            else:
                raise RuntimeError(
                    f'pipelinewrapper.uri union type argument with no '
                    f'valid types found in it, class: {uri_cls.__name__}.')

        else:
            entry['optional'] = False
            entry['types'] = [type_name]

        if arg.default is not inspect.Parameter.empty:
            schema[name]['default'] = arg.default

        if name in args_options:
            if not isinstance(args_options[name], collections.abc.Collection):
                raise RuntimeError(
                    f'URI class: {uri_cls.__name__} OPTION_ARGS for argument: {name} '
                    f'must be a collection, got: {type(args_options[name]).__name__}'
                )

            schema[name]['options'] = args_options[name]

        if name in args_files:
            schema[name]['files'] = args_files[name]

    return schema


class UnknownQuantizerName(Exception):
    """Raised upon referencing an unknown quantization backend name."""
    pass


def get_quantizer_uri_class(uri: str, exception: type[Exception] = UnknownQuantizerName):
    """
    Get the URI parser class needed for a particular quantizer URI
    :param uri: The URI
    :param exception: Exception type to raise on unsupported quantization backend.
    :return: Class from :py:mod:`dgenerate.pipelinewrapper.uris`
    """

    concept = uri.split(';')[0].strip()
    if concept in get_uri_names(_bnbquantizeruri.BNBQuantizerUri):
        if not diffusers.utils.is_bitsandbytes_available():
            raise exception(
                f'Cannot load quantization backend bitsandbytes, '
                f'as bitsandbytes is not installed.')
        return _bnbquantizeruri.BNBQuantizerUri
    elif concept in get_uri_names(_sdnqquantizeruri.SDNQQuantizerUri):
        return _sdnqquantizeruri.SDNQQuantizerUri
    else:
        raise exception(f'Unknown quantization backend: {concept}')


def quantizer_help(names: _types.Names,
                   throw: bool =False,
                   log_error: bool =True) -> int:
    """
    Implements ``--quantizer-help`` command line argument.

    :param names: backend names, may be an empty list.
    :param throw: Raise :py:exc:`UnknownQuantizerName` instead of handling unknown names?
    :param log_error: Log errors to ``stderr``?
    :return: return code, 0 for success, 1 for failure
    """
    qnames = ['bnb', 'sdnq']

    if len(names) == 0:
        available = ('\n' + ' ' * 4).join(_textprocessing.quote(name) for name in qnames)
        _messages.log(
            f'Available quantizers:\n\n{" " * 4}{available}')
        return 0

    help_strs = []
    for name in names:
        try:
            help_strs.append(get_uri_help(get_quantizer_uri_class(name)))
        except UnknownQuantizerName:
            if log_error:
                _messages.error(
                    f'A quantizer backend with the name of "{name}" could not be found.'
                )
            if throw:
                raise
            return 1
    for help_str in help_strs:
        _messages.log(help_str + '\n', underline=True)
    return 0


def _patch_module_to_for_sized_cache(cache: _memory.SizedConstrainedObjectCache, module):
    """
    Patch ``module.to`` to update cache CPU side memory usage based on the modules current location.

    :param cache: Module cache
    :param module: The module to patch
    """

    if hasattr(module, '_DGENERATE_ORIGINAL_TO_CACHE_PATCHED'):
        return  # Already patched

    module._DGENERATE_ORIGINAL_TO_CACHE_PATCHED = module.to

    import dgenerate.pipelinewrapper as _pipelinewrapper

    def new_to(*args, **kwargs):
        if _pipelinewrapper_util.is_loaded_in_8bit_bnb(module):
            _messages.debug_log(f'Module "{module.__name__}" is loaded in 8bit, .to() call is a no-op.')
            return module

        device = None
        if len(args) == 1:
            if isinstance(args[0], (str, torch.device)):
                device = torch.device(args[0])
        elif 'device' in kwargs:
            device = torch.device(kwargs['device'])

        if device is not None and not _torchutil.devices_equal(
                _pipelinewrapper.get_torch_device(module), device):
            try:
                metadata = cache.get_metadata(module)
                if device.type != 'cpu':
                    cache.size -= metadata.size

                    _messages.debug_log(
                        f'Cached {_types.class_and_id_string(module)} Size = '
                        f'{metadata.size} Bytes '
                        f'({_memory.bytes_best_human_unit(metadata.size)}) '
                        f'is leaving CPU side memory, '
                        f'cache size is now '
                        f'{cache.size} Bytes '
                        f'({_memory.bytes_best_human_unit(cache.size)})')
                else:
                    cache.size += metadata.size

                    _messages.debug_log(
                        f'Cached {_types.class_and_id_string(module)} Size = '
                        f'{metadata.size} Bytes '
                        f'({_memory.bytes_best_human_unit(metadata.size)}) '
                        f'is entering CPU side memory, '
                        f'cache size is now '
                        f'{cache.size} Bytes '
                        f'({_memory.bytes_best_human_unit(cache.size)})')

            except _memoize.ObjectCacheKeyError:
                # does not exist in the cache
                pass

        return module._DGENERATE_ORIGINAL_TO_CACHE_PATCHED(*args, **kwargs)

    module.to = new_to


__all__ = _types.module_all()
