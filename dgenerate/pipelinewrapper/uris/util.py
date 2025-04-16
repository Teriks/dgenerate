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
import itertools

import diffusers
import torch

import dgenerate.memoize as _memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.uris.bnbquantizeruri as _bnbquantizeruri
import dgenerate.pipelinewrapper.uris.torchaoquantizeruri as _torchaoquantizeruri
import dgenerate.torchutil as _torchutil
import dgenerate.types as _types
import dgenerate.textprocessing as _textprocessing
import dgenerate.pipelinewrapper.util as _pipelinewrapper_util


def uri_hash_with_parser(parser, exclude: set[str] | None = None):
    """
    Create a hash function from a particular URI parser function that hashes a URI string.

    The URI is parsed and then the object that results from parsing is hashed with
    :py:func:`dgenerate.memoize.struct_hasher`.

    If the parser returns a string, it is regarded as the hash value instead of being
    passed to :py:func:`dgenerate.memoize.struct_hasher`.

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

        return _memoize.struct_hasher(parser_result,
                                      exclude=exclude)

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


def get_uri_accepted_args_schema(uri_cls: type):
    """
    Get the accepted arguments as a schema dict for a URI class.

    Keyed by argument ``name``, content keys include:

    ``default`` contains any default value, this key may not exist if the argument has no default value.

    ``types`` contains all accepted types for the argument in string form.

    ``optional`` can the argument accept the value ``None``?

    :return: dict
    """
    schema = dict()
    for name, arg in inspect.signature(uri_cls.__init__).parameters.items():
        if name == 'self':
            continue

        name = _textprocessing.dashup(name)

        entry = {}
        schema[name] = entry

        def _type_name(t):
            return (str(t) if t.__module__ != 'builtins' else t.__name__).strip()

        def _resolve_union(t):
            name = _type_name(t)
            if _types.is_union(t):
                return set(itertools.chain.from_iterable(
                    [_resolve_union(t) for t in arg.annotation.__args__]))
            return [name]

        type_name = _type_name(arg.annotation)

        if _types.is_union(arg.annotation):
            union_args = _resolve_union(arg.annotation)
            if 'NoneType' in union_args:
                entry['optional'] = True
                union_args.remove('NoneType')

            entry['types'] = list(sorted(union_args))

        else:
            entry['optional'] = False
            entry['types'] = [type_name]

        if arg.default is not inspect.Parameter.empty:
            schema[name]['default'] = arg.default
    return schema


def get_quantizer_uri_class(uri: str, exception: type[Exception] = ValueError):
    """
    Get the URI parser class needed for a particular quantizer URI
    :param uri: The URI
    :param exception: Exception type to raise on unsupported quantization backend.
    :return: Class from :py:mod:`dgenerate.pipelinewrapper.uris`
    """

    concept = uri.split(';')[0].strip()
    if concept in {'bnb', 'bitsandbytes'}:
        if not diffusers.utils.is_bitsandbytes_available():
            raise exception(
                f'Cannot load quantization backend bitsandbytes, '
                f'as bitsandbytes is not installed.')
        return _bnbquantizeruri.BNBQuantizerUri
    elif concept == 'torchao':
        if not diffusers.utils.is_torchao_available():
            raise exception(
                f'Cannot load quantization backend torchao, '
                f'as torchao is not installed.')
        return _torchaoquantizeruri.TorchAOQuantizerUri
    else:
        raise exception(f'Unknown quantization backend: {concept}')


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
