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

import torch
import dgenerate.memory as _memory
import dgenerate.memoize as _memoize
import dgenerate.messages as _messages
import dgenerate.types as _types


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


def _patch_module_to_for_sized_cache(cache: _memory.SizedConstrainedObjectCache, module):
    """
    Patch ``module.to`` to update cache CPU side memory usage based on the modules current location.

    :param cache: Module cache
    :param module: The module to patch
    """
    old_to = module.to

    # prevent recursive import
    from dgenerate.pipelinewrapper import get_torch_device

    def new_to(*args, **kwargs):
        device = None
        if len(args) == 1:
            if isinstance(args[0], (str, torch.device)):
                device = torch.device(args[0])
        elif 'device' in kwargs:
            device = torch.device(kwargs['device'])

        if device is not None and get_torch_device(module) != device:
            try:
                metadata = cache.get_metadata(module)
                if device.type != 'cpu':
                    cache.size -= metadata.size

                    _messages.debug_log(
                        f'Cached {_types.class_and_id_string(module)} Size = '
                        f'{metadata} Bytes '
                        f'({_memory.bytes_best_human_unit(metadata.size)}) '
                        f'is leaving CPU side memory, '
                        f'cache size is now '
                        f'{cache.size} Bytes '
                        f'({_memory.bytes_best_human_unit(cache.size)})')
                else:
                    cache.size += metadata.size

                    _messages.debug_log(
                        f'Cached {_types.class_and_id_string(module)} Size = '
                        f'{metadata} Bytes '
                        f'({_memory.bytes_best_human_unit(metadata.size)}) '
                        f'is entering CPU side memory, '
                        f'cache size is now '
                        f'{cache.size} Bytes '
                        f'({_memory.bytes_best_human_unit(cache.size)})')

            except _memoize.ObjectCacheKeyError:
                # does not exist in the cache
                pass

        old_to(*args, **kwargs)

    module.to = new_to


__all__ = _types.module_all()
