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
import numbers


def args_cache_key(args_dict, custom_hashes=None):
    def value_hash(obj):
        if isinstance(obj, dict):
            return '{' + args_cache_key(obj) + '}'
        elif isinstance(obj, list):
            return f'[{",".join(args_cache_key(o) if o is isinstance(o, (dict, list)) else value_hash(o) for o in obj)}]'
        elif obj is None or isinstance(obj, (str, numbers.Number)):
            return str(obj)
        else:
            return f'<{obj.__class__.__name__}:{str(id(obj))}>'

    if custom_hashes:
        # Only for the top level, let user control recursion
        return ','.join(f'{k}={value_hash(v) if k not in custom_hashes else custom_hashes[k](v)}' for k, v in sorted(args_dict.items()))
    else:
        return ','.join(f'{k}={value_hash(v)}' for k, v in sorted(args_dict.items()))


def memoize(cache, exceptions=None, hasher=args_cache_key, on_hit=None, on_create=None):
    if exceptions is None:
        exceptions = {}

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

            unprovided_args = [(arg, idx - args_len) for idx, arg in enumerate(args_after_positionals) if
                               arg not in kwargs]

            if spec.defaults is not None:
                try:
                    kwargs.update({k: spec.defaults[v] for k, v in unprovided_args})
                except IndexError:
                    raise ValueError(f'Missing positional argument.')
            else:
                if unprovided_args:
                    raise ValueError(
                        f'Missing arguments: {", ".join(x[0] for x in unprovided_args)}')

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

            on_create(cache_key, val)
            return val

        return wrapper

    return decorate
