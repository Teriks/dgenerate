import inspect
import numbers


def args_cache_key(args_dict):
    def value_hash(obj):
        if isinstance(obj, dict):
            return '{' + args_cache_key(obj) + '}'
        elif obj is None or isinstance(obj, (str, numbers.Number)):
            return str(obj)
        else:
            return f'<{obj.__class__.__name__}:{str(id(obj))}>'

    return ','.join(f'{k}={value_hash(v)}' for k, v in sorted(args_dict.items()))


def memoize(cache, exceptions=None, hasher=args_cache_key, on_hit=None):
    if exceptions is None:
        exceptions = {}

    def _on_hit(key, hit):
        if on_hit is not None:
            on_hit(key, hit)

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
            cache_key = hasher({k: v for k, v in named_provided_arguments.items() if k not in exceptions})

            cache_hit = cache.get(cache_key, None)
            if cache_hit is not None:
                _on_hit(cache_key, cache_hit)
                return cache_hit

            val = func(**named_provided_arguments)
            cache[cache_key] = val
            return val

        return wrapper

    return decorate
