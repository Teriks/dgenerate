import os
import typing

import psutil

import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types


class MemoryConstraintSyntaxError(Exception):
    """
    Thrown by :py:meth:`.memory_constraints` on syntax errors or
    if an expression returns a non-boolean value
    """
    pass


def memory_constraints(expressions: typing.Optional[typing.Union[str, list]],
                       extra_vars: typing.Optional[typing.Dict[str, typing.Union[int, float]]] = None,
                       mode=any,
                       pid: typing.Optional[int] = None) -> bool:
    """
    Evaluate a user boolean expression involving the processes used memory in bytes,
    used memory percent, and available system memory in bytes.

    Available functions are:
        * kb(bytes to kilobytes)
        * mb(bytes to megabytes)
        * gb(bytes to gigabytes)
        * kib(bytes to kibibytes)
        * mib(bytes to mebibytes)
        * gib(bytes to gibibytes)

    Available values are:
        * used / u (memory currently used by the process in bytes)
        * used_percent_total / upt (memory used by the process, as percent of total system memory, example: 25.4)
        * used_percent / up (memory used by the process, as a percent of used + available memory, example 75.4)
        * available / a (available memory remaining on the system in bytes that can be used without going to the swap)
        * total / t (total memory on the system in bytes)

    Example expressions:
        * ``used > gb(1)`` (when the process has used more than 1GB of memory)
        * ``used_percent_total > 25`` (when the process has used more than 25 percent of system memory)
        * ``used_percent > 25`` (when the process has used more than 25 percent of virtual memory available to it)
        * ``available < gb(2)`` (when the available memory on the system is less than 2GB)

    :raise: :py:exc:`ValueError` if extra_vars overwrites a reserved variable name

    :raise: :py:exc:`.MemoryConstraintSyntaxError` on syntax errors or if the return value
        of an expression is not a boolean value.

    :param expressions: a string containing an expression or a list of expressions,
        If expressions is None or empty this function will return False.
    :param extra_vars: extra integer or float variables
    :param mode: the standard library function 'any' (equating to OR all expressions) or
        the standard library function 'all' (equating to AND all expressions). The default
        is 'any' which ORs all expressions.
    :param pid: PID of the process from which to acquire the 'used' and 'used_percent' variable
        values from, defaults to the current process.
    :return: Boolean result of the expression
    """

    if not expressions:
        return False

    if isinstance(expressions, str):
        expressions = [expressions]

    if pid is None:
        pid = os.getpid()

    p_info = psutil.Process(pid)
    used = p_info.memory_info().rss
    used_percent_total = p_info.memory_percent()

    mem_info = psutil.virtual_memory()

    available = mem_info.available
    total = mem_info.total

    used_percent = (used / (used + available)) * 100.0

    eval_globals = {'gb': lambda x: x * 1000 ** 3,
                    'mb': lambda x: x * 1000 ** 2,
                    'kb': lambda x: x * 1000,
                    'gib': lambda x: x * 1024 ** 3,
                    'mib': lambda x: x * 1024 ** 2,
                    'kib': lambda x: x * 1024}

    eval_locals = {
        'used': used,
        'u': used,
        'used_percent': used_percent,
        'up': used_percent,
        'used_percent_total': used_percent_total,
        'upt': used_percent_total,
        'available': available,
        'a': available,
        'total': total,
        't': total
    }

    _messages.debug_log(
        f'{_types.fullname(memory_constraints)} constraint = '
        f'[{", ".join(_textprocessing.quote_spaces(expressions))}], '
        f'vars = {str(eval_locals)}')

    if extra_vars:
        for key, value in extra_vars.items():
            if key in eval_locals:
                raise ValueError(
                    f'extra_vars cannot redefine reserved attribute: {key}')
            eval_locals[key] = value

    try:
        value = mode(eval(e, eval_globals, eval_locals) for e in expressions)
        if not isinstance(value, bool):
            raise MemoryConstraintSyntaxError('Memory constraint must return a boolean value.')
        return value
    except Exception as e:
        raise MemoryConstraintSyntaxError(e)


_MEM_FACTORS = {
    'b': 1,
    'kb': 1000,
    'mb': 1000 ** 2,
    'gb': 1000 ** 3,
    'kib': 1024,
    'mib': 1024 ** 2,
    'gib': 1024 ** 3,
}


def get_used_memory(measure='b', pid: typing.Optional[int] = None) -> int:
    """
    Get the memory used by a process in a selectable unit.

    :param measure: one of (case insensitive): b (bytes), kb (kilobytes),
        mb (megabytes), gb (gigabytes), kib (kibibytes),
        mib (mebibytes), gib (gibibytes)

    :param pid: The process PID to retrieve this information from, defaults to the current process.

    :return: Requested value.
    """

    if pid is None:
        pid = os.getpid()

    return psutil.Process(pid).memory_info().rss / _MEM_FACTORS[measure.strip().lower()]


def get_used_memory_percent_total(pid: typing.Optional[int] = None) -> float:
    """
    Get the percentage of memory used by a process as a percentage of total system memory.

    :param pid: PID of the process, defaults to the current process.
    :return: A whole percentage, for example: 25.4
    """

    if pid is None:
        pid = os.getpid()

    return psutil.Process(pid).memory_percent()


def get_used_memory_percent(pid: typing.Optional[int] = None) -> float:
    """
    Get the percentage of memory used by a process as a percentage of
    already used memory plus available virtual memory.

    :param pid: PID of the process, defaults to the current process.
    :return: A whole percentage, for example: 25.4
    """

    if pid is None:
        pid = os.getpid()

    p_info = psutil.Process(pid)
    used = p_info.memory_info().rss
    mem_info = psutil.virtual_memory()
    available = mem_info.available

    return (used / (used + available)) * 100.0


def get_available_memory(measure='b'):
    """
    Get the available memory remaining on the system in a selectable unit.

    :param measure: one of (case insensitive): b (bytes), kb (kilobytes),
        mb (megabytes), gb (gigabytes), kib (kibibytes),
        mib (mebibytes), gib (gibibytes)

    :return: Requested value.
    """
    return psutil.virtual_memory().available / _MEM_FACTORS[measure.strip().lower()]


def get_total_memory(measure='b'):
    """
    Get the total physical memory on the system.

    :param measure: one of (case insensitive): b (bytes), kb (kilobytes),
        mb (megabytes), gb (gigabytes), kib (kibibytes),
        mib (mebibytes), gib (gibibytes)

    :return: Requested value.
    """

    return psutil.virtual_memory().total / _MEM_FACTORS[measure.strip().lower()]


def bytes_best_human_unit(byte_count: int, delimiter='') -> str:
    """
    Return a string for humans from a byte count using an appropriate unit: IE 1KB, 1MB, 1GB etc.

    :param delimiter: add this string between the value and the unit
    :param byte_count: the byte count
    :return: formatted string
    """
    gb = byte_count / 1000 ** 3
    mb = byte_count / 1000 ** 2
    kb = byte_count / 1000

    if gb > 1:
        return f'{round(gb, 2)}{delimiter}GB'
    if mb > 1:
        return f'{round(mb, 2)}{delimiter}MB'
    if kb > 1:
        return f'{round(kb, 2)}{delimiter}KB'

    return f'{byte_count}{delimiter}B'


def memory_use_debug_string(pid=None):
    """
    Return a debug string using describing the memory consumption of a process and also
    available system memory.

    Example:
        "Used Memory: 465.25MB, Available Memory: 50.94GB, Used Percent: 0.91%, Total Memory: 68.64GB, Used Total Percent: 0.68%"

    Where:
        * Used Memory = :py:meth:`.get_used_memory`
        * Available Memory = :py:meth:`.get_available_memory`
        * Used Percent = :py:meth:`.get_used_memory_percent`
        * Total Memory = :py:meth:`.get_total_memory`
        * Used Percent Total = :py:meth:`.get_used_memory_percent_total`


    :param pid: PID of the process to describe, defaults to the current process.
    :return: formatted string
    """

    if pid is None:
        pid = os.getpid()

    return (f'Used Memory: '
            f'{bytes_best_human_unit(get_used_memory(pid=pid))}, '
            f'Available Memory: '
            f'{bytes_best_human_unit(get_available_memory())}, '
            f'Used Percent: '
            f'{round(get_used_memory_percent(pid=pid), 2)}%, '
            f'Total Memory: '
            f'{bytes_best_human_unit(get_total_memory())}, '
            f'Used Total Percent: '
            f'{round(get_used_memory_percent_total(pid=pid), 2)}%')
