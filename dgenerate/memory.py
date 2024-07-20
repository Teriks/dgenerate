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
import ast
import collections.abc
import os

import asteval
import psutil
import torch

import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types

__doc__ = """
System memory information and memory constraint expressions.
"""


class MemoryConstraintSyntaxError(Exception):
    """
    Thrown by :py:func:`.memory_constraints` on syntax errors or
    if an expression returns a non-boolean value
    """
    pass


def memory_constraint_syntax_check(expression: str):
    """
    Syntax check an expression given to :py:func:`memory_constraints`

    :param expression: the expression string
    :raises MemoryConstraintSyntaxError: on syntax errors.
    """

    if len(expression) > 128:
        raise MemoryConstraintSyntaxError(f'Given expression "{expression[:24]} ..." is too long.')
    try:
        tree = ast.parse(expression)
        if tree.body:
            if not isinstance(tree.body[0], ast.Expr):
                raise MemoryConstraintSyntaxError(
                    f'Expression "{expression}" is invalid. '
                    f'Only simple expressions accepted, no control statements, etc.')
            if not isinstance(tree.body[0].value, (ast.BoolOp, ast.Compare)):
                raise MemoryConstraintSyntaxError(
                    f'Expression "{expression}" is invalid. '
                    'Only expressions returning boolean values accepted.')
    except SyntaxError as e:
        raise MemoryConstraintSyntaxError(
            f'Syntax error in expression "{expression}": {str(e).strip()}')


def memory_constraints(expressions: collections.abc.Iterable[str],
                       extra_vars: dict[str, int | float] | None = None,
                       mode=any,
                       pid: int | None = None) -> bool:
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
        * used_total_percent / utp (memory used by the process, as percent of total system memory, example: 25.4)
        * used_percent / up (memory used by the process, as a percent of used + available memory, example 75.4)
        * available / a (available memory remaining on the system in bytes that can be used without going to the swap)
        * total / t (total memory on the system in bytes)

    Example expressions:
        * ``used > gb(1)`` (when the process has used more than 1GB of memory)
        * ``used_total_percent > 25`` (when the process has used more than 25 percent of system memory)
        * ``used_percent > 25`` (when the process has used more than 25 percent of virtual memory available to it)
        * ``available < gb(2)`` (when the available memory on the system is less than 2GB)

    Expressions may not be longer than 128 characters. However, multiple expressions may be provided.

    :raise ValueError: if extra_vars overwrites a reserved variable name

    :raise MemoryConstraintSyntaxError: on syntax errors or if the return value
        of an expression is not a boolean value.

    :param expressions: a list of expressions, if expressions is ``None`` or empty this
        function will return ``False``.
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

    for expr in expressions:
        memory_constraint_syntax_check(expr)

    if pid is None:
        pid = os.getpid()

    p_info = psutil.Process(pid)
    used = p_info.memory_info().rss
    used_total_percent = p_info.memory_percent()

    mem_info = psutil.virtual_memory()

    available = mem_info.available
    total = mem_info.total

    used_percent = (used / (used + available)) * 100.0

    functions = {
        'gb': lambda x: x * 1000 ** 3,
        'mb': lambda x: x * 1000 ** 2,
        'kb': lambda x: x * 1000,
        'gib': lambda x: x * 1024 ** 3,
        'mib': lambda x: x * 1024 ** 2,
        'kib': lambda x: x * 1024
    }

    variables = {
        'used': used,
        'u': used,
        'used_percent': used_percent,
        'up': used_percent,
        'used_total_percent': used_total_percent,
        'utp': used_total_percent,
        'available': available,
        'a': available,
        'total': total,
        't': total
    }

    if extra_vars:
        for key, value in extra_vars.items():
            if key in variables or key in functions:
                raise ValueError(
                    f'extra_vars cannot redefine reserved attribute: {key}')
            variables[key] = value

    interpreter = asteval.Interpreter(
        minimal=True,
        symtable=variables.copy())

    if 'print' in interpreter.symtable:
        del interpreter.symtable['print']

    interpreter.symtable.update(functions)

    _messages.debug_log(
        f'{_types.fullname(memory_constraints)} constraint = '
        f'[{", ".join(_textprocessing.quote_spaces(expressions))}], '
        f'vars = {str(variables)}')

    try:
        value = mode(interpreter(
            e, raise_errors=True, show_errors=False) for e in expressions)
        if not isinstance(value, bool):
            raise MemoryConstraintSyntaxError('Memory constraint must return a boolean value.')
        return value
    except (Exception, NameError):
        raise MemoryConstraintSyntaxError(
            f'Memory constraint syntax error: {e}')


_MEM_FACTORS = {
    'b': 1,
    'kb': 1000,
    'mb': 1000 ** 2,
    'gb': 1000 ** 3,
    'kib': 1024,
    'mib': 1024 ** 2,
    'gib': 1024 ** 3,
}


def get_used_memory(unit='b', pid: int | None = None) -> int:
    """
    Get the memory used by a process in a selectable unit.

    :param unit: one of (case insensitive): b (bytes), kb (kilobytes),
        mb (megabytes), gb (gigabytes), kib (kibibytes),
        mib (mebibytes), gib (gibibytes)

    :param pid: The process PID to retrieve this information from, defaults to the current process.

    :return: Requested value.
    """

    if pid is None:
        pid = os.getpid()

    return psutil.Process(pid).memory_info().rss / _MEM_FACTORS[unit.strip().lower()]


def get_used_total_memory_percent(pid: int | None = None) -> float:
    """
    Get the percentage of memory used by a process as a percentage of total system memory.

    :param pid: PID of the process, defaults to the current process.
    :return: A whole percentage, for example: 25.4
    """

    if pid is None:
        pid = os.getpid()

    return psutil.Process(pid).memory_percent()


def get_used_memory_percent(pid: int | None = None) -> float:
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


def get_available_memory(unit='b'):
    """
    Get the available memory remaining on the system in a selectable unit.

    :param unit: one of (case insensitive): b (bytes), kb (kilobytes),
        mb (megabytes), gb (gigabytes), kib (kibibytes),
        mib (mebibytes), gib (gibibytes)

    :return: Requested value.
    """
    return psutil.virtual_memory().available / _MEM_FACTORS[unit.strip().lower()]


def get_total_memory(unit='b'):
    """
    Get the total physical memory on the system.

    :param unit: one of (case insensitive): b (bytes), kb (kilobytes),
        mb (megabytes), gb (gigabytes), kib (kibibytes),
        mib (mebibytes), gib (gibibytes)

    :return: Requested value.
    """

    return psutil.virtual_memory().total / _MEM_FACTORS[unit.strip().lower()]


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
        * Used Memory = :py:func:`.get_used_memory`
        * Available Memory = :py:func:`.get_available_memory`
        * Used Percent = :py:func:`.get_used_memory_percent`
        * Total Memory = :py:func:`.get_total_memory`
        * Used Percent Total = :py:func:`.get_used_total_memory_percent`


    :param pid: PID of the process to describe, defaults to the current process.
    :return: formatted string
    """

    if pid is None:
        pid = os.getpid()

    return (f'Used Memory (CPU Side): '
            f'{bytes_best_human_unit(get_used_memory(pid=pid))}, '
            f'Available Memory: '
            f'{bytes_best_human_unit(get_available_memory())}, '
            f'Used Percent: '
            f'{round(get_used_memory_percent(pid=pid), 2)}%, '
            f'Total Memory: '
            f'{bytes_best_human_unit(get_total_memory())}, '
            f'Used Total Percent: '
            f'{round(get_used_total_memory_percent(pid=pid), 2)}%')


def calculate_chunk_size(file_size):
    """
    Calculate the chunk size for downloading / copying
    a file based on the file size and available memory.

    :param file_size: The size of the file to be downloaded / copied.
    :return: The calculated chunk size.
    """
    # Get the total available virtual memory (in bytes)
    total_memory = psutil.virtual_memory().available

    # If the file size is less than 1% of the total memory, all in one chunk
    if file_size <= total_memory * 0.01:
        return file_size

    # If the file size is between 1% and 10% of the total memory, use 1% of the total memory as the chunk size
    elif file_size <= total_memory * 0.1:
        return int(total_memory * 0.01)

    # If the file size is larger than 10% of the total memory, use 0.1% of the total memory as the chunk size
    else:
        return int(total_memory * 0.001)


def cuda_memory_constraints(expressions: collections.abc.Iterable[str],
                            extra_vars: dict[str, int | float] | None = None,
                            mode=any,
                            device: str | torch.device = 'cuda:0') -> bool:
    """
    Evaluate a user boolean expression involving the CUDA device's memory in bytes,
    used memory percent, and available CUDA memory in bytes.

    If you pass a non cuda device identifier to this method, it will always return ``False``

    Available functions are:
        * kb(bytes to kilobytes)
        * mb(bytes to megabytes)
        * gb(bytes to gigabytes)
        * kib(bytes to kibibytes)
        * mib(bytes to mebibytes)
        * gib(bytes to gibibytes)

    Available values are:
        * used / u (memory currently used by the CUDA device in bytes)
        * used_total_percent / utp (memory used by the CUDA device, as percent of total CUDA memory, example: 25.4)
        * available / a (available memory remaining on the CUDA device in bytes that can be used)
        * total / t (total memory on the CUDA device in bytes)

    Example expressions:
        * ``used > gb(1)`` (when the device has used more than 1GB of memory)
        * ``used_total_percent > 25`` (when the device has used more than 25 percent of CUDA memory)
        * ``available < gb(2)`` (when the available memory on the device is less than 2GB)

    Expressions may not be longer than 128 characters. However, multiple expressions may be provided.

    :raise ValueError: if extra_vars overwrites a reserved variable name,
        or if ``device`` is not a ``str`` or ``torch.device`` object.

    :raise MemoryConstraintSyntaxError: on syntax errors or if the return value
        of an expression is not a boolean value.

    :param expressions: a list of expressions, if expressions is ``None`` or empty this
        function will return ``False``.
    :param extra_vars: extra integer or float variables
    :param mode: the standard library function 'any' (equating to OR all expressions) or
        the standard library function 'all' (equating to AND all expressions). The default
        is 'any' which ORs all expressions.
    :param device: CUDA device string or torch.device object, defaults to 'cuda:0'.
    :return: Boolean result of the expression
    """

    if not expressions:
        return False

    for expr in expressions:
        memory_constraint_syntax_check(expr)

    if isinstance(device, str):
        if not device.startswith('cuda'):
            return False
        if ':' in device:
            device_index = int(device.split(':')[1])
        else:
            device_index = 0  # default to device 0 if no index is specified
    elif isinstance(device, torch.device):
        if device.type != 'cuda':
            return False
        device_index = device.index if device.index is not None else 0
    else:
        raise ValueError('device must be a str or torch.device object.')

    total_memory = torch.cuda.get_device_properties(device_index).total_memory

    with torch.cuda.device(device_index):
        reserved_memory = torch.cuda.memory_reserved()
        allocated_memory = torch.cuda.memory_allocated()
        free_memory = total_memory - reserved_memory

    used = allocated_memory
    used_total_percent = (used / total_memory) * 100.0
    available = free_memory
    total = total_memory

    functions = {
        'gb': lambda x: x * 1000 ** 3,
        'mb': lambda x: x * 1000 ** 2,
        'kb': lambda x: x * 1000,
        'gib': lambda x: x * 1024 ** 3,
        'mib': lambda x: x * 1024 ** 2,
        'kib': lambda x: x * 1024
    }

    variables = {
        'used': used,
        'u': used,
        'used_total_percent': used_total_percent,
        'utp': used_total_percent,
        'available': available,
        'a': available,
        'total': total,
        't': total
    }

    if extra_vars:
        for key, value in extra_vars.items():
            if key in variables or key in functions:
                raise ValueError(
                    f'extra_vars cannot redefine reserved attribute: {key}')
            variables[key] = value

    interpreter = asteval.Interpreter(
        minimal=True,
        symtable=variables.copy())

    if 'print' in interpreter.symtable:
        del interpreter.symtable['print']

    interpreter.symtable.update(functions)

    _messages.debug_log(
        f'{_types.fullname(cuda_memory_constraints)} constraint = '
        f'[{", ".join(_textprocessing.quote_spaces(expressions))}], '
        f'vars = {str(variables)}')

    try:
        value = mode(interpreter(
            e, raise_errors=True, show_errors=False) for e in expressions)
        if not isinstance(value, bool):
            raise MemoryConstraintSyntaxError('Memory constraint must return a boolean value.')
        return value
    except (Exception, NameError) as e:
        raise MemoryConstraintSyntaxError(
            f'Memory constraint syntax error: {e}')


