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
import typing

import psutil
import torch

import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
import dgenerate.memoize as _memoize
import dgenerate.eval as _eval
import dgenerate.torchutil as _torchutil

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
            # noinspection PyUnresolvedReferences
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

    interpreter = _eval.standard_interpreter(
        symtable=_eval.safe_builtins() | variables.copy()
    )

    interpreter.symtable.update(functions)

    _messages.debug_log(
        f'CPU MEMORY CONSTRAINT TEST: {_types.fullname(memory_constraints)} constraint = '
        f'[{", ".join(_textprocessing.quote_spaces(expressions))}], '
        f'vars = {str(variables)}, mode={mode.__name__}')

    try:
        value = mode(interpreter(
            e, raise_errors=True, show_errors=False) for e in expressions)
        if not isinstance(value, bool):
            raise MemoryConstraintSyntaxError('Memory constraint must return a boolean value.')

        _messages.debug_log(f'CPU MEMORY CONSTRAINT RESULT: {value}')
        return value
    except (Exception, NameError) as e:
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


def get_used_memory(unit: str = 'b', pid: int | None = None):
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


def get_available_memory(unit: str = 'b'):
    """
    Get the available memory remaining on the system in a selectable unit.

    :param unit: one of (case insensitive): b (bytes), kb (kilobytes),
        mb (megabytes), gb (gigabytes), kib (kibibytes),
        mib (mebibytes), gib (gibibytes)

    :return: Requested value.
    """
    return psutil.virtual_memory().available / _MEM_FACTORS[unit.strip().lower()]


def get_total_memory(unit: str = 'b'):
    """
    Get the total physical memory on the system.

    :param unit: one of (case insensitive): b (bytes), kb (kilobytes),
        mb (megabytes), gb (gigabytes), kib (kibibytes),
        mib (mebibytes), gib (gibibytes)

    :return: Requested value.
    """

    return psutil.virtual_memory().total / _MEM_FACTORS[unit.strip().lower()]


def is_supported_gpu_device(device: str | torch.device) -> bool:
    """
    Check if a device is a supported GPU device (CUDA or XPU) that can be used
    with the GPU memory functions in this module.

    MPS statistics are unsupported due to using a unified memory model.

    :param device: The device to check (string like 'cuda:0', 'xpu:1' or torch.device object)
    :return: True if the device is a supported GPU device, False otherwise
    """
    if isinstance(device, str):
        device = device.strip()
        return device.startswith('cuda') or device.startswith('xpu')
    elif isinstance(device, torch.device):
        return device.type in ('cuda', 'xpu')
    else:
        return False


def _parse_gpu_device(device: str | torch.device) -> tuple[str, int]:
    """
    Parse a GPU device identifier and return the device type and index.
    
    :param device: The device to parse (string like 'cuda:0', 'xpu:1' or torch.device object)
    :return: Tuple of (device_type, device_index)
    :raises ValueError: If device is not a valid GPU device identifier
    """
    if isinstance(device, str):
        device = device.strip()
        device_type = device.split(':')[0]
        if ':' in device:
            device_index = int(device.split(':')[1].strip())
        else:
            device_index = 0  # default to device 0 if no index is specified
    elif isinstance(device, torch.device):
        device_type = device.type
        device_index = device.index if device.index is not None else 0
    else:
        raise ValueError('device must be a str or torch.device object.')
    
    return device_type, device_index


def get_gpu_total_memory(device: str | torch.device, unit: str = 'b'):
    """
    Return the total memory processed by a GPU device.

    Non GPU devices always return 0.

    :param device: The device.
    :param unit: one of (case insensitive): b (bytes), kb (kilobytes),
        mb (megabytes), gb (gigabytes), kib (kibibytes),
        mib (mebibytes), gib (gibibytes)
    :return: Requested value.
    """
    if not is_supported_gpu_device(device):
        return 0
    
    device_type, device_index = _parse_gpu_device(device)

    if device_type == 'cuda':
        return torch.cuda.get_device_properties(device_index).total_memory / _MEM_FACTORS[unit.strip().lower()]
    elif device_type == 'xpu':
        return torch.xpu.get_device_properties(device_index).total_memory / _MEM_FACTORS[unit.strip().lower()]


def get_gpu_allocated_memory(device: str | torch.device, unit: str = 'b'):
    """
    Return the total memory allocated on a GPU device.

    Non GPU devices always return 0.

    :param device: The device.
    :param unit: one of (case insensitive): b (bytes), kb (kilobytes),
        mb (megabytes), gb (gigabytes), kib (kibibytes),
        mib (mebibytes), gib (gibibytes)
    :return: Requested value.
    """
    if not is_supported_gpu_device(device):
        return 0
    
    device_type, device_index = _parse_gpu_device(device)

    if device_type == 'cuda':
        with torch.cuda.device(device_index):
            return torch.cuda.memory_allocated() / _MEM_FACTORS[unit.strip().lower()]
    elif device_type == 'xpu':
        with torch.xpu.device(device_index):
            return torch.xpu.memory_allocated() / _MEM_FACTORS[unit.strip().lower()]


def get_gpu_free_memory(device: str | torch.device, unit: str = 'b'):
    """
    Return the amount of free memory available on a GPU device.

    Non GPU devices always return 0.

    :param device: The device.
    :param unit: one of (case insensitive): b (bytes), kb (kilobytes),
        mb (megabytes), gb (gigabytes), kib (kibibytes),
        mib (mebibytes), gib (gibibytes)
    :return: Requested value.
    """
    if not is_supported_gpu_device(device):
        return 0
    
    device_type, device_index = _parse_gpu_device(device)

    if device_type == 'cuda':
        total_memory = torch.cuda.get_device_properties(device_index).total_memory
        with torch.cuda.device(device_index):
            reserved_memory = torch.cuda.memory_reserved()
            return (total_memory - reserved_memory) / _MEM_FACTORS[unit.strip().lower()]
    elif device_type == 'xpu':
        total_memory = torch.xpu.get_device_properties(device_index).total_memory
        with torch.xpu.device(device_index):
            reserved_memory = torch.xpu.memory_reserved()
            return (total_memory - reserved_memory) / _MEM_FACTORS[unit.strip().lower()]


def get_gpu_reserved_memory(device: str | torch.device, unit: str = 'b'):
    """
    Return the amount of reserved memory on a GPU device.

    Non GPU devices always return 0.

    :param device: The device.
    :param unit: one of (case insensitive): b (bytes), kb (kilobytes),
        mb (megabytes), gb (gigabytes), kib (kibibytes),
        mib (mebibytes), gib (gibibytes)
    :return: Requested value.
    """
    if not is_supported_gpu_device(device):
        return 0
    
    device_type, device_index = _parse_gpu_device(device)

    if device_type == 'cuda':
        with torch.cuda.device(device_index):
            return torch.cuda.memory_reserved() / _MEM_FACTORS[unit.strip().lower()]
    elif device_type == 'xpu':
        with torch.xpu.device(device_index):
            return torch.xpu.memory_reserved() / _MEM_FACTORS[unit.strip().lower()]


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


def gpu_memory_constraints(expressions: collections.abc.Iterable[str],
                            extra_vars: dict[str, int | float] | None = None,
                            mode=any,
                            device: str | torch.device = 'cuda:0') -> bool:
    """
    Evaluate a user boolean expression involving a GPU device's memory in bytes,
    used memory percent, and available VRAM memory in bytes.

   If you pass a non GPU device identifier to this method, it will always return ``False``

    Available functions are:
        * kb(bytes to kilobytes)
        * mb(bytes to megabytes)
        * gb(bytes to gigabytes)
        * kib(bytes to kibibytes)
        * mib(bytes to mebibytes)
        * gib(bytes to gibibytes)

    Available values are:
        * used / u (memory currently used by the GPU device in bytes)
        * used_total_percent / utp (memory used by the GPU device, as percent of total VRAM memory, example: 25.4)
        * available / a (available memory remaining on the GPU device in bytes that can be used)
        * total / t (total memory on the GPU device in bytes)

    Example expressions:
        * ``used > gb(1)`` (when the device has used more than 1GB of memory)
        * ``used_total_percent > 25`` (when the device has used more than 25 percent of VRAM memory)
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
    :param device: GPU device string or torch.device object, defaults to 'cuda:0'. Can be CUDA (e.g., 'cuda:0') or XPU (e.g., 'xpu:0') devices.
    :return: Boolean result of the expression
    """

    if not expressions:
        return False

    for expr in expressions:
        memory_constraint_syntax_check(expr)

    if not is_supported_gpu_device(device):
        return False
    
    device_type, device_index = _parse_gpu_device(device)

    if device_type == 'cuda':
        total_memory = torch.cuda.get_device_properties(device_index).total_memory
        with torch.cuda.device(device_index):
            reserved_memory = torch.cuda.memory_reserved()
            allocated_memory = torch.cuda.memory_allocated()
            free_memory = total_memory - reserved_memory
    elif device_type == 'xpu':
        total_memory = torch.xpu.get_device_properties(device_index).total_memory
        with torch.xpu.device(device_index):
            reserved_memory = torch.xpu.memory_reserved()
            allocated_memory = torch.xpu.memory_allocated()
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

    interpreter = _eval.standard_interpreter(
        symtable=_eval.safe_builtins() | variables.copy()
    )

    interpreter.symtable.update(functions)

    _messages.debug_log(
        f'GPU MEMORY CONSTRAINT TEST: {_types.fullname(gpu_memory_constraints)} constraint = '
        f'[{", ".join(_textprocessing.quote_spaces(expressions))}], '
        f'vars = {str(variables)}, mode={mode.__name__}')

    try:
        value = mode(interpreter(
            e, raise_errors=True, show_errors=False) for e in expressions)
        if not isinstance(value, bool):
            raise MemoryConstraintSyntaxError('Memory constraint must return a boolean value.')

        _messages.debug_log(f'GPU MEMORY CONSTRAINT TEST RESULT: {value}')
        return value
    except (Exception, NameError) as e:
        raise MemoryConstraintSyntaxError(
            f'Memory constraint syntax error: {e}')


class SizedConstrainedObjectCache(_memoize.ObjectCache):
    """
    An object cache that can track cache memory use via the cached objects returned metadata.

    Your memoized function should return at least: ``object, dgenerate.memoize.CachedObjectMetadata(size=the_size)``

    You must return a metadata object with the attribute ``size`` at the minimum.

    You may attach other metadata to the object as needed.
    """

    def __init__(self, name):
        super().__init__(name)
        self._size = 0
        self.register_on_un_cache(self._on_un_cache)
        self.register_on_cache(self._on_cache)
        self.register_on_clear(self._on_clear)

    @property
    def size(self):
        """
        Return the current cache size.
        """
        return self._size

    @size.setter
    def size(self, value):
        """
        Set the current cache size.
        """
        self._size = value
        if self._size < 0:
            self._size = 0

    def enforce_cpu_mem_constraints(
            self,
            constraints: typing.Iterable[str],
            size_var: str,
            new_object_size: int,
            mode: typing.Callable[[typing.Iterable], bool] = any
    ):
        """
        Clear the cache if these CPU side memory constraints are met.

        See: :py:func:`memory_constraints`

        The constraint variable ``cache_size`` equates to the current cache size.

        :param constraints:
        :param size_var: Memory constraint expression variable name containing the ``new_object_size`` value.
        :param new_object_size: Size of the new object.
        :param mode: Logical and/or function on constraint expressions, ``any`` for or, ``all`` for and.
        :return: ``True`` if the cache was cleared, ``False`` otherwise
        """

        _messages.debug_log(
            f'Object Cache: "{self.name}", enforcing CPU side memory constraints: {constraints}, mode={mode.__name__}')

        if memory_constraints(constraints,
                              {size_var: new_object_size, 'cache_size': self.size},
                              mode=mode):
            _messages.debug_log(
                f'Object Cache: "{self.name}", cleared due to CPU side memory constraints being met.')
            self.clear()
            return True
        return False

    def enforce_gpu_mem_constraints(
            self,
            constraints: typing.Iterable[str],
            size_var: str,
            new_object_size: int,
            device: str | torch.device,
            mode: typing.Callable[[typing.Iterable], bool] = any
    ):
        """
        Clear the cache if these GPU side memory constraints are met.

        See: :py:func:`gpu_memory_constraints`

        The constraint variable ``cache_size`` equates to the current cache size.

        :param constraints:
        :param size_var: Memory constraint expression variable name containing the ``new_object_size`` value.
        :param new_object_size: Size of the new object.
        :param device: Device to check
        :param mode: Logical and/or function on constraint expressions, ``any`` for or, ``all`` for and.
        :return: ``True`` if the cache was cleared, ``False`` otherwise
        """
        _messages.debug_log(
            f'Object Cache: "{self.name}", enforcing GPU side memory constraints: {constraints}, mode={mode.__name__}')

        if gpu_memory_constraints(constraints,
                                   {size_var: new_object_size, 'cache_size': self.size},
                                   mode=mode, device=device):
            _messages.debug_log(
                f'Object Cache: "{self.name}", cleared due to GPU side memory constraints being met.')
            self.clear()
            return True
        return False

    def _on_un_cache(self, cache, cached_object):
        self.size -= self.get_metadata(cached_object).size

    def _on_clear(self, cache):
        self.size = 0

    def _on_cache(self, cache, cached_object):
        self.size += self.get_metadata(cached_object).size


def torch_gc():
    """
    Call ``torch.cuda.empty_cache()`` and ``torch.cuda.ipc_collect()`` for CUDA,
    and ``torch.xpu.empty_cache()`` for XPU devices.
    """

    if _torchutil.is_cuda_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if _torchutil.is_xpu_available():
        torch.xpu.empty_cache()
        # Note: torch.xpu does not have ipc_collect() equivalent
