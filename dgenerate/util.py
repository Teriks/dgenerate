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
import os
import pathlib
import typing

import portalocker
import psutil

PathMaker = typing.Callable[[typing.Optional[str], typing.Optional[int]], typing.Union[str, typing.List[str]]]


def suffix_path_maker(filenames, suffix):
    """
    To be used with :py:meth:`.touch_avoid_duplicate`, a pathmaker implementation that
    appends a suffix and a number to a filename or list of files when a duplicate is detected for
    any of them in the directory.

    :param filename: Original filename, or a list of filenames
    :param suffix: Suffix to append if needed, a trailing number will be appended
    :return:
    """

    if isinstance(filenames, str):
        filenames = [filenames]

    def pathmaker(base_filename, attempt_number=None):
        if base_filename is None and attempt_number is None:
            # Requesting a list of files involved
            return filenames

        # Requesting we fix a filename to try to make
        # it unique, will be requested again with a new
        # attempt_number if the filename does not turn
        # out to be unique.
        base, ext = os.path.splitext(base_filename)
        return f'{base}{suffix}{attempt_number}{ext}'

    return pathmaker


def touch_avoid_duplicate(directory: str,
                          pathmaker: PathMaker,
                          lockname: str = '.lock',
                          return_list=False):
    """
    Generate a filename in a directory and avoid duplicates using a file lock in that directory
    with a known name. Use to ensure duplicate checking in a directory is multiprocess safe,
    at least for processes using this function to write to the same directory.


    :param return_list: Always return a list even if generated paths is only of length 1,
        defaults to False, which means that a single string will be returned if only one
        path was generated by the pathmaker

    :param directory: The directory to create the lockfile in

    :param pathmaker: Callback that generates paths until a non-existent path is found,
        first argument is the base filename and the second is attempt number. On the first attempt
        to create the files both arguments will be none, in which case the callback should return
        a single filename or list of filenames to touch with duplicate avoidance. Calls to the callback
        thereafter will have non None values for both arguments and the callback should take the passed
        base filename and apply a suffix using the attempt number.

    :param lockname: Name of the lock file to be used as a mutex

    :return: Unique path that has been touched (created but empty), or a tuple of paths
        if the path maker requested duplicate checks on multiple files
    """
    with temp_file_lock(os.path.join(directory, lockname)):
        paths = pathmaker(None, None)

        if isinstance(paths, str):
            paths = [paths]

        for idx, path in enumerate(paths):
            if not os.path.exists(path):
                pathlib.Path(path).touch()
                continue

            unmodified_path = path
            duplicate_number = 1
            while os.path.exists(path):
                path = pathmaker(unmodified_path, duplicate_number)
                duplicate_number += 1

            paths[idx] = path
            pathlib.Path(path).touch()

        if len(paths) == 1 and not return_list:
            return paths[0]
        return paths


class MemoryConstraintSyntaxError:
    """
    Thrown by :py:meth:`.memory_constraints` on syntax errors or
    if an expression returns a non boolean value
    """
    pass


def memory_constraints(expressions: typing.Optional[typing.Union[str, list]],
                       extra_vars: typing.Optional[typing.Dict[str, typing.Union[int, float]]] = None,
                       mode=any,
                       pid=os.getpid()) -> bool:
    """
    Evaluate a user boolean expression involving the the processes used memory in bytes, used memory percent, and available system memory in bytes.

    Available functions are:
        * kb(bytes to kilobytes)
        * mb(bytes to megabytes)
        * gb(bytes to gigabytes)
        * kib(bytes to kibibytes)
        * mib(bytes to mebibytes)
        * gib(bytes to gibibytes)

    Available values are:

        * used / u (memory used by the process in bytes)
        * used_percent / up (memory used by the process, as whole percent, example: 25.4)
        * available / a (available memory remaining on the system in bytes)
        * total / t (total memory on the system in bytes)

    Example expressions:

        * ``used > gb(1)`` (when the process has used more than 1GB of memory)
        * ``used_percent > 25`` (when the process has used more than 25 percent of system memory)
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
    :param pid: PID of the process from which to aquire the 'used' and 'used_percent' variable
        values from, defaults to the current process.
    :return: Boolean result of the expression
    """

    if not expressions:
        return False

    if isinstance(expressions, str):
        expressions = [expressions]

    p_info = psutil.Process(os.getpid())
    used = p_info.memory_info().rss
    used_percent = p_info.memory_percent()

    mem_info = psutil.virtual_memory()

    available = mem_info.available
    total = mem_info.total

    globals = {'gb': lambda x: x * 1000 ** 3,
               'mb': lambda x: x * 1000 ** 2,
               'kb': lambda x: x * 1000,
               'gib': lambda x: x * 1024 ** 3,
               'mib': lambda x: x * 1024 ** 2,
               'kib': lambda x: x * 1024}

    locals = {
        'used': used,
        'u': used,
        'used_percent': used_percent,
        'up': used_percent,
        'available': available,
        'a': available,
        'total': total,
        't': total
    }

    if extra_vars:
        for key, value in extra_vars.items():
            if key in locals:
                raise ValueError(
                    f'extra_vars cannot redefine reserved attribute: {key}')
            locals[key] = value

    try:
        value = mode(eval(e, globals, locals) for e in expressions)
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


def get_used_memory(measure='b', pid = os.getpid()) -> float:
    """
    Get the memory used by a process in a selectable unit.

    :param measure: one of (case insensitive): b (bytes), kb (kilobytes),
        mb (megabytes), gb (gigabytes), kib (kibibytes),
        mib (mebibytes), gib (gibibytes)

    :param pid: The process PID to retrieve this information from, defaults to the current process.

    :return: Requested value.
    """
    return psutil.Process(pid).memory_info().rss / _MEM_FACTORS[measure.strip().lower()]


def get_used_memory_percent(pid = os.getpid()) -> float:
    """
    Get the percentage of memory used by a process.

    :param pid: PID of the process, defaults to the current process.
    :return: A whole percentage, for example: 25.4
    """
    return psutil.Process(pid).memory_percent()


def get_available_memory(measure='b'):
    """
    Get the available memory remaining on the system in a selectable unit.

    :param measure: one of (case insensitive): b (bytes), kb (kilobytes),
        mb (megabytes), gb (gigabytes), kib (kibibytes),
        mib (mebibytes), gib (gibibytes)

    :return: Requested value.
    """
    return psutil.virtual_memory().available / _MEM_FACTORS[measure.strip().lower()]


@contextlib.contextmanager
def temp_file_lock(path):
    """
    Multiprocess synchronization utility.

    Get a lock on an empty file as a context manager, delete the lock file if possible when done.

    :param path: Path where the lock file will be created.
    :return: Lock as a context manager
    """
    try:
        with portalocker.Lock(path):
            yield
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
