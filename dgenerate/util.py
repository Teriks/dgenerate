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


def suffix_path_maker(filename, suffix):
    """
    To be used with :py:meth:`.touch_avoid_duplicate`, a pathmaker implementation that
    appends a suffix and a number to a file when a duplicate is detected.

    :param filename: Original filename
    :param suffix: Suffix to append if needed, a trailing number will be appended
    :return:
    """

    def pathmaker(attempt=None):
        base, ext = os.path.splitext(filename)
        if attempt is not None:
            return f"{base}{suffix}{attempt}{ext}"
        return filename

    return pathmaker


def touch_avoid_duplicate(directory: str,
                          pathmaker: typing.Callable[[typing.Optional[int]], str],
                          lockname: str = '.duplicate_check_lock'):
    """
    Generate a filename in a directory and avoid duplicates using a file lock in that directory
    with a known name. Use to ensure duplicate checking in a directory is multiprocess safe,
    at least for processes using this function to write to the same directory.


    :param directory: The directory to create the lockfile in
    :param pathmaker: Callback that enerates paths until a non-existent path is found,
        first argument is the attempt number. On the first attempt it will be None
    :param lockname: Name of the lock file to be used as a mutex

    :return: Unique path
    """
    with temp_file_lock(os.path.join(directory, lockname)):
        path = pathmaker(None)

        if not os.path.exists(path):
            return path

        duplicate_number = 1
        while os.path.exists(path):
            path = pathmaker(duplicate_number)
            duplicate_number += 1

        pathlib.Path(path).touch()
        return path


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
