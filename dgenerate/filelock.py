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
import contextlib
import os
import pathlib
import typing

import portalocker

__doc__ = """
Thread / Multiprocess safe file locking utilities.
"""

PathMaker = typing.Callable[
    [typing.Optional[str], typing.Optional[int]], typing.Union[str, collections.abc.Iterable[str]]]


def suffix_path_maker(filenames: typing.Union[str, collections.abc.Iterable[str]], suffix: str) -> PathMaker:
    """
    To be used with :py:func:`.touch_avoid_duplicate`, a pathmaker implementation that
    appends a suffix and a number to a filename or list of files when a duplicate is detected for
    any of them in the directory.

    :param filenames: Original filename, or a list of filenames
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
                          path_maker: PathMaker,
                          lock_name: str = '.lock',
                          return_list=False):
    """
    Generate a filename in a directory and avoid duplicates using a file lock in that directory
    with a known name. Use to ensure duplicate checking in a directory is multiprocess safe,
    at least for processes using this function to write to the same directory.


    :param return_list: Always return a list even if generated paths is only of length 1,
        defaults to False, which means that a single string will be returned if only one
        path was generated by the pathmaker

    :param directory: The directory to create the lockfile in

    :param path_maker: Callback that generates paths until a non-existent path is found,
        first argument is the base filename and the second is attempt number. On the first attempt
        to create the files both arguments will be none, in which case the callback should return
        a single filename or iterable of filenames to touch with duplicate avoidance. Calls to the callback
        thereafter will have non None values for both arguments and the callback should take the passed
        base filename and apply a suffix using the attempt number.

    :param lock_name: Name of the lock file to be used as a mutex

    :return: Unique path that has been touched (created but empty), or a tuple of paths
        if the path maker requested duplicate checks on multiple files
    """
    with temp_file_lock(os.path.join(directory, lock_name)):
        paths = path_maker(None, None)

        if isinstance(paths, str):
            paths = [paths]

        for idx, path in enumerate(paths):
            if not os.path.exists(path):
                pathlib.Path(path).touch()
                continue

            unmodified_path = path
            duplicate_number = 1
            while os.path.exists(path):
                path = path_maker(unmodified_path, duplicate_number)
                duplicate_number += 1

            paths[idx] = path
            pathlib.Path(path).touch()

        if len(paths) == 1 and not return_list:
            return paths[0]
        return paths


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
        except OSError:
            pass
