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
import sys
import typing

import dgenerate.types as _types

__doc__ = """
Utilities for file like objects.
"""


class PeekReader:
    """
    Read from a file like iterator object while peeking at the next line.

    This is an iterable reader wrapper that yields the tuple ``(current_line, next_line)``

    **next_line** will be ``None`` if the next line is the end of iterator / file.
    """

    def __init__(self, iterator: typing.Iterator[str]):
        """
        :param iterator: The ``typing.Iterator`` capable reader to wrap.
        """
        self._iterator = iterator
        self._last_next_line = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._last_next_line is not None:
            self._cur_line = self._last_next_line
            self._last_next_line = None
        else:
            self._cur_line = next(self._iterator)

        try:
            self._next_line = next(self._iterator)
            self._last_next_line = self._next_line
        except StopIteration:
            self._next_line = None

        return self._cur_line, self._next_line


class Unbuffered:
    """File wrapper which auto flushes a stream on write"""

    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def stdin_is_tty():
    """
    Safely checks if stdin is a tty

    :return: `True` or `False`
    """
    return sys.stdin is not None and hasattr(sys.stdin, 'isatty') and sys.stdin.isatty()


class GCFile:
    """File object wrapper, close file on garbage collection"""

    def __init__(self, file):
        self.file = file

    def __del__(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def __iter__(self):
        return self.file.__iter__()

    def __next__(self):
        return self.file.__next__()

    def __getattr__(self, item):
        return getattr(self.file, item)

    def __setattr__(self, key, value):
        if key == "file":
            self.__dict__[key] = value
        else:
            setattr(self.file, key, value)

    def __delattr__(self, item):
        delattr(self.file, item)


class TerminalLineReader:
    """
    Reads lines from a binary stream, typically `stdout` or `stderr` of a subprocess.

    Breaks on newlines and carriage return, preserves
    newlines and carriage return in the output as is.
    """

    pushback_byte: bytes | None
    """
    Byte on the stack which will be prepended to the next line if needed.
    
    Should be set to ``None`` if file was provided a callable 
    and the underlying reader has changed to a new instance.
    """

    def __init__(self, file: typing.BinaryIO | typing.Callable[[], typing.IO]):
        """
        :param file: Binary IO object, or a function that returns one.
        """
        self._file = file
        self.pushback_byte = None

    @property
    def file(self) -> typing.BinaryIO:
        """
        The current file object being read.
        """
        if callable(self._file):
            return self._file()
        return self._file

    def readline(self):
        line = bytearray()
        if self.pushback_byte is not None:
            line.append(ord(self.pushback_byte))
            self.pushback_byte = None

        while True:
            byte = self.file.read(1)
            if not byte:
                break
            line.append(ord(byte))
            if byte == b'\n':
                break
            if byte == b'\r':
                next_byte = self.file.read(1)
                if next_byte == b'\n':
                    line.append(ord(next_byte))
                else:
                    self.pushback_byte = next_byte
                break

        return bytes(line) if line else b''


__all__ = _types.module_all()
