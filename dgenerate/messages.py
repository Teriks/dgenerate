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
import os
import sys
import typing

import dgenerate.textprocessing as _textprocessing

__doc__ = """
Library logging / informational output.
"""

LEVEL = 0
"""Current Log Level (set-able)"""

INFO = 0
"""Log level INFO"""
WARNING = 1
"""Log Level WARNING"""
ERROR = 2
"""Log Level ERROR"""
DEBUG = 3
"""Log Level DEBUG"""

_ERROR_FILE = sys.stderr
_MESSAGE_FILE = sys.stdout

AUTO_FLUSH_MESSAGES = True
"""
Whether to auto flush the output stream when printing to ``stdout`` 
or the output file assigned with :py:func:`.set_message_file`.

Errors are printed to ``stderr`` which is unbuffered by default.
"""

_handlers = []

_level_stack = []


def push_level(level):
    """
    Set :py:attr:`LEVEL` and save the previous value to a stack.

    :param level: one of :py:attr:`.INFO`, :py:attr:`.WARNING`, :py:attr:`.ERROR`, , :py:attr:`.DEBUG`
    """
    global LEVEL
    _level_stack.append(LEVEL)
    LEVEL = level


def pop_level():
    """
    Pop ``LEVEL`` value last saved by :py:func:`.push_level` and assign it to :py:attr:`.LEVEL`.

    If no previous level was saved, no-op.
    """

    global LEVEL, _level_stack

    if _level_stack:
        LEVEL = _level_stack.pop()


def set_error_file(file: typing.TextIO):
    """
    Set a file stream or file like object for dgenerates error output.

    :param file: The file stream
    """
    global _ERROR_FILE
    _ERROR_FILE = file


def set_message_file(file: typing.TextIO):
    """
    Set a file stream or file like object for dgenerates normal (non error) output.

    :param file: The file stream
    """
    global _MESSAGE_FILE
    _MESSAGE_FILE = file


def messages_to_null():
    """
    Force dgenerates normal output to a null file.
    """
    global _MESSAGE_FILE
    _MESSAGE_FILE = open(os.devnull, "w")


def errors_to_null():
    """
    Force dgenerates error output to a null file.
    """
    global _ERROR_FILE
    _ERROR_FILE = open(os.devnull, "w")


def add_logging_handler(callback: typing.Callable[[typing.ParamSpecArgs, int, bool, str], None]):
    """
    Add your own logging handler callback.

    :param callback: Callback accepting ``(*args, LEVEL, underline (bool), underline_char)``
    """
    _handlers.append(callback)


def remove_logging_handler(callback: typing.Callable[[typing.ParamSpecArgs, int, bool, str], None]):
    """
    Remove a logging handler callback by reference.

    :param callback: The previously registered callback
    """
    _handlers.remove(callback)


def log(*args: typing.Any, level=INFO, underline=False, underline_char='='):
    """
    Write a message to dgenerates log

    :param args: args, objects that will be stringified and joined with a space
    :param level: Log level, one of:
        :py:attr:`.INFO`, :py:attr:`.WARNING`, :py:attr:`.ERROR`, :py:attr:`.DEBUG`
    :param underline: Underline this message?
    :param underline_char: Underline character
    :return:
    """
    global _MESSAGE_FILE, _ERROR_FILE

    file = _MESSAGE_FILE
    if level != INFO and LEVEL == INFO:
        if level != ERROR and level != WARNING:
            return
        else:
            file = _ERROR_FILE

    prefix = ''
    if level == DEBUG:
        prefix = 'DEBUG: '
    if level == WARNING:
        prefix = 'WARNING: '

    if underline:
        print(_textprocessing.underline(prefix + ' '.join(str(a) for a in args),
                                        underline_char=underline_char), file=file, flush=AUTO_FLUSH_MESSAGES)
    else:
        print(prefix + ' '.join(str(a) for a in args), file=file, flush=AUTO_FLUSH_MESSAGES)

    for handler in _handlers:
        handler(*args,
                level=level,
                underline=underline,
                underline_char=underline_char)


def debug_log(*func_or_str: typing.Union[typing.Callable[[], typing.Any], typing.Any],
              underline=False, underline_char='='):
    """
    Conditionally log strings or possibly expensive functions if :py:attr:`.LEVEL` is
    set to :py:attr:`.DEBUG`.

    :param func_or_str: objects to be stringified and printed or callables that return said objects
    :param underline: Underline this message?
    :param underline_char: Underline character.
    """
    if LEVEL == DEBUG:
        vals = []
        for val in func_or_str:
            if callable(val):
                vals.append(val())
            else:
                vals.append(val)
        log(*vals, level=DEBUG, underline=underline, underline_char=underline_char)
