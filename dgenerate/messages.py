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
import sys
import typing

import dgenerate.textprocessing as _textprocessing

__doc__ = """
Library logging / informational output.
"""

LEVEL = 1
"""
Current Log Level (set-able)

Setting to :py:attr:`INFO` means print all messages except :py:attr:`DEBUG` messages.

Setting to :py:attr:`ERROR` means only print :py:attr:`ERROR` messages.

Setting to :py:attr:`WARNING` means only print :py:attr:`WARNING` messages.

Setting to :py:attr:`DEBUG` means print every message.

Levels are a bitfield, so you can set: ``LEVEL = WARNING | ERROR`` etc.
"""

INFO = 1
"""Log level ``INFO``"""
WARNING = 2
"""Log Level ``WARNING``"""
ERROR = 4
"""Log Level ``ERROR``"""
DEBUG = 8
"""Log Level ``DEBUG``"""

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
    Set :py:attr:`dgenerate.messages.LEVEL` and save the previous value to a stack.

    :param level: one of :py:attr:`.INFO`, :py:attr:`.WARNING`, :py:attr:`.ERROR`, , :py:attr:`.DEBUG`
    """
    global LEVEL
    _level_stack.append(LEVEL)
    LEVEL = level


def pop_level():
    """
    Pop ``dgenerate.messages.LEVEL`` value last saved by :py:func:`.push_level` and assign it to :py:attr:`.LEVEL`.

    If no previous level was saved, no-op.
    """

    global LEVEL, _level_stack

    if _level_stack:
        LEVEL = _level_stack.pop()


@contextlib.contextmanager
def with_level(level):
    """
    Context manager which pushes a ``dgenerate.messages.LEVEL`` to
    the stack and pops it when the ``with`` context ends.

    This affects logging output level within the context.

    :param level: log level
    """
    try:
        push_level(level)
        yield
    finally:
        pop_level()


def set_error_file(file: typing.TextIO):
    """
    Set a file stream or file like object for dgenerate's error output.

    :param file: The file stream
    """
    global _ERROR_FILE
    _ERROR_FILE = file


def set_message_file(file: typing.TextIO):
    """
    Set a file stream or file like object for dgenerate's normal (non error) output.

    :param file: The file stream
    """
    global _MESSAGE_FILE
    _MESSAGE_FILE = file


def get_error_file():
    """
    Get the file stream or file like object for dgenerate's error output.

    """
    global _ERROR_FILE
    return _ERROR_FILE


def get_message_file():
    """
    Get the file stream or file like object for dgenerate's normal (non error) output.

    """
    global _MESSAGE_FILE
    return _MESSAGE_FILE


def messages_to_null():
    """
    Force dgenerate's normal output to a null file.
    """
    global _MESSAGE_FILE
    _MESSAGE_FILE = open(os.devnull, "w")


def errors_to_null():
    """
    Force dgenerate's error output to a null file.
    """
    global _ERROR_FILE
    _ERROR_FILE = open(os.devnull, "w")


@contextlib.contextmanager
def silence():
    """
    Context manager to silence all dgenerate logging / messages.

    This will redirect all messages to a null file temporarily.
    """
    global _MESSAGE_FILE, _ERROR_FILE

    old_message_file = _MESSAGE_FILE
    old_error_file = _ERROR_FILE

    messages_to_null()
    errors_to_null()

    try:
        yield
    finally:
        _MESSAGE_FILE = old_message_file
        _ERROR_FILE = old_error_file


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
    Write a message to dgenerate's log

    :param args: args, objects that will be stringified and joined with a space
    :param level: Log level, one of:
        :py:attr:`.INFO`, :py:attr:`.WARNING`, :py:attr:`.ERROR`, :py:attr:`.DEBUG`
    :param underline: Underline this message?
    :param underline_char: Underline character
    """
    global _MESSAGE_FILE, _ERROR_FILE

    file = _ERROR_FILE if level == ERROR else _MESSAGE_FILE

    allowed = set()

    if LEVEL & INFO:
        allowed.update({INFO, ERROR, WARNING})
    if LEVEL & ERROR:
        allowed.update({ERROR})
    if LEVEL & WARNING:
        allowed.update({WARNING})
    if LEVEL & DEBUG:
        allowed.update({INFO, ERROR, WARNING, DEBUG})

    if level not in allowed:
        return

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


def error(*args: typing.Any, underline=False, underline_char='='):
    """
    Write an error message to dgenerate's log

    :param args: args, objects that will be stringified and joined with a space
    :param underline: Underline this message?
    :param underline_char: Underline character
    """
    log(*args, level=ERROR, underline=underline, underline_char=underline_char)


def warning(*args: typing.Any, underline=False, underline_char='='):
    """
    Write a warning message to dgenerate's log

    :param args: args, objects that will be stringified and joined with a space
    :param underline: Underline this message?
    :param underline_char: Underline character
    """
    log(*args, level=WARNING, underline=underline, underline_char=underline_char)


def debug_log(*func_or_str: typing.Callable[[], typing.Any] | typing.Any,
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
