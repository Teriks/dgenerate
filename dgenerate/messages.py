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

LEVEL = 0

INFO = 0
WARNING = 1
ERROR = 2
DEBUG = 3

_err_file = sys.stderr
_msg_file = sys.stdout

_handlers = []


def set_error_file(file: typing.TextIO):
    _errfile = file


def set_message_file(file: typing.TextIO):
    _msgfile = file


def messages_to_null():
    _msgfile = open(os.devnull, "w")


def errors_to_null():
    _errfile = open(os.devnull, "w")


def add_logging_handler(callback: typing.Callable[[typing.ParamSpecArgs, int, bool, str], None]):
    _handlers.append(callback)


def remove_logging_handler(callback: typing.Callable[[typing.ParamSpecArgs, int, bool, str], None]):
    _handlers.remove(callback)


def log(*args: typing.Any, level=INFO, underline=False, underline_char='='):
    file = _msg_file
    if level != INFO and LEVEL == INFO:
        if level != ERROR and level != WARNING:
            return
        else:
            file = _err_file

    if underline:
        print(_textprocessing.underline(' '.join(str(a) for a in args),
                                        underline_char=underline_char), file=file)
    else:
        print(' '.join(str(a) for a in args), file=file)

    for handler in _handlers:
        handler(*args,
                level=level,
                underline=underline,
                underline_char=underline_char)


def debug_log(*func_or_str: typing.Union[typing.Callable[[], typing.Any], typing.Any],
              underline=False, underline_char='='):
    if LEVEL == DEBUG:
        vals = []
        for val in func_or_str:
            if callable(val):
                vals.append(val())
            else:
                vals.append(val)
        log(*vals, level=DEBUG, underline=underline, underline_char=underline_char)
