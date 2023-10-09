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

import dgenerate.textprocessing as _textprocessing

LEVEL = 0

INFO = 0
WARNING = 1
ERROR = 2
DEBUG = 3


def log(*args: typing.Any, **kwargs):
    level = kwargs.get('level', INFO)
    underline_me = kwargs.get('underline', False)
    file = sys.stdout
    if level != INFO and LEVEL == INFO:
        if level != ERROR and level != WARNING:
            return
        else:
            file = sys.stderr

    if underline_me:
        print(_textprocessing.underline(' '.join(str(a) for a in args)), file=file)
    else:
        print(' '.join(str(a) for a in args), file=file)


def debug_log(*func_or_str: typing.Union[typing.Callable[[], typing.Any], typing.Any], **kwargs):
    if LEVEL == DEBUG:
        vals = []
        for val in func_or_str:
            if callable(val):
                vals.append(val())
            else:
                vals.append(val)
        log(*vals, level=DEBUG, **kwargs)
