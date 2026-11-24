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

"""MIOpen find mode and log level for one generation."""

import contextlib
import os

# MIOpen caches both values the first time it reads them. Enter this before
# the first convolution. Find mode 2 is FAST: it does not probe GemmFwdRest,
# which warns when PyTorch's workspace is smaller than that solver wants.
# Log level 3 is Error. Log level 4 is Warning.
_FIND_MODE = 'MIOPEN_FIND_MODE'
_LOG_LEVEL = 'MIOPEN_LOG_LEVEL'
_FIND_FAST = '2'
_LOG_ERROR = '3'
_LOG_WARNING = '4'


def _blank(name: str) -> bool:
    return not os.environ.get(name, '').strip()


@contextlib.contextmanager
def applied(verbose: bool):
    """
    Use MIOpen's fast convolution find, and hide its warnings unless ``verbose``.

    A value already in the environment is left in place. Values this sets are
    removed when the context ends.
    """
    previous = {}
    if _blank(_FIND_MODE):
        previous[_FIND_MODE] = os.environ.get(_FIND_MODE)
        os.environ[_FIND_MODE] = _FIND_FAST
    if _blank(_LOG_LEVEL):
        previous[_LOG_LEVEL] = os.environ.get(_LOG_LEVEL)
        os.environ[_LOG_LEVEL] = _LOG_WARNING if verbose else _LOG_ERROR
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
