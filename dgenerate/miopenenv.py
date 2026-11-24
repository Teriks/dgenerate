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

"""MIOpen find mode and log level for ROCm convolutions."""

import os
import sys

# MIOpen caches both values on first use. Set them before the first convolution.
# Find mode 2 is FAST: it does not probe GemmFwdRest, which warns when PyTorch's
# workspace is smaller than that solver wants and then uses another solver.
# Log level 3 is Error. Log level 4 is Warning.
_FIND_MODE = '2'
_LOG_ERROR = '3'
_LOG_WARNING = '4'

_log_level_owned = False


def argv_is_verbose(argv: list[str] | None = None) -> bool:
    """True when ``argv`` contains ``-v`` or ``--verbose``."""
    if argv is None:
        argv = sys.argv[1:]
    return any(arg == '-v' or arg == '--verbose' for arg in argv)


def configure(verbose: bool) -> None:
    """
    Use MIOpen's fast convolution find, and hide its warnings unless ``verbose``.

    An ``MIOPEN_FIND_MODE`` or ``MIOPEN_LOG_LEVEL`` already set in the environment
    is left alone. A log level this function set earlier is updated when ``verbose``
    changes, which covers ``-v`` on a config line after import.
    """
    global _log_level_owned

    if not os.environ.get('MIOPEN_FIND_MODE', '').strip():
        os.environ['MIOPEN_FIND_MODE'] = _FIND_MODE

    if os.environ.get('MIOPEN_LOG_LEVEL', '').strip() and not _log_level_owned:
        return
    os.environ['MIOPEN_LOG_LEVEL'] = _LOG_WARNING if verbose else _LOG_ERROR
    _log_level_owned = True
