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

"""
Raise Hugging Face's file-download read timeout.

``HF_HUB_DOWNLOAD_TIMEOUT`` is a single number of seconds. Hugging Face uses it
as the longest silence allowed between socket reads while streaming a file.
The library default is 10 seconds, which is short for multi-gigabyte files.
Hub downloads already retry and resume; this only makes a stall less likely
to interrupt them. A value already set in the environment is left alone.
"""

import os

_ENV = 'HF_HUB_DOWNLOAD_TIMEOUT'
_DEFAULT_SECONDS = 60


def apply_hf_download_timeout(seconds: int = _DEFAULT_SECONDS) -> int:
    """
    Apply ``seconds`` as the Hub file-download timeout unless the environment
    already sets one.

    :param seconds: Timeout in seconds used when the environment variable is unset.
    :return: The timeout now stored on ``huggingface_hub.constants``.
    """
    raw = os.environ.get(_ENV)
    if raw is None or not str(raw).strip():
        chosen = seconds
        os.environ[_ENV] = str(seconds)
    else:
        try:
            chosen = int(str(raw).strip())
        except ValueError:
            chosen = seconds
            os.environ[_ENV] = str(seconds)

    import huggingface_hub.constants as constants
    constants.HF_HUB_DOWNLOAD_TIMEOUT = chosen
    return chosen


apply_hf_download_timeout()
