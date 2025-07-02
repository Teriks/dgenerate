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

import dgenerate.types as _types
from .argos import ArgosTranslator
from .exceptions import TranslatorLoadError, TranslationError
from .mariana import MarianaTranslator
from .util import get_language_code

__doc__ = """
Translation backends for language translation via local inference.
"""

_offline_mode = False

def is_offline_mode() -> bool:
    """
    Check if the translators module is in offline mode.

    :return: ``True`` if in offline mode, ``False`` otherwise.
    """
    global _offline_mode
    return _offline_mode


def enable_offline_mode():
    """
    Enable offline mode for the translators module.

    This will prevent any network requests from being made.
    """
    global _offline_mode
    _offline_mode = True
    ArgosTranslator._offline_mode = True
    MarianaTranslator._offline_mode = True


def disable_offline_mode():
    """
    Disable offline mode for the translators module.

    This will allow network requests to be made again.
    """
    global _offline_mode
    _offline_mode = False
    ArgosTranslator._offline_mode = False
    MarianaTranslator._offline_mode = False


@contextlib.contextmanager
def offline_mode_context(enabled=True):
    """
    Context manager to temporarily enable or disable offline mode for the translators module.

    :param enabled: If `True`, enables offline mode. If `False`, disables it.
    """
    global _offline_mode
    original_mode = _offline_mode

    if enabled:
        enable_offline_mode()
    else:
        disable_offline_mode()
    try:
        yield
    finally:
        if original_mode:
            enable_offline_mode()
        else:
            disable_offline_mode()

__all__ = _types.module_all()
