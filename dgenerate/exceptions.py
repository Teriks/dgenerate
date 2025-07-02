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

__doc__ = 'Common exceptions'

import torch.cuda
import dgenerate.types as _types


class OutOfMemoryError(Exception):
    """
    Raised when a GPU or processing device runs out of memory.
    """

    def __init__(self, message):
        super().__init__(f'Device Out Of Memory: {message}')


TORCH_CUDA_OOM_EXCEPTIONS = (torch.cuda.OutOfMemoryError, RuntimeError)
"""Exceptions which can be raised by ``torch.cuda`` for OOM conditions."""


def raise_if_not_cuda_oom(e):
    """
    Decide if a :py:attr:`TORCH_CUDA_OOM_EXCEPTIONS` exception was caused by OOM
    and re-raise it if not.

    :param e: the exception
    """
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return
    if isinstance(e, RuntimeError):
        msg = str(e).lower()
        if 'cuda' in msg and 'memory' in msg:
            return
    raise e


class ModelNotFoundError(Exception):
    """Raised when a specified model can not be located either locally or remotely"""
    pass


class ConfigNotFoundError(Exception):
    """Raised when a specified config can not be located either locally or remotely"""
    pass


__all__ = _types.module_all()

