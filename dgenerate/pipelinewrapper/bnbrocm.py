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

"""ROCm workaround for the bitsandbytes 8-bit GEMM."""

import dgenerate.messages as _messages
import dgenerate.torchutil as _torchutil

_installed = False


def _arch_is_gfx950(arch: str) -> bool:
    """``gcnArchName`` is ``gfx950`` or ``gfx950:sramecc+:xnack-``."""
    return arch.split(':', 1)[0] == 'gfx950'


def _has_gfx950() -> bool:
    """True when a visible ROCm GPU is gfx950."""
    import torch

    for index in range(torch.cuda.device_count()):
        arch = getattr(torch.cuda.get_device_properties(index), 'gcnArchName', '') or ''
        if _arch_is_gfx950(arch):
            return True
    return False


def use_dequantized_int8_matmul() -> None:
    """
    On gfx950, run 8-bit bitsandbytes layers with a dequantized matmul.

    bitsandbytes 0.50.2 passes a rocBLAS handle to hipBLASLt. On gfx950 that
    8-bit GEMM writes past its buffer. The same kernel runs on other AMD GPUs.
    ``MatMul8bitFp`` dequantizes the stored int8 weights and uses an ordinary
    linear, which is the path bitsandbytes already uses on CPU and Intel GPU.
    The weights stay int8. The LLM.int8 outlier split is not applied.
    """
    global _installed
    if _installed or not _torchutil.is_rocm_available() or not _has_gfx950():
        return

    import bitsandbytes as bnb
    import bitsandbytes.autograd._functions as functions

    def matmul(A, B, out=None, state=None, threshold=0.0, bias=None):
        state = state or functions.MatmulLtState()
        if threshold > 0.0:
            state.threshold = threshold
        return functions.MatMul8bitFp.apply(A, B, out, bias, state)

    bnb.matmul = matmul
    functions.matmul = matmul
    _installed = True
    _messages.log(
        '8-bit bitsandbytes GEMM writes past its buffer on gfx950. '
        'Using a dequantized matmul instead.',
        level=_messages.WARNING)
