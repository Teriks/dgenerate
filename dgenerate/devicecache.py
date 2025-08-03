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
import gc
import typing

import torch

import dgenerate.memory as _memory
import dgenerate.types as _types

__doc__ = """
High level utilitys for interacting with objects cached by dgenerate in GPU side memory.

These objects may be cached in small quantity by various dgenerate sub-modules.
"""

__eviction_methods = []


def register_eviction_method(method: typing.Callable[[torch.device | None], None]):
    """
    Register a method for evicting an object cached on the GPU.

    This will be called upon calling :py:meth:`clear_device_cache`.

    :param method: Eviction method, first argument is the ``torch.device`` being considered,
        if this value is ``None``, any torch device with cached objects should be considered.
    """
    __eviction_methods.append(method)


def clear_device_cache(device: torch.device | str | None = None):
    """
    Clear every object cached by dgenerate in its GPU side cache for a specific device.

    :param device: The device the objects are cached on, specifying ``None`` clears all devices.
    """

    device = torch.device(device)

    for method in __eviction_methods:
        method(device)

    gc.collect()
    _memory.torch_gc()


__all__ = _types.module_all()
