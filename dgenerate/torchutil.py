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


__doc__ = """
Commonly used torch utilities.
"""

import re
import torch


class InvalidDeviceOrdinalException(Exception):
    """
    GPU in device specification (cuda:N) does not exist
    """
    pass


def default_device() -> str:
    """
    Return a string representing the systems default accelerator device.

    Possible Values:

        * ``"cuda"``
        * ``"mps"``
        * ``"cpu"``

    :return: ``"cuda"``, ``"mps"``, etc.
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def is_valid_device_string(device, raise_ordinal=True):
    """
    Is a device string valid? including the device ordinal specified?

    Other than cuda, "mps" (MacOS metal performance shaders) is experimentally supported.

    :param device: device string, such as ``cpu``, or ``cuda``, or ``cuda:N``
    :param raise_ordinal: Raise :py:exc:`.InvalidDeviceOrdinalException` if
        a specified CUDA device ordinal is found to not exist?

    :raises InvalidDeviceOrdinalException: If ``raise_ordinal=True`` and a the
        device ordinal specified in a CUDA device string does not exist.

    :return: ``True`` or ``False``
    """

    match = re.match(r'^(?:cpu|cuda(?::([0-9]+))?)$', device)

    if match:
        if match.lastindex:
            ordinal = int(match[1])
            valid_ordinal = ordinal < torch.cuda.device_count()
            if raise_ordinal and not valid_ordinal:
                raise InvalidDeviceOrdinalException(f'CUDA device ordinal {ordinal} is invalid, no such device exists.')
            return valid_ordinal
        return True

    if device == 'mps' and hasattr(torch.backends, 'mps') \
            and torch.backends.mps.is_available():
        return True

    return False


def devices_equal(device1: torch.device | str, device2: torch.device | str):
    """
    Compare if two devices are the same device.

    This considers ``cuda`` and ``cuda:{torch.cuda.current_device()}`` to be the same device.

    :param device1: Device 1.
    :param device2: Device 2.
    :return: Equality?
    """
    d1 = torch.device(device1)
    d2 = torch.device(device2)

    default_device_index = torch.cuda.current_device()

    if d1.type == 'cuda' and d1.index is None:
        d1 = torch.device(f'cuda:{default_device_index}')
    if d2.type == 'cuda' and d2.index is None:
        d2 = torch.device(f'cuda:{default_device_index}')

    return d1 == d2


def estimate_module_memory_usage(module: torch.nn.Module) -> str:
    """
    Estimate the static memory use of a torch module.

    :param module: the module

    :return: static memory use in bytes
    """

    dtype = next(module.parameters()).dtype
    dtype_sizes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1
    }
    bytes_per_param = dtype_sizes.get(dtype, 4)
    num_params = sum(p.numel() for p in module.parameters())
    return num_params * bytes_per_param
