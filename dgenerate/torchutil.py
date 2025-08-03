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

import dgenerate.types as _types
import dgenerate.textprocessing as _textprocessing


def is_cuda_available() -> bool:
    """
    Check if CUDA is available on this system.
    
    :return: True if CUDA is available, False otherwise
    """
    return hasattr(torch, 'cuda') and torch.cuda.is_available()


def is_xpu_available() -> bool:
    """
    Check if Intel XPU is available on this system.
    
    :return: True if XPU is available, False otherwise
    """
    return hasattr(torch, 'xpu') and torch.xpu.is_available()


def is_mps_available() -> bool:
    """
    Check if Apple Metal Performance Shaders (MPS) is available on this system.
    
    :return: True if MPS is available, False otherwise
    """
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


class InvalidDeviceOrdinalException(Exception):
    """
    Device in device specification (cuda:N, xpu:N) does not exist
    """
    pass


def default_device() -> str:
    """
    Return a string representing the systems default accelerator device.

    Possible Values:

        * ``"cuda"``
        * ``"mps"``
        * ``"xpu"``
        * ``"cpu"``

    :return: ``"cuda"``, ``"mps"``, ``"xpu"``, etc.
    """
    if is_cuda_available():
        return 'cuda'
    elif is_xpu_available():
        return 'xpu'
    elif is_mps_available():
        return 'mps'
    else:
        return 'cpu'


def available_device_types() -> list[str]:
    """
    Return a list of available torch device type strings.

    Such as: cpu, cuda, xpu, mps

    :return: List of device type strings
    """

    devices = ['cpu']
    if is_cuda_available():
        devices.append('cuda')
    if is_xpu_available():
        devices.append('xpu')
    if is_mps_available():
        devices.append('mps')

    return devices


def is_valid_device_string(device: str, raise_ordinal=False):
    """
    Is a device string valid? including the device ordinal specified?

    Other than cuda, "mps" (MacOS metal performance shaders) and "xpu" (Intel) is experimentally supported.

    :param device: device string, such as ``cpu``, or ``cuda``, or ``cuda:N``, or ``xpu:N``
    :param raise_ordinal: Raise :py:exc:`.InvalidDeviceOrdinalException` if
        a specified CUDA or XPU device ordinal is found to not exist?

    :raises InvalidDeviceOrdinalException: If ``raise_ordinal=True`` and a the
        device ordinal specified in a CUDA or XPU device string does not exist.

    :return: ``True`` or ``False``
    """

    match = re.match(r'^(?:cpu|cuda|xpu)(?::([0-9]+))?$', device)

    if match:
        device_type = device.split(':')[0]
        if match.lastindex:
            ordinal = int(match[1])
            if device_type == 'cuda':
                valid_ordinal = ordinal < torch.cuda.device_count()
                if raise_ordinal and not valid_ordinal:
                    raise InvalidDeviceOrdinalException(
                        f'CUDA device ordinal {ordinal} is invalid, no such device exists.')
                return valid_ordinal
            elif device_type == 'xpu':
                # Check if XPU is available and validate ordinal
                if is_xpu_available():
                    valid_ordinal = ordinal < torch.xpu.device_count()
                    if raise_ordinal and not valid_ordinal:
                        raise InvalidDeviceOrdinalException(
                            f'XPU device ordinal {ordinal} is invalid, no such device exists.')
                    return valid_ordinal
                else:
                    return False
        else:
            # No ordinal specified
            if device_type == 'cuda':
                return is_cuda_available()
            elif device_type == 'xpu':
                return is_xpu_available()
            elif device_type == 'cpu':
                return True
        return True

    if device == 'mps' and is_mps_available():
        return True

    return False

def invalid_device_message(device: torch.device | str, cap: bool = True) -> str:
    """
    Generate a standard invalid device message.

    For example: ``Must be ...., unknown value: (given value)"``

    Or: ``CUDA device ordinal 2 is invalid, no such device exists.``

    The content is hardware / platform / selected device specific.

    :param device: The device given that was invalid
    :param cap: The message starts with a capital?
    :return: Invalid device message string.
    """

    try:
        is_valid_device_string(device, raise_ordinal=True)
    except InvalidDeviceOrdinalException as e:
        return str(e)

    d_device = torch.device(default_device())

    cap = 'M' if cap else 'm'

    if d_device.type == 'mps':
        return f'{cap}ust be "cpu" or "mps", unknown value: {str(device)}'
    else:
        return f'{cap}ust be one of: {_textprocessing.oxford_comma(available_device_types(), "or")}, '\
               f'Optionally with a device ordinal, for example ({d_device}:0, {d_device}:1, etc..). '\
               f'Unknown value: {str(device)}'


def devices_equal(device1: torch.device | str, device2: torch.device | str):
    """
    Compare if two devices are the same device.

    This considers ``cuda`` and ``cuda:{torch.cuda.current_device()}`` to be the same device,
    and ``xpu`` and ``xpu:{torch.xpu.current_device()}`` to be the same device.

    :param device1: Device 1.
    :param device2: Device 2.
    :return: Equality?
    """
    d1 = torch.device(device1)
    d2 = torch.device(device2)

    if d1.type == 'cuda' and d1.index is None:
        default_cuda_index = torch.cuda.current_device()
        d1 = torch.device(f'cuda:{default_cuda_index}')
    if d2.type == 'cuda' and d2.index is None:
        default_cuda_index = torch.cuda.current_device()
        d2 = torch.device(f'cuda:{default_cuda_index}')

    if d1.type == 'xpu' and d1.index is None:
        if is_xpu_available():
            default_xpu_index = torch.xpu.current_device()
            d1 = torch.device(f'xpu:{default_xpu_index}')
    if d2.type == 'xpu' and d2.index is None:
        if is_xpu_available():
            default_xpu_index = torch.xpu.current_device()
            d2 = torch.device(f'xpu:{default_xpu_index}')

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


def is_tensor(obj) -> bool:
    """
    Check if an object is a PyTorch tensor.
    
    :param obj: Object to check
    :return: True if the object is a torch.Tensor, False otherwise
    """
    return isinstance(obj, torch.Tensor)


__all__ = _types.module_all()
