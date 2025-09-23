# Copyright (c) 2023, Teriks
# BSD 3-Clause License

"""
Patch for accelerate.cpu_offload and accelerate.cpu_offload_with_hook to automatically
fallback to available devices when the requested device is not available.

This prevents crashes when third-party code tries to use CUDA on systems without CUDA,
MPS on non-macOS systems, etc.
"""

import torch
import torch.nn as nn
import accelerate
from accelerate.hooks import CpuOffload
from typing import Optional, Union, Tuple

# Try to import the correct hook type
try:
    from accelerate.hooks import UserCpuOffloadHook
except ImportError:
    # Fallback to CpuOffload if UserCpuOffloadHook is not available
    UserCpuOffloadHook = CpuOffload


def _is_cuda_available() -> bool:
    """Check if CUDA is available on this system."""
    return hasattr(torch, 'cuda') and torch.cuda.is_available()


def _is_xpu_available() -> bool:
    """Check if Intel XPU is available on this system."""
    return hasattr(torch, 'xpu') and torch.xpu.is_available()


def _is_mps_available() -> bool:
    """Check if Apple Metal Performance Shaders (MPS) is available on this system."""
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def _default_device() -> str:
    """
    Return a string representing the systems default accelerator device.
    
    Possible Values: "cuda", "mps", "xpu", "cpu"
    """
    if _is_cuda_available():
        return 'cuda'
    elif _is_xpu_available():
        return 'xpu'
    elif _is_mps_available():
        return 'mps'
    else:
        return 'cpu'


# Store original functions
_ORIG_CPU_OFFLOAD = accelerate.cpu_offload
_ORIG_CPU_OFFLOAD_WITH_HOOK = accelerate.cpu_offload_with_hook


def _get_fallback_device(requested_device: Union[int, str, torch.device]) -> torch.device:
    """
    Get a fallback device if the requested device is not available.
    
    Args:
        requested_device: The originally requested device
        
    Returns:
        A valid torch.device that is available on the system
    """
    if isinstance(requested_device, (str, int)):
        device_obj = torch.device(requested_device)
    else:
        device_obj = requested_device
    
    device_type = device_obj.type
    
    # Check if the requested device type is actually available
    if device_type == 'cuda' and not _is_cuda_available():
        fallback = _default_device()
        # print(f"Warning: CUDA requested but not available, falling back to {fallback}")
        return torch.device(fallback)
    elif device_type == 'mps' and not _is_mps_available():
        fallback = _default_device()
        # print(f"Warning: MPS requested but not available, falling back to {fallback}")
        return torch.device(fallback)
    elif device_type == 'xpu' and not _is_xpu_available():
        fallback = _default_device()
        # print(f"Warning: XPU requested but not available, falling back to {fallback}")
        return torch.device(fallback)
    
    # Device is available, return as-is
    return device_obj


def _patched_cpu_offload(
    model: nn.Module,
    execution_device: Optional[torch.device] = None,
    offload_buffers: bool = False,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
    preload_module_classes: Optional[list[str]] = None,
):
    """
    Patched version of accelerate.cpu_offload that falls back to available devices.
    """
    if execution_device is not None:
        execution_device = _get_fallback_device(execution_device)
    
    return _ORIG_CPU_OFFLOAD(
        model=model,
        execution_device=execution_device,
        offload_buffers=offload_buffers,
        state_dict=state_dict,
        preload_module_classes=preload_module_classes
    )


def _patched_cpu_offload_with_hook(
    model: torch.nn.Module,
    execution_device: Optional[Union[int, str, torch.device]] = None,
    prev_module_hook: Optional[UserCpuOffloadHook] = None,
):
    """
    Patched version of accelerate.cpu_offload_with_hook that falls back to available devices.
    """
    if execution_device is not None:
        execution_device = _get_fallback_device(execution_device)
    
    return _ORIG_CPU_OFFLOAD_WITH_HOOK(
        model=model,
        execution_device=execution_device,
        prev_module_hook=prev_module_hook
    )


# Apply patches
accelerate.cpu_offload = _patched_cpu_offload
accelerate.cpu_offload_with_hook = _patched_cpu_offload_with_hook

# Also patch the functions in their original module locations
try:
    import accelerate.big_modeling
    accelerate.big_modeling.cpu_offload = _patched_cpu_offload
    accelerate.big_modeling.cpu_offload_with_hook = _patched_cpu_offload_with_hook
except (ImportError, AttributeError):
    pass

try:
    import accelerate.hooks
    # The hooks module might import these functions
    if hasattr(accelerate.hooks, 'cpu_offload'):
        accelerate.hooks.cpu_offload = _patched_cpu_offload
    if hasattr(accelerate.hooks, 'cpu_offload_with_hook'):
        accelerate.hooks.cpu_offload_with_hook = _patched_cpu_offload_with_hook
except (ImportError, AttributeError):
    pass
