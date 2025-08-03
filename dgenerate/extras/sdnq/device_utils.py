"""
Device utilities for SDNQ quantization backend.
Provides device and memory management functionality.
"""

import sys
import torch
from typing import Optional, Union

# Default dtype - can be overridden
dtype = torch.float16

# Detect backend
def _detect_backend():
    """Detect the computation backend"""
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"

backend = _detect_backend()

# Current device
if backend == "cuda":
    device = torch.device("cuda")
elif backend == "xpu":
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

# CPU device
cpu = torch.device("cpu")

def same_device(device1: torch.device, device2: torch.device) -> bool:
    """Check if two devices are the same"""
    device1 = torch.device(device1)
    device2 = torch.device(device2)
    return device1.type == device2.type and device1.index == device2.index

def torch_gc(force: bool = False, reason: Optional[str] = None):
    """Perform garbage collection for torch tensors"""
    if backend == "cuda":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if force:
                torch.cuda.ipc_collect()
    elif backend == "xpu":
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()
    
    # Also perform regular garbage collection
    import gc
    gc.collect()

def set_dtype(new_dtype: torch.dtype):
    """Set the default dtype"""
    global dtype
    dtype = new_dtype

def set_device(new_device: Union[str, torch.device]):
    """Set the default device"""
    global device, backend
    if isinstance(new_device, str):
        device = torch.device(new_device)
    else:
        device = new_device
    
    backend = device.type 