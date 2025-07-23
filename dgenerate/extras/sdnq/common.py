"""
Common utilities and constants for SDNQ quantization backend.
"""

import sys
import torch
from . import device_utils as devices

torch_version = float(torch.__version__[:3])

dtype_dict = {
    "int8": {"min": -128, "max": 127, "num_bits": 8, "target_dtype": torch.int8, "torch_dtype": torch.int8, "storage_dtype": torch.int8, "is_unsigned": False, "is_integer": True},
    "int7": {"min": -64, "max": 63, "num_bits": 7, "target_dtype": "int7", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True},
    "int6": {"min": -32, "max": 31, "num_bits": 6, "target_dtype": "int6", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True},
    "int5": {"min": -16, "max": 15, "num_bits": 5, "target_dtype": "int5", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True},
    "int4": {"min": -8, "max": 7, "num_bits": 4, "target_dtype": "int4", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True},
    "int3": {"min": -4, "max": 3, "num_bits": 3, "target_dtype": "int3", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True},
    "int2": {"min": -2, "max": 1, "num_bits": 2, "target_dtype": "int2", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True},
    "uint8": {"min": 0, "max": 255, "num_bits": 8, "target_dtype": torch.uint8, "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True},
    "uint7": {"min": 0, "max": 127, "num_bits": 7, "target_dtype": "uint7", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True},
    "uint6": {"min": 0, "max": 63, "num_bits": 6, "target_dtype": "uint6", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True},
    "uint5": {"min": 0, "max": 31, "num_bits": 5, "target_dtype": "uint5", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True},
    "uint4": {"min": 0, "max": 15, "num_bits": 4, "target_dtype": "uint4", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True},
    "uint3": {"min": 0, "max": 7, "num_bits": 3, "target_dtype": "uint3", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True},
    "uint2": {"min": 0, "max": 3, "num_bits": 2, "target_dtype": "uint2", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True},
    "uint1": {"min": 0, "max": 1, "num_bits": 1, "target_dtype": torch.bool, "torch_dtype": torch.bool, "storage_dtype": torch.bool, "is_unsigned": True, "is_integer": True},
    "float8_e4m3fn": {"min": -448, "max": 448, "num_bits": 8, "target_dtype": torch.float8_e4m3fn, "torch_dtype": torch.float8_e4m3fn, "storage_dtype": torch.float8_e4m3fn, "is_unsigned": False, "is_integer": False},
    "float8_e5m2": {"min": -57344, "max": 57344, "num_bits": 8, "target_dtype": torch.float8_e5m2, "torch_dtype": torch.float8_e5m2, "storage_dtype": torch.float8_e5m2, "is_unsigned": False, "is_integer": False},
}
dtype_dict["bool"] = dtype_dict["uint1"]
if hasattr(torch, "float8_e4m3fnuz"):
    dtype_dict["float8_e4m3fnuz"] = {"min": -240, "max": 240, "num_bits": 8, "target_dtype": "fp8", "torch_dtype": torch.float8_e4m3fnuz, "storage_dtype": torch.float8_e4m3fnuz, "is_unsigned": False, "is_integer": False}
if hasattr(torch, "float8_e5m2fnuz"):
    dtype_dict["float8_e5m2fnuz"] = {"min": -57344, "max": 57344, "num_bits": 8, "target_dtype": "fp8", "torch_dtype": torch.float8_e5m2fnuz, "storage_dtype": torch.float8_e5m2fnuz, "is_unsigned": False, "is_integer": False}

use_tensorwise_fp8_matmul = torch_version < 2.5 or devices.backend in {"cpu", "openvino"} or (devices.backend == "cuda" and sys.platform == "win32" and torch_version <= 2.7 and torch.cuda.get_device_capability(devices.device) == (8,9))
quantized_matmul_dtypes = ("int8", "int7", "int6", "int5", "int4", "int3", "int2", "float8_e4m3fn", "float8_e5m2")
if devices.backend in {"cpu", "openvino"}:
    quantized_matmul_dtypes += ("float8_e4m3fnuz", "float8_e5m2fnuz")

linear_types = ("Linear",)
conv_types = ("Conv1d", "Conv2d", "Conv3d")
conv_transpose_types = ("ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d")
allowed_types = linear_types + conv_types + conv_transpose_types 