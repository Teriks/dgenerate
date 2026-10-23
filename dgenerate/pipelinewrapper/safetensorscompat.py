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
Load ComfyUI pre-quantized ``.safetensors`` checkpoints with Diffusers.

ComfyUI stores each quantized linear as packed ``weight`` storage plus
``weight_scale`` (and per format ``weight_scale_2``, ``input_scale``,
``pre_quant_scale``), described by a ``comfy_quant`` JSON marker. Diffusers
knows none of this: it drops the scales as unexpected keys and casts the raw
packed values to the model dtype, which renders noise.

Supported formats: ``float8_e4m3fn`` and ``float8_e5m2`` (per tensor or per
row scale, including legacy ``scaled_fp8`` files with no marker),
``int8_tensorwise`` (per row scale, optional ConvRot Hadamard rotation), and
``nvfp4`` (E2M1 with swizzled FP8 block scales and a tensor scale, including
unmarked ModelOpt exports such as LTX-2 FP4). Legacy scaled fp8 files name
their scales ``scale_weight`` and ``scale_input``.

Quantized linears stay packed in :py:class:`ComfyQuantLinear`. Where the
hardware has a matching matmul (INT8 and FP8 on CUDA, ROCm, XPU, and MPS,
NVFP4 on Blackwell and MPS), including ``kernels`` Hub kernels on MPS and ROCm,
it is used once verified for accuracy and speed.
Otherwise the weight is decoded to the activation dtype on each forward with
plain torch ops, which runs on CUDA, ROCm, XPU, MPS, and CPU alike. Every packed tensor, float32 scales
included (as int32 bit views), is an integer parameter: Diffusers' load-time
cast and later dtype casts leave it untouched, and offload hooks move it like
any other parameter.
"""

import contextlib
import enum
import functools
import importlib
import json
import math
import os
import struct
import time

import torch

import dgenerate.messages as _messages

FLOAT8_FORMATS = ('float8_e4m3fn', 'float8_e5m2')
SUPPORTED_FORMATS = FLOAT8_FORMATS + ('int8_tensorwise', 'nvfp4')

_FLOAT8_DTYPES = {
    'float8_e4m3fn': torch.float8_e4m3fn,
    'float8_e5m2': torch.float8_e5m2,
}

_SAFETENSORS_FLOAT8 = {
    'F8_E4M3': 'float8_e4m3fn',
    'F8_E5M2': 'float8_e5m2',
}

_AUX_SUFFIXES = (
    '.weight_scale', '.weight_scale_2', '.input_scale',
    '.pre_quant_scale', '.comfy_quant')

# Marker tensor of legacy (pre comfy_quant) scaled fp8 files.
_LEGACY_MARKER = 'scaled_fp8'

# Legacy scaled fp8 files name the scales ``<layer>.scale_weight`` and
# ``<layer>.scale_input``.
_LEGACY_SCALE_NAMES = {
    '.scale_weight': '.weight_scale',
    '.scale_input': '.input_scale',
}


def _scale_owner(key: str) -> str | None:
    """The layer name of a weight scale key, current or legacy naming."""
    for suffix in ('.weight_scale', '.scale_weight'):
        if key.endswith(suffix):
            return key[:-len(suffix)]
    return None


def _rename_legacy_scales(checkpoint: dict) -> dict:
    """Legacy ``scale_weight`` / ``scale_input`` keys -> ``weight_scale`` / ``input_scale``."""
    renamed = {}
    for key, value in checkpoint.items():
        for old, new in _LEGACY_SCALE_NAMES.items():
            if key.endswith(old) and key[:-len(old)] + '.weight' in checkpoint:
                key = key[:-len(old)] + new
                break
        renamed[key] = value
    return renamed

_NVFP4_BLOCK = 16

# E2M1 code -> value. NVFP4 packs two codes per byte, high nibble first.
_E2M1_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

_lut_cache = {}
_hadamard_cache = {}

# Device types where viewing uint8 storage as a float8 dtype and casting it
# failed once. Those decode through a lookup table instead (MPS has no float8).
_no_native_float8 = {'mps'}


class ComfyQuantError(Exception):
    """A ComfyUI quantized checkpoint cannot be loaded."""


def _read_header(path: str) -> dict | None:
    try:
        with open(path, 'rb') as file:
            size = struct.unpack('<Q', file.read(8))[0]
            return json.loads(file.read(size))
    except (OSError, ValueError, struct.error):
        return None


def is_comfy_quantized(path: str) -> bool:
    """
    True when a local ``.safetensors`` file holds ComfyUI quantized linears.

    Only the header is read.
    """
    if not path.lower().endswith('.safetensors') or not os.path.isfile(path):
        return False
    header = _read_header(path)
    if not header:
        return False
    header.pop('__metadata__', None)
    if _LEGACY_MARKER in header:
        return True
    for key, info in header.items():
        if key.endswith('.comfy_quant'):
            return True
        if (key.endswith('.weight') and info.get('dtype') in _SAFETENSORS_FLOAT8
                and len(info.get('shape', ())) == 2):
            return True
        name = _scale_owner(key)
        if name is not None:
            weight = header.get(name + '.weight')
            if weight and weight.get('dtype') in ('I8', 'U8', *_SAFETENSORS_FLOAT8):
                return True
    return False


def _lookup_table(kind: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (kind, str(device), dtype)
    table = _lut_cache.get(key)
    if table is None:
        if kind == 'nvfp4':
            # One row per byte: [high nibble value, low nibble value].
            values = torch.tensor(_E2M1_VALUES, dtype=torch.float32)
            codes = torch.arange(256)
            table = torch.stack([values[codes >> 4], values[codes & 0x0F]], dim=-1)
        else:
            table = torch.arange(256, dtype=torch.uint8).view(_FLOAT8_DTYPES[kind]).float()
            table = table.unsqueeze(-1)
        table = table.to(device=device, dtype=dtype)
        _lut_cache[key] = table
    return table


def _decode_float8(storage: torch.Tensor, fmt: str, dtype: torch.dtype) -> torch.Tensor:
    """Decode float8 bits held in a uint8 tensor."""
    device_type = storage.device.type
    if device_type not in _no_native_float8:
        try:
            return storage.view(_FLOAT8_DTYPES[fmt]).to(dtype)
        except (RuntimeError, TypeError, NotImplementedError):
            _no_native_float8.add(device_type)
            _messages.debug_log(
                f'No native float8 cast on "{device_type}", decoding through a lookup table.')
    table = _lookup_table(fmt, storage.device, dtype)
    return torch.nn.functional.embedding(storage.int(), table).view(storage.shape)


def _decode_nvfp4(storage: torch.Tensor, block_scale: torch.Tensor,
                  tensor_scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    rows, packed = storage.shape
    blocks = packed * 2 // _NVFP4_BLOCK
    table = _lookup_table('nvfp4', storage.device, dtype)
    values = torch.nn.functional.embedding(storage.int(), table)
    values = values.view(rows, blocks, _NVFP4_BLOCK)
    # Block scales stay in the tiled layout the hardware matmul takes.
    block_scale = _unswizzle_block_scales(block_scale, rows, blocks)
    scale = _decode_float8(block_scale, 'float8_e4m3fn', torch.float32)
    scale = scale * _broadcast_scale(tensor_scale.to(torch.float32), rows)
    return (values * scale.to(dtype).unsqueeze(-1)).view(rows, packed * 2)


def _float_bits(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """float32 values as an int32 tensor of the same bits."""
    if tensor is None:
        return None
    return tensor.to(torch.float32).contiguous().view(torch.int32)


def _bits_float(tensor: torch.Tensor | None) -> torch.Tensor | None:
    return None if tensor is None else tensor.view(torch.float32)


def _broadcast_scale(scale: torch.Tensor, rows: int) -> torch.Tensor:
    if scale.numel() == 1:
        return scale.reshape(())
    if scale.numel() == rows:
        return scale.reshape(rows, 1)
    raise ComfyQuantError(
        f'weight_scale of {scale.numel()} values does not fit a weight of {rows} rows.')


def _decode(fmt: str, storage: torch.Tensor, scale: torch.Tensor,
            scale_2: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor:
    """
    Packed storage -> weight in ``dtype``, still padded and rotated.

    ``scale`` is the NVFP4 block scale grid as uint8, otherwise float32 bits
    as int32, like ``scale_2``.
    """
    scale_2 = _bits_float(scale_2)
    if fmt == 'nvfp4':
        return _decode_nvfp4(storage, scale, scale_2, dtype)
    if fmt == 'int8_tensorwise':
        weight = storage.to(dtype)
    else:
        weight = _decode_float8(storage, fmt, dtype)
    return weight * _broadcast_scale(_bits_float(scale), storage.shape[0]).to(dtype)


def _hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    The normalized regular Hadamard matrix used by ComfyUI ConvRot.

    It is the Kronecker power of a symmetric 4x4 Hadamard matrix, so it is
    symmetric and orthogonal: its own inverse.
    """
    key = (size, str(device), dtype)
    matrix = _hadamard_cache.get(key)
    if matrix is None:
        h4 = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=torch.float64)
        matrix = h4
        while matrix.shape[0] < size:
            matrix = torch.kron(matrix, h4)
        matrix = (matrix / math.sqrt(size)).to(device=device, dtype=dtype)
        _hadamard_cache[key] = matrix
    return matrix


def _rotate(tensor: torch.Tensor, group_size: int) -> torch.Tensor:
    """Multiply each ``group_size`` slice of the last dimension by the Hadamard matrix."""
    shape = tensor.shape
    grouped = tensor.reshape(-1, shape[-1] // group_size, group_size)
    return (grouped @ _hadamard(group_size, tensor.device, tensor.dtype)).reshape(shape)


def _valid_group_size(size: int) -> bool:
    return size >= 4 and math.log(size, 4).is_integer()


class _QuantLayer:
    """
    One quantized linear from the checkpoint, before it is placed in a module.

    Tensors are in module storage form: float8 weights as uint8, float32
    scales as int32 bits, NVFP4 block scales in their tiled layout as uint8.
    ``scale`` and the NVFP4 tensor scale ``scale_2`` hold one value or one per
    row.
    """

    def __init__(self, name: str, fmt: str, weight: torch.Tensor, scale: torch.Tensor,
                 scale_2: torch.Tensor | None = None,
                 pre_quant_scale: torch.Tensor | None = None,
                 convrot_group_size: int = 0, full_precision: bool = False):
        self.name = name
        self.format = fmt
        self.weight = weight
        self.scale = scale
        self.scale_2 = scale_2
        self.pre_quant_scale = pre_quant_scale
        self.convrot_group_size = convrot_group_size
        # The file asks for a dequantized matmul instead of a low precision one.
        self.full_precision = full_precision

    def logical_shape(self) -> tuple[int, int]:
        rows, cols = self.weight.shape
        return (rows, cols * 2) if self.format == 'nvfp4' else (rows, cols)

    def fits(self, out_features: int, in_features: int) -> bool:
        rows, cols = self.logical_shape()
        if self.format == 'nvfp4':
            # NVFP4 pads both dimensions to a multiple of 16.
            return (rows == _round_up(out_features, _NVFP4_BLOCK)
                    and cols == _round_up(in_features, _NVFP4_BLOCK))
        return (rows, cols) == (out_features, in_features)

    def dequantize(self, dtype: torch.dtype, shape: tuple[int, int] | None = None) -> torch.Tensor:
        """The dense weight, unrotated and trimmed to ``shape``."""
        weight = _decode(self.format, self.weight, self.scale, self.scale_2, torch.float32)
        if self.convrot_group_size:
            weight = _rotate(weight, self.convrot_group_size)
        if shape is not None:
            weight = weight[:shape[0], :shape[1]]
        if self.pre_quant_scale is not None:
            # (x * s) @ W.T == x @ (W * s).T
            weight = weight * _bits_float(self.pre_quant_scale).reshape(1, -1)
        return weight.to(dtype).contiguous()

    def row_scales(self) -> torch.Tensor:
        """float32 weight scale for each stored row (INT8, FP8)."""
        scale = _bits_float(self.scale).reshape(-1)
        return scale.expand(self.weight.shape[0]) if scale.numel() == 1 else scale

    def tensor_scales(self) -> torch.Tensor:
        """float32 NVFP4 tensor scale for each stored row."""
        scale = _bits_float(self.scale_2).reshape(-1)
        return scale.expand(self.weight.shape[0]) if scale.numel() == 1 else scale

    def block_scales(self) -> torch.Tensor:
        """NVFP4 block scales, ``(rows, inputs / 16)`` uint8, row major."""
        rows, packed = self.weight.shape
        return _unswizzle_block_scales(self.scale, rows, packed * 2 // _NVFP4_BLOCK)


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _unswizzle_block_scales(blocked: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """
    Undo the cuBLAS tiled layout ComfyUI writes NVFP4 block scales in.

    The file stores a ``(roundup(rows, 128), roundup(cols, 4))`` grid of
    128x4 tiles. Returns ``(rows, cols)``.
    """
    row_tiles = -(-rows // 128)
    col_tiles = -(-cols // 4)
    tiles = blocked.reshape(-1, 32, 16).reshape(-1, 32, 4, 4).transpose(1, 2)
    tiles = tiles.reshape(row_tiles, col_tiles, 128, 4).permute(0, 2, 1, 3)
    return tiles.reshape(row_tiles * 128, col_tiles * 4)[:rows, :cols].contiguous()


def _layer_config(checkpoint: dict, name: str) -> dict:
    marker = checkpoint.get(name + '.comfy_quant')
    if marker is None:
        return {}
    try:
        return json.loads(bytes(marker.tolist()).decode('utf-8'))
    except (ValueError, UnicodeDecodeError) as e:
        raise ComfyQuantError(f'"{name}.comfy_quant" is not valid JSON: {e}') from e


def _infer_format(checkpoint: dict, name: str) -> str | None:
    """The format of a layer with no ``comfy_quant`` marker, from its storage."""
    weight = checkpoint[name + '.weight']
    if weight.dtype == torch.int8:
        return 'int8_tensorwise'
    if weight.dtype == torch.uint8 and name + '.weight_scale_2' in checkpoint:
        # NVFP4 as ModelOpt exports it (LTX-2 FP4, for example): the same layout.
        return 'nvfp4'
    for fmt, dtype in _FLOAT8_DTYPES.items():
        if weight.dtype == dtype:
            return fmt
    return None


def _extract_layer(checkpoint: dict, name: str) -> _QuantLayer:
    config = _layer_config(checkpoint, name)
    weight = checkpoint[name + '.weight']
    scale = checkpoint.get(name + '.weight_scale')
    fmt = config.get('format') or _infer_format(checkpoint, name)

    if fmt not in SUPPORTED_FORMATS:
        raise ComfyQuantError(
            f'Layer "{name}" uses the ComfyUI quantization format "{fmt}", which is not '
            f'supported. Supported formats: {", ".join(SUPPORTED_FORMATS)}.')
    full_precision = bool(config.get('full_precision_matrix_mult', False))
    if scale is None and fmt in FLOAT8_FORMATS and not config:
        # Plain float8 storage. ComfyUI casts these weights for a full precision
        # matmul, and so does Diffusers, so keep that numerically: scale 1.
        scale = torch.ones(())
        full_precision = True
    if scale is None:
        raise ComfyQuantError(f'Layer "{name}" is missing weight_scale.')

    pre_quant_scale = _float_bits(checkpoint.get(name + '.pre_quant_scale'))
    if weight.ndim != 2:
        raise ComfyQuantError(f'Quantized layer "{name}" is not a 2D linear weight.')

    if fmt in FLOAT8_FORMATS:
        if weight.dtype not in (torch.uint8, _FLOAT8_DTYPES[fmt]):
            raise ComfyQuantError(f'Layer "{name}" is marked {fmt} but stored as {weight.dtype}.')
        storage = weight.view(torch.uint8)
        if fmt == 'float8_e4m3fn':
            # -0 (0x80) is NaN when the bits are read as float8_e4m3fnuz (ROCm MI300).
            # +0 decodes to the same value everywhere.
            storage.masked_fill_(storage == 0x80, 0)
        return _QuantLayer(name, fmt, storage, _float_bits(scale),
                           pre_quant_scale=pre_quant_scale, full_precision=full_precision)

    if fmt == 'int8_tensorwise':
        if weight.dtype != torch.int8:
            raise ComfyQuantError(f'Layer "{name}" is marked int8 but stored as {weight.dtype}.')
        params = config.get('params') if isinstance(config.get('params'), dict) else {}
        group_size = 0
        if config.get('convrot', params.get('convrot', False)):
            group_size = int(config.get('convrot_groupsize', params.get('convrot_groupsize', 256)))
            if not _valid_group_size(group_size) or weight.shape[1] % group_size:
                raise ComfyQuantError(
                    f'Layer "{name}" has ConvRot group size {group_size}, which does not '
                    f'divide its {weight.shape[1]} inputs or is not a power of 4.')
        return _QuantLayer(name, fmt, weight, _float_bits(scale),
                           pre_quant_scale=pre_quant_scale,
                           convrot_group_size=group_size, full_precision=full_precision)

    # nvfp4
    scale_2 = checkpoint.get(name + '.weight_scale_2')
    if weight.dtype != torch.uint8 or scale_2 is None:
        raise ComfyQuantError(
            f'NVFP4 layer "{name}" needs uint8 weight storage and weight_scale_2.')
    rows, cols = weight.shape[0], weight.shape[1] * 2
    if rows % _NVFP4_BLOCK or cols % _NVFP4_BLOCK:
        raise ComfyQuantError(f'NVFP4 layer "{name}" is not padded to 16x16 blocks.')
    tiled = (_round_up(rows, 128), _round_up(cols // _NVFP4_BLOCK, 4))
    if tuple(scale.shape) != tiled:
        raise ComfyQuantError(
            f'NVFP4 layer "{name}" has block scales of shape {tuple(scale.shape)}, '
            f'expected the tiled shape {tiled}.')
    return _QuantLayer(name, fmt, weight, scale.view(torch.uint8).contiguous(),
                       scale_2=_float_bits(scale_2.reshape(())),
                       pre_quant_scale=pre_quant_scale, full_precision=full_precision)


def _one_or_per_row(values: torch.Tensor) -> torch.Tensor:
    """A single value when every row has the same one, otherwise one per row."""
    if bool((values == values[0]).all()):
        return values[:1].reshape(())
    return values


def _assemble_layer(name: str, segments: list, storage: torch.Tensor) -> _QuantLayer | None:
    """
    The quantized layer a checkpoint conversion builds from rows of source layers.

    ``segments`` are ``(layer, start, stop)`` stored row ranges in output order,
    and ``storage`` is the converted packed weight: those rows stacked. Rows of
    packed storage are independent in every format, so only the scales need
    rebuilding. ``None`` when the sources cannot share one layer.
    """
    first = segments[0][0]
    for layer, _, _ in segments:
        if (layer.format != first.format
                or layer.convrot_group_size != first.convrot_group_size
                or layer.pre_quant_scale is not first.pre_quant_scale
                or layer.weight.shape[1] != first.weight.shape[1]):
            return None
    rows = sum(stop - start for _, start, stop in segments)
    if (storage.dtype != first.weight.dtype or storage.ndim != 2
            or tuple(storage.shape) != (rows, first.weight.shape[1])):
        return None
    if len(segments) == 1 and storage is first.weight:
        return first
    full_precision = any(layer.full_precision for layer, _, _ in segments)

    if first.format == 'nvfp4':
        blocks = torch.cat([layer.block_scales()[start:stop] for layer, start, stop in segments])
        tensor = torch.cat([layer.tensor_scales()[start:stop] for layer, start, stop in segments])
        padded = _round_up(rows, _NVFP4_BLOCK)
        if padded != rows:
            storage = _pad_rows(storage, padded)
            blocks = _pad_rows(blocks, padded)
            tensor = torch.nn.functional.pad(tensor, (0, padded - rows), value=1.0)
        return _QuantLayer(name, 'nvfp4', storage, _to_blocked(blocks).contiguous(),
                           scale_2=_float_bits(_one_or_per_row(tensor)),
                           pre_quant_scale=first.pre_quant_scale, full_precision=full_precision)

    scales = torch.cat([layer.row_scales()[start:stop] for layer, start, stop in segments])
    scale = _one_or_per_row(scales)
    return _QuantLayer(name, first.format, storage,
                       _float_bits(scale if scale.ndim == 0 else scale.reshape(-1, 1)),
                       pre_quant_scale=first.pre_quant_scale,
                       convrot_group_size=first.convrot_group_size, full_precision=full_precision)


def _dequantize_segments(segments: list, dtype: torch.dtype,
                         shape: tuple[int, int] | None = None) -> torch.Tensor:
    """Dense rows of source layers, stacked and trimmed to ``shape``."""
    weight = torch.cat([layer.dequantize(torch.float32)[start:stop]
                        for layer, start, stop in segments])
    if shape is not None:
        weight = weight[:shape[0], :shape[1]]
    return weight.to(dtype).contiguous()


class _Trace:
    """Sources a checkpoint conversion used in a way rows cannot follow."""

    def __init__(self):
        self.tainted = set()


# Torch functions whose outputs are row views of their first argument.
_ROW_VIEW_FUNCTIONS = frozenset({
    torch.chunk, torch.Tensor.chunk,
    torch.split, torch.Tensor.split,
    torch.tensor_split, torch.Tensor.tensor_split,
    torch.narrow, torch.Tensor.narrow,
    torch.Tensor.__getitem__,
})

# Torch functions that keep every row in place.
_ROW_COPY_FUNCTIONS = frozenset({
    torch.Tensor.contiguous, torch.Tensor.clone, torch.clone, torch.Tensor.detach,
    torch.Tensor.to, torch.Tensor.view, torch.Tensor.reshape, torch.reshape,
})

_CAT_FUNCTIONS = frozenset({torch.cat, torch.concat, torch.concatenate})


class _RowTrace(torch.Tensor):
    """
    A meta tensor that records which source rows it holds.

    Diffusers checkpoint conversions split fused projections (qkv) and reorder
    modulation weights with row slices and concatenations. Packed storage
    keeps rows intact, so those operations carry over to it unchanged and only
    the scales need following. Any other use of a traced tensor marks its
    sources untraceable; they are dequantized before converting.
    """

    @staticmethod
    def wrap(meta: torch.Tensor, segments: tuple, trace: _Trace) -> '_RowTrace':
        tensor = torch.Tensor._make_subclass(_RowTrace, meta, False)
        tensor.segments = tuple(segments)
        tensor.trace = trace
        return tensor

    def taint(self) -> None:
        self.trace.tainted.update(source for source, _, _ in self.segments)
        self.segments = ()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        traced = [value for value in _flatten((args, kwargs)) if isinstance(value, _RowTrace)]
        with torch._C.DisableTorchFunctionSubclass():
            result = func(*args, **kwargs)
        if not any(isinstance(value, torch.Tensor) for value in _flatten(result)):
            if func is torch.Tensor.__setitem__:
                for value in traced:
                    value.taint()
            # Shapes, dtypes, and other metadata.
            return result
        followed = _follow_rows(func, args, kwargs, result)
        if followed is not None:
            return followed
        for value in traced:
            value.taint()
        return _plain(result)


def _flatten(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten(item)
    else:
        yield value


def _plain(value):
    if isinstance(value, _RowTrace):
        return value.as_subclass(torch.Tensor)
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(item) for item in value)
    return value


def _row_range(source: torch.Tensor, view: torch.Tensor) -> tuple[int, int] | None:
    """The rows of contiguous ``source`` that ``view`` covers, if it is a whole row slab."""
    if view.ndim != source.ndim or view.shape[1:] != source.shape[1:] or view.stride() != source.stride():
        return None
    row = source.stride(0) if source.ndim > 1 else 1
    offset = view.storage_offset() - source.storage_offset()
    if offset % row:
        return None
    start = offset // row
    return start, start + view.shape[0]


def _slice_segments(segments: tuple, start: int, stop: int) -> tuple:
    """Source row ranges for rows ``start:stop`` of a tensor made of ``segments``."""
    result = []
    position = 0
    for source, begin, end in segments:
        length = end - begin
        low, high = max(start, position), min(stop, position + length)
        if low < high:
            result.append((source, begin + low - position, begin + high - position))
        position += length
    return tuple(result)


def _follow_rows(func, args, kwargs, result):
    """``result`` as row traces when ``func`` only moved whole rows, else ``None``."""
    source = args[0] if args else None
    if func in _ROW_VIEW_FUNCTIONS and isinstance(source, _RowTrace) and source.segments:
        if not source.is_contiguous():
            return None
        outputs = result if isinstance(result, (list, tuple)) else (result,)
        traced = []
        for output in outputs:
            rows = _row_range(source, output) if isinstance(output, torch.Tensor) else None
            if rows is None:
                return None
            traced.append(_RowTrace.wrap(output, _slice_segments(source.segments, *rows), source.trace))
        return type(result)(traced) if isinstance(result, (list, tuple)) else traced[0]

    if func in _ROW_COPY_FUNCTIONS and isinstance(source, _RowTrace) and source.segments:
        if (isinstance(result, torch.Tensor) and result.shape == source.shape
                and result.dtype == source.dtype):
            return _RowTrace.wrap(result, source.segments, source.trace)
        return None

    if func in _CAT_FUNCTIONS:
        tensors = args[0] if args else kwargs.get('tensors')
        dim = args[1] if len(args) > 1 else kwargs.get('dim', 0)
        if (not tensors or not all(isinstance(t, _RowTrace) and t.segments for t in tensors)
                or dim not in (0, -tensors[0].ndim)):
            return None
        segments = tuple(segment for t in tensors for segment in t.segments)
        return _RowTrace.wrap(result, segments, tensors[0].trace)
    return None


class ComfyQuantCheckpoint:
    """
    A ComfyUI quantized checkpoint prepared for ``from_single_file``.

    ``checkpoint`` holds the original tensor names with every auxiliary scale
    removed, so Diffusers' checkpoint conversion sees an ordinary state dict.
    Quantized weights keep their packed storage (float8 as uint8); their
    layers, scales included, are in ``sources`` by weight name.
    """

    def __init__(self, checkpoint: dict, sources: dict[str, _QuantLayer], dtype: torch.dtype):
        self.checkpoint = checkpoint
        self.sources = sources
        self.dtype = dtype
        # Converted weight name -> source row ranges, once a conversion is traced.
        self.plan = None

    @property
    def quantization_config(self):
        """Pass as ``quantization_config``, or ``None`` when every layer is dense."""
        if not self.sources:
            return None
        return _quantization_config(self, self.dtype)

    def _densify(self, key: str) -> None:
        layer = self.sources.pop(key)
        self.checkpoint[key] = layer.dequantize(self.dtype)

    def densify_all(self) -> None:
        """Dequantize every layer to a dense ``dtype`` weight."""
        for key in list(self.sources):
            self._densify(key)

    def segments_for(self, key: str, value) -> list | None:
        """``(layer, start, stop)`` source rows behind a converted weight, or ``None``."""
        if self.plan is not None:
            segments = self.plan.get(key)
            if segments is None:
                return None
            return [(self.sources[source], start, stop) for source, start, stop in segments]
        # Diffusers skipped conversion: the names already matched the model.
        layer = self.sources.get(key)
        if layer is not None and layer.weight is value:
            return [(layer, 0, layer.weight.shape[0])]
        return None

    def _plan_conversion(self, mapping_fn, kwargs: dict) -> None:
        """
        Trace a Diffusers checkpoint conversion through the quantized weights.

        The conversion runs on meta tensors; quantized weights are row traces.
        Layers the conversion only renames, splits, or concatenates by rows
        stay packed. Layers it uses any other way are dequantized first.
        """
        trace = _Trace()
        probe = {}
        for key, value in self.checkpoint.items():
            if not isinstance(value, torch.Tensor):
                probe[key] = value
                continue
            meta = torch.empty_like(value, device='meta')
            if key in self.sources:
                meta = _RowTrace.wrap(meta, ((key, 0, value.shape[0]),), trace)
            probe[key] = meta
        try:
            converted = mapping_fn(**dict(kwargs, checkpoint=probe))
        except Exception as e:
            _messages.debug_log(
                f'Could not trace the checkpoint conversion ({e}), '
                f'dequantizing every ComfyUI quantized layer.')
            self.densify_all()
            self.plan = {}
            return
        plan = {}
        for key, value in converted.items():
            if isinstance(value, _RowTrace) and value.segments:
                if key.endswith('.weight'):
                    plan[key] = value.segments
                else:
                    value.taint()
        tainted = trace.tainted & self.sources.keys()
        if tainted:
            _messages.debug_log(
                f'Dequantizing {len(tainted)} ComfyUI quantized layers that the '
                f'checkpoint conversion does not just split or join by rows.')
        for key in tainted:
            self._densify(key)
        self.plan = {
            key: segments for key, segments in plan.items()
            if not any(source in tainted for source, _, _ in segments)
        }

    @contextlib.contextmanager
    def conversion_guard(self, model_class):
        """
        Within this context, ``model_class.from_single_file`` traces its
        checkpoint conversion (see :py:meth:`_plan_conversion`) before converting.
        """
        import diffusers.loaders.single_file_model as single_file_model

        name = single_file_model._get_single_file_loadable_mapping_class(model_class)
        if name is None or not self.sources:
            yield
            return
        entry = single_file_model.SINGLE_FILE_LOADABLE_CLASSES[name]
        original = entry['checkpoint_mapping_fn']

        # Diffusers passes only the kwargs named in the mapping function's
        # signature; wraps() keeps that signature visible.
        @functools.wraps(original)
        def mapping_fn(*args, **kwargs):
            if not args and kwargs.get('checkpoint') is self.checkpoint:
                self._plan_conversion(original, kwargs)
            return original(*args, **kwargs)

        entry['checkpoint_mapping_fn'] = mapping_fn
        try:
            yield
        finally:
            entry['checkpoint_mapping_fn'] = original


def _quantized_layer_names(raw: dict) -> list[str]:
    """
    Names of the quantized linears in a raw checkpoint.

    A layer is quantized when it has a ``comfy_quant`` marker; or, in files
    without markers, a scale beside int8 / float8 storage, or uint8 storage
    with a second scale (NVFP4); or a 2D float8 weight with no scale at all.
    Other ``weight_scale`` params are left alone.
    """
    packed_dtypes = (torch.int8, *_FLOAT8_DTYPES.values())
    names = {key[:-len('.comfy_quant')] for key in raw if key.endswith('.comfy_quant')}
    names.update(
        key[:-len('.weight_scale')] for key in raw
        if key.endswith('.weight_scale')
        and (getattr(raw.get(key[:-len('_scale')]), 'dtype', None) in packed_dtypes
             or key + '_2' in raw))
    names.update(
        key[:-len('.weight')] for key, value in raw.items()
        if key.endswith('.weight') and value.ndim == 2
        and value.dtype in _FLOAT8_DTYPES.values())
    return sorted(name for name in names if name + '.weight' in raw)


def _read_tensors(path: str, load=None) -> dict:
    """
    The tensors of a ``.safetensors`` file. With ``load``, a predicate on the
    name, other tensors are meta tensors of the stored shape and dtype.
    """
    import safetensors.torch

    if load is None:
        return safetensors.torch.load_file(path)
    from safetensors import safe_open

    tensors = {}
    with safe_open(path, 'pt') as file:
        for key in file.keys():
            if load(key):
                tensors[key] = file.get_tensor(key)
            else:
                piece = file.get_slice(key)
                tensors[key] = torch.empty(
                    piece.get_shape(), dtype=_TORCH_DTYPES[piece.get_dtype()], device='meta')
    return tensors


_TORCH_DTYPES = {
    'F64': torch.float64, 'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16,
    'F8_E4M3': torch.float8_e4m3fn, 'F8_E5M2': torch.float8_e5m2,
    'I64': torch.int64, 'I32': torch.int32, 'I16': torch.int16, 'I8': torch.int8,
    'U8': torch.uint8, 'BOOL': torch.bool,
}


def load_checkpoint(path: str, dtype: torch.dtype | None,
                    dequantize: bool = False, load=None) -> ComfyQuantCheckpoint:
    """
    Read a ComfyUI quantized ``.safetensors`` file.

    :param path: local file path
    :param dtype: dtype for unquantized and dequantized weights, ``None`` for bfloat16
    :param dequantize: Dequantize every layer, for example so another
        quantizer or LoRA fusion can work on dense weights.
    :param load: Optional predicate on tensor names; others are read as meta
        tensors (shape and dtype only). ``comfy_quant`` markers are always read.
    :return: :py:class:`ComfyQuantCheckpoint`
    """
    dtype = dtype or torch.bfloat16
    raw = _read_tensors(
        path, None if load is None else lambda key: key.endswith('.comfy_quant') or load(key))
    raw.pop(_LEGACY_MARKER, None)
    raw = _rename_legacy_scales(raw)

    names = _quantized_layer_names(raw)
    sources = {}
    for name in names:
        layer = _extract_layer(raw, name)
        sources[name + '.weight'] = layer
        raw[name + '.weight'] = layer.weight

    checkpoint = {}
    for key, value in raw.items():
        if key.endswith(_AUX_SUFFIXES) and key.rsplit('.', 1)[0] in names:
            continue
        if key not in sources and value.dtype in _FLOAT8_DTYPES.values():
            # Other float8 tensors (biases, norms) are plain low precision storage.
            value = value.to(dtype)
        checkpoint[key] = value

    loaded = ComfyQuantCheckpoint(checkpoint, sources, dtype)
    formats = sorted({layer.format for layer in sources.values()})
    _messages.debug_log(
        f'ComfyUI quantized checkpoint "{path}": {len(sources)} quantized layers '
        f'({", ".join(formats)}){", dequantizing" if dequantize else ""}.')
    if dequantize:
        loaded.densify_all()
    return loaded


# ==== Components read through Diffusers' single file loader ====
#
# Pipelines and dgenerate load text encoders, VAEs, and UNets out of a single
# file with ``load_single_file_checkpoint``, which knows nothing of ComfyUI
# quantization. For a ComfyUI quantized file it returns the checkpoint with the
# quantized layers dequantized. Within :py:func:`single_file_components` with
# ``transformer_loaded`` set, dgenerate loads the transformer itself (packed,
# through ``--transformer`` or :py:class:`TransformerUri`), so only text encoder
# and VAE layers are dequantized and the rest become meta placeholders: a
# transformer only file costs nothing, and a stray load of those weights fails
# instead of loading garbage.

# Key prefixes of components other than the denoiser in ComfyUI / A1111 checkpoints.
_COMPONENT_PREFIXES = (
    'text_encoders.', 'cond_stage_model.', 'conditioner.', 'first_stage_model.', 'vae.')

_single_file_scope = None


@contextlib.contextmanager
def single_file_components(transformer_loaded: bool):
    """
    Scope one pipeline creation: cache dequantized checkpoints per path, and
    leave the denoiser to dgenerate when ``transformer_loaded``.
    """
    global _single_file_scope
    previous = _single_file_scope
    _single_file_scope = {'transformer_loaded': transformer_loaded, 'cache': {}}
    try:
        yield
    finally:
        _single_file_scope = previous


def _component_checkpoint(path: str) -> dict:
    scope = _single_file_scope
    keep_denoiser = bool(scope and scope['transformer_loaded'])
    if scope is not None and path in scope['cache']:
        return dict(scope['cache'][path])
    # With the denoiser left to dgenerate, only component weights are read.
    loaded = load_checkpoint(
        path, torch.bfloat16,
        load=(lambda key: key.startswith(_COMPONENT_PREFIXES)) if keep_denoiser else None)
    for key in list(loaded.sources):
        if keep_denoiser and not key.startswith(_COMPONENT_PREFIXES):
            layer = loaded.sources.pop(key)
            loaded.checkpoint[key] = torch.empty(
                layer.logical_shape(), dtype=torch.bfloat16, device='meta')
        else:
            loaded._densify(key)
    if scope is not None:
        scope['cache'][path] = loaded.checkpoint
    # Conversions pop keys, so each caller gets its own dict.
    return dict(loaded.checkpoint)


@functools.cache
def install_single_file_patch() -> None:
    """Route Diffusers' single file checkpoint reads of ComfyUI quantized files through here."""
    import diffusers.loaders.single_file as single_file
    import diffusers.loaders.single_file_model as single_file_model
    import diffusers.loaders.single_file_utils as single_file_utils

    original = single_file_utils.load_single_file_checkpoint

    @functools.wraps(original)
    def load_single_file_checkpoint(pretrained_model_link_or_path, *args, **kwargs):
        path = pretrained_model_link_or_path
        if isinstance(path, str) and is_comfy_quantized(path):
            return _component_checkpoint(path)
        return original(pretrained_model_link_or_path, *args, **kwargs)

    for module in (single_file_utils, single_file, single_file_model):
        if getattr(module, 'load_single_file_checkpoint', None) is original:
            module.load_single_file_checkpoint = load_single_file_checkpoint


# ==== Hardware matmul ====
#
# Each format can run as a hardware matmul. Candidates are tried in order for
# the format of the layer and the device:
#
#   INT8   MPS    MLX ``quantized_matmul`` affine 8 bit (Hub), then PyTorch
#                 ``_weight_int8pack_mm``
#          ROCm   AITER ``gemm_a8w8`` Triton kernel (Hub), then ``torch._int_mm``
#          CUDA   ``torch._int_mm`` (sm75+)
#          XPU    ``torch._int_mm``
#   FP8    MPS    MLX ``quantized_matmul`` mxfp8 (Hub)
#          ROCm   AITER ``gemm_a8w8``, then ``torch._scaled_mm`` (MI300 reads the
#                 e4m3fn bits as e4m3fnuz, half the value; MI350, RDNA4 e4m3fn)
#          CUDA   ``torch._scaled_mm`` (sm89+)
#          XPU    ``torch._scaled_mm``
#   NVFP4  MPS    MLX ``quantized_matmul`` nvfp4 (Hub)
#          CUDA   block scaled ``scaled_mm`` (Blackwell)
#
# The MPS kernels are weight only: the weight is decoded inside the matmul and
# the activations keep their dtype, so they also serve layers marked
# ``full_precision_matrix_mult``. The others quantize the activations on the
# fly as ComfyUI does (W8A8, W4A4), so those layers skip them, as in ComfyUI.
#
# Hub kernels come from the ``kernels`` package, like the bitsandbytes MPS
# kernels, and are skipped when ``DIFFUSERS_DISABLE_REMOTE_CODE`` is set or
# they cannot be loaded (offline without a cached copy, no build for this torch).
#
# The first call of each candidate for a layer shape (rows bucketed by powers
# of two) is compared against the dequantized matmul, for accuracy and for
# speed. A failure or a mismatch turns the candidate off on the device; a
# slower kernel is skipped for that shape. The next candidate is tried either
# way, so a kernel or layout difference on untested hardware costs speed, not
# correctness.

SOFTWARE_MATMUL_ENVVAR = 'DGENERATE_COMFY_QUANT_SOFTWARE'
"""Set to ``1`` to always use the dequantized matmul."""

_HUB_KERNELS = {
    'mlx': ('kernels-community/mlx-quantization-metal-kernels', 2),
    'aiter': ('kernels-community/aiter-kernels', 2),
}

_hub_kernel_cache = {}

_HARDWARE_DEVICE_TYPES = ('cuda', 'xpu', 'mps')

# (candidate, device) -> False once a kernel fails or computes a wrong result.
_hardware_state = {}

# (candidate, device, rows bucket, outputs, inputs) -> whether the kernel is
# accurate and faster than the dequantized matmul for that layer shape.
_shape_state = {}

# Upper bounds of the E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4 (6 above).
_E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _remote_code_disabled() -> bool:
    value = os.environ.get('DIFFUSERS_DISABLE_REMOTE_CODE', '').strip().lower()
    return value in ('1', 'true', 'yes', 'on')


def _hub_kernel(name: str):
    """A ``kernels`` Hub kernel module, or ``None`` when it cannot be used here."""
    if name in _hub_kernel_cache:
        return _hub_kernel_cache[name]
    module = None
    repo, version = _HUB_KERNELS[name]
    if not _remote_code_disabled():
        try:
            from kernels import get_kernel
            module = get_kernel(repo, version=version)
        except Exception as e:
            _messages.debug_log(f'Hub kernel "{repo}" is not available: {e}')
    _hub_kernel_cache[name] = module
    return module


def _aiter_gemm_a8w8():
    key = 'aiter.gemm_a8w8'
    if key not in _hub_kernel_cache:
        function = None
        module = _hub_kernel('aiter')
        if module is not None:
            try:
                function = importlib.import_module(
                    f'{module.__name__}.gemm.basic.gemm_a8w8').gemm_a8w8
            except Exception as e:
                _messages.debug_log(f'AITER gemm_a8w8 is not available: {e}')
        _hub_kernel_cache[key] = function
    return _hub_kernel_cache[key]


def _rocm_arch(device: torch.device) -> str:
    return getattr(torch.cuda.get_device_properties(device), 'gcnArchName', '')


def _float8_matmul_format(device: torch.device) -> tuple[torch.dtype, float, float] | None:
    """
    ``(float8 dtype, its largest value, weight scale factor)`` for FP8 matmuls on ``device``.

    MI300 only multiplies float8_e4m3fnuz, whose exponent bias is one higher:
    the same bits are half the e4m3fn value, so the weight scale doubles.
    """
    if device.type == 'cuda' and torch.version.hip:
        arch = _rocm_arch(device)
        if 'gfx950' in arch or 'gfx12' in arch:
            return torch.float8_e4m3fn, 448.0, 1.0
        if any(name in arch for name in ('gfx940', 'gfx941', 'gfx942')):
            return torch.float8_e4m3fnuz, 240.0, 2.0
        return None
    return torch.float8_e4m3fn, 448.0, 1.0


# Kernels that decode the weight inside the matmul and leave activations alone.
_WEIGHT_ONLY = frozenset({'int8-mlx', 'int8-pack', 'fp8-mlx', 'nvfp4-mlx'})


def _candidates(module: 'ComfyQuantLinear', device: torch.device) -> tuple[str, ...]:
    """Hardware matmuls to try for ``module`` on ``device``, best first."""
    candidates = _format_candidates(module.quant_format, device)
    if module.full_precision_mm:
        candidates = tuple(name for name in candidates if name in _WEIGHT_ONLY)
    return candidates


def _format_candidates(fmt: str, device: torch.device) -> tuple[str, ...]:
    hip = device.type == 'cuda' and bool(torch.version.hip)
    if fmt == 'int8_tensorwise':
        if device.type == 'mps':
            return 'int8-mlx', 'int8-pack'
        if hip:
            return 'int8-aiter', 'int8-mm'
        if device.type == 'cuda':
            return ('int8-mm',) if torch.cuda.get_device_capability(device) >= (7, 5) else ()
        if device.type == 'xpu':
            return ('int8-mm',)
    elif fmt == 'float8_e4m3fn':
        if device.type == 'mps':
            return ('fp8-mlx',)
        if hip:
            return ('fp8-aiter', 'fp8-scaled-mm') if _float8_matmul_format(device) else ()
        if device.type == 'cuda':
            return ('fp8-scaled-mm',) if torch.cuda.get_device_capability(device) >= (8, 9) else ()
        if device.type == 'xpu':
            return ('fp8-scaled-mm',)
    elif fmt == 'nvfp4':
        if device.type == 'mps':
            return ('nvfp4-mlx',)
        if (device.type == 'cuda' and not hip
                and torch.cuda.get_device_capability(device)[0] in (10, 12)
                and hasattr(torch.nn.functional, 'scaled_mm')
                and hasattr(torch, 'float4_e2m1fn_x2')):
            return ('nvfp4',)
    return ()


def _pad_rows(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    if tensor.shape[0] == rows:
        return tensor
    return torch.nn.functional.pad(tensor, (0, 0, 0, rows - tensor.shape[0]))


def _row_absmax(x: torch.Tensor, dim=None, keepdim=False) -> torch.Tensor:
    # One read of ``x``, no ``abs()`` temporary.
    return torch.linalg.vector_norm(x, ord=math.inf, dim=dim, keepdim=keepdim)


def _weight_row_scales(module: 'ComfyQuantLinear') -> torch.Tensor:
    """float32 weight scale per output row, ``(outputs,)``."""
    outputs = module.weight.shape[0]
    scale = _broadcast_scale(_bits_float(module.weight_scale), outputs)
    return scale.reshape(-1).expand(outputs)


def _quantize_int8_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per row INT8 activations and their float32 ``(rows, 1)`` scales."""
    x_scale = _row_absmax(x, dim=-1, keepdim=True).float().clamp_(min=1e-6) / 127
    # Scale in float32: casting small bfloat16 activations (|x| < 6e-5) to float16
    # first makes them float16 subnormals, which lose most of their bits.
    quantized = (x.float() * (1 / x_scale)).round_().to(torch.int8)
    return quantized, x_scale


def _int8_mm(module: 'ComfyQuantLinear', x: torch.Tensor) -> torch.Tensor | None:
    """W8A8 through ``torch._int_mm``."""
    rows, inputs = x.shape
    outputs = module.weight.shape[0]
    if inputs % 8 or outputs % 8:
        return None
    quantized, x_scale = _quantize_int8_rows(x)
    # cuBLASLt INT8 needs more than 16 rows; pad to a multiple of 32.
    quantized = _pad_rows(quantized, max(32, _round_up(rows, 32)))
    accumulated = torch._int_mm(quantized, module.weight.t())[:rows]
    w_scale = _weight_row_scales(module).reshape(1, -1)
    if x.dtype == torch.bfloat16:
        # Sums reach about 127 * 127 * inputs: fine for the range of bfloat16, which
        # keeps the scaling to one conversion and two in-place passes.
        output = accumulated.to(x.dtype)
        return output.mul_(x_scale.to(x.dtype)).mul_(w_scale.to(x.dtype))
    # Too large for float16.
    return (accumulated.float() * x_scale * w_scale).to(x.dtype)


def _int8_aiter(module: 'ComfyQuantLinear', x: torch.Tensor) -> torch.Tensor | None:
    """W8A8 through the AITER Triton ``gemm_a8w8``, scales applied in the kernel."""
    gemm = _aiter_gemm_a8w8()
    if gemm is None:
        raise RuntimeError('AITER gemm_a8w8 is not available')
    if x.dtype not in (torch.bfloat16, torch.float16):
        return None
    quantized, x_scale = _quantize_int8_rows(x)
    return gemm(quantized, module.weight, x_scale, _weight_row_scales(module).contiguous(),
                dtype=x.dtype)


def _mlx_group_size(inputs: int) -> int | None:
    for size in (128, 64, 32):
        if inputs % size == 0:
            return size
    return None


def _mlx_kernel():
    kernel = _hub_kernel('mlx')
    if kernel is None:
        raise RuntimeError('MLX quantization Metal kernels are not available')
    return kernel


def _int8_mlx(module: 'ComfyQuantLinear', x: torch.Tensor) -> torch.Tensor | None:
    """
    INT8 through the MLX affine 8 bit ``quantized_matmul`` Metal kernel.

    MLX affine weights are unsigned codes ``q`` with ``w = scale * q + bias``,
    four to a little endian uint32. A signed code ``c`` is ``q = c + 128`` (the sign
    bit flipped), so ``bias = -128 * scale`` reproduces the weight exactly.
    """
    kernel = _mlx_kernel()
    outputs, inputs = module.weight.shape
    group_size = _mlx_group_size(inputs)
    if group_size is None:
        return None
    scales = _weight_row_scales(module).to(x.dtype).reshape(-1, 1)
    scales = scales.expand(outputs, inputs // group_size).contiguous()
    return kernel.quantized_matmul(
        x.contiguous(), _mlx_int8_codes(module.weight), scales, scales * -128,
        transpose=True, group_size=group_size, bits=8, mode='affine')


def _mlx_int8_codes(weight: torch.Tensor) -> torch.Tensor:
    """Signed INT8 ``(outputs, inputs)`` -> MLX 8 bit codes ``(outputs, inputs / 4)`` uint32."""
    return (weight.view(torch.uint8) ^ 0x80).view(torch.uint32)


def _fp8_mlx(module: 'ComfyQuantLinear', x: torch.Tensor) -> torch.Tensor | None:
    """
    FP8 through the MLX mxfp8 ``quantized_matmul`` Metal kernel.

    mxfp8 is e4m3 bytes, four to a little endian uint32, with a power of two
    (e8m0) scale per 32 inputs. Every block scale is 2^0 (127), so the weight
    bytes are used as they are, and the per tensor or per row scale of the
    file is applied to the output.
    """
    kernel = _mlx_kernel()
    outputs, inputs = module.weight.shape
    if inputs % 32:
        return None
    unit_scales = torch.full((outputs, inputs // 32), 127, dtype=torch.uint8, device=x.device)
    output = kernel.quantized_matmul(
        x.contiguous(), module.weight.view(torch.uint32), unit_scales, None,
        transpose=True, group_size=32, bits=8, mode='mxfp8')
    return output * _weight_row_scales(module).to(x.dtype).reshape(1, -1)


def _mlx_nvfp4_codes(weight: torch.Tensor) -> torch.Tensor:
    """ComfyUI NVFP4 bytes (high nibble first) -> MLX 4 bit codes (low nibble first) as uint32."""
    return ((weight << 4) | (weight >> 4)).view(torch.uint32)


def _nvfp4_mlx(module: 'ComfyQuantLinear', x: torch.Tensor) -> torch.Tensor | None:
    """
    NVFP4 through the MLX nvfp4 ``quantized_matmul`` Metal kernel.

    The same E2M1 codes and e4m3 block scales per 16 inputs, but MLX packs the
    low nibble first and takes the block scales row major, not tiled. The
    tensor scale is applied to the output.
    """
    kernel = _mlx_kernel()
    rows, packed = module.weight.shape
    inputs = packed * 2
    if x.shape[1] != inputs:
        # The weight was padded to a multiple of 16 inputs.
        x = torch.nn.functional.pad(x, (0, inputs - x.shape[1]))
    block_scales = _unswizzle_block_scales(module.weight_scale, rows, inputs // _NVFP4_BLOCK)
    output = kernel.quantized_matmul(
        x.contiguous(), _mlx_nvfp4_codes(module.weight), block_scales, None,
        transpose=True, group_size=_NVFP4_BLOCK, bits=4, mode='nvfp4')
    tensor_scale = _bits_float(module.weight_scale_2).to(x.dtype)
    if tensor_scale.numel() > 1:
        tensor_scale = tensor_scale.reshape(1, -1)[:, :module.out_features]
    return output[:, :module.out_features] * tensor_scale


def _int8_pack(module: 'ComfyQuantLinear', x: torch.Tensor) -> torch.Tensor | None:
    """Weight only INT8 through the PyTorch ``_weight_int8pack_mm`` (Metal on MPS)."""
    scales = _weight_row_scales(module).to(x.dtype).contiguous()
    return torch._weight_int8pack_mm(x.contiguous(), module.weight, scales)


def _quantize_fp8_tensor(x: torch.Tensor, float8: torch.dtype, largest: float):
    x_scale = (_row_absmax(x).float() / largest).clamp_(min=1e-12)
    # |x| / x_scale <= the largest finite float8 value.
    return (x * (1 / x_scale).to(x.dtype)).to(float8), x_scale


def _fp8_scaled_mm(module: 'ComfyQuantLinear', x: torch.Tensor) -> torch.Tensor | None:
    """
    W8A8 through ``torch._scaled_mm`` with a dynamic per tensor activation scale.

    ComfyUI uses the static ``input_scale`` of the file, which clips activations
    above it; a dynamic scale cannot clip.
    """
    w_scale = _bits_float(module.weight_scale)
    rows, inputs = x.shape
    outputs = module.weight.shape[0]
    if w_scale.numel() != 1 or inputs % 16 or outputs % 16:
        return None
    float8, largest, weight_factor = _float8_matmul_format(x.device)
    quantized, x_scale = _quantize_fp8_tensor(x, float8, largest)
    quantized = _pad_rows(quantized, _round_up(rows, 16))
    output = torch._scaled_mm(
        quantized, module.weight.view(float8).t(),
        scale_a=x_scale.reshape(()), scale_b=(w_scale * weight_factor).reshape(()),
        out_dtype=x.dtype)
    return output[:rows]


def _fp8_aiter(module: 'ComfyQuantLinear', x: torch.Tensor) -> torch.Tensor | None:
    """W8A8 through the AITER Triton ``gemm_a8w8``."""
    gemm = _aiter_gemm_a8w8()
    if gemm is None:
        raise RuntimeError('AITER gemm_a8w8 is not available')
    if x.dtype not in (torch.bfloat16, torch.float16):
        return None
    float8, largest, weight_factor = _float8_matmul_format(x.device)
    quantized, x_scale = _quantize_fp8_tensor(x, float8, largest)
    w_scale = _weight_row_scales(module) * weight_factor
    return gemm(quantized, module.weight.view(float8),
                x_scale.reshape(1, 1).expand(x.shape[0], 1).contiguous(),
                w_scale.contiguous(), dtype=x.dtype)


def _to_blocked(scales: torch.Tensor) -> torch.Tensor:
    """``(rows, cols)`` block scales -> the cuBLAS 128x4 tiled layout (inverse of the unswizzle)."""
    rows, cols = scales.shape
    row_tiles, col_tiles = -(-rows // 128), -(-cols // 4)
    padded = scales.new_zeros(row_tiles * 128, col_tiles * 4)
    padded[:rows, :cols] = scales
    tiles = padded.view(row_tiles, 128, col_tiles, 4).permute(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(row_tiles * 128, col_tiles * 4)


def _quantize_nvfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize ``(rows, cols)`` activations like the checkpoint weights.

    :return: packed E2M1 codes (high nibble first) ``(roundup(rows, 16), cols / 2)``,
        tiled float8 block scales as uint8, float32 tensor scale
    """
    rows, cols = x.shape
    x = _pad_rows(x.float(), _round_up(rows, _NVFP4_BLOCK))
    tensor_scale = (x.abs().amax() / (448 * 6)).clamp_(min=1e-12)
    blocks = x.view(x.shape[0], cols // _NVFP4_BLOCK, _NVFP4_BLOCK)
    block_scale = (blocks.abs().amax(dim=-1) / 6 / tensor_scale).clamp_(max=448)
    block_scale = block_scale.to(torch.float8_e4m3fn)
    total = block_scale.float() * tensor_scale
    total = torch.where(total == 0, torch.ones_like(total), total)
    scaled = blocks / total.unsqueeze(-1)
    bounds = torch.tensor(_E2M1_BOUNDS, device=x.device)
    codes = torch.bucketize(scaled.abs(), bounds) | ((scaled < 0).long() << 3)
    codes = codes.view(x.shape[0], cols).to(torch.uint8)
    packed = (codes[:, 0::2] << 4) | codes[:, 1::2]
    return packed, _to_blocked(block_scale.view(torch.uint8)), tensor_scale


def _nvfp4_scaled_mm(module: 'ComfyQuantLinear', x: torch.Tensor) -> torch.Tensor | None:
    """W4A4 with block scaled ``scaled_mm`` (Blackwell)."""
    if module.weight_scale_2.numel() != 1:
        # Rows joined from layers with different tensor scales; scaled_mm takes one.
        return None
    functional = torch.nn.functional
    rows = x.shape[0]
    inputs = module.weight.shape[1] * 2
    if x.shape[1] != inputs:
        # The weight was padded to a multiple of 16 inputs.
        x = functional.pad(x, (0, inputs - x.shape[1]))
    packed, block_scale, tensor_scale = _quantize_nvfp4(x)
    fp4 = torch.float4_e2m1fn_x2
    recipe = [functional.ScalingType.BlockWise1x16, functional.ScalingType.TensorWise]
    swizzle = [functional.SwizzleType.SWIZZLE_32_4_4, functional.SwizzleType.NO_SWIZZLE]
    output = functional.scaled_mm(
        packed.view(fp4), module.weight.view(fp4).t(),
        scale_a=[block_scale.view(torch.float8_e4m3fn).view(-1), tensor_scale],
        scale_recipe_a=recipe,
        scale_b=[module.weight_scale.view(torch.float8_e4m3fn).view(-1),
                 _bits_float(module.weight_scale_2).reshape(())],
        scale_recipe_b=recipe,
        swizzle_a=swizzle, swizzle_b=swizzle,
        output_dtype=x.dtype)
    return output[:rows, :module.out_features]


_HARDWARE_MATMULS = {
    'int8-mlx': _int8_mlx,
    'int8-pack': _int8_pack,
    'fp8-mlx': _fp8_mlx,
    'nvfp4-mlx': _nvfp4_mlx,
    'int8-aiter': _int8_aiter,
    'int8-mm': _int8_mm,
    'fp8-aiter': _fp8_aiter,
    'fp8-scaled-mm': _fp8_scaled_mm,
    'nvfp4': _nvfp4_scaled_mm,
}

# Relative error allowed against the dequantized matmul. Weight only kernels
# are exact up to rounding; W8A8 and W4A4 add activation quantization error.
_HARDWARE_TOLERANCE = {
    'int8-mlx': 0.02, 'int8-pack': 0.02, 'fp8-mlx': 0.02, 'nvfp4-mlx': 0.02,
    'int8-aiter': 0.05, 'int8-mm': 0.05,
    'fp8-aiter': 0.08, 'fp8-scaled-mm': 0.08,
    'nvfp4': 0.25,
}


def _relative_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    reference = reference.float()
    return ((output.float() - reference).norm() / reference.norm().clamp(min=1e-12)).item()


def _synchronize(device: torch.device) -> None:
    backend = getattr(torch, device.type, None)
    if hasattr(backend, 'synchronize'):
        backend.synchronize()


def _seconds(function, device: torch.device) -> float:
    """Best of two timings."""
    best = math.inf
    for _ in range(2):
        _synchronize(device)
        start = time.perf_counter()
        function()
        _synchronize(device)
        best = min(best, time.perf_counter() - start)
    return best


def _shape_key(candidate: str, module, x: torch.Tensor) -> tuple:
    """Speed depends on the layer and the number of rows: bucket rows by powers of two."""
    rows = 1 << max(0, int(x.shape[0]) - 1).bit_length()
    return candidate, str(x.device), rows, module.out_features, module.in_features


def _verify(candidate: str, module, x: torch.Tensor, output: torch.Tensor, software) -> bool:
    """
    Accept ``candidate`` for this layer shape if it matches the dequantized
    matmul and is faster. A mismatch turns the candidate off on the device.
    """
    reference = software(x)
    error = _relative_error(output, reference)
    if not math.isfinite(error) or error > _HARDWARE_TOLERANCE[candidate]:
        _hardware_state[(candidate, str(x.device))] = False
        _messages.warning(
            f'{candidate} hardware matmul on "{x.device}" differs from the dequantized '
            f'result (relative error {error:.3f}), not using it.')
        return False
    # Both paths have run once, so neither timing includes first call setup.
    kernel = _HARDWARE_MATMULS[candidate]
    hardware_seconds = _seconds(lambda: kernel(module, x), x.device)
    software_seconds = _seconds(lambda: software(x), x.device)
    faster = hardware_seconds < software_seconds
    _shape_state[_shape_key(candidate, module, x)] = faster
    _messages.debug_log(
        f'{candidate} hardware matmul on "{x.device}" for {tuple(x.shape)} x '
        f'{(module.out_features, module.in_features)}: '
        f'{hardware_seconds * 1000:.2f} ms vs {software_seconds * 1000:.2f} ms dequantized, '
        f'relative error {error:.4f}, {"using it" if faster else "not faster"}.')
    return faster


def _hardware_matmul(module: 'ComfyQuantLinear', x: torch.Tensor, software) -> torch.Tensor | None:
    """``x @ W.T`` on low precision hardware, or ``None`` to use ``software``."""
    if x.device.type not in _HARDWARE_DEVICE_TYPES or x.shape[0] == 0:
        return None
    if os.environ.get(SOFTWARE_MATMUL_ENVVAR) == '1':
        return None
    for candidate in _candidates(module, x.device):
        if _hardware_state.get((candidate, str(x.device))) is False:
            continue
        shape_state = _shape_state.get(_shape_key(candidate, module, x))
        if shape_state is False:
            continue
        try:
            output = _HARDWARE_MATMULS[candidate](module, x)
        except Exception as e:
            _hardware_state[(candidate, str(x.device))] = False
            _messages.debug_log(f'{candidate} hardware matmul failed on "{x.device}": {e}')
            continue
        if output is None:
            # This layer shape does not suit the kernel.
            continue
        if shape_state or _verify(candidate, module, x, output, software):
            return output
    return None


class ComfyQuantLinear(torch.nn.Linear):
    """
    A linear layer whose weight stays in ComfyUI's packed storage.

    Forward uses a hardware low precision matmul where one is available (see
    :py:func:`_hardware_matmul`), otherwise the weight is decoded to the
    input's dtype. ConvRot is
    undone on the input side: ``x @ (W_rot @ H).T == (x @ H) @ W_rot.T``
    for the symmetric Hadamard matrix ``H``, which is cheaper than
    unrotating the weight.
    """

    def __init__(self, in_features: int, out_features: int, layer: _QuantLayer,
                 bias: torch.nn.Parameter | None = None):
        torch.nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.quant_format = layer.format
        self.convrot_group_size = layer.convrot_group_size
        self.full_precision_mm = layer.full_precision

        def placeholder(value):
            if value is None:
                return None
            return torch.nn.Parameter(torch.empty_like(value, device='meta'), requires_grad=False)

        self.register_parameter('weight', placeholder(layer.weight))
        self.register_parameter('weight_scale', placeholder(layer.scale))
        self.register_parameter('weight_scale_2', placeholder(layer.scale_2))
        self.register_parameter('pre_quant_scale', placeholder(layer.pre_quant_scale))
        self.register_parameter('bias', bias)

    def extra_repr(self) -> str:
        rotation = f', convrot={self.convrot_group_size}' if self.convrot_group_size else ''
        return f'{super().extra_repr()}, format={self.quant_format}{rotation}'

    def _layer(self) -> _QuantLayer:
        return _QuantLayer('', self.quant_format, self.weight, self.weight_scale,
                           self.weight_scale_2, self.pre_quant_scale, self.convrot_group_size,
                           self.full_precision_mm)

    def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        """The dense ``(out_features, in_features)`` weight."""
        return self._layer().dequantize(dtype, (self.out_features, self.in_features))

    def _dequantized_matmul(self, input: torch.Tensor) -> torch.Tensor:
        weight = _decode(self.quant_format, self.weight, self.weight_scale,
                         self.weight_scale_2, input.dtype)
        if weight.shape != (self.out_features, self.in_features):
            # NVFP4 padding.
            weight = weight[:self.out_features, :self.in_features]
        return torch.nn.functional.linear(input, weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dtype = input.dtype
        if self.pre_quant_scale is not None:
            input = input * _bits_float(self.pre_quant_scale).to(dtype)
        if self.convrot_group_size:
            input = _rotate(input, self.convrot_group_size)
        shape = input.shape
        input = input.reshape(-1, shape[-1])
        output = _hardware_matmul(self, input, self._dequantized_matmul)
        if output is None:
            output = self._dequantized_matmul(input)
        if self.bias is not None:
            output = output + self.bias.to(dtype)
        return output.reshape(*shape[:-1], self.out_features)


# Diffusers looks quantizers up by quant_method. Like Diffusers' own
# QuantizationMethod, the str mixin makes the member hash as its value.
class _QuantMethod(str, enum.Enum):
    COMFY_QUANT = 'dgenerate_comfy_quant'


def _quantizer_classes():
    from diffusers.quantizers.base import DiffusersQuantizer
    from diffusers.quantizers.quantization_config import QuantizationConfigMixin

    class _Config(QuantizationConfigMixin):
        def __init__(self, checkpoint: 'ComfyQuantCheckpoint', dtype: torch.dtype):
            self.quant_method = _QuantMethod.COMFY_QUANT
            self.checkpoint = checkpoint
            self.dtype = dtype

        def to_dict(self):
            return {'quant_method': self.quant_method.value}

    class _Quantizer(DiffusersQuantizer):
        """
        Swap linears for :py:class:`ComfyQuantLinear` once the checkpoint is
        converted. Their tensors are all integer, so they load as is.
        """
        requires_calibration = False

        def validate_environment(self, *args, **kwargs):
            pass

        def _process_model_before_weight_loading(self, model, state_dict=None, **kwargs):
            config = self.quantization_config
            # Models cast activations to the dtype of these modules' parameters
            # (Wan casts the timestep to time_embedder's), which packed integer
            # storage would turn into a cast to uint8. Keep them dense.
            keep_dense = set(getattr(model, '_keep_in_fp32_modules', None) or [])
            for key, value in list(state_dict.items()):
                segments = config.checkpoint.segments_for(key, value)
                if segments is None:
                    continue
                module_name = key[:-len('.weight')] if key.endswith('.weight') else None
                module = _get_submodule(model, module_name) if module_name else None
                layer = None
                if (isinstance(module, torch.nn.Linear)
                        and keep_dense.isdisjoint(module_name.split('.'))):
                    layer = _assemble_layer(module_name, segments, value)
                if layer is None or not layer.fits(module.out_features, module.in_features):
                    _messages.debug_log(
                        f'Dequantizing ComfyUI quantized tensor "{key}", it does not load '
                        f'into a linear as one packed layer.')
                    target = getattr(module, 'weight', None)
                    shape = tuple(target.shape) if target is not None and target.ndim == 2 else None
                    state_dict[key] = _dequantize_segments(segments, config.dtype, shape)
                    continue
                replacement = ComfyQuantLinear(
                    module.in_features, module.out_features, layer, bias=module.bias)
                _set_submodule(model, module_name, replacement)
                state_dict[key] = layer.weight
                for name, tensor in (('weight_scale', layer.scale),
                                     ('weight_scale_2', layer.scale_2),
                                     ('pre_quant_scale', layer.pre_quant_scale)):
                    if tensor is not None:
                        state_dict[f'{module_name}.{name}'] = tensor
            return model

        def _process_model_after_weight_loading(self, model, **kwargs):
            # The checkpoint tensors now live in the model.
            self.quantization_config.checkpoint = None
            return model

        def _dequantize(self, model):
            for name, module in list(model.named_modules()):
                if isinstance(module, ComfyQuantLinear):
                    dense = torch.nn.Linear(
                        module.in_features, module.out_features, bias=module.bias is not None,
                        device=module.weight.device, dtype=module.bias.dtype
                        if module.bias is not None else torch.bfloat16)
                    with torch.no_grad():
                        dense.weight.copy_(module.dequantize(dense.weight.dtype))
                        if module.bias is not None:
                            dense.bias.copy_(module.bias)
                    _set_submodule(model, name, dense)
            return model

        @property
        def is_serializable(self):
            return False

        @property
        def is_trainable(self):
            return False

    return _Config, _Quantizer


def _get_submodule(model: torch.nn.Module, name: str) -> torch.nn.Module | None:
    try:
        return model.get_submodule(name)
    except AttributeError:
        return None


def _set_submodule(model: torch.nn.Module, name: str, module: torch.nn.Module) -> None:
    parent_name, _, child = name.rpartition('.')
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, module)


@functools.cache
def _register():
    import diffusers.quantizers.auto as auto

    config_class, quantizer_class = _quantizer_classes()
    auto.AUTO_QUANTIZER_MAPPING[_QuantMethod.COMFY_QUANT.value] = quantizer_class
    auto.AUTO_QUANTIZATION_CONFIG_MAPPING[_QuantMethod.COMFY_QUANT.value] = config_class
    return config_class


def _quantization_config(checkpoint: ComfyQuantCheckpoint, dtype: torch.dtype):
    """Diffusers ``quantization_config`` that keeps the layers of ``checkpoint`` packed."""
    return _register()(checkpoint, dtype)


def _resolve_module(model: torch.nn.Module, name: str) -> tuple[str, torch.nn.Module | None]:
    """A submodule by checkpoint name, also without the ``text_model.`` prefix flattened CLIP drops."""
    module = _get_submodule(model, name)
    if module is None and name.startswith('text_model.'):
        name = name[len('text_model.'):]
        module = _get_submodule(model, name)
    return name, module


def prepare_text_encoder(model: torch.nn.Module, path: str, dtype: torch.dtype | None,
                         pack: bool = True, rename=None) -> dict:
    """
    Put the ComfyUI quantized linears of a text encoder file into ``model``.

    Linears become :py:class:`ComfyQuantLinear` holding the file's packed
    tensors, except modules the model keeps in float32 (T5 ``wo``: its forward
    casts activations to the weight dtype) and any linear when ``pack`` is
    false, for example under another quantizer. Those are dequantized.

    :param rename: Optional checkpoint name -> model name mapping.
    :return: the state dict of everything not packed, dense, by checkpoint name,
        for the caller's regular load
    """
    loaded = load_checkpoint(path, dtype)
    keep_fp32 = set(getattr(model, '_keep_in_fp32_modules', None) or [])
    state = loaded.checkpoint
    packed = 0
    for key, layer in loaded.sources.items():
        target = rename(key) if rename else key
        module_name, module = _resolve_module(model, target[:-len('.weight')])
        if (pack and type(module) is torch.nn.Linear
                and module_name.rsplit('.', 1)[-1] not in keep_fp32
                and layer.fits(module.out_features, module.in_features)):
            replacement = ComfyQuantLinear(
                module.in_features, module.out_features, layer, bias=module.bias)
            for name, tensor in (('weight', layer.weight), ('weight_scale', layer.scale),
                                 ('weight_scale_2', layer.scale_2),
                                 ('pre_quant_scale', layer.pre_quant_scale)):
                if tensor is not None:
                    setattr(replacement, name, torch.nn.Parameter(tensor, requires_grad=False))
            _set_submodule(model, module_name, replacement)
            del state[key]
            packed += 1
            continue
        target = getattr(module, 'weight', None)
        shape = tuple(target.shape) if target is not None and target.ndim == 2 else None
        precision = torch.float32 if module_name.rsplit('.', 1)[-1] in keep_fp32 else loaded.dtype
        state[key] = layer.dequantize(precision, shape)
    _messages.debug_log(
        f'ComfyUI quantized text encoder "{path}": {packed} packed linears, '
        f'{len(loaded.sources) - packed} dequantized.')
    return state


@contextlib.contextmanager
def accelerate_checkpoint(path: str, state: dict):
    """Within this context, accelerate reads ``state`` instead of the file at ``path``."""
    import accelerate.utils.modeling as modeling

    original = modeling.load_state_dict

    def load_state_dict(checkpoint_file, device_map=None):
        if os.path.abspath(str(checkpoint_file)) == os.path.abspath(path):
            return dict(state)
        return original(checkpoint_file, device_map=device_map)

    modeling.load_state_dict = load_state_dict
    try:
        yield
    finally:
        modeling.load_state_dict = original


def module_has_comfy_quant(module) -> bool:
    """True when any submodule of ``module`` keeps ComfyUI packed weights."""
    if not isinstance(module, torch.nn.Module):
        return False
    return any(isinstance(part, ComfyQuantLinear) for part in module.modules())
