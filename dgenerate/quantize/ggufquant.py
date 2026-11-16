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
Quantize tensors to the GGUF types dgenerate loads.

``gguf.quantize`` writes Q4_0, Q5_0, Q8_0, and the plain floats. The K-quants
and IQ4 types in that package only dequantize, so those byte layouts are
written here. Each packer is the inverse of ``gguf``'s own dequantizer.
"""

import numpy


def quantize(array: numpy.ndarray, qtype) -> numpy.ndarray:
    """Packed row-major bytes for ``qtype``, or the array ``gguf.quantize`` returns."""
    import gguf

    array = numpy.ascontiguousarray(array, dtype=numpy.float32)
    try:
        return gguf.quantize(array, qtype)
    except NotImplementedError:
        pass
    pack = _PACKERS.get(qtype.name)
    if pack is None:
        raise NotImplementedError(f'Quantization for {qtype.name} is not implemented.')
    block, type_size = gguf.GGML_QUANT_SIZES[qtype]
    if array.shape[-1] % block:
        raise ValueError(
            f'Cannot quantize shape {array.shape} to {qtype.name}: '
            f'the last dimension is not a multiple of {block}.')
    rows = array.reshape(-1, array.shape[-1])
    blocks = rows.reshape(-1, block)
    packed = numpy.ascontiguousarray(pack(blocks))
    if packed.shape != (blocks.shape[0], type_size):
        raise RuntimeError(
            f'{qtype.name} packed {packed.shape}, expected {(blocks.shape[0], type_size)}.')
    out_last = array.shape[-1] // block * type_size
    return numpy.ascontiguousarray(packed.reshape(*array.shape[:-1], out_last))


def _f16_bytes(values: numpy.ndarray) -> numpy.ndarray:
    """``(n, 1)`` float values as ``(n, 2)`` little-endian float16 bytes."""
    stored = numpy.ascontiguousarray(values.reshape(-1, 1).astype(numpy.float16))
    return stored.view(numpy.uint8).reshape(values.shape[0], 2)


def _rint_div(num: numpy.ndarray, den: numpy.ndarray) -> numpy.ndarray:
    out = numpy.zeros(numpy.broadcast(num, den).shape, dtype=numpy.float32)
    numpy.divide(num, den, out=out, where=numpy.abs(den) > 1e-12)
    return numpy.rint(out)


def _pack_q4k_scales(scale: numpy.ndarray, minimum: numpy.ndarray) -> numpy.ndarray:
    """8 scales and 8 mins, each 0..63, into the 12-byte Q4_K / Q5_K block."""
    scale = scale.astype(numpy.uint8)
    minimum = minimum.astype(numpy.uint8)
    low_scale, high_scale = scale[:, :4], scale[:, 4:]
    low_min, high_min = minimum[:, :4], minimum[:, 4:]
    first = low_scale | ((high_scale & numpy.uint8(0x30)) << numpy.uint8(2))
    second = low_min | ((high_min & numpy.uint8(0x30)) << numpy.uint8(2))
    third = (high_scale & numpy.uint8(0x0F)) | ((high_min & numpy.uint8(0x0F)) << numpy.uint8(4))
    return numpy.concatenate([first, second, third], axis=1)


def _super_codes(per_group: numpy.ndarray, levels: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    ``per_group`` of shape ``(blocks, groups)`` as ``super * code``.

    ``code`` is an integer in ``0 .. levels``. ``super`` is one float per block
    and may be negative, so a signed group value still fits in an unsigned code.
    """
    magnitude = numpy.max(numpy.abs(per_group), axis=1, keepdims=True)
    sign = numpy.sign(numpy.sum(per_group, axis=1, keepdims=True)).astype(numpy.float32)
    sign = numpy.where(sign == 0, 1, sign)
    super_scale = sign * magnitude / levels
    codes = numpy.clip(_rint_div(per_group, super_scale), 0, levels)
    return super_scale, codes


def _pack_nibbles_low_then_high(codes: numpy.ndarray) -> numpy.ndarray:
    """
    Codes shaped ``(blocks, pairs, 2, low_width)`` into bytes.

    The dequantizer reads every low nibble of a byte group, then every high nibble.
    """
    low = codes[:, :, 0, :].astype(numpy.uint8)
    high = codes[:, :, 1, :].astype(numpy.uint8)
    return (low | (high << numpy.uint8(4))).reshape(codes.shape[0], -1)


def _q4_k(blocks: numpy.ndarray) -> numpy.ndarray:
    count = blocks.shape[0]
    groups = blocks.reshape(count, 8, 32)
    lowest = groups.min(axis=-1)
    highest = groups.max(axis=-1)
    scale, scale_code = _super_codes((highest - lowest) / 15, 63)
    minimum, min_code = _super_codes(-lowest, 63)
    scale_code = scale_code.astype(numpy.float32)
    min_code = min_code.astype(numpy.float32)
    decoded_scale = scale * scale_code
    decoded_min = minimum * min_code
    quant = numpy.clip(
        _rint_div(groups + decoded_min[:, :, None], decoded_scale[:, :, None]), 0, 15)
    codes = quant.reshape(count, 4, 2, 32).astype(numpy.uint8)
    return numpy.concatenate([
        _f16_bytes(scale),
        _f16_bytes(minimum),
        _pack_q4k_scales(scale_code, min_code),
        _pack_nibbles_low_then_high(codes),
    ], axis=1)


def _q5_k(blocks: numpy.ndarray) -> numpy.ndarray:
    count = blocks.shape[0]
    groups = blocks.reshape(count, 8, 32)
    lowest = groups.min(axis=-1)
    highest = groups.max(axis=-1)
    scale, scale_code = _super_codes((highest - lowest) / 31, 63)
    minimum, min_code = _super_codes(-lowest, 63)
    decoded_scale = scale * scale_code
    decoded_min = minimum * min_code
    quant = numpy.clip(
        _rint_div(groups + decoded_min[:, :, None], decoded_scale[:, :, None]), 0, 31)
    quant = quant.astype(numpy.uint8)
    low = (quant & numpy.uint8(0x0F)).reshape(count, 4, 2, 32)
    high = ((quant >> numpy.uint8(4)) & numpy.uint8(1)).reshape(count, 8, 32)
    high_bytes = numpy.zeros((count, 32), dtype=numpy.uint8)
    for shift in range(8):
        high_bytes |= high[:, shift, :] << numpy.uint8(shift)
    return numpy.concatenate([
        _f16_bytes(scale),
        _f16_bytes(minimum),
        _pack_q4k_scales(scale_code, min_code),
        high_bytes,
        _pack_nibbles_low_then_high(low),
    ], axis=1)


def _q6_k(blocks: numpy.ndarray) -> numpy.ndarray:
    count = blocks.shape[0]
    groups = blocks.reshape(count, 16, 16)
    peak = numpy.max(numpy.abs(groups), axis=-1)
    super_scale = numpy.max(peak, axis=1, keepdims=True) / (127 * 31)
    scales = numpy.clip(_rint_div(peak, super_scale * 31), -128, 127)
    decoded = super_scale * scales
    quant = numpy.clip(_rint_div(groups, decoded[:, :, None]), -32, 31)
    codes = (quant + 32).astype(numpy.uint8).reshape(count, 8, 32)
    low = (codes & numpy.uint8(0x0F)).reshape(count, 2, 2, 64)
    high = ((codes >> numpy.uint8(4)) & numpy.uint8(0x03)).reshape(count, 2, 4, 32)
    high_bytes = numpy.zeros((count, 2, 32), dtype=numpy.uint8)
    for shift in range(4):
        high_bytes |= high[:, :, shift, :] << numpy.uint8(2 * shift)
    return numpy.concatenate([
        _pack_nibbles_low_then_high(low),
        high_bytes.reshape(count, 64),
        scales.astype(numpy.int8).view(numpy.uint8),
        _f16_bytes(super_scale),
    ], axis=1)


def _q2_k(blocks: numpy.ndarray) -> numpy.ndarray:
    count = blocks.shape[0]
    groups = blocks.reshape(count, 16, 16)
    lowest = groups.min(axis=-1)
    highest = groups.max(axis=-1)
    scale, scale_code = _super_codes((highest - lowest) / 3, 15)
    minimum, min_code = _super_codes(-lowest, 15)
    decoded_scale = scale * scale_code
    decoded_min = minimum * min_code
    quant = numpy.clip(
        _rint_div(groups + decoded_min[:, :, None], decoded_scale[:, :, None]), 0, 3)
    expanded = quant.astype(numpy.uint8).reshape(count, 2, 4, 32)
    packed = numpy.zeros((count, 2, 32), dtype=numpy.uint8)
    for shift in range(4):
        packed |= expanded[:, :, shift, :] << numpy.uint8(2 * shift)
    scales = (scale_code.astype(numpy.uint8) & numpy.uint8(0x0F)) | (
        (min_code.astype(numpy.uint8) & numpy.uint8(0x0F)) << numpy.uint8(4))
    return numpy.concatenate([
        scales.astype(numpy.uint8),
        packed.reshape(count, 64),
        _f16_bytes(scale),
        _f16_bytes(minimum),
    ], axis=1)


def _q3_k(blocks: numpy.ndarray) -> numpy.ndarray:
    count = blocks.shape[0]
    groups = blocks.reshape(count, 16, 16)
    peak = numpy.max(numpy.abs(groups), axis=-1)
    # Stored scale is (code - 32), and the 2-bit magnitude is subtracted by 0 or 4.
    super_scale = numpy.max(peak, axis=1, keepdims=True) / (32 * 4)
    scales = numpy.clip(_rint_div(peak, super_scale * 4), -32, 31)
    decoded = super_scale * scales
    quant = numpy.clip(_rint_div(groups, decoded[:, :, None]), -4, 3)
    # q = ql - 4 * (stored_bit ^ 1). ql is 0..3.
    signed = quant.astype(numpy.int16)
    high = (signed < 0).astype(numpy.uint8)
    low = numpy.where(signed < 0, signed + 4, signed).astype(numpy.uint8)
    stored_bit = high ^ numpy.uint8(1)
    low_bytes = _pack_shifted(low.reshape(count, 2, 4, 32), (0, 2, 4, 6)).reshape(count, 64)
    high_bytes = _pack_shifted(stored_bit.reshape(count, 1, 8, 32), tuple(range(8))).reshape(count, 32)
    scale_codes = numpy.clip(scales + 32, 0, 63).astype(numpy.uint8)
    return numpy.concatenate([
        high_bytes,
        low_bytes,
        _pack_q3_scales(scale_codes),
        _f16_bytes(super_scale),
    ], axis=1)


def _pack_shifted(bits: numpy.ndarray, shifts: tuple[int, ...]) -> numpy.ndarray:
    """``(blocks, chunks, len(shifts), width)`` bit fields into bytes."""
    packed = numpy.zeros(bits.shape[:2] + bits.shape[3:], dtype=numpy.uint8)
    for index, shift in enumerate(shifts):
        packed |= bits[:, :, index, :].astype(numpy.uint8) << numpy.uint8(shift)
    return packed


def _pack_q3_scales(codes: numpy.ndarray) -> numpy.ndarray:
    """16 codes of 0..63 into the 12-byte Q3_K scale block."""
    low = codes & numpy.uint8(0x0F)
    high = (codes >> numpy.uint8(4)) & numpy.uint8(0x03)
    low_bytes = low[:, :8] | (low[:, 8:] << numpy.uint8(4))
    high_bytes = numpy.zeros((codes.shape[0], 4), dtype=numpy.uint8)
    for shift in range(4):
        high_bytes |= high[:, shift * 4:(shift + 1) * 4] << numpy.uint8(2 * shift)
    return numpy.concatenate([low_bytes, high_bytes], axis=1)


def _nearest_code(values: numpy.ndarray, levels: numpy.ndarray) -> numpy.ndarray:
    """Index of the closest entry in ``levels`` for every value. ``levels`` is 1D."""
    table = levels.reshape((1,) * values.ndim + (-1,))
    return numpy.abs(values[..., None] - table).argmin(axis=-1).astype(numpy.uint8)


def _iq4_nl(blocks: numpy.ndarray) -> numpy.ndarray:
    import gguf
    levels = numpy.array(gguf.quants.IQ4_NL.kvalues, dtype=numpy.float32)
    peak = numpy.max(numpy.abs(blocks), axis=1, keepdims=True) / numpy.max(numpy.abs(levels))
    codes = _nearest_code(blocks / numpy.maximum(peak, 1e-12), levels)
    pairs = codes.reshape(codes.shape[0], 2, 16)
    # Dequantize reads the low nibble of all 16 bytes, then the high nibble.
    packed = pairs[:, 0, :] | (pairs[:, 1, :] << numpy.uint8(4))
    return numpy.concatenate([_f16_bytes(peak), packed], axis=1)


def _iq4_xs(blocks: numpy.ndarray) -> numpy.ndarray:
    import gguf
    levels = numpy.array(gguf.quants.IQ4_NL.kvalues, dtype=numpy.float32)
    count = blocks.shape[0]
    groups = blocks.reshape(count, 8, 32)
    peak = numpy.max(numpy.abs(groups), axis=-1) / numpy.max(numpy.abs(levels))
    super_scale = numpy.max(numpy.abs(peak), axis=1, keepdims=True) / 32
    sign = numpy.sign(numpy.sum(peak, axis=1, keepdims=True))
    sign = numpy.where(sign == 0, 1, sign).astype(numpy.float32)
    super_scale = sign * super_scale
    scales = numpy.clip(_rint_div(peak, super_scale), -32, 31)
    decoded = super_scale * scales
    codes = _nearest_code(
        groups / numpy.maximum(decoded[:, :, None], 1e-12), levels).astype(numpy.uint8)
    arranged = codes.reshape(count, 8, 2, 16)
    qs = (arranged[:, :, 0, :] | (arranged[:, :, 1, :] << numpy.uint8(4))).reshape(count, 128)
    stored = numpy.clip(scales + 32, 0, 63).astype(numpy.uint8)
    low = stored & numpy.uint8(0x0F)
    high = (stored >> numpy.uint8(4)) & numpy.uint8(0x03)
    # 8 low nibbles in 4 bytes: low nibble then high nibble of the next scale.
    scale_low = low[:, 0::2] | (low[:, 1::2] << numpy.uint8(4))
    # 8 high 2-bit fields packed into one uint16, low field in the low bits.
    scale_high = numpy.zeros((count,), dtype=numpy.uint16)
    for index in range(8):
        scale_high |= high[:, index].astype(numpy.uint16) << numpy.uint16(2 * index)
    return numpy.concatenate([
        _f16_bytes(super_scale),
        numpy.ascontiguousarray(scale_high).view(numpy.uint8).reshape(count, 2),
        scale_low.astype(numpy.uint8),
        qs,
    ], axis=1)


_PACKERS = {
    'Q2_K': _q2_k,
    'Q3_K': _q3_k,
    'Q4_K': _q4_k,
    'Q5_K': _q5_k,
    'Q6_K': _q6_k,
    'IQ4_NL': _iq4_nl,
    'IQ4_XS': _iq4_xs,
}
