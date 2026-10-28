import json
import math
import os
import tempfile
import unittest

import safetensors.torch
import torch

import dgenerate.pipelinewrapper.safetensorscompat as _safetensorscompat


def _marker(config: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(config).encode('utf-8')), dtype=torch.uint8)


def _to_blocked(scales: torch.Tensor) -> torch.Tensor:
    """ComfyUI / cuBLAS 128x4 tiled block scale layout (comfy_kitchen.float_utils.to_blocked)."""
    rows, cols = scales.shape
    row_tiles, col_tiles = -(-rows // 128), -(-cols // 4)
    padded = torch.zeros(row_tiles * 128, col_tiles * 4, dtype=scales.dtype)
    padded[:rows, :cols] = scales
    blocks = padded.view(row_tiles, 128, col_tiles, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(row_tiles * 128, col_tiles * 4)


def _quantize_fp8(weight):
    scale = weight.abs().amax() / 448
    return (weight / scale).to(torch.float8_e4m3fn), scale.reshape(())


def _quantize_int8_convrot(weight, group):
    hadamard = _safetensorscompat._hadamard(group, 'cpu', torch.float32)
    rotated = (weight.view(weight.shape[0], -1, group) @ hadamard.T).view(weight.shape)
    scale = rotated.abs().amax(dim=-1, keepdim=True) / 127
    return torch.round(rotated / scale).clamp(-128, 127).to(torch.int8), scale


def _quantize_mxfp8(weight):
    """FP8 E4M3 with E8M0 block scales, padded to 32 and swizzled like ComfyUI."""
    rows, cols = weight.shape
    padded_rows = _safetensorscompat._round_up(rows, 32)
    padded_cols = _safetensorscompat._round_up(cols, 32)
    padded = weight.new_zeros(padded_rows, padded_cols)
    padded[:rows, :cols] = weight
    blocks = padded.view(padded_rows, padded_cols // 32, 32)
    max_abs = blocks.abs().amax(-1)
    need = (max_abs / 448).clamp(min=2.0 ** -127)
    exponents = (torch.ceil(torch.log2(need)).to(torch.int32) + 127).clamp(0, 254).to(torch.uint8)
    zero = max_abs == 0
    scale = (exponents.to(torch.int32) << 23).view(torch.float32)
    scale = torch.where(zero, torch.ones_like(scale), scale)
    quant = (blocks / scale.unsqueeze(-1)).reshape(padded_rows, padded_cols).clamp(-448, 448)
    exponents = torch.where(zero, torch.zeros_like(exponents), exponents)
    return quant.to(torch.float8_e4m3fn), _to_blocked(exponents).contiguous()


def _pack_signed_int4(values):
    codes = values.to(torch.int32)
    return ((codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)).to(torch.int8)


def _quantize_int4_convrot(weight, group, grouped=False):
    hadamard = _safetensorscompat._hadamard(group, 'cpu', torch.float32)
    rotated = (weight.view(weight.shape[0], -1, group) @ hadamard.T).reshape_as(weight)
    if grouped:
        blocks = rotated.view(rotated.shape[0], -1, 64)
        scale = blocks.abs().amax(-1).clamp(min=1e-10) / 7
        quant = torch.round(blocks / scale.unsqueeze(-1)).clamp(-7, 7).reshape_as(rotated)
        scale = scale.contiguous()
    else:
        scale = rotated.abs().amax(-1, keepdim=True).clamp(min=1e-10) / 7
        quant = torch.round(rotated / scale).clamp(-7, 7)
    return _pack_signed_int4(quant), scale.to(torch.float32)


def _quantize_w4a8(weight, group=16, convrot=256, fp8_scales=False):
    """Symmetric codebook W4A8, the layout comfy-kitchen writes for ``asym_w4a8_int8``."""
    hadamard = _safetensorscompat._hadamard(convrot, 'cpu', torch.float32)
    rotated = (weight.view(weight.shape[0], -1, convrot) @ hadamard.T).reshape_as(weight)
    rows, inputs = rotated.shape
    groups = inputs // group
    grouped = rotated.view(rows, groups, group)
    codebook = torch.tensor(_safetensorscompat._W4A8_CODEBOOK)
    group_scale = grouped.abs().amax(-1, keepdim=True).clamp(min=1e-8)

    def assign(normalized):
        return (normalized.unsqueeze(-1) - codebook).abs().argmin(-1)

    codes = assign(grouped / group_scale)
    for _ in range(2):
        levels = codebook[codes]
        group_scale = ((grouped * levels).sum(-1, keepdim=True)
                       / levels.square().sum(-1, keepdim=True).clamp(min=1e-8)).clamp(min=1e-8)
        codes = assign(grouped / group_scale)
    channel = (codebook[codes] * group_scale).abs().amax(dim=(1, 2)).clamp(min=1e-8) / 127
    rel = (group_scale.squeeze(-1) / channel.unsqueeze(1)).float()
    if fp8_scales:
        rel = rel.to(torch.float8_e4m3fn).float()
    grid = (codebook.view(1, 1, -1) * rel.unsqueeze(-1)).round().clamp(-127, 127)
    target = grouped / channel.view(rows, 1, 1)
    codes = (target.unsqueeze(-1) - grid.unsqueeze(2)).abs().argmin(-1).reshape(rows, inputs)
    packed = ((codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)).to(torch.int8)
    stored_rel = rel.to(torch.float8_e4m3fn) if fp8_scales else rel
    return packed, stored_rel.contiguous(), channel.contiguous(), codebook


def _pack_int6(codes):
    """Unsigned 6-bit codes ``(rows, inputs)`` -> int8 ``(rows, inputs * 3 / 4)``."""
    low = ((codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)).to(torch.int32)
    high = (codes[:, 0::4] >> 4) & 0x3
    for shift in range(1, 4):
        high = high | (((codes[:, shift::4] >> 4) & 0x3) << (2 * shift))
    return torch.cat((low, high), dim=1).to(torch.int8).contiguous()


def _quantize_w6a8(weight, group=16, convrot=256):
    """Uniform 6-bit W6A8: levels -31..31 stored as ``level + 32``."""
    hadamard = _safetensorscompat._hadamard(convrot, 'cpu', torch.float32)
    rotated = (weight.view(weight.shape[0], -1, convrot) @ hadamard.T).reshape_as(weight)
    rows, inputs = rotated.shape
    groups = inputs // group
    grouped = rotated.view(rows, groups, group)
    group_scale = (grouped.abs().amax(-1, keepdim=True) / 31).clamp(min=1e-8)

    def assign(normalized):
        return normalized.round().clamp(-31, 31)

    signed = assign(grouped / group_scale)
    for _ in range(2):
        group_scale = ((grouped * signed).sum(-1, keepdim=True)
                       / signed.square().sum(-1, keepdim=True).clamp(min=1e-8)).clamp(min=1e-8)
        signed = assign(grouped / group_scale)
    channel = (signed * group_scale).abs().amax(dim=(1, 2)).clamp(min=1e-8) / 127
    rel = (group_scale.squeeze(-1) / channel.unsqueeze(1)).float()
    level_table = torch.arange(-31, 32)
    grid = (level_table.view(1, 1, -1) * rel.unsqueeze(-1)).round().clamp(-127, 127)
    target = grouped / channel.view(rows, 1, 1)
    index = (target.unsqueeze(-1) - grid.unsqueeze(2)).abs().argmin(-1)
    return _pack_int6((index + 1).reshape(rows, inputs)), rel.contiguous(), channel.contiguous()


def _quantize_nvfp4(weight):
    values = torch.tensor(_safetensorscompat._E2M1_VALUES[:8])
    rows, cols = weight.shape
    tensor_scale = weight.abs().amax() / (448 * 6)
    blocks = weight.view(rows, cols // 16, 16)
    block_scale = (blocks.abs().amax(-1) / 6 / tensor_scale).to(torch.float8_e4m3fn)
    total = block_scale.float() * tensor_scale
    scaled = blocks / total.clamp(min=1e-12).unsqueeze(-1)
    codes = (scaled.abs().unsqueeze(-1) - values).abs().argmin(-1)
    codes = (codes | ((scaled < 0).long() << 3)).view(rows, cols)
    packed = ((codes[:, 0::2] << 4) | codes[:, 1::2]).to(torch.uint8)
    return packed, _to_blocked(block_scale).contiguous(), tensor_scale.reshape(())


class TestSafetensorsCompat(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.weight = torch.randn(128, 256) * 0.02
        self.x = torch.randn(3, 256)
        self.reference = self.x @ self.weight.T

    def _write(self, tensors, metadata=None):
        handle, path = tempfile.mkstemp(suffix='.safetensors')
        os.close(handle)
        self.addCleanup(os.remove, path)
        safetensors.torch.save_file(tensors, path, metadata=metadata)
        return path

    def _kitchen_metadata(self, layers):
        return {
            '_quantization_metadata': json.dumps({
                'format_version': '1.0',
                'layers': layers,
            }),
            'converted_by': 'ComfyUI Kitchen NVFP4 Converter',
            'quant_format': 'INT8_CONVROT',
        }

    def _layer(self, tensors):
        path = self._write(tensors)
        self.assertTrue(_safetensorscompat.is_comfy_quantized(path))
        loaded = _safetensorscompat.load_checkpoint(path, torch.float32)
        self.assertEqual(set(loaded.checkpoint), {'layer.weight'})
        layer = next(iter(loaded.sources.values()))
        module = _safetensorscompat.ComfyQuantLinear(256, 128, layer)
        state = {'weight': layer.weight, 'weight_scale': layer.scale}
        for attr, key in (('scale_2', 'weight_scale_2'), ('codebook', 'weight_codebook'),
                          ('correction', 'weight_correction')):
            value = getattr(layer, attr)
            if value is not None:
                state[key] = value
        module.load_state_dict(state, assign=True)
        for _, param in module.named_parameters():
            self.assertFalse(param.is_floating_point())
        return module

    def _relative_error(self, output):
        return ((output - self.reference).norm() / self.reference.norm()).item()

    def test_float8(self):
        weight, scale = _quantize_fp8(self.weight)
        module = self._layer({
            'layer.weight': weight, 'layer.weight_scale': scale,
            'layer.input_scale': torch.tensor(1.0),
            'layer.comfy_quant': _marker({'format': 'float8_e4m3fn'})})
        self.assertLess(self._relative_error(module(self.x)), 0.05)

    def test_legacy_scaled_fp8(self):
        weight, scale = _quantize_fp8(self.weight)
        module = self._layer({
            'scaled_fp8': torch.empty(0, dtype=torch.float8_e4m3fn),
            'layer.weight': weight, 'layer.weight_scale': scale})
        self.assertLess(self._relative_error(module(self.x)), 0.05)

    def test_int8_convrot(self):
        weight, scale = _quantize_int8_convrot(self.weight, 64)
        module = self._layer({
            'layer.weight': weight, 'layer.weight_scale': scale,
            'layer.comfy_quant': _marker(
                {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 64})})
        self.assertLess(self._relative_error(module(self.x)), 0.02)
        dense = module.dequantize(torch.float32)
        self.assertLess(((dense - self.weight).norm() / self.weight.norm()).item(), 0.02)

    def test_kitchen_metadata_supplies_convrot_for_any_prefix(self):
        # Kitchen omits per-layer comfy_quant and strips model.diffusion_model.
        # The same metadata key must match a bare name, that prefix, and any
        # other single stored prefix.
        weight, scale = _quantize_int8_convrot(self.weight, 64)
        config = {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 64}
        cases = (
            ('layer', 'layer'),
            ('blocks.0.attn.q', 'model.diffusion_model.blocks.0.attn.q'),
            ('blocks.0.attn.q', 'transformer.blocks.0.attn.q'),
        )
        for meta_name, tensor_name in cases:
            with self.subTest(tensor_name=tensor_name):
                path = self._write(
                    {tensor_name + '.weight': weight,
                     tensor_name + '.weight_scale': scale},
                    metadata=self._kitchen_metadata({meta_name: config}))
                self.assertTrue(_safetensorscompat.is_comfy_quantized(path))
                loaded = _safetensorscompat.load_checkpoint(path, torch.float32)
                layer = loaded.sources[tensor_name + '.weight']
                self.assertEqual(layer.convrot_group_size, 64)
                dense = layer.dequantize(torch.float32)
                self.assertLess(((dense - self.weight).norm() / self.weight.norm()).item(), 0.02)

    def test_comfy_quant_marker_overrides_kitchen_metadata(self):
        weight, scale = _quantize_int8_convrot(self.weight, 64)
        path = self._write({
            'layer.weight': weight, 'layer.weight_scale': scale,
            'layer.comfy_quant': _marker(
                {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 64}),
        }, metadata=self._kitchen_metadata({
            'layer': {'format': 'int8_tensorwise'},
        }))
        layer = next(iter(_safetensorscompat.load_checkpoint(path, torch.float32).sources.values()))
        self.assertEqual(layer.convrot_group_size, 64)

    def test_nvfp4(self):
        weight, block_scale, tensor_scale = _quantize_nvfp4(self.weight)
        module = self._layer({
            'layer.weight': weight, 'layer.weight_scale': block_scale,
            'layer.weight_scale_2': tensor_scale,
            'layer.comfy_quant': _marker({'format': 'nvfp4'})})
        self.assertLess(self._relative_error(module(self.x)), 0.15)

    def test_legacy_scale_weight_naming(self):
        # SD3.5 / Wan / HunyuanVideo 1.5 "fp8_scaled": scale_weight and scale_input.
        weight, scale = _quantize_fp8(self.weight)
        module = self._layer({
            'scaled_fp8': torch.empty(0, dtype=torch.float8_e4m3fn),
            'layer.weight': weight, 'layer.scale_weight': scale.reshape(1),
            'layer.scale_input': torch.tensor(1.0)})
        self.assertLess(self._relative_error(module(self.x)), 0.05)

    def test_unmarked_nvfp4(self):
        # ModelOpt exports (LTX-2 FP4) have no comfy_quant marker.
        weight, block_scale, tensor_scale = _quantize_nvfp4(self.weight)
        module = self._layer({
            'layer.weight': weight, 'layer.weight_scale': block_scale,
            'layer.weight_scale_2': tensor_scale, 'layer.input_scale': torch.tensor(1.0)})
        self.assertEqual(module.quant_format, 'nvfp4')
        self.assertLess(self._relative_error(module(self.x)), 0.15)

    def test_nvfp4_activation_quantize_round_trip(self):
        packed, block_scale, tensor_scale = _safetensorscompat._quantize_nvfp4(self.x)
        decoded = _safetensorscompat._decode_nvfp4(packed, block_scale, tensor_scale, torch.float32)
        decoded = decoded[:self.x.shape[0]]
        self.assertLess(((decoded - self.x).norm() / self.x.norm()).item(), 0.15)

    def test_int8_activation_quantize_small_bfloat16(self):
        # Rows below the float16 normal range, like LTX audio to video attention inputs.
        x = (torch.randn(8, 256) * 1e-5).to(torch.bfloat16)
        quantized, x_scale = _safetensorscompat._quantize_int8_rows(x)
        decoded = quantized.float() * x_scale
        self.assertLess(((decoded - x.float()).norm() / x.float().norm()).item(), 0.02)

    def _module(self, fmt, device='cpu'):
        if fmt == 'int8':
            weight, scale = _quantize_int8_convrot(self.weight, 64)
            marker = {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 64}
        else:
            weight, scale = _quantize_fp8(self.weight)
            marker = {'format': 'float8_e4m3fn'}
        module = self._layer({'layer.weight': weight, 'layer.weight_scale': scale,
                              'layer.comfy_quant': _marker(marker)})
        for state in (_safetensorscompat._hardware_state, _safetensorscompat._shape_state):
            state.clear()
            self.addCleanup(state.clear)
        return module.to(device)

    def test_hardware_matmul_matches_dequantized(self):
        if not torch.cuda.is_available():
            self.skipTest('no CUDA device')
        device = torch.device('cuda')
        for fmt in ('int8', 'fp8'):
            module = self._module(fmt, device)
            for candidate in _safetensorscompat._candidates(module, device):
                with self.subTest(candidate=candidate):
                    x = torch.randn(40, 256, device=device, dtype=torch.bfloat16)
                    output = _safetensorscompat._HARDWARE_MATMULS[candidate](module, x)
                    reference = module._dequantized_matmul(x)
                    self.assertLess(_safetensorscompat._relative_error(output, reference),
                                    _safetensorscompat._HARDWARE_TOLERANCE[candidate])

    def test_candidates_fall_through_to_the_next(self):
        from unittest import mock

        module = self._module('int8')
        x = torch.randn(40, 256, dtype=torch.bfloat16)
        reference = module._dequantized_matmul(x)

        def broken(module, x):
            raise RuntimeError('no kernel')

        def wrong(module, x):
            return torch.zeros_like(reference)

        def good(module, x):
            return module._dequantized_matmul(x)

        with mock.patch.object(_safetensorscompat, '_HARDWARE_DEVICE_TYPES', ('cpu',)),                 mock.patch.dict(_safetensorscompat._HARDWARE_MATMULS,
                             {'a': broken, 'b': wrong, 'c': good}), \
                mock.patch.dict(_safetensorscompat._HARDWARE_TOLERANCE, {'a': 1, 'b': 0.05, 'c': 0.05}), \
                mock.patch.object(_safetensorscompat, '_candidates', return_value=('a', 'b', 'c')), \
                mock.patch.object(_safetensorscompat, '_seconds', side_effect=[1.0, 2.0]), \
                mock.patch.object(_safetensorscompat, '_synchronize'):
            output = _safetensorscompat._hardware_matmul(module, x, module._dequantized_matmul)
        self.assertTrue(torch.equal(output, reference))
        state = _safetensorscompat._hardware_state
        self.assertEqual((state[('a', 'cpu')], state[('b', 'cpu')]), (False, False))
        self.assertNotIn(('c', 'cpu'), state)
        self.assertTrue(_safetensorscompat._shape_state[('c', 'cpu', 64, 128, 256)])

    def test_slower_hardware_matmul_is_not_used(self):
        from unittest import mock

        module = self._module('int8')
        x = torch.randn(40, 256, dtype=torch.bfloat16)
        with mock.patch.object(_safetensorscompat, '_HARDWARE_DEVICE_TYPES', ('cpu',)),                 mock.patch.dict(_safetensorscompat._HARDWARE_MATMULS,
                                {'c': lambda module, x: module._dequantized_matmul(x)}), \
                mock.patch.dict(_safetensorscompat._HARDWARE_TOLERANCE, {'c': 0.05}), \
                mock.patch.object(_safetensorscompat, '_candidates', return_value=('c',)), \
                mock.patch.object(_safetensorscompat, '_seconds', side_effect=[2.0, 1.0]):
            output = _safetensorscompat._hardware_matmul(module, x, module._dequantized_matmul)
        self.assertIsNone(output)
        # Slower for this shape only; the kernel stays available for others.
        self.assertFalse(_safetensorscompat._shape_state[('c', 'cpu', 64, 128, 256)])
        self.assertNotIn(('c', 'cpu'), _safetensorscompat._hardware_state)

    def test_full_precision_marker_skips_hardware(self):
        weight, scale = _quantize_fp8(self.weight)
        module = self._layer({
            'layer.weight': weight, 'layer.weight_scale': scale,
            'layer.comfy_quant': _marker({'format': 'float8_e4m3fn', 'full_precision_matrix_mult': True})})
        self.assertEqual(_safetensorscompat._candidates(module, torch.device('cuda')), ())

    def test_mlx_int8_codes_are_exact(self):
        # MLX: w = scale * q + bias, q unpacked little endian from uint32.
        weight = torch.randint(-128, 128, (8, 64), dtype=torch.int8)
        codes = _safetensorscompat._mlx_int8_codes(weight)
        self.assertEqual(tuple(codes.shape), (8, 16))
        words = codes.view(torch.int32).long() & 0xFFFFFFFF
        unpacked = torch.stack([(words >> (8 * i)) & 0xFF for i in range(4)], dim=-1).view(8, 64)
        # With bias = -128 * scale, scale * q + bias == scale * (q - 128) == scale * weight.
        self.assertTrue(torch.equal(unpacked - 128, weight.long()))

    def test_mlx_nvfp4_layout_matches_comfy(self):
        # MLX fp_quantized.h: 4 bit codes low nibble first, e4m3 scales row major per 16.
        weight, block_scale, tensor_scale = _quantize_nvfp4(self.weight)
        module = self._layer({
            'layer.weight': weight, 'layer.weight_scale': block_scale,
            'layer.weight_scale_2': tensor_scale,
            'layer.comfy_quant': _marker({'format': 'nvfp4'})})
        codes = _safetensorscompat._mlx_nvfp4_codes(module.weight).view(torch.uint8)
        values = torch.tensor(_safetensorscompat._E2M1_VALUES)
        unpacked = torch.stack([values[(codes & 0x0F).long()], values[(codes >> 4).long()]], -1)
        scales = _safetensorscompat._unswizzle_block_scales(module.weight_scale, 128, 16)
        scales = scales.view(torch.float8_e4m3fn).float()
        mlx = (unpacked.view(128, 16, 16) * scales.unsqueeze(-1)).view(128, 256) * tensor_scale
        self.assertTrue(torch.allclose(mlx, module.dequantize(torch.float32), atol=1e-7))

    def test_mlx_mxfp8_unit_scales_match_comfy(self):
        # MLX mxfp8: e4m3 bytes, e8m0 scale 127 == 2^0; the file scale goes on the output.
        weight, scale = _quantize_fp8(self.weight)
        module = self._layer({'layer.weight': weight, 'layer.weight_scale': scale,
                              'layer.comfy_quant': _marker({'format': 'float8_e4m3fn'})})
        bytes_ = module.weight.view(torch.uint32).view(torch.uint8)
        e8m0_one = torch.tensor([127 << 23], dtype=torch.int32).view(torch.float32)
        mlx = bytes_.view(torch.float8_e4m3fn).float() * e8m0_one * scale
        self.assertTrue(torch.allclose(mlx, module.dequantize(torch.float32), atol=1e-7))
        self.assertEqual(e8m0_one.item(), 1.0)

    def test_float8_e4m3fnuz_reads_half_the_value(self):
        weight, _ = _quantize_fp8(self.weight)
        weight.view(torch.uint8)[0, 0] = 0x80
        path = self._write({'layer.weight': weight, 'layer.weight_scale': torch.tensor(1.0)})
        layer = next(iter(_safetensorscompat.load_checkpoint(path, torch.float32).sources.values()))
        self.assertFalse((layer.weight == 0x80).any())
        fn = layer.weight.view(torch.float8_e4m3fn).float()
        fnuz = layer.weight.view(torch.float8_e4m3fnuz).float()
        self.assertTrue(torch.equal(fn, fnuz * 2))

    def test_lookup_table_matches_native_float8(self):
        storage = torch.arange(256, dtype=torch.uint8)
        native = _safetensorscompat._decode_float8(storage, 'float8_e4m3fn', torch.float32)
        table = _safetensorscompat._lookup_table('float8_e4m3fn', storage.device, torch.float32)
        lookup = torch.nn.functional.embedding(storage.int(), table).view(-1)
        self.assertTrue(torch.equal(native.nan_to_num(), lookup.nan_to_num()))

    def _dense_error(self, layer, weight=None):
        weight = self.weight if weight is None else weight
        dense = layer.dequantize(torch.float32)[:weight.shape[0], :weight.shape[1]]
        return ((dense - weight).norm() / weight.norm()).item()

    def _loaded_layer(self, tensors, metadata=None):
        path = self._write(tensors, metadata=metadata)
        self.assertTrue(_safetensorscompat.is_comfy_quantized(path))
        return next(iter(_safetensorscompat.load_checkpoint(path, torch.float32).sources.values()))

    def test_mxfp8_marker_and_kitchen_metadata(self):
        weight, scale = _quantize_mxfp8(self.weight)
        tensors = {'layer.weight': weight, 'layer.weight_scale': scale}
        marker = {'layer.comfy_quant': _marker({'format': 'mxfp8'})}
        for metadata, extra in (
                (None, marker),
                (self._kitchen_metadata({'layer': {'format': 'mxfp8'}}), {})):
            with self.subTest(kitchen=metadata is not None):
                layer = self._loaded_layer({**tensors, **extra}, metadata)
                self.assertEqual(layer.format, 'mxfp8')
                self.assertLess(self._dense_error(layer), 0.05)
                module = _safetensorscompat.ComfyQuantLinear(256, 128, layer)
                module.load_state_dict(
                    {'weight': layer.weight, 'weight_scale': layer.scale}, assign=True)
                self.assertLess(self._relative_error(module(self.x)), 0.05)

    def test_mxfp8_padding_trims_to_the_linear(self):
        weight = torch.randn(100, 200) * 0.02
        stored, scale = _quantize_mxfp8(weight)
        self.assertEqual(tuple(stored.shape), (128, 224))
        layer = self._loaded_layer({
            'layer.weight': stored, 'layer.weight_scale': scale,
            'layer.comfy_quant': _marker({'format': 'mxfp8'})})
        self.assertTrue(layer.fits(100, 200))
        dense = layer.dequantize(torch.float32, (100, 200))
        self.assertLess(((dense - weight).norm() / weight.norm()).item(), 0.05)

    def test_mxfp8_without_metadata_is_inferred(self):
        # No comfy_quant and no Kitchen entry: uint8 E8M0 scales name the format.
        weight, scale = _quantize_mxfp8(self.weight)
        layer = self._loaded_layer({'layer.weight': weight, 'layer.weight_scale': scale})
        self.assertEqual(layer.format, 'mxfp8')
        self.assertLess(self._dense_error(layer), 0.05)

    def test_e8m0_zero_is_zero_and_bias_is_127(self):
        exponents = torch.tensor([0, 127, 128], dtype=torch.uint8)
        self.assertEqual(
            _safetensorscompat._e8m0_to_float(exponents).tolist(), [0.0, 1.0, 2.0])

    def test_int4_convrot_marker_and_kitchen_metadata(self):
        packed, scale = _quantize_int4_convrot(self.weight, 64)
        tensors = {'layer.weight': packed, 'layer.weight_scale': scale}
        config = {'format': 'convrot_w4a4', 'convrot_groupsize': 64}
        for metadata, extra in (
                (None, {'layer.comfy_quant': _marker(config)}),
                (self._kitchen_metadata({'layer': config}), {})):
            with self.subTest(kitchen=metadata is not None):
                layer = self._loaded_layer({**tensors, **extra}, metadata)
                self.assertEqual(layer.convrot_group_size, 64)
                self.assertLess(self._dense_error(layer), 0.15)
                module = _safetensorscompat.ComfyQuantLinear(256, 128, layer)
                module.load_state_dict(
                    {'weight': layer.weight, 'weight_scale': layer.scale}, assign=True)
                self.assertLess(self._relative_error(module(self.x)), 0.15)

    def test_int4_convrot_group_scales(self):
        # One scale per 64 inputs, rather than the per-row scale kitchen writes.
        packed, scale = _quantize_int4_convrot(self.weight, 64, grouped=True)
        self.assertEqual(tuple(scale.shape), (128, 4))
        layer = self._loaded_layer({
            'layer.weight': packed, 'layer.weight_scale': scale,
            'layer.comfy_quant': _marker(
                {'format': 'convrot_w4a4', 'convrot_groupsize': 64})})
        self.assertEqual(layer.group_size, 64)
        self.assertLess(self._dense_error(layer), 0.15)

    def test_w4a8_marker_kitchen_and_inferred(self):
        packed, rel, channel, codebook = _quantize_w4a8(self.weight, fp8_scales=True)
        tensors = {
            'layer.weight': packed, 'layer.weight_s_rel': rel,
            'layer.weight_s_channel': channel, 'layer.weight_codebook': codebook}
        config = {'format': 'asym_w4a8_int8', 'group_size': 16, 'convrot_groupsize': 256}
        cases = (
            ('marker', None, {'layer.comfy_quant': _marker(config)}),
            ('kitchen', self._kitchen_metadata({'layer': config}), {}),
            ('inferred', None, {}),
        )
        for name, metadata, extra in cases:
            with self.subTest(source=name):
                layer = self._loaded_layer({**tensors, **extra}, metadata)
                self.assertEqual((layer.format, layer.group_size, layer.convrot_group_size),
                                 ('asym_w4a8_int8', 16, 256))
                self.assertLess(self._dense_error(layer), 0.15)
                module = _safetensorscompat.ComfyQuantLinear(256, 128, layer)
                state = {'weight': layer.weight, 'weight_scale': layer.scale,
                         'weight_scale_2': layer.scale_2, 'weight_codebook': layer.codebook}
                module.load_state_dict(state, assign=True)
                for _, param in module.named_parameters():
                    self.assertFalse(param.is_floating_point())
                self.assertLess(self._relative_error(module(self.x)), 0.15)

    def test_w6a8_marker_kitchen_and_inferred(self):
        packed, rel, channel = _quantize_w6a8(self.weight)
        self.assertEqual(packed.shape[1], 192)
        tensors = {'layer.weight': packed, 'layer.weight_s_rel': rel,
                   'layer.weight_s_channel': channel}
        config = {'format': 'w6a8_int8', 'group_size': 16, 'convrot_groupsize': 256}
        cases = (
            ('marker', None, {'layer.comfy_quant': _marker(config)}),
            ('kitchen', self._kitchen_metadata({'layer': config}), {}),
            ('inferred', None, {}),
        )
        for name, metadata, extra in cases:
            with self.subTest(source=name):
                layer = self._loaded_layer({**tensors, **extra}, metadata)
                self.assertEqual((layer.format, layer.group_size, layer.convrot_group_size),
                                 ('w6a8_int8', 16, 256))
                self.assertIsNone(layer.codebook)
                self.assertLess(self._dense_error(layer), 0.1)
                module = _safetensorscompat.ComfyQuantLinear(256, 128, layer)
                module.load_state_dict(
                    {'weight': layer.weight, 'weight_scale': layer.scale,
                     'weight_scale_2': layer.scale_2}, assign=True)
                self.assertLess(self._relative_error(module(self.x)), 0.1)

    def test_w6a8_kitchen_group_size_and_exact_codes(self):
        packed, rel, channel = _quantize_w6a8(self.weight, group=32, convrot=64)
        layer = self._loaded_layer({
            'layer.weight': packed, 'layer.weight_s_rel': rel, 'layer.weight_s_channel': channel,
        }, self._kitchen_metadata({'layer': {
            'format': 'w6a8_int8', 'group_size': 32, 'convrot_groupsize': 64}}))
        self.assertEqual((layer.group_size, layer.convrot_group_size), (32, 64))
        self.assertLess(self._dense_error(layer), 0.1)

        rows, inputs, group, rotation = 4, 32, 16, 16
        codes = torch.randint(0, 64, (rows, inputs))
        rel = torch.rand(rows, inputs // group) + 0.05
        channel = torch.rand(rows) + 0.01
        levels = (codes.float().view(rows, -1, group) - 32) * rel.unsqueeze(-1)
        rotated = levels.round().clamp(-127, 127) * channel.view(rows, 1, 1)
        expected = _safetensorscompat._rotate(rotated.reshape(rows, inputs), rotation)
        exact = self._loaded_layer({
            'layer.weight': _pack_int6(codes), 'layer.weight_s_rel': rel,
            'layer.weight_s_channel': channel,
            'layer.comfy_quant': _marker(
                {'format': 'w6a8_int8', 'group_size': group, 'convrot_groupsize': rotation})})
        self.assertTrue(torch.allclose(exact.dequantize(torch.float32), expected, atol=1e-5))

    def test_w4a8_kitchen_group_size_is_not_the_default(self):
        # Metadata must supply the group. The default of 16 does not match this grid.
        packed, rel, channel, codebook = _quantize_w4a8(self.weight, group=32, convrot=64)
        layer = self._loaded_layer({
            'layer.weight': packed, 'layer.weight_s_rel': rel,
            'layer.weight_s_channel': channel, 'layer.weight_codebook': codebook,
        }, self._kitchen_metadata({'layer': {
            'format': 'asym_w4a8_int8', 'group_size': 32, 'convrot_groupsize': 64}}))
        self.assertEqual((layer.group_size, layer.convrot_group_size), (32, 64))
        self.assertLess(self._dense_error(layer), 0.15)

    def test_w4a8_correction_and_missing_codebook(self):
        rows, inputs, group, rotation = 8, 64, 16, 64
        codes = torch.randint(0, 16, (rows, inputs))
        packed = ((codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)).to(torch.int8)
        rel = torch.rand(rows, inputs // group) + 0.05
        channel = torch.rand(rows) + 0.01
        correction = torch.randn(inputs // group, rows) * 0.01
        codebook = torch.linspace(-1, 1, 16)
        levels = codebook[codes].view(rows, -1, group) * rel.unsqueeze(-1)
        rotated = levels.round().clamp(-127, 127) * channel.view(rows, 1, 1)
        rotated = rotated + correction.t().unsqueeze(-1)
        expected = _safetensorscompat._rotate(rotated.reshape(rows, inputs), rotation)
        config = {'format': 'asym_w4a8_int8', 'group_size': group, 'convrot_groupsize': rotation}
        base = {'layer.weight': packed, 'layer.weight_s_rel': rel,
                'layer.weight_s_channel': channel, 'layer.weight_correction': correction,
                'layer.comfy_quant': _marker(config)}
        with_codebook = self._loaded_layer({**base, 'layer.weight_codebook': codebook})
        self.assertTrue(torch.allclose(with_codebook.dequantize(torch.float32), expected, atol=1e-5))
        uniform = codes.float().view(rows, -1, group) - 8
        uniform = uniform * rel.unsqueeze(-1)
        uniform = uniform.round().clamp(-127, 127) * channel.view(rows, 1, 1)
        uniform = uniform + correction.t().unsqueeze(-1)
        uniform = _safetensorscompat._rotate(uniform.reshape(rows, inputs), rotation)
        without = self._loaded_layer(base)
        self.assertIsNone(without.codebook)
        self.assertTrue(torch.allclose(without.dequantize(torch.float32), uniform, atol=1e-5))

    def test_packed_formats_split_by_rows(self):
        # Fused projections are split while the storage is still packed.
        weight = torch.randn(128, 256) * 0.02
        mx_weight, mx_scale = _quantize_mxfp8(weight)
        int4, int4_scale = _quantize_int4_convrot(weight, 64)
        w4, rel, channel, codebook = _quantize_w4a8(weight, convrot=64)
        w6, w6_rel, w6_channel = _quantize_w6a8(weight, convrot=64)
        cases = {
            'mxfp8': {'weight': mx_weight, 'weight_scale': mx_scale,
                      'comfy_quant': _marker({'format': 'mxfp8'})},
            'int4': {'weight': int4, 'weight_scale': int4_scale,
                     'comfy_quant': _marker(
                         {'format': 'convrot_w4a4', 'convrot_groupsize': 64})},
            'w4a8': {'weight': w4, 'weight_s_rel': rel, 'weight_s_channel': channel,
                     'weight_codebook': codebook,
                     'comfy_quant': _marker(
                         {'format': 'asym_w4a8_int8', 'group_size': 16,
                          'convrot_groupsize': 64})},
            'w6a8': {'weight': w6, 'weight_s_rel': w6_rel, 'weight_s_channel': w6_channel,
                     'comfy_quant': _marker(
                         {'format': 'w6a8_int8', 'group_size': 16, 'convrot_groupsize': 64})},
        }
        for fmt, tensors in cases.items():
            with self.subTest(fmt=fmt):
                loaded, converted = self._convert(
                    {f'qkv.{name}': value for name, value in tensors.items()},
                    lambda checkpoint, **kwargs: dict(zip(
                        ('a.weight', 'b.weight'),
                        torch.chunk(checkpoint.pop('qkv.weight'), 2, dim=0))))
                for index, key in enumerate(('a.weight', 'b.weight')):
                    layer = self._assembled(loaded, converted, key)
                    expected = weight[index * 64:(index + 1) * 64]
                    error = (layer.dequantize(torch.float32, (64, 256)) - expected).norm() / expected.norm()
                    self.assertLess(error.item(), 0.15)

    def test_unsupported_format_is_an_error(self):
        path = self._write({
            'layer.weight': torch.zeros(128, 256, dtype=torch.int8),
            'layer.weight_scale': torch.ones(128, 1),
            'layer.comfy_quant': _marker({'format': 'not_a_format'})})
        with self.assertRaises(_safetensorscompat.ComfyQuantError):
            _safetensorscompat.load_checkpoint(path, torch.float32)

    def test_plain_checkpoint_is_not_comfy_quantized(self):
        path = self._write({'layer.weight': self.weight, 'layer.weight_scale': torch.ones(1)})
        self.assertFalse(_safetensorscompat.is_comfy_quantized(path))

    def _formats(self):
        """Quantized checkpoint tensors of a (384, 256) weight in every format, by name."""
        torch.manual_seed(1)
        weight = torch.randn(384, 256) * 0.02
        fp8, fp8_scale = _quantize_fp8(weight)
        int8, int8_scale = _quantize_int8_convrot(weight, 64)
        nvfp4, block_scale, tensor_scale = _quantize_nvfp4(weight)
        return weight, {
            'fp8': {'weight_scale': fp8_scale, 'weight': fp8,
                    'comfy_quant': _marker({'format': 'float8_e4m3fn'})},
            'int8': {'weight_scale': int8_scale, 'weight': int8,
                     'comfy_quant': _marker({'format': 'int8_tensorwise', 'convrot': True,
                                             'convrot_groupsize': 64})},
            'nvfp4': {'weight_scale': block_scale, 'weight_scale_2': tensor_scale, 'weight': nvfp4,
                      'comfy_quant': _marker({'format': 'nvfp4'})},
        }

    def _convert(self, tensors: dict, convert):
        """Load ``tensors``, then run ``convert`` as a Diffusers conversion under the guard."""
        import diffusers.loaders.single_file_model as single_file_model

        loaded = _safetensorscompat.load_checkpoint(self._write(tensors), torch.float32)
        name = next(iter(single_file_model.SINGLE_FILE_LOADABLE_CLASSES))
        model_class = getattr(__import__('diffusers'), name)
        entry = single_file_model.SINGLE_FILE_LOADABLE_CLASSES[name]
        original = entry['checkpoint_mapping_fn']
        entry['checkpoint_mapping_fn'] = convert
        try:
            with loaded.conversion_guard(model_class):
                converted = entry['checkpoint_mapping_fn'](config=None, checkpoint=loaded.checkpoint)
        finally:
            entry['checkpoint_mapping_fn'] = original
        return loaded, converted

    def _assembled(self, loaded, converted, key):
        segments = loaded.segments_for(key, converted[key])
        self.assertIsNotNone(segments, key)
        layer = _safetensorscompat._assemble_layer(key, segments, converted[key])
        self.assertIsNotNone(layer, key)
        return layer

    def test_conversion_split_keeps_layers_packed(self):
        # Fused qkv split into q, k, v (Flux, SD3, Z-Image, HunyuanVideo).
        weight, formats = self._formats()
        for fmt, tensors in formats.items():
            with self.subTest(fmt=fmt):
                loaded, converted = self._convert(
                    {f'qkv.{name}': value for name, value in tensors.items()},
                    lambda checkpoint, **kwargs: dict(zip(
                        ('to_q.weight', 'to_k.weight', 'to_v.weight'),
                        torch.chunk(checkpoint.pop('qkv.weight'), 3, dim=0))))
                self.assertIn('qkv.weight', loaded.sources)
                for index, key in enumerate(('to_q.weight', 'to_k.weight', 'to_v.weight')):
                    layer = self._assembled(loaded, converted, key)
                    self.assertFalse(layer.weight.is_floating_point())
                    expected = weight[index * 128:(index + 1) * 128]
                    error = (layer.dequantize(torch.float32, (128, 256)) - expected).norm() / expected.norm()
                    self.assertLess(error.item(), 0.15)

    def test_conversion_join_of_differently_scaled_sources(self):
        # Rows of two layers with their own scales, swapped and joined (swap_scale_shift).
        weight, formats = self._formats()
        for fmt, tensors in formats.items():
            with self.subTest(fmt=fmt):
                other = {name: value.clone() for name, value in tensors.items()}
                if fmt == 'nvfp4':
                    other['weight_scale_2'] = tensors['weight_scale_2'] * 2
                    reference_b = weight * 2
                else:
                    other['weight_scale'] = tensors['weight_scale'] * 2
                    reference_b = weight * 2
                source = {f'a.{name}': value for name, value in tensors.items()}
                source.update({f'b.{name}': value for name, value in other.items()})

                def convert(checkpoint, **kwargs):
                    shift, _ = checkpoint.pop('a.weight').chunk(2, dim=0)
                    _, scale = checkpoint.pop('b.weight').chunk(2, dim=0)
                    return {'out.weight': torch.cat([scale, shift], dim=0)}

                loaded, converted = self._convert(source, convert)
                layer = self._assembled(loaded, converted, 'out.weight')
                expected = torch.cat([reference_b[192:], weight[:192]])
                error = (layer.dequantize(torch.float32, (384, 256)) - expected).norm() / expected.norm()
                self.assertLess(error.item(), 0.15)
                per_row = layer.scale_2 if fmt == 'nvfp4' else layer.scale
                self.assertEqual(per_row.numel(), 384)

    def test_conversion_untraceable_use_dequantizes(self):
        weight, formats = self._formats()
        tensors = formats['fp8']
        loaded, converted = self._convert(
            {f'w.{name}': value for name, value in tensors.items()},
            lambda checkpoint, **kwargs: {'out.weight': checkpoint.pop('w.weight').t().contiguous()})
        self.assertNotIn('w.weight', loaded.sources)
        self.assertTrue(converted['out.weight'].is_floating_point())
        self.assertIsNone(loaded.segments_for('out.weight', converted['out.weight']))

    def test_nvfp4_rows_not_a_multiple_of_16_are_padded(self):
        weight, formats = self._formats()
        loaded, converted = self._convert(
            {f'w.{name}': value for name, value in formats['nvfp4'].items()},
            lambda checkpoint, **kwargs: {'out.weight': checkpoint.pop('w.weight')[:100]})
        layer = self._assembled(loaded, converted, 'out.weight')
        self.assertEqual(layer.weight.shape[0], 112)
        self.assertTrue(layer.fits(100, 256))
        expected = weight[:100]
        error = (layer.dequantize(torch.float32, (100, 256)) - expected).norm() / expected.norm()
        self.assertLess(error.item(), 0.15)

    def test_plain_float8_weights_are_packed_full_precision(self):
        weight = self.weight.to(torch.float8_e4m3fn)
        path = self._write({'layer.weight': weight, 'layer.bias': torch.zeros(128)})
        self.assertTrue(_safetensorscompat.is_comfy_quantized(path))
        layer = next(iter(_safetensorscompat.load_checkpoint(path, torch.float32).sources.values()))
        self.assertTrue(layer.full_precision)
        self.assertTrue(torch.equal(layer.dequantize(torch.float32), weight.float()))

    def _comfy_fp8_state(self, state: dict, prefix: str = '', legacy: bool = False) -> dict:
        """A ComfyUI style file: 2D linear weights as scaled float8, the rest as they are."""
        tensors = {}
        for key, value in state.items():
            value = value.detach().clone()
            if key.endswith('.weight') and value.ndim == 2 and 'embed' not in key and 'shared' not in key \
                    and 'lm_head' not in key:
                weight, scale = _quantize_fp8(value.float())
                tensors[prefix + key] = weight
                tensors[prefix + key[:-len('weight')] + ('scale_weight' if legacy else 'weight_scale')] = scale
            else:
                tensors[prefix + key] = value.float()
        if legacy:
            tensors['scaled_fp8'] = torch.empty(0, dtype=torch.float8_e4m3fn)
        return tensors

    def test_language_model_text_encoder_single_file(self):
        import json
        import transformers
        from dgenerate.pipelinewrapper.uris import textencoderuri

        torch.manual_seed(0)
        config = transformers.Qwen3Config(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True)
        reference = transformers.Qwen3ForCausalLM(config).eval()
        # ComfyUI stores the language model under ``model.``, without lm_head when tied.
        state = {k: v for k, v in reference.state_dict().items() if not k.startswith('lm_head')}
        path = self._write(self._comfy_fp8_state(state))
        handle, config_path = tempfile.mkstemp(suffix='.json')
        os.close(handle)
        self.addCleanup(os.remove, config_path)
        with open(config_path, 'w') as file:
            json.dump(config.to_dict(), file)

        model = textencoderuri._load_language_model_from_single_file(
            transformers.Qwen3ForCausalLM, path, config_path, torch.float32)
        packed = [m for m in model.modules() if isinstance(m, _safetensorscompat.ComfyQuantLinear)]
        self.assertEqual(len(packed), 2 * 7)
        tokens = torch.randint(0, 128, (1, 12))
        with torch.no_grad():
            expected = reference(tokens, output_hidden_states=True).hidden_states[-1]
            output = model(tokens, output_hidden_states=True).hidden_states[-1]
        self.assertLess(((output - expected).norm() / expected.norm()).item(), 0.05)

    def test_trimmed_language_model_matches_full_model_hidden_state(self):
        # ComfyUI's Flux.2 Mistral file keeps 30 of 40 layers and no final norm;
        # hidden_states[30] of the trimmed model must equal the full model's.
        import json
        import transformers
        from dgenerate.pipelinewrapper.uris import textencoderuri

        torch.manual_seed(0)
        config = transformers.Qwen3Config(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16)
        reference = transformers.Qwen3ForCausalLM(config).eval()
        state = {k: v for k, v in reference.state_dict().items()
                 if not k.startswith(('lm_head', 'model.norm', 'model.layers.3.'))}
        path = self._write(self._comfy_fp8_state(state))
        handle, config_path = tempfile.mkstemp(suffix='.json')
        os.close(handle)
        self.addCleanup(os.remove, config_path)
        with open(config_path, 'w') as file:
            json.dump(config.to_dict(), file)

        model = textencoderuri._load_language_model_from_single_file(
            transformers.Qwen3ForCausalLM, path, config_path, torch.float32)
        self.assertEqual(len(model.model.layers), 3)
        self.assertIsInstance(model.model.norm, torch.nn.Identity)
        tokens = torch.randint(0, 128, (1, 12))
        with torch.no_grad():
            expected = reference(tokens, output_hidden_states=True).hidden_states[3]
            output = model(tokens, output_hidden_states=True).hidden_states[3]
        self.assertLess(((output - expected).norm() / expected.norm()).item(), 0.05)

    def test_language_model_rename_resolves_each_tensor(self):
        from dgenerate.pipelinewrapper.uris import textencoderuri

        # Qwen2.5-VL: language model and vision tower under different prefixes.
        model_keys = {'model.language_model.layers.0.mlp.up_proj.weight',
                      'model.language_model.embed_tokens.weight',
                      'model.visual.blocks.0.attn.qkv.weight', 'lm_head.weight'}
        rename = textencoderuri._language_model_rename(
            model_keys, ['model.layers.0.mlp.up_proj.weight', 'model.embed_tokens.weight',
                         'visual.blocks.0.attn.qkv.weight', 'lm_head.weight'])
        self.assertEqual(rename('model.layers.0.mlp.up_proj.weight'),
                         'model.language_model.layers.0.mlp.up_proj.weight')
        self.assertEqual(rename('visual.blocks.0.attn.qkv.weight'), 'model.visual.blocks.0.attn.qkv.weight')
        self.assertEqual(rename('lm_head.weight'), 'lm_head.weight')

    def test_monolithic_t5_keeps_wo_dense(self):
        import transformers
        from dgenerate.pipelinewrapper.uris import textencoderuri

        torch.manual_seed(0)
        config = transformers.T5Config(
            vocab_size=128, d_model=64, d_kv=16, d_ff=128, num_layers=2, num_heads=4,
            feed_forward_proj='gated-gelu')
        reference = transformers.T5EncoderModel(config).eval()
        path = self._write(self._comfy_fp8_state(reference.state_dict(), legacy=True))
        model = transformers.T5EncoderModel(config).eval()
        model = textencoderuri._load_monolithic_checkpoint(
            model, path, device_map=None, dtype=torch.float32, no_split_module_classes=['T5Block'])
        layer = model.encoder.block[0].layer[1].DenseReluDense
        self.assertIsInstance(layer.wi_0, _safetensorscompat.ComfyQuantLinear)
        self.assertNotIsInstance(layer.wo, _safetensorscompat.ComfyQuantLinear)
        self.assertTrue(layer.wo.weight.is_floating_point())
        tokens = torch.randint(0, 128, (1, 12))
        with torch.no_grad():
            expected = reference(tokens).last_hidden_state
            output = model(tokens).last_hidden_state
        self.assertLess(((output - expected).norm() / expected.norm()).item(), 0.05)

    def test_single_file_components_dequantize_encoders_only(self):
        weight, scale = _quantize_fp8(self.weight)
        path = self._write({
            'model.diffusion_model.block.weight': weight, 'model.diffusion_model.block.weight_scale': scale,
            'text_encoders.t5xxl.layer.weight': weight.clone(),
            'text_encoders.t5xxl.layer.scale_weight': scale.clone(),
            'scaled_fp8': torch.empty(0, dtype=torch.float8_e4m3fn)})
        _safetensorscompat.install_single_file_patch()
        import diffusers.loaders.single_file as single_file
        with _safetensorscompat.single_file_components(transformer_loaded=True):
            checkpoint = single_file.load_single_file_checkpoint(path)
            again = single_file.load_single_file_checkpoint(path)
        self.assertEqual(checkpoint['model.diffusion_model.block.weight'].device.type, 'meta')
        encoder = checkpoint['text_encoders.t5xxl.layer.weight']
        self.assertEqual(encoder.dtype, torch.bfloat16)
        self.assertLess(((encoder.float() - self.weight).norm() / self.weight.norm()).item(), 0.05)
        self.assertIsNot(checkpoint, again)
        self.assertIs(checkpoint['text_encoders.t5xxl.layer.weight'], again['text_encoders.t5xxl.layer.weight'])
        # A pipeline load without a supplied transformer reads the denoiser too.
        with _safetensorscompat.single_file_components(transformer_loaded=True):
            with _safetensorscompat.single_file_components(transformer_loaded=False):
                nested = single_file.load_single_file_checkpoint(path)
        self.assertEqual(nested['model.diffusion_model.block.weight'].device.type, 'cpu')
        # Outside a scope every layer is dequantized.
        everything = single_file.load_single_file_checkpoint(path)
        self.assertEqual(everything['model.diffusion_model.block.weight'].device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()
