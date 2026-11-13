import unittest

import torch

import dgenerate.pipelinewrapper.ggufcompat as _ggufcompat


def _plant_half(data, type_size, offset):
    rows, width = data.shape
    blocks = data.view(rows, width // type_size, type_size)
    values = torch.randn(blocks.shape[0], blocks.shape[1], dtype=torch.float16)
    blocks[:, :, offset:offset + 2] = values.contiguous().view(torch.uint8).view(
        blocks.shape[0], blocks.shape[1], 2)


class TestGGUFKernel(unittest.TestCase):
    def test_cpu_and_bf16_stay_on_pytorch(self):
        import gguf
        from diffusers.quantizers.gguf.utils import GGUFParameter

        data = torch.zeros(2, 18, dtype=torch.uint8)
        param = GGUFParameter(data, quant_type=gguf.GGMLQuantizationType.Q4_0)
        self.assertIsNone(_ggufcompat.dequantize(param))
        if not torch.cuda.is_available():
            return
        bf16 = GGUFParameter(
            torch.zeros(4, 8, dtype=torch.uint8, device='cuda'),
            quant_type=gguf.GGMLQuantizationType.BF16)
        self.assertIsNone(_ggufcompat.dequantize(bf16))

    def test_dequantize_matches_diffusers(self):
        if not torch.cuda.is_available():
            self.skipTest('no CUDA device')
        if not _ggufcompat._has_triton():
            self.skipTest('triton is not installed')
        import gguf
        from diffusers.quantizers.gguf.utils import (
            GGML_QUANT_SIZES, GGUFParameter, dequantize_functions)

        half_offsets = {
            'Q4_0': (0,), 'Q4_1': (0, 2), 'Q5_0': (0,), 'Q5_1': (0, 2), 'Q8_0': (0,),
            'IQ4_NL': (0,), 'Q2_K': (80, 82), 'Q3_K': (108,), 'Q4_K': (0, 2),
            'Q5_K': (0, 2), 'Q6_K': (208,), 'IQ4_XS': (0,),
        }
        rows = 4
        for name, offsets in half_offsets.items():
            qtype = getattr(gguf.GGMLQuantizationType, name)
            block, type_size = GGML_QUANT_SIZES[qtype]
            per_row = max(1, 256 // block)
            data = torch.randint(0, 256, (rows, per_row * type_size), dtype=torch.uint8)
            for offset in offsets:
                _plant_half(data, type_size, offset)
            reference = dequantize_functions[qtype](
                data.reshape(-1, type_size), block, type_size).float()
            reference = reference.reshape(rows, per_row * block)
            param = GGUFParameter(data.cuda(), quant_type=qtype)
            with self.subTest(quant=name):
                output = _ggufcompat.dequantize(param)
                self.assertEqual(tuple(output.shape), (rows, per_row * block))
                error = (output.cpu() - reference).norm() / reference.norm().clamp(min=1e-6)
                self.assertLess(error.item(), 0.01)

    def test_linear_and_comfy_transpose(self):
        if not torch.cuda.is_available():
            self.skipTest('no CUDA device')
        if not _ggufcompat._has_triton():
            self.skipTest('triton is not installed')
        import gguf
        from diffusers.quantizers.gguf.utils import GGML_QUANT_SIZES, GGUFParameter

        qtype = gguf.GGMLQuantizationType.Q8_0
        block, type_size = GGML_QUANT_SIZES[qtype]
        data = torch.randint(0, 256, (8, type_size), dtype=torch.uint8)
        _plant_half(data, type_size, 0)
        reference = _ggufcompat.dequantize(GGUFParameter(data.cuda(), quant_type=qtype))
        self.assertEqual(tuple(reference.shape), (8, block))

        class Layer:
            bias = None
            compute_dtype = torch.bfloat16

        layer = Layer()
        layer.weight = GGUFParameter(data.cuda(), quant_type=qtype)
        activation = torch.randn(3, block, device='cuda', dtype=torch.bfloat16)
        output = _ggufcompat.linear(layer, activation)
        expected = torch.nn.functional.linear(activation, reference.to(torch.bfloat16))
        error = (output.float() - expected.float()).norm() / expected.float().norm().clamp(min=1e-6)
        self.assertLess(error.item(), 0.02)

        layer.weight._dgenerate_comfy_transpose = True
        layer.bias = torch.randn(block, device='cuda')
        activation = torch.randn(3, 8, device='cuda', dtype=torch.bfloat16)
        output = _ggufcompat.linear(layer, activation)
        expected = torch.nn.functional.linear(
            activation, reference.transpose(0, 1).to(torch.bfloat16), layer.bias.to(torch.bfloat16))
        error = (output.float() - expected.float()).norm() / expected.float().norm().clamp(min=1e-6)
        self.assertLess(error.item(), 0.02)


if __name__ == '__main__':
    unittest.main()
