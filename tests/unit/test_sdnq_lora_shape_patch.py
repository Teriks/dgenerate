import unittest

import torch

import dgenerate  # noqa: F401  applies patches
import diffusers.loaders.lora_pipeline as lora_pipeline


class _Dequantizer:
    original_shape = (8, 4)


class _PackedLinear:
    def __init__(self):
        self.weight = torch.empty(16, dtype=torch.uint8)
        self.sdnq_dequantizer = _Dequantizer()


class TestSdnqLoraShapePatch(unittest.TestCase):
    def test_packed_sdnq_weight_reports_the_original_shape(self):
        module = _PackedLinear()
        shape = lora_pipeline.FluxLoraLoaderMixin._calculate_module_shape(
            None, base_module=module)
        out_features, in_features = shape
        self.assertEqual((out_features, in_features), (8, 4))

    def test_plain_linear_shape_is_unchanged(self):
        linear = torch.nn.Linear(4, 8, bias=False)
        shape = lora_pipeline.FluxLoraLoaderMixin._calculate_module_shape(
            None, base_module=linear)
        self.assertEqual(tuple(shape), tuple(linear.weight.shape))
