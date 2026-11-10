import unittest
from unittest.mock import patch

import torch
from torch import nn

import dgenerate.pipelinewrapper.pipelines as pipelines
from dgenerate.pipelinewrapper.pipelines import (
    eager_sdnq_on_cpu,
    pipeline_has_sdnq,
)


class _Quantized(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.linear.sdnq_dequantizer = object()


class _Pipeline:
    def __init__(self, text_encoder):
        self.components = {'text_encoder': text_encoder}
        self.image_encoder = None


class TestSdnqCpuEager(unittest.TestCase):

    def test_plain_module_is_not_sdnq(self):
        self.assertFalse(pipeline_has_sdnq(nn.Linear(2, 2)))

    def test_pipeline_components_are_searched(self):
        linear = nn.Linear(2, 2)
        linear.sdnq_dequantizer = object()
        self.assertTrue(pipeline_has_sdnq(_Pipeline(linear)))
        self.assertFalse(pipeline_has_sdnq(_Pipeline(nn.Linear(2, 2))))

    def test_cpu_sdnq_uses_the_eager_dequantizer(self):
        import sdnq.dequantizer as dequantizer

        pipeline = _Quantized()
        self.assertTrue(pipeline_has_sdnq(pipeline))
        compiled = dequantizer.dequantize_weight_compiled
        dynamo = torch._dynamo.config.disable
        try:
            with eager_sdnq_on_cpu(pipeline, 'cpu'):
                self.assertIs(
                    dequantizer.dequantize_weight_compiled,
                    dequantizer.dequantize_weight)
                self.assertEqual(torch._dynamo.config.disable, dynamo)
            self.assertIs(dequantizer.dequantize_weight_compiled, compiled)
            with eager_sdnq_on_cpu(pipeline, 'cuda'):
                self.assertIs(dequantizer.dequantize_weight_compiled, compiled)
        finally:
            dequantizer.dequantize_weight_compiled = compiled

    def test_missing_compiler_warns_once(self):
        pipeline = _Quantized()
        pipelines._CPU_CXX_COMPILER = False
        pipelines._SDNQ_CPU_COMPILER_WARNED = False
        try:
            with patch('dgenerate.pipelinewrapper.pipelines._messages.warning') as warning:
                with eager_sdnq_on_cpu(pipeline, 'cpu'):
                    pass
                with eager_sdnq_on_cpu(pipeline, 'cpu'):
                    pass
            self.assertEqual(warning.call_count, 1)
            self.assertIn('running eager', warning.call_args.args[0])
        finally:
            pipelines._CPU_CXX_COMPILER = None
            pipelines._SDNQ_CPU_COMPILER_WARNED = False

    def test_cpu_compile_follows_the_compiler_check(self):
        saved = pipelines._CPU_CXX_COMPILER
        try:
            pipelines._CPU_CXX_COMPILER = False
            self.assertFalse(pipelines.inductor_can_compile('cpu'))
            pipelines._CPU_CXX_COMPILER = True
            self.assertTrue(pipelines.inductor_can_compile('cpu'))
        finally:
            pipelines._CPU_CXX_COMPILER = saved
