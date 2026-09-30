import unittest

import torch
from accelerate.hooks import remove_hook_from_module
from transformers.utils.quantization_config import QuantizationMethod

import dgenerate.pipelinewrapper.pipelines as _pipelines


class TestSequentialOffloadReentry(unittest.TestCase):
    def test_pipeline_method_reinstalls_dgenerate_hooks(self):
        unet = torch.nn.Linear(2, 2)
        quantized = torch.nn.Linear(2, 2)
        quantized.is_loaded_in_8bit = True
        quantized.quantization_method = QuantizationMethod.BITS_AND_BYTES

        class Pipe:
            def __init__(self):
                self.components = {'unet': unet, 'quantized': quantized}
                self._exclude_from_cpu_offload = []

            def enable_sequential_cpu_offload(self, gpu_id=None, device=None):
                raise AssertionError('Diffusers sequential offload must not run')

            def remove_all_hooks(self):
                for module in self.components.values():
                    if hasattr(module, '_hf_hook'):
                        remove_hook_from_module(module, recurse=True)

        pipe = Pipe()
        _pipelines.enable_sequential_cpu_offload(pipe, 'cpu')
        self.assertTrue(hasattr(unet, '_hf_hook'))
        self.assertFalse(hasattr(quantized, '_hf_hook'))
        self.assertTrue(hasattr(unet, '_DGENERATE_ORIGINAL_TO_DISABLED'))

        pipe.remove_all_hooks()
        self.assertFalse(hasattr(unet, '_hf_hook'))
        pipe.enable_sequential_cpu_offload()
        self.assertTrue(hasattr(unet, '_hf_hook'))
        self.assertFalse(hasattr(quantized, '_hf_hook'))


if __name__ == '__main__':
    unittest.main()
