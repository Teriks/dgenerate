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


class TestGroupOffloadDtype(unittest.TestCase):
    def test_stream_snapshot_refresh_replaces_stale_dtype(self):
        from dgenerate._patches.diffusers_group_offload_dtype_patch import (
            refresh_cpu_snapshots,
        )

        linear = torch.nn.Linear(4, 4, dtype=torch.float32)
        reference = {name: param.detach().clone() for name, param in linear.named_parameters()}

        class _Group:
            def __init__(self):
                self.stream = object()
                self.low_cpu_mem_usage = True
                self.modules = [linear]
                self.parameters = []
                self.buffers = []
                self.cpu_param_dict = {
                    param: param.detach().to(dtype=torch.bfloat16).clone()
                    for param in linear.parameters()
                }

            @staticmethod
            def _to_cpu(tensor, low_cpu_mem_usage):
                return tensor.data.detach().cpu()

        group = _Group()
        self.assertTrue(refresh_cpu_snapshots(group))
        self.assertFalse(refresh_cpu_snapshots(group))
        for name, param in linear.named_parameters():
            snap = group.cpu_param_dict[param]
            self.assertEqual(snap.dtype, torch.float32)
            self.assertTrue(torch.equal(snap, reference[name]))

        group.stream = None
        linear.weight.data = linear.weight.data.to(dtype=torch.float64)
        self.assertFalse(refresh_cpu_snapshots(group))

    def test_group_offload_without_stream_keeps_dtype_cast(self):
        from diffusers.hooks import apply_group_offloading

        layer = torch.nn.Linear(4, 4, dtype=torch.float32)
        reference = torch.nn.Linear(4, 4, dtype=torch.float32)
        reference.load_state_dict(layer.state_dict())
        apply_group_offloading(
            layer,
            onload_device=torch.device('cpu'),
            offload_device=torch.device('cpu'),
            offload_type='leaf_level',
            use_stream=False,
        )
        layer.to(dtype=torch.float64)
        reference.to(dtype=torch.float64)
        sample = torch.randn(2, 4, dtype=torch.float64)
        self.assertEqual(layer.bias.dtype, torch.float64)
        self.assertTrue(torch.allclose(layer(sample), reference(sample)))

    def test_streamed_group_offload_keeps_dtype_cast(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA stream group offload')
        from diffusers.hooks import apply_group_offloading

        layer = torch.nn.Conv3d(2, 2, kernel_size=1, dtype=torch.bfloat16)
        reference = torch.nn.Conv3d(2, 2, kernel_size=1, dtype=torch.bfloat16)
        reference.load_state_dict({
            name: param.detach().clone() for name, param in layer.state_dict().items()})
        apply_group_offloading(
            layer,
            onload_device=torch.device('cuda'),
            offload_device=torch.device('cpu'),
            offload_type='leaf_level',
            use_stream=True,
        )
        layer.to(dtype=torch.float32)
        reference.to(device='cuda', dtype=torch.float32)
        sample = torch.randn(1, 2, 2, 4, 4, dtype=torch.float32)
        self.assertEqual(layer.bias.dtype, torch.float32)
        output = layer(sample)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(layer.bias.dtype, torch.float32)
        self.assertTrue(torch.allclose(output, reference(sample.cuda())))
        again = layer(sample)
        self.assertEqual(layer.bias.dtype, torch.float32)
        self.assertTrue(torch.allclose(again, reference(sample.cuda())))

        layer.to(dtype=torch.bfloat16)
        reference.to(dtype=torch.bfloat16)
        sample16 = sample.to(dtype=torch.bfloat16)
        cast_back = layer(sample16)
        self.assertEqual(layer.bias.dtype, torch.bfloat16)
        self.assertEqual(cast_back.dtype, torch.bfloat16)
        self.assertTrue(torch.allclose(cast_back, reference(sample16.cuda())))

    def test_snapshot_refresh_leaves_packed_weights(self):
        from dgenerate._patches.diffusers_group_offload_dtype_patch import (
            refresh_cpu_snapshots,
        )

        linear = torch.nn.Linear(2, 2)
        linear.weight.quant_type = 8

        class _Group:
            def __init__(self):
                self.stream = object()
                self.low_cpu_mem_usage = True
                self.modules = [linear]
                self.parameters = []
                self.buffers = []
                self.cpu_param_dict = {
                    param: param.detach().to(dtype=torch.bfloat16).clone()
                    for param in linear.parameters()
                }

            @staticmethod
            def _to_cpu(tensor, low_cpu_mem_usage):
                raise AssertionError('packed weights must not be copied into a snapshot')

        self.assertFalse(refresh_cpu_snapshots(_Group()))


class _RecordsTo(torch.nn.Module):
    def __init__(self, **marks):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 2))
        for name, value in marks.items():
            setattr(self, name, value)
        self.calls = []

    def to(self, *args, **kwargs):
        self.calls.append(args[0] if args else kwargs.get('device'))
        return self


class TestGroupOffloadQuantPlacement(unittest.TestCase):
    def test_packed_markers_skip_group_offload(self):
        ordinary = torch.nn.Linear(2, 2)
        self.assertFalse(_pipelines.module_skips_group_offload(ordinary))

        class GGUFLinear(torch.nn.Linear):
            pass

        class SDNQLinear(torch.nn.Linear):
            pass

        self.assertTrue(_pipelines.module_skips_group_offload(GGUFLinear(2, 2)))
        self.assertTrue(_pipelines.module_skips_group_offload(SDNQLinear(2, 2)))

        marked = torch.nn.Linear(2, 2)
        marked.sdnq_dequantizer = object()
        parent = torch.nn.Sequential(marked)
        self.assertTrue(_pipelines.module_skips_group_offload(parent))

        packed = torch.nn.Linear(2, 2)
        packed.weight.quant_type = 2
        self.assertTrue(_pipelines.module_skips_group_offload(packed))
        packed.weight.quant_type = None
        packed.weight.quant_state = object()
        self.assertTrue(_pipelines.module_skips_group_offload(packed))

    def test_quantized_module_is_placed_not_hooked(self):
        quant = _RecordsTo(quantization_config={'quant_method': 'gguf'})
        bit8 = _RecordsTo(
            is_loaded_in_8bit=True,
            quantization_method=QuantizationMethod.BITS_AND_BYTES)
        plain = torch.nn.Linear(2, 2)

        class Pipe:
            def __init__(self):
                self.quant = quant
                self.bit8 = bit8
                self.plain = plain
                self._exclude_from_cpu_offload = []

            @property
            def components(self):
                return {'quant': self.quant, 'bit8': self.bit8, 'plain': self.plain}

            def remove_all_hooks(self):
                return None

            def to(self, *args, **kwargs):
                raise AssertionError('pipeline.to must not run')

        pipe = Pipe()
        _pipelines.enable_group_offload(pipe, 'cpu')
        self.assertTrue(_pipelines.is_group_offload_enabled(pipe))
        self.assertTrue(_pipelines.is_group_offload_enabled(plain))
        self.assertFalse(_pipelines.is_group_offload_enabled(quant))
        self.assertFalse(_pipelines.is_group_offload_enabled(bit8))
        self.assertEqual(quant.calls, [])
        self.assertEqual(bit8.calls, [])

        quant.calls.clear()
        bit8.calls.clear()
        _pipelines.place_quantized_module(quant, 'cuda')
        _pipelines.place_quantized_module(bit8, 'cuda')
        self.assertEqual(quant.calls, [torch.device('cuda')])
        self.assertEqual(bit8.calls, [])

        _pipelines.pipeline_to(pipe, 'cuda')
        self.assertEqual(quant.calls, [torch.device('cuda'), torch.device('cuda')])
        self.assertEqual(bit8.calls, [])
        self.assertEqual(plain.weight.device.type, 'cpu')


class TestOtherOffloadStillMovesQuant(unittest.TestCase):
    def test_sequential_offload_still_hooks_gguf_marker(self):
        gguf = torch.nn.Linear(2, 2)
        gguf.quantization_config = {'quant_method': 'gguf'}
        bit8 = torch.nn.Linear(2, 2)
        bit8.is_loaded_in_8bit = True
        bit8.quantization_method = QuantizationMethod.BITS_AND_BYTES

        class Pipe:
            def __init__(self):
                self.components = {'gguf': gguf, 'bit8': bit8}
                self._exclude_from_cpu_offload = []

            def remove_all_hooks(self):
                for module in self.components.values():
                    if hasattr(module, '_hf_hook'):
                        remove_hook_from_module(module, recurse=True)

        _pipelines.enable_sequential_cpu_offload(Pipe(), 'cpu')
        self.assertTrue(hasattr(gguf, '_hf_hook'))
        self.assertFalse(hasattr(bit8, '_hf_hook'))

    def test_sequential_offload_promotes_bare_parameter_tensors(self):
        layer = torch.nn.Linear(4, 4)
        layer._parameters['weight'] = torch.zeros(4, 4)

        class Pipe:
            def __init__(self):
                self.components = {'transformer': layer}
                self._exclude_from_cpu_offload = []

            def remove_all_hooks(self):
                for module in self.components.values():
                    if hasattr(module, '_hf_hook'):
                        remove_hook_from_module(module, recurse=True)

        _pipelines.enable_sequential_cpu_offload(Pipe(), 'cpu')
        self.assertIsInstance(layer.weight, torch.nn.Parameter)
        self.assertEqual(type(layer.weight).__name__, 'Parameter')
        self.assertEqual(layer.weight.device.type, 'meta')
        self.assertTrue(hasattr(layer, '_hf_hook'))

    def test_model_cpu_offload_still_hooks_gguf_marker(self):
        gguf = torch.nn.Linear(2, 2)
        gguf.quantization_config = {'quant_method': 'sdnq'}
        bit8 = torch.nn.Linear(2, 2)
        bit8.is_loaded_in_8bit = True
        bit8.quantization_method = QuantizationMethod.BITS_AND_BYTES

        class Pipe:
            model_cpu_offload_seq = 'gguf->bit8'

            def __init__(self):
                self.components = {'gguf': gguf, 'bit8': bit8}
                self._exclude_from_cpu_offload = []
                self.device = torch.device('cpu')

            def remove_all_hooks(self):
                for module in self.components.values():
                    if hasattr(module, '_hf_hook'):
                        remove_hook_from_module(module, recurse=True)

            def to(self, *args, **kwargs):
                return self

        _pipelines.enable_model_cpu_offload(Pipe(), 'cpu')
        self.assertTrue(hasattr(gguf, '_hf_hook'))
        self.assertFalse(hasattr(bit8, '_hf_hook'))


if __name__ == '__main__':
    unittest.main()
