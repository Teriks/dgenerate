import os
import tempfile
import unittest
from unittest import mock

import torch

from dgenerate.pipelinewrapper.videopipelines import diskoffload
from dgenerate.pipelinewrapper.videopipelines.diskoffload import (
    _loras_for_component,
    offload_module_weights,
    pipeline_fits_in_cpu_memory,
    should_stream_weights_from_disk,
    uncounted_component_names,
)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, hidden):
        return self.linear(hidden)


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _Block()

    def forward(self, hidden):
        return self.block(hidden)


class TestCpuMemoryFit(unittest.TestCase):
    def test_margin_keeps_a_small_pipeline_resident(self):
        available = 64 * 1024 ** 3
        self.assertTrue(pipeline_fits_in_cpu_memory(20 * 1024 ** 3, available))

    def test_margin_streams_a_pipeline_that_fills_memory(self):
        available = 48 * 1024 ** 3
        self.assertFalse(pipeline_fits_in_cpu_memory(45 * 1024 ** 3, available))

    def test_page_file_budget_keeps_the_pipeline_resident(self):
        # LTX-2.5 is about 66 GiB of files and ran in 64 GiB of RAM.
        weights = 66 * 1024 ** 3
        physical = 64 * 1024 ** 3
        commit = 132 * 1024 ** 3
        self.assertTrue(pipeline_fits_in_cpu_memory(
            weights, commit, physical_bytes=physical))

    def test_weights_larger_than_ram_stream_despite_page_file(self):
        # Wan 2.2 I2V is about 117.5 GiB. The page file can commit that,
        # and the run then holds about 225 GB for every step.
        weights = int(117.5 * 1024 ** 3)
        physical = 64 * 1024 ** 3
        commit = int(146.5 * 1024 ** 3)
        self.assertFalse(pipeline_fits_in_cpu_memory(
            weights, commit, physical_bytes=physical))

    def test_packed_weights_stay_on_the_normal_load_path(self):
        estimate = 80 * 1024 ** 3
        available = 16 * 1024 ** 3
        self.assertFalse(should_stream_weights_from_disk(
            estimate, quantizer_uri='bnb', available_bytes=available))
        self.assertFalse(should_stream_weights_from_disk(
            estimate, transformer_uri='org/model;quantizer=bnb',
            available_bytes=available))
        self.assertTrue(should_stream_weights_from_disk(
            estimate, available_bytes=available))


class TestComponentLoraSelection(unittest.TestCase):
    def test_default_lora_targets_the_high_noise_expert(self):
        uris = [
            'org/loras;scale=1.0;weight-name=high.safetensors',
            'org/loras;scale=0.5;weight-name=low.safetensors;component=transformer_2',
        ]
        high = _loras_for_component('transformer', uris)
        low = _loras_for_component('transformer_2', uris)
        self.assertEqual(len(high), 1)
        self.assertEqual(high[0].weight_name, 'high.safetensors')
        self.assertEqual(len(low), 1)
        self.assertEqual(low[0].component, 'transformer_2')
        self.assertEqual(_loras_for_component('vae', uris), [])


class TestUncountedComponents(unittest.TestCase):
    def test_second_wan_expert_is_part_of_the_footprint(self):
        index = {
            '_class_name': 'WanImageToVideoPipeline',
            'transformer': ['diffusers', 'WanTransformer3DModel'],
            'transformer_2': ['diffusers', 'WanTransformer3DModel'],
            'text_encoder': ['transformers', 'UMT5EncoderModel'],
            'vae': ['diffusers', 'AutoencoderKLWan'],
            'image_encoder': ['transformers', 'CLIPVisionModel'],
        }
        names = uncounted_component_names(index)
        self.assertEqual(names, ['transformer_2', 'image_encoder'])
        self.assertEqual(
            uncounted_component_names(index, replaced={'transformer_2'}),
            ['image_encoder'])


class TestDiskOffload(unittest.TestCase):
    def _save(self, module, folder):
        path = os.path.join(folder, 'model.safetensors')
        import safetensors.torch
        safetensors.torch.save_file(module.state_dict(), path)
        return path

    def test_reused_checkpoint_stays_on_disk_between_loads(self):
        module = _Net()
        sample = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        expected = module(sample).detach().clone()
        with tempfile.TemporaryDirectory() as folder:
            self._save(module, folder)
            offload = os.path.join(folder, 'offload')
            offload_module_weights(
                module, folder, offload, torch.device('cpu'), dirty=False)
            self.assertFalse(os.path.exists(os.path.join(offload, 'weights.safetensors')))
            self.assertTrue(module.block.linear.weight.is_meta)
            other = _Net()
            self.assertFalse(other.block.linear.weight.is_meta)
            result = module(sample)
            self.assertTrue(torch.allclose(result, expected))
            self.assertTrue(module.block.linear.weight.is_meta)
            module.to(torch.device('cpu'))
            self.assertTrue(module.block.linear.weight.is_meta)

    def test_dirty_weights_are_the_ones_read_back(self):
        module = _Net()
        with torch.no_grad():
            module.block.linear.bias.zero_()
        sample = torch.ones(2, 4)
        expected = module(sample).detach().clone()
        with tempfile.TemporaryDirectory() as folder:
            self._save(_Net(), folder)
            offload = os.path.join(folder, 'offload')
            offload_module_weights(
                module, folder, offload, torch.device('cpu'), dirty=True)
            self.assertTrue(os.path.isfile(os.path.join(offload, 'weights.safetensors')))
            result = module(sample)
            self.assertTrue(torch.allclose(result, expected))

    def test_replacement_checkpoints_do_not_share_a_disk_cache(self):
        first = _Net()
        second = _Net()
        with torch.no_grad():
            second.block.linear.weight.add_(1)
            second.block.linear.bias.add_(1)
        sample = torch.ones(2, 4)
        first_expected = first(sample).detach().clone()
        second_expected = second(sample).detach().clone()
        with tempfile.TemporaryDirectory() as folder:
            def cache_dir(key):
                path = os.path.join(folder, 'cache', key)
                os.makedirs(path, exist_ok=True)
                return path

            with mock.patch.object(diskoffload, '_persistent_cache_dir', cache_dir):
                offload_module_weights(
                    first, None, os.path.join(folder, 'off-a'), torch.device('cpu'),
                    cache_identity=('repo', 'transformer-a', (), None))
                offload_module_weights(
                    second, None, os.path.join(folder, 'off-b'), torch.device('cpu'),
                    cache_identity=('repo', 'transformer-b', (), None))
                self.assertTrue(torch.allclose(second(sample), second_expected))
                self.assertFalse(torch.allclose(second(sample), first_expected))

                reused = _Net()
                with torch.no_grad():
                    reused.block.linear.weight.zero_()
                    reused.block.linear.bias.zero_()
                offload_module_weights(
                    reused, None, os.path.join(folder, 'off-c'), torch.device('cpu'),
                    cache_identity=('repo', 'transformer-a', (), None))
                self.assertTrue(torch.allclose(reused(sample), first_expected))

    def test_sharded_index_is_reused(self):
        module = torch.nn.Linear(4, 4)
        sample = torch.ones(1, 4)
        expected = module(sample).detach().clone()
        with tempfile.TemporaryDirectory() as folder:
            import json
            import safetensors.torch
            shard = 'diffusion_pytorch_model-00001-of-00001.safetensors'
            safetensors.torch.save_file(module.state_dict(), os.path.join(folder, shard))
            with open(os.path.join(folder, 'diffusion_pytorch_model.safetensors.index.json'),
                      'w', encoding='utf-8') as handle:
                json.dump({'weight_map': {key: shard for key in module.state_dict()}}, handle)
            offload = os.path.join(folder, 'offload')
            offload_module_weights(
                module, folder, offload, torch.device('cpu'), dirty=False)
            self.assertFalse(os.path.exists(offload))
            self.assertTrue(torch.allclose(module(sample), expected))
