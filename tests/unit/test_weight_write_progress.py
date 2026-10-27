import json
import os
import tempfile
import unittest

import safetensors.torch
import torch

from dgenerate.pipelinewrapper.weightsave import (
    save_module_with_progress,
    save_pipeline_with_progress,
    save_state_dict_with_progress,
)


class _Saveable(torch.nn.Linear):
    def save_pretrained(self, directory, safe_serialization=True, max_shard_size=None, **kwargs):
        os.makedirs(directory, exist_ok=True)
        safetensors.torch.save_file(
            self.state_dict(), os.path.join(directory, 'model.safetensors'))


class TestWeightWriteProgress(unittest.TestCase):
    def test_module_save_restores_safetensors_writer(self):
        original = safetensors.torch.save_file
        module = _Saveable(4, 4)
        with tempfile.TemporaryDirectory() as folder:
            save_module_with_progress(module, folder, desc='Writing linear')
            path = os.path.join(folder, 'model.safetensors')
            self.assertTrue(os.path.isfile(path))
            loaded = safetensors.torch.load_file(path)
        self.assertEqual(set(loaded), set(module.state_dict()))
        self.assertIs(safetensors.torch.save_file, original)

    def test_pipeline_save_uses_the_weight_bar(self):
        original = safetensors.torch.save_file

        class Pipe:
            def __init__(self):
                self.components = {'unet': _Saveable(2, 2)}

            def save_pretrained(self, directory, safe_serialization=True,
                                max_shard_size=None, variant=None):
                self.components['unet'].save_pretrained(directory)

        with tempfile.TemporaryDirectory() as folder:
            save_pipeline_with_progress(Pipe(), folder, desc='Writing pipeline')
            self.assertTrue(os.path.isfile(os.path.join(folder, 'model.safetensors')))
        self.assertIs(safetensors.torch.save_file, original)

    def test_sharded_state_dict_records_every_tensor(self):
        state = {
            'a': torch.zeros(8, dtype=torch.float32),
            'b': torch.ones(8, dtype=torch.float32),
        }
        with tempfile.TemporaryDirectory() as folder:
            index = save_state_dict_with_progress(
                state, folder, desc='Writing shards', max_shard_bytes=16)
            self.assertEqual(set(index), {'a', 'b'})
            self.assertTrue(os.path.isfile(
                os.path.join(folder, 'weights.safetensors.index.json')))
            with open(os.path.join(folder, 'weights.safetensors.index.json'),
                      encoding='utf-8') as handle:
                weight_map = json.load(handle)['weight_map']
        self.assertEqual(set(weight_map), {'a', 'b'})
        self.assertNotEqual(weight_map['a'], weight_map['b'])
