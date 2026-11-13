import json
import os
import tempfile
import unittest
from unittest import mock

import dgenerate.pipelinewrapper.util as _util


def _touch(folder, name, size):
    path = os.path.join(folder, name)
    with open(path, 'wb') as handle:
        handle.write(b'0' * size)
    return size


class TestComponentFilesForEstimate(unittest.TestCase):
    def test_index_keeps_one_shard_layout(self):
        files = {}
        for index in range(1, 5):
            files[f'diffusion_pytorch_model-{index:05d}-of-00004.safetensors'] = 100
        for index in range(1, 9):
            files[f'diffusion_pytorch_model-{index:05d}-of-00008.safetensors'] = 50
        weight_map = {
            f'layer.{index}': f'diffusion_pytorch_model-{index:05d}-of-00004.safetensors'
            for index in range(1, 5)
        }
        selected = _util.component_files_for_estimate(files, weight_map)
        self.assertEqual(sum(selected.values()), 400)
        self.assertTrue(all(name.endswith('of-00004.safetensors') for name in selected))

    def test_index_ignores_duplicate_single_file(self):
        files = {
            'diffusion_pytorch_model-00001-of-00002.safetensors': 40,
            'diffusion_pytorch_model-00002-of-00002.safetensors': 10,
            'diffusion_pytorch_model.safetensors': 50,
        }
        weight_map = {
            'a': 'diffusion_pytorch_model-00001-of-00002.safetensors',
            'b': 'diffusion_pytorch_model-00002-of-00002.safetensors',
        }
        selected = _util.component_files_for_estimate(files, weight_map)
        self.assertEqual(sum(selected.values()), 50)
        self.assertNotIn('diffusion_pytorch_model.safetensors', selected)

    def test_missing_index_keeps_the_single_file(self):
        files = {
            'diffusion_pytorch_model-00001-of-00002.safetensors': 40,
            'diffusion_pytorch_model-00002-of-00002.safetensors': 10,
            'diffusion_pytorch_model.safetensors': 50,
        }
        selected = _util.component_files_for_estimate(files, None)
        self.assertEqual(list(selected), ['diffusion_pytorch_model.safetensors'])


class TestEstimateModelMemoryUse(unittest.TestCase):
    def test_directory_estimate_follows_the_weight_index(self):
        with tempfile.TemporaryDirectory() as root:
            folder = os.path.join(root, 'transformer')
            os.makedirs(folder)
            for index in range(1, 5):
                _touch(folder, f'diffusion_pytorch_model-{index:05d}-of-00004.safetensors', 100)
            for index in range(1, 9):
                _touch(folder, f'diffusion_pytorch_model-{index:05d}-of-00008.safetensors', 50)
            weight_map = {
                f'layer.{index}': f'diffusion_pytorch_model-{index:05d}-of-00004.safetensors'
                for index in range(1, 5)
            }
            with open(os.path.join(folder, 'diffusion_pytorch_model.safetensors.index.json'),
                      'w', encoding='utf-8') as handle:
                json.dump({'weight_map': weight_map}, handle)
            size = _util.estimate_model_memory_use(root, local_files_only=True)
        self.assertEqual(size, 400)

    def test_cached_subfolder_weight_is_counted_once(self):
        with tempfile.TemporaryDirectory() as root:
            folder = os.path.join(root, 'split_files', 'text_encoders')
            os.makedirs(folder)
            path = os.path.join(folder, 'gemma.safetensors')
            with open(path, 'wb') as handle:
                handle.write(b'0' * 128)

            def cached(repo_id, revision, filename):
                name = filename.replace('\\', '/')
                if name == 'split_files/text_encoders/gemma.safetensors':
                    return path
                return None

            with mock.patch.object(
                    _util, '_hf_try_to_load_from_cache', side_effect=cached):
                size = _util.estimate_model_memory_use(
                    'org/model',
                    subfolder='split_files/text_encoders',
                    weight_name='gemma.safetensors',
                    local_files_only=True)
        self.assertEqual(size, 128)

    def test_gguf_file_is_counted(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, 'model.gguf')
            with open(path, 'wb') as handle:
                handle.write(b'0' * 64)
            size = _util.estimate_model_memory_use(path, local_files_only=True)
        self.assertEqual(size, 64)

    def test_subfolder_file_is_counted_when_the_caller_has_no_subfolder(self):
        with mock.patch.object(
                _util, 'fetch_model_files_with_size',
                return_value=[('Diffusion Model/Full/model.gguf', 80)]):
            size = _util.estimate_model_memory_use(
                'https://huggingface.co/org/repo/blob/main/Diffusion%20Model/Full/model.gguf')
        self.assertEqual(size, 80)
