import json
import os
import tempfile
import types
import unittest
import unittest.mock

import dgenerate  # noqa: F401  applies patches
import dgenerate._patches.diffusers_local_files_only_patch as _patch
import diffusers.models.modeling_utils as modeling_utils
import diffusers.utils.hub_utils as hub_utils
from huggingface_hub.utils import httpx as hub_httpx

_SHARDS = ['m-00001-of-00002.safetensors', 'm-00002-of-00002.safetensors']


class TestDiffusersLocalFilesOnlyPatch(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.index = os.path.join(self._dir.name, 'index.json')
        with open(self.index, 'w') as f:
            json.dump({'metadata': {}, 'weight_map': {'a': _SHARDS[0], 'b': _SHARDS[1]}}, f)

    def tearDown(self):
        self._dir.cleanup()

    def _online(self):
        siblings = [types.SimpleNamespace(rfilename=name) for name in _SHARDS]
        return unittest.mock.patch.multiple(
            hub_utils, HF_HUB_OFFLINE=False, HF_HUB_DISABLE_TELEMETRY=False), \
            unittest.mock.patch.object(
                _patch, 'model_info', return_value=types.SimpleNamespace(siblings=siblings))

    def test_patch_is_installed(self):
        self.assertIs(hub_utils._get_checkpoint_shard_files,
                      _patch._patched_get_checkpoint_shard_files)
        self.assertIs(modeling_utils._get_checkpoint_shard_files,
                      _patch._patched_get_checkpoint_shard_files)

    def test_the_active_hub_http_client_errors_are_caught(self):
        self.assertIn(hub_httpx.HTTPError, _patch._HTTP_ERRORS)

    def test_a_connection_error_becomes_an_environment_error(self):
        flags, info = self._online()
        error = hub_httpx.ConnectError('refused')
        with flags, info, unittest.mock.patch.object(_patch, 'snapshot_download', side_effect=error):
            with self.assertRaises(EnvironmentError) as raised:
                _patch._patched_get_checkpoint_shard_files(
                    'owner/repo', self.index, local_files_only=False)
        self.assertIn("couldn't connect", str(raised.exception))
        self.assertIs(raised.exception.__cause__, error)

    def test_local_files_only_skips_model_info(self):
        folder = self._dir.name
        for name in _SHARDS:
            open(os.path.join(folder, name), 'wb').close()
        with unittest.mock.patch.object(_patch, 'model_info') as model_info, \
                unittest.mock.patch.object(_patch, 'snapshot_download', return_value=folder) as download:
            files, metadata = _patch._patched_get_checkpoint_shard_files(
                'owner/repo', self.index, local_files_only=True)
        model_info.assert_not_called()
        self.assertTrue(download.call_args.kwargs['local_files_only'])
        self.assertEqual([os.path.normpath(f) for f in files],
                         [os.path.join(folder, name) for name in _SHARDS])
        self.assertEqual(metadata['all_checkpoint_keys'], ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
