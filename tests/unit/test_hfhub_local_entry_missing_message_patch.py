import tempfile
import unittest

import dgenerate  # noqa: F401  applies patches
import dgenerate._patches.hfhub_local_entry_missing_message_patch as _patch
import huggingface_hub
import huggingface_hub.errors
import huggingface_hub.file_download


class TestHfHubLocalEntryMissingMessagePatch(unittest.TestCase):
    def test_patch_is_installed(self):
        self.assertIs(huggingface_hub.file_download._hf_hub_download_to_cache_dir,
                      _patch._hf_hub_download_to_cache_dir)
        self.assertIs(huggingface_hub.file_download._hf_hub_download_to_local_dir,
                      _patch._hf_hub_download_to_local_dir)

    def test_offline_cache_miss_names_the_file_and_repo(self):
        with tempfile.TemporaryDirectory() as cache:
            with self.assertRaises(huggingface_hub.errors.LocalEntryNotFoundError) as raised:
                huggingface_hub.hf_hub_download(
                    'owner/repo', 'weights.safetensors', cache_dir=cache, local_files_only=True)
        message = str(raised.exception)
        self.assertIn('Offline mode is enabled', message)
        self.assertIn('"weights.safetensors"', message)
        self.assertIn('"owner/repo"', message)

    def test_offline_local_dir_miss_names_the_file_and_repo(self):
        with tempfile.TemporaryDirectory() as local:
            with self.assertRaises(huggingface_hub.errors.LocalEntryNotFoundError) as raised:
                huggingface_hub.hf_hub_download(
                    'owner/repo', 'weights.safetensors', local_dir=local, local_files_only=True)
        self.assertIn('"weights.safetensors"', str(raised.exception))


if __name__ == '__main__':
    unittest.main()
