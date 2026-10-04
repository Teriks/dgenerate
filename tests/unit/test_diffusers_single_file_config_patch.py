import os
import unittest

import dgenerate  # noqa: F401  applies patches
import dgenerate._patches.diffusers_single_file_config_patch as _patch
import huggingface_hub


class TestDiffusersSingleFileConfigPatch(unittest.TestCase):
    def test_patch_is_installed(self):
        self.assertIs(huggingface_hub.hf_hub_download, _patch._patched_hf_hub_download)

    def test_moved_sd2_repos_are_remapped(self):
        self.assertEqual(
            _patch._resolve_repo_id('stabilityai/stable-diffusion-2-1'),
            'sd2-community/stable-diffusion-2-1',
        )
        self.assertEqual(
            _patch._resolve_repo_id('stabilityai/stable-diffusion-2-inpainting'),
            'sd2-community/stable-diffusion-2-inpainting',
        )
        self.assertEqual(
            _patch._resolve_repo_id('stable-diffusion-v1-5/stable-diffusion-v1-5'),
            'stable-diffusion-v1-5/stable-diffusion-v1-5',
        )

    def test_vendored_sd2_model_index_is_served_online(self):
        path = huggingface_hub.hf_hub_download(
            'stabilityai/stable-diffusion-2-1',
            filename='model_index.json',
            local_files_only=False,
        )
        self.assertTrue(os.path.isfile(path))
        self.assertIn(
            os.path.join('hub_configs', 'models--stabilityai--stable-diffusion-2-1'),
            path.replace('/', os.sep),
        )


if __name__ == '__main__':
    unittest.main()
