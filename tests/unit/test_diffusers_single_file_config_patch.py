import json
import os
import unittest

import dgenerate  # noqa: F401  applies patches
import dgenerate._patches.diffusers_single_file_config_patch as _patch
import dgenerate.pipelinewrapper.pipelines as _pipelines
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

    def test_single_file_model_index_stays_vendored(self):
        path = huggingface_hub.hf_hub_download(
            'stabilityai/stable-diffusion-x4-upscaler',
            filename='model_index.json',
            local_files_only=True,
        )
        self.assertIn(
            os.path.join(
                'hub_configs',
                'models--stabilityai--stable-diffusion-x4-upscaler',
                'model_index.json',
            ),
            path.replace('/', os.sep),
        )


class TestRepoPipelineVariant(unittest.TestCase):
    def _upscaler_index(self):
        path = os.path.join(
            os.path.dirname(dgenerate.pipelinewrapper.__file__),
            'hub_configs',
            'models--stabilityai--stable-diffusion-x4-upscaler',
            'model_index.json',
        )
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)

    def test_variant_is_omitted_when_weight_modules_are_loaded(self):
        index = self._upscaler_index()
        variant = _pipelines._repo_pipeline_variant(
            'fp16',
            index,
            {'text_encoder': object(), 'unet': object(), 'vae': object()},
        )
        self.assertIsNone(variant)

    def test_variant_is_kept_when_a_weight_module_is_still_loaded_by_diffusers(self):
        index = self._upscaler_index()
        variant = _pipelines._repo_pipeline_variant(
            'fp16',
            index,
            {'text_encoder': object(), 'vae': object()},
        )
        self.assertEqual(variant, 'fp16')


if __name__ == '__main__':
    unittest.main()
