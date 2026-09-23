import importlib
import inspect
import os
import unittest


class TestModernizationCompat(unittest.TestCase):

    def test_tqdm_hub_patch_accepts_tqdm_class(self):
        import huggingface_hub.file_download as file_download

        params = inspect.signature(file_download._get_progress_bar_context).parameters
        self.assertIn('tqdm_class', params)

    def test_t5_tokenizer_fast_resolves(self):
        from dgenerate.extras.compel.embeddings_provider import T5TokenizerFast
        from transformers import T5Tokenizer

        self.assertTrue(issubclass(T5TokenizerFast, T5Tokenizer) or T5TokenizerFast is T5Tokenizer)

    def test_hidiffusion_uses_packaging_version(self):
        import dgenerate.extras.hidiffusion.hidiffusion as hidiffusion

        self.assertFalse(hidiffusion.old_diffusers)
        self.assertIn('sd2-community/stable-diffusion-2-1-base', hidiffusion._sd15_strategy_models)
        self.assertIn('stabilityai/stable-diffusion-2-1-base', hidiffusion._sd15_strategy_models)

    def test_sada_imports_chunked_feed_forward(self):
        from diffusers.models.attention import _chunked_feed_forward as expected
        from dgenerate.extras.sada.module import _chunked_feed_forward

        self.assertIs(_chunked_feed_forward, expected)

    def test_ultraedit_uses_stable_diffusion_lora_mixin(self):
        from diffusers.loaders import StableDiffusionLoraLoaderMixin
        from dgenerate.extras.ultraedit.pipeline_stablediffusion_instruct_pix2pix import (
            StableDiffusionInstructPix2PixPipeline,
        )

        self.assertTrue(issubclass(StableDiffusionInstructPix2PixPipeline, StableDiffusionLoraLoaderMixin))

    def test_pipeline_class_supports_lora_matches_new_mixin(self):
        from diffusers import StableDiffusionPipeline
        from dgenerate.pipelinewrapper.pipelines import pipeline_class_supports_lora

        self.assertTrue(pipeline_class_supports_lora(StableDiffusionPipeline))

    def test_bnb_transformers_config_is_transformers_class(self):
        import diffusers
        import transformers
        from dgenerate.pipelinewrapper.uris.bnbquantizeruri import BNBQuantizerUri

        uri = BNBQuantizerUri.parse('bnb;bits=4;bits4-compute-dtype=float16')
        df_cfg = uri.to_config()
        tf_cfg = uri.to_transformers_config()
        self.assertIsInstance(df_cfg, diffusers.BitsAndBytesConfig)
        self.assertIsInstance(tf_cfg, transformers.BitsAndBytesConfig)
        self.assertTrue(tf_cfg.load_in_4bit)
        self.assertFalse(isinstance(df_cfg, transformers.BitsAndBytesConfig))

    def test_sdnq_is_not_vendored(self):
        extras_sdnq = os.path.join(
            os.path.dirname(importlib.import_module('dgenerate').__file__),
            'extras',
            'sdnq',
        )
        self.assertFalse(os.path.exists(extras_sdnq), extras_sdnq)
        import sdnq
        self.assertTrue(hasattr(sdnq, 'SDNQConfig'))

    def test_clip_tokenizer_ids_regression(self):
        from pathlib import Path
        from transformers import CLIPTokenizer

        tokenizer_dir = (
            Path(__file__).resolve().parents[2]
            / 'dgenerate'
            / 'pipelinewrapper'
            / 'hub_configs'
            / 'models--stable-diffusion-v1-5--stable-diffusion-v1-5'
            / 'tokenizer'
        )
        tokenizer = CLIPTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
        ids = tokenizer('an astronaut riding a horse', add_special_tokens=True).input_ids
        # transformers 5 CLIP ids for this prompt; fails if special-token handling drifts
        self.assertEqual(ids[0], tokenizer.bos_token_id)
        self.assertEqual(ids[-1], tokenizer.eos_token_id)
        self.assertEqual(ids, [49406, 550, 18376, 6765, 320, 4558, 49407])

    def test_hf_constants_reset_survives_transformers5(self):
        import huggingface_hub.constants as hf_constants
        import transformers.utils.hub as tf_hub
        from dgenerate import _reset_hf_constants

        previous = hf_constants.HF_HOME
        try:
            os.environ['HF_HOME'] = os.path.join(os.path.expanduser('~'), '.cache', 'dgenerate-hf-test-home')
            _reset_hf_constants()
            self.assertIn('dgenerate-hf-test-home', hf_constants.HF_HOME)
            if hasattr(tf_hub, 'HF_MODULES_CACHE'):
                self.assertTrue(tf_hub.HF_MODULES_CACHE)
        finally:
            if previous:
                os.environ['HF_HOME'] = previous
            _reset_hf_constants()

    def test_sd2_community_recipe_default(self):
        from pathlib import Path

        recipe = Path(__file__).resolve().parents[2] / 'dgenerate' / 'console' / 'recipes' / 'stable-diffusion-2.recipe'
        text = recipe.read_text(encoding='utf-8')
        self.assertIn('sd2-community/stable-diffusion-2-1', text)
        self.assertNotIn('stabilityai/stable-diffusion-2-1', text)

    def test_hub_configs_builder_maps_moved_sd2_repos(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / 'assetgen' / 'builders' / 'hf_configs_builder.py'
        spec = importlib.util.spec_from_file_location('hf_configs_builder', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        builder = module.HfConfigsBuilder()
        self.assertEqual(
            builder.moved_repositories['stabilityai/stable-diffusion-2-1'],
            'sd2-community/stable-diffusion-2-1',
        )


if __name__ == '__main__':
    unittest.main()
