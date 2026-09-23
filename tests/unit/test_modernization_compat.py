import importlib
import inspect
import os
import unittest


class TestModernizationCompat(unittest.TestCase):

    def test_tqdm_hub_patch_accepts_tqdm_class(self):
        import huggingface_hub.file_download as file_download

        params = inspect.signature(file_download._get_progress_bar_context).parameters
        self.assertIn('tqdm_class', params)

    def test_compel_pads_without_tokenizer_max_length(self):
        from dgenerate.extras.compel.embeddings_provider import EmbeddingsProvider

        class Tokenizer:
            bos_token_id = 0
            eos_token_id = 1
            pad_token_id = 2
            model_max_length = 8

            def __call__(self, texts, truncation=None, padding='do_not_pad', return_tensors=None):
                self.padding = padding
                if padding == 'max_length':
                    # The failure mode: a 77-token CLIP window comes back as 78.
                    ids = [self.bos_token_id, 10, self.eos_token_id] + [self.pad_token_id] * 6
                else:
                    ids = [self.bos_token_id, 10, 11, self.eos_token_id]
                return {'input_ids': [ids]}

        class Encoder:
            device = 'cpu'

        tokenizer = Tokenizer()
        provider = EmbeddingsProvider(tokenizer=tokenizer, text_encoder=Encoder(), device='cpu')
        token_ids = provider.get_token_ids(['a cat'], padding='max_length')
        self.assertEqual(tokenizer.padding, 'do_not_pad')
        self.assertEqual(len(token_ids[0]), tokenizer.model_max_length)
        self.assertEqual(token_ids[0][0], tokenizer.bos_token_id)
        self.assertEqual(token_ids[0][3], tokenizer.eos_token_id)
        self.assertTrue(all(token == tokenizer.pad_token_id for token in token_ids[0][4:]))

    def test_compel_sentence_split_respects_combined_flags(self):
        from dgenerate.extras.compel.embeddings_provider import EmbeddingsProvider, SplitLongTextMode

        class Tokenizer:
            bos_token_id = 0
            eos_token_id = 1
            pad_token_id = 2
            model_max_length = 8

            def convert_ids_to_tokens(self, token_ids):
                names = {10: 'word</w>', 11: '.</w>', 12: 'more</w>', 13: ',</w>', 14: 'end</w>'}
                return [names[token] for token in token_ids]

        class Encoder:
            device = 'cpu'

        provider = EmbeddingsProvider(
            tokenizer=Tokenizer(),
            text_encoder=Encoder(),
            device='cpu',
            truncate=False,
            split_long_text_mode=SplitLongTextMode.SENTENCES | SplitLongTextMode.COPY_FIRST_CLS_TOKEN,
        )
        # Reverse scan must stop at the sentence marker, not the later comma.
        split_at = provider._find_next_best_split_point([10, 11, 12, 13, 14], max_length=4)
        self.assertEqual(split_at, 2)

    def test_compel_sequential_offload_keeps_embeddings_off_meta(self):
        import accelerate
        import torch
        from diffusers.models.modeling_utils import ModelMixin
        from torch import nn

        from dgenerate.extras.compel.embeddings_provider import text_encoder_device

        class Encoder(ModelMixin, nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 4)

            def forward(self, hidden):
                return self.lin(hidden)

        encoder = Encoder()
        accelerate.cpu_offload(encoder, execution_device=torch.device('cpu'))
        self.assertEqual(encoder.device.type, 'meta')

        target = text_encoder_device(encoder)
        self.assertNotEqual(target.type, 'meta')

        encoded = encoder(torch.ones(1, 4))
        placed = encoded.to(target, dtype=encoder.dtype)
        self.assertNotEqual(placed.device.type, 'meta')
        self.assertEqual(tuple(placed.shape), (1, 4))

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
