import contextlib
import types
import unittest
from unittest import mock

import torch

from dgenerate.promptupscalers.magicpromptupscaler import (
    MagicPromptUpscaler,
    _TextGenerationPipeline,
    _attn_implementation_for_config,
    _legacy_rope_scaling_to_parameters,
    _sampling_generate_kwargs,
    _try_native_causal_lm_config,
)
import dgenerate.pipelinewrapper.promptenhance as _promptenhance


class TestMagicPromptAttn(unittest.TestCase):
    def test_sliding_window_uses_eager(self):
        config = types.SimpleNamespace(sliding_window=262144)
        self.assertEqual(_attn_implementation_for_config(config), 'eager')

    def test_no_sliding_window_keeps_default(self):
        config = types.SimpleNamespace()
        self.assertIsNone(_attn_implementation_for_config(config))
        config.sliding_window = None
        self.assertIsNone(_attn_implementation_for_config(config))


class TestMagicPromptPhi3NativeConfig(unittest.TestCase):
    def test_su_rope_scaling_converts_to_longrope_parameters(self):
        params = _legacy_rope_scaling_to_parameters({
            'max_position_embeddings': 131072,
            'original_max_position_embeddings': 4096,
            'rope_theta': 10000.0,
            'rope_scaling': {
                'type': 'su',
                'short_factor': [1.05, 1.1],
                'long_factor': [2.0, 3.0],
                'original_max_position_embeddings': 4096,
            },
        })
        self.assertIsNotNone(params)
        self.assertEqual(params['rope_type'], 'longrope')
        self.assertEqual(params['original_max_position_embeddings'], 4096)
        self.assertEqual(params['factor'], 32.0)
        self.assertEqual(params['short_factor'], [1.05, 1.1])
        self.assertEqual(params['long_factor'], [2.0, 3.0])
        self.assertEqual(params['rope_theta'], 10000.0)

    def test_existing_rope_parameters_skipped(self):
        self.assertIsNone(_legacy_rope_scaling_to_parameters({
            'rope_parameters': {'rope_type': 'default'},
            'rope_scaling': {'type': 'su'},
        }))

    def test_try_native_builds_phi3_without_auto_map(self):
        hub_dict = {
            'model_type': 'phi3',
            'architectures': ['Phi3ForCausalLM'],
            'vocab_size': 32000,
            'hidden_size': 64,
            'intermediate_size': 128,
            'num_hidden_layers': 1,
            'num_attention_heads': 4,
            'num_key_value_heads': 4,
            'max_position_embeddings': 256,
            'original_max_position_embeddings': 64,
            'sliding_window': 512,
            'rope_theta': 10000.0,
            'torch_dtype': 'float16',
            'auto_map': {
                'AutoConfig': 'microsoft/Phi-3-mini-128k-instruct--configuration_phi3.Phi3Config',
                'AutoModelForCausalLM':
                    'microsoft/Phi-3-mini-128k-instruct--modeling_phi3.Phi3ForCausalLM',
            },
            'rope_scaling': {
                'type': 'su',
                'short_factor': [1.05] * 8,
                'long_factor': [1.1] * 8,
                'original_max_position_embeddings': 64,
            },
            'bos_token_id': 1,
            'eos_token_id': 0,
            'pad_token_id': 0,
        }

        with mock.patch(
                'transformers.PretrainedConfig.get_config_dict',
                return_value=(hub_dict, None),
        ):
            config = _try_native_causal_lm_config('fake/phi3', local_files_only=True)

        self.assertIsNotNone(config)
        self.assertEqual(type(config).__name__, 'Phi3Config')
        self.assertEqual(config.rope_parameters['rope_type'], 'longrope')
        self.assertEqual(config.rope_parameters['factor'], 4.0)
        self.assertIsNone(getattr(config, 'auto_map', None))

    def test_try_native_ignores_non_phi3(self):
        with mock.patch(
                'transformers.PretrainedConfig.get_config_dict',
                return_value=({'model_type': 'gpt2'}, None),
        ):
            self.assertIsNone(
                _try_native_causal_lm_config('fake/gpt2', local_files_only=True))


class TestMagicPromptLayerSplit(unittest.TestCase):
    def test_gpt2_blocks_are_split_units(self):
        class GPT2Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.wte = torch.nn.Embedding(4, 2)
                self.h = torch.nn.ModuleList([GPT2Block(), GPT2Block()])
                self.lm_head = torch.nn.Linear(2, 4)

        embeddings, layers, _norms, heads = _promptenhance._placement_units(Model())
        self.assertEqual([name for name, _module in layers], ['h.0', 'h.1'])
        self.assertEqual([name for name, _module in embeddings], ['wte'])
        self.assertEqual([name for name, _module in heads], ['lm_head'])

    def test_unquantized_upscale_uses_the_layer_split(self):
        model = object()
        seen = {}

        @contextlib.contextmanager
        def fast_encoder(encoder, device, offload, **kwargs):
            seen['encoder'] = encoder
            seen['device'] = device
            seen['offload'] = offload
            seen['label'] = kwargs.get('label')
            seen['debug'] = kwargs.get('debug')
            seen['headroom_bytes'] = kwargs.get('headroom_bytes')
            yield

        upscaler = MagicPromptUpscaler.__new__(MagicPromptUpscaler)
        upscaler._quantizer = None
        upscaler._gpu_headroom = 1.5
        upscaler._pipeline = types.SimpleNamespace(model=model)
        upscaler._PromptUpscaler__device = 'cuda'
        upscaler._PromptUpscaler__size_estimate = 1
        upscaler.memory_guard_device = lambda *args, **kwargs: None

        with mock.patch.object(_promptenhance, '_fast_encoder', fast_encoder):
            with upscaler._with_device():
                seen['inside'] = True

        self.assertIs(seen['encoder'], model)
        self.assertEqual(seen['device'], 'cuda')
        self.assertTrue(seen['offload'])
        self.assertEqual(seen['label'], 'magicprompt')
        self.assertTrue(seen['debug'])
        self.assertEqual(seen['headroom_bytes'], int(1.5 * 1024 ** 3))
        self.assertTrue(seen['inside'])

    def test_quantized_upscale_stays_on_its_device_map(self):
        upscaler = MagicPromptUpscaler.__new__(MagicPromptUpscaler)
        upscaler._quantizer = 'bnb'

        def fail_split(*args, **kwargs):
            raise AssertionError('quantized magicprompt should not layer-split')

        with mock.patch.object(_promptenhance, '_fast_encoder', fail_split):
            with upscaler._with_device():
                pass


class TestMagicPromptPresencePenalty(unittest.TestCase):
    def test_sampling_kwargs_apply_the_penalty_once_per_new_token(self):
        quiet = _sampling_generate_kwargs(4, 0.0, 50, 1.0, 0.0, 16)
        self.assertFalse(quiet['do_sample'])
        self.assertNotIn('logits_processor', quiet)
        self.assertNotIn('temperature', quiet)

        sampled = _sampling_generate_kwargs(4, 0.7, 20, 0.8, 1.5, 16)
        self.assertTrue(sampled['do_sample'])
        self.assertEqual(sampled['top_k'], 20)
        processor = sampled['logits_processor'][0]
        scores = torch.zeros(1, 6)
        changed = processor(torch.tensor([[1, 1, 1, 1, 3, 3]]), scores)
        self.assertEqual(changed[0, 3].item(), -1.5)
        self.assertEqual(changed[0, 1].item(), 0.0)

    def test_text_pipeline_does_not_forward_presence_penalty(self):
        class Encoded(dict):
            def to(self, device):
                return self

        class Tokenizer:
            def __call__(self, batch, return_tensors='pt', padding=True,
                         truncation=True, max_length=100):
                return Encoded(input_ids=torch.tensor([[1, 2, 3]]))

            def decode(self, output):
                return 'ok'

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return kwargs['input_ids']

        model = Model()
        pipeline = _TextGenerationPipeline(model, Tokenizer(), quantized=True)
        pipeline(
            ['fox', 'horse'],
            max_length=20,
            temperature=0.7,
            top_k=20,
            top_p=0.8,
            presence_penalty=1.5,
            batch_size=2)
        call = model.calls[0]
        self.assertNotIn('presence_penalty', call)
        self.assertEqual(call['logits_processor'][0].prompt_length, 3)
        self.assertEqual(call['temperature'], 0.7)
        self.assertTrue(call['do_sample'])


if __name__ == '__main__':
    unittest.main()
