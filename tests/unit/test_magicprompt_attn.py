import types
import unittest
from unittest import mock

from dgenerate.promptupscalers.magicpromptupscaler import (
    _attn_implementation_for_config,
    _legacy_rope_scaling_to_parameters,
    _try_native_causal_lm_config,
)


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


if __name__ == '__main__':
    unittest.main()
