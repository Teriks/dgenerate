import json
import os
import tempfile
import unittest
import unittest.mock

import gguf
import numpy as np
import torch
import transformers

import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.ggufcompat as _ggufcompat
import dgenerate.pipelinewrapper.safetensorscompat as _safetensorscompat
from dgenerate.pipelinewrapper.uris import textencoderuri
from dgenerate.pipelinewrapper.uris.exceptions import (
    InvalidTextEncoderUriError,
    TextEncoderUriLoadError,
)


def _write_gguf(path, architecture, tensors):
    """llama.cpp-style GGUF. 2D weights other than embeddings are Q8_0."""
    writer = gguf.GGUFWriter(path, architecture)
    quant = gguf.GGMLQuantizationType.Q8_0
    for name, value in tensors.items():
        array = np.ascontiguousarray(value.detach().float().cpu().numpy())
        embedding = 'embed' in name or name.startswith('token_embd') or name.startswith('shared')
        if array.ndim == 2 and not embedding and array.shape[-1] % 32 == 0:
            packed = gguf.quantize(array, quant)
            writer.add_tensor(name, packed, raw_dtype=quant)
        else:
            writer.add_tensor(name, array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _replace(key, pairs):
    for source, target in pairs:
        key = key.replace(source, target)
    return key


_T5_TO_GGUF = (
    ('layer.0.SelfAttention.relative_attention_bias', 'attn_rel_b'),
    ('layer.0.SelfAttention.q', 'attn_q'),
    ('layer.0.SelfAttention.k', 'attn_k'),
    ('layer.0.SelfAttention.v', 'attn_v'),
    ('layer.0.SelfAttention.o', 'attn_o'),
    ('layer.0.layer_norm', 'attn_norm'),
    ('layer.1.DenseReluDense.wi_0', 'ffn_gate'),
    ('layer.1.DenseReluDense.wi_1', 'ffn_up'),
    ('layer.1.DenseReluDense.wo', 'ffn_down'),
    ('layer.1.layer_norm', 'ffn_norm'),
    ('final_layer_norm', 'output_norm'),
    ('.block.', '.blk.'),
    ('encoder.', 'enc.'),
    ('shared', 'token_embd'),
)

_QWEN_TO_GGUF = (
    ('self_attn.q_norm.', 'attn_q_norm.'),
    ('self_attn.k_norm.', 'attn_k_norm.'),
    ('self_attn.v_norm.', 'attn_v_norm.'),
    ('self_attn.q_proj', 'attn_q'),
    ('self_attn.k_proj', 'attn_k'),
    ('self_attn.v_proj', 'attn_v'),
    ('self_attn.o_proj', 'attn_output'),
    ('mlp.up_proj', 'ffn_up'),
    ('mlp.down_proj', 'ffn_down'),
    ('mlp.gate_proj', 'ffn_gate'),
    ('input_layernorm', 'attn_norm'),
    ('post_attention_layernorm', 'ffn_norm'),
    ('model.layers.', 'blk.'),
    ('model.embed_tokens', 'token_embd'),
    ('model.norm', 'output_norm'),
    ('lm_head.weight', 'output.weight'),
)


def _config_file(config):
    handle, path = tempfile.mkstemp(suffix='.json')
    os.close(handle)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(config.to_dict(), file)
    return path


def _comfy_fp8(state):
    tensors = {}
    for key, value in state.items():
        value = value.detach().float().clone()
        if (key.endswith('.weight') and value.ndim == 2 and 'embed' not in key
                and 'shared' not in key and 'lm_head' not in key):
            scale = value.abs().amax() / 448
            tensors[key] = (value / scale).to(torch.float8_e4m3fn)
            tensors[key[:-len('weight')] + 'scale_weight'] = scale.reshape(())
        else:
            tensors[key] = value
    tensors['scaled_fp8'] = torch.empty(0, dtype=torch.float8_e4m3fn)
    return tensors


class TextEncoderFileTest(unittest.TestCase):
    def setUp(self):
        self._files = []

    def tearDown(self):
        for path in self._files:
            if os.path.exists(path):
                os.remove(path)

    def _temp(self, suffix):
        handle, path = tempfile.mkstemp(suffix=suffix)
        os.close(handle)
        self._files.append(path)
        return path

    def test_t5_gguf_names_match_city96(self):
        state = {
            'enc.blk.0.attn_q.weight': torch.zeros(4, 4),
            'enc.blk.0.ffn_gate.weight': torch.zeros(8, 4),
            'enc.blk.0.ffn_down.weight': torch.zeros(4, 8),
            'enc.blk.0.attn_rel_b.weight': torch.zeros(8, 4),
            'enc.output_norm.weight': torch.zeros(4),
            'token_embd.weight': torch.zeros(16, 4),
        }
        remapped = _ggufcompat._remap_keys(state, _ggufcompat._T5_GGUF_KEYS)
        self.assertEqual(set(remapped), {
            'encoder.block.0.layer.0.SelfAttention.q.weight',
            'encoder.block.0.layer.1.DenseReluDense.wi_0.weight',
            'encoder.block.0.layer.1.DenseReluDense.wo.weight',
            'encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight',
            'encoder.final_layer_norm.weight',
            'shared.weight',
        })

    def test_shipped_configs_match_published_encoders(self):
        t5, other = textencoderuri._lookup_shipped_config(
            transformers.T5EncoderModel, 4096, 32128, 24)
        self.assertIsNotNone(t5)
        self.assertIsNone(other)
        umt5, _ = textencoderuri._lookup_shipped_config(
            transformers.UMT5EncoderModel, 4096, 256384, 24)
        self.assertIn('Wan', umt5)
        gemma, _ = textencoderuri._lookup_shipped_config(
            transformers.Gemma3ForConditionalGeneration, 3840, 262208, 48)
        self.assertIn('LTX-2', gemma)
        qwen, _ = textencoderuri._lookup_shipped_config(
            transformers.Qwen2_5_VLForConditionalGeneration, 3584, 152064, 28)
        self.assertIn('Qwen-Image', qwen)
        krea, _ = textencoderuri._lookup_shipped_config(
            transformers.Qwen3VLModel, 2560, 151936, 36)
        self.assertIn('Krea-2', krea)
        qwen21, _ = textencoderuri._lookup_shipped_config(
            transformers.Qwen3VLForConditionalGeneration, 4096, 151936, 36)
        self.assertIn('Qwen-Image-2.1', qwen21)
        klein9, _ = textencoderuri._lookup_shipped_config(
            transformers.Qwen3ForCausalLM, 4096, 151936, 36)
        self.assertIn('klein', klein9.lower())
        hunyuan, _ = textencoderuri._lookup_shipped_config(
            transformers.Qwen2_5_VLTextModel, 3584, 152064, 28)
        self.assertIn('HunyuanVideo', hunyuan)
        with self.assertRaises(TextEncoderUriLoadError) as caught:
            textencoderuri._require_shipped_config(
                transformers.T5EncoderModel, (256384, 4096), 24, 'umt5.gguf')
        self.assertIn('UMT5EncoderModel', str(caught.exception))

    def test_llama_row_permute_keeps_the_quantized_bytes(self):
        torch.manual_seed(0)
        weight = torch.randn(64, 96)
        packed = gguf.quantize(weight.numpy(), gguf.GGMLQuantizationType.Q8_0)
        from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor
        parameter = GGUFParameter(torch.from_numpy(packed), quant_type=gguf.GGMLQuantizationType.Q8_0)
        state = _ggufcompat.permute_llama_attention(
            {'model.layers.0.self_attn.q_proj.weight': parameter}, 4, 4)
        dense = dequantize_gguf_tensor(parameter).float()
        permuted = dense.reshape(4, 8, 2, 96).swapaxes(1, 2).reshape(64, 96)
        decoded = dequantize_gguf_tensor(state['model.layers.0.self_attn.q_proj.weight']).float()
        self.assertTrue(torch.allclose(decoded, permuted))

    def test_gemma3_norm_correction(self):
        corrected = _ggufcompat.correct_gemma3_norms({
            'model.layers.0.input_layernorm.weight': torch.ones(4),
            'model.layers.0.self_attn.q_proj.weight': torch.ones(4, 4),
        })
        self.assertTrue(torch.equal(
            corrected['model.layers.0.input_layernorm.weight'], torch.zeros(4)))
        self.assertTrue(torch.equal(
            corrected['model.layers.0.self_attn.q_proj.weight'], torch.ones(4, 4)))

    def test_gguf_t5_matches_full_precision(self):
        torch.manual_seed(0)
        config = transformers.T5Config(
            vocab_size=128, d_model=64, d_kv=16, d_ff=128, num_layers=2, num_heads=4,
            feed_forward_proj='gated-gelu', tie_word_embeddings=False)
        reference = transformers.T5EncoderModel(config).eval()
        path = self._temp('.gguf')
        config_path = _config_file(config)
        self._files.append(config_path)
        _write_gguf(path, 't5encoder', {
            _replace(key, _T5_TO_GGUF): value
            for key, value in reference.state_dict().items()})

        with unittest.mock.patch.object(
                textencoderuri, '_require_shipped_config', return_value=config_path):
            loaded = textencoderuri.TextEncoderUri.parse(
                f'T5EncoderModel;model={path}').load(
                dtype_fallback=_enums.DataType.FLOAT32, no_cache=True)

        layer = loaded.encoder.block[0].layer[1].DenseReluDense
        self.assertEqual(type(layer.wi_0).__name__, 'GGUFLinear')
        self.assertNotEqual(type(layer.wo).__name__, 'GGUFLinear')
        self.assertEqual(layer.wo.weight.dtype, torch.float32)
        tokens = torch.randint(0, 128, (1, 8))
        with torch.no_grad():
            expected = reference(tokens).last_hidden_state
            output = loaded(tokens).last_hidden_state
        self.assertLess(((output - expected).norm() / expected.norm()).item(), 0.08)

    def test_gguf_qwen_matches_full_precision(self):
        torch.manual_seed(0)
        config = transformers.Qwen3Config(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=False)
        reference = transformers.Qwen3ForCausalLM(config).eval()
        path = self._temp('.gguf')
        config_path = _config_file(config)
        self._files.append(config_path)
        _write_gguf(path, 'qwen3', {
            _replace(key, _QWEN_TO_GGUF): value for key, value in reference.state_dict().items()})

        with unittest.mock.patch.object(
                textencoderuri, '_require_shipped_config', return_value=config_path):
            loaded = textencoderuri.TextEncoderUri.parse(
                f'Qwen3ForCausalLM;model={path}').load(
                dtype_fallback=_enums.DataType.FLOAT32, no_cache=True)

        packed = [module for module in loaded.modules()
                  if type(module).__name__ == 'GGUFLinear']
        self.assertGreater(len(packed), 0)
        tokens = torch.randint(0, 128, (1, 8))
        with torch.no_grad():
            expected = reference(tokens, output_hidden_states=True).hidden_states[-1]
            output = loaded(tokens, output_hidden_states=True).hidden_states[-1]
        self.assertLess(((output - expected).norm() / expected.norm()).item(), 0.08)

    def test_gguf_llama_restores_permuted_attention(self):
        # llama.cpp interleaves query and key heads. Architecture "llama" is Mistral.
        torch.manual_seed(0)
        config = transformers.Qwen3Config(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=False)
        reference = transformers.Qwen3ForCausalLM(config).eval()
        path = self._temp('.gguf')
        config_path = _config_file(config)
        self._files.append(config_path)

        def llama_permute(value, heads):
            # llama.cpp groups pairs before the head split. The loader reverses it.
            return value.reshape(
                heads, 2, value.shape[0] // heads // 2, *value.shape[1:]).swapaxes(1, 2).reshape(value.shape).contiguous()

        tensors = {}
        for key, value in reference.state_dict().items():
            if key.endswith(('q_proj.weight', 'q_proj.bias')):
                value = llama_permute(value, 4)
            elif key.endswith(('k_proj.weight', 'k_proj.bias')):
                value = llama_permute(value, 2)
            tensors[_replace(key, _QWEN_TO_GGUF)] = value
        _write_gguf(path, 'llama', tensors)

        with unittest.mock.patch.object(
                textencoderuri, '_require_shipped_config', return_value=config_path):
            loaded = textencoderuri.TextEncoderUri.parse(
                f'Qwen3ForCausalLM;model={path}').load(
                dtype_fallback=_enums.DataType.FLOAT32, no_cache=True)
        tokens = torch.randint(0, 128, (1, 8))
        with torch.no_grad():
            expected = reference(tokens, output_hidden_states=True).hidden_states[-1]
            output = loaded(tokens, output_hidden_states=True).hidden_states[-1]
        self.assertLess(((output - expected).norm() / expected.norm()).item(), 0.08)

    def test_gguf_rejects_a_quantizer(self):
        path = self._temp('.gguf')
        with self.assertRaises(InvalidTextEncoderUriError):
            textencoderuri.TextEncoderUri.parse(
                f'T5EncoderModel;model={path};quantizer=bnb')

    def test_comfy_umt5_safetensors_stay_packed(self):
        import safetensors.torch

        torch.manual_seed(0)
        config = transformers.UMT5Config(
            vocab_size=128, d_model=64, d_kv=16, d_ff=128, num_layers=2, num_heads=4,
            feed_forward_proj='gated-gelu', tie_word_embeddings=False)
        reference = transformers.UMT5EncoderModel(config).eval()
        path = self._temp('.safetensors')
        config_path = _config_file(config)
        self._files.append(config_path)
        safetensors.torch.save_file(_comfy_fp8(reference.state_dict()), path)

        with unittest.mock.patch.object(
                textencoderuri, '_require_shipped_config', return_value=config_path):
            loaded = textencoderuri.TextEncoderUri.parse(
                f'UMT5EncoderModel;model={path}').load(
                dtype_fallback=_enums.DataType.FLOAT32, no_cache=True)

        layer = loaded.encoder.block[0].layer[1].DenseReluDense
        self.assertIsInstance(layer.wi_0, _safetensorscompat.ComfyQuantLinear)
        self.assertNotIsInstance(layer.wo, _safetensorscompat.ComfyQuantLinear)
        self.assertEqual(layer.wo.weight.dtype, torch.float32)
        tokens = torch.randint(0, 128, (1, 8))
        with torch.no_grad():
            expected = reference(tokens).last_hidden_state
            output = loaded(tokens).last_hidden_state
        self.assertLess(((output - expected).norm() / expected.norm()).item(), 0.08)

    def test_comfy_t5_safetensors_stay_packed(self):
        import safetensors.torch

        torch.manual_seed(0)
        config = transformers.T5Config(
            vocab_size=128, d_model=64, d_kv=16, d_ff=128, num_layers=2, num_heads=4,
            feed_forward_proj='gated-gelu', tie_word_embeddings=False)
        reference = transformers.T5EncoderModel(config).eval()
        path = self._temp('.safetensors')
        config_path = _config_file(config)
        self._files.append(config_path)
        safetensors.torch.save_file(_comfy_fp8(reference.state_dict()), path)

        with unittest.mock.patch.object(
                textencoderuri, '_require_shipped_config', return_value=config_path):
            loaded = textencoderuri.TextEncoderUri.parse(
                f'T5EncoderModel;model={path}').load(
                dtype_fallback=_enums.DataType.FLOAT32, no_cache=True)

        layer = loaded.encoder.block[0].layer[1].DenseReluDense
        self.assertIsInstance(layer.wi_0, _safetensorscompat.ComfyQuantLinear)
        self.assertNotIsInstance(layer.wo, _safetensorscompat.ComfyQuantLinear)
        self.assertEqual(layer.wo.weight.dtype, torch.float32)
        tokens = torch.randint(0, 128, (1, 8))
        with torch.no_grad():
            expected = reference(tokens).last_hidden_state
            output = loaded(tokens).last_hidden_state
        self.assertLess(((output - expected).norm() / expected.norm()).item(), 0.08)

    def test_gemma3_comfy_names_resolve(self):
        model_keys = {
            'model.language_model.layers.0.mlp.up_proj.weight',
            'model.language_model.embed_tokens.weight',
            'model.vision_tower.encoder.layers.0.self_attn.q_proj.weight',
            'model.multi_modal_projector.mm_input_projection_weight',
        }
        rename = textencoderuri._language_model_rename(model_keys, [
            'model.layers.0.mlp.up_proj.weight',
            'model.embed_tokens.weight',
            'vision_model.encoder.layers.0.self_attn.q_proj.weight',
            'multi_modal_projector.mm_input_projection_weight',
        ])
        self.assertEqual(
            rename('vision_model.encoder.layers.0.self_attn.q_proj.weight'),
            'model.vision_tower.encoder.layers.0.self_attn.q_proj.weight')
        self.assertEqual(
            rename('model.layers.0.mlp.up_proj.weight'),
            'model.language_model.layers.0.mlp.up_proj.weight')
        self.assertEqual(
            rename('multi_modal_projector.mm_input_projection_weight'),
            'model.multi_modal_projector.mm_input_projection_weight')

    def test_qwen3_vl_comfy_names_resolve_for_both_wrappers(self):
        # Krea 2's Qwen3VLModel keeps the language model beside the vision tower.
        base_keys = {
            'language_model.layers.0.self_attn.q_proj.weight',
            'language_model.embed_tokens.weight',
            'visual.blocks.0.attn.qkv.weight',
        }
        rename = textencoderuri._language_model_rename(base_keys, [
            'model.layers.0.self_attn.q_proj.weight',
            'model.embed_tokens.weight',
            'model.visual.blocks.0.attn.qkv.weight',
        ])
        self.assertEqual(rename('model.layers.0.self_attn.q_proj.weight'),
                         'language_model.layers.0.self_attn.q_proj.weight')
        self.assertEqual(rename('model.visual.blocks.0.attn.qkv.weight'),
                         'visual.blocks.0.attn.qkv.weight')
        # Qwen-Image 2.1 wraps the same checkpoint under model.language_model.
        wrapped_keys = {
            'model.language_model.layers.0.self_attn.q_proj.weight',
            'model.language_model.embed_tokens.weight',
            'model.visual.blocks.0.attn.qkv.weight',
        }
        rename = textencoderuri._language_model_rename(wrapped_keys, [
            'model.layers.0.self_attn.q_proj.weight',
            'model.embed_tokens.weight',
            'model.visual.blocks.0.attn.qkv.weight',
        ])
        self.assertEqual(rename('model.layers.0.self_attn.q_proj.weight'),
                         'model.language_model.layers.0.self_attn.q_proj.weight')
        self.assertEqual(rename('model.visual.blocks.0.attn.qkv.weight'),
                         'model.visual.blocks.0.attn.qkv.weight')

    def test_text_tower_ignores_a_bundled_vision_tower(self):
        # HunyuanVideo 1.5 reads Qwen2_5_VLTextModel. ComfyUI's file also
        # stores the vision tower, which is most of the tensors.
        model_keys = {
            'layers.0.self_attn.q_proj.weight',
            'embed_tokens.weight',
            'norm.weight',
        }
        checkpoint = [
            'model.layers.0.self_attn.q_proj.weight',
            'model.embed_tokens.weight',
            'model.norm.weight',
        ] + [f'visual.blocks.{index}.attn.qkv.weight' for index in range(10)]
        rename = textencoderuri._language_model_rename(model_keys, checkpoint)
        self.assertEqual(rename('model.layers.0.self_attn.q_proj.weight'),
                         'layers.0.self_attn.q_proj.weight')
        self.assertEqual(rename('visual.blocks.0.attn.qkv.weight'),
                         'visual.blocks.0.attn.qkv.weight')


if __name__ == '__main__':
    unittest.main()
