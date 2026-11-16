import json
import os
import tempfile
import unittest

import gguf
import torch

import dgenerate.quantize.comfykeys as comfykeys
import dgenerate.pipelinewrapper.ggufcompat as ggufcompat
import dgenerate.quantize.quantexport as quantexport
import dgenerate.subcommands.exceptions as subcommand_exceptions
import dgenerate.subcommands.quantize as quantize_command


def _filled(shape):
    return torch.arange(int(torch.tensor(shape).prod().item()), dtype=torch.float32).reshape(shape)


def _clone(state):
    return {key: value.clone() for key, value in state.items()}


def _same(test, actual, expected):
    test.assertEqual(set(actual), set(expected))
    for key, value in expected.items():
        test.assertTrue(torch.equal(actual[key], value), key)


class TestComfyKeys(unittest.TestCase):
    def test_flux_round_trip(self):
        from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers
        state = _flux_comfy_state()
        converted = convert_flux_transformer_checkpoint_to_diffusers(_clone(state))
        restored, label = comfykeys.comfy_state(
            converted, {'_class_name': 'FluxTransformer2DModel'}, gguf=True)
        self.assertEqual(label, 'flux')
        _same(self, restored, state)

    def test_sd3_round_trip(self):
        from diffusers.loaders.single_file_utils import convert_sd3_transformer_checkpoint_to_diffusers
        state = _sd3_comfy_state()
        converted = convert_sd3_transformer_checkpoint_to_diffusers(_clone(state))
        restored, label = comfykeys.comfy_state(
            converted, {'_class_name': 'SD3Transformer2DModel'}, gguf=True)
        self.assertEqual(label, 'sd3')
        _same(self, restored, state)

    def test_wan_round_trip(self):
        from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers
        state = {
            'time_embedding.0.weight': _filled((4, 4)),
            'blocks.0.self_attn.q.weight': _filled((4, 4)),
            'blocks.0.self_attn.k.weight': _filled((4, 4)),
            'blocks.0.cross_attn.k_img.weight': _filled((4, 4)),
            'blocks.0.norm2.weight': _filled((4,)),
            'blocks.0.norm3.weight': _filled((4,)),
            'blocks.0.ffn.0.weight': _filled((8, 4)),
            'blocks.0.modulation': _filled((4,)),
            'head.modulation': _filled((4,)),
            'head.head.weight': _filled((4, 4)),
            'blocks.0.after_proj.weight': _filled((4, 4)),
            'img_emb.proj.1.weight': _filled((4, 4)),
            'img_emb.emb_pos': _filled((1, 2, 4)),
        }
        converted = convert_wan_transformer_to_diffusers(_clone(state))
        restored, label = comfykeys.comfy_state(
            converted, {'_class_name': 'WanTransformer3DModel'}, gguf=True)
        self.assertEqual(label, 'wan')
        _same(self, restored, state)

    def test_wan_gguf_patch_embedding_keeps_its_shape(self):
        weight = _filled((5120, 16, 1, 2, 2))
        modulation = _filled((5120, 2, 1))
        state = {
            'patch_embedding.weight': weight,
            'patch_embedding.bias': _filled((5120,)),
            'condition_embedder.time_embedder.linear_1.weight': _filled((32, 32)),
            'scale_shift_table': modulation,
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = quantexport.export_component(
                state, ['Q8_0'], directory, 'transformer',
                {'_class_name': 'WanTransformer3DModel'}, comfy=True)
            reader = gguf.GGUFReader(paths[0])
            patch = next(tensor for tensor in reader.tensors if tensor.name == 'patch_embedding.weight')
            table = next(tensor for tensor in reader.tensors if tensor.name == 'head.modulation')
            linear = next(tensor for tensor in reader.tensors if tensor.name == 'time_embedding.0.weight')
            meta = tuple(int(dim) for dim in patch.shape)
            copied = torch.from_numpy(patch.data.copy())
            patch_type = patch.tensor_type
            table_type = table.tensor_type
            linear_type = linear.tensor_type
            layout = ggufcompat.detect_gguf_layout(ggufcompat.read_gguf_shapes(paths[0]))
            reader.data._mmap.close()
            del reader
        self.assertEqual(patch_type, gguf.GGMLQuantizationType.F32)
        self.assertEqual(tuple(copied.shape), tuple(weight.shape))
        self.assertEqual(int(table_type), int(gguf.GGMLQuantizationType.F32))
        self.assertEqual(linear_type, gguf.GGMLQuantizationType.Q8_0)
        self.assertEqual(meta, (2, 2, 1, 16, 5120))
        self.assertEqual(layout.kind, 'wan-t2v-14B')
        self.assertTrue(torch.equal(copied, weight))

    def test_zimage_safetensors_round_trip_and_gguf_refuses(self):
        from diffusers.loaders.single_file_utils import convert_z_image_transformer_checkpoint_to_diffusers
        state = {
            'x_embedder.weight': _filled((4, 4)),
            'final_layer.linear.weight': _filled((4, 4)),
            'layers.0.attention.qkv.weight': _filled((12, 4)),
            'layers.0.attention.q_norm.weight': _filled((4,)),
            'layers.0.attention.k_norm.weight': _filled((4,)),
            'layers.0.attention.out.weight': _filled((4, 4)),
            'layers.0.attention.out.bias': _filled((4,)),
        }
        converted = convert_z_image_transformer_checkpoint_to_diffusers(_clone(state))
        config = {'_class_name': 'ZImageTransformer2DModel'}
        restored, label = comfykeys.comfy_state(converted, config, gguf=False)
        self.assertEqual(label, 'z-image')
        _same(self, restored, state)
        with self.assertRaises(quantexport.QuantExportError):
            comfykeys.comfy_state(converted, config, gguf=True)

    def test_qwen21_fuses_the_swiglu_pair(self):
        gate = _filled((6, 4))
        proj = _filled((6, 4)) + 3
        state = {
            'img_in.weight': _filled((4, 4)),
            'transformer_blocks.0.img_mlp.gate_layer.weight': gate,
            'transformer_blocks.0.img_mlp.proj.weight': proj,
        }
        converted, label = comfykeys.comfy_state(
            state, {'_class_name': 'QwenImage21Transformer2DModel'}, gguf=True)
        self.assertEqual(label, 'qwen_image')
        self.assertTrue(torch.equal(
            converted['transformer_blocks.0.img_mlp.gate_up.weight'], torch.cat([gate, proj], 0)))
        self.assertIn('img_in.weight', converted)

    def test_t5_and_llama_names_round_trip_through_the_loader_map(self):
        t5 = {
            'shared.weight': _filled((8, 32)),
            'encoder.block.0.layer.0.SelfAttention.q.weight': _filled((32, 32)),
            'encoder.block.0.layer.1.DenseReluDense.wi_0.weight': _filled((32, 32)),
            'encoder.final_layer_norm.weight': _filled((32,)),
        }
        restored, label = comfykeys.comfy_state(t5, {'model_type': 't5'}, gguf=True)
        self.assertEqual(label, 't5')
        self.assertEqual(
            set(restored),
            {
                'token_embd.weight',
                'enc.blk.0.attn_q.weight',
                'enc.blk.0.ffn_gate.weight',
                'enc.output_norm.weight',
            })
        self.assertTrue(torch.equal(
            ggufcompat._remap_keys(restored, ggufcompat._T5_GGUF_KEYS)['shared.weight'],
            t5['shared.weight']))

        rows = 64
        llama = {
            'model.layers.0.self_attn.q_proj.weight': _filled((rows, 8)),
            'model.layers.0.self_attn.k_proj.weight': _filled((rows, 8)),
            'model.norm.weight': _filled((8,)),
        }
        config = {'model_type': 'llama', 'num_attention_heads': 32, 'num_key_value_heads': 8}
        restored, label = comfykeys.comfy_state(llama, config, gguf=True)
        self.assertEqual(label, 'llama')
        self.assertIn('blk.0.attn_q.weight', restored)
        remapped = ggufcompat._remap_keys(restored, ggufcompat._LLAMA_GGUF_KEYS)
        undone = ggufcompat.permute_llama_attention(remapped, 32, 8)
        self.assertTrue(torch.equal(undone['model.layers.0.self_attn.q_proj.weight'], llama[
            'model.layers.0.self_attn.q_proj.weight']))

    def test_qwen2_comfy_gguf_is_labeled_qwen3(self):
        state = {
            'model.embed_tokens.weight': _filled((8, 8)),
            'model.layers.0.self_attn.q_proj.weight': _filled((8, 8)),
            'model.layers.0.self_attn.k_proj.bias': _filled((4,)),
            'model.norm.weight': _filled((8,)),
        }
        restored, label = comfykeys.comfy_state(state, {'model_type': 'qwen2'}, gguf=True)
        self.assertEqual(label, 'qwen3')
        self.assertIn('blk.0.attn_q.weight', restored)
        self.assertIn('blk.0.attn_k.bias', restored)
        plain, plain_label = comfykeys.comfy_state(state, {'model_type': 'qwen2'}, gguf=False)
        self.assertEqual(plain_label, 'qwen2')
        self.assertIn('model.layers.0.self_attn.q_proj.weight', plain)

    def test_llama_mode_round_trips_and_allows_other_head_counts(self):
        t5 = {
            'shared.weight': _filled((8, 32)),
            'encoder.block.0.layer.0.SelfAttention.q.weight': _filled((32, 32)),
        }
        restored, label = comfykeys.llama_state(t5, {'model_type': 't5'})
        self.assertEqual(label, 't5')
        self.assertIn('enc.blk.0.attn_q.weight', restored)
        self.assertIn('token_embd.weight', restored)

        weight = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
        llama = {
            'model.layers.0.self_attn.q_proj.weight': weight,
            'model.norm.weight': torch.ones(8),
        }
        config = {'model_type': 'llama', 'num_attention_heads': 4, 'num_key_value_heads': 2}
        with self.assertRaises(quantexport.QuantExportError):
            comfykeys.comfy_state(llama, config, gguf=True)
        restored, label = comfykeys.llama_state(llama, config)
        self.assertEqual(label, 'llama')
        remapped = ggufcompat._remap_keys(restored, ggufcompat._LLAMA_GGUF_KEYS)
        undone = ggufcompat.permute_llama_attention(remapped, 4, 2)
        self.assertTrue(torch.equal(
            undone['model.layers.0.self_attn.q_proj.weight'], weight))

        gemma = {
            'model.layers.0.input_layernorm.weight': torch.ones(4),
            'model.embed_tokens.weight': _filled((8, 4)),
        }
        restored, label = comfykeys.llama_state(gemma, {'model_type': 'gemma4'})
        self.assertEqual(label, 'gemma4')
        # Gemma 4 stores the norm as written. Gemma 3 adds one, matching Gemma3RMSNorm.
        self.assertTrue(torch.equal(restored['blk.0.attn_norm.weight'], torch.ones(4)))
        gemma3 = {
            'model.layers.0.input_layernorm.weight': torch.ones(4),
            'model.embed_tokens.weight': _filled((8, 4)),
        }
        restored, label = comfykeys.llama_state(gemma3, {'model_type': 'gemma3'})
        self.assertEqual(label, 'gemma3')
        self.assertTrue(torch.equal(restored['blk.0.attn_norm.weight'], torch.full((4,), 2.0)))

    def test_gemma4_llama_file_loads_back_to_huggingface_names(self):
        state = {
            'model.embed_tokens.weight': _filled((8, 4)),
            'model.layers.0.input_layernorm.weight': torch.ones(4),
            'model.layers.0.self_attn.q_norm.weight': torch.full((4,), 0.5),
            'model.layers.0.layer_scalar': torch.tensor([1.25]),
            'model.layers.0.per_layer_input_gate.weight': torch.full((4, 4), 0.3),
            'model.layers.0.per_layer_projection.weight': torch.full((4, 4), 0.4),
            'model.layers.0.post_per_layer_input_norm.weight': torch.full((4,), 0.7),
            'model.embed_tokens_per_layer.weight': torch.full((8, 4), 0.2),
            'model.per_layer_model_projection.weight': torch.full((4, 4), 0.6),
            'model.per_layer_projection_norm.weight': torch.full((4,), 0.8),
            'model.norm.weight': torch.full((4,), 2.0),
        }
        converted, label = comfykeys.llama_state(state, {
            'model_type': 'gemma4',
            'global_head_dim': 8,
            'rope_parameters': {'full_attention': {'partial_rotary_factor': 0.25}},
        })
        self.assertEqual(label, 'gemma4')
        handle, path = tempfile.mkstemp(suffix='.gguf')
        os.close(handle)
        self.addCleanup(os.remove, path)
        writer = gguf.GGUFWriter(path, label)
        for name, value in converted.items():
            writer.add_tensor(name, value.numpy())
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        _, loaded, llama_names = ggufcompat.load_text_encoder_gguf(path)
        self.assertTrue(llama_names)
        self.assertNotIn('rope_freqs.weight', loaded)
        for key, value in state.items():
            self.assertIn(key, loaded, key)
            self.assertTrue(torch.equal(loaded[key].float(), value), key)
        with self.assertRaises(quantexport.QuantExportError):
            comfykeys.llama_state(
                {'down_blocks.0.resnets.0.norm1.weight': _filled((4,))},
                {'_class_name': 'UNet2DConditionModel'})

    def test_gemma4v_mmproj_uses_the_full_tower_names(self):
        patch = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
        query = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        depthwise = torch.arange(10, dtype=torch.float32).reshape(2, 5)
        state = {
            'v.patch_embd.weight': patch,
            'v.position_embd.weight': torch.arange(16, dtype=torch.float32).reshape(2, 4, 2),
            'v.blk.0.attn_q.weight': query,
            'v.blk.0.attn_q.input_min': torch.tensor([1.5]),
            'v.blk.0.ln1.weight': torch.ones(2),
            'v.blk.0.attn_post_norm.weight': torch.ones(2),
            'v.blk.0.ffn_down.weight': query,
            'mm.input_projection.weight': torch.ones(2, 2),
            'mm.a.input_projection.weight': torch.ones(2, 2),
            'a.blk.0.per_dim_scale.weight': torch.full((2,), 0.693147),
            'a.blk.0.conv_dw.weight': depthwise,
            'a.blk.0.attn_out.weight': query,
            'a.conv1d.0.weight': torch.ones(2, 1, 3, 3),
            'a.pre_encode.out.bias': torch.zeros(2),
        }
        restored = ggufcompat.restore_mmproj(state)
        expected = patch.permute(0, 2, 3, 1).contiguous().reshape(2, 12)
        self.assertTrue(torch.equal(
            restored['model.vision_tower.patch_embedder.input_proj.weight'], expected))
        self.assertTrue(torch.equal(
            restored['model.vision_tower.patch_embedder.position_embedding_table'],
            state['v.position_embd.weight']))
        self.assertTrue(torch.equal(
            restored['model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight'], query))
        self.assertEqual(
            restored['model.vision_tower.encoder.layers.0.self_attn.q_proj.input_min'].shape, ())
        self.assertIn('model.vision_tower.encoder.layers.0.input_layernorm.weight', restored)
        self.assertIn('model.vision_tower.encoder.layers.0.post_attention_layernorm.weight', restored)
        self.assertIn('model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight', restored)
        self.assertIn('model.embed_vision.embedding_projection.weight', restored)
        self.assertIn('model.embed_audio.embedding_projection.weight', restored)
        self.assertEqual(
            tuple(restored['model.audio_tower.layers.0.lconv1d.depthwise_conv1d.weight'].shape),
            (2, 1, 5))
        self.assertIn('model.audio_tower.layers.0.self_attn.post.linear.weight', restored)
        self.assertIn('model.audio_tower.subsample_conv_projection.layer0.conv.weight', restored)
        self.assertIn('model.audio_tower.output_proj.bias', restored)
        self.assertTrue(torch.allclose(
            restored['model.audio_tower.layers.0.self_attn.per_dim_scale'],
            torch.zeros(2), atol=1e-4))

    def test_gemma4v_mmproj_round_trips_from_the_checkpoint(self):
        patch = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
        query = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        state = {
            'v.patch_embd.weight': patch,
            'v.position_embd.weight': torch.arange(16, dtype=torch.float32).reshape(2, 4, 2),
            'v.blk.0.attn_q.weight': query,
            'v.blk.0.attn_q.input_min': torch.tensor([1.5]),
            'v.blk.0.ln1.weight': torch.ones(2),
            'mm.input_projection.weight': torch.ones(2, 2),
            'mm.a.input_projection.weight': torch.ones(2, 2),
            'a.blk.0.per_dim_scale.weight': torch.full((2,), 0.693147),
            'a.blk.0.conv_dw.weight': torch.arange(10, dtype=torch.float32).reshape(2, 5),
            'a.blk.0.attn_out.weight': query,
            'a.conv1d.0.weight': torch.ones(2, 1, 3, 3),
            'a.pre_encode.out.bias': torch.zeros(2),
        }
        checkpoint = ggufcompat.restore_mmproj(state)
        checkpoint['model.embed_tokens.weight'] = torch.zeros(4, 4)
        written = comfykeys.mmproj_state(checkpoint, {'model_type': 'gemma4'})
        self.assertEqual(tuple(written['v.patch_embd.weight'].shape), (2, 3, 2, 2))
        self.assertIn('v.blk.0.attn_q.weight', written)
        self.assertIn('a.blk.0.per_dim_scale.weight', written)
        self.assertNotIn('model.embed_tokens.weight', written)
        again = ggufcompat.restore_mmproj(written)
        for key, value in checkpoint.items():
            if key == 'model.embed_tokens.weight':
                self.assertNotIn(key, again)
                continue
            self.assertIn(key, again, key)
            self.assertTrue(torch.allclose(again[key].float(), value.float(), atol=1e-4), key)
        language, label = comfykeys.llama_state(checkpoint, {'model_type': 'gemma4'})
        self.assertEqual(label, 'gemma4')
        self.assertFalse(any(
            'vision_tower' in key or 'audio_tower' in key or 'embed_vision' in key
            for key in language))

    def test_gemma4_mmproj_round_trips(self):
        length = 2 * 2 * 3
        ln1 = torch.arange(length, dtype=torch.float32)
        dense = torch.arange(4 * length, dtype=torch.float32).reshape(4, length)
        pos = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
        state = {
            'model.embed_tokens.weight': torch.zeros(4, 4),
            'model.embed_vision.patch_ln1.weight': ln1,
            'model.embed_vision.patch_ln1.bias': ln1 + 1,
            'model.embed_vision.patch_dense.weight': dense,
            'model.embed_vision.patch_ln2.weight': torch.ones(4),
            'model.embed_vision.pos_norm.weight': torch.full((4,), 0.5),
            'model.embed_vision.pos_embedding': pos,
            'model.embed_vision.multimodal_embedder.embedding_projection.weight': torch.arange(8, dtype=torch.float32).reshape(2, 4),
            'model.embed_audio.embedding_projection.weight': torch.arange(8, dtype=torch.float32).reshape(2, 4),
        }
        converted = comfykeys.mmproj_state(state, {'model_type': 'gemma4'})
        self.assertIn('v.patch_norm.1.weight', converted)
        self.assertIn('v.patch_embd.weight', converted)
        self.assertIn('v.position_embd.weight', converted)
        self.assertIn('mm.input_projection.weight', converted)
        self.assertIn('mm.a.input_projection.weight', converted)
        self.assertNotIn('model.embed_tokens.weight', converted)
        restored = ggufcompat.restore_mmproj(converted)
        for key, value in state.items():
            if key == 'model.embed_tokens.weight':
                self.assertNotIn(key, restored)
                continue
            self.assertIn(key, restored, key)
            self.assertTrue(torch.equal(restored[key], value), key)
        language, label = comfykeys.llama_state(state, {'model_type': 'gemma4'})
        self.assertEqual(label, 'gemma4')
        self.assertFalse(any('embed_vision' in key or 'embed_audio' in key for key in language))

    def test_qwen3vl_mmproj_uses_llama_cpp_names(self):
        patch = torch.arange(2 * 2 * 2 * 2 * 2, dtype=torch.float32).reshape(2, 2, 2, 2, 2)
        state = {
            'model.visual.blocks.5.attn.qkv.weight': _filled((6, 4)),
            'model.visual.blocks.5.mlp.linear_fc1.weight': _filled((8, 4)),
            'model.visual.blocks.5.mlp.linear_fc2.weight': _filled((4, 8)),
            'model.visual.merger.linear_fc1.weight': _filled((4, 8)),
            'model.visual.merger.norm.weight': _filled((4,)),
            'model.visual.deepstack_merger_list.0.linear_fc1.weight': _filled((4, 4)),
            'model.visual.patch_embed.proj.weight': patch,
            'model.visual.pos_embed.weight': _filled((3, 4)),
        }
        config = {'vision_config': {'deepstack_visual_indexes': [5, 11, 17]}}
        converted = comfykeys.mmproj_state(state, config)
        self.assertIn('v.blk.5.attn_qkv.weight', converted)
        self.assertIn('v.blk.5.ffn_up.weight', converted)
        self.assertIn('v.blk.5.ffn_down.weight', converted)
        self.assertIn('mm.0.weight', converted)
        self.assertIn('v.post_ln.weight', converted)
        self.assertIn('v.deepstack.5.fc1.weight', converted)
        self.assertIn('v.patch_embd.weight', converted)
        self.assertIn('v.patch_embd.weight.1', converted)
        self.assertIn('v.position_embd.weight', converted)
        self.assertTrue(torch.equal(converted['v.patch_embd.weight'], patch[:, :, 0]))
        restored = ggufcompat.restore_mmproj(converted, [5, 11, 17])
        self.assertTrue(torch.equal(restored['visual.blocks.5.attn.qkv.weight'], state['model.visual.blocks.5.attn.qkv.weight']))
        self.assertIn('visual.deepstack_merger_list.0.linear_fc1.weight', restored)
        self.assertTrue(torch.equal(restored['visual.patch_embed.proj.weight'], patch))

    def test_qwen3vl_language_model_uses_llama_cpp_block_names(self):
        state = {
            'model.language_model.layers.0.self_attn.q_proj.weight': _filled((4, 4)),
            'model.language_model.embed_tokens.weight': _filled((8, 4)),
            'model.language_model.norm.weight': _filled((4,)),
        }
        converted, label = comfykeys.llama_state(
            state, {'model_type': 'qwen3_vl', 'text_config': {
                'num_attention_heads': 2, 'num_key_value_heads': 1}})
        self.assertEqual(label, 'qwen3vl')
        self.assertIn('blk.0.attn_q.weight', converted)
        self.assertIn('token_embd.weight', converted)
        self.assertIn('output_norm.weight', converted)

    def test_qwen_image_keeps_its_names(self):
        weight = _filled((4, 4))
        state = {'transformer_blocks.0.img_mlp.net.0.proj.weight': weight}
        converted, label = comfykeys.comfy_state(
            state, {'_class_name': 'QwenImageTransformer2DModel'}, gguf=True)
        self.assertEqual(label, 'qwen_image')
        self.assertTrue(torch.equal(
            converted['transformer_blocks.0.img_mlp.net.0.proj.weight'], weight))

    def test_clip_safetensors_keep_hugging_face_names(self):
        state = {'text_model.encoder.layers.0.self_attn.q_proj.weight': _filled((4, 4))}
        converted, label = comfykeys.comfy_state(state, {'model_type': 'clip'}, gguf=False)
        self.assertEqual(label, 'clip')
        self.assertIn('text_model.encoder.layers.0.self_attn.q_proj.weight', converted)
        with self.assertRaises(quantexport.QuantExportError):
            comfykeys.comfy_state(state, {'model_type': 'clip'}, gguf=True)

    def test_sd_unet_round_trip(self):
        from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
        ldm = _sd_ldm_state()
        prefixed = {'model.diffusion_model.' + key: value for key, value in ldm.items()}
        converted = convert_ldm_unet_checkpoint(prefixed, {'layers_per_block': 1})
        restored, label = comfykeys.comfy_state(
            converted, {'_class_name': 'UNet2DConditionModel', 'layers_per_block': 1}, gguf=True)
        self.assertEqual(label, 'sd1')
        _same(self, restored, ldm)

    def test_sdxl_unet_is_labeled_from_the_added_embedding(self):
        state = {
            'add_embedding.linear_1.weight': _filled((4, 4)),
            'add_embedding.linear_1.bias': _filled((4,)),
            'add_embedding.linear_2.weight': _filled((4, 4)),
            'add_embedding.linear_2.bias': _filled((4,)),
            'down_blocks.0.resnets.0.norm1.weight': _filled((4,)),
        }
        restored, label = comfykeys.comfy_state(
            state, {'_class_name': 'UNet2DConditionModel', 'layers_per_block': 1,
                    'addition_embed_type': 'text_time'}, gguf=True)
        self.assertEqual(label, 'sdxl')
        self.assertIn('label_emb.0.0.weight', restored)

    def test_flux2_safetensors_round_trip_and_gguf_refuses(self):
        from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
        state = _flux2_comfy_state()
        converted = convert_flux2_transformer_checkpoint_to_diffusers(_clone(state))
        config = {'_class_name': 'Flux2Transformer2DModel'}
        restored, label = comfykeys.comfy_state(converted, config, gguf=False)
        self.assertEqual(label, 'flux2')
        _same(self, restored, state)
        with self.assertRaises(quantexport.QuantExportError):
            comfykeys.comfy_state(converted, config, gguf=True)

    def test_mmproj_names_round_trip_and_the_file_is_written(self):
        qkv = _filled((12, 4))
        patch = _filled((2, 2, 2, 2, 2))
        state = {
            'model.layers.0.self_attn.q_proj.weight': _filled((64, 8)),
            'visual.blocks.0.attn.qkv.weight': qkv,
            'visual.blocks.0.attn.proj.weight': _filled((4, 4)),
            'visual.blocks.0.mlp.up_proj.weight': _filled((4, 4)),
            'visual.blocks.0.mlp.gate_proj.weight': _filled((4, 4)),
            'visual.blocks.0.mlp.down_proj.weight': _filled((4, 4)),
            'visual.blocks.0.norm1.weight': _filled((4,)),
            'visual.patch_embed.proj.weight': patch,
            'visual.merger.ln_q.weight': _filled((4,)),
            'visual.merger.mlp.0.weight': _filled((4, 4)),
        }
        vision = comfykeys.mmproj_state(state)
        self.assertIn('v.blk.0.attn_q.weight', vision)
        self.assertIn('v.blk.0.attn_out.weight', vision)
        self.assertIn('v.blk.0.ffn_up.weight', vision)
        self.assertIn('v.blk.0.ln1.weight', vision)
        self.assertIn('v.post_ln.weight', vision)
        self.assertIn('mm.0.weight', vision)
        self.assertTrue(torch.equal(
            torch.stack([vision['v.patch_embd.weight'], vision['v.patch_embd.weight.1']], dim=2),
            patch))
        rebuilt = ggufcompat.restore_mmproj(vision)
        self.assertTrue(torch.equal(rebuilt['visual.blocks.0.attn.qkv.weight'], qkv))
        self.assertTrue(torch.equal(rebuilt['visual.patch_embed.proj.weight'], patch))
        config = {
            'model_type': 'qwen2_vl',
            'num_attention_heads': 32,
            'num_key_value_heads': 8,
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = quantexport.export_component(
                state, ['Q8_0'], directory, 'text_encoder', config, comfy=True)
            mmproj = [path for path in paths if os.path.basename(path).startswith('mmproj-')]
            text_path = next(
                path for path in paths if not os.path.basename(path).startswith('mmproj-'))
            self.assertEqual(len(mmproj), 1)
            reader = gguf.GGUFReader(mmproj[0])
            kind = reader.get_field('general.type')
            self.assertEqual(str(bytes(kind.parts[kind.data[-1]]), encoding='utf-8'), 'mmproj')
            names = {tensor.name for tensor in reader.tensors}
            self.assertIn('v.blk.0.attn_q.weight', names)
            text = gguf.GGUFReader(text_path)
            text_names = {tensor.name for tensor in text.tensors}
            text.data._mmap.close()
            reader.data._mmap.close()
            del text, reader
            self.assertNotIn('v.blk.0.attn_q.weight', text_names)
        state = {'weight': _filled((4, 4))}
        with self.assertRaises(quantexport.QuantExportError):
            comfykeys.comfy_state(state, {'_class_name': 'UNet2DConditionModel'}, gguf=False)
        with self.assertRaises(quantexport.QuantExportError):
            comfykeys.comfy_state(
                {'text_model.encoder.layers.0.self_attn.q_proj.weight': _filled((4, 4))},
                {'model_type': 'clip_text_model'},
                gguf=True)

    def test_comfy_gguf_records_the_logical_shape(self):
        state = {
            'shared.weight': torch.randn(8, 32),
            'encoder.block.0.layer.0.SelfAttention.q.weight': torch.randn(32, 32),
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = quantexport.export_component(
                state, ['Q8_0'], directory, 'text_encoder', {'model_type': 't5'}, comfy=True)
            self.assertTrue(paths[0].endswith('text_encoder-Q8_0-comfy.gguf'))
            reader = gguf.GGUFReader(paths[0])
            names = {tensor.name for tensor in reader.tensors}
            self.assertIn('enc.blk.0.attn_q.weight', names)
            field = reader.get_field('comfy.gguf.orig_shape.enc.blk.0.attn_q.weight')
            shape = tuple(int(field.parts[index][0]) for index in field.data)
            architecture = reader.get_field('general.architecture')
            arch = str(bytes(architecture.parts[architecture.data[-1]]), encoding='utf-8')
            reader.data._mmap.close()
            del reader
            self.assertEqual(shape, (32, 32))
            self.assertEqual(arch, 't5')

    def test_subcommand_writes_wan_comfy_names(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            folder = os.path.join(root, 'transformer')
            os.makedirs(folder)
            import safetensors.torch
            safetensors.torch.save_file(
                {'blocks.0.attn1.to_q.weight': torch.randn(32, 32)},
                os.path.join(folder, 'diffusion_pytorch_model.safetensors'))
            with open(os.path.join(folder, 'config.json'), 'w', encoding='utf-8') as handle:
                json.dump({'_class_name': 'WanTransformer3DModel'}, handle)
            command = quantize_command.QuantizeSubCommand(
                loaded_by_name='quantize', plugin_module_paths=[],
                args=[root, '--subfolders', 'transformer', '--formats', 'fp8', '--comfy', '-o', output])
            self.assertEqual(command(), 0)
            path = os.path.join(output, 'transformer-float8_e4m3fn-comfy.safetensors')
            loaded = safetensors.torch.load_file(path)
            self.assertIn('blocks.0.self_attn.q.weight', loaded)
            self.assertNotIn('blocks.0.attn1.to_q.weight', loaded)

    def test_subcommand_refuses_an_unmapped_model(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            folder = os.path.join(root, 'unet')
            os.makedirs(folder)
            import safetensors.torch
            safetensors.torch.save_file(
                {'down_blocks.0.weight': torch.randn(4, 4)},
                os.path.join(folder, 'diffusion_pytorch_model.safetensors'))
            with open(os.path.join(folder, 'config.json'), 'w', encoding='utf-8') as handle:
                json.dump({'_class_name': 'UNet2DConditionModel'}, handle)
            command = quantize_command.QuantizeSubCommand(
                loaded_by_name='quantize', plugin_module_paths=[],
                args=[root, '--formats', 'fp8', '--comfy', '-o', output])
            with self.assertRaises(subcommand_exceptions.SubCommandArgumentError):
                command()


def _flux_comfy_state():
    def pair(rows, cols):
        return _filled((rows, cols)), _filled((rows,))

    state = {}

    def put(key, rows, cols=None):
        if cols is None:
            state[key] = _filled((rows,))
            return
        weight, bias = pair(rows, cols)
        state[key + '.weight'] = weight
        state[key + '.bias'] = bias

    for name in (
            'time_in.in_layer', 'time_in.out_layer', 'vector_in.in_layer', 'vector_in.out_layer',
            'guidance_in.in_layer', 'guidance_in.out_layer', 'txt_in', 'img_in'):
        put(name, 4, 4)
    block = 'double_blocks.0.'
    for name in ('img_mod.lin', 'txt_mod.lin', 'img_mlp.0', 'img_mlp.2', 'txt_mlp.0', 'txt_mlp.2',
                 'img_attn.proj', 'txt_attn.proj'):
        put(block + name, 4, 4)
    put(block + 'img_attn.qkv', 12, 4)
    put(block + 'txt_attn.qkv', 12, 4)
    for name in ('img_attn.norm.query_norm.scale', 'img_attn.norm.key_norm.scale',
                 'txt_attn.norm.query_norm.scale', 'txt_attn.norm.key_norm.scale'):
        put(block + name, 4)
    single = 'single_blocks.0.'
    put(single + 'modulation.lin', 8, 4)
    inner = 3072
    put(single + 'linear1', inner * 7, 4)
    put(single + 'norm.query_norm.scale', 4)
    put(single + 'norm.key_norm.scale', 4)
    put(single + 'linear2', 4, inner)
    put('final_layer.linear', 4, 4)
    put('final_layer.adaLN_modulation.1', 8, 4)
    return state


def _sd3_layer(state, index, last, attn2):
    block = f'joint_blocks.{index}.'

    def put(key, rows, cols=None):
        if cols is None:
            state[key] = _filled((rows,)) + index
            return
        state[key + '.weight'] = _filled((rows, cols)) + index
        state[key + '.bias'] = _filled((rows,)) + index

    for name in ('x_block.attn.qkv', 'context_block.attn.qkv'):
        put(block + name, 12, 4)
    for name in ('x_block.attn.ln_q.weight', 'x_block.attn.ln_k.weight',
                 'context_block.attn.ln_q.weight', 'context_block.attn.ln_k.weight'):
        put(block + name, 4)
    put(block + 'x_block.attn.proj', 4, 4)
    if not last:
        put(block + 'context_block.attn.proj', 4, 4)
    if attn2:
        put(block + 'x_block.attn2.qkv', 12, 4)
        put(block + 'x_block.attn2.ln_q.weight', 4)
        put(block + 'x_block.attn2.ln_k.weight', 4)
        put(block + 'x_block.attn2.proj', 4, 4)
    put(block + 'x_block.adaLN_modulation.1', 8, 4)
    put(block + 'context_block.adaLN_modulation.1', 8, 4)
    put(block + 'x_block.mlp.fc1', 8, 4)
    put(block + 'x_block.mlp.fc2', 4, 8)
    if not last:
        put(block + 'context_block.mlp.fc1', 8, 4)
        put(block + 'context_block.mlp.fc2', 4, 8)


def _sd3_comfy_state():
    state = {
        'pos_embed': _filled((1, 2, 4)),
        'x_embedder.proj.weight': _filled((4, 4)),
        'x_embedder.proj.bias': _filled((4,)),
        't_embedder.mlp.0.weight': _filled((4, 4)),
        't_embedder.mlp.0.bias': _filled((4,)),
        't_embedder.mlp.2.weight': _filled((4, 4)),
        't_embedder.mlp.2.bias': _filled((4,)),
        'context_embedder.weight': _filled((8, 4)),
        'context_embedder.bias': _filled((8,)),
        'y_embedder.mlp.0.weight': _filled((4, 4)),
        'y_embedder.mlp.0.bias': _filled((4,)),
        'y_embedder.mlp.2.weight': _filled((4, 4)),
        'y_embedder.mlp.2.bias': _filled((4,)),
        'final_layer.linear.weight': _filled((4, 4)),
        'final_layer.linear.bias': _filled((4,)),
        'final_layer.adaLN_modulation.1.weight': _filled((8, 4)),
        'final_layer.adaLN_modulation.1.bias': _filled((8,)),
    }
    _sd3_layer(state, 0, last=False, attn2=True)
    _sd3_layer(state, 1, last=True, attn2=False)
    return state


def _pair(rows, cols=None):
    if cols is None:
        return _filled((rows,))
    return _filled((rows, cols))


def _sd_ldm_state():
    state = {}

    def put(key, rows, cols=None):
        state[key] = _pair(rows, cols)

    for key in (
            'time_embed.0.weight', 'time_embed.0.bias', 'time_embed.2.weight', 'time_embed.2.bias',
            'input_blocks.0.0.weight', 'input_blocks.0.0.bias',
            'out.0.weight', 'out.0.bias', 'out.2.weight', 'out.2.bias'):
        put(key, 4, 4)
    for key in (
            'input_blocks.1.0.in_layers.0.weight', 'input_blocks.1.0.in_layers.2.weight',
            'input_blocks.1.0.out_layers.0.weight', 'input_blocks.1.0.out_layers.3.weight',
            'input_blocks.1.0.emb_layers.1.weight', 'input_blocks.1.0.skip_connection.weight',
            'input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight',
            'input_blocks.2.0.op.weight', 'input_blocks.2.0.op.bias',
            'middle_block.0.in_layers.0.weight', 'middle_block.1.transformer_blocks.0.attn1.to_q.weight',
            'output_blocks.0.0.in_layers.0.weight',
            'output_blocks.0.1.transformer_blocks.0.attn1.to_q.weight',
            'output_blocks.1.0.in_layers.0.weight',
            'output_blocks.1.1.conv.weight', 'output_blocks.1.1.conv.bias'):
        put(key, 4, 4)
    return state


def _flux2_comfy_state():
    state = {}

    def put(key, rows, cols=None, scale=False):
        if scale:
            state[key + '.scale'] = _filled((rows,))
            return
        state[key + '.weight'] = _filled((rows, cols))
        state[key + '.bias'] = _filled((rows,))

    for name in (
            'img_in', 'txt_in', 'time_in.in_layer', 'time_in.out_layer',
            'double_stream_modulation_img.lin', 'single_stream_modulation.lin',
            'final_layer.linear'):
        put(name, 4, 4)
    put('final_layer.adaLN_modulation.1', 8, 4)
    block = 'double_blocks.0.'
    put(block + 'img_attn.qkv', 12, 4)
    put(block + 'txt_attn.qkv', 12, 4)
    put(block + 'img_attn.norm.query_norm', 4, scale=True)
    put(block + 'img_attn.norm.key_norm', 4, scale=True)
    put(block + 'txt_attn.norm.query_norm', 4, scale=True)
    put(block + 'txt_attn.norm.key_norm', 4, scale=True)
    for name in ('img_attn.proj', 'img_mlp.0', 'img_mlp.2', 'txt_attn.proj', 'txt_mlp.0', 'txt_mlp.2'):
        put(block + name, 4, 4)
    single = 'single_blocks.0.'
    put(single + 'linear1', 8, 4)
    put(single + 'linear2', 4, 4)
    put(single + 'norm.query_norm', 4, scale=True)
    put(single + 'norm.key_norm', 4, scale=True)
    return state

