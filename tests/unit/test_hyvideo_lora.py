import json
import os
import tempfile
import unittest

import safetensors.torch
import torch

import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.videopipelines as _videopipelines
from dgenerate.pipelinewrapper.videopipelines.hyvideo_lora import (
    HunyuanVideo15LoraLoaderMixin,
    convert_hunyuan_video15_lora_state_dict,
)


def _pair(out_features, in_features, rank=4):
    return {
        'A': torch.ones(rank, in_features),
        'B': torch.ones(out_features, rank),
    }


class TestHunyuanVideo15Lora(unittest.TestCase):
    def test_pipeline_classes_accept_loras(self):
        import diffusers

        for pipeline_class in (
                diffusers.HunyuanVideo15Pipeline,
                diffusers.HunyuanVideo15ImageToVideoPipeline):
            self.assertTrue(hasattr(pipeline_class, 'load_lora_weights'), pipeline_class.__name__)
            self.assertTrue(_pipelines.pipeline_class_supports_lora(pipeline_class))
        self.assertIs(
            _videopipelines._single_image_video_pipeline_class('hyvideo-15', 'text-to-video'),
            diffusers.HunyuanVideo15Pipeline)

    def test_original_attention_keys_map_onto_diffusers_modules(self):
        state = {}
        for name, out_features, in_features in (
                ('double_blocks.0.img_attn_q', 16, 16),
                ('double_blocks.0.img_attn_proj', 16, 16),
                ('double_blocks.0.txt_attn_k', 16, 16),
                ('double_blocks.0.txt_attn_proj', 16, 16),
                ('double_blocks.0.img_mod.linear', 96, 16),
                ('double_blocks.0.txt_mod.linear', 96, 16),
                ('double_blocks.0.img_mlp.fc1', 64, 16),
                ('double_blocks.0.txt_mlp.fc2', 16, 64),
                ('txt_in.individual_token_refiner.blocks.0.adaLN_modulation.1', 32, 16),
                ('final_layer.adaLN_modulation.1', 32, 16),
                ('final_layer.linear', 32, 16)):
            pair = _pair(out_features, in_features)
            state[f'{name}.lora_A.weight'] = pair['A']
            state[f'{name}.lora_B.weight'] = pair['B']
        # Original final AdaLN is shift then scale. Diffusers reads scale then shift.
        state['final_layer.adaLN_modulation.1.lora_B.weight'] = torch.cat((
            torch.full((16, 4), 1.0),
            torch.full((16, 4), 2.0),
        ), dim=0)
        refiner = state[
            'txt_in.individual_token_refiner.blocks.0.adaLN_modulation.1.lora_B.weight'].clone()

        converted, _, _ = convert_hunyuan_video15_lora_state_dict(state)
        self.assertEqual(
            converted['transformer.transformer_blocks.0.attn.to_q.lora_A.weight'].shape,
            (4, 16))
        self.assertIn('transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight', converted)
        self.assertIn('transformer.transformer_blocks.0.attn.add_k_proj.lora_A.weight', converted)
        self.assertIn('transformer.transformer_blocks.0.attn.to_add_out.lora_B.weight', converted)
        self.assertIn('transformer.transformer_blocks.0.norm1.linear.lora_B.weight', converted)
        self.assertIn('transformer.transformer_blocks.0.norm1_context.linear.lora_A.weight', converted)
        self.assertIn('transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight', converted)
        self.assertIn('transformer.transformer_blocks.0.ff_context.net.2.lora_B.weight', converted)
        self.assertIn(
            'transformer.context_embedder.token_refiner.refiner_blocks.0.norm_out.linear.lora_A.weight',
            converted)
        self.assertIn('transformer.norm_out.linear.lora_A.weight', converted)
        self.assertIn('transformer.proj_out.lora_B.weight', converted)

        final = converted['transformer.norm_out.linear.lora_B.weight']
        self.assertTrue(torch.equal(final[:16], torch.full((16, 4), 2.0)))
        self.assertTrue(torch.equal(final[16:], torch.full((16, 4), 1.0)))
        self.assertTrue(torch.equal(
            converted[
                'transformer.context_embedder.token_refiner.refiner_blocks.0.'
                'norm_out.linear.lora_B.weight'],
            refiner))

    def test_fused_qkv_and_musubi_alpha(self):
        fused_up = torch.arange(48 * 2, dtype=torch.float32).reshape(48, 2)
        state = {
            'diffusion_model.double_blocks.0.img_attn_qkv.lora_A.weight': torch.ones(2, 16),
            'diffusion_model.double_blocks.0.img_attn_qkv.lora_B.weight': fused_up,
            'lora_unet_double_blocks_0_txt_attn_q.lora_down.weight': torch.ones(4, 16),
            'lora_unet_double_blocks_0_txt_attn_q.lora_up.weight': torch.ones(16, 4),
            'lora_unet_double_blocks_0_txt_attn_q.alpha': torch.tensor(8.0),
        }
        metadata = {
            'lora_adapter_metadata': json.dumps({'r': 4, 'lora_alpha': 32, 'alpha_pattern': {}}),
        }
        converted, _, metadata = convert_hunyuan_video15_lora_state_dict(state, metadata)
        for projection in ('to_q', 'to_k', 'to_v'):
            self.assertTrue(torch.equal(
                converted[f'transformer.transformer_blocks.0.attn.{projection}.lora_A.weight'],
                torch.ones(2, 16)))
        self.assertTrue(torch.equal(
            converted['transformer.transformer_blocks.0.attn.to_q.lora_B.weight'],
            fused_up[:16]))
        self.assertTrue(torch.equal(
            converted['transformer.transformer_blocks.0.attn.to_v.lora_B.weight'],
            fused_up[32:]))
        # Kohya alpha 8 / rank 4 scales the up weight. The PEFT header must not scale it again.
        self.assertTrue(torch.equal(
            converted['transformer.transformer_blocks.0.attn.add_q_proj.lora_B.weight'],
            torch.full((16, 4), 2.0)))
        header = json.loads(metadata['lora_adapter_metadata'])
        self.assertEqual(header['lora_alpha'], header['r'])

    def test_peft_alpha_is_baked_when_metadata_would_be_dropped(self):
        state = {
            'transformer.transformer_blocks.0.attn.to_q.lora_A.weight': torch.ones(4, 16),
            'transformer.transformer_blocks.0.attn.to_q.lora_B.weight': torch.ones(16, 4),
        }
        metadata = {'lora_adapter_metadata': json.dumps({
            'r': 4, 'lora_alpha': 8, 'alpha_pattern': {}, 'rank_pattern': {}})}
        converted, _, metadata = convert_hunyuan_video15_lora_state_dict(state, metadata)
        self.assertTrue(torch.equal(
            converted['transformer.transformer_blocks.0.attn.to_q.lora_B.weight'],
            torch.full((16, 4), 2.0)))
        self.assertEqual(json.loads(metadata['lora_adapter_metadata'])['lora_alpha'], 4)

    def test_comfy_dotted_lora_maps_onto_diffusers_modules(self):
        hidden = 16
        qkv_up = torch.arange(hidden * 3 * 2, dtype=torch.float32).reshape(hidden * 3, 2)
        state = {
            'diffusion_model.double_blocks.0.img_attn.qkv.lora_down.weight': torch.ones(2, hidden),
            'diffusion_model.double_blocks.0.img_attn.qkv.lora_up.weight': qkv_up,
            'diffusion_model.double_blocks.0.img_attn.proj.lora_down.weight': torch.ones(4, hidden),
            'diffusion_model.double_blocks.0.img_attn.proj.lora_up.weight': torch.ones(hidden, 4),
            'diffusion_model.double_blocks.0.img_mod.lin.lora_down.weight': torch.ones(4, hidden),
            'diffusion_model.double_blocks.0.img_mod.lin.lora_up.weight': torch.ones(hidden * 6, 4),
            'diffusion_model.double_blocks.0.img_mlp.0.lora_down.weight': torch.ones(4, hidden),
            'diffusion_model.double_blocks.0.img_mlp.0.lora_up.weight': torch.ones(hidden * 4, 4),
            'diffusion_model.double_blocks.0.img_mlp.0.diff_b': torch.ones(hidden * 4),
            'diffusion_model.time_in.in_layer.lora_down.weight': torch.ones(4, 256),
            'diffusion_model.time_in.in_layer.lora_up.weight': torch.ones(hidden, 4),
            'diffusion_model.txt_in.c_embedder.out_layer.lora_down.weight': torch.ones(4, hidden),
            'diffusion_model.txt_in.c_embedder.out_layer.lora_up.weight': torch.ones(hidden, 4),
        }
        converted, _, _ = convert_hunyuan_video15_lora_state_dict(state)
        self.assertNotIn(
            'transformer.transformer_blocks.0.img_mod.lin.lora_A.weight', converted)
        self.assertTrue(torch.equal(
            converted['transformer.transformer_blocks.0.attn.to_q.lora_B.weight'],
            qkv_up[:hidden]))
        self.assertIn('transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight', converted)
        self.assertIn('transformer.transformer_blocks.0.norm1.linear.lora_A.weight', converted)
        self.assertIn('transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight', converted)
        self.assertIn('transformer.time_embed.timestep_embedder.linear_1.lora_A.weight', converted)
        self.assertIn(
            'transformer.context_embedder.time_text_embed.text_embedder.linear_2.lora_B.weight',
            converted)
        self.assertFalse(any('diff' in key for key in converted))
        final = torch.cat((torch.ones(16, 4), torch.full((16, 4), 2.0)), dim=0)
        _lora, updates, _metadata = convert_hunyuan_video15_lora_state_dict({
            'diffusion_model.double_blocks.0.img_attn.qkv.diff_b': torch.arange(48),
            'diffusion_model.final_layer.adaLN_modulation.1.diff': final,
            'diffusion_model.cond_type_embedding.lora_down.weight': torch.ones(3, 8),
            'diffusion_model.cond_type_embedding.lora_up.weight': torch.eye(3),
        })
        self.assertTrue(torch.equal(
            updates['transformer.transformer_blocks.0.attn.to_q.bias'],
            torch.arange(16)))
        swapped = updates['transformer.norm_out.linear.weight']
        self.assertTrue(torch.equal(swapped[:16], torch.full((16, 4), 2.0)))
        self.assertTrue(torch.equal(swapped[16:], torch.ones(16, 4)))
        self.assertTrue(torch.equal(
            updates['transformer.cond_type_embed.weight'],
            torch.eye(3) @ torch.ones(3, 8)))
        self.assertFalse(any('cond_type_embed' in key for key in _lora))

    def test_single_stream_lora_is_rejected(self):
        state = {
            'single_blocks.0.linear1.lora_A.weight': torch.ones(4, 16),
            'single_blocks.0.linear1.lora_B.weight': torch.ones(16, 4),
        }
        with self.assertRaises(ValueError) as caught:
            convert_hunyuan_video15_lora_state_dict(state)
        self.assertIn('single-stream', str(caught.exception))

    def test_loader_wraps_the_matching_linear(self):
        from diffusers.models.transformers.transformer_hunyuan_video15 import (
            HunyuanVideo15Transformer3DModel,
        )

        model = HunyuanVideo15Transformer3DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=8,
            num_layers=1,
            num_refiner_layers=1,
            text_embed_dim=8,
            text_embed_2_dim=8,
            image_embed_dim=8,
            rope_axes_dim=(2, 3, 3),
        )
        hidden = 16

        class MarkerLinear(torch.nn.Linear):
            pass

        original = model.transformer_blocks[0].attn.to_q
        marked = MarkerLinear(original.in_features, original.out_features, bias=original.bias is not None)
        with torch.no_grad():
            marked.weight.copy_(original.weight)
            if original.bias is not None:
                marked.bias.copy_(original.bias)
        model.transformer_blocks[0].attn.to_q = marked

        state = {
            'double_blocks.0.img_attn_q.lora_A.weight': torch.full((4, hidden), 0.25),
            'double_blocks.0.img_attn_q.lora_B.weight': torch.full((hidden, 4), 0.5),
            'double_blocks.0.img_mod.linear.lora_A.weight': torch.ones(4, hidden),
            'double_blocks.0.img_mod.linear.lora_B.weight': torch.ones(hidden * 6, 4),
            'final_layer.adaLN_modulation.1.lora_A.weight': torch.ones(4, hidden),
            'final_layer.adaLN_modulation.1.lora_B.weight': torch.ones(hidden * 2, 4),
            'txt_in.individual_token_refiner.blocks.0.adaLN_modulation.1.lora_A.weight': torch.ones(4, hidden),
            'txt_in.individual_token_refiner.blocks.0.adaLN_modulation.1.lora_B.weight': torch.ones(hidden * 2, 4),
        }

        class _Pipeline(HunyuanVideo15LoraLoaderMixin):
            def __init__(self, transformer):
                self.transformer = transformer
                self.hf_device_map = None
                self.components = {'transformer': transformer}

        _Pipeline(model).load_lora_weights(state, adapter_name='style')
        wrapped = model.transformer_blocks[0].attn.to_q
        self.assertIs(type(wrapped.base_layer), MarkerLinear)
        self.assertIn('style', model.peft_config)
        sample = torch.randn(2, hidden)
        with torch.no_grad():
            adapted = wrapped(sample)
            plain = wrapped.base_layer(sample)
        self.assertFalse(torch.allclose(adapted, plain))

    def test_lightx2v_diffs_are_merged_when_the_lora_is_fused(self):
        from diffusers.models.transformers.transformer_hunyuan_video15 import (
            HunyuanVideo15Transformer3DModel,
        )

        model = HunyuanVideo15Transformer3DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=8,
            num_layers=1,
            num_refiner_layers=1,
            text_embed_dim=8,
            text_embed_2_dim=8,
            image_embed_dim=8,
            rope_axes_dim=(2, 3, 3),
        )
        layer = model.transformer_blocks[0].norm1.linear
        bias_before = layer.bias.detach().clone()
        embed_before = model.cond_type_embed.weight.detach().clone()
        hidden = layer.in_features
        state = {
            'diffusion_model.double_blocks.0.img_mod.lin.lora_down.weight': torch.zeros(4, hidden),
            'diffusion_model.double_blocks.0.img_mod.lin.lora_up.weight': torch.zeros(layer.out_features, 4),
            'diffusion_model.double_blocks.0.img_mod.lin.diff_b': torch.ones_like(layer.bias),
            'diffusion_model.cond_type_embedding.lora_down.weight': torch.ones(3, embed_before.shape[1]),
            'diffusion_model.cond_type_embedding.lora_up.weight': torch.eye(3) * 0.25,
        }

        class _Pipeline(HunyuanVideo15LoraLoaderMixin):
            def __init__(self, transformer):
                self.transformer = transformer
                self.hf_device_map = None
                self.components = {'transformer': transformer}

        pipeline = _Pipeline(model)
        pipeline.load_lora_weights(state, adapter_name='style', lora_strength=0.5)
        pipeline.fuse_lora(lora_scale=2.0)
        self.assertTrue(torch.allclose(layer.bias, bias_before + 1))
        embed_delta = (torch.eye(3) * 0.25) @ torch.ones(3, embed_before.shape[1])
        self.assertTrue(torch.allclose(model.cond_type_embed.weight, embed_before + embed_delta))
        self.assertFalse(pipeline._hy15_weight_updates)

    def test_peft_header_does_not_ask_the_transformer_for_model_type(self):
        from diffusers.models.transformers.transformer_hunyuan_video15 import (
            HunyuanVideo15Transformer3DModel,
        )

        model = HunyuanVideo15Transformer3DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=8,
            num_layers=1,
            num_refiner_layers=1,
            text_embed_dim=8,
            text_embed_2_dim=8,
            image_embed_dim=8,
            rope_axes_dim=(2, 3, 3),
        )
        hidden = 16
        handle, path = tempfile.mkstemp(suffix='.safetensors')
        os.close(handle)
        self.addCleanup(os.remove, path)
        safetensors.torch.save_file(
            {
                'double_blocks.0.img_attn_q.lora_A.weight': torch.ones(4, hidden),
                'double_blocks.0.img_attn_q.lora_B.weight': torch.ones(hidden, 4),
            },
            path,
            metadata={'lora_adapter_metadata': json.dumps({
                'r': 4,
                'lora_alpha': 8,
                'alpha_pattern': {},
                'rank_pattern': {},
                'target_modules': ['img_attn_q'],
                'peft_type': 'LORA',
                'task_type': 'FEATURE_EXTRACTION',
            })},
        )

        class _Pipeline(HunyuanVideo15LoraLoaderMixin):
            def __init__(self, transformer):
                self.transformer = transformer
                self.hf_device_map = None
                self.components = {'transformer': transformer}

        _Pipeline(model).load_lora_weights(path, adapter_name='style')
        layer = model.transformer_blocks[0].attn.to_q
        self.assertIn('style', model.peft_config)
        self.assertEqual(model.peft_config['style'].target_modules, {
            'transformer_blocks.0.attn.to_q'})
        # alpha 8 / rank 4 is stored in the up weight. The config scale stays 1.
        self.assertTrue(torch.equal(layer.lora_B['style'].weight, torch.full((hidden, 4), 2.0)))
        self.assertEqual(layer.scaling['style'], 1.0)
