import unittest

import torch

import dgenerate.pipelinewrapper.ggufcompat as _ggufcompat


class TestGGUFCompat(unittest.TestCase):
    def test_flux2_width_selects_klein_or_dev(self):
        klein = _ggufcompat.detect_gguf_layout({
            'single_stream_modulation.lin.weight': (3072, 9216),
            'img_in.weight': (128, 3072),
        })
        self.assertEqual(klein.kind, 'flux2-klein-4b')
        self.assertEqual(klein.config_repo, 'black-forest-labs/FLUX.2-klein-4B')
        self.assertFalse(klein.adapt)

        nine = _ggufcompat.detect_gguf_layout({
            'model.diffusion_model.single_stream_modulation.lin.weight': (4096, 12288),
            'model.diffusion_model.img_in.weight': (128, 4096),
        })
        self.assertEqual(nine.kind, 'flux2-klein-9b')
        self.assertTrue(nine.comfy)
        self.assertEqual(nine.config_repo, 'black-forest-labs/FLUX.2-klein-base-9B')

        dev = _ggufcompat.detect_gguf_layout({
            'single_stream_modulation.lin.weight': (6144, 18432),
            'double_blocks.0.img_attn.qkv.weight': (6144, 18432),
        })
        self.assertEqual(dev.kind, 'flux2-dev')
        self.assertEqual(dev.config_repo, 'black-forest-labs/FLUX.2-dev')

    def test_flux1_is_not_flux2(self):
        self.assertIsNone(_ggufcompat.detect_gguf_layout({
            'double_blocks.0.img_attn.qkv.weight': (3072, 9216),
        }))

    def test_qwen_zimage_and_ltx(self):
        qwen = _ggufcompat.detect_gguf_layout({
            'img_in.weight': (64, 3072),
            'time_text_embed.timestep_embedder.linear_1.weight': (3072, 256),
        })
        self.assertEqual(qwen.kind, 'qwen-image')
        self.assertEqual(qwen.config_repo, 'Qwen/Qwen-Image')
        self.assertFalse(qwen.adapt)

        layered = _ggufcompat.detect_gguf_layout({
            'img_in.weight': (64, 3072),
            'time_text_embed.timestep_embedder.linear_1.weight': (256, 3072),
            'time_text_embed.addition_t_embedding.weight': (3072, 2),
        })
        self.assertEqual(layered.kind, 'qwen-image-layered')
        self.assertEqual(layered.config_repo, 'Qwen/Qwen-Image-Layered')
        # Always adapt so addition_t_embedding is materialized for nn.Embedding.
        self.assertTrue(layered.adapt)

        comfy_qwen = _ggufcompat.detect_gguf_layout({
            'model.diffusion_model.img_in.weight': (64, 3072),
            'model.diffusion_model.time_text_embed.timestep_embedder.linear_1.weight': (3072, 256),
        })
        self.assertTrue(comfy_qwen.comfy)
        self.assertTrue(comfy_qwen.adapt)

        zimage = _ggufcompat.detect_gguf_layout({
            'cap_embedder.0.weight': (2560,),
            'layers.0.adaLN_modulation.0.weight': (256, 15360),
        })
        self.assertEqual(zimage.kind, 'z-image-turbo')
        self.assertEqual(zimage.config_repo, 'Tongyi-MAI/Z-Image-Turbo')

        ltx = _ggufcompat.detect_gguf_layout({
            'keyframes_abs_pos_embedding': (4096,),
            'audio_patchify_proj.weight': (128, 2048),
        })
        self.assertEqual(ltx.kind, 'ltx-2.5')
        self.assertEqual(ltx.config_repo, 'Lightricks/LTX-2.5-Diffusers')
        self.assertTrue(ltx.adapt)

    def test_wan_layout_from_patch_embedding(self):
        t2v_small = _ggufcompat.detect_gguf_layout({
            'head.modulation': (1536,),
            'patch_embedding.weight': (1536, 16, 2, 2),
        })
        self.assertEqual(t2v_small.kind, 'wan-t2v-1.3B')
        self.assertEqual(t2v_small.config_repo, 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers')
        self.assertFalse(t2v_small.adapt)

        ti2v = _ggufcompat.detect_gguf_layout({
            'head.modulation': (3072,),
            'patch_embedding.weight': (3072, 48, 2, 2),
        })
        self.assertEqual(ti2v.kind, 'wan-ti2v-5B')
        self.assertEqual(ti2v.config_repo, 'Wan-AI/Wan2.2-TI2V-5B-Diffusers')

        i2v = _ggufcompat.detect_gguf_layout({
            'model.diffusion_model.head.modulation': (5120,),
            'model.diffusion_model.patch_embedding.weight': (5120, 36, 2, 2),
        })
        self.assertEqual(i2v.kind, 'wan-i2v-14B')
        self.assertTrue(i2v.comfy)
        self.assertEqual(i2v.config_repo, 'Wan-AI/Wan2.1-I2V-14B-480P-Diffusers')

        t2v = _ggufcompat.detect_gguf_layout({
            'head.modulation': (5120,),
            'patch_embedding.weight': (5120, 16, 2, 2),
        })
        self.assertEqual(t2v.kind, 'wan-t2v-14B')

        vace = _ggufcompat.detect_gguf_layout({
            'head.modulation': (1536,),
            'patch_embedding.weight': (1536, 16, 2, 2),
            'vace_blocks.0.after_proj.bias': (1536,),
        })
        self.assertEqual(vace.kind, 'wan-vace-1.3B')

        animate = _ggufcompat.detect_gguf_layout({
            'head.modulation': (5120,),
            'patch_embedding.weight': (5120, 16, 2, 2),
            'motion_encoder.dec.direction.weight': (5120, 512),
        })
        self.assertEqual(animate.kind, 'wan-animate-14B')
        self.assertEqual(animate.config_repo, 'Wan-AI/Wan2.2-Animate-14B-Diffusers')
        self.assertTrue(animate.adapt)

        animate2 = _ggufcompat.detect_gguf_layout({
            'head.modulation': (5120,),
            'patch_embedding.weight': (5120, 36, 1, 2, 2),
            'blocks.0.block.self_attn.q.weight': (5120, 5120),
        })
        self.assertEqual(animate2.kind, 'wan-animate-2-14B')
        self.assertEqual(animate2.config_repo, 'Wan-AI/Wan2.2-Animate-2-14B-Diffusers')
        self.assertFalse(animate2.adapt)

        # Plain Wan self_attn keys must not classify as Animate-2.
        t2v_with_attn = _ggufcompat.detect_gguf_layout({
            'head.modulation': (5120,),
            'patch_embedding.weight': (5120, 16, 2, 2),
            'blocks.0.self_attn.to_q.weight': (5120, 5120),
        })
        self.assertEqual(t2v_with_attn.kind, 'wan-t2v-14B')

        i2v_with_attn = _ggufcompat.detect_gguf_layout({
            'head.modulation': (5120,),
            'patch_embedding.weight': (5120, 36, 2, 2),
            'blocks.0.self_attn.to_q.weight': (5120, 5120),
        })
        self.assertEqual(i2v_with_attn.kind, 'wan-i2v-14B')

        animate2_comfy = _ggufcompat.detect_gguf_layout({
            'model.diffusion_model.head.modulation': (5120,),
            'model.diffusion_model.patch_embedding.weight': (5120, 36, 1, 2, 2),
            'model.diffusion_model.blocks.0.block.self_attn.q.weight': (5120, 5120),
        })
        self.assertEqual(animate2_comfy.kind, 'wan-animate-2-14B')
        self.assertTrue(animate2_comfy.comfy)

    def test_wan_animate_dequantizes_linear1_kv(self):
        import gguf
        from diffusers.quantizers.gguf.utils import GGML_QUANT_SIZES, GGUFParameter

        _ggufcompat.install_gguf_patches()
        quant_type = gguf.GGMLQuantizationType.Q8_0
        _block_size, type_size = GGML_QUANT_SIZES[quant_type]
        packed = torch.zeros(4, type_size, dtype=torch.uint8)
        param = GGUFParameter(packed, quant_type=quant_type)
        checkpoint = {
            'face_adapter.fuser_blocks.0.linear1_kv.weight': param,
            'other.weight': param,
            'model.diffusion_model.motion_encoder.enc.net_app.convs.0.bias': param,
        }
        adapted = _ggufcompat.adapt_wan_animate_checkpoint(checkpoint)
        linear = adapted['face_adapter.fuser_blocks.0.linear1_kv.weight']
        bias = adapted['motion_encoder.enc.net_app.convs.0.bias']
        other = adapted['other.weight']
        self.assertFalse(hasattr(linear, 'quant_type'))
        self.assertFalse(hasattr(bias, 'quant_type'))
        self.assertTrue(hasattr(other, 'quant_type'))
        half = linear.shape[0] // 2
        split = linear[:half]
        self.assertEqual(tuple(split.shape)[0], half)

    def test_qwen_prefix_strip_and_ltx_plain_shapes(self):
        stripped = _ggufcompat.adapt_qwen_checkpoint({
            'model.diffusion_model.img_in.weight': torch.zeros(64, 3072),
            'img_in.bias': torch.zeros(64),
        })
        self.assertIn('img_in.weight', stripped)
        self.assertNotIn('model.diffusion_model.img_in.weight', stripped)

        table = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        flipped = _ggufcompat._align_tensor(table, (2, 4))
        self.assertEqual(tuple(flipped.shape), (2, 4))
        self.assertTrue(torch.equal(flipped, table.transpose(0, 1)))

        embedding = torch.arange(4, dtype=torch.float32)
        expanded = _ggufcompat._align_tensor(embedding, (1, 4))
        self.assertEqual(tuple(expanded.shape), (1, 4))

    def test_qwen_layered_materializes_addition_t_embedding(self):
        import gguf
        from diffusers.quantizers.gguf.utils import GGUFParameter

        # BF16 storage is uint8 with last dim 2x the logical width. nn.Embedding
        # would otherwise return width 6144 instead of 3072.
        logical = torch.arange(2 * 3072, dtype=torch.bfloat16).reshape(2, 3072)
        packed = logical.contiguous().view(torch.uint8)
        param = GGUFParameter(packed, quant_type=gguf.GGMLQuantizationType.BF16)
        self.assertEqual(tuple(param.shape), (2, 6144))
        self.assertEqual(tuple(param.quant_shape), (2, 3072))

        adapted = _ggufcompat.adapt_qwen_layered_checkpoint({
            'model.diffusion_model.time_text_embed.addition_t_embedding.weight': param,
            'img_in.weight': torch.zeros(64, 3072),
        })
        weight = adapted['time_text_embed.addition_t_embedding.weight']
        self.assertEqual(tuple(weight.shape), (2, 3072))
        self.assertFalse(hasattr(weight, 'quant_type'))
        self.assertTrue(torch.equal(weight.cpu().bfloat16(), logical))

        transposed = _ggufcompat.adapt_qwen_layered_checkpoint({
            'time_text_embed.addition_t_embedding.weight': logical.transpose(0, 1).contiguous(),
        })
        self.assertEqual(
            tuple(transposed['time_text_embed.addition_t_embedding.weight'].shape),
            (2, 3072))

    def test_gguf_parameter_keeps_quant_type_when_rebuilt_for_offload(self):
        import gguf
        from diffusers.quantizers.gguf.utils import GGML_QUANT_SIZES, GGUFParameter

        _ggufcompat.install_gguf_patches()
        quant_type = gguf.GGMLQuantizationType.Q8_0
        _block_size, type_size = GGML_QUANT_SIZES[quant_type]
        packed = torch.zeros(2, type_size, dtype=torch.uint8)
        original = GGUFParameter(packed, quant_type=quant_type)
        original._dgenerate_comfy_transpose = True

        moved = original.to('meta')
        rebuilt = type(original)(moved, requires_grad=False)
        self.assertEqual(rebuilt.quant_type, quant_type)
        self.assertTrue(rebuilt._dgenerate_comfy_transpose)
        self.assertEqual(rebuilt.device.type, 'meta')


if __name__ == '__main__':
    unittest.main()
