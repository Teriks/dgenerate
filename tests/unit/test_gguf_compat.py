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
        self.assertFalse(layered.adapt)

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
