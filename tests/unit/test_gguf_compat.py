import unittest
from unittest import mock

import torch

import dgenerate.pipelinewrapper.ggufcompat as _ggufcompat
import dgenerate.pipelinewrapper.singlefilecompat as _singlefilecompat


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

    def test_kandinsky5_gguf_loads_standard_and_flattened_layouts(self):
        import json
        import os
        import tempfile

        import diffusers
        import gguf
        import numpy

        config = {
            'in_visual_dim': 16, 'out_visual_dim': 16, 'time_dim': 256,
            'patch_size': [1, 2, 2], 'model_dim': 256, 'ff_dim': 512,
            'num_text_blocks': 1, 'num_visual_blocks': 1, 'axes_dims': [32, 48, 48],
            'visual_cond': True, 'in_text_dim': 256, 'in_text_dim2': 256,
            'attention_type': 'regular'}
        torch.manual_seed(0)
        model = diffusers.Kandinsky5Transformer3DModel.from_config(config)
        state = {name: value.detach().float().numpy() for name, value in model.state_dict().items()}
        quant = gguf.GGMLQuantizationType.Q8_0

        def write(path, flatten=False, corrupt=False):
            writer = gguf.GGUFWriter(path, 'kandinsky5')
            for name, value in state.items():
                if value.ndim == 2 and value.shape[1] % 32 == 0:
                    packed = gguf.quants.quantize(value, quant)
                    if corrupt:
                        # Q8_0 blocks start with a float16 scale; 0x7e00 is NaN.
                        packed = packed.copy()
                        packed.reshape(-1, 34)[:, :2] = numpy.array(
                            [0x7e00], dtype=numpy.uint16).view(numpy.uint8)
                    if flatten:
                        packed = packed.reshape(-1)
                    writer.add_tensor(name, packed, raw_dtype=quant)
                elif name == 'out_layer.modulation.out_layer.weight':
                    # Standard files can carry an extra singleton dimension.
                    writer.add_tensor(name, value.reshape(1, *value.shape))
                else:
                    writer.add_tensor(name, value)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

        def load(path, config_dir):
            _ggufcompat.install_gguf_patches()
            checkpoint, config_repo, subfolder, converted = _ggufcompat.plan_gguf_load(
                path, config_dir, 'transformer')
            self.assertIsNotNone(checkpoint)
            self.assertFalse(converted)
            return diffusers.Kandinsky5Transformer3DModel.from_single_file(
                checkpoint, config=config_repo, subfolder=subfolder,
                torch_dtype=torch.float32,
                quantization_config=diffusers.GGUFQuantizationConfig(
                    compute_dtype=torch.float32))

        with tempfile.TemporaryDirectory() as directory:
            os.makedirs(os.path.join(directory, 'transformer'))
            with open(os.path.join(directory, 'transformer', 'config.json'), 'w') as file:
                json.dump(dict(config, _class_name='Kandinsky5Transformer3DModel'), file)

            standard = os.path.join(directory, 'standard.gguf')
            write(standard)
            layout = _ggufcompat.detect_gguf_layout(_ggufcompat.read_gguf_shapes(standard))
            self.assertEqual(layout.kind, 'kandinsky5')
            flattened = os.path.join(directory, 'flat.gguf')
            write(flattened, flatten=True)

            from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor
            for path in (standard, flattened):
                with self.subTest(path=os.path.basename(path)):
                    loaded = load(path, directory)
                    weight = loaded.visual_transformer_blocks[0].self_attention.to_query.weight
                    self.assertEqual(tuple(weight.quant_shape), (256, 256))
                    expected = model.visual_transformer_blocks[0].self_attention.to_query.weight
                    self.assertTrue(torch.allclose(
                        dequantize_gguf_tensor(weight).float(), expected, atol=0.01))
                    self.assertTrue(torch.equal(
                        loaded.out_layer.modulation.out_layer.weight.float(),
                        model.out_layer.modulation.out_layer.weight))

            corrupt = os.path.join(directory, 'corrupt.gguf')
            write(corrupt, corrupt=True)
            with self.assertRaisesRegex(ValueError, 'NaN/Inf'):
                load(corrupt, directory)

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

    def test_krea2_gguf_layout(self):
        layout = _ggufcompat.detect_gguf_layout({
            'txtfusion.projector.weight': (12,),
            'first.weight': (128, 64),
        })
        self.assertEqual(layout.kind, 'krea2')
        self.assertEqual(layout.config_repo, 'krea/Krea-2-Raw')
        self.assertTrue(layout.adapt)

    def test_krea2_projector_vector_is_aligned(self):
        checkpoint = {'txtfusion.projector.weight': torch.zeros(12)}
        with mock.patch(
                'diffusers.loaders.single_file_utils.convert_krea2_transformer_checkpoint_to_diffusers',
                return_value={'text_fusion.projector.weight': checkpoint['txtfusion.projector.weight']}), \
             mock.patch.object(
                 _ggufcompat, '_target_shapes',
                 return_value={'text_fusion.projector.weight': (1, 12)}):
            adapted = _ggufcompat.adapt_krea2_checkpoint(
                checkpoint, 'krea/Krea-2-Raw', 'transformer')
        self.assertEqual(tuple(adapted['text_fusion.projector.weight'].shape), (1, 12))

    def test_hunyuan_video15_converter_maps_15_components(self):
        source = {
            'time_text_embed.timestep_embedder.linear_1.weight': torch.zeros(2, 2),
            'cond_type_embedding.weight': torch.zeros(3, 2),
            'vision_in.proj.0.weight': torch.zeros(2, 2),
            'byt5_in.net.0.proj.weight': torch.zeros(2, 2),
            'transformer_blocks.0.norm1.weight': torch.zeros(2),
        }
        with mock.patch(
                'diffusers.loaders.single_file_utils.convert_hunyuan_video_transformer_to_diffusers',
                side_effect=lambda checkpoint, **kwargs: checkpoint):
            converted = _singlefilecompat.convert_hunyuan_video15_transformer_checkpoint(source)
        self.assertIn('time_embed.timestep_embedder.linear_1.weight', converted)
        self.assertIn('cond_type_embed.weight', converted)
        self.assertIn('image_embedder.norm_in.weight', converted)
        self.assertIn('context_embedder_2.linear_1.weight', converted)
        self.assertIn('transformer_blocks.0.norm1.weight', converted)

    def test_ltx25_converter_maps_prompt_adaln(self):
        import diffusers.loaders.single_file_model as single_file_model

        source = {
            'model.diffusion_model.adaln_single.linear.weight': torch.zeros(2, 2),
            'model.diffusion_model.prompt_adaln_single.linear.weight': torch.zeros(2, 2),
            'model.diffusion_model.audio_prompt_adaln_single.emb.timestep_embedder.linear_1.bias':
                torch.zeros(2),
        }
        converted = _singlefilecompat.convert_ltx2_transformer_checkpoint(source)
        self.assertEqual(set(converted), {
            'time_embed.linear.weight',
            'prompt_adaln.linear.weight',
            'audio_prompt_adaln.emb.timestep_embedder.linear_1.bias',
        })

        _singlefilecompat.register_single_file_classes()
        entry = single_file_model.SINGLE_FILE_LOADABLE_CLASSES['LTX2VideoTransformer3DModel']
        self.assertIs(entry['checkpoint_mapping_fn'],
                      _singlefilecompat.convert_ltx2_transformer_checkpoint)

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

        flf = _ggufcompat.detect_gguf_layout({
            'head.modulation': (5120,),
            'patch_embedding.weight': (5120, 36, 2, 2),
            'img_emb.emb_pos': (1, 514, 1280),
        })
        self.assertEqual(flf.kind, 'wan-flf2v-14B')
        self.assertEqual(flf.config_repo, 'Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers')
        self.assertTrue(flf.adapt)

        flf_comfy = _ggufcompat.detect_gguf_layout({
            'model.diffusion_model.head.modulation': (5120,),
            'model.diffusion_model.patch_embedding.weight': (5120, 36, 2, 2),
            'model.diffusion_model.img_emb.emb_pos': (1, 514, 1280),
        })
        self.assertEqual(flf_comfy.kind, 'wan-flf2v-14B')
        self.assertTrue(flf_comfy.comfy)

        animate2_comfy = _ggufcompat.detect_gguf_layout({
            'model.diffusion_model.head.modulation': (5120,),
            'model.diffusion_model.patch_embedding.weight': (5120, 36, 1, 2, 2),
            'model.diffusion_model.blocks.0.block.self_attn.q.weight': (5120, 5120),
        })
        self.assertEqual(animate2_comfy.kind, 'wan-animate-2-14B')
        self.assertTrue(animate2_comfy.comfy)

    def test_wan_flf_maps_emb_pos(self):
        table = torch.arange(514 * 1280, dtype=torch.float32).reshape(1, 514, 1280)
        adapted = _ggufcompat.adapt_wan_flf_checkpoint({
            'img_emb.emb_pos': table,
            'patch_embedding.weight': torch.zeros(1),
        })
        self.assertNotIn('img_emb.emb_pos', adapted)
        mapped = adapted['condition_embedder.image_embedder.pos_embed']
        self.assertEqual(tuple(mapped.shape), (1, 514, 1280))
        self.assertTrue(torch.equal(mapped, table))

        flat = torch.arange(514 * 1280, dtype=torch.float32).reshape(514, 1280)
        comfy = _ggufcompat.adapt_wan_flf_checkpoint({
            'model.diffusion_model.img_emb.emb_pos': flat,
        })
        mapped = comfy['condition_embedder.image_embedder.pos_embed']
        self.assertEqual(tuple(mapped.shape), (1, 514, 1280))
        self.assertTrue(torch.equal(mapped.reshape(514, 1280), flat))

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

    def test_dequantized_gguf_weight_is_stored_as_parameter(self):
        import gguf
        from diffusers import GGUFQuantizationConfig
        from diffusers.quantizers.gguf.gguf_quantizer import GGUFQuantizer
        from diffusers.quantizers.gguf.utils import GGML_QUANT_SIZES, GGUFParameter

        _ggufcompat.install_gguf_patches()
        quant_type = gguf.GGMLQuantizationType.Q8_0
        _block_size, type_size = GGML_QUANT_SIZES[quant_type]
        packed = torch.zeros(4, type_size, dtype=torch.uint8)
        fp32 = GGUFParameter(packed.clone(), quant_type=quant_type)
        packed_weight = GGUFParameter(packed.clone(), quant_type=quant_type)

        parent = torch.nn.Module()
        parent.norm1 = torch.nn.Linear(32, 4, bias=False)
        parent.attn = torch.nn.Linear(32, 4, bias=False)

        config = GGUFQuantizationConfig(compute_dtype=torch.float32)
        quantizer = GGUFQuantizer(config)
        quantizer.modules_to_not_convert = ['norm1']
        quantizer.keep_in_fp32_modules = ['norm1']
        quantizer.compute_dtype = torch.float32

        quantizer.create_quantized_param(
            parent, fp32, 'norm1.weight', torch.device('cpu'))
        quantizer.create_quantized_param(
            parent, packed_weight, 'attn.weight', torch.device('cpu'))

        stored = parent.norm1._parameters['weight']
        self.assertIsInstance(stored, torch.nn.Parameter)
        self.assertEqual(type(stored).__name__, 'Parameter')
        self.assertFalse(hasattr(stored, 'quant_type') and stored.quant_type)
        self.assertEqual(type(parent.attn.weight).__name__, 'GGUFParameter')
        self.assertEqual(parent.attn.weight.quant_type, quant_type)


if __name__ == '__main__':
    unittest.main()
