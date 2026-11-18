import os
import tempfile
import types
import unittest

import torch

import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.featuredefaults as _featuredefaults
import dgenerate.pipelinewrapper.ggufcompat as _ggufcompat
import dgenerate.pipelinewrapper.minimaxh3 as _minimax
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.videopipelines as _videopipelines
import dgenerate.quantize.comfykeys as _comfykeys
import dgenerate.quantize.quantexport as _quantexport


def _parsed(**values):
    parsed = types.SimpleNamespace(
        images=None, end_image=None, reference_images=None, mask_images=None,
        control_images=None, adapter_images=None, floyd_image=None,
        latents=None, audio_latents=None, multi_image_mode=False,
        wan_pose_video=None, wan_face_video=None, wan_background_video=None,
        wan_driving_video=None)
    for name, value in values.items():
        setattr(parsed, name, value)
    return parsed


class TestMiniMaxH3(unittest.TestCase):
    def test_video_prompt_enhance_uses_the_loaded_encoder(self):
        import dgenerate.pipelinewrapper.videopipelines.minimax as _call
        import dgenerate.pipelinewrapper.videopipelines.media as _media

        prompt = types.SimpleNamespace(positive='a fox', negative=None)
        user_args = types.SimpleNamespace(
            sigmas=None, timesteps=None, video_frames=None, end_video_frames=None,
            prompt=prompt, guidance_scale=1.0, video_fps=24, inference_steps=30,
            images=None, end_images=None, reference_media_paths=None,
            scheduler_uri=None, second_model_scheduler_uri=None,
            video_length=5, width=1024, height=576, vae_slicing=False,
            latents=None, audio_latents=None, pipeline_prompt_enhance=True, seed=1)
        wrapper = types.SimpleNamespace(
            controlnet_uris=None, model_cpu_offload=True,
            model_sequential_offload=False, model_group_offload=False, device='cpu')
        pipe = object()

        def enhance(wrapper, pipe_arg, args, references=None):
            self.assertIs(pipe_arg, pipe)
            self.assertIsNone(references)
            args.prompt = types.SimpleNamespace(positive='rewritten fox', negative=None)

        def invoke(wrapper, pipe_arg, kwargs, user_args=None):
            self.assertEqual(kwargs['prompt'], 'rewritten fox')
            return {'videos': [], 'audio': None, 'sampling_rate': 24000}

        with unittest.mock.patch.object(
                _videopipelines, '_video_pipeline', return_value=(pipe, None)), \
                unittest.mock.patch(
                    'dgenerate.pipelinewrapper.videopipelines.single_image_video._enhance_loaded_video_prompt',
                    enhance), \
                unittest.mock.patch.object(_videopipelines, '_invoke', invoke), \
                unittest.mock.patch.object(_media, 'frames_from_output', return_value=[]), \
                unittest.mock.patch.object(_media, 'audio_to_numpy', return_value=None):
            _call.call_minimax_h3(wrapper, user_args)

    def test_model_type_registration(self):
        self.assertEqual(
            _enums.get_model_type_enum('minimax-h3-video'), _enums.ModelType.MINIMAX_H3)
        self.assertIn('minimax-h3-video', _enums.supported_model_type_strings())
        self.assertTrue(_enums.model_type_is_video_family('minimax-h3-video'))
        self.assertTrue(_enums.model_type_is_minimax_h3('minimax-h3-video'))
        self.assertIn('minimax-h3-video', _pipelinewrapper.CACHE_DIT_MODEL_TYPES)
        self.assertEqual(_featuredefaults.video_fps_default('minimax-h3-video'), 24.0)
        schema = __import__(
            'dgenerate.pipelinewrapper.textencoderschema', fromlist=['text_encoder_schema']
        ).text_encoder_schema()
        self.assertEqual(schema['minimax-h3-video'][0]['name'], 'text_encoder')
        self.assertIn('Qwen3VLForConditionalGeneration', schema['minimax-h3-video'][0]['encoders'])

    def test_workflow_from_the_image_seed(self):
        self.assertEqual(_videopipelines.classify_video_seed('minimax-h3-video', None), 't2va')
        self.assertEqual(
            _videopipelines.classify_video_seed('minimax-h3-video', _parsed(images=['first.png'])),
            'fl2va')
        self.assertEqual(
            _videopipelines.classify_video_seed('minimax-h3-video', _parsed(end_image='last.png')),
            'fl2va')
        self.assertEqual(
            _videopipelines.classify_video_seed(
                'minimax-h3-video', _parsed(reference_images=['a.png', 'b.mp4', 'c.wav'])),
            'ref2va')
        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError):
            _videopipelines.classify_video_seed(
                'minimax-h3-video', _parsed(images=['first.png'], reference_images=['a.png']))
        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError):
            _videopipelines.classify_video_seed(
                'minimax-h3-video', _parsed(reference_images=['only.wav']))
        self.assertEqual(
            _videopipelines.classify_video_seed(
                'minimax-h3-video', _parsed(images=['canny.mp4']), controlnet=True),
            't2va')
        self.assertEqual(
            _videopipelines.classify_video_seed(
                'minimax-h3-video',
                _parsed(images=['source.mp4'], mask_images=['mask.mp4']),
                controlnet=True),
            't2va')
        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError):
            _videopipelines.classify_video_seed(
                'minimax-h3-video',
                _parsed(images=['first.png'], control_images=['canny.mp4']),
                controlnet=True)
        bare = _parsed(images=['canny.mp4'], multi_image_mode=False)
        bare.is_single_spec = True
        self.assertEqual(
            _videopipelines.video_seed_slots(bare, control_guidance=True),
            (None, None, 'canny.mp4'))

    def test_controlnet_is_allowed_on_minimax_video(self):
        import dgenerate.renderloopconfig as _renderloopconfig

        gif = 'examples/media/rickroll-roll.gif'
        uri = ('alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0;'
               'weight-name=MiniMax-H3-Fun-Controlnet-Union-2.0.safetensors')
        config = _renderloopconfig.RenderLoopConfig()
        config.model_path = 'MiniMaxAI/MiniMax-H3'
        config.model_type = _enums.ModelType.MINIMAX_H3
        config.controlnet_uris = [uri]
        config.image_seeds = [gif]
        config.control_image_processors = [
            'openpose;include-hand=true;include-face=true']
        config.quantizer_uri = 'bnb;bits=8'
        config.quantizer_map = ['controlnet']
        config.check()

        blocked = _renderloopconfig.RenderLoopConfig()
        blocked.model_path = 'org/ltx'
        blocked.model_type = _enums.ModelType.LTX2
        blocked.controlnet_uris = [uri]
        blocked.image_seeds = [gif]
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as caught:
            blocked.check()
        self.assertIn('cannot be used with --model-type ltx2', str(caught.exception))

    def test_frame_count_stays_inside_five_to_fifteen_seconds(self):
        self.assertEqual(_minimax.frames_for_seconds(5), 124)
        self.assertEqual(
            round(_minimax.DEFAULT_SECONDS * _minimax.FPS),
            _minimax.frames_for_seconds(_minimax.DEFAULT_SECONDS))
        self.assertLessEqual(_minimax.frames_for_seconds(15) / 24.0, 15.0)
        self.assertGreaterEqual(_minimax.frames_for_seconds(15) / 24.0, 5.0)
        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError):
            _minimax.frames_for_seconds(4)
        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError):
            _minimax.frames_for_seconds(16)

    def test_preprocessed_reference_keeps_its_own_size(self):
        from PIL import Image

        image = Image.new('RGB', (512, 320))
        frames = [Image.new('RGB', (256, 256))]
        _references, plan = _minimax.build_references([
            {'kind': 'image', 'frames': [image], 'honor_size': True},
            {'kind': 'video', 'frames': frames, 'fps': 24.0, 'honor_size': True},
        ])
        self.assertEqual(plan['images'], [320])
        self.assertEqual(plan['videos'], [{'edge': 256, 'pixels': 256 * 256}])

        class _Config:
            reference_image_short_edge = 2048

        class _Pipe:
            config = _Config()

        with _minimax.honor_reference_canvas(_Pipe(), plan):
            self.assertEqual(_Pipe.config.reference_image_short_edge / 320, 1.0)
            self.assertAlmostEqual(
                _Pipe.config.reference_image_short_edge / 100, 2048 / 100)
        self.assertEqual(_Pipe.config.reference_image_short_edge, 2048)

    def test_comfy_names_round_trip_through_diffusers(self):
        import diffusers

        config = {
            'num_attention_heads': 2,
            'attention_head_dim': 16,
            'hidden_size': 32,
            'num_layers': 1,
            'num_refiner_layers': 1,
            'ffn_dim': 32,
            'in_channels': 4,
            'audio_in_channels': 4,
            'patch_size': [1, 2, 2],
            'text_dim': 16,
            'freq_dim': 8,
            'time_embed_hidden_dim': 16,
            'time_embed_dim': 16,
            'rope_freq_dim': 16,
        }
        model = diffusers.MiniMaxH3Transformer3DModel.from_config(config)
        original = {key: value.detach().clone() for key, value in model.state_dict().items()}
        comfy = _minimax.diffusers_to_comfy(original)
        self.assertIn('blocks.0.attn.qkv_proj.weight', comfy)
        self.assertIn('video_patch_proj.weight', comfy)
        self.assertIn('rope.inv_freq', comfy)
        self.assertNotIn('transformer_blocks.0.attn.to_q.weight', comfy)
        restored = _minimax.comfy_to_diffusers(comfy)
        self.assertNotIn('rope.inv_freq', restored)
        for key, value in original.items():
            self.assertIn(key, restored)
            self.assertTrue(torch.equal(restored[key], value), key)
        exported, label = _comfykeys.comfy_state(
            original, {'_class_name': 'MiniMaxH3Transformer3DModel'}, gguf=False)
        self.assertEqual(label, 'minimax_h3')
        self.assertIn('blocks.0.attn.qkv_proj.weight', exported)

    def test_quantize_comfy_keeps_published_minimax_names(self):
        import dgenerate.quantize.comfykeys as _comfykeys
        import dgenerate.quantize.quantexport as _quantexport

        published = {
            'video_patch_proj.weight': torch.zeros(4, 8),
            'blocks.0.attn.qkv_proj.weight': torch.zeros(12, 4),
        }
        exported, label = _comfykeys.comfy_state(
            published, {'_class_name': 'MiniMaxH3DiTModel'}, gguf=False)
        self.assertEqual(label, 'minimax_h3')
        self.assertIn('blocks.0.attn.qkv_proj.weight', exported)
        self.assertIn('video_patch_proj.weight', exported)
        with self.assertRaises(_quantexport.QuantExportError):
            _comfykeys.comfy_state(
                {'encoder.conv_in.weight': torch.zeros(4, 3, 3, 3)},
                {'_class_name': 'AutoencoderKLMiniMaxH3'},
                gguf=False)

    def test_empty_tensors_gain_storage_and_rope_is_restored(self):
        import torch.nn as nn

        class _Rope(nn.Module):
            def __init__(self):
                super().__init__()
                self.rope_freq_dim = 4
                self.register_buffer('inv_freq', torch.empty(4, device='meta'), persistent=False)

        class _Host(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type('C', (), {'rope_theta': 10000.0})()
                self.linear = nn.Linear(2, 2)
                self.linear.weight = nn.Parameter(torch.empty(2, 2, device='meta'))
                self.rope = _Rope()

        host = _Host()
        missing = _minimax._materialize_meta_tensors(host)
        self.assertIn('linear.weight', missing)
        self.assertEqual(host.linear.weight.device.type, 'cpu')
        self.assertEqual(host.rope.inv_freq.device.type, 'cpu')
        self.assertGreater(float(host.rope.inv_freq.abs().sum()), 0)

    def test_image_vae_rope_is_restored_before_a_device_move(self):
        import torch.nn as nn

        class MiniMaxH3ImageRotaryPosEmbed(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('inv_freq', torch.empty(8, device='meta'), persistent=False)

        class _Vae(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type('C', (), {
                    'decoder_attention_head_dim': 64,
                    'decoder_rope_dim_ratio': 0.5,
                    'decoder_rope_theta': 100.0,
                })()
                self.rope = MiniMaxH3ImageRotaryPosEmbed()
                self.norm = nn.Linear(2, 2)
                self.norm.weight = nn.Parameter(torch.empty(2, 2, device='meta'))

        vae = _Vae()

        class _Pipe:
            components = {'vae': vae}

        _minimax.prepare_device_move(_Pipe())
        self.assertEqual(vae.rope.inv_freq.device.type, 'cpu')
        self.assertEqual(tuple(vae.rope.inv_freq.shape), (8,))
        self.assertGreater(float(vae.rope.inv_freq.abs().sum()), 0)
        self.assertEqual(vae.norm.weight.device.type, 'meta')

    def test_curve_embedder_does_not_contain_the_transformer(self):
        import torch.nn as nn

        class _Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.adaln_proj = nn.Linear(4, 24)
                self.adaln_proj.hidden_size = 4

        class _Host(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([_Block()])
                self.norm_out = nn.Linear(4, 8)

        host = _Host()
        _minimax._install_curve(host, torch.zeros(4, 2))
        self.assertNotIn(host, host.time_embedder.modules())
        self.assertEqual(host.adaln_t_table.shape, (4, 2))
        host.state_dict()
        import diffusers.loaders.single_file_model as single_file
        original = single_file.dispatch_model
        with _minimax.keep_missing_weights_addressable():
            self.assertIsNot(single_file.dispatch_model, original)
        self.assertIs(single_file.dispatch_model, original)

    def test_gguf_layout_detects_diffusers_and_comfy_names(self):
        diffusers_layout = _ggufcompat.detect_gguf_layout({
            'audio_proj_in.weight': (32, 4),
            'transformer_blocks.0.attn.to_q.weight': (32, 32),
        })
        self.assertEqual(diffusers_layout.kind, 'minimax-h3')
        self.assertEqual(diffusers_layout.config_repo, 'MiniMaxAI/MiniMax-H3')
        comfy_layout = _ggufcompat.detect_gguf_layout({
            'model.diffusion_model.video_patch_proj.weight': (32, 16),
            'model.diffusion_model.audio_patch_proj.weight': (32, 8),
            'model.diffusion_model.blocks.0.attn.qkv_proj.weight': (96, 32),
        })
        self.assertEqual(comfy_layout.kind, 'minimax-h3')
        self.assertTrue(comfy_layout.comfy)
        self.assertTrue(comfy_layout.adapt)
        image_layout = _ggufcompat.detect_gguf_layout({
            'video_patch_proj.weight': (96, 5376),
            'blocks.0.attn.qkv_proj.weight': (5376, 21504),
            'final_layer.video_out.weight': (5376, 576),
        })
        self.assertEqual(image_layout.kind, 'minimax-h3-image')
        self.assertEqual(image_layout.config_repo, 'MiniMaxAI/MiniMax-H3-Image')

    def test_pruned_video_checkpoint_keeps_the_curve(self):
        import dgenerate.pipelinewrapper.singlefilecompat as _singlefilecompat
        import diffusers.loaders.single_file_model as single_file

        _singlefilecompat.register_single_file_classes()
        mapping = single_file.SINGLE_FILE_LOADABLE_CLASSES['MiniMaxH3Transformer3DModel']
        self.assertIs(
            mapping['checkpoint_mapping_fn'],
            _singlefilecompat.convert_minimax_h3_transformer_checkpoint)
        converted = mapping['checkpoint_mapping_fn']({
            'adaln_t_table': torch.zeros(4, 8),
            'video_patch_proj.weight': torch.zeros(4, 2),
        })
        self.assertNotIn('adaln_t_table', converted)
        self.assertIsNotNone(_minimax.convert_minimax_h3_checkpoint.last_curve_table)

    def test_packed_lora_adds_the_low_rank_residual(self):
        import torch.nn as nn

        linear = nn.Linear(4, 6, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.eye(6, 4))
        down = torch.zeros(2, 4)
        down[0, 0] = 1
        up = torch.zeros(6, 2)
        up[0, 0] = 3
        _minimax._attach_lora_delta(linear, down, up, 1.0)
        sample = torch.zeros(1, 4)
        sample[0, 0] = 1
        output = linear(sample).detach()
        self.assertAlmostEqual(float(output[0, 0]), 4.0, places=5)
        self.assertAlmostEqual(float(output[0, 1]), 0.0, places=5)

    def test_comfy_character_lora_names_convert_onto_the_diffusers_linears(self):
        from diffusers.loaders.lora_conversion_utils import (
            _convert_non_diffusers_minimax_h3_lora_to_diffusers)

        converted = _convert_non_diffusers_minimax_h3_lora_to_diffusers({
            'diffusion_model.blocks.0.attn.out_proj.lora_A.weight': torch.zeros(2, 4),
            'diffusion_model.blocks.0.attn.out_proj.lora_B.weight': torch.zeros(4, 2),
            'diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight': torch.zeros(2, 4),
            'diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight': torch.zeros(6, 2),
        })
        self.assertIn(
            'transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight', converted)
        self.assertIn(
            'transformer.transformer_blocks.0.attn.to_q.lora_B.weight', converted)
        self.assertEqual(
            tuple(converted['transformer.transformer_blocks.0.attn.to_q.lora_B.weight'].shape),
            (2, 2))

    def test_scheduler_arguments_overlay_video_and_audio_separately(self):
        import diffusers
        import dgenerate.pipelinewrapper.videopipelines.minimax as _video

        video = diffusers.MiniMaxH3Scheduler(shift=12.0)
        audio = diffusers.MiniMaxH3Scheduler(shift=3.0)

        class _Pipe:
            pass

        pipe = _Pipe()
        pipe.scheduler = video
        pipe.audio_scheduler = audio
        _video._configure_schedulers(
            pipe,
            'MiniMaxH3Scheduler;shift=8;stochastic-sampling=true',
            'MiniMaxH3Scheduler;shift=1.5')
        self.assertEqual(pipe.scheduler.config.shift, 8.0)
        self.assertTrue(pipe.scheduler.config.stochastic_sampling)
        self.assertEqual(pipe.audio_scheduler.config.shift, 1.5)
        self.assertFalse(pipe.audio_scheduler.config.stochastic_sampling)

    def test_quantizer_lora_fuses_to_disk_only_when_asked(self):
        plan = _minimax._quantized_denoiser_lora_plan
        self.assertEqual(
            plan(['org/lora'], 'bnb', None, False, 'transformer', None), 'refuse')
        self.assertEqual(
            plan(['org/lora'], 'bnb', None, True, 'transformer', None), 'fuse')
        self.assertEqual(
            plan(['org/lora'], 'bnb', 'model.gguf', True, 'transformer', None), 'attach')
        self.assertEqual(
            plan(None, 'bnb', None, True, 'transformer', None), 'attach')
        self.assertEqual(
            plan(['org/lora'], 'bnb', None, True, 'transformer_ref', None), 'fuse')

    def test_cache_dit_adapter_uses_pattern_3(self):
        import cache_dit

        class MiniMaxH3Transformer3DModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity()])

        pipe = types.SimpleNamespace(transformer=MiniMaxH3Transformer3DModel())
        adapter = _pipelines._cache_dit_custom_block_adapter(cache_dit, pipe)
        self.assertIs(adapter.forward_pattern, cache_dit.ForwardPattern.Pattern_3)
        self.assertFalse(adapter.has_separate_cfg)
        ref_pipe = types.SimpleNamespace(
            transformer=None, transformer_ref=MiniMaxH3Transformer3DModel())
        ref_adapter = _pipelines._cache_dit_custom_block_adapter(cache_dit, ref_pipe)
        self.assertIs(ref_adapter.transformer, ref_pipe.transformer_ref)

    def test_published_comfy_and_curve_layouts(self):
        qkv = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        scale = torch.arange(6, dtype=torch.float32).reshape(6, 1)
        marker = torch.tensor([1, 2, 3], dtype=torch.uint8)
        table = torch.arange(8 * 1025, dtype=torch.float32).reshape(8, 1025)
        converted = _minimax.comfy_to_diffusers({
            'adaln_t_table': table,
            'video_patch_proj.weight': torch.ones(4, 4),
            'blocks.0.adaln_proj.linear.weight': torch.ones(64, 8),
            'blocks.0.adaln_proj.folded_bias': torch.arange(64, dtype=torch.float32),
            'blocks.0.attn.qkv_proj.weight': qkv,
            'blocks.0.attn.qkv_proj.weight_scale': scale,
            'blocks.0.attn.qkv_proj.comfy_quant': marker,
            'extra.not_a_weight': torch.zeros(1),
        })
        self.assertNotIn('adaln_t_table', converted)
        self.assertEqual(tuple(_minimax.convert_minimax_h3_checkpoint.last_curve_table.shape), (1025, 8))
        self.assertEqual(tuple(converted['transformer_blocks.0.attn.to_q.weight'].shape), (2, 2))
        self.assertTrue(torch.equal(
            converted['transformer_blocks.0.attn.to_q.weight'], qkv[:2]))
        self.assertEqual(tuple(converted['transformer_blocks.0.attn.to_v.weight_scale'].shape), (2, 1))
        self.assertTrue(torch.equal(
            converted['transformer_blocks.0.attn.to_k.comfy_quant'], marker))
        self.assertTrue(torch.equal(
            converted['transformer_blocks.0.adaln_proj.linear.bias'],
            torch.arange(64, dtype=torch.float32)))
        self.assertEqual(_minimax.curve_time_embed_dim(converted), 8)

        full = _minimax.comfy_to_diffusers({
            'video_patch_proj.weight': torch.ones(4, 4),
            'time_embedder.proj_in.weight': torch.ones(4, 2),
            'blocks.0.attn.qkv_proj.weight': qkv,
        })
        self.assertIn('time_embedder.linear_1.weight', full)
        self.assertIsNone(_minimax.convert_minimax_h3_checkpoint.last_curve_table)
        self.assertIsNone(_minimax.curve_time_embed_dim(full))

    def test_empty_parameters_can_be_copied_for_offload(self):
        import accelerate
        import torch.nn as nn

        with accelerate.init_empty_weights():
            layer = nn.Linear(4, 4)
        missing = _minimax._materialize_meta_tensors(layer)
        self.assertIn('weight', missing)
        copied = {name: tensor.to('cpu') for name, tensor in layer.state_dict().items()}
        self.assertEqual(tuple(copied['weight'].shape), (4, 4))
        self.assertFalse(copied['weight'].is_meta)

    def test_comfy_swiglu_stores_the_gate_first(self):
        gate = torch.ones(4, 2)
        value = torch.full((4, 2), 2.0)
        converted = _minimax.comfy_to_diffusers({
            'blocks.0.mlp.fc1.weight': torch.cat([gate, value], dim=0),
        })
        weight = converted['transformer_blocks.0.ff.net.0.proj.weight']
        self.assertTrue(torch.equal(weight[:4], value))
        self.assertTrue(torch.equal(weight[4:], gate))
        self.assertTrue(torch.equal(
            _minimax._swap_swiglu_halves(weight), torch.cat([gate, value], dim=0)))

    def test_curve_adaln_reads_a_bf16_gguf_weight(self):
        import gguf
        import torch.nn as nn
        from diffusers.quantizers.gguf.utils import GGUFParameter

        class _Host(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList()
                self.norm_out = nn.Linear(4, 8)
                linear = nn.Linear(8, 24, bias=False)
                values = torch.zeros(24, 8, dtype=torch.bfloat16)
                values[0, 0] = 1
                linear.weight = GGUFParameter(
                    values.view(torch.uint8),
                    quant_type=gguf.GGMLQuantizationType.BF16)

                class _Proj(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.hidden_size = 4
                        self.linear = linear

                self.adaln_proj = _Proj()
                self.transformer_blocks.append(self)

        host = _Host()
        _minimax._install_curve(host, torch.zeros(4, 8))
        temb = torch.zeros(1, 8)
        temb[0, 0] = -2
        shift = host.adaln_proj(temb)[0]
        self.assertTrue(torch.allclose(shift[0, 0].float(), torch.tensor(-2.0)))

    def test_curve_adaln_reads_a_bf16_gguf_bias(self):
        import gguf
        import torch.nn as nn
        from diffusers.quantizers.gguf.utils import GGUFParameter

        class _Host(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList()
                self.norm_out = nn.Linear(4, 8)
                linear = nn.Linear(8, 24)
                linear.weight = nn.Parameter(torch.zeros(24, 8))
                bias = torch.zeros(24, dtype=torch.bfloat16)
                bias[0] = 5
                linear.bias = GGUFParameter(
                    bias.view(torch.uint8),
                    quant_type=gguf.GGMLQuantizationType.BF16)

                class _Proj(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.hidden_size = 4
                        self.linear = linear

                self.adaln_proj = _Proj()
                self.transformer_blocks.append(self)

        host = _Host()
        _minimax._install_curve(host, torch.zeros(4, 8))
        shift = host.adaln_proj(torch.zeros(1, 8))[0]
        self.assertTrue(torch.allclose(shift[0, 0].float(), torch.tensor(5.0)))

    def test_curve_table_replaces_the_time_embedder(self):
        import diffusers

        config = {
            'num_attention_heads': 2,
            'attention_head_dim': 16,
            'hidden_size': 32,
            'num_layers': 1,
            'num_refiner_layers': 1,
            'ffn_dim': 32,
            'in_channels': 4,
            'audio_in_channels': 4,
            'patch_size': [1, 2, 2],
            'text_dim': 16,
            'freq_dim': 8,
            'time_embed_hidden_dim': 16,
            'time_embed_dim': 8,
            'rope_freq_dim': 16,
        }
        model = diffusers.MiniMaxH3Transformer3DModel.from_config(config)
        table = torch.zeros(5, 8)
        table[1] = 2
        _minimax._install_curve(model, table)
        sampled = model.time_embedder(model.time_proj(torch.tensor([0.25])))
        self.assertTrue(torch.allclose(sampled, table[1]))
        projection = model.transformer_blocks[0].adaln_proj
        with torch.no_grad():
            projection.linear.weight.zero_()
            projection.linear.bias.zero_()
            projection.linear.weight[0, 0] = 1
        temb = torch.zeros(1, 8)
        temb[0, 0] = -2
        shift = projection(temb)[0]
        self.assertTrue(torch.allclose(shift[0, 0], torch.tensor(-2.0)))

    def test_modular_index_selects_component_folders(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ('transformer', 'transformer_ref', 'text_encoder', 'tokenizer'):
                folder = os.path.join(root, name)
                os.makedirs(folder)
                if name != 'tokenizer':
                    open(os.path.join(folder, 'diffusion_pytorch_model.safetensors'), 'wb').close()
            with open(os.path.join(root, 'modular_model_index.json'), 'w', encoding='utf-8') as handle:
                handle.write(
                    '{"_class_name": "MiniMaxH3ModularPipeline", '
                    '"transformer": ["diffusers", "MiniMaxH3Transformer3DModel", {}], '
                    '"transformer_ref": ["diffusers", "MiniMaxH3Transformer3DModel", {}], '
                    '"text_encoder": ["transformers", "Qwen3VLForConditionalGeneration", {}], '
                    '"tokenizer": ["transformers", "Qwen2Tokenizer", {}]}')
            found = dict(_quantexport.component_directories(root))
        self.assertEqual(set(found), {'transformer', 'transformer_ref', 'text_encoder'})

    def test_control_branch_layout_and_identity_forward(self):
        import torch.nn as nn
        import diffusers

        import dgenerate.pipelinewrapper.minimaxh3control as _control

        self.assertEqual(
            _control.checkpoint_time_embed_dim(
                {'control_blocks.0.adaln_proj.linear.weight': torch.zeros(4, 2688)}),
            2688)
        places, mode = _control.branch_layout(
            [f'control_blocks.{index}.after_proj.weight' for index in range(10)]
            + ['control_proj_in.weight'])
        self.assertEqual(places, (0, 5, 10, 15, 20, 25, 30, 35, 40, 45))
        self.assertEqual(mode, 'post_norm')
        with self.assertRaises(_pipelinewrapper.UnsupportedPipelineConfigError):
            _control.branch_layout(['control_blocks.0.after_proj.weight', 'control_proj_in.weight'])

        rows = torch.zeros(2, 16)
        padded = _control.pad_control_rows(rows, 49, (1, 2, 2))
        self.assertEqual(tuple(padded.shape), (2, 49 * 4))

        config = {
            'num_attention_heads': 2,
            'attention_head_dim': 24,
            'hidden_size': 48,
            'num_layers': 4,
            'num_refiner_layers': 1,
            'ffn_dim': 48,
            'in_channels': 4,
            'audio_in_channels': 4,
            'patch_size': [1, 2, 2],
            'text_dim': 16,
            'freq_dim': 8,
            'time_embed_hidden_dim': 16,
            'time_embed_dim': 8,
            'rope_freq_dim': 4,
        }
        model = diffusers.MiniMaxH3Transformer3DModel.from_config(config).eval()
        block_cls = _control._control_block_class()
        kwargs = dict(
            hidden_size=48, num_attention_heads=2, attention_head_dim=24,
            ffn_dim=48, time_embed_dim=8, norm_eps=1e-5, qk_norm_eps=1e-5)
        model.control_blocks = nn.ModuleList(block_cls(index, **kwargs) for index in range(2))
        model.control_proj_in = nn.Linear(16, 48)
        with torch.no_grad():
            for block in model.control_blocks:
                block.after_proj.weight.zero_()
                block.after_proj.bias.zero_()
        model._dgenerate_control_places = (0, 2)
        model._dgenerate_control_apply_audio = False
        hidden = torch.randn(1, 2, 16)
        audio = torch.randn(1, 2, 4)
        text = torch.randn(1, 2, 16)
        tags = torch.tensor([1, 1, 2, 2, 0, 0])
        steps = torch.zeros(6, dtype=torch.long)
        positions = torch.zeros(6, 3)
        videos = torch.tensor([4, 5])
        audios = torch.tensor([2, 3])
        texts = torch.tensor([0, 1])
        timestep = torch.tensor([0.4])
        base = model(
            hidden, audio, text, timestep, steps, tags, positions,
            videos, audios, texts, return_dict=False)
        controlled = _control._control_forward(
            model, hidden, audio, text, timestep, steps, tags, positions,
            videos, audios, texts, torch.randn(1, 2, 16), 1.0, None, False)
        self.assertTrue(torch.allclose(base[0], controlled[0], atol=1e-5))
        self.assertTrue(torch.allclose(base[1], controlled[1], atol=1e-5))
        with torch.no_grad():
            model.control_blocks[0].after_proj.weight.fill_(0.05)
        changed = _control._control_forward(
            model, hidden, audio, text, timestep, steps, tags, positions,
            videos, audios, texts, torch.randn(1, 2, 16), 1.0, None, False)
        self.assertFalse(torch.allclose(base[0], changed[0], atol=1e-5))
        state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items() if key.startswith('control_')}
        missing, unexpected = model.load_state_dict(state, strict=False)
        self.assertFalse([key for key in missing if key.startswith('control_')])
        self.assertFalse(list(unexpected))
        self.assertIn('control_blocks.0.before_proj.weight', state)
        self.assertNotIn('control_blocks.1.before_proj.weight', state)


if __name__ == '__main__':
    unittest.main()
