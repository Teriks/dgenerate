import types
import unittest
import unittest.mock

import torch

import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.pipelinewrapper.featuredefaults as _featuredefaults
import dgenerate.pipelinewrapper.flux3 as _flux3
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.ggufcompat as _ggufcompat
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.singlefilecompat as _singlefilecompat
import dgenerate.pipelinewrapper.textencoderschema as _textencoderschema
import dgenerate.pipelinewrapper.videopipelines.loading as _video_loading
import dgenerate.pipelinewrapper.videopipelines.single_image_video as _single_image_video


class TestFlux3Kandinsky6(unittest.TestCase):
    def test_model_type_strings(self):
        self.assertEqual(_enums.get_model_type_enum('flux3'), _enums.ModelType.FLUX3)
        self.assertEqual(_enums.get_model_type_string(_enums.ModelType.FLUX3), 'flux3')
        self.assertEqual(
            _enums.get_model_type_enum('kandinsky6-video'), _enums.ModelType.KANDINSKY6_VIDEO)
        self.assertIn('flux3', _enums.supported_model_type_strings())
        self.assertIn('kandinsky6-video', _enums.supported_model_type_strings())
        self.assertTrue(_enums.model_type_is_flux3('flux3'))
        self.assertFalse(_enums.model_type_is_flux('flux3'))
        self.assertFalse(_enums.model_type_is_flux2('flux3'))
        self.assertTrue(_enums.model_type_is_flow_image_family('flux3'))
        self.assertTrue(_enums.model_type_image_is_condition('flux3'))
        self.assertTrue(_enums.model_type_is_video_family('kandinsky6-video'))
        self.assertFalse(_enums.model_type_is_kandinsky5_video('kandinsky6-video'))
        self.assertTrue(_enums.model_type_supports_spectrum('flux3'))
        self.assertTrue(_enums.model_type_supports_spectrum('kandinsky6-video'))
        self.assertTrue(_enums.model_type_supports_dype('flux3'))
        self.assertTrue(_enums.model_type_supports_dype('kandinsky6-video'))
        self.assertTrue(_enums.model_type_supports_pipeline_prompt_enhance('kandinsky6-video'))
        self.assertIn('qwen3vl', _ggufcompat._TEXT_GGUF_ARCHITECTURES)
        self.assertIn('qwen2vl', _ggufcompat._TEXT_GGUF_ARCHITECTURES)
        self.assertEqual(_featuredefaults.video_fps_default('kandinsky6-video'), 24.0)

    def test_flux3_pipeline_rejects_mask_and_controlnet(self):
        self.assertIs(
            _pipelines.get_pipeline_class(_enums.ModelType.FLUX3),
            __import__('diffusers').Flux3ModularPipeline)
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _pipelines.get_pipeline_class(
                _enums.ModelType.FLUX3,
                pipeline_type=_enums.PipelineType.INPAINT)
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _pipelines.get_pipeline_class(
                _enums.ModelType.FLUX3, controlnet_uris=['org/controlnet'])

    def test_flux3_gguf_layout_and_key_rename(self):
        distilled = _ggufcompat.detect_gguf_layout({
            'model.diffusion_model.emb_in.image.weight': (3072, 128),
            'model.diffusion_model.ref_time_in.in_layer.weight': (3072, 256),
        })
        self.assertEqual(distilled.kind, 'flux3')
        self.assertTrue(distilled.adapt)
        self.assertEqual(distilled.config_repo, 'black-forest-labs/flux-3-image-klein')

        base = _ggufcompat.detect_gguf_layout({
            'x_embedders.image.weight': (3072, 128),
            'mode_blocks.image.0.attn.to_q.weight': (3072, 3072),
        })
        self.assertEqual(base.kind, 'flux3')
        self.assertFalse(base.adapt)
        self.assertEqual(base.config_repo, 'black-forest-labs/flux-3-image-klein-base')

        converted = _ggufcompat.adapt_flux3_checkpoint({
            'emb_in.image.weight': torch.zeros(4, 4),
            'txt_in.weight': torch.zeros(4, 4),
            'content_mode_blocks.image.0.q_proj.weight': torch.zeros(4, 4),
            'single_blocks.0.mlp_in.weight': torch.zeros(4, 4),
            'time_in.in_layer.weight': torch.zeros(4, 4),
            'final_layer.image.linear.weight': torch.zeros(4, 4),
        })
        self.assertIn('x_embedders.image.weight', converted)
        self.assertIn('context_embedder.weight', converted)
        self.assertIn('mode_blocks.image.0.attn.to_q.weight', converted)
        self.assertIn('single_transformer_blocks.0.ff.linear_in.weight', converted)
        self.assertIn('timestep_embedder.linear_1.weight', converted)
        self.assertIn('proj_out.weight', converted)

    def test_flux3_comfy_fused_qkv_is_split(self):
        import diffusers.loaders.single_file_model as single_file_model

        _singlefilecompat.register_single_file_classes()
        weight = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        separate = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        converted = _singlefilecompat.convert_flux3_transformer_checkpoint({
            'content_mode_blocks.image.0.qkv.weight': weight,
            'single_blocks.0.qkv.weight': weight.clone(),
            'txt_mode_blocks.0.qkv.weight': weight.clone(),
            'content_mode_blocks.image.1.q_proj.weight': separate,
        })
        self.assertTrue(torch.equal(
            converted['mode_blocks.image.0.attn.to_q.weight'], weight[:2]))
        self.assertTrue(torch.equal(
            converted['mode_blocks.image.0.attn.to_k.weight'], weight[2:4]))
        self.assertTrue(torch.equal(
            converted['mode_blocks.image.0.attn.to_v.weight'], weight[4:]))
        self.assertIn('single_transformer_blocks.0.attn.to_q.weight', converted)
        self.assertIn('txt_mode_blocks.0.attn.to_q.weight', converted)
        self.assertNotIn('mode_blocks.image.0.qkv.weight', converted)
        self.assertTrue(torch.equal(
            converted['mode_blocks.image.1.attn.to_q.weight'], separate))
        mapping = single_file_model.SINGLE_FILE_LOADABLE_CLASSES[
            'Flux3Transformer2DModel']['checkpoint_mapping_fn']
        self.assertIs(mapping, _singlefilecompat.convert_flux3_transformer_checkpoint)

    def test_kandinsky6_gguf_does_not_steal_kandinsky5(self):
        lite = _ggufcompat.detect_gguf_layout({
            'audio_embeddings.in_layer.weight': (1792, 20),
            'visual_embeddings.in_layer.weight': (1792, 132),
        })
        self.assertEqual(lite.kind, 'kandinsky6')
        self.assertEqual(
            lite.config_repo, 'kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers')
        pro = _ggufcompat.detect_gguf_layout({
            'model.diffusion_model.video_text_transformer_blocks.0.attn.to_q.weight': (4096, 4096),
            'model.diffusion_model.visual_embeddings.in_layer.weight': (4096, 132),
        })
        self.assertEqual(pro.kind, 'kandinsky6')
        self.assertTrue(pro.comfy)
        self.assertTrue(pro.adapt)
        weight = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
        aligned = _ggufcompat._align_tensor(weight, (4, 1))
        self.assertEqual(tuple(aligned.shape), (4, 1))
        modulation = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1)
        transposed = _ggufcompat._align_tensor(modulation, (4, 2))
        self.assertEqual(tuple(transposed.shape), (4, 2))
        import diffusers.loaders.single_file_model as single_file_model
        import dgenerate.pipelinewrapper.singlefilecompat as _singlefilecompat
        _singlefilecompat.register_single_file_classes()
        self.assertIn(
            'Kandinsky6Transformer3DModel',
            single_file_model.SINGLE_FILE_LOADABLE_CLASSES)
        import diffusers
        bound = _singlefilecompat.bind_from_single_file(diffusers.Kandinsky6Transformer3DModel)
        self.assertIs(bound.__self__, diffusers.Kandinsky6Transformer3DModel)
        self.assertEqual(
            pro.config_repo, 'kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers')
        kandinsky5 = _ggufcompat.detect_gguf_layout({
            'visual_transformer_blocks.0.visual_modulation.out_layer.weight': (1, 1792),
            'visual_transformer_blocks.0.self_attention.to_query.bias': (1792,),
        })
        self.assertEqual(kandinsky5.kind, 'kandinsky5')

    def test_kandinsky6_video_class(self):
        import diffusers
        family, pipeline_class, moe = _video_loading._resolve_video_load(
            _enums.ModelType.KANDINSKY6_VIDEO,
            {'_class_name': 'Kandinsky6TI2VAPipeline'},
            None)
        self.assertEqual(family, 'kandinsky6-video')
        self.assertIs(pipeline_class, diffusers.Kandinsky6TI2VAPipeline)
        self.assertFalse(moe)
        self.assertIs(
            _single_image_video._single_image_video_transformer_class('kandinsky6-video'),
            diffusers.Kandinsky6Transformer3DModel)
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            _video_loading._resolve_video_load(
                _enums.ModelType.KANDINSKY5_VIDEO,
                {'_class_name': 'Kandinsky6TI2VAPipeline'},
                None)
        schema = _textencoderschema.text_encoder_schema()
        self.assertEqual(schema['flux3'][0]['encoders'], ['Qwen3VLForConditionalGeneration'])
        names = [slot['name'] for slot in schema['kandinsky6-video']]
        self.assertEqual(names, ['text_encoder', 'text_encoder_2'])

    def test_flux3_latents_round_trip_to_flux2_layout(self):
        wrapper = _pipelinewrapper.PipelineWrapper.__new__(
            _pipelinewrapper.PipelineWrapper)
        spatial = torch.arange(32 * 8 * 8, dtype=torch.float32).reshape(1, 32, 8, 8)
        patchified = wrapper._patchify_flux2_latents(spatial)
        self.assertEqual(tuple(patchified.shape), (1, 128, 4, 4))
        restored = wrapper._unpack_flux3_latents(patchified[0])
        self.assertEqual(tuple(restored.shape), (32, 8, 8))
        self.assertTrue(torch.equal(restored, spatial[0]))
        repacked = wrapper._repack_flux3_latents(spatial)
        self.assertTrue(torch.equal(repacked, patchified))

    def test_upsampler_generate_uses_real_weights(self):
        module = torch.nn.Linear(2, 2)
        seen = {}

        def generate(**kwargs):
            seen['weight'] = module.weight.device.type
            seen['ids'] = kwargs['input_ids'].device.type
            seen['meta'] = module.weight.is_meta
            seen['streamer'] = kwargs.get('streamer')
            return torch.zeros(1)

        module.generate = generate
        _flux3._bind_upsampler_generate(module, 'cpu')
        module.generate(input_ids=torch.zeros(1, 1, dtype=torch.long))
        self.assertEqual(seen['weight'], 'cpu')
        self.assertEqual(seen['ids'], 'cpu')
        self.assertFalse(seen['meta'])
        self.assertEqual(module.weight.device.type, 'cpu')
        self.assertIsNotNone(seen['streamer'])

    def test_parked_upsampler_splits_onto_gpu_by_default(self):
        import contextlib

        import dgenerate.pipelinewrapper.promptenhance as _promptenhance

        module = torch.nn.Linear(2, 2)

        def generate(**kwargs):
            return torch.zeros(1)

        module.generate = generate
        entered = {}

        @contextlib.contextmanager
        def split(encoder, device, offload, **kwargs):
            entered['device'] = torch.device(device)
            entered['offload'] = offload
            entered['headroom'] = kwargs.get('headroom_bytes')
            yield

        with unittest.mock.patch.object(_promptenhance, '_fast_encoder', split):
            _flux3._bind_upsampler_generate(module, 'cuda')
            module.generate(input_ids=torch.zeros(1, 1, dtype=torch.long))
        self.assertEqual(entered['device'].type, 'cuda')
        self.assertTrue(entered['offload'])
        self.assertIsNone(entered['headroom'])
        self.assertEqual(module.weight.device.type, 'cpu')

    def test_prompt_upsampler_block_can_be_removed(self):
        class Blocks:
            def __init__(self):
                self.sub_blocks = {'prompt_upsampler': object(), 'denoise': object()}

        class Pipe:
            def __init__(self):
                self._blocks = Blocks()

        pipe = Pipe()
        _flux3._drop_prompt_upsampler_block(pipe)
        self.assertEqual(list(pipe._blocks.sub_blocks), ['denoise'])

    def test_denoiser_stays_unplaced_until_the_upsampler_returns(self):
        class Track(torch.nn.Linear):
            def __init__(self):
                super().__init__(1, 1)
                self.destinations = []

            def to(self, *args, **kwargs):
                if args:
                    self.destinations.append(args[0])
                return super().to(*args, **kwargs)

        class Step:
            def __call__(self, components, state):
                components.during = tuple(components.transformer.destinations)
                return components, state

        class Pipe:
            def __init__(self):
                self.transformer = Track()
                self.vae = Track()
                self.text_encoder = Track()
                self.prompt_upsampler = Track()
                self.prompt_upsampler.generate = lambda **kwargs: torch.zeros(1)
                self._blocks = types.SimpleNamespace(sub_blocks={'prompt_upsampler': Step()})
                self.during = None

            @property
            def components(self):
                return {
                    'transformer': self.transformer,
                    'vae': self.vae,
                    'text_encoder': self.text_encoder,
                    'prompt_upsampler': self.prompt_upsampler,
                }

        pipe = Pipe()
        _flux3.place_flux3(pipe, 'cpu', False, False, False)
        self.assertEqual(pipe.transformer.destinations, [])
        self.assertEqual(pipe.vae.destinations, [])
        self.assertEqual(pipe.text_encoder.destinations, [])
        pipe._blocks.sub_blocks['prompt_upsampler'](pipe, None)
        self.assertEqual(pipe.during, ())
        self.assertEqual(pipe.transformer.destinations, ['cpu'])
        self.assertEqual(pipe.vae.destinations, ['cpu'])
        self.assertEqual(pipe.text_encoder.destinations, ['cpu'])

    def test_comfy_quant_transformer_stays_resident_under_offload(self):
        import dgenerate.pipelinewrapper.pipelines as pipelines
        import dgenerate.pipelinewrapper.safetensorscompat as safetensorscompat

        class Pipe:
            def __init__(self):
                self.transformer = torch.nn.Linear(1, 1)
                self.text_encoder = torch.nn.Linear(1, 1)
                self.vae = torch.nn.Linear(1, 1)

            @property
            def components(self):
                return {
                    'transformer': self.transformer,
                    'text_encoder': self.text_encoder,
                    'vae': self.vae,
                }

        pipe = Pipe()
        placed = []
        offloaded = []

        def has_comfy(module):
            return module is pipe.transformer

        with unittest.mock.patch.object(
                pipelines, 'place_quantized_module',
                side_effect=lambda module, device: placed.append(module)), \
                unittest.mock.patch(
                    'accelerate.cpu_offload',
                    side_effect=lambda module, *args, **kwargs: offloaded.append(module)), \
                unittest.mock.patch.object(
                    safetensorscompat, 'module_has_comfy_quant', side_effect=has_comfy):
            _flux3.place_flux3(pipe, 'cuda', False, True, False)

        self.assertIs(placed[0], pipe.transformer)
        self.assertIn(pipe.text_encoder, offloaded)
        self.assertNotIn(pipe.transformer, offloaded)

    def test_upsampler_step_logs_the_enhanced_prompt(self):
        class Track(torch.nn.Linear):
            def __init__(self):
                super().__init__(1, 1)

            def to(self, *args, **kwargs):
                return super().to(*args, **kwargs)

        class State:
            def __init__(self):
                self.values = {'prompt': 'a cat'}

            def get(self, key, default=None):
                return self.values.get(key, default)

            def set(self, key, value, kwargs_type=None):
                self.values[key] = value

        class Step:
            def get_block_state(self, state):
                return {'prompt': state.get('prompt')}

            def __call__(self, components, state):
                state.set('prompt', 'a red fox sitting in snow')
                return components, state

        class Pipe:
            def __init__(self):
                self.transformer = Track()
                self.vae = Track()
                self.text_encoder = Track()
                self.prompt_upsampler = Track()
                self._blocks = types.SimpleNamespace(
                    sub_blocks={'prompt_upsampler': Step()})

            @property
            def components(self):
                return {
                    'transformer': self.transformer,
                    'vae': self.vae,
                    'text_encoder': self.text_encoder,
                    'prompt_upsampler': self.prompt_upsampler,
                }

        pipe = Pipe()
        logs = []
        with unittest.mock.patch(
                'dgenerate.messages.log', side_effect=logs.append):
            _flux3.place_flux3(pipe, 'cpu', False, False, False)
            pipe._blocks.sub_blocks['prompt_upsampler'](pipe, State())
        self.assertIn('Pipeline prompt enhance: a red fox sitting in snow', logs)


if __name__ == '__main__':
    unittest.main()
