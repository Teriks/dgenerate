import inspect
import os
import tempfile
import unittest
from unittest import mock

import diffusers
import torch

from dgenerate.extras.hydit.image_encoder import (
    _clip_encoder_weight_path,
    _hydit_clip_encoder_cache,
    _resident_dtype,
    _transformers_vision_state,
    load_hunyuan_dit_clip_image_encoder,
)
from dgenerate.extras.hydit.lora import (
    _kohya_text_encoder_target,
    _load_state_dict,
    _lora_delta_scale,
    _normalize_hydit_lora_module_key,
)
from dgenerate.extras.hydit.ip_adapter import (
    _load_ipa_state_dict,
    _normalize_ip_adapter_images,
    _unwrap_ip_adapter_load_arg,
)
from dgenerate.extras.hydit.lora import HunyuanDiTLoraLoaderMixin
from dgenerate.extras.hydit.pipelines import (
    HunyuanDiTControlNetPipeline,
    HunyuanDiTPipeline,
    _control_guidance_keep,
    _hydit_denoising_steps,
    _scaled_controlnet_conditioning,
    align_ip_adapter_clip_embeds,
)
from dgenerate.pipelinewrapper import util as _putil
from dgenerate.pipelinewrapper.uris.lorauri import _activate_loaded_loras
from dgenerate.pipelinewrapper.pipelines import (
    _sequential_offload_preload_classes,
    get_pipeline_modules,
    validate_model_index_class,
)
import dgenerate.pipelinewrapper.singlefilecompat as _hydit_sf
import dgenerate.pipelinewrapper.enums as _enums
from diffusers.models.embeddings import HunyuanDiTAttentionPool


class TestHyditPipelines(unittest.TestCase):
    def test_get_pipeline_modules_includes_attached_image_encoder(self):
        class _Pipe:
            @property
            def components(self):
                return {'transformer': torch.nn.Linear(1, 1)}

            def __init__(self):
                self.image_encoder = torch.nn.Linear(2, 2)

        modules = get_pipeline_modules(_Pipe())
        self.assertIn('image_encoder', modules)
        self.assertIn('transformer', modules)

    def test_pipeline_init_exposes_diffusers_components(self):
        for pipeline_class in (HunyuanDiTPipeline, HunyuanDiTControlNetPipeline):
            parameters = inspect.signature(pipeline_class.__init__).parameters
            self.assertIn('transformer', parameters)
            self.assertIn('vae', parameters)
            self.assertNotIn('args', parameters)
            self.assertNotIn('kwargs', parameters)

    def test_call_exposes_ip_adapter_image_arguments(self):
        pairs = (
            (HunyuanDiTPipeline, diffusers.HunyuanDiTPipeline),
            (HunyuanDiTControlNetPipeline, diffusers.HunyuanDiTControlNetPipeline),
        )
        for pipeline_class, base_pipeline_class in pairs:
            patched = inspect.signature(pipeline_class.__call__)
            base = inspect.signature(base_pipeline_class.__call__.__wrapped__)
            self.assertEqual(
                list(patched.parameters)[-2:],
                ['ip_adapter_image', 'ip_adapter_images'])
            self.assertEqual(
                list(patched.parameters)[:len(base.parameters)],
                list(base.parameters))
            call = pipeline_class.__call__
            func = getattr(call, '__wrapped__', None) or call
            self.assertIn('ip_adapter_image', inspect.getfullargspec(func).args)
            self.assertIn('guidance_rescale', inspect.getfullargspec(func).args)
            self.assertIs(func.__wrapped__, base_pipeline_class.__call__.__wrapped__)
        control = inspect.signature(HunyuanDiTControlNetPipeline.__call__)
        self.assertIn('controlnet_conditioning_scale', control.parameters)
        self.assertIn('control_guidance_start', control.parameters)
        self.assertIn('control_guidance_end', control.parameters)
        text = inspect.signature(HunyuanDiTPipeline.__call__)
        self.assertNotIn('control_guidance_start', text.parameters)

    def test_control_guidance_window_scales_conditioning(self):
        self.assertEqual(_control_guidance_keep(0, 10, 0.2, 0.8), 0.0)
        self.assertEqual(_control_guidance_keep(2, 10, 0.2, 0.8), 1.0)
        self.assertEqual(_control_guidance_keep(8, 10, 0.2, 0.8), 0.0)
        self.assertEqual(
            _scaled_controlnet_conditioning(0.6, [0.0], multi=False), 0.0)
        self.assertEqual(
            _scaled_controlnet_conditioning([0.5, 1.0], [1.0, 0.0], multi=True),
            [0.5, 0.0])

    def test_attention_pool_is_preloaded_for_sequential_offload(self):
        pool = HunyuanDiTAttentionPool(spacial_dim=4, embed_dim=8, num_heads=2, output_dim=8)
        root = torch.nn.Module()
        root.pooler = pool
        self.assertIn('HunyuanDiTAttentionPool', _sequential_offload_preload_classes(root))

    def test_cpu_half_precision_request_is_stored_as_float32(self):
        self.assertEqual(_resident_dtype('cpu', torch.float16), torch.float32)
        self.assertEqual(_resident_dtype('cpu', torch.bfloat16), torch.float32)
        self.assertEqual(_resident_dtype('cpu', torch.float32), torch.float32)
        self.assertEqual(_resident_dtype('cuda', torch.float16), torch.float16)

    def _tiny_visual_checkpoint(self):
        width, patch, proj = 80, 8, 16
        positions = 2
        middle = width * 4
        weights = {
            'conv1.weight': torch.randn(width, 3, patch, patch),
            'class_embedding': torch.randn(width),
            'positional_embedding': torch.randn(positions, width),
            'ln_pre.weight': torch.ones(width),
            'ln_pre.bias': torch.zeros(width),
            'ln_post.weight': torch.ones(width),
            'ln_post.bias': torch.zeros(width),
            'proj': torch.randn(width, proj),
            'transformer.resblocks.0.ln_1.weight': torch.ones(width),
            'transformer.resblocks.0.ln_1.bias': torch.zeros(width),
            'transformer.resblocks.0.ln_2.weight': torch.ones(width),
            'transformer.resblocks.0.ln_2.bias': torch.zeros(width),
            'transformer.resblocks.0.attn.in_proj_weight': torch.randn(width * 3, width),
            'transformer.resblocks.0.attn.in_proj_bias': torch.randn(width * 3),
            'transformer.resblocks.0.attn.out_proj.weight': torch.randn(width, width),
            'transformer.resblocks.0.attn.out_proj.bias': torch.randn(width),
            'transformer.resblocks.0.mlp.c_fc.weight': torch.randn(middle, width),
            'transformer.resblocks.0.mlp.c_fc.bias': torch.randn(middle),
            'transformer.resblocks.0.mlp.c_proj.weight': torch.randn(width, middle),
            'transformer.resblocks.0.mlp.c_proj.bias': torch.randn(width),
        }
        return {'state_dict': {f'module.visual.{key}': value for key, value in weights.items()}}

    def test_openclip_qkv_and_projection_map_onto_transformers(self):
        checkpoint = self._tiny_visual_checkpoint()['state_dict']
        visual = {key[len('module.visual.'):]: value for key, value in checkpoint.items()}
        converted, config = _transformers_vision_state(visual)
        qkv = checkpoint['module.visual.transformer.resblocks.0.attn.in_proj_weight']
        q_weight, _, _ = qkv.chunk(3, dim=0)
        self.assertTrue(torch.equal(
            converted['vision_model.encoder.layers.0.self_attn.q_proj.weight'], q_weight))
        self.assertTrue(torch.equal(
            converted['visual_projection.weight'],
            checkpoint['module.visual.proj'].transpose(0, 1)))
        self.assertEqual(config['num_attention_heads'], 1)
        self.assertEqual(config['image_size'], 8)
        self.assertEqual(config['hidden_act'], 'gelu')

    def test_clip_encoder_cache_reuses_one_tower(self):
        checkpoint = self._tiny_visual_checkpoint()
        _hydit_clip_encoder_cache.clear()
        try:
            with tempfile.TemporaryDirectory() as folder:
                weight = os.path.join(folder, 'clip_img_encoder.pt')
                torch.save(checkpoint, weight)
                first = load_hunyuan_dit_clip_image_encoder(
                    weight, device='cpu', dtype=torch.float16)
                second = load_hunyuan_dit_clip_image_encoder(
                    weight, device=torch.device('cpu'), dtype=torch.bfloat16)
            self.assertIs(first, second)
            self.assertEqual(first.dtype, torch.float32)
            self.assertGreater(_hydit_clip_encoder_cache.size, 0)
        finally:
            _hydit_clip_encoder_cache.clear()

    def test_ip_adapter_cfg_prepends_zero_clip_batch(self):
        clip = torch.ones(1, 1024)
        with_cfg = align_ip_adapter_clip_embeds(clip, 1, 1, 5.0)
        self.assertEqual(tuple(with_cfg.shape), (2, 1024))
        self.assertTrue(torch.all(with_cfg[0] == 0))
        self.assertTrue(torch.all(with_cfg[1] == 1))

    def test_ip_adapter_cfg_repeats_one_image_across_the_batch(self):
        clip = torch.ones(1, 4)
        expanded = align_ip_adapter_clip_embeds(clip, 1, 2, 5.0)
        self.assertEqual(tuple(expanded.shape), (4, 4))
        self.assertTrue(torch.all(expanded[:2] == 0))
        self.assertTrue(torch.all(expanded[2:] == 1))

        prompts = torch.tensor([[1.0], [2.0]])
        paired = align_ip_adapter_clip_embeds(prompts, 2, 2, 5.0)
        self.assertEqual(paired[4:, 0].tolist(), [1.0, 1.0, 2.0, 2.0])
        self.assertTrue(torch.all(paired[:4] == 0))

    def test_control_guidance_uses_scheduler_timestep_count(self):
        class _Scheduler:
            timesteps = [0, 1, 2, 3, 4, 5]

        class _Pipeline:
            scheduler = _Scheduler()

        self.assertEqual(_hydit_denoising_steps(_Pipeline(), 2), 6)
        self.assertEqual(_hydit_denoising_steps(object(), 2), 2)

    def test_pooler_q_proj_keeps_its_parameter_name(self):
        class _Pool(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(2, 2, bias=False)

        class _Extra(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pooler = _Pool()

        class _Attn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(2, 2, bias=False)

        class _Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.time_extra_emb = _Extra()
                self.attn2 = _Attn()

        class _Pipe(HunyuanDiTLoraLoaderMixin):
            def __init__(self):
                self.transformer = _Net()

        eye = torch.eye(2)
        deltas = _Pipe()._extract_lora_deltas({
            'pooler.q_proj.lora_up.weight': eye,
            'pooler.q_proj.lora_down.weight': eye,
            'attn2.q_proj.lora_up.weight': eye,
            'attn2.q_proj.lora_down.weight': eye,
        })
        self.assertIn('transformer\ntime_extra_emb.pooler.q_proj.weight', deltas)
        self.assertIn('transformer\nattn2.to_q.weight', deltas)
        self.assertNotIn('transformer\npooler.to_q.weight', deltas)

    def test_lora_skips_suffix_matches_and_fuse_is_idempotent(self):
        class _Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block0 = torch.nn.Linear(2, 2, bias=False)
                self.block1 = torch.nn.Linear(2, 2, bias=False)

        class _Pipe(HunyuanDiTLoraLoaderMixin):
            def __init__(self):
                self.transformer = _Net()

        pipe = _Pipe()
        skipped = {}
        pipe._store_delta(skipped, 'transformer', 'weight', torch.ones(2, 2))
        self.assertEqual(skipped, {})

        pipe._lora_adapters = {
            'a': {'transformer\nblock0.weight': torch.ones(2, 2)}}
        pipe._lora_adapter_weights = {'a': 1.0}
        pipe._unfused_lora_weights = {}
        pipe._hydit_lora_fused = False
        original = pipe.transformer.block0.weight.detach().clone()
        untouched = pipe.transformer.block1.weight.detach().clone()
        pipe.fuse_lora()
        pipe.fuse_lora()
        self.assertTrue(torch.equal(
            pipe.transformer.block0.weight.detach(), original + 1))
        self.assertTrue(torch.equal(
            pipe.transformer.block1.weight.detach(), untouched))

    def test_get_list_adapters_names_each_component(self):
        class _Pipe(HunyuanDiTLoraLoaderMixin):
            pass

        pipe = _Pipe()
        pipe._lora_adapters = {
            '0': {
                'transformer\nblocks.0.attn1.to_q.weight': torch.zeros(1),
                'text_encoder\nbert.encoder.layer.0.weight': torch.zeros(1),
            },
            '1': {'transformer\nblocks.0.attn1.to_q.weight': torch.zeros(1)},
        }

        self.assertEqual(pipe.get_list_adapters(), {
            'transformer': ['0', '1'],
            'text_encoder': ['0'],
        })

    def test_mixed_component_loras_fuse_on_the_pipeline(self):
        class _Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(2, 2, bias=False)

        class _Pipe(HunyuanDiTLoraLoaderMixin):
            def __init__(self):
                self.transformer = _Net()
                self.text_encoder = _Net()

        pipe = _Pipe()
        pipe._lora_adapters = {
            '0': {'transformer\nproj.weight': torch.ones(2, 2)},
            '1': {'text_encoder\nproj.weight': torch.ones(2, 2)},
        }
        pipe._lora_adapter_weights = {'0': 1.0, '1': 1.0}
        pipe._unfused_lora_weights = {}
        pipe._hydit_lora_fused = False
        transformer_before = pipe.transformer.proj.weight.detach().clone()
        text_before = pipe.text_encoder.proj.weight.detach().clone()

        _activate_loaded_loras(pipe, ['0', '1'], [1.0, 0.5], True, 1.0)

        self.assertTrue(torch.equal(
            pipe.transformer.proj.weight.detach(), transformer_before + 1))
        self.assertTrue(torch.equal(
            pipe.text_encoder.proj.weight.detach(), text_before + 0.5))

    def test_clip_encoder_path_uses_a_local_sibling(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = os.path.join(directory, 'ipa.pt')
            encoder = os.path.join(directory, 'clip_img_encoder.pt')
            open(weights, 'wb').close()
            open(encoder, 'wb').close()

            self.assertEqual(_clip_encoder_weight_path(weights), encoder)
            self.assertEqual(_clip_encoder_weight_path(directory), encoder)
            self.assertEqual(_clip_encoder_weight_path(encoder), encoder)

    def test_clip_encoder_path_rejects_a_local_file_without_a_sibling(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = os.path.join(directory, 'ipa.pt')
            open(weights, 'wb').close()
            with self.assertRaises(FileNotFoundError):
                _clip_encoder_weight_path(weights)

    @mock.patch('dgenerate.extras.hydit.image_encoder.huggingface_hub.hf_hub_download')
    def test_clip_encoder_path_downloads_a_hub_repo(self, hf_hub_download):
        hf_hub_download.return_value = '/cache/clip_img_encoder.pt'

        found = _clip_encoder_weight_path(
            'org/ip-adapter-not-a-local-path', token='t', local_files_only=True)

        self.assertEqual(found, '/cache/clip_img_encoder.pt')
        hf_hub_download.assert_called_once_with(
            repo_id='org/ip-adapter-not-a-local-path',
            filename='clip_img_encoder.pt',
            token='t',
            local_files_only=True,
        )

    def test_normalize_ip_adapter_images_flattens_dgenerate_groups(self):
        sentinel = object()
        self.assertIs(_normalize_ip_adapter_images([[sentinel]]), sentinel)
        self.assertEqual(_normalize_ip_adapter_images([[sentinel, sentinel]]), [sentinel, sentinel])

    def test_ip_adapter_load_unwraps_diffusers_style_lists(self):
        self.assertEqual(
            _unwrap_ip_adapter_load_arg(['Tencent-Hunyuan/IP-Adapter'], 'model', allow_none=False),
            'Tencent-Hunyuan/IP-Adapter')
        self.assertIsNone(_unwrap_ip_adapter_load_arg(['.'], 'subfolder'))

    @mock.patch('dgenerate.extras.hydit.ip_adapter.huggingface_hub.hf_hub_download')
    @mock.patch('dgenerate.extras.hydit.ip_adapter.torch.load')
    def test_load_ipa_state_dict_accepts_model_list(self, torch_load, hf_hub_download):
        hf_hub_download.return_value = '/tmp/ipa.pt'
        torch_load.return_value = {}

        _load_ipa_state_dict(
            ['Tencent-Hunyuan/IP-Adapter'],
            weight_name=['ipa.pt'],
            subfolder=['.'],
        )

        hf_hub_download.assert_called_once()
        self.assertEqual(
            hf_hub_download.call_args.kwargs['repo_id'], 'Tencent-Hunyuan/IP-Adapter')
        self.assertEqual(hf_hub_download.call_args.kwargs['filename'], 'ipa.pt')
        self.assertIsNone(hf_hub_download.call_args.kwargs['subfolder'])

    def test_peft_lora_without_alpha_uses_full_scale(self):
        self.assertEqual(_lora_delta_scale(None, 64), 1.0)
        self.assertEqual(_lora_delta_scale(32.0, 64), 0.5)

    def test_kohya_bert_text_encoder_target(self):
        self.assertEqual(
            _kohya_text_encoder_target(
                'lora_te1_encoder_layer_3_attention_self_query'),
            ('text_encoder', 'encoder.layer.3.attention.self.query'))

    def test_normalize_kohya_hydit_lora_module_key(self):
        self.assertEqual(
            _normalize_hydit_lora_module_key('lora_unet_blocks_3_attn2_kv_proj'),
            'blocks.3.attn2.kv_proj')
        self.assertEqual(
            _normalize_hydit_lora_module_key(
                'base_model.model.blocks.0.attn1.Wqkv'),
            'blocks.0.attn1.Wqkv')

    @mock.patch('safetensors.torch.load_file')
    @mock.patch('dgenerate.extras.hydit.lora.huggingface_hub.hf_hub_download')
    def test_load_lora_state_dict_falls_back_to_adapter_model(self, hf_hub_download, load_file):
        def _download(*, filename, **kwargs):
            if filename == 'adapter_model.safetensors':
                return '/tmp/adapter_model.safetensors'
            raise OSError(filename)

        hf_hub_download.side_effect = _download
        load_file.return_value = {}

        _load_state_dict('Tencent-Hunyuan/HYDiT-LoRA', subfolder='jade')

        load_file.assert_called_once_with('/tmp/adapter_model.safetensors', device='cpu')
        self.assertEqual(
            hf_hub_download.call_args.kwargs['filename'], 'adapter_model.safetensors')
        self.assertEqual(hf_hub_download.call_args.kwargs['subfolder'], 'jade')


class TestHyditSingleFileCompat(unittest.TestCase):
    def _checkpoint(self):
        import os
        path = os.environ.get('DGENERATE_HYDIT_CHECKPOINT', '')
        if not path or not os.path.isfile(path):
            self.skipTest(
                'Set DGENERATE_HYDIT_CHECKPOINT to a native Hunyuan-DiT '
                'transformer file to run this test.')
        return path

    def test_native_hydit_checkpoint_detection(self):
        self.assertTrue(_hydit_sf.is_native_hydit_checkpoint({
            'blocks.0.attn1.Wqkv.weight': None,
            'x_embedder.proj.weight': None,
        }))
        self.assertFalse(_hydit_sf.is_native_hydit_checkpoint({
            'blocks.0.attn1.to_q.weight': None,
        }))

    def test_fetch_model_index_uses_hydit_diffusers_repo(self):
        index = _putil.fetch_model_index_dict(self._checkpoint())
        self.assertEqual(index['_class_name'], 'HunyuanDiTPipeline')
        validate_model_index_class(_enums.ModelType.HYDIT, index['_class_name'])

    def test_transformer_only_single_file_uses_diffusers_repo_for_aux(self):
        repo = _hydit_sf.single_file_pretrained_repo_for_model_path(self._checkpoint())
        self.assertEqual(repo, 'Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers')

    def test_hydit_registers_from_single_file(self):
        _hydit_sf.register_single_file_classes()
        self.assertTrue(hasattr(diffusers.HunyuanDiT2DModel, 'from_single_file'))
        self.assertIsNotNone(diffusers.HunyuanDiT2DModel.from_single_file)

    def test_convert_native_hydit_matches_diffusers_transformer_keys(self):
        import diffusers.loaders.single_file as _sf
        checkpoint = _sf.load_single_file_checkpoint(self._checkpoint())
        converted = _hydit_sf.convert_hydit_transformer_for_single_file(checkpoint)
        reference = diffusers.HunyuanDiT2DModel.from_pretrained(
            'Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers',
            subfolder='transformer',
        )
        expected = reference.state_dict().keys()
        self.assertEqual(set(converted.keys()), set(expected))


if __name__ == '__main__':
    unittest.main()
