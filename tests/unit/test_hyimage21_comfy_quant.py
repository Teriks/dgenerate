import unittest

import torch

import dgenerate.pipelinewrapper.singlefilecompat as _hyimage_sf
import dgenerate.pipelinewrapper.safetensorscompat as _safetensorscompat

# Comfy-Org repacks for HyImage 2.1 (see examples/hyimage21/comfy_quant and recipes).
_COMFY_TRANSFORMERS = (
    'https://huggingface.co/Comfy-Org/Hunyuan-Image-2.1/resolve/main/split_files/diffusion_models'
)
HYIMAGE21_BASE_FP8 = f'{_COMFY_TRANSFORMERS}/hunyuanimage2.1_fp8_e4m3fn.safetensors'
DIFFUSERS_BASE_SLUG = 'hunyuanvideo-community/HunyuanImage-2.1-Diffusers'


class TestHyImage21ComfyQuant(unittest.TestCase):
    def test_comfy_fp8_url_for_base_diffusers_slug(self):
        self.assertTrue(HYIMAGE21_BASE_FP8.startswith('https://huggingface.co/'))
        self.assertIn('hunyuanimage2.1_fp8_e4m3fn.safetensors', HYIMAGE21_BASE_FP8)
        # Documented pairing used by examples (Diffusers slug + Comfy transformer file).
        self.assertEqual(DIFFUSERS_BASE_SLUG,
                         'hunyuanvideo-community/HunyuanImage-2.1-Diffusers')

    def test_fp8_header_is_comfy_quantized(self):
        import json
        import struct
        import urllib.request

        request = urllib.request.Request(HYIMAGE21_BASE_FP8, headers={'Range': 'bytes=0-500000'})
        data = urllib.request.urlopen(request, timeout=120).read()
        header_size = struct.unpack('<Q', data[:8])[0]
        header = json.loads(data[8:8 + header_size])
        header.pop('__metadata__', None)
        fp8_2d = sum(
            1 for key, info in header.items()
            if isinstance(info, dict) and info.get('dtype') == 'F8_E4M3'
            and key.endswith('.weight') and len(info.get('shape', ())) == 2)
        self.assertGreater(fp8_2d, 100)

    def test_fused_qkv_splits_to_per_head_keys(self):
        weight = torch.randn(96, 32)
        prepared = _hyimage_sf.prepare_hunyuanimage_checkpoint({
            'double_blocks.0.img_attn_qkv.weight': weight,
        })
        self.assertIn('double_blocks.0.img_attn_q.weight', prepared)
        self.assertIn('double_blocks.0.img_attn_k.weight', prepared)
        self.assertIn('double_blocks.0.img_attn_v.weight', prepared)
        self.assertNotIn('double_blocks.0.img_attn_weightq.weight', prepared)

    def test_fp8_norm_scales_map_to_qk_norm_weights(self):
        prepared = _hyimage_sf.prepare_hunyuanimage_checkpoint({
            'model.model.double_blocks.0.img_attn.norm.query_norm.scale': torch.ones(128),
            'model.model.double_blocks.0.img_attn.norm.key_norm.scale': torch.ones(128),
        })
        self.assertIn('double_blocks.0.img_attn_q_norm.weight', prepared)
        self.assertIn('double_blocks.0.img_attn_k_norm.weight', prepared)

    def test_mapping_fn_does_not_mutate_shared_checkpoint(self):
        weight = torch.randn(96, 32)
        q, k, v = torch.chunk(weight, 3, dim=0)
        shared = {
            'img_in.proj.weight': torch.randn(3584, 64),
            'img_in.proj.bias': torch.zeros(3584),
            'double_blocks.0.img_attn_qkv.weight': weight,
            'double_blocks.0.img_attn_qkv.bias': torch.zeros(96),
            'double_blocks.0.img_attn_q.weight': q,
            'double_blocks.0.img_attn_k.weight': k,
            'double_blocks.0.img_attn_v.weight': v,
        }
        before = set(shared)
        try:
            _hyimage_sf.convert_hunyuanimage_transformer_for_single_file(shared)
        except KeyError:
            pass
        self.assertEqual(set(shared), before)

    def test_infer_config_from_fp8_distilled_names(self):
        names = {
            'model.model.guidance_in.mlp.0.weight',
            'model.model.time_r_in.mlp.0.weight',
            'byt5_in.fc1.weight',
            'txt_in.input_embedder.weight',
        }
        repo = _hyimage_sf.hunyuanimage_gguf_config_repo(
            {name: (1,) for name in names})
        self.assertEqual(
            repo, 'hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers')

    def test_synthetic_fp8_is_comfy_quantized(self):
        import os
        import tempfile

        import safetensors.torch

        weight = torch.randn(64, 128).to(torch.float8_e4m3fn)
        handle, path = tempfile.mkstemp(suffix='.safetensors')
        os.close(handle)
        self.addCleanup(os.remove, path)
        safetensors.torch.save_file({'layer.weight': weight}, path)
        self.assertTrue(_safetensorscompat.is_comfy_quantized(path))
