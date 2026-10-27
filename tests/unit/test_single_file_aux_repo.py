import os
import tempfile
import unittest
from unittest.mock import patch

from dgenerate.pipelinewrapper.enums import ModelType
from dgenerate.pipelinewrapper.pipelines import estimate_pipeline_cache_footprint
from dgenerate.pipelinewrapper.singlefilecompat import (
    single_file_pretrained_repo_for_model_path,
    single_file_pretrained_repo_for_shapes,
)


class TestSingleFileAuxRepo(unittest.TestCase):
    def repo(self, hint=None, **shapes):
        return single_file_pretrained_repo_for_shapes(shapes, hint=hint)

    def test_flux_dev_and_schnell(self):
        dev = {
            'double_blocks.0.img_attn.norm.key_norm.scale': (128,),
            'guidance_in.in_layer.bias': (3072,),
            'img_in.weight': (3072, 64),
        }
        self.assertEqual(self.repo(**dev), 'black-forest-labs/FLUX.1-dev')
        schnell = dict(dev)
        schnell.pop('guidance_in.in_layer.bias')
        self.assertEqual(self.repo(**schnell), 'black-forest-labs/FLUX.1-schnell')

    def test_embedded_text_encoders_stay_on_from_single_file(self):
        self.assertIsNone(self.repo(**{
            'double_blocks.0.img_attn.norm.key_norm.scale': (128,),
            'text_encoders.clip_l.transformer.text_model.embeddings.position_embedding.weight': (77, 768),
        }))
        self.assertIsNone(self.repo(**{
            'cond_stage_model.transformer.text_model.embeddings.position_embedding.weight': (77, 768),
        }))

    def test_sd3_medium_and_35(self):
        joint = {'joint_blocks.0.context_block.adaLN_modulation.1.bias': (9216,)}
        self.assertEqual(
            self.repo(**joint, pos_embed=(1, 36864)),
            'stabilityai/stable-diffusion-3-medium-diffusers')
        self.assertEqual(
            self.repo(**joint, pos_embed=(1, 147456)),
            'stabilityai/stable-diffusion-3.5-medium')

    def test_filename_picks_sibling_repos(self):
        z = {
            'layers.0.adaLN_modulation.0.weight': (1,),
            'cap_embedder.0.weight': (1,),
        }
        self.assertEqual(self.repo(**z), 'Tongyi-MAI/Z-Image-Turbo')
        self.assertEqual(
            self.repo(hint='z_image-Q4_K_S.gguf', **z),
            'Tongyi-MAI/Z-Image')
        hy = {'cond_type_embedding.weight': (3, 2048)}
        self.assertEqual(
            self.repo(**hy),
            'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v')
        self.assertEqual(
            self.repo(hint='hunyuanvideo1.5_480p_i2v_distilled-Q4.gguf', **hy),
            'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_distilled')
        self.assertEqual(
            self.repo(hint='hunyuanvideo1.5_720p_i2v-Q4_K_S.gguf', **hy),
            'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v')
        self.assertEqual(
            self.repo(hint='hunyuanvideo1.5_720p_t2v_distilled.gguf', **hy),
            'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v_distilled')
        self.assertEqual(
            self.repo(hint='hunyuanvideo1.5_720p_i2v_distilled.gguf', **hy),
            'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled')
        self.assertEqual(
            self.repo(hint='hunyuanvideo1.5_1720_steps.gguf', **hy),
            'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v')

    def test_hyvideo_is_not_hunyuanimage(self):
        self.assertEqual(self.repo(**{
            'cond_type_embedding.weight': (3, 2048),
            'txt_in.input_embedder.weight': (3584, 3584),
            'img_in.proj.weight': (3584, 64),
            'byt5_in.fc1.weight': (2048, 1472),
        }), 'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v')

    def test_hydit_and_known_layouts(self):
        self.assertEqual(self.repo(**{
            'blocks.0.attn1.Wqkv.weight': (1,),
            'x_embedder.proj.weight': (1,),
        }), 'Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers')
        self.assertEqual(self.repo(**{
            'single_transformer_blocks.0.cross_attn.to_q.weight': (1, 1),
        }), 'Motif-Technologies/Motif-Video-2B')
        self.assertEqual(self.repo(**{
            'txt_in.text_norm.weight': (1,),
        }), 'Qwen/Qwen-Image-2.1')
        self.assertEqual(self.repo(
            hint='qwen_image_2.1_turbo_bf16.safetensors',
            **{'txt_in.text_norm.weight': (1,)}),
            'Qwen/Qwen-Image-2.1-Turbo')
        self.assertEqual(self.repo(**{
            'single_stream_modulation.lin.weight': (1,),
            'img_in.weight': (3072, 128),
        }), 'black-forest-labs/FLUX.2-klein-4B')

    def test_wan_i2v_720p_filename(self):
        shapes = {
            'head.modulation': (1,),
            'patch_embedding.weight': (5120, 36, 1, 2, 2),
            'img_emb.proj.0.weight': (5120, 1280),
        }
        self.assertEqual(
            self.repo(**shapes),
            'Wan-AI/Wan2.1-I2V-14B-480P-Diffusers')
        self.assertEqual(
            self.repo(hint='wan21I2v14b720pFP8_v10.safetensors', **shapes),
            'Wan-AI/Wan2.1-I2V-14B-720P-Diffusers')
        self.assertEqual(
            self.repo(hint='wan21_i2v_14b_7200steps.safetensors', **shapes),
            'Wan-AI/Wan2.1-I2V-14B-480P-Diffusers')
        wan22_i2v = {
            'head.modulation': (1,),
            'patch_embedding.weight': (5120, 36, 1, 2, 2),
        }
        self.assertEqual(
            self.repo(hint='wan2.2_i2v_a14b_720p.safetensors', **wan22_i2v),
            'Wan-AI/Wan2.2-I2V-A14B-Diffusers')
        t2v = {
            'head.modulation': (1,),
            'patch_embedding.weight': (5120, 16, 1, 2, 2),
        }
        self.assertEqual(
            self.repo(hint='wan21_t2v_14b_720p.safetensors', **t2v),
            'Wan-AI/Wan2.1-T2V-14B-Diffusers')
        self.assertEqual(
            self.repo(hint='wan2.2_t2v_high_noise_14B.safetensors', **t2v),
            'Wan-AI/Wan2.2-T2V-A14B-Diffusers')
        self.assertEqual(
            self.repo(hint='wan2.1_t2v_14b.safetensors', **t2v),
            'Wan-AI/Wan2.1-T2V-14B-Diffusers')
        flf = {
            'head.modulation': (1,),
            'patch_embedding.weight': (5120, 36, 1, 2, 2),
            'img_emb.emb_pos': (1, 514, 1280),
        }
        self.assertEqual(
            self.repo(hint='wan_i2v_720p_flf.safetensors', **flf),
            'Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers')

    def test_civitai_url_is_downloaded_before_layout_detection(self):
        url = 'https://civitai.com/api/download/models/1666198?fileId=1567048'
        shapes = {
            'head.modulation': (1,),
            'patch_embedding.weight': (5120, 16, 1, 2, 2),
        }
        with patch(
                'dgenerate.pipelinewrapper.singlefilecompat._hfhub.webcache_or_hf_blob_download',
                return_value='cached.safetensors') as download, \
                patch(
                    'dgenerate.pipelinewrapper.singlefilecompat._safetensors_weight_shapes',
                    return_value=shapes):
            repo = single_file_pretrained_repo_for_model_path(url)
        download.assert_called_once()
        self.assertEqual(download.call_args.args[0], url)
        self.assertEqual(repo, 'Wan-AI/Wan2.1-T2V-14B-Diffusers')


class TestAuxRepoPipelineEstimate(unittest.TestCase):
    def test_offload_counts_file_and_repo_components_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'transformer.safetensors')
            payload = b'x' * 1000
            with open(path, 'wb') as handle:
                handle.write(payload)
            with patch(
                    'dgenerate.pipelinewrapper.pipelines._util.estimate_model_memory_use',
                    return_value=50) as estimate:
                size = estimate_pipeline_cache_footprint(
                    model_path=path,
                    model_type=ModelType.QWEN_IMAGE_21,
                    include_unet_or_transformer=True,
                    include_vae=True,
                    include_text_encoders=True,
                    component_repo='Qwen/Qwen-Image-2.1-Turbo',
                    extra_args={'text_encoder_3': None},
                )
            self.assertEqual(size, len(payload) + 50)
            kwargs = estimate.call_args.kwargs
            self.assertEqual(kwargs['repo_id'], 'Qwen/Qwen-Image-2.1-Turbo')
            self.assertFalse(kwargs['include_unet_or_transformer'])
            self.assertTrue(kwargs['include_vae'])
            self.assertTrue(kwargs['include_text_encoder'])
            self.assertFalse(kwargs['include_text_encoder_3'])

    def test_separate_caches_do_not_count_the_transformer_file_again(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'transformer.safetensors')
            with open(path, 'wb') as handle:
                handle.write(b'x' * 1000)
            with patch(
                    'dgenerate.pipelinewrapper.pipelines._util.estimate_model_memory_use',
                    return_value=0):
                size = estimate_pipeline_cache_footprint(
                    model_path=path,
                    model_type=ModelType.QWEN_IMAGE_21,
                    include_unet_or_transformer=False,
                    include_vae=False,
                    include_text_encoders=False,
                    component_repo='Qwen/Qwen-Image-2.1-Turbo',
                )
            self.assertEqual(size, 0)


if __name__ == '__main__':
    unittest.main()
