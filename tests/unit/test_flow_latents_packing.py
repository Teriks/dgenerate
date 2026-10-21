import types
import unittest

import torch

import dgenerate.pipelinewrapper.denoise_range as denoise_range
from dgenerate.pipelinewrapper.enums import ModelType
from dgenerate.pipelinewrapper.wrapper import DiffusionPipelineWrapper


class TestFlowLatentsPacking(unittest.TestCase):
    """Round-trip tests for packed-flow image latent layouts."""

    def test_flux_style_pack_unpack_roundtrip(self):
        # External spatial [B, C, H, W] <-> packed [B, L, C*4]
        spatial = torch.randn(2, 16, 64, 64)
        packed = DiffusionPipelineWrapper._repack_flux_latents(spatial)
        self.assertEqual(packed.shape, (2, 32 * 32, 64))

        # Mimic wrapper unpack geometry for 512px / vae_scale 8.
        batch_size, num_patches, channels = packed.shape
        height = width = 64
        unpacked = packed.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        unpacked = unpacked.permute(0, 3, 1, 4, 2, 5)
        unpacked = unpacked.reshape(batch_size, channels // 4, height, width)

        self.assertTrue(torch.allclose(unpacked, spatial, atol=1e-6))

    def test_qwen21_flatten_pack_unpack_roundtrip(self):
        spatial = torch.randn(2, 16, 64, 96)
        packed = DiffusionPipelineWrapper._repack_qwen21_latents(spatial)
        restored = DiffusionPipelineWrapper._unpack_qwen21_latents(
            packed, height=1024, width=1536)

        self.assertEqual(packed.shape, (2, 64 * 96, 16))
        self.assertTrue(torch.allclose(restored, spatial, atol=1e-6))

        wrapper = DiffusionPipelineWrapper.__new__(DiffusionPipelineWrapper)
        wrapper._model_type = ModelType.QWEN_IMAGE_21
        wrapper._pipeline = types.SimpleNamespace(vae_scale_factor=16)
        self.assertTrue(wrapper._latents_use_qwen21_packing())
        self.assertTrue(torch.allclose(
            wrapper._repack_input_latents(spatial), packed, atol=1e-6))
        self.assertTrue(torch.allclose(
            wrapper._unpack_output_latents(packed[0], height=1024, width=1536),
            spatial[:1], atol=1e-6))

    def test_kandinsky5_image_layout_roundtrip(self):
        spatial = torch.randn(2, 16, 64, 96)
        pipeline_layout = DiffusionPipelineWrapper._repack_kandinsky5_image_latents(spatial)
        restored = DiffusionPipelineWrapper._unpack_kandinsky5_image_latents(pipeline_layout)

        self.assertEqual(pipeline_layout.shape, (2, 1, 64, 96, 16))
        self.assertTrue(torch.allclose(restored, spatial, atol=1e-6))

        wrapper = DiffusionPipelineWrapper.__new__(DiffusionPipelineWrapper)
        wrapper._model_type = ModelType.KANDINSKY5_IMAGE
        self.assertTrue(wrapper._latents_use_kandinsky5_image_layout())
        self.assertTrue(torch.allclose(
            wrapper._repack_input_latents(spatial), pipeline_layout, atol=1e-6))
        self.assertTrue(torch.allclose(
            wrapper._unpack_output_latents(pipeline_layout[0]), spatial[:1], atol=1e-6))

        wrapper._inference_height = 512
        wrapper._inference_width = 768
        output = type('Output', (), {})()
        output.image = pipeline_layout
        result = wrapper._create_pipeline_result(output, output_type='latent')
        self.assertEqual(len(result.latents), 2)
        self.assertTrue(torch.allclose(result.latents[0], spatial[:1], atol=1e-6))

    def test_krea2_uses_flux_style_packing(self):
        from types import SimpleNamespace
        wrapper = DiffusionPipelineWrapper.__new__(DiffusionPipelineWrapper)
        wrapper._model_type = ModelType.KREA2
        wrapper._pipeline = SimpleNamespace(default_sample_size=64, vae_scale_factor=8)
        spatial = torch.randn(1, 16, 64, 64)
        packed = wrapper._repack_input_latents(spatial)
        unpacked = wrapper._unpack_output_latents(packed[0], height=512, width=512)

        self.assertTrue(wrapper._latents_use_flux_style_packing())
        self.assertEqual(packed.shape, (1, 32 * 32, 64))
        self.assertTrue(torch.allclose(unpacked, spatial, atol=1e-6))

    def test_new_flow_pipelines_are_recognized_for_denoise_splits(self):
        for name in ('Krea2Pipeline', 'Kandinsky5T2IPipeline', 'Kandinsky5I2IPipeline'):
            with self.subTest(pipeline=name):
                pipeline = type(name, (), {})()
                self.assertTrue(denoise_range._supports_denoise_range_flow_matching(pipeline))

    def test_flux2_patchify_unpatchify_roundtrip(self):
        spatial = torch.randn(2, 32, 128, 128)
        patchified = DiffusionPipelineWrapper._patchify_flux2_latents(spatial)
        self.assertEqual(patchified.shape, (2, 128, 64, 64))
        restored = DiffusionPipelineWrapper._unpatchify_flux2_latents(patchified)
        self.assertEqual(restored.shape, spatial.shape)
        self.assertTrue(torch.allclose(restored, spatial, atol=1e-6))

    def test_flux2_dense_pack_inverse_matches_unpatchify(self):
        # Full Flux.2: prepare_latents packs patchified [B, C*4, H/2, W/2]
        # to [B, H*W, C*4]. Dense inverse must recover patchified, then spatial.
        spatial = torch.randn(1, 32, 128, 128)
        patchified = DiffusionPipelineWrapper._patchify_flux2_latents(spatial)
        packed = patchified.reshape(1, 128, 64 * 64).permute(0, 2, 1)  # Flux2._pack_latents

        batch_size, num_patches, channels = packed.shape
        restored_patch = packed.permute(0, 2, 1).reshape(batch_size, channels, 64, 64)
        restored = DiffusionPipelineWrapper._unpatchify_flux2_latents(restored_patch)

        self.assertTrue(torch.allclose(restored_patch, patchified, atol=1e-6))
        self.assertTrue(torch.allclose(restored, spatial, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
