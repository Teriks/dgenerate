import unittest

import torch

from dgenerate.pipelinewrapper.wrapper import DiffusionPipelineWrapper


class TestFlowLatentsPacking(unittest.TestCase):
    """Round-trip tests for Flux.1 / Qwen / Flux.2 latent pack helpers."""

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
