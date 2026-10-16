import math
import unittest
from unittest import mock

import torch

import dgenerate.extras.dype.patch as _dype_patch
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.renderloopconfig as _renderloopconfig


def _config(**values):
    config = _renderloopconfig.RenderLoopConfig()
    for name, value in values.items():
        setattr(config, name, value)
    return config


class TestDyPE(unittest.TestCase):
    def test_config_rejects_step_reuse_features(self):
        flux = _pipelinewrapper.ModelType.FLUX
        for kwargs in (
            {'cache_dit': True},
            {'cache_dit_config': 'cache-dit.yaml'},
            {'tea_cache': True},
            {'spectrum': True},
            {'sada': True},
        ):
            config = _config(
                model_path='black-forest-labs/FLUX.1-dev',
                model_type=flux,
                dype=True,
                prompts=['a horse'],
                **kwargs)
            with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
                config.check()

        implied = _config(
            model_path='black-forest-labs/FLUX.1-dev',
            model_type=flux,
            dype_methods=['yarn'],
            tea_cache_rel_l1_thresholds=[0.6],
            prompts=['a horse'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            implied.check()

    def test_config_rejects_unsupported_model_and_resolution(self):
        unsupported = _config(
            model_path='stable-diffusion-v1-5/stable-diffusion-v1-5',
            model_type=_pipelinewrapper.ModelType.SD,
            dype=True,
            prompts=['a horse'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            unsupported.check()
        self.assertIn('dype', str(raised.exception))

        resolution = _config(
            model_path='black-forest-labs/FLUX.1-dev',
            model_type=_pipelinewrapper.ModelType.FLUX,
            dype_base_resolutions=[8],
            prompts=['a horse'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError) as raised:
            resolution.check()
        self.assertIn('16', str(raised.exception))

    def test_wrapper_rejects_cache_dit(self):
        wrapper = _pipelinewrapper.DiffusionPipelineWrapper.__new__(
            _pipelinewrapper.DiffusionPipelineWrapper)
        wrapper._model_type = _pipelinewrapper.ModelType.FLUX
        args = _pipelinewrapper.DiffusionArguments()
        args.dype = True
        args.cache_dit = True
        with self.assertRaises(_pipelines.UnsupportedPipelineConfigError):
            wrapper._auto_dype_check(args)

    def test_flux_high_resolution_matches_dype_scheduler_shift(self):
        from diffusers import FluxPipeline
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        class _Pipeline(FluxPipeline):
            vae_scale_factor = 8

            def __init__(self, scheduler):
                self.scheduler = scheduler
                self.transformer = object()

        sigmas = [1.0 - (index / 49) * 0.98 for index in range(50)]
        scheduler = FlowMatchEulerDiscreteScheduler(
            use_dynamic_shifting=True,
            base_image_seq_len=256,
            max_image_seq_len=4096,
            base_shift=0.5,
            max_shift=1.15)
        pipeline = _Pipeline(scheduler)
        with mock.patch.object(_dype_patch, '_patch_transformer', return_value=lambda: None):
            with _dype_patch.dype_context(
                    pipeline, 'flux1', height=3072, width=3072):
                self.assertEqual(scheduler.config.max_image_seq_len, 36864)
                mu = calculate_shift(
                    36864,
                    scheduler.config.base_image_seq_len,
                    scheduler.config.max_image_seq_len,
                    scheduler.config.base_shift,
                    scheduler.config.max_shift)
                scheduler.set_timesteps(50, sigmas=sigmas, mu=mu)
                self.assertLess(float(scheduler.timesteps[-1]), 100.0)

        self.assertEqual(scheduler.config.max_image_seq_len, 4096)

    def test_resolution_shift_leaves_calibrated_sizes_unchanged(self):
        from diffusers import FluxPipeline
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler(
            use_dynamic_shifting=True,
            max_image_seq_len=4096)

        class _Pipeline(FluxPipeline):
            vae_scale_factor = 8

            def __init__(self):
                self.scheduler = scheduler
                self.transformer = object()

        with mock.patch.object(_dype_patch, '_patch_transformer', return_value=lambda: None):
            with _dype_patch.dype_context(
                    _Pipeline(), 'flux1', height=1024, width=1024):
                self.assertEqual(scheduler.config.max_image_seq_len, 4096)

    def test_resolution_shift_does_not_apply_to_other_pipeline_types(self):
        from diffusers import DiffusionPipeline
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler(
            use_dynamic_shifting=True,
            max_image_seq_len=4096)

        class _Pipeline(DiffusionPipeline):
            vae_scale_factor = 8

            def __init__(self):
                self.scheduler = scheduler
                self.transformer = object()

        with mock.patch.object(_dype_patch, '_patch_transformer', return_value=lambda: None):
            with _dype_patch.dype_context(
                    _Pipeline(), 'flux1', height=3072, width=3072):
                self.assertEqual(scheduler.config.max_image_seq_len, 4096)

    def test_zimage_yarn_keeps_magnitude_scale(self):
        class _Rope:
            axes_dims = (16, 56, 56)
            theta = 256.0

        class _Transformer:
            def __init__(self):
                self.rope_embedder = _Rope()

        transformer = _Transformer()
        restore = _dype_patch._patch_zimage_rope(transformer, 256, 'yarn')
        try:
            freqs = transformer.rope_embedder(torch.tensor([[0, 40, 0]]))
        finally:
            restore()
        self.assertGreater(float(freqs.abs().max()), 1.01)

    def test_yarn_mscale_modes(self):
        from dgenerate.extras.dype.rope import dype_rope_1d

        def magnitude(mode, timestep):
            cos, sin = dype_rope_1d(
                torch.arange(192), 56, theta=10000.0, base_length=64, current_length=192,
                timestep=timestep, method='yarn', mscale=mode)
            return float((cos ** 2 + sin ** 2).sqrt().max())

        constant = 0.1 * math.log(3.0) + 1.0
        self.assertAlmostEqual(magnitude('constant', 1.0), constant, places=5)
        self.assertAlmostEqual(magnitude('constant', 0.1), constant, places=5)
        self.assertAlmostEqual(magnitude('decay', 1.0), constant, places=5)
        self.assertAlmostEqual(magnitude('decay', 0.5), 1.0 + (constant - 1.0) * 0.25, places=5)
        self.assertAlmostEqual(magnitude('off', 1.0), 1.0, places=5)
        with self.assertRaises(ValueError):
            magnitude('bogus', 1.0)

    def test_config_mscales_imply_dype_and_validate(self):
        config = _config(
            model_path='black-forest-labs/FLUX.1-dev',
            model_type=_pipelinewrapper.ModelType.FLUX,
            dype_mscales=['decay', 'off'],
            prompts=['a horse'])
        config.check()
        self.assertTrue(config.dype)

        invalid = _config(
            model_path='black-forest-labs/FLUX.1-dev',
            model_type=_pipelinewrapper.ModelType.FLUX,
            dype_mscales=['bogus'],
            prompts=['a horse'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            invalid.check()

    def test_shared_axis_scaling(self):
        from dgenerate.extras.dype.rope import dype_rope_1d

        def rope(length, shared_scale):
            return dype_rope_1d(
                torch.arange(length), 56, theta=10000.0, base_length=64, current_length=length,
                timestep=0.7, method='yarn', shared_scale=shared_scale)

        # The long axis and square images are unchanged by sharing.
        self.assertTrue(torch.equal(rope(128, None)[0], rope(128, 2.0)[0]))
        # A short axis at the base length is left alone independently,
        # but receives the image-wide rebalancing and magnitude when shared.
        independent, _ = rope(64, None)
        shared, _ = rope(64, 2.0)
        self.assertAlmostEqual(float(independent.abs().max()), 1.0, places=5)
        self.assertAlmostEqual(float(shared.abs().max()), 0.1 * math.log(2.0) + 1.0, places=5)
        self.assertFalse(torch.allclose(independent, shared))

    def test_flux_shared_axis_scaling_uses_long_side(self):
        class _PosEmbed:
            axes_dim = (16, 56, 56)
            theta = 10000

        class _Transformer:
            def __init__(self):
                self.pos_embed = _PosEmbed()

        image = torch.zeros(128, 64, 3)
        image[..., 1] = torch.arange(128)[:, None]
        image[..., 2] = torch.arange(64)[None, :]
        ids = image.reshape(-1, 3)

        outputs = {}
        for axis_scaling in ('independent', 'shared'):
            transformer = _Transformer()
            restore = _dype_patch._patch_flux_pos_embed(
                transformer, None, 'yarn', axis_scaling=axis_scaling)
            try:
                outputs[axis_scaling] = transformer.pos_embed.forward(ids)[0]
            finally:
                restore()
        height = slice(16, 72)
        width = slice(72, 128)
        self.assertTrue(torch.equal(
            outputs['independent'][:, height], outputs['shared'][:, height]))
        self.assertFalse(torch.allclose(
            outputs['independent'][:, width], outputs['shared'][:, width]))

    def test_config_axis_scalings_imply_dype_and_validate(self):
        config = _config(
            model_path='black-forest-labs/FLUX.1-dev',
            model_type=_pipelinewrapper.ModelType.FLUX,
            dype_axis_scalings=['shared'],
            prompts=['a horse'])
        config.check()
        self.assertTrue(config.dype)

        invalid = _config(
            model_path='black-forest-labs/FLUX.1-dev',
            model_type=_pipelinewrapper.ModelType.FLUX,
            dype_axis_scalings=['bogus'],
            prompts=['a horse'])
        with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
            invalid.check()

    def test_wan_base_matches_precomputed_rope(self):
        from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed

        rope = WanRotaryPosEmbed(64, (1, 2, 2), 64)

        class _Holder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rope = rope

        holder = _Holder()
        hidden = torch.zeros(1, 4, 2, 16, 16)
        original = rope.forward(hidden)
        restore = _dype_patch._patch_wan_rope(holder, None, 'base')
        try:
            patched = holder.rope.forward(hidden)
        finally:
            restore()
        self.assertTrue(torch.equal(original[0], patched[0]))
        self.assertTrue(torch.equal(original[1], patched[1]))

    def test_wan_base_extends_past_precomputed_buffer(self):
        from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed

        rope = WanRotaryPosEmbed(64, (1, 2, 2), 4)

        class _Holder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rope = rope

        holder = _Holder()
        hidden = torch.zeros(1, 4, 1, 16, 16)
        restore = _dype_patch._patch_wan_rope(holder, None, 'base')
        try:
            cos, sin = holder.rope.forward(hidden)
        finally:
            restore()
        self.assertEqual(tuple(cos.shape), (1, 64, 1, 64))
        self.assertEqual(sin.shape, cos.shape)
