import unittest

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
