import unittest

import torch

import dgenerate.arguments as _arguments
import dgenerate.extras.spectrum as _spectrum
import dgenerate.extras.spectrum.forecaster as _forecaster
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.renderloopconfig as _renderloopconfig


def _settings(**overrides):
    values = dict(
        weight=1.0,
        order=1,
        lam=1e-6,
        warmup_steps=2,
        window_size=2.0,
        flex_window=0.75,
        fallback_steps=8,
    )
    values.update(overrides)
    return _spectrum.SpectrumSettings(**values)


class _PairBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden, encoder_hidden_states=None):
        self.calls += 1
        return encoder_hidden_states, hidden + float(self.calls)


class _TensorBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden, encoder_hidden_states=None):
        self.calls += 1
        return hidden + float(self.calls)


class _Denoiser(torch.nn.Module):
    def __init__(self, mismatch=False, block_attr='transformer_blocks'):
        super().__init__()
        self.early = _PairBlock()
        self.cache_block = _TensorBlock() if mismatch else _PairBlock()
        blocks = torch.nn.ModuleList([self.early, self.cache_block])
        setattr(self, block_attr, blocks)
        self.mismatch = mismatch
        self.last_early = None

    def forward(self, hidden, timestep=None, encoder_hidden_states=None):
        encoder, early = self.early(hidden, encoder_hidden_states=encoder_hidden_states)
        self.last_early = early
        if self.mismatch:
            return self.cache_block(hidden[..., :2], encoder_hidden_states=encoder)
        encoder, hidden = self.cache_block(early, encoder_hidden_states=encoder)
        return hidden


class _Pipeline(torch.nn.Module):
    def __init__(self, mismatch=False, block_attr='transformer_blocks'):
        super().__init__()
        self.transformer = _Denoiser(mismatch=mismatch, block_attr=block_attr)


class TestSpectrum(unittest.TestCase):
    def test_linear_chebyshev_fit(self):
        forecaster = _forecaster.ChebyshevForecaster(M=1, K=10, lam=1e-8)
        for time in (0.0, 0.25, 0.5, 0.75):
            forecaster.update(time, torch.tensor([time, time]))
        predicted = forecaster.predict(1.0)
        self.assertTrue(torch.allclose(predicted, torch.ones(1, 2), atol=1e-4))

    def test_short_history_does_not_use_degree_four(self):
        # Four full passes is the whole Krea Turbo budget. T_4 is not
        # identified yet; the fit has to stay on a lower degree.
        forecaster = _forecaster.ChebyshevForecaster(M=4, K=100, lam=0.1)
        for step in (0.0, 1.0, 3.0, 5.0):
            forecaster.update(step / 7.0, torch.tensor([step]))
        self.assertEqual(forecaster._active_degree(), 2)
        predicted = forecaster.predict(6.0 / 7.0)
        self.assertLess(abs(predicted.item() - 6.0), 1.5)

    def test_skip_schedule_and_restore(self):
        pipe = _Pipeline()
        hidden = torch.zeros(1, 2, 4)
        encoder = torch.zeros(1, 1, 3)
        full_steps = []
        with _spectrum.apply(pipe, 8, _settings()):
            for timestep in torch.linspace(1, 0, 8):
                before = pipe.transformer.cache_block.calls
                pipe.transformer(
                    hidden, timestep=timestep, encoder_hidden_states=encoder)
                full_steps.append(pipe.transformer.cache_block.calls > before)
        self.assertEqual(
            full_steps,
            [True, True, False, True, False, True, False, False])
        self.assertEqual(pipe.transformer.early.calls, 4)
        restored = pipe.transformer.cache_block.calls
        pipe.transformer(hidden, timestep=torch.tensor(0.1), encoder_hidden_states=encoder)
        self.assertEqual(pipe.transformer.cache_block.calls, restored + 1)
        self.assertEqual(pipe.transformer.early.calls, 5)

    def test_tail_forecast_does_not_ring(self):
        pipe = _Pipeline()
        hidden = torch.zeros(1, 2, 4)
        encoder = torch.zeros(1, 1, 3)
        outputs = []
        with _spectrum.apply(pipe, 8, _settings(weight=0.5, order=4, lam=0.1)):
            for timestep in torch.linspace(1, 0, 8):
                outputs.append(pipe.transformer(
                    hidden, timestep=timestep, encoder_hidden_states=encoder))
        # Full passes land on 2, 4, 6, 8. An underdetermined T_4 turns the
        # last predicted step back down through 5; the capped fit stays above
        # the last real pass.
        self.assertGreater(outputs[-1].flatten()[0].item(), 7.5)

    def test_schedule_warning_names_the_tail_and_guidance(self):
        from unittest.mock import patch

        import dgenerate.extras.spectrum.runtime as _runtime

        tuned = _spectrum.resolve_settings(
            _settings(warmup_steps=None, flex_window=None), 28, None)
        self.assertEqual((tuned.warmup_steps, tuned.flex_window), (4, 0.0))
        guided = _spectrum.resolve_settings(
            _settings(warmup_steps=None, flex_window=None), 50, 7.5)
        self.assertEqual((guided.warmup_steps, guided.flex_window), (5, 0.0))
        paper = _spectrum.resolve_settings(
            _settings(warmup_steps=None, flex_window=None), 50, None)
        self.assertEqual((paper.warmup_steps, paper.flex_window), (5, 0.75))

        full, skipped, trailing = _spectrum.schedule_counts(28, _settings())
        self.assertEqual((full, skipped, trailing), (8, 20, 5))
        paper = _spectrum.schedule_counts(50, _settings(warmup_steps=5))
        self.assertEqual(paper[2], 3)

        class _Config:
            def get(self, name, default=None):
                return default

        class _Pipe:
            class transformer:
                config = _Config()

        _runtime._WARNED.clear()
        with patch('dgenerate.extras.spectrum.runtime._messages.warning') as warning:
            _spectrum.warn_schedule(_Pipe(), 28, _settings(), guidance_scale=7.5)
            _spectrum.warn_schedule(_Pipe(), 28, _settings(), guidance_scale=7.5)
        self.assertEqual(warning.call_count, 1)
        text = warning.call_args.args[0]
        self.assertIn('last 5 steps', text)
        self.assertIn('--spectrum-flex-windows 0', text)
        self.assertIn('--inference-steps 50', text)
        self.assertIn('this flex was measured on', text)
        self.assertNotIn('Guidance', text)

        _runtime._WARNED.clear()
        with patch('dgenerate.extras.spectrum.runtime._messages.warning') as warning:
            _spectrum.warn_schedule(
                _Pipe(), 8, _settings(window_size=8.0, flex_window=0.0, warmup_steps=2))
        self.assertEqual(warning.call_count, 1)
        custom = warning.call_args.args[0]
        self.assertNotIn('this flex was measured on', custom)
        self.assertIn('--spectrum-warmup-steps 5', custom)

        class _Embeds:
            def get(self, name, default=None):
                return True if name == 'guidance_embeds' else default

        class _Flux:
            class transformer:
                config = _Embeds()

        _runtime._WARNED.clear()
        with patch('dgenerate.extras.spectrum.runtime._messages.warning') as warning:
            _spectrum.warn_schedule(_Flux(), 50, _settings(warmup_steps=5), guidance_scale=3.5)
        warning.assert_not_called()

        _runtime._WARNED.clear()
        with patch('dgenerate.extras.spectrum.runtime._messages.warning') as warning:
            _spectrum.warn_schedule(
                _Pipe(), 28, _settings(flex_window=0.0), guidance_scale=1)
        warning.assert_not_called()

        _runtime._WARNED.clear()
        with patch('dgenerate.extras.spectrum.runtime._messages.warning') as warning:
            _spectrum.warn_schedule(_Pipe(), 50, _settings(warmup_steps=5), cfg=7.5)
        warning.assert_not_called()

    def test_image_first_pair_keeps_the_forecast_in_the_image_slot(self):
        class _ImageFirst(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, hidden, encoder_hidden_states=None):
                self.calls += 1
                return hidden + float(self.calls), encoder_hidden_states

        class _Denoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block = _ImageFirst()
                self.transformer_blocks = torch.nn.ModuleList([self.block])

            def forward(self, hidden, timestep=None, encoder_hidden_states=None):
                image, _encoder = self.block(
                    hidden, encoder_hidden_states=encoder_hidden_states)
                return image

        pipe = torch.nn.Module()
        pipe.transformer = _Denoiser()
        hidden = torch.zeros(1, 2, 4)
        encoder = torch.ones(1, 1, 3)
        skipped = None
        with _spectrum.apply(pipe, 4, _settings(warmup_steps=2, window_size=2.0, flex_window=0.0)):
            for timestep in (1.0, 0.6, 0.3, 0.0):
                before = pipe.transformer.block.calls
                output = pipe.transformer(
                    hidden, timestep=torch.tensor(timestep), encoder_hidden_states=encoder)
                if skipped is None and pipe.transformer.block.calls == before:
                    skipped = output
        self.assertIsNotNone(skipped)
        self.assertEqual(tuple(skipped.shape), tuple(hidden.shape))
        self.assertLess((skipped - 3).abs().max().item(), 0.15)

    def test_face_adapter_does_not_add_on_a_predicted_step(self):
        class _Face(torch.nn.Module):
            def forward(self, hidden_states, motion=None):
                return torch.ones_like(hidden_states)

        class _Denoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block = _TensorBlock()
                self.blocks = torch.nn.ModuleList([self.block])
                self.face_adapter = torch.nn.ModuleList([_Face()])

            def forward(self, hidden, timestep=None, encoder_hidden_states=None):
                hidden = self.block(hidden, encoder_hidden_states=encoder_hidden_states)
                return hidden + self.face_adapter[0](hidden, encoder_hidden_states)

        pipe = torch.nn.Module()
        pipe.transformer = _Denoiser()
        hidden = torch.zeros(1, 2, 4)
        encoder = torch.zeros(1, 1, 3)
        outputs = []
        with _spectrum.apply(pipe, 4, _settings(warmup_steps=2, window_size=2.0, flex_window=0.0, weight=0.0)):
            for timestep in (1.0, 0.6, 0.3):
                outputs.append(pipe.transformer(
                    hidden, timestep=torch.tensor(timestep), encoder_hidden_states=encoder))
        # Full steps add the face residual. The predicted step must not.
        self.assertTrue(torch.equal(outputs[0], torch.full_like(hidden, 2)))
        self.assertTrue(torch.equal(outputs[1], torch.full_like(hidden, 3)))
        self.assertLess((outputs[2] - 3).abs().max().item(), 0.15)

    def test_hydit_blocks_module_list(self):
        pipe = _Pipeline(block_attr='blocks')
        hidden = torch.zeros(1, 2, 4)
        encoder = torch.zeros(1, 1, 3)
        with _spectrum.apply(pipe, 4, _settings(warmup_steps=0, window_size=2.0)):
            for timestep in torch.linspace(1, 0, 4):
                pipe.transformer(hidden, timestep=timestep, encoder_hidden_states=encoder)
        self.assertGreater(pipe.transformer.cache_block.calls, 0)

    def test_shape_mismatch_identity_pass(self):
        pipe = _Pipeline(mismatch=True)
        hidden = torch.zeros(1, 4)
        encoder = torch.zeros(1, 3)
        with _spectrum.apply(pipe, 4, _settings(warmup_steps=2, fallback_steps=4)):
            for timestep in (1.0, 0.6, 0.3):
                pipe.transformer(
                    hidden, timestep=torch.tensor(timestep), encoder_hidden_states=encoder)
            self.assertEqual(pipe.transformer.early.calls, 2)
            self.assertEqual(pipe.transformer.cache_block.calls, 2)
            self.assertTrue(torch.equal(pipe.transformer.last_early, hidden))

    def test_sdxl_unet_skip_keeps_output_convolution(self):
        from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

        unet = UNet2DConditionModel(
            sample_size=16,
            in_channels=4,
            out_channels=4,
            layers_per_block=1,
            block_out_channels=(32, 64),
            down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
            up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'),
            cross_attention_dim=32,
            attention_head_dim=4,
            norm_num_groups=4,
        )
        calls = {'resnet': 0, 'conv_out': 0}

        def _count_resnet(_module, _inputs, _output):
            calls['resnet'] += 1

        def _count_conv(_module, _inputs, _output):
            calls['conv_out'] += 1

        unet.down_blocks[0].resnets[0].register_forward_hook(_count_resnet)
        unet.conv_out.register_forward_hook(_count_conv)
        pipe = torch.nn.Module()
        pipe.unet = unet
        sample = torch.zeros(1, 4, 16, 16)
        encoder = torch.zeros(1, 4, 32)
        full_steps = []
        with _spectrum.apply(pipe, 8, _settings()):
            for timestep in torch.linspace(1, 0, 8):
                before = calls['resnet']
                pipe.unet(sample, timestep, encoder)
                full_steps.append(calls['resnet'] > before)
        self.assertEqual(
            full_steps,
            [True, True, False, True, False, True, False, False])
        self.assertEqual(calls['conv_out'], 8)

    def test_missing_blocks(self):
        class _Empty(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = torch.nn.Linear(2, 2)

        with self.assertRaises(_spectrum.SpectrumUnsupported):
            with _spectrum.apply(_Empty(), 4, _settings()):
                pass

    def test_cli_and_unsupported_model_types(self):
        parsed, _unknown = _arguments.parse_known_args([
            '--spectrum',
            '--spectrum-weights', '0.5',
            '--spectrum-orders', '4',
            '--spectrum-lambdas', '0.1',
            '--spectrum-warmup-steps', '2',
            '--spectrum-window-sizes', '2',
            '--spectrum-flex-windows', '0.75',
        ], throw=True, log_error=False)
        self.assertTrue(parsed.spectrum)
        self.assertEqual(parsed.spectrum_weights, [0.5])
        self.assertEqual(parsed.spectrum_orders, [4])
        self.assertEqual(parsed.spectrum_flex_windows, [0.75])

        self.assertTrue(_pipelinewrapper.model_type_supports_spectrum('flux'))
        self.assertTrue(_pipelinewrapper.model_type_supports_spectrum('wan'))
        self.assertTrue(_pipelinewrapper.model_type_supports_spectrum('sdxl'))
        self.assertTrue(_pipelinewrapper.model_type_supports_spectrum('sdxl-pix2pix'))
        self.assertTrue(_pipelinewrapper.model_type_supports_spectrum('kolors'))
        self.assertTrue(_pipelinewrapper.model_type_supports_dype('krea2'))
        for model_type in ('qwen-image-21', 'krea2', 'kandinsky5-image',
                           'kandinsky5-video', 'hyvideo-15', 'motif-video',
                           'hyimage-21', 'hyimage-21-refiner'):
            self.assertTrue(_pipelinewrapper.model_type_supports_spectrum(model_type))
        self.assertFalse(_pipelinewrapper.model_type_supports_spectrum('sd'))
        self.assertFalse(_pipelinewrapper.model_type_supports_spectrum('ltx2'))
        self.assertFalse(_pipelinewrapper.model_type_supports_spectrum('hydit'))
        self.assertIsInstance(_pipelinewrapper.SPECTRUM_MODEL_TYPES, tuple)
        self.assertTrue(_pipelinewrapper.model_type_supports_tea_cache('flux-fill'))
        self.assertFalse(_pipelinewrapper.model_type_supports_tea_cache('flux2'))
        self.assertTrue(_pipelinewrapper.model_type_supports_deep_cache('upscaler-x4'))
        self.assertFalse(_pipelinewrapper.model_type_supports_deep_cache('upscaler-x2'))
        self.assertTrue(_pipelinewrapper.model_type_supports_hi_diffusion('kolors'))
        self.assertFalse(_pipelinewrapper.model_type_supports_hi_diffusion('pix2pix'))
        self.assertTrue(_pipelinewrapper.model_type_supports_sada('flux-kontext'))
        self.assertFalse(_pipelinewrapper.model_type_supports_sada('sdxl-pix2pix'))
        self.assertTrue(_pipelinewrapper.model_type_supports_ras('sd3-pix2pix'))
        self.assertFalse(_pipelinewrapper.model_type_supports_ras('sdxl'))
        self.assertTrue(_pipelinewrapper.model_type_supports_freeu('upscaler-x2'))
        self.assertFalse(_pipelinewrapper.model_type_supports_freeu('sd3'))
        self.assertTrue(_pipelinewrapper.model_type_supports_pag('sd3'))
        self.assertFalse(_pipelinewrapper.model_type_supports_pag('flux'))

    def test_wrapper_uses_krea2_schedule_defaults(self):
        import types

        import dgenerate.pipelinewrapper.arguments as _diffusion_arguments
        import dgenerate.pipelinewrapper.wrapper as _wrapper

        def settings(distilled, **overrides):
            pipe = types.SimpleNamespace(
                _model_type=_pipelinewrapper.ModelType.KREA2,
                _pipeline=types.SimpleNamespace(config={'is_distilled': distilled}))
            args = _diffusion_arguments.GenerationArguments()
            args.spectrum = True
            for name, value in overrides.items():
                setattr(args, name, value)
            return _wrapper.PipelineWrapper._spectrum_settings(pipe, args)

        turbo = settings(True)
        self.assertEqual(turbo.warmup_steps, 2)
        self.assertEqual(turbo.flex_window, 0.0)
        self.assertEqual(turbo.window_size, 2.0)

        raw = settings(False)
        self.assertEqual(raw.warmup_steps, 4)
        self.assertEqual(raw.flex_window, 0.0)

        short_raw = settings(False, inference_steps=8)
        self.assertEqual(short_raw.warmup_steps, 2)

        long_turbo = settings(True, inference_steps=28)
        self.assertEqual(long_turbo.warmup_steps, 4)

        explicit = settings(True, spectrum_flex_window=3.0, spectrum_warmup_steps=1)
        self.assertEqual(explicit.flex_window, 3.0)
        self.assertEqual(explicit.warmup_steps, 1)

        unloaded = types.SimpleNamespace(
            _model_type=_pipelinewrapper.ModelType.KREA2, _pipeline=None)
        omitted = _diffusion_arguments.GenerationArguments()
        omitted.spectrum = True
        pending = _wrapper.PipelineWrapper._spectrum_settings(unloaded, omitted)
        self.assertIsNone(pending.warmup_steps)
        self.assertIsNone(pending.flex_window)
        counted = _diffusion_arguments.GenerationArguments()
        counted.spectrum = True
        counted.inference_steps = 8
        counted_settings = _wrapper.PipelineWrapper._spectrum_settings(unloaded, counted)
        self.assertEqual(counted_settings.warmup_steps, 2)
        self.assertEqual(counted_settings.flex_window, 0.0)

        for model_type in ('sd', 'ltx2', 'ltx-video'):
            config = _renderloopconfig.RenderLoopConfig()
            config.model_type = model_type
            config.spectrum = True
            with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
                config._check_optimization_features(lambda name: name)
