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
    def __init__(self, mismatch=False):
        super().__init__()
        self.early = _PairBlock()
        self.cache_block = _TensorBlock() if mismatch else _PairBlock()
        self.transformer_blocks = torch.nn.ModuleList([self.early, self.cache_block])
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
    def __init__(self, mismatch=False):
        super().__init__()
        self.transformer = _Denoiser(mismatch=mismatch)


class TestSpectrum(unittest.TestCase):
    def test_linear_chebyshev_fit(self):
        forecaster = _forecaster.ChebyshevForecaster(M=1, K=10, lam=1e-8)
        for time in (0.0, 0.25, 0.5, 0.75):
            forecaster.update(time, torch.tensor([time, time]))
        predicted = forecaster.predict(1.0)
        self.assertTrue(torch.allclose(predicted, torch.ones(1, 2), atol=1e-4))

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
        self.assertFalse(_pipelinewrapper.model_type_supports_spectrum('sd'))
        self.assertFalse(_pipelinewrapper.model_type_supports_spectrum('ltx'))
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

        for model_type in ('sd', 'ltx'):
            config = _renderloopconfig.RenderLoopConfig()
            config.model_type = model_type
            config.spectrum = True
            with self.assertRaises(_renderloopconfig.RenderLoopConfigError):
                config._check_optimization_features(lambda name: name)
