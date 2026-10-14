import unittest

import torch

import dgenerate.extras.sada.patch as _sada
import dgenerate.pipelinewrapper.pipelines as _pipelines


class _Bus:
    pass


class _Denoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        bus = _Bus()
        bus.step = 12
        bus.skip_this_step = True
        bus.prev_epsilon = torch.ones(1)
        bus.prev_f = [torch.ones(1)]
        self._cache_bus = bus


class _Pipeline:
    def __init__(self):
        self.unet = _Denoiser()


class TestSadaReset(unittest.TestCase):
    def test_reset_clears_step_cache(self):
        pipeline = _Pipeline()
        _sada.reset_cache(pipeline)
        bus = pipeline.unet._cache_bus
        self.assertEqual(bus.step, 0)
        self.assertFalse(bus.skip_this_step)
        self.assertIsNone(bus.prev_epsilon)
        self.assertEqual(bus.prev_f, [None, None])

    def test_call_helper_resets_only_patched_denoisers(self):
        patched = _Pipeline()
        plain = _Pipeline()
        del plain.unet._cache_bus
        _pipelines._reset_sada_cache(patched)
        _pipelines._reset_sada_cache(plain)
        self.assertEqual(patched.unet._cache_bus.step, 0)
        self.assertFalse(hasattr(plain.unet, '_cache_bus'))
