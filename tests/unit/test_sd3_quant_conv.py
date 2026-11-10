import unittest

import torch

import dgenerate.pipelinewrapper.pipelines as _pipelines


class _TinyVae(torch.nn.Module):
    pass


class _KlVae(torch.nn.Module):
    def __init__(self, quant_conv, post_quant_conv):
        super().__init__()
        self.quant_conv = quant_conv
        self.post_quant_conv = post_quant_conv


class TestSd3QuantConv(unittest.TestCase):
    def test_tiny_vae_is_left_alone(self):
        vae = _TinyVae()
        _pipelines.fill_unset_sd3_quant_convs(vae)
        with self.assertRaises(AttributeError):
            vae.quant_conv
        with self.assertRaises(AttributeError):
            vae.post_quant_conv

    def test_unset_kl_convolutions_become_identity(self):
        vae = _KlVae(None, None)
        _pipelines.fill_unset_sd3_quant_convs(vae)
        sample = torch.zeros(1)
        self.assertTrue(torch.equal(vae.quant_conv(sample), sample))
        self.assertTrue(torch.equal(vae.post_quant_conv(sample), sample))

    def test_existing_kl_convolutions_stay(self):
        quant = torch.nn.Identity()
        post = torch.nn.Identity()
        vae = _KlVae(quant, post)
        _pipelines.fill_unset_sd3_quant_convs(vae)
        self.assertIs(vae.quant_conv, quant)
        self.assertIs(vae.post_quant_conv, post)


if __name__ == '__main__':
    unittest.main()
