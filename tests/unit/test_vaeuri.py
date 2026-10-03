import tempfile
import unittest

import diffusers

import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.uris.vaeuri as _vaeuri
from dgenerate.pipelinewrapper.uris.exceptions import InvalidVaeUriError


class TestVAEUri(unittest.TestCase):

    def test_basic_parsing(self):
        # Test basic model path parsing
        uri = "AutoencoderKL;model=path/to/model"
        result = _vaeuri.VAEUri.parse(uri)
        self.assertEqual(result.encoder, "AutoencoderKL")
        self.assertEqual(result.model, "path/to/model")
        self.assertEqual(result.revision, None)
        self.assertEqual(result.variant, None)
        self.assertEqual(result.subfolder, None)
        self.assertEqual(result.dtype, None)
        self.assertEqual(result.extract, False)

    def test_full_options_parsing(self):
        # Test parsing with all options
        uri = "AutoencoderKL;model=path/to/model;revision=v1.0;variant=fp16;subfolder=models;dtype=float16"
        
        result = _vaeuri.VAEUri.parse(uri)
        self.assertEqual(result.encoder, "AutoencoderKL")
        self.assertEqual(result.model, "path/to/model")
        self.assertEqual(result.revision, "v1.0")
        self.assertEqual(result.variant, "fp16")
        self.assertEqual(result.subfolder, "models")
        self.assertEqual(result.dtype, _enums.DataType.FLOAT16)

    def test_encoder_validation(self):
        # Test encoder validation
        
        # Valid encoders
        uri = "AutoencoderKL;model=path/to/model"
        result = _vaeuri.VAEUri.parse(uri)
        self.assertEqual(result.encoder, "AutoencoderKL")
        
        uri = "AsymmetricAutoencoderKL;model=path/to/model"
        result = _vaeuri.VAEUri.parse(uri)
        self.assertEqual(result.encoder, "AsymmetricAutoencoderKL")
        
        uri = "AutoencoderTiny;model=path/to/model"
        result = _vaeuri.VAEUri.parse(uri)
        self.assertEqual(result.encoder, "AutoencoderTiny")
        
        uri = "ConsistencyDecoderVAE;model=path/to/model"
        result = _vaeuri.VAEUri.parse(uri)
        self.assertEqual(result.encoder, "ConsistencyDecoderVAE")

        for encoder in _vaeuri.VAEUri.supported_encoder_names():
            result = _vaeuri.VAEUri.parse(f"{encoder};model=path/to/model")
            self.assertEqual(result.encoder, encoder)
        
        # Invalid encoder
        with self.assertRaises(InvalidVaeUriError):
            _vaeuri.VAEUri.parse("InvalidEncoder;model=path/to/model")

    def test_dtype_validation(self):
        # Test dtype validation
        
        # Valid dtypes
        uri = "AutoencoderKL;model=path/to/model;dtype=float16"
        result = _vaeuri.VAEUri.parse(uri)
        self.assertEqual(result.dtype, _enums.DataType.FLOAT16)
        
        uri = "AutoencoderKL;model=path/to/model;dtype=float32"
        result = _vaeuri.VAEUri.parse(uri)
        self.assertEqual(result.dtype, _enums.DataType.FLOAT32)
        
        # Invalid dtype
        with self.assertRaises(InvalidVaeUriError):
            _vaeuri.VAEUri.parse("AutoencoderKL;model=path/to/model;dtype=invalid_dtype")

    def test_model_required(self):
        # Test that model is required
        with self.assertRaises(InvalidVaeUriError):
            _vaeuri.VAEUri.parse("AutoencoderKL")

    def _roundtrip_load(self, encoder_name, model):
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = _vaeuri.VAEUri.parse(
                f'{encoder_name};model={directory};dtype=float32'
            ).load(local_files_only=True, no_cache=True)
            self.assertEqual(type(loaded).__name__, encoder_name)

    def test_load_autoencoder_kl_wan(self):
        config = {
            'base_dim': 8,
            'z_dim': 4,
            'dim_mult': [1, 1],
            'num_res_blocks': 1,
            'temperal_downsample': [False],
            'dropout': 0.0,
            'attn_scales': [],
            'latents_mean': [0.0] * 4,
            'latents_std': [1.0] * 4,
        }
        self._roundtrip_load(
            'AutoencoderKLWan',
            diffusers.AutoencoderKLWan.from_config(config))

    def test_load_autoencoder_kl_ltx_video(self):
        config = {
            'in_channels': 3,
            'out_channels': 3,
            'latent_channels': 4,
            'block_out_channels': [8, 8],
            'layers_per_block': [1, 1, 1],
            'patch_size': 4,
            'patch_size_t': 1,
            'spatio_temporal_scaling': [False, False],
            'encoder_causal': True,
            'decoder_causal': False,
            'resnet_norm_eps': 1e-6,
            'scaling_factor': 1.0,
        }
        self._roundtrip_load(
            'AutoencoderKLLTXVideo',
            diffusers.AutoencoderKLLTXVideo.from_config(config))

    def test_load_autoencoder_kl_ltx2_video(self):
        config = {
            'in_channels': 3,
            'out_channels': 3,
            'latent_channels': 4,
            'block_out_channels': [8, 8],
            'decoder_block_out_channels': [8, 8],
            'layers_per_block': [1, 1, 1],
            'decoder_layers_per_block': [1, 1, 1],
            'down_block_types': ['LTX2VideoDownBlock3D', 'LTX2VideoDownBlock3D'],
            'downsample_type': ['spatial', 'spatial'],
            'spatio_temporal_scaling': [False, False],
            'decoder_spatio_temporal_scaling': [False, False],
            'decoder_inject_noise': [False, False, False],
            'upsample_type': ['spatial', 'spatial'],
            'upsample_factor': [1, 1],
            'upsample_residual': [False, False],
            'patch_size': 4,
            'patch_size_t': 1,
            'encoder_causal': True,
            'decoder_causal': False,
            'resnet_norm_eps': 1e-6,
            'scaling_factor': 1.0,
            'spatial_compression_ratio': 4,
            'temporal_compression_ratio': 1,
            'timestep_conditioning': False,
            'decoder_spatial_padding_mode': 'zeros',
            'encoder_spatial_padding_mode': 'zeros',
        }
        self._roundtrip_load(
            'AutoencoderKLLTX2Video',
            diffusers.AutoencoderKLLTX2Video.from_config(config))

    def test_load_autoencoder_kl_flux2(self):
        config = {
            'in_channels': 3,
            'out_channels': 3,
            'down_block_types': ['DownEncoderBlock2D', 'DownEncoderBlock2D'],
            'up_block_types': ['UpDecoderBlock2D', 'UpDecoderBlock2D'],
            'block_out_channels': [8, 8],
            'layers_per_block': 1,
            'latent_channels': 4,
            'norm_num_groups': 4,
            'sample_size': 32,
        }
        self._roundtrip_load(
            'AutoencoderKLFlux2',
            diffusers.AutoencoderKLFlux2.from_config(config))

    def test_load_autoencoder_kl_qwen_image(self):
        config = {
            'base_dim': 8,
            'z_dim': 4,
            'dim_mult': [1, 1],
            'num_res_blocks': 1,
            'temperal_downsample': [False],
            'dropout': 0.0,
            'latents_mean': [0.0] * 4,
            'latents_std': [1.0] * 4,
        }
        self._roundtrip_load(
            'AutoencoderKLQwenImage',
            diffusers.AutoencoderKLQwenImage.from_config(config))


if __name__ == '__main__':
    unittest.main() 