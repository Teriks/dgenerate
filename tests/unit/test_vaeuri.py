import unittest

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


if __name__ == '__main__':
    unittest.main() 