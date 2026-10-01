import unittest
import unittest.mock as mock

import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.uris.textencoderuri as _textencoderuri
import dgenerate.hfhub as _hfhub
from dgenerate.pipelinewrapper.uris.exceptions import InvalidTextEncoderUriError


class TestTextEncoderUri(unittest.TestCase):

    def test_basic_parsing(self):
        # Test basic model path parsing
        uri = "CLIPTextModel;model=path/to/model"
        result = _textencoderuri.TextEncoderUri.parse(uri)
        self.assertEqual(result.encoder, "CLIPTextModel")
        self.assertEqual(result.model, "path/to/model")
        self.assertEqual(result.revision, None)
        self.assertEqual(result.variant, None)
        self.assertEqual(result.subfolder, None)
        self.assertEqual(result.dtype, None)
        self.assertEqual(result.quantizer, False)  # Default is False, not None
        self.assertEqual(result.mode, None)

    def test_full_options_parsing(self):
        # Test parsing with all options
        uri = "CLIPTextModel;model=path/to/model;revision=v1.0;variant=fp16;subfolder=models;dtype=float16;quantizer=bnb"
        
        result = _textencoderuri.TextEncoderUri.parse(uri)
        self.assertEqual(result.encoder, "CLIPTextModel")
        self.assertEqual(result.model, "path/to/model")
        self.assertEqual(result.revision, "v1.0")
        self.assertEqual(result.variant, "fp16")
        self.assertEqual(result.subfolder, "models")
        self.assertEqual(result.dtype, _enums.DataType.FLOAT16)
        self.assertEqual(result.quantizer, "bnb")
        self.assertEqual(result.mode, None)

    def test_encoder_validation(self):
        # Test encoder validation
        
        # Valid encoders
        for encoder in _textencoderuri.TextEncoderUri.supported_encoder_names():
            uri = f"{encoder};model=path/to/model"
            result = _textencoderuri.TextEncoderUri.parse(uri)
            self.assertEqual(result.encoder, encoder)
        
        # Invalid encoder
        with self.assertRaises(InvalidTextEncoderUriError) as context:
            _textencoderuri.TextEncoderUri.parse("InvalidEncoder;model=path/to/model")
        
        # Check that the error message is descriptive
        self.assertIn("Unknown TextEncoder encoder class", str(context.exception))

    def test_dtype_validation(self):
        # Test dtype validation
        
        # Valid dtypes
        uri = "CLIPTextModel;model=path/to/model;dtype=float16"
        result = _textencoderuri.TextEncoderUri.parse(uri)
        self.assertEqual(result.dtype, _enums.DataType.FLOAT16)
        
        uri = "CLIPTextModel;model=path/to/model;dtype=float32"
        result = _textencoderuri.TextEncoderUri.parse(uri)
        self.assertEqual(result.dtype, _enums.DataType.FLOAT32)
        
        # Invalid dtype
        with self.assertRaises(InvalidTextEncoderUriError) as context:
            _textencoderuri.TextEncoderUri.parse("CLIPTextModel;model=path/to/model;dtype=invalid_dtype")
        
        # Check that the error message is descriptive
        self.assertIn("must be", str(context.exception))

    def test_mode_validation(self):
        # Test mode validation
        
        # Valid modes
        for mode in _textencoderuri.TextEncoderUri._valid_modes():
            uri = f"CLIPTextModel;model=path/to/model;mode={mode}"
            result = _textencoderuri.TextEncoderUri.parse(uri)
            self.assertEqual(result.mode, mode)
        
        # Invalid mode
        with self.assertRaises(InvalidTextEncoderUriError) as context:
            _textencoderuri.TextEncoderUri.parse("CLIPTextModel;model=path/to/model;mode=invalid_mode")
        
        # Check that the error message is descriptive
        self.assertIn("Unknown TextEncoder load mode", str(context.exception))

    def test_mode_incompatible_options(self):
        # Test that mode is incompatible with variant, revision, and subfolder
        mode = _textencoderuri.TextEncoderUri._valid_modes()[0]  # Get a valid mode
        
        # Mode + variant
        with self.assertRaises(InvalidTextEncoderUriError) as context:
            _textencoderuri.TextEncoderUri.parse(f"CLIPTextModel;model=path/to/model;mode={mode};variant=fp16")
        self.assertIn("cannot use variant with mode", str(context.exception))
        
        # Mode + revision
        with self.assertRaises(InvalidTextEncoderUriError) as context:
            _textencoderuri.TextEncoderUri.parse(f"CLIPTextModel;model=path/to/model;mode={mode};revision=v1.0")
        self.assertIn("cannot use revision with mode", str(context.exception))
        
        # Mode + subfolder
        with self.assertRaises(InvalidTextEncoderUriError) as context:
            _textencoderuri.TextEncoderUri.parse(f"CLIPTextModel;model=path/to/model;mode={mode};subfolder=models")
        self.assertIn("cannot use subfolder with mode", str(context.exception))

    def test_single_file_with_quantizer_validation(self):
        # Test that single file loads with quantizer are not supported unless a valid mode is specified
        
        # Mock is_single_file_model_load to return True
        with mock.patch.object(_hfhub, 'is_single_file_model_load', return_value=True):
            # Valid: Single file with mode and quantizer
            mode = _textencoderuri.TextEncoderUri._valid_modes()[0]
            uri = f"CLIPTextModel;model=path/to/model;mode={mode};quantizer=bnb"
            result = _textencoderuri.TextEncoderUri.parse(uri)
            self.assertEqual(result.quantizer, "bnb")
            
            # Invalid: Single file with quantizer but no mode
            with self.assertRaises(InvalidTextEncoderUriError) as context:
                _textencoderuri.TextEncoderUri.parse("CLIPTextModel;model=path/to/model;quantizer=bnb")
            self.assertIn("single file loads are not supported", str(context.exception))

    def test_string_representation(self):
        # Test string representation
        uri = "CLIPTextModel;model=path/to/model;variant=fp16;revision=v1.0"
        result = _textencoderuri.TextEncoderUri.parse(uri)
        string_repr = str(result)
        
        # Just verify that string conversion works and returns a string
        self.assertTrue(isinstance(string_repr, str))

    def test_clip_skip_uses_final_layer_norm_on_flattened_clip(self):
        import torch
        from transformers import CLIPTextConfig, CLIPTextModel

        config = CLIPTextConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            max_position_embeddings=16,
            eos_token_id=2,
            bos_token_id=None,
        )
        text_encoder = CLIPTextModel(config).eval()
        self.assertIs(text_encoder.text_model.final_layer_norm, text_encoder.final_layer_norm)

        ids = torch.randint(0, config.vocab_size, (2, 8))
        ids[:, -1] = config.eos_token_id
        normal = text_encoder(ids)[0]
        hidden = text_encoder(ids, output_hidden_states=True)
        for skip in (0, 1):
            layer = hidden[-1][-(skip + 1)]
            skipped = text_encoder.text_model.final_layer_norm(layer)
            self.assertEqual(tuple(skipped.shape), tuple(normal.shape))
            if skip == 0:
                self.assertTrue(torch.allclose(skipped, normal, atol=1e-5))
            else:
                self.assertFalse(torch.allclose(skipped, normal, atol=1e-5))


if __name__ == '__main__':
    unittest.main() 