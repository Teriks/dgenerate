import unittest
import unittest.mock as mock

import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.uris.imageencoderuri as _imageencoderuri
import dgenerate.hfhub as _hfhub
from dgenerate.pipelinewrapper.uris.exceptions import InvalidImageEncoderUriError


class TestImageEncoderUri(unittest.TestCase):

    def test_basic_parsing(self):
        # Test basic model path parsing
        uri = "path/to/model"
        
        # Mock is_single_file_model_load to return False to avoid the validation error
        with mock.patch.object(_hfhub, 'is_single_file_model_load', return_value=False):
            result = _imageencoderuri.ImageEncoderUri.parse(uri)
            self.assertEqual(result.model, "path/to/model")
            self.assertEqual(result.revision, None)
            self.assertEqual(result.variant, None)
            self.assertEqual(result.subfolder, None)
            self.assertEqual(result.dtype, None)

    def test_full_options_parsing(self):
        # Test parsing with all options
        uri = "path/to/model;revision=v1.0;variant=fp16;subfolder=models;dtype=float16"
        
        # Mock is_single_file_model_load to return False to avoid the validation error
        with mock.patch.object(_hfhub, 'is_single_file_model_load', return_value=False):
            result = _imageencoderuri.ImageEncoderUri.parse(uri)
            self.assertEqual(result.model, "path/to/model")
            self.assertEqual(result.revision, "v1.0")
            self.assertEqual(result.variant, "fp16")
            self.assertEqual(result.subfolder, "models")
            self.assertEqual(result.dtype, _enums.DataType.FLOAT16)

    def test_dtype_validation(self):
        # Test dtype validation
        
        # Valid dtypes
        uri = "model;dtype=float16"
        with mock.patch.object(_hfhub, 'is_single_file_model_load', return_value=False):
            result = _imageencoderuri.ImageEncoderUri.parse(uri)
            self.assertEqual(result.dtype, _enums.DataType.FLOAT16)
        
        uri = "model;dtype=float32"
        with mock.patch.object(_hfhub, 'is_single_file_model_load', return_value=False):
            result = _imageencoderuri.ImageEncoderUri.parse(uri)
            self.assertEqual(result.dtype, _enums.DataType.FLOAT32)
        
        # Invalid dtype
        with mock.patch.object(_hfhub, 'is_single_file_model_load', return_value=False):
            with self.assertRaises(InvalidImageEncoderUriError) as context:
                _imageencoderuri.ImageEncoderUri.parse("model;dtype=invalid_dtype")
            
            # Check that the error message is descriptive
            self.assertIn("must be", str(context.exception))

    def test_single_file_validation(self):
        # Test validation for single file models
        uri = "path/to/single/file/model"
        
        # Mock is_single_file_model_load to return True to trigger the validation error
        with mock.patch.object(_hfhub, 'is_single_file_model_load', return_value=True):
            with self.assertRaises(InvalidImageEncoderUriError) as context:
                _imageencoderuri.ImageEncoderUri.parse(uri)
            
            # Check that the error message is descriptive
            self.assertIn("single file is not supported", str(context.exception))

    def test_variant_parsing(self):
        # Test variant parsing
        
        # With variant
        uri = "model;variant=fp16"
        with mock.patch.object(_hfhub, 'is_single_file_model_load', return_value=False):
            result = _imageencoderuri.ImageEncoderUri.parse(uri)
            self.assertEqual(result.variant, "fp16")
        
        # With null variant
        uri = "model;variant=null"
        with mock.patch.object(_hfhub, 'is_single_file_model_load', return_value=False):
            result = _imageencoderuri.ImageEncoderUri.parse(uri)
            self.assertEqual(result.variant, "null")

    def test_string_representation(self):
        # Test string representation
        uri = "path/to/model;variant=fp16;revision=v1.0"
        with mock.patch.object(_hfhub, 'is_single_file_model_load', return_value=False):
            result = _imageencoderuri.ImageEncoderUri.parse(uri)
            string_repr = str(result)
            
            # Just verify that string conversion works and returns a string
            self.assertTrue(isinstance(string_repr, str))


if __name__ == '__main__':
    unittest.main() 