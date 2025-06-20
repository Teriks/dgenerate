import unittest

import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.uris.sdxlrefineruri as _sdxlrefineruri
from dgenerate.pipelinewrapper.uris.exceptions import InvalidSDXLRefinerUriError


class TestSDXLRefinerUri(unittest.TestCase):

    def test_basic_parsing(self):
        # Test basic model path parsing
        uri = "path/to/model"
        result = _sdxlrefineruri.SDXLRefinerUri.parse(uri)
        self.assertEqual(result.model, "path/to/model")
        self.assertEqual(result.revision, None)
        self.assertEqual(result.variant, None)
        self.assertEqual(result.subfolder, None)
        self.assertEqual(result.dtype, None)

    def test_full_options_parsing(self):
        # Test parsing with all options
        uri = "path/to/model;revision=v1.0;variant=fp16;subfolder=models;dtype=float16"
        
        result = _sdxlrefineruri.SDXLRefinerUri.parse(uri)
        self.assertEqual(result.model, "path/to/model")
        self.assertEqual(result.revision, "v1.0")
        self.assertEqual(result.variant, "fp16")
        self.assertEqual(result.subfolder, "models")
        self.assertEqual(result.dtype, _enums.DataType.FLOAT16)

    def test_dtype_validation(self):
        # Test dtype validation
        
        # Valid dtypes
        uri = "model;dtype=float16"
        result = _sdxlrefineruri.SDXLRefinerUri.parse(uri)
        self.assertEqual(result.dtype, _enums.DataType.FLOAT16)
        
        uri = "model;dtype=float32"
        result = _sdxlrefineruri.SDXLRefinerUri.parse(uri)
        self.assertEqual(result.dtype, _enums.DataType.FLOAT32)
        
        # Invalid dtype
        with self.assertRaises(InvalidSDXLRefinerUriError) as context:
            _sdxlrefineruri.SDXLRefinerUri.parse("model;dtype=invalid_dtype")
        
        # Check that the error message is descriptive
        self.assertIn("must be", str(context.exception))
        
    def test_variant_parsing(self):
        # Test variant parsing
        
        # With variant
        uri = "model;variant=fp16"
        result = _sdxlrefineruri.SDXLRefinerUri.parse(uri)
        self.assertEqual(result.variant, "fp16")
        
        # With null variant
        uri = "model;variant=null"
        result = _sdxlrefineruri.SDXLRefinerUri.parse(uri)
        self.assertEqual(result.variant, "null")

    def test_string_representation(self):
        # Test string representation
        uri = "path/to/model;variant=fp16;revision=v1.0"
        result = _sdxlrefineruri.SDXLRefinerUri.parse(uri)
        string_repr = str(result)
        
        # Just verify that string conversion works and returns a string
        self.assertTrue(isinstance(string_repr, str))


if __name__ == '__main__':
    unittest.main() 