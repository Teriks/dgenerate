import unittest

import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.uris.transformeruri as _transformeruri
from dgenerate.pipelinewrapper.uris.exceptions import InvalidTransformerUriError


class TestTransformerUri(unittest.TestCase):

    def test_basic_parsing(self):
        # Test basic model path parsing
        uri = "path/to/model"
        result = _transformeruri.TransformerUri.parse(uri)
        self.assertEqual(result.model, "path/to/model")
        self.assertEqual(result.revision, None)
        self.assertEqual(result.variant, None)
        self.assertEqual(result.subfolder, None)
        self.assertEqual(result.dtype, None)
        self.assertEqual(result.quantizer, False)  # Default is False, not None

    def test_full_options_parsing(self):
        # Test parsing with all options
        uri = "path/to/model;revision=v1.0;variant=fp16;subfolder=models;dtype=float16;quantizer=bnb"
        
        result = _transformeruri.TransformerUri.parse(uri)
        self.assertEqual(result.model, "path/to/model")
        self.assertEqual(result.revision, "v1.0")
        self.assertEqual(result.variant, "fp16")
        self.assertEqual(result.subfolder, "models")
        self.assertEqual(result.dtype, _enums.DataType.FLOAT16)
        self.assertEqual(result.quantizer, "bnb")

    def test_dtype_validation(self):
        # Test dtype validation
        
        # Valid dtypes
        uri = "model;dtype=float16"
        result = _transformeruri.TransformerUri.parse(uri)
        self.assertEqual(result.dtype, _enums.DataType.FLOAT16)
        
        uri = "model;dtype=float32"
        result = _transformeruri.TransformerUri.parse(uri)
        self.assertEqual(result.dtype, _enums.DataType.FLOAT32)
        
        # Invalid dtype
        with self.assertRaises(InvalidTransformerUriError) as context:
            _transformeruri.TransformerUri.parse("model;dtype=invalid_dtype")
        
        # Check that the error message is descriptive
        self.assertIn("must be", str(context.exception))
        
    def test_variant_parsing(self):
        # Test variant parsing
        
        # With variant
        uri = "model;variant=fp16"
        result = _transformeruri.TransformerUri.parse(uri)
        self.assertEqual(result.variant, "fp16")
        
        # With null variant
        uri = "model;variant=null"
        result = _transformeruri.TransformerUri.parse(uri)
        self.assertEqual(result.variant, "null")

    def test_quantizer_parsing(self):
        # Test quantizer parsing
        
        # With quantizer
        uri = "model;quantizer=bnb"
        result = _transformeruri.TransformerUri.parse(uri)
        self.assertEqual(result.quantizer, "bnb")
        
        # Without quantizer (default is False)
        uri = "model"
        result = _transformeruri.TransformerUri.parse(uri)
        self.assertEqual(result.quantizer, False)


if __name__ == '__main__':
    unittest.main() 