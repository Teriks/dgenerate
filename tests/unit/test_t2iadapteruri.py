import unittest

import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.uris.t2iadapteruri as _t2iadapteruri
from dgenerate.pipelinewrapper.uris.exceptions import InvalidT2IAdapterUriError


class TestT2IAdapterUri(unittest.TestCase):

    def test_basic_parsing(self):
        # Test basic model path parsing
        uri = "path/to/model"
        result = _t2iadapteruri.T2IAdapterUri.parse(uri)
        self.assertEqual(result.model, "path/to/model")
        self.assertEqual(result.revision, None)
        self.assertEqual(result.variant, None)
        self.assertEqual(result.subfolder, None)
        self.assertEqual(result.dtype, None)
        self.assertEqual(result.scale, 1.0)  # Default scale

    def test_full_options_parsing(self):
        # Test parsing with all options
        uri = "path/to/model;revision=v1.0;variant=fp16;subfolder=models;dtype=float16;scale=0.75"
        
        result = _t2iadapteruri.T2IAdapterUri.parse(uri)
        self.assertEqual(result.model, "path/to/model")
        self.assertEqual(result.revision, "v1.0")
        self.assertEqual(result.variant, "fp16")
        self.assertEqual(result.subfolder, "models")
        self.assertEqual(result.dtype, _enums.DataType.FLOAT16)
        self.assertEqual(result.scale, 0.75)

    def test_dtype_validation(self):
        # Test dtype validation
        
        # Valid dtypes
        uri = "model;dtype=float16"
        result = _t2iadapteruri.T2IAdapterUri.parse(uri)
        self.assertEqual(result.dtype, _enums.DataType.FLOAT16)
        
        uri = "model;dtype=float32"
        result = _t2iadapteruri.T2IAdapterUri.parse(uri)
        self.assertEqual(result.dtype, _enums.DataType.FLOAT32)
        
        # Invalid dtype
        with self.assertRaises(InvalidT2IAdapterUriError) as context:
            _t2iadapteruri.T2IAdapterUri.parse("model;dtype=invalid_dtype")
        
        # Check that the error message is descriptive
        self.assertIn("must be", str(context.exception))
        
    def test_scale_validation(self):
        # Test scale validation
        
        # Valid scales
        uri = "model;scale=0.5"
        result = _t2iadapteruri.T2IAdapterUri.parse(uri)
        self.assertEqual(result.scale, 0.5)
        
        uri = "model;scale=1.25"
        result = _t2iadapteruri.T2IAdapterUri.parse(uri)
        self.assertEqual(result.scale, 1.25)
        
        # Default scale
        uri = "model"
        result = _t2iadapteruri.T2IAdapterUri.parse(uri)
        self.assertEqual(result.scale, 1.0)
        
        # Invalid scale format
        with self.assertRaises(InvalidT2IAdapterUriError) as context:
            _t2iadapteruri.T2IAdapterUri.parse("model;scale=invalid")
        
        # Check that the error message is descriptive
        self.assertIn("must be a floating point number", str(context.exception))


if __name__ == '__main__':
    unittest.main() 