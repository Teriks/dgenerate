import unittest

import dgenerate.pipelinewrapper.uris.bnbquantizeruri as _bnbquantizeruri
from dgenerate.pipelinewrapper.uris.exceptions import InvalidBNBQuantizerUriError


class TestBNBQuantizerUri(unittest.TestCase):

    def test_basic_parsing(self):
        # Test basic parsing with default values
        uri = "bnb"
        result = _bnbquantizeruri.BNBQuantizerUri.parse(uri)
        self.assertEqual(result.bits, 8)  # Default bits
        self.assertEqual(result.bits4_compute_dtype, None)
        self.assertEqual(result.bits4_quant_type, "fp4")
        self.assertEqual(result.bits4_use_double_quant, False)
        self.assertEqual(result.bits4_quant_storage, None)

    def test_full_options_parsing(self):
        # Test parsing with all options
        uri = "bnb;bits=4;bits4-compute-dtype=float16;bits4-quant-type=nf4;bits4-use-double-quant=true;bits4-quant-storage=float32"
        
        result = _bnbquantizeruri.BNBQuantizerUri.parse(uri)
        self.assertEqual(result.bits, 4)
        self.assertEqual(result.bits4_compute_dtype, "float16")
        self.assertEqual(result.bits4_quant_type, "nf4")
        self.assertEqual(result.bits4_use_double_quant, True)
        self.assertEqual(result.bits4_quant_storage, "float32")

    def test_bits_validation(self):
        # Test bits validation
        
        # Valid bits
        uri = "bnb;bits=4"
        result = _bnbquantizeruri.BNBQuantizerUri.parse(uri)
        self.assertEqual(result.bits, 4)
        
        uri = "bnb;bits=8"
        result = _bnbquantizeruri.BNBQuantizerUri.parse(uri)
        self.assertEqual(result.bits, 8)
        
        # Invalid bits
        with self.assertRaises(InvalidBNBQuantizerUriError) as context:
            _bnbquantizeruri.BNBQuantizerUri.parse("bnb;bits=16")
        
        # Check that the error message is descriptive
        self.assertIn("must be 4 or 8", str(context.exception))

    def test_quant_type_validation(self):
        # Test quant_type validation
        
        # Valid quant_types
        uri = "bnb;bits4-quant-type=fp4"
        result = _bnbquantizeruri.BNBQuantizerUri.parse(uri)
        self.assertEqual(result.bits4_quant_type, "fp4")
        
        uri = "bnb;bits4-quant-type=nf4"
        result = _bnbquantizeruri.BNBQuantizerUri.parse(uri)
        self.assertEqual(result.bits4_quant_type, "nf4")
        
        # Invalid quant_type
        with self.assertRaises(InvalidBNBQuantizerUriError) as context:
            _bnbquantizeruri.BNBQuantizerUri.parse("bnb;bits4-quant-type=invalid")
        
        # Check that the error message is descriptive
        self.assertIn("must be fp4 or nf4", str(context.exception))

    def test_compute_dtype_validation(self):
        # Test compute_dtype validation
        
        # Valid compute_dtypes
        uri = "bnb;bits4-compute-dtype=float16"
        result = _bnbquantizeruri.BNBQuantizerUri.parse(uri)
        self.assertEqual(result.bits4_compute_dtype, "float16")
        
        uri = "bnb;bits4-compute-dtype=float32"
        result = _bnbquantizeruri.BNBQuantizerUri.parse(uri)
        self.assertEqual(result.bits4_compute_dtype, "float32")
        
        uri = "bnb;bits4-compute-dtype=bfloat16"
        result = _bnbquantizeruri.BNBQuantizerUri.parse(uri)
        self.assertEqual(result.bits4_compute_dtype, "bfloat16")
        
        # Invalid compute_dtype
        with self.assertRaises(InvalidBNBQuantizerUriError) as context:
            _bnbquantizeruri.BNBQuantizerUri.parse("bnb;bits4-compute-dtype=invalid")
        
        # Check that the error message is descriptive
        self.assertIn("must be one of", str(context.exception))

    def test_quant_storage_validation(self):
        # Test quant_storage validation
        
        # Valid quant_storage values
        uri = "bnb;bits4-quant-storage=float16"
        result = _bnbquantizeruri.BNBQuantizerUri.parse(uri)
        self.assertEqual(result.bits4_quant_storage, "float16")
        
        uri = "bnb;bits4-quant-storage=int8"
        result = _bnbquantizeruri.BNBQuantizerUri.parse(uri)
        self.assertEqual(result.bits4_quant_storage, "int8")
        
        # Invalid quant_storage
        with self.assertRaises(InvalidBNBQuantizerUriError) as context:
            _bnbquantizeruri.BNBQuantizerUri.parse("bnb;bits4-quant-storage=invalid")
        
        # Check that the error message is descriptive
        self.assertIn("must be one of", str(context.exception))

    def test_use_double_quant_parsing(self):
        # Test use_double_quant parsing
        
        # True values
        uri = "bnb;bits4-use-double-quant=true"
        result = _bnbquantizeruri.BNBQuantizerUri.parse(uri)
        self.assertEqual(result.bits4_use_double_quant, True)
        
        uri = "bnb;bits4-use-double-quant=True"
        result = _bnbquantizeruri.BNBQuantizerUri.parse(uri)
        self.assertEqual(result.bits4_use_double_quant, True)
        
        # False values
        uri = "bnb;bits4-use-double-quant=false"
        result = _bnbquantizeruri.BNBQuantizerUri.parse(uri)
        self.assertEqual(result.bits4_use_double_quant, False)
        
        uri = "bnb;bits4-use-double-quant=False"
        result = _bnbquantizeruri.BNBQuantizerUri.parse(uri)
        self.assertEqual(result.bits4_use_double_quant, False)

    def test_alternative_concept_name(self):
        # Test alternative concept name (bitsandbytes)
        uri = "bitsandbytes;bits=4"
        result = _bnbquantizeruri.BNBQuantizerUri.parse(uri)
        self.assertEqual(result.bits, 4)
        
        # Invalid concept name
        with self.assertRaises(InvalidBNBQuantizerUriError) as context:
            _bnbquantizeruri.BNBQuantizerUri.parse("invalid;bits=4")
        
        # Check that the error message is descriptive
        self.assertIn("Unknown quantization backend", str(context.exception))

    def test_to_config(self):
        # Test to_config method
        uri = "bnb;bits=4;bits4-compute-dtype=float16"
        result = _bnbquantizeruri.BNBQuantizerUri.parse(uri)
        config = result.to_config()
        
        # Check that the config has the correct values
        self.assertEqual(config.load_in_4bit, True)
        self.assertEqual(config.load_in_8bit, False)
        self.assertEqual(str(config.bnb_4bit_compute_dtype), "torch.float16")  # Compare string representation
        self.assertEqual(config.bnb_4bit_quant_type, "fp4")
        self.assertEqual(config.bnb_4bit_use_double_quant, False)


if __name__ == '__main__':
    unittest.main() 