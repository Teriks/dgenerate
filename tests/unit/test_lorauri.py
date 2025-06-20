import unittest

import dgenerate.pipelinewrapper.constants as _pipelinewrapper_constants
import dgenerate.pipelinewrapper.uris.lorauri as _lorauri
from dgenerate.pipelinewrapper.uris.exceptions import InvalidLoRAUriError


class TestLoraUri(unittest.TestCase):

    def test_basic_parsing(self):
        # Test basic model path parsing
        uri = "path/to/model"
        result = _lorauri.LoRAUri.parse(uri)
        self.assertEqual(result.model, "path/to/model")
        self.assertEqual(result.revision, None)
        self.assertEqual(result.subfolder, None)
        self.assertEqual(result.weight_name, None)
        self.assertEqual(result.scale, 1.0)

    def test_full_options_parsing(self):
        # Test parsing with all options
        uri = "path/to/model;revision=v1.0;subfolder=models;weight-name=weights.pt;scale=0.8"
        
        result = _lorauri.LoRAUri.parse(uri)
        self.assertEqual(result.model, "path/to/model")
        self.assertEqual(result.revision, "v1.0")
        self.assertEqual(result.subfolder, "models")
        self.assertEqual(result.weight_name, "weights.pt")
        self.assertEqual(result.scale, 0.8)

    def test_scale_validation(self):
        # Test scale validation
        
        # Valid scale
        uri = "model;scale=0.75"
        result = _lorauri.LoRAUri.parse(uri)
        self.assertEqual(result.scale, 0.75)
        
        # Invalid scale format
        with self.assertRaises(InvalidLoRAUriError) as context:
            _lorauri.LoRAUri.parse("model;scale=invalid")
        
        # Check that the error message is descriptive
        self.assertIn("must be a floating point number", str(context.exception))


if __name__ == '__main__':
    unittest.main() 