import unittest

import dgenerate.pipelinewrapper.uris.textualinversionuri as _textualinversionuri
from dgenerate.pipelinewrapper.uris.exceptions import InvalidTextualInversionUriError


class TestTextualInversionUri(unittest.TestCase):

    def test_basic_parsing(self):
        # Test basic model path parsing
        uri = "path/to/model"
        result = _textualinversionuri.TextualInversionUri.parse(uri)
        self.assertEqual(result.model, "path/to/model")
        self.assertEqual(result.token, None)
        self.assertEqual(result.revision, None)
        self.assertEqual(result.subfolder, None)
        self.assertEqual(result.weight_name, None)

    def test_full_options_parsing(self):
        # Test parsing with all options
        uri = "path/to/model;token=my_token;revision=v1.0;subfolder=models;weight-name=weights.pt"
        
        result = _textualinversionuri.TextualInversionUri.parse(uri)
        self.assertEqual(result.model, "path/to/model")
        self.assertEqual(result.token, "my_token")
        self.assertEqual(result.revision, "v1.0")
        self.assertEqual(result.subfolder, "models")
        self.assertEqual(result.weight_name, "weights.pt")

    def test_token_parsing(self):
        # Test token parsing
        
        # With token
        uri = "model;token=cat"
        result = _textualinversionuri.TextualInversionUri.parse(uri)
        self.assertEqual(result.token, "cat")
        
        # With token containing spaces
        uri = "model;token=fluffy cat"
        result = _textualinversionuri.TextualInversionUri.parse(uri)
        self.assertEqual(result.token, "fluffy cat")
        
        # With quoted token
        uri = "model;token=\"fluffy cat\""
        result = _textualinversionuri.TextualInversionUri.parse(uri)
        self.assertEqual(result.token, "fluffy cat")

    def test_string_representation(self):
        # Test string representation
        uri = "path/to/model;token=cat;revision=v1.0"
        result = _textualinversionuri.TextualInversionUri.parse(uri)
        string_repr = str(result)
        
        # Check that the string representation contains all the attributes
        self.assertIn("'model': 'path/to/model'", string_repr)
        self.assertIn("'token': 'cat'", string_repr)
        self.assertIn("'revision': 'v1.0'", string_repr)


if __name__ == '__main__':
    unittest.main() 