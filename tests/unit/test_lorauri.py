import unittest
import unittest.mock

import torch

import dgenerate.messages as _messages
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

    def test_pipeline_has_quantized_modules(self):
        class Pipe:
            def __init__(self, transformer):
                self.components = {'transformer': transformer}

        plain = torch.nn.Linear(2, 2)
        self.assertFalse(_lorauri._pipeline_has_quantized_modules(Pipe(plain)))

        quantized = torch.nn.Linear(2, 2)
        quantized.quantization_config = {'quant_method': 'sdnq'}
        self.assertTrue(_lorauri._pipeline_has_quantized_modules(Pipe(quantized)))

    def test_fuse_warns_when_pipeline_quantized(self):
        warnings = []

        class Pipe:
            def __init__(self):
                transformer = torch.nn.Linear(2, 2)
                transformer.quantization_config = {'quant_method': 'gguf'}
                self.components = {'transformer': transformer}

            def load_lora_weights(self, *args, **kwargs):
                pass

            def get_list_adapters(self):
                return {'transformer': ['0']}

            def set_adapters(self, *args, **kwargs):
                pass

            def fuse_lora(self, *args, **kwargs):
                pass

        with unittest.mock.patch.object(
                _messages, 'warning', side_effect=lambda msg: warnings.append(msg)), \
                unittest.mock.patch.object(
                    _lorauri, '_load_one_lora', return_value=None), \
                unittest.mock.patch.object(
                    _lorauri._hfhub, 'download_non_hf_slug_model',
                    return_value='lora.safetensors'):
            _lorauri.LoRAUri.load_on_pipeline(
                Pipe(), ['org/lora'], fuse=True)

        self.assertTrue(any('quantized or GGUF' in msg for msg in warnings))


if __name__ == '__main__':
    unittest.main() 