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
        self.assertIsNone(result.component)

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

    def test_component_selects_the_wan_low_noise_expert(self):
        parsed = _lorauri.LoRAUri.parse('org/low;component=transformer_2;scale=1')
        self.assertEqual(parsed.component, 'transformer_2')
        with self.assertRaises(InvalidLoRAUriError):
            _lorauri.LoRAUri.parse('org/low;component=vae')

        class Expert:
            def __init__(self):
                self.adapters = None
                self.fused = None

            def set_adapters(self, names, weights):
                self.adapters = (list(names), list(weights))

            def fuse_lora(self, lora_scale=1.0, adapter_names=None):
                self.fused = (lora_scale, list(adapter_names))

        class Pipe:
            def __init__(self):
                self.transformer = Expert()
                self.transformer_2 = Expert()
                self.components = {
                    'transformer': self.transformer,
                    'transformer_2': self.transformer_2,
                }
                self.loaded = []

            def load_lora_weights(self, model_path, adapter_name=None, **kwargs):
                self.loaded.append((adapter_name, kwargs.get('load_into_transformer_2', False)))

            def get_list_adapters(self):
                listed = {}
                for name, into_second in self.loaded:
                    key = 'transformer_2' if into_second else 'transformer'
                    listed.setdefault(key, []).append(name)
                return listed

            def set_adapters(self, *args, **kwargs):
                raise AssertionError('split LoRAs must be activated per component')

            def fuse_lora(self, *args, **kwargs):
                raise AssertionError('split LoRAs must be fused per component')

        pipeline = Pipe()
        with unittest.mock.patch.object(
                _lorauri._hfhub, 'download_non_hf_slug_model',
                side_effect=lambda model, **kwargs: model):
            _lorauri.LoRAUri.load_on_pipeline(pipeline, [
                'org/high;scale=1.0',
                'org/low;scale=0.5;component=transformer_2',
            ], fuse_scale=1.0, fuse=True)

        self.assertEqual(pipeline.loaded, [('0', False), ('1', True)])
        self.assertEqual(pipeline.transformer.adapters, (['0'], [1.0]))
        self.assertEqual(pipeline.transformer.fused, (1.0, ['0']))
        self.assertEqual(pipeline.transformer_2.adapters, (['1'], [0.5]))
        self.assertEqual(pipeline.transformer_2.fused, (1.0, ['1']))

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