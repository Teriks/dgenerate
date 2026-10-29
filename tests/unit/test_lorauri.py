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

    def test_quantized_weights_are_not_fused(self):
        class Pipe:
            def __init__(self):
                transformer = torch.nn.Linear(2, 2)
                transformer.quantization_config = {'quant_method': 'gguf'}
                self.components = {'transformer': transformer}
                self.adapter_weights = None
                self.fused = None

            def load_lora_weights(self, *args, **kwargs):
                pass

            def get_list_adapters(self):
                return {'transformer': ['0']}

            def set_adapters(self, names, adapter_weights=None, **kwargs):
                self.adapter_weights = list(adapter_weights)

            def fuse_lora(self, *args, **kwargs):
                self.fused = True

        pipe = Pipe()
        with unittest.mock.patch.object(
                _messages, 'warning') as warning, \
                unittest.mock.patch.object(
                    _lorauri, '_load_one_lora', return_value=None), \
                unittest.mock.patch.object(
                    _lorauri._hfhub, 'download_non_hf_slug_model',
                    return_value='lora.safetensors'):
            _lorauri.LoRAUri.load_on_pipeline(
                pipe, ['org/lora;scale=0.5'], fuse_scale=2.0, fuse=True)

        warning.assert_called_once()
        message = warning.call_args.args[0]
        self.assertIn('transformer', message)
        self.assertIn('Quality may be affected', message)
        self.assertEqual(pipe.adapter_weights, [1.0])
        self.assertIsNone(pipe.fused)

    def test_quantized_text_encoder_does_not_block_unet_fuse(self):
        class Expert(torch.nn.Module):
            def __init__(self, quantized=False):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)
                if quantized:
                    self.quantization_config = {'quant_method': 'gguf'}
                self.adapters = None
                self.fused = None

            def set_adapters(self, names, weights):
                self.adapters = (list(names), list(weights))

            def fuse_lora(self, lora_scale=1.0, adapter_names=None):
                self.fused = (lora_scale, list(adapter_names))

        class Pipe:
            def __init__(self):
                self.transformer = Expert()
                self.text_encoder = Expert(quantized=True)
                self.components = {
                    'transformer': self.transformer,
                    'text_encoder': self.text_encoder,
                }

            def load_lora_weights(self, *args, **kwargs):
                pass

            def get_list_adapters(self):
                return {'transformer': ['0'], 'text_encoder': ['0']}

            def set_adapters(self, *args, **kwargs):
                raise AssertionError('mixed components are activated separately')

            def fuse_lora(self, *args, **kwargs):
                raise AssertionError('mixed components are fused separately')

        pipeline = Pipe()
        with unittest.mock.patch.object(
                _lorauri, '_load_one_lora', return_value=None), \
                unittest.mock.patch.object(
                    _lorauri._hfhub, 'download_non_hf_slug_model',
                    return_value='lora.safetensors'):
            _lorauri.LoRAUri.load_on_pipeline(
                pipeline, ['org/lora;scale=0.5'], fuse_scale=2.0, fuse=True)

        self.assertEqual(pipeline.transformer.adapters, (['0'], [0.5]))
        self.assertEqual(pipeline.transformer.fused, (2.0, ['0']))
        self.assertEqual(pipeline.text_encoder.adapters, (['0'], [1.0]))
        self.assertIsNone(pipeline.text_encoder.fused)

    def test_comfy_quant_weights_are_not_fused(self):
        import dgenerate.pipelinewrapper.safetensorscompat as _safetensorscompat

        weight = torch.zeros(4, 8, dtype=torch.int8)
        scale = _safetensorscompat._float_bits(torch.ones(4))
        layer = _safetensorscompat._QuantLayer('linear', 'int8_tensorwise', weight, scale)
        packed = _safetensorscompat.ComfyQuantLinear(8, 4, layer)

        class Transformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = packed

        class Pipe:
            def __init__(self):
                self.transformer = Transformer()
                self.components = {'transformer': self.transformer}
                self.adapter_weights = None
                self.fused = None

            def load_lora_weights(self, *args, **kwargs):
                pass

            def get_list_adapters(self):
                return {'transformer': ['0']}

            def set_adapters(self, names, adapter_weights=None, **kwargs):
                self.adapter_weights = list(adapter_weights)

            def fuse_lora(self, *args, **kwargs):
                self.fused = True

        pipe = Pipe()
        with unittest.mock.patch.object(
                _messages, 'warning') as warning, \
                unittest.mock.patch.object(
                    _lorauri, '_load_one_lora', return_value=None), \
                unittest.mock.patch.object(
                    _lorauri._hfhub, 'download_non_hf_slug_model',
                    return_value='lora.safetensors'):
            _lorauri.LoRAUri.load_on_pipeline(
                pipe, ['org/lora;scale=0.5'], fuse_scale=2.0, fuse=True)

        warning.assert_called_once()
        message = warning.call_args.args[0]
        self.assertIn('transformer', message)
        self.assertIn('Quality may be affected', message)
        self.assertEqual(pipe.adapter_weights, [1.0])
        self.assertIsNone(pipe.fused)

    def test_unfused_lora_runs_on_comfy_quant_linear(self):
        import dgenerate.pipelinewrapper.safetensorscompat as _safetensorscompat
        from peft import LoraConfig
        from peft.tuners.lora.layer import Linear as LoraLinear

        torch.manual_seed(0)
        dense = torch.randn(4, 8) * 0.02
        scale = dense.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127
        packed_weight = torch.round(dense / scale).clamp(-128, 127).to(torch.int8)
        layer = _safetensorscompat._QuantLayer(
            'linear', 'int8_tensorwise', packed_weight,
            _safetensorscompat._float_bits(scale.reshape(4)))
        module = _safetensorscompat.ComfyQuantLinear(8, 4, layer)
        module.load_state_dict(
            {'weight': layer.weight, 'weight_scale': layer.scale}, assign=True)
        config = LoraConfig(r=2, lora_alpha=2, target_modules=['linear'])
        wrapped = LoraLinear(module, 'default', config, r=2, lora_alpha=2)
        with torch.no_grad():
            wrapped.lora_A['default'].weight.normal_()
            wrapped.lora_B['default'].weight.normal_()
        x = torch.randn(2, 8)
        with torch.no_grad():
            got = wrapped(x)
            bare = module(x)
        self.assertEqual(got.shape, bare.shape)
        self.assertEqual(got.dtype, x.dtype)
        self.assertFalse(torch.allclose(got, bare))
        with self.assertRaises(RuntimeError):
            wrapped.merge()


if __name__ == '__main__':
    unittest.main() 