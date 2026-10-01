import unittest
import unittest.mock as mock

import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.uris.textencoderuri as _textencoderuri
import dgenerate.hfhub as _hfhub
from dgenerate.pipelinewrapper.uris.exceptions import InvalidTextEncoderUriError


class TestTextEncoderUri(unittest.TestCase):

    def test_basic_parsing(self):
        # Test basic model path parsing
        uri = "CLIPTextModel;model=path/to/model"
        result = _textencoderuri.TextEncoderUri.parse(uri)
        self.assertEqual(result.encoder, "CLIPTextModel")
        self.assertEqual(result.model, "path/to/model")
        self.assertEqual(result.revision, None)
        self.assertEqual(result.variant, None)
        self.assertEqual(result.subfolder, None)
        self.assertEqual(result.dtype, None)
        self.assertEqual(result.quantizer, False)  # Default is False, not None
        self.assertEqual(result.mode, None)

    def test_full_options_parsing(self):
        # Test parsing with all options
        uri = "CLIPTextModel;model=path/to/model;revision=v1.0;variant=fp16;subfolder=models;dtype=float16;quantizer=bnb"
        
        result = _textencoderuri.TextEncoderUri.parse(uri)
        self.assertEqual(result.encoder, "CLIPTextModel")
        self.assertEqual(result.model, "path/to/model")
        self.assertEqual(result.revision, "v1.0")
        self.assertEqual(result.variant, "fp16")
        self.assertEqual(result.subfolder, "models")
        self.assertEqual(result.dtype, _enums.DataType.FLOAT16)
        self.assertEqual(result.quantizer, "bnb")
        self.assertEqual(result.mode, None)

    def test_encoder_validation(self):
        # Test encoder validation
        
        # Valid encoders
        for encoder in _textencoderuri.TextEncoderUri.supported_encoder_names():
            uri = f"{encoder};model=path/to/model"
            result = _textencoderuri.TextEncoderUri.parse(uri)
            self.assertEqual(result.encoder, encoder)
        
        # Invalid encoder
        with self.assertRaises(InvalidTextEncoderUriError) as context:
            _textencoderuri.TextEncoderUri.parse("InvalidEncoder;model=path/to/model")
        
        # Check that the error message is descriptive
        self.assertIn("Unknown TextEncoder encoder class", str(context.exception))

    def test_dtype_validation(self):
        # Test dtype validation
        
        # Valid dtypes
        uri = "CLIPTextModel;model=path/to/model;dtype=float16"
        result = _textencoderuri.TextEncoderUri.parse(uri)
        self.assertEqual(result.dtype, _enums.DataType.FLOAT16)
        
        uri = "CLIPTextModel;model=path/to/model;dtype=float32"
        result = _textencoderuri.TextEncoderUri.parse(uri)
        self.assertEqual(result.dtype, _enums.DataType.FLOAT32)
        
        # Invalid dtype
        with self.assertRaises(InvalidTextEncoderUriError) as context:
            _textencoderuri.TextEncoderUri.parse("CLIPTextModel;model=path/to/model;dtype=invalid_dtype")
        
        # Check that the error message is descriptive
        self.assertIn("must be", str(context.exception))

    def test_mode_validation(self):
        # Test mode validation
        
        # Valid modes
        for mode in _textencoderuri.TextEncoderUri._valid_modes():
            uri = f"CLIPTextModel;model=path/to/model;mode={mode}"
            result = _textencoderuri.TextEncoderUri.parse(uri)
            self.assertEqual(result.mode, mode)
        
        # Invalid mode
        with self.assertRaises(InvalidTextEncoderUriError) as context:
            _textencoderuri.TextEncoderUri.parse("CLIPTextModel;model=path/to/model;mode=invalid_mode")
        
        # Check that the error message is descriptive
        self.assertIn("Unknown TextEncoder load mode", str(context.exception))

    def test_mode_incompatible_options(self):
        # Test that mode is incompatible with variant, revision, and subfolder
        mode = _textencoderuri.TextEncoderUri._valid_modes()[0]  # Get a valid mode
        
        # Mode + variant
        with self.assertRaises(InvalidTextEncoderUriError) as context:
            _textencoderuri.TextEncoderUri.parse(f"CLIPTextModel;model=path/to/model;mode={mode};variant=fp16")
        self.assertIn("cannot use variant with mode", str(context.exception))
        
        # Mode + revision
        with self.assertRaises(InvalidTextEncoderUriError) as context:
            _textencoderuri.TextEncoderUri.parse(f"CLIPTextModel;model=path/to/model;mode={mode};revision=v1.0")
        self.assertIn("cannot use revision with mode", str(context.exception))
        
        # Mode + subfolder
        with self.assertRaises(InvalidTextEncoderUriError) as context:
            _textencoderuri.TextEncoderUri.parse(f"CLIPTextModel;model=path/to/model;mode={mode};subfolder=models")
        self.assertIn("cannot use subfolder with mode", str(context.exception))

    def test_single_file_with_quantizer_validation(self):
        # Test that single file loads with quantizer are not supported unless a valid mode is specified
        
        # Mock is_single_file_model_load to return True
        with mock.patch.object(_hfhub, 'is_single_file_model_load', return_value=True):
            # Valid: Single file with mode and quantizer
            mode = _textencoderuri.TextEncoderUri._valid_modes()[0]
            uri = f"CLIPTextModel;model=path/to/model;mode={mode};quantizer=bnb"
            result = _textencoderuri.TextEncoderUri.parse(uri)
            self.assertEqual(result.quantizer, "bnb")
            
            # Invalid: Single file with quantizer but no mode
            with self.assertRaises(InvalidTextEncoderUriError) as context:
                _textencoderuri.TextEncoderUri.parse("CLIPTextModel;model=path/to/model;quantizer=bnb")
            self.assertIn("single file loads are not supported", str(context.exception))

    def test_string_representation(self):
        # Test string representation
        uri = "CLIPTextModel;model=path/to/model;variant=fp16;revision=v1.0"
        result = _textencoderuri.TextEncoderUri.parse(uri)
        string_repr = str(result)
        
        # Just verify that string conversion works and returns a string
        self.assertTrue(isinstance(string_repr, str))

    def test_clip_skip_uses_final_layer_norm_on_flattened_clip(self):
        import torch
        from transformers import CLIPTextConfig, CLIPTextModel

        config = CLIPTextConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            max_position_embeddings=16,
            eos_token_id=2,
            bos_token_id=None,
        )
        text_encoder = CLIPTextModel(config).eval()
        self.assertIs(text_encoder.text_model, text_encoder)
        self.assertIs(text_encoder.text_model.embeddings, text_encoder.embeddings)
        self.assertIs(text_encoder.text_model.final_layer_norm, text_encoder.final_layer_norm)
        # Single-file checkpoints still use text_model.* key prefixes. Accelerate
        # walks getattr, so the prefix must land on the real parameter.
        module = text_encoder
        dotted = 'text_model.embeddings.token_embedding.weight'
        for part in dotted.split('.')[:-1]:
            module = getattr(module, part)
        self.assertIn(dotted.split('.')[-1], module._parameters)

        ids = torch.randint(0, config.vocab_size, (2, 8))
        ids[:, -1] = config.eos_token_id
        normal = text_encoder(ids)[0]
        hidden = text_encoder(ids, output_hidden_states=True)
        for skip in (0, 1):
            layer = hidden[-1][-(skip + 1)]
            skipped = text_encoder.text_model.final_layer_norm(layer)
            self.assertEqual(tuple(skipped.shape), tuple(normal.shape))
            if skip == 0:
                self.assertTrue(torch.allclose(skipped, normal, atol=1e-5))
            else:
                self.assertFalse(torch.allclose(skipped, normal, atol=1e-5))

    def test_single_file_clip_load_does_not_see_the_text_model_alias(self):
        from transformers import CLIPTextConfig, CLIPTextModel

        from dgenerate._patches.transformers_clip_text_model_patch import (
            _call_without_flattened_text_model_alias,
        )

        config = CLIPTextConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=8,
            eos_token_id=2,
            bos_token_id=None,
        )

        def seen_as_wrapper():
            model = CLIPTextModel(config)
            return hasattr(model, 'text_model')

        self.assertTrue(seen_as_wrapper())
        self.assertFalse(_call_without_flattened_text_model_alias(seen_as_wrapper))
        self.assertTrue(seen_as_wrapper())

    def test_flattened_clip_checkpoint_drops_text_model_prefix(self):
        import tempfile

        import torch
        from safetensors.torch import save_file
        from transformers import CLIPTextConfig, CLIPTextModel

        config = CLIPTextConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=8,
            eos_token_id=2,
            bos_token_id=None,
        )
        donor = CLIPTextModel(config).eval()
        with torch.no_grad():
            for parameter in donor.parameters():
                parameter.normal_()

        prefixed = {
            'text_model.' + key: value.detach().cpu().contiguous().clone()
            for key, value in donor.state_dict().items()
        }
        target = CLIPTextModel(config).eval()
        projection = target.encoder.layers[0].self_attn.q_proj
        projection.weight = torch.nn.Parameter(
            torch.empty_like(projection.weight, device='meta'),
            requires_grad=False,
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = f'{directory}/clip_l.safetensors'
            save_file(prefixed, checkpoint)
            loaded = _textencoderuri._load_monolithic_checkpoint(
                target,
                checkpoint=checkpoint,
                device_map={'': 'cpu'},
                dtype=torch.float32,
                no_split_module_classes=['CLIPEncoderLayer'],
            )

        weight = loaded.encoder.layers[0].self_attn.q_proj.weight
        self.assertFalse(weight.is_meta)
        self.assertTrue(torch.allclose(
            weight, donor.encoder.layers[0].self_attn.q_proj.weight))
        self.assertTrue(torch.allclose(
            loaded.embeddings.token_embedding.weight,
            donor.embeddings.token_embedding.weight,
        ))

    def test_flattened_clip_checkpoint_keeps_unprefixed_keys(self):
        loaded = {'embeddings.token_embedding.weight': 'exact',
                  'text_model.encoder.layers.0.mlp.fc1.weight': 'prefixed',
                  'unused.weight': 'unused'}
        model_keys = {
            'embeddings.token_embedding.weight',
            'encoder.layers.0.mlp.fc1.weight',
        }
        translated = _textencoderuri._translate_flattened_clip_checkpoint_keys(
            loaded, model_keys)
        self.assertEqual(translated['embeddings.token_embedding.weight'], 'exact')
        self.assertEqual(translated['encoder.layers.0.mlp.fc1.weight'], 'prefixed')
        self.assertEqual(translated['unused.weight'], 'unused')
        self.assertNotIn('text_model.encoder.layers.0.mlp.fc1.weight', translated)

    def test_missing_clip_text_projection_becomes_identity(self):
        import tempfile

        import torch
        from safetensors.torch import save_file
        from transformers import CLIPTextConfig, CLIPTextModelWithProjection

        config = CLIPTextConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=8,
            projection_dim=32,
            eos_token_id=2,
            bos_token_id=None,
        )
        donor = CLIPTextModelWithProjection(config).eval()
        state = {
            key: value.detach().cpu().contiguous().clone()
            for key, value in donor.state_dict().items()
            if key != 'text_projection.weight'
        }
        target = CLIPTextModelWithProjection(config).eval()
        projection = target.text_projection
        projection.weight = torch.nn.Parameter(
            torch.empty_like(projection.weight, device='meta'),
            requires_grad=False,
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = f'{directory}/clip_l.safetensors'
            save_file(state, checkpoint)
            loaded = _textencoderuri._load_monolithic_checkpoint(
                target,
                checkpoint=checkpoint,
                device_map={'': 'cpu'},
                dtype=torch.float32,
                no_split_module_classes=['CLIPEncoderLayer'],
            )

        weight = loaded.text_projection.weight
        self.assertFalse(weight.is_meta)
        self.assertTrue(torch.equal(weight, torch.eye(32)))

    def test_openclip_text_projection_is_transposed_onto_the_linear(self):
        import torch

        bare = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        loaded = _textencoderuri._supply_missing_clip_text_projection(
            {'text_projection': bare},
            {'text_projection.weight'},
            type('Projection', (), {'weight': torch.empty(3, 2)})(),
        )
        self.assertTrue(torch.equal(loaded['text_projection.weight'], bare.T))

    def test_quantized_t5_keeps_wo_in_fp32(self):
        import os
        import tempfile

        import torch
        from safetensors.torch import save_file
        from transformers import T5Config, T5EncoderModel

        if not torch.cuda.is_available():
            self.skipTest('bitsandbytes 4-bit needs CUDA')
        try:
            import bitsandbytes as bnb
            import diffusers
        except ImportError:
            self.skipTest('bitsandbytes is not installed')

        config = T5Config(
            vocab_size=128,
            d_model=32,
            d_kv=8,
            d_ff=64,
            num_layers=2,
            num_heads=4,
            feed_forward_proj='gated-gelu',
            is_gated_act=True,
            relative_attention_num_buckets=8,
        )
        donor = T5EncoderModel(config).eval()
        state = {
            key: value.detach().cpu().contiguous().clone()
            for key, value in donor.state_dict().items()
        }
        target = T5EncoderModel(config).to(dtype=torch.bfloat16)
        quant_config = diffusers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        device_map = _textencoderuri._monolithic_auto_quant_device_map(quant_config, 'cuda')
        quantizer = diffusers.quantizers.auto.DiffusersAutoQuantizer().from_config(quant_config)
        quantizer.preprocess_model(
            target,
            device_map=device_map,
            keep_in_fp32_modules=_textencoderuri._keep_in_fp32_module_names(target),
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, 't5.safetensors')
            save_file(state, checkpoint)
            loaded = _textencoderuri._load_monolithic_checkpoint(
                target,
                checkpoint=checkpoint,
                device_map=device_map,
                dtype=torch.bfloat16,
                no_split_module_classes=['T5Block'],
            )
        _textencoderuri._cast_kept_fp32_modules(loaded)
        quantizer.postprocess_model(loaded)

        wo = loaded.encoder.block[0].layer[1].DenseReluDense.wo
        q = loaded.encoder.block[0].layer[0].SelfAttention.q
        self.assertNotIsInstance(wo, bnb.nn.Linear4bit)
        self.assertIsInstance(q, bnb.nn.Linear4bit)
        self.assertEqual(wo.weight.dtype, torch.float32)
        # The checkpoint load casts to bf16, then this layer is restored to fp32.
        reference = donor.encoder.block[0].layer[1].DenseReluDense.wo.weight.detach().bfloat16().float().cpu()
        self.assertTrue(torch.allclose(
            wo.weight.detach().float().cpu(),
            reference,
            atol=1e-3,
            rtol=1e-2,
        ))

    def test_sdnq_t5_quantizes_linears_and_keeps_wo(self):
        import os
        import tempfile

        import torch
        from safetensors.torch import save_file
        from transformers import T5Config, T5EncoderModel

        try:
            from sdnq import SDNQConfig
            import diffusers
        except ImportError:
            self.skipTest('sdnq is not installed')

        config = T5Config(
            vocab_size=128,
            d_model=64,
            d_kv=16,
            d_ff=128,
            num_layers=1,
            num_heads=4,
            feed_forward_proj='gated-gelu',
            is_gated_act=True,
            relative_attention_num_buckets=8,
        )
        donor = T5EncoderModel(config).eval()
        state = {
            key: value.detach().cpu().contiguous().clone()
            for key, value in donor.state_dict().items()
        }
        target = T5EncoderModel(config).to(dtype=torch.bfloat16)
        quant_config = SDNQConfig(
            weights_dtype='int8',
            minimum_allowed_numel=1,
            minimum_allowed_channel_size=1,
        )
        quantizer = _textencoderuri._monolithic_hf_quantizer(quant_config)
        quantizer.preprocess_model(
            target,
            device_map=None,
            keep_in_fp32_modules=_textencoderuri._keep_in_fp32_module_names(target),
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, 't5.safetensors')
            save_file(state, checkpoint)
            loaded = _textencoderuri._load_monolithic_checkpoint(
                target,
                checkpoint=checkpoint,
                device_map={'': 'cpu'},
                dtype=torch.bfloat16,
                no_split_module_classes=['T5Block'],
            )
        loaded = _textencoderuri._finish_monolithic_quantization(loaded, quantizer, torch.bfloat16)

        wo = loaded.encoder.block[0].layer[1].DenseReluDense.wo
        q = loaded.encoder.block[0].layer[0].SelfAttention.q
        self.assertIsInstance(wo, torch.nn.Linear)
        self.assertEqual(wo.weight.dtype, torch.float32)
        self.assertTrue(hasattr(q, 'sdnq_dequantizer'))


if __name__ == '__main__':
    unittest.main() 