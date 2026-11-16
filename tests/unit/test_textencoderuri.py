import unittest
import unittest.mock as mock

import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.uris.textencoderuri as _textencoderuri
import dgenerate.hfhub as _hfhub
from dgenerate.pipelinewrapper.uris.exceptions import InvalidTextEncoderUriError


class TestTextEncoderUri(unittest.TestCase):

    def test_mmproj_url_downloads_that_file(self):
        import tempfile
        url = ('https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/'
               'mmproj-F16.gguf')
        with tempfile.NamedTemporaryFile(suffix='.gguf') as cached:
            with mock.patch(
                    'dgenerate.hfhub.huggingface_hub.hf_hub_download',
                    return_value=cached.name) as hub:
                path = _textencoderuri._materialize_mmproj(url, False, 'token')
            self.assertEqual(path, cached.name)
            self.assertEqual(hub.call_args.kwargs['filename'], 'mmproj-F16.gguf')
            self.assertEqual(hub.call_args.kwargs['repo_id'], 'unsloth/gemma-4-E4B-it-GGUF')

    def test_mmproj_file_name_is_not_taken_from_the_model_repository(self):
        from dgenerate.pipelinewrapper.uris.exceptions import TextEncoderUriLoadError
        with mock.patch('dgenerate.hfhub.huggingface_hub.hf_hub_download') as hub:
            with self.assertRaises(TextEncoderUriLoadError):
                _textencoderuri._materialize_mmproj('mmproj-F16.gguf', False, None)
        hub.assert_not_called()

    def test_mmproj_is_parsed_for_a_gguf_encoder(self):
        uri = "Qwen2_5_VLForConditionalGeneration;model=text.gguf;mmproj=mmproj-text.gguf"
        result = _textencoderuri.TextEncoderUri.parse(uri)
        self.assertEqual(result.mmproj, "mmproj-text.gguf")
        with self.assertRaises(InvalidTextEncoderUriError):
            _textencoderuri.TextEncoderUri.parse(
                "CLIPTextModel;model=encoder.safetensors;mmproj=mmproj-text.gguf")

    def test_mmproj_is_rejected_for_a_text_only_encoder(self):
        for uri in (
                "CLIPTextModel;model=encoder.gguf;mmproj=tower.gguf",
                "Qwen2_5_VLTextModel;model=encoder.gguf;mmproj=tower.gguf",
                "T5EncoderModel;model=org/repo;weight-name=encoder.gguf;mmproj=tower.gguf"):
            with self.assertRaises(InvalidTextEncoderUriError) as caught:
                _textencoderuri.TextEncoderUri.parse(uri)
            self.assertIn('has no vision tower', str(caught.exception))
        result = _textencoderuri.TextEncoderUri.parse(
            "Qwen3VLModel;model=text.gguf;mmproj=tower.gguf")
        self.assertEqual(result.mmproj, "tower.gguf")

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

    def test_flattened_clip_lora_loads_kohya_text_model_prefix(self):
        # Transformers 5 flattens CLIPTextModel; Kohya LoRA keys still use
        # text_model.*. Without stripping, PEFT rank discovery stays empty and
        # diffusers raises IndexError in get_peft_kwargs.
        import torch
        from transformers import CLIPTextConfig, CLIPTextModel
        import diffusers.loaders.lora_base as lora_base
        import diffusers.loaders.lora_pipeline as lora_pipeline

        self.assertTrue(
            getattr(lora_base._load_lora_into_text_encoder,
                    '_dgenerate_strips_flattened_clip_prefix', False))
        self.assertIs(
            lora_base._load_lora_into_text_encoder,
            lora_pipeline._load_lora_into_text_encoder)

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
        text_encoder = CLIPTextModel(config).eval()
        self.assertNotIn('text_model', text_encoder._modules)
        self.assertTrue(hasattr(text_encoder, 'text_model'))

        rank = 2
        state_dict = {}
        with torch.no_grad():
            for name, module in text_encoder.named_modules():
                if name.endswith(('.q_proj', '.k_proj', '.v_proj', '.out_proj', '.fc1', '.fc2')):
                    state_dict[f'text_encoder.text_model.{name}.lora_A.weight'] = (
                        torch.randn(rank, module.weight.shape[1]))
                    state_dict[f'text_encoder.text_model.{name}.lora_B.weight'] = (
                        torch.randn(module.weight.shape[0], rank))

        lora_base._load_lora_into_text_encoder(
            state_dict,
            None,
            text_encoder,
            prefix='text_encoder',
            adapter_name='0',
        )
        self.assertIn('0', getattr(text_encoder, 'peft_config', {}))

    def test_all_in_one_checkpoint_is_not_one_clip(self):
        import tempfile

        import torch
        from safetensors.torch import save_file
        from transformers import CLIPTextModelWithProjection

        from dgenerate.pipelinewrapper.uris.exceptions import TextEncoderUriLoadError

        towers = {}
        for name, hidden in (('clip_g', 1280), ('clip_l', 768)):
            prefix = f'text_encoders.{name}.transformer.'
            towers[prefix + 'text_model.embeddings.token_embedding.weight'] = torch.zeros(8, hidden)
            towers[prefix + 'text_model.encoder.layers.0.layer_norm1.weight'] = torch.zeros(hidden)
        towers['model.diffusion_model.pos_embed'] = torch.zeros(1, 4)

        with tempfile.TemporaryDirectory() as directory:
            path = f'{directory}/sd3.safetensors'
            save_file(towers, path)
            self.assertIsNone(_textencoderuri._clip_family_config(
                path, CLIPTextModelWithProjection))

            one = {
                'text_model.embeddings.token_embedding.weight': torch.zeros(8, 32),
                'text_model.encoder.layers.0.layer_norm1.weight': torch.zeros(32),
            }
            save_file(one, f'{directory}/clip.safetensors')
            with self.assertRaises(TextEncoderUriLoadError):
                _textencoderuri._clip_family_config(
                    f'{directory}/clip.safetensors', CLIPTextModelWithProjection)

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

    def _roundtrip_load(self, encoder_name, model):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = _textencoderuri.TextEncoderUri.parse(
                f'{encoder_name};model={directory};dtype=float32'
            ).load(local_files_only=True, no_cache=True)
            self.assertEqual(type(loaded).__name__, encoder_name)

    def test_load_umt5_encoder_model(self):
        import transformers
        from transformers import UMT5Config

        model = transformers.UMT5EncoderModel(UMT5Config(
            d_model=32,
            d_kv=8,
            d_ff=64,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            vocab_size=100,
            is_encoder_decoder=False,
        ))
        self._roundtrip_load('UMT5EncoderModel', model)

    def test_load_qwen3_model(self):
        import transformers
        from transformers import Qwen3Config

        model = transformers.Qwen3Model(Qwen3Config(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=100,
            max_position_embeddings=64,
        ))
        self._roundtrip_load('Qwen3Model', model)

    def test_load_qwen3_for_causal_lm(self):
        import transformers
        from transformers import Qwen3Config

        model = transformers.Qwen3ForCausalLM(Qwen3Config(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=100,
            max_position_embeddings=64,
        ))
        self._roundtrip_load('Qwen3ForCausalLM', model)

    def test_load_qwen2_5_vl_for_conditional_generation(self):
        import transformers

        config = transformers.Qwen2_5_VLConfig(
            vision_config={
                'depth': 1,
                'hidden_size': 32,
                'intermediate_size': 64,
                'num_heads': 4,
                'out_hidden_size': 32,
                'patch_size': 14,
                'spatial_merge_size': 2,
                'temporal_patch_size': 2,
                'in_channels': 3,
            },
            text_config={
                'hidden_size': 32,
                'intermediate_size': 64,
                'num_hidden_layers': 1,
                'num_attention_heads': 4,
                'num_key_value_heads': 2,
                'vocab_size': 128,
                'max_position_embeddings': 64,
                'bos_token_id': 0,
                'eos_token_id': 1,
            },
            image_token_id=2,
            video_token_id=3,
            vision_start_token_id=4,
            vision_end_token_id=5,
            bos_token_id=0,
            eos_token_id=1,
        )
        model = transformers.Qwen2_5_VLForConditionalGeneration(config)
        self._roundtrip_load('Qwen2_5_VLForConditionalGeneration', model)

    def test_load_mistral3_for_conditional_generation(self):
        import transformers

        try:
            config = transformers.Mistral3Config.from_pretrained(
                'black-forest-labs/FLUX.2-dev',
                subfolder='text_encoder',
                local_files_only=True,
            )
        except Exception as exc:
            self.skipTest(f'Mistral3 config unavailable offline: {exc}')
        text = config.text_config
        for attr, value in (
            ('hidden_size', 32),
            ('intermediate_size', 64),
            ('num_hidden_layers', 1),
            ('num_attention_heads', 4),
            ('num_key_value_heads', 2),
            ('vocab_size', 128),
            ('head_dim', 8),
        ):
            if hasattr(text, attr):
                setattr(text, attr, value)
        vision = config.vision_config
        if vision is not None:
            for attr, value in (
                ('hidden_size', 32),
                ('intermediate_size', 64),
                ('num_hidden_layers', 1),
                ('num_attention_heads', 4),
                ('num_channels', 3),
            ):
                if hasattr(vision, attr):
                    setattr(vision, attr, value)
        try:
            model = transformers.Mistral3ForConditionalGeneration(config)
        except Exception as exc:
            self.skipTest(f'cannot build tiny Mistral3: {exc}')
        self._roundtrip_load('Mistral3ForConditionalGeneration', model)

    def test_load_gemma4_unified_for_conditional_generation(self):
        import tempfile
        import unittest.mock as mock

        import torch
        import transformers

        sentinel = torch.nn.Linear(1, 1)

        with mock.patch.object(
                transformers.Gemma4UnifiedForConditionalGeneration,
                'from_pretrained',
                return_value=sentinel) as from_pretrained:
            with tempfile.TemporaryDirectory() as directory:
                loaded = _textencoderuri.TextEncoderUri.parse(
                    f'Gemma4UnifiedForConditionalGeneration;model={directory};dtype=float32'
                ).load(local_files_only=True, no_cache=True)

        self.assertIs(loaded, sentinel)
        self.assertTrue(from_pretrained.called)
        kwargs = from_pretrained.call_args.kwargs
        self.assertIn('dtype', kwargs)
        self.assertTrue(kwargs.get('local_files_only'))

    def test_gguf_class_follows_the_architecture_and_shape(self):
        choose = _textencoderuri._gguf_encoder_class
        self.assertEqual(choose('qwen3', (151936, 2560), 36).__name__, 'Qwen3ForCausalLM')
        self.assertEqual(
            choose('qwen35', (248320, 4096), 32).__name__, 'Qwen3_5ForConditionalGeneration')
        self.assertEqual(choose('t5encoder', (32128, 4096), 24).__name__, 'T5EncoderModel')
        self.assertEqual(choose('t5encoder', (256384, 4096), 24).__name__, 'UMT5EncoderModel')
        self.assertEqual(choose('gemma4', (262144, 1536), 35).__name__, 'Gemma4ForConditionalGeneration')
        self.assertEqual(
            choose('qwen2vl', (152064, 3584), 28).__name__,
            'Qwen2_5_VLForConditionalGeneration')
        self.assertEqual(choose('clip', (49408, 768), 12).__name__, 'CLIPTextModel')
        self.assertEqual(
            choose('clip', (49408, 1280), 32).__name__, 'CLIPTextModelWithProjection')
        self.assertIsNone(choose('not-an-encoder', (1, 1), 1))

    def test_gguf_warns_when_the_named_class_does_not_match(self):
        mismatch = _textencoderuri._gguf_class_mismatch
        named = _textencoderuri.TextEncoderUri._resolve_encoder_class
        qwen = named('Qwen3ForCausalLM')
        self.assertIsNone(mismatch(qwen, 'qwen3', (151936, 2560), 36))
        t5_on_qwen = mismatch(named('T5EncoderModel'), 'qwen3', (151936, 2560), 36)
        self.assertIn('Qwen3ForCausalLM', t5_on_qwen)
        self.assertIn('T5EncoderModel', t5_on_qwen)
        self.assertIsNone(mismatch(
            named('Qwen2_5_VLTextModel'), 'qwen2vl', (152064, 3584), 28))
        self.assertIsNone(mismatch(
            named('Qwen3VLModel'), 'qwen3vl', (151936, 2560), 36))
        clip_g = mismatch(named('CLIPTextModel'), 'clip', (49408, 1280), 32)
        self.assertIn('CLIPTextModelWithProjection', clip_g)
        umt5_on_t5 = mismatch(named('UMT5EncoderModel'), 't5encoder', (32128, 4096), 24)
        self.assertIn('T5EncoderModel', umt5_on_t5)

    def test_shared_key_value_layers_are_not_a_mismatch(self):
        model_keys = {
            f'model.language_model.layers.{index}.self_attn.q_proj.weight'
            for index in range(35)}
        model_keys.update(
            f'model.language_model.layers.{index}.self_attn.k_proj.weight'
            for index in range(15))
        checkpoint = [f'model.layers.{index}.self_attn.q_proj.weight' for index in range(35)]
        checkpoint += [f'model.layers.{index}.self_attn.k_proj.weight' for index in range(35)]
        rename = _textencoderuri._language_model_rename(model_keys, checkpoint)
        self.assertEqual(
            rename('model.layers.0.self_attn.q_proj.weight'),
            'model.language_model.layers.0.self_attn.q_proj.weight')

    def test_unrelated_tensors_are_still_rejected(self):
        from dgenerate.pipelinewrapper.uris.exceptions import TextEncoderUriLoadError
        model_keys = {'model.language_model.layers.0.self_attn.q_proj.weight'}
        checkpoint = [f'other.{index}.block.weight' for index in range(20)]
        with self.assertRaises(TextEncoderUriLoadError):
            _textencoderuri._language_model_rename(model_keys, checkpoint)


if __name__ == '__main__':
    unittest.main() 