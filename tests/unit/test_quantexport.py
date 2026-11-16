import json
import os
import tempfile
import unittest

import gguf
import numpy
import torch

import dgenerate.messages as messages
import dgenerate.pipelinewrapper.ggufcompat as ggufcompat
import dgenerate.quantize.ggufquant as ggufquant
import dgenerate.quantize.quantexport as quantexport
import dgenerate.pipelinewrapper.safetensorscompat as safetensorscompat
import dgenerate.subcommands.exceptions as subcommand_exceptions
import dgenerate.subcommands.quantize as quantize_command


def numpy_weight():
    torch.manual_seed(1)
    return torch.randn(4, 256) * 0.05


def _relative(actual, expected) -> float:
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    if actual.shape != expected.shape and actual.numel() == expected.numel():
        if actual.ndim == 2 and tuple(actual.shape) == tuple(reversed(expected.shape)):
            actual = actual.transpose(0, 1).contiguous()
        else:
            actual = actual.reshape(expected.shape)
    return ((actual - expected).norm() / expected.norm().clamp(min=1e-8)).item()


def _read_gguf_tensor(path: str, name: str) -> tuple[torch.Tensor, gguf.GGMLQuantizationType]:
    reader = gguf.GGUFReader(path)
    for tensor in reader.tensors:
        if tensor.name != name:
            continue
        data = tensor.data
        if tensor.tensor_type == gguf.GGMLQuantizationType.F32:
            values = data.astype('float32')
        elif tensor.tensor_type == gguf.GGMLQuantizationType.F16:
            values = data.astype('float32')
        else:
            values = gguf.dequantize(data, tensor.tensor_type)
        return torch.from_numpy(values.copy()).float(), tensor.tensor_type
    raise AssertionError(name)


def _gguf_field(path_or_reader, key: str):
    reader = path_or_reader if hasattr(path_or_reader, 'get_field') else gguf.GGUFReader(path_or_reader)
    field = reader.get_field(key)
    if field is None:
        raise AssertionError(key)
    return field.contents()


def _gguf_scalar(reader, key: str):
    value = _gguf_field(reader, key)
    return int(value[0] if hasattr(value, '__getitem__') and not isinstance(value, (str, bytes)) else value)


def _gguf_text(reader, key: str) -> str:
    value = _gguf_field(reader, key)
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, str):
        return value
    part = value[0]
    return part.decode('utf-8') if isinstance(part, bytes) else str(part)


def _gguf_array(reader, key: str):
    value = _gguf_field(reader, key)
    items = []
    for item in value:
        if isinstance(item, bytes):
            items.append(item.decode('utf-8'))
        else:
            items.append(int(item) if not isinstance(item, str) else item)
    return items


class TestQuantExportFormats(unittest.TestCase):
    def test_presets_and_short_names_expand_once(self):
        resolved = quantexport.resolve_formats(['dgenerate', 'fp8', 'q4', 'Q8_0'])
        self.assertEqual(resolved, ['Q4_K', 'Q8_0', 'float8_e4m3fn'])
        plain = quantexport.resolve_formats(['fp16', 'float16', 'fp32', 'float32', 'F16'])
        self.assertEqual(plain, ['F16', 'float16', 'F32', 'float32'])

    def test_every_advertised_gguf_type_can_be_quantized(self):
        for name in quantexport.gguf_format_names():
            self.assertIn(name, quantexport.preset_formats()['gguf'] + ('BF16', 'F16', 'F32'))

    def test_text_architectures_match_the_loader(self):
        for architecture in ('t5', 't5encoder', 'qwen2', 'qwen2vl', 'qwen3', 'qwen3vl', 'gemma3', 'gemma4', 'llama'):
            self.assertIn(architecture, ggufcompat._TEXT_GGUF_ARCHITECTURES)
        self.assertEqual(quantexport._text_architecture('qwen3_5'), 'qwen35')
        self.assertEqual(quantexport._text_architecture('qwen3_5_moe'), 'qwen35moe')
        self.assertIn('qwen35', ggufcompat._TEXT_GGUF_ARCHITECTURES)
        self.assertEqual(quantexport._text_architecture('gemma4'), 'gemma4')

    def test_qwen35_linear_attention_round_trips(self):
        torch.manual_seed(0)
        hidden = 8
        num_k, num_v, head = 2, 4, 4
        key_dim = head * num_k
        value_dim = head * num_v
        text = {
            'linear_num_key_heads': num_k,
            'linear_num_value_heads': num_v,
            'linear_key_head_dim': head,
            'linear_value_head_dim': head,
        }
        original = {
            'model.layers.0.input_layernorm.weight': torch.zeros(hidden),
            'model.layers.0.post_attention_layernorm.weight': torch.zeros(hidden),
            'model.layers.0.linear_attn.norm.weight': torch.zeros(head),
            'model.layers.0.linear_attn.A_log': torch.linspace(-1.0, 0.5, num_v),
            'model.layers.0.linear_attn.dt_bias': torch.arange(num_v).float(),
            'model.layers.0.linear_attn.in_proj_qkv.weight': torch.randn(key_dim * 2 + value_dim, hidden),
            'model.layers.0.linear_attn.in_proj_z.weight': torch.randn(value_dim, hidden),
            'model.layers.0.linear_attn.in_proj_a.weight': torch.randn(num_v, hidden),
            'model.layers.0.linear_attn.in_proj_b.weight': torch.randn(num_v, hidden),
            'model.layers.0.linear_attn.out_proj.weight': torch.randn(hidden, value_dim),
            'model.layers.0.linear_attn.conv1d.weight': torch.randn(key_dim * 2 + value_dim, 1, 4),
            'model.norm.weight': torch.zeros(hidden),
        }
        from dgenerate.quantize.comfykeys import llama_state

        config = {'model_type': 'qwen3_5', 'text_config': text}
        written, kind = llama_state(original, config)
        self.assertEqual(kind, 'qwen35')
        self.assertIn('blk.0.ssm_a', written)
        self.assertIn('blk.0.post_attention_norm.weight', written)
        self.assertNotIn('blk.0.ffn_norm.weight', written)
        heads = (num_k, num_v, head, head)
        restored = ggufcompat._remap_keys(
            ggufcompat.qwen35_from_llama(written, heads), ggufcompat._LLAMA_GGUF_KEYS)
        for key, value in original.items():
            self.assertIn(key, restored)
            self.assertTrue(torch.allclose(restored[key].float(), value.float(), atol=1e-5), key)

    def test_text_sidecars_are_copied_beside_a_quant(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as output:
            for name in ('config.json', 'tokenizer_config.json', 'preprocessor_config.json'):
                with open(os.path.join(source, name), 'w', encoding='utf-8') as handle:
                    handle.write('{}')
            quantexport._copy_text_sidecars(source, output, {'model_type': 'gemma4'})
            self.assertTrue(os.path.isfile(os.path.join(output, 'config.json')))
            self.assertTrue(os.path.isfile(os.path.join(output, 'preprocessor_config.json')))
            quantexport._copy_text_sidecars(source, output, {'model_type': 'qwen3_5'})
            with open(os.path.join(output, 'config.json'), encoding='utf-8') as handle:
                self.assertEqual(handle.read(), '{}')
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as output:
            with open(os.path.join(source, 'config.json'), 'w', encoding='utf-8') as handle:
                handle.write('transformer')
            quantexport._copy_text_sidecars(source, output, {'model_type': 'flux'})
            self.assertFalse(os.path.isfile(os.path.join(output, 'config.json')))

    def test_every_gguf_type_dequantizes(self):
        import gguf
        weight = numpy_weight()
        for name in quantexport.gguf_format_names():
            if name in {'F32', 'F16', 'BF16'}:
                continue
            qtype = getattr(gguf.GGMLQuantizationType, name)
            block = gguf.GGML_QUANT_SIZES[qtype][0]
            cols = block if block >= 32 else 32
            sample = weight[:, :cols] if cols <= weight.shape[1] else weight
            if sample.shape[1] % block:
                sample = weight[:, :block]
            packed = ggufquant.quantize(sample.numpy(), qtype)
            restored = torch.from_numpy(gguf.dequantize(packed, qtype).copy())
            with self.subTest(name=name):
                self.assertLess(_relative(restored, sample), 0.35)
        with self.assertRaises(quantexport.QuantExportError) as caught:
            quantexport.resolve_formats(['nope'])
        self.assertIn('--list-formats', str(caught.exception))


class TestQuantExportFiles(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.weight = torch.randn(32, 256) * 0.05
        self.bias = torch.randn(32)
        self.narrow = torch.randn(4, 17) * 0.05
        self.state = {
            'blocks.0.attn.to_q.weight': self.weight,
            'blocks.0.attn.to_q.bias': self.bias,
            'embed_tokens.weight': torch.randn(8, 32),
            'blocks.0.attn.to_k.weight': self.narrow,
        }

    def _comfy_weight(self, path, name, shape):
        loaded = safetensorscompat.load_checkpoint(path, torch.float32)
        self.assertIn(name, loaded.sources)
        return loaded.sources[name].dequantize(torch.float32, shape=shape)

    def test_gguf_round_trip_and_text_encoder_architecture(self):
        state = {
            'encoder.block.0.layer.0.SelfAttention.q.weight': self.weight,
            'shared.weight': torch.randn(8, 32),
            'encoder.block.0.layer.0.SelfAttention.q.bias': self.bias,
        }
        config = {'model_type': 't5', 'architectures': ['T5EncoderModel']}
        with tempfile.TemporaryDirectory() as directory:
            paths = quantexport.export_component(
                state, ['Q8_0', 'Q4_K', 'BF16'], directory, 'text_encoder', config)
            by_suffix = {os.path.basename(path): path for path in paths}
            weight, qtype = _read_gguf_tensor(
                by_suffix['text_encoder-Q8_0.gguf'],
                'encoder.block.0.layer.0.SelfAttention.q.weight')
            self.assertEqual(qtype, gguf.GGMLQuantizationType.Q8_0)
            self.assertLess(_relative(weight, self.weight), 0.05)

            weight, qtype = _read_gguf_tensor(
                by_suffix['text_encoder-Q4_K.gguf'],
                'encoder.block.0.layer.0.SelfAttention.q.weight')
            self.assertEqual(qtype, gguf.GGMLQuantizationType.Q4_K)
            self.assertLess(_relative(weight, self.weight), 0.12)

            narrow, narrow_type = _read_gguf_tensor(
                by_suffix['text_encoder-Q4_K.gguf'],
                'encoder.block.0.layer.0.SelfAttention.q.bias')
            self.assertEqual(narrow_type, gguf.GGMLQuantizationType.F32)
            self.assertLess(_relative(narrow, self.bias), 1e-5)

            restored, bf16_type = _read_gguf_tensor(
                by_suffix['text_encoder-BF16.gguf'],
                'encoder.block.0.layer.0.SelfAttention.q.weight')
            self.assertEqual(bf16_type, gguf.GGMLQuantizationType.BF16)
            self.assertLess(_relative(restored, self.weight), 1e-2)

            self.assertEqual(
                ggufcompat._gguf_architecture(by_suffix['text_encoder-Q8_0.gguf']), 't5encoder')
            architecture, loaded, _ = ggufcompat.load_text_encoder_gguf(
                by_suffix['text_encoder-Q8_0.gguf'])
            self.assertEqual(architecture, 't5encoder')
            self.assertIn('encoder.block.0.layer.0.SelfAttention.q.weight', loaded)

    def test_plain_f16_and_f32_store_every_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = quantexport.export_component(
                self.state, ['F16', 'F32', 'float16', 'float32'], directory, 'transformer', None)
            by_name = {os.path.basename(path): path for path in paths}
            weight, qtype = _read_gguf_tensor(
                by_name['transformer-F16.gguf'], 'blocks.0.attn.to_q.weight')
            self.assertEqual(qtype, gguf.GGMLQuantizationType.F16)
            self.assertLess(_relative(weight, self.weight), 1e-2)
            bias, bias_type = _read_gguf_tensor(
                by_name['transformer-F16.gguf'], 'blocks.0.attn.to_q.bias')
            self.assertEqual(bias_type, gguf.GGMLQuantizationType.F16)
            self.assertLess(_relative(bias, self.bias), 1e-2)
            exact, exact_type = _read_gguf_tensor(
                by_name['transformer-F32.gguf'], 'blocks.0.attn.to_k.weight')
            self.assertEqual(exact_type, gguf.GGMLQuantizationType.F32)
            self.assertLess(_relative(exact, self.narrow), 1e-6)

            import safetensors.torch
            half = safetensors.torch.load_file(by_name['transformer-float16.safetensors'])
            full = safetensors.torch.load_file(by_name['transformer-float32.safetensors'])
            self.assertEqual(half['blocks.0.attn.to_q.weight'].dtype, torch.float16)
            self.assertEqual(full['blocks.0.attn.to_q.weight'].dtype, torch.float32)
            self.assertFalse(safetensorscompat.is_comfy_quantized(
                by_name['transformer-float16.safetensors']))
            self.assertLess(_relative(half['blocks.0.attn.to_q.weight'], self.weight), 1e-2)
            self.assertLess(_relative(full['blocks.0.attn.to_k.weight'], self.narrow), 1e-6)

    def test_comfy_float16_converts_names_without_a_quant(self):
        state = {
            'single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight': self.weight[:, :32],
            'single_transformer_blocks.0.attn.to_qkv_mlp_proj.bias': self.bias,
        }
        config = {'_class_name': 'Flux2Transformer2DModel'}
        with tempfile.TemporaryDirectory() as directory:
            paths = quantexport.export_component(
                state, ['float16'], directory, 'transformer', config, comfy=True)
            self.assertEqual(os.path.basename(paths[0]), 'transformer-float16-comfy.safetensors')
            self.assertFalse(safetensorscompat.is_comfy_quantized(paths[0]))
            import safetensors.torch
            loaded = safetensors.torch.load_file(paths[0])
            self.assertIn('single_blocks.0.linear1.weight', loaded)
            self.assertEqual(loaded['single_blocks.0.linear1.weight'].dtype, torch.float16)
            self.assertLess(
                _relative(loaded['single_blocks.0.linear1.weight'], state[
                    'single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight']),
                1e-2)
            with self.assertRaises(quantexport.QuantExportError):
                quantexport.export_component(
                    state, ['float16'], directory, 'transformer', config, llama=True)

    def test_unaligned_matrix_stays_bf16(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = quantexport.export_component(
                self.state, ['Q8_0'], directory, 'transformer', None)
            _, qtype = _read_gguf_tensor(paths[0], 'blocks.0.attn.to_k.weight')
            self.assertEqual(qtype, gguf.GGMLQuantizationType.BF16)
            values, _ = _read_gguf_tensor(paths[0], 'blocks.0.attn.to_k.weight')
            self.assertLess(_relative(values, self.narrow), 1e-2)

    def test_comfy_formats_round_trip(self):
        limits = {
            'float8_e4m3fn': 0.08,
            'float8_e5m2': 0.2,
            'int8_tensorwise': 0.05,
            'nvfp4': 0.35,
            'mxfp8': 0.15,
            'convrot_w4a4': 0.2,
            'asym_w4a8_int8': 0.25,
            'w6a8_int8': 0.2,
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = quantexport.export_component(
                self.state, list(limits), directory, 'transformer', None)
            self.assertEqual(len(paths), len(limits))
            for path, (fmt, limit) in zip(paths, limits.items()):
                with self.subTest(fmt=fmt):
                    self.assertTrue(safetensorscompat.is_comfy_quantized(path))
                    dense = self._comfy_weight(
                        path, 'blocks.0.attn.to_q.weight', tuple(self.weight.shape))
                    self.assertLess(_relative(dense, self.weight), limit)
                    loaded = safetensorscompat.load_checkpoint(path, torch.float32)
                    self.assertNotIn('embed_tokens.weight', loaded.sources)

    def test_component_folders_come_from_the_index(self):
        with tempfile.TemporaryDirectory() as root:
            transformer = os.path.join(root, 'transformer')
            tokenizer = os.path.join(root, 'tokenizer')
            os.makedirs(transformer)
            os.makedirs(tokenizer)
            torch.save({'weight': torch.zeros(2, 2)}, os.path.join(transformer, 'diffusion_pytorch_model.bin'))
            with open(os.path.join(root, 'model_index.json'), 'w', encoding='utf-8') as handle:
                json.dump({
                    '_class_name': 'Test',
                    'transformer': ['diffusers', 'Model'],
                    'tokenizer': ['transformers', 'Tokenizer'],
                }, handle)
            self.assertEqual(
                quantexport.component_directories(root),
                [('transformer', transformer)])
            with self.assertRaises(quantexport.QuantExportError):
                quantexport.component_directories(root, ['vae'])

    def test_weight_name_selects_one_file(self):
        with tempfile.TemporaryDirectory() as root:
            folder = os.path.join(root, 'text_encoder')
            os.makedirs(folder)
            import safetensors.torch
            safetensors.torch.save_file(
                {'text_model.embeddings.token_embedding.weight': torch.zeros(4, 4)},
                os.path.join(folder, 'model.safetensors'))
            safetensors.torch.save_file(
                {'text_model.embeddings.token_embedding.weight': torch.ones(4, 4)},
                os.path.join(folder, 'model.fp16.safetensors'))
            targets = quantexport.weight_targets(root, weight_name='model.fp16.safetensors')
            self.assertEqual(targets, [('text_encoder', folder, 'model.fp16.safetensors')])
            state = quantexport.load_component_state(folder, weight_name='model.fp16.safetensors')
            self.assertEqual(float(state['text_model.embeddings.token_embedding.weight'].sum()), 16.0)
            named = quantexport.weight_targets(
                root, weight_name='text_encoder/model.fp16.safetensors')
            self.assertEqual(named[0][0], 'text_encoder')
            self.assertEqual(named[0][2], 'model.fp16.safetensors')

    def test_single_weight_reference_and_name(self):
        self.assertTrue(quantexport.single_weight_reference(
            'https://huggingface.co/org/repo/blob/main/text_encoder/model.fp16.safetensors'))
        self.assertFalse(quantexport.single_weight_reference('org/repo'))
        self.assertEqual(
            quantexport.component_name_for_file(
                r'C:\models\text_encoder\model.fp16.safetensors'),
            'text_encoder')
        self.assertEqual(
            quantexport.component_name_for_file(
                r'C:\models\clip_g.safetensors'),
            'clip_g')
        self.assertEqual(
            quantexport.component_name_for_file(
                r'C:\cache\aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\model.safetensors',
                'text_encoder_2'),
            'text_encoder_2')


class TestQuantizeSubCommand(unittest.TestCase):
    def _command(self, args):
        return quantize_command.QuantizeSubCommand(
            loaded_by_name='quantize', plugin_module_paths=[], args=args)

    def test_list_formats_and_missing_formats(self):
        self.assertEqual(self._command(['--list-formats'])(), 0)
        with self.assertRaises(subcommand_exceptions.SubCommandArgumentError):
            self._command(['model'])()

    def test_writes_the_requested_files(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            folder = os.path.join(root, 'transformer')
            os.makedirs(folder)
            weight = torch.randn(32, 256) * 0.05
            import safetensors.torch
            safetensors.torch.save_file(
                {'blocks.0.attn.to_q.weight': weight, 'blocks.0.attn.to_q.bias': torch.randn(32)},
                os.path.join(folder, 'diffusion_pytorch_model.safetensors'))
            with open(os.path.join(folder, 'config.json'), 'w', encoding='utf-8') as handle:
                json.dump({'model_type': 'flux'}, handle)
            status = self._command([
                root, '--subfolders', 'transformer', '--formats', 'q8', 'fp8', '-o', output,
            ])()
            self.assertEqual(status, 0)
            self.assertTrue(os.path.isfile(os.path.join(output, 'transformer-Q8_0.gguf')))
            safetensors_path = os.path.join(output, 'transformer-float8_e4m3fn.safetensors')
            self.assertTrue(safetensorscompat.is_comfy_quantized(safetensors_path))
            loaded = safetensorscompat.load_checkpoint(safetensors_path, torch.float32)
            dense = loaded.sources['blocks.0.attn.to_q.weight'].dequantize(
                torch.float32, shape=tuple(weight.shape))
            self.assertLess(_relative(dense, weight), 0.08)

    def test_clip_gguf_is_labeled_clip(self):
        state = {
            'text_model.embeddings.token_embedding.weight': torch.randn(32, 64),
            'text_model.encoder.layers.0.self_attn.q_proj.weight': torch.randn(64, 64),
        }
        config = {'model_type': 'clip_text_model', 'architectures': ['CLIPTextModel']}
        with tempfile.TemporaryDirectory() as directory, messages_warning() as warned:
            paths = quantexport.export_component(
                state, ['Q8_0'], directory, 'text_encoder', config)
            self.assertFalse(any('text encoder' in note for note in warned))
            import gguf
            reader = gguf.GGUFReader(paths[0])
            field = reader.get_field('general.architecture')
            architecture = str(bytes(field.parts[field.data[-1]]), encoding='utf-8')
            reader.data._mmap.close()
            del reader
        self.assertEqual(architecture, 'clip')

    def test_single_file_and_weight_name_arguments(self):
        import safetensors.torch
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            folder = os.path.join(root, 'text_encoder')
            os.makedirs(folder)
            weight = os.path.join(folder, 'model.fp16.safetensors')
            safetensors.torch.save_file(
                {'text_model.embeddings.token_embedding.weight': torch.ones(8, 32),
                 'text_model.encoder.layers.0.self_attn.q_proj.weight': torch.randn(32, 32)},
                weight)
            with open(os.path.join(folder, 'config.json'), 'w', encoding='utf-8') as handle:
                json.dump({'model_type': 'clip_text_model', 'architectures': ['CLIPTextModel']}, handle)
            status = self._command([
                weight, '--formats', 'q8', '-o', output, '--offline-mode',
            ])()
            self.assertEqual(status, 0)
            self.assertTrue(os.path.isfile(os.path.join(output, 'text_encoder-Q8_0.gguf')))
            with self.assertRaises(subcommand_exceptions.SubCommandArgumentError):
                self._command([
                    weight, '--subfolders', 'text_encoder', '--formats', 'q8', '-o', output,
                ])()
            other = os.path.join(output, 'by-name')
            status = self._command([
                root, '--weight-name', 'model.fp16.safetensors', '--formats', 'q8',
                '-o', other, '--offline-mode',
            ])()
            self.assertEqual(status, 0)
            self.assertTrue(os.path.isfile(os.path.join(other, 'text_encoder-Q8_0.gguf')))

    def test_llama_flag_writes_llama_cpp_names_and_excludes_comfy(self):
        import safetensors.torch
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            folder = os.path.join(root, 'text_encoder')
            os.makedirs(folder)
            safetensors.torch.save_file(
                {'shared.weight': torch.ones(8, 32),
                 'encoder.block.0.layer.0.SelfAttention.q.weight': torch.randn(32, 32)},
                os.path.join(folder, 'model.safetensors'))
            with open(os.path.join(folder, 'config.json'), 'w', encoding='utf-8') as handle:
                json.dump({'model_type': 't5'}, handle)
            status = self._command([
                root, '--subfolders', 'text_encoder', '--formats', 'q8', '--llama', '-o', output,
            ])()
            self.assertEqual(status, 0)
            path = os.path.join(output, 'text_encoder-Q8_0-llama.gguf')
            self.assertTrue(os.path.isfile(path))
            reader = gguf.GGUFReader(path)
            names = {tensor.name for tensor in reader.tensors}
            self.assertIn('enc.blk.0.attn_q.weight', names)
            self.assertIsNone(reader.get_field('comfy.gguf.orig_shape.enc.blk.0.attn_q.weight'))
            reader.data._mmap.close()
            del reader

    def test_llama_file_stores_tokenizer_hparams_and_tied_output(self):
        import safetensors.torch
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            folder = os.path.join(root, 'text_encoder')
            os.makedirs(folder)
            torch.manual_seed(0)
            safetensors.torch.save_file({
                'model.embed_tokens.weight': torch.arange(6 * 32, dtype=torch.float32).reshape(6, 32),
                'model.layers.0.self_attn.q_proj.weight': torch.randn(32, 32),
                'model.layers.0.self_attn.k_proj.weight': torch.randn(16, 32),
                'model.layers.0.self_attn.v_proj.weight': torch.randn(16, 32),
                'model.layers.0.self_attn.o_proj.weight': torch.randn(32, 32),
                'model.norm.weight': torch.ones(32),
            }, os.path.join(folder, 'model.safetensors'))
            with open(os.path.join(folder, 'config.json'), 'w', encoding='utf-8') as handle:
                json.dump({
                    'model_type': 'qwen3_vl',
                    'tie_word_embeddings': True,
                    'text_config': {
                        'num_hidden_layers': 1,
                        'hidden_size': 32,
                        'intermediate_size': 64,
                        'num_attention_heads': 4,
                        'num_key_value_heads': 2,
                        'head_dim': 8,
                        'rms_norm_eps': 1e-6,
                        'rope_theta': 5000000,
                        'max_position_embeddings': 128,
                        'vocab_size': 8,
                        'rope_scaling': {'mrope_section': [4, 2, 2]},
                    },
                }, handle)
            with open(os.path.join(folder, 'tokenizer.json'), 'w', encoding='utf-8') as handle:
                json.dump({
                    'model': {
                        'type': 'BPE',
                        'vocab': {'<|endoftext|>': 0, 'hello': 1, '<0x0A>': 2},
                        'merges': ['h ello'],
                    },
                    'added_tokens': [
                        {'id': 0, 'content': '<|endoftext|>', 'special': True},
                        {'id': 5, 'content': '<|im_end|>', 'special': True},
                    ],
                }, handle)
            with open(os.path.join(folder, 'tokenizer_config.json'), 'w', encoding='utf-8') as handle:
                json.dump({
                    'tokenizer_class': 'Qwen2Tokenizer',
                    'eos_token': '<|im_end|>',
                    'add_bos_token': False,
                }, handle)
            with open(os.path.join(folder, 'chat_template.json'), 'w', encoding='utf-8') as handle:
                json.dump({'chat_template': 'TEMPLATE'}, handle)
            status = self._command([
                root, '--subfolders', 'text_encoder', '--formats', 'f32', '--llama', '-o', output,
            ])()
            self.assertEqual(status, 0)
            path = os.path.join(output, 'text_encoder-F32-llama.gguf')
            reader = gguf.GGUFReader(path)
            names = {tensor.name for tensor in reader.tensors}
            self.assertIn('output.weight', names)
            self.assertIn('token_embd.weight', names)
            self.assertEqual(_gguf_scalar(reader, 'qwen3vl.block_count'), 1)
            self.assertEqual(_gguf_scalar(reader, 'qwen3vl.attention.head_count'), 4)
            self.assertEqual(_gguf_scalar(reader, 'qwen3vl.attention.head_count_kv'), 2)
            self.assertEqual(list(_gguf_array(reader, 'qwen3vl.rope.dimension_sections')), [4, 2, 2, 0])
            self.assertEqual(_gguf_text(reader, 'tokenizer.ggml.model'), 'gpt2')
            self.assertEqual(_gguf_text(reader, 'tokenizer.ggml.pre'), 'qwen2')
            self.assertEqual(_gguf_text(reader, 'tokenizer.chat_template'), 'TEMPLATE')
            self.assertEqual(_gguf_scalar(reader, 'tokenizer.ggml.eos_token_id'), 5)
            tokens = _gguf_array(reader, 'tokenizer.ggml.tokens')
            self.assertEqual(len(tokens), 8)
            self.assertEqual(tokens[1], 'hello')
            self.assertEqual(tokens[5], '<|im_end|>')
            self.assertTrue(tokens[6].startswith('<unused'))
            reader.data._mmap.close()
            del reader
        status = self._command([
            'model', '--formats', 'q8', '-o', 'out', '--comfy', '--llama',
        ])()
        self.assertNotEqual(status, 0)


class messages_warning:
    """Collect dgenerate warnings. ``assertLogs`` watches the logging module, which this does not use."""

    def __enter__(self):
        self.notes = []
        self.original = messages.warning
        messages.warning = lambda *args, **kwargs: self.notes.append(' '.join(str(arg) for arg in args))
        return self.notes

    def __exit__(self, exc_type, exc, traceback):
        messages.warning = self.original
        return False
