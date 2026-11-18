import json
import os
import tempfile
import unittest
from unittest.mock import patch

import dgenerate.quantize.hfcard as hfcard
import dgenerate.quantize.repoquant as repoquant
import dgenerate.subcommands.exceptions as subcommand_exceptions
import dgenerate.subcommands.quantize as quantize_command


def _command(args):
    return quantize_command.QuantizeSubCommand(
        loaded_by_name='quantize', plugin_module_paths=[], args=args)


class TestModelCard(unittest.TestCase):
    def test_non_tty_upload_follows_transferred_bytes(self):
        import threading
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'weights.gguf')
            with open(path, 'wb') as handle:
                handle.write(b'x' * 1000)

            class Operation:
                path_or_fileobj = path

            class Display:
                _tty = True
                _active = False
                _xet_bytes = 0
                _lock = threading.Lock()
                _stop_event = threading.Event()

                def _render_loop(self):
                    pass

                def close(self):
                    self.closed = True

            class Pipeline:
                display = Display()
                add_operations = [Operation()]
                abort_event = threading.Event()

            pipeline = Pipeline()
            hfcard._attach_upload_bar(pipeline)
            bar = pipeline.display._dgenerate_bar
            self.assertEqual(bar.total, 1000)
            pipeline.display._xet_bytes = 400
            pipeline.display._sync_upload_bar()
            self.assertGreaterEqual(bar.n, 400)
            pipeline.display.close()
            self.assertTrue(pipeline.display.closed)
            self.assertEqual(bar.n, 1000)

    def test_bare_upload_card_builds_the_generated_card(self):
        self.assertEqual(hfcard.parse_upload_card('').kind, 'generate')
        self.assertEqual(hfcard.parse_upload_card('generate').kind, 'generate')
        command = hfcard.quantize_command_line(
            'org/model', formats=['q4'], hf_upload_card='')
        self.assertIn('--hf-upload-card', command)
        self.assertNotIn('--hf-upload-card generate', command)

    def test_card_records_quant_settings_command_and_omits_the_token(self):
        spec = repoquant.parse_quantizer(
            'sdnq;type=uint4;group-size=32;quant-conv=true;use-svd=true')
        bnb = repoquant.parse_quantizer(
            'bnb;bits=4;bits4-quant-type=nf4;bits4-use-double-quant=true')
        command = hfcard.quantize_command_line(
            'black-forest-labs/FLUX.1-dev',
            subfolders=['transformer'],
            formats=['q4'],
            quantizer_uris=[spec.uri, bnb.uri],
            output='quants',
            revision='main',
            hf_upload='me/flux-q4')
        card = hfcard.build_model_card(
            model='black-forest-labs/FLUX.1-dev',
            repo_id='me/flux-q4',
            revision='main',
            subfolders=['transformer'],
            formats_requested=['q4'],
            formats=['Q4_K'],
            quantizers=[spec, bnb],
            comfy=False,
            llama=False,
            variant=None,
            weight_name=None,
            command=command,
            preview=True)
        self.assertIn('base_model: black-forest-labs/FLUX.1-dev', card)
        self.assertIn('base_model_relation: quantized', card)
        self.assertIn('https://huggingface.co/black-forest-labs/FLUX.1-dev', card)
        self.assertIn('| `type` | `uint4` |', card)
        self.assertIn('| `group-size` | `32` |', card)
        self.assertIn('| `quant-conv` | `true` |', card)
        self.assertIn('| `use-svd` | `true` |', card)
        self.assertIn('| `quantized-matmul` | `false` |', card)
        self.assertIn('| `bits` | `4` |', card)
        self.assertIn('| `bits4-quant-type` | `nf4` |', card)
        self.assertIn('| `bits4-use-double-quant` | `true` |', card)
        self.assertIn('| `bits4-compute-dtype` | `default` |', card)
        self.assertIn('`Q4_K`', card)
        self.assertIn('transformer-Q4_K.gguf', card)
        self.assertIn('sdnq-uint4-g32-conv-svd/', card)
        self.assertIn('bnb-4-nf4-dq/', card)
        self.assertIn('--quantizer', card)
        self.assertIn('sdnq;type=uint4;group-size=32;quant-conv=true;use-svd=true', card)
        self.assertIn('--hf-upload me/flux-q4', card)
        self.assertIn('--revision main', card)
        self.assertNotIn('--auth-token', card)
        self.assertNotIn('hf_secret', card)
        self.assertNotIn('would be uploaded', card)
        self.assertNotIn('Nothing was quantized', card)

    def test_uploaded_card_lists_file_sizes(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, 'transformer-Q4_K.gguf')
            with open(path, 'wb') as handle:
                handle.write(b'x' * 2048)
            files = hfcard.listed_files(root, ['transformer-Q4_K.gguf', 'README.md'])
            card = hfcard.build_model_card(
                model='org/model',
                repo_id='me/model-q4',
                revision=None,
                subfolders=['transformer'],
                formats_requested=['q4'],
                formats=['Q4_K'],
                quantizers=[],
                comfy=True,
                llama=False,
                variant=None,
                weight_name=None,
                command='dgenerate --sub-command quantize org/model --formats q4 --comfy',
                files=files,
                preview=False)
        self.assertIn('library_name: gguf', card)
        self.assertIn('Tensor names follow the ComfyUI layout', card)
        self.assertIn('| `transformer-Q4_K.gguf` | 2.0 KB |', card)
        self.assertNotIn('would be uploaded', card)
        self.assertNotIn('Nothing was quantized', card)

    def test_one_repository_uploads_at_its_root(self):
        with tempfile.TemporaryDirectory() as output:
            folder = os.path.join(output, 'sdnq-int4')
            os.makedirs(os.path.join(folder, 'transformer'))
            with open(os.path.join(folder, 'model_index.json'), 'w', encoding='utf-8') as handle:
                handle.write('{}')
            with open(os.path.join(folder, 'transformer', 'config.json'), 'w', encoding='utf-8') as handle:
                handle.write('{}')
            root, patterns = hfcard.upload_layout(output, [folder])
            self.assertEqual(root, folder)
            self.assertIsNone(patterns)
            self.assertEqual(
                [path for path, _size in hfcard.listed_files(root, patterns)],
                ['model_index.json', 'transformer/config.json'])

    def test_repo_url_normalizes_to_an_id(self):
        self.assertEqual(
            hfcard.normalize_repo_id('https://huggingface.co/me/model-q4/tree/main'),
            'me/model-q4')
        with self.assertRaises(Exception):
            hfcard.normalize_repo_id('not a repo')

    def test_local_preview_reads_pipeline_and_component_class(self):
        with tempfile.TemporaryDirectory() as model:
            with open(os.path.join(model, 'model_index.json'), 'w', encoding='utf-8') as handle:
                json.dump({'_class_name': 'FluxPipeline'}, handle)
            folder = os.path.join(model, 'transformer')
            os.makedirs(folder)
            with open(os.path.join(folder, 'config.json'), 'w', encoding='utf-8') as handle:
                json.dump({'_class_name': 'FluxTransformer2DModel'}, handle)
            card = hfcard.build_model_card(
                model=model,
                repo_id=None,
                revision=None,
                subfolders=['transformer'],
                formats_requested=['q8'],
                formats=['Q8_0'],
                quantizers=[],
                comfy=False,
                llama=False,
                variant='fp16',
                weight_name=None,
                command='dgenerate --sub-command quantize ./model --formats q8',
                preview=True)
        self.assertIn('`FluxPipeline`', card)
        self.assertIn('`transformer` (`FluxTransformer2DModel`)', card)
        self.assertIn('`fp16`', card)
        self.assertNotIn('base_model:', card)


class TestQuantizeUploadCommand(unittest.TestCase):
    def test_dry_run_writes_the_readme_and_does_not_quantize(self):
        token = 'hf_dryrun_secret_token_value'
        with tempfile.TemporaryDirectory() as output:
            status = _command([
                'black-forest-labs/FLUX.1-dev',
                '--subfolders', 'transformer',
                '--formats', 'q4', 'fp8',
                '--quantizer', 'sdnq;type=uint4;group-size=32;use-svd=true',
                '--hf-upload', 'https://huggingface.co/me/flux-q4',
                '--hf-upload-card-preview',
                '--auth-token', token,
                '--revision', 'main',
                '-o', output,
            ])()
            self.assertEqual(status, 0)
            self.assertFalse(os.path.isfile(os.path.join(output, 'transformer-Q4_K.gguf')))
            with open(os.path.join(output, 'README.md'), encoding='utf-8') as handle:
                card = handle.read()
        self.assertNotIn(token, card)
        self.assertIn('me/flux-q4', card)
        self.assertIn('| `use-svd` | `true` |', card)
        self.assertIn('transformer-Q4_K.gguf', card)
        self.assertIn('transformer-float8_e4m3fn.safetensors', card)
        self.assertIn('sdnq-uint4-g32-svd/', card)
        self.assertNotIn('Nothing was quantized', card)
        self.assertNotIn('would be uploaded', card)

    def test_upload_requires_a_token_before_downloading(self):
        with patch('dgenerate.subcommands.quantize._hfhub.resolve_auth_token', return_value=None), \
                patch('dgenerate.subcommands.quantize.huggingface_hub.snapshot_download',
                      side_effect=AssertionError('downloaded')):
            with self.assertRaises(subcommand_exceptions.SubCommandArgumentError) as caught:
                _command([
                    'org/model', '--formats', 'q4', '-o', 'unused',
                    '--hf-upload', 'me/model-q4',
                ])()
        self.assertIn('--auth-token', str(caught.exception))

    def test_private_without_a_repository_is_rejected(self):
        with self.assertRaises(subcommand_exceptions.SubCommandArgumentError) as caught:
            _command([
                'org/model', '--formats', 'q4', '--hf-private', '--hf-upload-card-preview',
            ])()
        self.assertIn('--hf-private', str(caught.exception))

    def test_upload_sends_the_quant_files_and_the_card(self):
        captured = {}

        class FakeApi:
            def __init__(self, token=None):
                captured['token'] = token

            def create_repo(self, **kwargs):
                captured['create'] = kwargs

            def upload_folder(self, **kwargs):
                captured['upload'] = kwargs

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            folder = os.path.join(root, 'transformer')
            os.makedirs(folder)
            import torch
            import safetensors.torch
            weight = torch.randn(32, 256) * 0.05
            safetensors.torch.save_file(
                {'blocks.0.attn.to_q.weight': weight},
                os.path.join(folder, 'diffusion_pytorch_model.safetensors'))
            with open(os.path.join(folder, 'config.json'), 'w', encoding='utf-8') as handle:
                json.dump({'model_type': 'flux'}, handle)
            with open(os.path.join(root, 'model_index.json'), 'w', encoding='utf-8') as handle:
                json.dump({'_class_name': 'FluxPipeline'}, handle)
            with patch('dgenerate.quantize.hfcard.huggingface_hub.HfApi', FakeApi):
                status = _command([
                    root, '--subfolders', 'transformer', '--formats', 'q8',
                    '-o', output, '--hf-upload', 'me/tiny-q8',
                    '--hf-upload-card',
                    '--auth-token', 'hf_upload_secret_token_value',
                ])()
            self.assertEqual(status, 0)
            self.assertTrue(os.path.isfile(os.path.join(output, 'transformer-Q8_0.gguf')))
            with open(os.path.join(output, 'README.md'), encoding='utf-8') as handle:
                card = handle.read()
        self.assertNotIn('hf_upload_secret_token_value', card)
        self.assertIn('FluxPipeline', card)
        self.assertIn('transformer-Q8_0.gguf', card)
        self.assertIn('| `transformer-Q8_0.gguf` |', card)
        self.assertEqual(captured['token'], 'hf_upload_secret_token_value')
        self.assertEqual(captured['create']['repo_id'], 'me/tiny-q8')
        self.assertIsNone(captured['create']['private'])
        self.assertTrue(captured['create']['exist_ok'])
        self.assertEqual(captured['upload']['repo_id'], 'me/tiny-q8')
        self.assertEqual(captured['upload']['folder_path'], output)
        self.assertIn('README.md', captured['upload']['allow_patterns'])
        self.assertIn('transformer-Q8_0.gguf', captured['upload']['allow_patterns'])
        self.assertIn('Q8_0', captured['upload']['commit_message'])
        self.assertNotIn('hf_upload_secret_token_value', captured['upload']['commit_message'])
        self.assertIn('--hf-upload-card', card)
        self.assertNotIn('--hf-upload-card generate', card)

    def test_upload_without_a_card_writes_no_readme(self):
        captured = {}

        class FakeApi:
            def __init__(self, token=None):
                captured['token'] = token

            def create_repo(self, **kwargs):
                captured['create'] = kwargs

            def upload_folder(self, **kwargs):
                captured['upload'] = kwargs

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            folder = os.path.join(root, 'transformer')
            os.makedirs(folder)
            import torch
            import safetensors.torch
            safetensors.torch.save_file(
                {'blocks.0.attn.to_q.weight': torch.randn(32, 256)},
                os.path.join(folder, 'diffusion_pytorch_model.safetensors'))
            with open(os.path.join(folder, 'config.json'), 'w', encoding='utf-8') as handle:
                json.dump({'model_type': 'flux'}, handle)
            with patch('dgenerate.quantize.hfcard.huggingface_hub.HfApi', FakeApi):
                status = _command([
                    root, '--subfolders', 'transformer', '--formats', 'q8',
                    '-o', output, '--hf-upload', 'me/tiny-q8',
                    '--auth-token', 'hf_upload_secret_token_value',
                ])()
            self.assertEqual(status, 0)
            self.assertFalse(os.path.isfile(os.path.join(output, 'README.md')))
        self.assertNotIn('README.md', captured['upload']['allow_patterns'])

    def test_text_card_is_what_gets_uploaded(self):
        captured = {}

        class FakeApi:
            def __init__(self, token=None):
                pass

            def create_repo(self, **kwargs):
                pass

            def upload_folder(self, **kwargs):
                captured['upload'] = kwargs

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            folder = os.path.join(root, 'transformer')
            os.makedirs(folder)
            import torch
            import safetensors.torch
            safetensors.torch.save_file(
                {'blocks.0.attn.to_q.weight': torch.randn(32, 256)},
                os.path.join(folder, 'diffusion_pytorch_model.safetensors'))
            with open(os.path.join(folder, 'config.json'), 'w', encoding='utf-8') as handle:
                json.dump({'model_type': 'flux'}, handle)
            with patch('dgenerate.quantize.hfcard.huggingface_hub.HfApi', FakeApi):
                status = _command([
                    root, '--subfolders', 'transformer', '--formats', 'q8',
                    '-o', output, '--hf-upload', 'me/tiny-q8',
                    '--hf-upload-card', 'text:# Hello quant',
                    '--auth-token', 'hf_upload_secret_token_value',
                ])()
            self.assertEqual(status, 0)
            with open(os.path.join(output, 'README.md'), encoding='utf-8') as handle:
                card = handle.read()
        self.assertEqual(card, '# Hello quant\n')
        self.assertIn('README.md', captured['upload']['allow_patterns'])

    def test_file_card_preview_prints_the_file(self):
        with tempfile.TemporaryDirectory() as output, tempfile.TemporaryDirectory() as cards:
            path = os.path.join(cards, 'card.md')
            with open(path, 'w', encoding='utf-8', newline='\n') as handle:
                handle.write('# From a file\n')
            status = _command([
                'org/model', '--formats', 'q4',
                '--hf-upload-card', 'file:' + path,
                '--hf-upload-card-preview',
                '-o', output,
            ])()
            self.assertEqual(status, 0)
            self.assertFalse(os.path.isfile(os.path.join(output, 'transformer-Q4_K.gguf')))
            with open(os.path.join(output, 'README.md'), encoding='utf-8') as handle:
                self.assertEqual(handle.read(), '# From a file\n')

    def test_unknown_card_value_is_rejected(self):
        with self.assertRaises(subcommand_exceptions.SubCommandArgumentError) as caught:
            _command([
                'org/model', '--formats', 'q4', '--hf-upload-card', 'nope',
                '--hf-upload-card-preview',
            ])()
        self.assertIn('file:PATH', str(caught.exception))

    def test_extra_text_is_inserted_into_the_generated_card(self):
        with tempfile.TemporaryDirectory() as output:
            status = _command([
                'black-forest-labs/FLUX.1-dev',
                '--subfolders', 'transformer',
                '--formats', 'q4',
                '--hf-upload-card', 'extra:Use guidance 3.5.',
                '--hf-upload-card-preview',
                '-o', output,
            ])()
            self.assertEqual(status, 0)
            with open(os.path.join(output, 'README.md'), encoding='utf-8') as handle:
                card = handle.read()
        intro = card.index('The license of the source model applies to these weights.')
        note = card.index('Use guidance 3.5.')
        source = card.index('## Source')
        self.assertLess(intro, note)
        self.assertLess(note, source)
        self.assertIn('base_model: black-forest-labs/FLUX.1-dev', card)
        self.assertIn('## Quantization', card)
        self.assertIn('## Command', card)

    def test_extra_file_is_inserted_into_the_generated_card(self):
        with tempfile.TemporaryDirectory() as output, tempfile.TemporaryDirectory() as notes:
            path = os.path.join(notes, 'notes.md')
            with open(path, 'w', encoding='utf-8', newline='\n') as handle:
                handle.write('## Sampling\n\n4 steps, guidance 0.\n')
            status = _command([
                'org/model', '--formats', 'q8',
                '--hf-upload-card', 'extra-file:' + path,
                '--hf-upload-card-preview',
                '-o', output,
            ])()
            self.assertEqual(status, 0)
            with open(os.path.join(output, 'README.md'), encoding='utf-8') as handle:
                card = handle.read()
        self.assertIn('## Sampling\n\n4 steps, guidance 0.', card)
        self.assertLess(card.index('## Sampling'), card.index('## Source'))
        self.assertIn('| `Q8_0` | GGUF |', card)
