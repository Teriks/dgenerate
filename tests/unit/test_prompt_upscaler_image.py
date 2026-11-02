import os
import tempfile
import unittest

import dgenerate.promptupscalers.exceptions as _exceptions
import dgenerate.promptupscalers.util as _util
from dgenerate.promptupscalers import create_prompt_upscaler, prompt_upscaler_names
from dgenerate.promptupscalers.llmupscalermixin import LLMPromptUpscalerMixin
from dgenerate.promptupscalers.promptupscalerloader import PromptUpscalerLoader

_MEDIA = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'examples', 'media', 'dog-on-bench.png'))


class TestPromptUpscalerImageHelpers(unittest.TestCase):
    def test_load_rgb_image_and_data_url(self):
        image = _util.load_rgb_image(_MEDIA)
        self.assertEqual(image.mode, 'RGB')
        url = _util.png_data_url(image)
        self.assertTrue(url.startswith('data:image/png;base64,'))
        self.assertGreater(len(url), 100)

    def test_missing_image_path(self):
        with self.assertRaises(ValueError) as ctx:
            _util.resolve_input_image('no-such-prompt-image.png', local_files_only=True)
        self.assertIn('image', str(ctx.exception))


class TestMagicPromptImageArgument(unittest.TestCase):
    def test_schema_accepts_image(self):
        schema = PromptUpscalerLoader().get_accepted_args_schema(include_bases=True)
        image = schema['magicprompt']['image']
        self.assertTrue(image['optional'])
        self.assertIsNone(image['default'])
        self.assertEqual(image['files']['mode'], 'in')

    def test_missing_image_is_an_argument_error(self):
        with self.assertRaises(_exceptions.PromptUpscalerArgumentError) as ctx:
            create_prompt_upscaler(
                'magicprompt;image=no-such-prompt-image.png',
                local_files_only=True,
            )
        self.assertIn('image', str(ctx.exception))

    def test_schema_accepts_gpu_headroom(self):
        schema = PromptUpscalerLoader().get_accepted_args_schema(include_bases=True)
        headroom = schema['magicprompt']['gpu-headroom']
        self.assertTrue(headroom['optional'])
        self.assertIsNone(headroom['default'])

    def test_schema_accepts_presence_penalty(self):
        schema = PromptUpscalerLoader().get_accepted_args_schema(include_bases=True)
        penalty = schema['magicprompt']['presence-penalty']
        self.assertEqual(penalty['default'], 0.0)
        self.assertIn('float', penalty['types'])

    def test_negative_gpu_headroom_is_an_argument_error(self):
        with self.assertRaises(_exceptions.PromptUpscalerArgumentError) as ctx:
            create_prompt_upscaler(
                'magicprompt;gpu-headroom=-1',
                local_files_only=True,
            )
        self.assertIn('gpu-headroom', str(ctx.exception))

    def test_system_file_argument_is_read_before_the_model_loads(self):
        with self.assertRaises(_exceptions.PromptUpscalerArgumentError) as ctx:
            create_prompt_upscaler(
                'magicprompt;system=file:no-such-system-prompt.txt',
                local_files_only=True,
            )
        message = str(ctx.exception)
        self.assertIn('system', message)
        self.assertIn('no-such-system-prompt.txt', message)


class _FileArgumentStub:
    def argument_error(self, message):
        return _exceptions.PromptUpscalerArgumentError(message)


class TestLLMFileArguments(unittest.TestCase):
    def test_file_argument_reads_utf8_text(self):
        handle = tempfile.NamedTemporaryFile(
            'w', encoding='utf-8', delete=False, suffix=' prompt.txt')
        handle.write('  be specific \n')
        handle.close()
        try:
            loaded = LLMPromptUpscalerMixin._resolve_file_argument(
                _FileArgumentStub(), f'file:{handle.name}', 'preamble')
        finally:
            os.remove(handle.name)
        self.assertEqual(loaded, 'be specific')

    def test_plain_text_is_left_unchanged(self):
        stub = _FileArgumentStub()
        self.assertEqual(
            LLMPromptUpscalerMixin._resolve_file_argument(stub, 'stay literal', 'system'),
            'stay literal')
        self.assertIsNone(
            LLMPromptUpscalerMixin._resolve_file_argument(stub, None, 'system'))

    def test_file_argument_requires_a_path_and_text(self):
        stub = _FileArgumentStub()
        with self.assertRaises(_exceptions.PromptUpscalerArgumentError) as missing:
            LLMPromptUpscalerMixin._resolve_file_argument(stub, 'file:', 'system')
        self.assertIn('system', str(missing.exception))

        handle = tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False)
        handle.close()
        try:
            with self.assertRaises(_exceptions.PromptUpscalerArgumentError) as empty:
                LLMPromptUpscalerMixin._resolve_file_argument(
                    stub, f'file:{handle.name}', 'preamble')
        finally:
            os.remove(handle.name)
        self.assertIn('preamble', str(empty.exception))
        self.assertIn('empty', str(empty.exception))


@unittest.skipUnless('llama' in prompt_upscaler_names(), 'xllamacpp is not installed')
class TestLlamaImageArgument(unittest.TestCase):
    def test_schema_accepts_image_and_mmproj(self):
        schema = PromptUpscalerLoader().get_accepted_args_schema(include_bases=True)
        self.assertNotIn('xllamacpp', schema)
        image = schema['llama']
        self.assertTrue(image['image']['optional'])
        self.assertTrue(image['mmproj']['optional'])
        self.assertEqual(image['image']['files']['mode'], 'in')
        self.assertEqual(image['mmproj']['files']['mode'], 'in')
        self.assertFalse(image['thinking']['default'])
        self.assertEqual(image['thinking']['types'], ['bool'])
        self.assertEqual(image['presence-penalty']['default'], 0.0)

    def test_image_requires_mmproj(self):
        with self.assertRaises(_exceptions.PromptUpscalerArgumentError) as ctx:
            create_prompt_upscaler('llama;image=no-such-prompt-image.png', local_files_only=True)
        self.assertIn('mmproj', str(ctx.exception))

    def test_mmproj_requires_image(self):
        with self.assertRaises(_exceptions.PromptUpscalerArgumentError) as ctx:
            create_prompt_upscaler('llama;mmproj=no-such-projector.gguf', local_files_only=True)
        self.assertIn('image', str(ctx.exception))

    def test_missing_image_file_is_reported_before_the_model_downloads(self):
        with self.assertRaises(_exceptions.PromptUpscalerArgumentError) as ctx:
            create_prompt_upscaler(
                'llama;image=no-such-prompt-image.png;mmproj=no-such-projector.gguf',
                local_files_only=True,
            )
        self.assertIn('image', str(ctx.exception))
        self.assertNotIn('model', str(ctx.exception))

    def test_missing_mmproj_file(self):
        image = _MEDIA.replace('\\', '/')
        with self.assertRaises(_exceptions.PromptUpscalerArgumentError) as ctx:
            create_prompt_upscaler(
                f'llama;image={image};mmproj=no-such-projector.gguf',
                local_files_only=True,
            )
        self.assertIn('mmproj', str(ctx.exception))

    def test_chat_request_sends_presence_penalty(self):
        from dgenerate.promptupscalers.xllamacpppromptupscaler import XllamaCppPromptUpscaler
        upscaler = XllamaCppPromptUpscaler.__new__(XllamaCppPromptUpscaler)
        upscaler._max_length = 12
        upscaler._temperature = 0.7
        upscaler._top_k = 40
        upscaler._top_p = 0.4
        upscaler._min_p = 0.0
        upscaler._presence_penalty = 1.25
        upscaler._thinking = False
        seen = {}

        class Server:
            def handle_chat_completions(self, body):
                seen.update(body)
                return {'choices': [{'message': {'content': 'a red fox'}}]}

        upscaler._server = Server()
        self.assertEqual(upscaler._complete([{'role': 'user', 'content': 'fox'}]), 'a red fox')
        self.assertEqual(seen['presence_penalty'], 1.25)


if __name__ == '__main__':
    unittest.main()
