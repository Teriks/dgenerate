import os
import unittest

import dgenerate.promptupscalers.exceptions as _exceptions
import dgenerate.promptupscalers.util as _util
from dgenerate.promptupscalers import create_prompt_upscaler, prompt_upscaler_names
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


@unittest.skipUnless('llama' in prompt_upscaler_names(), 'xllamacpp is not installed')
class TestLlamaImageArgument(unittest.TestCase):
    def test_schema_accepts_image_and_mmproj(self):
        schema = PromptUpscalerLoader().get_accepted_args_schema(include_bases=True)
        for name in ('llama', 'xllamacpp'):
            self.assertTrue(schema[name]['image']['optional'])
            self.assertTrue(schema[name]['mmproj']['optional'])
            self.assertEqual(schema[name]['image']['files']['mode'], 'in')
            self.assertEqual(schema[name]['mmproj']['files']['mode'], 'in')
            self.assertFalse(schema[name]['thinking']['default'])
            self.assertEqual(schema[name]['thinking']['types'], ['bool'])

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


if __name__ == '__main__':
    unittest.main()
