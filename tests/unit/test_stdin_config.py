import unittest

import dgenerate


class TestStdinConfigSkip(unittest.TestCase):

    def test_help_and_subcommands_skip_stdin_config(self):
        for args in (
            ['--help'],
            ['-h'],
            ['--version'],
            ['--image-processor-help'],
            ['--image-processor-help', 'canny'],
            ['--quantizer-help=bnb'],
            ['--sub-command', 'transpile'],
            ['--sub-command=transpile'],
            ['--templates-help'],
            ['--directives-help'],
            ['--functions-help'],
            ['--prompt-weighter-help'],
            ['--latents-processor-help'],
            ['--prompt-upscaler-help'],
            ['--sub-command-help'],
        ):
            with self.subTest(args=args):
                self.assertTrue(dgenerate._skips_stdin_config(args))

    def test_generation_and_shell_still_read_stdin(self):
        for args in (
            [],
            ['--shell'],
            ['--set', 'a=b'],
            ['--setp', 'a=1'],
            ['stabilityai/sd-turbo', '--prompts', 'hi'],
            ['--offline-mode'],
        ):
            with self.subTest(args=args):
                self.assertFalse(dgenerate._skips_stdin_config(args))


if __name__ == '__main__':
    unittest.main()
