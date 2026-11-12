import unittest

from dgenerate.transpile import transpile


class TestTranspileGlobs(unittest.TestCase):

    def test_unquoted_glob_is_one_runtime_argument(self):
        source = (
            'sd2-community/stable-diffusion-2-1\n'
            '--variant fp16\n'
            '--prompts "In the style of picaso"\n'
            '--image-seeds ../media/*.png kept.png\n'
            '--output-path ./output\n'
        )
        output = transpile(source)
        self.assertNotIn('script.step', output)
        self.assertIn("*script.shell('../media/*.png')", output)
        self.assertIn("'kept.png'", output)
        self.assertIn("Prompt.parse('In the style of picaso')", output)
        self.assertNotIn("typed_list('--prompts'", output)
        self.assertNotIn('glob.glob', output)

    def test_quoted_star_stays_literal(self):
        source = (
            'sd2-community/stable-diffusion-2-1\n'
            '--prompts "a * b"\n'
        )
        output = transpile(source)
        self.assertNotIn('script.shell', output)
        self.assertNotIn('script.step', output)
        self.assertIn("a * b", output)

    def test_env_home_and_percent_are_one_argument(self):
        source = (
            'sd2-community/stable-diffusion-2-1\n'
            '--image-seeds ~/$DIR/%NAME%/*.png\n'
            '--prompts "$CAPTION"\n'
            "--output-path '~/literal'\n"
        )
        output = transpile(source)
        self.assertNotIn('script.step', output)
        self.assertIn("*script.shell('~/$DIR/%NAME%/*.png')", output)
        self.assertIn("*script.typed_list('--prompts', script.shell('\"$CAPTION\"'))", output)
        self.assertIn("~/literal", output)
        self.assertNotIn("script.shell('~/literal')", output)

    def test_glob_inside_loop_does_not_shell_the_invocation(self):
        source = (
            '{% for n in [1] %}\n'
            'sd2-community/stable-diffusion-2-1\n'
            '--image-seeds imgs/*.png\n'
            '--prompts hi\n'
            '{% endfor %}\n'
        )
        output = transpile(source)
        self.assertNotIn('script.step', output)
        self.assertNotIn('script.run_argv', output)
        self.assertIn("*script.shell('imgs/*.png')", output)
        self.assertIn("Prompt.parse('hi')", output)
        self.assertNotIn("typed_list('--prompts'", output)

    def test_static_prompt_stays_parsed_beside_a_shell_prompt(self):
        source = (
            'sd2-community/stable-diffusion-2-1\n'
            '--prompts hi "$CAPTION"\n'
            '--gen-seeds 1\n'
            '--image-seeds imgs/*.png\n'
        )
        output = transpile(source)
        self.assertIn("Prompt.parse('hi')", output)
        self.assertIn("*script.typed_list('--prompts', script.shell('\"$CAPTION\"'))", output)
        self.assertIn("script.typed('--gen-seeds', '1')", output)
        self.assertNotIn('script.step', output)


if __name__ == '__main__':
    unittest.main()
