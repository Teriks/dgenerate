import ast
import unittest

from dgenerate.transpile import transpile


class TestTranspileFormat(unittest.TestCase):

    def test_format_wraps_and_separates_without_changing_the_program(self):
        prompt = 'a ' + ('very detailed painting ' * 8)
        source = (
            'sd2-community/stable-diffusion-2-1\n'
            f'--prompts "{prompt}" "second {prompt}"\n'
            '\n'
            'sd2-community/stable-diffusion-2-1\n'
            '--prompts hi\n'
        )
        plain = transpile(source)
        formatted = transpile(source, format_output=True)
        self.assertIn('config.prompts = [', formatted)
        self.assertIn('\n    Prompt.parse(', formatted)
        self.assertIn('script.run(config)\n\nconfig:', formatted)
        self.assertEqual(
            ast.dump(ast.parse(plain)),
            ast.dump(ast.parse(formatted)),
        )

    def test_statements_are_grouped(self):
        source = '\n'.join([
            '#! dgenerate 6.0.0',
            '#! dgenerate 6.0.0',
            r'\set a 1',
            r'\setp b 2',
            r'\print {{ a }}',
            r'\exit',
            r'\echo hi',
            '{% if a == 1 %}',
            r'\print inside',
            '{% endif %}',
            '{% if a == 2 %}',
            r'\print other',
            '{% endif %}',
            '',
        ])
        output = transpile(source, relaxed=True)
        version = output.split("script.version_check(0, '#! dgenerate 6.0.0')", 1)[1]
        self.assertTrue(version.startswith("\nscript.version_check(1, '#! dgenerate 6.0.0')\n\n"))
        self.assertIn("script.directive('set', 'a', '1')\nscript.directive('setp', 'b', '2')\n\n", output)
        self.assertIn("script.directive('print', '{{ a }}')\nscript.directive('exit')\n\n", output)
        self.assertIn("script.directive('exit')\n\nscript.directive('echo', 'hi')\n\n", output)
        blocks = output.split('with script.block():')
        self.assertEqual(len(blocks), 3)
        self.assertIn('\n\nwith script.block():', output)
        self.assertNotIn('with script.block():\n\n', output)

    def test_a_comment_has_a_blank_line_above_and_below(self):
        source = '\n'.join([
            '#! dgenerate 6.0.0',
            '# before the set',
            '# still the set',
            r'\set a 1',
            '{% if a == 1 %}',
            r'\print inside',
            r'\exit',
            '{% endif %}',
            '# before the model',
            'sd2-community/stable-diffusion-2-1 --prompts hi',
            '',
        ])
        output = transpile(source, relaxed=True, comments=True)
        self.assertIn(
            "script.version_check(0, '#! dgenerate 6.0.0')\n\n"
            '# before the set\n'
            '# still the set\n\n',
            output,
        )
        self.assertIn("script.directive('exit')\n\n# before the model\n\n", output)
        self.assertIn('if (script.eager(\'a\') == 1):\n        script.directive(', output)

    def test_short_script_stays_one_assignment_per_line(self):
        source = 'sd2-community/stable-diffusion-2-1 --prompts hi\n'
        formatted = transpile(source, format_output=True)
        self.assertIn("config.prompts = [Prompt.parse('hi')]", formatted)


if __name__ == '__main__':
    unittest.main()
