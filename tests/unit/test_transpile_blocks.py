import unittest

from dgenerate.transpile import ParseError, transpile


class TestTranspileBlocks(unittest.TestCase):

    def test_set_is_not_stepped(self):
        output = transpile('{% set x = 5 %}\n\\print {{ x }}\n', relaxed=True)
        self.assertNotIn('script.step', output)
        self.assertIn('\n5\n', output)
        self.assertIn("script.directive('print', '{{ x }}')", output)

    def test_filter_renders_then_runs(self):
        output = transpile('{% filter upper %}hi{% endfilter %}\n\\print after\n', relaxed=True)
        self.assertNotIn('script.step', output)
        self.assertIn("script.run_rendered(script.filter('upper', 'hi'))", output)
        self.assertIn("script.directive('print', 'after')", output)

    def test_with_renders_then_runs(self):
        output = transpile('{% with x = 1 %}{{ x }}{% endwith %}\n', relaxed=True)
        self.assertNotIn('script.step', output)
        self.assertIn('def _render_1():', output)
        self.assertIn('x = 1', output)
        self.assertIn('script.run_rendered(str(x))', output)

    def test_tuple_for_is_a_python_loop(self):
        source = (
            "{% for a, b in [('x', 1), ('y', 2)] %}\n"
            '\\print {{ a }}={{ b }}\n'
            '{% endfor %}\n'
        )
        output = transpile(source, relaxed=True)
        self.assertNotIn('script.step', output)
        self.assertIn("('x', 1)", output)
        self.assertIn('for _loop0, (a, b) in enumerate(_items0):', output)

    def test_parentheses_still_group(self):
        output = transpile('\\print {{ (1 + 2) }}\n', relaxed=True)
        self.assertIn('(1 + 2)', output)
        self.assertNotIn('(1 + 2,)', output)

    def test_broken_for_is_a_syntax_error(self):
        source = '{% for in in range(0, 100) %}\n\\print {{ i }}\n{% endfor %}\n'
        with self.assertRaises(ParseError) as caught:
            transpile(source, relaxed=True)
        message = str(caught.exception)
        self.assertIn('Template syntax error on line 1', message)
        self.assertNotIn('script.step', message)

    def test_valid_for_still_lowers(self):
        source = '{% for i in range(0, 100) %}\n\\print {{ i }}\n{% endfor %}\n'
        output = transpile(source, relaxed=True)
        self.assertIn('for _loop0, i in enumerate', output)
        self.assertNotIn('script.step', output)

    def test_scanner_syntax_error_stops_translation(self):
        with self.assertRaises(ParseError) as caught:
            transpile('{% raw %}\n\\print inside\n')
        message = str(caught.exception)
        self.assertIn('Template Syntax Error', message)
        self.assertNotIn('BatchProcessError', message)

    def test_uncompiled_tag_still_steps(self):
        output = transpile('{% macro name() %}hi{% endmacro %}\n', relaxed=True)
        self.assertIn('script.step', output)
        self.assertIn('{% macro name() %}', output)
