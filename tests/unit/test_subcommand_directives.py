import contextlib
import io
import json
import pathlib
import unittest

import dgenerate.batchprocess as _batchprocess


_SCHEMA = pathlib.Path(__file__).parents[2] / 'dgenerate' / 'console' / 'schemas' / 'directives.json'
_NAMES = ('assistant', 'kernel_warmup', 'quantize', 'transpile')


class TestSubcommandDirectives(unittest.TestCase):
    def test_subcommands_have_shell_directives(self):
        runner = _batchprocess.ConfigRunner()
        schema = json.loads(_SCHEMA.read_text(encoding='utf-8'))
        for name in _NAMES:
            self.assertIn(name, runner.directives)
            self.assertEqual(schema['\\' + name], runner.generate_directives_help(
                [name], help_wrap_width=100))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runner.directives[name](['--help']), 0)


if __name__ == '__main__':
    unittest.main()
