"""The ANTLR parser rebuilds only when a grammar file changed."""

import os
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

from assetgen.builders.antlr_parser_builder import AntlrParserBuilder


def _tree(root: Path, grammar_newer: bool):
    grammar = root / 'dgenerate' / 'transpile' / 'grammar'
    grammar.mkdir(parents=True)
    now = time.time()
    for name in AntlrParserBuilder.GRAMMARS:
        path = grammar / name
        path.write_text('grammar\n', encoding='utf-8')
        os.utime(path, (now, now))
    later = now + (100 if grammar_newer else 10)
    earlier = now - 10
    stamp = earlier if grammar_newer else later
    for name in AntlrParserBuilder.OUTPUTS:
        path = grammar / name
        path.write_text('generated\n', encoding='utf-8')
        os.utime(path, (stamp, stamp))
    return grammar


class TestAntlrParserBuilder(unittest.TestCase):

    def test_up_to_date_parser_does_not_require_java(self):
        with tempfile.TemporaryDirectory() as tmp:
            _tree(Path(tmp), grammar_newer=False)
            builder = AntlrParserBuilder(tmp)
            with unittest.mock.patch(
                    'assetgen.builders.antlr_parser_builder.shutil.which',
                    return_value=None):
                builder.build()

    def test_changed_grammar_fails_without_java(self):
        with tempfile.TemporaryDirectory() as tmp:
            _tree(Path(tmp), grammar_newer=True)
            builder = AntlrParserBuilder(tmp)
            with unittest.mock.patch(
                    'assetgen.builders.antlr_parser_builder.shutil.which',
                    return_value=None):
                with self.assertRaises(SystemExit) as raised:
                    builder.build()
            self.assertIn('Java is not installed', str(raised.exception))

    def test_missing_parser_fails_without_java(self):
        with tempfile.TemporaryDirectory() as tmp:
            grammar = _tree(Path(tmp), grammar_newer=False)
            (grammar / 'Config.py').unlink()
            builder = AntlrParserBuilder(tmp)
            with unittest.mock.patch(
                    'assetgen.builders.antlr_parser_builder.shutil.which',
                    return_value=None):
                with self.assertRaises(SystemExit):
                    builder.build()


if __name__ == '__main__':
    unittest.main()
