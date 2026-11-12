import json
import tempfile
import unittest
from pathlib import Path

import dgenerate.console.transpileform as transpileform
from dgenerate.console.transpileform import (
    load_transpile_preferences,
    save_transpile_preferences,
    transpile_command,
)


class TestTranspileCommand(unittest.TestCase):

    def test_strict_reads_stdin(self):
        self.assertEqual(
            transpile_command('dgenerate', False),
            ['dgenerate', '--sub-command', 'transpile', '--stdin'],
        )

    def test_relaxed_flag(self):
        self.assertEqual(
            transpile_command('dgenerate', True),
            ['dgenerate', '--sub-command', 'transpile', '--relaxed', '--stdin'],
        )

    def test_format_flag(self):
        self.assertEqual(
            transpile_command('dgenerate', False, True),
            ['dgenerate', '--sub-command', 'transpile', '--format', '--stdin'],
        )
        self.assertEqual(
            transpile_command('dgenerate', True, format_output=True),
            ['dgenerate', '--sub-command', 'transpile', '--relaxed', '--format', '--stdin'],
        )

    def test_comments_are_off_unless_requested(self):
        self.assertEqual(
            transpile_command('dgenerate', True, comments=False),
            ['dgenerate', '--sub-command', 'transpile', '--relaxed', '--stdin'],
        )
        self.assertEqual(
            transpile_command('dgenerate', True, comments=True),
            ['dgenerate', '--sub-command', 'transpile', '--relaxed', '--comments', '--stdin'],
        )

    def test_preferences_round_trip_without_dropping_other_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'console_settings.json'
            path.write_text(json.dumps({'theme': 'dgenerate'}), encoding='utf-8')
            original = transpileform.transpile_settings_path
            transpileform.transpile_settings_path = lambda: path
            try:
                self.assertEqual(
                    load_transpile_preferences(),
                    {'relaxed': False, 'format': False, 'comments': False},
                )
                save_transpile_preferences(True, True, True)
                self.assertEqual(
                    load_transpile_preferences(),
                    {'relaxed': True, 'format': True, 'comments': True},
                )
                self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['theme'], 'dgenerate')
            finally:
                transpileform.transpile_settings_path = original


if __name__ == '__main__':
    unittest.main()
