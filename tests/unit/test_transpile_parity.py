"""Transpiled scripts must match the config interpreter without generating images."""

import shutil
import tempfile
import unittest
from pathlib import Path

import dgenerate.renderloop as renderloop
import dgenerate.renderloopconfig as renderloopconfig

from tests.unit import transpile_parity as parity

# Every combination of the transpile switches. The library default is the
# first entry: strict, unformatted, comments kept.
MODES = [
    {'relaxed': relaxed, 'format_output': format_output, 'comments': comments}
    for relaxed in (False, True)
    for format_output in (False, True)
    for comments in (False, True)
]


def _mode_label(mode):
    return (
        f"{'relaxed' if mode['relaxed'] else 'strict'}, "
        f"{'formatted' if mode['format_output'] else 'plain'}, "
        f"{'comments' if mode['comments'] else 'no comments'}"
    )


class TestTranspileParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parity._write_png(parity.IMAGE)
        renderloopconfig.gen_seeds = parity._deterministic_seeds
        renderloop.gen_seeds = parity._deterministic_seeds
        cls.work = Path(tempfile.mkdtemp(prefix='dgen-transpile-parity-'))
        parity._prepare_dir(cls.work)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def test_adversarial_cases(self):
        for mode in MODES:
            for name, source in parity.iter_cases():
                with self.subTest(mode=_mode_label(mode), case=name):
                    difference = parity.compare(source, self.work, **mode)
                    self.assertIsNone(difference, difference)

    def test_example_configs(self):
        for mode in MODES:
            with self.subTest(mode=_mode_label(mode)):
                broken = parity.run_examples(echo=False, **mode)
                self.assertFalse(
                    broken,
                    '\n'.join(
                        f'{path}: {difference}' for path, difference in broken[:20]
                    ),
                )


if __name__ == '__main__':
    unittest.main()
