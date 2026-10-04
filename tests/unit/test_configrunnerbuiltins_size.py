import os
import tempfile
import unittest

import PIL.Image

import dgenerate.batchprocess.batchprocessor as _batchprocessor
import dgenerate.batchprocess.configrunnerbuiltins as _builtins


class TestConfigRunnerBuiltinsSize(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.image_path = os.path.join(self._temp_dir.name, 'sample.png')
        PIL.Image.new('RGB', (320, 240), color=(255, 0, 0)).save(self.image_path)

    def test_image_width_height(self):
        self.assertEqual(_builtins.image_width(self.image_path), 320)
        self.assertEqual(_builtins.image_height(self.image_path), 240)

    def test_scale_size_from_string(self):
        self.assertEqual(_builtins.scale_size('512x768', 2), '1024x1536')
        self.assertEqual(_builtins.scale_size('512x768', 1.5), '768x1152')
        self.assertEqual(
            _builtins.scale_size('512x768', 2, format_size=False),
            (1024, 1536))

    def test_scale_size_from_tuple(self):
        self.assertEqual(_builtins.scale_size((100, 50), 2), '200x100')
        self.assertEqual(
            _builtins.scale_size((100, 50), 0.5, format_size=False),
            (50, 25))

    def test_scale_size_from_image_file(self):
        self.assertEqual(_builtins.scale_size(self.image_path, 2), '640x480')
        self.assertEqual(
            _builtins.scale_size(self.image_path, 2, format_size=False),
            (640, 480))

    def test_scale_size_clamps_to_one(self):
        self.assertEqual(_builtins.scale_size('10x10', 0.01), '1x1')

    def test_scale_size_independent_axes(self):
        self.assertEqual(_builtins.scale_size('512x768', (2, 1)), '1024x768')
        self.assertEqual(_builtins.scale_size('512x768', [2, 1.5]), '1024x1152')
        self.assertEqual(_builtins.scale_size('512x768', '2x1.5'), '1024x1152')
        self.assertEqual(
            _builtins.scale_size(self.image_path, (2, 0.5), format_size=False),
            (640, 120))

    def test_scale_size_rejects_bad_scale(self):
        with self.assertRaises(_batchprocessor.BatchProcessError):
            _builtins.scale_size('512x512', 'nope')
        with self.assertRaises(_batchprocessor.BatchProcessError):
            _builtins.scale_size('512x512', (2, 1, 1))
        with self.assertRaises(_batchprocessor.BatchProcessError):
            _builtins.scale_size('512x512', True)

    def test_configrunner_registers_functions(self):
        from dgenerate.batchprocess import ConfigRunner
        runner = ConfigRunner()
        self.assertIs(runner.template_functions['image_width'], _builtins.image_width)
        self.assertIs(runner.template_functions['image_height'], _builtins.image_height)
        self.assertIs(runner.template_functions['scale_size'], _builtins.scale_size)


if __name__ == '__main__':
    unittest.main()
