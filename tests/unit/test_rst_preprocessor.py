import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from assetgen.core.rst_preprocessor import RSTPreprocessor


class TestCommandCache(unittest.TestCase):
    def test_bypassing_the_cache_rewrites_the_entry_and_keeps_the_rest(self):
        directory = tempfile.mkdtemp()
        path = Path(directory) / 'command.cache.json'
        path.write_text(json.dumps({
            'dgenerate --help': 'old help',
            'other': 'leave this',
        }), encoding='utf-8')

        preprocessor = RSTPreprocessor(command_cache_path=path, verbose=False, use_cache=False)
        completed = unittest.mock.Mock(stdout='new help\x1b[0m')
        with unittest.mock.patch('assetgen.core.rst_preprocessor.subprocess.run', return_value=completed) as run:
            text = preprocessor._execute_command(Path(directory), 'dgenerate --help')
        self.assertEqual(text, 'new help')
        run.assert_called_once()
        preprocessor._save_cache()

        saved = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(saved['dgenerate --help'], 'new help')
        self.assertEqual(saved['other'], 'leave this')

    def test_a_cache_hit_does_not_run_the_command(self):
        directory = tempfile.mkdtemp()
        path = Path(directory) / 'command.cache.json'
        path.write_text(json.dumps({'dgenerate --help': 'cached help'}), encoding='utf-8')

        preprocessor = RSTPreprocessor(command_cache_path=path, verbose=False)
        with unittest.mock.patch('assetgen.core.rst_preprocessor.subprocess.run') as run:
            text = preprocessor._execute_command(Path(directory), 'dgenerate --help')
        self.assertEqual(text, 'cached help')
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
