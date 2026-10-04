import io
import logging
import os
import threading
import unittest

import dgenerate  # noqa: F401  applies the patch
import dgenerate._patches.tqdm_huggingface_hub_patch as _patch
import huggingface_hub.file_download as _file_download
import huggingface_hub.utils as _hf_utils
from huggingface_hub.utils.tqdm import _create_progress_bar, tqdm as hf_tqdm


def _bar(log_level=logging.INFO):
    return _file_download._get_progress_bar_context(
        desc='file.bin', log_level=log_level, total=10, name='huggingface_hub.http_get')


class TestTqdmHuggingfaceHubPatch(unittest.TestCase):
    def test_patch_is_installed(self):
        self.assertIs(_file_download._get_progress_bar_context, _patch._get_progress_bar_context)

    def test_disable_progress_bars_is_honored(self):
        _hf_utils.disable_progress_bars()
        try:
            with _bar() as bar:
                self.assertTrue(bar.disable)
        finally:
            _hf_utils.enable_progress_bars()

        with _bar() as bar:
            self.assertFalse(bar.disable)
            self.assertIsInstance(bar, _patch.MainThreadTqdm)

    def test_notset_log_level_disables(self):
        with _bar(logging.NOTSET) as bar:
            self.assertTrue(bar.disable)

    def test_thread_flag_from_diffusers_loading(self):
        import diffusers.pipelines.pipeline_utils as pipeline_utils

        scope = {'__name__': pipeline_utils.__name__, 'threading': threading}
        exec(compile(
            'def make():\n    return threading.Thread(target=lambda: None)\n',
            pipeline_utils.__file__, 'exec'), scope)

        self.assertTrue(scope['make']()._dgenerate_no_tqdm_thread)
        self.assertFalse(threading.Thread(target=lambda: None)._dgenerate_no_tqdm_thread)

    def test_force_tqdm_enables_without_nested_position(self):
        previous_force = os.environ.get('DGENERATE_FORCE_TQDM')
        previous_position = os.environ.get('TQDM_POSITION')
        os.environ['DGENERATE_FORCE_TQDM'] = '1'
        os.environ.pop('TQDM_POSITION', None)
        try:
            self.assertIs(_patch.is_tqdm_disabled(logging.INFO), False)

            buf = io.StringIO()
            bar = _create_progress_bar(
                cls=hf_tqdm,
                log_level=logging.WARNING,
                name='huggingface_hub.http_get',
                desc='model.safetensors.index.json',
                total=19900,
                unit='B',
                unit_scale=True,
                file=buf,
            )
            self.assertFalse(bar.disable)
            self.assertEqual(bar.pos, 0)
            bar.update(19900)
            bar.close()
            self.assertNotIn('\x1b[A', buf.getvalue())
        finally:
            if previous_force is None:
                os.environ.pop('DGENERATE_FORCE_TQDM', None)
            else:
                os.environ['DGENERATE_FORCE_TQDM'] = previous_force
            if previous_position is None:
                os.environ.pop('TQDM_POSITION', None)
            else:
                os.environ['TQDM_POSITION'] = previous_position


if __name__ == '__main__':
    unittest.main()
