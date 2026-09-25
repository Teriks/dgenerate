import logging
import threading
import unittest

import dgenerate  # noqa: F401  applies the patch
import dgenerate._patches.tqdm_huggingface_hub_patch as _patch
import huggingface_hub.file_download as _file_download
import huggingface_hub.utils as _hf_utils


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


if __name__ == '__main__':
    unittest.main()
