import io
import os
import unittest

import dgenerate  # noqa: F401  applies the patch
import tqdm
import tqdm.auto
import tqdm.std


def _pipe():
    buf = io.StringIO()
    buf.isatty = lambda: False
    return buf


class TestTqdmPatch(unittest.TestCase):
    def setUp(self):
        self._previous_force = os.environ.get('DGENERATE_FORCE_TQDM')
        self._previous_disable = os.environ.get('DGENERATE_DISABLE_TQDM')
        os.environ.pop('DGENERATE_FORCE_TQDM', None)
        os.environ.pop('DGENERATE_DISABLE_TQDM', None)

    def tearDown(self):
        if self._previous_force is None:
            os.environ.pop('DGENERATE_FORCE_TQDM', None)
        else:
            os.environ['DGENERATE_FORCE_TQDM'] = self._previous_force
        if self._previous_disable is None:
            os.environ.pop('DGENERATE_DISABLE_TQDM', None)
        else:
            os.environ['DGENERATE_DISABLE_TQDM'] = self._previous_disable

    def test_plain_and_auto_bars_default_on_without_a_terminal(self):
        for cls in (tqdm.tqdm, tqdm.auto.tqdm, tqdm.std.tqdm):
            bar = cls(total=4, file=_pipe(), desc='hash')
            self.assertFalse(bar.disable, cls)
            bar.update(4)
            self.assertEqual(bar.n, 4)
            bar.close()

        os.environ['DGENERATE_FORCE_TQDM'] = '0'
        for cls in (tqdm.tqdm, tqdm.auto.tqdm, tqdm.std.tqdm):
            bar = cls(total=4, file=_pipe(), desc='hash')
            self.assertTrue(bar.disable, cls)
            bar.close()

    def test_explicit_disable_wins_over_force(self):
        os.environ['DGENERATE_FORCE_TQDM'] = '1'
        bar = tqdm.tqdm(total=1, file=_pipe(), disable=True)
        self.assertTrue(bar.disable)
        bar.close()

    def test_disable_tqdm_hides_every_bar(self):
        from huggingface_hub.utils.tqdm import tqdm as hf_tqdm

        class _Tty(io.StringIO):
            def isatty(self):
                return True

        os.environ['DGENERATE_FORCE_TQDM'] = '1'
        os.environ['DGENERATE_DISABLE_TQDM'] = '1'

        class _Widget(tqdm.std.tqdm):
            pass

        _Widget.__module__ = 'tqdm.notebook'
        for cls in (tqdm.tqdm, tqdm.auto.tqdm, tqdm.std.tqdm, hf_tqdm, _Widget):
            bar = cls(total=4, file=_Tty(), desc='step')
            self.assertTrue(bar.disable, cls)
            bar.close()

    def test_notebook_widgets_are_not_coerced(self):
        class _Widget(tqdm.std.tqdm):
            pass

        _Widget.__module__ = 'tqdm.notebook'
        bar = _Widget(total=1, file=_pipe(), disable=False)
        self.assertFalse(bar.disable)
        bar.close()


if __name__ == '__main__':
    unittest.main()
