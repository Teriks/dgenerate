import logging
import os
import unittest

import dgenerate  # noqa: F401  applies patches
import huggingface_hub._snapshot_download as _snapshot_download
from huggingface_hub.utils._xet_progress_reporting import XetDownloadProgressReporter
from huggingface_hub.utils.tqdm import tqdm as hf_tqdm


class TestHfHubXetProgressPatch(unittest.TestCase):
    def test_standalone_xet_reporter_uses_one_bar(self):
        previous = os.environ.get('TQDM_POSITION')
        os.environ['TQDM_POSITION'] = '-1'
        try:
            self._run_single_bar_case()
        finally:
            if previous is None:
                os.environ.pop('TQDM_POSITION', None)
            else:
                os.environ['TQDM_POSITION'] = previous

    def _run_single_bar_case(self):
        with XetDownloadProgressReporter(
                reconstruction_desc='model.safetensors: reconstructing file',
                transfer_desc='model.safetensors: downloading bytes',
                total=1000,
                log_level=logging.WARNING,
                name='huggingface_hub.xet_get',
        ) as progress:
            self.assertIs(progress.transfer_bar, progress.reconstruction_bar)
            self.assertFalse(progress._owns_transfer_bar)
            self.assertTrue(getattr(progress.reconstruction_bar, '_dgenerate_xet_single_bar', False))

            class Report:
                total_bytes_completed = 100
                total_transfer_bytes_completed = 250
                total_bytes_completion_rate = 10.0
                total_transfer_bytes_completion_rate = 50.0
                total_bytes = 1000

            progress.update_progress(Report())
            # Transfer drives the bar; reconstruction increments are ignored.
            self.assertEqual(progress.reconstruction_bar.n, 250)
            self.assertEqual(progress.reconstruction_bar.total, 1000)
            bar = progress.reconstruction_bar

        # On close, fill to the known file size instead of flashing at a
        # compressed/deduped transfer total.
        self.assertEqual(bar.n, 1000)

    def test_snapshot_download_progress_bars_share_one_display(self):
        previous = os.environ.get('TQDM_POSITION')
        os.environ['TQDM_POSITION'] = '-1'
        try:
            self._run_snapshot_bar_case()
        finally:
            if previous is None:
                os.environ.pop('TQDM_POSITION', None)
            else:
                os.environ['TQDM_POSITION'] = previous

    def _run_snapshot_bar_case(self):
        transfer = _snapshot_download._create_progress_bar(
            cls=hf_tqdm,
            log_level=logging.WARNING,
            name='huggingface_hub.snapshot_download.transfer',
            desc='Downloading bytes',
            total=0,
            initial=0,
            unit='B',
            unit_scale=True,
        )
        reconstruct = _snapshot_download._create_progress_bar(
            cls=hf_tqdm,
            log_level=logging.WARNING,
            name='huggingface_hub.snapshot_download',
            desc='Reconstructing (incomplete total...)',
            total=0,
            initial=0,
            unit='B',
            unit_scale=True,
        )

        self.assertTrue(getattr(transfer, '_dgenerate_xet_snapshot_proxy', False))
        self.assertTrue(getattr(reconstruct, '_dgenerate_xet_snapshot_proxy', False))
        self.assertIs(transfer._bar, reconstruct._bar)
        self.assertEqual(transfer.desc, 'Downloading')

        # Mimic _AggregatedTqdm: both sides get += total, but only reconstruct
        # may accumulate on the shared bar.
        reconstruct.total = (reconstruct.total or 0) + 1000
        transfer.total = (transfer.total or 0) + 1000
        _snapshot_download._update_transfer_bar(transfer, 250)
        reconstruct.update(100)

        self.assertEqual(transfer._bar.n, 250)
        self.assertEqual(transfer._bar.total, 1000)

        # _finish_transfer_bar style snap should fill to file size, not shrink.
        transfer.total = transfer.n
        self.assertEqual(transfer._bar.n, 1000)
        self.assertEqual(transfer._bar.total, 1000)

        transfer.close()

    def test_snapshot_empty_download_complete_is_discarded(self):
        previous = os.environ.get('TQDM_POSITION')
        os.environ['TQDM_POSITION'] = '-1'
        try:
            transfer = _snapshot_download._create_progress_bar(
                cls=hf_tqdm,
                log_level=logging.WARNING,
                name='huggingface_hub.snapshot_download.transfer',
                desc='Downloading bytes',
                total=0,
                initial=0,
                unit='B',
                unit_scale=True,
            )
            _snapshot_download._create_progress_bar(
                cls=hf_tqdm,
                log_level=logging.WARNING,
                name='huggingface_hub.snapshot_download',
                desc='Reconstructing (incomplete total...)',
                total=0,
                initial=0,
                unit='B',
                unit_scale=True,
            )
            transfer.set_description_str('Download complete')
            self.assertFalse(transfer._bar.leave)
            self.assertTrue(transfer._bar.disable)
        finally:
            if previous is None:
                os.environ.pop('TQDM_POSITION', None)
            else:
                os.environ['TQDM_POSITION'] = previous


if __name__ == '__main__':
    unittest.main()
