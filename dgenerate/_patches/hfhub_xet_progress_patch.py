# Copyright (c) 2023, Teriks
#
# dgenerate is distributed under the following BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Collapse huggingface_hub Xet dual progress bars into one.

Xet shows separate transfer and reconstruction bars for standalone downloads and
for ``snapshot_download``. In the Console UI and many Windows/IDE terminals those
bars fight over the same lines and flash. A single bar driven by network transfer
stays readable, similar to how ``tqdm_huggingface_hub_patch`` aggregates threaded
http downloads.
"""

from __future__ import annotations

import importlib

import huggingface_hub._snapshot_download as _snapshot_download
from huggingface_hub.utils._xet_progress_reporting import (
    XET_BYTES_BAR_FORMAT,
    XetDownloadProgressReporter,
)
from huggingface_hub.utils.tqdm import tqdm as _hf_tqdm

_tqdm_utils = importlib.import_module('huggingface_hub.utils.tqdm')

_original_reporter_init = XetDownloadProgressReporter.__init__
_original_create_progress_bar = _tqdm_utils._create_progress_bar
_original_snapshot_update_transfer_bar = _snapshot_download._update_transfer_bar

_SNAPSHOT_TRANSFER_NAME = 'huggingface_hub.snapshot_download.transfer'
_SNAPSHOT_RECONSTRUCT_NAME = 'huggingface_hub.snapshot_download'

# Active display bar while snapshot_download builds its transfer/reconstruct pair.
_snapshot_display_bar = None


def _grow_transfer_total(bar, inc: int) -> None:
    n_after = getattr(bar, 'n', 0) + inc
    current_total = getattr(bar, 'total', 0) or 0
    if n_after > 0 and current_total < n_after:
        bar.total = max(current_total, int(n_after * 1.25) + 1)


def _fill_bar_to_total(bar) -> None:
    total = getattr(bar, 'total', None)
    n = getattr(bar, 'n', 0) or 0
    if total and n < total:
        bar.n = total
        bar.refresh()


class _SnapshotTransferProxy:
    """Receives snapshot transfer updates and drives the shared display bar."""

    _dgenerate_xet_snapshot_proxy = True

    def __init__(self, bar):
        self._bar = bar

    def update(self, n=1):
        inc = int(n or 0)
        _grow_transfer_total(self._bar, inc)
        return self._bar.update(inc)

    def refresh(self):
        return self._bar.refresh()

    def close(self):
        _fill_bar_to_total(self._bar)
        return self._bar.close()

    def set_description_str(self, desc):
        bar = self._bar
        # Cached snapshots never move bytes; upstream still finishes with
        # "Download complete" and leave=True, which sticks a useless
        # "0.00B / 0.00B" line in the Console UI.
        if not (bar.n or bar.total):
            bar.leave = False
            bar.close()
            return
        return bar.set_description_str(desc)

    def set_description(self, desc):
        return self._bar.set_description(desc)

    def set_postfix_str(self, s, refresh=False):
        return self._bar.set_postfix_str(s, refresh=refresh)

    @property
    def n(self):
        return self._bar.n

    @n.setter
    def n(self, value):
        self._bar.n = value

    @property
    def total(self):
        return self._bar.total

    @total.setter
    def total(self, value):
        # Reconstruct proxy accumulates file-size totals. Ignore transfer-side
        # accumulation from _AggregatedTqdm (it would double-count on a shared bar).
        # _finish_transfer_bar sets total=n to snap to 100% — fill instead.
        bar = self._bar
        if (value is not None and bar.total and value == bar.n and value < bar.total):
            _fill_bar_to_total(bar)

    @property
    def format_dict(self):
        return self._bar.format_dict

    def __getattr__(self, name):
        return getattr(self._bar, name)


class _SnapshotReconstructProxy:
    """Owns snapshot totals; ignores reconstruction byte updates (transfer drives)."""

    _dgenerate_xet_snapshot_proxy = True

    def __init__(self, bar):
        self._bar = bar

    def update(self, n=1):
        return True

    def refresh(self):
        return self._bar.refresh()

    def close(self):
        return None

    def set_description_str(self, desc):
        return None

    def set_description(self, desc):
        # Upstream finishes with "Reconstruction complete"; the transfer proxy
        # already sets "Download complete".
        return None

    def set_postfix_str(self, s, refresh=False):
        return self._bar.set_postfix_str(s, refresh=refresh)

    @property
    def n(self):
        return self._bar.n

    @n.setter
    def n(self, value):
        self._bar.n = value

    @property
    def total(self):
        return self._bar.total

    @total.setter
    def total(self, value):
        self._bar.total = value

    @property
    def format_dict(self):
        return self._bar.format_dict

    def __getattr__(self, name):
        return getattr(self._bar, name)


def _xet_single_bar_class(base):
    if getattr(base, '_dgenerate_xet_single_bar', False):
        return base

    class _XetSingleBar(base):
        _dgenerate_xet_single_bar = True

        def update_transfer(self, n=1):
            # Drive the single bar from network bytes. Do not call the shared
            # _update_transfer_bar helper: it ends in self.update(), which we
            # intentionally no-op for reconstruction increments.
            inc = int(n or 0)
            _grow_transfer_total(self, inc)
            return super().update(inc)

        def update(self, n=1):
            # Reconstruction bytes would double-count against transfer updates.
            return True

        def set_transfer_postfix_str(self, s, refresh=False):
            self.set_postfix_str(s, refresh=refresh)

        def close(self):
            _fill_bar_to_total(self)
            return super().close()

    _XetSingleBar._dgenerate_xet_single_bar = True
    return _XetSingleBar


def _patched_reporter_init(
        self,
        *,
        reconstruction_desc,
        transfer_desc='Downloading bytes',
        total=None,
        log_level,
        name=None,
        tqdm_class=None,
        external_reconstruction_bar=None,
        position=0):
    # snapshot_download / reused bars already aggregate into parent bars.
    if external_reconstruction_bar is not None or (
            tqdm_class is not None and callable(getattr(tqdm_class, 'update_transfer', None))):
        return _original_reporter_init(
            self,
            reconstruction_desc=reconstruction_desc,
            transfer_desc=transfer_desc,
            total=total,
            log_level=log_level,
            name=name,
            tqdm_class=tqdm_class,
            external_reconstruction_bar=external_reconstruction_bar,
            position=position,
        )

    desc = reconstruction_desc
    if desc.endswith(': reconstructing file'):
        desc = desc[:-len(': reconstructing file')]

    # Upstream places the reconstruction bar at position+1. Shift so the single
    # bar lands on the requested position instead of leaving a blank line above.
    return _original_reporter_init(
        self,
        reconstruction_desc=desc,
        transfer_desc=transfer_desc,
        total=total,
        log_level=log_level,
        name=name,
        tqdm_class=_xet_single_bar_class(tqdm_class or _hf_tqdm),
        external_reconstruction_bar=None,
        position=position - 1,
    )


def _patched_create_progress_bar(*, cls, log_level, name=None, **kwargs):
    """Collapse snapshot_download's transfer + reconstruction bars into one."""
    global _snapshot_display_bar

    if name == _SNAPSHOT_TRANSFER_NAME:
        kwargs = dict(kwargs)
        kwargs['desc'] = 'Downloading'
        kwargs['bar_format'] = XET_BYTES_BAR_FORMAT
        bar = _original_create_progress_bar(
            cls=cls, log_level=log_level, name=name, **kwargs)
        _snapshot_display_bar = bar
        return _SnapshotTransferProxy(bar)

    if name == _SNAPSHOT_RECONSTRUCT_NAME:
        bar = _snapshot_display_bar
        _snapshot_display_bar = None
        if bar is not None:
            return _SnapshotReconstructProxy(bar)

    return _original_create_progress_bar(
        cls=cls, log_level=log_level, name=name, **kwargs)


def _patched_snapshot_update_transfer_bar(bar, inc: int) -> None:
    # Avoid _update_transfer_bar writing transfer.total (duplicate of reconstruct).
    if getattr(bar, '_dgenerate_xet_snapshot_proxy', False):
        bar.update(inc)
        return
    return _original_snapshot_update_transfer_bar(bar, inc)


XetDownloadProgressReporter.__init__ = _patched_reporter_init
_tqdm_utils._create_progress_bar = _patched_create_progress_bar
_snapshot_download._create_progress_bar = _patched_create_progress_bar
_snapshot_download._update_transfer_bar = _patched_snapshot_update_transfer_bar
