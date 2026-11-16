# Copyright (c) 2023, Teriks
# BSD 3-Clause License

"""
Show tqdm bars when stderr is not a terminal.

``tqdm`` draws a bar only for ``disable=False``. ``disable=None`` means
"draw only if the output file is a terminal". dgenerate's own bars and
dependency bars (HTTP downloads, Xet, diffusers, transformers) all construct
that class. ``DGENERATE_FORCE_TQDM`` defaults to ``1``, which keeps
``disable=False`` so pipes still show bars. Set it to ``0`` to use the
terminal check.

``DGENERATE_DISABLE_TQDM=1`` sets ``disable=True`` on every bar, including
notebook widgets, and it overrides ``DGENERATE_FORCE_TQDM``. An explicit
``disable=True`` is also left alone. ``HF_HUB_DISABLE_PROGRESS_BARS``,
``disable_progress_bars()``, a ``NOTSET`` log level, and dgenerate's worker
thread bars use that.

Notebook widgets are not terminal bars. Unless ``DGENERATE_DISABLE_TQDM=1``,
instances whose class comes from ``tqdm.notebook`` keep tqdm's own behavior.
"""

from __future__ import annotations

import os

import tqdm.std as _tqdm_std

_original_init = _tqdm_std.tqdm.__init__


def _tqdm_disabled() -> bool:
    return os.environ.get('DGENERATE_DISABLE_TQDM') == '1'


def _coerce_disable(disable):
    if _tqdm_disabled() or disable is True:
        return True
    # Unset defaults to on. ``0`` restores tqdm's terminal check.
    if os.environ.get('DGENERATE_FORCE_TQDM', '1') == '1':
        return False
    return None


def _is_notebook_bar(bar) -> bool:
    return any(cls.__module__ == 'tqdm.notebook' for cls in type(bar).__mro__)


def _patched_init(self, *args, **kwargs):
    # disable is the 11th parameter of tqdm.tqdm.__init__, after self.
    # A global disable covers notebook widgets too.
    if _tqdm_disabled() or not _is_notebook_bar(self):
        if len(args) > 10:
            args = list(args)
            args[10] = _coerce_disable(args[10])
            args = tuple(args)
        else:
            kwargs['disable'] = _coerce_disable(kwargs.get('disable', False))
    return _original_init(self, *args, **kwargs)


_tqdm_std.tqdm.__init__ = _patched_init
