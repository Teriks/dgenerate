Progress Bars
=============

Download, hash, upscale, and generation progress bars are drawn when stderr
is not a terminal. That includes the Console UI and output redirected to a
log. The Console UI output pane handles carriage return and cursor-movement
sequences, so the bars redraw in place.

``DGENERATE_FORCE_TQDM`` defaults to ``1``. ``\env DGENERATE_FORCE_TQDM=0``
draws a bar only when stderr is a terminal. Set it in the environment or in
``init.dgen``.

``\env DGENERATE_DISABLE_TQDM=1`` hides every tqdm bar: downloads, Xet,
hashing, upscaling, and generation. It overrides ``DGENERATE_FORCE_TQDM``.

``HF_HUB_DISABLE_PROGRESS_BARS=1`` hides Hugging Face download bars only.
