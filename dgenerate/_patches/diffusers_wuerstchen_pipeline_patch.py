# Copyright (c) 2023, Teriks
# BSD 3-Clause License

"""
Restore ``diffusers.pipelines.wuerstchen`` after it moved to ``pipelines.deprecated``.

Stable Cascade's ``model_index.json`` still names the decoder VQGAN as library
``wuerstchen`` (``PaellaVQModel``). Current diffusers only exposes that package
under ``diffusers.pipelines.deprecated.wuerstchen``, so ``from_pretrained``
treats ``wuerstchen`` as a missing custom module and refuses to load
``stabilityai/stable-cascade``.
"""

import sys

import diffusers.pipelines as _pipelines
import diffusers.pipelines.deprecated.wuerstchen as _wuerstchen


def _alias_wuerstchen_pipeline_module():
    if hasattr(_pipelines, "wuerstchen"):
        return
    _pipelines.wuerstchen = _wuerstchen
    sys.modules.setdefault("diffusers.pipelines.wuerstchen", _wuerstchen)


_alias_wuerstchen_pipeline_module()
