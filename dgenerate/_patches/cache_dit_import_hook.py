# Copyright (c) 2023, Teriks
# BSD 3-Clause License

"""
Apply dgenerate's Cache-DiT patches as soon as ``cache_dit`` is imported.

Importing Cache-DiT takes several seconds, and most dgenerate runs never use it,
so it is not imported at startup. Instead, a meta path finder waits for the first
``import cache_dit`` and loads the patches right after the package finishes
executing, whoever imports it.
"""

import importlib.abc
import importlib.util
import sys


def _apply_patches():
    try:
        import dgenerate._patches.cache_dit_ltx2_patch
    except Exception:
        # Importing Cache-DiT must not fail because of a patch. cache_dit_context
        # imports the patch again when Cache-DiT is enabled, and reports the error.
        pass


class _CacheDitFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != 'cache_dit':
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None or not hasattr(spec.loader, 'exec_module'):
            return spec
        original_exec_module = spec.loader.exec_module

        def exec_module(module):
            original_exec_module(module)
            _apply_patches()

        spec.loader.exec_module = exec_module
        return spec


def _install():
    if 'cache_dit' in sys.modules:
        _apply_patches()
    elif not any(isinstance(finder, _CacheDitFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _CacheDitFinder())


_install()
