# Copyright (c) 2025 Guy Yariv
#
# Portions of this module adapt the DyPE frequency interpolation described in
# the upstream project. See LICENSE for the upstream MIT license and README.md
# for attribution and integration notes.

"""Optional, training-free dynamic RoPE scaling for selected DiT pipelines."""

from .patch import DypeError, dype_context

__all__ = ['DypeError', 'dype_context']
