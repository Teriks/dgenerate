# Copyright (c) 2023, Teriks
# BSD 3-Clause License

"""
Keep a one-step flow-match schedule finite when ``shift_terminal`` is set.

``FlowMatchEulerDiscreteScheduler.set_timesteps(1)`` builds a linspace of one
value, ``sigma_max``, which is 1. ``stretch_shift_to_terminal`` then divides
the gap ``1 - t`` by that same gap. The gap is 0, so the sigma is NaN. The
transformer denoises NaN latents and the AAC encoder rejects the NaN audio.
"""

import numpy

import diffusers


def _stretch_shift_to_terminal(self, t):
    one_minus_z = 1 - t
    terminal_gap = one_minus_z[-1]
    dest_gap = 1 - self.config.shift_terminal
    if dest_gap == 0:
        return t
    # A one-step schedule is only sigma_max. There is no interval to scale,
    # and 0/0 is NaN. That lone sigma is the terminal sample. A longer
    # schedule whose last sigma is already 1 is an overflow, and it has to
    # stay NaN so the caller can reject it.
    if float(terminal_gap) == 0.0 and len(t) == 1:
        value = float(self.config.shift_terminal)
        if hasattr(t, 'new_full'):
            return t.new_full(t.shape, value)
        return numpy.full_like(t, value)
    scale_factor = terminal_gap / dest_gap
    return 1 - (one_minus_z / scale_factor)


def _install():
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler
    if getattr(scheduler, '_dgenerate_terminal_stretch', False):
        return
    scheduler.stretch_shift_to_terminal = _stretch_shift_to_terminal
    scheduler._dgenerate_terminal_stretch = True


_install()
