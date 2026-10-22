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

"""Schedule checks for the earlier LTX-Video pipeline."""

from __future__ import annotations

import warnings

import numpy
import torch

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.pipelines as _pipelines


def _legacy_ltx_schedule_finite(scheduler, width, height, num_frames, steps,
                                spatial=32, temporal=8) -> bool:
    """
    The earlier LTX pipeline shifts timesteps from the latent token count.
    Past a point that shift makes the terminal sigma 1 and the schedule NaN,
    which later crashes the sampler with an empty timestep index.
    """
    return bool(torch.isfinite(
        _legacy_ltx_timesteps(scheduler, width, height, num_frames, steps, spatial, temporal)).all())


def _legacy_ltx_timesteps(scheduler, width, height, num_frames, steps,
                          spatial=32, temporal=8) -> torch.Tensor:
    """
    The timesteps the earlier LTX text and image pipelines build for this clip.
    """
    cfg = scheduler.config
    latent_frames = (int(num_frames) - 1) // int(temporal) + 1
    seq = latent_frames * (int(height) // int(spatial)) * (int(width) // int(spatial))
    base_seq = cfg.get('base_image_seq_len', 256)
    max_seq = cfg.get('max_image_seq_len', 4096)
    base_shift = cfg.get('base_shift', 0.5)
    max_shift = cfg.get('max_shift', 1.15)
    slope = (max_shift - base_shift) / (max_seq - base_seq)
    mu = seq * slope + (base_shift - slope * base_seq)
    probe = scheduler.__class__.from_config(cfg)
    sigmas = numpy.linspace(1.0, 1.0 / int(steps), int(steps))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        probe.set_timesteps(int(steps), device='cpu', sigmas=sigmas, mu=float(mu))
    return probe.timesteps


def _legacy_ltx_compression(pipe) -> tuple[int, int]:
    spatial = int(getattr(pipe, 'vae_spatial_compression_ratio', 32) or 32)
    temporal = int(getattr(pipe, 'vae_temporal_compression_ratio', 8) or 8)
    return spatial, temporal


def _fix_legacy_ltx_condition_schedule(pipe, width, height, num_frames, steps) -> list[float] | None:
    """
    ``LTXConditionPipeline`` sets its own timesteps and never passes ``mu``, so a
    scheduler with dynamic shifting (the LTX-Video 0.9.0 / 0.9.1 checkpoints) raises.

    Resolve the checkpoint's shifted schedule here, then give ``pipe`` a copy of the
    scheduler that applies no further shift, so the timesteps are used as they are.
    ``pipe`` must be the per-call condition pipeline, not the cached one.

    :return: timesteps to pass, or ``None`` when the scheduler needs no change
    """
    scheduler = pipe.scheduler
    if not scheduler.config.get('use_dynamic_shifting', False):
        return None
    timesteps = _legacy_ltx_timesteps(
        scheduler, width, height, num_frames, steps, *_legacy_ltx_compression(pipe))
    pipe.scheduler = scheduler.__class__.from_config({
        **scheduler.config,
        'use_dynamic_shifting': False,
        'shift': 1.0,
        'shift_terminal': None,
        'use_karras_sigmas': False,
        'use_exponential_sigmas': False,
        'use_beta_sigmas': False,
    })
    return [float(value) for value in timesteps]


def _require_legacy_ltx_schedule(pipe, width, height, num_frames, steps):
    if _legacy_ltx_schedule_finite(
            pipe.scheduler, width, height, num_frames, steps, *_legacy_ltx_compression(pipe)):
        return
    raise _pipelines.UnsupportedPipelineConfigError(
        f'LTX-Video cannot build a noise schedule for {width}x{height} and {num_frames} frames. '
        f'The resolution-dependent timestep shift overflows at this size. '
        f'Use a smaller output size or a shorter clip.')


def _warn_legacy_ltx_canvas(width, height, num_frames, fps):
    short = num_frames is not None and int(num_frames) < 121
    small = (
        width is not None and height is not None
        and (int(width) < 704 or int(height) < 480)
    )
    if not short and not small:
        return
    _messages.warning(
        'LTX-Video follows the prompt at about 768x512, 25 fps, and 121 frames. '
        f'This clip is {width}x{height}, {num_frames} frames, {fps} fps.')
