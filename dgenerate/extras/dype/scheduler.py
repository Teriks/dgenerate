# Copyright (c) 2025, Teriks
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

"""Resolution-aware bounds for model flow-scheduler timestep shifts."""

import contextlib

import diffusers

import dgenerate.messages as _messages


@contextlib.contextmanager
def resolution_shift_context(
        pipeline: diffusers.DiffusionPipeline,
        *,
        height: int | None,
        width: int | None):
    """Match DyPE's Flux pipeline by using the current image length as the max length."""
    scheduler = getattr(pipeline, 'scheduler', None)
    config = getattr(scheduler, 'config', None)
    max_seq_len = config.get('max_image_seq_len') if config is not None else None
    vae_scale_factor = getattr(pipeline, 'vae_scale_factor', None)
    if (not isinstance(height, int) or not isinstance(width, int) or
            not isinstance(vae_scale_factor, int) or vae_scale_factor <= 0 or
            config is None or
            not config.get('use_dynamic_shifting', False) or
            not isinstance(max_seq_len, int) or
            not callable(getattr(scheduler, 'register_to_config', None))):
        yield
        return

    packed_stride = vae_scale_factor * 2
    image_seq_len = (height // packed_stride) * (width // packed_stride)
    if image_seq_len <= max_seq_len:
        yield
        return

    register_to_config = getattr(scheduler, 'register_to_config', None)
    _messages.debug_log(
        'Matching the DyPE Flux pipeline scheduler shift calculation; '
        f'image sequence length {image_seq_len} exceeds configured maximum {max_seq_len}.')
    register_to_config(max_image_seq_len=image_seq_len)
    try:
        yield
    finally:
        register_to_config(max_image_seq_len=max_seq_len)


__all__ = ['resolution_shift_context']
