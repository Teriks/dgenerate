# Copyright (c) 2025 Guy Yariv
#
# This file adapts the YARN / NTK / DyPE inverse-frequency methods from
# https://github.com/guyyariv/DyPE. See LICENSE for the upstream MIT license.

"""Small PyTorch implementation of DyPE's RoPE frequency schedules."""

from __future__ import annotations

import math

import torch


def _correction_factor(num_rotations: float, dim: int, theta: float, base_length: float) -> float:
    return (dim * math.log(base_length / (num_rotations * 2.0 * math.pi))) / (2.0 * math.log(theta))


def _correction_range(low_ratio: float,
                      high_ratio: float,
                      dim: int,
                      theta: float,
                      base_length: float) -> tuple[int, int]:
    low = math.floor(_correction_factor(low_ratio, dim, theta, base_length))
    high = math.ceil(_correction_factor(high_ratio, dim, theta, base_length))
    return max(low, 0), min(high, dim - 1)


def _linear_ramp_mask(low: int, high: int, dim: int, *, device: torch.device, dtype: torch.dtype):
    if low == high:
        high += 0.001
    ramp = (torch.arange(dim, device=device, dtype=torch.float32) - low) / (high - low)
    return ramp.clamp(0, 1).to(dtype=dtype)


def dype_rope_1d(positions: torch.Tensor,
                 dim: int,
                 *,
                 theta: float,
                 base_length: float,
                 current_length: int,
                 timestep: float,
                 method: str = 'yarn',
                 output: str = 'real',
                 complex_mscale: bool = False,
                 freqs_dtype: torch.dtype | None = None):
    """Build DyPE frequencies for one position axis using YARN, NTK, or base RoPE.

    `timestep` is normalized so 1 is the noisiest step and 0 is the final
    denoising step. `output` may be `real` (cos/sin), `complex`, or `angles`.
    """
    if dim % 2 or dim < 2:
        raise ValueError(f'RoPE axis dimension must be an even integer >= 2, got {dim}.')
    if base_length <= 0:
        raise ValueError(f'RoPE base length must be positive, got {base_length}.')
    if method not in {'base', 'ntk', 'yarn'}:
        raise ValueError(f'Unknown DyPE position encoding method {method!r}.')
    if output not in {'real', 'complex', 'angles'}:
        raise ValueError(f'Unknown RoPE output format {output!r}.')

    positions = positions.reshape(-1)
    device = positions.device
    if freqs_dtype is None:
        freqs_dtype = torch.float64 if device.type not in {'mps', 'npu'} else torch.float32
    exponents = torch.arange(0, dim, 2, device=device, dtype=freqs_dtype) / dim
    freqs_base = torch.pow(torch.tensor(theta, device=device, dtype=freqs_dtype), -exponents)

    scale_value = max(1.0, float(current_length) / float(base_length))
    mscale = 1.0
    if method == 'ntk' and scale_value > 1.0:
        if dim <= 2:
            raise ValueError('DyPE NTK scaling requires a RoPE axis dimension greater than 2.')
        timestep_squared = min(1.0, max(0.0, float(timestep))) ** 2.0
        ntk_factor = scale_value ** (dim / (dim - 2))
        ntk_factor = ntk_factor ** (2.0 * timestep_squared)
        scaled_theta = torch.tensor(theta * ntk_factor, device=device, dtype=freqs_dtype)
        freqs = torch.pow(scaled_theta, -exponents)
    elif method == 'yarn' and scale_value > 1.0:
        scale = torch.tensor(scale_value, device=device, dtype=freqs_dtype)
        freqs_linear = freqs_base / scale
        new_base = theta * scale.pow(dim / (dim - 2)) if dim > 2 else theta
        freqs_ntk = torch.pow(torch.as_tensor(new_base, device=device, dtype=freqs_dtype), -exponents)

        kappa = min(1.0, max(0.0, float(timestep))) ** 2.0
        beta_0 = max(1.25 * kappa, 1e-6)
        beta_1 = max(0.75 * kappa, 1e-6)
        low, high = _correction_range(beta_0, beta_1, dim, theta, base_length)
        high = min(dim // 2, high)
        freqs_mask = 1.0 - _linear_ramp_mask(low, high, dim // 2, device=device, dtype=freqs_dtype)
        freqs = freqs_linear * (1.0 - freqs_mask) + freqs_ntk * freqs_mask

        gamma_0 = max(16.0 * kappa, 1e-6)
        gamma_1 = max(2.0 * kappa, 1e-6)
        low, high = _correction_range(gamma_0, gamma_1, dim, theta, base_length)
        high = min(dim // 2, high)
        freqs_mask = 1.0 - _linear_ramp_mask(low, high, dim // 2, device=device, dtype=freqs_dtype)
        freqs = freqs * (1.0 - freqs_mask) + freqs_base * freqs_mask
        mscale = 0.1 * math.log(scale_value) + 1.0
    else:
        freqs = freqs_base

    angles = positions.to(device=device, dtype=freqs_dtype).unsqueeze(-1) * freqs.unsqueeze(0)
    if output == 'angles':
        return angles.float()
    if output == 'complex':
        angles = angles.float()
        frequencies = torch.polar(torch.ones_like(angles), angles)
        if complex_mscale:
            frequencies = frequencies * mscale
        return frequencies.to(torch.complex64)

    cos = angles.cos().repeat_interleave(2, dim=-1).float() * mscale
    sin = angles.sin().repeat_interleave(2, dim=-1).float() * mscale
    return cos, sin
