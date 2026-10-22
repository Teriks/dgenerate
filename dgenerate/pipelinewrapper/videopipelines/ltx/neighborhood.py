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

"""Neighborhood attention for the LTX diffusion decoder when no NATTEN build is available."""

from __future__ import annotations

import torch

import dgenerate.messages as _messages


# Bytes of gathered keys or values kept live for one neighborhood chunk.
# Stage 5 attends to an 11x11x11 cube, so the chunk stays well under a gigabyte.
_NEIGHBORHOOD_BUDGET_BYTES = 256 * 1024 * 1024


_TRITON_NEIGHBORHOOD_KERNEL = None


_triton_neighborhood_failed = False


def _neighborhood_budget_bytes() -> int:
    return _NEIGHBORHOOD_BUDGET_BYTES


def _neighborhood_query_chunk(tokens, heads, dim, element_size, window_volume):
    bytes_per_query = max(window_volume * heads * dim * element_size, dim * element_size)
    return max(1, min(tokens, _neighborhood_budget_bytes() // max(bytes_per_query, 1)))


def _triton_neighborhood_ready() -> bool:
    """True when this process can compile a CUDA Triton neighborhood kernel."""
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _neighborhood_kernel():
    """Compile the Triton neighborhood kernel once."""
    global _TRITON_NEIGHBORHOOD_KERNEL
    if _TRITON_NEIGHBORHOOD_KERNEL is not None:
        return _TRITON_NEIGHBORHOOD_KERNEL

    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
            query_ptr, key_ptr, value_ptr, output_ptr,
            frames, height, width, dim,
            kernel_t, kernel_h, kernel_w,
            token_offset, chunk_tokens,
            stride_bh, stride_token,
            BLOCK_D: tl.constexpr,
            OUT_DTYPE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        plane = height * width
        local_token = pid % chunk_tokens
        token = token_offset + local_token
        group = pid // chunk_tokens
        frame = token // plane
        remainder = token - frame * plane
        row = remainder // width
        column = remainder - row * width
        start_frame = tl.minimum(tl.maximum(frame - (kernel_t // 2), 0), frames - kernel_t)
        start_row = tl.minimum(tl.maximum(row - (kernel_h // 2), 0), height - kernel_h)
        start_column = tl.minimum(tl.maximum(column - (kernel_w // 2), 0), width - kernel_w)

        offsets = tl.arange(0, BLOCK_D)
        mask = offsets < dim
        query = tl.load(
            query_ptr + group * stride_bh + token * stride_token + offsets,
            mask=mask, other=0.0).to(tl.float32)
        maximum = tl.full([1], -1.0e30, dtype=tl.float32)
        normalizer = tl.zeros([1], dtype=tl.float32)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)

        for frame_offset in range(kernel_t):
            for row_offset in range(kernel_h):
                for column_offset in range(kernel_w):
                    neighbor = (
                        (start_frame + frame_offset) * plane
                        + (start_row + row_offset) * width
                        + (start_column + column_offset))
                    base = group * stride_bh + neighbor * stride_token
                    key = tl.load(key_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
                    value = tl.load(value_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
                    score = tl.sum(query * key, axis=0)
                    next_max = tl.maximum(maximum, score)
                    alpha = tl.exp(maximum - next_max)
                    weight = tl.exp(score - next_max)
                    normalizer = normalizer * alpha + weight
                    accumulator = accumulator * alpha + value * weight
                    maximum = next_max

        tl.store(
            output_ptr + group * stride_bh + token * stride_token + offsets,
            (accumulator / normalizer).to(OUT_DTYPE),
            mask=mask)

    _TRITON_NEIGHBORHOOD_KERNEL = kernel
    return kernel


def _triton_dtype(dtype):
    import triton.language as tl
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    return tl.float32


def _grouped_kv(key, value, batch, heads, tokens, dim):
    grouped_key = key.permute(0, 4, 1, 2, 3, 5).contiguous().view(batch * heads, tokens, dim)
    grouped_value = value.permute(0, 4, 1, 2, 3, 5).contiguous().view(batch * heads, tokens, dim)
    return grouped_key, grouped_value


def _neighborhood_attention_triton(query, key, value, kernel_t, kernel_h, kernel_w):
    """One program per query. The window matches NATTEN's inward shift."""
    import triton

    batch, frames, height, width, heads, dim = query.shape
    tokens = frames * height * width
    volume = kernel_t * kernel_h * kernel_w
    grouped_query = query.permute(0, 4, 1, 2, 3, 5).contiguous().view(batch * heads, tokens, dim)
    grouped_key, grouped_value = _grouped_kv(key, value, batch, heads, tokens, dim)
    output = torch.empty_like(grouped_query)
    chunk = _neighborhood_query_chunk(
        tokens, heads, dim, query.element_size(), volume)
    kernel = _neighborhood_kernel()
    block_d = triton.next_power_of_2(dim)
    out_dtype = _triton_dtype(query.dtype)
    stride_bh = grouped_query.stride(0)
    stride_token = grouped_query.stride(1)
    for start in range(0, tokens, chunk):
        stop = min(start + chunk, tokens)
        chunk_tokens = stop - start
        kernel[(batch * heads * chunk_tokens,)](
            grouped_query, grouped_key, grouped_value, output,
            frames, height, width, dim,
            kernel_t, kernel_h, kernel_w,
            start, chunk_tokens,
            stride_bh, stride_token,
            BLOCK_D=block_d,
            OUT_DTYPE=out_dtype,
            num_warps=1,
        )
    return output.view(batch, heads, frames, height, width, dim).permute(0, 2, 3, 4, 1, 5).contiguous()


def _neighborhood_attention_gather(query, key, value, kernel_t, kernel_h, kernel_w):
    """PyTorch gather used where Triton cannot run."""
    batch, frames, height, width, heads, dim = query.shape
    tokens = frames * height * width
    plane = height * width
    query_flat = query.reshape(batch, tokens, heads, dim)
    key_flat = key.reshape(batch, tokens, heads, dim)
    value_flat = value.reshape(batch, tokens, heads, dim)
    output = torch.empty_like(query_flat)

    volume = kernel_t * kernel_h * kernel_w
    chunk = _neighborhood_query_chunk(
        tokens, heads, dim, query.element_size(), volume)
    device = query.device
    offset_t = torch.arange(kernel_t, device=device)
    offset_h = torch.arange(kernel_h, device=device)
    offset_w = torch.arange(kernel_w, device=device)

    for start in range(0, tokens, chunk):
        stop = min(start + chunk, tokens)
        index = torch.arange(start, stop, device=device)
        query_t = index // plane
        remainder = index - query_t * plane
        query_h = remainder // width
        query_w = remainder - query_h * width
        start_t = torch.clamp(query_t - (kernel_t // 2), min=0, max=frames - kernel_t)
        start_h = torch.clamp(query_h - (kernel_h // 2), min=0, max=height - kernel_h)
        start_w = torch.clamp(query_w - (kernel_w // 2), min=0, max=width - kernel_w)
        neighbor_t = start_t[:, None, None, None] + offset_t[None, :, None, None]
        neighbor_h = start_h[:, None, None, None] + offset_h[None, None, :, None]
        neighbor_w = start_w[:, None, None, None] + offset_w[None, None, None, :]
        linear = (neighbor_t * plane + neighbor_h * width + neighbor_w).reshape(stop - start, volume)

        gathered_key = key_flat[:, linear]
        gathered_value = value_flat[:, linear]
        query_chunk = query_flat[:, start:stop]
        scores = torch.einsum('bnhd,bnkhd->bnhk', query_chunk, gathered_key)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=gathered_value.dtype)
        output[:, start:stop] = torch.einsum('bnhk,bnkhd->bnhd', weights, gathered_value)

    return output.reshape(batch, frames, height, width, heads, dim)


def _neighborhood_attention(query, key, value, kernel_size):
    """
    Neighborhood attention with NATTEN's inward-shifted window.

    ``query``, ``key``, and ``value`` are ``(batch, frames, height, width, heads, dim)``.
    Each query attends to a cube of ``kernel_size`` that slides inward at the border
    so the cube always stays full. CUDA uses a Triton kernel. Other devices gather
    the cube in PyTorch. Neither path builds the FlexAttention mask.
    """
    global _triton_neighborhood_failed
    if query.ndim != 6:
        raise ValueError(
            'Neighborhood attention expected query shaped '
            f'(batch, frames, height, width, heads, dim), got {tuple(query.shape)}.')
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    batch, frames, height, width, heads, dim = query.shape
    kernel_t, kernel_h, kernel_w = (int(part) for part in kernel_size)
    kernel_t = min(max(kernel_t, 1), frames)
    kernel_h = min(max(kernel_h, 1), height)
    kernel_w = min(max(kernel_w, 1), width)
    if (query.is_cuda and dim <= 128 and heads > 0
            and not _triton_neighborhood_failed and _triton_neighborhood_ready()):
        try:
            return _neighborhood_attention_triton(
                query, key, value, kernel_t, kernel_h, kernel_w)
        except Exception as error:
            _triton_neighborhood_failed = True
            _messages.log(
                'Triton neighborhood attention failed '
                f'({error}). Gathering the window in PyTorch.')
    return _neighborhood_attention_gather(
        query, key, value, kernel_t, kernel_h, kernel_w)


class _LTXWindowNeighborhoodProcessor:
    """
    LTX diffusion-decoder attention when the NATTEN Hub kernel cannot load.

    The call shape matches ``LTX2VideoVaeNeighborhoodNattenProcessor``. CUDA
    runs the window with Triton. The FlexAttention processor is not used:
    its mask does not fit in GPU memory.
    """

    tiles = True

    def __call__(self, attn, hidden_states, block_mask=None):
        batch_size, num_frames, height, width, channels = hidden_states.shape
        query, key, value = attn.project_qkv(hidden_states)
        attended = _neighborhood_attention(query, key, value, attn.kernel_size)
        attended = attended.reshape(batch_size, num_frames, height, width, channels)
        return attn.to_out[0](attended)


def _ltx_diffusion_natten_processor():
    """
    Attention processor for the LTX diffusion decoder.

    Linux uses the ``shi-labs/natten`` kernel from the Hub. Windows has no
    build of that kernel. The same window then runs with Triton, or is
    gathered in PyTorch when Triton is not installed.
    """
    try:
        from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
            LTX2VideoVaeNeighborhoodNattenProcessor)
        return LTX2VideoVaeNeighborhoodNattenProcessor()
    except Exception as error:
        if _triton_neighborhood_ready():
            path = 'Running that window with Triton.'
        else:
            path = 'Gathering that window in PyTorch.'
        _messages.log(
            f'LTX diffusion decoder has no NATTEN build for this system. {path}')
        _messages.debug_log(error)
        return _LTXWindowNeighborhoodProcessor()
