import torch
from typing import Type, Dict, Any, Tuple, Callable, List, Optional, Union, Literal
import torch.nn.functional as F
import math
from .utils import init_generator

import os

from torch import Tensor
# DEBUG_MODE = False


def compute_prune(x: torch.Tensor, mode: str, tome_info: Dict[str, Any], cache: Any) -> Tuple[Callable, ...]:
    """
    Optimized to avoid re-calculation of pruning and reconstruction function
    """

    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]

    if downsample <= args["max_downsample"]:
        if cache.cache_bus.ind_step == cache.cache_bus.step:
            m, u = cache.cache_bus.m_a, cache.cache_bus.u_a

        else:  # when reaching a new step
            # Update indicator
            cache.cache_bus.ind_step = cache.cache_bus.step

            w = int(math.ceil(original_w / downsample))
            h = int(math.ceil(original_h / downsample))
            r = 0.0 # dummy

            # Re-init the generator if it hasn't already been initialized or device has changed.
            if args["generator"] is None:
                args["generator"] = init_generator(x.device)
            elif args["generator"].device != x.device:
                args["generator"] = init_generator(x.device, fallback=args["generator"])

            # the function defines the indices to prune
            m, u = prune_2d(x, w, h, args["sx"], args["sy"], tome_info,
                            no_rand=False, unmerge_mode=mode,
                            cache=cache, rand_indices=None,
                            generator=args["generator"], )

            cache.cache_bus.m_a, cache.cache_bus.u_a = m, u

    else:
        m, u = (do_nothing, do_nothing)

    return m, u


def do_nothing(x: torch.Tensor, mode: str = None, prune: bool = None, unmerge_mode = None, cache=None, ids=None):
    """
    A versatile placeholder...
    """
    if ids:
        return x, None

    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def downsample_temporal_score(temporal_score: torch.Tensor, target_length: int, 
                              original_h: int, original_w: int, target_h: int, target_w: int) -> torch.Tensor:
    """
    Downsample temporal score from original rectangular dimensions to target rectangular dimensions.
    
    Args:
        temporal_score: Tensor with shape [1, N, 1] where N = original_h * original_w
        target_length: Target number of tokens (target_h * target_w)
        original_h: Original height dimension
        original_w: Original width dimension  
        target_h: Target height dimension
        target_w: Target width dimension
    """
    assert temporal_score.dim() == 3 and temporal_score.shape[0] == 1, \
        "temporal_score must have shape [1, N, 1]"

    N = temporal_score.shape[1]
    assert N == original_h * original_w, \
        f"temporal_score length {N} doesn't match original dimensions {original_h}x{original_w}={original_h * original_w}"
    assert target_length == target_h * target_w, \
        f"target_length {target_length} doesn't match target dimensions {target_h}x{target_w}={target_h * target_w}"

    # Calculate downsampling factors for height and width
    factor_h = original_h / target_h
    factor_w = original_w / target_w
    
    # Reshape temporal_score to 2D spatial layout
    x = temporal_score.view(1, 1, original_h, original_w).float()
    
    # Use adaptive average pooling to handle non-integer factors
    pooled = F.adaptive_avg_pool2d(x, (target_h, target_w))
    
    # Apply threshold to create binary mask
    interpolated = (pooled > 0.25).long()
    interpolated = interpolated.view(1, target_length, 1)

    return interpolated

def prune_2d(metric: torch.Tensor,
             w: int, h: int, sx: int, sy: int,
             tome_info: dict,
             no_rand: bool = False,
             unmerge_mode: str = 'token_merge',
             cache: any = None,
             rand_indices: list = None,
             generator: torch.Generator = None
             ) -> Tuple[Callable, Callable]:
    """
    Core algorithm - V2
    """

    def push_all(x: torch.Tensor, cache=cache):
        cache.push(x)
        return x

    B, N, _ = metric.shape

    if cache.cache_bus.step not in range(tome_info['args']['acc_range'][0], tome_info['args']['acc_range'][1]):
        return do_nothing, do_nothing
    else:
        acc_start = tome_info['args']['acc_range'][0]
        max_interval = tome_info['args']['max_interval']

        if acc_start > cache.cache_bus.last_skip_step and (cache.cache_bus.step - acc_start) % max_interval == 0:
            return do_nothing, push_all
        elif acc_start <= cache.cache_bus.last_skip_step and (cache.cache_bus.step - cache.cache_bus.last_skip_step - 1) % max_interval == 0: # should be off by one
            return do_nothing, push_all

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        assert no_rand is False
        rand_idx = torch.randint(sy * sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)

        idx_buffer_view = torch.zeros(hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)
        del idx_buffer, idx_buffer_view
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :]  # src
        b_idx = rand_idx[:, :num_dst, :]  # dst

        temporal_score = cache.cache_bus.temporal_score.reshape(1, -1, 1)

        if temporal_score.shape[1] != N:
            # Get original dimensions from tome_info
            original_h, original_w = tome_info["size"]
            temporal_score = downsample_temporal_score(temporal_score, N, original_h, original_w, h, w)

        a_idx_flat = a_idx.view(-1)
        b_idx_flat = b_idx.view(-1)
        score_flat = temporal_score.view(-1)

        b_mask = torch.zeros(score_flat.size(0), dtype=torch.bool, device=temporal_score.device)
        b_mask[b_idx_flat] = True
        move_mask = (score_flat[a_idx_flat] == 1) & (~b_mask[a_idx_flat])
        indices_to_move = a_idx_flat[move_mask]

        a_idx_flat = a_idx_flat[~move_mask] # Update a_idx_flat by removing indices_to_move
        b_idx_flat = torch.cat([b_idx_flat, indices_to_move]) # Update b_idx_flat by adding indices_to_move
        a_idx = a_idx_flat.view(1, -1, 1)
        b_idx = b_idx_flat.view(1, -1, 1)

        r = a_idx.size(1)
        num_dst = b_idx.size(1)

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        src_idx = torch.arange(r, device=metric.device).view(1, -1, 1)  # Indices in a_idx
        unm_idx = torch.tensor([], device=metric.device, dtype=torch.long).view(1, 0, 1)  # No unmerged tokens

    def prune(x: torch.Tensor, ids=False) -> tuple[Tensor, Tensor] | Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        out = torch.cat([unm, dst], dim=1)

        if ids: # useful when dealing RoPE
            def get_pruned_ids(b_idx: torch.Tensor, h: int, w: int) -> torch.Tensor:
                num_tokens = b_idx.numel()

                new_h = int(round(math.sqrt(num_tokens * (h / w))))
                new_h = max(new_h, 1)  # Ensure at least one row
                new_w = int(math.ceil(num_tokens / new_h))

                rows = torch.arange(new_h)
                cols = torch.arange(new_w)
                grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing='ij')

                grid_rows = grid_rows.flatten()[:num_tokens]
                grid_cols = grid_cols.flatten()[:num_tokens]

                zeros = torch.zeros_like(grid_rows)

                ids = torch.stack([zeros, grid_rows, grid_cols], dim=-1).to('cuda')

                return ids

            ids = get_pruned_ids(b_idx, h, w)
            return out, ids

        return out

    def reconstruct(x: torch.Tensor, unmerge_mode=unmerge_mode, cache=cache) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        if unmerge_mode == 'cache_merge' and cache.cache_bus.step in range(tome_info['args']['acc_range'][0], tome_info['args']['acc_range'][1]):
            cache.push(dst, index=b_idx.expand(B, num_dst, c))
            src = cache.pop(index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c))
        else:
            raise RuntimeError

        # == Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out

    return prune, reconstruct




