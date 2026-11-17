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

"""MiniMax-H3 joint video and audio generation.

The checkpoint is a modular pipeline. ``t2va`` and ``fl2va`` read ``transformer/``.
``ref2va`` reads ``transformer_ref/``. One call writes the clip and its soundtrack.
Guidance is distilled into the weights.
"""

from __future__ import annotations

import contextlib
import os

import torch

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.pipelines as _pipelines

FPS = 24.0
MIN_SECONDS = 5.0
MAX_SECONDS = 15.0
CANVAS_MULTIPLE = 32
# 5 seconds is 120 frames and is not a 17n+5 count. The shortest clip
# the video VAE can decode is 124 frames, 5.17 seconds at 24 fps.
DEFAULT_FRAMES = 124
DEFAULT_SECONDS = DEFAULT_FRAMES / FPS
DEFAULT_STEPS = 30
MAX_REFERENCES = 12
MAX_IMAGE_REFERENCES = 9
MAX_VIDEO_REFERENCES = 3
MAX_AUDIO_REFERENCES = 3

_AUDIO_EXTENSIONS = frozenset({
    'wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'opus', 'wma', 'aiff', 'aif',
})

# ComfyUI stores one fused qkv. Diffusers stores to_q, to_k, and to_v.
# The split is contiguous: q, then k, then v, along the output rows.
_QKV_PARTS = ('to_q', 'to_k', 'to_v')


def _extension(path: str) -> str:
    cleaned = str(path).split('?', 1)[0].split('#', 1)[0]
    return os.path.splitext(cleaned)[1].lstrip('.').lower()


def reference_kind(path: str) -> str:
    """``image``, ``video``, or ``audio`` for one ``reference=`` path."""
    import dgenerate.filetypes as _filetypes

    extension = _extension(path)
    if extension in _AUDIO_EXTENSIONS:
        return 'audio'
    if (extension in _filetypes.video_read_extension_set()
            or extension in _filetypes.animated_image_read_extension_set()):
        return 'video'
    return 'image'


def classify_seed(parsed, controlnet: bool = False) -> str:
    """
    ``t2va``, ``fl2va``, or ``ref2va`` for one image seed.

    A still image and ``last-frame=`` are keyframes. ``reference=`` is an
    ordered omni-reference list and uses the other transformer. With
    ``--control-nets`` the seed is a control video, or an inpaint source plus
    ``mask=``, and the workflow stays ``t2va``.
    """
    if controlnet:
        return _classify_control_seed(parsed)
    if parsed is None:
        return 't2va'
    if parsed.mask_images or parsed.control_images or parsed.adapter_images or parsed.floyd_image:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-video does not accept masks, control images, IP adapter images, '
            'or a floyd image.')
    if parsed.multi_image_mode or (parsed.images and len(parsed.images) != 1):
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-video accepts one opening still. Extra images belong in reference=.')
    references = list(parsed.reference_images or ())
    has_keyframe = bool(parsed.images) or bool(parsed.end_image)
    if references and has_keyframe:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-video keyframes and reference= are different workflows. '
            'Use an opening image or last-frame= for fl2va, or reference= alone for ref2va.')
    if references:
        _check_reference_counts(references)
        return 'ref2va'
    if has_keyframe:
        return 'fl2va'
    return 't2va'


def _classify_control_seed(parsed) -> str:
    """``t2va`` when ``--control-nets`` supplies the control video."""
    if parsed is None:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-video --control-nets needs a control video in --image-seeds, '
            'or an inpaint source with mask=.')
    if parsed.adapter_images or parsed.floyd_image:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-video --control-nets does not accept IP adapter images or a floyd image.')
    if parsed.reference_images:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-video --control-nets cannot be combined with reference=. '
            'The control branch was trained on text to video.')
    if parsed.end_image:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-video --control-nets cannot be combined with last-frame=. '
            'The control branch was trained on text to video.')
    if parsed.multi_image_mode or (parsed.images and len(parsed.images) != 1):
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-video --control-nets accepts one control video, '
            'or one inpaint source.')
    bare = bool(parsed.images) and not parsed.mask_images and not parsed.control_images
    if bare:
        return 't2va'
    if parsed.images and not parsed.mask_images:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-video --control-nets reads a control video. '
            'A seed image next to control= is a keyframe, which this branch was not trained on. '
            'Pass the control video alone, or add mask= for inpaint.')
    if parsed.mask_images and not parsed.images:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-video inpaint needs the source video before mask=.')
    if parsed.control_images and len(parsed.control_images) != 1:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-video --control-nets accepts one control video.')
    return 't2va'


def _check_reference_counts(paths: list[str]) -> None:
    counts = {'image': 0, 'video': 0, 'audio': 0}
    for path in paths:
        counts[reference_kind(path)] += 1
    if len(paths) > MAX_REFERENCES:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'minimax-h3-video accepts at most {MAX_REFERENCES} references.')
    if counts['image'] > MAX_IMAGE_REFERENCES:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'minimax-h3-video accepts at most {MAX_IMAGE_REFERENCES} image references.')
    if counts['video'] > MAX_VIDEO_REFERENCES:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'minimax-h3-video accepts at most {MAX_VIDEO_REFERENCES} video references.')
    if counts['audio'] > MAX_AUDIO_REFERENCES:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'minimax-h3-video accepts at most {MAX_AUDIO_REFERENCES} audio references.')
    if counts['audio'] and not counts['image'] and not counts['video']:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-video audio references have to be paired with an image or a video reference.')


def frames_for_seconds(seconds: float) -> int:
    """Snap a duration to a ``17 * n + 5`` frame count MiniMax-H3 can decode.

    The ceiling is the aligned duration. A request whose upward snap would pass
    15 seconds steps back one video-VAE chunk instead.
    """
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import align_num_frames

    seconds = float(seconds)
    if seconds < MIN_SECONDS or seconds > MAX_SECONDS:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'minimax-h3-video accepts --video-lengths from {MIN_SECONDS:g} to {MAX_SECONDS:g} seconds. '
            f'Got {seconds}.')
    requested = max(1, int(round(seconds * FPS)))
    aligned = align_num_frames(requested, 17, 5)
    if aligned / FPS > MAX_SECONDS:
        aligned -= 17
    duration = aligned / FPS
    if duration < MIN_SECONDS or duration > MAX_SECONDS:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'minimax-h3-video accepts --video-lengths from {MIN_SECONDS:g} to {MAX_SECONDS:g} seconds. '
            f'{seconds} has no 17n+5 frame count in that window.')
    if aligned != requested:
        _messages.warning(
            f'minimax-h3-video snapped --video-lengths {seconds} to {aligned} frames '
            f'({duration:.2f}s) so the video VAE can decode it.')
    return aligned


def _warn_transformer_partition(uri, workflow: str) -> None:
    """The two published partitions are not interchangeable."""
    name = str(uri).lower()
    if 'ref2va' in name and workflow != 'ref2va':
        _messages.warning(
            'This transformer file name says ref2va, but this request uses the '
            'text and keyframe transformer. Pass reference= to load it as transformer_ref.')
    elif ('fl2va' in name or 't2va' in name) and workflow == 'ref2va':
        _messages.warning(
            'This transformer file name says fl2va, but this request uses the '
            'reference transformer. Leave reference= out to load it as transformer/.')


def denoiser_name(workflow: str) -> str:
    """Component slot for this workflow."""
    if workflow == 'ref2va':
        return 'transformer_ref'
    return 'transformer'


def _is_comfy_state(keys) -> bool:
    for key in keys:
        if (key == 'video_patch_proj.weight'
                or key.startswith('blocks.')
                or '.qkv_proj.' in key
                or key.startswith('token_refiner.blocks.')):
            return True
    return False


_SKIP_KEYS = frozenset({
    'rope.inv_freq',
    'adaln_basis',
    'adaln_mean',
    'scaled_fp8',
})
_QKV_ROW_SUFFIXES = ('weight', 'bias', 'weight_scale')


def _logical_shape(value) -> tuple[int, ...]:
    quant_shape = getattr(value, 'quant_shape', None)
    if quant_shape is not None:
        return tuple(int(item) for item in quant_shape)
    return tuple(int(item) for item in value.shape)


def _split_qkv(weight: torch.Tensor):
    """Split fused qkv on the output axis. Packed GGUF rows stay packed."""
    if weight.ndim == 1:
        if weight.shape[0] % 3:
            raise ValueError(
                f'MiniMax-H3 qkv projection has {weight.shape[0]} elements, which is not divisible by 3.')
        return weight.chunk(3, dim=0)
    if weight.ndim != 2:
        raise ValueError(
            f'MiniMax-H3 qkv projection has shape {tuple(weight.shape)}, expected a matrix.')
    logical = _logical_shape(weight)
    if len(logical) != 2:
        raise ValueError(
            f'MiniMax-H3 qkv projection has shape {logical}, expected a matrix.')
    output_axis = 0 if logical[0] >= logical[1] else 1
    if logical[output_axis] % 3:
        output_axis = 1 - output_axis
    if logical[output_axis] % 3:
        raise ValueError(
            f'MiniMax-H3 qkv projection shape {logical} is not divisible by 3.')
    # Storage follows the logical row axis for a packed GGUF weight.
    storage_axis = 0 if weight.shape[0] == logical[0] else output_axis
    if weight.shape[storage_axis] % 3:
        storage_axis = output_axis
    parts = weight.chunk(3, dim=storage_axis)
    if storage_axis == 1:
        parts = tuple(part.transpose(0, 1).contiguous() for part in parts)
    return parts


def _swap_swiglu_halves(weight: torch.Tensor) -> torch.Tensor:
    """Swap the two halves of a fused SwiGLU projection.

    ComfyUI stores ``mlp.fc1`` as gate then value. Diffusers' SwiGLU reads
    value then gate.
    """
    if not isinstance(weight, torch.Tensor) or weight.ndim not in (1, 2):
        return weight
    if weight.ndim == 1:
        if weight.shape[0] % 2:
            return weight
        gate, value = weight.chunk(2, dim=0)
        return torch.cat([value, gate], dim=0)
    logical = _logical_shape(weight)
    if len(logical) != 2:
        return weight
    output_axis = 0 if logical[0] >= logical[1] else 1
    if logical[output_axis] % 2:
        output_axis = 1 - output_axis
    if logical[output_axis] % 2:
        return weight
    storage_axis = 0 if weight.shape[0] == logical[0] else output_axis
    if weight.shape[storage_axis] % 2:
        storage_axis = output_axis
    gate, value = weight.chunk(2, dim=storage_axis)
    return torch.cat([value, gate], dim=storage_axis)


def _prepare_comfy_key(key: str) -> str | None:
    """Normalize one Comfy key, or ``None`` when the tensor is not a weight."""
    if key in _SKIP_KEYS or key.endswith('.rope.inv_freq'):
        return None
    if key.endswith('.adaln_proj.folded_bias'):
        key = key[:-len('folded_bias')] + 'linear.bias'
    elif key == 'norm_out.folded_bias':
        key = 'final_layer.adaln_proj.linear.bias'
    if key in ('time_embedder.table', 'adaln_t_table'):
        return 'adaln_t_table'
    return key


def _orient_table(table: torch.Tensor) -> torch.Tensor:
    """Curve table as ``(timesteps, basis)``, 1025 by 8 in the published files."""
    if table.ndim == 2 and table.shape[0] < table.shape[1]:
        table = table.transpose(0, 1).contiguous()
    return table


def _stash_curve_table(converted: dict) -> dict:
    """Hold the pruned timestep table aside so single-file load does not drop it."""
    table = converted.pop('adaln_t_table', None)
    if table is None:
        return converted
    table = _orient_table(table).detach().float().contiguous()
    for key in list(converted):
        if key.startswith('time_embedder.'):
            converted.pop(key)
    convert_minimax_h3_checkpoint.last_curve_table = table
    return converted


def comfy_to_diffusers(checkpoint: dict) -> dict:
    """Rename a ComfyUI MiniMax-H3 transformer onto ``MiniMaxH3Transformer3DModel``.

    Full checkpoints keep ``time_embedder``. Pruned curve checkpoints replace it
    with ``adaln_t_table`` and 8-wide AdaLN projections. Both use fused ``qkv_proj``.
    """
    import dgenerate.pipelinewrapper.singlefilecompat as _singlefilecompat

    checkpoint = _singlefilecompat.strip_comfy_prefix(dict(checkpoint))
    if any(key.startswith('diffusion_model.') for key in checkpoint):
        checkpoint = {
            key[len('diffusion_model.'):] if key.startswith('diffusion_model.') else key: value
            for key, value in checkpoint.items()}
    convert_minimax_h3_checkpoint.last_curve_table = None
    if any('to_gate_compress' in key for key in checkpoint):
        raise ValueError(
            'This MiniMax-H3 file has attn.to_gate_compress, which the Diffusers '
            'transformer does not implement.')

    converted = {}
    skipped = []
    for key, value in checkpoint.items():
        prepared = _prepare_comfy_key(key)
        if prepared is None:
            continue
        if prepared == 'adaln_t_table':
            converted[prepared] = value
            continue
        try:
            renamed = _comfy_key_to_diffusers(prepared)
        except ValueError:
            skipped.append(key)
            continue
        if '.qkv_proj.' in renamed:
            suffix = renamed.rsplit('.', 1)[-1]
            base = renamed[:-(len('qkv_proj.') + len(suffix))]
            if suffix == 'comfy_quant':
                for part in ('to_q', 'to_k', 'to_v'):
                    converted[base + part + '.comfy_quant'] = value
                continue
            if suffix not in _QKV_ROW_SUFFIXES:
                skipped.append(key)
                continue
            query, key_weight, value_weight = _split_qkv(value)
            converted[base + 'to_q.' + suffix] = query
            converted[base + 'to_k.' + suffix] = key_weight
            converted[base + 'to_v.' + suffix] = value_weight
            continue
        if '.ff.net.0.proj.' in renamed and renamed.rsplit('.', 1)[-1] in ('weight', 'bias'):
            value = _swap_swiglu_halves(value)
        converted[renamed] = value
    if skipped:
        shown = ', '.join(skipped[:6])
        if len(skipped) > 6:
            shown += f', and {len(skipped) - 6} more'
        _messages.debug_log(f'Skipping unrecognized MiniMax-H3 tensors: {shown}')
    return _stash_curve_table(converted)


def _comfy_key_to_diffusers(key: str) -> str:
    simple = (
        ('video_patch_proj.', 'proj_in.'),
        ('audio_patch_proj.', 'audio_proj_in.'),
        ('condition_proj.', 'context_embedder.'),
        ('time_embedder.proj_in.', 'time_embedder.linear_1.'),
        ('time_embedder.proj_out.', 'time_embedder.linear_2.'),
        ('final_layer.norm.', 'norm_out.norm.'),
        ('final_layer.adaln_proj.linear.', 'norm_out.linear.'),
        ('final_layer.video_out.', 'proj_out.'),
        ('final_layer.audio_out.', 'audio_proj_out.'),
    )
    for source, target in simple:
        if key.startswith(source):
            return target + key[len(source):]
    if key.startswith('token_refiner.blocks.'):
        return 'token_refiner.refiner_blocks.' + _comfy_block_tail(key[len('token_refiner.blocks.'):])
    if key.startswith('blocks.'):
        return 'transformer_blocks.' + _comfy_block_tail(key[len('blocks.'):])
    if key.startswith('token_refiner.final_norm.'):
        return key
    raise ValueError(f'No Diffusers name for MiniMax-H3 tensor "{key}".')


def _comfy_block_tail(rest: str) -> str:
    """``0.attn.q_norm.weight`` to the Diffusers tail, leaving ``qkv_proj`` for the split."""
    index, dot, tail = rest.partition('.')
    if not dot:
        return rest
    tail = tail.replace('attn.q_norm.', 'attn.norm_q.')
    tail = tail.replace('attn.k_norm.', 'attn.norm_k.')
    tail = tail.replace('attn.out_proj.', 'attn.to_out.0.')
    tail = tail.replace('mlp.fc1.', 'ff.net.0.proj.')
    tail = tail.replace('mlp.fc2.', 'ff.net.2.')
    return index + '.' + tail


def _fuse_image_output_heads(state: dict) -> dict:
    """Join ``pdd_proj_out.0`` through ``pdd_proj_out.5`` into one ``proj_out``.

    The image transformer keeps one linear per distilled step. ComfyUI stores
    those six heads as ``final_layer.video_out``, which the video rename writes
    from ``proj_out``.
    """
    grouped: dict[str, dict[int, torch.Tensor]] = {}
    for key, value in state.items():
        if not key.startswith('pdd_proj_out.'):
            continue
        index, dot, suffix = key[len('pdd_proj_out.'):].partition('.')
        if not index.isdigit() or not dot or not isinstance(value, torch.Tensor):
            continue
        grouped.setdefault(suffix, {})[int(index)] = value
    if not grouped:
        return state
    converted = {key: value for key, value in state.items() if not key.startswith('pdd_proj_out.')}
    for suffix, parts in grouped.items():
        if suffix == 'comfy_quant':
            converted['proj_out.comfy_quant'] = next(iter(parts.values()))
            continue
        ordered = _pack_pdd_head_bank([parts[index] for index in sorted(parts)])
        converted['proj_out.' + suffix] = torch.cat(ordered, dim=0).contiguous()
    return converted


def diffusers_to_comfy(state: dict) -> dict:
    """Rename a Diffusers MiniMax-H3 transformer onto ComfyUI names and fuse qkv.

    A checkpoint that is already Comfy names (``blocks.`` and ``qkv_proj``) is
    returned as it is. The published image ``transformer/`` folder and the
    video ``FL2VA`` and ``Ref2VA`` folders are that layout.
    """
    if _is_comfy_state(state):
        pending = {
            key: value.detach().cpu().contiguous()
            for key, value in state.items() if isinstance(value, torch.Tensor)}
        if 'rope.inv_freq' not in pending and _hidden_size(pending) is not None:
            pending['rope.inv_freq'] = _rope_inv_freq()
        return pending
    state = _fuse_image_output_heads(state)
    pending = {}
    fused = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        renamed, fuse = _diffusers_key_to_comfy(key)
        if (not fuse and '.mlp.fc1.' in renamed
                and renamed.rsplit('.', 1)[-1] in ('weight', 'bias')):
            value = _swap_swiglu_halves(value)
        if fuse:
            prefix, part, suffix = fuse
            slot = fused.setdefault((prefix, suffix), {})
            slot[part] = value
            continue
        pending[renamed] = value.detach().cpu().contiguous()
    for (prefix, suffix), parts in fused.items():
        missing = [part for part in _QKV_PARTS if part not in parts]
        if missing:
            raise ValueError(
                f'MiniMax-H3 attention at "{prefix}" is missing {", ".join(missing)}.')
        pending[prefix + 'qkv_proj.' + suffix] = torch.cat(
            [parts[part] for part in _QKV_PARTS], dim=0).contiguous()
    hidden = _hidden_size(pending)
    if hidden is not None:
        pending['rope.inv_freq'] = _rope_inv_freq()
    return pending


def _hidden_size(state: dict) -> int | None:
    weight = state.get('video_patch_proj.weight')
    if weight is not None and weight.ndim == 2:
        return int(weight.shape[0])
    return None


def _rope_inv_freq() -> torch.Tensor:
    """The 16 frequencies ComfyUI stores and Diffusers recomputes."""
    rope_freq_dim = 16
    positions = torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32)
    return 1.0 / (10000.0 ** (positions / (2 * rope_freq_dim)))


def _diffusers_key_to_comfy(key: str):
    """Return ``(comfy key, None)`` or ``(unused, (prefix, q|k|v, suffix))``."""
    simple = (
        ('proj_in.', 'video_patch_proj.'),
        ('audio_proj_in.', 'audio_patch_proj.'),
        ('context_embedder.', 'condition_proj.'),
        ('time_embedder.linear_1.', 'time_embedder.proj_in.'),
        ('time_embedder.linear_2.', 'time_embedder.proj_out.'),
        ('norm_out.norm.', 'final_layer.norm.'),
        ('norm_out.linear.', 'final_layer.adaln_proj.linear.'),
        ('proj_out.', 'final_layer.video_out.'),
        ('audio_proj_out.', 'final_layer.audio_out.'),
    )
    for source, target in simple:
        if key.startswith(source):
            return target + key[len(source):], None
    if key.startswith('token_refiner.final_norm.'):
        return key, None
    if key.startswith('token_refiner.refiner_blocks.'):
        return _diffusers_block_key(
            'token_refiner.blocks.', key[len('token_refiner.refiner_blocks.'):])
    if key.startswith('transformer_blocks.'):
        return _diffusers_block_key('blocks.', key[len('transformer_blocks.'):])
    raise ValueError(f'No ComfyUI name for MiniMax-H3 tensor "{key}".')


def _diffusers_block_key(prefix: str, rest: str):
    index, dot, tail = rest.partition('.')
    if not dot:
        raise ValueError(f'No ComfyUI name for MiniMax-H3 tensor tail "{rest}".')
    block = prefix + index + '.'
    for part in _QKV_PARTS:
        marker = f'attn.{part}.'
        if tail.startswith(marker):
            return '', (block + 'attn.', part, tail[len(marker):])
    tail = tail.replace('attn.norm_q.', 'attn.q_norm.')
    tail = tail.replace('attn.norm_k.', 'attn.k_norm.')
    tail = tail.replace('attn.to_out.0.', 'attn.out_proj.')
    tail = tail.replace('ff.net.0.proj.', 'mlp.fc1.')
    tail = tail.replace('ff.net.2.', 'mlp.fc2.')
    return block + tail, None


def convert_minimax_h3_checkpoint(checkpoint: dict, **kwargs) -> dict:
    """Single-file entry. Comfy names are rewritten. Diffusers names pass through."""
    del kwargs
    import dgenerate.pipelinewrapper.singlefilecompat as _singlefilecompat

    checkpoint = _singlefilecompat.strip_comfy_prefix(dict(checkpoint))
    if not _is_comfy_state(checkpoint) and not _has_curve_key(checkpoint):
        convert_minimax_h3_checkpoint.last_curve_table = None
        return checkpoint
    if _is_comfy_state(checkpoint):
        return comfy_to_diffusers(checkpoint)
    renamed = dict(checkpoint)
    if 'time_embedder.table' in renamed and 'adaln_t_table' not in renamed:
        renamed['adaln_t_table'] = renamed.pop('time_embedder.table')
    return _stash_curve_table(renamed)


convert_minimax_h3_checkpoint.last_curve_table = None


def _split_even(weight: torch.Tensor, parts: int, part_size: int | None = None):
    """Split one fused projection into ``parts`` along its output axis.

    ``part_size`` is one head's output width. The fused GGUF matrix is
    ``(heads * part_size, hidden)`` even when hidden is the longer side, so
    the longer side is not the output.
    """
    if weight.ndim == 1:
        if weight.shape[0] % parts:
            raise ValueError(
                f'MiniMax-H3-Image output projection has {weight.shape[0]} elements, '
                f'which is not divisible by {parts}.')
        return weight.chunk(parts, dim=0)
    if weight.ndim != 2:
        raise ValueError(
            f'MiniMax-H3-Image output projection has shape {tuple(weight.shape)}, expected a matrix.')
    logical = _logical_shape(weight)
    target = None if part_size is None else parts * part_size
    if target is not None and logical[0] == target:
        output_axis = 0
    elif target is not None and len(logical) > 1 and logical[1] == target:
        output_axis = 1
    else:
        output_axis = 0 if logical[0] >= logical[1] else 1
    if logical[output_axis] % parts:
        output_axis = 1 - output_axis
    if logical[output_axis] % parts:
        raise ValueError(
            f'MiniMax-H3-Image output projection shape {logical} is not divisible by {parts}.')
    storage_axis = 0 if weight.shape[0] == logical[0] else output_axis
    if weight.shape[storage_axis] % parts:
        storage_axis = output_axis
    chunks = weight.chunk(parts, dim=storage_axis)
    if storage_axis == 1:
        chunks = tuple(part.transpose(0, 1).contiguous() for part in chunks)
    return chunks


def _expand_pdd_head_bank(parts):
    """Turn a ComfyUI PDD bank into one full head per step.

    Row block 0 is the first step's head. Every later block is that step's
    head minus the first, which is how ComfyUI stores ``final_layer.video_out``.
    """
    if not parts or not parts[0].is_floating_point():
        return parts
    base = parts[0]
    return [base, *[base + part for part in parts[1:]]]


def _pack_pdd_head_bank(parts):
    """Store full per-step heads as a ComfyUI bank of the first head plus offsets."""
    if not parts or not parts[0].is_floating_point():
        return parts
    base = parts[0]
    return [base, *[part - base for part in parts[1:]]]


def split_image_output_heads(checkpoint: dict, heads: int = 6) -> dict:
    """Turn one fused ``proj_out`` into the image transformer's per-step heads.

    ComfyUI stores the six distilled output heads as ``final_layer.video_out``.
    The first block of rows is step 0's head. Each later block is that step's
    head minus the first. ``MiniMaxH3ImageTransformer2DModel`` keeps one full
    head per step as ``pdd_proj_out.0`` through ``pdd_proj_out.5``.
    """
    fused = [key for key in checkpoint if key == 'proj_out' or key.startswith('proj_out.')]
    if not fused:
        return checkpoint
    part_size = None
    proj_in = checkpoint.get('proj_in.weight')
    if torch.is_tensor(proj_in) and proj_in.ndim == 2:
        logical = _logical_shape(proj_in)
        if len(logical) == 2:
            part_size = int(min(logical))
    converted = dict(checkpoint)
    for key in fused:
        value = converted.pop(key)
        suffix = 'weight' if key == 'proj_out' else key[len('proj_out.'):]
        if suffix == 'comfy_quant':
            for index in range(heads):
                converted[f'pdd_proj_out.{index}.comfy_quant'] = value
            continue
        parts = _expand_pdd_head_bank(list(_split_even(value, heads, part_size)))
        for index, part in enumerate(parts):
            converted[f'pdd_proj_out.{index}.{suffix}'] = part
    return converted


def convert_minimax_h3_image_checkpoint(checkpoint: dict, **kwargs) -> dict:
    """ComfyUI MiniMax-H3-Image names onto ``MiniMaxH3ImageTransformer2DModel``."""
    return split_image_output_heads(convert_minimax_h3_checkpoint(checkpoint, **kwargs))


def _has_curve_key(keys) -> bool:
    if isinstance(keys, dict):
        keys = keys.keys()
    return any(key in ('adaln_t_table', 'time_embedder.table') or key.endswith('.adaln_t_table')
               for key in keys)


def _table_from_source(source):
    if not isinstance(source, dict):
        return None
    for key in ('adaln_t_table', 'time_embedder.table'):
        if key in source and torch.is_tensor(source[key]):
            return source[key]
    return None


def _narrow_adaln_width(tensors: dict) -> int | None:
    for key, value in tensors.items():
        if not torch.is_tensor(value):
            continue
        if not (key.endswith('adaln_proj.linear.weight') or key.endswith('norm_out.linear.weight')):
            continue
        shape = _logical_shape(value)
        if len(shape) == 2 and min(shape) <= 32 < max(shape):
            return int(min(shape))
    return None


def curve_time_embed_dim(source) -> int | None:
    """Basis width of a pruned checkpoint, or ``None`` for a full time embedder.

    ``source`` is a state dict, a ``.safetensors`` path, or a ``.gguf`` path.
    A table already removed from a converted state dict is read from the stash
    only when that dict still has the narrow AdaLN projections.
    """
    table = _table_from_source(source)
    if table is not None:
        return int(_orient_table(table).shape[-1])
    if isinstance(source, dict):
        width = _narrow_adaln_width(source)
        stashed = getattr(convert_minimax_h3_checkpoint, 'last_curve_table', None)
        if width is not None and torch.is_tensor(stashed):
            return int(_orient_table(stashed).shape[-1])
        return None
    if not isinstance(source, str):
        return None
    shapes = _curve_shapes(source)
    if shapes is None:
        return None
    for name in ('adaln_t_table', 'time_embedder.table'):
        shape = shapes.get(name)
        if shape is not None and len(shape) == 2:
            return int(min(shape))
    return None


def _curve_shapes(path: str) -> dict | None:
    lower = path.lower()
    try:
        if lower.endswith('.gguf'):
            import dgenerate.pipelinewrapper.ggufcompat as _ggufcompat
            return _ggufcompat.read_gguf_shapes(path)
        if lower.endswith('.safetensors'):
            import safetensors
            with safetensors.safe_open(path, framework='pt', device='cpu') as handle:
                present = set(handle.keys())
                shapes = {}
                for key in present:
                    plain = key
                    for prefix in ('model.diffusion_model.', 'diffusion_model.'):
                        if plain.startswith(prefix):
                            plain = plain[len(prefix):]
                    if plain in ('adaln_t_table', 'time_embedder.table'):
                        shapes[plain] = tuple(handle.get_slice(key).get_shape())
                return shapes
    except (OSError, ValueError):
        return None
    return None


def _interpolate_curve(table: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    """Sample the pruned AdaLN curve. ``timestep`` is in ``[0, 1]``."""
    grid = table.shape[0] - 1
    position = timestep.detach().float().reshape(-1).clamp(0.0, 1.0) * grid
    lower = position.floor().long().clamp(max=table.shape[0] - 2)
    fraction = (position - lower.float()).unsqueeze(1).to(table.dtype)
    return torch.lerp(table[lower], table[lower + 1], fraction)


def _rope_frequencies(rope_freq_dim: int, rope_theta: float) -> torch.Tensor:
    positions = torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32)
    return 1.0 / (rope_theta ** (positions / (2 * rope_freq_dim)))


def _vae_rope_frequencies(config, rope) -> torch.Tensor | None:
    """Frequencies for a MiniMax VAE decoder rope that was built on the meta device."""
    if not hasattr(config, 'decoder_rope_theta'):
        return None
    num_axes = 2 if 'Image' in type(rope).__name__ else 3
    dim = int(config.decoder_attention_head_dim * config.decoder_rope_dim_ratio)
    if dim % (2 * num_axes) != 0:
        return None
    step = 2 * num_axes / dim
    return 1.0 / float(config.decoder_rope_theta) ** torch.arange(0, 1, step, dtype=torch.float32)


def _restore_ropes(root) -> None:
    """Replace meta rotary tables. They are not in the checkpoint and stay empty after load."""
    config = getattr(root, 'config', None)
    for module in root.modules():
        buffer = module._buffers.get('inv_freq') if hasattr(module, '_buffers') else None
        if buffer is None or buffer.device.type != 'meta':
            continue
        if hasattr(module, 'rope_freq_dim'):
            theta = float(getattr(config, 'rope_theta', 10000.0)) if config is not None else 10000.0
            table = _rope_frequencies(int(module.rope_freq_dim), theta)
        elif config is not None:
            table = _vae_rope_frequencies(config, module)
        else:
            table = None
        if table is None or tuple(table.shape) != tuple(buffer.shape):
            continue
        module.register_buffer('inv_freq', table, persistent=False)


def _materialize_meta_tensors(model) -> list[str]:
    """Give empty parameters and buffers storage so the module can move devices.

    A pruned checkpoint has no ``time_embedder``. Those parameters stay meta,
    and moving the module then fails. The curve replaces that embedder after
    the load. Rotary frequencies are computed again, not left as zeros.
    """
    import accelerate

    _restore_ropes(model)
    missing = []
    named = list(model.named_parameters(recurse=True)) + list(model.named_buffers(recurse=True))
    for name, tensor in named:
        if tensor.device.type != 'meta':
            continue
        value = torch.zeros(tensor.shape, dtype=tensor.dtype)
        accelerate.utils.set_module_tensor_to_device(
            model, name, 'cpu', value=value, dtype=tensor.dtype)
        missing.append(name)
    if missing:
        _messages.debug_log(
            f'MiniMax checkpoint left {len(missing)} tensors empty '
            f'(for example {missing[0]}). They now have storage.')
    return missing


def prepare_device_move(pipe) -> None:
    """Restore empty rotary tables before a MiniMax module is moved.

    The image and video VAEs build that table in ``__init__`` and do not save
    it. A low-memory load leaves it empty, and ``.to()`` then fails. Other
    empty tensors are left alone: filling them with zeros would decode from
    a checkpoint whose names did not match the module.
    """
    modules = []
    if isinstance(pipe, torch.nn.Module):
        modules.append(pipe)
    for module in _pipelines.get_pipeline_modules(pipe).values():
        if isinstance(module, torch.nn.Module):
            modules.append(module)
    seen = set()
    for module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        _restore_ropes(module)


@contextlib.contextmanager
def keep_missing_weights_addressable():
    """Let a single-file load move a model that the checkpoint did not fully fill.

    Diffusers binds ``dispatch_model`` when its loader is imported, so replacing
    Accelerate's function later does not change the call that moves the module.
    """
    import diffusers.loaders.single_file_model as single_file

    original = single_file.dispatch_model

    def dispatch_model(model, *args, **kwargs):
        _materialize_meta_tensors(model)
        return original(model, *args, **kwargs)

    single_file.dispatch_model = dispatch_model
    try:
        yield
    finally:
        single_file.dispatch_model = original


def install_cached_curve(model) -> None:
    """Attach a pruned timestep table stashed while the checkpoint was renamed."""
    table = getattr(convert_minimax_h3_checkpoint, 'last_curve_table', None)
    convert_minimax_h3_checkpoint.last_curve_table = None
    if not torch.is_tensor(table):
        return
    _install_curve(model, _orient_table(table).detach().float().contiguous())


def _bind_packed_linear(linear) -> None:
    """Decode a GGUF weight inside ``forward``, after CPU offload has loaded it.

    Offload parks the packed bytes and only puts them back when ``forward``
    starts. Checking the weight before that call still sees the empty meta
    tensor, so the decode has to live in ``forward`` itself.
    """
    if getattr(linear, '_dgenerate_packed_linear', False):
        return

    def forward(self, x):
        import dgenerate.pipelinewrapper.ggufcompat as _gguf

        weight = _gguf.as_float_tensor(self.weight)
        bias = _gguf.as_float_tensor(self.bias, weight.dtype)
        return torch.nn.functional.linear(x.to(dtype=weight.dtype), weight, bias)

    linear.forward = forward.__get__(linear, type(linear))
    linear._dgenerate_packed_linear = True


def _install_curve(model, table: torch.Tensor) -> None:
    """Run AdaLN from the curve table instead of ``time_embedder``."""
    import torch.nn as nn

    model.register_buffer('adaln_t_table', table, persistent=False)

    class _CurveEmbed(nn.Module):
        def __init__(self, curve):
            super().__init__()
            # A reference to the parent module would be registered as a child
            # and state_dict() would recurse through the transformer forever.
            self.register_buffer('table', curve, persistent=False)

        def forward(self, timestep):
            return _interpolate_curve(self.table, timestep)

    model.time_proj = nn.Identity()
    model.time_embedder = _CurveEmbed(table)

    def modulation_forward(self, temb):
        projected = self.linear(temb)
        projected = projected.view(-1, 6 * self.hidden_size)
        return projected.chunk(6, dim=-1)

    def output_forward(self, hidden_states, temb, timestep_indices):
        shift, scale = self.linear(temb).chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices)

    for block in model.transformer_blocks:
        packed = getattr(block.adaln_proj, 'linear', None)
        if packed is not None:
            _bind_packed_linear(packed)
        block.adaln_proj.forward = modulation_forward.__get__(block.adaln_proj, type(block.adaln_proj))
    packed = getattr(model.norm_out, 'linear', None)
    if packed is not None:
        _bind_packed_linear(packed)
    model.norm_out.forward = output_forward.__get__(model.norm_out, type(model.norm_out))
    _messages.debug_log(
        f'MiniMax-H3 pruned timestep curve loaded ({table.shape[0]} steps, '
        f'basis {table.shape[1]}).')


def build_references(items):
    """Ordered ``reference=`` media as MiniMax-H3 reference objects.

    A string is a file path. A dict is media dgenerate already decoded and
    preprocessed: ``kind``, ``frames``, and, for a video, ``fps``. ``path``
    on a video keeps the file's soundtrack. ``honor_size`` asks the pipeline
    to encode those pixels instead of scaling an image to a 2048 short edge
    or a video onto the 768 canvas.

    The second value is that canvas plan. ``None`` entries keep the checkpoint
    rule. Image values are short edges. Video values are ``edge`` and ``pixels``.
    """
    import numpy

    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )

    built = []
    plan = {'images': [], 'videos': []}
    for item in items:
        if isinstance(item, str):
            kind = reference_kind(item)
            if kind == 'audio':
                built.append(MiniMaxH3AudioReference.from_file(item))
            elif kind == 'video':
                built.append(MiniMaxH3VideoReference.from_file(item))
                plan['videos'].append(None)
            else:
                built.append(MiniMaxH3ImageReference.from_file(item))
                plan['images'].append(None)
            continue
        kind = item['kind']
        honor = bool(item.get('honor_size'))
        if kind == 'audio':
            built.append(MiniMaxH3AudioReference.from_file(item['path']))
        elif kind == 'image':
            image = item['frames'][0]
            built.append(MiniMaxH3ImageReference(image=image))
            plan['images'].append(min(image.size) if honor else None)
        else:
            frames = item['frames']
            array = numpy.stack([numpy.asarray(frame.convert('RGB')) for frame in frames])
            if item.get('path'):
                reference = MiniMaxH3VideoReference.from_file(item['path'])
                reference.frames = array
                if item.get('fps'):
                    reference.fps = float(item['fps'])
            else:
                reference = MiniMaxH3VideoReference(
                    frames=array, fps=item.get('fps'))
            built.append(reference)
            if honor:
                height, width = array.shape[1], array.shape[2]
                plan['videos'].append({
                    'edge': min(height, width),
                    'pixels': height * width,
                })
            else:
                plan['videos'].append(None)
    return built, plan


class _ImageShortEdge:
    """One short edge per image reference. ``None`` keeps the checkpoint edge."""

    def __init__(self, edges, default):
        self._edges = list(edges)
        self._default = default
        self._index = 0

    def __truediv__(self, other):
        edge = self._edges[self._index] if self._index < len(self._edges) else None
        self._index += 1
        if edge is None:
            return self._default / other
        return edge / other


@contextlib.contextmanager
def honor_reference_canvas(pipe, plan):
    """Encode preprocessed references at the size dgenerate handed in.

    An image scales by 1, then snaps to a multiple of 32. A video uses its own
    short edge instead of 768. References that were not preprocessed keep the
    checkpoint canvas.
    """
    images = list((plan or {}).get('images') or ())
    videos = list((plan or {}).get('videos') or ())
    if not any(edge is not None for edge in images) and not any(videos):
        yield
        return

    import diffusers.modular_pipelines.minimax_h3.before_encoder as _encoder

    step = _encoder.MiniMaxH3Ref2VASetupStep
    original_video = step._normalize_video_condition
    config = pipe.config
    original_edge = config.reference_image_short_edge
    pending = list(videos)

    def _normalize_video(frames, fps, num_frames, canvas_multiple,
                         canvas_short_edge, canvas_max_pixels, target_fps):
        spec = pending.pop(0) if pending else None
        if spec:
            canvas_short_edge = spec['edge']
            canvas_max_pixels = max(int(canvas_max_pixels), int(spec['pixels']))
        return original_video(
            frames, fps, num_frames, canvas_multiple,
            canvas_short_edge, canvas_max_pixels, target_fps)

    if any(edge is not None for edge in images):
        config.reference_image_short_edge = _ImageShortEdge(images, original_edge)
    step._normalize_video_condition = staticmethod(_normalize_video)
    try:
        yield
    finally:
        config.reference_image_short_edge = original_edge
        step._normalize_video_condition = original_video


def _quantized(module) -> bool:
    if module is None:
        return False
    name = type(module).__name__.lower()
    return 'gguf' in name or 'sdnq' in name or 'bitsandbytes' in name or 'quant' in name


def _place(pipe, device, offload: bool):
    """Put the text encoder and denoiser on ``device``, or hook them for one visit each."""
    import accelerate

    prepare_device_move(pipe)

    heavy = ('text_encoder', 'transformer', 'transformer_ref')
    if offload:
        _messages.warning(
            'minimax-h3-video uses one placement for --model-cpu-offload, '
            '--model-sequential-offload, and --model-group-offload. '
            'The text encoder and transformer visit the GPU for each forward '
            'and return to CPU.')
    for name, module in _pipelines.get_pipeline_modules(pipe).items():
        if module is None or not hasattr(module, 'to'):
            continue
        # A pruned curve file has no time embedder. Those parameters stay on
        # the meta device, and cpu_offload copies the state dict.
        _materialize_meta_tensors(module)
        if offload and name in heavy and not _pipelines._module_is_8bit_bnb(module):
            accelerate.cpu_offload(module, device, offload_buffers=len(module._parameters) > 0)
            _pipelines._set_cpu_offload_flag(module, True)
            continue
        if _pipelines.module_skips_group_offload(module):
            _pipelines.place_quantized_module(module, device)
        else:
            module.to(device)
    if offload:
        _pipelines._set_cpu_offload_flag(pipe, True)


def _quantized_denoiser_lora_plan(lora_uris, quantizer_uri, transformer_uri,
                                  lora_disk_fuse, denoiser: str, quantizer_map) -> str:
    """How a repo denoiser quantized with ``--quantizer`` takes ``--loras``.

    ``fuse`` writes a full-precision copy before quantization.
    ``refuse`` is that same request without ``--lora-disk-fuse``.
    ``attach`` is every other case, including a GGUF or Comfy ``--transformer``.
    """
    import dgenerate.pipelinewrapper.videopipelines.quantization as _quantization

    if (not lora_uris or transformer_uri or not quantizer_uri
            or not _quantization._quantize_component(denoiser, quantizer_uri, quantizer_map)):
        return 'attach'
    if lora_disk_fuse:
        return 'fuse'
    return 'refuse'


def _has_packed_linear(module) -> bool:
    """True when a denoiser keeps its linears packed (GGUF or Comfy quant)."""
    if module is None:
        return False
    for child in module.modules():
        if type(child).__name__ in ('GGUFLinear', 'ComfyQuantLinear'):
            return True
    return False


def _attach_lora_delta(module, down: torch.Tensor, up: torch.Tensor, scale: float) -> None:
    """Add ``scale * up @ down @ x`` after a packed linear's own forward.

    The base weight stays packed. Only the low-rank factors are dense, and
    they are moved with the activation for that call.
    """
    previous = module.forward
    down = down.detach().float().contiguous()
    up = up.detach().float().contiguous()

    def forward(x, _previous=previous, _down=down, _up=up, _scale=scale):
        result = _previous(x)
        if _scale == 0.0:
            return result
        dtype = result.dtype
        device = result.device
        hidden = torch.nn.functional.linear(x, _down.to(device=device, dtype=dtype))
        delta = torch.nn.functional.linear(hidden, _up.to(device=device, dtype=dtype))
        return result + delta * _scale

    module.forward = forward


def _apply_packed_loras(pipe, denoiser_name: str, lora_uris, fuse_scale, auth_token, local_files_only) -> None:
    """Apply MiniMax-H3 LoRAs to a packed denoiser without expanding it."""
    import dgenerate.hfhub as _hfhub
    import dgenerate.pipelinewrapper.uris as _uris
    from diffusers.loaders.lora_pipeline import MiniMaxH3LoraLoaderMixin

    denoiser = getattr(pipe, denoiser_name, None)
    if denoiser is None:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'minimax-h3-video has no {denoiser_name} to load a LoRA into.')
    fuse = 1.0 if fuse_scale is None else float(fuse_scale)
    matched = 0
    for raw in lora_uris:
        parsed = raw if isinstance(raw, _uris.LoRAUri) else _uris.LoRAUri.parse(raw)
        model_path = _hfhub.download_non_hf_slug_model(
            parsed.model, local_files_only=local_files_only, auth_token=auth_token)
        state, metadata = MiniMaxH3LoraLoaderMixin.lora_state_dict(
            model_path,
            weight_name=parsed.weight_name,
            revision=parsed.revision,
            subfolder=parsed.subfolder,
            local_files_only=local_files_only,
            token=auth_token,
            return_lora_metadata=True)
        alpha = None
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if str(key).endswith('lora_alpha') and isinstance(value, (int, float)):
                    alpha = float(value)
                    break
        pairs = {}
        for key, value in state.items():
            body = key.removeprefix('transformer.')
            if body.endswith('.lora_A.weight'):
                slot = body[:-len('.lora_A.weight')]
                pairs.setdefault(slot, {})['down'] = value
            elif body.endswith('.lora_B.weight'):
                slot = body[:-len('.lora_B.weight')]
                pairs.setdefault(slot, {})['up'] = value
        for slot, pair in pairs.items():
            down, up = pair.get('down'), pair.get('up')
            if down is None or up is None:
                continue
            try:
                layer = denoiser.get_submodule(slot)
            except AttributeError:
                _messages.debug_log(
                    f'MiniMax-H3 LoRA has no packed layer for "{slot}".')
                continue
            rank = int(down.shape[0])
            network_alpha = float(rank) if alpha is None else alpha
            scale = float(parsed.scale) * fuse * (network_alpha / rank)
            _attach_lora_delta(layer, down, up, scale)
            matched += 1
    if matched == 0:
        raise _pipelines.UnsupportedPipelineConfigError(
            'This MiniMax-H3 LoRA did not match any packed linear on '
            f'{denoiser_name}.')
    _messages.debug_log(
        f'Applied a MiniMax-H3 LoRA to {matched} packed linears on {denoiser_name}.')


def create_pipeline(model_path,
                    model_type,
                    revision,
                    variant,
                    subfolder,
                    dtype,
                    device,
                    model_cpu_offload,
                    sequential_cpu_offload,
                    model_group_offload,
                    torch_compile,
                    local_files_only,
                    auth_token,
                    transformer_uri=None,
                    lora_uris=None,
                    lora_fuse_scale=None,
                    quantizer_uri=None,
                    quantizer_map=None,
                    vae_uri=None,
                    text_encoder_uris=None,
                    minimax_workflow='t2va',
                    lora_disk_fuse=False,
                    **_ignored):
    """Load one MiniMax-H3 workflow and return ``(held, cache metadata)``."""
    import diffusers

    import dgenerate.hfhub as _hfhub
    import dgenerate.pipelinewrapper.enums as _enums
    import dgenerate.pipelinewrapper.singlefilecompat as _singlefilecompat
    import dgenerate.pipelinewrapper.uris as _uris
    import dgenerate.pipelinewrapper.videopipelines.loading as _loading
    import dgenerate.pipelinewrapper.videopipelines.lora as _lora
    import dgenerate.pipelinewrapper.videopipelines.quantization as _quantization
    import dgenerate.memoize as _d_memoize

    workflow = minimax_workflow or 't2va'
    if workflow not in ('t2va', 'fl2va', 'ref2va'):
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Unknown minimax-h3-video workflow {workflow!r}.')
    denoiser = denoiser_name(workflow)
    lora_plan = _quantized_denoiser_lora_plan(
        lora_uris, quantizer_uri, transformer_uri, lora_disk_fuse, denoiser, quantizer_map)
    if lora_plan == 'refuse':
        raise _pipelines.UnsupportedPipelineConfigError(
            '--loras with --quantizer on minimax-h3-video needs --lora-disk-fuse. '
            'That fuses the LoRA into a full-precision copy and caches it before quantization.')

    aux_repo = None
    if _hfhub.is_single_file_model_load(model_path):
        aux_repo = _singlefilecompat.single_file_pretrained_repo_for_model_path(
            model_path,
            auth_token=auth_token,
            local_files_only=local_files_only,
            model_type=model_type)
        if aux_repo is None:
            raise _pipelines.UnsupportedPipelineConfigError(
                'This file is not a MiniMax-H3 transformer. '
                'Pass MiniMaxAI/MiniMax-H3 as the model and the file as --transformer.')
        if not transformer_uri:
            transformer_uri = model_path
    repo = aux_repo or model_path
    if dtype is _enums.DataType.AUTO or dtype is None:
        dtype = _enums.DataType.BFLOAT16
    model_torch = _enums.get_torch_dtype(dtype) or torch.bfloat16
    offload = bool(model_cpu_offload or sequential_cpu_offload or model_group_offload)

    index = {
        'text_encoder': ['transformers', 'Qwen3VLForConditionalGeneration'],
        denoiser: ['diffusers', 'MiniMaxH3Transformer3DModel'],
    }
    injected = {}
    te_uris = list(text_encoder_uris or ())
    _loading._check_video_text_encoder_uris(index, te_uris)
    for name, module in _loading._inject_video_text_encoders(
            index, te_uris, dtype, auth_token, local_files_only, device, offload).items():
        if module is None:
            raise _pipelines.UnsupportedPipelineConfigError(
                'minimax-h3-video requires its Qwen3-VL text encoder. '
                'Do not pass null in --text-encoders.')
        injected[name] = module

    if (quantizer_uri
            and 'text_encoder' not in injected
            and _quantization._quantize_component('text_encoder', quantizer_uri, quantizer_map)):
        from transformers import Qwen3VLForConditionalGeneration
        injected['text_encoder'] = _quantization._load_quantized_module(
            Qwen3VLForConditionalGeneration, repo, 'text_encoder', revision, variant, dtype,
            quantizer_uri, auth_token, local_files_only,
            _quantization._quantized_device_map(device, offload, quantizer_uri),
            offload=offload)

    if lora_plan == 'fuse':
        import diffusers.modular_pipelines.minimax_h3 as _minimax_pipe
        fuse_kwargs = {
            'local_files_only': local_files_only,
            'torch_dtype': model_torch,
            'workflow': workflow,
        }
        if revision:
            fuse_kwargs['revision'] = revision
        if auth_token:
            fuse_kwargs['token'] = auth_token
        if subfolder and not aux_repo:
            fuse_kwargs['subfolder'] = subfolder
        if 'text_encoder' in injected:
            fuse_kwargs['text_encoder'] = injected['text_encoder']
        fused_paths, fused_manifest = _lora._preflight_quantized_video_loras(
            _minimax_pipe.MiniMaxH3ModularPipeline, repo, fuse_kwargs, injected,
            model_type, list(lora_uris), lora_fuse_scale, auth_token, local_files_only,
            'minimax-h3-video', components=(denoiser,))
        injected[denoiser] = _lora._load_fused_video_component(
            denoiser, fused_paths[denoiser], fused_manifest[denoiser], dtype,
            quantizer_uri, auth_token, local_files_only, device, offload)
    elif transformer_uri:
        _warn_transformer_partition(transformer_uri, workflow)
        injected[denoiser] = _loading._load_replacement_transformer(
            model_type, transformer_uri, dtype, variant, auth_token, local_files_only,
            quantizer_uri=quantizer_uri, quantizer_map=quantizer_map, device=device,
            offload=offload, component_name=denoiser, config_repo=repo,
            family='minimax-h3-video')
    elif quantizer_uri and _quantization._quantize_component(denoiser, quantizer_uri, quantizer_map):
        injected[denoiser] = _quantization._load_quantized_module(
            diffusers.MiniMaxH3Transformer3DModel, repo, denoiser, revision, variant, dtype,
            quantizer_uri, auth_token, local_files_only,
            _quantization._quantized_device_map(device, offload, quantizer_uri),
            offload=offload)

    if vae_uri:
        parsed_vae = _uris.VAEUri.parse(vae_uri)
        vae_dtype = dtype if parsed_vae.dtype is not None else _enums.DataType.FLOAT32
        injected['vae'] = parsed_vae.load(
            dtype_fallback=vae_dtype,
            auth_token=auth_token,
            local_files_only=local_files_only)

    load_kwargs = {'local_files_only': local_files_only}
    if revision:
        load_kwargs['revision'] = revision
    if auth_token:
        load_kwargs['token'] = auth_token
    if subfolder and not aux_repo:
        load_kwargs['subfolder'] = subfolder
    pipe = diffusers.ModularPipeline.from_pretrained(repo, workflow=workflow, **load_kwargs)
    modules = {
        name: value for name, value in injected.items()
        if value is not None and name in getattr(pipe, '_component_specs', {})}
    if modules:
        pipe.update_components(**modules)

    component_kwargs = dict(load_kwargs)
    if variant:
        component_kwargs['variant'] = variant

    def unloaded(name):
        spec = pipe._component_specs.get(name)
        if spec is None or spec.default_creation_method != 'from_pretrained':
            return False
        if getattr(pipe, name, None) is not None:
            return False
        return True

    if unloaded('vae'):
        vae_kwargs = dict(component_kwargs)
        vae_kwargs['torch_dtype'] = torch.float32
        pipe.load_components(names=['vae'], **vae_kwargs)
    elif getattr(pipe, 'vae', None) is not None and not _quantized(pipe.vae):
        pipe.vae.to(dtype=torch.float32)
    remaining = [name for name in pipe._component_specs if unloaded(name)]
    if remaining:
        pipe.load_components(names=remaining, torch_dtype=model_torch, **component_kwargs)

    if lora_uris and lora_plan != 'fuse':
        if _has_packed_linear(getattr(pipe, denoiser, None)):
            _apply_packed_loras(
                pipe, denoiser, list(lora_uris), lora_fuse_scale, auth_token, local_files_only)
        else:
            _lora._apply_video_loras(
                pipe, model_type, list(lora_uris), lora_fuse_scale, auth_token, local_files_only)

    compile_on = _pipelines.torch_compile_enabled(torch_compile)
    if compile_on or _pipelines._compile_requested_names(pipe):
        _pipelines.apply_torch_compile(
            pipe, device,
            names=_pipelines.torch_compile_module_names(torch_compile) if compile_on else ())
    _place(pipe, device, offload)
    _quantization._pin_video_8bit_modules(pipe)
    _loading._enable_vae_tiling(pipe)
    audio_vae = getattr(pipe, 'audio_vae', None)
    if audio_vae is not None and hasattr(audio_vae, 'enable_tiling'):
        audio_vae.enable_tiling()

    estimate = 0
    counted = ['text_encoder', 'vae', 'audio_vae']
    if denoiser not in injected:
        counted.append(denoiser)
    for name in counted:
        estimate += _loading._weight_bytes(
            repo, auth_token, local_files_only,
            revision=revision or 'main', variant=variant, directory=name)
    if transformer_uri:
        estimate += _loading._parsed_weight_bytes(
            _uris.TransformerUri, transformer_uri, auth_token, local_files_only)
    held = _loading._VideoPipeline(pipe, 'minimax-h3-video', model_mode=workflow)
    return held, _d_memoize.CachedObjectMetadata(size=max(estimate, 1))
