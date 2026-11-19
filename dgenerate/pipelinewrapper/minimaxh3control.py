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

"""MiniMax-H3 Fun control branch loaded through ``--control-nets``.

The published file is not a Diffusers ControlNet. It is ``control_proj_in`` plus
a stack of transformer blocks. Those blocks run on the packed sequence with the
control video written over the video rows, and each block adds a skip into the
main stack. Union 2.0 uses ten blocks at layers 0, 5, ..., 45 and a 49-channel
input so the same file also reads an inpaint mask.
"""

from __future__ import annotations

import inspect
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.pipelines as _pipelines

_BLOCK_INDEX = re.compile(r'^control_blocks\.(\d+)\.')
_PLACES = {
    5: (0, 10, 20, 30, 40),
    10: (0, 5, 10, 15, 20, 25, 30, 35, 40, 45),
}
# Five-block Union zeroes masked pixels before ImageNet normalization.
# Ten-block Union 2.0 zeroes them after, so the holes sit at 0.
_MASK_MODE = {5: 'pre_norm', 10: 'post_norm'}


def _control_block_class():
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock

    class _ControlBlock(MiniMaxH3TransformerBlock):
        """A base block plus the skip projections the control file stores on it."""

        def __init__(self, block_id: int, **kwargs):
            super().__init__(**kwargs)
            self.block_id = block_id
            hidden = int(kwargs['hidden_size'])
            if block_id == 0:
                self.before_proj = nn.Linear(hidden, hidden)
            self.after_proj = nn.Linear(hidden, hidden)

        def run(self, stream, packed, temb, adaln_indices, rotary_emb):
            if self.block_id == 0:
                current = self.before_proj(stream) + packed
                carried = []
            else:
                carried = list(torch.unbind(stream))
                current = carried.pop(-1)
            current = super().forward(current, temb, adaln_indices, rotary_emb)
            skip = self.after_proj(current)
            carried.extend((skip, current))
            return torch.stack(carried)

    return _ControlBlock


def branch_layout(keys) -> tuple[tuple[int, ...], str]:
    """``(layer places, mask mode)`` read from a control-branch state dict."""
    indices = set()
    for key in keys:
        match = _BLOCK_INDEX.match(key)
        if match:
            indices.add(int(match.group(1)))
    if not indices:
        raise _pipelines.UnsupportedPipelineConfigError(
            'This --control-nets file has no MiniMax-H3 control blocks.')
    count = max(indices) + 1
    if indices != set(range(count)) or count not in _PLACES:
        raise _pipelines.UnsupportedPipelineConfigError(
            'This --control-nets file is not MiniMax-H3 Fun ControlNet Union '
            f'or Union 2.0. Found control blocks {sorted(indices)}.')
    if 'control_proj_in.weight' not in keys:
        raise _pipelines.UnsupportedPipelineConfigError(
            'This --control-nets file has no control_proj_in weight.')
    return _PLACES[count], _MASK_MODE[count]


def checkpoint_time_embed_dim(state: dict) -> int:
    """AdaLN input width stored on the control branch.

    A pruned curve transformer narrows this to 8. The control file keeps the
    full width, 2688 on the released checkpoint.
    """
    weight = state.get('control_blocks.0.adaln_proj.linear.weight')
    if weight is None or weight.ndim != 2:
        raise _pipelines.UnsupportedPipelineConfigError(
            'This --control-nets file has no control block AdaLN projection.')
    return int(weight.shape[1])


def control_in_dim_from_weight(weight: torch.Tensor, patch_size) -> int:
    """Channel count of ``control_proj_in`` from its input columns."""
    patch = int(patch_size[0]) * int(patch_size[1]) * int(patch_size[2])
    columns = int(weight.shape[1])
    if patch <= 0 or columns % patch:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'control_proj_in has {columns} input columns, which is not a multiple '
            f'of the patch {tuple(patch_size)}.')
    return columns // patch


def attach_control_branch(transformer, state: dict, source: str, dtype=None,
                          model_path: str | None = None, revision: str | None = None,
                          auth_token: str | None = None, local_files_only: bool = False) -> None:
    """Load the control branch onto a MiniMax-H3 transformer.

    A later call with the same ``source`` keeps the weights. A different file
    replaces them. The base weights stay as they were.
    """
    if not hasattr(transformer, 'transformer_blocks') or not hasattr(transformer, 'proj_in'):
        raise _pipelines.UnsupportedPipelineConfigError(
            'This MiniMax-H3 transformer cannot take a Fun control branch.')
    config = transformer.config
    patch_size = tuple(config.patch_size)
    foreign = [key for key in state if not key.startswith('control_')]
    if foreign:
        raise _pipelines.UnsupportedPipelineConfigError(
            'This --control-nets file is not a MiniMax-H3 control branch. '
            f'It also contains {foreign[:4]}.')
    places, mask_mode = branch_layout(state.keys())
    weight = state.get('control_proj_in.weight')
    if weight is None:
        raise _pipelines.UnsupportedPipelineConfigError(
            'This --control-nets file has no control_proj_in weight.')
    control_in_dim = control_in_dim_from_weight(weight, patch_size)
    if int(weight.shape[0]) != int(config.hidden_size):
        raise _pipelines.UnsupportedPipelineConfigError(
            'control_proj_in does not match this transformer. '
            f'It writes {tuple(weight.shape)} and the hidden size is {config.hidden_size}.')
    if places[-1] >= int(config.num_layers):
        raise _pipelines.UnsupportedPipelineConfigError(
            f'The control branch addresses layer {places[-1]}, and this transformer '
            f'has {config.num_layers} layers.')

    previous = getattr(transformer, '_dgenerate_control_source', None)
    if previous == source and hasattr(transformer, 'control_blocks'):
        return
    if hasattr(transformer, 'control_blocks'):
        del transformer.control_blocks
        del transformer.control_proj_in

    example = transformer.transformer_blocks[0]
    adaln_file = state['control_blocks.0.adaln_proj.linear.weight']
    block_dtype = dtype or (
        adaln_file.dtype if adaln_file.is_floating_point() else torch.bfloat16)
    proj_dtype = dtype or (weight.dtype if weight.is_floating_point() else torch.float32)
    device = example.adaln_proj.linear.weight.device
    if device.type == 'meta':
        device = torch.device('cpu')
    time_dim = checkpoint_time_embed_dim(state)
    block_cls = _control_block_class()
    block_kwargs = dict(
        hidden_size=int(config.hidden_size),
        num_attention_heads=int(config.num_attention_heads),
        attention_head_dim=int(config.attention_head_dim),
        ffn_dim=int(config.ffn_dim),
        time_embed_dim=time_dim,
        norm_eps=float(config.norm_eps),
        qk_norm_eps=float(config.qk_norm_eps),
    )
    blocks = nn.ModuleList(
        block_cls(block_id, **block_kwargs) for block_id, _place in enumerate(places))
    proj = nn.Linear(
        control_in_dim * patch_size[0] * patch_size[1] * patch_size[2],
        int(config.hidden_size),
        bias=True,
    )
    transformer.control_blocks = blocks.to(device=device, dtype=block_dtype)
    transformer.control_proj_in = proj.to(device=device, dtype=proj_dtype)
    missing, unexpected = transformer.load_state_dict(state, strict=False)
    missing_control = [key for key in missing if key.startswith('control_')]
    if unexpected or missing_control:
        raise _pipelines.UnsupportedPipelineConfigError(
            'The --control-nets file does not match this MiniMax-H3 control branch. '
            f'Unexpected keys: {unexpected[:8]}. Missing keys: {missing_control[:8]}.')
    transformer._dgenerate_control_places = places
    transformer._dgenerate_control_in_dim = control_in_dim
    transformer._dgenerate_control_mask_mode = mask_mode
    transformer._dgenerate_control_apply_audio = False
    if time_dim != int(config.time_embed_dim):
        _attach_full_time_embedder(
            transformer, config, time_dim, device,
            model_path, revision, auth_token, local_files_only)
    transformer._dgenerate_control_source = source
    if not getattr(transformer, '_dgenerate_control_rows', None):
        transformer._dgenerate_control_rows = None
    transformer._dgenerate_control_scale = 1.0
    _install_forward(transformer)
    _messages.debug_log(
        f'Loaded a MiniMax-H3 control branch from "{source}" '
        f'({len(places)} blocks, control_in_dim={control_in_dim}, {mask_mode}).')


def _module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


def _place_control_modules(transformer, device) -> None:
    """Move the control branch onto the activation device.

    It is attached after CPU offload hooks the transformer, so the hook's
    weight map does not include it. The timestep still arrives on the GPU.
    """
    if _module_device(transformer.control_blocks) != device:
        transformer.control_blocks.to(device)
    proj = transformer.control_proj_in
    if proj.weight.device != device:
        proj.to(device)
    embedder = getattr(transformer, 'control_time_embedder', None)
    if embedder is not None and _module_device(embedder) != device:
        embedder.to(device)
        transformer.control_time_proj.to(device)


def _timestep_embedding(proj, embedder, timestep: torch.Tensor) -> torch.Tensor:
    from diffusers.models.modeling_utils import get_parameter_dtype

    device = _module_device(embedder)
    sample = proj(timestep if device is None else timestep.to(device))
    try:
        dtype = get_parameter_dtype(embedder)
    except Exception:
        dtype = sample.dtype
    if device is None:
        return embedder(sample.to(dtype))
    return embedder(sample.to(device=device, dtype=dtype))


_TIME_EMBEDDER_KEYS = (
    'time_embedder.linear_1.weight',
    'time_embedder.linear_1.bias',
    'time_embedder.linear_2.weight',
    'time_embedder.linear_2.bias',
)


def _attach_full_time_embedder(transformer, config, time_dim: int, device,
                               model_path, revision, auth_token, local_files_only) -> None:
    """Give the control branch the full timestep MLP a curve checkpoint removed."""
    if getattr(transformer, 'control_time_embedder', None) is not None:
        return
    if not model_path:
        raise _pipelines.UnsupportedPipelineConfigError(
            'This transformer uses the pruned timestep curve. The control branch '
            'needs the full time embedder from the MiniMax-H3 repository. '
            'Pass that repository as the model path.')
    _messages.debug_log(
        'This transformer uses the pruned timestep curve. '
        f'The control branch reads the full time embedder from "{model_path}".')
    tensors = _read_time_embedder(model_path, revision, auth_token, local_files_only)
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps

    freq = int(getattr(config, 'freq_dim', 256))
    hidden = int(getattr(config, 'time_embed_hidden_dim', 5376))
    embedder = TimestepEmbedding(in_channels=freq, time_embed_dim=hidden, out_dim=time_dim)
    mapped = {key.split('.', 1)[1]: value for key, value in tensors.items()}
    try:
        embedder.load_state_dict(mapped)
    except RuntimeError as error:
        raise _pipelines.UnsupportedPipelineConfigError(
            'The model repository time embedder does not match this control branch. '
            f'{error}') from error
    transformer.control_time_proj = Timesteps(
        num_channels=freq, flip_sin_to_cos=True, downscale_freq_shift=0).to(device)
    transformer.control_time_embedder = embedder.to(device=device, dtype=torch.float32)


def _read_time_embedder(model_path, revision, auth_token, local_files_only) -> dict:
    import json

    import safetensors

    import dgenerate.hfhub as _hfhub

    subfolder = 'transformer'
    filename = 'diffusion_pytorch_model.safetensors'
    try:
        index_path, _ignored = _hfhub.materialize_named_weight(
            model_path, weight_name='diffusion_pytorch_model.safetensors.index.json',
            revision=revision, subfolder=subfolder,
            local_files_only=local_files_only, auth_token=auth_token)
        with open(index_path, encoding='utf-8') as handle:
            filename = json.load(handle)['weight_map'][_TIME_EMBEDDER_KEYS[0]]
    except Exception:
        filename = 'diffusion_pytorch_model.safetensors'
    try:
        path, _ignored = _hfhub.materialize_named_weight(
            model_path, weight_name=filename, revision=revision, subfolder=subfolder,
            local_files_only=local_files_only, auth_token=auth_token)
    except Exception as error:
        raise _pipelines.UnsupportedPipelineConfigError(
            'The control branch needs the full time embedder, and it was not found in '
            f'"{model_path}" transformer/. A pruned curve file does not contain it.') from error
    with safetensors.safe_open(path, framework='pt', device='cpu') as handle:
        present = {}
        for key in handle.keys():
            plain = key
            for prefix in ('model.diffusion_model.', 'diffusion_model.'):
                if plain.startswith(prefix):
                    plain = plain[len(prefix):]
                    break
            if plain in _TIME_EMBEDDER_KEYS:
                present[plain] = handle.get_tensor(key)
    missing = [key for key in _TIME_EMBEDDER_KEYS if key not in present]
    if missing:
        raise _pipelines.UnsupportedPipelineConfigError(
            'The model repository transformer is missing the full time embedder '
            f'({", ".join(missing)}).')
    return present


def _install_forward(transformer) -> None:
    if getattr(transformer, '_dgenerate_control_forward_installed', False):
        return
    base = transformer._old_forward if hasattr(transformer, '_old_forward') else transformer.forward

    def runner(
            hidden_states,
            audio_hidden_states,
            encoder_hidden_states,
            timestep,
            timestep_indices,
            token_tags,
            position_ids,
            video_indices,
            audio_indices,
            text_indices,
            attention_kwargs=None,
            return_dict=True):
        rows = getattr(transformer, '_dgenerate_control_rows', None)
        if rows is None:
            return base(
                hidden_states, audio_hidden_states, encoder_hidden_states, timestep,
                timestep_indices, token_tags, position_ids, video_indices, audio_indices,
                text_indices, attention_kwargs=attention_kwargs, return_dict=return_dict)
        return _control_forward(
            transformer, hidden_states, audio_hidden_states, encoder_hidden_states, timestep,
            timestep_indices, token_tags, position_ids, video_indices, audio_indices,
            text_indices, rows, float(getattr(transformer, '_dgenerate_control_scale', 1.0)),
            attention_kwargs, return_dict)

    try:
        runner.__signature__ = inspect.signature(base)
    except (TypeError, ValueError):
        pass
    if hasattr(transformer, '_old_forward'):
        transformer._old_forward = runner
    else:
        transformer.forward = runner
    transformer._dgenerate_control_forward_installed = True


def _control_forward(
        transformer, hidden_states, audio_hidden_states, encoder_hidden_states, timestep,
        timestep_indices, token_tags, position_ids, video_indices, audio_indices,
        text_indices, control_rows, scale, attention_kwargs, return_dict):
    from diffusers.models.modeling_utils import get_parameter_dtype
    from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM
    from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

    kwargs = dict(attention_kwargs or {})
    lora_scale = kwargs.pop('scale', 1.0)
    if USE_PEFT_BACKEND:
        scale_lora_layers(transformer, lora_scale)
    try:
        if control_rows.shape[1] != video_indices.shape[0]:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'The control video has {control_rows.shape[1]} rows and the packed '
                f'sequence has {video_indices.shape[0]} video rows.')
        sequence_length = position_ids.shape[0]
        video = transformer.proj_in(hidden_states.to(get_parameter_dtype(transformer.proj_in)))
        audio = transformer.audio_proj_in(
            audio_hidden_states.to(get_parameter_dtype(transformer.audio_proj_in)))
        text = transformer.context_embedder(
            encoder_hidden_states.to(get_parameter_dtype(transformer.context_embedder)))
        text = transformer.token_refiner(text)
        packed = text.new_zeros((text.shape[0], sequence_length, text.shape[-1]))
        packed = packed.index_copy(1, text_indices, text)
        packed = packed.index_copy(1, video_indices, video.to(text.dtype))
        packed = packed.index_copy(1, audio_indices, audio.to(text.dtype))

        _place_control_modules(transformer, packed.device)
        main_temb = _timestep_embedding(transformer.time_proj, transformer.time_embedder, timestep)
        control_temb = main_temb
        if getattr(transformer, 'control_time_embedder', None) is not None:
            control_temb = _timestep_embedding(
                transformer.control_time_proj, transformer.control_time_embedder, timestep)
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
        rotary = transformer.rope(position_ids)

        # A Spectrum forecast already holds the last real block output, hints
        # included. Running the branch again would add them a second time.
        hints = None
        places = {}
        if getattr(transformer, '_dgenerate_spectrum_actual', True):
            proj = transformer.control_proj_in
            control = proj(control_rows.to(device=proj.weight.device, dtype=get_parameter_dtype(proj)))
            stream = packed.index_copy(1, video_indices, control.to(text.dtype))
            for block in transformer.control_blocks:
                stream = block.run(stream, packed, control_temb, adaln_indices, rotary)
            hints = torch.unbind(stream)[:-1]
            if (not transformer._dgenerate_control_apply_audio) and audio_indices.numel():
                keep = torch.ones(sequence_length, dtype=hints[0].dtype, device=hints[0].device)
                keep[audio_indices.to(keep.device)] = 0
                hints = tuple(hint * keep[None, :, None] for hint in hints)
            places = {
                layer: index for index, layer in enumerate(transformer._dgenerate_control_places)}
        hidden = packed
        for index, block in enumerate(transformer.transformer_blocks):
            hidden = block(hidden, main_temb, adaln_indices, rotary)
            hint_index = places.get(index)
            if hints is not None and hint_index is not None and scale:
                hint = hints[hint_index].to(device=hidden.device, dtype=hidden.dtype)
                hidden = hidden + hint * scale

        hidden = transformer.norm_out(hidden, main_temb, timestep_indices).to(
            get_parameter_dtype(transformer.proj_out))
        video_output = transformer.proj_out(hidden).index_select(1, video_indices)
        audio_output = transformer.audio_proj_out(hidden).index_select(1, audio_indices)
    finally:
        if USE_PEFT_BACKEND:
            unscale_lora_layers(transformer, lora_scale)
    if not return_dict:
        return video_output, audio_output
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerOutput
    return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)


def set_control_rows(transformer, rows: torch.Tensor | None, scale: float) -> None:
    """Hand the next forwards a control row tensor, or clear it."""
    transformer._dgenerate_control_rows = rows
    transformer._dgenerate_control_scale = float(scale)


def fit_clip(pixels: torch.Tensor, height: int, width: int, num_frames: int) -> torch.Tensor:
    """Pad or trim frames, then bilinear-resize onto the canvas."""
    count = pixels.shape[2]
    if count < num_frames:
        tail = pixels[:, :, -1:].expand(-1, -1, num_frames - count, -1, -1)
        pixels = torch.cat((pixels, tail), dim=2)
    elif count > num_frames:
        pixels = pixels[:, :, :num_frames]
    if pixels.shape[-2:] != (height, width):
        resized = F.interpolate(
            pixels[0].permute(1, 0, 2, 3), size=(height, width),
            mode='bilinear', align_corners=False)
        pixels = resized.permute(1, 0, 2, 3).unsqueeze(0)
    return pixels


def frames_to_tensor(frames, channels: int) -> torch.Tensor:
    """``(1, channels, frames, height, width)`` in ``[0, 1]``."""
    import numpy

    arrays = []
    for frame in frames:
        image = frame.convert('RGB' if channels == 3 else 'L')
        arrays.append(numpy.asarray(image, dtype=numpy.float32) / 255.0)
    stack = numpy.stack(arrays, axis=0)
    tensor = torch.from_numpy(stack)
    if channels == 1:
        tensor = tensor[..., None]
    return tensor.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def pad_control_rows(rows: torch.Tensor, control_in_dim: int, patch_size) -> torch.Tensor:
    """Zero-fill mask channels when the rows are a plain control video."""
    patch = int(patch_size[0]) * int(patch_size[1]) * int(patch_size[2])
    expected = int(control_in_dim) * patch
    width = rows.shape[-1]
    if width < expected:
        return F.pad(rows, (0, expected - width))
    if width > expected:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'The control rows have {width} columns and this branch expects {expected}.')
    return rows


def encode_control_rows(pipe, control_frames, inpaint_frames, mask_frames,
                        height: int, width: int, num_frames: int) -> torch.Tensor:
    """Patchified control rows for one denoising loop, shape ``(1, video_rows, columns)``."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import video_latent_num_frames

    transformer = pipe.transformer
    device = next(pipe.vae.parameters()).device
    patch_size = tuple(pipe.patch_size)
    mode = transformer._dgenerate_control_mask_mode
    control_in_dim = transformer._dgenerate_control_in_dim
    pixel_mean = torch.tensor(pipe.pixel_mean, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(pipe.pixel_std, device=device).view(1, -1, 1, 1, 1)
    latents_mean = torch.tensor(pipe.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(pipe.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)

    def _encode(pixels):
        autocast = torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda')
        with torch.no_grad(), autocast:
            encoded = pipe.vae.encode(pixels.to(device)).latent_dist.mode()
        return (encoded.float() - latents_mean) / latents_std

    rows = None
    if control_frames:
        pixels = fit_clip(frames_to_tensor(control_frames, 3), height, width, num_frames)
        pixels = (pixels.to(device) - pixel_mean) / pixel_std
        rows = patchify_video_latents(_encode(pixels), patch_size)
    if mask_frames:
        mask = fit_clip(frames_to_tensor(mask_frames, 1), height, width, num_frames)
        mask = (mask.to(device) > 0.5).to(torch.float32)
        if inpaint_frames:
            source = fit_clip(frames_to_tensor(inpaint_frames, 3), height, width, num_frames).to(device)
            if mode == 'post_norm':
                source = (source - pixel_mean) / pixel_std
                source = source * (1.0 - mask)
            else:
                source = source * (1.0 - mask)
                source = (source - pixel_mean) / pixel_std
        else:
            _messages.warning(
                'minimax-h3-video inpaint has a mask and no source video. '
                'The kept region carries no content.')
            source = torch.zeros(1, 3, num_frames, height, width, device=device)
            if mode != 'post_norm':
                source = (source - pixel_mean) / pixel_std
        masked = _encode(source)
        visible = F.interpolate(1.0 - mask, size=masked.shape[2:], mode='trilinear', align_corners=False)
        inpaint_rows = torch.cat((
            patchify_video_latents(visible, patch_size),
            patchify_video_latents(masked, patch_size),
        ), dim=-1)
        if rows is None:
            latent_frames = video_latent_num_frames(
                num_frames, pipe.vae_frames_per_chunk, pipe.vae_latents_per_chunk)
            latent_h = height // pipe.vae_spatial_compression_ratio
            latent_w = width // pipe.vae_spatial_compression_ratio
            video_rows = (
                (latent_frames // patch_size[0])
                * (latent_h // patch_size[1])
                * (latent_w // patch_size[2]))
            video_columns = pipe.vae_latent_channels * patch_size[0] * patch_size[1] * patch_size[2]
            rows = torch.zeros(video_rows, video_columns, device=inpaint_rows.device)
        rows = torch.cat((rows, inpaint_rows), dim=-1)
    if rows is None:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-video --control-nets needs a control video or an inpaint mask.')
    rows = pad_control_rows(rows, control_in_dim, patch_size)
    return rows.unsqueeze(0)


def quantize_control_branch(transformer, quantizer_uri: str, dtype=None) -> None:
    """Quantize the loaded control blocks. ``control_proj_in`` stays full precision.

    ``--quantizer-map controlnet`` or a ``quantizer`` argument on the
    ``--control-nets`` URI selects this. The same URI is not applied twice.
    """
    if not quantizer_uri:
        return
    previous = getattr(transformer, '_dgenerate_control_quantizer', None)
    if previous == quantizer_uri:
        return
    if previous:
        raise _pipelines.UnsupportedPipelineConfigError(
            'The MiniMax-H3 control branch is already quantized with '
            f'"{previous}".')
    blocks = getattr(transformer, 'control_blocks', None)
    if blocks is None:
        return
    parsed_name = str(quantizer_uri).split(';', 1)[0].strip().lower()
    try:
        if parsed_name in {'bnb', 'bitsandbytes'}:
            _quantize_control_bnb(blocks, quantizer_uri, dtype)
        elif parsed_name == 'sdnq':
            _quantize_control_sdnq(blocks, quantizer_uri)
        else:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'Unknown quantizer "{parsed_name}" for the MiniMax-H3 control branch. '
                'Use bnb or sdnq.')
    except _pipelines.UnsupportedPipelineConfigError:
        raise
    except Exception as error:
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Could not quantize the MiniMax-H3 control branch with "{quantizer_uri}". '
            f'{error}') from error
    transformer._dgenerate_control_quantizer = quantizer_uri
    _messages.debug_log(f'Quantized the MiniMax-H3 control branch with "{quantizer_uri}".')


def _quantize_control_sdnq(blocks, quantizer_uri: str) -> None:
    from sdnq.quantizer import sdnq_post_load_quant

    import dgenerate.pipelinewrapper.uris as _uris

    config = _uris.SDNQQuantizerUri.parse(quantizer_uri).to_config()
    sdnq_post_load_quant(blocks, quantization_config=config, add_skip_keys=False)


def _quantize_control_bnb(blocks, quantizer_uri: str, dtype) -> None:
    import bitsandbytes as bnb

    import dgenerate.pipelinewrapper.uris as _uris

    config = _uris.BNBQuantizerUri.parse(quantizer_uri).to_config(dtype)
    _replace_linears_with_bnb(blocks, config)


def _replace_linears_with_bnb(module, config) -> None:
    import bitsandbytes as bnb

    for name, child in list(module.named_children()):
        if type(child) is not nn.Linear:
            _replace_linears_with_bnb(child, config)
            continue
        has_bias = child.bias is not None
        weight = child.weight.detach()
        if config.load_in_8bit:
            new = bnb.nn.Linear8bitLt(
                child.in_features, child.out_features, bias=has_bias, has_fp16_weights=False)
            new.weight = bnb.nn.Int8Params(weight, requires_grad=False, has_fp16_weights=False)
        else:
            compute = config.bnb_4bit_compute_dtype or torch.bfloat16
            new = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=has_bias,
                compute_dtype=compute,
                compress_statistics=config.bnb_4bit_use_double_quant,
                quant_type=config.bnb_4bit_quant_type,
            )
            new.weight = bnb.nn.Params4bit(
                weight,
                requires_grad=False,
                quant_type=config.bnb_4bit_quant_type,
                compress_statistics=config.bnb_4bit_use_double_quant,
            )
        if has_bias:
            new.bias = nn.Parameter(child.bias.detach(), requires_grad=False)
        new.requires_grad_(False)
        setattr(module, name, new)
        new.to(weight.device)


def load_control_file(uri, auth_token, local_files_only) -> tuple[dict, str, float, torch.dtype | None, str | None]:
    """Read one ``--control-nets`` URI and return ``(state, source, scale, dtype, quantizer)``."""
    import safetensors.torch

    import dgenerate.hfhub as _hfhub
    import dgenerate.pipelinewrapper.enums as _enums
    import dgenerate.pipelinewrapper.uris as _uris

    try:
        parsed = _uris.ControlNetUri.parse(uri, _enums.ModelType.MINIMAX_H3)
    except _uris.InvalidControlNetUriError as error:
        raise _pipelines.UnsupportedPipelineConfigError(str(error)) from error
    if parsed.mode is not None or parsed.start != 0.0 or parsed.end != 1.0 or parsed.compile:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-video --control-nets accepts scale and quantizer. '
            'start, end, mode, and compile are not used.')
    path, _subfolder = _hfhub.materialize_named_weight(
        parsed.model,
        weight_name=parsed.weight_name,
        revision=parsed.revision,
        subfolder=parsed.subfolder,
        local_files_only=local_files_only,
        auth_token=auth_token)
    if not _hfhub.is_single_file_model_load(path):
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-video --control-nets loads one weights file. '
            f'"{parsed.model}" is a folder or repository. Pass weight-name= '
            'for the safetensors file.')
    state = safetensors.torch.load_file(path, device='cpu')
    cast = _enums.get_torch_dtype(parsed.dtype) if parsed.dtype is not None else None
    return state, path, float(parsed.scale), cast, parsed.quantizer
