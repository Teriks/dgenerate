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

import math
import os
import pickle
import typing

import huggingface_hub
import torch
import torch.nn as nn

import dgenerate.memoize as _memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
from dgenerate.memoize import memoize as _memoize_call

_CLIP_ENCODER_FILENAME = 'clip_img_encoder.pt'
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# OpenCLIP ViT-H/14 uses an attention head of this width.
_HEAD_WIDTH = 80
_CPU_UNSAFE_DTYPES = frozenset({torch.float16, torch.bfloat16})
_hydit_clip_encoder_cache = _memoize.create_object_cache(
    'hydit_clip_image_encoder',
    cache_type=_memory.SizedConstrainedObjectCache)


def _image_preprocess(image_size: int):
    """OpenCLIP's shortest-edge resize, center crop, and dataset normalization."""
    from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor

    return Compose([
        Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        CenterCrop(image_size),
        ToTensor(),
        Normalize(_CLIP_MEAN, _CLIP_STD),
    ])


class HunyuanDiTCLIPImageEncoder(nn.Module):
    """
    ViT-H/14 image tower from Tencent's ``clip_img_encoder.pt``.

    The checkpoint is an OpenCLIP vision tower. It is loaded into
    ``CLIPVisionModelWithProjection`` so the OpenCLIP package is not required.
    """

    def __init__(self, vision_model, preprocess):
        super().__init__()
        self._model = vision_model
        self.preprocess = preprocess

    @property
    def dtype(self) -> torch.dtype:
        return next(self._model.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self._model.parameters()).device

    def encode_pil_images(self, image: typing.Any) -> torch.Tensor:
        if isinstance(image, (list, tuple)):
            tensors = torch.stack([self.preprocess(img.convert('RGB')) for img in image])
        else:
            tensors = self.preprocess(image.convert('RGB')).unsqueeze(0)
        tensors = tensors.to(
            device=next(self._model.parameters()).device,
            dtype=self.dtype)
        with torch.no_grad():
            embeds = self._model(pixel_values=tensors).image_embeds
        return embeds / embeds.norm(dim=-1, keepdim=True)

    def forward(self, pixel_values: torch.Tensor):
        with torch.no_grad():
            return self._model(pixel_values=pixel_values).image_embeds


def _clip_encoder_weight_path(
        repo_id: str,
        *,
        token: str | None = None,
        local_files_only: bool = False) -> str:
    """
    Path to ``clip_img_encoder.pt``.

    A local IP-Adapter file or directory is not a Hub repo id. Tencent ships
    the encoder next to ``ipa.pt``.
    """
    path = str(repo_id)
    if os.path.isfile(path):
        path = os.path.abspath(path)
        if os.path.basename(path) == _CLIP_ENCODER_FILENAME:
            return path
        sibling = os.path.join(os.path.dirname(path), _CLIP_ENCODER_FILENAME)
        if os.path.isfile(sibling):
            return sibling
        raise FileNotFoundError(
            f'Hunyuan-DiT IP-Adapter image encoder "{_CLIP_ENCODER_FILENAME}" '
            f'was not found next to "{repo_id}".')
    if os.path.isdir(path):
        candidate = os.path.join(os.path.abspath(path), _CLIP_ENCODER_FILENAME)
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(
            f'Hunyuan-DiT IP-Adapter image encoder "{_CLIP_ENCODER_FILENAME}" '
            f'was not found in "{repo_id}".')
    return huggingface_hub.hf_hub_download(
        repo_id=path,
        filename=_CLIP_ENCODER_FILENAME,
        token=token,
        local_files_only=local_files_only,
    )


def _resident_dtype(device: torch.device | str, dtype: torch.dtype) -> torch.dtype:
    """
    Dtype the CLIP vision tower is stored in.

    CPU convolutions do not implement float16 or bfloat16. A half-precision
    request on CPU is stored as float32; the caller casts the embedding.
    """
    if torch.device(device).type == 'cpu' and dtype in _CPU_UNSAFE_DTYPES:
        return torch.float32
    return dtype


def _module_nbytes(module: torch.nn.Module) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in (*module.parameters(), *module.buffers()))


def load_hunyuan_dit_clip_image_encoder(
        repo_id: str,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        token: str | None = None,
        local_files_only: bool = False) -> HunyuanDiTCLIPImageEncoder:
    requested = torch.device(device)
    dtype = _resident_dtype(requested, dtype)
    encoder = _load_hunyuan_dit_clip_image_encoder(
        str(repo_id), requested, dtype, token, local_files_only)
    # The cached copy is built on CPU. Move it after insertion so the
    # sized cache can drop the CPU-side footprint.
    if requested.type != 'cpu' and encoder.device.type != requested.type:
        encoder.to(device=requested)
    return encoder


def _encoder_cache_key(args: dict) -> str:
    return _memoize.args_cache_key(args, {
        'device': lambda device: str(torch.device(device)),
        'dtype': str,
    })


def _visual_weights(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """OpenCLIP ``visual.*`` tensors, without the ``module.`` prefix."""
    visual = {}
    for key, value in state_dict.items():
        for prefix in ('module.visual.', 'visual.'):
            if key.startswith(prefix):
                visual[key[len(prefix):]] = value
                break
    if not visual:
        raise RuntimeError(
            'clip_img_encoder.pt has no visual tower weights. '
            'Expected Tencent\'s ViT-H/14 checkpoint.')
    return visual


def _copy_linear(converted: dict[str, torch.Tensor], source: dict[str, torch.Tensor],
                 source_prefix: str, target_prefix: str) -> None:
    converted[f'{target_prefix}.weight'] = source.pop(f'{source_prefix}.weight')
    converted[f'{target_prefix}.bias'] = source.pop(f'{source_prefix}.bias')


def _transformers_vision_state(
        visual: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict]:
    """
    Map an OpenCLIP vision tower onto ``CLIPVisionModelWithProjection``.

    The fused QKV projection is stored as one matrix. Transformers keeps
    separate q, k, and v projections, in that order. ``proj`` is a right-multiply
    matrix, so the linear weight is its transpose.
    """
    converted: dict[str, torch.Tensor] = {}
    visual = dict(visual)

    def take(key: str) -> torch.Tensor:
        try:
            return visual.pop(key)
        except KeyError:
            raise RuntimeError(
                f'clip_img_encoder.pt is missing visual weight "{key}".') from None

    patch = take('conv1.weight')
    converted['vision_model.embeddings.patch_embedding.weight'] = patch
    converted['vision_model.embeddings.class_embedding'] = take('class_embedding')
    positional = take('positional_embedding')
    converted['vision_model.embeddings.position_embedding.weight'] = positional
    _copy_linear(converted, visual, 'ln_pre', 'vision_model.pre_layrnorm')
    _copy_linear(converted, visual, 'ln_post', 'vision_model.post_layernorm')
    projection = take('proj')
    converted['visual_projection.weight'] = projection.transpose(0, 1).contiguous()

    blocks = sorted({
        int(key.split('.')[2]) for key in visual
        if key.startswith('transformer.resblocks.')
    })
    if not blocks or blocks != list(range(len(blocks))):
        raise RuntimeError('clip_img_encoder.pt visual tower has no contiguous transformer blocks.')
    for index in blocks:
        source = f'transformer.resblocks.{index}'
        target = f'vision_model.encoder.layers.{index}'
        _copy_linear(converted, visual, f'{source}.ln_1', f'{target}.layer_norm1')
        _copy_linear(converted, visual, f'{source}.ln_2', f'{target}.layer_norm2')
        qkv_weight = take(f'{source}.attn.in_proj_weight')
        qkv_bias = take(f'{source}.attn.in_proj_bias')
        q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)
        converted[f'{target}.self_attn.q_proj.weight'] = q_weight
        converted[f'{target}.self_attn.k_proj.weight'] = k_weight
        converted[f'{target}.self_attn.v_proj.weight'] = v_weight
        converted[f'{target}.self_attn.q_proj.bias'] = q_bias
        converted[f'{target}.self_attn.k_proj.bias'] = k_bias
        converted[f'{target}.self_attn.v_proj.bias'] = v_bias
        _copy_linear(converted, visual, f'{source}.attn.out_proj', f'{target}.self_attn.out_proj')
        _copy_linear(converted, visual, f'{source}.mlp.c_fc', f'{target}.mlp.fc1')
        _copy_linear(converted, visual, f'{source}.mlp.c_proj', f'{target}.mlp.fc2')

    if visual:
        shown = ', '.join(sorted(visual)[:4])
        extra = f' (+{len(visual) - 4} more)' if len(visual) > 4 else ''
        raise RuntimeError(
            'clip_img_encoder.pt visual tower has weights this loader does not map: '
            f'{shown}{extra}.')

    width = int(patch.shape[0])
    patch_size = int(patch.shape[-1])
    if width % _HEAD_WIDTH != 0:
        raise RuntimeError(
            f'clip_img_encoder.pt vision width {width} is not a ViT-H/14 tower '
            f'(expected a multiple of {_HEAD_WIDTH}).')
    positions = int(positional.shape[0])
    grid = int(round(math.sqrt(positions - 1)))
    if grid * grid + 1 != positions:
        raise RuntimeError(
            f'clip_img_encoder.pt position embedding has {positions} tokens, '
            'which is not a square grid plus a class token.')
    config = {
        'hidden_size': width,
        'intermediate_size': int(converted['vision_model.encoder.layers.0.mlp.fc1.weight'].shape[0]),
        'num_hidden_layers': len(blocks),
        'num_attention_heads': width // _HEAD_WIDTH,
        'num_channels': int(patch.shape[1]),
        'image_size': grid * patch_size,
        'patch_size': patch_size,
        'hidden_act': 'gelu',
        'layer_norm_eps': 1e-5,
        'projection_dim': int(projection.shape[1]),
    }
    return converted, config


@_memoize_call(
    _hydit_clip_encoder_cache,
    exceptions={'token', 'local_files_only'},
    hasher=_encoder_cache_key,
    on_hit=lambda key, hit: _memoize.simple_cache_hit_debug(
        'Hunyuan-DiT CLIP image encoder', key, hit),
    on_create=lambda key, new: _memoize.simple_cache_miss_debug(
        'Hunyuan-DiT CLIP image encoder', key, new))
def _load_hunyuan_dit_clip_image_encoder(
        repo_id: str,
        device: torch.device,
        dtype: torch.dtype,
        token: str | None = None,
        local_files_only: bool = False) -> HunyuanDiTCLIPImageEncoder:
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

    file_path = _clip_encoder_weight_path(
        repo_id, token=token, local_files_only=local_files_only)
    from dgenerate.pipelinewrapper.constants import IMAGE_ENCODER_CACHE_MEMORY_CONSTRAINTS

    _hydit_clip_encoder_cache.enforce_cpu_mem_constraints(
        IMAGE_ENCODER_CACHE_MEMORY_CONSTRAINTS,
        size_var='image_encoder_size',
        new_object_size=os.path.getsize(file_path))
    try:
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
    except (pickle.UnpicklingError, RuntimeError):
        # Some releases wrap the state dict in objects weights_only rejects.
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    converted, config = _transformers_vision_state(_visual_weights(state_dict))
    model = CLIPVisionModelWithProjection(CLIPVisionConfig(**config))
    incompatible = model.load_state_dict(converted, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            'Converted clip_img_encoder.pt did not match CLIPVisionModelWithProjection: '
            f'missing {incompatible.missing_keys}, unexpected {incompatible.unexpected_keys}.')
    # Stay on CPU until the sized cache records the footprint. Moving onto
    # another device afterwards updates that accounting.
    model = model.to(device='cpu', dtype=dtype)
    model.eval()

    encoder = HunyuanDiTCLIPImageEncoder(model, _image_preprocess(config['image_size']))
    nbytes = _module_nbytes(encoder)
    from dgenerate.pipelinewrapper.uris.util import _patch_module_to_for_sized_cache

    _patch_module_to_for_sized_cache(_hydit_clip_encoder_cache, encoder)
    _messages.debug_log(
        f'Loaded Hunyuan-DiT IP-Adapter image encoder from "{file_path}" '
        f'for {device}.')
    return encoder, _memoize.CachedObjectMetadata(size=nbytes)
