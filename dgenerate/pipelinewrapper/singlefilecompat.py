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

"""
Diffusers single file loader fixes shared by every single file transformer
load: dense, GGUF, and ComfyUI quantized ``.safetensors`` alike.

Registers transformers Diffusers can build but not load from a single file,
and replaces checkpoint conversions that miss tensors of newer releases.
"""

import functools
import json
import os
import re
import struct

import torch

import dgenerate.hfhub as _hfhub

COMFY_PREFIX = 'model.diffusion_model.'


def strip_comfy_prefix(checkpoint: dict) -> dict:
    """Drop the Comfy ``model.diffusion_model.`` key prefix, if present."""
    if not any(key.startswith(COMFY_PREFIX) for key in checkpoint):
        return checkpoint
    renamed = {}
    for key, value in checkpoint.items():
        new_key = key[len(COMFY_PREFIX):] if key.startswith(COMFY_PREFIX) else key
        renamed[new_key] = value
    return renamed


def rename_ltx25_prompt_adaln(checkpoint: dict) -> dict:
    """
    LTX 2.5 checkpoints keep the original ``*_adaln_single`` prefix. Diffusers
    drops ``_single`` for the prompt embeddings and maps the main ones itself,
    but its converter (as of 0.41) does not know the prompt ones.
    """
    renamed = {}
    for key, value in checkpoint.items():
        new_key = key.replace('audio_prompt_adaln_single.', 'audio_prompt_adaln.')
        new_key = new_key.replace('prompt_adaln_single.', 'prompt_adaln.')
        renamed[new_key] = value
    return renamed


def convert_ltx2_transformer_checkpoint(checkpoint: dict, **kwargs) -> dict:
    """Diffusers' LTX 2 conversion, plus the LTX 2.5 prompt modulation names."""
    from diffusers.loaders.single_file_utils import convert_ltx2_transformer_to_diffusers
    return convert_ltx2_transformer_to_diffusers(rename_ltx25_prompt_adaln(checkpoint), **kwargs)


def convert_hunyuan_video15_transformer_checkpoint(checkpoint: dict, **kwargs) -> dict:
    """Convert original Hunyuan Video weights to the Diffusers 1.5 names."""
    from diffusers.loaders.single_file_utils import convert_hunyuan_video_transformer_to_diffusers

    converted = convert_hunyuan_video_transformer_to_diffusers(
        strip_comfy_prefix(checkpoint), **kwargs)
    renamed = {}
    vision_projection = {
        'vision_in.proj.0.': 'image_embedder.norm_in.',
        'vision_in.proj.1.': 'image_embedder.linear_1.',
        'vision_in.proj.3.': 'image_embedder.linear_2.',
        'vision_in.proj.4.': 'image_embedder.norm_out.',
    }
    byt5_projection = {
        'byt5_in.net.0.proj.': 'context_embedder_2.linear_1.',
        'byt5_in.net.2.': 'context_embedder_2.linear_2.',
        'byt5_in.fc3.': 'context_embedder_2.linear_3.',
        'byt5_in.layernorm.': 'context_embedder_2.norm.',
    }
    for name, value in converted.items():
        for source, target in byt5_projection.items():
            if name.startswith(source):
                name = target + name[len(source):]
                break
        if name.startswith('time_text_embed.'):
            name = 'time_embed.' + name[len('time_text_embed.'):]
        elif name.startswith('cond_type_embedding.'):
            name = 'cond_type_embed.' + name[len('cond_type_embedding.'):]
        else:
            for source, target in vision_projection.items():
                if name.startswith(source):
                    name = target + name[len(source):]
                    break
        renamed[name] = value
    return renamed


# HunyuanImage 2.1 (Comfy / Tencent / GGUF -> Diffusers). See diffusers
# scripts/convert_hunyuan_image_to_diffusers.py.

_HYIMAGE_CONFIG_BASE = 'hunyuanvideo-community/HunyuanImage-2.1-Diffusers'
_HYIMAGE_CONFIG_DISTILLED = 'hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers'
_HYIMAGE_CONFIG_REFINER = 'hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers'

_HYIMAGE_STRIP_PREFIXES = (
    COMFY_PREFIX,
    'model.model.',
    'model.diffusion_model.',
    'diffusion_model.',
)


def _hyimage_strip_known_prefixes(key: str) -> str:
    while True:
        for prefix in _HYIMAGE_STRIP_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        else:
            return key


def _hyimage_canonicalize_key(key: str) -> str:
    key = _hyimage_strip_known_prefixes(key)
    replacements = (
        ('.in_layer.', '.mlp.0.'),
        ('.out_layer.', '.mlp.2.'),
        ('.img_mod.lin.', '.img_mod.linear.'),
        ('.txt_mod.lin.', '.txt_mod.linear.'),
        ('.modulation.lin.', '.modulation.linear.'),
        ('.img_mlp.0.', '.img_mlp.fc1.'),
        ('.img_mlp.2.', '.img_mlp.fc2.'),
        ('.txt_mlp.0.', '.txt_mlp.fc1.'),
        ('.txt_mlp.2.', '.txt_mlp.fc2.'),
        ('.self_attn.proj.', '.self_attn_proj.'),
        ('.self_attn.qkv.', '.self_attn_qkv.'),
        ('.img_attn.qkv.', '.img_attn_qkv.'),
        ('.img_attn.proj.', '.img_attn_proj.'),
        ('.txt_attn.qkv.', '.txt_attn_qkv.'),
        ('.txt_attn.proj.', '.txt_attn_proj.'),
        ('.img_attn.norm.query_norm.scale', '.img_attn_q_norm.weight'),
        ('.img_attn.norm.key_norm.scale', '.img_attn_k_norm.weight'),
        ('.txt_attn.norm.query_norm.scale', '.txt_attn_q_norm.weight'),
        ('.txt_attn.norm.key_norm.scale', '.txt_attn_k_norm.weight'),
        ('.norm.query_norm.scale', '.q_norm.weight'),
        ('.norm.key_norm.scale', '.k_norm.weight'),
    )
    for old, new in replacements:
        key = key.replace(old, new)
    key = key.replace('.c_embedder.mlp.0.', '.c_embedder.linear_1.')
    key = key.replace('.c_embedder.mlp.2.', '.c_embedder.linear_2.')
    if '.individual_token_refiner.blocks.' in key:
        key = key.replace('.mlp.0.', '.mlp.fc1.')
        key = key.replace('.mlp.2.', '.mlp.fc2.')
    return key


def _hyimage_is_auxiliary_key(key: str) -> bool:
    if '.norm.key_norm.scale' in key or '.norm.query_norm.scale' in key:
        return True
    if key.endswith('.comfy_quant') or key == 'scaled_fp8':
        return True
    return False


def prepare_hunyuanimage_checkpoint(checkpoint: dict) -> dict:
    """Normalize Comfy / Tencent / GGUF key names to the converter input layout."""
    prepared = {}
    for key, value in checkpoint.items():
        key = _hyimage_canonicalize_key(key)
        if _hyimage_is_auxiliary_key(key):
            continue
        prepared[key] = value
    return _hyimage_split_fused_linear_weights(prepared)


def _hyimage_split_fused_qkv_base(key: str, fused_suffix: str, out_prefix: str) -> str | None:
    if not key.endswith(('.weight', '.bias')):
        return None
    stem, _kind = key.rsplit('.', 1)
    if not stem.endswith(fused_suffix):
        return None
    return stem[: -len(fused_suffix)] + out_prefix


def _hyimage_split_fused_linear_weights(state_dict: dict) -> dict:
    result = {}
    for key, weight in state_dict.items():
        base = _hyimage_split_fused_qkv_base(key, '.img_attn_qkv', '.img_attn_')
        if base is not None and key.startswith('double_blocks'):
            q, k, v = torch.chunk(weight, 3, dim=0)
            kind = key.rsplit('.', 1)[1]
            result[base + f'q.{kind}'] = q
            result[base + f'k.{kind}'] = k
            result[base + f'v.{kind}'] = v
            continue
        base = _hyimage_split_fused_qkv_base(key, '.txt_attn_qkv', '.txt_attn_')
        if base is not None and key.startswith('double_blocks'):
            q, k, v = torch.chunk(weight, 3, dim=0)
            kind = key.rsplit('.', 1)[1]
            result[base + f'q.{kind}'] = q
            result[base + f'k.{kind}'] = k
            result[base + f'v.{kind}'] = v
            continue
        if key.startswith('single_blocks') and '.linear1.' in key and 'linear1_' not in key:
            linear2_key = key.replace('.linear1.', '.linear2.')
            if linear2_key not in state_dict:
                result[key] = weight
                continue
            hidden_size = state_dict[linear2_key].shape[0]
            if key.endswith('.weight'):
                q, k, v, mlp = torch.split(
                    weight, [hidden_size, hidden_size, hidden_size, weight.shape[0] - 3 * hidden_size], dim=0)
                prefix = key.replace('.linear1.', '.')
                block = prefix.rsplit('.', 1)[0]
                result[block + '.linear1_q.weight'] = q
                result[block + '.linear1_k.weight'] = k
                result[block + '.linear1_v.weight'] = v
                result[block + '.linear1_mlp.weight'] = mlp
            else:
                q, k, v, mlp = torch.split(
                    weight, [hidden_size, hidden_size, hidden_size, weight.shape[0] - 3 * hidden_size], dim=0)
                prefix = key.replace('.linear1.', '.')
                block = prefix.rsplit('.', 1)[0]
                result[block + '.linear1_q.bias'] = q
                result[block + '.linear1_k.bias'] = k
                result[block + '.linear1_v.bias'] = v
                result[block + '.linear1_mlp.bias'] = mlp
            continue
        if key.startswith('single_blocks') and '.linear2.' in key and 'linear2.fc' not in key:
            result[key.replace('.linear2.', '.linear2.fc.')] = weight
            continue
        result[key] = weight
    return result


def _hyimage_block_count(state_dict: dict, prefix: str) -> int:
    pattern = re.compile(rf'^{re.escape(prefix)}\.(\d+)\.')
    indices = [int(match.group(1)) for key in state_dict for match in [pattern.match(key)] if match]
    return max(indices) + 1 if indices else 0


def infer_hunyuanimage_config_repo(state_dict: dict) -> str:
    keys = set(state_dict)
    has_byt5 = any(key.startswith('byt5_in.') for key in keys)
    has_meanflow = any(key.startswith('time_r_in.') for key in keys)
    has_guidance = any(key.startswith('guidance_in.') for key in keys)
    if not has_byt5 and has_guidance:
        return _HYIMAGE_CONFIG_REFINER
    if has_byt5 and (has_guidance or has_meanflow):
        return _HYIMAGE_CONFIG_DISTILLED
    return _HYIMAGE_CONFIG_BASE


def _hyimage_is_comfy_checkpoint(state_dict: dict) -> bool:
    for key in state_dict:
        if key.startswith(('model.model.', 'model.diffusion_model.', 'img_in.', 'double_blocks.')):
            return True
        canonical = _hyimage_canonicalize_key(key)
        if canonical.startswith(('img_in.', 'double_blocks.', 'single_blocks.')):
            return True
    return False


def _hyimage_pop_or_split_qkv_triple(state_dict: dict, block: str, attn: str, kind: str):
    base = f'{block}.{attn}_'
    q_key = f'{base}q.{kind}'
    if q_key in state_dict:
        return (
            state_dict.pop(q_key),
            state_dict.pop(f'{base}k.{kind}'),
            state_dict.pop(f'{base}v.{kind}'),
        )
    qkv_key = f'{base}qkv.{kind}'
    fused = state_dict.pop(qkv_key)
    return torch.chunk(fused, 3, dim=0)


def infer_hunyuanimage_convert_flags(state_dict: dict) -> dict:
    keys = set(state_dict)
    return {
        'use_byt5': any(key.startswith('byt5_in.') for key in keys),
        'use_meanflow': any(key.startswith('time_r_in.') for key in keys),
        'guidance_distilled': any(key.startswith('guidance_in.') for key in keys),
    }


def convert_hunyuanimage_transformer_checkpoint_to_diffusers(
    original_state_dict,
    use_byt5=True,
    guidance_distilled=False,
    use_meanflow=False,
):
    if _hyimage_is_comfy_checkpoint(original_state_dict):
        original_state_dict = prepare_hunyuanimage_checkpoint(
            strip_comfy_prefix(dict(original_state_dict)))

    converted_state_dict = {}
    num_layers = _hyimage_block_count(original_state_dict, 'double_blocks')
    num_single_layers = _hyimage_block_count(original_state_dict, 'single_blocks')
    num_refiner_layers = _hyimage_block_count(original_state_dict, 'txt_in.individual_token_refiner.blocks')
    if num_refiner_layers == 0:
        num_refiner_layers = 2

    if use_byt5:
        converted_state_dict['context_embedder_2.norm.weight'] = original_state_dict.pop('byt5_in.layernorm.weight')
        converted_state_dict['context_embedder_2.norm.bias'] = original_state_dict.pop('byt5_in.layernorm.bias')
        converted_state_dict['context_embedder_2.linear_1.weight'] = original_state_dict.pop('byt5_in.fc1.weight')
        converted_state_dict['context_embedder_2.linear_1.bias'] = original_state_dict.pop('byt5_in.fc1.bias')
        converted_state_dict['context_embedder_2.linear_2.weight'] = original_state_dict.pop('byt5_in.fc2.weight')
        converted_state_dict['context_embedder_2.linear_2.bias'] = original_state_dict.pop('byt5_in.fc2.bias')
        converted_state_dict['context_embedder_2.linear_3.weight'] = original_state_dict.pop('byt5_in.fc3.weight')
        converted_state_dict['context_embedder_2.linear_3.bias'] = original_state_dict.pop('byt5_in.fc3.bias')

    converted_state_dict['x_embedder.proj.weight'] = original_state_dict.pop('img_in.proj.weight')
    converted_state_dict['x_embedder.proj.bias'] = original_state_dict.pop('img_in.proj.bias')

    converted_state_dict['context_embedder.proj_in.weight'] = original_state_dict.pop('txt_in.input_embedder.weight')
    converted_state_dict['context_embedder.proj_in.bias'] = original_state_dict.pop('txt_in.input_embedder.bias')

    converted_state_dict['context_embedder.time_text_embed.timestep_embedder.linear_1.weight'] = (
        original_state_dict.pop('txt_in.t_embedder.mlp.0.weight'))
    converted_state_dict['context_embedder.time_text_embed.timestep_embedder.linear_1.bias'] = (
        original_state_dict.pop('txt_in.t_embedder.mlp.0.bias'))
    converted_state_dict['context_embedder.time_text_embed.timestep_embedder.linear_2.weight'] = (
        original_state_dict.pop('txt_in.t_embedder.mlp.2.weight'))
    converted_state_dict['context_embedder.time_text_embed.timestep_embedder.linear_2.bias'] = (
        original_state_dict.pop('txt_in.t_embedder.mlp.2.bias'))

    converted_state_dict['context_embedder.time_text_embed.text_embedder.linear_1.weight'] = (
        original_state_dict.pop('txt_in.c_embedder.linear_1.weight'))
    converted_state_dict['context_embedder.time_text_embed.text_embedder.linear_1.bias'] = (
        original_state_dict.pop('txt_in.c_embedder.linear_1.bias'))
    converted_state_dict['context_embedder.time_text_embed.text_embedder.linear_2.weight'] = (
        original_state_dict.pop('txt_in.c_embedder.linear_2.weight'))
    converted_state_dict['context_embedder.time_text_embed.text_embedder.linear_2.bias'] = (
        original_state_dict.pop('txt_in.c_embedder.linear_2.bias'))

    for i in range(num_refiner_layers):
        block_prefix = f'context_embedder.token_refiner.refiner_blocks.{i}.'
        converted_state_dict[f'{block_prefix}norm1.weight'] = original_state_dict.pop(
            f'txt_in.individual_token_refiner.blocks.{i}.norm1.weight')
        converted_state_dict[f'{block_prefix}norm1.bias'] = original_state_dict.pop(
            f'txt_in.individual_token_refiner.blocks.{i}.norm1.bias')
        converted_state_dict[f'{block_prefix}norm2.weight'] = original_state_dict.pop(
            f'txt_in.individual_token_refiner.blocks.{i}.norm2.weight')
        converted_state_dict[f'{block_prefix}norm2.bias'] = original_state_dict.pop(
            f'txt_in.individual_token_refiner.blocks.{i}.norm2.bias')

        qkv_weight = original_state_dict.pop(f'txt_in.individual_token_refiner.blocks.{i}.self_attn_qkv.weight')
        qkv_bias = original_state_dict.pop(f'txt_in.individual_token_refiner.blocks.{i}.self_attn_qkv.bias')
        q_weight, k_weight, v_weight = torch.chunk(qkv_weight, 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(qkv_bias, 3, dim=0)
        converted_state_dict[f'{block_prefix}attn.to_q.weight'] = q_weight
        converted_state_dict[f'{block_prefix}attn.to_q.bias'] = q_bias
        converted_state_dict[f'{block_prefix}attn.to_k.weight'] = k_weight
        converted_state_dict[f'{block_prefix}attn.to_k.bias'] = k_bias
        converted_state_dict[f'{block_prefix}attn.to_v.weight'] = v_weight
        converted_state_dict[f'{block_prefix}attn.to_v.bias'] = v_bias

        converted_state_dict[f'{block_prefix}attn.to_out.0.weight'] = original_state_dict.pop(
            f'txt_in.individual_token_refiner.blocks.{i}.self_attn_proj.weight')
        converted_state_dict[f'{block_prefix}attn.to_out.0.bias'] = original_state_dict.pop(
            f'txt_in.individual_token_refiner.blocks.{i}.self_attn_proj.bias')

        converted_state_dict[f'{block_prefix}ff.net.0.proj.weight'] = original_state_dict.pop(
            f'txt_in.individual_token_refiner.blocks.{i}.mlp.fc1.weight')
        converted_state_dict[f'{block_prefix}ff.net.0.proj.bias'] = original_state_dict.pop(
            f'txt_in.individual_token_refiner.blocks.{i}.mlp.fc1.bias')
        converted_state_dict[f'{block_prefix}ff.net.2.weight'] = original_state_dict.pop(
            f'txt_in.individual_token_refiner.blocks.{i}.mlp.fc2.weight')
        converted_state_dict[f'{block_prefix}ff.net.2.bias'] = original_state_dict.pop(
            f'txt_in.individual_token_refiner.blocks.{i}.mlp.fc2.bias')

        converted_state_dict[f'{block_prefix}norm_out.linear.weight'] = original_state_dict.pop(
            f'txt_in.individual_token_refiner.blocks.{i}.adaLN_modulation.1.weight')
        converted_state_dict[f'{block_prefix}norm_out.linear.bias'] = original_state_dict.pop(
            f'txt_in.individual_token_refiner.blocks.{i}.adaLN_modulation.1.bias')

    converted_state_dict['time_guidance_embed.timestep_embedder.linear_1.weight'] = original_state_dict.pop(
        'time_in.mlp.0.weight')
    converted_state_dict['time_guidance_embed.timestep_embedder.linear_1.bias'] = original_state_dict.pop(
        'time_in.mlp.0.bias')
    converted_state_dict['time_guidance_embed.timestep_embedder.linear_2.weight'] = original_state_dict.pop(
        'time_in.mlp.2.weight')
    converted_state_dict['time_guidance_embed.timestep_embedder.linear_2.bias'] = original_state_dict.pop(
        'time_in.mlp.2.bias')

    if use_meanflow:
        converted_state_dict['time_guidance_embed.timestep_embedder_r.linear_1.weight'] = original_state_dict.pop(
            'time_r_in.mlp.0.weight')
        converted_state_dict['time_guidance_embed.timestep_embedder_r.linear_1.bias'] = original_state_dict.pop(
            'time_r_in.mlp.0.bias')
        converted_state_dict['time_guidance_embed.timestep_embedder_r.linear_2.weight'] = original_state_dict.pop(
            'time_r_in.mlp.2.weight')
        converted_state_dict['time_guidance_embed.timestep_embedder_r.linear_2.bias'] = original_state_dict.pop(
            'time_r_in.mlp.2.bias')

    if guidance_distilled:
        converted_state_dict['time_guidance_embed.guidance_embedder.linear_1.weight'] = original_state_dict.pop(
            'guidance_in.mlp.0.weight')
        converted_state_dict['time_guidance_embed.guidance_embedder.linear_1.bias'] = original_state_dict.pop(
            'guidance_in.mlp.0.bias')
        converted_state_dict['time_guidance_embed.guidance_embedder.linear_2.weight'] = original_state_dict.pop(
            'guidance_in.mlp.2.weight')
        converted_state_dict['time_guidance_embed.guidance_embedder.linear_2.bias'] = original_state_dict.pop(
            'guidance_in.mlp.2.bias')

    for i in range(num_layers):
        block_prefix = f'transformer_blocks.{i}.'
        converted_state_dict[f'{block_prefix}norm1.linear.weight'] = original_state_dict.pop(
            f'double_blocks.{i}.img_mod.linear.weight')
        converted_state_dict[f'{block_prefix}norm1.linear.bias'] = original_state_dict.pop(
            f'double_blocks.{i}.img_mod.linear.bias')
        converted_state_dict[f'{block_prefix}norm1_context.linear.weight'] = original_state_dict.pop(
            f'double_blocks.{i}.txt_mod.linear.weight')
        converted_state_dict[f'{block_prefix}norm1_context.linear.bias'] = original_state_dict.pop(
            f'double_blocks.{i}.txt_mod.linear.bias')

        block = f'double_blocks.{i}'
        q_w, k_w, v_w = _hyimage_pop_or_split_qkv_triple(original_state_dict, block, 'img_attn', 'weight')
        q_b, k_b, v_b = _hyimage_pop_or_split_qkv_triple(original_state_dict, block, 'img_attn', 'bias')
        converted_state_dict[f'{block_prefix}attn.to_q.weight'] = q_w
        converted_state_dict[f'{block_prefix}attn.to_q.bias'] = q_b
        converted_state_dict[f'{block_prefix}attn.to_k.weight'] = k_w
        converted_state_dict[f'{block_prefix}attn.to_k.bias'] = k_b
        converted_state_dict[f'{block_prefix}attn.to_v.weight'] = v_w
        converted_state_dict[f'{block_prefix}attn.to_v.bias'] = v_b
        converted_state_dict[f'{block_prefix}attn.norm_q.weight'] = original_state_dict.pop(
            f'double_blocks.{i}.img_attn_q_norm.weight')
        converted_state_dict[f'{block_prefix}attn.norm_k.weight'] = original_state_dict.pop(
            f'double_blocks.{i}.img_attn_k_norm.weight')
        converted_state_dict[f'{block_prefix}attn.to_out.0.weight'] = original_state_dict.pop(
            f'double_blocks.{i}.img_attn_proj.weight')
        converted_state_dict[f'{block_prefix}attn.to_out.0.bias'] = original_state_dict.pop(
            f'double_blocks.{i}.img_attn_proj.bias')
        converted_state_dict[f'{block_prefix}ff.net.0.proj.weight'] = original_state_dict.pop(
            f'double_blocks.{i}.img_mlp.fc1.weight')
        converted_state_dict[f'{block_prefix}ff.net.0.proj.bias'] = original_state_dict.pop(
            f'double_blocks.{i}.img_mlp.fc1.bias')
        converted_state_dict[f'{block_prefix}ff.net.2.weight'] = original_state_dict.pop(
            f'double_blocks.{i}.img_mlp.fc2.weight')
        converted_state_dict[f'{block_prefix}ff.net.2.bias'] = original_state_dict.pop(
            f'double_blocks.{i}.img_mlp.fc2.bias')

        q_w, k_w, v_w = _hyimage_pop_or_split_qkv_triple(original_state_dict, block, 'txt_attn', 'weight')
        q_b, k_b, v_b = _hyimage_pop_or_split_qkv_triple(original_state_dict, block, 'txt_attn', 'bias')
        converted_state_dict[f'{block_prefix}attn.add_q_proj.weight'] = q_w
        converted_state_dict[f'{block_prefix}attn.add_q_proj.bias'] = q_b
        converted_state_dict[f'{block_prefix}attn.add_k_proj.weight'] = k_w
        converted_state_dict[f'{block_prefix}attn.add_k_proj.bias'] = k_b
        converted_state_dict[f'{block_prefix}attn.add_v_proj.weight'] = v_w
        converted_state_dict[f'{block_prefix}attn.add_v_proj.bias'] = v_b
        converted_state_dict[f'{block_prefix}attn.norm_added_q.weight'] = original_state_dict.pop(
            f'double_blocks.{i}.txt_attn_q_norm.weight')
        converted_state_dict[f'{block_prefix}attn.norm_added_k.weight'] = original_state_dict.pop(
            f'double_blocks.{i}.txt_attn_k_norm.weight')
        converted_state_dict[f'{block_prefix}attn.to_add_out.weight'] = original_state_dict.pop(
            f'double_blocks.{i}.txt_attn_proj.weight')
        converted_state_dict[f'{block_prefix}attn.to_add_out.bias'] = original_state_dict.pop(
            f'double_blocks.{i}.txt_attn_proj.bias')
        converted_state_dict[f'{block_prefix}ff_context.net.0.proj.weight'] = original_state_dict.pop(
            f'double_blocks.{i}.txt_mlp.fc1.weight')
        converted_state_dict[f'{block_prefix}ff_context.net.0.proj.bias'] = original_state_dict.pop(
            f'double_blocks.{i}.txt_mlp.fc1.bias')
        converted_state_dict[f'{block_prefix}ff_context.net.2.weight'] = original_state_dict.pop(
            f'double_blocks.{i}.txt_mlp.fc2.weight')
        converted_state_dict[f'{block_prefix}ff_context.net.2.bias'] = original_state_dict.pop(
            f'double_blocks.{i}.txt_mlp.fc2.bias')

    for i in range(num_single_layers):
        block_prefix = f'single_transformer_blocks.{i}.'
        converted_state_dict[f'{block_prefix}norm.linear.weight'] = original_state_dict.pop(
            f'single_blocks.{i}.modulation.linear.weight')
        converted_state_dict[f'{block_prefix}norm.linear.bias'] = original_state_dict.pop(
            f'single_blocks.{i}.modulation.linear.bias')
        converted_state_dict[f'{block_prefix}attn.to_q.weight'] = original_state_dict.pop(
            f'single_blocks.{i}.linear1_q.weight')
        converted_state_dict[f'{block_prefix}attn.to_q.bias'] = original_state_dict.pop(
            f'single_blocks.{i}.linear1_q.bias')
        converted_state_dict[f'{block_prefix}attn.to_k.weight'] = original_state_dict.pop(
            f'single_blocks.{i}.linear1_k.weight')
        converted_state_dict[f'{block_prefix}attn.to_k.bias'] = original_state_dict.pop(
            f'single_blocks.{i}.linear1_k.bias')
        converted_state_dict[f'{block_prefix}attn.to_v.weight'] = original_state_dict.pop(
            f'single_blocks.{i}.linear1_v.weight')
        converted_state_dict[f'{block_prefix}attn.to_v.bias'] = original_state_dict.pop(
            f'single_blocks.{i}.linear1_v.bias')
        converted_state_dict[f'{block_prefix}attn.norm_q.weight'] = original_state_dict.pop(
            f'single_blocks.{i}.q_norm.weight')
        converted_state_dict[f'{block_prefix}attn.norm_k.weight'] = original_state_dict.pop(
            f'single_blocks.{i}.k_norm.weight')
        converted_state_dict[f'{block_prefix}proj_mlp.weight'] = original_state_dict.pop(
            f'single_blocks.{i}.linear1_mlp.weight')
        converted_state_dict[f'{block_prefix}proj_mlp.bias'] = original_state_dict.pop(
            f'single_blocks.{i}.linear1_mlp.bias')
        converted_state_dict[f'{block_prefix}proj_out.weight'] = original_state_dict.pop(
            f'single_blocks.{i}.linear2.fc.weight')
        converted_state_dict[f'{block_prefix}proj_out.bias'] = original_state_dict.pop(
            f'single_blocks.{i}.linear2.fc.bias')

    converted_state_dict['proj_out.weight'] = original_state_dict.pop('final_layer.linear.weight')
    converted_state_dict['proj_out.bias'] = original_state_dict.pop('final_layer.linear.bias')
    shift_w, scale_w = original_state_dict.pop('final_layer.adaLN_modulation.1.weight').chunk(2, dim=0)
    shift_b, scale_b = original_state_dict.pop('final_layer.adaLN_modulation.1.bias').chunk(2, dim=0)
    converted_state_dict['norm_out.linear.weight'] = torch.cat([scale_w, shift_w], dim=0)
    converted_state_dict['norm_out.linear.bias'] = torch.cat([scale_b, shift_b], dim=0)

    return converted_state_dict


def convert_hunyuanimage_transformer_for_single_file(checkpoint: dict, **kwargs) -> dict:
    """Entry point for Diffusers ``from_single_file`` registration."""
    prepared = prepare_hunyuanimage_checkpoint(strip_comfy_prefix(dict(checkpoint)))
    flags = infer_hunyuanimage_convert_flags(prepared)
    flags.update({k: v for k, v in kwargs.items() if k in flags})
    working = dict(prepared)
    return convert_hunyuanimage_transformer_checkpoint_to_diffusers(working, **flags)


def hunyuanimage_gguf_config_repo(shapes: dict[str, tuple[int, ...]]) -> str:
    """Pick a Diffusers config repo from GGUF / safetensors tensor names."""
    canonical = {_hyimage_canonicalize_key(name) for name in shapes}
    return infer_hunyuanimage_config_repo({key: None for key in canonical})


_HYDIT_CONFIG_V12 = 'Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers'
_HYDIT_CONFIG_V11 = 'Tencent-Hunyuan/HunyuanDiT-Diffusers'


def is_native_hydit_checkpoint(checkpoint: dict) -> bool:
    """True for Tencent native Hunyuan-DiT transformer checkpoints (Civitai / .pt)."""
    return (
        'blocks.0.attn1.Wqkv.weight' in checkpoint
        and 'x_embedder.proj.weight' in checkpoint
    )


def infer_hydit_config_repo(checkpoint: dict) -> str:
    """Pick the Diffusers pipeline repo for a native Hunyuan-DiT transformer file."""
    if 'style_embedder.weight' in checkpoint:
        return _HYDIT_CONFIG_V11
    return _HYDIT_CONFIG_V12


def _hydit_block_count(state_dict: dict) -> int:
    return _hyimage_block_count(state_dict, 'blocks')


def _hydit_swap_scale_shift(weight: torch.Tensor) -> torch.Tensor:
    shift, scale = weight.chunk(2, dim=0)
    return torch.cat([scale, shift], dim=0)


def convert_hydit_transformer_checkpoint_to_diffusers(
        checkpoint: dict,
        **kwargs) -> dict:
    """
    Map Tencent Hunyuan-DiT transformer weights to Diffusers ``HunyuanDiT2DModel`` names.

    Based on Hugging Face ``scripts/convert_hunyuandit_to_diffusers.py``.
    """
    state_dict = dict(checkpoint)
    num_layers = _hydit_block_count(state_dict)
    for i in range(num_layers):
        q, k, v = torch.chunk(state_dict[f'blocks.{i}.attn1.Wqkv.weight'], 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(state_dict[f'blocks.{i}.attn1.Wqkv.bias'], 3, dim=0)
        state_dict[f'blocks.{i}.attn1.to_q.weight'] = q
        state_dict[f'blocks.{i}.attn1.to_q.bias'] = q_bias
        state_dict[f'blocks.{i}.attn1.to_k.weight'] = k
        state_dict[f'blocks.{i}.attn1.to_k.bias'] = k_bias
        state_dict[f'blocks.{i}.attn1.to_v.weight'] = v
        state_dict[f'blocks.{i}.attn1.to_v.bias'] = v_bias
        state_dict.pop(f'blocks.{i}.attn1.Wqkv.weight')
        state_dict.pop(f'blocks.{i}.attn1.Wqkv.bias')

        state_dict[f'blocks.{i}.attn1.norm_q.weight'] = state_dict.pop(f'blocks.{i}.attn1.q_norm.weight')
        state_dict[f'blocks.{i}.attn1.norm_q.bias'] = state_dict.pop(f'blocks.{i}.attn1.q_norm.bias')
        state_dict[f'blocks.{i}.attn1.norm_k.weight'] = state_dict.pop(f'blocks.{i}.attn1.k_norm.weight')
        state_dict[f'blocks.{i}.attn1.norm_k.bias'] = state_dict.pop(f'blocks.{i}.attn1.k_norm.bias')

        state_dict[f'blocks.{i}.attn1.to_out.0.weight'] = state_dict.pop(f'blocks.{i}.attn1.out_proj.weight')
        state_dict[f'blocks.{i}.attn1.to_out.0.bias'] = state_dict.pop(f'blocks.{i}.attn1.out_proj.bias')

        k, v = torch.chunk(state_dict[f'blocks.{i}.attn2.kv_proj.weight'], 2, dim=0)
        k_bias, v_bias = torch.chunk(state_dict[f'blocks.{i}.attn2.kv_proj.bias'], 2, dim=0)
        state_dict[f'blocks.{i}.attn2.to_k.weight'] = k
        state_dict[f'blocks.{i}.attn2.to_k.bias'] = k_bias
        state_dict[f'blocks.{i}.attn2.to_v.weight'] = v
        state_dict[f'blocks.{i}.attn2.to_v.bias'] = v_bias
        state_dict.pop(f'blocks.{i}.attn2.kv_proj.weight')
        state_dict.pop(f'blocks.{i}.attn2.kv_proj.bias')

        state_dict[f'blocks.{i}.attn2.to_q.weight'] = state_dict.pop(f'blocks.{i}.attn2.q_proj.weight')
        state_dict[f'blocks.{i}.attn2.to_q.bias'] = state_dict.pop(f'blocks.{i}.attn2.q_proj.bias')

        state_dict[f'blocks.{i}.attn2.norm_q.weight'] = state_dict.pop(f'blocks.{i}.attn2.q_norm.weight')
        state_dict[f'blocks.{i}.attn2.norm_q.bias'] = state_dict.pop(f'blocks.{i}.attn2.q_norm.bias')
        state_dict[f'blocks.{i}.attn2.norm_k.weight'] = state_dict.pop(f'blocks.{i}.attn2.k_norm.weight')
        state_dict[f'blocks.{i}.attn2.norm_k.bias'] = state_dict.pop(f'blocks.{i}.attn2.k_norm.bias')

        state_dict[f'blocks.{i}.attn2.to_out.0.weight'] = state_dict.pop(f'blocks.{i}.attn2.out_proj.weight')
        state_dict[f'blocks.{i}.attn2.to_out.0.bias'] = state_dict.pop(f'blocks.{i}.attn2.out_proj.bias')

        norm2_weight = state_dict[f'blocks.{i}.norm2.weight']
        norm2_bias = state_dict[f'blocks.{i}.norm2.bias']
        state_dict[f'blocks.{i}.norm2.weight'] = state_dict[f'blocks.{i}.norm3.weight']
        state_dict[f'blocks.{i}.norm2.bias'] = state_dict[f'blocks.{i}.norm3.bias']
        state_dict[f'blocks.{i}.norm3.weight'] = norm2_weight
        state_dict[f'blocks.{i}.norm3.bias'] = norm2_bias

        state_dict[f'blocks.{i}.norm1.norm.weight'] = state_dict.pop(f'blocks.{i}.norm1.weight')
        state_dict[f'blocks.{i}.norm1.norm.bias'] = state_dict.pop(f'blocks.{i}.norm1.bias')
        state_dict[f'blocks.{i}.norm1.linear.weight'] = state_dict.pop(f'blocks.{i}.default_modulation.1.weight')
        state_dict[f'blocks.{i}.norm1.linear.bias'] = state_dict.pop(f'blocks.{i}.default_modulation.1.bias')

        state_dict[f'blocks.{i}.ff.net.0.proj.weight'] = state_dict.pop(f'blocks.{i}.mlp.fc1.weight')
        state_dict[f'blocks.{i}.ff.net.0.proj.bias'] = state_dict.pop(f'blocks.{i}.mlp.fc1.bias')
        state_dict[f'blocks.{i}.ff.net.2.weight'] = state_dict.pop(f'blocks.{i}.mlp.fc2.weight')
        state_dict[f'blocks.{i}.ff.net.2.bias'] = state_dict.pop(f'blocks.{i}.mlp.fc2.bias')

    state_dict['time_extra_emb.pooler.positional_embedding'] = state_dict.pop('pooler.positional_embedding')
    for proj in ('k_proj', 'q_proj', 'v_proj', 'c_proj'):
        state_dict[f'time_extra_emb.pooler.{proj}.weight'] = state_dict.pop(f'pooler.{proj}.weight')
        state_dict[f'time_extra_emb.pooler.{proj}.bias'] = state_dict.pop(f'pooler.{proj}.bias')

    for src, dst in (('0', 'linear_1'), ('2', 'linear_2')):
        state_dict[f'time_extra_emb.timestep_embedder.{dst}.bias'] = state_dict.pop(f't_embedder.mlp.{src}.bias')
        state_dict[f'time_extra_emb.timestep_embedder.{dst}.weight'] = state_dict.pop(f't_embedder.mlp.{src}.weight')

    state_dict['pos_embed.proj.weight'] = state_dict.pop('x_embedder.proj.weight')
    state_dict['pos_embed.proj.bias'] = state_dict.pop('x_embedder.proj.bias')

    for src, dst in (('0', 'linear_1'), ('2', 'linear_2')):
        state_dict[f'text_embedder.{dst}.bias'] = state_dict.pop(f'mlp_t5.{src}.bias')
        state_dict[f'text_embedder.{dst}.weight'] = state_dict.pop(f'mlp_t5.{src}.weight')

    for src, dst in (('0', 'linear_1'), ('2', 'linear_2')):
        state_dict[f'time_extra_emb.extra_embedder.{dst}.bias'] = state_dict.pop(f'extra_embedder.{src}.bias')
        state_dict[f'time_extra_emb.extra_embedder.{dst}.weight'] = state_dict.pop(f'extra_embedder.{src}.weight')

    state_dict['norm_out.linear.weight'] = _hydit_swap_scale_shift(
        state_dict.pop('final_layer.adaLN_modulation.1.weight'))
    state_dict['norm_out.linear.bias'] = _hydit_swap_scale_shift(
        state_dict.pop('final_layer.adaLN_modulation.1.bias'))

    state_dict['proj_out.weight'] = state_dict.pop('final_layer.linear.weight')
    state_dict['proj_out.bias'] = state_dict.pop('final_layer.linear.bias')

    if 'style_embedder.weight' in state_dict:
        state_dict['time_extra_emb.style_embedder.weight'] = state_dict.pop('style_embedder.weight')[0:1]

    return state_dict


def convert_hydit_transformer_for_single_file(checkpoint: dict, **kwargs) -> dict:
    """Entry point for Diffusers ``from_single_file`` registration."""
    return convert_hydit_transformer_checkpoint_to_diffusers(strip_comfy_prefix(dict(checkpoint)), **kwargs)


# Diffusers model types whose default pipeline repo supplies text encoders and
# a VAE for a transformer-only checkpoint. UNet families stay on from_single_file.
_AUX_DIFFUSERS_TYPES = frozenset({
    'sd3', 'sd35_large', 'sd35_medium',
    'flux-dev', 'flux-fill', 'flux-depth', 'flux-schnell',
    'flux-2-dev',
    'z-image-turbo',
    'ltx-video', 'ltx-video-0.9.1', 'ltx-video-0.9.5', 'ltx-video-0.9.7',
    'ltx2-dev',
    'wan-t2v-1.3B', 'wan-t2v-14B', 'wan-i2v-14B', 'wan-animate-14B',
    'wan-vace-1.3B', 'wan-vace-14B',
    'qwen-image-2.1',
    'krea2',
})

_EMBEDDED_TEXT_ENCODER_PREFIXES = (
    'cond_stage_model.',
    'conditioner.embedders.',
    'text_encoders.',
)

# Diffusers ``LDM_VAE_KEYS``; CivitAI / A1111 single-file checkpoints use these prefixes.
_EMBEDDED_VAE_PREFIXES = (
    'first_stage_model.',
    'vae.',
)

# Built-in VAE repos for SD 1.x / SD 2 / SDXL checkpoints with no embedded VAE
# (issue #62). Not used for Flux, SD3, or transformer-only aux-repo loads.
_SD_FAMILY_BUILTIN_SD15_VAE_REPO = 'stabilityai/sd-vae-ft-mse'
_SD_FAMILY_BUILTIN_SD2_VAE_REPO = 'sd2-community/stable-diffusion-2-1'
_SD_FAMILY_BUILTIN_SD2_VAE_SUBFOLDER = 'vae'
_SD_FAMILY_BUILTIN_SDXL_VAE_REPO = 'madebyollin/sdxl-vae-fp16-fix'

# Diffusers' SD2 fingerprint: OpenCLIP cross-attention width is 1024.
# SD 1.5 uses the same key with width 768.
_SD_FAMILY_SD2_CROSS_ATTENTION_KEY = (
    'model.diffusion_model.input_blocks.2.1.'
    'transformer_blocks.0.attn2.to_k.weight'
)

_ZIMAGE_TURBO = 'Tongyi-MAI/Z-Image-Turbo'
_KREA2_RAW = 'krea/Krea-2-Raw'
_QWEN_IMAGE = 'Qwen/Qwen-Image'
_QWEN_IMAGE_EDIT = 'Qwen/Qwen-Image-Edit'
_QWEN_IMAGE_EDIT_2509 = 'Qwen/Qwen-Image-Edit-2509'
_QWEN21 = 'Qwen/Qwen-Image-2.1'
_QWEN21_TURBO = 'Qwen/Qwen-Image-2.1-Turbo'
_FLUX_DEV = 'black-forest-labs/FLUX.1-dev'
_FLUX_KONTEXT = 'black-forest-labs/FLUX.1-Kontext-dev'
_FLUX2_KLEIN_9B = 'black-forest-labs/FLUX.2-klein-base-9B'
_FLUX2_KLEIN_KV = 'black-forest-labs/FLUX.2-klein-9b-kv'
_WAN_I2V_480 = 'Wan-AI/Wan2.1-I2V-14B-480P-Diffusers'
_WAN_I2V_720 = 'Wan-AI/Wan2.1-I2V-14B-720P-Diffusers'
_WAN_T2V_14 = 'Wan-AI/Wan2.1-T2V-14B-Diffusers'
_WAN_T2V_A14B = 'Wan-AI/Wan2.2-T2V-A14B-Diffusers'


class _ShapeTensor:
    """Stand-in so Diffusers can read ``.shape`` without loading weights."""

    def __init__(self, shape):
        self.shape = torch.Size(shape)


def _checkpoint_has_text_encoder(keys) -> bool:
    return any(key.startswith(_EMBEDDED_TEXT_ENCODER_PREFIXES) for key in keys)


def _sd_family_checkpoint_has_embedded_vae(keys) -> bool:
    """True when an SD-family single-file checkpoint embeds an LDM VAE."""
    return any(
        key.startswith(prefix) for key in keys for prefix in _EMBEDDED_VAE_PREFIXES
    )


def _sd_family_model_type_supports_builtin_vae(model_type) -> bool:
    """SD 1.x, SD 2, and SDXL --model-type values (issue #62)."""
    import dgenerate.pipelinewrapper.enums as _enums

    return (
        _enums.model_type_is_sd15(model_type)
        or _enums.model_type_is_sd2(model_type)
        or _enums.model_type_is_sdxl(model_type)
    )


def _sd_family_checkpoint_shapes_are_sd2(shapes: dict) -> bool:
    """True for an SD 2.x UNet (cross-attention width 1024)."""
    shape = shapes.get(_SD_FAMILY_SD2_CROSS_ATTENTION_KEY)
    return bool(shape) and int(shape[-1]) == 1024


def _sd_family_unet_config_is_sd2(config: dict) -> bool:
    dim = config.get('cross_attention_dim')
    if isinstance(dim, (list, tuple)):
        return False
    return dim == 1024


def _sd_family_checkpoint_path_is_sd2(model_path: str, *, local_files_only: bool) -> bool:
    """
    True when a ``--model-type sd`` checkpoint is SD 2.x rather than SD 1.x.

    Single-file safetensors are classified from the header. Other single-file
    checkpoints are classified from their tensor shapes. Diffusers folders and
    repository ids are classified from ``unet/config.json``.
    """
    import dgenerate.hfhub as _hfhub

    if _hfhub.is_single_file_model_load(model_path):
        path = model_path
        if not os.path.isfile(path):
            try:
                path = _hfhub.webcache_or_hf_blob_download(
                    path, local_files_only=local_files_only)
            except Exception:
                return False
        lower = path.lower()
        try:
            if lower.endswith('.safetensors'):
                shapes = _safetensors_weight_shapes(path)
            elif lower.endswith('.gguf'):
                return False
            else:
                import diffusers.loaders.single_file as _single_file
                checkpoint = _single_file.load_single_file_checkpoint(
                    path, local_files_only=local_files_only)
                shapes = _shapes_from_checkpoint(checkpoint)
        except Exception:
            return False
        return _sd_family_checkpoint_shapes_are_sd2(shapes)

    config_path = os.path.join(model_path, 'unet', 'config.json')
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding='utf-8') as file:
                return _sd_family_unet_config_is_sd2(json.load(file))
        except (OSError, json.JSONDecodeError):
            return False

    if os.path.isdir(model_path) or os.path.isfile(model_path):
        return False

    try:
        import huggingface_hub
        downloaded = huggingface_hub.hf_hub_download(
            model_path,
            'config.json',
            subfolder='unet',
            local_files_only=local_files_only,
        )
        with open(downloaded, encoding='utf-8') as file:
            return _sd_family_unet_config_is_sd2(json.load(file))
    except Exception:
        return False


def _sd_family_use_sd2_builtin_vae(
        model_type,
        model_path: str | None,
        *,
        local_files_only: bool) -> bool:
    """SD 2.x checkpoints need the SD 2.1 VAE, not the SD 1.5 fine-tune."""
    import dgenerate.pipelinewrapper.enums as _enums

    if _enums.model_type_is_sdxl(model_type):
        return False
    sd15 = _enums.model_type_is_sd15(model_type)
    sd2 = _enums.model_type_is_sd2(model_type)
    if sd2 and not sd15:
        return True
    if not (sd15 and sd2) or not model_path:
        return False
    return _sd_family_checkpoint_path_is_sd2(
        model_path, local_files_only=local_files_only)


def _sd_family_builtin_vae_source(
        model_type,
        model_path: str | None = None,
        *,
        local_files_only: bool = False) -> tuple[str, str | None] | None:
    """``(repo, subfolder)`` for the SD-family built-in VAE, or ``None``."""
    import dgenerate.pipelinewrapper.enums as _enums

    if _enums.model_type_is_sdxl(model_type):
        return _SD_FAMILY_BUILTIN_SDXL_VAE_REPO, None
    if not (
            _enums.model_type_is_sd15(model_type)
            or _enums.model_type_is_sd2(model_type)):
        return None
    if _sd_family_use_sd2_builtin_vae(
            model_type, model_path, local_files_only=local_files_only):
        return (
            _SD_FAMILY_BUILTIN_SD2_VAE_REPO,
            _SD_FAMILY_BUILTIN_SD2_VAE_SUBFOLDER)
    return _SD_FAMILY_BUILTIN_SD15_VAE_REPO, None


def _sd_family_should_load_builtin_vae(
        vae_uri: str | None,
        model_type,
        local_files_only: bool) -> bool:
    """When no ``--vae`` was given, load the SD-family built-in VAE if online."""
    if vae_uri is not None or local_files_only:
        return False
    return _sd_family_model_type_supports_builtin_vae(model_type)


def _sd_family_builtin_vae_uri(
        model_type,
        dtype,
        model_path: str | None = None,
        *,
        local_files_only: bool = False):
    """
    :class:`~dgenerate.pipelinewrapper.uris.VAEUri` for the SD-family built-in VAE.

    :raises ValueError: if ``model_type`` is outside SD 1.x / SD 2 / SDXL.
    """
    import dgenerate.pipelinewrapper.uris as _uris

    source = _sd_family_builtin_vae_source(
        model_type, model_path, local_files_only=local_files_only)
    if source is None:
        raise ValueError(
            f'no SD-family built-in VAE for model type {model_type!r}')
    repo, subfolder = source
    return _uris.VAEUri(
        encoder='AutoencoderKL',
        model=repo,
        subfolder=subfolder,
        dtype=dtype)


def _sd_family_missing_embedded_vae_message(
        local_files_only: bool,
        model_type,
        model_path: str | None = None) -> str:
    """Error text when an SD-family checkpoint has no embedded VAE."""
    import dgenerate.pipelinewrapper.enums as _enums

    model_type_name = _enums.get_model_type_string(model_type)
    source = _sd_family_builtin_vae_source(
        model_type, model_path, local_files_only=local_files_only)
    repo, subfolder = (None, None) if source is None else source
    if repo and subfolder:
        repo = f'{repo};subfolder={subfolder}'
    if local_files_only:
        hint = (
            f'Download a VAE manually with --vae "AutoencoderKL;model={repo}" '
            if repo else
            'Specify a VAE with --vae.'
        )
        return (
            f'The checkpoint has no embedded VAE and --local-files-only is set, '
            f'so the default VAE cannot be fetched for --model-type {model_type_name}. '
            f'{hint}'
        )
    if repo:
        return (
            f'The checkpoint has no embedded VAE for --model-type {model_type_name}. '
            f'Specify one with --vae "AutoencoderKL;model={repo}" or another Hugging Face VAE.'
        )
    return (
        f'The checkpoint has no embedded VAE for --model-type {model_type_name}. '
        'Specify one with --vae.'
    )


def _shapes_from_checkpoint(checkpoint: dict) -> dict[str, tuple[int, ...]]:
    shapes = {}
    for key, value in checkpoint.items():
        shape = getattr(value, 'shape', None)
        shapes[key] = tuple(int(item) for item in shape) if shape is not None else ()
    return shapes


def _name_marks_720p(name: str) -> bool:
    """True for ``720`` or ``720p``. ``7200`` and ``1720`` do not count."""
    return re.search(r'(?<![0-9])720(?![0-9])', name) is not None


def _name_marks_i2v(name: str) -> bool:
    """True for an ``i2v`` task token. ``ti2v`` does not count."""
    return re.search(r'(?<![a-z])i2v(?![a-z])', name) is not None


def _name_marks_wan22(name: str) -> bool:
    """True for a Wan 2.2 MoE name. ``wan2.1`` and ``wan21`` do not count."""
    folded = re.sub(r'[^a-z0-9]', '', name.lower())
    return 'wan22' in folded or 'a14b' in folded or 'highnoise' in folded or 'lownoise' in folded


def _refine_aux_repo(repo: str, hint: str | None) -> str:
    """Pick a sibling repo when the file name names the variant."""
    name = os.path.basename(hint or '').lower()
    if not name:
        return repo
    if repo.startswith('hunyuanvideo-community/HunyuanVideo-1.5'):
        # 480p and 720p, and t2v and i2v, publish the same transformer keys.
        # Resolution and task each come from the file name; one must not
        # replace the other. ``720p_i2v`` is a real community repo.
        resolution = '720p' if _name_marks_720p(name) else '480p'
        task = 'i2v' if _name_marks_i2v(name) else 't2v'
        refined = (
            'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-'
            f'{resolution}_{task}')
        if 'distill' in name:
            refined += '_distilled'
        return refined
    if repo == _ZIMAGE_TURBO and 'turbo' not in name and (
            'z_image' in name or 'z-image' in name):
        return 'Tongyi-MAI/Z-Image'
    if repo == _KREA2_RAW and 'turbo' in name:
        return 'krea/Krea-2-Turbo'
    # Qwen-Image 2.1 and 2.1 Turbo use one transformer layout. Turbo's
    # repository is the distilled checkpoint with the short sigma schedule.
    if repo == _QWEN21 and 'turbo' in name:
        return _QWEN21_TURBO
    # 14B I2V 480p and 720p share the 5120 x 36 patch embedding and the CLIP
    # image embedder. The download name is the only marker, including the
    # Content-Disposition name stored for a web-cache file. Wan 2.2 I2V has
    # no image embedder, so it never reaches this branch. Wan 2.2 T2V uses
    # the same 5120 x 16 patch as Wan 2.1 T2V 14B; the file name is the marker.
    if repo == _WAN_T2V_14 and _name_marks_wan22(name):
        return _WAN_T2V_A14B
    if repo == _WAN_I2V_480 and _name_marks_720p(name):
        return _WAN_I2V_720
    return repo


def _name_marks_qwen_edit_2509(name: str) -> bool:
    """True for the 2509 edit-plus release. ``25090`` does not count."""
    return re.search(r'(?<![0-9])2509(?![0-9])', name) is not None


def _refine_aux_repo_for_model_type(repo: str, model_type, hint: str | None) -> str:
    """
    Apply ``--model-type`` as a hint when a single-file fingerprint matches
    more than one pipeline.

    Shape detection cannot tell Qwen-Image from Qwen-Image-Edit, Flux dev
    from Kontext, or Flux.2 Klein 9B from Klein KV. Other detected repos
    are left alone, including a Qwen-Image 2.1 checkpoint asked for as
    ``qwen-image-edit``.
    """
    if model_type is None:
        return repo
    import dgenerate.pipelinewrapper.enums as _enums

    if repo == _QWEN_IMAGE and _enums.model_type_is_qwen_image_edit(model_type):
        name = os.path.basename(hint or '').lower()
        if _name_marks_qwen_edit_2509(name):
            return _QWEN_IMAGE_EDIT_2509
        return _QWEN_IMAGE_EDIT
    if (repo == _FLUX_DEV
            and _enums.get_model_type_enum(model_type) == _enums.ModelType.FLUX_KONTEXT):
        return _FLUX_KONTEXT
    if repo == _FLUX2_KLEIN_9B and _enums.model_type_is_flux2_klein_kv(model_type):
        return _FLUX2_KLEIN_KV
    return repo


def _diffusers_aux_repo(shapes: dict[str, tuple[int, ...]]) -> str | None:
    from diffusers.loaders.single_file_utils import (
        DIFFUSERS_DEFAULT_PIPELINE_PATHS,
        infer_diffusers_model_type,
    )

    checkpoint = {key: _ShapeTensor(shape) for key, shape in shapes.items()}
    try:
        model_type = infer_diffusers_model_type(checkpoint)
    except Exception:
        return None
    if model_type not in _AUX_DIFFUSERS_TYPES:
        return None
    info = DIFFUSERS_DEFAULT_PIPELINE_PATHS.get(model_type) or {}
    if info.get('subfolder'):
        return None
    return info.get('pretrained_model_name_or_path')


def single_file_pretrained_repo_for_shapes(
        shapes: dict[str, tuple[int, ...]],
        hint: str | None = None,
        model_type=None) -> str | None:
    """
    Hugging Face pipeline repo for a transformer-only checkpoint.

    Checkpoints that already ship text encoders return ``None`` so
    ``from_single_file`` keeps extracting them. ``--text-encoders`` ``+`` and
    ``null`` still apply on top of the repo chosen here.

    ``model_type`` is an optional ``--model-type`` hint for that single-file
    case. It picks an edit or Klein KV sibling when the fingerprint only
    identifies the shared transformer. ``None`` keeps the fingerprint repo.
    A full pipeline repository does not come through here.
    """
    if not shapes or _checkpoint_has_text_encoder(shapes):
        return None
    if is_native_hydit_checkpoint(shapes):
        return infer_hydit_config_repo(shapes)

    from dgenerate.pipelinewrapper.ggufcompat import detect_gguf_layout
    layout = detect_gguf_layout(shapes)
    repo = layout.config_repo if layout is not None else _diffusers_aux_repo(shapes)
    if repo is None:
        return None
    return _refine_aux_repo_for_model_type(_refine_aux_repo(repo, hint), model_type, hint)


def single_file_pretrained_repo_for_checkpoint(
        checkpoint: dict,
        hint: str | None = None,
        model_type=None) -> str | None:
    """Override Diffusers' inferred config repo when dgenerate recognizes the layout."""
    return single_file_pretrained_repo_for_shapes(
        _shapes_from_checkpoint(checkpoint), hint=hint, model_type=model_type)


def load_hunyuan_dit_transformer_single_file(
        transformer_class: type,
        pretrained_model_link_or_path_or_dict: str | dict,
        *,
        config: str | None = None,
        subfolder: str = 'transformer',
        torch_dtype: torch.dtype | None = None,
        token: str | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        device_map: str | None = None,
        quantization_config=None,
        **kwargs) -> torch.nn.Module:
    """
    Load ``HunyuanDiT2DModel`` from a single file.

    Diffusers' ``HunyuanDiT2DModel`` does not mix in ``FromOriginalModelMixin``, so the
    generic single-file path fails even when the class is registered.
    """
    import diffusers.loaders.single_file as _single_file

    if kwargs:
        pass  # accepted for API compatibility with ``from_single_file`` callers

    if isinstance(pretrained_model_link_or_path_or_dict, dict):
        checkpoint = pretrained_model_link_or_path_or_dict
    else:
        checkpoint = _single_file.load_single_file_checkpoint(
            pretrained_model_link_or_path_or_dict,
            token=token,
            local_files_only=local_files_only,
            revision=revision,
        )

    if is_native_hydit_checkpoint(checkpoint):
        state_dict = convert_hydit_transformer_for_single_file(checkpoint)
        config_repo = config or infer_hydit_config_repo(checkpoint)
    else:
        state_dict = checkpoint
        if config is None:
            config_repo = _single_file.fetch_diffusers_config(checkpoint)['pretrained_model_name_or_path']
        else:
            config_repo = config

    load_kwargs = {
        'torch_dtype': torch_dtype,
        'token': token,
        'local_files_only': local_files_only,
        'revision': revision,
        'subfolder': subfolder or 'transformer',
    }
    if device_map is not None:
        load_kwargs['device_map'] = device_map
    if quantization_config is not None:
        load_kwargs['quantization_config'] = quantization_config

    transformer = transformer_class.from_pretrained(config_repo, **load_kwargs)
    transformer.load_state_dict(state_dict, strict=True)
    return transformer


def _safetensors_weight_shapes(path: str) -> dict[str, tuple[int, ...]]:
    with open(path, 'rb') as file:
        header_size = struct.unpack('<Q', file.read(8))[0]
        header = json.loads(file.read(header_size))
    header.pop('__metadata__', None)
    shapes = {}
    for key, info in header.items():
        shape = info.get('shape') if isinstance(info, dict) else None
        shapes[key] = tuple(int(item) for item in shape) if shape else ()
    return shapes


def single_file_pretrained_repo_for_model_path(
        path: str,
        auth_token: str | None = None,
        local_files_only: bool = False,
        model_type=None) -> str | None:
    """
    Hugging Face repo for pipeline parts not stored in a single-file checkpoint.

    Transformer-only files (GGUF, Comfy quantized, native Hunyuan-DiT, and the
    same layouts Diffusers already fingerprints) load text encoders and the VAE
    from that repo. Checkpoints that already contain text encoders return
    ``None`` and keep the ``from_single_file`` path.

    ``model_type`` is an optional ``--model-type`` hint for this single-file
    path. It picks an edit or Klein KV sibling when the tensors match more
    than one pipeline. It is ignored for a pipeline repository.
    """
    if not _hfhub.is_single_file_model_load(path):
        return None
    hint = path
    # A CivitAI link does not end in ``.safetensors``. Diffusers then tries to
    # parse it as a Hugging Face URL and fails with repo id ``(None,)``.
    # Download first, then classify the file that was saved.
    if not os.path.isfile(path):
        path = _hfhub.webcache_or_hf_blob_download(
            path, local_files_only=local_files_only, auth_token=auth_token)
    lower = path.lower()
    if lower.endswith('.gguf'):
        from dgenerate.pipelinewrapper.ggufcompat import read_gguf_shapes
        shapes = read_gguf_shapes(path)
    elif lower.endswith('.safetensors'):
        shapes = _safetensors_weight_shapes(path)
    else:
        import diffusers.loaders.single_file as _single_file
        checkpoint = _single_file.load_single_file_checkpoint(
            path,
            token=auth_token,
            local_files_only=local_files_only,
        )
        return single_file_pretrained_repo_for_shapes(
            _shapes_from_checkpoint(checkpoint),
            hint=_download_name_hint(path, hint),
            model_type=model_type)
    return single_file_pretrained_repo_for_shapes(
        shapes, hint=_download_name_hint(path, hint), model_type=model_type)


def _download_name_hint(path: str, fallback: str) -> str:
    """
    Prefer the ``Content-Disposition`` name stored for a web-cache file.

    The path itself is a generated id after a CivitAI download.
    """
    import dgenerate.webcache as _webcache
    stored = _webcache.cached_content_filename(path)
    return stored or fallback


@functools.cache
def register_single_file_classes() -> None:
    """Install dgenerate's entries in Diffusers' single file loadable class table."""
    import diffusers
    import diffusers.loaders.single_file_model as single_file_model

    loadable = single_file_model.SINGLE_FILE_LOADABLE_CLASSES

    # Diffusers provides the Hunyuan 1.5 architecture but (as of 0.41) does
    # not expose it through the single-file loader. Its source checkpoint
    # converter is reusable once the 1.5-only component names are remapped.
    if hasattr(diffusers, 'HunyuanVideo15Transformer3DModel'):
        loadable.setdefault(
            'HunyuanVideo15Transformer3DModel', {
                'checkpoint_mapping_fn': convert_hunyuan_video15_transformer_checkpoint,
                'default_subfolder': 'transformer',
            })

    # Kandinsky 5 is likewise not registered. Its checkpoints use the Diffusers
    # tensor names, so they load unconverted.
    if hasattr(diffusers, 'Kandinsky5Transformer3DModel'):
        loadable.setdefault(
            'Kandinsky5Transformer3DModel', {
                'checkpoint_mapping_fn': lambda checkpoint, **kwargs: checkpoint,
                'default_subfolder': 'transformer',
            })

    # LTX 2.5 adds prompt modulation tensors Diffusers' LTX 2 conversion
    # leaves unrenamed, so they would stay on the meta device.
    ltx2 = loadable.get('LTX2VideoTransformer3DModel')
    if ltx2 is not None:
        ltx2['checkpoint_mapping_fn'] = convert_ltx2_transformer_checkpoint

    if hasattr(diffusers, 'HunyuanImageTransformer2DModel'):
        loadable['HunyuanImageTransformer2DModel'] = {
            'checkpoint_mapping_fn': convert_hunyuanimage_transformer_for_single_file,
            'default_subfolder': 'transformer',
        }

    if hasattr(diffusers, 'HunyuanDiT2DModel'):
        loadable['HunyuanDiT2DModel'] = {
            'checkpoint_mapping_fn': convert_hydit_transformer_for_single_file,
            'default_subfolder': 'transformer',
        }

        @classmethod
        def _hydit_from_single_file(cls, pretrained_model_link_or_path_or_dict=None, **kwargs):
            return load_hunyuan_dit_transformer_single_file(
                cls, pretrained_model_link_or_path_or_dict, **kwargs)

        diffusers.HunyuanDiT2DModel.from_single_file = _hydit_from_single_file
