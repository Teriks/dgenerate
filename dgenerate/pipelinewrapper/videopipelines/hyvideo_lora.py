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

"""LoRA loading for HunyuanVideo 1.5.

Diffusers 0.41 builds ``HunyuanVideo15Pipeline`` without a LoRA mixin. The
checkpoints published for that model use the original module names
(``double_blocks.*.img_attn_q`` and friends, sometimes fused QKV, sometimes
Musubi's ``lora_unet_`` keys). This maps those names onto the Diffusers
transformer and installs the loader on the pipeline classes.
"""

from __future__ import annotations

import json

import torch
from diffusers.loaders.lora_base import _fetch_state_dict
from diffusers.loaders.lora_pipeline import HunyuanVideoLoraLoaderMixin
from huggingface_hub.utils import validate_hf_hub_args

import dgenerate.messages as _messages


_PREFIXES = (
    'base_model.model.',
    'model.diffusion_model.',
    'diffusion_model.',
    'transformer.',
    'lora_unet_',
)

# Longest match wins. These are the underscore tokens in original and Musubi keys.
_KOHYA_TOKENS = (
    'individual_token_refiner',
    'adaLN_modulation',
    'cond_type_embedding',
    'input_embedder',
    'img_attn_q_norm',
    'img_attn_k_norm',
    'txt_attn_q_norm',
    'txt_attn_k_norm',
    'self_attn_q_norm',
    'self_attn_k_norm',
    'img_attn_qkv',
    'txt_attn_qkv',
    'self_attn_qkv',
    'img_attn_proj',
    'txt_attn_proj',
    'self_attn_proj',
    'img_attn_q',
    'img_attn_k',
    'img_attn_v',
    'txt_attn_q',
    'txt_attn_k',
    'txt_attn_v',
    'double_blocks',
    'single_blocks',
    'final_layer',
    'norm_final',
    'img_norm1',
    'img_norm2',
    'txt_norm1',
    'txt_norm2',
    'img_mod',
    'txt_mod',
    'img_mlp',
    'txt_mlp',
    'guidance_in',
    'vector_in',
    'byt5_in',
    'vision_in',
    't_embedder',
    'c_embedder',
    'time_in',
    'txt_in',
    'cond_type_embedding',
)

# Fused attention projections. lora_A is copied; lora_B is split on the output.
_FUSED_QKV = {
    'img_attn_qkv': ('attn.to_q', 'attn.to_k', 'attn.to_v'),
    'txt_attn_qkv': ('attn.add_q_proj', 'attn.add_k_proj', 'attn.add_v_proj'),
    'self_attn_qkv': ('attn.to_q', 'attn.to_k', 'attn.to_v'),
}

# Dotted forms used by some trainers, rewritten to the underscore tokens above.
_DOTTED_QKV = (
    ('.img_attn.qkv.', '.img_attn_qkv.'),
    ('.txt_attn.qkv.', '.txt_attn_qkv.'),
    ('.self_attn.qkv.', '.self_attn_qkv.'),
    ('.img_attn.proj.', '.img_attn_proj.'),
    ('.txt_attn.proj.', '.txt_attn_proj.'),
    ('.self_attn.proj.', '.self_attn_proj.'),
    ('.img_attn.q.', '.img_attn_q.'),
    ('.img_attn.k.', '.img_attn_k.'),
    ('.img_attn.v.', '.img_attn_v.'),
    ('.txt_attn.q.', '.txt_attn_q.'),
    ('.txt_attn.k.', '.txt_attn_k.'),
    ('.txt_attn.v.', '.txt_attn_v.'),
)

# ComfyUI writes the same modules with shorter names than the original checkpoint.
_COMFY_ALIASES = (
    ('.img_mod.lin.', '.img_mod.linear.'),
    ('.txt_mod.lin.', '.txt_mod.linear.'),
    ('.modulation.lin.', '.modulation.linear.'),
    ('.img_mlp.0.', '.img_mlp.fc1.'),
    ('.img_mlp.2.', '.img_mlp.fc2.'),
    ('.txt_mlp.0.', '.txt_mlp.fc1.'),
    ('.txt_mlp.2.', '.txt_mlp.fc2.'),
    ('.mlp.0.', '.mlp.fc1.'),
    ('.mlp.2.', '.mlp.fc2.'),
    ('byt5_in.fc1.', 'byt5_in.net.0.proj.'),
    ('byt5_in.fc2.', 'byt5_in.net.2.'),
)

# Applied longest-prefix first. ``final_layer`` must be renamed before the
# generic ``adaLN_modulation.1`` rule, which belongs to the token refiner.
_RENAMES = (
    ('final_layer.adaLN_modulation.1', 'norm_out.linear'),
    ('final_layer.norm_final', 'norm_out.norm'),
    ('final_layer.linear', 'proj_out'),
    ('img_attn_q_norm', 'attn.norm_q'),
    ('img_attn_k_norm', 'attn.norm_k'),
    ('txt_attn_q_norm', 'attn.norm_added_q'),
    ('txt_attn_k_norm', 'attn.norm_added_k'),
    ('self_attn_q_norm', 'attn.norm_q'),
    ('self_attn_k_norm', 'attn.norm_k'),
    ('img_attn_proj', 'attn.to_out.0'),
    ('txt_attn_proj', 'attn.to_add_out'),
    ('self_attn_proj', 'attn.to_out.0'),
    ('img_attn_q', 'attn.to_q'),
    ('img_attn_k', 'attn.to_k'),
    ('img_attn_v', 'attn.to_v'),
    ('txt_attn_q', 'attn.add_q_proj'),
    ('txt_attn_k', 'attn.add_k_proj'),
    ('txt_attn_v', 'attn.add_v_proj'),
    ('img_mod.linear', 'norm1.linear'),
    ('txt_mod.linear', 'norm1_context.linear'),
    ('individual_token_refiner.blocks', 'token_refiner.refiner_blocks'),
    ('adaLN_modulation.1', 'norm_out.linear'),
    ('img_mlp', 'ff'),
    ('txt_mlp', 'ff_context'),
    ('.mlp.', '.ff.'),
    ('fc1', 'net.0.proj'),
    ('fc2', 'net.2'),
    ('double_blocks', 'transformer_blocks'),
    ('txt_in.t_embedder.in_layer', 'context_embedder.time_text_embed.timestep_embedder.linear_1'),
    ('txt_in.t_embedder.out_layer', 'context_embedder.time_text_embed.timestep_embedder.linear_2'),
    ('txt_in.c_embedder.in_layer', 'context_embedder.time_text_embed.text_embedder.linear_1'),
    ('txt_in.c_embedder.out_layer', 'context_embedder.time_text_embed.text_embedder.linear_2'),
    ('txt_in.', 'context_embedder.'),
    ('input_embedder', 'proj_in'),
    ('time_in.in_layer', 'time_embed.timestep_embedder.linear_1'),
    ('time_in.out_layer', 'time_embed.timestep_embedder.linear_2'),
    ('time_in.mlp.0', 'time_embed.timestep_embedder.linear_1'),
    ('time_in.mlp.2', 'time_embed.timestep_embedder.linear_2'),
    ('img_in.proj', 'x_embedder.proj'),
    ('cond_type_embedding.', 'cond_type_embed.'),
    ('vision_in.proj.0.', 'image_embedder.norm_in.'),
    ('vision_in.proj.1.', 'image_embedder.linear_1.'),
    ('vision_in.proj.3.', 'image_embedder.linear_2.'),
    ('vision_in.proj.4.', 'image_embedder.norm_out.'),
    ('byt5_in.net.0.proj.', 'context_embedder_2.linear_1.'),
    ('byt5_in.net.2.', 'context_embedder_2.linear_2.'),
    ('byt5_in.fc3.', 'context_embedder_2.linear_3.'),
    ('byt5_in.layernorm.', 'context_embedder_2.norm.'),
)

_STALE_TOKENS = (
    'double_blocks',
    'single_blocks',
    'img_attn_',
    'txt_attn_',
    'txt_in.',
    'final_layer.',
    'lora_down',
    'lora_up',
    'img_attn_qkv',
    'txt_attn_qkv',
)


def _strip_prefix(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
                break
    return key


def _kohya_module(module: str) -> str:
    """Turn ``double_blocks_0_img_attn_q`` into ``double_blocks.0.img_attn_q``."""
    placeholders = []
    for token in sorted(_KOHYA_TOKENS, key=len, reverse=True):
        if token not in module:
            continue
        module = module.replace(token, f'\x00{len(placeholders)}\x00')
        placeholders.append(token)
    module = module.replace('_', '.')
    for index, token in enumerate(placeholders):
        module = module.replace(f'\x00{index}\x00', token)
    return module


_KOHYA_MARKERS = (
    'double_blocks', 'single_blocks', 'txt_in', 'final_layer', 'img_in',
    'time_in', 'byt5_in', 'vision_in', 'cond_type_embedding',
)


def _normalize_key(key: str) -> str:
    key = _strip_prefix(key)
    for source, target in _DOTTED_QKV:
        key = key.replace(source, target)
    for source, target in _COMFY_ALIASES:
        key = key.replace(source, target)
    for suffix in (
            '.lora_down.weight', '.lora_down.bias',
            '.lora_up.weight', '.lora_up.bias',
            '.lora_A.weight', '.lora_A.bias',
            '.lora_B.weight', '.lora_B.bias',
            '.alpha'):
        if key.endswith(suffix):
            module = key[:-len(suffix)]
            if '.' not in module and any(marker in module for marker in _KOHYA_MARKERS):
                key = _kohya_module(module) + suffix
            break
    return key.replace('.lora_down.', '.lora_A.').replace('.lora_up.', '.lora_B.')


def _swap_scale_shift(weight: torch.Tensor) -> torch.Tensor:
    """Original final-layer AdaLN emits shift then scale; Diffusers emits scale then shift."""
    if weight.ndim < 1 or weight.shape[0] % 2:
        raise ValueError(
            'HunyuanVideo 1.5 final-layer LoRA is not divisible into scale and shift.')
    shift, scale = weight.chunk(2, dim=0)
    return torch.cat((scale, shift), dim=0)


def _apply_alphas(state: dict) -> tuple[dict, bool]:
    """Fold Kohya ``.alpha`` into ``lora_B`` and drop the alpha tensors."""
    alphas = {}
    kept = {}
    for key, value in state.items():
        if key.endswith('.alpha'):
            alphas[key[:-len('.alpha')]] = value
        else:
            kept[key] = value
    baked = False
    for module, alpha in alphas.items():
        down_key = module + '.lora_A.weight'
        up_key = module + '.lora_B.weight'
        if down_key not in kept or up_key not in kept:
            raise ValueError(f'LoRA alpha for "{module}" has no down/up pair.')
        rank = int(kept[down_key].shape[0])
        scale = float(alpha.detach().float().reshape(-1)[0].item()) / float(rank)
        if scale != 1.0:
            kept[up_key] = kept[up_key] * scale
            baked = True
    return kept, baked or bool(alphas)


def _metadata_adapter(metadata) -> dict | None:
    """PEFT adapter fields, from a safetensors header or the parsed header."""
    if not isinstance(metadata, dict) or not metadata:
        return None
    if 'lora_adapter_metadata' in metadata:
        raw = metadata['lora_adapter_metadata']
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        return parsed if isinstance(parsed, dict) else None
    if 'lora_alpha' in metadata or 'r' in metadata:
        return metadata
    return None


def _apply_metadata_alpha(state: dict, metadata):
    """Bake a PEFT ``lora_alpha / r`` that is not 1 into ``lora_B``.

    Diffusers drops the safetensors header while filtering keys by the
    ``transformer.`` prefix, so the alpha stored there would otherwise be ignored.
    """
    parsed = _metadata_adapter(metadata)
    if not parsed or parsed.get('alpha_pattern') or parsed.get('rank_pattern'):
        return state, metadata
    alpha = parsed.get('lora_alpha')
    rank = parsed.get('r')
    if alpha is None or not rank or float(alpha) == float(rank):
        return state, metadata
    scale = float(alpha) / float(rank)
    scaled = {
        key: value * scale if 'lora_B' in key and getattr(value, 'ndim', 0) > 1 else value
        for key, value in state.items()
    }
    return scaled, _neutralize_metadata_scale(metadata)


def _neutralize_metadata_scale(metadata):
    """Keep PEFT from scaling again after alpha was folded into the weights."""
    if not metadata or 'lora_adapter_metadata' not in metadata:
        return metadata
    raw = metadata['lora_adapter_metadata']
    parsed = json.loads(raw) if isinstance(raw, str) else dict(raw)
    if not isinstance(parsed, dict):
        return metadata
    parsed = dict(parsed)
    if 'r' in parsed:
        parsed['lora_alpha'] = parsed['r']
    else:
        parsed.pop('lora_alpha', None)
    updated = dict(metadata)
    updated['lora_adapter_metadata'] = json.dumps(parsed)
    return updated


def _split_fused_qkv(state: dict) -> dict:
    converted = {}
    for key, value in state.items():
        fused = next((token for token in _FUSED_QKV if token in key), None)
        if fused is None:
            converted[key] = value
            continue
        targets = _FUSED_QKV[fused]
        if 'lora_A' in key or key.endswith('.alpha'):
            for target in targets:
                converted[key.replace(fused, target)] = value
            continue
        if 'lora_B' not in key:
            converted[key] = value
            continue
        if value.shape[0] % len(targets):
            raise ValueError(
                f'Fused QKV LoRA "{key}" has output size {value.shape[0]}, '
                f'which does not split across {len(targets)} projections.')
        for target, part in zip(targets, value.chunk(len(targets), dim=0)):
            converted[key.replace(fused, target)] = part
    return converted


def _rename_key(key: str, value: torch.Tensor, *, swap_output: bool = False) -> tuple[str, torch.Tensor]:
    if 'final_layer.adaLN_modulation.1' in key and (swap_output or 'lora_B' in key):
        value = _swap_scale_shift(value)
    for source, target in _RENAMES:
        key = key.replace(source, target)
    if not key.startswith('transformer.'):
        key = 'transformer.' + key
    return key, value


def _diff_parameter_key(key: str) -> str | None:
    """``.diff`` / ``.diff_b`` / ``.diff_m`` become ``.weight`` / ``.bias`` / ``.modulation``."""
    for source, target in (('.diff_b', '.bias'), ('.diff_m', '.modulation'), ('.diff', '.weight')):
        if key.endswith(source):
            return key[:-len(source)] + target
    return None


def _chunk_fused_output(key: str, value: torch.Tensor) -> dict:
    """Split a fused QKV weight or bias across the three Diffusers projections."""
    fused = next((token for token in _FUSED_QKV if token in key), None)
    if fused is None:
        return {key: value}
    targets = _FUSED_QKV[fused]
    if value.shape[0] % len(targets):
        raise ValueError(
            f'Fused QKV tensor "{key}" has size {value.shape[0]} on the output axis, '
            f'which does not split across {len(targets)} projections.')
    return {
        key.replace(fused, target): part
        for target, part in zip(targets, value.chunk(len(targets), dim=0))
    }


def _convert_weight_diffs(diff_state: dict) -> dict:
    """Map LightX2V ``.diff`` and ``.diff_b`` tensors onto Diffusers parameters."""
    converted = {}
    for key, value in diff_state.items():
        parameter_key = _diff_parameter_key(_normalize_key(key))
        if parameter_key is None:
            continue
        for split_key, split_value in _chunk_fused_output(parameter_key, value).items():
            renamed, renamed_value = _rename_key(split_key, split_value, swap_output=True)
            converted[renamed] = renamed_value
    return converted


def _embedding_weight_updates(lora_state: dict) -> tuple[dict, dict]:
    """Merge a LoRA pair into ``cond_type_embed.weight``. PEFT cannot wrap an embedding."""
    prefixes = {
        key.split('.lora_A.', 1)[0]
        for key in lora_state
        if 'cond_type_embed.' in key and '.lora_A.' in key
    }
    if not prefixes:
        return lora_state, {}
    kept = {
        key: value for key, value in lora_state.items()
        if not any(key.startswith(prefix + '.') for prefix in prefixes)
    }
    updates = {}
    for prefix in prefixes:
        down = lora_state.get(prefix + '.lora_A.weight')
        up = lora_state.get(prefix + '.lora_B.weight')
        if down is None or up is None or down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0]:
            _messages.warning(
                'Skipping HunyuanVideo 1.5 LoRA weights for the condition-type embedding. '
                'The up and down matrices do not form a weight update.')
            continue
        updates[prefix + '.weight'] = up @ down
    return kept, updates


def convert_hunyuan_video15_lora_state_dict(state_dict, metadata=None):
    """
    Map an original, Comfy, or Musubi HunyuanVideo 1.5 LoRA onto Diffusers keys.

    Diffusers-format keys (``transformer.transformer_blocks...attn.to_q``) pass
    through. Kohya ``.alpha`` tensors and a PEFT ``lora_alpha`` other than the
    rank are folded into ``lora_B``. The stored PEFT alpha is then set equal to
    the rank so that scale is not applied twice.

    :param state_dict: LoRA tensors
    :param metadata: safetensors header, or ``None``
    :return: ``(lora_state, weight_updates, metadata)``. ``weight_updates`` maps
        Diffusers parameter names to tensors that are added into the module,
        which is how LightX2V applies ``.diff`` and ``.diff_b``.
    """
    if any('dora_scale' in key for key in state_dict):
        _messages.warning(
            'Dropping DoRA scale tensors from a HunyuanVideo 1.5 LoRA. '
            'DoRA is not applied.')
        state_dict = {key: value for key, value in state_dict.items() if 'dora_scale' not in key}

    diff_state = {key: value for key, value in state_dict.items() if _diff_parameter_key(key)}
    state_dict = {key: value for key, value in state_dict.items() if key not in diff_state}
    weight_updates = _convert_weight_diffs(diff_state)

    normalized = {_normalize_key(key): value for key, value in state_dict.items()}
    if any('single_blocks' in key for key in normalized):
        raise ValueError(
            'This LoRA targets HunyuanVideo single-stream blocks. '
            'HunyuanVideo 1.5 has none; use a HunyuanVideo 1.5 LoRA.')
    normalized, folded_alpha = _apply_alphas(normalized)
    if folded_alpha:
        metadata = _neutralize_metadata_scale(metadata)
    else:
        normalized, metadata = _apply_metadata_alpha(normalized, metadata)
    renamed = dict(
        _rename_key(key, value) for key, value in _split_fused_qkv(normalized).items())
    renamed, embedding_updates = _embedding_weight_updates(renamed)
    for key, value in embedding_updates.items():
        weight_updates[key] = weight_updates.get(key, 0) + value
    stale = [
        key for key in (*renamed, *weight_updates)
        if any(token in key for token in _STALE_TOKENS)
    ]
    if stale:
        sample = ', '.join(stale[:3])
        raise ValueError(
            'Could not convert HunyuanVideo 1.5 LoRA keys'
            + (f', for example: {sample}' if sample else '') + '.')
    if renamed and not all('lora' in key for key in renamed):
        raise ValueError('No usable HunyuanVideo 1.5 LoRA weights were found.')
    if not renamed and not weight_updates:
        raise ValueError('No usable HunyuanVideo 1.5 LoRA weights were found.')
    return renamed, weight_updates, metadata


def _read_hunyuan_video15_lora(pretrained_model_name_or_path_or_dict, **kwargs):
    """Fetch a LoRA file and return ``(lora_state, weight_updates, metadata)``."""
    cache_dir = kwargs.pop('cache_dir', None)
    force_download = kwargs.pop('force_download', False)
    proxies = kwargs.pop('proxies', None)
    local_files_only = kwargs.pop('local_files_only', None)
    token = kwargs.pop('token', None)
    revision = kwargs.pop('revision', None)
    subfolder = kwargs.pop('subfolder', None)
    weight_name = kwargs.pop('weight_name', None)
    use_safetensors = kwargs.pop('use_safetensors', None)
    kwargs.pop('return_lora_metadata', None)

    allow_pickle = False
    if use_safetensors is None:
        use_safetensors = True
        allow_pickle = True

    state_dict, metadata = _fetch_state_dict(
        pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
        weight_name=weight_name,
        use_safetensors=use_safetensors,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        token=token,
        revision=revision,
        subfolder=subfolder,
        user_agent={'file_type': 'attn_procs_weights', 'framework': 'pytorch'},
        allow_pickle=allow_pickle,
    )
    return convert_hunyuan_video15_lora_state_dict(state_dict, metadata)


class HunyuanVideo15LoraLoaderMixin(HunyuanVideoLoraLoaderMixin):
    """LoRA loader for HunyuanVideo 1.5 pipelines. The transformer is the only target."""

    @classmethod
    @validate_hf_hub_args
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: str | dict[str, torch.Tensor],
        **kwargs,
    ):
        return_lora_metadata = kwargs.pop('return_lora_metadata', False)
        state_dict, _updates, _metadata = _read_hunyuan_video15_lora(
            pretrained_model_name_or_path_or_dict, **kwargs)
        # A PEFT header lists the original module names (img_attn_q, and so on).
        # Diffusers keeps only keys prefixed with "transformer." before building
        # LoraConfig, so that header becomes {} and PEFT then asks the transformer
        # config for model_type. The scale is already in the weights. Leave the
        # config to be inferred from the converted keys.
        if return_lora_metadata:
            return state_dict, None
        return state_dict

    def load_lora_weights(
            self, pretrained_model_name_or_path_or_dict, adapter_name=None,
            hotswap=False, lora_strength=1.0, **kwargs):
        """Load LoRA matrices and keep LightX2V weight diffs until they are fused."""
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()
        low_cpu_mem_usage = kwargs.pop('low_cpu_mem_usage', False)
        kwargs.pop('return_lora_metadata', None)
        state_dict, updates, _metadata = _read_hunyuan_video15_lora(
            pretrained_model_name_or_path_or_dict, **kwargs)
        if state_dict:
            self.load_lora_into_transformer(
                state_dict,
                transformer=self.transformer,
                adapter_name=adapter_name,
                metadata=None,
                _pipeline=self,
                low_cpu_mem_usage=low_cpu_mem_usage,
                hotswap=hotswap,
            )
        if updates:
            pending = getattr(self, '_hy15_weight_updates', None)
            if pending is None:
                pending = {}
                self._hy15_weight_updates = pending
            pending[adapter_name or 'default'] = (updates, float(lora_strength))

    def fuse_lora(
            self, components=('transformer',), lora_scale=1.0, safe_fusing=False,
            adapter_names=None, **kwargs):
        transformer = getattr(self, 'transformer', None)
        if getattr(transformer, 'peft_config', None):
            super().fuse_lora(
                components=list(components),
                lora_scale=lora_scale,
                safe_fusing=safe_fusing,
                adapter_names=adapter_names,
                **kwargs)
        self.commit_lora_weight_updates(lora_scale, adapter_names)

    def commit_lora_weight_updates(self, extra_scale=1.0, adapter_names=None):
        """Add stored ``.diff`` / ``.diff_b`` updates into the transformer parameters."""
        pending = getattr(self, '_hy15_weight_updates', None)
        if not pending:
            return
        names = list(adapter_names) if adapter_names is not None else list(pending)
        for name in names:
            item = pending.pop(name, None)
            if item is None:
                continue
            updates, strength = item
            _apply_parameter_updates(self.transformer, updates, strength * float(extra_scale))


def _apply_parameter_updates(transformer, updates, strength):
    """Add converted LightX2V diffs into floating-point parameters."""
    skipped = []
    applied = 0
    for key, value in updates.items():
        parameter_key = key[len('transformer.'):] if key.startswith('transformer.') else key
        module_name, _, param_name = parameter_key.rpartition('.')
        try:
            module = transformer.get_submodule(module_name) if module_name else transformer
        except AttributeError:
            skipped.append(key)
            continue
        param = getattr(module, param_name, None)
        if (type(module).__name__ == 'ComfyQuantLinear' and param_name == 'weight') or (
                not isinstance(param, torch.Tensor) or tuple(param.shape) != tuple(value.shape)):
            skipped.append(key)
            continue
        delta = value.to(device=param.device, dtype=param.dtype)
        if strength != 1.0:
            delta = delta * strength
        param.data.add_(delta)
        applied += 1
    if skipped:
        _messages.warning(
            f'Skipped {len(skipped)} HunyuanVideo 1.5 weight diffs that did not match a '
            f'floating-point parameter, for example: {skipped[0]}')
    elif applied:
        _messages.debug_log(
            f'Merged {applied} HunyuanVideo 1.5 weight diffs into the transformer.')


def install_hunyuan_video15_lora_loader() -> None:
    """Give the Diffusers HunyuanVideo 1.5 pipelines ``load_lora_weights``."""
    import diffusers

    for name in ('HunyuanVideo15Pipeline', 'HunyuanVideo15ImageToVideoPipeline'):
        pipeline_class = getattr(diffusers, name, None)
        if pipeline_class is None:
            continue
        if HunyuanVideo15LoraLoaderMixin in pipeline_class.__mro__:
            continue
        # A later Diffusers release may ship its own loader. Leave that in place.
        if 'load_lora_weights' in pipeline_class.__dict__:
            continue
        pipeline_class.__bases__ = (HunyuanVideo15LoraLoaderMixin,) + tuple(
            pipeline_class.__bases__)
