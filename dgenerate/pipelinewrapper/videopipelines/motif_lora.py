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

"""LoRA loading for Motif-Video.

Diffusers builds the Motif pipelines without a LoRA mixin. The transformer
already uses Diffusers module names, so a PEFT file loads once the mixin
exists. Musubi writes those same names with ``lora_unet_`` and underscores.
"""

from __future__ import annotations

import torch
from diffusers.loaders.lora_base import _fetch_state_dict
from diffusers.loaders.lora_pipeline import LoraBaseMixin
from huggingface_hub.utils import validate_hf_hub_args

import dgenerate.messages as _messages


_PREFIXES = (
    'base_model.model.',
    'model.diffusion_model.',
    'diffusion_model.',
    'transformer.',
    'lora_unet_',
)

# Longest first when applied. ``single_transformer_blocks`` contains
# ``transformer_blocks`` as a substring, so the longer token has to win.
_KOHYA_TOKENS = (
    'single_transformer_blocks',
    'transformer_blocks',
    'norm_added_q',
    'norm_added_k',
    'norm1_context',
    'norm2_context',
    'time_text_embed',
    'timestep_embedder',
    'context_embedder',
    'image_embedder',
    'ff_context',
    'add_q_proj',
    'add_k_proj',
    'add_v_proj',
    'to_add_out',
    'cross_attn',
    'proj_mlp',
    'proj_out',
    'norm_out',
    'x_embedder',
    'to_out',
    'norm_q',
    'norm_k',
    'norm1',
    'norm2',
    'to_q',
    'to_k',
    'to_v',
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


def _normalize_key(key: str) -> str:
    key = _strip_prefix(key)
    for suffix in (
            '.lora_down.weight', '.lora_down.bias',
            '.lora_up.weight', '.lora_up.bias',
            '.lora_A.weight', '.lora_A.bias',
            '.lora_B.weight', '.lora_B.bias',
            '.alpha'):
        if key.endswith(suffix):
            module = key[:-len(suffix)]
            if '.' not in module:
                key = _kohya_module(module) + suffix
            break
    return key.replace('.lora_down.', '.lora_A.').replace('.lora_up.', '.lora_B.')


def _apply_alphas(state: dict) -> tuple[dict, bool]:
    alphas = {}
    kept = {}
    for key, value in state.items():
        if key.endswith('.alpha'):
            alphas[key[:-len('.alpha')]] = value
        else:
            kept[key] = value
    folded = False
    for module, alpha in alphas.items():
        down_key = module + '.lora_A.weight'
        up_key = module + '.lora_B.weight'
        if down_key not in kept or up_key not in kept:
            raise ValueError(f'LoRA alpha for "{module}" has no down/up pair.')
        rank = int(kept[down_key].shape[0])
        scale = float(alpha.detach().float().reshape(-1)[0].item()) / float(rank)
        if scale != 1.0:
            kept[up_key] = kept[up_key] * scale
            folded = True
    return kept, folded or bool(alphas)


def convert_motif_video_lora_state_dict(state_dict, metadata=None):
    """
    Map a Diffusers or Musubi Motif-Video LoRA onto ``transformer.`` keys.

    Kohya ``.alpha`` tensors are folded into ``lora_B``. When that happens the
    PEFT header is dropped so its scale is not applied a second time.

    :param state_dict: LoRA tensors
    :param metadata: safetensors adapter header, or ``None``
    :return: ``(state_dict, metadata)``
    """
    if any('dora_scale' in key for key in state_dict):
        _messages.warning(
            'Dropping DoRA scale tensors from a Motif-Video LoRA. DoRA is not applied.')
        state_dict = {key: value for key, value in state_dict.items() if 'dora_scale' not in key}

    normalized = {_normalize_key(key): value for key, value in state_dict.items()}
    normalized, folded_alpha = _apply_alphas(normalized)
    if folded_alpha:
        metadata = None
    converted = {}
    for key, value in normalized.items():
        if not key.startswith('transformer.'):
            key = 'transformer.' + key
        converted[key] = value
    stale = [key for key in converted if 'lora_unet_' in key or '.lora_down.' in key or '.lora_up.' in key]
    if stale:
        raise ValueError(
            'Could not convert Motif-Video LoRA keys, for example: ' + ', '.join(stale[:3]))
    if not converted or not all('lora' in key for key in converted):
        raise ValueError('No usable Motif-Video LoRA weights were found.')
    return converted, metadata


class MotifVideoLoraLoaderMixin(LoraBaseMixin):
    """LoRA loader for Motif-Video pipelines. The transformer is the only target."""

    _lora_loadable_modules = ['transformer']
    transformer_name = 'transformer'

    @classmethod
    @validate_hf_hub_args
    def lora_state_dict(cls, pretrained_model_name_or_path_or_dict, **kwargs):
        cache_dir = kwargs.pop('cache_dir', None)
        force_download = kwargs.pop('force_download', False)
        proxies = kwargs.pop('proxies', None)
        local_files_only = kwargs.pop('local_files_only', None)
        token = kwargs.pop('token', None)
        revision = kwargs.pop('revision', None)
        subfolder = kwargs.pop('subfolder', None)
        weight_name = kwargs.pop('weight_name', None)
        use_safetensors = kwargs.pop('use_safetensors', None)
        return_lora_metadata = kwargs.pop('return_lora_metadata', False)

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
        state_dict, metadata = convert_motif_video_lora_state_dict(state_dict, metadata)
        if return_lora_metadata:
            return state_dict, metadata
        return state_dict

    def load_lora_weights(self, pretrained_model_name_or_path_or_dict, adapter_name=None, hotswap=False, **kwargs):
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()
        kwargs['return_lora_metadata'] = True
        low_cpu_mem_usage = kwargs.pop('low_cpu_mem_usage', False)
        state_dict, metadata = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)
        self.load_lora_into_transformer(
            state_dict,
            transformer=self.transformer,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=self,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
        )

    @classmethod
    def load_lora_into_transformer(
            cls, state_dict, transformer, adapter_name=None, _pipeline=None,
            low_cpu_mem_usage=False, hotswap=False, metadata=None):
        transformer.load_lora_adapter(
            state_dict,
            network_alphas=None,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=_pipeline,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
        )


def install_motif_video_lora_loader() -> None:
    """Give the Diffusers Motif-Video pipelines ``load_lora_weights``."""
    import diffusers

    for name in ('MotifVideoPipeline', 'MotifVideoImage2VideoPipeline'):
        pipeline_class = getattr(diffusers, name, None)
        if pipeline_class is None:
            continue
        if MotifVideoLoraLoaderMixin in pipeline_class.__mro__:
            continue
        if 'load_lora_weights' in pipeline_class.__dict__:
            continue
        pipeline_class.__bases__ = (MotifVideoLoraLoaderMixin,) + tuple(pipeline_class.__bases__)
