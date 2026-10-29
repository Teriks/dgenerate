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

"""LoRA loading for Krea 2.

Diffusers accepts Diffusers ``transformer.*`` keys and the reference
``diffusion_model.`` / AI-Toolkit ``base_model.model.`` layouts. Musubi writes
``lora_unet_`` keys with Kohya ``lora_down`` / ``lora_up`` / ``alpha`` tensors.
LoKr checkpoints are handed to the LoKr loader.
"""

from __future__ import annotations

import typing

import torch

from dgenerate.pipelinewrapper.lokr import (
    apply_lokr_state_dict,
    fetch_lora_state_dict,
    state_dict_has_lokr,
)

if typing.TYPE_CHECKING:
    from diffusers import DiffusionPipeline


def _is_krea2_diffusers_lora(state_dict: dict[str, torch.Tensor]) -> bool:
    return bool(state_dict) and all(
        key.startswith("transformer.") and "lora" in key for key in state_dict)


# Musubi flattens ``.`` to ``_``. These module names already contain underscores,
# so they have to be restored before the remaining underscores become dots.
_KREA2_KOHYA_TOKENS = (
    "layerwise_blocks",
    "refiner_blocks",
)

_KREA2_LORA_SUFFIXES = (
    ".lora_down.weight",
    ".lora_up.weight",
    ".lora_down.bias",
    ".lora_up.bias",
    ".lora_A.weight",
    ".lora_B.weight",
    ".lora_A.bias",
    ".lora_B.bias",
    ".alpha",
)


def _unflatten_krea2_module(module: str) -> str:
    placeholders = []
    for token in _KREA2_KOHYA_TOKENS:
        if token not in module:
            continue
        module = module.replace(token, f"\x00{len(placeholders)}\x00")
        placeholders.append(token)
    module = module.replace("_", ".")
    for index, token in enumerate(placeholders):
        module = module.replace(f"\x00{index}\x00", token)
    return module


def _normalize_krea2_lora_key(key: str) -> str:
    """Turn a Musubi ``lora_unet_`` key into the original dotted module name."""
    if key.startswith("lora_unet_"):
        body = key[len("lora_unet_"):]
        for suffix in _KREA2_LORA_SUFFIXES:
            if body.endswith(suffix):
                module = body[:-len(suffix)]
                if "." not in module:
                    module = _unflatten_krea2_module(module)
                key = module + suffix
                break
    return key.replace(".lora_down.", ".lora_A.").replace(".lora_up.", ".lora_B.")


def _fold_krea2_alphas(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fold Kohya ``.alpha`` into ``lora_B`` and drop the alpha tensors.

    Diffusers loads Krea 2 with ``network_alphas=None``. A leftover ``.alpha``
    key is treated as an unknown weight and the load fails. PEFT then uses
    ``alpha == rank``, so the Kohya ``alpha / rank`` factor has to live in
    the weights.
    """
    alphas = {}
    kept = {}
    for key, value in state_dict.items():
        if key.endswith(".alpha"):
            alphas[key[:-len(".alpha")]] = value
        else:
            kept[key] = value
    for module, alpha in alphas.items():
        down_key = module + ".lora_A.weight"
        up_key = module + ".lora_B.weight"
        if down_key not in kept or up_key not in kept:
            raise ValueError(f'LoRA alpha for "{module}" has no down/up pair.')
        rank = int(kept[down_key].shape[0])
        scale = float(alpha.detach().float().reshape(-1)[0].item()) / float(rank)
        if scale != 1.0:
            kept[up_key] = kept[up_key] * scale
    return kept


def _prepare_krea2_community_lora(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {_normalize_krea2_lora_key(key): value for key, value in state_dict.items()}
    return _fold_krea2_alphas(normalized)


def convert_krea2_lora_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map a community Krea 2 LoRA onto diffusers names.

    Accepts Diffusers ``transformer.*`` keys, the reference ``diffusion_model.``
    and AI-Toolkit ``base_model.model.`` layouts, and Musubi ``lora_unet_``
    files that store Kohya ``lora_down`` / ``lora_up`` / ``alpha`` tensors.
    """
    if _is_krea2_diffusers_lora(state_dict):
        return state_dict
    if state_dict_has_lokr(state_dict):
        raise ValueError(
            "This Krea 2 LoRA uses LoKr weights. Load it through the LoKr loader.")
    from diffusers.loaders.lora_conversion_utils import (
        _convert_non_diffusers_krea2_lora_to_diffusers,
    )
    return _convert_non_diffusers_krea2_lora_to_diffusers(
        _prepare_krea2_community_lora(state_dict))


def load_krea2_lora_on_pipeline(
        pipeline: DiffusionPipeline,
        model_path,
        adapter_name: str,
        *,
        lora_strength: float = 1.0,
        fuse: bool = True,
        fuse_scale: float = 1.0,
        **kwargs) -> str:
    """Load a Krea 2 LoRA. LoKr checkpoints are merged through PEFT."""
    state_dict, metadata = fetch_lora_state_dict(
        model_path, return_lora_metadata=True, **kwargs)
    if state_dict_has_lokr(state_dict):
        return apply_lokr_state_dict(
            pipeline,
            state_dict,
            adapter_name,
            scale=lora_strength * fuse_scale,
            fuse=fuse,
        )

    converted = convert_krea2_lora_state_dict(state_dict)
    pipeline.load_lora_into_transformer(
        converted,
        transformer=pipeline.transformer,
        adapter_name=adapter_name,
        metadata=metadata,
        _pipeline=pipeline,
    )
    return "peft_lora"
