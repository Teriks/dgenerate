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

"""Load LyCORIS / AI-Toolkit LoKr checkpoints."""

from __future__ import annotations

import os
import typing
from collections import defaultdict

import torch
import torch.nn as nn

import dgenerate.messages as _messages
from diffusers.loaders.lora_base import _fetch_state_dict

if typing.TYPE_CHECKING:
    from diffusers import DiffusionPipeline

_LOKR_SUFFIXES = (
    ".lokr_w1",
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w2",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_t2",
    ".alpha",
)

_COMPONENTS = (
    "transformer",
    "transformer_2",
    "unet",
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
)


def state_dict_has_lokr(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(".lokr_w1" in key for key in state_dict)


def _checkpoint_file(model_path, weight_name) -> str | None:
    if model_path and os.path.isfile(model_path):
        return model_path
    if not model_path or not os.path.isdir(model_path):
        return None
    names = [weight_name] if weight_name else []
    names.extend((
        "pytorch_lora_weights.safetensors",
        "pytorch_lora_weights.bin",
    ))
    for name in names:
        if not name:
            continue
        candidate = os.path.join(model_path, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def _file_has_lokr(path: str) -> bool:
    if path.endswith(".safetensors"):
        from safetensors import safe_open
        with safe_open(path, framework="pt") as handle:
            return any(".lokr_w1" in key for key in handle.keys())
    return False


def fetch_lora_state_dict(
        model_path,
        *,
        return_lora_metadata: bool = True,
        **kwargs) -> tuple[dict[str, torch.Tensor], dict] | dict[str, torch.Tensor]:
    weight_name = kwargs.pop("weight_name", None)
    use_safetensors = kwargs.pop("use_safetensors", True)
    allow_pickle = use_safetensors is None
    if use_safetensors is None:
        use_safetensors = True
    fetch_kwargs = dict(
        cache_dir=kwargs.pop("cache_dir", None),
        force_download=kwargs.pop("force_download", False),
        proxies=kwargs.pop("proxies", None),
        token=kwargs.pop("token", None),
        revision=kwargs.pop("revision", None),
        subfolder=kwargs.pop("subfolder", None),
        local_files_only=kwargs.pop("local_files_only", None),
    )
    state_dict, metadata = _fetch_state_dict(
        pretrained_model_name_or_path_or_dict=model_path,
        weight_name=weight_name,
        use_safetensors=use_safetensors,
        allow_pickle=allow_pickle,
        user_agent={"file_type": "attn_procs_weights", "framework": "pytorch"},
        **fetch_kwargs,
    )
    state_dict = {key: value for key, value in state_dict.items() if "dora_scale" not in key}
    if return_lora_metadata:
        return state_dict, metadata
    return state_dict


def _group_lokr(state_dict: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    grouped: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for key, value in state_dict.items():
        suffix = next((item for item in _LOKR_SUFFIXES if key.endswith(item)), None)
        if suffix is None:
            continue
        grouped[key[: -len(suffix)]][suffix] = value
    return {
        stem: tensors for stem, tensors in grouped.items()
        if ".lokr_w1" in tensors or ".lokr_w1_a" in tensors
    }


def _probe_modules(pipeline, stem: str) -> list[str]:
    """Rename one LoKr module stem with the pipeline's own LoRA converter."""
    trials = []
    if stem.startswith(("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_")):
        trials.append("down")
    trials.append("ab")
    if "down" not in trials:
        trials.append("down")

    for style in trials:
        if style == "ab":
            dummy = {
                stem + ".lora_A.weight": torch.zeros(1, 1),
                stem + ".lora_B.weight": torch.zeros(1, 1),
            }
        else:
            # Flux Kohya splits fused qkv when the up-weight has 3 rows.
            dummy = {
                stem + ".lora_down.weight": torch.zeros(4, 16),
                stem + ".lora_up.weight": torch.zeros(3, 4),
                stem + ".alpha": torch.tensor(4.0),
            }
        try:
            loaded = pipeline.lora_state_dict(dummy, return_lora_metadata=True)
        except Exception:
            continue
        state = loaded[0] if isinstance(loaded, tuple) else loaded
        if state and not any("lora_A" in key or "lora_B" in key for key in state):
            try:
                from diffusers.utils import convert_unet_state_dict_to_peft
                state = convert_unet_state_dict_to_peft(dict(state))
            except Exception:
                pass
        modules = []
        for key in state:
            key = key.replace("..", ".")
            if ".lora_A" in key:
                modules.append(key.split(".lora_A", 1)[0])
            elif ".lora.down." in key:
                modules.append(key.split(".lora.down.", 1)[0])
        ordered = list(dict.fromkeys(modules))
        if ordered:
            return ordered
    return [stem]


def _find_module(pipeline, qualified: str) -> tuple[nn.Module, str, nn.Module] | None:
    """Return ``(component, relative_path, module)`` for a converted LoRA key."""
    if "." in qualified:
        head, tail = qualified.split(".", 1)
        if head in _COMPONENTS:
            component = getattr(pipeline, head, None)
            if isinstance(component, nn.Module):
                try:
                    return component, tail, _resolve(component, tail)
                except AttributeError:
                    return None
    for name in _COMPONENTS:
        component = getattr(pipeline, name, None)
        if not isinstance(component, nn.Module):
            continue
        try:
            return component, qualified, _resolve(component, qualified)
        except AttributeError:
            continue
    return None


def _resolve(root: nn.Module, path: str) -> nn.Module:
    module = root
    for part in path.split("."):
        module = getattr(module, part)
    return module


def _make_kron(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    if w1.dim() != w2.dim():
        for _ in range(w2.dim() - w1.dim()):
            w1 = w1.unsqueeze(-1)
    return torch.kron(w1.contiguous(), w2.contiguous())


def _lokr_w1(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    if ".lokr_w1" in tensors:
        return tensors[".lokr_w1"]
    if ".lokr_w1_a" in tensors and ".lokr_w1_b" in tensors:
        return tensors[".lokr_w1_a"] @ tensors[".lokr_w1_b"]
    raise ValueError("LoKr module is missing lokr_w1 weights.")


def _lokr_w2(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    if ".lokr_w2" in tensors:
        return tensors[".lokr_w2"]
    if ".lokr_w2_a" in tensors and ".lokr_w2_b" in tensors:
        if ".lokr_t2" in tensors:
            return torch.einsum(
                "i j ..., i p, j r -> p r ...",
                tensors[".lokr_t2"], tensors[".lokr_w2_a"], tensors[".lokr_w2_b"])
        return tensors[".lokr_w2_a"] @ tensors[".lokr_w2_b"]
    raise ValueError("LoKr module is missing lokr_w2 weights.")


def _lokr_dim(tensors: dict[str, torch.Tensor]) -> int | None:
    """LyCORIS ``lora_dim`` for a factorized side."""
    if ".lokr_t2" in tensors:
        return int(tensors[".lokr_t2"].shape[0])
    if ".lokr_w2_a" in tensors and ".lokr_w2_b" in tensors:
        w2_a = tensors[".lokr_w2_a"]
        w2_b = tensors[".lokr_w2_b"]
        if w2_a.shape[-1] == w2_b.shape[0]:
            return int(w2_a.shape[-1])
    if ".lokr_w1_a" in tensors and ".lokr_w1_b" in tensors:
        w1_a = tensors[".lokr_w1_a"]
        w1_b = tensors[".lokr_w1_b"]
        if w1_a.shape[-1] == w1_b.shape[0]:
            return int(w1_a.shape[-1])
    return None


def _lokr_scale(tensors: dict[str, torch.Tensor]) -> float:
    """LyCORIS scale inside the Kronecker product.

    Full ``lokr_w1`` and ``lokr_w2`` are unscaled: LyCORIS forces ``alpha``
    to the rank so the factor is 1. A factorized side uses ``alpha / lora_dim``.
    A missing or zero alpha adds no extra scale.
    """
    if ".lokr_w1" in tensors and ".lokr_w2" in tensors:
        return 1.0
    alpha = tensors.get(".alpha")
    if alpha is None:
        return 1.0
    value = float(alpha.detach().float().reshape(-1)[0].item())
    if value == 0.0:
        return 1.0
    dim = _lokr_dim(tensors)
    if not dim:
        return 1.0
    return value / float(dim)


def _factor_for(linear: nn.Linear, w1: torch.Tensor, w2: torch.Tensor) -> int | None:
    from peft.tuners.lokr.layer import factorization

    if w1.ndim != 2 or w2.ndim < 2:
        return None
    out_features, in_features = linear.weight.shape[:2]
    out_l, out_k = int(w1.shape[0]), int(w2.shape[0])
    in_m, in_n = int(w1.shape[1]), int(w2.shape[1])
    if out_l * out_k != out_features or in_m * in_n != in_features:
        return None
    factor = out_l
    if factorization(out_features, factor) != (out_l, out_k):
        return None
    if factorization(in_features, factor) != (in_m, in_n):
        return None
    return factor


def _full_rank_needed(w2: torch.Tensor) -> int:
    return int(max(w2.shape[:2]))


def _fuse_split_delta(pipeline, modules: list[str], w1, w2, scale: float) -> None:
    """Fused projections (Flux qkv) are one LoKr matrix split across several linears."""
    delta = _make_kron(w1.float(), w2.float()) * scale
    if delta.shape[0] % len(modules) != 0:
        raise RuntimeError(
            "This LoKr tensor maps to "
            f"{len(modules)} modules and its rows cannot be split evenly.")
    chunk = delta.shape[0] // len(modules)
    for index, qualified in enumerate(modules):
        found = _find_module(pipeline, qualified)
        if found is None or not isinstance(found[2], nn.Linear):
            raise RuntimeError(
                f"LoKr weights mapped to {qualified}, which is not a linear layer.")
        linear = found[2]
        piece = delta[index * chunk:(index + 1) * chunk]
        if tuple(piece.shape) != tuple(linear.weight.shape):
            piece = piece.reshape(linear.weight.shape)
        linear.weight.data.add_(piece.to(dtype=linear.weight.dtype, device=linear.weight.device))


def _component_rejects_lokr_fuse(component: nn.Module) -> bool:
    """Packed weights cannot take a LoKr write into ``weight.data``."""
    from dgenerate.pipelinewrapper.pipelines import module_skips_group_offload
    from dgenerate.pipelinewrapper.safetensorscompat import module_has_comfy_quant

    return module_skips_group_offload(component) or module_has_comfy_quant(component)


def _warn_lokr_left_active() -> None:
    _messages.warning(
        'Leaving LoKr adapters active on packed weights. ComfyUI, GGUF, '
        'bitsandbytes, and SDNQ checkpoints cannot take a fused write, so the '
        'LoKr is applied on top of already quantized weights. Quality may be '
        'affected. If the result looks poor, pass --lora-disk-fuse.')


def _restore_merged_linears(component: nn.Module, adapter_name: str, scale: float) -> int:
    from peft.tuners.lokr.layer import LoKrLayer

    merged = 0
    layers = [
        (name, module) for name, module in component.named_modules()
        if isinstance(module, LoKrLayer)
    ]
    for name, layer in layers:
        layer.set_scale(adapter_name, scale)
        layer.merge(adapter_names=[adapter_name])
        parent_name, _, child = name.rpartition(".")
        parent = component.get_submodule(parent_name) if parent_name else component
        setattr(parent, child, layer.get_base_layer())
        merged += 1
    return merged


def apply_lokr_state_dict(
        pipeline,
        state_dict: dict[str, torch.Tensor],
        adapter_name: str,
        *,
        scale: float = 1.0,
        fuse: bool = True) -> str:
    """
    Inject PEFT LoKr for modules that map onto one linear, and merge when ``fuse`` is set.

    Returns ``\"fused_lokr\"`` when the adapters were merged into the base weights.
    """
    from peft import LoKrConfig, inject_adapter_in_model, set_peft_model_state_dict

    grouped = _group_lokr(state_dict)
    if not grouped:
        raise RuntimeError("LoKr checkpoint did not contain lokr_w1 weights.")

    placements = []
    split_jobs = []
    for stem, tensors in grouped.items():
        if ".lokr_w2" not in tensors and ".lokr_w2_a" not in tensors:
            continue
        try:
            w1 = _lokr_w1(tensors)
            w2 = _lokr_w2(tensors)
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        lycoris_scale = _lokr_scale(tensors)
        if lycoris_scale != 1.0:
            w2 = w2 * lycoris_scale
        modules = _probe_modules(pipeline, stem)
        if len(modules) > 1:
            split_jobs.append((modules, w1, w2))
            continue
        found = _find_module(pipeline, modules[0])
        if found is None or not isinstance(found[2], nn.Linear):
            raise RuntimeError(
                f'LoKr module "{stem}" could not be matched to a linear layer '
                f'(mapped name: {modules[0]}).')
        component, relative, linear = found
        factor = _factor_for(linear, w1, w2)
        if factor is None:
            raise RuntimeError(
                f'LoKr shapes for "{relative}" do not match '
                f"{tuple(linear.weight.shape)} with w1 {tuple(w1.shape)} "
                f"and w2 {tuple(w2.shape)}.")
        placements.append((component, relative, factor, w1, w2, _full_rank_needed(w2)))

    by_component: dict[int, list] = defaultdict(list)
    for item in placements:
        by_component[id(item[0])].append(item)

    loaded = 0
    left_active = False
    for items in by_component.values():
        component = items[0][0]
        merge = fuse and not _component_rejects_lokr_fuse(component)
        if fuse and not merge:
            left_active = True
        by_factor: dict[int, list] = defaultdict(list)
        for item in items:
            by_factor[item[2]].append(item)
        for factor, group in by_factor.items():
            rank = max(item[5] for item in group)
            targets = [item[1] for item in group]
            config = LoKrConfig(
                r=rank,
                alpha=rank,
                decompose_factor=factor,
                target_modules=targets,
                init_weights=False,
            )
            inject_adapter_in_model(config, component, adapter_name=adapter_name)
            peft_state = {}
            for _, relative, _, w1, w2, _ in group:
                peft_state[f"{relative}.lokr_w1.{adapter_name}"] = w1
                peft_state[f"{relative}.lokr_w2.{adapter_name}"] = w2
            incompatible = set_peft_model_state_dict(
                component, peft_state, adapter_name=adapter_name)
            unexpected = getattr(incompatible, "unexpected_keys", None) or []
            if unexpected:
                raise RuntimeError(
                    "Could not load part of the LoKr adapter, for example: "
                    + ", ".join(unexpected[:3]))
            if merge:
                loaded += _restore_merged_linears(component, adapter_name, scale)
            else:
                from peft.tuners.lokr.layer import LoKrLayer
                for module in component.modules():
                    if isinstance(module, LoKrLayer):
                        module.set_scale(adapter_name, scale)
                loaded += len(group)

    for modules, w1, w2 in split_jobs:
        packed = False
        for qualified in modules:
            found = _find_module(pipeline, qualified)
            if found is not None and _component_rejects_lokr_fuse(found[0]):
                packed = True
                break
        if packed:
            raise RuntimeError(
                'This LoKr tensor maps to several packed layers and cannot '
                'stay active as one adapter. Pass --lora-disk-fuse to merge '
                'it before quantization.')
        _fuse_split_delta(pipeline, modules, w1, w2, scale)
        loaded += len(modules)

    if left_active:
        _warn_lokr_left_active()
    if loaded == 0:
        raise RuntimeError("No LoKr weights could be loaded onto this pipeline.")
    _messages.debug_log(f"Loaded {loaded} LoKr modules with scale {scale}.")
    if left_active:
        return "lokr"
    if fuse or split_jobs:
        return "fused_lokr"
    return "lokr"


def load_lokr_if_present(
        pipeline: DiffusionPipeline,
        model_path,
        adapter_name: str,
        *,
        lora_strength: float = 1.0,
        fuse: bool = True,
        fuse_scale: float = 1.0,
        **kwargs) -> str | None:
    """Load a LoKr checkpoint when the weight file contains ``lokr_w1`` keys."""
    weight_file = _checkpoint_file(model_path, kwargs.get("weight_name"))
    if weight_file is None or not _file_has_lokr(weight_file):
        return None
    state_dict, _metadata = fetch_lora_state_dict(
        weight_file, return_lora_metadata=True, **kwargs)
    return apply_lokr_state_dict(
        pipeline,
        state_dict,
        adapter_name,
        scale=lora_strength * fuse_scale,
        fuse=fuse,
    )
