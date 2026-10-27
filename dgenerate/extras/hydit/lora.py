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

import os
import re
import typing
import torch
import torch.nn as nn
from diffusers.loaders.lora_pipeline import LoraLoaderMixin
import huggingface_hub
import dgenerate.messages as _messages

_LORA_WEIGHT_CANDIDATES = (
    'pytorch_lora_weights.safetensors',
    'pytorch_lora_weights.bin',
    'adapter_model.safetensors',
    'adapter_model.bin',
    'lora.safetensors',
    'lora.pt',
)


def _lora_weight_candidate_names(weight_name: str | None) -> list[str]:
    if weight_name:
        return [weight_name]
    return list(_LORA_WEIGHT_CANDIDATES)


_KOHYA_HYDIT_TAIL_MAP = {
    'attn1_Wqkv': 'attn1.Wqkv',
    'attn1_out_proj': 'attn1.out_proj',
    'attn2_kv_proj': 'attn2.kv_proj',
    'attn2_out_proj': 'attn2.out_proj',
    'attn2_q_proj': 'attn2.q_proj',
    'mlp_fc1': 'mlp.fc1',
    'mlp_fc2': 'mlp.fc2',
    'default_modulation_1': 'norm1.linear',
    'skip_linear': 'skip_linear',
}

_KOHYA_HYDIT_UNET_RE = re.compile(r'^lora_unet_blocks_(\d+)_(.+)$')

_KOHYA_BERT_RE = re.compile(
    r'^lora_te(\d+)_encoder_layer_(\d+)_('
    r'attention_self_query|attention_self_key|attention_self_value|'
    r'attention_output_dense|intermediate_dense|output_dense)$')

_KOHYA_BERT_TAIL = {
    'attention_self_query': 'attention.self.query',
    'attention_self_key': 'attention.self.key',
    'attention_self_value': 'attention.self.value',
    'attention_output_dense': 'attention.output.dense',
    'intermediate_dense': 'intermediate.dense',
    'output_dense': 'output.dense',
}


def _lora_delta_scale(alpha: float | None, rank: int) -> float:
    """Kohya stores alpha explicitly. PEFT files omit it and use alpha == rank."""
    if alpha is None:
        return 1.0
    return alpha / rank


def _kohya_text_encoder_target(base_k: str) -> tuple[str, str] | None:
    """Map ``lora_te1_encoder_layer_*`` to a BERT ``text_encoder`` module path."""
    match = _KOHYA_BERT_RE.match(base_k)
    if match is None:
        return None
    encoder_index = int(match.group(1))
    module_name = 'text_encoder' if encoder_index == 1 else f'text_encoder_{encoder_index}'
    layer = match.group(2)
    tail = _KOHYA_BERT_TAIL[match.group(3)]
    return module_name, f'encoder.layer.{layer}.{tail}'


def _normalize_hydit_lora_module_key(base_k: str) -> str:
    """Map PEFT, Tencent, or Kohya (lora_unet_*) keys to diffusers module paths."""
    if base_k.startswith('base_model.model.'):
        return base_k[len('base_model.model.'):]
    if base_k.startswith('unet.'):
        return base_k[len('unet.'):]
    if base_k.startswith('transformer.'):
        return base_k[len('transformer.'):]

    match = _KOHYA_HYDIT_UNET_RE.match(base_k)
    if not match:
        if base_k.startswith('lora_unet_'):
            return base_k[len('lora_unet_'):].replace('_', '.')
        return base_k

    block_id, tail = match.group(1), match.group(2)
    mapped_tail = _KOHYA_HYDIT_TAIL_MAP.get(tail)
    if mapped_tail is None:
        _messages.warning(
            f'Unknown Kohya Hunyuan-DiT LoRA module {base_k!r}; skipping.')
        return ''
    return f'blocks.{block_id}.{mapped_tail}'


def _load_state_dict(model_path_or_dict, weight_name=None, subfolder=None, token=None, local_files_only=False):
    if isinstance(model_path_or_dict, dict):
        return model_path_or_dict

    path = str(model_path_or_dict)
    if os.path.isfile(path):
        file_path = path
    elif os.path.isdir(path):
        candidate_names = _lora_weight_candidate_names(weight_name)
        file_path = None
        for name in candidate_names:
            p = os.path.join(path, name)
            if os.path.isfile(p):
                file_path = p
                break
        if file_path is None:
            # Pick any safetensors or bin file
            for f in os.listdir(path):
                if f.endswith(".safetensors") or f.endswith(".bin") or f.endswith(".pt"):
                    file_path = os.path.join(path, f)
                    break
        if file_path is None:
            raise FileNotFoundError(f"Could not find LoRA weights in directory: {path}")
    else:
        file_path = None
        last_error: BaseException | None = None
        for filename in _lora_weight_candidate_names(weight_name):
            try:
                file_path = huggingface_hub.hf_hub_download(
                    repo_id=path,
                    filename=filename,
                    subfolder=subfolder,
                    token=token,
                    local_files_only=local_files_only,
                )
                break
            except Exception as error:
                last_error = error
        if file_path is None:
            if last_error is not None:
                raise last_error
            raise FileNotFoundError(f'Could not find LoRA weights in repository: {path}')

    if file_path.endswith(".safetensors"):
        import safetensors.torch
        return safetensors.torch.load_file(file_path, device="cpu")
    else:
        return torch.load(file_path, map_location="cpu", weights_only=True)


class HunyuanDiTLoraLoaderMixin(LoraLoaderMixin):
    """
    LoRA loader and fuser mixin for HunyuanDiT pipelines.
    Supports Tencent native hydit LoRA checkpoints, Kohya LoRAs, and PEFT/Diffusers LoRAs.
    """

    def get_list_adapters(self) -> dict[str, list[str]]:
        """
        Adapters per pipeline component.

        Diffusers returns this same mapping. A stored delta key is
        ``component\\nparameter``, so a text-encoder LoRA is listed on
        ``text_encoder`` rather than only on ``transformer``.
        """
        listed: dict[str, list[str]] = {}
        for adapter_name, deltas in (getattr(self, '_lora_adapters', None) or {}).items():
            components: list[str] = []
            seen: set[str] = set()
            for delta_key in deltas or ():
                component, separator, _param = str(delta_key).partition('\n')
                if not separator:
                    component = 'transformer'
                if component not in seen:
                    seen.add(component)
                    components.append(component)
            if not components:
                components.append('transformer')
            for component in components:
                listed.setdefault(component, []).append(adapter_name)
        return listed

    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: str | dict[str, torch.Tensor],
        adapter_name: str | None = None,
        weight_name: str | None = None,
        subfolder: str | None = None,
        token: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        if not hasattr(self, "_lora_adapters"):
            self._lora_adapters = {}
            self._lora_adapter_weights = {}
            self._unfused_lora_weights = {}
            self._hydit_lora_fused = False

        if adapter_name is None:
            adapter_name = f"adapter_{len(self._lora_adapters)}"

        state_dict = _load_state_dict(
            pretrained_model_name_or_path_or_dict,
            weight_name=weight_name,
            subfolder=subfolder,
            token=token,
            local_files_only=local_files_only,
        )

        deltas = self._extract_lora_deltas(state_dict)
        if not deltas:
            raise RuntimeError(f"No usable LoRA weights found in {pretrained_model_name_or_path_or_dict}.")

        self._lora_adapters[adapter_name] = deltas
        self._lora_adapter_weights[adapter_name] = 1.0

    def _parameter_banks(self) -> dict[str, dict[str, torch.nn.Parameter]]:
        banks = {'transformer': dict(self.transformer.named_parameters())}
        for name in ('text_encoder', 'text_encoder_2'):
            module = getattr(self, name, None)
            if module is not None and hasattr(module, 'named_parameters'):
                banks[name] = dict(module.named_parameters())
        return banks

    @staticmethod
    def _delta_key(module_name: str, param_name: str) -> str:
        return f'{module_name}\n{param_name}'

    def _store_delta(
            self,
            deltas: dict[str, torch.Tensor],
            module_name: str,
            param_name: str,
            delta: torch.Tensor,
            banks: dict[str, dict[str, torch.nn.Parameter]] | None = None):
        if banks is None:
            banks = self._parameter_banks()
        params = banks.get(module_name)
        if params is None:
            return
        if param_name not in params and module_name.startswith('text_encoder'):
            bert_name = f'bert.{param_name}'
            if bert_name in params:
                param_name = bert_name
        if param_name in params:
            deltas[self._delta_key(module_name, param_name)] = delta
            return
        skipped = getattr(self, '_hydit_lora_skipped', None)
        if skipped is None:
            skipped = []
            self._hydit_lora_skipped = skipped
        skipped.append(f'{module_name}.{param_name}')

    def _extract_lora_deltas(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        banks = self._parameter_banks()
        deltas: dict[str, torch.Tensor] = {}
        self._hydit_lora_skipped = []

        # Scan for up/down pairs
        pairs: dict[str, dict[str, typing.Any]] = {}
        alphas: dict[str, float] = {}

        for k, v in state_dict.items():
            if k.endswith(".alpha"):
                base_k = k[:-len(".alpha")]
                alphas[base_k] = float(v.item() if hasattr(v, "item") else v)
                continue

            is_up = any(k.endswith(s) for s in [".lora.up.weight", ".lora_up.weight", ".lora_B.weight", ".lora_b.weight"])
            is_down = any(k.endswith(s) for s in [".lora.down.weight", ".lora_down.weight", ".lora_A.weight", ".lora_a.weight"])

            if is_up:
                for suffix in [".lora.up.weight", ".lora_up.weight", ".lora_B.weight", ".lora_b.weight"]:
                    if k.endswith(suffix):
                        base_k = k[:-len(suffix)]
                        pairs.setdefault(base_k, {})["up"] = v
                        break
            elif is_down:
                for suffix in [".lora.down.weight", ".lora_down.weight", ".lora_A.weight", ".lora_a.weight"]:
                    if k.endswith(suffix):
                        base_k = k[:-len(suffix)]
                        pairs.setdefault(base_k, {})["down"] = v
                        break

        for base_k, p in pairs.items():
            if "up" not in p or "down" not in p:
                continue

            up = p["up"].float()
            down = p["down"].float()
            rank = down.shape[0] if down.ndim >= 2 else up.shape[1]
            alpha = alphas.get(base_k, None)
            scale = _lora_delta_scale(alpha, rank)

            text_encoder_target = _kohya_text_encoder_target(base_k)
            if text_encoder_target is not None:
                module_name, stem = text_encoder_target
                self._store_delta(
                    deltas, module_name, f'{stem}.weight', (up @ down) * scale)
                continue

            norm_key = _normalize_hydit_lora_module_key(base_k)
            if not norm_key:
                continue

            # Map Tencent-specific module names if applicable
            if "Wqkv" in norm_key:
                # Wqkv chunks into to_q, to_k, to_v
                q_up, k_up, v_up = torch.chunk(up, 3, dim=0)
                for sub_name, sub_up in [("to_q", q_up), ("to_k", k_up), ("to_v", v_up)]:
                    target_param = norm_key.replace("Wqkv", sub_name) + ".weight"
                    self._store_delta(
                        deltas, 'transformer', target_param, (sub_up @ down) * scale, banks)
                continue

            if "kv_proj" in norm_key:
                k_up, v_up = torch.chunk(up, 2, dim=0)
                for sub_name, sub_up in [("to_k", k_up), ("to_v", v_up)]:
                    target_param = norm_key.replace("kv_proj", sub_name) + ".weight"
                    self._store_delta(
                        deltas, 'transformer', target_param, (sub_up @ down) * scale, banks)
                continue

            if norm_key == "pooler.q_proj" or norm_key.endswith("pooler.q_proj"):
                self._store_delta(
                    deltas, 'transformer', 'time_extra_emb.pooler.q_proj.weight',
                    (up @ down) * scale, banks)
                continue

            if norm_key.endswith(".q_proj") or norm_key.endswith("q_proj"):
                target_param = norm_key.replace("q_proj", "to_q") + ".weight"
                self._store_delta(
                    deltas, 'transformer', target_param, (up @ down) * scale, banks)
                continue

            if "out_proj" in norm_key:
                target_param = norm_key.replace("out_proj", "to_out.0") + ".weight"
            elif "mlp.fc1" in norm_key:
                target_param = norm_key.replace("mlp.fc1", "ff.net.0.proj") + ".weight"
            elif "mlp.fc2" in norm_key:
                target_param = norm_key.replace("mlp.fc2", "ff.net.2") + ".weight"
            else:
                target_param = norm_key + ".weight" if not norm_key.endswith(".weight") else norm_key

            self._store_delta(
                deltas, 'transformer', target_param, (up @ down) * scale, banks)

        skipped = getattr(self, '_hydit_lora_skipped', None) or []
        if skipped:
            shown = ', '.join(skipped[:4])
            extra = f' (+{len(skipped) - 4} more)' if len(skipped) > 4 else ''
            _messages.warning(
                'Hunyuan-DiT LoRA weights did not match a parameter and were skipped: '
                f'{shown}{extra}.')
        return deltas

    def set_adapters(self, adapter_names: list[str], adapter_weights: list[float] | None = None):
        if not hasattr(self, "_lora_adapters"):
            return
        if adapter_weights is None:
            adapter_weights = [1.0] * len(adapter_names)
        for name, weight in zip(adapter_names, adapter_weights):
            if name in self._lora_adapters:
                self._lora_adapter_weights[name] = float(weight)
        if getattr(self, '_hydit_lora_fused', False):
            self.fuse_lora(lora_scale=getattr(self, '_hydit_lora_fuse_scale', 1.0))

    def _add_lora_deltas(self, adapter_names: list[str] | None = None, lora_scale: float = 1.0):
        if not hasattr(self, "_lora_adapters") or not self._lora_adapters:
            return
        if not hasattr(self, "_unfused_lora_weights"):
            self._unfused_lora_weights = {}

        if adapter_names is None:
            adapter_names = list(self._lora_adapters.keys())

        banks = self._parameter_banks()

        for name in adapter_names:
            if name not in self._lora_adapters:
                continue
            weight_scale = self._lora_adapter_weights.get(name, 1.0) * lora_scale
            deltas = self._lora_adapters[name]

            for delta_key, delta in deltas.items():
                module_name, param_name = delta_key.split('\n', 1)
                params = banks.get(module_name, {})
                if param_name not in params:
                    continue
                param = params[param_name]
                if delta_key not in self._unfused_lora_weights:
                    self._unfused_lora_weights[delta_key] = param.data.detach().clone()
                param.data.add_(delta.to(device=param.device, dtype=param.dtype) * weight_scale)

    def _restore_lora_snapshots(self):
        snapshots = getattr(self, '_unfused_lora_weights', None)
        if not snapshots:
            return
        banks = self._parameter_banks()
        for delta_key, orig in snapshots.items():
            module_name, param_name = delta_key.split('\n', 1)
            params = banks.get(module_name, {})
            if param_name in params:
                params[param_name].data.copy_(orig)
        snapshots.clear()

    def fuse_lora(self, adapter_names: list[str] | None = None, lora_scale: float = 1.0):
        self._hydit_lora_fuse_scale = lora_scale
        if getattr(self, '_hydit_lora_fused', False):
            # A second fuse used to add the same delta again.
            self._restore_lora_snapshots()
            self._hydit_lora_fused = False
        self._add_lora_deltas(adapter_names, lora_scale)
        if getattr(self, '_lora_adapters', None):
            self._hydit_lora_fused = True

    def unload_lora_weights(self):
        self._restore_lora_snapshots()
        self._hydit_lora_fused = False

        if hasattr(self, "_lora_adapters"):
            self._lora_adapters.clear()
            self._lora_adapter_weights.clear()
