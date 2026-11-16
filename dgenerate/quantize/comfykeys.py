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
Diffusers tensor names rewritten to the names ComfyUI loaders look up.

Diffusion models are the inverse of the Diffusers single-file converters, so a
round trip through that converter restores the original keys and values.
GGUF text encoders use the llama.cpp names ComfyUI-GGUF remaps. Safetensors
text encoders keep Hugging Face names, which is what Comfy's safetensors text
encoder loader reads.
"""

import torch

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.ggufcompat as _ggufcompat
import dgenerate.quantize.quantexport as _quantexport

_QuantExportError = _quantexport.QuantExportError

# ComfyUI-GGUF refuses an image GGUF whose general.architecture is not in this set.
_GGUF_IMAGE_ARCHITECTURES = frozenset({
    'flux', 'sd1', 'sdxl', 'sd3', 'aura', 'hidream', 'cosmos',
    'ltxv', 'hyvid', 'wan', 'lumina2', 'qwen_image',
})

# ComfyUI-GGUF un-permutes llama GGUFs with these head counts hardcoded.
_COMFY_LLAMA_HEADS = (32, 8)

# Wan renames, in the order Diffusers applies them. head.modulation, modulation,
# head.head, and after_proj are applied separately: each pair shares a target.
_WAN_PAIRS = (
    ('time_embedding.0', 'condition_embedder.time_embedder.linear_1'),
    ('time_embedding.2', 'condition_embedder.time_embedder.linear_2'),
    ('text_embedding.0', 'condition_embedder.text_embedder.linear_1'),
    ('text_embedding.2', 'condition_embedder.text_embedder.linear_2'),
    ('time_projection.1', 'condition_embedder.time_proj'),
    ('cross_attn', 'attn2'),
    ('self_attn', 'attn1'),
    ('.o.', '.to_out.0.'),
    ('.q.', '.to_q.'),
    ('.k.', '.to_k.'),
    ('.v.', '.to_v.'),
    ('.k_img.', '.add_k_proj.'),
    ('.v_img.', '.add_v_proj.'),
    ('.norm_k_img.', '.norm_added_k.'),
    ('ffn.0', 'ffn.net.0.proj'),
    ('ffn.2', 'ffn.net.2'),
    ('norm2', 'norm__placeholder'),
    ('norm3', 'norm2'),
    ('norm__placeholder', 'norm3'),
    ('img_emb.proj.0', 'condition_embedder.image_embedder.norm1'),
    ('img_emb.proj.1', 'condition_embedder.image_embedder.ff.net.0.proj'),
    ('img_emb.proj.3', 'condition_embedder.image_embedder.ff.net.2'),
    ('img_emb.proj.4', 'condition_embedder.image_embedder.norm2'),
    ('before_proj', 'proj_in'),
)

_ZIMAGE_PAIRS = (
    ('all_final_layer.2-1.', 'final_layer.'),
    ('all_x_embedder.2-1.', 'x_embedder.'),
    ('.attention.to_out.0.bias', '.attention.out.bias'),
    ('.attention.norm_k.weight', '.attention.k_norm.weight'),
    ('.attention.norm_q.weight', '.attention.q_norm.weight'),
    ('.attention.to_out.0.weight', '.attention.out.weight'),
)


def comfy_state(state: dict, config: dict | None, *, gguf: bool) -> tuple[dict, str]:
    """
    A copy of ``state`` under ComfyUI names, and the GGUF architecture string.

    The architecture is what ComfyUI-GGUF reads from ``general.architecture``.
    Safetensors text encoders are returned unchanged, with that label still set
    so the log can name the component.

    :raises QuantExportError: This component has no Comfy key map, or a GGUF of
        it is not a ComfyUI-GGUF architecture.
    """
    copied = {
        key: value.detach().cpu().contiguous()
        for key, value in state.items() if isinstance(value, torch.Tensor)}
    kind = _kind(copied, config)
    converted, label = _convert(copied, config, kind, gguf)
    if gguf and label not in _GGUF_IMAGE_ARCHITECTURES and label not in _ggufcompat._TEXT_GGUF_ARCHITECTURES:
        raise _QuantExportError(
            f'ComfyUI-GGUF does not load a {label} GGUF. '
            f'Safetensors quants of this component still use Comfy names.')
    if gguf and label == 'gemma4':
        raise _QuantExportError(
            'ComfyUI-GGUF does not load a Gemma 4 GGUF. '
            'A safetensors quant keeps the Hugging Face names Comfy can read.')
    return converted, label


def _kind(state: dict, config: dict | None) -> str:
    text = _text_kind(config, state)
    if text:
        return text
    name = ''
    if config:
        name = str(config.get('_class_name') or '')
        if not name:
            architectures = config.get('architectures') or []
            name = str(architectures[0]) if architectures else ''
    keys = set(state)
    if 'Flux2' in name or any('to_qkv_mlp_proj' in key for key in keys):
        return 'flux2'
    if ('Flux' in name or (
            'x_embedder.weight' in keys
            and any(key.startswith('single_transformer_blocks.') for key in keys)
            and 'pos_embed.pos_embed' not in keys)):
        return 'flux'
    if 'SD3' in name or 'pos_embed.pos_embed' in keys:
        return 'sd3'
    if 'Wan' in name or 'condition_embedder.time_embedder.linear_1.weight' in keys:
        return 'wan'
    if 'ZImage' in name or any(key.startswith('all_x_embedder.2-1.') for key in keys):
        return 'z-image'
    if 'QwenImage21' in name or 'QwenImage2' in name or any(
            key.endswith('img_mlp.gate_layer.weight') for key in keys):
        return 'qwen_image'
    if 'QwenImage' in name or any(
            key.startswith('transformer_blocks.') and 'img_mlp.net.' in key for key in keys):
        return 'qwen_image'
    if 'UNet' in name or any('.resnets.' in key and key.startswith('down_blocks.') for key in keys):
        return 'unet'
    raise _QuantExportError(
        'No ComfyUI key map for this component. '
        '--comfy converts FLUX.1, FLUX.2 safetensors, SD1, SDXL, SD3, Wan, Qwen-Image, '
        'and Z-Image safetensors, plus text encoders T5, UMT5, Llama 3 / Mistral, Qwen2, '
        'Qwen3, Gemma 3, and CLIP safetensors. A Qwen2-VL vision tower is written as mmproj.')


def _text_kind(config: dict | None, state: dict) -> str | None:
    model_type = ''
    if config:
        model_type = str(config.get('model_type') or '')
        if not model_type:
            architectures = config.get('architectures') or []
            if architectures:
                model_type = str(architectures[0])
    architecture = _quantexport._encoder_only_t5(
        config, _quantexport._text_architecture(model_type)) if model_type else None
    if architecture:
        return architecture
    if any(key.startswith('encoder.block.') or key == 'shared.weight' for key in state):
        return 't5'
    lowered = model_type.lower()
    if lowered.startswith('clip'):
        return 'clip'
    return None


def _convert(state: dict, config: dict | None, kind: str, gguf: bool) -> tuple[dict, str]:
    if kind == 'clip':
        if gguf:
            raise _QuantExportError(
                'ComfyUI-GGUF does not load a CLIP GGUF. '
                'A safetensors quant keeps the Hugging Face names Comfy\'s CLIP loader reads.')
        return state, 'clip'
    if kind == 'flux2':
        if gguf:
            raise _QuantExportError(
                'ComfyUI-GGUF does not load a FLUX.2 GGUF. '
                'A safetensors quant uses the Comfy tensor names.')
        return _flux2(state), 'flux2'
    if kind == 'flux':
        return _flux(state), 'flux'
    if kind == 'sd3':
        return _sd3(state), 'sd3'
    if kind == 'wan':
        return _wan(state), 'wan'
    if kind == 'z-image':
        return _zimage(state), 'z-image'
    if kind == 'qwen_image':
        return _qwen21(state), 'qwen_image'
    if kind == 'unet':
        return _unet(state, config)
    if kind in {'t5', 't5encoder', 'llama', 'qwen2', 'qwen2vl', 'qwen3', 'qwen3vl', 'gemma3', 'gemma4'}:
        converted = _text(state, config, kind, gguf)
        if gguf and kind == 'qwen2':
            # ComfyUI-GGUF's text list has qwen3 and not qwen2. Both use LLAMA_SD_MAP,
            # and qwen3 does not permute attention or require an mmproj. Comfy then
            # tells Qwen2.5 from model.layers.*.k_proj.bias.
            _messages.warning(
                'ComfyUI-GGUF has no qwen2 architecture. This GGUF is labeled qwen3, '
                'which uses the same tensor map and does not require an mmproj.')
            return converted, 'qwen3'
        return converted, kind
    raise _QuantExportError(f'No ComfyUI key map for {kind}.')


def _take(state: dict, key: str) -> torch.Tensor:
    try:
        return state.pop(key)
    except KeyError:
        raise _QuantExportError(f'Comfy conversion expected "{key}".') from None


def _cat(parts: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(parts, dim=0)


def _swap_halves(weight: torch.Tensor) -> torch.Tensor:
    """Inverse of Diffusers' scale/shift swap. Swapping twice restores the tensor."""
    first, second = weight.chunk(2, dim=0)
    return torch.cat([second, first], dim=0)


def _block_indices(state: dict, prefix: str) -> list[int]:
    found = set()
    for key in state:
        if not key.startswith(prefix):
            continue
        head = key[len(prefix):].split('.', 1)[0]
        if head.isdigit():
            found.add(int(head))
    return sorted(found)


def _store(out: dict, key: str, weight: torch.Tensor, bias: torch.Tensor | None):
    out[key + '.weight'] = weight
    if bias is not None:
        out[key + '.bias'] = bias


def _pair(state: dict, key: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    return _take(state, key + '.weight'), state.pop(key + '.bias', None)


def _leftover(state: dict):
    if not state:
        return
    shown = ', '.join(list(state)[:6])
    raise _QuantExportError(
        f'Comfy conversion left {len(state)} tensors unused ({shown}).')


def _flux(state: dict) -> dict:
    out = {}
    for source, target in (
            ('time_text_embed.timestep_embedder.linear_1', 'time_in.in_layer'),
            ('time_text_embed.timestep_embedder.linear_2', 'time_in.out_layer'),
            ('time_text_embed.text_embedder.linear_1', 'vector_in.in_layer'),
            ('time_text_embed.text_embedder.linear_2', 'vector_in.out_layer')):
        weight, bias = _pair(state, source)
        _store(out, target, weight, bias)
    if 'time_text_embed.guidance_embedder.linear_1.weight' in state:
        for source, target in (
                ('time_text_embed.guidance_embedder.linear_1', 'guidance_in.in_layer'),
                ('time_text_embed.guidance_embedder.linear_2', 'guidance_in.out_layer')):
            weight, bias = _pair(state, source)
            _store(out, target, weight, bias)
    for source, target in (('context_embedder', 'txt_in'), ('x_embedder', 'img_in')):
        weight, bias = _pair(state, source)
        _store(out, target, weight, bias)

    for index in _block_indices(state, 'transformer_blocks.'):
        prefix = f'transformer_blocks.{index}.'
        block = f'double_blocks.{index}.'
        for source, target in (
                ('norm1.linear', 'img_mod.lin'),
                ('norm1_context.linear', 'txt_mod.lin')):
            weight, bias = _pair(state, prefix + source)
            _store(out, block + target, weight, bias)
        for sources, target in (
                (('attn.to_q', 'attn.to_k', 'attn.to_v'), 'img_attn.qkv'),
                (('attn.add_q_proj', 'attn.add_k_proj', 'attn.add_v_proj'), 'txt_attn.qkv')):
            weights = []
            biases = []
            for source in sources:
                weight, bias = _pair(state, prefix + source)
                weights.append(weight)
                biases.append(bias)
            out[block + target + '.weight'] = _cat(weights)
            if all(bias is not None for bias in biases):
                out[block + target + '.bias'] = _cat(biases)
        for source, target in (
                ('attn.norm_q.weight', 'img_attn.norm.query_norm.scale'),
                ('attn.norm_k.weight', 'img_attn.norm.key_norm.scale'),
                ('attn.norm_added_q.weight', 'txt_attn.norm.query_norm.scale'),
                ('attn.norm_added_k.weight', 'txt_attn.norm.key_norm.scale')):
            out[block + target] = _take(state, prefix + source)
        for source, target in (
                ('ff.net.0.proj', 'img_mlp.0'),
                ('ff.net.2', 'img_mlp.2'),
                ('ff_context.net.0.proj', 'txt_mlp.0'),
                ('ff_context.net.2', 'txt_mlp.2'),
                ('attn.to_out.0', 'img_attn.proj'),
                ('attn.to_add_out', 'txt_attn.proj')):
            weight, bias = _pair(state, prefix + source)
            _store(out, block + target, weight, bias)

    for index in _block_indices(state, 'single_transformer_blocks.'):
        prefix = f'single_transformer_blocks.{index}.'
        block = f'single_blocks.{index}.'
        weight, bias = _pair(state, prefix + 'norm.linear')
        _store(out, block + 'modulation.lin', weight, bias)
        parts = ['attn.to_q', 'attn.to_k', 'attn.to_v', 'proj_mlp']
        weights = []
        biases = []
        for source in parts:
            weight, bias = _pair(state, prefix + source)
            weights.append(weight)
            biases.append(bias)
        out[block + 'linear1.weight'] = _cat(weights)
        if all(item is not None for item in biases):
            out[block + 'linear1.bias'] = _cat(biases)
        out[block + 'norm.query_norm.scale'] = _take(state, prefix + 'attn.norm_q.weight')
        out[block + 'norm.key_norm.scale'] = _take(state, prefix + 'attn.norm_k.weight')
        weight, bias = _pair(state, prefix + 'proj_out')
        _store(out, block + 'linear2', weight, bias)

    weight, bias = _pair(state, 'proj_out')
    _store(out, 'final_layer.linear', weight, bias)
    out['final_layer.adaLN_modulation.1.weight'] = _swap_halves(_take(state, 'norm_out.linear.weight'))
    bias = state.pop('norm_out.linear.bias', None)
    if bias is not None:
        out['final_layer.adaLN_modulation.1.bias'] = _swap_halves(bias)
    _leftover(state)
    return out


def _sd3(state: dict) -> dict:
    out = {}
    out['pos_embed'] = _take(state, 'pos_embed.pos_embed')
    weight, bias = _pair(state, 'pos_embed.proj')
    _store(out, 'x_embedder.proj', weight, bias)
    for source, target in (
            ('time_text_embed.timestep_embedder.linear_1', 't_embedder.mlp.0'),
            ('time_text_embed.timestep_embedder.linear_2', 't_embedder.mlp.2'),
            ('time_text_embed.text_embedder.linear_1', 'y_embedder.mlp.0'),
            ('time_text_embed.text_embedder.linear_2', 'y_embedder.mlp.2')):
        weight, bias = _pair(state, source)
        _store(out, target, weight, bias)
    weight, bias = _pair(state, 'context_embedder')
    _store(out, 'context_embedder', weight, bias)

    layers = _block_indices(state, 'transformer_blocks.')
    if not layers:
        raise _QuantExportError('Comfy conversion expected SD3 transformer blocks.')
    last = layers[-1]
    for index in layers:
        prefix = f'transformer_blocks.{index}.'
        block = f'joint_blocks.{index}.'
        for sources, target in (
                (('attn.to_q', 'attn.to_k', 'attn.to_v'), 'x_block.attn.qkv'),
                (('attn.add_q_proj', 'attn.add_k_proj', 'attn.add_v_proj'), 'context_block.attn.qkv')):
            weights = []
            biases = []
            for source in sources:
                weight, bias = _pair(state, prefix + source)
                weights.append(weight)
                biases.append(bias)
            out[block + target + '.weight'] = _cat(weights)
            if all(item is not None for item in biases):
                out[block + target + '.bias'] = _cat(biases)
        if prefix + 'attn.norm_q.weight' in state:
            for source, target in (
                    ('attn.norm_q.weight', 'x_block.attn.ln_q.weight'),
                    ('attn.norm_k.weight', 'x_block.attn.ln_k.weight'),
                    ('attn.norm_added_q.weight', 'context_block.attn.ln_q.weight'),
                    ('attn.norm_added_k.weight', 'context_block.attn.ln_k.weight')):
                out[block + target] = _take(state, prefix + source)
        weight, bias = _pair(state, prefix + 'attn.to_out.0')
        _store(out, block + 'x_block.attn.proj', weight, bias)
        if index != last:
            weight, bias = _pair(state, prefix + 'attn.to_add_out')
            _store(out, block + 'context_block.attn.proj', weight, bias)
        if prefix + 'attn2.to_q.weight' in state:
            weights = []
            biases = []
            for source in ('attn2.to_q', 'attn2.to_k', 'attn2.to_v'):
                weight, bias = _pair(state, prefix + source)
                weights.append(weight)
                biases.append(bias)
            out[block + 'x_block.attn2.qkv.weight'] = _cat(weights)
            if all(item is not None for item in biases):
                out[block + 'x_block.attn2.qkv.bias'] = _cat(biases)
            if prefix + 'attn2.norm_q.weight' in state:
                out[block + 'x_block.attn2.ln_q.weight'] = _take(state, prefix + 'attn2.norm_q.weight')
                out[block + 'x_block.attn2.ln_k.weight'] = _take(state, prefix + 'attn2.norm_k.weight')
            weight, bias = _pair(state, prefix + 'attn2.to_out.0')
            _store(out, block + 'x_block.attn2.proj', weight, bias)
        weight, bias = _pair(state, prefix + 'norm1.linear')
        _store(out, block + 'x_block.adaLN_modulation.1', weight, bias)
        weight, bias = _pair(state, prefix + 'norm1_context.linear')
        if index == last:
            weight = _swap_halves(weight)
            if bias is not None:
                bias = _swap_halves(bias)
        _store(out, block + 'context_block.adaLN_modulation.1', weight, bias)
        for source, target in (
                ('ff.net.0.proj', 'x_block.mlp.fc1'),
                ('ff.net.2', 'x_block.mlp.fc2')):
            weight, bias = _pair(state, prefix + source)
            _store(out, block + target, weight, bias)
        if index != last:
            for source, target in (
                    ('ff_context.net.0.proj', 'context_block.mlp.fc1'),
                    ('ff_context.net.2', 'context_block.mlp.fc2')):
                weight, bias = _pair(state, prefix + source)
                _store(out, block + target, weight, bias)

    weight, bias = _pair(state, 'proj_out')
    _store(out, 'final_layer.linear', weight, bias)
    out['final_layer.adaLN_modulation.1.weight'] = _swap_halves(_take(state, 'norm_out.linear.weight'))
    bias = state.pop('norm_out.linear.bias', None)
    if bias is not None:
        out['final_layer.adaLN_modulation.1.bias'] = _swap_halves(bias)
    _leftover(state)
    return out


def _wan_key(key: str) -> str:
    # The loader copies this onto condition_embedder.image_embedder.pos_embed.
    # ComfyUI stores the same table as img_emb.emb_pos.
    if key == 'condition_embedder.image_embedder.pos_embed':
        return 'img_emb.emb_pos'
    if key == 'scale_shift_table' or key.startswith('scale_shift_table.'):
        key = 'head.modulation' + key[len('scale_shift_table'):]
    else:
        key = key.replace('.scale_shift_table', '.modulation')
    if key == 'proj_out' or key.startswith('proj_out.'):
        key = 'head.head' + key[len('proj_out'):]
    else:
        key = key.replace('.proj_out', '.after_proj')
    for source, target in reversed(_WAN_PAIRS):
        key = key.replace(target, source)
    return key


def _wan(state: dict) -> dict:
    if any('face_adapter' in key or 'motion_encoder' in key for key in state):
        raise _QuantExportError(
            'Wan face-adapter and motion-encoder checkpoints have no ComfyUI key map yet.')
    return {_wan_key(key): value for key, value in state.items()}


def _zimage(state: dict) -> dict:
    if 'norm_final.weight' in state:
        state.pop('norm_final.weight')
        _messages.warning(
            'Z-Image norm_final.weight has no Comfy name. It was left out of the Comfy file.')
    fused = {}
    consumed = set()
    for key in list(state):
        if not key.endswith('.attention.to_q.weight'):
            continue
        base = key[:-len('to_q.weight')]
        fused[base + 'qkv.weight'] = _cat([
            state[key],
            state[base + 'to_k.weight'],
            state[base + 'to_v.weight'],
        ])
        consumed.update({key, base + 'to_k.weight', base + 'to_v.weight'})
    out = {}
    for key, value in state.items():
        if key in consumed:
            continue
        renamed = key
        for source, target in _ZIMAGE_PAIRS:
            renamed = renamed.replace(source, target)
        out[renamed] = value
    out.update(fused)
    return out


def _qwen21(state: dict) -> dict:
    out = {}
    consumed = set()
    for key, value in state.items():
        if not key.endswith('img_mlp.gate_layer.weight'):
            continue
        proj = key.replace('gate_layer.weight', 'proj.weight')
        if proj not in state:
            raise _QuantExportError(f'Comfy conversion expected "{proj}" next to "{key}".')
        out[key.replace('gate_layer.weight', 'gate_up.weight')] = _cat([value, state[proj]])
        consumed.update({key, proj})
    for key, value in state.items():
        if key not in consumed:
            out[key] = value
    return out


def _flatten_language_model(state: dict) -> dict:
    """``model.language_model.layers`` is the same stack llama.cpp calls ``blk``."""
    out = {}
    for key, value in state.items():
        new = key
        if new.startswith('model.language_model.'):
            new = 'model.' + new[len('model.language_model.'):]
        elif new.startswith('language_model.'):
            new = 'model.' + new[len('language_model.'):]
        out[new] = value
    return out


def _invert_names(state: dict, pairs: tuple) -> dict:
    renamed = {}
    for key, value in state.items():
        new = key
        for source, target in reversed(pairs):
            new = new.replace(target, source)
        renamed[new] = value
    return renamed


_LLAMA_KINDS = frozenset({
    't5', 't5encoder', 'llama', 'qwen2', 'qwen2vl', 'qwen3', 'qwen3vl', 'qwen35', 'qwen35moe',
    'gemma3', 'gemma4',
})


def llama_state(state: dict, config: dict | None) -> tuple[dict, str]:
    """
    A copy of ``state`` under llama.cpp names, and the GGUF architecture string.

    This is the layout Unsloth text-encoder GGUFs use. dgenerate reads T5,
    UMT5, Llama, Qwen, and Gemma, including Gemma 4 and a Llama whose head
    counts are not the 32 and 8 that ComfyUI-GGUF hardcodes.

    :raises QuantExportError: This component is not one of those text encoders.
    """
    copied = {
        key: value.detach().cpu().contiguous()
        for key, value in state.items() if isinstance(value, torch.Tensor)}
    kind = _text_kind(config, copied)
    if kind not in _LLAMA_KINDS:
        raise _QuantExportError(
            '--llama writes the llama.cpp names dgenerate reads for T5, UMT5, '
            'Llama, Qwen, and Gemma. This component is not one of those.')
    return _llama_cpp_text(copied, config, kind), kind


def _drop_vision(state: dict, destination: str) -> dict:
    vision = [
        key for key in state
        if key.startswith(('visual.', 'vision_', 'embed_vision.'))
        or '.visual.' in key or '.vision_' in key or '.embed_vision.' in key]
    if not vision:
        return state
    _messages.warning(
        f'{len(vision)} vision tensors go in a separate mmproj file. '
        f'{destination} loads that beside the text encoder.')
    kept = {key: value for key, value in state.items() if key not in set(vision)}
    if not kept:
        raise _QuantExportError('The text encoder had no language tensors to write as a GGUF.')
    return kept


def _text(state: dict, config: dict | None, kind: str, gguf: bool) -> dict:
    if not gguf:
        return state
    state = _drop_vision(state, 'ComfyUI-GGUF')
    if kind == 'gemma4':
        return state
    if kind in {'llama', 'qwen2', 'qwen2vl', 'qwen3', 'qwen3vl', 'gemma3'}:
        if kind == 'llama':
            heads, kv = _attention_heads(config, 'A Llama Comfy GGUF')
            if (heads, kv) != _COMFY_LLAMA_HEADS:
                raise _QuantExportError(
                    'ComfyUI-GGUF un-permutes Llama GGUFs as 32 query heads and 8 key-value heads. '
                    f'This text encoder has {heads} and {kv}.')
            state = _ggufcompat.permute_llama_attention(state, heads, kv)
        if kind == 'gemma3':
            state = _gemma_norms(state)
        pairs = _ggufcompat._GEMMA3_GGUF_KEYS if kind == 'gemma3' else _ggufcompat._LLAMA_GGUF_KEYS
        return _invert_names(_flatten_language_model(state), pairs)
    return _invert_names(state, _ggufcompat._T5_GGUF_KEYS)


def _t5_llama_names(state: dict) -> dict:
    """Encoder and decoder names llama.cpp uses. Encoder names match the loader."""
    encoder = (
        ('layer.0.SelfAttention.relative_attention_bias', 'attn_rel_b'),
        ('layer.0.SelfAttention.q', 'attn_q'),
        ('layer.0.SelfAttention.k', 'attn_k'),
        ('layer.0.SelfAttention.v', 'attn_v'),
        ('layer.0.SelfAttention.o', 'attn_o'),
        ('layer.0.layer_norm', 'attn_norm'),
        ('layer.1.DenseReluDense.wi_1', 'ffn_up'),
        ('layer.1.DenseReluDense.wi_0', 'ffn_gate'),
        ('layer.1.DenseReluDense.wi', 'ffn_up'),
        ('layer.1.DenseReluDense.wo', 'ffn_down'),
        ('layer.1.layer_norm', 'ffn_norm'),
        ('final_layer_norm', 'output_norm'),
    )
    decoder = (
        ('layer.1.EncDecAttention.relative_attention_bias', 'cross_attn_rel_b'),
        ('layer.1.EncDecAttention.q', 'cross_attn_q'),
        ('layer.1.EncDecAttention.k', 'cross_attn_k'),
        ('layer.1.EncDecAttention.v', 'cross_attn_v'),
        ('layer.1.EncDecAttention.o', 'cross_attn_o'),
        ('layer.1.layer_norm', 'cross_attn_norm'),
        ('layer.2.DenseReluDense.wi_1', 'ffn_up'),
        ('layer.2.DenseReluDense.wi_0', 'ffn_gate'),
        ('layer.2.DenseReluDense.wi', 'ffn_up'),
        ('layer.2.DenseReluDense.wo', 'ffn_down'),
        ('layer.2.layer_norm', 'ffn_norm'),
    ) + encoder
    out = {}
    for key, value in state.items():
        name = key
        pairs = encoder
        if name.startswith('decoder.'):
            name = 'dec.' + name[len('decoder.'):]
            pairs = decoder
        elif name.startswith('encoder.'):
            name = 'enc.' + name[len('encoder.'):]
        name = name.replace('.block.', '.blk.')
        for source, target in pairs:
            name = name.replace(source, target)
        if name == 'shared.weight':
            name = 'token_embd.weight'
        elif name == 'lm_head.weight':
            name = 'output.weight'
        out[name] = value
    return out


def _drop_audio(state: dict) -> dict:
    audio = [
        key for key in state
        if '.audio_' in key or key.startswith('audio_') or '.embed_audio.' in key]
    if not audio:
        return state
    _messages.warning(
        f'{len(audio)} audio tensors are left out of the text GGUF. '
        'llama.cpp reads the language model from this file.')
    return {key: value for key, value in state.items() if key not in set(audio)}


def _gemma4_language_names(state: dict) -> dict:
    """Names that are not the Gemma 3 stack. ``embed_tokens`` must not swallow the per-layer table."""
    renamed = {}
    for key, value in state.items():
        name = key
        if 'embed_tokens_per_layer.' in name:
            name = 'per_layer_token_embd.' + name.split('embed_tokens_per_layer.', 1)[1]
        elif 'per_layer_model_projection.' in name:
            name = 'per_layer_model_proj.' + name.split('per_layer_model_projection.', 1)[1]
        elif 'per_layer_projection_norm.' in name:
            name = 'per_layer_proj_norm.' + name.split('per_layer_projection_norm.', 1)[1]
        elif name.endswith('layer_scalar'):
            name = name[:-len('layer_scalar')] + 'layer_output_scale.weight'
        else:
            name = name.replace('per_layer_input_gate', 'inp_gate')
            name = name.replace('per_layer_projection', 'proj')
            name = name.replace('post_per_layer_input_norm', 'post_norm')
        renamed[name] = value
    return renamed


def _gemma4_rope_freqs(config: dict | None):
    """Freq factors for full-attention proportional RoPE. One tensor, shared by those layers."""
    text = config.get('text_config') if isinstance(config, dict) and isinstance(config.get('text_config'), dict) else (config or {})
    rope = text.get('rope_parameters') if isinstance(text.get('rope_parameters'), dict) else {}
    full = rope.get('full_attention') if isinstance(rope.get('full_attention'), dict) else {}
    head = text.get('global_head_dim')
    factor = full.get('partial_rotary_factor', 0.25)
    if not isinstance(head, int) or not isinstance(factor, (int, float)):
        return None
    rotated = int(head * float(factor) / 2)
    total = head // 2
    values = [1.0] * rotated + [1e30] * (total - rotated)
    return torch.tensor(values, dtype=torch.float32)


def _llama_cpp_text(state: dict, config: dict | None, kind: str) -> dict:
    """llama.cpp names for a text encoder dgenerate loads, including Gemma 4."""
    state = _drop_vision(state, 'llama.cpp')
    state = _drop_audio(state)
    if kind == 'llama':
        heads, kv = _attention_heads(config, 'A llama.cpp Llama GGUF')
        state = _ggufcompat.permute_llama_attention(state, heads, kv)
    if kind in {'gemma3', 'gemma4'}:
        # Gemma 3 RMSNorm stores weights one below the runtime value. Gemma 4 does not.
        if kind == 'gemma3':
            state = _gemma_norms(state)
        state = _flatten_language_model(state)
        if kind == 'gemma4':
            state = _gemma4_language_names(state)
        state = _invert_names(state, _ggufcompat._GEMMA3_GGUF_KEYS)
        if kind == 'gemma4':
            freqs = _gemma4_rope_freqs(config)
            if freqs is not None:
                state = dict(state)
                state['rope_freqs.weight'] = freqs
        return state
    if kind in {'llama', 'qwen2', 'qwen2vl', 'qwen3', 'qwen3vl', 'qwen35', 'qwen35moe'}:
        state = _flatten_language_model(state)
        if kind in {'qwen35', 'qwen35moe'}:
            state = _qwen35_norms(state)
        state = _invert_names(state, _ggufcompat._LLAMA_GGUF_KEYS)
        if kind in {'qwen35', 'qwen35moe'}:
            text = config.get('text_config') if isinstance(config, dict) and isinstance(config.get('text_config'), dict) else config
            try:
                state = _ggufcompat.qwen35_to_llama(state, text)
            except ValueError as error:
                raise _QuantExportError(str(error)) from error
        return state
    return _t5_llama_names(state)


def _qwen35_norms(state: dict) -> dict:
    """llama.cpp multiplies RMSNorm weights directly. Qwen3.5 stores ``weight`` and adds one."""
    shifted = {}
    for key, value in state.items():
        if key.endswith('norm.weight') and 'linear_attn.norm.' not in key:
            value = value.float() + 1
        shifted[key] = value
    return shifted


def _attention_heads(config: dict | None, label: str) -> tuple[int, int]:
    if not config:
        raise _QuantExportError(f'{label} needs num_attention_heads in config.json.')
    source = config.get('text_config') if isinstance(config.get('text_config'), dict) else config
    heads = source.get('num_attention_heads')
    kv = source.get('num_key_value_heads', heads)
    if not isinstance(heads, int) or not isinstance(kv, int):
        raise _QuantExportError(f'{label} needs num_attention_heads in config.json.')
    return heads, kv


def _gemma_norms(state: dict) -> dict:
    updated = {}
    for key, value in state.items():
        if any(marker in key for marker in _ggufcompat._GEMMA3_NORM_MARKERS):
            value = value.float() + 1
        updated[key] = value
    return updated


def _layers_per_block(state: dict, config: dict | None) -> int:
    if config and isinstance(config.get('layers_per_block'), int):
        return config['layers_per_block']
    indices = []
    for key in state:
        if key.startswith('down_blocks.0.resnets.'):
            head = key[len('down_blocks.0.resnets.'):].split('.', 1)[0]
            if head.isdigit():
                indices.append(int(head))
    if not indices:
        raise _QuantExportError('Comfy conversion expected UNet resnets in down_blocks.0.')
    return max(indices) + 1


def _resnet_to_ldm(key: str) -> str:
    for source, target in (
            ('conv_shortcut', 'skip_connection'),
            ('time_emb_proj', 'emb_layers.1'),
            ('norm1', 'in_layers.0'),
            ('conv1', 'in_layers.2'),
            ('norm2', 'out_layers.0'),
            ('conv2', 'out_layers.3')):
        key = key.replace(source, target)
    return key


def _consume_prefix(state: dict, source: str, target: str, resnet: bool) -> dict:
    moved = {}
    prefix = source + '.'
    for key in list(state):
        if not key.startswith(prefix):
            continue
        renamed = target + key[len(source):]
        if resnet:
            renamed = _resnet_to_ldm(renamed)
        moved[renamed] = state.pop(key)
    return moved


def _unet(state: dict, config: dict | None) -> tuple[dict, str]:
    from diffusers.loaders.single_file_utils import DIFFUSERS_TO_LDM_MAPPING

    sdxl = any(key.startswith('add_embedding.') for key in state) or (
        bool(config) and config.get('addition_embed_type') == 'text_time')
    out = {}
    groups = DIFFUSERS_TO_LDM_MAPPING['unet']
    for group in ('layers', 'class_embed_type', 'addition_embed_type'):
        for diffusers_key, ldm_key in groups[group].items():
            if diffusers_key in state:
                out[ldm_key] = state.pop(diffusers_key)

    layers = _layers_per_block(state, config)
    stride = layers + 1
    for block in _block_indices(state, 'down_blocks.'):
        for layer in _block_indices(state, f'down_blocks.{block}.resnets.'):
            index = block * stride + layer + 1
            out.update(_consume_prefix(
                state,
                f'down_blocks.{block}.resnets.{layer}',
                f'input_blocks.{index}.0',
                True))
            out.update(_consume_prefix(
                state,
                f'down_blocks.{block}.attentions.{layer}',
                f'input_blocks.{index}.1',
                False))
        sampler = block * stride + layers + 1
        out.update(_consume_prefix(
            state,
            f'down_blocks.{block}.downsamplers.0.conv',
            f'input_blocks.{sampler}.0.op',
            False))

    for layer in _block_indices(state, 'mid_block.resnets.'):
        out.update(_consume_prefix(
            state, f'mid_block.resnets.{layer}', f'middle_block.{2 * layer}', True))
    for layer in _block_indices(state, 'mid_block.attentions.'):
        out.update(_consume_prefix(
            state, f'mid_block.attentions.{layer}', f'middle_block.{2 * layer + 1}', False))

    for block in _block_indices(state, 'up_blocks.'):
        for layer in _block_indices(state, f'up_blocks.{block}.resnets.'):
            index = block * stride + layer
            out.update(_consume_prefix(
                state, f'up_blocks.{block}.resnets.{layer}', f'output_blocks.{index}.0', True))
            out.update(_consume_prefix(
                state,
                f'up_blocks.{block}.attentions.{layer}',
                f'output_blocks.{index}.1',
                False))
        index = block * stride + layers
        attended = any(key.startswith(f'output_blocks.{index}.1.') for key in out)
        slot = '2' if attended else '1'
        out.update(_consume_prefix(
            state,
            f'up_blocks.{block}.upsamplers.0.conv',
            f'output_blocks.{index}.{slot}.conv',
            False))

    _leftover(state)
    return out, 'sdxl' if sdxl else 'sd1'


def _move_module(state: dict, out: dict, source: str, target: str, scale: bool):
    weight = source + '.weight'
    bias = source + '.bias'
    if weight not in state and bias not in state:
        return
    if weight in state:
        out[target + ('.scale' if scale else '.weight')] = state.pop(weight)
    if bias in state:
        out[target + '.bias'] = state.pop(bias)


def _fuse3(state: dict, out: dict, query: str, key: str, value: str, target: str):
    if query + '.weight' not in state:
        return
    weights = [state.pop(query + '.weight'), state.pop(key + '.weight'), state.pop(value + '.weight')]
    out[target + '.weight'] = _cat(weights)
    biases = [state.pop(query + '.bias', None), state.pop(key + '.bias', None), state.pop(value + '.bias', None)]
    if all(item is not None for item in biases):
        out[target + '.bias'] = _cat(biases)


def _flux2(state: dict) -> dict:
    out = {}
    for index in _block_indices(state, 'single_transformer_blocks.'):
        prefix = f'single_transformer_blocks.{index}.attn.'
        block = f'single_blocks.{index}.'
        for source, target, scale in (
                ('to_qkv_mlp_proj', 'linear1', False),
                ('to_out', 'linear2', False),
                ('norm_q', 'norm.query_norm', True),
                ('norm_k', 'norm.key_norm', True)):
            _move_module(state, out, prefix + source, block + target, scale)
    for index in _block_indices(state, 'transformer_blocks.'):
        prefix = f'transformer_blocks.{index}.'
        block = f'double_blocks.{index}.'
        _fuse3(state, out, prefix + 'attn.to_q', prefix + 'attn.to_k', prefix + 'attn.to_v', block + 'img_attn.qkv')
        _fuse3(
            state, out,
            prefix + 'attn.add_q_proj', prefix + 'attn.add_k_proj', prefix + 'attn.add_v_proj',
            block + 'txt_attn.qkv')
        for source, target, scale in (
                ('attn.norm_q', 'img_attn.norm.query_norm', True),
                ('attn.norm_k', 'img_attn.norm.key_norm', True),
                ('attn.to_out.0', 'img_attn.proj', False),
                ('ff.linear_in', 'img_mlp.0', False),
                ('ff.linear_out', 'img_mlp.2', False),
                ('attn.norm_added_q', 'txt_attn.norm.query_norm', True),
                ('attn.norm_added_k', 'txt_attn.norm.key_norm', True),
                ('attn.to_add_out', 'txt_attn.proj', False),
                ('ff_context.linear_in', 'txt_mlp.0', False),
                ('ff_context.linear_out', 'txt_mlp.2', False)):
            _move_module(state, out, prefix + source, block + target, scale)
    if 'norm_out.linear.weight' in state:
        out['final_layer.adaLN_modulation.1.weight'] = _swap_halves(state.pop('norm_out.linear.weight'))
    if 'norm_out.linear.bias' in state:
        out['final_layer.adaLN_modulation.1.bias'] = _swap_halves(state.pop('norm_out.linear.bias'))
    renames = (
        ('time_guidance_embed.timestep_embedder.linear_1', 'time_in.in_layer'),
        ('time_guidance_embed.timestep_embedder.linear_2', 'time_in.out_layer'),
        ('time_guidance_embed.guidance_embedder.linear_1', 'guidance_in.in_layer'),
        ('time_guidance_embed.guidance_embedder.linear_2', 'guidance_in.out_layer'),
        ('double_stream_modulation_img.linear', 'double_stream_modulation_img.lin'),
        ('double_stream_modulation_txt.linear', 'double_stream_modulation_txt.lin'),
        ('single_stream_modulation.linear', 'single_stream_modulation.lin'),
        ('context_embedder', 'txt_in'),
        ('x_embedder', 'img_in'),
        ('proj_out', 'final_layer.linear'),
    )
    for key in list(state):
        renamed = key
        for source, target in renames:
            if renamed.startswith(source):
                renamed = target + renamed[len(source):]
                break
        out[renamed] = state.pop(key)
    _leftover(state)
    return out


_MMPROJ_PAIRS = (
    ('visual.merger.mlp.', 'mm.'),
    ('visual.merger.ln_q.', 'v.post_ln.'),
    ('visual.patch_embed.proj', 'v.patch_embd'),
    ('visual.blocks.', 'v.blk.'),
    ('mlp.up_proj', 'ffn_up'),
    ('mlp.down_proj', 'ffn_down'),
    ('mlp.gate_proj', 'ffn_gate'),
    ('attn.proj.', 'attn_out.'),
    ('norm1.', 'ln1.'),
    ('norm2.', 'ln2.'),
)


def _gemma4_embed_key(key: str) -> bool:
    return (
        key.startswith(('embed_vision.', 'embed_audio.'))
        or '.embed_vision.' in key or '.embed_audio.' in key)


def _gemma4_patch_perm(length: int) -> torch.Tensor | None:
    """Column order llama.cpp uses so a Gemma 4 patch lines up with CHW pixels."""
    if length % 3:
        return None
    side = length // 3
    patch = int(side ** 0.5)
    if patch * patch != side:
        return None
    index = torch.arange(length)
    channel = index // side
    row = (index % side) // patch
    col = index % patch
    return row * patch * 3 + col * 3 + channel


def _gemma4_mmproj(state: dict) -> dict:
    """Gemma 4 unified vision and audio tensors in the llama.cpp mmproj names."""
    out = {}
    for key, value in state.items():
        if '.embed_audio.' in key or key.startswith('embed_audio.'):
            marker = 'embed_audio.'
            tail = key.split(marker, 1)[1]
            if tail == 'embedding_projection.weight':
                out['mm.a.input_projection.weight'] = value
            continue
        marker = 'embed_vision.'
        if marker not in key and not key.startswith('embed_vision.'):
            continue
        tail = key.split('embed_vision.', 1)[1]
        if tail == 'patch_dense.weight':
            perm = _gemma4_patch_perm(value.shape[-1])
            out['v.patch_embd.weight'] = value.index_select(-1, perm) if perm is not None else value
        elif tail == 'patch_dense.bias':
            out['v.patch_embd.bias'] = value
        elif tail in ('patch_ln1.weight', 'patch_ln1.bias'):
            perm = _gemma4_patch_perm(value.shape[0])
            suffix = 'weight' if tail.endswith('weight') else 'bias'
            out['v.patch_norm.1.' + suffix] = value.index_select(0, perm) if perm is not None else value
        elif tail == 'patch_ln2.weight':
            out['v.patch_norm.2.weight'] = value
        elif tail == 'patch_ln2.bias':
            out['v.patch_norm.2.bias'] = value
        elif tail == 'pos_norm.weight':
            out['v.patch_norm.3.weight'] = value
        elif tail == 'pos_norm.bias':
            out['v.patch_norm.3.bias'] = value
        elif tail == 'pos_embedding':
            out['v.position_embd.weight'] = value.permute(1, 0, 2).contiguous()
        elif tail == 'multimodal_embedder.embedding_projection.weight':
            out['mm.input_projection.weight'] = value
        else:
            raise _QuantExportError(f'Unexpected Gemma 4 vision tensor "{key}".')
    return out


def _gemma4_tower_key(key: str) -> bool:
    return (
        'vision_tower.' in key or key.startswith('vision_tower.')
        or 'audio_tower.' in key or key.startswith('audio_tower.')
        or _gemma4_embed_key(key))


def _gemma4v_block(table: dict, index: str, prefix: str, tail: str, value, out: dict) -> bool:
    """One Hugging Face block tensor as the llama.cpp ``gemma4v`` / ``gemma4a`` name."""
    from dgenerate.pipelinewrapper.ggufcompat import _GEMMA4_CLAMP

    for stem, (path, clippable) in table.items():
        if clippable:
            if tail == path + '.linear.weight':
                out[f'{prefix}{index}.{stem}.weight'] = value
                return True
            if tail == path + '.linear.bias':
                out[f'{prefix}{index}.{stem}.bias'] = value
                return True
            for clamp in _GEMMA4_CLAMP:
                if tail == path + '.' + clamp:
                    out[f'{prefix}{index}.{stem}.{clamp}'] = value.reshape(1)
                    return True
            continue
        if stem == 'out_scale' and tail in (path, path + '.weight'):
            out[f'{prefix}{index}.out_scale'] = value.reshape(1) if value.ndim == 0 else value
            return True
        for suffix in ('weight', 'bias'):
            if tail == path + '.' + suffix:
                out[f'{prefix}{index}.{stem}.{suffix}'] = value
                return True
    return False


def _gemma4v_patch(value):
    """``(out, 3*ph*pw)`` as the ``(out, 3, ph, pw)`` tensor llama.cpp stores."""
    if value.ndim != 2 or value.shape[-1] % 3:
        return None
    side = value.shape[-1] // 3
    patch = int(side ** 0.5)
    if patch * patch != side:
        return None
    return value.reshape(value.shape[0], patch, patch, 3).permute(0, 3, 1, 2).contiguous()


def _gemma4v_mmproj(state: dict) -> dict:
    """Full Gemma 4 vision and audio towers as an Unsloth-style mmproj."""
    import torch.nn.functional as functional

    from dgenerate.pipelinewrapper.ggufcompat import _GEMMA4V_AUDIO, _GEMMA4V_VISION

    out = {}
    unknown = []
    for key, value in state.items():
        if not _gemma4_tower_key(key):
            continue
        if 'rotary' in key or 'inv_freq' in key:
            continue
        tail = key.split('embed_vision.', 1)[1] if 'embed_vision.' in key else ''
        if tail == 'embedding_projection.weight':
            out['mm.input_projection.weight'] = value
            continue
        tail = key.split('embed_audio.', 1)[1] if 'embed_audio.' in key else ''
        if tail == 'embedding_projection.weight':
            out['mm.a.input_projection.weight'] = value
            continue
        if key.endswith('vision_tower.patch_embedder.input_proj.weight'):
            patch = _gemma4v_patch(value)
            if patch is None:
                raise _QuantExportError(
                    f'Gemma 4 patch embedding "{key}" is not (out, 3*patch*patch).')
            out['v.patch_embd.weight'] = patch
            continue
        if key.endswith('vision_tower.patch_embedder.position_embedding_table'):
            out['v.position_embd.weight'] = value
            continue
        if key.endswith('vision_tower.std_bias'):
            out['v.std_bias'] = value.reshape(-1)
            continue
        if key.endswith('vision_tower.std_scale'):
            out['v.std_scale'] = value.reshape(-1)
            continue
        if key.endswith('audio_tower.subsample_conv_projection.input_proj_linear.weight'):
            out['a.input_projection.weight'] = value
            continue
        if '.audio_tower.output_proj.' in key or key.endswith('audio_tower.output_proj.weight') \
                or key.endswith('audio_tower.output_proj.bias'):
            out['a.pre_encode.out.' + key.rsplit('.', 1)[1]] = value
            continue
        conv = key.split('subsample_conv_projection.layer', 1)
        if len(conv) == 2 and conv[1][:1].isdigit():
            layer, rest = conv[1].split('.', 1)
            if rest == 'conv.weight':
                out[f'a.conv1d.{layer}.weight'] = value
                continue
            if rest == 'norm.weight':
                out[f'a.conv1d.{layer}.norm.weight'] = value
                continue
        vision = key.split('vision_tower.encoder.layers.', 1)
        if len(vision) == 2:
            index, block_tail = vision[1].split('.', 1)
            if _gemma4v_block(_GEMMA4V_VISION, index, 'v.blk.', block_tail, value, out):
                continue
        audio = key.split('audio_tower.layers.', 1)
        if len(audio) == 2:
            index, block_tail = audio[1].split('.', 1)
            if block_tail == 'lconv1d.depthwise_conv1d.weight':
                squeezed = value.squeeze(1) if value.ndim == 3 and value.shape[1] == 1 else value
                out[f'a.blk.{index}.conv_dw.weight'] = squeezed
                continue
            if block_tail == 'self_attn.per_dim_scale':
                out[f'a.blk.{index}.per_dim_scale.weight'] = functional.softplus(value)
                continue
            if _gemma4v_block(_GEMMA4V_AUDIO, index, 'a.blk.', block_tail, value, out):
                continue
        unknown.append(key)
    if unknown:
        sample = unknown[0]
        raise _QuantExportError(
            f'Unexpected Gemma 4 tower tensor "{sample}". '
            'The mmproj writer knows the full vision and audio towers.')
    if not out:
        return None
    return out


def _is_vision_key(key: str) -> bool:
    return (
        key.startswith(('visual.', 'vision_', 'model.visual.', 'model.vision_'))
        or '.visual.' in key or '.vision_' in key)


def _mmproj_name(key: str) -> str:
    for source, target in _MMPROJ_PAIRS:
        key = key.replace(source, target)
    return key


def mmproj_state(state: dict, config: dict | None = None) -> dict | None:
    """
    Vision tensors in llama.cpp mmproj names.

    Qwen towers use the names ComfyUI-GGUF remaps. A full Gemma 4 tower
    (the LTX-2.5 prompt enhancer) uses the ``gemma4v`` / ``gemma4a`` names.
    Gemma 4 unified vision and audio use the ``gemma4uv`` / ``gemma4ua`` names.
    ``None`` when the component has no vision tower. A 5D Qwen patch embedding
    is split into the two 4D tensors that loader stacks back together.

    :raises QuantExportError: The vision MLP is not the Qwen2.5-VL gate/up/down layout.
    """
    if any(_gemma4_tower_key(key) and ('vision_tower.' in key or 'audio_tower.' in key)
           for key in state):
        tower = {
            key: value.detach().cpu().contiguous()
            for key, value in state.items()
            if isinstance(value, torch.Tensor) and _gemma4_tower_key(key)}
        written = _gemma4v_mmproj(tower)
        if written:
            return written
    gemma4 = {
        key: value.detach().cpu().contiguous()
        for key, value in state.items()
        if isinstance(value, torch.Tensor) and _gemma4_embed_key(key)}
    if gemma4:
        return _gemma4_mmproj(gemma4)
    vision = {
        key: value.detach().cpu().contiguous()
        for key, value in state.items()
        if isinstance(value, torch.Tensor) and _is_vision_key(key)}
    if not vision:
        return None
    if any(key.endswith('mlp.linear_fc1.weight') for key in vision):
        return _qwen3vl_mmproj(vision, config)
    if any(key.endswith('mlp.fc1.weight') for key in vision):
        raise _QuantExportError(
            'This vision tower uses fc1/fc2. ComfyUI-GGUF rebuilds an mmproj from the '
            'Qwen2.5-VL gate, up, and down projections.')
    normalized = {}
    for key, value in vision.items():
        renamed = key
        for prefix in ('model.visual.', 'model.vision_tower.'):
            if renamed.startswith(prefix):
                renamed = 'visual.' + renamed[len(prefix):]
                break
        normalized[renamed] = value

    out = {}
    for key, value in normalized.items():
        if key.endswith('.attn.qkv.weight') or key.endswith('.attn.qkv.bias'):
            suffix = 'weight' if key.endswith('.weight') else 'bias'
            stem = key[: -len('.qkv.' + suffix)]
            for name, part in zip(('q', 'k', 'v'), value.chunk(3, dim=0)):
                out[_mmproj_name(stem[: -len('attn')] + f'attn_{name}.{suffix}')] = part.contiguous()
            continue
        if key.endswith('patch_embed.proj.weight') and value.ndim == 5:
            if value.shape[2] != 2:
                raise _QuantExportError(
                    'ComfyUI-GGUF stacks a patch embedding of temporal size 2. '
                    f'This one has temporal size {value.shape[2]}.')
            base = _mmproj_name(key[:-len('.weight')])
            out[base + '.weight'] = value[:, :, 0].contiguous()
            out[base + '.weight.1'] = value[:, :, 1].contiguous()
            continue
        renamed = key
        for source, target in (
                ('.attn.q_proj.', '.attn_q.'),
                ('.attn.k_proj.', '.attn_k.'),
                ('.attn.v_proj.', '.attn_v.')):
            renamed = renamed.replace(source, target)
        if renamed.endswith('patch_embed.proj.bias') or renamed.endswith('patch_embd.bias'):
            # llama.cpp's Qwen2-VL graph rejects a patch bias.
            continue
        out[_mmproj_name(renamed)] = value
    return out


def _qwen3vl_mmproj(vision: dict, config: dict | None) -> dict:
    """Qwen3-VL tower names llama.cpp loads: fused ``attn_qkv``, ``ffn_up``/``ffn_down``, DeepStack."""
    normalized = {}
    for key, value in vision.items():
        renamed = key
        if renamed.startswith('model.visual.'):
            renamed = 'visual.' + renamed[len('model.visual.'):]
        normalized[renamed] = value

    indexes = ((config or {}).get('vision_config') or {}).get('deepstack_visual_indexes') or []
    out = {}
    for key, value in normalized.items():
        if key.startswith('visual.deepstack_merger_list.'):
            rest = key[len('visual.deepstack_merger_list.'):]
            index, tail = rest.split('.', 1)
            absolute = indexes[int(index)] if int(index) < len(indexes) else int(index)
            tail = tail.replace('linear_fc1.', 'fc1.').replace('linear_fc2.', 'fc2.')
            out[f'v.deepstack.{absolute}.{tail}'] = value
            continue
        if key.startswith('visual.merger.'):
            suffix = key[len('visual.merger.'):]
            if suffix.startswith('linear_fc1.'):
                out['mm.0.' + suffix[len('linear_fc1.'):]] = value
            elif suffix.startswith('linear_fc2.'):
                out['mm.2.' + suffix[len('linear_fc2.'):]] = value
            elif suffix.startswith('norm.'):
                out['v.post_ln.' + suffix[len('norm.'):]] = value
            else:
                raise _QuantExportError(f'Unexpected Qwen3-VL merger tensor "{key}".')
            continue
        if key == 'visual.pos_embed.weight':
            out['v.position_embd.weight'] = value
            continue
        if key.endswith('patch_embed.proj.weight') and value.ndim == 5:
            if value.shape[2] != 2:
                raise _QuantExportError(
                    'llama.cpp stacks a patch embedding of temporal size 2. '
                    f'This one has temporal size {value.shape[2]}.')
            out['v.patch_embd.weight'] = value[:, :, 0].contiguous()
            out['v.patch_embd.weight.1'] = value[:, :, 1].contiguous()
            continue
        if key.endswith('patch_embed.proj.bias'):
            out['v.patch_embd.bias'] = value
            continue
        renamed = key.replace('.mlp.linear_fc1.', '.ffn_up.').replace('.mlp.linear_fc2.', '.ffn_down.')
        renamed = renamed.replace('.attn.qkv.', '.attn_qkv.')
        out[_mmproj_name(renamed)] = value
    return out

