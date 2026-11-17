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
Write GGUF and ComfyUI safetensors quants from a diffusers component state dict.

GGUF types are the ones :py:mod:`dgenerate.pipelinewrapper.ggufcompat` can load.
Comfy formats are the ones :py:mod:`dgenerate.pipelinewrapper.safetensorscompat`
can load. Tensor names are left as the component stored them.
"""

import functools
import json
import os
import shutil

import torch

import dgenerate.messages as _messages
import dgenerate.quantize.ggufquant as _ggufquant
import dgenerate.pipelinewrapper.safetensorscompat as _safetensorscompat

# Short names people actually type. q4 is Q4_K; Q4_0 is requested by that name.
# Text-encoder architecture strings stay in sync with ggufcompat._TEXT_GGUF_ARCHITECTURES.
_ALIASES = (
    ('q2', 'Q2_K'),
    ('q3', 'Q3_K'),
    ('q4', 'Q4_K'),
    ('q5', 'Q5_K'),
    ('q6', 'Q6_K'),
    ('q8', 'Q8_0'),
    ('iq4', 'IQ4_NL'),
    ('bf16', 'BF16'),
    ('f16', 'F16'),
    ('fp16', 'F16'),
    ('f32', 'F32'),
    ('fp32', 'F32'),
    ('fp8', 'float8_e4m3fn'),
    ('fp8-e4m3', 'float8_e4m3fn'),
    ('fp8-e5m2', 'float8_e5m2'),
    ('int8', 'int8_tensorwise'),
    ('int4', 'convrot_w4a4'),
    ('w4a4', 'convrot_w4a4'),
    ('w4a8', 'asym_w4a8_int8'),
    ('w6a8', 'w6a8_int8'),
)

_GGUF_ORDER = (
    'Q2_K', 'Q3_K', 'Q4_0', 'Q4_1', 'Q4_K', 'Q5_0', 'Q5_1', 'Q5_K', 'Q6_K', 'Q8_0',
    'IQ4_NL', 'IQ4_XS', 'BF16', 'F16', 'F32',
)

_WEIGHT_STEMS = ('diffusion_pytorch_model', 'model', 'pytorch_model')

# 2D tables that are embeddings, not linears. A Comfy quant marker on one of
# these is dequantized at load, so leaving them dense keeps the table exact.
_EMBEDDING_NAMES = {
    'embed_tokens', 'token_embedding', 'token_embd', 'wte', 'wpe',
    'shared', 'pos_embed', 'position_embedding', 'position_embeddings',
}

# Safetensors with every tensor stored in one dtype and no quant metadata.
_PLAIN_DTYPES = {
    'float16': torch.float16,
    'float32': torch.float32,
}


class QuantExportError(Exception):
    """A component could not be read or a quant file could not be written."""


def comfy_format_names() -> tuple[str, ...]:
    """ComfyUI safetensors quant formats dgenerate can load."""
    return _safetensorscompat.SUPPORTED_FORMATS


@functools.cache
def gguf_format_names() -> tuple[str, ...]:
    """
    GGUF types dgenerate can load and this exporter can write.

    That is the Diffusers GGUF set, plus F16 and F32, limited to types the
    installed ``gguf`` package knows how to quantize.
    """
    import gguf
    from diffusers.quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES

    writable = set(gguf.quants._type_traits)
    writable.update({
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.BF16,
    })
    loadable = set(SUPPORTED_GGUF_QUANT_TYPES)
    loadable.update({
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.BF16,
    })
    by_name = {}
    for qtype in writable & loadable:
        by_name[qtype.name] = qtype
    return tuple(name for name in _GGUF_ORDER if name in by_name)


def preset_formats() -> dict[str, tuple[str, ...]]:
    """Named format lists. ``dgenerate`` and ``comfy`` are the short ones."""
    quants = tuple(name for name in gguf_format_names() if name not in {'F32', 'F16', 'BF16'})
    return {
        'dgenerate': ('Q4_K', 'Q8_0'),
        'comfy': ('float8_e4m3fn', 'int8_tensorwise'),
        'gguf': quants,
        'comfy-all': comfy_format_names(),
    }


def format_catalog() -> str:
    """Text for ``--list-formats``."""
    lines = [
        'GGUF types dgenerate loads:',
        '  ' + ', '.join(gguf_format_names()),
        '',
        'ComfyUI safetensors quants dgenerate loads:',
        '  ' + ', '.join(comfy_format_names()),
        '',
        'Short names:',
    ]
    lines.extend(f'  {alias} -> {target}' for alias, target in _ALIASES)
    lines.append('')
    lines.append('Presets:')
    for name, formats in preset_formats().items():
        lines.append(f'  {name}: {", ".join(formats)}')
    lines.append('')
    lines.append('Plain safetensors, no quant:')
    lines.append('  ' + ', '.join(_PLAIN_DTYPES))
    lines.append('')
    lines.append('Files:')
    lines.append('  transformer-Q4_K.gguf')
    lines.append('  transformer-F16.gguf')
    lines.append('  transformer-float16.safetensors')
    lines.append('  text_encoder-F16-llama.gguf')
    lines.append('  mmproj-text_encoder-F16-llama.gguf')
    lines.append('  transformer-F16-comfy.gguf')
    lines.append('  transformer-float16-comfy.safetensors')
    lines.append('')
    lines.append(
        'F16, F32, BF16, float16, and float32 store every tensor in that dtype. '
        'The gguf and comfy presets leave those out. '
        '--llama adds -llama to the file name. --comfy adds -comfy.')
    return '\n'.join(lines)


def resolve_formats(names: list[str]) -> list[str]:
    """
    Expand presets and short names to canonical format names, in order.

    A format named twice is written once, at its first position.

    :raises QuantExportError: A name is not a format, short name, or preset.
    """
    gguf_names = {name.lower(): name for name in gguf_format_names()}
    comfy_names = {name.lower(): name for name in comfy_format_names()}
    plain_names = {name.lower(): name for name in _PLAIN_DTYPES}
    aliases = {alias.lower(): target for alias, target in _ALIASES}
    presets = {name.lower(): formats for name, formats in preset_formats().items()}
    resolved = []
    seen = set()
    unknown = []
    for raw in names:
        for piece in raw.split(','):
            piece = piece.strip()
            if not piece:
                continue
            key = piece.lower()
            if key in presets:
                chosen = presets[key]
            elif key in aliases:
                chosen = (aliases[key],)
            elif key in gguf_names:
                chosen = (gguf_names[key],)
            elif key in comfy_names:
                chosen = (comfy_names[key],)
            elif key in plain_names:
                chosen = (plain_names[key],)
            else:
                unknown.append(piece)
                continue
            for item in chosen:
                if item not in seen:
                    seen.add(item)
                    resolved.append(item)
    if unknown:
        raise QuantExportError(
            f'Unknown format: {", ".join(unknown)}. '
            f'Run with --list-formats to see GGUF types, Comfy safetensors quants, '
            f'plain float16 and float32, short names, and presets.')
    if not resolved:
        raise QuantExportError('No formats were given.')
    return resolved


def _weight_candidates(variant: str | None) -> list[str]:
    """Index files first, so a sharded checkpoint wins over a stray single file."""
    names = []
    for stem in _WEIGHT_STEMS:
        if variant:
            names.append(f'{stem}.safetensors.index.{variant}.json')
            names.append(f'{stem}.{variant}.safetensors.index.json')
            names.append(f'{stem}.bin.index.{variant}.json')
            names.append(f'{stem}.{variant}.safetensors')
            names.append(f'{stem}.{variant}.bin')
        else:
            names.append(f'{stem}.safetensors.index.json')
            names.append(f'{stem}.bin.index.json')
            names.append(f'{stem}.safetensors')
            names.append(f'{stem}.bin')
    return names


def find_weight_file(directory: str,
                     variant: str | None = None,
                     weight_name: str | None = None) -> str | None:
    """The weight file or index in a component folder, or ``None``.

    ``weight_name`` is a file in that folder, such as ``model.fp16.safetensors``.
    """
    if weight_name:
        path = os.path.join(directory, weight_name)
        return path if os.path.isfile(path) else None
    for name in _weight_candidates(variant):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    return None


def component_directories(root: str,
                          subfolders: list[str] | None = None,
                          variant: str | None = None,
                          weight_name: str | None = None) -> list[tuple[str, str]]:
    """
    ``(name, directory)`` for each component to quantize.

    ``subfolders`` selects those folders. Without it, components named in
    ``model_index.json`` that have weights are used. A folder of one component,
    with the weights at its root, is that one component.

    :raises QuantExportError: A requested subfolder is missing or has no weights,
        or the folder has no components.
    """
    if subfolders:
        found = []
        for name in subfolders:
            folder = os.path.join(root, name)
            if not os.path.isdir(folder):
                raise QuantExportError(f'Subfolder "{name}" was not found in "{root}".')
            if find_weight_file(folder, variant, weight_name) is None:
                if weight_name:
                    raise QuantExportError(
                        f'Subfolder "{name}" has no weight file "{weight_name}".')
                raise QuantExportError(
                    f'Subfolder "{name}" has no weights'
                    + (f' for variant "{variant}".' if variant else '.'))
            found.append((name.replace('\\', '/'), folder))
        return found

    index_path = os.path.join(root, 'model_index.json')
    if not os.path.isfile(index_path):
        index_path = os.path.join(root, 'modular_model_index.json')
    names = None
    if os.path.isfile(index_path):
        with open(index_path, encoding='utf-8') as handle:
            index = json.load(handle)
        names = [
            key for key, value in index.items()
            if not key.startswith('_') and isinstance(value, list)]
    if names:
        found = []
        for name in names:
            folder = os.path.join(root, name)
            if os.path.isdir(folder) and find_weight_file(folder, variant, weight_name) is not None:
                found.append((name, folder))
        if found:
            return found

    if find_weight_file(root, variant, weight_name) is not None:
        stem = os.path.basename(os.path.abspath(root)) or 'model'
        return [(stem, root)]

    found = []
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if os.path.isdir(folder) and find_weight_file(folder, variant, weight_name) is not None:
            found.append((name, folder))
    if not found:
        if weight_name:
            raise QuantExportError(f'"{root}" has no weight file "{weight_name}".')
        variant_note = f' for variant "{variant}"' if variant else ''
        raise QuantExportError(f'"{root}" has no diffusers component weights{variant_note}.')
    return found


def _progress(items, desc: str, unit: str):
    """A bar with a count and a percent. One item is returned as it is."""
    from tqdm.auto import tqdm

    items = list(items)
    if len(items) <= 1:
        yield from items
        return
    with tqdm(total=len(items), desc=desc, unit=unit, dynamic_ncols=True) as bar:
        for item in items:
            yield item
            bar.update()


def _load_shard(path: str) -> dict:
    if path.endswith('.safetensors'):
        import safetensors.torch
        return safetensors.torch.load_file(path)
    try:
        loaded = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location='cpu')
    if not isinstance(loaded, dict):
        raise QuantExportError(f'"{path}" is not a state dict.')
    if 'state_dict' in loaded and not any(str(key).endswith('.weight') for key in loaded):
        loaded = loaded['state_dict']
    tensors = {}
    for key, value in loaded.items():
        if isinstance(value, torch.Tensor):
            tensors[str(key)] = value
    if not tensors:
        raise QuantExportError(f'"{path}" has no tensors.')
    return tensors


def load_component_state(directory: str,
                         variant: str | None = None,
                         weight_name: str | None = None) -> dict[str, torch.Tensor]:
    """The component state dict. Sharded checkpoints are joined."""
    path = find_weight_file(directory, variant, weight_name)
    if path is None:
        if weight_name:
            raise QuantExportError(f'"{directory}" has no weight file "{weight_name}".')
        raise QuantExportError(
            f'"{directory}" has no weights' + (f' for variant "{variant}".' if variant else '.'))
    if not path.endswith('.index.json'):
        return _load_shard(path)
    with open(path, encoding='utf-8') as handle:
        weight_map = json.load(handle)['weight_map']
    folder = os.path.dirname(path)
    label = os.path.basename(os.path.abspath(directory)) or 'weights'
    unique = list(dict.fromkeys(weight_map.values()))
    shards = {
        shard: _load_shard(os.path.join(folder, shard))
        for shard in _progress(unique, f'Loading {label}', 'shard')}
    return {key: shards[shard][key] for key, shard in weight_map.items()}


def load_component_config(directory: str) -> dict | None:
    """``config.json`` next to the weights, or ``None``."""
    path = os.path.join(directory, 'config.json')
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


_SINGLE_WEIGHT_SUFFIXES = (
    '.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.index.json',
)

_GENERIC_WEIGHT_STEMS = ('model', 'pytorch_model', 'diffusion_pytorch_model')


def single_weight_reference(model: str) -> bool:
    """True when ``model`` is one weight file or a URL to one weight file."""
    cleaned = str(model).split('?', 1)[0].split('#', 1)[0].lower()
    if not cleaned.endswith(_SINGLE_WEIGHT_SUFFIXES):
        return False
    if cleaned.startswith(('http://', 'https://')):
        return True
    return os.path.isfile(model)


def component_name_for_file(path: str, subfolder: str | None = None) -> str:
    """Output stem for one weight file.

    A blob under ``text_encoder/`` is named ``text_encoder``. A generic file
    such as ``model.safetensors`` takes its parent folder. A named file keeps
    its own stem.
    """
    if subfolder:
        return subfolder.replace('\\', '/').strip('/').split('/')[-1]
    name = os.path.basename(path)
    lower = name.lower()
    for suffix in ('.safetensors.index.json', '.bin.index.json'):
        if lower.endswith(suffix):
            stem = name[:-len(suffix)]
            break
    else:
        stem = os.path.splitext(name)[0]
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    generic = stem.split('.', 1)[0] in _GENERIC_WEIGHT_STEMS
    parent_is_hash = len(parent) >= 16 and all(character in '0123456789abcdef' for character in parent)
    if generic and parent and not parent_is_hash:
        return parent
    return stem or 'model'


def weight_targets(root: str,
                   subfolders: list[str] | None = None,
                   variant: str | None = None,
                   weight_name: str | None = None) -> list[tuple[str, str, str | None]]:
    """``(component name, directory, weight file name in that directory)``."""
    if weight_name and ('/' in weight_name or '\\' in weight_name):
        if subfolders:
            raise QuantExportError(
                '--weight-name includes a path, which already selects the component. '
                'Do not also pass --subfolders.')
        relative = weight_name.replace('\\', '/').strip('/')
        path = os.path.normpath(os.path.join(root, *relative.split('/')))
        if not os.path.isfile(path):
            raise QuantExportError(f'Weight file "{weight_name}" was not found in "{root}".')
        parent = os.path.dirname(relative)
        return [(
            component_name_for_file(path, parent or None),
            os.path.dirname(path),
            os.path.basename(path),
        )]
    found = component_directories(root, subfolders, variant, weight_name)
    return [(name, folder, weight_name) for name, folder in found]


def _text_architecture(model_type: str) -> str | None:
    """llama.cpp architecture for a transformers ``model_type``, when one exists."""
    name = model_type.lower().replace('-', '_')
    if name in {'t5', 'mt5', 'umt5', 't5encoder'}:
        return 't5'
    if name.startswith('gemma4') or name.startswith('gemma_4'):
        return 'gemma4'
    if name.startswith('gemma3') or name.startswith('gemma_3'):
        return 'gemma3'
    if name.startswith('qwen3_vl') or name.startswith('qwen3vl'):
        return 'qwen3vl'
    # Qwen3.5 (the FLUX.3 prompt upsampler) is a hybrid language stack.
    # llama.cpp loads that as qwen35. The vision file stays a Qwen3-VL mmproj.
    if name.startswith('qwen3_5') or name.startswith('qwen35'):
        return 'qwen35moe' if 'moe' in name else 'qwen35'
    if name.startswith('qwen3'):
        return 'qwen3'
    if name.startswith('qwen2') and 'vl' in name:
        return 'qwen2vl'
    if name.startswith('qwen2'):
        return 'qwen2'
    if name.startswith('llama') or name.startswith('mistral'):
        return 'llama'
    return None


def _encoder_only_t5(config: dict | None, architecture: str | None) -> str | None:
    """UMT5EncoderModel has no decoder stack. llama.cpp loads that as ``t5encoder``."""
    if architecture != 't5' or not config:
        return architecture
    names = ''.join(str(item) for item in (config.get('architectures') or []))
    model = str(config.get('model_type') or '')
    folded = (names + model).lower().replace('-', '').replace('_', '')
    if 'encodermodel' in folded or folded.endswith('t5encoder'):
        return 't5encoder'
    return architecture


def gguf_architecture(config: dict | None, keys) -> str:
    """
    ``general.architecture`` for a GGUF of this component.

    Text encoders dgenerate reads keep their llama.cpp architecture when the
    tensors already use Hugging Face names. Everything else is ``diffusion``,
    and the tensor names identify it.
    """
    model_type = ''
    if config:
        model_type = str(config.get('model_type') or '')
        if not model_type:
            architectures = config.get('architectures') or []
            if architectures:
                model_type = str(architectures[0])
    architecture = _encoder_only_t5(config, _text_architecture(model_type))
    if architecture is None and (
            model_type.lower().startswith('clip')
            or any(key.endswith('embeddings.token_embedding.weight') and 'vision' not in key
                   for key in keys)):
        return 'clip'
    if architecture is None:
        return 'diffusion'
    if architecture in {'t5', 't5encoder'}:
        hugging_face = any(
            key.startswith('encoder.block.') or key == 'shared.weight' for key in keys)
    else:
        hugging_face = any(
            key.endswith('embed_tokens.weight') or '.self_attn.' in key for key in keys)
    if hugging_face:
        return architecture
    return 'diffusion'


def _is_text_encoder(name: str, config: dict | None) -> bool:
    if 'text_encoder' in name.replace('\\', '/'):
        return True
    if not config:
        return False
    architectures = ' '.join(str(item) for item in (config.get('architectures') or []))
    model_type = str(config.get('model_type') or '')
    return 'Text' in architectures or model_type.startswith('clip')


def _convrot_group(inputs: int) -> int | None:
    """Largest power of 4, at most 256, that divides ``inputs``."""
    group = 256
    while group >= 4:
        if inputs % group == 0:
            return group
        group //= 4
    return None


def _embedding_weight(name: str) -> bool:
    module = name[:-len('.weight')].rsplit('.', 1)[-1].lower() if name.endswith('.weight') else ''
    return module in _EMBEDDING_NAMES


def _floating(tensor: torch.Tensor) -> bool:
    return tensor.is_floating_point() or tensor.dtype in {
        torch.float8_e4m3fn, torch.float8_e5m2}


def _linear_weight(name: str, tensor: torch.Tensor) -> bool:
    return (name.endswith('.weight') and tensor.ndim == 2 and _floating(tensor)
            and not _embedding_weight(name) and tensor.shape[0] > 0 and tensor.shape[1] > 0)


def _marker(config: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(config, separators=(',', ':')).encode('utf-8')), dtype=torch.uint8)


def _clear_neg_zero(storage: torch.Tensor) -> torch.Tensor:
    """float8 e4m3 ``-0`` is NaN when those bits are read as e4m3fnuz."""
    bits = storage.view(torch.uint8)
    return bits.masked_fill(bits == 0x80, 0).view(storage.dtype)


def _quantize_float8(weight: torch.Tensor, fmt: str) -> dict:
    dtype = _safetensorscompat._FLOAT8_DTYPES[fmt]
    largest = float(torch.finfo(dtype).max)
    values = weight.float()
    # ComfyUI's TensorCore FP8 layout is one scale for the whole matrix.
    scale = (values.abs().amax() / largest).clamp(min=1e-12)
    quant = (values / scale).to(dtype)
    if fmt == 'float8_e4m3fn':
        quant = _clear_neg_zero(quant)
    return {
        'weight': quant.contiguous(),
        'weight_scale': scale.reshape(()).to(torch.float32),
        'comfy_quant': _marker({'format': fmt}),
    }


def _quantize_int8(weight: torch.Tensor) -> dict:
    values = weight.float()
    group = _convrot_group(values.shape[1])
    if group is not None:
        values = _safetensorscompat._rotate(values, group)
        # ConvRot INT8 is per output row. Shape (rows, 1) broadcasts over the row.
        scale = (values.abs().amax(dim=-1, keepdim=True) / 127).clamp(min=1e-12)
    else:
        scale = (values.abs().amax() / 127).clamp(min=1e-12).reshape(())
    quant = torch.round(values / scale).clamp(-128, 127).to(torch.int8)
    config = {'format': 'int8_tensorwise'}
    if group is not None:
        config['convrot'] = True
        config['convrot_groupsize'] = group
    return {
        'weight': quant.contiguous(),
        'weight_scale': scale.to(torch.float32).contiguous(),
        'comfy_quant': _marker(config),
    }


def _pack_signed_int4(values: torch.Tensor) -> torch.Tensor:
    codes = values.to(torch.int32)
    packed = (codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)
    return packed.to(torch.int8).contiguous()


def _quantize_int4(weight: torch.Tensor) -> dict | None:
    inputs = weight.shape[1]
    group = _convrot_group(inputs)
    if group is None or inputs % 2:
        return None
    rotated = _safetensorscompat._rotate(weight.float(), group)
    scale = (rotated.abs().amax(dim=-1) / 7).clamp(min=1e-12)
    quant = torch.round(rotated / scale.unsqueeze(-1)).clamp(-7, 7)
    return {
        'weight': _pack_signed_int4(quant),
        'weight_scale': scale.to(torch.float32).contiguous(),
        'comfy_quant': _marker({'format': 'convrot_w4a4', 'convrot_groupsize': group}),
    }


def _pad(weight: torch.Tensor, multiple: int) -> torch.Tensor:
    rows, cols = weight.shape
    padded_rows = _safetensorscompat._round_up(rows, multiple)
    padded_cols = _safetensorscompat._round_up(cols, multiple)
    if padded_rows == rows and padded_cols == cols:
        return weight
    padded = weight.new_zeros(padded_rows, padded_cols)
    padded[:rows, :cols] = weight
    return padded


def _quantize_nvfp4(weight: torch.Tensor) -> dict:
    values = _pad(weight.float(), _safetensorscompat._NVFP4_BLOCK)
    rows, cols = values.shape
    levels = torch.tensor(_safetensorscompat._E2M1_VALUES[:8])
    tensor_scale = (values.abs().amax() / (448 * 6)).clamp(min=1e-12)
    blocks = values.view(rows, cols // _safetensorscompat._NVFP4_BLOCK, _safetensorscompat._NVFP4_BLOCK)
    block_scale = (blocks.abs().amax(-1) / 6 / tensor_scale).clamp(max=448).to(torch.float8_e4m3fn)
    total = (block_scale.float() * tensor_scale).clamp(min=1e-12)
    scaled = blocks / total.unsqueeze(-1)
    codes = (scaled.abs().unsqueeze(-1) - levels).abs().argmin(-1)
    codes = (codes | ((scaled < 0).long() << 3)).reshape(rows, cols)
    packed = ((codes[:, 0::2] << 4) | codes[:, 1::2]).to(torch.uint8)
    return {
        'weight': packed.contiguous(),
        'weight_scale': _safetensorscompat._to_blocked(block_scale).contiguous(),
        'weight_scale_2': tensor_scale.reshape(()).to(torch.float32),
        'comfy_quant': _marker({'format': 'nvfp4'}),
    }


def _quantize_mxfp8(weight: torch.Tensor) -> dict:
    values = _pad(weight.float(), _safetensorscompat._MXFP8_BLOCK)
    rows, cols = values.shape
    blocks = values.view(rows, cols // _safetensorscompat._MXFP8_BLOCK, _safetensorscompat._MXFP8_BLOCK)
    max_abs = blocks.abs().amax(-1)
    need = (max_abs / 448).clamp(min=2.0 ** -127)
    exponents = (torch.ceil(torch.log2(need)).to(torch.int32) + 127).clamp(0, 254).to(torch.uint8)
    zero = max_abs == 0
    scale = (exponents.to(torch.int32) << 23).view(torch.float32)
    scale = torch.where(zero, torch.ones_like(scale), scale)
    quant = (blocks / scale.unsqueeze(-1)).reshape(rows, cols).clamp(-448, 448)
    exponents = torch.where(zero, torch.zeros_like(exponents), exponents)
    return {
        'weight': _clear_neg_zero(quant.to(torch.float8_e4m3fn)).contiguous(),
        'weight_scale': _safetensorscompat._to_blocked(exponents).contiguous(),
        'comfy_quant': _marker({'format': 'mxfp8'}),
    }


def _quantize_w4a8(weight: torch.Tensor) -> dict | None:
    group = 16
    inputs = weight.shape[1]
    rotation = _convrot_group(inputs)
    if rotation is None or inputs % group:
        return None
    rotated = _safetensorscompat._rotate(weight.float(), rotation)
    rows = rotated.shape[0]
    groups = inputs // group
    grouped = rotated.view(rows, groups, group)
    codebook = torch.tensor(_safetensorscompat._W4A8_CODEBOOK)
    group_scale = grouped.abs().amax(-1, keepdim=True).clamp(min=1e-8)

    def assign(normalized):
        return (normalized.unsqueeze(-1) - codebook).abs().argmin(-1)

    codes = assign(grouped / group_scale)
    for _ in range(2):
        levels = codebook[codes]
        group_scale = ((grouped * levels).sum(-1, keepdim=True)
                       / levels.square().sum(-1, keepdim=True).clamp(min=1e-8)).clamp(min=1e-8)
        codes = assign(grouped / group_scale)
    channel = (codebook[codes] * group_scale).abs().amax(dim=(1, 2)).clamp(min=1e-8) / 127
    rel = (group_scale.squeeze(-1) / channel.unsqueeze(1)).float()
    grid = (codebook.view(1, 1, -1) * rel.unsqueeze(-1)).round().clamp(-127, 127)
    target = grouped / channel.view(rows, 1, 1)
    codes = (target.unsqueeze(-1) - grid.unsqueeze(2)).abs().argmin(-1).reshape(rows, inputs)
    packed = ((codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)).to(torch.int8)
    return {
        'weight': packed.contiguous(),
        'weight_s_rel': rel.contiguous(),
        'weight_s_channel': channel.reshape(rows).contiguous(),
        'weight_codebook': codebook.contiguous(),
        'comfy_quant': _marker({
            'format': 'asym_w4a8_int8', 'group_size': group, 'convrot_groupsize': rotation}),
    }


def _pack_int6(codes: torch.Tensor) -> torch.Tensor:
    low = ((codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)).to(torch.int32)
    high = (codes[:, 0::4] >> 4) & 0x3
    for shift in range(1, 4):
        high = high | (((codes[:, shift::4] >> 4) & 0x3) << (2 * shift))
    return torch.cat((low, high), dim=1).to(torch.int8).contiguous()


def _quantize_w6a8(weight: torch.Tensor) -> dict | None:
    group = 16
    inputs = weight.shape[1]
    rotation = _convrot_group(inputs)
    if rotation is None or inputs % 32 or inputs % group:
        return None
    rotated = _safetensorscompat._rotate(weight.float(), rotation)
    rows = rotated.shape[0]
    groups = inputs // group
    grouped = rotated.view(rows, groups, group)
    group_scale = (grouped.abs().amax(-1, keepdim=True) / 31).clamp(min=1e-8)

    def assign(normalized):
        return normalized.round().clamp(-31, 31)

    signed = assign(grouped / group_scale)
    for _ in range(2):
        group_scale = ((grouped * signed).sum(-1, keepdim=True)
                       / signed.square().sum(-1, keepdim=True).clamp(min=1e-8)).clamp(min=1e-8)
        signed = assign(grouped / group_scale)
    channel = (signed * group_scale).abs().amax(dim=(1, 2)).clamp(min=1e-8) / 127
    rel = (group_scale.squeeze(-1) / channel.unsqueeze(1)).float()
    level_table = torch.arange(-31, 32)
    grid = (level_table.view(1, 1, -1) * rel.unsqueeze(-1)).round().clamp(-127, 127)
    target = grouped / channel.view(rows, 1, 1)
    index = (target.unsqueeze(-1) - grid.unsqueeze(2)).abs().argmin(-1)
    return {
        'weight': _pack_int6((index + 1).reshape(rows, inputs)),
        'weight_s_rel': rel.contiguous(),
        'weight_s_channel': channel.reshape(rows).contiguous(),
        'comfy_quant': _marker({
            'format': 'w6a8_int8', 'group_size': group, 'convrot_groupsize': rotation}),
    }


_COMFY_QUANTIZERS = {
    'float8_e4m3fn': lambda weight: _quantize_float8(weight, 'float8_e4m3fn'),
    'float8_e5m2': lambda weight: _quantize_float8(weight, 'float8_e5m2'),
    'int8_tensorwise': _quantize_int8,
    'nvfp4': _quantize_nvfp4,
    'mxfp8': _quantize_mxfp8,
    'convrot_w4a4': _quantize_int4,
    'asym_w4a8_int8': _quantize_w4a8,
    'w6a8_int8': _quantize_w6a8,
}


def _dense_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_floating_point() and tensor.dtype not in {
            torch.float16, torch.float32, torch.float64, torch.bfloat16}:
        return tensor.detach().to(torch.float32).cpu().contiguous()
    return tensor.detach().cpu().contiguous()


def _write_plain(state: dict, path: str, fmt: str) -> int:
    """Every tensor in ``fmt``, with no quant metadata."""
    dtype = _PLAIN_DTYPES[fmt]
    tensors = {}
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        value = tensor.detach().cpu().contiguous()
        if value.is_floating_point() and value.dtype != dtype:
            value = value.to(dtype)
        tensors[name] = value
    if not tensors:
        return 0
    import safetensors.torch
    safetensors.torch.save_file(tensors, path)
    return len(tensors)


def _write_comfy(state: dict, path: str, fmt: str) -> int | None:
    quantize = _COMFY_QUANTIZERS[fmt]
    tensors = {}
    layers = {}
    quantized = 0
    entries = [
        (name, tensor) for name, tensor in state.items()
        if isinstance(tensor, torch.Tensor)]
    for name, tensor in _progress(entries, fmt, 'tensor'):
        packed = quantize(tensor.detach().cpu()) if _linear_weight(name, tensor) else None
        if packed is None:
            tensors[name] = _dense_tensor(tensor)
            continue
        stem = name[:-len('.weight')]
        layers[stem] = json.loads(bytes(packed['comfy_quant'].tolist()).decode('utf-8'))
        for suffix, value in packed.items():
            tensors[f'{stem}.{suffix}' if suffix != 'weight' else name] = value
        quantized += 1
    if not quantized:
        return None
    import safetensors.torch
    safetensors.torch.save_file(tensors, path, metadata={
        '_quantization_metadata': json.dumps({
            'format_version': '1.0',
            'layers': layers,
        }, separators=(',', ':')),
        'dgenerate_quant_format': fmt,
    })
    return quantized


def _gguf_qtype(name: str):
    import gguf
    return getattr(gguf.GGMLQuantizationType, name)


def _comfy_f32_key(name: str) -> bool:
    """Keys ComfyUI-GGUF's converter keeps in F32 (``keys_hiprec``)."""
    return (
        '.modulation' in name
        or name.endswith('modulation')
        or 'scale_shift_table' in name
        or 'img_emb.emb_pos' in name
        or '.ff_i.gate.weight' in name
        or 'pos_embedder' in name)


def _write_gguf(state: dict,
                path: str,
                fmt: str,
                architecture: str,
                record_shapes: bool = False,
                extra_strings: dict | None = None,
                metadata=None) -> tuple[int, int]:
    import gguf
    import numpy

    qtype = _gguf_qtype(fmt)
    block = gguf.GGML_QUANT_SIZES[qtype][0]
    writer = gguf.GGUFWriter(path, architecture, use_temp_file=True)
    for key, value in (extra_strings or {}).items():
        writer.add_string(key, value)
    quantized = 0
    stored = 0
    tensors = [
        (name, tensor) for name, tensor in state.items()
        if isinstance(tensor, torch.Tensor)]
    try:
        for name, tensor in _progress(tensors, fmt, 'tensor'):
            array = tensor.detach().to(torch.float32).cpu().contiguous().numpy()
            logical_shape = tuple(int(dim) for dim in array.shape)
            raw = None
            did_quant = False
            if array.size == 0 or qtype == gguf.GGMLQuantizationType.F32:
                stored_array = numpy.ascontiguousarray(array.astype(numpy.float32, copy=False))
            elif qtype == gguf.GGMLQuantizationType.F16:
                stored_array = numpy.ascontiguousarray(array.astype(numpy.float16))
            elif qtype == gguf.GGMLQuantizationType.BF16:
                stored_array = _ggufquant.quantize(array, qtype)
                raw = qtype
            elif 'patch_embd' in name or 'position_embd' in name or 'ssm_conv' in name or (
                    (extra_strings or {}).get('general.type') == 'mmproj' and 'conv' in name):
                # Qwen3-VL resizes v.position_embd through ggml_interpolate, which
                # requires F32. The patch conv accepts F16 and BF16; F32 is what
                # llama.cpp's own mmproj keeps for both of these small tensors.
                # Gemma 4 audio convs are stored F32 the same way.
                stored_array = numpy.ascontiguousarray(array.astype(numpy.float32, copy=False))
            elif tensor.ndim <= 1 or tensor.ndim > 4 or _comfy_f32_key(name):
                # ComfyUI-GGUF quantizes at most four dimensions, so a Wan patch
                # conv stays F32 and the file shape is the tensor shape. Its
                # converter also keeps ``.modulation`` in F32. A BF16 pack of
                # either one is not what that loader reads.
                stored_array = numpy.ascontiguousarray(array.astype(numpy.float32, copy=False))
            elif (tensor.ndim == 2 and _floating(tensor)
                  and array.shape[-1] % block == 0 and array.shape[-1] > 0):
                stored_array = _ggufquant.quantize(array, qtype)
                raw = qtype
                did_quant = True
            else:
                stored_array = _ggufquant.quantize(array, gguf.GGMLQuantizationType.BF16)
                raw = gguf.GGMLQuantizationType.BF16
            if raw is None:
                writer.add_tensor(name, stored_array)
            else:
                writer.add_tensor(name, stored_array, raw_dtype=raw)
            if record_shapes and logical_shape:
                writer.add_key_value(
                    f'comfy.gguf.orig_shape.{name}',
                    list(logical_shape),
                    gguf.GGUFValueType.ARRAY,
                    gguf.GGUFValueType.INT32)
            if did_quant:
                quantized += 1
            else:
                stored += 1
        if metadata is not None:
            metadata(writer)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    except Exception:
        writer.close()
        if os.path.exists(path):
            os.remove(path)
        raise
    writer.close()
    return quantized, stored


def _count(amount: int, noun: str) -> str:
    return f'{amount} {noun}' if amount == 1 else f'{amount} {noun}s'


def _output_path(output_dir: str, stem: str, fmt: str, marker: str = '') -> str:
    suffix = '.gguf' if fmt in set(gguf_format_names()) else '.safetensors'
    safe = stem.replace('\\', '/').replace('/', '-').replace(' ', '_')
    tag = f'-{marker}' if marker else ''
    return os.path.join(output_dir, f'{safe}-{fmt}{tag}{suffix}')


def export_component(state: dict[str, torch.Tensor],
                     formats: list[str],
                     output_dir: str,
                     stem: str,
                     config: dict | None = None,
                     comfy: bool = False,
                     llama: bool = False,
                     source_dir: str | None = None) -> list[str]:
    """
    Write one file per format. Returns the paths that were written.

    A Comfy format that cannot quantize any linear in ``state`` is skipped.
    ``float16`` and ``float32`` store every tensor in that dtype, with no quant.
    GGUF ``F16`` and ``F32`` do the same inside a GGUF. Matrices whose width is
    not a multiple of the quant block are stored as BF16. 1D tensors are stored
    as F32. A tensor with more than four dimensions, and Comfy's high-precision
    keys such as ``.modulation``, are stored as F32. ComfyUI-GGUF does not
    quantize a 5D Wan patch conv, and it cannot load ``.modulation`` from BF16.

    :param comfy: Write ComfyUI tensor names. A component with no key map is an error.
    :param llama: Write llama.cpp tensor names for a text encoder dgenerate loads.
        Mutually exclusive with ``comfy``.
    :raises QuantExportError: A file could not be written, or the naming mode has no key map.
    """
    if comfy and llama:
        raise QuantExportError('--llama and --comfy choose different tensor names.')
    os.makedirs(output_dir, exist_ok=True)
    gguf_names = set(gguf_format_names())
    layouts = {}
    name_marker = 'comfy' if comfy else 'llama' if llama else ''
    llama_metadata = None
    if llama and any(fmt not in gguf_names for fmt in formats):
        raise QuantExportError(
            '--llama writes llama.cpp GGUF names. Leave safetensors formats out of --formats, '
            'or run them without --llama.')
    if comfy:
        from dgenerate.quantize.comfykeys import comfy_state
        if any(fmt in gguf_names for fmt in formats):
            layouts['gguf'] = comfy_state(state, config, gguf=True)
        if any(fmt not in gguf_names for fmt in formats):
            layouts['safetensors'] = comfy_state(state, config, gguf=False)
        label = next(iter(layouts.values()))[1]
        _messages.log(f'ComfyUI tensor names for "{stem}" ({label}).')
    elif llama:
        from dgenerate.quantize.comfykeys import llama_state
        from dgenerate.quantize.llamameta import llama_sidecar, tie_output
        converted, kind = llama_state(state, config)
        layouts['gguf'] = (tie_output(converted, config), kind)
        _messages.log(f'llama.cpp tensor names for "{stem}" ({kind}).')
        llama_metadata = llama_sidecar(config, source_dir, stem)
    else:
        architecture = gguf_architecture(config, state.keys())
        if (architecture == 'diffusion' and _is_text_encoder(stem, config)
                and any(fmt in gguf_names for fmt in formats)):
            _messages.warning(
                f'"{stem}" is a text encoder outside the GGUF text encoders dgenerate reads '
                f'(T5, UMT5, Qwen, Mistral, Gemma, and CLIP). The GGUF is still written. '
                f'A Comfy safetensors quant of this component loads in dgenerate.')
    written = []
    gguf_count = sum(fmt in gguf_names for fmt in formats)
    told_mmproj = False
    for fmt in formats:
        path = _output_path(output_dir, stem, fmt, name_marker)
        try:
            if comfy or llama:
                source, architecture = layouts['gguf' if fmt in gguf_names else 'safetensors']
            else:
                source = state
            if fmt in gguf_names:
                quantized, stored = _write_gguf(
                    source, path, fmt, architecture, record_shapes=comfy,
                    metadata=llama_metadata if llama else None)
                if comfy or llama:
                    from dgenerate.quantize.comfykeys import mmproj_state
                    from dgenerate.quantize.llamameta import (
                        gemma4_mmproj_metadata, qwen3vl_mmproj_metadata)
                    vision = mmproj_state(state, config)
                    if vision:
                        folder, filename = os.path.split(path)
                        vision_path = os.path.join(folder, 'mmproj-' + filename)
                        vision_metadata = qwen3vl_mmproj_metadata(config, source_dir)
                        if vision_metadata is None:
                            vision_metadata = gemma4_mmproj_metadata(config, vision)
                        _write_gguf(
                            vision, vision_path, fmt,
                            'clip' if vision_metadata is not None else architecture,
                            record_shapes=True,
                            extra_strings={'general.type': 'mmproj'},
                            metadata=vision_metadata)
                        written.append(vision_path)
                        _messages.log(f'Wrote {vision_path} (mmproj).')
                        if gguf_count > 1 and not told_mmproj:
                            _messages.warning(
                                'ComfyUI-GGUF picks one mmproj when several quants share a folder. '
                                'Write one GGUF format per folder to pair the vision tower.')
                            told_mmproj = True
                if fmt in {'F32', 'F16', 'BF16'}:
                    detail = _count(stored, 'tensor') + f' stored as {fmt}'
                else:
                    detail = (
                        _count(quantized, 'tensor') + ' quantized, '
                        + _count(stored, 'tensor') + ' stored full precision')
            else:
                if fmt in _PLAIN_DTYPES:
                    stored = _write_plain(source, path, fmt)
                    if not stored:
                        _messages.warning(
                            f'"{stem}" has no tensors to store as {fmt}. That file was not written.')
                        continue
                    detail = _count(stored, 'tensor') + f' stored as {fmt}'
                else:
                    quantized = _write_comfy(source, path, fmt)
                    if not quantized:
                        _messages.warning(
                            f'No linear in "{stem}" can be stored as {fmt}. That file was not written.')
                        continue
                    detail = _count(quantized, 'linear') + ' quantized'
        except QuantExportError:
            raise
        except Exception as error:
            if os.path.exists(path):
                os.remove(path)
            raise QuantExportError(f'Could not write "{path}": {error}') from error
        _messages.log(f'Wrote {path} ({detail}).')
        written.append(path)
    if written:
        _copy_text_sidecars(source_dir, output_dir, config)
    return written


_TEXT_SIDECARS = (
    'config.json',
    'generation_config.json',
    'preprocessor_config.json',
    'processor_config.json',
    'tokenizer_config.json',
    'tokenizer.json',
    'tokenizer.model',
    'special_tokens_map.json',
    'added_tokens.json',
    'vocab.json',
    'merges.txt',
    'chat_template.json',
    'chat_template.jinja',
)


def _copy_text_sidecars(source_dir: str | None, output_dir: str, config: dict | None) -> None:
    """Config, tokenizer, and processor files beside a text quant.

    Loading the quant for generation does not read ``config.json``. The copy
    is there for a later quantize of that file, and for anything else that
    looks beside the weights for the architecture. A folder that already has
    ``config.json`` is left alone.
    """
    if not source_dir or not isinstance(config, dict):
        return
    if _text_architecture(str(config.get('model_type') or '')) is None:
        return
    if os.path.isfile(os.path.join(output_dir, 'config.json')):
        return
    copied = []
    for name in _TEXT_SIDECARS:
        source = os.path.join(source_dir, name)
        destination = os.path.join(output_dir, name)
        if os.path.isfile(source) and not os.path.exists(destination):
            shutil.copy2(source, destination)
            copied.append(name)
    if copied:
        _messages.log('Copied the config and processor beside the quant.')
