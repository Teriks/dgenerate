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
import contextlib
import json
import logging
import os
import pathlib
import typing

import accelerate
import diffusers.loaders
import diffusers.pipelines.kolors
import torch
import transformers.models.clip

import dgenerate.exceptions as _d_exceptions
import dgenerate.hfhub as _hfhub
import dgenerate.memoize as _d_memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.safetensorscompat as _safetensorscompat
import dgenerate.pipelinewrapper.sdnqload as _sdnqload
import dgenerate.pipelinewrapper.util as _pipelinewrapper_util
import dgenerate.textprocessing as _textprocessing
import dgenerate.torchutil as _torchutil
import dgenerate.types as _types
from dgenerate.memoize import memoize as _memoize
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions
from dgenerate.pipelinewrapper.uris import util as _util
import dgenerate.pipelinewrapper.models as _models


_text_encoder_uri_parser = _textprocessing.ConceptUriParser(
    'TextEncoder', [
        'model',
        'revision',
        'variant',
        'subfolder',
        'dtype',
        'quantizer',
        'mode'
    ]
)

_text_encoder_cache = _d_memoize.create_object_cache(
    'text_encoder', cache_type=_memory.SizedConstrainedObjectCache
)


@contextlib.contextmanager
def _suppress_accelerate_warnings():
    """Context manager to temporarily suppress Accelerate warnings."""
    # Get the logger
    logger = logging.getLogger("accelerate.utils.modeling")
    # Store original level
    original_level = logger.level
    # Set level to ERROR (suppressing warnings)
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        # Restore original level
        logger.setLevel(original_level)


def _read_hub_config(repo_name, subfolder, filename='config.json'):
    hub_configs_dir = pathlib.Path(__file__).parent.parent / 'hub_configs'
    config_path = hub_configs_dir / repo_name / subfolder / filename
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Config not found: {config_path}")


def _monolithic_auto_quant_device_map(
        quantization_config, device_map: str | None
):
    if device_map is None:
        return "auto"

    if device_map == 'cpu':
        return {"": device_map}

    return "auto"


def _translate_flattened_clip_checkpoint_keys(loaded, model_keys):
    """Drop a monolithic ``text_model.`` prefix when the module no longer has that child.

    Transformers 5 flattens ``CLIPTextModel``: ``embeddings`` and ``encoder`` sit on the
    module itself. ComfyUI / single-file clip-l checkpoints still store
    ``text_model.embeddings.*``. Accelerate refuses any checkpoint key that is not in
    ``state_dict()``, so those weights are skipped. After bitsandbytes replaces the
    linears with empty meta modules, ``dispatch_model`` then dies with
    "Cannot copy out of meta tensor".
    """
    prefix = 'text_model.'
    translated = {}
    stripped_count = 0
    for key, value in loaded.items():
        if key in model_keys:
            translated[key] = value
            continue

        stripped = None
        marker = key.rfind(prefix)
        if marker != -1:
            candidate = key[marker + len(prefix):]
            if candidate in model_keys:
                stripped = candidate

        if stripped is None:
            translated[key] = value
            continue

        if stripped not in translated:
            translated[stripped] = value
            stripped_count += 1

    if stripped_count:
        _messages.debug_log(
            'Translated '
            f'{stripped_count} monolithic CLIP checkpoint key(s) '
            'by dropping the "text_model." prefix.'
        )
    return translated


def _supply_missing_clip_text_projection(loaded, model_keys, projection):
    """Give ``CLIPTextModelWithProjection`` a projection when the file has none.

    ComfyUI ``clip_l.safetensors`` stores the text tower only. Diffusers fills
    the missing ``text_projection`` with an identity matrix for that file.
    Bitsandbytes has already replaced the linear with an empty meta module, so
    leaving the key absent makes ``dispatch_model`` die on ``model.to()``.

    An OpenCLIP matrix stored as ``text_projection`` is transposed onto
    ``text_projection.weight``, which is what Diffusers does for that layout.
    """
    target = 'text_projection.weight'
    if target not in model_keys or target in loaded:
        return loaded

    bare = loaded.get('text_projection')
    if torch.is_tensor(bare) and bare.ndim == 2:
        loaded = dict(loaded)
        loaded[target] = bare.transpose(0, 1).contiguous()
        _messages.debug_log(
            'Mapped monolithic CLIP "text_projection" onto "text_projection.weight".'
        )
        return loaded

    weight = getattr(projection, 'weight', None)
    if weight is None or weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        return loaded

    loaded = dict(loaded)
    loaded[target] = torch.eye(weight.shape[0])
    _messages.debug_log(
        'Monolithic CLIP checkpoint has no text_projection; using an identity projection.'
    )
    return loaded


@contextlib.contextmanager
def _flattened_clip_checkpoint_keys(model):
    model_keys = set(model.state_dict().keys())
    flattened = (
            'embeddings.token_embedding.weight' in model_keys and
            'text_model.embeddings.token_embedding.weight' not in model_keys
    )
    projection = getattr(model, 'text_projection', None)
    if not flattened and projection is None:
        yield
        return

    modeling = accelerate.utils.modeling
    original = modeling.load_state_dict

    def load_state_dict(checkpoint_file, device_map=None):
        loaded = original(checkpoint_file, device_map=device_map)
        if flattened:
            loaded = _translate_flattened_clip_checkpoint_keys(loaded, model_keys)
        if projection is not None:
            loaded = _supply_missing_clip_text_projection(loaded, model_keys, projection)
        return loaded

    modeling.load_state_dict = load_state_dict
    try:
        yield
    finally:
        modeling.load_state_dict = original


_LANGUAGE_MODEL_PREFIXES = (
    # checkpoint prefix -> model prefix, tried in order for each tensor
    ('', ''),
    ('model.', ''),
    ('', 'model.'),
    ('model.', 'model.language_model.'),
    ('model.', 'language_model.model.'),
    ('visual.', 'model.visual.'),
)


def _language_model_layout(header: dict) -> tuple[str, int, int, int] | None:
    """``(embedding name, vocab size, hidden size, decoder layers)`` from a file header."""
    embed = next((key for key in header if key.endswith('embed_tokens.weight')
                  and 'visual' not in key), None)
    if embed is None:
        return None
    vocab, hidden = header[embed]['shape'][:2]
    prefix = embed[:-len('embed_tokens.weight')] + 'layers.'
    indices = {key[len(prefix):].split('.', 1)[0] for key in header if key.startswith(prefix)}
    layers = 1 + max((int(index) for index in indices if index.isdigit()), default=-1)
    return embed, vocab, hidden, layers


def _language_model_config(model_path: str, encoder) -> str | None:
    """
    A config shipped with dgenerate for a single file language model text
    encoder (Qwen3, Mistral 3, ...), matched by model type, vocabulary, and
    hidden size in the file header. A file may hold fewer decoder layers than
    the config: ComfyUI trims encoders to the layers a pipeline reads.
    ``None`` when there is no match.
    """
    config_class = getattr(encoder, 'config_class', None)
    model_type = getattr(config_class, 'model_type', None)
    if not model_type or not model_path.lower().endswith('.safetensors'):
        return None
    header = _safetensorscompat._read_header(model_path) if os.path.isfile(model_path) else None
    if not header:
        return None
    header.pop('__metadata__', None)
    layout = _language_model_layout(header)
    if layout is None:
        return None
    _, vocab, hidden, layers = layout

    configs = pathlib.Path(__file__).parent.parent / 'hub_configs'
    best = None
    for config_path in sorted(configs.glob('*/text_encoder*/config.json')):
        with open(config_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        if config.get('model_type') != model_type:
            continue
        text = config.get('text_config') or config
        config_layers = text.get('num_hidden_layers') or 0
        if (text.get('hidden_size'), text.get('vocab_size')) != (hidden, vocab) or config_layers < layers:
            continue
        # The closest layer count: an exact match, else the least trimming.
        if best is None or config_layers < best[0]:
            best = (config_layers, str(config_path))
    return best[1] if best else None


def _language_model_rename(model_keys: set, checkpoint_keys) -> typing.Callable[[str], str]:
    """
    Checkpoint -> model tensor names.

    Each tensor takes the first prefix mapping that names a model tensor, so a
    language model and a vision tower under different prefixes both resolve
    (Qwen2.5-VL: ``model.`` -> ``model.language_model.``, ``visual.`` ->
    ``model.visual.``).
    """
    def make(source, target):
        def rename(key):
            if source and not key.startswith(source):
                return None
            return target + key[len(source):]
        return rename

    rules = [make(source, target) for source, target in _LANGUAGE_MODEL_PREFIXES]

    def rename(key):
        for rule in rules:
            target = rule(key)
            if target is not None and target in model_keys:
                return target
        return key

    checkpoint_keys = list(checkpoint_keys)
    matches = sum(1 for key in checkpoint_keys if rename(key) in model_keys)
    if matches < 0.9 * len(checkpoint_keys):
        raise _exceptions.TextEncoderUriLoadError(
            f'Only {matches} of {len(checkpoint_keys)} tensors in the text encoder file '
            f'match the model, it is not a checkpoint of this text encoder.')
    return rename


def _load_language_model_from_single_file(model_class, model_path: str, config_path: str,
                                          dtype: torch.dtype | None):
    """
    A transformers language model text encoder from a single ``.safetensors``
    file (as ComfyUI distributes them), with a config shipped with dgenerate.

    ComfyUI quantized linears stay packed. Weights the file does not hold, such
    as the vision tower of Mistral 3, which text encoding does not use, are
    zeros.

    A file trimmed to fewer decoder layers than the config (ComfyUI's Flux.2
    Mistral keeps the 30 of 40 layers whose hidden states the pipeline reads)
    builds that many layers. Without the final norm in the file, the norm is
    an identity, so the last hidden state is the raw output of the last layer,
    as it is at that index in the full model.
    """
    config = model_class.config_class.from_json_file(config_path)
    dtype = dtype or torch.bfloat16
    header = _safetensorscompat._read_header(model_path)
    header.pop('__metadata__', None)
    embed, _, _, layers = _language_model_layout(header)

    text_config = getattr(config, 'text_config', None) or config
    trimmed = layers < text_config.num_hidden_layers
    if trimmed:
        text_config.num_hidden_layers = layers
        layer_types = getattr(text_config, 'layer_types', None)
        if layer_types:
            text_config.layer_types = list(layer_types)[:layers]

    with accelerate.init_empty_weights(include_buffers=False):
        model = model_class(config)

    model_keys = set(model.state_dict().keys())
    # Model tensors: not scales or markers, and not uint8 blobs such as the
    # tokenizer some ComfyUI files embed (NVFP4 weights are uint8 ``.weight``).
    weight_keys = [key for key in header
                   if key != _safetensorscompat._LEGACY_MARKER
                   and not key.endswith(_safetensorscompat._AUX_SUFFIXES)
                   and not key.endswith(tuple(_safetensorscompat._LEGACY_SCALE_NAMES))
                   and (header[key]['dtype'] != 'U8' or key.endswith('.weight'))]
    rename = _language_model_rename(model_keys, weight_keys)

    if trimmed:
        norm_name = rename(embed)[:-len('embed_tokens.weight')] + 'norm'
        norm_weight = norm_name + '.weight'
        if norm_weight in model_keys and not any(rename(key) == norm_weight for key in weight_keys):
            _safetensorscompat._set_submodule(model, norm_name, torch.nn.Identity())
            model_keys = set(model.state_dict().keys())
            _messages.debug_log(
                f'Text encoder file "{model_path}" holds {layers} of '
                f'{text_config.num_hidden_layers} layers and no final norm; '
                f'its last hidden state is the raw output of layer {layers}.')

    state = _safetensorscompat.prepare_text_encoder(model, model_path, dtype, rename=rename)
    keep_fp32 = set(_keep_in_fp32_module_names(model))
    for key, value in state.items():
        target = rename(key)
        if target not in model_keys:
            continue
        module = target.rsplit('.', 2)[-2] if target.count('.') else ''
        value_dtype = None
        if value.is_floating_point():
            value_dtype = torch.float32 if module in keep_fp32 else dtype
        accelerate.utils.set_module_tensor_to_device(model, target, 'cpu', value=value, dtype=value_dtype)

    model.tie_weights()
    missing = {name: tensor for name, tensor in
               list(model.named_parameters()) + list(model.named_buffers())
               if tensor.device.type == 'meta'}
    for name, tensor in missing.items():
        accelerate.utils.set_module_tensor_to_device(
            model, name, 'cpu', value=torch.zeros(tensor.shape, dtype=tensor.dtype))
    missing = list(missing)
    if missing:
        _messages.debug_log(
            f'Text encoder file "{model_path}" holds no weights for {len(missing)} tensors '
            f'(for example {missing[0]}); they are zeros.')
    return model.eval()


def _load_monolithic_checkpoint(
        model,
        checkpoint,
        device_map,
        dtype,
        no_split_module_classes,
):
    if _safetensorscompat.is_comfy_quantized(checkpoint):
        # Keep its quantized linears packed, unless another quantizer already
        # replaced the linears and needs dense weights.
        quantized = getattr(model.config, 'quantization_config', None) is not None
        state = _safetensorscompat.prepare_text_encoder(
            model, checkpoint, dtype, pack=not quantized)
        with _safetensorscompat.accelerate_checkpoint(checkpoint, state):
            return _load_monolithic_state(
                model, checkpoint, device_map, dtype, no_split_module_classes)
    return _load_monolithic_state(model, checkpoint, device_map, dtype, no_split_module_classes)


def _load_monolithic_state(
        model,
        checkpoint,
        device_map,
        dtype,
        no_split_module_classes,
):
    with _flattened_clip_checkpoint_keys(model):
        return accelerate.load_checkpoint_and_dispatch(
            model,
            checkpoint=checkpoint,
            device_map=device_map,
            dtype=dtype,
            no_split_module_classes=no_split_module_classes,
        )


def _keep_in_fp32_module_names(model) -> list[str]:
    """Module names transformers refuses to quantize, such as T5 ``wo``."""
    names = getattr(model, '_keep_in_fp32_modules', None) or []
    return [name for name in names if name]


def _cast_kept_fp32_modules(model) -> None:
    """Put skipped modules back in float32 after a dtype-cast checkpoint load.

    T5-XXL keeps ``DenseReluDense.wo`` in float32 during 4-bit and 8-bit
    loads. Quantizing that projection, or leaving it in bf16 next to nf4
    linears, scrambles the hidden states Flux uses as the prompt.
    """
    names = set(_keep_in_fp32_module_names(model))
    if not names:
        return

    for module_name, module in model.named_modules():
        short = module_name.rsplit('.', 1)[-1]
        if short not in names and module_name not in names:
            continue
        for param_name, param in list(module.named_parameters(recurse=False)):
            if not param.is_floating_point() or param.dtype == torch.float32:
                continue
            module._parameters[param_name] = torch.nn.Parameter(
                param.detach().to(dtype=torch.float32),
                requires_grad=False,
            )


def _monolithic_hf_quantizer(quantization_config):
    """Build the quantizer for a full-precision monolithic checkpoint.

    SDNQ defaults ``pre_quantized=True``, which packs the randomly initialized
    weights before the checkpoint tensor is copied in. These files are full
    precision, so the pack has to happen after the load.
    """
    kwargs = {}
    if type(quantization_config).__name__ == 'SDNQConfig':
        kwargs['pre_quantized'] = False
    return diffusers.quantizers.auto.DiffusersAutoQuantizer().from_config(
        quantization_config, **kwargs)


def _finish_monolithic_quantization(text_encoder, hf_quantizer, dtype):
    """Quantize a loaded monolithic encoder and restore layers that must stay in fp32.

    Bitsandbytes quantizes while the checkpoint is copied onto the module.
    SDNQ quantizes afterwards, and only if this step runs. Both honor T5's
    ``wo`` skip so the prompt hidden states stay intact.
    """
    if hf_quantizer.__class__.__name__ == 'SDNQQuantizer' and not getattr(hf_quantizer, 'pre_quantized', False):
        from sdnq.quantizer import apply_sdnq_to_module
        text_encoder, quant_config = apply_sdnq_to_module(
            text_encoder,
            hf_quantizer.quantization_config,
            torch_dtype=dtype,
        )
        hf_quantizer.quantization_config = quant_config
    _cast_kept_fp32_modules(text_encoder)
    hf_quantizer.postprocess_model(text_encoder)
    return text_encoder

def _load_clip_l_from_single_file(
        model_class: transformers.CLIPTextModel | transformers.CLIPTextModelWithProjection,
        model_path: str,
        dtype: torch.dtype,
        quantization_config=None,
        device_map: str | None = None
):
    try:
        with _suppress_accelerate_warnings():
            config = transformers.CLIPTextConfig.from_dict(
                _read_hub_config('models--stable-diffusion-v1-5--stable-diffusion-v1-5', 'text_encoder')
            )

            config.torch_dtype = dtype

            if quantization_config:
                config.quantization_config = quantization_config

            text_encoder = model_class(config)

        _messages.log(
            f'Loading monolithic clip-l state dict from: "{model_path}", this may take a while, please wait...',
            level=_messages.WARNING
        )

        device_map = _monolithic_auto_quant_device_map(quantization_config, device_map)

        if quantization_config:
            hf_quantizer = _monolithic_hf_quantizer(quantization_config)
            hf_quantizer.preprocess_model(
                text_encoder,
                device_map=device_map,
                keep_in_fp32_modules=_keep_in_fp32_module_names(text_encoder),
            )

        with _suppress_accelerate_warnings():
            # Load state dict and update weights
            text_encoder = _load_monolithic_checkpoint(
                text_encoder,
                checkpoint=model_path,
                device_map=device_map,
                dtype=dtype,
                no_split_module_classes=["CLIPEncoderLayer"]
            )

        if quantization_config:
            text_encoder = _finish_monolithic_quantization(text_encoder, hf_quantizer, dtype)

        return text_encoder.eval()

    except Exception as e:
        raise _exceptions.TextEncoderUriLoadError(
            f"Failed to load CLIP-L model: {e}"
        ) from e


def _load_clip_l_sd3_from_single_file(
        model_class: transformers.CLIPTextModel | transformers.CLIPTextModelWithProjection,
        model_path: str,
        dtype: torch.dtype,
        quantization_config=None,
        device_map: str | None = None
):
    """
    Load a CLIP-L text encoder from a monolithic checkpoint for SD3/SD3.5 models.
    This function is used for both SD3 and SD3.5 variants (non-large).
    """
    try:
        with _suppress_accelerate_warnings():
            # Create a config manually for SD3/SD3.5 CLIP-L with exact parameters

            config = transformers.CLIPTextConfig.from_dict(
                _read_hub_config('models--stabilityai--stable-diffusion-3-medium-diffusers', 'text_encoder')
            )

            if quantization_config:
                config.quantization_config = quantization_config

            text_encoder = model_class(config)

        _messages.log(
            f'Loading monolithic CLIP-L SD3/SD3.5 state dict from: "{model_path}", this may take a while, please wait...',
            level=_messages.WARNING
        )

        device_map = _monolithic_auto_quant_device_map(quantization_config, device_map)

        if quantization_config:
            hf_quantizer = _monolithic_hf_quantizer(quantization_config)
            hf_quantizer.preprocess_model(
                text_encoder,
                device_map=device_map,
                keep_in_fp32_modules=_keep_in_fp32_module_names(text_encoder),
            )

        with _suppress_accelerate_warnings():
            # Load state dict and update weights
            text_encoder = _load_monolithic_checkpoint(
                text_encoder,
                checkpoint=model_path,
                device_map=device_map,
                dtype=dtype,
                no_split_module_classes=["CLIPEncoderLayer"]
            )

        if quantization_config:
            text_encoder = _finish_monolithic_quantization(text_encoder, hf_quantizer, dtype)

        return text_encoder.eval()

    except Exception as e:
        raise _exceptions.TextEncoderUriLoadError(
            f"Failed to load CLIP-L SD3/SD3.5 model: {e}"
        ) from e


def _load_clip_g_sd3_from_single_file(
        model_class: transformers.CLIPTextModel | transformers.CLIPTextModelWithProjection,
        model_path: str,
        dtype: torch.dtype,
        quantization_config=None,
        device_map: str | None = None
):
    """
    Load a CLIP-G text encoder from a monolithic checkpoint for SD3/SD3.5 models.
    This function is used for both SD3 and SD3.5 variants (non-large).
    """
    try:
        with _suppress_accelerate_warnings():
            # Create a config manually for SD3/SD3.5 CLIP-G with exact parameters

            config = transformers.CLIPTextConfig.from_dict(
                _read_hub_config('models--stabilityai--stable-diffusion-3-medium-diffusers', 'text_encoder_2')
            )

            if quantization_config:
                config.quantization_config = quantization_config

            text_encoder = model_class(config)

        _messages.log(
            f'Loading monolithic CLIP-G SD3/SD3.5 state dict from: "{model_path}", this may take a while, please wait...',
            level=_messages.WARNING
        )

        device_map = _monolithic_auto_quant_device_map(quantization_config, device_map)

        if quantization_config:
            hf_quantizer = _monolithic_hf_quantizer(quantization_config)
            hf_quantizer.preprocess_model(
                text_encoder,
                device_map=device_map,
                keep_in_fp32_modules=_keep_in_fp32_module_names(text_encoder),
            )

        with _suppress_accelerate_warnings():
            text_encoder = _load_monolithic_checkpoint(
                text_encoder,
                checkpoint=model_path,
                device_map=device_map,
                dtype=dtype,
                no_split_module_classes=["CLIPEncoderLayer"]
            )

        if quantization_config:
            text_encoder = _finish_monolithic_quantization(text_encoder, hf_quantizer, dtype)

        return text_encoder.eval()

    except Exception as e:
        raise _exceptions.TextEncoderUriLoadError(
            f"Failed to load CLIP-G SD3/SD3.5 model: {e}"
        ) from e


def _load_t5_xxl_sd3_from_single_file(
        model_class: transformers.models.t5.T5EncoderModel | _models.DistillT5EncoderModel,
        model_path: str,
        dtype: torch.dtype,
        quantization_config=None,
        device_map: str | None = None
):
    """
    Load a T5-XXL text encoder from a monolithic checkpoint specifically for SD3/SD3.5 models.
    """
    try:
        with _suppress_accelerate_warnings():
            # Create a config specifically for SD3/SD3.5 T5-XXL based on the exact SD3 configuration
            config = transformers.T5Config.from_dict(
                _read_hub_config('models--stabilityai--stable-diffusion-3-medium-diffusers', 'text_encoder_3')
            )

            if quantization_config:
                config.quantization_config = quantization_config

            text_encoder = model_class(config)

        # Load state dict and update weights
        _messages.log(
            f'Loading monolithic T5-XXL SD3/SD3.5 state dict from: '
            f'"{model_path}", this may take a while, please wait...',
            level=_messages.WARNING
        )

        device_map = _monolithic_auto_quant_device_map(quantization_config, device_map)

        if quantization_config:
            hf_quantizer = _monolithic_hf_quantizer(quantization_config)
            hf_quantizer.preprocess_model(
                text_encoder,
                device_map=device_map,
                keep_in_fp32_modules=_keep_in_fp32_module_names(text_encoder),
            )

        with _suppress_accelerate_warnings():
            text_encoder = _load_monolithic_checkpoint(
                text_encoder,
                checkpoint=model_path,
                device_map=device_map,
                dtype=dtype,
                no_split_module_classes=["T5Block"]
            )

        if quantization_config:
            text_encoder = _finish_monolithic_quantization(text_encoder, hf_quantizer, dtype)

        return text_encoder.eval()

    except Exception as e:
        raise _exceptions.TextEncoderUriLoadError(
            f"Failed to load T5-XXL SD3/SD3.5 model: {e}"
        ) from e


def _load_t5_xxl_from_single_file(
        model_class: transformers.models.t5.T5EncoderModel | _models.DistillT5EncoderModel,
        model_path: str,
        dtype: torch.dtype,
        quantization_config=None,
        device_map: str | None = None
):
    try:

        with _suppress_accelerate_warnings():
            config = transformers.T5Config.from_dict(
                _read_hub_config('models--black-forest-labs--FLUX.1-dev', 'text_encoder_2')
            )

            config.torch_dtype = dtype

            if quantization_config:
                config.quantization_config = quantization_config

            text_encoder = model_class(config)

        # Load state dict and update weights
        _messages.log(
            f'Loading monolithic t5-xxl state dict from: '
            f'"{model_path}", this may take a while, please wait...',
            level=_messages.WARNING
        )

        device_map = _monolithic_auto_quant_device_map(quantization_config, device_map)

        if quantization_config:
            hf_quantizer = _monolithic_hf_quantizer(quantization_config)
            hf_quantizer.preprocess_model(
                text_encoder,
                device_map=device_map,
                keep_in_fp32_modules=_keep_in_fp32_module_names(text_encoder),
            )

        with _suppress_accelerate_warnings():
            text_encoder = _load_monolithic_checkpoint(
                text_encoder,
                checkpoint=model_path,
                device_map=device_map,
                dtype=dtype,
                no_split_module_classes=["T5Block"]
            )

        if quantization_config:
            text_encoder = _finish_monolithic_quantization(text_encoder, hf_quantizer, dtype)

        return text_encoder.eval()

    except Exception as e:
        raise _exceptions.TextEncoderUriLoadError(
            f"Failed to load T5-XXL model: {e}"
        ) from e


def _load_clip_l_sd35_large_from_single_file(
        model_class: transformers.CLIPTextModel | transformers.CLIPTextModelWithProjection,
        model_path: str,
        dtype: torch.dtype,
        quantization_config=None,
        device_map: str | None = None
):
    try:
        with _suppress_accelerate_warnings():
            # Create a config manually for SD3.5 Large CLIP-L
            config = transformers.CLIPTextConfig.from_dict(
                _read_hub_config('models--stabilityai--stable-diffusion-3.5-large', 'text_encoder')
            )

            if quantization_config:
                config.quantization_config = quantization_config

            text_encoder = model_class(config)

        _messages.log(
            f'Loading monolithic clip-l-sd35-large state dict from: "{model_path}", this may take a while, please wait...',
            level=_messages.WARNING
        )

        device_map = _monolithic_auto_quant_device_map(quantization_config, device_map)

        if quantization_config:
            hf_quantizer = _monolithic_hf_quantizer(quantization_config)
            hf_quantizer.preprocess_model(
                text_encoder,
                device_map=device_map,
                keep_in_fp32_modules=_keep_in_fp32_module_names(text_encoder),
            )

        with _suppress_accelerate_warnings():
            # Load state dict and update weights
            text_encoder = _load_monolithic_checkpoint(
                text_encoder,
                checkpoint=model_path,
                device_map=device_map,
                dtype=dtype,
                no_split_module_classes=["CLIPEncoderLayer"]
            )

        if quantization_config:
            text_encoder = _finish_monolithic_quantization(text_encoder, hf_quantizer, dtype)

        return text_encoder.eval()

    except Exception as e:
        raise _exceptions.TextEncoderUriLoadError(
            f"Failed to load CLIP-L SD3.5 Large model: {e}"
        ) from e


def _load_clip_g_sd35_large_from_single_file(
        model_class: transformers.CLIPTextModel | transformers.CLIPTextModelWithProjection,
        model_path: str,
        dtype: torch.dtype,
        quantization_config=None,
        device_map: str | None = None
):
    try:
        with _suppress_accelerate_warnings():
            # Create a config manually for SD3.5 Large CLIP-G
            config = transformers.CLIPTextConfig.from_dict(
                _read_hub_config('models--stabilityai--stable-diffusion-3.5-large', 'text_encoder_2')
            )

            if quantization_config:
                config.quantization_config = quantization_config

            text_encoder = model_class(config)

        _messages.log(
            f'Loading monolithic clip-g-sd35-large state dict from: "{model_path}", this may take a while, please wait...',
            level=_messages.WARNING
        )

        device_map = _monolithic_auto_quant_device_map(quantization_config, device_map)

        if quantization_config:
            hf_quantizer = _monolithic_hf_quantizer(quantization_config)
            hf_quantizer.preprocess_model(
                text_encoder,
                device_map=device_map,
                keep_in_fp32_modules=_keep_in_fp32_module_names(text_encoder),
            )

        with _suppress_accelerate_warnings():
            # Load state dict and update weights
            text_encoder = _load_monolithic_checkpoint(
                text_encoder,
                checkpoint=model_path,
                device_map=device_map,
                dtype=dtype,
                no_split_module_classes=["CLIPEncoderLayer"]
            )

        if quantization_config:
            text_encoder = _finish_monolithic_quantization(text_encoder, hf_quantizer, dtype)

        return text_encoder.eval()

    except Exception as e:
        raise _exceptions.TextEncoderUriLoadError(
            f"Failed to load CLIP-G SD3.5 Large model: {e}"
        ) from e


class TextEncoderUri:
    """
    Representation of ``--text-encoders`` URI.
    """

    @property
    def encoder(self) -> str:
        """
        Encoder class name such as "CLIPTextModel"
        """
        return self._encoder

    @property
    def model(self) -> str:
        """
        Model path, huggingface slug
        """
        return self._model

    @property
    def revision(self) -> _types.OptionalString:
        """
        Model repo revision
        """
        return self._revision

    @property
    def variant(self) -> _types.OptionalString:
        """
        Model repo revision
        """
        return self._variant

    @property
    def subfolder(self) -> _types.OptionalPath:
        """
        Model repo subfolder
        """
        return self._subfolder

    @property
    def dtype(self) -> _enums.DataType | None:
        """
        Model dtype (precision)
        """
        return self._dtype

    @property
    def quantizer(self) -> _types.OptionalUri:
        """
        --quantizer URI override
        """
        return self._quantizer

    @property
    def mode(self) -> _types.OptionalString:
        """
        Model loading mode for single file checkpoints, for example 'clip-l', 'clip-g', or 't5-xxl'

        The default behavior is to extract the sub model from an 
        assumed to be combined checkpoint, which is not compatible
        with quantization.
        """
        return self._mode

    _encoders = {
        'CLIPTextModel': transformers.models.clip.CLIPTextModel,
        'CLIPTextModelWithProjection': transformers.models.clip.CLIPTextModelWithProjection,
        'T5EncoderModel': transformers.models.t5.T5EncoderModel,
        'UMT5EncoderModel': transformers.UMT5EncoderModel,
        'DistillT5EncoderModel': _models.DistillT5EncoderModel,
        'ChatGLMModel': diffusers.pipelines.kolors.ChatGLMModel,
        'Gemma4UnifiedForConditionalGeneration': transformers.Gemma4UnifiedForConditionalGeneration,
        'Mistral3ForConditionalGeneration': transformers.Mistral3ForConditionalGeneration,
        'Qwen2_5_VLForConditionalGeneration': transformers.Qwen2_5_VLForConditionalGeneration,
        'Qwen3ForCausalLM': transformers.Qwen3ForCausalLM,
        'Qwen3Model': transformers.Qwen3Model,
    }

    _clip_modes = (
        'clip-l',
        'clip-l-sd3',
        'clip-g-sd3',
        'clip-l-sd35-large',
        'clip-g-sd35-large'
    )

    _t5_modes = (
        't5-xxl',
        't5-xxl-sd3',
    )

    @staticmethod
    def _valid_modes():
        return TextEncoderUri._clip_modes + TextEncoderUri._t5_modes

    @staticmethod
    def supported_encoder_names() -> list[str]:
        return list(TextEncoderUri._encoders.keys())

    @staticmethod
    def _resolve_encoder_class(name: str):
        """
        Resolve a text encoder class.

        Named encoders stay in ``_encoders``. Z-Image does not name its language
        model on the pipeline, so a ``transformers.PreTrainedModel`` class from
        ``model_index.json`` is accepted as well.
        """
        known = TextEncoderUri._encoders.get(name)
        if known is not None:
            return known
        candidate = getattr(transformers, name, None)
        if isinstance(candidate, type) and issubclass(candidate, transformers.PreTrainedModel):
            return candidate
        return None

    # pipelinewrapper.uris.util.get_uri_accepted_args_schema metadata

    NAMES = ['Text Encoder']

    @staticmethod
    def help():
        import dgenerate.arguments as _a
        return _a.get_raw_help_text('--text-encoders')

    OPTION_ARGS = {
        'encoder': list(_encoders.keys()),
        'mode': _clip_modes + _t5_modes,
        'dtype': ['auto', 'float16', 'bfloat16', 'float32']
    }

    FILE_ARGS = {
        'model': {'mode': ['in', 'dir'], 'filetypes': [('Models', ['*.safetensors', '*.pt', '*.pth', '*.cpkt', '*.bin'])]}
    }

    # ===

    def __init__(self,
                 encoder: str,
                 model: str,
                 revision: _types.OptionalString = None,
                 variant: _types.OptionalString = None,
                 subfolder: _types.OptionalString = None,
                 dtype: _enums.DataType | str | None = None,
                 quantizer: _types.OptionalUri = None,
                 mode: _types.OptionalString = None):
        """
        :param encoder: encoder class name, for example ``CLIPTextModel``
        :param model: model path
        :param revision: model revision (branch name)
        :param variant: model variant, for example ``fp16``
        :param subfolder: model subfolder
        :param dtype: model data type (precision)
        :param mode: model loading mode, for example ``clip-l`` for single file ''clip-l'' checkpoints.

        :raises InvalidTextEncoderUriError: If ``dtype`` is passed an invalid data type string, or if
            ``model`` points to a single file and the specified ``encoder`` class name does not
            support loading from a single file.
        """

        if TextEncoderUri._resolve_encoder_class(encoder) is None:
            raise _exceptions.InvalidTextEncoderUriError(
                f'Unknown TextEncoder encoder class {encoder}, must be one of: {_textprocessing.oxford_comma(self._encoders.keys(), "or")}, '
                f'or a transformers.PreTrainedModel class name from model_index.json.')

        mode = mode.lower() if mode is not None else mode

        valid_modes = TextEncoderUri._valid_modes()

        if _hfhub.is_single_file_model_load(model):
            if quantizer and mode not in valid_modes:
                raise _exceptions.InvalidTextEncoderUriError(
                    'specifying a Text Encoder quantizer URI is only supported for Hugging Face '
                    'repository loads from a repo slug or disk path, single file loads are not supported.')

        if mode is not None:

            if mode not in valid_modes:
                raise _exceptions.InvalidTextEncoderUriError(
                    f'Unknown TextEncoder load mode "{mode}", must be one of: {_textprocessing.oxford_comma(valid_modes, "or")}')

            # Validate special modes don't use variant or revision
            if mode in valid_modes:
                if variant is not None:
                    raise _exceptions.InvalidTextEncoderUriError(
                        f'TextEncoder cannot use variant with mode "{mode}", these are incompatible options')
                if revision is not None:
                    raise _exceptions.InvalidTextEncoderUriError(
                        f'TextEncoder cannot use revision with mode "{mode}", these are incompatible options')
                if subfolder is not None:
                    raise _exceptions.InvalidTextEncoderUriError(
                        f'TextEncoder cannot use subfolder with mode "{mode}", these are incompatible options')

        self._encoder = encoder
        self._model = model
        self._revision = revision
        self._variant = variant
        self._subfolder = subfolder
        self._quantizer = quantizer
        self._mode = mode

        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise _exceptions.InvalidTextEncoderUriError(
                f'invalid dtype string, must be one of: {_textprocessing.oxford_comma(_enums.supported_data_type_strings(), "or")}')

    def load(self,
             variant_fallback: _types.OptionalString = None,
             dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
             original_config: _types.OptionalPath = None,
             auth_token: _types.OptionalString = None,
             local_files_only: bool = False,
             no_cache: bool = False,
             missing_ok: bool = False,
             device_map: str | None = None
             ) -> \
            typing.Union[
                transformers.models.clip.CLIPTextModel,
                transformers.models.clip.CLIPTextModelWithProjection,
                transformers.models.t5.T5EncoderModel,
                _models.DistillT5EncoderModel,
                diffusers.pipelines.kolors.ChatGLMModel, None]:
        """
        Load a torch Text Encoder of type :py:class:`transformers.models.clip.CLIPTextModel`,
        :py:class:`transformers.models.clip.CLIPTextModelWithProjection`,
        :py:class:`transformers.models.t5.T5EncoderModel`, or
        :py:class:`diffusers.pipelines.kolors.ChatGLMModel` from this URI

        :param variant_fallback: If the URI does not specify a variant, use this variant.
        :param dtype_fallback: If the URI does not specify a dtype, use this dtype.
        :param original_config: Path to original model configuration for single file checkpoints, URL or `.yaml` file on disk.
        :param auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug or blob link
        :param no_cache: If ``True``, force the returned object not to be cached by the memoize decorator.
        :param missing_ok: If ``True``, when a VAE is not found inside a single file checkpoint as a sub model,
            just return ``None`` instead of throwing an error.
        :param device_map: device placement strategy for quantized models, defaults to ``None``

        :raises ModelNotFoundError: If the model could not be found.

        :return: :py:class:`transformers.models.clip.CLIPTextModel`,
            :py:class:`transformers.models.clip.CLIPTextModelWithProjection`,
            :py:class:`transformers.models.t5.T5EncoderModel`, or
            :py:class:`diffusers.pipelines.kolors.ChatGLMModel`
        """
        def cache_all(e):
            raise _exceptions.TextEncoderUriLoadError(
                f'error loading text encoder "{self.model}": {e}') from e

        with _hfhub.with_hf_errors_as_model_not_found(cache_all):
            args = locals()
            args.pop('self')
            args.pop('cache_all')
            loaded = self._load(**args)

            def reload():
                fresh = dict(args)
                fresh['no_cache'] = True
                return self._load(**fresh)

            return _util.reuse_lora_free_module(loaded, reload)

    @staticmethod
    def _enforce_cache_size(new_text_encoder_size):
        _text_encoder_cache.enforce_cpu_mem_constraints(
            _constants.TEXT_ENCODER_CACHE_MEMORY_CONSTRAINTS,
            size_var='text_encoder_size',
            new_object_size=new_text_encoder_size)

    # noinspection PyTypeChecker
    @_memoize(_text_encoder_cache,
              exceptions={'local_files_only', 'missing_ok'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _d_memoize.property_hasher}),
              on_hit=lambda key, hit: _d_memoize.simple_cache_hit_debug("Torch TextEncoder", key, hit),
              on_create=lambda key, new: _d_memoize.simple_cache_miss_debug("Torch TextEncoder", key, new))
    def _load(self,
              variant_fallback: _types.OptionalString = None,
              dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
              original_config: _types.OptionalPath = None,
              auth_token: _types.OptionalString = None,
              local_files_only: bool = False,
              no_cache: bool = False,
              missing_ok: bool = False,
              device_map: str | None = None
              ) -> \
            typing.Union[
                transformers.models.clip.CLIPTextModel,
                transformers.models.clip.CLIPTextModelWithProjection,
                transformers.models.t5.T5EncoderModel,
                _models.DistillT5EncoderModel,
                diffusers.pipelines.kolors.ChatGLMModel, None]:

        if self.dtype is None:
            torch_dtype = _enums.get_torch_dtype(dtype_fallback)
        else:
            torch_dtype = _enums.get_torch_dtype(self.dtype)

        if self.variant is None:
            variant = variant_fallback
        elif self.variant == 'null':
            variant = None
        else:
            variant = self.variant

        if self.encoder == 'ChatGLMModel':
            encoder_library = 'diffusers.pipelines.kolors'
        else:
            encoder_library = 'transformers'

        encoder = self._resolve_encoder_class(self.encoder)
        if encoder is None:
            raise _exceptions.TextEncoderUriLoadError(
                f'Unknown TextEncoder encoder class {self.encoder}.')

        # Validate mode and encoder class compatibility
        clip_encoders = (transformers.CLIPTextModel, transformers.CLIPTextModelWithProjection)
        t5_encoders = (transformers.T5EncoderModel, _models.DistillT5EncoderModel)

        if self.mode in TextEncoderUri._clip_modes and encoder not in clip_encoders:
            raise _exceptions.TextEncoderUriLoadError(
                f'Encoder "{self.encoder}" does not support loading with mode "{self.mode}".')

        if self.mode in TextEncoderUri._t5_modes and encoder not in t5_encoders:
            raise _exceptions.TextEncoderUriLoadError(
                f'Encoder "{self.encoder}" does not support loading with mode "{self.mode}".')

        model_path = _hfhub.download_non_hf_slug_model(self.model, local_files_only=local_files_only, auth_token=auth_token)

        if self.quantizer:
            quant_uri = _util.get_quantizer_uri_class(
                self.quantizer,
                _exceptions.InvalidTextEncoderUriError
            ).parse(self.quantizer)
        else:
            quant_uri = None

        if _hfhub.is_single_file_model_load(model_path):
            # single file loads quantize with the diffusers auto quantizer
            quant_config = quant_uri.to_config(torch_dtype) if quant_uri else None

            # Ensure these modes are only used with safetensors files

            if self.mode in TextEncoderUri._valid_modes():
                if not model_path.endswith('.safetensors'):
                    raise _exceptions.TextEncoderUriLoadError(
                        f'TextEncoder mode "{self.mode}" only supports loading '
                        f'from safetensors files, but got: "{model_path}"')

            if self.mode == 'clip-l':
                # Estimate memory directly based on model size
                # CLIP-L: ~123M parameters

                estimated_memory_use = 123_000_000 * 4
                if torch_dtype == torch.float16:
                    estimated_memory_use = estimated_memory_use // 2

                _messages.debug_log(
                    f'Estimated CLIP-L Memory Use: {_memory.bytes_best_human_unit(estimated_memory_use)}')
                self._enforce_cache_size(estimated_memory_use)

                text_encoder = _load_clip_l_from_single_file(
                    model_class=encoder,
                    model_path=model_path,
                    dtype=torch_dtype,
                    quantization_config=quant_config,
                    device_map=device_map
                )

                estimated_memory_use = _torchutil.estimate_module_memory_usage(text_encoder)
            elif self.mode == 'clip-l-sd3':
                # Estimate memory directly based on model size
                # CLIP-L for SD3: ~123M parameters (same architecture as CLIP-L)

                estimated_memory_use = 123_000_000 * 4
                if torch_dtype == torch.float16:
                    estimated_memory_use = estimated_memory_use // 2

                _messages.debug_log(
                    f'Estimated CLIP-L SD3 Memory Use: {_memory.bytes_best_human_unit(estimated_memory_use)}')
                self._enforce_cache_size(estimated_memory_use)

                text_encoder = _load_clip_l_sd3_from_single_file(
                    model_class=encoder,
                    model_path=model_path,
                    dtype=torch_dtype,
                    quantization_config=quant_config,
                    device_map=device_map
                )

                estimated_memory_use = _torchutil.estimate_module_memory_usage(text_encoder)
            elif self.mode == 'clip-l-sd35-large':
                # Estimate memory directly based on model size
                # CLIP-L for SD3.5 Large: ~320M parameters (larger architecture)

                estimated_memory_use = 320_000_000 * 4
                if torch_dtype == torch.float16:
                    estimated_memory_use = estimated_memory_use // 2

                _messages.debug_log(
                    f'Estimated CLIP-L SD3.5 Large Memory Use: {_memory.bytes_best_human_unit(estimated_memory_use)}')

                self._enforce_cache_size(estimated_memory_use)

                text_encoder = _load_clip_l_sd35_large_from_single_file(
                    model_class=encoder,
                    model_path=model_path,
                    dtype=torch_dtype,
                    quantization_config=quant_config,
                    device_map=device_map
                )

                estimated_memory_use = _torchutil.estimate_module_memory_usage(text_encoder)
            elif self.mode == 'clip-g-sd3':
                # Estimate memory directly based on model size
                # CLIP-G for SD3: ~1.2B parameters

                estimated_memory_use = 1_200_000_000 * 4
                if torch_dtype == torch.float16:
                    estimated_memory_use = estimated_memory_use // 2

                _messages.debug_log(
                    f'Estimated CLIP-G SD3 Memory Use: {_memory.bytes_best_human_unit(estimated_memory_use)}')
                self._enforce_cache_size(estimated_memory_use)

                text_encoder = _load_clip_g_sd3_from_single_file(
                    model_class=encoder,
                    model_path=model_path,
                    dtype=torch_dtype,
                    quantization_config=quant_config,
                    device_map=device_map
                )

                estimated_memory_use = _torchutil.estimate_module_memory_usage(text_encoder)
            elif self.mode == 'clip-g-sd35-large':
                # Estimate memory directly based on model size
                # CLIP-G for SD3.5 Large: ~1.2B parameters (similar to CLIP-G)

                estimated_memory_use = 1_200_000_000 * 4
                if torch_dtype == torch.float16:
                    estimated_memory_use = estimated_memory_use // 2

                _messages.debug_log(
                    f'Estimated CLIP-G SD3.5 Large Memory Use: {_memory.bytes_best_human_unit(estimated_memory_use)}')
                self._enforce_cache_size(estimated_memory_use)

                text_encoder = _load_clip_g_sd35_large_from_single_file(
                    model_class=encoder,
                    model_path=model_path,
                    dtype=torch_dtype,
                    quantization_config=quant_config,
                    device_map=device_map
                )

                estimated_memory_use = _torchutil.estimate_module_memory_usage(text_encoder)
            elif self.mode == 't5-xxl':
                # Estimate memory directly based on model size
                # T5-XXL: ~4.6B parameters

                estimated_memory_use = 4_600_000_000 * 4
                if torch_dtype == torch.float16:
                    estimated_memory_use = estimated_memory_use // 2

                _messages.debug_log(
                    f'Estimated T5-XXL Memory Use: {_memory.bytes_best_human_unit(estimated_memory_use)}')

                self._enforce_cache_size(estimated_memory_use)

                text_encoder = _load_t5_xxl_from_single_file(
                    model_class=encoder,
                    model_path=model_path,
                    dtype=torch_dtype,
                    quantization_config=quant_config,
                    device_map=device_map
                )

                estimated_memory_use = _torchutil.estimate_module_memory_usage(text_encoder)
            elif self.mode == 't5-xxl-sd3':
                # Estimate memory directly based on model size
                # T5-XXL: ~4.6B parameters

                estimated_memory_use = 4_600_000_000 * 4

                if torch_dtype == torch.float16:
                    estimated_memory_use = estimated_memory_use // 2

                _messages.debug_log(
                    f'Estimated T5-XXL SD3 Memory Use: {_memory.bytes_best_human_unit(estimated_memory_use)}')
                self._enforce_cache_size(estimated_memory_use)

                text_encoder = _load_t5_xxl_sd3_from_single_file(
                    model_class=encoder,
                    model_path=model_path,
                    dtype=torch_dtype,
                    quantization_config=quant_config,
                    device_map=device_map
                )

                estimated_memory_use = _torchutil.estimate_module_memory_usage(text_encoder)
            elif (encoder_library == 'transformers' and not self.mode
                  and (language_model_config := _language_model_config(model_path, encoder))):
                # Language model encoders (Qwen3, Mistral 3, ...) as single files,
                # which Diffusers' single file loader does not handle.
                if quant_config is not None:
                    raise _exceptions.TextEncoderUriLoadError(
                        'A quantizer is not supported for single file language model '
                        'text encoders. Use a pre-quantized file, or load the encoder '
                        'from a Hugging Face repository.')
                self._enforce_cache_size(os.path.getsize(model_path))
                text_encoder = _load_language_model_from_single_file(
                    encoder, model_path, language_model_config, torch_dtype)
                estimated_memory_use = _torchutil.estimate_module_memory_usage(text_encoder)
            else:
                try:
                    original_config = _hfhub.download_non_hf_slug_config(
                        original_config, local_files_only=local_files_only, auth_token=auth_token) if original_config else None
                except _hfhub.NonHFConfigDownloadError as e:
                    raise _exceptions.TextEncoderUriLoadError(
                        f'original config file "{original_config}" for Text Encoder could not be downloaded: {e}'
                    ) from e

                estimated_memory_use = _pipelinewrapper_util.estimate_model_memory_use(
                    repo_id=model_path,
                    revision=self.revision,
                    local_files_only=local_files_only,
                    auth_token=auth_token
                )

                self._enforce_cache_size(estimated_memory_use)

                try:
                    text_encoder = _pipelinewrapper_util.single_file_load_sub_module(
                        path=model_path,
                        class_name=self.encoder,
                        library_name=encoder_library,
                        name=self.subfolder if self.subfolder else 'text_encoder',
                        auth_token=auth_token,
                        original_config=original_config,
                        local_files_only=local_files_only,
                        revision=self.revision,
                        dtype=torch_dtype
                    )
                except FileNotFoundError as e:
                    # cannot find configs
                    raise _d_exceptions.ModelNotFoundError(e) from e
                except diffusers.loaders.single_file.SingleFileComponentError as e:
                    if missing_ok:
                        # noinspection PyTypeChecker
                        return None, _d_memoize.CachedObjectMetadata(
                            size=0,
                            skip=True
                        )
                    raise _exceptions.TextEncoderUriLoadError(
                        f'Failed to load Text Encoder from single file checkpoint {model_path}, '
                        f'make sure the file contains a Text Encoders.') from e

                estimated_memory_use = _torchutil.estimate_module_memory_usage(text_encoder)

        else:
            if original_config:
                raise _exceptions.TextEncoderUriLoadError(
                    'specifying original_config file for Text Encoder '
                    'is only supported for single file loads.')

            estimated_memory_use = _pipelinewrapper_util.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                variant=variant,
                subfolder=self.subfolder,
                local_files_only=local_files_only,
                auth_token=auth_token
            )

            self._enforce_cache_size(estimated_memory_use)

            prequantized = _sdnqload.sdnq_config_for_component(
                model_path,
                subfolder=self.subfolder,
                revision=self.revision,
                token=auth_token,
                local_files_only=local_files_only)
            requantize_error = _sdnqload.sdnq_requantize_error(self.quantizer, prequantized)
            if requantize_error:
                raise _exceptions.TextEncoderUriLoadError(requantize_error)

            load_kwargs = dict(
                revision=self.revision,
                variant=variant,
                dtype=torch_dtype,
                subfolder=self.subfolder if self.subfolder else "",
                token=auth_token,
                local_files_only=local_files_only,
                device_map=device_map)
            if prequantized is None and quant_uri:
                load_kwargs['quantization_config'] = quant_uri.to_transformers_config(torch_dtype)

            text_encoder = encoder.from_pretrained(model_path, **load_kwargs)

        _messages.debug_log('Estimated Torch TextEncoder Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_use))

        _util._patch_module_to_for_sized_cache(_text_encoder_cache, text_encoder)

        # noinspection PyTypeChecker
        return text_encoder, _d_memoize.CachedObjectMetadata(
            size=estimated_memory_use,
            skip=self.quantizer or no_cache
        )

    @staticmethod
    def parse(uri: _types.Uri) -> 'TextEncoderUri':
        """
        Parse a ``--text-encoders*`` uri and return an object representing its constituents

        :param uri: string with ``--text-encoders*`` uri syntax

        :raise InvalidTextEncoderUriError:

        :return: :py:class:`.TorchTextEncoderUri`
        """
        try:
            r = _text_encoder_uri_parser.parse(uri)

            model = r.args.get('model')
            if model is None:
                raise _exceptions.InvalidTextEncoderUriError(
                    'model argument for torch TextEncoder specification must be defined.')

            dtype = r.args.get('dtype')

            supported_dtypes = _enums.supported_data_type_strings()
            if dtype is not None and dtype not in supported_dtypes:
                raise _exceptions.InvalidTextEncoderUriError(
                    f'Torch TextEncoder "dtype" must be {", ".join(supported_dtypes)}, '
                    f'or left undefined, received: {dtype}')

            return TextEncoderUri(encoder=r.concept,
                                  model=model,
                                  revision=r.args.get('revision', None),
                                  variant=r.args.get('variant', None),
                                  dtype=dtype,
                                  subfolder=r.args.get('subfolder', None),
                                  quantizer=r.args.get('quantizer', False),
                                  mode=r.args.get('mode', None)
                                  )
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidTextEncoderUriError(e) from e
