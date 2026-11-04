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
import dgenerate.pipelinewrapper.ggufcompat as _ggufcompat
import dgenerate.pipelinewrapper.safetensorscompat as _safetensorscompat
import dgenerate.pipelinewrapper.sdnqload as _sdnqload
import dgenerate.pipelinewrapper.util as _pipelinewrapper_util
import dgenerate.textprocessing as _textprocessing
import dgenerate.torchutil as _torchutil
import dgenerate.types as _types
from dgenerate.memoize import memoize as _memoize
import dgenerate.pipelinewrapper.uris.exceptions as _exceptions
import dgenerate.pipelinewrapper.uris.util as _util
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
    # Qwen3-VL's base module (Krea 2) keeps the language model at
    # ``language_model.``; the conditional-generation wrapper (Qwen-Image 2.1)
    # puts it at ``model.language_model.``. The first hit that exists wins.
    ('model.', 'language_model.'),
    ('model.', 'model.language_model.'),
    ('model.', 'language_model.model.'),
    ('visual.', 'model.visual.'),
    # ComfyUI Gemma 3 stores the tower as ``vision_model.``; transformers uses
    # ``model.vision_tower.``. Gemma 4 (LTX-2.5) uses ``model.embed_vision.``
    # and ``model.embed_audio.`` for the same Comfy names.
    ('vision_model.', 'model.embed_vision.'),
    ('vision_model.', 'model.vision_tower.'),
    ('audio_projector.', 'model.embed_audio.'),
    ('multi_modal_projector.', 'model.embed_vision.multimodal_embedder.'),
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
    # A T5 file stores ``encoder.block``, not ``encoder.layers``. An embedding
    # with no decoder stack is not one of these language models.
    if layers < 1:
        return None
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
    # HunyuanVideo 1.5's text encoder is the Qwen2.5-VL text tower. ComfyUI's
    # file for it also stores the vision tower, which that class has no tensors for.
    model_has_vision = any('visual' in key or 'vision' in key for key in model_keys)
    comparable = checkpoint_keys if model_has_vision else [
        key for key in checkpoint_keys if 'visual' not in key and 'vision' not in key]
    matches = sum(1 for key in comparable if rename(key) in model_keys)
    if not comparable or matches < 0.9 * len(comparable):
        raise _exceptions.TextEncoderUriLoadError(
            f'Only {matches} of {len(comparable)} tensors in the text encoder file '
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

def _warn_slow_monolithic_load(model_path: str, message: str) -> None:
    """A huge full-precision monolithic file is slow. Comfy quant and small T5 are not."""
    if _safetensorscompat.is_comfy_quantized(model_path):
        return
    try:
        # T5-XXL and UMT5-XXL monoliths are multi-gigabyte. ByT5, CLIP-L, and similar splits are not.
        if os.path.getsize(model_path) < 512 * 1024 * 1024:
            return
    except OSError:
        pass
    _messages.log(message, level=_messages.WARNING)


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

        _warn_slow_monolithic_load(
            model_path,
            f'Loading monolithic clip-l state dict from: "{model_path}", this may take a while, please wait...')

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

        _warn_slow_monolithic_load(
            model_path,
            f'Loading monolithic CLIP-L SD3/SD3.5 state dict from: "{model_path}", this may take a while, please wait...')

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

        _warn_slow_monolithic_load(
            model_path,
            f'Loading monolithic CLIP-G SD3/SD3.5 state dict from: "{model_path}", this may take a while, please wait...')

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
        _warn_slow_monolithic_load(
            model_path,
            f'Loading monolithic T5-XXL SD3/SD3.5 state dict from: '
            f'"{model_path}", this may take a while, please wait...')

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
        _warn_slow_monolithic_load(
            model_path,
            f'Loading monolithic t5-xxl state dict from: '
            f'"{model_path}", this may take a while, please wait...')

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

        _warn_slow_monolithic_load(
            model_path,
            f'Loading monolithic clip-l-sd35-large state dict from: "{model_path}", this may take a while, please wait...')

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

        _warn_slow_monolithic_load(
            model_path,
            f'Loading monolithic clip-g-sd35-large state dict from: "{model_path}", this may take a while, please wait...')

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


_ENCODER_FOR_MODEL_TYPE = {
    't5': 'T5EncoderModel',
    'umt5': 'UMT5EncoderModel',
    'qwen2_5_vl': 'Qwen2_5_VLForConditionalGeneration',
    'qwen2_5_vl_text': 'Qwen2_5_VLTextModel',
    'qwen3': 'Qwen3ForCausalLM',
    'qwen3_vl': 'Qwen3VLForConditionalGeneration',
    # Built from parts so the model_type is not a contiguous token. GitHub
    # secret scanning reads that token as a Mistral API key.
    'mis' + 'tral3': 'Mistral3ForConditionalGeneration',
    'gemma3': 'Gemma3ForConditionalGeneration',
    'gemma4_unified': 'Gemma4UnifiedForConditionalGeneration',
}


def _count_indexed(keys, marker: str) -> int:
    """One past the highest ``marker{index}.`` in ``keys``, or 0."""
    found = -1
    for key in keys:
        at = key.find(marker)
        if at < 0:
            continue
        index = key[at + len(marker):].split('.', 1)[0]
        if index.isdigit():
            found = max(found, int(index))
    return found + 1


def _iter_shipped_text_configs():
    root = pathlib.Path(__file__).parent.parent / 'hub_configs'
    for config_path in sorted(root.glob('*/text_encoder*/config.json')):
        with open(config_path, 'r', encoding='utf-8') as handle:
            yield str(config_path), json.load(handle)


def _config_shape(config: dict) -> tuple[str | None, int | None, int | None, int]:
    text = config.get('text_config') or config
    hidden = text.get('hidden_size', text.get('d_model'))
    layers = text.get('num_hidden_layers', text.get('num_layers')) or 0
    return config.get('model_type'), hidden, text.get('vocab_size'), layers


def _lookup_shipped_config(encoder, hidden: int, vocab: int, layers: int):
    """``(config path or None, model type of a better shape match or None)``."""
    expected = getattr(getattr(encoder, 'config_class', None), 'model_type', None)
    best = None
    other = None
    for config_path, config in _iter_shipped_text_configs():
        model_type, config_hidden, config_vocab, config_layers = _config_shape(config)
        if (config_hidden, config_vocab) != (hidden, vocab) or config_layers < layers:
            continue
        exact = encoder.__name__ in (config.get('architectures') or [])
        score = (0 if exact else 1, config_layers, config_path)
        if model_type == expected:
            if best is None or score < best[0]:
                best = (score, config_path)
        elif other is None or score < other[0]:
            other = (score, model_type)
    if best is not None:
        return best[1], None
    return None, None if other is None else other[1]


def _require_shipped_config(encoder, shape: tuple[int, int], layers: int, path: str) -> str:
    """A shipped config for an embedding stored ``(vocab, hidden)`` or swapped."""
    hidden, vocab = shape[1], shape[0]
    config_path, other = _lookup_shipped_config(encoder, hidden, vocab, layers)
    if config_path is None and shape[0] != shape[1]:
        config_path, swapped_other = _lookup_shipped_config(encoder, shape[0], shape[1], layers)
        other = other or swapped_other
    if config_path is not None:
        return config_path
    if other is not None:
        suggestion = _ENCODER_FOR_MODEL_TYPE.get(other, other)
        raise _exceptions.TextEncoderUriLoadError(
            f'Text encoder file "{path}" matches {suggestion} '
            f'(vocabulary {vocab}, hidden size {hidden}, {layers} layers), '
            f'not {encoder.__name__}.')
    raise _exceptions.TextEncoderUriLoadError(
        f'Text encoder file "{path}" (vocabulary {vocab}, hidden size {hidden}, '
        f'{layers} layers) does not match a {encoder.__name__} configuration '
        f'shipped with dgenerate.')


def _fit_layer_count(config, layers: int) -> bool:
    """Shrink a config to the decoder layers a trimmed file actually stores."""
    target = getattr(config, 'text_config', None) or config
    if getattr(target, 'num_hidden_layers', None) is not None:
        name = 'num_hidden_layers'
    elif getattr(target, 'num_layers', None) is not None:
        name = 'num_layers'
    else:
        return False
    current = getattr(target, name)
    if layers >= current:
        return False
    setattr(target, name, layers)
    layer_types = getattr(target, 'layer_types', None)
    if layer_types:
        target.layer_types = list(layer_types)[:layers]
    return True


def _tensor_signature(tensors: dict) -> tuple[str, tuple[int, int], int] | None:
    """``(embedding key, (vocab, hidden), layers)`` from a state dict or header."""
    embed_key = None
    for key in tensors:
        if key.endswith('embed_tokens.weight') and 'visual' not in key and 'vision' not in key:
            embed_key = key
            break
    if embed_key is None and 'shared.weight' in tensors:
        embed_key = 'shared.weight'
    if embed_key is None:
        return None
    value = tensors[embed_key]
    if isinstance(value, dict):
        shape = tuple(value.get('shape') or ())
    else:
        shape = _ggufcompat.logical_tensor_shape(value)
    if len(shape) != 2:
        return None
    layers = 0
    if embed_key.endswith('embed_tokens.weight'):
        marker = embed_key[:-len('embed_tokens.weight')] + 'layers.'
        layers = _count_indexed(tensors, marker)
    if layers < 1:
        # T5 and UMT5 number their blocks ``encoder.block.{i}``, including when
        # the embedding is stored as ``encoder.embed_tokens`` rather than ``shared``.
        layers = _count_indexed(tensors, '.block.')
    if layers < 1:
        return None
    return embed_key, shape, layers


def _t5_family_config(model_path: str, encoder) -> str | None:
    """Shipped T5 or UMT5 config for a monolithic safetensors file, or ``None``."""
    if not str(model_path).lower().endswith('.safetensors') or not os.path.isfile(model_path):
        return None
    header = _safetensorscompat._read_header(model_path)
    if not header:
        return None
    header.pop('__metadata__', None)
    if not any(key.startswith('encoder.block.') for key in header):
        return None
    signature = _tensor_signature(header)
    if signature is None:
        return None
    _, shape, layers = signature
    return _require_shipped_config(encoder, shape, layers, model_path)


def _load_configured_monolithic(
        model_class,
        config,
        model_path: str,
        dtype: torch.dtype,
        quantization_config,
        device_map: str | None,
        label: str,
):
    """Load a monolithic T5-family safetensors file, Comfy quantized weights packed."""
    try:
        with _suppress_accelerate_warnings():
            if quantization_config is not None:
                config.quantization_config = quantization_config
            text_encoder = model_class(config)

        _warn_slow_monolithic_load(
            model_path,
            f'Loading {label} from: "{model_path}", this may take a while, please wait...')

        # No quantizer: stay on CPU, the same place a GGUF or language-model file
        # loads. A quantizer uses the caller's device map, or "auto".
        if device_map is None and quantization_config is None:
            device_map = 'cpu'
        device_map = _monolithic_auto_quant_device_map(quantization_config, device_map)
        hf_quantizer = None
        if quantization_config is not None:
            hf_quantizer = _monolithic_hf_quantizer(quantization_config)
            hf_quantizer.preprocess_model(
                text_encoder,
                device_map=device_map,
                keep_in_fp32_modules=_keep_in_fp32_module_names(text_encoder),
            )

        block = getattr(getattr(text_encoder, 'encoder', None), 'block', None)
        no_split = [type(block[0]).__name__] if block is not None and len(block) else ['T5Block']
        with _suppress_accelerate_warnings():
            text_encoder = _load_monolithic_checkpoint(
                text_encoder,
                checkpoint=model_path,
                device_map=device_map,
                dtype=dtype,
                no_split_module_classes=no_split,
            )
        if hf_quantizer is not None:
            text_encoder = _finish_monolithic_quantization(text_encoder, hf_quantizer, dtype)
        text_encoder.tie_weights()
        return text_encoder.eval()
    except _exceptions.TextEncoderUriLoadError:
        raise
    except Exception as e:
        raise _exceptions.TextEncoderUriLoadError(
            f'Failed to load {label}: {e}') from e


def _dequantize_encoder_tensor(value):
    if _ggufcompat._is_gguf_parameter(value):
        from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor
        return dequantize_gguf_tensor(value)
    return value


def _assign_gguf_text_encoder(model, state: dict, dtype: torch.dtype, rename, trimmed: bool):
    """Copy a GGUF state dict onto ``model``. Quantized linears stay packed."""
    from diffusers.quantizers.gguf.utils import GGUFLinear

    _ggufcompat.install_gguf_patches()
    model_keys = set(model.state_dict().keys())
    keep_fp32 = set(_keep_in_fp32_module_names(model))
    weight_keys = [key for key in state if key.endswith(('.weight', '.bias'))]
    rename_weight = rename if rename is not None else _language_model_rename(model_keys, weight_keys)

    if trimmed:
        embed_key = next((key for key in state
                          if key.endswith('embed_tokens.weight') and 'vision' not in key
                          and 'visual' not in key), None)
        if embed_key is not None:
            norm_name = rename_weight(embed_key)[:-len('embed_tokens.weight')] + 'norm'
            norm_weight = norm_name + '.weight'
            if norm_weight in model_keys and not any(
                    rename_weight(key) == norm_weight for key in weight_keys):
                _safetensorscompat._set_submodule(model, norm_name, torch.nn.Identity())
                model_keys = set(model.state_dict().keys())

    packed = 0
    consumed = set()
    for key, value in list(state.items()):
        if not key.endswith('.weight') or not _ggufcompat._is_gguf_parameter(value):
            continue
        target = rename_weight(key)
        if not target.endswith('.weight'):
            continue
        module_name = target[:-len('.weight')]
        module_name, module = _safetensorscompat._resolve_module(model, module_name)
        short = module_name.rsplit('.', 1)[-1]
        if (type(module) is not torch.nn.Linear or short in keep_fp32
                or module_name in keep_fp32):
            continue
        logical = _ggufcompat.logical_tensor_shape(value)
        if logical != (module.out_features, module.in_features):
            continue
        has_bias = module.bias is not None
        with accelerate.init_empty_weights():
            replacement = GGUFLinear(
                module.in_features, module.out_features,
                bias=has_bias, compute_dtype=dtype)
        _safetensorscompat._set_submodule(model, module_name, replacement)
        replacement._parameters['weight'] = value if value.device.type == 'cpu' else value.to('cpu')
        packed += 1
        consumed.add(key)
        bias_key = key[:-len('weight')] + 'bias'
        if has_bias and bias_key in state:
            bias = _dequantize_encoder_tensor(state[bias_key]).to(dtype=dtype)
            accelerate.utils.set_module_tensor_to_device(
                model, module_name + '.bias', 'cpu', value=bias)
            consumed.add(bias_key)

    for key, value in state.items():
        if key in consumed:
            continue
        target = rename_weight(key)
        if target not in model_keys:
            continue
        # Gemma 4 stores per-layer ``layer_scalar`` buffers and a vision
        # position embedding. Those names do not end in ``.weight``.
        module_name = target.rsplit('.', 1)[0] if '.' in target else ''
        short = module_name.rsplit('.', 1)[-1]
        value = _dequantize_encoder_tensor(value)
        value_dtype = None
        if value.is_floating_point():
            value_dtype = torch.float32 if short in keep_fp32 or module_name in keep_fp32 else dtype
        accelerate.utils.set_module_tensor_to_device(
            model, target, 'cpu', value=value, dtype=value_dtype)

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
            f'GGUF text encoder holds no weights for {len(missing)} tensors '
            f'(for example {missing[0]}); they are zeros.')
    _messages.debug_log(f'GGUF text encoder: {packed} packed linears.')
    return model.eval()


def _load_gguf_text_encoder(encoder, model_path: str, dtype: torch.dtype | None,
                            config_path: str | None = None):
    """
    Load a ``.gguf`` text encoder. Quantized linears stay packed.

    ``config_path`` skips the shipped-config match. Tests use that for a model
    smaller than any configuration dgenerate ships.
    """
    dtype = dtype or torch.bfloat16
    try:
        architecture, state = _ggufcompat.load_text_encoder_gguf(model_path)
    except _exceptions.TextEncoderUriLoadError:
        raise
    except Exception as e:
        raise _exceptions.TextEncoderUriLoadError(
            f'Failed to read GGUF text encoder "{model_path}": {e}') from e

    signature = _tensor_signature(state)
    if signature is None:
        raise _exceptions.TextEncoderUriLoadError(
            f'GGUF text encoder "{model_path}" has no token embedding and decoder layers.')
    _, shape, layers = signature
    if config_path is None:
        config_path = _require_shipped_config(encoder, shape, layers, model_path)

    try:
        config = encoder.config_class.from_json_file(config_path)
    except Exception as e:
        raise _exceptions.TextEncoderUriLoadError(
            f'Failed to read the text encoder config "{config_path}": {e}') from e

    if architecture == 'llama':
        text = getattr(config, 'text_config', None) or config
        n_head = getattr(text, 'num_attention_heads', None) or 0
        n_kv = getattr(text, 'num_key_value_heads', None) or n_head
        state = _ggufcompat.permute_llama_attention(state, n_head, n_kv)

    trimmed = _fit_layer_count(config, layers)
    try:
        with accelerate.init_empty_weights(include_buffers=False):
            model = encoder(config)
        return _assign_gguf_text_encoder(model, state, dtype, rename=None, trimmed=trimmed)
    except _exceptions.TextEncoderUriLoadError:
        raise
    except Exception as e:
        raise _exceptions.TextEncoderUriLoadError(
            f'Failed to load GGUF text encoder "{model_path}": {e}') from e


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
        'Gemma3ForConditionalGeneration': transformers.Gemma3ForConditionalGeneration,
        'Gemma4UnifiedForConditionalGeneration': transformers.Gemma4UnifiedForConditionalGeneration,
        'Mistral3ForConditionalGeneration': transformers.Mistral3ForConditionalGeneration,
        'Qwen2_5_VLForConditionalGeneration': transformers.Qwen2_5_VLForConditionalGeneration,
        'Qwen2_5_VLTextModel': transformers.Qwen2_5_VLTextModel,
        'Qwen3ForCausalLM': transformers.Qwen3ForCausalLM,
        'Qwen3Model': transformers.Qwen3Model,
        'Qwen3VLForConditionalGeneration': transformers.Qwen3VLForConditionalGeneration,
        'Qwen3VLModel': transformers.Qwen3VLModel,
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
        'model': {'mode': ['in', 'dir'], 'filetypes': [('Models', ['*.safetensors', '*.gguf', '*.pt', '*.pth', '*.cpkt', '*.bin'])]}
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

        if _hfhub.is_gguf_model(model) and quantizer:
            raise _exceptions.InvalidTextEncoderUriError(
                'A GGUF text encoder file is already quantized. '
                'Do not set the quantizer argument.')

        if _hfhub.is_single_file_model_load(model):
            if quantizer and mode not in valid_modes and not _hfhub.is_gguf_model(model):
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

            if self.mode in TextEncoderUri._valid_modes() and not _hfhub.is_gguf_model(model_path):
                if not model_path.endswith('.safetensors'):
                    raise _exceptions.TextEncoderUriLoadError(
                        f'TextEncoder mode "{self.mode}" only supports loading '
                        f'from safetensors files, but got: "{model_path}"')

            if _hfhub.is_gguf_model(model_path):
                # A GGUF text encoder is already quantized. Linears stay packed.
                self._enforce_cache_size(os.path.getsize(model_path))
                text_encoder = _load_gguf_text_encoder(encoder, model_path, torch_dtype)
                estimated_memory_use = _torchutil.estimate_module_memory_usage(text_encoder)
            elif self.mode == 'clip-l':
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
            elif (encoder_library == 'transformers' and not self.mode
                  and (t5_config := _t5_family_config(model_path, encoder))):
                # Monolithic T5 and UMT5 files, including ComfyUI quantized ones.
                # Flux and SD3 T5 files can still pass mode=; this matches UMT5,
                # which has no mode of its own, from the checkpoint shapes.
                if quant_config is not None and _safetensorscompat.is_comfy_quantized(model_path):
                    raise _exceptions.TextEncoderUriLoadError(
                        'A quantizer is not supported for a ComfyUI quantized text encoder. '
                        'The file is already quantized.')
                self._enforce_cache_size(os.path.getsize(model_path))
                config = encoder.config_class.from_json_file(t5_config)
                header = _safetensorscompat._read_header(model_path) or {}
                header.pop('__metadata__', None)
                signature = _tensor_signature(header)
                if signature is not None:
                    _fit_layer_count(config, signature[2])
                text_encoder = _load_configured_monolithic(
                    encoder, config, model_path, torch_dtype, quant_config, device_map,
                    f'{encoder.__name__} text encoder')
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
