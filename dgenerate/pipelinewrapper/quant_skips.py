"""
Layers that must stay full precision when a diffusion transformer is quantized.

SDNQ already records these per architecture. Bitsandbytes does not read that
record, so it quantizes them and the sample collapses to noise. Qwen has a
second modulation linear next to the one SDNQ skips. That one produces the
text-stream scale and shift. Quantizing it does the same thing.

Qwen ControlNet reuses the same block stack. When it is quantized (only via
``--quantizer-map controlnet`` or a ControlNet URI quantizer), it needs the
same skips. By default ControlNet is left full precision.
"""


def _bnb_module_name(key: str) -> str:
    if key.startswith('.'):
        key = key[1:]
    if key.endswith('.weight'):
        key = key[:-len('.weight')]
    return key


_QWEN_TEXT_MODULATION = 'transformer_blocks.0.txt_mod.1.weight'
_QWEN_CONTROLNET_DROP = frozenset({'proj_out', 'norm_out'})

# SDNQ's table is keyed by exact class name. Animate transformers reuse the
# Wan DiT stack (patch embed, condition embedder, scale/shift, output head).
_ARCHITECTURE_SKIP_ALIASES = {
    'WanAnimateTransformer3DModel': 'WanTransformer3DModel',
    'WanAnimate2Transformer3DModel': 'WanTransformer3DModel',
}

# Classes SDNQ does not list. LTX-Video (1.x) shares the non-audio embed /
# projection layout with LTX-2's video path, plus AdaLN scale_shift_table.
_ARCHITECTURE_SKIP_EXTRA = {
    'LTXVideoTransformer3DModel': [
        'time_embed',
        'caption_projection',
        'proj_in',
        'proj_out',
        'scale_shift_table',
        'norm_out',
    ],
}

_QWEN_CLASSES = frozenset({
    'QwenImageTransformer2DModel',
    'QwenImageControlNetModel',
})

# SDNQ does not look these up by class name; write modules_to_not_convert.
_SDNQ_WRITE_CLASSES = frozenset(_ARCHITECTURE_SKIP_ALIASES) | frozenset(
    _ARCHITECTURE_SKIP_EXTRA)


def architecture_skip_keys(class_name: str) -> list[str]:
    """
    Parameter names that stay full precision for ``class_name``.

    The list is SDNQ's architecture table, local extras (LTX-Video), Animate
    aliases to Wan, and Qwen's text-stream modulation weight. An empty list
    means this class has no special case.
    """
    try:
        from sdnq.common import module_skip_keys_dict
    except ImportError:
        module_skip_keys_dict = {}

    if class_name == 'QwenImageControlNetModel':
        # Same sensitive layers as the transformer; ControlNet has no head.
        entry = module_skip_keys_dict.get('QwenImageTransformer2DModel')
        keys = [
            key for key in (list(entry[0]) if entry else [])
            if _bnb_module_name(key) not in _QWEN_CONTROLNET_DROP
        ]
        if _QWEN_TEXT_MODULATION not in keys:
            keys.append(_QWEN_TEXT_MODULATION)
        return keys

    if class_name in _ARCHITECTURE_SKIP_EXTRA:
        return list(_ARCHITECTURE_SKIP_EXTRA[class_name])

    lookup = _ARCHITECTURE_SKIP_ALIASES.get(class_name, class_name)
    entry = module_skip_keys_dict.get(lookup)
    keys = list(entry[0]) if entry else []
    if class_name == 'QwenImageTransformer2DModel':
        if _QWEN_TEXT_MODULATION not in keys:
            keys.append(_QWEN_TEXT_MODULATION)
    return keys


def apply_architecture_quant_skips(quant_config, model_class: type | str | None):
    """
    Merge :func:`architecture_skip_keys` into a bitsandbytes or SDNQ config.

    Bitsandbytes reads ``llm_int8_skip_modules`` for both 4-bit and 8-bit.
    SDNQ reads ``modules_to_not_convert`` and also adds its own table later
    by exact class name, so only Qwen extras, Animate aliases, and local
    extras (LTX-Video) need writing there.
    """
    if quant_config is None or model_class is None:
        return quant_config

    class_name = model_class if isinstance(model_class, str) else model_class.__name__
    keys = architecture_skip_keys(class_name)
    if not keys:
        return quant_config

    if hasattr(quant_config, 'llm_int8_skip_modules'):
        existing = list(quant_config.llm_int8_skip_modules or [])
        for key in keys:
            name = _bnb_module_name(key)
            if name and name not in existing:
                existing.append(name)
        quant_config.llm_int8_skip_modules = existing
        return quant_config

    if hasattr(quant_config, 'modules_to_not_convert'):
        if class_name in _SDNQ_WRITE_CLASSES:
            keys_to_add = keys
        elif class_name in _QWEN_CLASSES and _QWEN_TEXT_MODULATION in keys:
            keys_to_add = [_QWEN_TEXT_MODULATION]
        else:
            return quant_config
        current = quant_config.modules_to_not_convert
        if current is None:
            current = []
        else:
            current = list(current)
        for key in keys_to_add:
            if key not in current:
                current.append(key)
        quant_config.modules_to_not_convert = current

    return quant_config
