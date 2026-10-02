"""
Layers that must stay full precision when a diffusion transformer is quantized.

SDNQ already records these per architecture. Bitsandbytes does not read that
record, so it quantizes them and the sample collapses to noise. Qwen has a
second modulation linear next to the one SDNQ skips. That one produces the
text-stream scale and shift. Quantizing it does the same thing.
"""


def _bnb_module_name(key: str) -> str:
    if key.startswith('.'):
        key = key[1:]
    if key.endswith('.weight'):
        key = key[:-len('.weight')]
    return key


def architecture_skip_keys(class_name: str) -> list[str]:
    """
    Parameter names that stay full precision for ``class_name``.

    The list is SDNQ's architecture table, plus Qwen's text-stream modulation
    weight. An empty list means this class has no special case.
    """
    try:
        from sdnq.common import module_skip_keys_dict
    except ImportError:
        module_skip_keys_dict = {}

    entry = module_skip_keys_dict.get(class_name)
    keys = list(entry[0]) if entry else []
    if class_name == 'QwenImageTransformer2DModel':
        text_modulation = 'transformer_blocks.0.txt_mod.1.weight'
        if text_modulation not in keys:
            keys.append(text_modulation)
    return keys


def apply_architecture_quant_skips(quant_config, model_class: type | str | None):
    """
    Merge :func:`architecture_skip_keys` into a bitsandbytes or SDNQ config.

    Bitsandbytes reads ``llm_int8_skip_modules`` for both 4-bit and 8-bit.
    SDNQ reads ``modules_to_not_convert`` and also adds its own table later,
    so only the Qwen text-stream weight has to be written there.
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
        extra = 'transformer_blocks.0.txt_mod.1.weight'
        if class_name != 'QwenImageTransformer2DModel' or extra not in keys:
            return quant_config
        current = quant_config.modules_to_not_convert
        if current is None:
            current = []
        else:
            current = list(current)
        if extra not in current:
            current.append(extra)
        quant_config.modules_to_not_convert = current

    return quant_config
