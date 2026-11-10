# Copyright (c) 2023, Teriks
# BSD 3-Clause License

"""
Give Diffusers the original SDNQ linear shape when a LoRA is loaded.

SDNQ int4 packs two values into each byte and stores that vector as one
dimension. ``FluxLoraLoaderMixin`` reads ``weight.shape`` and unpacks it as
``(out_features, in_features)``. A packed vector has one dimension, so that
unpack raises ``not enough values to unpack (expected 2, got 1)``. The
dequantizer still has the shape from before packing.
"""

import diffusers.loaders.lora_pipeline as _lora_pipeline

_original_calculate_module_shape = _lora_pipeline.FluxLoraLoaderMixin._calculate_module_shape


def _sdnq_original_shape(module):
    dequantizer = getattr(module, 'sdnq_dequantizer', None)
    return getattr(dequantizer, 'original_shape', None)


def _calculate_module_shape(model, base_module=None, base_weight_param_name=None):
    if base_module is not None:
        shape = _sdnq_original_shape(base_module)
        if shape is not None:
            return shape
    elif (
            base_weight_param_name is not None
            and str(base_weight_param_name).endswith('.weight')
            and model is not None):
        module_path = base_weight_param_name.rsplit('.weight', 1)[0]
        try:
            submodule = model.get_submodule(module_path)
        except AttributeError:
            submodule = None
        if submodule is not None:
            shape = _sdnq_original_shape(submodule)
            if shape is not None:
                return shape
    return _original_calculate_module_shape(
        model,
        base_module=base_module,
        base_weight_param_name=base_weight_param_name)


_lora_pipeline.FluxLoraLoaderMixin._calculate_module_shape = staticmethod(_calculate_module_shape)
