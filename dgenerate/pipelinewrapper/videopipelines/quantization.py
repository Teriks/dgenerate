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

"""Quantized component loading for video pipelines."""

from __future__ import annotations

import gc
import importlib

import torch

import dgenerate.hfhub as _hfhub
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.pipelinewrapper.util as _util
import dgenerate.pipelinewrapper.videopipelines as _vp


_DEFAULT_QUANT_NAMES = {
    'transformer',
    'transformer_2',
    'text_encoder',
    'text_encoder_2',
    'text_encoder_3',
    'image_encoder',
    'connectors',
}


# Loaded as None so from_pretrained does not pull the unused Gemma
# prompt enhancer (about 9.5 GiB) onto the GPU.
_LTX_SKIP_OPTIONAL_MODULES = {
    'prompt_enhancer': None,
}


def _quantize_component(name: str, quantizer_uri, quantizer_map) -> bool:
    if not quantizer_uri:
        return False
    if quantizer_map is None:
        return name in _DEFAULT_QUANT_NAMES
    return name in quantizer_map


def _quantizer_config(quantizer_uri, dtype, transformers_module: bool):
    parsed = _uris.get_quantizer_uri_class(quantizer_uri).parse(quantizer_uri)
    torch_dtype = _enums.get_torch_dtype(dtype) if isinstance(dtype, _enums.DataType) else None
    if transformers_module and hasattr(parsed, 'to_transformers_config'):
        return parsed.to_transformers_config(torch_dtype)
    return parsed.to_config(torch_dtype)


def _resolve_index_class(library: str, class_name: str):
    import diffusers.pipelines as pipelines

    if hasattr(pipelines, library):
        module = importlib.import_module(f'diffusers.pipelines.{library}')
    else:
        module = importlib.import_module(library)
    return getattr(module, class_name)


def _load_quantized_module(component_class, model_path, subfolder, revision, variant,
                           dtype, quantizer_uri, auth_token, local_files_only, device_map,
                           offload=False):
    module_name = getattr(component_class, '__module__', '')
    transformers_module = module_name.startswith('transformers')
    config = _vp._quantizer_config(quantizer_uri, dtype, transformers_module)
    from dgenerate.pipelinewrapper.quant_skips import apply_architecture_quant_skips
    apply_architecture_quant_skips(config, component_class)
    torch_dtype = _enums.get_torch_dtype(dtype) if isinstance(dtype, _enums.DataType) else None
    _messages.debug_log(
        f'Quantizing {component_class.__name__} from "{model_path}" '
        f'subfolder "{subfolder}" with "{quantizer_uri}".')
    load_kwargs = {
        'subfolder': subfolder or '',
        'revision': revision,
        'variant': variant,
        'token': auth_token,
        'local_files_only': local_files_only,
        'quantization_config': config,
        'device_map': device_map,
        'low_cpu_mem_usage': True,
    }
    if torch_dtype is not None:
        if transformers_module:
            load_kwargs['dtype'] = torch_dtype
        else:
            load_kwargs['torch_dtype'] = torch_dtype
    with _hfhub.with_hf_errors_as_model_not_found():
        module = component_class.from_pretrained(model_path, **load_kwargs)
    # 8-bit BnB weights cannot leave the GPU. Pin .to() like still pipelines.
    if _util.is_loaded_in_8bit_bnb(module):
        _pipelines._disable_to(module)
    elif offload:
        module.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return module


def _pin_video_8bit_modules(pipe):
    """Disable ``.to()`` on 8-bit BnB modules after the video pipeline is built."""
    for module in _pipelines.get_pipeline_modules(pipe).values():
        if _util.is_loaded_in_8bit_bnb(module):
            _pipelines._disable_to(module)


def _quantized_device_map(device, offload: bool, quantizer_uri=None):
    # bitsandbytes rejects a CPU or disk device_map, so it has to be
    # created on the GPU. SDNQ weights can be quantized on the CPU and
    # streamed later, which is what keeps a 16GB card from filling up.
    del offload
    if not quantizer_uri or not str(quantizer_uri).split(';', 1)[0] == 'bnb':
        return None
    if device is None or str(device) == 'cpu':
        return None
    return {'': str(device)}


def _quantized_classic_modules(model_path, revision, variant, subfolder, dtype,
                               quantizer_uri, quantizer_map, auth_token,
                               local_files_only, device, offload, skip_names=()):
    if not quantizer_uri:
        return {}
    index = _util.fetch_model_index_dict(
        model_path,
        subfolder=subfolder,
        revision=revision,
        auth_token=auth_token,
        local_files_only=local_files_only)
    device_map = _quantized_device_map(device, offload, quantizer_uri)
    modules = {}
    for name, spec in index.items():
        if name.startswith('_') or name in skip_names:
            continue
        if not isinstance(spec, (list, tuple)) or len(spec) != 2:
            continue
        library_name, class_name = spec
        # Optional slots such as Wan 2.2's image_encoder are [null, null].
        if not isinstance(library_name, str) or not isinstance(class_name, str):
            continue
        if not _quantize_component(name, quantizer_uri, quantizer_map):
            continue
        component_class = _resolve_index_class(library_name, class_name)
        folder = f'{subfolder}/{name}' if subfolder else name
        modules[name] = _vp._load_quantized_module(
            component_class, model_path, folder, revision, variant, dtype,
            quantizer_uri, auth_token, local_files_only, device_map, offload=offload)
    return modules
