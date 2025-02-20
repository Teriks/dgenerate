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


__doc__ = "Configure dgenerate's model garbage collection heuristics."

import contextlib
import importlib
import json
import yaml
import toml
import dgenerate.types as _types

_cache_memory_constraints = {
    "pipeline_wrapper_cache_gc_constraints": "dgenerate.pipelinewrapper.PIPELINE_WRAPPER_CACHE_GC_CONSTRAINTS",
    "pipeline_cache_memory_constraints": "dgenerate.pipelinewrapper.PIPELINE_CACHE_MEMORY_CONSTRAINTS",
    "unet_cache_memory_constraints": "dgenerate.pipelinewrapper.UNET_CACHE_MEMORY_CONSTRAINTS",
    "vae_cache_memory_constraints": "dgenerate.pipelinewrapper.VAE_CACHE_MEMORY_CONSTRAINTS",
    "controlnet_cache_memory_constraints": "dgenerate.pipelinewrapper.CONTROLNET_CACHE_MEMORY_CONSTRAINTS",
    "adapter_cache_memory_constraints": "dgenerate.pipelinewrapper.ADAPTER_CACHE_MEMORY_CONSTRAINTS",
    "transformer_cache_memory_constraints": "dgenerate.pipelinewrapper.TRANSFORMER_CACHE_MEMORY_CONSTRAINTS",
    "text_encoder_cache_memory_constraints": "dgenerate.pipelinewrapper.TEXT_ENCODER_CACHE_MEMORY_CONSTRAINTS",
    "image_encoder_cache_memory_constraints": "dgenerate.pipelinewrapper.IMAGE_ENCODER_CACHE_MEMORY_CONSTRAINTS",
    "image_processor_cache_gc_constraints": "dgenerate.imageprocessors.IMAGE_PROCESSOR_CACHE_GC_CONSTRAINTS",
    "image_processor_cache_memory_constraints": "dgenerate.imageprocessors.IMAGE_PROCESSOR_CACHE_MEMORY_CONSTRAINTS",
    "image_processor_cuda_memory_constraints": "dgenerate.imageprocessors.IMAGE_PROCESSOR_CUDA_MEMORY_CONSTRAINTS",
    "prompt_weighter_cache_gc_constraints": "dgenerate.promptweighters.PROMPT_WEIGHTER_CACHE_GC_CONSTRAINTS",
    "prompt_weighter_cache_memory_constraints": "dgenerate.promptweighters.PROMPT_WEIGHTER_CACHE_MEMORY_CONSTRAINTS",
    "prompt_weighter_cuda_memory_constraints": "dgenerate.promptweighters.PROMPT_WEIGHTER_CUDA_MEMORY_CONSTRAINTS"
}


def _get_constant_by_string(path: str):
    module_path, _, attr_name = path.rpartition(".")

    if not module_path or not attr_name:
        raise ValueError(f"Invalid path: {path}")

    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not retrieve '{path}': {e}")


def _set_constant_by_string(path: str, value):
    module_path, _, attr_name = path.rpartition(".")

    if not module_path or not attr_name:
        raise ValueError(f"Invalid path: {path}")

    try:
        module = importlib.import_module(module_path)
        return setattr(module, attr_name, value)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not retrieve '{path}': {e}")


def _get_config_dict():
    config_dict = dict()
    for name, location in _cache_memory_constraints.items():
        config_dict[name] = _get_constant_by_string(location)
    return config_dict


def get_config_dict():
    """
    Return a dictionary representation of the GC configuration.

    :return: config dictionary
    """
    return _types.partial_deep_copy_container(_get_config_dict())


def set_from_config_dict(config_dict: dict):
    """
    Set the current GC config from a dictionary object.

    This dictionary may be partial, i.e. an incomplete set of settings
    as long as the key names mentioned are correct.

    :param config_dict: The config dictionary
    :raise KeyError: If a configuration key name is not valid.
    """
    for name, value in config_dict.items():
        _set_constant_by_string(_cache_memory_constraints[name], value)


def serialize_current_config(stream=None, mode: str = 'json') -> str | None:
    """
    Serialize the current GC config.

    :param stream: File like object, if not provided this function will return a string.
    :param mode: ``json``, ``yaml``, or ``toml``
    :return: the serialized config
    """
    config_dict = _get_config_dict()

    if mode == 'json':
        if stream:
            json.dump(config_dict, stream, indent=4)
        else:
            return json.dumps(config_dict, indent=4)
    elif mode == 'yaml':
        if stream:
            yaml.dump(config_dict, stream=stream, default_flow_style=False)
        else:
            return yaml.dump(config_dict, default_flow_style=False)
    elif mode == 'toml':
        if stream:
            toml.dump(config_dict, stream)
        else:
            return toml.dumps(config_dict)
    else:
        raise ValueError(f'Unknown serialization mode: {mode}')

    return None


__config_stack = []


def push_config():
    """
    Save the current configuration to the stack.
    """
    __config_stack.append(_get_config_dict())


def pop_config():
    """
    Pop the last saved configuration off the stack and restore it.

    :raise IndexError: if the stack is empty.
    """

    config_dict = __config_stack.pop()

    set_from_config_dict(config_dict)


@contextlib.contextmanager
def restore_config_context():
    """
    Context manager which pushes the current GC configuration to the stack
    and pops it when the ``with`` context ends.
    """
    try:
        push_config()
        yield
    finally:
        pop_config()


def load_config(content_or_stream, mode: str = 'json'):
    """
    Load GC config from a string.

    :param content_or_stream: string content or file like object.
    :param mode: ``json``, ``yaml``, or ``toml``
    """
    if mode == 'json':
        config = json.loads(content_or_stream) \
            if isinstance(content_or_stream, str) else json.load(content_or_stream)
    elif mode == 'yaml':
        config = yaml.safe_load(content_or_stream)
    elif mode == 'toml':
        config = toml.loads(content_or_stream) \
            if isinstance(content_or_stream, str) else toml.load(content_or_stream)
    else:
        raise ValueError(f'Unknown deserialization mode: {mode}')

    for name, value in config.items():
        _set_constant_by_string(_cache_memory_constraints[name], value)


__all__ = _types.module_all()
