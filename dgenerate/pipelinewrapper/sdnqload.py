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

"""Detect prequantized SDNQ checkpoints and import ``sdnq`` before ``from_pretrained``."""

import json
import os.path

import huggingface_hub
import huggingface_hub.errors

import dgenerate.hfhub as _hfhub
import dgenerate.messages as _messages

_SDNQ_ALREADY_QUANTIZED = (
    'This checkpoint is already SDNQ-quantized. '
    'Do not pass --quantizer sdnq; that would quantize it again.'
)

_sdnq_imported = False


def quantizer_uri_is_sdnq(quantizer_uri: str | None) -> bool:
    """Return whether a ``--quantizer`` URI selects the SDNQ backend."""
    if not quantizer_uri:
        return False
    return quantizer_uri.split(';', 1)[0].strip().lower() == 'sdnq'


def import_sdnq():
    """
    Import ``sdnq`` once so its quantizer class is registered.

    The import is lazy. SDNQ warns about optional kernels at import time.
    """
    global _sdnq_imported
    if _sdnq_imported:
        return
    import sdnq  # noqa: F401
    _sdnq_imported = True
    _messages.debug_log('Imported sdnq so prequantized checkpoints can load.')


def config_method_is_sdnq(data) -> bool:
    """Return whether a config dict describes SDNQ quantization."""
    if not isinstance(data, dict):
        return False
    method = data.get('quant_method', data.get('quantization_method'))
    if hasattr(method, 'value'):
        method = method.value
    return str(method).lower() == 'sdnq'


def sdnq_config_in_directory(directory: str) -> dict | None:
    """
    Read an SDNQ quantization config from a local component directory.

    ``quantization_config.json`` is the file ``sdnq.load_sdnq_model`` reads.
    ``config.json`` may instead nest that object under ``quantization_config``.
    """
    if not directory or not os.path.isdir(directory):
        return None

    quant_file = os.path.join(directory, 'quantization_config.json')
    if os.path.isfile(quant_file):
        with open(quant_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        if config_method_is_sdnq(data):
            return data

    config_file = os.path.join(directory, 'config.json')
    if os.path.isfile(config_file):
        with open(config_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        nested = data.get('quantization_config') if isinstance(data, dict) else None
        if isinstance(nested, dict) and config_method_is_sdnq(nested):
            return nested
        if config_method_is_sdnq(data):
            return data
    return None


def _download_config_json(
        repo_id: str,
        filename: str,
        subfolder: str | None,
        revision: str | None,
        token: str | None,
        local_files_only: bool
) -> dict | None:
    try:
        path = huggingface_hub.hf_hub_download(
            repo_id,
            filename=filename,
            subfolder=subfolder or None,
            revision=revision,
            token=token,
            local_files_only=local_files_only
        )
    except (huggingface_hub.errors.EntryNotFoundError,
            huggingface_hub.errors.LocalEntryNotFoundError):
        return None
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def sdnq_config_for_component(
        model_path: str,
        subfolder: str | None = None,
        revision: str | None = None,
        token: str | None = None,
        local_files_only: bool = False
) -> dict | None:
    """
    Return the SDNQ quantization config for a pipeline repo or component directory.

    Local directories are read in place. A Hugging Face repo downloads only
    ``quantization_config.json`` or ``config.json``, not the weights.
    """
    if not model_path or _hfhub.is_single_file_model_load(model_path):
        return None

    if os.path.isdir(model_path):
        directory = os.path.join(model_path, subfolder) if subfolder else model_path
        return sdnq_config_in_directory(directory)

    quant = _download_config_json(
        model_path, 'quantization_config.json', subfolder, revision, token, local_files_only)
    if config_method_is_sdnq(quant):
        return quant

    config = _download_config_json(
        model_path, 'config.json', subfolder, revision, token, local_files_only)
    if not isinstance(config, dict):
        return None
    nested = config.get('quantization_config')
    if isinstance(nested, dict) and config_method_is_sdnq(nested):
        return nested
    if config_method_is_sdnq(config):
        return config
    return None


def sdnq_requantize_error(quantizer_uri: str | None, sdnq_config: dict | None) -> str | None:
    """
    Return an error when ``--quantizer sdnq`` is aimed at a checkpoint that is already SDNQ.

    Import ``sdnq`` when a prequantized config is present so the quantizer class
    is registered before ``from_pretrained``.
    """
    if sdnq_config is None:
        return None
    import_sdnq()
    if quantizer_uri_is_sdnq(quantizer_uri):
        return _SDNQ_ALREADY_QUANTIZED
    return None


def quantization_config_of(module):
    """
    Quantization config attached to a loaded module.

    SDNQ assigns ``quantization_config`` onto the module. Diffusers also stores
    it on ``module.config``, and warns when that field is read as
    ``module.quantization_config``.
    """
    if module is None:
        return None
    values = getattr(module, '__dict__', None)
    if isinstance(values, dict) and 'quantization_config' in values:
        return values['quantization_config']
    config = values.get('config') if isinstance(values, dict) else None
    if config is None:
        return None
    config_values = getattr(config, '__dict__', None)
    if isinstance(config_values, dict) and 'quantization_config' in config_values:
        return config_values['quantization_config']
    getter = getattr(config, 'get', None)
    if getter is None:
        return None
    try:
        return getter('quantization_config', None)
    except Exception:
        return None


def module_is_sdnq(module) -> bool:
    """Return whether a loaded module carries an SDNQ quantization config."""
    candidate = quantization_config_of(module)
    if candidate is None:
        return False
    if type(candidate).__name__ == 'SDNQConfig':
        return True
    if isinstance(candidate, dict):
        return config_method_is_sdnq(candidate)
    method = getattr(candidate, 'quant_method', None)
    if hasattr(method, 'value'):
        method = method.value
    return str(method).lower() == 'sdnq'
