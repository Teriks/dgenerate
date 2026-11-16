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
Write SDNQ and bitsandbytes weights as a diffusers repository.

``--quantizer`` takes the same URI as image generation. The folder is a model
dgenerate can load with no quantizer of its own. Components that were not
selected are copied at full precision.
"""

import os
import shutil

import torch

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.uris.bnbquantizeruri as _bnburi
import dgenerate.pipelinewrapper.uris.exceptions as _uri_exceptions
import dgenerate.pipelinewrapper.uris.sdnqquantizeruri as _sdnquri
import dgenerate.quantize.quantexport as _quantexport

_WEIGHT_SUFFIXES = {'.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.gguf', '.msgpack'}


class QuantizerSpec:
    """One ``--quantizer`` URI and the repository folder it writes."""

    def __init__(self, backend: str, uri: str, folder: str, parsed):
        self.backend = backend
        self.uri = uri
        self.folder = folder
        self.parsed = parsed

    def config(self, library: str):
        if self.backend == 'sdnq':
            return self.parsed.to_config()
        if library == 'transformers':
            return self.parsed.to_transformers_config(torch.float16)
        return self.parsed.to_config(torch.float16)


def _sdnq_folder(uri) -> str:
    parts = ['sdnq']
    if uri.type != 'int8':
        parts.append(uri.type)
    if uri.group_size:
        parts.append(f'g{uri.group_size}')
    if uri.quant_conv:
        parts.append('conv')
    if uri.quantized_matmul:
        parts.append('matmul')
    if uri.quantized_matmul_conv:
        parts.append('matmulconv')
    if uri.use_svd:
        parts.append('svd')
    return '-'.join(parts)


def _bnb_folder(uri) -> str:
    parts = ['bnb', str(uri.bits)]
    if uri.bits == 4 and uri.bits4_quant_type != 'fp4':
        parts.append(uri.bits4_quant_type)
    if uri.bits == 4 and uri.bits4_use_double_quant:
        parts.append('dq')
    if uri.bits4_compute_dtype:
        parts.append('compute-' + uri.bits4_compute_dtype)
    if uri.bits4_quant_storage:
        parts.append('storage-' + uri.bits4_quant_storage)
    return '-'.join(parts)


def parse_quantizer(uri: str) -> QuantizerSpec:
    """
    Parse an SDNQ or bitsandbytes quantizer URI.

    :raises QuantExportError: The URI is not one of those backends, or an argument is invalid.
    """
    text = uri.strip()
    scheme = text.split(';', 1)[0].strip().lower()
    try:
        if scheme == 'sdnq':
            parsed = _sdnquri.SDNQQuantizerUri.parse(text)
            return QuantizerSpec('sdnq', text, _sdnq_folder(parsed), parsed)
        if scheme in {'bnb', 'bitsandbytes'}:
            parsed = _bnburi.BNBQuantizerUri.parse(text)
            return QuantizerSpec('bnb', text, _bnb_folder(parsed), parsed)
    except (_uri_exceptions.InvalidSDNQQuantizerUriError,
            _uri_exceptions.InvalidBNBQuantizerUriError) as error:
        raise _quantexport.QuantExportError(str(error)) from error
    raise _quantexport.QuantExportError(
        f'Quantizer "{text}" is not sdnq or bnb. '
        f'A repository is written from a URI such as '
        f'"sdnq;type=uint4;group-size=32;use-svd=true" or '
        f'"bnb;bits=4;bits4-quant-type=nf4;bits4-use-double-quant=true".')


def _model_class(config: dict):
    import diffusers
    import transformers

    names = []
    class_name = config.get('_class_name')
    if class_name:
        names.append(str(class_name))
    names.extend(str(name) for name in (config.get('architectures') or []))
    for name in names:
        diffusers_class = getattr(diffusers, name, None)
        if diffusers_class is not None:
            return diffusers_class, 'diffusers'
        transformers_class = getattr(transformers, name, None)
        if transformers_class is not None:
            return transformers_class, 'transformers'
    label = names[0] if names else 'this component'
    raise _quantexport.QuantExportError(
        f'"{label}" cannot be saved as an SDNQ or bitsandbytes repository. '
        f'The component needs a diffusers or transformers class.')


def _remove_weights(directory: str) -> None:
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        if name.endswith('.index.json') or os.path.splitext(name)[1].lower() in _WEIGHT_SUFFIXES:
            os.remove(path)


def _quantize_component(source: str,
                        target: str,
                        spec: QuantizerSpec,
                        variant: str | None) -> None:
    config = _quantexport.load_component_config(source)
    if not config:
        raise _quantexport.QuantExportError(
            f'"{source}" has no config.json, so it cannot be saved as a prequantized repository.')
    model_class, library = _model_class(config)
    quant_config = spec.config(library)
    kwargs = {'quantization_config': quant_config, 'low_cpu_mem_usage': True}
    if variant:
        kwargs['variant'] = variant
    if spec.backend == 'bnb':
        if not torch.cuda.is_available():
            raise _quantexport.QuantExportError(
                'bitsandbytes repositories are quantized on CUDA, and CUDA is not available.')
        kwargs['device_map'] = {'': 'cuda:0'}
    try:
        model = model_class.from_pretrained(source, **kwargs)
    except _quantexport.QuantExportError:
        raise
    except Exception as error:
        raise _quantexport.QuantExportError(
            f'Could not quantize "{source}" with "{spec.uri}": {error}') from error
    os.makedirs(target, exist_ok=True)
    _remove_weights(target)
    try:
        if spec.backend == 'sdnq':
            from sdnq import save_sdnq_model
            save_sdnq_model(model, target, sdnq_config=quant_config)
        else:
            model.save_pretrained(target)
    except Exception as error:
        raise _quantexport.QuantExportError(
            f'Could not write "{target}": {error}') from error
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def export_pretrained_repo(root: str,
                           components: list[tuple[str, str]],
                           spec: QuantizerSpec,
                           output_dir: str,
                           variant: str | None = None) -> str:
    """
    Copy ``root`` and replace the selected components with one quantizer URI.

    The returned directory is a diffusers folder. A pipeline keeps
    ``model_index.json`` and the components that were not quantized.

    :raises QuantExportError: CUDA is required and missing, or a component cannot be quantized.
    """
    destination = os.path.join(output_dir, spec.folder)
    root_abs = os.path.abspath(root)
    destination_abs = os.path.abspath(destination)
    if destination_abs == root_abs or destination_abs.startswith(root_abs + os.sep):
        raise _quantexport.QuantExportError(
            f'Output "{destination}" is inside the model folder. Choose a directory outside it.')
    if os.path.isdir(destination):
        shutil.rmtree(destination)
    shutil.copytree(root, destination)
    names = []
    try:
        for name, folder in components:
            relative = os.path.relpath(folder, root)
            target = destination if relative == '.' else os.path.join(destination, relative)
            _messages.log(f'Quantizing {name} with {spec.uri}.')
            _quantize_component(folder, target, spec, variant)
            names.append(name)
    except Exception:
        if os.path.isdir(destination):
            shutil.rmtree(destination)
        raise
    _messages.log(f'Wrote {destination} ({", ".join(names)} with {spec.uri}).')
    return destination
