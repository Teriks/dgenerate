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

"""Progress bars for writing large weight files."""

from __future__ import annotations

import inspect
import os
from contextlib import contextmanager

import torch
from tqdm.auto import tqdm


def tensor_nbytes(tensors) -> int:
    """Byte size of tensor values in a state dict or tensor mapping."""
    total = 0
    for value in tensors.values():
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
    return total


@contextmanager
def safetensors_write_progress(total_bytes: int, desc: str):
    """
    Advance a byte bar each time ``safetensors.torch.save_file`` returns.

    Diffusers looks the function up on the module. Transformers binds its own
    name at import, so that binding is patched too when the module is loaded.
    """
    import safetensors.torch as safetensors_torch
    import sys

    bar = tqdm(
        total=max(int(total_bytes), 1),
        desc=desc,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        leave=True,
    )
    original = safetensors_torch.save_file

    def wrapped(tensors, filename, metadata=None):
        original(tensors, filename, metadata=metadata)
        bar.update(tensor_nbytes(tensors))

    targets = [(safetensors_torch, 'save_file', original)]
    transformers_utils = sys.modules.get('transformers.modeling_utils')
    if transformers_utils is not None and hasattr(transformers_utils, 'safe_save_file'):
        targets.append((
            transformers_utils, 'safe_save_file', transformers_utils.safe_save_file))
    for module, name, _previous in targets:
        setattr(module, name, wrapped)
    try:
        yield bar
    finally:
        for module, name, previous in targets:
            setattr(module, name, previous)
        bar.close()


def save_module_with_progress(module, directory, variant=None, desc: str | None = None):
    """
    ``save_pretrained`` with a bar that advances as each shard is written.

    Shards are 2 GiB so a multi-tens-of-gigabytes component does not sit on
    one silent write.
    """
    label = desc or f'Writing {type(module).__name__}'
    total = tensor_nbytes(module.state_dict())
    signature = inspect.signature(module.save_pretrained)
    kwargs = {}
    if 'safe_serialization' in signature.parameters:
        kwargs['safe_serialization'] = True
    if variant and 'variant' in signature.parameters:
        kwargs['variant'] = variant
    if 'max_shard_size' in signature.parameters:
        kwargs['max_shard_size'] = '2GB'
    module_name = getattr(type(module), '__module__', '')
    if module_name.startswith('transformers'):
        import transformers.modeling_utils  # noqa: F401
    with safetensors_write_progress(total, label):
        module.save_pretrained(directory, **kwargs)


def save_pipeline_with_progress(pipeline, directory, variant=None, desc: str = 'Writing pipeline'):
    """
    Save a diffusers pipeline and advance a byte bar as each weight shard is written.

    ``max_shard_size`` is 2 GiB so the bar moves during a large export instead of
    waiting on one multi-tens-of-gigabytes file.
    """
    components = getattr(pipeline, 'components', None) or {}
    total = 0
    needs_transformers = False
    for value in components.values():
        if not isinstance(value, torch.nn.Module):
            continue
        total += tensor_nbytes(value.state_dict())
        if type(value).__module__.startswith('transformers'):
            needs_transformers = True
    if needs_transformers:
        import transformers.modeling_utils  # noqa: F401
    with safetensors_write_progress(total, desc):
        pipeline.save_pretrained(
            directory,
            safe_serialization=True,
            variant=variant,
            max_shard_size='2GB',
        )


def save_state_dict_with_progress(state: dict,
                                  folder: str,
                                  desc: str = 'Writing weights',
                                  max_shard_bytes: int = 2 * 1024 ** 3) -> dict | str:
    """
    Write ``state`` under ``folder`` and return an offload index.

    One shard stays ``weights.safetensors``. Several shards are
    ``weights-00001-of-N.safetensors`` plus ``weights.safetensors.index.json``.
    Shared-storage tensors fall back to accelerate's per-tensor files and
    the return value is ``folder``.
    """
    import json

    import safetensors.torch

    os.makedirs(folder, exist_ok=True)
    groups = []
    current = {}
    current_bytes = 0
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        nbytes = value.numel() * value.element_size()
        if current and current_bytes + nbytes > max_shard_bytes:
            groups.append(current)
            current = {}
            current_bytes = 0
        current[key] = value if value.is_contiguous() else value.contiguous()
        current_bytes += nbytes
    if current:
        groups.append(current)
    if not groups:
        return {}

    total = tensor_nbytes(state)
    weight_map = {}
    index = {}
    try:
        with safetensors_write_progress(total, desc):
            width = max(5, len(str(len(groups))))
            for number, group in enumerate(groups, 1):
                if len(groups) == 1:
                    filename = 'weights.safetensors'
                else:
                    filename = (
                        f'weights-{number:0{width}d}-of-'
                        f'{len(groups):0{width}d}.safetensors')
                path = os.path.join(folder, filename)
                safetensors.torch.save_file(group, path)
                absolute = os.path.abspath(path)
                for key in group:
                    weight_map[key] = filename
                    index[key] = {
                        'safetensors_file': absolute,
                        'weight_name': key,
                    }
    except RuntimeError as error:
        message = str(error).lower()
        if 'share memory' not in message and 'shared' not in message and 'alias' not in message:
            raise
        for name in os.listdir(folder):
            if name.startswith('weights'):
                os.remove(os.path.join(folder, name))
        from accelerate.utils.offload import offload_weight, save_offload_index
        accelerate_index = {}
        with safetensors_write_progress(total, desc):
            for key, value in state.items():
                if not torch.is_tensor(value):
                    continue
                accelerate_index = offload_weight(
                    value.detach(), key, folder, index=accelerate_index)
        save_offload_index(accelerate_index, folder)
        return folder

    if len(groups) > 1:
        with open(os.path.join(folder, 'weights.safetensors.index.json'),
                  'w', encoding='utf-8') as handle:
            json.dump({'weight_map': weight_map}, handle)
    return index
