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

"""Load a video pipeline when its weights do not fit in CPU memory.

Sequential and group offload keep every weight in RAM and only save GPU
memory. Wan 2.2's two 14B experts overflow that RAM while the second
expert is still loading. This loader brings in one component, leaves its
weights on disk, and frees the RAM before the next component is read.
A forward pass copies one layer back onto the execution device.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import struct
import tempfile
import weakref

import torch

import dgenerate.hfhub as _hfhub
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.uris as _uris


# One extra shard is resident while that shard is copied into the model.
_LOAD_HEADROOM_BYTES = 10 * 1024 ** 3

# File size can sit a little above physical RAM when the loaded dtype is
# narrower. LTX-2.5 is about 66 GiB of files and ran in 64 GiB of RAM.
# Wan 2.2's two experts are about 117 GiB and stay paged for every step.
_RESIDENT_SLACK_BYTES = 16 * 1024 ** 3

# In-RAM footprint reported to the pipeline cache. The weights themselves
# are files, so the cache must not treat them as resident.
DISK_PIPELINE_CACHE_BYTES = 256 * 1024 * 1024

_SAFETENSORS_DTYPE = {
    torch.float64: 'F64',
    torch.float32: 'F32',
    torch.float16: 'F16',
    torch.bfloat16: 'BF16',
    torch.int64: 'I64',
    torch.int32: 'I32',
    torch.int16: 'I16',
    torch.int8: 'I8',
    torch.uint8: 'U8',
    torch.bool: 'BOOL',
}


def format_gib(num_bytes: int) -> str:
    """Format a byte count in gibibytes for log messages."""
    return f'{num_bytes / (1024 ** 3):.1f} GiB'


def pipeline_fits_in_cpu_memory(estimate_bytes: int,
                                available_bytes: int | None = None,
                                physical_bytes: int | None = None) -> bool:
    """
    True when ``estimate_bytes`` can stay resident for the whole run.

    The page file may absorb the load spike, but weights that cannot sit in
    physical RAM are read back from that page file on every layer. Those
    loads stream from the checkpoint instead. A model only a little larger
    than RAM, such as LTX-2.5's 66 GiB of files on a 64 GiB machine, still
    stays resident. Another 10 GiB of the commit budget is reserved for the
    shard copy a sharded load holds briefly.

    :param estimate_bytes: estimated pipeline weight bytes
    :param available_bytes: committable bytes, or the live budget when omitted
    :param physical_bytes: installed RAM. When omitted, the live total is
        used only if ``available_bytes`` is also omitted.
    """
    estimate_bytes = int(estimate_bytes)
    query_live = available_bytes is None and physical_bytes is None
    if query_live:
        physical_bytes = int(_memory.get_total_memory('b'))
    if available_bytes is None:
        available_bytes = int(_memory.get_available_commit_memory('b'))
    if (physical_bytes is not None
            and estimate_bytes > int(physical_bytes) + _RESIDENT_SLACK_BYTES):
        return False
    return estimate_bytes + _LOAD_HEADROOM_BYTES < int(available_bytes)


_COUNTED_COMPONENT_NAMES = frozenset({
    'unet', 'prior', 'decoder', 'transformer', 'vae',
    'text_encoder', 'text_encoder_2', 'text_encoder_3',
})


def uncounted_component_names(index, extra_dirs=(), load_kwargs=None, replaced=()) -> list[str]:
    """
    Component folders the pipeline estimate does not already include.

    Wan 2.2's ``transformer_2`` is one of these. Leaving it out makes a
    two-expert checkpoint look small enough to keep in RAM.
    """
    present = set(extra_dirs or ())
    skipped = set(replaced or ())
    provided = load_kwargs or {}
    names = []
    for name, spec in (index or {}).items():
        if _component_spec(spec) is None or name in skipped:
            continue
        if name in provided and provided[name] is None:
            continue
        if name in _COUNTED_COMPONENT_NAMES or name in present:
            continue
        names.append(name)
    return names


def _uses_packed_weights(*uris) -> bool:
    for uri in uris:
        if not uri:
            continue
        text = str(uri)
        if _hfhub.is_gguf_model(text.split(';', 1)[0]):
            return True
    return False


def should_stream_weights_from_disk(estimate_bytes: int,
                                    quantizer_uri=None,
                                    quantizer_map=None,
                                    model_path=None,
                                    transformer_uri=None,
                                    second_transformer_uri=None,
                                    vae_uri=None,
                                    available_bytes: int | None = None) -> bool:
    """
    True when a normal load would exhaust CPU memory.

    Quantized and GGUF weights are left on the ordinary load path. Packed
    parameters cannot be rebuilt from a plain safetensors copy.
    """
    if pipeline_fits_in_cpu_memory(estimate_bytes, available_bytes):
        return False
    if quantizer_uri or quantizer_map:
        return False
    if _uses_packed_weights(model_path, transformer_uri, second_transformer_uri, vae_uri):
        return False
    for uri in (transformer_uri, second_transformer_uri):
        if uri and _uris.TransformerUri.parse(uri).quantizer:
            return False
    if vae_uri and getattr(_uris.VAEUri.parse(vae_uri), 'quantizer', None):
        return False
    return True


def _release_cpu_cache() -> None:
    gc.collect()
    empty = getattr(torch._C, '_host_emptyCache', None)
    if empty is not None:
        empty()


def _safetensors_header(path: str) -> dict | None:
    try:
        with open(path, 'rb') as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                return None
            (size,) = struct.unpack('<Q', raw)
            if size <= 0 or size > 50_000_000:
                return None
            payload = handle.read(size)
        header = json.loads(payload)
    except (OSError, json.JSONDecodeError, struct.error, UnicodeDecodeError):
        return None
    if not isinstance(header, dict):
        return None
    header.pop('__metadata__', None)
    return header


def _index_from_weight_map(folder: str, weight_map: dict, state: dict) -> dict | None:
    if not weight_map or any(key not in weight_map for key in state):
        return None
    headers: dict[str, dict] = {}
    index = {}
    for key, value in state.items():
        expected = _SAFETENSORS_DTYPE.get(value.dtype)
        if expected is None:
            return None
        filename = weight_map[key]
        if not isinstance(filename, str):
            return None
        path = os.path.join(folder, filename)
        header = headers.get(path)
        if header is None:
            header = _safetensors_header(path)
            if header is None:
                return None
            headers[path] = header
        entry = header.get(key)
        if not isinstance(entry, dict) or entry.get('dtype') != expected:
            return None
        index[key] = {
            'safetensors_file': os.path.abspath(path),
            'weight_name': key,
        }
    return index


def _reuse_checkpoint_index(folder: str | None, state: dict) -> dict | None:
    """Point at the checkpoint files when they already match the loaded tensors."""
    if not folder or not os.path.isdir(folder) or not state:
        return None
    try:
        names = os.listdir(folder)
    except OSError:
        return None
    for name in names:
        if not name.endswith('.safetensors.index.json'):
            continue
        try:
            with open(os.path.join(folder, name), encoding='utf-8') as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        weight_map = payload.get('weight_map') if isinstance(payload, dict) else None
        if isinstance(weight_map, dict):
            index = _index_from_weight_map(folder, weight_map, state)
            if index is not None:
                return index
    for name in names:
        if not name.endswith('.safetensors') or name.endswith('.index.json'):
            continue
        index = _index_from_weight_map(folder, {key: name for key in state}, state)
        if index is not None:
            return index
    return None


def _write_weights(state: dict, folder: str, desc: str | None = None) -> dict | str:
    """
    Store ``state`` under ``folder``.

    Returns a safetensors index, or the folder path when the tensors share
    storage and were written as accelerate memory-mapped files instead.
    """
    os.makedirs(folder, exist_ok=True)
    from dgenerate.pipelinewrapper.weightsave import save_state_dict_with_progress
    nbytes = sum(
        value.numel() * value.element_size()
        for value in state.values() if torch.is_tensor(value))
    gib = nbytes / (1024 ** 3)
    return save_state_dict_with_progress(
        state, folder, desc=desc or f'Writing weights ({gib:.1f} GiB)')


def _freeze_to(module) -> None:
    def frozen(*args, **kwargs):
        return module

    module.to = frozen


def _install_disk_hooks(module, weights: dict | str, execution_device: torch.device) -> None:
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module, attach_align_device_hook
    from accelerate.utils.offload import OffloadedWeightsLoader

    if isinstance(weights, str):
        loader = OffloadedWeightsLoader(save_folder=weights)
    else:
        device = None if execution_device.type == 'cpu' else str(execution_device)
        loader = OffloadedWeightsLoader(index=weights, device=device)
    _pipelines._promote_bare_parameter_tensors(module)
    # The root hook sends outputs back to the input device. The child hooks
    # load each parameter for the forward and drop it afterward.
    add_hook_to_module(module, AlignDevicesHook(io_same_device=True), append=True)
    attach_align_device_hook(
        module,
        execution_device=execution_device,
        offload=True,
        offload_buffers=True,
        weights_map=loader,
        preload_module_classes=_pipelines._sequential_offload_preload_classes(module),
    )
    _pipelines._set_sequential_cpu_offload_flag(module, True)
    module._dgenerate_weights_on_disk = True
    _freeze_to(module)


def _persistent_cache_dir(cache_key: str) -> str:
    parent = os.path.dirname(_pipelines.get_converted_checkpoint_cache_dir())
    path = os.path.join(parent, 'disk_offload', 'cache', cache_key)
    os.makedirs(path, exist_ok=True)
    return path


def _saved_copy_index(folder: str, state: dict) -> dict | str | None:
    """Reuse a copy written by an earlier run when it still matches ``state``."""
    weights = os.path.join(folder, 'weights.safetensors')
    if os.path.isfile(weights):
        return _index_from_weight_map(
            folder, {key: 'weights.safetensors' for key in state}, state)
    shard_index = os.path.join(folder, 'weights.safetensors.index.json')
    if os.path.isfile(shard_index):
        try:
            with open(shard_index, encoding='utf-8') as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            payload = None
        weight_map = payload.get('weight_map') if isinstance(payload, dict) else None
        if isinstance(weight_map, dict):
            reused = _index_from_weight_map(folder, weight_map, state)
            if reused is not None:
                return reused
    index_path = os.path.join(folder, 'index.json')
    if not os.path.isfile(index_path):
        return None
    try:
        with open(index_path, encoding='utf-8') as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if isinstance(payload, dict) and all(key in payload for key in state):
        return folder
    return None


def _checkpoint_source(load_kwargs, name) -> str:
    """Identity of a component loaded from the pipeline repository."""
    return repr((
        load_kwargs.get('revision'),
        load_kwargs.get('variant'),
        load_kwargs.get('subfolder'),
        name,
    ))


def _cache_key(repo, source, state, loras, fuse_scale) -> str:
    dtypes = tuple(sorted(str(value.dtype) for value in state.values()))
    shapes = tuple(sorted(
        (key, tuple(value.shape)) for key, value in state.items()))
    lora_ids = tuple(str(lora) for lora in loras)
    identity = repr((repo, source, dtypes, shapes, lora_ids, fuse_scale))
    return hashlib.sha256(identity.encode('utf-8')).hexdigest()


def offload_module_weights(module,
                           checkpoint_folder: str | None,
                           offload_dir: str,
                           execution_device: torch.device,
                           dirty: bool = False,
                           cache_identity=None) -> bool:
    """
    Leave ``module`` weights on disk and drop the RAM copy.

    When ``dirty`` is false and ``checkpoint_folder`` already contains
    matching safetensors, those files are used directly. Otherwise the
    loaded tensors are written to the persistent disk cache, or under
    ``offload_dir`` when no cache identity is given.

    :param cache_identity: ``(repo, source, loras, fuse_scale)`` for a
        persistent copy. ``source`` identifies the checkpoint that produced
        the tensors, including a replacement ``--transformer`` or ``--vae``.
        A later load with the same identity reads that copy instead of
        writing the weights again.
    :return: True when the module had weights to offload
    """
    state = {key: value.detach() for key, value in module.state_dict().items()}
    if not state:
        return False
    index: dict | str | None = None
    if not dirty:
        index = _reuse_checkpoint_index(checkpoint_folder, state)
        if index is not None:
            _messages.debug_log(
                f'Using checkpoint files for {type(module).__name__} '
                'instead of copying them.')
    store = offload_dir
    if index is None and cache_identity is not None:
        repo, source, loras, fuse_scale = cache_identity
        store = _persistent_cache_dir(
            _cache_key(repo, source, state, loras, fuse_scale))
        index = _saved_copy_index(store, state)
        if index is not None:
            _messages.debug_log(
                f'Using cached on-disk weights for {type(module).__name__}.')
    if index is None:
        nbytes = sum(value.numel() * value.element_size() for value in state.values())
        _messages.log(
            f'Writing {type(module).__name__} weights to disk '
            f'({format_gib(nbytes)}) so they can be released from memory.')
        index = _write_weights(
            state, store, desc=f'Writing {type(module).__name__}')
    del state
    _install_disk_hooks(module, index, execution_device)
    _release_cpu_cache()
    return True


def _loras_for_component(component_name: str, lora_uris) -> list:
    selected = []
    for uri in lora_uris or ():
        parsed = uri if isinstance(uri, _uris.LoRAUri) else _uris.LoRAUri.parse(uri)
        target = parsed.component or 'transformer'
        if target == component_name:
            selected.append(parsed)
    return selected


def _read_lora_state(pipeline_class, lora: _uris.LoRAUri, token, local_files_only):
    model_path = _hfhub.download_non_hf_slug_model(
        lora.model, local_files_only=local_files_only, auth_token=token)
    kwargs = dict(
        return_lora_metadata=True,
        weight_name=lora.weight_name,
        subfolder=lora.subfolder,
        revision=lora.revision,
        local_files_only=local_files_only,
        token=token,
    )
    try:
        return pipeline_class.lora_state_dict(model_path, use_safetensors=True, **kwargs)
    except OSError:
        return pipeline_class.lora_state_dict(model_path, **kwargs)


def _fuse_loras(pipeline_class, module, loras, fuse_scale, token, local_files_only) -> None:
    if not hasattr(pipeline_class, 'load_lora_into_transformer'):
        raise _pipelines.UnsupportedPipelineConfigError(
            'This pipeline does not fit in CPU memory, and its LoRAs cannot be '
            'merged one component at a time. Free some memory, or load a smaller checkpoint.')
    names = []
    weights = []
    for index, lora in enumerate(loras):
        state_dict, metadata = _read_lora_state(
            pipeline_class, lora, token, local_files_only)
        expand = getattr(pipeline_class, '_maybe_expand_t2v_lora_for_i2v', None)
        if expand is not None:
            state_dict = expand(module, state_dict)
        adapter_name = str(index)
        pipeline_class.load_lora_into_transformer(
            state_dict,
            transformer=module,
            adapter_name=adapter_name,
            metadata=metadata,
            low_cpu_mem_usage=False,
        )
        names.append(adapter_name)
        weights.append(lora.scale)
        _messages.debug_log(f'Fused LoRA "{lora}" into {type(module).__name__}.')
    module.set_adapters(names, weights)
    scale = 1.0 if fuse_scale is None else fuse_scale
    module.fuse_lora(lora_scale=scale, adapter_names=names)
    unload = getattr(module, 'unload_lora', None)
    if not callable(unload):
        raise _pipelines.UnsupportedPipelineConfigError(
            f'{type(module).__name__} kept its LoRA wrappers after fusing, '
            'so the merged weights cannot be stored on disk.')
    unload()


def _component_spec(spec) -> tuple[str, str] | None:
    if not isinstance(spec, (list, tuple)) or len(spec) != 2:
        return None
    library_name, class_name = spec
    if not isinstance(library_name, str) or not isinstance(class_name, str):
        return None
    return library_name, class_name


def _load_one_component(pipeline_class, name, library_name, class_name,
                        cached_folder, torch_dtype, model_variants):
    import diffusers.pipelines as diffusers_pipelines
    from diffusers.pipelines.pipeline_loading_utils import (
        ALL_IMPORTABLE_CLASSES,
        load_sub_model,
    )

    return load_sub_model(
        library_name=library_name,
        class_name=class_name,
        importable_classes=ALL_IMPORTABLE_CLASSES,
        pipelines=diffusers_pipelines,
        is_pipeline_module=hasattr(diffusers_pipelines, library_name),
        pipeline_class=pipeline_class,
        dtype=torch_dtype,
        provider=None,
        sess_options=None,
        device_map=None,
        max_memory=None,
        offload_folder=None,
        offload_state_dict=False,
        model_variants=model_variants,
        name=name,
        variant=None,
        low_cpu_mem_usage=True,
        cached_folder=cached_folder,
        use_safetensors=True,
        provider_options=None,
        disable_mmap=False,
        quantization_config=None,
        use_flashpack=False,
        trust_remote_code=False,
    )


def _offload_directory() -> str:
    parent = os.path.dirname(_pipelines.get_converted_checkpoint_cache_dir())
    root = os.path.join(parent, 'disk_offload')
    os.makedirs(root, exist_ok=True)
    return tempfile.mkdtemp(prefix='pipeline-', dir=root)


def _discard_directory(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _keep_disk_offload(pipeline, device):
    del pipeline, device
    _messages.debug_log('Pipeline weights stay on disk.')


def load_video_pipeline_low_ram(pipeline_class,
                                pretrained_repo: str,
                                load_kwargs: dict,
                                index: dict,
                                estimate: int,
                                device,
                                lora_uris,
                                lora_fuse_scale,
                                prepare_module=None,
                                weight_sources=None,
                                defer_lora_fusion: bool = False):
    """
    Load ``pipeline_class`` without holding every component in RAM at once.

    ``load_kwargs`` gains each component. LoRAs are fused into a component
    before that component's weights leave RAM. The returned pipeline reads
    those weights from disk during its forward pass.

    ``defer_lora_fusion`` is ``--lora-no-disk-fuse``. LoRA targets stay in
    memory and are not merged into a cached copy. The caller attaches those
    adapters after the pipeline exists.

    :param prepare_module: called with ``(name, module)`` after LoRAs are
        fused and before the weights are released. Used for the Wan VAE cast.
    :param weight_sources: component name to the checkpoint that supplied its
        weights. Replacement URIs belong here. Components omitted from the
        map are identified by the pipeline repository, revision, and variant.
    :param defer_lora_fusion: Keep LoRA targets resident instead of fusing them.
    """
    available = int(_memory.get_available_commit_memory('b'))
    notice = (
        f'This pipeline needs about {format_gib(estimate)} of CPU memory '
        f'and about {format_gib(available)} can be committed, including the page file. '
        'Each component is loaded, then its weights stay on disk and are '
        'copied to the device one layer at a time.'
    )
    if lora_uris and defer_lora_fusion:
        notice += (
            ' --lora-no-disk-fuse keeps LoRA targets in memory instead of '
            'writing a fused copy.'
        )
    elif lora_uris:
        notice += (
            ' LoRAs are merged into a component before it is released, and '
            'the merged weights are written to the dgenerate cache.'
        )
    _messages.warning(notice)
    if (lora_uris and not defer_lora_fusion
            and not hasattr(pipeline_class, 'load_lora_into_transformer')):
        raise _pipelines.UnsupportedPipelineConfigError(
            'This pipeline does not fit in CPU memory, and its LoRAs cannot be '
            'merged one component at a time. Free some memory, or load a smaller checkpoint.')

    token = load_kwargs.get('token')
    local_files_only = bool(load_kwargs.get('local_files_only', False))
    download_kwargs = {
        'local_files_only': local_files_only,
    }
    for key in ('revision', 'variant', 'subfolder', 'token'):
        if load_kwargs.get(key):
            download_kwargs[key] = load_kwargs[key]
    cached_folder = pipeline_class.download(pretrained_repo, **download_kwargs)
    from diffusers.pipelines.pipeline_loading_utils import _identify_model_variants
    model_variants = _identify_model_variants(
        cached_folder, load_kwargs.get('variant'), index or {})
    torch_dtype = load_kwargs.get('torch_dtype') or torch.float32
    execution_device = torch.device(device)
    offload_root = _offload_directory()
    weight_sources = weight_sources or {}
    try:
        def finish(name, module, reuse_folder, injected):
            if not isinstance(module, torch.nn.Module):
                return
            loras = _loras_for_component(name, lora_uris)
            if defer_lora_fusion and loras:
                # Packed weights cannot take a fused write, and a full-precision
                # fuse would be the cached copy --lora-no-disk-fuse skips.
                # The caller attaches the adapter after the pipeline is built.
                if prepare_module is not None:
                    prepare_module(name, module)
                _messages.debug_log(
                    f'Leaving "{name}" in memory so its LoRA stays active.')
                return
            if _pipelines.module_skips_group_offload(module):
                _messages.debug_log(
                    f'Leaving "{name}" in memory because its weights are packed.')
                return
            if loras:
                _fuse_loras(
                    pipeline_class, module, loras, lora_fuse_scale,
                    token, local_files_only)
            if prepare_module is not None:
                prepare_module(name, module)
            offload_module_weights(
                module,
                None if injected or loras else reuse_folder,
                os.path.join(offload_root, name),
                execution_device,
                dirty=bool(loras),
                cache_identity=(
                    pretrained_repo,
                    weight_sources.get(name) or _checkpoint_source(load_kwargs, name),
                    loras,
                    lora_fuse_scale,
                ),
            )

        for name, module in list(load_kwargs.items()):
            if _component_spec((index or {}).get(name)) is None:
                continue
            if not isinstance(module, torch.nn.Module):
                continue
            folder = os.path.join(cached_folder, name)
            finish(name, module, folder if os.path.isdir(folder) else None, True)

        for name, spec in (index or {}).items():
            parsed = _component_spec(spec)
            if parsed is None or name in load_kwargs:
                continue
            library_name, class_name = parsed
            _messages.log(f'Loading {name} ({class_name}).')
            module = _load_one_component(
                pipeline_class, name, library_name, class_name,
                cached_folder, torch_dtype, model_variants)
            load_kwargs[name] = module
            folder = os.path.join(cached_folder, name)
            finish(name, module, folder if os.path.isdir(folder) else None, False)
            _release_cpu_cache()

        pipe = pipeline_class.from_pretrained(pretrained_repo, **load_kwargs)
    except Exception:
        _discard_directory(offload_root)
        raise

    pipe._dgenerate_weights_on_disk = True
    pipe._dgenerate_disk_offload_dir = offload_root
    pipe._dgenerate_offload_device = execution_device
    _pipelines._set_sequential_cpu_offload_flag(pipe, True)
    for method in (
            'enable_sequential_cpu_offload',
            'enable_model_cpu_offload',
            'enable_group_offload'):
        if hasattr(pipe, method):
            _pipelines._bind_pipeline_offload_method(pipe, method, _keep_disk_offload)
    weakref.finalize(pipe, _discard_directory, offload_root)
    return pipe
