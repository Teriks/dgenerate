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

"""Video LoRA loading, and the fused-LoRA cache used with quantized components."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import pathlib
import random
import shutil
import tempfile

import filelock
import torch

import dgenerate.filecache as _filecache
import dgenerate.hfhub as _hfhub
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.pipelinewrapper.videopipelines as _vp
from .quantization import (
    _quantize_component,
    _quantized_device_map,
)


def _apply_video_loras(pipe, model_type, lora_uris, fuse_scale, auth_token, local_files_only):
    if not lora_uris:
        return
    del model_type
    _uris.LoRAUri.load_on_pipeline(
        pipeline=pipe,
        uris=lora_uris,
        fuse_scale=1.0 if fuse_scale is None else fuse_scale,
        auth_token=auth_token,
        local_files_only=local_files_only,
        fuse=True)


def _preflight_quantized_video_loras(
        pipeline_class, model_path, load_kwargs, injected, model_type, lora_uris,
        fuse_scale, auth_token, local_files_only, family, component_uris=(),
        components=()):
    """Serialize creation of each fused-LoRA cache entry across processes."""
    cache_root = pathlib.Path(_pipelines.get_converted_checkpoint_cache_dir()) / 'video_lora_fused'
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_identity = repr((
        model_path, str(model_type), load_kwargs.get('revision'),
        load_kwargs.get('variant'), load_kwargs.get('subfolder'),
        load_kwargs.get('torch_dtype'), tuple(lora_uris),
        1.0 if fuse_scale is None else fuse_scale,
        tuple(component_uris), tuple(components)))
    cache_hash = hashlib.sha256(cache_identity.encode('utf-8')).hexdigest()
    lock_root = cache_root / '.locks'
    lock_root.mkdir(parents=True, exist_ok=True)
    # Keep the lock file in place: unlinking a lock after releasing it can let
    # another process lock a new inode while a waiter still holds the old one.
    with filelock.FileLock(str(lock_root / f'{cache_hash}.lock')):
        return _preflight_quantized_video_loras_locked(
            pipeline_class, model_path, load_kwargs, injected, model_type,
            lora_uris, fuse_scale, auth_token, local_files_only, family,
            component_uris, components)


def _preflight_quantized_video_loras_locked(
        pipeline_class, model_path, load_kwargs, injected, model_type, lora_uris,
        fuse_scale, auth_token, local_files_only, family, component_uris=(),
        components=()):
    """Fuse video LoRAs and cache only the pipeline components they can affect."""
    if not lora_uris:
        return {}

    cache_root = pathlib.Path(_pipelines.get_converted_checkpoint_cache_dir()) / 'video_lora_fused'
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_identity = repr((
        model_path, str(model_type), load_kwargs.get('revision'),
        load_kwargs.get('variant'), load_kwargs.get('subfolder'),
        load_kwargs.get('torch_dtype'), tuple(lora_uris),
        1.0 if fuse_scale is None else fuse_scale,
        tuple(component_uris), tuple(components)))
    cache_hash = hashlib.sha256(cache_identity.encode('utf-8')).hexdigest()
    cache_store = _filecache.KeyValueStore(str(cache_root / 'cache.db'))
    cached_path = cache_store.get(cache_identity)
    if cached_path is not None:
        manifest_path = pathlib.Path(cached_path) / 'components.json'
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if not isinstance(manifest, dict) or not manifest:
                raise ValueError('LoRA component manifest must be a nonempty object.')
            if any(
                    not isinstance(name, str)
                    or name not in components
                    or not isinstance(info, dict)
                    or not isinstance(info.get('module'), str)
                    or not info.get('module')
                    or not isinstance(info.get('class'), str)
                    or not info.get('class')
                    for name, info in manifest.items()):
                raise ValueError('LoRA component manifest has invalid component metadata.')
            component_dirs = {
                path.name for path in pathlib.Path(cached_path).iterdir() if path.is_dir()}
            if component_dirs != set(manifest):
                raise ValueError('LoRA component manifest does not match cached component directories.')
            paths = {name: str(pathlib.Path(cached_path) / name) for name in manifest}
            if all(pathlib.Path(path).is_dir() for path in paths.values()):
                _messages.debug_log(f'Using cached LoRA-fused video components: {cached_path}')
                return paths, manifest
        except (OSError, ValueError):
            pass
        try:
            del cache_store[cache_identity]
        except KeyError:
            pass

    # Match the static component cache: SQLite is keyed by the full identity,
    # while the hash is only a preferred directory name. If that path already
    # belongs to another entry (including a hash collision), choose a unique one.
    with cache_store:
        cache_uuid = cache_hash
        fused_path = cache_root / cache_uuid
        while fused_path.exists():
            cache_uuid = hashlib.sha256(
                (cache_identity + str(random.random())).encode('utf-8')).hexdigest()
            fused_path = cache_root / cache_uuid

    temporary_path = pathlib.Path(tempfile.mkdtemp(prefix=f'{cache_hash}.', dir=cache_root))
    try:
        _messages.warning(
            f'Fusing LoRAs into unquantized video components and caching them at "{fused_path}". '
            'This is a one-time task for this model and LoRA scale.')
        with _hfhub.with_hf_errors_as_model_not_found():
            if str(family).startswith('wan-animate-2'):
                from . import wan
                pipeline = wan.load_wan_animate_2_pipeline(
                    pipeline_class, model_path, load_kwargs, injected, load_kwargs.get('torch_dtype'))
            else:
                pipeline = pipeline_class.from_pretrained(model_path, **load_kwargs)
        _vp._apply_video_loras(
            pipeline, model_type, lora_uris, fuse_scale, auth_token, local_files_only)
        # Pipelines advertise every component *capable* of loading a LoRA, but
        # a particular adapter can target only a subset (for example, LTX
        # adapters commonly leave ``connectors`` untouched).  Capture that
        # subset before unloading PEFT so the cache contains only fused
        # components and the regular checkpoint continues to supply the rest.
        affected_components = tuple(
            name for name in components
            if getattr(getattr(pipeline, name, None), 'peft_config', None))
        unload_lora_weights = getattr(pipeline, 'unload_lora_weights', None)
        if not callable(unload_lora_weights):
            raise _pipelines.UnsupportedPipelineConfigError(
                f'{pipeline_class.__name__} cannot unload fused LoRA adapters before caching.')
        # fuse_lora merges the adapter deltas into the base weights, but leaves
        # PEFT wrappers in place. Save the merged base modules without those
        # wrappers so from_pretrained sees the checkpoint's expected keys.
        unload_lora_weights()
        manifest = {}
        for name in affected_components:
            module = getattr(pipeline, name, None)
            # Diffusers pipeline classes can advertise optional LoRA targets
            # (for example Wan's transformer_2 for MoE checkpoints) even when
            # the selected checkpoint does not provide that component.
            if module is None:
                continue
            if not hasattr(module, 'save_pretrained'):
                raise _pipelines.UnsupportedPipelineConfigError(
                    f'Cannot cache LoRA-affected video component "{name}" before quantization.')
            if getattr(module, 'peft_config', None):
                raise _pipelines.UnsupportedPipelineConfigError(
                    f'LoRA adapters remain attached to video component "{name}" after unloading.')
            component_path = temporary_path / name
            from dgenerate.pipelinewrapper.weightsave import save_module_with_progress
            save_module_with_progress(
                module, str(component_path), desc=f'Writing {name}')
            manifest[name] = {
                'module': type(module).__module__,
                'class': type(module).__name__,
            }
        if not manifest:
            raise _pipelines.UnsupportedPipelineConfigError(
                f'{pipeline_class.__name__} exposes no saveable LoRA components for quantization preflight.')
        (temporary_path / 'components.json').write_text(
            json.dumps(manifest, indent=2), encoding='utf-8')
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if fused_path.exists():
            if fused_path.is_dir():
                shutil.rmtree(fused_path)
            else:
                fused_path.unlink()
        temporary_path.replace(fused_path)
        cache_store[cache_identity] = str(fused_path)
    except Exception:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise
    return ({name: str(fused_path / name) for name in manifest}, manifest)


def _load_fused_video_component(name, component_path, component_info, dtype,
                                quantizer_uri, auth_token,
                                local_files_only, device, offload):
    module_class = getattr(
        importlib.import_module(component_info['module']), component_info['class'])
    if quantizer_uri:
        return _vp._load_quantized_module(
            module_class, component_path, None, None, None, dtype, quantizer_uri,
            auth_token, True, _quantized_device_map(device, offload, quantizer_uri),
            offload=offload)

    kwargs = {'local_files_only': True}
    torch_dtype = _enums.get_torch_dtype(dtype)
    if torch_dtype is not None:
        if getattr(module_class, '__module__', '').startswith('transformers'):
            kwargs['dtype'] = torch_dtype
        else:
            kwargs['torch_dtype'] = torch_dtype
    with _hfhub.with_hf_errors_as_model_not_found():
        module = module_class.from_pretrained(component_path, **kwargs)
    if offload:
        module.to('cpu')
    else:
        module.to(device)
    return module


def _video_lora_component_quantizers(
        pipeline_class, quantizer_uri, quantizer_map, transformer_uri,
        wan_second_transformer_uri, text_encoder_slots, text_encoder_uris):
    """Resolve quantizers for LoRA-loadable video components and replacements."""
    lora_components = tuple(getattr(
        pipeline_class, '_lora_loadable_modules', ('transformer',)))
    component_quantizers = {
        name: quantizer_uri
        for name in lora_components
        if _quantize_component(name, quantizer_uri, quantizer_map)
    }
    for name, uri in (
            ('transformer', transformer_uri),
            ('transformer_2', wan_second_transformer_uri)):
        if name in lora_components and uri:
            parsed = _uris.TransformerUri.parse(uri)
            if parsed.quantizer:
                component_quantizers[name] = parsed.quantizer
    for index, name in enumerate(text_encoder_slots):
        if name not in lora_components or index >= len(text_encoder_uris):
            continue
        uri = text_encoder_uris[index]
        if _pipelines._text_encoder_default(uri):
            continue
        if _pipelines._text_encoder_null(uri):
            component_quantizers.pop(name, None)
            continue
        parsed = _uris.TextEncoderUri.parse(uri)
        if parsed.quantizer:
            component_quantizers[name] = parsed.quantizer
        else:
            # Global --quantizer applies to the checkpoint's default encoder,
            # not to a replacement supplied through --text-encoders.
            component_quantizers.pop(name, None)
    return lora_components, component_quantizers


def _video_lora_component_uri_cache_key(
        lora_components, transformer_uri, wan_second_transformer_uri,
        text_encoder_slots, text_encoder_uris):
    """Build a LoRA cache key from component sources, excluding quantizers."""
    component_uris = []
    for name, uri in (
            ('transformer', transformer_uri),
            ('transformer_2', wan_second_transformer_uri)):
        if name not in lora_components or not uri:
            continue
        parsed = _uris.TransformerUri.parse(uri)
        component_uris.append((
            name, parsed.model, parsed.revision, parsed.variant,
            parsed.subfolder, parsed.dtype))
    for index, name in enumerate(text_encoder_slots):
        if name not in lora_components or index >= len(text_encoder_uris):
            continue
        uri = text_encoder_uris[index]
        if (_pipelines._text_encoder_default(uri)
                or _pipelines._text_encoder_null(uri)):
            continue
        parsed = _uris.TextEncoderUri.parse(uri)
        component_uris.append((
            name, parsed.encoder, parsed.model, parsed.revision,
            parsed.variant, parsed.subfolder, parsed.dtype, parsed.mode))
    return tuple(component_uris)


# Every pipeline dgenerate loads for a video ``--model-type``.
_VIDEO_LORA_PIPELINE_CLASSES = (
    'LTX2Pipeline',
    'LTX2ImageToVideoPipeline',
    'LTX2ConditionPipeline',
    'LTX2InContextPipeline',
    'LTXPipeline',
    'LTXImageToVideoPipeline',
    'LTXConditionPipeline',
    'WanPipeline',
    'WanImageToVideoPipeline',
    'WanVideoToVideoPipeline',
    'WanVACEPipeline',
    'WanAnimatePipeline',
    'WanAnimate2ModularPipeline',
    'WanAnimate2DistilledModularPipeline',
    'HunyuanVideo15Pipeline',
    'HunyuanVideo15ImageToVideoPipeline',
    'MotifVideoPipeline',
    'MotifVideoImage2VideoPipeline',
    'Kandinsky5T2VPipeline',
    'Kandinsky5I2VPipeline',
)


def _peft_adapter_header(metadata):
    """The parsed PEFT adapter config, or ``None`` when metadata is not one."""
    if not isinstance(metadata, dict) or not metadata:
        return None
    if any(torch.is_tensor(value) for value in metadata.values()):
        return None
    if 'lora_adapter_metadata' in metadata:
        raw = metadata['lora_adapter_metadata']
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        return parsed if isinstance(parsed, dict) else None
    if any(key in metadata for key in ('peft_type', 'target_modules', 'lora_alpha', 'r')):
        return metadata
    return None


def _scale_lora_b(state, header):
    """Fold ``lora_alpha / r`` into ``lora_B``. Rank-equal adapters are unchanged."""
    alpha = header.get('lora_alpha')
    rank = header.get('r')
    alpha_pattern = header.get('alpha_pattern') or {}
    rank_pattern = header.get('rank_pattern') or {}
    if not alpha_pattern and not rank_pattern and (
            alpha is None or not rank or float(alpha) == float(rank)):
        return state
    scaled = {}
    for key, value in state.items():
        if 'lora_B' in key and getattr(value, 'ndim', 0) > 1:
            module = key.split('.lora_B', 1)[0].removeprefix('transformer.')
            module_rank = float(rank_pattern.get(module, rank or value.shape[1]))
            module_alpha = float(alpha_pattern.get(
                module, alpha if alpha is not None else module_rank))
            scale = module_alpha / module_rank if module_rank else 1.0
            scaled[key] = value * scale if scale != 1.0 else value
        else:
            scaled[key] = value
    return scaled


def _guard_video_lora_state_dict(pipeline_class):
    """Drop PEFT config headers before Diffusers builds ``LoraConfig``.

    Diffusers keeps only metadata keys prefixed with ``transformer.``. A PEFT
    header (``r``, ``lora_alpha``, ``target_modules``) becomes ``{}``, and PEFT
    then looks up ``model_type`` on a transformer config that does not have it.
    """
    if not callable(getattr(pipeline_class, 'lora_state_dict', None)):
        return
    current = getattr(pipeline_class.lora_state_dict, '__func__', None)
    if getattr(current, '_dgenerate_video_lora_guard', False):
        return
    original = pipeline_class.lora_state_dict.__func__

    def lora_state_dict(cls, pretrained_model_name_or_path_or_dict, **kwargs):
        want_metadata = kwargs.pop('return_lora_metadata', False)
        kwargs['return_lora_metadata'] = True
        loaded = original(cls, pretrained_model_name_or_path_or_dict, **kwargs)
        if isinstance(loaded, tuple):
            state, metadata = loaded[0], loaded[1] if len(loaded) > 1 else None
        else:
            state, metadata = loaded, None
        header = _peft_adapter_header(metadata)
        if header is not None:
            state, metadata = _scale_lora_b(state, header), None
        if want_metadata:
            return state, metadata
        return state

    lora_state_dict._dgenerate_video_lora_guard = True
    pipeline_class.lora_state_dict = classmethod(lora_state_dict)


def install_video_lora_support() -> None:
    """Install Motif-Video LoRA loading and the shared metadata guard."""
    from .motif_lora import install_motif_video_lora_loader
    from .hyvideo_lora import install_hunyuan_video15_lora_loader

    install_hunyuan_video15_lora_loader()
    install_motif_video_lora_loader()
    import diffusers
    for name in _VIDEO_LORA_PIPELINE_CLASSES:
        pipeline_class = getattr(diffusers, name, None)
        if pipeline_class is not None:
            _guard_video_lora_state_dict(pipeline_class)


install_video_lora_support()
