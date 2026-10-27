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

"""Video pipeline loading, caching, and placement."""

from __future__ import annotations

import inspect

import os

import dgenerate.hfhub as _hfhub
import dgenerate.memoize as _d_memoize
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.singlefilecompat as _singlefilecompat
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.pipelinewrapper.util as _util
from dgenerate.memoize import memoize as _memoize
from .ltx.conditions import _ic_lora_downscale_factor
from .single_image_video import (
    SINGLE_IMAGE_VIDEO_FAMILIES,
    SINGLE_IMAGE_VIDEO_MODEL_TYPES,
    _resolve_single_image_video_load,
    _single_image_video_pipeline_class,
    _single_image_video_transformer_class,
)
import dgenerate.pipelinewrapper.videopipelines as _vp
from .quantization import (
    _LTX_SKIP_OPTIONAL_MODULES,
    _pin_video_8bit_modules,
    _quantize_component,
    _quantized_classic_modules,
    _quantized_device_map,
)
from .lora import (
    _load_fused_video_component,
    _preflight_quantized_video_loras,
    _video_lora_component_quantizers,
    _video_lora_component_uri_cache_key,
)


_LTX_FALLBACK_EXTRA_DIRS = ('audio_vae', 'vocoder')


_VIDEO_CORE_INDEX_NAMES = frozenset({
    'transformer',
    'transformer_2',
    'unet',
    'vae',
    'text_encoder',
    'text_encoder_2',
    'text_encoder_3',
    'image_encoder',
    'scheduler',
    'tokenizer',
    'tokenizer_2',
    'tokenizer_3',
    'feature_extractor',
    'safety_checker',
})


def extra_weight_directories_from_index(index: dict | None) -> list[str]:
    """
    Repo folders that belong on the LTX pipeline cache estimate besides
    transformer, VAE, and text encoders.

    :param index: ``model_index.json`` dict, or ``None``
    :return: folder names such as ``audio_vae`` and ``vocoder``
    """
    if not index:
        return list(_LTX_FALLBACK_EXTRA_DIRS)
    extras = []
    for name, spec in index.items():
        if name.startswith('_') or name in _VIDEO_CORE_INDEX_NAMES:
            continue
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            extras.append(name)
    return extras


class _VideoPipeline:
    """A video pipeline kept in the shared diffusion pipeline cache."""

    def __init__(self, pipeline, family: str, moe: bool = False, model_mode: str | None = None):
        self.pipeline = pipeline
        self.family = family
        self.moe = moe
        self.model_mode = model_mode
        self.reference_downscale_factor = None


def _pretrained_kwargs(revision, variant, subfolder, local_files_only, auth_token, dtype, include_dtype):
    kwargs = {'local_files_only': local_files_only}
    if revision:
        kwargs['revision'] = revision
    if variant:
        kwargs['variant'] = variant
    if subfolder:
        kwargs['subfolder'] = subfolder
    if auth_token:
        kwargs['token'] = auth_token
    if include_dtype:
        torch_dtype = _enums.get_torch_dtype(dtype)
        if torch_dtype is not None:
            kwargs['torch_dtype'] = torch_dtype
    return kwargs


def _enable_vae_tiling(pipe):
    vae = getattr(pipe, 'vae', None)
    if vae is not None and hasattr(vae, 'enable_tiling'):
        vae.enable_tiling()


def _disable_rope_double_precision_on_mps(pipe, device):
    """
    Make LTX2 rotary embeddings build their frequencies in float32 on MPS.

    The LTX2 checkpoints ship ``rope_double_precision: true``, and diffusers
    then calls ``torch.linspace(..., dtype=torch.float64, device=device)`` in
    the connectors and the transformer. MPS has no float64 support, so that
    raises ``TypeError: Cannot convert a MPS Tensor to float64 dtype``.

    The frequencies are cast to float32 right after they are built, so this
    only drops the float64 intermediate. ``double_precision`` is a plain
    attribute read inside ``forward``, so it can be switched off after loading.

    :param pipe: freshly loaded video pipeline
    :param device: execution device (string or ``torch.device``)
    """
    if str(device).split(':')[0] != 'mps':
        return
    patched = 0
    for component_name in ('connectors', 'transformer'):
        component = getattr(pipe, component_name, None)
        if component is None or not hasattr(component, 'modules'):
            continue
        for module in component.modules():
            if ('RotaryPosEmbed' in type(module).__name__
                    and getattr(module, 'double_precision', False)):
                module.double_precision = False
                patched += 1
    if patched:
        _messages.debug_log(
            f'MPS has no float64 support, disabled rope_double_precision '
            f'on {patched} rotary embedding module(s).')


def _apply_video_offload(pipe, device, model_cpu_offload, sequential_cpu_offload,
                         model_group_offload=False):
    if getattr(pipe, '_dgenerate_weights_on_disk', False):
        _messages.debug_log(
            'Pipeline weights are on disk; leaving the disk hooks in place.')
        return
    if sequential_cpu_offload:
        _pipelines.enable_sequential_cpu_offload(pipe, device)
    elif model_cpu_offload:
        _pipelines.enable_model_cpu_offload(pipe, device)
    elif model_group_offload:
        _pipelines.enable_group_offload(pipe, device)


def _video_transformer_class(model_type, family: str = 'ltx2'):
    model_type = _enums.get_model_type_enum(model_type)
    if _enums.model_type_is_ltx_family(model_type):
        if family == 'ltx':
            from diffusers import LTXVideoTransformer3DModel
            return LTXVideoTransformer3DModel
        from diffusers import LTX2VideoTransformer3DModel
        return LTX2VideoTransformer3DModel
    if _enums.model_type_is_wan_family(model_type):
        from . import wan
        return wan._wan_transformer_class(model_type, family)
    if model_type in SINGLE_IMAGE_VIDEO_MODEL_TYPES:
        return _single_image_video_transformer_class(family)
    raise _pipelines.UnsupportedPipelineConfigError(
        f'{_enums.get_model_type_string(model_type)} does not take --transformer.')


def _load_replacement_transformer(model_type, transformer_uri, dtype, variant,
                                  auth_token, local_files_only, quantizer_uri=None,
                                  quantizer_map=None, device=None, offload=False,
                                  component_name='transformer', config_repo=None,
                                  family: str = 'ltx2', force_unquantized=False):
    if not isinstance(dtype, _enums.DataType):
        dtype = _enums.DataType.AUTO
    parsed = _uris.TransformerUri.parse(transformer_uri)
    if (not _hfhub.is_gguf_model(parsed.model)
            and not force_unquantized
            and quantizer_uri and not parsed.quantizer
            and _quantize_component(component_name, quantizer_uri, quantizer_map)):
        parsed.quantizer = quantizer_uri
    if force_unquantized:
        if _hfhub.is_gguf_model(parsed.model):
            raise _pipelines.UnsupportedPipelineConfigError(
                f'Cannot fuse LoRAs into quantized GGUF replacement component "{component_name}".')
        parsed.quantizer = None
    _messages.debug_log(f'Loading replacement video transformer "{transformer_uri}".')
    return parsed.load(
        variant_fallback=variant,
        dtype_fallback=dtype,
        auth_token=auth_token,
        local_files_only=local_files_only,
        device_map=_quantized_device_map(
            device, offload, parsed.quantizer) if parsed.quantizer else None,
        config=config_repo,
        config_subfolder=component_name,
        model_type=model_type,
        # LoRAs are fused into this copy, which packed ComfyUI weights cannot take.
        dequantize=force_unquantized,
        transformer_class=_video_transformer_class(model_type, family))


def pipeline_for_mode(pipe, mode: str, family: str = 'ltx2'):
    """
    Return a video pipeline of the requested class that shares ``pipe.components``.

    ``from_pipe`` is not used: it calls ``.to(dtype=...)``, which raises on
    quantized modules (SDNQ / bitsandbytes).

    :param pipe: cached video pipeline
    :param mode: a pipeline mode such as ``ltx-image``, ``wan-i2v``,
        ``text-to-video``, or ``image-to-video``
    :return: pipeline of the class for ``mode``
    """
    if family in SINGLE_IMAGE_VIDEO_FAMILIES:
        cls = _single_image_video_pipeline_class(family, mode)
    elif str(mode).startswith('wan'):
        from . import wan
        if str(family).startswith('wan-animate-2'):
            wan._require_image_encoder(pipe, 'wan-animate-2')
            return pipe
        cls = wan._wan_pipeline_class(mode, family)
        wan._require_image_encoder(pipe, mode)
    else:
        cls = _vp._ltx_pipeline_class(mode, family)
    if type(pipe) is cls:
        return pipe
    accepted = set(inspect.signature(cls.__init__).parameters)
    accepted.discard('self')
    kwargs = {name: value for name, value in pipe.components.items() if name in accepted}
    return cls(**kwargs)


def _ltx_extra_weight_directories(model_path, revision, subfolder, auth_token, local_files_only):
    try:
        index = _util.fetch_model_index_dict(
            model_path,
            subfolder=subfolder,
            revision=revision,
            auth_token=auth_token,
            local_files_only=local_files_only)
    except Exception:
        return list(_LTX_FALLBACK_EXTRA_DIRS)
    return extra_weight_directories_from_index(index)


def _torch_compile_for_cache(wrapper) -> bool:
    """
    Cache identity for block compilation.

    Wan-Animate-2 compiles on every load, so its cache entry is compiled
    whether or not ``--torch-compile`` was passed. ``DGENERATE_TORCH_COMPILE=0``
    leaves that entry eager. Other models compile only when the flag is set.
    """
    if _pipelines.torch_compile_disabled():
        return False
    model_type = getattr(wrapper, 'model_type', None)
    if model_type == _enums.ModelType.WAN_ANIMATE_2:
        return True
    if isinstance(model_type, str):
        try:
            model_type = _enums.get_model_type_enum(model_type)
        except ValueError:
            model_type = None
    if model_type == _enums.ModelType.WAN_ANIMATE_2:
        return True
    return bool(getattr(wrapper, 'torch_compile', False))


def _cache_kwargs(wrapper) -> dict:
    quantizer_map = wrapper.quantizer_map
    return {
        'model_path': wrapper.model_path,
        'model_type': wrapper.model_type,
        'revision': wrapper._revision,
        'variant': wrapper._variant,
        'subfolder': wrapper._subfolder,
        'dtype': wrapper._dtype,
        'device': wrapper.device,
        'model_cpu_offload': bool(wrapper.model_cpu_offload),
        'sequential_cpu_offload': bool(wrapper.model_sequential_offload),
        'model_group_offload': bool(getattr(wrapper, 'model_group_offload', False)),
        'torch_compile': _torch_compile_for_cache(wrapper),
        'local_files_only': bool(wrapper._local_files_only),
        'auth_token': wrapper._auth_token,
        'quantizer_uri': wrapper.quantizer_uri,
        'quantizer_map': tuple(quantizer_map) if quantizer_map else None,
    }


def _video_on_hit(key, hit):
    _d_memoize.simple_cache_hit_debug('Torch Video Pipeline', key, hit.pipeline)


def _video_on_create(key, new):
    _d_memoize.simple_cache_miss_debug('Torch Video Pipeline', key, new.pipeline)


def _resolve_video_load(model_type, index, ltx_ic_lora_uri):
    model_type = _enums.get_model_type_enum(model_type)
    if _enums.model_type_is_wan_family(model_type):
        from . import wan
        return wan._resolve_wan_load(model_type, index)
    if model_type in SINGLE_IMAGE_VIDEO_MODEL_TYPES:
        return _resolve_single_image_video_load(model_type, index)
    from . import ltx
    family = ltx.ltx_family_for_model_type(model_type, index)
    mode = 'ltx-control' if ltx_ic_lora_uri else 'ltx-txt'
    return family, _vp._ltx_pipeline_class(mode, family), False


def _text_encoder_slots(index: dict | None) -> list[str]:
    """Ordered ``text_encoder*`` component names from ``model_index.json``."""
    if not index:
        return []
    return [
        name for name, spec in sorted(index.items())
        if _pipelines._is_text_encoder_component(name)
        and isinstance(spec, (list, tuple))
        and len(spec) >= 2
        and spec[0] is not None
    ]


def _inject_video_text_encoders(index,
                                text_encoder_uris,
                                dtype,
                                auth_token,
                                local_files_only,
                                device,
                                offload,
                                force_unquantized=()):
    """
    Load ``--text-encoders`` replacements for a video pipeline.

    ``+`` keeps the checkpoint default for that slot. ``null`` skips loading
    that encoder. Trailing omitted slots keep their defaults.
    """
    if not text_encoder_uris:
        return {}
    slots = _text_encoder_slots(index)
    if not slots:
        raise _pipelines.UnsupportedPipelineConfigError(
            '--text-encoders cannot be used with this video model; '
            'the checkpoint has no text encoder.')
    uris = list(text_encoder_uris)
    if len(uris) > len(slots):
        raise _pipelines.UnsupportedPipelineConfigError(
            f'Too many --text-encoders values for this video model '
            f'(got {len(uris)}, max {len(slots)}).')
    injected = {}
    for name, uri in zip(slots, uris):
        if _pipelines._text_encoder_default(uri):
            continue
        if _pipelines._text_encoder_null(uri):
            injected[name] = None
            _messages.debug_log(f'Video pipeline skipping "{name}" (--text-encoders null).')
            continue
        parsed = _uris.TextEncoderUri.parse(uri)
        if name in force_unquantized:
            parsed.quantizer = None
        # Global --quantizer only applies when the slot is left as +.
        # Put quantizer= on the URI to quantize a replacement encoder.
        device_map = (
            _quantized_device_map(device, offload, parsed.quantizer)
            if parsed.quantizer else None)
        injected[name] = parsed.load(
            dtype_fallback=dtype,
            auth_token=auth_token,
            local_files_only=local_files_only,
            device_map=device_map)
        _messages.debug_log(f'Video pipeline using --text-encoders for "{name}": "{uri}".')
    return injected


@_memoize(_pipelines._pipeline_cache,
          exceptions={'local_files_only'},
          hasher=_d_memoize.args_cache_key,
          extra_identities=[lambda held: held.pipeline],
          on_hit=_video_on_hit,
          on_create=_video_on_create)
def _create_cached_video_pipeline(model_path,
                                  model_type,
                                  revision,
                                  variant,
                                  subfolder,
                                  dtype,
                                  device,
                                  model_cpu_offload,
                                  sequential_cpu_offload,
                                  model_group_offload,
                                  torch_compile,
                                  local_files_only,
                                  auth_token,
                                  transformer_uri=None,
                                  lora_uris=None,
                                  lora_fuse_scale=None,
                                  quantizer_uri=None,
                                  quantizer_map=None,
                                  ltx_ic_lora_uri=None,
                                  ic_lora_downscale=None,
                                  wan_second_transformer_uri=None,
                                  vae_uri=None,
                                  text_encoder_uris=None):
    all_lora_uris = list(lora_uris or ()) + ([ltx_ic_lora_uri] if ltx_ic_lora_uri else [])
    aux_repo = None
    if _hfhub.is_single_file_model_load(model_path):
        aux_repo = _singlefilecompat.single_file_pretrained_repo_for_model_path(
            model_path,
            auth_token=auth_token,
            local_files_only=local_files_only,
            model_type=model_type)
    if aux_repo and subfolder:
        raise _pipelines.UnsupportedPipelineConfigError(
            'Single file model loads do not support the subfolder option.')
    # Transformer-only files use the detected repo for encoders, VAE, and
    # tokenizers. The file itself is the transformer unless --transformer
    # names a different one. Repository plus --transformer is unchanged.
    pretrained_repo = aux_repo or model_path
    load_subfolder = None if aux_repo else subfolder
    if aux_repo and not transformer_uri:
        transformer_uri = model_path
        _messages.debug_log(
            f'Single file transformer checkpoint; loading pipeline components from "{aux_repo}".')
    index = _util.fetch_model_index_dict(
        model_path,
        subfolder=load_subfolder,
        revision=revision,
        auth_token=auth_token,
        local_files_only=local_files_only,
        model_type=model_type)
    extra_dirs = extra_weight_directories_from_index(index)
    family, pipeline_class, moe = _resolve_video_load(model_type, index, ltx_ic_lora_uri)
    if transformer_uri and wan_second_transformer_uri is None and index.get('transformer_2'):
        extra_dirs = list(extra_dirs) + ['transformer_2']
    if dtype is _enums.DataType.AUTO:
        detected_dtype = _util.auto_dtype(
            model_path,
            revision=revision,
            subfolder=subfolder,
            auth_token=auth_token,
            local_files_only=local_files_only,
            model_index=index,
            model_type=model_type)
        if detected_dtype is not None:
            _messages.debug_log(f'--dtype auto selected: {_enums.get_data_type_string(detected_dtype)}')
            dtype = detected_dtype
    te_slots = _text_encoder_slots(index)
    te_uris = list(text_encoder_uris or ())
    lora_components, lora_component_quantizers = _video_lora_component_quantizers(
        pipeline_class, quantizer_uri, quantizer_map, transformer_uri,
        wan_second_transformer_uri, te_slots, te_uris)
    preflight_loras = bool(all_lora_uris and lora_component_quantizers)
    all_text_encoders_replaced = bool(
        te_slots
        and len(te_uris) >= len(te_slots)
        and all(
            not _pipelines._text_encoder_default(uri)
            for uri in te_uris[:len(te_slots)]))
    estimate_directories = extra_dirs
    if family == 'ltx2':
        # The Gemma prompt enhancer is passed as None and is not loaded.
        estimate_directories = [
            name for name in extra_dirs if name not in _LTX_SKIP_OPTIONAL_MODULES]
    estimate = _pipelines.estimate_pipeline_cache_footprint(
        model_path=pretrained_repo,
        model_type=model_type,
        revision=revision or 'main',
        variant=variant,
        subfolder=load_subfolder,
        include_unet_or_transformer=not transformer_uri and not aux_repo,
        include_vae=not vae_uri,
        include_text_encoders=not all_text_encoders_replaced,
        lora_uris=all_lora_uris,
        include_directories=estimate_directories,
        auth_token=auth_token,
        local_files_only=local_files_only)
    if aux_repo and os.path.isfile(model_path):
        estimate += os.path.getsize(model_path)
    _pipelines._enforce_pipeline_cache_size(estimate)

    load_kwargs = _pretrained_kwargs(
        revision, variant, load_subfolder, local_files_only, auth_token, dtype,
        include_dtype=True)
    if family == 'ltx2':
        load_kwargs.update(_LTX_SKIP_OPTIONAL_MODULES)

    offload = bool(model_cpu_offload or sequential_cpu_offload or model_group_offload)
    injected = {}
    if transformer_uri:
        force_unquantized = preflight_loras and 'transformer' in lora_component_quantizers
        injected['transformer'] = _load_replacement_transformer(
            model_type, transformer_uri, dtype, variant, auth_token, local_files_only,
            quantizer_uri=quantizer_uri, quantizer_map=quantizer_map, device=device,
            offload=offload, config_repo=pretrained_repo, family=family,
            force_unquantized=force_unquantized)
        load_kwargs['transformer'] = injected['transformer']
    if wan_second_transformer_uri:
        force_unquantized = preflight_loras and 'transformer_2' in lora_component_quantizers
        injected['transformer_2'] = _load_replacement_transformer(
            model_type, wan_second_transformer_uri, dtype, variant, auth_token, local_files_only,
            quantizer_uri=quantizer_uri, quantizer_map=quantizer_map, device=device,
            offload=offload, component_name='transformer_2', config_repo=pretrained_repo,
            family=family, force_unquantized=force_unquantized)
        load_kwargs['transformer_2'] = injected['transformer_2']
    if vae_uri:
        parsed_vae = _uris.VAEUri.parse(vae_uri)
        # The Wan VAE defaults to float32 even when replaced via --vae, unless
        # the URI sets dtype= explicitly.
        vae_dtype_fallback = dtype
        if (_enums.model_type_uses_wan_vae(model_type)
                and parsed_vae.dtype is None):
            vae_dtype_fallback = _constants.DEFAULT_WAN_VAE_DTYPE
        injected['vae'] = parsed_vae.load(
            dtype_fallback=vae_dtype_fallback,
            auth_token=auth_token,
            local_files_only=local_files_only)
        load_kwargs['vae'] = injected['vae']
        _messages.debug_log(f'Video pipeline using --vae "{vae_uri}".')
    injected.update(_inject_video_text_encoders(
        index, te_uris, dtype, auth_token, local_files_only, device, offload,
        force_unquantized=(lora_component_quantizers if preflight_loras else ())))
    load_kwargs.update(injected)
    fused_lora_components = {}
    if preflight_loras:
        fused_lora_cache = _preflight_quantized_video_loras(
            pipeline_class, pretrained_repo, load_kwargs, injected, model_type,
            all_lora_uris, lora_fuse_scale, auth_token, local_files_only, family,
            component_uris=_video_lora_component_uri_cache_key(
                lora_components, transformer_uri, wan_second_transformer_uri,
                te_slots, te_uris),
            components=lora_components)
        fused_lora_components, fused_lora_manifest = fused_lora_cache
        for component_name, component_path in fused_lora_components.items():
            load_kwargs.pop(component_name, None)
            injected.pop(component_name, None)
            injected[component_name] = _load_fused_video_component(
                component_name, component_path, fused_lora_manifest[component_name],
                dtype, lora_component_quantizers.get(component_name),
                auth_token, local_files_only,
                device, offload)
        load_kwargs.update(injected)
    injected.update(_quantized_classic_modules(
        pretrained_repo, revision, variant, load_subfolder, dtype, quantizer_uri, quantizer_map,
        auth_token, local_files_only, device, offload,
        skip_names=frozenset(load_kwargs)))
    load_kwargs.update(injected)
    _messages.debug_log(f'Loading {pipeline_class.__name__} from "{pretrained_repo}".')
    from . import diskoffload as _diskoffload
    replaced = set()
    if transformer_uri:
        replaced.add('transformer')
    if wan_second_transformer_uri:
        replaced.add('transformer_2')
    if vae_uri:
        replaced.add('vae')
    missing = _diskoffload.uncounted_component_names(
        index, extra_dirs, load_kwargs, replaced)
    footprint = estimate
    if missing:
        footprint += _util.estimate_model_memory_use(
            repo_id=pretrained_repo,
            revision=revision or 'main',
            variant=variant,
            subfolder=load_subfolder,
            include_unet_or_transformer=False,
            include_vae=False,
            include_text_encoder=False,
            include_text_encoder_2=False,
            include_text_encoder_3=False,
            include_directories=missing,
            auth_token=auth_token,
            local_files_only=local_files_only)
    stream_from_disk = (
        not str(family).startswith('wan-animate-2')
        and _diskoffload.should_stream_weights_from_disk(
            footprint,
            quantizer_uri=quantizer_uri,
            quantizer_map=quantizer_map,
            model_path=model_path,
            transformer_uri=transformer_uri,
            second_transformer_uri=wan_second_transformer_uri,
            vae_uri=vae_uri))
    with _hfhub.with_hf_errors_as_model_not_found():
        if str(family).startswith('wan-animate-2'):
            from . import wan
            pipe = wan.load_wan_animate_2_pipeline(
                pipeline_class, pretrained_repo, load_kwargs, injected, dtype)
        elif stream_from_disk:
            def prepare_module(name, module, _vae_uri=vae_uri, _model_type=model_type):
                if name != 'vae' or _vae_uri or not _enums.model_type_uses_wan_vae(_model_type):
                    return
                from . import wan
                wan._set_wan_vae_dtype(type('_VaeHost', (), {'vae': module})())

            weight_sources = {}
            if transformer_uri:
                weight_sources['transformer'] = str(transformer_uri)
            if wan_second_transformer_uri:
                weight_sources['transformer_2'] = str(wan_second_transformer_uri)
            if vae_uri:
                weight_sources['vae'] = str(vae_uri)
            for slot, uri in zip(te_slots, te_uris):
                if (uri
                        and not _pipelines._text_encoder_default(uri)
                        and not _pipelines._text_encoder_null(uri)):
                    weight_sources[slot] = str(uri)
            for component_name, component_path in fused_lora_components.items():
                weight_sources[component_name] = str(component_path)
            pipe = _diskoffload.load_video_pipeline_low_ram(
                pipeline_class, pretrained_repo, load_kwargs, index, footprint, device,
                all_lora_uris, lora_fuse_scale, prepare_module=prepare_module,
                weight_sources=weight_sources)
        else:
            pipe = pipeline_class.from_pretrained(pretrained_repo, **load_kwargs)
    if not fused_lora_components and not stream_from_disk:
        _vp._apply_video_loras(
            pipe, model_type, all_lora_uris, lora_fuse_scale, auth_token, local_files_only)
    _disable_rope_double_precision_on_mps(pipe, device)
    # The Wan VAE (also used by Motif-Video) defaults to float32. Cast before
    # offload so streamed group offload snapshots float32 weights. Skip when
    # --vae already chose a dtype (including the float32 fallback above), or
    # when the VAE is quantized. The disk loader casts before it releases the VAE.
    if _enums.model_type_uses_wan_vae(model_type) and not vae_uri and not stream_from_disk:
        from . import wan
        wan._set_wan_vae_dtype(pipe)
    if str(family).startswith('wan-animate-2'):
        from . import wan
        # Compiles on every load, before the offload hooks wrap forward.
        wan.place_wan_animate_2(
            pipe, device, model_cpu_offload, sequential_cpu_offload, model_group_offload)
    else:
        if torch_compile and stream_from_disk:
            _messages.warning(
                'Skipping --torch-compile because the pipeline weights are on disk.')
        elif torch_compile:
            _pipelines.apply_torch_compile(pipe, device)
        _apply_video_offload(
            pipe, device, model_cpu_offload, sequential_cpu_offload, model_group_offload)
    _pin_video_8bit_modules(pipe)
    _enable_vae_tiling(pipe)
    model_mode = None
    if family == 'kandinsky5-video':
        model_mode = 'image-to-video' if isinstance(pipe, _single_image_video_pipeline_class(
            family, 'image-to-video')) else 'text-to-video'
    held = _VideoPipeline(pipe, family, moe=moe, model_mode=model_mode)
    if ltx_ic_lora_uri:
        held.reference_downscale_factor = _ic_lora_downscale_factor(
            ltx_ic_lora_uri, ic_lora_downscale, auth_token, local_files_only)

    cache_size = _diskoffload.DISK_PIPELINE_CACHE_BYTES if stream_from_disk else estimate
    return held, _d_memoize.CachedObjectMetadata(size=cache_size)
