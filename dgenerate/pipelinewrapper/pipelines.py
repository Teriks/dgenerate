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

import collections.abc
import contextlib
import functools
import gc
import hashlib
import importlib
import inspect
import logging
import os
import os.path
import pathlib
import random
import re
import typing

import accelerate
import diffusers
import diffusers.loaders
import diffusers.loaders.single_file_utils
import diffusers.quantizers.quantization_config
import torch
import torch.nn
import transformers

import dgenerate.devicecache as _devicecache
import dgenerate.exceptions as _d_exceptions
import dgenerate.extras.hydit as _hydit
import dgenerate.extras.kolors as _kolors
import dgenerate.extras.spectrum as _spectrum
import dgenerate.extras.ultraedit
import dgenerate.filecache as _filecache
import dgenerate.hfhub as _hfhub
import dgenerate.memoize as _d_memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.safetensorscompat as _safetensorscompat
import dgenerate.pipelinewrapper.singlefilecompat as _singlefilecompat
import dgenerate.pipelinewrapper.sdnqload as _sdnqload
import dgenerate.pipelinewrapper.schedulers as _schedulers
import dgenerate.pipelinewrapper.uris as _uris
import dgenerate.pipelinewrapper.util as _util
import dgenerate.promptweighters as _promptweighters
import dgenerate.textprocessing as _textprocessing
import dgenerate.torchutil as _torchutil
import dgenerate.types as _types
from dgenerate.memoize import memoize as _memoize
from dgenerate.pipelinewrapper import constants as _constants
import dgenerate.pipelinewrapper.models as _models

# Text encoders, VAEs, and UNets read from a ComfyUI quantized single file
# are dequantized as Diffusers loads them.
_safetensorscompat.install_single_file_patch()

# Single file conversions Diffusers lacks or gets wrong.
_singlefilecompat.register_single_file_classes()


class UnsupportedPipelineConfigError(Exception):
    """
    Occurs when a diffusers pipeline is requested to be
    configured in a way that is unsupported by that pipeline.
    """
    pass


def _cache_dit_custom_block_adapter(cache_dit, pipeline):
    """Build Cache-DiT adapters for Wan-Animate transformers.

    Cache-DiT's built-in Wan adapter only accepts WanTransformer3DModel and
    WanVACETransformer3DModel. Wan-Animate uses the same WanTransformerBlock
    collection, while Wan-Animate-2 has its own single-hidden-state blocks.
    """
    transformer = getattr(pipeline, 'transformer', None)
    if transformer is None:
        return None

    transformer_type = type(transformer).__name__
    if transformer_type == 'WanAnimateTransformer3DModel':
        pattern = cache_dit.ForwardPattern.Pattern_2
    elif transformer_type == 'WanAnimate2Transformer3DModel':
        pattern = cache_dit.ForwardPattern.Pattern_3
    else:
        return None

    blocks = getattr(transformer, 'blocks', None)
    if not isinstance(blocks, torch.nn.ModuleList) or not blocks:
        raise ValueError(
            f'{transformer_type} does not expose a non-empty transformer.blocks ModuleList.')

    return cache_dit.BlockAdapter(
        pipe=pipeline,
        transformer=transformer,
        blocks=blocks,
        forward_pattern=pattern,
        has_separate_cfg=True,
        # dgenerate compiles repeated block forwards before call time; torch.compile
        # can hide the concrete method signature from Cache-DiT's introspection.
        # The exact transformer classes above establish these two known contracts.
        check_forward_pattern=False,
    )


def _configure_cache_dit_logging() -> None:
    """Keep Cache-DiT's directly attached loggers quiet unless dgenerate is verbose."""
    logger_module = importlib.import_module('cache_dit.logger')
    if _messages.LEVEL & _messages.DEBUG:
        configured_level = getattr(
            logging, logger_module.ENV.CACHE_DIT_LOG_LEVEL.upper(), logging.INFO)
    else:
        configured_level = logging.CRITICAL

    # Cache-DiT attaches handlers to its own logger tree and disables propagation,
    # so changing the root logger does not affect its output. Also update its
    # logger factory level in case Cache-DiT creates additional loggers later.
    logger_module._LOG_LEVEL = configured_level
    for name, logger in logging.root.manager.loggerDict.items():
        if not isinstance(logger, logging.Logger):
            continue
        if name == 'CACHE_DIT' or name == 'cache_dit' or name.startswith('cache_dit.'):
            logger.setLevel(configured_level)
            for handler in logger.handlers:
                handler.setLevel(configured_level)


def _capture_cache_dit_transformer_forwards(pipeline):
    """Capture forward entry points that temporary Cache-DiT setup may replace."""
    captured = []
    for name in ('transformer', 'transformer_2'):
        transformer = getattr(pipeline, name, None)
        if isinstance(transformer, torch.nn.Module):
            captured.append((transformer, transformer.forward))
    return captured


def _restore_cache_dit_transformer_forwards(captured) -> None:
    """Restore wrappers such as Accelerate's CPU-offload forward hook."""
    for transformer, original_forward in captured:
        transformer.forward = original_forward


@contextlib.contextmanager
def cache_dit_context(pipeline: diffusers.DiffusionPipeline,
                      enabled: bool = False,
                      config_path: _types.OptionalPath = None,
                      device: str = 'cuda',
                      offload_enabled: bool = False,
                      compile_enabled: bool = False,
                      quantizer_enabled: bool = False):
    """Enable Cache-DiT for one call and remove its temporary hooks afterward.

    Parallelism, quantization, and attention backend settings mutate the loaded
    pipeline. When any are configured, Cache-DiT and its cache hooks stay installed
    for later calls so those settings are not applied repeatedly.
    """
    if not enabled:
        if getattr(pipeline, '_dgenerate_cache_dit_setup', None) is not None:
            raise UnsupportedPipelineConfigError(
                'This pipeline already has persistent Cache-DiT settings applied and cannot be '
                'reused with Cache-DiT disabled.')
        yield pipeline
        return

    try:
        import cache_dit
    except ModuleNotFoundError as e:
        if e.name != 'cache_dit':
            raise
        raise UnsupportedPipelineConfigError(
            'Cache-DiT was requested, but the cache-dit package is not installed.'
        ) from e

    _configure_cache_dit_logging()

    import dgenerate._patches.cache_dit_ltx2_patch as _cache_dit_ltx2_patch
    _cache_dit_ltx2_patch.register_hooks(pipeline)

    try:
        import yaml

        raw_config = {}
        config_signature = None
        if config_path is not None:
            with open(config_path, 'r', encoding='utf-8') as config_file:
                loaded_config = yaml.safe_load(config_file)
            if loaded_config is None:
                raw_config = {}
            elif isinstance(loaded_config, dict):
                raw_config = loaded_config
            else:
                raise ValueError('The YAML document must contain a mapping at its root.')

        raw_config = dict(raw_config)
        layerwise_config = raw_config.pop('layerwise_offload', None)
        raw_modifiers = raw_config.pop('params_modifiers', None)

        # Cache-DiT's loader understands its native cache, calibrator, parallelism,
        # attention, and quantization sections. Preserve its legacy flat cache format,
        # while ensuring an empty cache_config is present for optimization-only files.
        native_config = dict(raw_config)
        native_sections = {
            'cache_config', 'calibrator_config', 'parallelism_config',
            'attention_backend', 'quantize_config'}
        if 'cache_config' not in native_config:
            legacy_options = {key: value for key, value in native_config.items()
                              if key not in native_sections}
            if not legacy_options:
                native_config['cache_config'] = {}
            elif any(key in native_config for key in native_sections - {'cache_config'}):
                native_config['cache_config'] = legacy_options
                for key in legacy_options:
                    native_config.pop(key)

        def _load_python_object(reference, key, callable_required=False):
            if not isinstance(reference, str) or not reference.startswith('python:'):
                raise ValueError(
                    f'{key} must use a python:module:attribute import reference.')
            module_name, separator, attribute_path = reference[7:].rpartition(':')
            if not separator or not module_name or not attribute_path:
                raise ValueError(
                    f'{key} must use the form python:module:attribute.')
            value = importlib.import_module(module_name)
            for attribute in attribute_path.split('.'):
                value = getattr(value, attribute)
            if callable_required and not callable(value):
                raise ValueError(f'{key} import reference must resolve to a callable.')
            return value

        parallelism_options = native_config.get('parallelism_config')
        if isinstance(parallelism_options, dict):
            for key in ('ray_transfer_fn', 'cp_plan'):
                reference = parallelism_options.get(key)
                if isinstance(reference, str) and reference.startswith('python:'):
                    parallelism_options[key] = _load_python_object(
                        reference, f'parallelism_config.{key}', callable_required=(key == 'ray_transfer_fn'))
                elif key == 'ray_transfer_fn' and reference is not None and not callable(reference):
                    raise ValueError(
                        'parallelism_config.ray_transfer_fn must resolve to a callable.')
        ray_enabled = bool(
            (parallelism_options or {}).get('use_ray', False))
        if isinstance(parallelism_options, dict) and not parallelism_options.get('use_ray', False):
            parallel_sizes = (
                parallelism_options.get('ulysses_size'),
                parallelism_options.get('ring_size'),
                parallelism_options.get('tp_size'))
            if any(size not in (None, 1, '1') for size in parallel_sizes) and \
                    not torch.distributed.is_initialized():
                raise ValueError(
                    'Native Cache-DiT parallelism requires an initialized torch.distributed '
                    'process group. dgenerate does not initialize process groups; use '
                    'parallelism_config.use_ray: true for the standalone CLI, or initialize '
                    'torch.distributed in the host application.')

        cache_configs = cache_dit.load_configs(native_config, return_dict=True)
        custom_block_adapter = _cache_dit_custom_block_adapter(cache_dit, pipeline)
        if ray_enabled and custom_block_adapter is not None:
            raise ValueError(
                'Cache-DiT Ray parallelism does not support custom BlockAdapter models, '
                'including Wan-Animate and Wan-Animate-2.')

        def _make_modifier(modifier_config):
            if isinstance(modifier_config, list):
                return [_make_modifier(item) for item in modifier_config]
            if not isinstance(modifier_config, dict):
                raise ValueError('Each params_modifiers entry must be a mapping.')

            modifier_config = dict(modifier_config)
            cache_options = modifier_config.pop('cache_config', None)
            calibrator_options = modifier_config.pop('calibrator_config', None)
            modifier_kwargs = modifier_config.pop('kwargs', {}) or {}
            if modifier_config:
                raise ValueError(
                    'Unknown params_modifiers keys: ' + ', '.join(sorted(modifier_config)))
            if ((cache_options is not None and not isinstance(cache_options, dict)) or
                    (calibrator_options is not None and not isinstance(calibrator_options, dict))):
                raise ValueError('params_modifiers cache_config and calibrator_config must be mappings.')
            if not isinstance(modifier_kwargs, dict):
                raise ValueError('params_modifiers kwargs must be a mapping.')

            modifier_native_config = {}
            if cache_options is not None:
                modifier_native_config['cache_config'] = cache_options
            if calibrator_options is not None:
                modifier_native_config['calibrator_config'] = calibrator_options
            if modifier_native_config:
                modifier_configs = cache_dit.load_configs(
                    modifier_native_config, return_dict=True, reset=True)
            else:
                modifier_configs = {'cache_config': None, 'calibrator_config': None}
            return cache_dit.ParamsModifier(
                cache_config=modifier_configs['cache_config'],
                calibrator_config=modifier_configs['calibrator_config'],
                **modifier_kwargs)

        if raw_modifiers is None:
            params_modifiers = None
        else:
            params_modifiers = _make_modifier(raw_modifiers)

        if layerwise_config is None:
            layerwise_components = {}
        elif not isinstance(layerwise_config, dict):
            raise ValueError('layerwise_offload must map pipeline component names to option mappings.')
        else:
            layerwise_components = layerwise_config
            for component_name, component_options in layerwise_components.items():
                if not isinstance(component_name, str) or not component_name:
                    raise ValueError('layerwise_offload component names must be non-empty strings.')
                if component_options is None:
                    layerwise_components[component_name] = {}
                elif not isinstance(component_options, dict):
                    raise ValueError(
                        f'layerwise_offload.{component_name} must be an option mapping.')
                elif isinstance(component_options.get('module_filter'), str):
                    component_options['module_filter'] = _load_python_object(
                        component_options['module_filter'],
                        f'layerwise_offload.{component_name}.module_filter',
                        callable_required=True)

        if layerwise_components and offload_enabled:
            raise ValueError(
                'Cache-DiT layerwise offload cannot be combined with dgenerate model offloading.')
        if cache_configs['quantize_config'] is not None and quantizer_enabled:
            raise ValueError(
                'Cache-DiT quantization cannot be combined with dgenerate --quantizer settings.')
        if ray_enabled and offload_enabled:
            raise ValueError(
                'Cache-DiT Ray parallelism cannot be combined with dgenerate model offloading.')

        model_mutating_config = bool(
            layerwise_components or
            cache_configs['parallelism_config'] is not None or
            cache_configs['attention_backend'] is not None or
            cache_configs['quantize_config'] is not None)
        if compile_enabled and model_mutating_config:
            raise ValueError(
                'Cache-DiT parallelism, attention backends, quantization, and layerwise offload '
                'must be configured before dgenerate compilation; they cannot be combined with '
                '--torch-compile. Cache-DiT Ray compilation can be configured with '
                'parallelism_config.ray_use_compile.')

        persistent_config = {
            'parallelism_config': native_config.get('parallelism_config'),
            'attention_backend': native_config.get('attention_backend'),
            'quantize_config': native_config.get('quantize_config'),
        }
        config_signature = hashlib.sha256(
            repr(persistent_config).encode('utf-8')).hexdigest()
        full_signature = hashlib.sha256(
            repr((native_config, raw_modifiers, layerwise_components)).encode('utf-8')).hexdigest()
        has_persistent_config = any(value is not None for value in persistent_config.values())
        if ray_enabled and layerwise_components:
            raise ValueError(
                'Cache-DiT layerwise offload cannot be combined with the Ray pipeline wrapper.')

        if model_mutating_config:
            target_names = list(layerwise_components)
            if (cache_configs['parallelism_config'] is not None or
                    cache_configs['attention_backend'] is not None):
                target_names.extend(('transformer', 'transformer_2'))
                extra_parallel_modules = (native_config.get('parallelism_config') or {}).get(
                    'extra_parallel_modules') or ()
                if isinstance(extra_parallel_modules, str):
                    target_names.append(extra_parallel_modules)
                else:
                    target_names.extend(extra_parallel_modules)
            if cache_configs['quantize_config'] is not None:
                quantize_components = getattr(
                    cache_configs['quantize_config'], 'components_to_quantize', None)
                if isinstance(quantize_components, str):
                    target_names.append(quantize_components)
                else:
                    target_names.extend(
                        quantize_components or ('transformer', 'transformer_2'))
            compiled_components = [
                name for name in target_names
                if getattr(pipeline, name, None) is not None and
                (hasattr(getattr(pipeline, name), '_orig_mod') or
                 getattr(getattr(pipeline, name), '_dgenerate_compiled', False))
            ]
            if compiled_components:
                raise ValueError(
                    'Cache-DiT parallelism, attention backends, quantization, and layerwise '
                    'offload must be configured before model compilation; compiled component(s): '
                    + ', '.join(compiled_components))

    except Exception as e:
        raise UnsupportedPipelineConfigError(
            f'Cache-DiT configuration could not be loaded: {e}'
        ) from e

    state_attribute = '_dgenerate_cache_dit_setup'
    state = getattr(pipeline, state_attribute, None)
    if state is not None and (
            state.get('signature') != config_signature or
            state.get('full_signature') != full_signature):
        raise UnsupportedPipelineConfigError(
            'This pipeline already has Cache-DiT settings configured. '
            'Its Cache-DiT config cannot be changed while reusing the loaded pipeline.')

    active_pipeline = state['pipeline'] if state is not None else pipeline
    cache_target = (
        state.get('cache_target', active_pipeline) if state is not None else pipeline)
    # Cache-DiT 1.5.2 restores Accelerate-offloaded transformers to _old_forward
    # when disabling its hooks. That skips Accelerate's pre-forward weight load,
    # leaving offloaded parameters on meta when this cached pipeline is reused.
    # Preserve the exact pre-Cache-DiT entry point for temporary configurations.
    original_transformer_forwards = (
        _capture_cache_dit_transformer_forwards(pipeline)
        if not has_persistent_config else [])
    if state is None:
        try:
            enable_kwargs = {
                'cache_config': cache_configs['cache_config'],
                'calibrator_config': cache_configs['calibrator_config'],
                'params_modifiers': params_modifiers,
                'parallelism_config': cache_configs['parallelism_config'],
                'attention_backend': cache_configs['attention_backend'],
                'quantize_config': cache_configs['quantize_config'],
            }
            cache_input = custom_block_adapter or active_pipeline
            enabled_target = cache_dit.enable_cache(cache_input, **enable_kwargs)
            if custom_block_adapter is not None:
                # Cache-DiT returns a BlockAdapter for custom models. The adapter
                # owns hooks on the original pipeline, which must still be called.
                active_pipeline = pipeline
                cache_target = enabled_target
            else:
                active_pipeline = enabled_target
                cache_target = enabled_target
            if has_persistent_config:
                setattr(pipeline, state_attribute, {
                    'signature': config_signature,
                    'full_signature': full_signature,
                    'pipeline': active_pipeline,
                    'cache_target': cache_target,
                    'ray': ray_enabled,
                })
        except Exception as e:
            _restore_cache_dit_transformer_forwards(original_transformer_forwards)
            raise UnsupportedPipelineConfigError(
                f'Cache-DiT could not be enabled for {pipeline.__class__.__name__}: {e}'
            ) from e

    handles = []
    try:
        for component_name, component_options in layerwise_components.items():
            component = getattr(active_pipeline, component_name, None)
            if component is None:
                raise UnsupportedPipelineConfigError(
                    f'Cache-DiT layerwise offload component "{component_name}" does not exist '
                    f'on {active_pipeline.__class__.__name__}.')
            options = dict(component_options)
            options.setdefault('onload_device', device)
            options.setdefault('offload_device', 'cpu')
            handles.append(cache_dit.layerwise_offload(component, **options))

        yield active_pipeline
    finally:
        try:
            try:
                for handle in reversed(handles):
                    handle.remove()
            finally:
                if not has_persistent_config:
                    try:
                        cache_dit.disable_cache(cache_target)
                    finally:
                        _restore_cache_dit_transformer_forwards(
                            original_transformer_forwards)
        except Exception as e:
            raise UnsupportedPipelineConfigError(
                f'Cache-DiT could not be disabled for {pipeline.__class__.__name__}: {e}'
            ) from e


class InvalidModelFileError(Exception):
    """
    Raised when a file is loaded from disk that is an invalid diffusers model format.

    This indicates that was a problem loading the primary diffusion model,
    This could also refer to an SDXL refiner model or Stable Cascade decoder
    model which are considered primary models.
    """
    pass


_pipeline_cache = _d_memoize.create_object_cache(
    'pipeline', cache_type=_memory.SizedConstrainedObjectCache
)


def _disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False] * num_images
    else:
        return images, False


def _floyd_disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False] * num_images, False
    else:
        return images, False, False


def _set_sd_safety_checker(pipeline: diffusers.DiffusionPipeline, safety_checker: bool):
    if not safety_checker:
        if hasattr(pipeline, 'safety_checker') and pipeline.safety_checker is not None:
            # If it's already None for some reason you'll get a call
            # to an unassigned feature_extractor by assigning it a value

            # The attribute will not exist for SDXL pipelines currently

            pipeline.safety_checker = _disabled_safety_checker


def _set_floyd_safety_checker(pipeline: diffusers.DiffusionPipeline, safety_checker: bool):
    if not safety_checker:
        if hasattr(pipeline, 'safety_checker') and pipeline.safety_checker is not None:
            pipeline.safety_checker = _floyd_disabled_safety_checker


# Counted by the include_* flags, or not a weight module.
_PIPELINE_ESTIMATE_SKIP_DIRS = {
    'scheduler',
    'tokenizer',
    'tokenizer_2',
    'tokenizer_3',
    'text_encoder',
    'text_encoder_2',
    'text_encoder_3',
    'vae',
    'unet',
    'transformer',
    'prior',
    'decoder',
    'safety_checker',
    'controlnet',
}


def extra_pipeline_weight_directories(model_index: dict | None) -> list[str]:
    """
    Repo folders that live on the pipeline object and are not in their own cache.

    Z-Image Omni's SigLIP encoder is the important case. Tokenizers, schedulers,
    and the transformer, VAE, and text encoders are left out. ControlNet is left
    out here because a ``--control-nets`` URI has its own cache, unless the
    pipeline keeps that module itself.
    """
    if not model_index:
        return []
    extras = []
    for name, spec in model_index.items():
        if name.startswith('_') or name in _PIPELINE_ESTIMATE_SKIP_DIRS:
            continue
        if not isinstance(spec, (list, tuple)) or len(spec) != 2:
            continue
        library, class_name = spec
        if not library or not isinstance(class_name, str):
            continue
        if 'Scheduler' in class_name or 'Tokenizer' in class_name or 'Processor' in class_name \
                or class_name.endswith('FeatureExtractor'):
            continue
        extras.append(name)
    return extras


def _is_text_encoder_component(name: str) -> bool:
    """Match encoder component slots, not similarly prefixed pipeline config fields."""
    return name == 'text_encoder' or bool(re.fullmatch(r'text_encoder_\d+', name))


def _reject_mixed_offload(model_cpu_offload: bool,
                          sequential_cpu_offload: bool,
                          model_group_offload: bool,
                          label: str = ''):
    chosen = [name for name, enabled in (
        (f'{label}model_cpu_offload', model_cpu_offload),
        (f'{label}sequential_cpu_offload', sequential_cpu_offload),
        (f'{label}model_group_offload', model_group_offload),
    ) if enabled]
    if len(chosen) > 1:
        raise UnsupportedPipelineConfigError(
            f'{chosen[0]} and {chosen[1]} may not be enabled simultaneously.')


def modules_live_on_pipeline(model_cpu_offload: bool,
                             sequential_cpu_offload: bool,
                             model_group_offload: bool = False) -> bool:
    """
    True when pipeline submodules stay inside the pipeline cache instead of
    their own caches. That is every CPU offload mode, including group offload.
    """
    return bool(model_cpu_offload or sequential_cpu_offload or model_group_offload)


def controlnet_is_owned_by_pipeline(model_type: _enums.ModelType,
                                    model_cpu_offload: bool,
                                    sequential_cpu_offload: bool,
                                    model_group_offload: bool = False) -> bool:
    """
    True when the ControlNet object should be counted with the pipeline cache.

    CPU offload already skips the ControlNet cache. Z-Image also has to, because
    ``ZImageControlNetModel.from_transformer`` writes the transformer modules
    onto that object.
    """
    return bool(
        modules_live_on_pipeline(
            model_cpu_offload, sequential_cpu_offload, model_group_offload)
        or _enums.model_type_is_z_image(model_type))


# Leaf classes and parameter types whose storage is not a plain floating tensor.
# Group offload assigns ``parameter.data`` from a CPU snapshot, which drops
# GGUFParameter, bitsandbytes Params4bit/Int8Params, and SDNQ packed weights.
_PACKED_MODULE_NAMES = frozenset({
    'Linear4bit', 'Linear8bitLt',
    'GGUFLinear',
    'SDNQLinear', 'SDNQEmbedding',
    'SDNQConv1d', 'SDNQConv2d', 'SDNQConv3d',
    'SDNQConvTranspose1d', 'SDNQConvTranspose2d', 'SDNQConvTranspose3d',
})
_PACKED_TENSOR_NAMES = frozenset({
    'Params4bit', 'Int8Params', 'GGUFParameter', 'SDNQTensor',
})


def _direct_param_is_packed(module) -> bool:
    try:
        params = module.parameters(recurse=False)
    except (TypeError, ValueError, RuntimeError):
        return False
    for param in params:
        if type(param).__name__ in _PACKED_TENSOR_NAMES:
            return True
        # GGUFParameter.quant_type and bitsandbytes quant_state survive even
        # when the parameter was rebuilt under another class name.
        if getattr(param, 'quant_type', None) is not None:
            return True
        if getattr(param, 'quant_state', None) is not None:
            return True
    return False


def _module_is_quantized(module) -> bool:
    quantized, _, _ = _util.check_bnb_status(module)
    if quantized or _sdnqload.module_is_sdnq(module):
        return True
    if _sdnqload.quantization_config_of(module) is not None:
        return True
    if type(module).__name__ in _PACKED_MODULE_NAMES:
        return True
    if getattr(module, 'sdnq_dequantizer', None) is not None:
        return True
    return _direct_param_is_packed(module)


def module_skips_group_offload(module) -> bool:
    """
    True when group offload must not hook this module.

    Group offload moves ordinary parameters. BitsAndBytes, SDNQ, GGUF, and
    other quantized weights are not those, including a quantized child of an
    otherwise ordinary module. Sequential offload and model CPU offload do
    not use this predicate; they already move these modules on their own.
    """
    if module is None or not isinstance(module, torch.nn.Module):
        return True
    return any(_module_is_quantized(part) for part in module.modules())


def _module_is_8bit_bnb(module) -> bool:
    return any(_util.is_loaded_in_8bit_bnb(part) for part in module.modules())


def place_quantized_module(module, device) -> None:
    """
    Move a quantized module onto ``device`` without group-offload hooks.

    8-bit bitsandbytes is left where ``device_map`` loaded it. ``.to()`` on
    those modules is either disabled or rejected. 4-bit bitsandbytes, SDNQ,
    and GGUF can move, and group offload will not stream them.
    """
    if module is None or not hasattr(module, 'to'):
        return
    if _module_is_8bit_bnb(module):
        _messages.debug_log(
            f'Not moving {type(module).__name__}: '
            f'8-bit bitsandbytes stays on its load device.')
        return
    target = torch.device(device)
    try:
        current = get_torch_device(module)
    except (ValueError, StopIteration, RuntimeError):
        try:
            current = next(module.parameters()).device
        except StopIteration:
            current = None
    if current is not None and _torchutil.devices_equal(current, target):
        return
    _messages.debug_log(
        f'Placing quantized {type(module).__name__} on {target} '
        f'without group offload.')
    module.to(target)


_DENOISER_NAMES = ('unet', 'transformer', 'transformer_2', 'controlnet')
_VAE_NAME = 'vae'
_COMPILE_PART_LABELS = {
    'unet': 'UNet',
    'transformer': 'Transformer',
    'transformer_2': 'Second Transformer',
    'controlnet': 'ControlNet',
    'vae': 'VAE',
}

# Used when a module inherits ModelMixin's empty ``_repeated_blocks``.
# SD3's joint blocks, ControlNet's transformer blocks, and Stable Cascade's
# three block classes. A declared list still wins, so a UNet keeps compiling
# only ``BasicTransformerBlock``.
_DENOISER_FALLBACK_BLOCKS = frozenset({
    'JointTransformerBlock',
    'BasicTransformerBlock',
    'SDCascadeResBlock',
    'SDCascadeTimestepBlock',
    'SDCascadeAttnBlock',
})

# Decoder and encoder blocks that repeat inside the VAEs dgenerate loads.
# The LTX diffusion decoder's neighborhood attention blocks are absent; that
# path stays on NATTEN.
_VAE_BLOCKS = frozenset({
    'ResnetBlock2D',
    'ResnetBlockCondNorm2D',
    'WanResidualBlock',
    'WanAttentionBlock',
    'LTXVideoResnetBlock3d',
    'LTX2VideoResnetBlock3d',
    'QwenImageResidualBlock',
    'QwenImageAttentionBlock',
})


def _blocks_present(module, allow: frozenset[str], minimum: int = 2) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for submod in module.modules():
        name = submod.__class__.__name__
        if name in allow:
            counts[name] = counts.get(name, 0) + 1
    return tuple(name for name, count in counts.items() if count >= minimum)


def repeated_block_names(module) -> tuple[str, ...]:
    """
    Class names of the denoiser or ControlNet blocks ``--torch-compile`` should compile.

    A declared ``_repeated_blocks`` list wins. Stable Diffusion 3, ControlNet,
    and Stable Cascade leave that list empty, so their repeated blocks are
    recognized by class name. A class has to appear at least twice.
    """
    declared = getattr(module, '_repeated_blocks', None)
    if declared:
        return tuple(declared)
    return _blocks_present(module, _DENOISER_FALLBACK_BLOCKS)


def vae_block_names(module) -> tuple[str, ...]:
    """Repeated VAE blocks ``--torch-compile`` should compile."""
    if module is None:
        return ()
    return _blocks_present(module, _VAE_BLOCKS)


def _torch_version_at_least(major: int, minor: int) -> bool:
    raw = torch.__version__.split('+')[0]
    parts = raw.split('.')
    try:
        got = (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return False
    return got >= (major, minor)


def torch_compile_disabled() -> bool:
    """
    True when ``DGENERATE_TORCH_COMPILE=0``.

    This leaves every model eager, including Wan-Animate-2, which otherwise
    compiles on every load.
    """
    return os.environ.get('DGENERATE_TORCH_COMPILE', '1').strip() == '0'


def inductor_can_compile(device) -> bool:
    """
    True when ``torch.compile`` can build an Inductor kernel on ``device``.

    CUDA and XPU use Triton. macOS uses the Metal Inductor path from
    PyTorch 2.13. CPU is left eager.
    """
    import importlib.util

    kind = torch.device(device).type
    if kind == 'mps':
        return _torch_version_at_least(2, 13)
    if kind in ('cuda', 'xpu'):
        return importlib.util.find_spec('triton') is not None
    return False


_COMPILE_LOG_QUIET: bool | None = None
_COMPILE_LOGS_ENV_OWNED = False


def _configure_torch_compile_logging() -> None:
    """
    Keep TorchDynamo, Inductor, and Triton quiet unless ``-v/--verbose`` is on.

    ``TORCH_LOGS``, when the user set it, is left alone. Errors still print.
    The env value is also set so Inductor's compile worker inherits it.
    """
    global _COMPILE_LOG_QUIET, _COMPILE_LOGS_ENV_OWNED

    if 'TORCH_LOGS' in os.environ and not _COMPILE_LOGS_ENV_OWNED:
        return
    quiet = not bool(_messages.LEVEL & _messages.DEBUG)
    if _COMPILE_LOG_QUIET is quiet:
        return
    if _COMPILE_LOGS_ENV_OWNED:
        os.environ.pop('TORCH_LOGS', None)
        _COMPILE_LOGS_ENV_OWNED = False
    import torch._logging
    level = logging.ERROR if quiet else logging.WARNING
    torch._logging.set_logs(dynamo=level, inductor=level, aot=level)
    logging.getLogger('triton').setLevel(level)
    if quiet:
        os.environ['TORCH_LOGS'] = '-dynamo,-inductor,-aot'
        _COMPILE_LOGS_ENV_OWNED = True
    _COMPILE_LOG_QUIET = quiet


def compile_repeated_forwards(module, block_names: tuple[str, ...] | None = None) -> int:
    """
    Compile ``forward`` on submodules named by :func:`repeated_block_names`.

    ``block_names`` overrides that list, which is how the VAE selects its
    own blocks. Compiling ``forward`` leaves group-offload hooks, which wrap
    ``_call_impl``, outside the graph. Returns how many blocks were
    compiled. A quantized module and an already compiled module compile
    nothing.
    """
    if module is None or getattr(module, '_dgenerate_compiled', False):
        return 0
    if torch_compile_disabled():
        return 0
    repeated = repeated_block_names(module) if block_names is None else block_names
    if not repeated or module_skips_group_offload(module):
        return 0
    # Repeated blocks share one compiled frame but their convolutions have
    # different widths. Leaving parameter shapes static recompiles that frame
    # until Dynamo's limit, then leaves the rest eager.
    torch._dynamo.config.force_parameter_static_shapes = False
    _configure_torch_compile_logging()
    compiled = 0
    for submod in module.modules():
        if submod.__class__.__name__ not in repeated:
            continue
        if getattr(submod, '_dgenerate_forward_compiled', False):
            continue
        submod.forward = torch.compile(submod.forward)
        submod._dgenerate_forward_compiled = True
        compiled += 1
    if compiled:
        module._dgenerate_compiled = True
    return compiled


def _modules_named(pipe, name: str):
    module = getattr(pipe, name, None)
    if isinstance(module, (list, tuple)):
        for item in module:
            if isinstance(item, torch.nn.Module):
                yield item
        return
    if isinstance(module, torch.nn.Module):
        yield module


def _uncompiled_block_counts(module, block_names: tuple[str, ...]) -> dict[str, int]:
    """How many of each block class ``compile_repeated_forwards`` will compile."""
    allowed = set(block_names)
    counts: dict[str, int] = {}
    for submod in module.modules():
        name = submod.__class__.__name__
        if name not in allowed or getattr(submod, '_dgenerate_forward_compiled', False):
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts


def _torch_compile_message(parts: list[tuple[str, dict[str, int]]]) -> str:
    """
    Name the pipeline parts and block classes that were compiled.

    ``parts`` is ``(attribute name, {class name: count})`` in compile order.
    Several modules on one attribute, such as a ControlNet list, are summed.
    """
    grouped: dict[str, dict[str, int]] = {}
    order: list[str] = []
    for name, counts in parts:
        if name not in grouped:
            grouped[name] = {}
            order.append(name)
        bucket = grouped[name]
        for cls, count in counts.items():
            bucket[cls] = bucket.get(cls, 0) + count
    clauses = []
    for name in order:
        label = _COMPILE_PART_LABELS.get(name, name)
        counts = grouped[name]
        total = sum(counts.values())
        kind = 'block' if total == 1 else 'blocks'
        listed = ', '.join(f'{count} {cls}' for cls, count in counts.items())
        clauses.append(f'{label} {kind} ({listed})')
    return (
        f'Compiling {_textprocessing.oxford_comma(clauses, "and")} (--torch-compile). The first use '
        'of each waits while the graph builds. A later change of resolution '
        'or frame count compiles once more, then reuses that graph.')


def apply_torch_compile(pipe, device) -> None:
    """
    Compile repeated denoiser, ControlNet, and VAE blocks for ``--torch-compile``.

    The log names only the pipeline parts and block classes compiled on this
    pipeline. Warns when the device cannot compile, the weights are quantized,
    or nothing repeated was found. Wan-Animate-2 compiles its transformer on
    every load and does not use this path for that module.
    """
    if torch_compile_disabled():
        _messages.warning(
            '--torch-compile was set, but DGENERATE_TORCH_COMPILE=0, '
            'so the blocks stay eager.')
        return
    if not inductor_can_compile(device):
        if torch.device(device).type in ('cuda', 'xpu'):
            detail = 'Triton is not installed, so the blocks stay eager.'
        else:
            detail = 'This device cannot compile the blocks, so they stay eager.'
        _messages.warning(f'--torch-compile was set. {detail}')
        return
    compiled = 0
    saw_repeated = False
    saw_quantized = False
    compiled_parts: list[tuple[str, dict[str, int]]] = []
    for name in (*_DENOISER_NAMES, _VAE_NAME):
        block_names_for = repeated_block_names if name != _VAE_NAME else vae_block_names
        for module in _modules_named(pipe, name):
            block_names = block_names_for(module)
            if not block_names:
                continue
            saw_repeated = True
            if module_skips_group_offload(module):
                saw_quantized = True
                continue
            counts = _uncompiled_block_counts(module, block_names)
            compiled_now = compile_repeated_forwards(module, block_names)
            compiled += compiled_now
            if compiled_now and counts:
                compiled_parts.append((name, counts))
    if compiled:
        _messages.log(_torch_compile_message(compiled_parts))
    if saw_quantized:
        _messages.warning(
            '--torch-compile was set. Quantized weights stay eager.')
        return
    if compiled:
        return
    if not saw_repeated:
        _messages.warning(
            '--torch-compile was set. This model has no repeated blocks '
            'to compile, so it stays eager.')


def estimate_pipeline_cache_footprint(
        model_path: str,
        model_type: _enums.ModelType,
        revision: _types.Name = 'main',
        variant: _types.OptionalName = None,
        subfolder: _types.OptionalPath = None,
        include_unet_or_transformer: bool = False,
        include_vae: bool = False,
        include_text_encoders: bool = False,
        lora_uris: _types.OptionalUris = None,
        image_encoder_uri: _types.OptionalUri = None,
        ip_adapter_uris: _types.OptionalUris = None,
        textual_inversion_uris: _types.OptionalUris = None,
        safety_checker: bool = False,
        auth_token: str | None = None,
        extra_args: dict[str, typing.Any] | None = None,
        local_files_only: bool = False,
        include_directories: collections.abc.Sequence[str] | None = None,
        component_repo: str | None = None) -> int:
    """
    Estimate the CPU side cache memory use of a pipeline.

    This does not include the UNet / Transformer, VAE, or Text Encoders
    as those have their own individual caches.

    :param model_path: huggingface slug, blob link, path to folder on disk, path to model file.
    :param model_type: :py:class:`dgenerate.pipelinewrapper.ModelType`
    :param revision: huggingface repo revision if using a huggingface slug
    :param variant: model file variant desired, for example "fp16"
    :param subfolder: huggingface repo subfolder if using a huggingface slug
        this is currently only supported for Stable Diffusion 3 and Flux models.
    :param include_unet_or_transformer: Include the unet / transformer?
        Under most conditions this is loaded separately and put into
        a cache of its own.
    :param include_text_encoders: Include text encoders?
        Under most conditions these are loaded separately and put into
        a cache of their own.
    :param include_vae: Include VAE? Under most conditions this is
        loaded separately and put into a cache its own.
    :param lora_uris: optional user specified ``--loras`` URIs that will be loaded on to the pipeline
    :param image_encoder_uri: optional user specified ``--image-encoder`` URI that will be loaded on to the pipeline
    :param ip_adapter_uris: optional user specified ``--ip-adapters`` URIs that will be loaded on to the pipeline
    :param textual_inversion_uris: optional user specified ``--textual-inversion`` URIs that will be loaded on to the pipeline
    :param safety_checker: consider the safety checker? dgenerate usually loads the safety checker and then retroactively
        disables it if needed, so it usually considers the size of the safety checker model.
    :param auth_token: optional huggingface auth token to access restricted repositories that your account has access to.
    :param extra_args: ``extra_args`` as to be passed to :py:func:`.create_diffusion_pipeline`
    :param local_files_only: Only ever attempt to look in the local huggingface cache? if ``False`` the huggingface
        API will be contacted when necessary.
    :param include_directories: extra repo folder names to count with the main estimate,
        for example LTX ``audio_vae`` and ``vocoder``.
    :param component_repo: Hugging Face repo that supplies the VAE and text encoders
        when ``model_path`` is only a transformer file. The file is counted only when
        ``include_unet_or_transformer`` is set, because that is when the transformer
        stays in the pipeline cache instead of the transformer cache. The repository's
        own transformer is not counted.
    :return: size estimate in bytes.
    """

    if extra_args is None:
        extra_args = dict()

    if component_repo:
        usage = 0
        if include_unet_or_transformer and os.path.isfile(model_path):
            usage += os.path.getsize(model_path)
            _messages.debug_log(
                'Estimate Considering:',
                os.path.basename(model_path) + f', Size: {usage} Bytes')

        def _kept(name: str) -> bool:
            return not (name in extra_args and extra_args[name] is None)

        usage += _util.estimate_model_memory_use(
            repo_id=component_repo,
            revision=revision,
            variant=variant,
            subfolder=subfolder,
            include_unet_or_transformer=False,
            include_vae=include_vae and _kept('vae'),
            safety_checker=safety_checker and 'safety_checker' not in extra_args,
            include_text_encoder=include_text_encoders and _kept('text_encoder'),
            include_text_encoder_2=include_text_encoders and _kept('text_encoder_2'),
            include_text_encoder_3=include_text_encoders and _kept('text_encoder_3'),
            include_directories=include_directories,
            auth_token=auth_token,
            local_files_only=local_files_only,
            sentencepiece=_enums.model_type_is_floyd(model_type)
        )
    else:
        usage = _util.estimate_model_memory_use(
            repo_id=_hfhub.download_non_hf_slug_model(model_path, local_files_only=local_files_only, auth_token=auth_token),
            revision=revision,
            variant=variant,
            subfolder=subfolder,
            include_unet_or_transformer=include_unet_or_transformer,
            include_vae=include_vae,
            safety_checker=safety_checker and 'safety_checker' not in extra_args,
            include_text_encoder=include_text_encoders,
            include_text_encoder_2=include_text_encoders,
            include_text_encoder_3=include_text_encoders,
            include_directories=include_directories,
            auth_token=auth_token,
            local_files_only=local_files_only,
            sentencepiece=_enums.model_type_is_floyd(model_type)
        )

    if image_encoder_uri:
        parsed = _uris.ImageEncoderUri.parse(image_encoder_uri)
        usage += _util.estimate_model_memory_use(
            repo_id=_hfhub.download_non_hf_slug_model(parsed.model, local_files_only=local_files_only, auth_token=auth_token),
            revision=parsed.revision,
            subfolder=parsed.subfolder,
            auth_token=auth_token,
            local_files_only=local_files_only
        )

    if lora_uris:
        for lora_uri in lora_uris:
            parsed = _uris.LoRAUri.parse(lora_uri)

            usage += _util.estimate_model_memory_use(
                repo_id=_hfhub.download_non_hf_slug_model(parsed.model, local_files_only=local_files_only, auth_token=auth_token),
                revision=parsed.revision,
                subfolder=parsed.subfolder,
                weight_name=parsed.weight_name,
                auth_token=auth_token,
                local_files_only=local_files_only
            )

    if ip_adapter_uris:
        for ip_adapter_uri in ip_adapter_uris:
            parsed = _uris.IPAdapterUri.parse(ip_adapter_uri)

            usage += _util.estimate_model_memory_use(
                repo_id=_hfhub.download_non_hf_slug_model(parsed.model, local_files_only=local_files_only, auth_token=auth_token),
                revision=parsed.revision,
                subfolder=parsed.subfolder,
                weight_name=parsed.weight_name,
                auth_token=auth_token,
                local_files_only=local_files_only
            )

    if textual_inversion_uris:
        for textual_inversion_uri in textual_inversion_uris:
            parsed = _uris.TextualInversionUri.parse(textual_inversion_uri)

            usage += _util.estimate_model_memory_use(
                repo_id=_hfhub.download_non_hf_slug_model(parsed.model, local_files_only=local_files_only, auth_token=auth_token),
                revision=parsed.revision,
                subfolder=parsed.subfolder,
                weight_name=parsed.weight_name,
                auth_token=auth_token,
                local_files_only=local_files_only
            )

    return usage


def set_vae_tiling_and_slicing(
        pipeline: diffusers.DiffusionPipeline,
        tiling: bool,
        slicing: bool
):
    """
    Set the ``vae_slicing`` and ``vae_tiling`` status on a diffusers pipeline.

    :raises UnsupportedPipelineConfigError: if the pipeline does not support one or both
        of the provided values for ``vae_tiling`` and ``vae_slicing``

    :param pipeline: pipeline object
    :param tiling: tiling status
    :param slicing: slicing status
    """

    has_vae = hasattr(pipeline, 'vae') and pipeline.vae is not None
    pipeline_class = pipeline.__class__

    if tiling:
        if has_vae:
            if hasattr(pipeline.vae, 'enable_tiling'):
                _messages.debug_log(f'Enabling VAE tiling on Pipeline: "{pipeline_class.__name__}",',
                                    f'VAE: "{pipeline.vae.__class__.__name__}"')
                pipeline.vae.enable_tiling()
            else:
                raise UnsupportedPipelineConfigError(
                    '--vae-tiling not supported as loaded VAE does not support it.'
                )
        else:
            raise UnsupportedPipelineConfigError(
                '--vae-tiling not supported as no VAE is present for the specified model.')
    elif has_vae:
        if hasattr(pipeline.vae, 'disable_tiling'):
            _messages.debug_log(f'Disabling VAE tiling on Pipeline: "{pipeline_class.__name__}",',
                                f'VAE: "{pipeline.vae.__class__.__name__}"')
            pipeline.vae.disable_tiling()

    if slicing:
        if has_vae:
            if hasattr(pipeline.vae, 'enable_slicing'):
                _messages.debug_log(f'Enabling VAE slicing on Pipeline: "{pipeline_class.__name__}",',
                                    f'VAE: "{pipeline.vae.__class__.__name__}"')
                pipeline.vae.enable_slicing()
            else:
                raise UnsupportedPipelineConfigError(
                    '--vae-slicing not supported as loaded VAE does not support it.'
                )
        else:
            raise UnsupportedPipelineConfigError(
                '--vae-slicing not supported as no VAE is present for the specified model.')
    elif has_vae:
        if hasattr(pipeline.vae, 'disable_slicing'):
            _messages.debug_log(f'Disabling VAE slicing on Pipeline: "{pipeline_class.__name__}",',
                                f'VAE: "{pipeline.vae.__class__.__name__}"')
            pipeline.vae.disable_slicing()


def get_pipeline_modules(pipeline: diffusers.DiffusionPipeline):
    """
    Get all component modules of a torch diffusers pipeline.

    :param pipeline: the pipeline
    :return: dictionary of modules by name
    """
    modules = {k: v for k, v in pipeline.components.items() if isinstance(v, torch.nn.Module)}
    # Hunyuan-DiT IP-Adapter attaches CLIP outside the diffusers component list.
    image_encoder = getattr(pipeline, 'image_encoder', None)
    if image_encoder is not None and isinstance(image_encoder, torch.nn.Module):
        modules.setdefault('image_encoder', image_encoder)
    return modules


def _set_sequential_cpu_offload_flag(module: diffusers.DiffusionPipeline | torch.nn.Module, value: bool):
    module.DGENERATE_SEQUENTIAL_CPU_OFFLOAD = bool(value)

    _messages.debug_log(
        f'setting DGENERATE_SEQUENTIAL_CPU_OFFLOAD={value} on module "{module.__class__.__name__}"')


def _set_cpu_offload_flag(module: diffusers.DiffusionPipeline | torch.nn.Module, value: bool):
    module.DGENERATE_MODEL_CPU_OFFLOAD = bool(value)

    _messages.debug_log(
        f'setting DGENERATE_MODEL_CPU_OFFLOAD={value} on module "{module.__class__.__name__}"')


def is_sequential_cpu_offload_enabled(module: diffusers.DiffusionPipeline | torch.nn.Module):
    """
    Test if a pipeline or torch neural net module created by dgenerate has sequential offload enabled.

    :param module: the module object
    :return: ``True`` or ``False``
    """
    return hasattr(module, 'DGENERATE_SEQUENTIAL_CPU_OFFLOAD') and bool(module.DGENERATE_SEQUENTIAL_CPU_OFFLOAD)


def is_model_cpu_offload_enabled(module: diffusers.DiffusionPipeline | torch.nn.Module):
    """
    Test if a pipeline or torch neural net module created by dgenerate has model cpu offload enabled.

    :param module: the module object
    :return: ``True`` or ``False``
    """
    return hasattr(module, 'DGENERATE_MODEL_CPU_OFFLOAD') and bool(module.DGENERATE_MODEL_CPU_OFFLOAD)


def _disable_to(module, vae=False):
    if hasattr(module, '_DGENERATE_ORIGINAL_TO_DISABLED'):
        return  # Already patched

    module._DGENERATE_ORIGINAL_TO_DISABLED = module.to

    def dummy(*args, **kwargs):
        # Wan and some other VAEs use FrozenDict configs without force_upcast.
        force_upcast = False
        if vae:
            config = getattr(module, 'config', None)
            if config is not None:
                force_upcast = bool(getattr(config, 'force_upcast', False))
                if not force_upcast and hasattr(config, 'get'):
                    force_upcast = bool(config.get('force_upcast', False))

        if vae and force_upcast and \
                (len(args) == 1 and isinstance(args[0], torch.dtype)) or \
                (len(kwargs) == 1 and 'dtype' in kwargs):

            # basically, is this a VAE that the pipeline needs to upcast
            # this has to happen even if it is described as 'meta'

            module._DGENERATE_ORIGINAL_TO_DISABLED(*args, **kwargs)
        else:
            pass

    module.to = dummy
    _messages.debug_log(
        f'Disabled .to() on module: {_types.fullname(module)}')


def _bind_pipeline_offload_method(pipeline, name, function):
    """
    Point ``pipeline.enable_*_cpu_offload`` at dgenerate's implementation.

    Diffusers calls those methods with no arguments after ``load_lora_weights``
    and, for model CPU offload, at the end of ``__call__``. Leaving the original
    methods in place replaces dgenerate's hooks, ``.to()`` guard, and
    bitsandbytes skip.
    """

    def entry(gpu_id=None, device=None, _function=function, _pipeline=pipeline):
        target = device
        if target is None:
            target = getattr(_pipeline, '_dgenerate_offload_device', None)
        if gpu_id is not None and device is None and target is not None:
            target = torch.device(torch.device(target).type, gpu_id)
        _function(_pipeline, target if target is not None else _torchutil.default_device())

    setattr(pipeline, name, entry)


def _promote_bare_parameter_tensors(module: torch.nn.Module) -> None:
    """
    Wrap plain tensors sitting in ``_parameters`` as ``nn.Parameter``.

    GGUF loading dequantizes Wan ``_keep_in_fp32_modules`` and assigns the
    float result directly. Accelerate rebuilds ``type(param)(value,
    requires_grad=...)`` during CPU offload, and ``torch.Tensor`` rejects
    that keyword.
    """
    for submodule in module.modules():
        bare = [
            name for name, value in submodule._parameters.items()
            if type(value) is torch.Tensor
        ]
        for name in bare:
            value = submodule._parameters[name]
            submodule._parameters[name] = torch.nn.Parameter(
                value, requires_grad=bool(value.requires_grad))


def _offload_as_one_unit(module: torch.nn.Module) -> bool:
    """
    This module reads child weights without calling those children.

    ``torch.nn.MultiheadAttention`` reads ``out_proj.weight`` directly.
    Hunyuan-DiT's text pooler reads ``q_proj`` / ``k_proj`` / ``v_proj`` /
    ``c_proj`` weights inside ``F.multi_head_attention_forward``. Leaf offload
    hooks on those linears never run, so the weights stay on the meta device.
    """
    return isinstance(module, torch.nn.MultiheadAttention) or \
        type(module).__name__ == 'HunyuanDiTAttentionPool'


def _sequential_offload_preload_classes(module: torch.nn.Module) -> list[str] | None:
    """
    Class names whose submodule weights accelerate must load as one unit.

    See :func:`_offload_as_one_unit`.
    """
    names = {type(m).__name__ for m in module.modules() if _offload_as_one_unit(m)}
    return sorted(names) if names else None


def enable_sequential_cpu_offload(pipeline: diffusers.DiffusionPipeline,
                                  device: torch.device | str = _torchutil.default_device()):
    """
    Enable sequential offloading on a torch pipeline, in a way dgenerate can keep track of.

    :param pipeline: the pipeline
    :param device: the device
    """
    torch_device = torch.device(device)
    
    # Check if the requested device type is actually available
    # If not, fall back to the system's default device
    if torch_device.type == 'cuda' and not _torchutil.is_cuda_available():
        fallback_device = _torchutil.default_device()
        _messages.debug_log(
            f'enable_sequential_cpu_offload: CUDA requested but not available, using {fallback_device} for execution device')
        torch_device = torch.device(fallback_device)
    elif torch_device.type == 'mps' and not _torchutil.is_mps_available():
        fallback_device = _torchutil.default_device()
        _messages.debug_log(
            f'enable_sequential_cpu_offload: MPS requested but not available, using {fallback_device} for execution device')
        torch_device = torch.device(fallback_device)
    elif torch_device.type == 'xpu' and not _torchutil.is_xpu_available():
        fallback_device = _torchutil.default_device()
        _messages.debug_log(
            f'enable_sequential_cpu_offload: XPU requested but not available, using {fallback_device} for execution device')
        torch_device = torch.device(fallback_device)
    
    pipeline.remove_all_hooks()

    pipeline._dgenerate_offload_device = torch_device
    _set_sequential_cpu_offload_flag(pipeline, True)
    for name, model in get_pipeline_modules(pipeline).items():
        quant, _, _ = _util.check_bnb_status(model)

        if name in pipeline._exclude_from_cpu_offload or quant:
            continue

        # The flag survives the hook removal Diffusers does before loading a LoRA.
        # Reinstall the hook when it is missing so that reload still uses this path.
        elif not is_sequential_cpu_offload_enabled(model) or not hasattr(model, '_hf_hook'):
            _set_sequential_cpu_offload_flag(model, True)
            _promote_bare_parameter_tensors(model)
            accelerate.cpu_offload(model, torch_device, offload_buffers=len(model._parameters) > 0,
                                   preload_module_classes=_sequential_offload_preload_classes(model))
            _disable_to(
                model,
                name == 'vae'
            )
    _bind_pipeline_offload_method(pipeline, 'enable_sequential_cpu_offload', enable_sequential_cpu_offload)


def enable_model_cpu_offload(pipeline: diffusers.DiffusionPipeline,
                             device: torch.device | str = _torchutil.default_device()):
    """
    Enable sequential model cpu offload on a torch pipeline, in a way dgenerate can keep track of.

    :param pipeline: the pipeline
    :param device: the device
    """

    if pipeline.model_cpu_offload_seq is None:
        raise ValueError(
            "Model CPU offload cannot be enabled because no `model_cpu_offload_seq` class attribute is set."
        )

    torch_device = torch.device(device)

    pipeline.remove_all_hooks()

    pipeline._offload_gpu_id = torch_device.index or getattr(pipeline, "_offload_gpu_id", 0)

    device_type = torch_device.type
    
    # Check if the requested device type is actually available
    # If not, fall back to the system's default device
    if device_type == 'cuda' and not _torchutil.is_cuda_available():
        fallback_device = _torchutil.default_device()
        _messages.debug_log(
            f'enable_model_cpu_offload: CUDA requested but not available, using {fallback_device} for execution device')
        device = torch.device(fallback_device)
    elif device_type == 'mps' and not _torchutil.is_mps_available():
        fallback_device = _torchutil.default_device()
        _messages.debug_log(
            f'enable_model_cpu_offload: MPS requested but not available, using {fallback_device} for execution device')
        device = torch.device(fallback_device)
    elif device_type == 'xpu' and not _torchutil.is_xpu_available():
        fallback_device = _torchutil.default_device()
        _messages.debug_log(
            f'enable_model_cpu_offload: XPU requested but not available, using {fallback_device} for execution device')
        device = torch.device(fallback_device)
    else:
        device = torch.device(f"{device_type}:{pipeline._offload_gpu_id}")

    if pipeline.device.type != "cpu":
        pipeline.to("cpu", silence_dtype_warnings=True)
        device_mod = getattr(torch, pipeline.device.type, None)
        if device_mod is not None and hasattr(device_mod, "empty_cache") and hasattr(device_mod, "is_available") and device_mod.is_available():
            device_mod.empty_cache()

    _set_cpu_offload_flag(pipeline, True)

    all_model_components = {k: v for k, v in pipeline.components.items() if isinstance(v, torch.nn.Module)}

    hook = None
    pipeline._all_hooks = []
    for model_str in pipeline.model_cpu_offload_seq.split("->"):
        model = all_model_components.pop(model_str, None)

        if not isinstance(model, torch.nn.Module):
            continue

        _, _, is_loaded_in_8bit_bnb = _util.check_bnb_status(model)

        if is_loaded_in_8bit_bnb:
            _messages.debug_log(
                f'Not cpu offloading pipeline module: {model_str}, due to bitsandbytes 8 bit quantization.')
            continue

        _promote_bare_parameter_tensors(model)
        _, hook = accelerate.cpu_offload_with_hook(model, device, prev_module_hook=hook)
        _set_cpu_offload_flag(model, True)
        pipeline._all_hooks.append(hook)

    for name, model in all_model_components.items():
        if not isinstance(model, torch.nn.Module):
            continue

        if name in pipeline._exclude_from_cpu_offload:
            model.to(device)
        else:
            _promote_bare_parameter_tensors(model)
            _, hook = accelerate.cpu_offload_with_hook(model, device)
            _set_cpu_offload_flag(model, True)
            pipeline._all_hooks.append(hook)

    pipeline._dgenerate_offload_device = device
    pipeline._offload_device = device
    _bind_pipeline_offload_method(pipeline, 'enable_model_cpu_offload', enable_model_cpu_offload)


def _set_group_offload_flag(module: diffusers.DiffusionPipeline | torch.nn.Module, value: bool):
    module.DGENERATE_GROUP_OFFLOAD = bool(value)

    _messages.debug_log(
        f'setting DGENERATE_GROUP_OFFLOAD={value} on module "{module.__class__.__name__}"')


def is_group_offload_enabled(module: diffusers.DiffusionPipeline | torch.nn.Module):
    """
    Test if a pipeline or torch neural net module created by dgenerate has group offload enabled.

    :param module: the module object
    :return: ``True`` or ``False``
    """
    return hasattr(module, 'DGENERATE_GROUP_OFFLOAD') and bool(module.DGENERATE_GROUP_OFFLOAD)


def _clear_group_offload_hooks(module: torch.nn.Module):
    from diffusers.hooks.group_offloading import (
        _GROUP_OFFLOADING,
        _LAZY_PREFETCH_GROUP_OFFLOADING,
        _LAYER_EXECUTION_TRACKER,
    )

    registry = getattr(module, '_diffusers_hook', None)
    if registry is None:
        return
    for name in (_GROUP_OFFLOADING, _LAZY_PREFETCH_GROUP_OFFLOADING, _LAYER_EXECUTION_TRACKER):
        if registry.get_hook(name) is not None:
            registry.remove_hook(name, recurse=True)


def _group_multihead_attention(module: torch.nn.Module):
    """
    Regroup modules that read child weights directly as one offload group.

    Leaf-level group offload hooks those children as their own groups, but the
    parent never calls them, so those groups are never onloaded. Run this after
    ``apply_group_offloading`` and before the first forward, which traces the
    prefetch order.
    """
    from diffusers.hooks.group_offloading import (
        _GROUP_OFFLOADING,
        ModuleGroup,
        _apply_group_offloading_hook,
    )

    for mha in module.modules():
        if not _offload_as_one_unit(mha):
            continue

        hooks = [m._diffusers_hook.get_hook(_GROUP_OFFLOADING)
                 for m in mha.modules() if hasattr(m, '_diffusers_hook')]
        hooks = [h for h in hooks if h is not None]
        if not hooks:
            continue

        old = hooks[0].group
        for m in mha.modules():
            registry = getattr(m, '_diffusers_hook', None)
            if registry is not None and registry.get_hook(_GROUP_OFFLOADING) is not None:
                registry.remove_hook(_GROUP_OFFLOADING, recurse=False)

        group = ModuleGroup(
            modules=[mha],
            offload_device=old.offload_device,
            onload_device=old.onload_device,
            offload_leader=mha,
            onload_leader=mha,
            non_blocking=old.non_blocking,
            stream=old.stream,
            record_stream=old.record_stream,
            low_cpu_mem_usage=old.low_cpu_mem_usage,
            onload_self=True,
        )
        _apply_group_offloading_hook(mha, group, config=hooks[0].config)


def enable_group_offload(pipeline: diffusers.DiffusionPipeline,
                         device: torch.device | str = _torchutil.default_device()):
    """
    Enable leaf-level group offload on a torch pipeline.

    This is the middle setting between model CPU offload and sequential offload.
    CUDA and XPU prefetch the next leaf on a side stream. BitsAndBytes, SDNQ,
    and GGUF modules are not hooked: packed weights cannot be snapshotted.
    Those modules are placed on ``device`` instead, except 8-bit bitsandbytes,
    which stays where it was loaded. Full-precision weights stay in CPU RAM,
    so the pipeline cache size still counts them. Disk offload is not used.

    A later ``module.to(dtype=...)`` is kept. Streamed group offload otherwise
    restores the dtype captured when the hooks were installed, so the module
    reports the new dtype while the next forward still runs the old one.

    :param pipeline: the pipeline
    :param device: the device layers move to for a forward pass
    """
    from diffusers.hooks.group_offloading import apply_group_offloading

    torch_device = torch.device(device)
    if torch_device.type == 'cuda' and not _torchutil.is_cuda_available():
        fallback_device = _torchutil.default_device()
        _messages.debug_log(
            f'enable_group_offload: CUDA requested but not available, using {fallback_device} for execution device')
        torch_device = torch.device(fallback_device)
    elif torch_device.type == 'mps' and not _torchutil.is_mps_available():
        fallback_device = _torchutil.default_device()
        _messages.debug_log(
            f'enable_group_offload: MPS requested but not available, using {fallback_device} for execution device')
        torch_device = torch.device(fallback_device)
    elif torch_device.type == 'xpu' and not _torchutil.is_xpu_available():
        fallback_device = _torchutil.default_device()
        _messages.debug_log(
            f'enable_group_offload: XPU requested but not available, using {fallback_device} for execution device')
        torch_device = torch.device(fallback_device)

    pipeline.remove_all_hooks()
    for _, model in get_pipeline_modules(pipeline).items():
        _clear_group_offload_hooks(model)

    use_stream = torch_device.type in ('cuda', 'xpu')
    _set_group_offload_flag(pipeline, True)
    pipeline._dgenerate_offload_device = torch_device
    excluded = set(getattr(pipeline, '_exclude_from_cpu_offload', ()) or ())

    # The transformer is hooked before a Z-Image ControlNet. ``from_transformer``
    # grafts transformer modules onto the ControlNet, and a second group-offload
    # pass leaves those existing hooks in place.
    modules = get_pipeline_modules(pipeline)
    ordered = [(name, model) for name, model in modules.items() if name != 'controlnet']
    if 'controlnet' in modules:
        ordered.append(('controlnet', modules['controlnet']))

    for name, model in ordered:
        if module_skips_group_offload(model):
            _messages.debug_log(
                f'Not group offloading pipeline module: {name}, '
                f'because it is quantized.')
            place_quantized_module(model, torch_device)
            continue
        if name in excluded:
            model.to(torch_device)
            continue
        apply_group_offloading(
            model,
            onload_device=torch_device,
            offload_device=torch.device('cpu'),
            offload_type='leaf_level',
            use_stream=use_stream,
            non_blocking=use_stream,
        )
        _group_multihead_attention(model)
        _set_group_offload_flag(model, True)

    _bind_pipeline_offload_method(pipeline, 'enable_group_offload', enable_group_offload)


def get_torch_device(component: diffusers.DiffusionPipeline | torch.nn.Module) -> torch.device:
    """
    Get the device that a pipeline or pipeline component exists on.

    :param component: pipeline or pipeline component.
    :return: :py:class:`torch.device`
    """
    if hasattr(component, 'device'):
        return component.device
    elif hasattr(component, 'get_device'):
        return component.get_device()

    raise ValueError(f'component type {component.__class__} did not have a '
                     f'device attribute or the function get_device()')


def get_torch_device_string(component: diffusers.DiffusionPipeline | torch.nn.Module) -> str:
    """
    Get the device string that a pipeline or pipeline component exists on.

    :param component: pipeline or pipeline component.
    :return: device string
    """
    return str(get_torch_device(component))


def _pipeline_to(pipeline, device: torch.device | str | None):
    if device is None:
        _messages.debug_log(
            f'pipeline_to() Not moving pipeline "{pipeline.__class__.__name__}" '
            f'as specified device was None.')
        return

    if not hasattr(pipeline, 'to'):
        _messages.debug_log(
            f'pipeline_to() Not moving pipeline "{pipeline.__class__.__name__}" to '
            f'"{device}" as it has no to() method.')
        return

    device = torch.device(device)

    # Group-offload weights stay in CPU memory for the life of the pipeline.
    # Moving the shell to the accelerator would drop them from the cache size
    # without freeing that memory. Quantized modules were not hooked, so they
    # still have to follow ``device``; sequential offload and model CPU offload
    # do that in their own paths below.
    if is_group_offload_enabled(pipeline):
        _messages.debug_log(
            f'pipeline_to() Not moving group-offloaded weights of '
            f'"{pipeline.__class__.__name__}" to "{device}". '
            f'Placing quantized modules that were left unhooked.')
        for name, model in get_pipeline_modules(pipeline).items():
            if is_group_offload_enabled(model) or not module_skips_group_offload(model):
                continue
            _messages.debug_log(
                f'pipeline_to() Placing quantized module "{name}" on "{device}".')
            place_quantized_module(model, device)
        return

    pipeline_device = get_torch_device(pipeline)

    all_modules_on_device = all(
        _torchutil.devices_equal(device, get_torch_device(m)) for m in get_pipeline_modules(pipeline).values())

    pipeline_on_device = _torchutil.devices_equal(get_torch_device(pipeline), pipeline_device)

    if pipeline_on_device and all_modules_on_device:
        _messages.debug_log(
            f'pipeline_to() Not moving pipeline "{pipeline.__class__.__name__}" to '
            f'"{device}" as it is already on that device.')
        return

    if pipeline_on_device != all_modules_on_device:
        # really the most likely way for this to occur is if
        # an OOM happened moving a pipeline to the GPU, which
        # is something we want to be able to recover from hence
        # the fall through above
        #
        # This also happens when the pipeline has cpu offload
        # enabled, we can fall through that harmlessly as its
        # modules can never be moved to anything but the CPU
        # and that is accounted for below
        _messages.debug_log(
            f'pipeline_to() Moving pipeline "{pipeline.__class__.__name__}" to "{device}", '
            f'pipeline_on_device={pipeline_on_device}, all_modules_on_device={all_modules_on_device}.')

    if not _torchutil.devices_equal(pipeline_device, device):
        try:
            cache_metadata = _pipeline_cache.get_metadata(pipeline)

            if device.type != 'cpu':
                _pipeline_cache.size -= cache_metadata.size

                _messages.debug_log(
                    f'Cached {_types.class_and_id_string(pipeline)} Size = '
                    f'{cache_metadata.size} Bytes '
                    f'({_memory.bytes_best_human_unit(cache_metadata.size)}) '
                    f'is leaving CPU side memory, '
                    f'cache size is now '
                    f'{_pipeline_cache.size} Bytes '
                    f'({_memory.bytes_best_human_unit(_pipeline_cache.size)})')

            else:
                _pipeline_cache.size += cache_metadata.size

                _messages.debug_log(
                    f'Cached {_types.class_and_id_string(pipeline)} Size = '
                    f'{cache_metadata.size} Bytes '
                    f'({_memory.bytes_best_human_unit(cache_metadata.size)}) '
                    f'is entering CPU side memory, '
                    f'cache size is now '
                    f'{_pipeline_cache.size} Bytes '
                    f'({_memory.bytes_best_human_unit(_pipeline_cache.size)})')
        except _d_memoize.ObjectCacheKeyError:
            # does not exist in the cache
            pass

    for name, value in get_pipeline_modules(pipeline).items():

        is_loaded_in_8bit_bnb = _util.is_loaded_in_8bit_bnb(value)

        if is_loaded_in_8bit_bnb:
            _messages.debug_log(
                f'pipeline_to() Not moving module "{name} = {value.__class__.__name__}" to "{device}" '
                f'as it is loaded in 8bit mode via bitsandbytes.')
            _disable_to(value)
            continue

        current_device = get_torch_device(value)

        if current_device.type == 'meta':
            _messages.debug_log(
                f'pipeline_to() Not moving module "{name} = {value.__class__.__name__}" to "{device}" '
                f'as its device value is "meta".')
            _disable_to(
                value,
                name == 'vae'
            )
            continue

        if _torchutil.devices_equal(current_device, device):
            _messages.debug_log(
                f'pipeline_to() Not moving module "{name} = {value.__class__.__name__}" to "{device}" '
                f'as it is already on that device.')
            continue

        if (is_model_cpu_offload_enabled(value) or is_group_offload_enabled(value)) \
                and device.type != 'cpu':
            _messages.debug_log(
                f'pipeline_to() Not moving module "{name} = {value.__class__.__name__}" to "{device}" '
                f'as it has cpu offload enabled and can only move to cpu.')
            continue

        _messages.debug_log(
            f'pipeline_to() Moving module "{name}" of pipeline {_types.fullname(pipeline)} '
            f'from device "{current_device}" to device "{device}"')

        value.to(device)

    if device.type == 'cpu':
        _memory.torch_gc()


def pipeline_to(pipeline, device: torch.device | str | None):
    """
    Move a diffusers pipeline to a device if possible, in a way that dgenerate can keep track of.

    This calls methods associated with updating the cache statistics such as
    :py:func:`dgenerate.pipelinewrapper.pipeline_off_cpu_update_cache_info` and
    :py:func:`dgenerate.pipelinewrapper.pipeline_to_cpu_update_cache_info` for you,
    as well as the associated cache update functions for the pipelines individual
    components as needed.

    If ``device==None`` this is a no-op.

    Modules which are meta tensors will not be moved (sequentially offloaded modules)

    Modules which have model cpu offload enabled will not be moved unless they are moving to "cpu"

    :raise dgenerate.OutOfMemoryError: if there is not enough memory on the specified device

    :param pipeline: the pipeline
    :param device: the device

    :return: the moved pipeline
    """

    try:
        _pipeline_to(pipeline=pipeline, device=device)
    except _d_exceptions.TORCH_CUDA_OOM_EXCEPTIONS as e:
        _d_exceptions.raise_if_not_cuda_oom(e)
        # attempt to recover VRAM before rethrowing
        # move any modules back to cpu which have entered VRAM

        _pipeline_to(pipeline=pipeline, device='cpu')
        _memory.torch_gc()
        gc.collect()

        raise _d_exceptions.OutOfMemoryError(e) from e
    except MemoryError as e:
        # probably out of RAM on a back
        # to CPU move not much we can do
        gc.collect()
        raise _d_exceptions.OutOfMemoryError('cpu (system memory)') from e


def _call_args_debug_transformer(key, value):
    if isinstance(value, torch.Generator):
        return f'torch.Generator(seed={value.initial_seed()})'
    if isinstance(value, torch.Tensor):
        return f'torch.Tensor({value.shape})'
    return value


def _prompt_token_counter(tokenizer):
    """
    Return an object that can ``tokenize`` a prompt.

    Flux.2 stores a ``PixtralProcessor`` on ``pipeline.tokenizer``. The
    processor has no ``model_max_length`` or ``tokenize``; those live on
    the wrapped tokenizer.
    """
    if hasattr(tokenizer, 'tokenize'):
        return tokenizer
    inner = getattr(tokenizer, 'tokenizer', None)
    if inner is not None and hasattr(inner, 'tokenize'):
        return inner
    return None


def _prompt_max_length(pipeline, tokenizer, tokenizer_attr, kwargs):
    name = pipeline.__class__.__name__
    model_max = getattr(tokenizer, 'model_max_length', None)
    if tokenizer_attr == 'tokenizer_3' and name.startswith('StableDiffusion3') and model_max is not None:
        return min(kwargs.get('max_sequence_length', 256), model_max)
    if tokenizer_attr == 'tokenizer_2' and name.startswith('Flux') and model_max is not None:
        return min(kwargs.get('max_sequence_length', 512), model_max)
    if model_max is not None:
        return model_max

    # Processors such as PixtralProcessor omit model_max_length. The
    # pipeline truncates at max_sequence_length, falling back to its
    # own tokenizer_max_length (512 for Flux.2).
    sequence = kwargs.get('max_sequence_length')
    if sequence is not None:
        return sequence
    if tokenizer_attr == 'tokenizer':
        pipeline_max = getattr(pipeline, 'tokenizer_max_length', None)
        if pipeline_max is not None:
            return pipeline_max
    inner = getattr(tokenizer, 'tokenizer', None)
    return getattr(inner, 'model_max_length', None)


def _warn_prompt_lengths(pipeline, **kwargs):
    prompts = [
        ('Primary positive prompt', kwargs.get('prompt'), 'tokenizer'),
        ('Primary negative prompt', kwargs.get('negative_prompt'), 'tokenizer'),
        ('Secondary positive prompt', kwargs.get('prompt_2'), 'tokenizer_2'),
        ('Secondary negative prompt', kwargs.get('negative_prompt_2'), 'tokenizer_2'),
        ('Tertiary positive prompt', kwargs.get('prompt_3'), 'tokenizer_3'),
        ('Tertiary negative prompt', kwargs.get('negative_prompt_3'), 'tokenizer_3')
    ]

    warned_prompts = {}

    for label, prompt, tokenizer_attr in prompts:
        if prompt and not isinstance(prompt, list):
            prompt = [prompt]

        if prompt:
            tokenizer = getattr(pipeline, tokenizer_attr, None)

            if tokenizer:
                counter = _prompt_token_counter(tokenizer)
                max_length = _prompt_max_length(pipeline, tokenizer, tokenizer_attr, kwargs)
                if counter is None or max_length is None:
                    continue

                for p in prompt:
                    if len(counter.tokenize(p)) > max_length:
                        key = f'{label}{tokenizer_attr}{p}'
                        if key not in warned_prompts:
                            _messages.warning(
                                f'{label} exceeds max token length '
                                f'of {max_length} for the model\'s tokenizer '
                                f'and will be truncated: "{p}"'
                            )
                            warned_prompts[key] = True


_LAST_CALLED_PIPELINE = None


def get_last_called_pipeline() -> diffusers.DiffusionPipeline | None:
    """
    Get a reference to the globally cached pipeline last called with :py:func:`call_pipeline`.

    This value may be ``None`` if a pipeline was never called.

    :return: diffusion pipeline object
    """
    return _LAST_CALLED_PIPELINE


def destroy_last_called_pipeline(collect=True):
    """
    Move to CPU and dereference the globally cached pipeline last called with :py:func:`call_pipeline`.

    This is a no-op if a pipeline has never been called with :py:func:`call_pipeline`

    :param collect: call ``gc.collect`` and :py:func:`dgenerate.memory.torch_gc` if
        there is a pipeline to dereference?
    """
    global _LAST_CALLED_PIPELINE

    if _LAST_CALLED_PIPELINE is not None:

        pipeline_to(_LAST_CALLED_PIPELINE, 'cpu')

        _LAST_CALLED_PIPELINE = None

        if collect:
            gc.collect()
            _memory.torch_gc()


def _evict_last_pipeline(device: torch.device | None):
    active_pipe = get_last_called_pipeline()

    if active_pipe is None:
        return

    if device is None or _torchutil.devices_equal(get_torch_device(active_pipe), device):
        # get rid of this reference immediately
        # noinspection PyUnusedLocal
        active_pipe = None

        _messages.debug_log(
            f'{_types.fullname(_devicecache.clear_device_cache)} is attempting to evacuate any previously '
            f'called diffusion pipeline in the VRAM of device: {device}.')

        # potentially free up VRAM on the GPU we are
        # about to move to
        destroy_last_called_pipeline()


def _evict_8bit_bnb_pipelines(device: torch.device | None):
    # clear out any 8bit bnb pipelines in the cache, they are
    # on the GPU and cannot be moved
    for cached_pipeline in _pipeline_cache.values():

        bit8 = any(_util.is_loaded_in_8bit_bnb(module) for
                   module in get_pipeline_modules(cached_pipeline.pipeline).values())

        if bit8 and (device is None or _torchutil.devices_equal(device, get_torch_device(cached_pipeline.pipeline))):
            _messages.debug_log(
                f'Clearing out cached 8bit pipeline from cache: {cached_pipeline.pipeline.__class__.__name__}')
            _pipeline_cache.un_cache(cached_pipeline)
            del cached_pipeline


_devicecache.register_eviction_method(_evict_last_pipeline)
_devicecache.register_eviction_method(_evict_8bit_bnb_pipelines)


def _reset_sada_cache(pipeline) -> None:
    """Start SADA's step cache over for one pipeline call.

    ADetailer calls the pipeline once per detected region. SADA's cache bus
    lives on the denoiser for the whole session, so a later region would
    otherwise skip steps using the previous region's cached output.
    """
    denoiser = getattr(pipeline, 'unet', None)
    if denoiser is None:
        denoiser = getattr(pipeline, 'transformer', None)
    if denoiser is None or getattr(denoiser, '_cache_bus', None) is None:
        return
    from dgenerate.extras.sada.patch import reset_cache
    reset_cache(pipeline)


# noinspection PyCallingNonCallable
@torch.inference_mode()
def call_pipeline(pipeline: diffusers.DiffusionPipeline,
                  device: torch.device | str | None = _torchutil.default_device(),
                  prompt_weighter: _promptweighters.PromptWeighter = None,
                  **kwargs):
    """
    Call a diffusers pipeline, offload the last called pipeline to CPU before
    doing so if the last pipeline is not being called in succession


    :param pipeline: The pipeline

    :param device: The device to move the pipeline to before calling, it will be
        moved to this device if it is not already on the device. If the pipeline
        does not support moving to specific device, such as with sequentially offloaded
        pipelines which cannot move at all, or cpu offloaded pipelines which can
        only move to CPU, this argument is ignored.

    :param kwargs: diffusers pipeline keyword arguments

    :param prompt_weighter: Optional prompt weighter for weighted prompt syntaxes

    :raises dgenerate.OutOfMemoryError: if there is not enough memory on the specified device

    :raises UnsupportedPipelineConfiguration:
        If the pipeline is missing certain required modules, such as text encoders.

    :return: the result of calling the diffusers pipeline
    """

    global _LAST_CALLED_PIPELINE

    if prompt_weighter is not None:
        if not _torchutil.devices_equal(device, prompt_weighter.device):
            raise UnsupportedPipelineConfigError(
                'dgenerate.pipelinewrapper.call_pipeline: prompt_weighter '
                'must specify the same compute device that pipeline is to be called on. '
                f'Got prompt_weighter={prompt_weighter.device}, and pipeline={device}')

    _messages.debug_log(
        f'Calling Pipeline: "{pipeline.__class__.__name__}",',
        f'Device: "{device}",',
        'Args:',
        lambda: _textprocessing.debug_format_args(
            kwargs, value_transformer=_call_args_debug_transformer))

    enable_retry_pipe = True

    def _cleanup_prompt_weighter():
        try:
            _messages.debug_log(
                f'Executing prompt weighter cleanup for "{prompt_weighter.__class__.__name__}"')
            prompt_weighter.cleanup()
        except Exception as e:
            _messages.debug_log(
                f'Ignoring prompt weighter cleanup '
                f'exception in "{prompt_weighter.__class__.__name__}.cleanup()": {e}')
            pass

    def _call_prompt_weighter():
        nonlocal enable_retry_pipe

        try:
            translated = prompt_weighter.translate_to_embeds(pipeline, kwargs)
        except _d_exceptions.TORCH_CUDA_OOM_EXCEPTIONS as e:
            _d_exceptions.raise_if_not_cuda_oom(e)
            _cleanup_prompt_weighter()
            _memory.torch_gc()
            gc.collect()
            raise _d_exceptions.OutOfMemoryError(e) from e
        except MemoryError as e:
            _cleanup_prompt_weighter()
            gc.collect()
            raise _d_exceptions.OutOfMemoryError('cpu (system memory)') from e
        except Exception:
            _cleanup_prompt_weighter()
            _memory.torch_gc()
            gc.collect()
            raise

        def _debug_string_func():
            return f'{prompt_weighter.__class__.__name__} translated pipeline call args to: ' + \
                _textprocessing.debug_format_args(
                    translated,
                    value_transformer=_call_args_debug_transformer)

        _messages.debug_log(_debug_string_func)

        return translated

    prompt_warning_issued = False

    def _call_pipeline_raw():
        nonlocal prompt_warning_issued

        def _invoke(call_kwargs):
            _reset_sada_cache(pipeline)
            settings = _spectrum.active_settings()
            if settings is None:
                return pipeline(**call_kwargs)
            steps = call_kwargs.get('num_inference_steps', settings.fallback_steps)
            try:
                steps = int(steps)
            except (TypeError, ValueError):
                steps = settings.fallback_steps
            try:
                with _spectrum.apply(pipeline, steps, settings):
                    return pipeline(**call_kwargs)
            except _spectrum.SpectrumUnsupported as error:
                raise UnsupportedPipelineConfigError(str(error)) from error

        try:
            if prompt_weighter is None:
                if not prompt_warning_issued:
                    _warn_prompt_lengths(pipeline, **kwargs)
                    prompt_warning_issued = True
                pipeline_to(pipeline, device)
                pipe_result = _invoke(kwargs)
            else:
                args = _call_prompt_weighter()
                pipeline_to(pipeline, device)
                pipe_result = _invoke(args)
                prompt_weighter.cleanup()
            return pipe_result
        except TypeError as e:
            null_call_name = _types.get_null_call_name(e)
            if null_call_name:
                raise UnsupportedPipelineConfigError(
                    'Missing pipeline module?, cannot call: ' + null_call_name)
            raise
        except AttributeError as e:
            null_attr_name = _types.get_null_attr_name(e)
            if null_attr_name:
                raise UnsupportedPipelineConfigError(
                    'Missing pipeline module?, cannot access: ' + null_attr_name)
            raise

    def _torch_oom_handler():
        global _LAST_CALLED_PIPELINE

        if pipeline is _LAST_CALLED_PIPELINE:
            _LAST_CALLED_PIPELINE = None

        # move the torch pipeline back to the CPU
        pipeline_to(pipeline, 'cpu')

        # empty the CUDA cache
        _memory.torch_gc()

        # force garbage collection
        gc.collect()

    def _call_pipeline():
        nonlocal enable_retry_pipe
        old_execution_device_property = None
        try:
            if hasattr(pipeline, '_execution_device'):
                # HACK
                # The device this returns is sometimes wrong and causes issues
                # with a randomly generated tensor (complaining about) being
                # generated on the wrong device as compared to the torch.Generator
                # object being used to generate it, this is a diffusers problem in
                # the code of this private property
                old_execution_device_property = pipeline.__class__._execution_device
                pipeline.__class__._execution_device = property(lambda s: torch.device(device))

            return _call_pipeline_raw()
        except _d_exceptions.TORCH_CUDA_OOM_EXCEPTIONS as e:
            _d_exceptions.raise_if_not_cuda_oom(e)
            _torch_oom_handler()
            raise _d_exceptions.OutOfMemoryError(e) from e
        except MemoryError as e:
            gc.collect()
            raise _d_exceptions.OutOfMemoryError('cpu (system memory)') from e
        except Exception:
            # same cleanup
            _torch_oom_handler()
            raise
        finally:
            if old_execution_device_property is not None:
                pipeline.__class__._execution_device = old_execution_device_property

    for cached_pipeline in _pipeline_cache.values():
        # hack to clear out cached 8bit pipelines, only one should ever
        # be in the object cache for sequential calls, this is necessary
        # for efficient main pipeline recall in DiffusionPipelineWrapper
        # which is used for the adetailer post processor
        if cached_pipeline.pipeline is not pipeline:
            bit8 = any(_util.is_loaded_in_8bit_bnb(module) for
                       module in get_pipeline_modules(cached_pipeline.pipeline).values())
            if bit8 and _torchutil.devices_equal(device, get_torch_device(cached_pipeline.pipeline)):
                _messages.debug_log(
                    f'Clearing out cached 8bit pipeline from cache: {cached_pipeline.pipeline.__class__.__name__}')
                _pipeline_cache.un_cache(cached_pipeline)
                del cached_pipeline
                gc.collect()
                _memory.torch_gc()

    if pipeline is _LAST_CALLED_PIPELINE:
        try:
            return _call_pipeline()
        except _d_exceptions.OutOfMemoryError:
            if not enable_retry_pipe:
                raise

            _messages.debug_log(
                f'Attempting to call pipeline '
                f'"{pipeline.__class__.__name__}" again after out '
                f'of memory condition and cleanup.')

            # retry after memory cleanup
            result = _call_pipeline()
            _LAST_CALLED_PIPELINE = pipeline
            return result

    if _LAST_CALLED_PIPELINE is not None and hasattr(_LAST_CALLED_PIPELINE, 'to'):
        _messages.debug_log(
            f'Moving previously called pipeline '
            f'"{_LAST_CALLED_PIPELINE.__class__.__name__}", back to the CPU.')

        pipeline_to(_LAST_CALLED_PIPELINE, 'cpu')

    try:
        result = _call_pipeline()
    except _d_exceptions.OutOfMemoryError:
        if not enable_retry_pipe:
            raise
        _messages.debug_log(
            f'Attempting to call pipeline '
            f'"{pipeline.__class__.__name__}" again after out '
            f'of memory condition and cleanup.')
        # allow for memory cleanup and try again
        # might be able to run now
        result = _call_pipeline()

    _LAST_CALLED_PIPELINE = pipeline
    return result


class PipelineCreationResult:

    model_path: _types.OptionalPath
    """
    Path the model was loaded from.
    """

    parsed_unet_uri: _uris.UNetUri | None
    """
    Parsed UNet URI if one was present
    """

    parsed_vae_uri: _uris.VAEUri | None
    """
    Parsed VAE URI if one was present
    """

    parsed_lora_uris: collections.abc.Sequence[_uris.LoRAUri]
    """
    Parsed LoRA URIs if any were present
    """

    parsed_ip_adapter_uris: collections.abc.Sequence[_uris.IPAdapterUri]
    """
    Parsed IP Adapter URIs if any were present
    """

    parsed_textual_inversion_uris: collections.abc.Sequence[_uris.TextualInversionUri]
    """
    Parsed Textual Inversion URIs if any were present
    """

    parsed_controlnet_uris: collections.abc.Sequence[_uris.ControlNetUri]
    """
    Parsed ControlNet URIs if any were present
    """

    parsed_t2i_adapter_uris: collections.abc.Sequence[_uris.T2IAdapterUri]
    """
    Parsed T2IAdapter URIs if any were present
    """

    parsed_image_encoder_uri: _uris.ImageEncoderUri | None
    """
    Parsed ImageEncoder URI if one was present
    """

    parsed_transformer_uri: _uris.TransformerUri | None
    """
    Parsed Transformer URI if one was present
    """

    def load_scheduler(self, scheduler_uri: _types.Uri | None):
        """
        Load a scheduler onto the pipeline using a URI specification.

        Passing ``None`` to the URI reloads the original scheduler that the model was loaded
        with, if no new scheduler has been set since then, this is a no-op.

        :param scheduler_uri: The scheduler URI
        """
        _schedulers.load_scheduler(self.pipeline, scheduler_uri)

    def set_vae_tiling_and_slicing(self, vae_tiling: bool, vae_slicing: bool):
        """
        Set the VAE tiling and slicing status of the pipeline.

        :param vae_tiling: vae tiling?
        :param vae_slicing: vae slicing?
        """
        set_vae_tiling_and_slicing(self.pipeline, tiling=vae_tiling, slicing=vae_slicing)

    def __init__(self,
                 model_path: _types.Path,
                 pipeline: diffusers.DiffusionPipeline,
                 parsed_unet_uri: _uris.UNetUri | None,
                 parsed_transformer_uri: _uris.TransformerUri | None,
                 parsed_vae_uri: _uris.VAEUri | None,
                 parsed_image_encoder_uri: _uris.ImageEncoderUri | None,
                 parsed_lora_uris: collections.abc.Sequence[_uris.LoRAUri],
                 parsed_ip_adapter_uris: collections.abc.Sequence[_uris.IPAdapterUri],
                 parsed_textual_inversion_uris: collections.abc.Sequence[_uris.TextualInversionUri],
                 parsed_controlnet_uris: collections.abc.Sequence[_uris.ControlNetUri],
                 parsed_t2i_adapter_uris: collections.abc.Sequence[_uris.T2IAdapterUri]):
        self.model_path = model_path
        self.parsed_unet_uri = parsed_unet_uri
        self.parsed_vae_uri = parsed_vae_uri
        self.parsed_lora_uris = parsed_lora_uris
        self.parsed_textual_inversion_uris = parsed_textual_inversion_uris
        self.parsed_controlnet_uris = parsed_controlnet_uris
        self.parsed_t2i_adapter_uris = parsed_t2i_adapter_uris
        self.parsed_ip_adapter_uris = parsed_ip_adapter_uris
        self.parsed_image_encoder_uri = parsed_image_encoder_uri
        self.parsed_transformer_uri = parsed_transformer_uri
        self._pipeline = pipeline

    @property
    def pipeline(self):
        return self._pipeline

    def get_pipeline_modules(self, names: collections.abc.Iterable[str]):
        """
        Get associated pipeline module such as ``vae`` etc, in
        a dictionary mapped from name to module value.

        Possible Module Names:

            * ``unet``
            * ``vae``
            * ``transformer``
            * ``text_encoder``
            * ``text_encoder_2``
            * ``text_encoder_3``
            * ``tokenizer``
            * ``tokenizer_2``
            * ``tokenizer_3``
            * ``safety_checker``
            * ``feature_extractor``
            * ``image_encoder``
            * ``adapter``
            * ``controlnet``
            * ``scheduler``


        If the module is not present or a recognized name, a :py:exc:`ValueError`
        will be thrown describing the module that is not part of the pipeline.

        :raise ValueError:

        :param names: module names, such as ``vae``, ``text_encoder``
        :return: dictionary
        """

        module_values = dict()

        acceptable_lookups = {
            'unet',
            'vae',
            'transformer',
            'text_encoder',
            'text_encoder_2',
            'text_encoder_3',
            'tokenizer',
            'tokenizer_2',
            'tokenizer_3',
            'safety_checker',
            'feature_extractor',
            'image_encoder',
            'adapter',
            'controlnet',
            'scheduler'
        }

        for name in names:
            if name not in acceptable_lookups:
                raise ValueError(f'"{name}" is not a recognized pipeline module name.')
            if not hasattr(self.pipeline, name):
                raise ValueError(f'Created pipeline does not possess a module named: "{name}".')
            module_values[name] = getattr(self.pipeline, name)

        return module_values

    def call(self,
             device: torch.device | str | None = _torchutil.default_device(),
             prompt_weighter: _promptweighters.PromptWeighter | None = None,
             **kwargs) -> diffusers.utils.BaseOutput:
        """
        Call **pipeline**, see: :py:func:`.call_pipeline`

        :param device: move the pipeline to this device before calling
        :param prompt_weighter: Optional prompt weighter for weighted prompt syntaxes
        :param kwargs: forward kwargs to pipeline
        :return: A subclass of :py:class:`diffusers.utils.BaseOutput`
        """
        return call_pipeline(self.pipeline,
                             device,
                             prompt_weighter,
                             **kwargs)


def create_diffusion_pipeline(
        model_path: str,
        model_type: _enums.ModelType = _enums.ModelType.SD,
        pipeline_type: _enums.PipelineType = _enums.PipelineType.TXT2IMG,
        revision: _types.OptionalString = None,
        variant: _types.OptionalString = None,
        subfolder: _types.OptionalString = None,
        dtype: _enums.DataType = _enums.DataType.AUTO,
        unet_uri: _types.OptionalUri = None,
        transformer_uri: _types.OptionalUri = None,
        vae_uri: _types.OptionalUri = None,
        lora_uris: _types.OptionalUris = None,
        lora_fuse_scale: _types.OptionalFloat = None,
        image_encoder_uri: _types.OptionalUri = None,
        ip_adapter_uris: _types.OptionalUris = None,
        textual_inversion_uris: _types.OptionalUris = None,
        text_encoder_uris: _types.OptionalUris = None,
        controlnet_uris: _types.OptionalUris = None,
        t2i_adapter_uris: _types.OptionalUris = None,
        quantizer_uri: _types.OptionalUri = None,
        quantizer_map: _types.OptionalStrings = None,
        pag: bool = False,
        safety_checker: bool = False,
        original_config: _types.OptionalString = None,
        auth_token: _types.OptionalString = None,
        device: str = _torchutil.default_device(),
        extra_modules: dict[str, typing.Any] | None = None,
        model_cpu_offload: bool = False,
        sequential_cpu_offload: bool = False,
        model_group_offload: bool = False,
        torch_compile: bool = False,
        local_files_only: bool = False,
        missing_submodules_ok: bool = False
) -> PipelineCreationResult:
    """
    Create a :py:class:`diffusers.DiffusionPipeline` in dgenerate's in memory cacheing system.

    :param model_type:  :py:class:`dgenerate.pipelinewrapper.ModelType` enum value
    :param model_path: huggingface slug, huggingface blob link, path to folder on disk, path to file on disk
    :param pipeline_type: :py:class:`dgenerate.pipelinewrapper.PipelineType` enum value
    :param revision: huggingface repo revision (branch)
    :param variant: model weights name variant, for example 'fp16'
    :param subfolder: huggingface repo subfolder if applicable
    :param dtype: Optional :py:class:`dgenerate.pipelinewrapper.DataType` enum value
    :param unet_uri: Optional ``--unet`` URI string for specifying a specific UNet
    :param transformer_uri: Optional ``--transformer`` URI string for specifying a specific Transformer.
        Supported for Stable Diffusion 3, Flux, Flux.2, Z-Image, Qwen-Image, and LTX.
    :param vae_uri: Optional ``--vae`` URI string for specifying a specific VAE
    :param lora_uris: Optional ``--loras`` URI strings for specifying LoRA weights
    :param lora_fuse_scale: Optional ``--lora-fuse-scale`` global LoRA fuse scale value.
        Once all LoRAs are merged with their individual scales, the merged weights will be fused
        into the pipeline at this scale. The default value is 1.0.
    :param image_encoder_uri: Optional ``--image-encoder`` URI for use with IP Adapter weights or Stable Cascade
    :param ip_adapter_uris: Optional ``--ip-adapters`` URI strings for specifying IP Adapter weights
    :param textual_inversion_uris: Optional ``--textual-inversions`` URI strings for specifying Textual Inversion weights
    :param text_encoder_uris: Optional user specified ``--text-encoders`` URIs that will be loaded on to the
        pipeline in order. A uri value of ``+`` or ``None`` indicates use default, a string value of ``null``
        indicates to explicitly not load any encoder all
    :param controlnet_uris: Optional ``--control-nets`` URI strings for specifying ControlNet models
    :param t2i_adapter_uris: Optional ``--t2i-adapters`` URI strings for specifying T2IAdapter models
    :param quantizer_uri: Optional ``--quantizer`` URI value
    :param quantizer_map: Collection of pipeline submodule names to which quantization should be applied when
        ``quantizer_uri`` is provided. Valid values are ``unet``, ``transformer``, ``text_encoder``,
        ``text_encoder_2``, ``text_encoder_3``, ``controlnet``, and ``connectors``. If ``None``, the UNet
        or transformer and every text encoder are quantized. ControlNet stays full precision unless named here.
    :param pag: Use perturbed attention guidance?
    :param safety_checker: Safety checker enabled? default is ``False``
    :param original_config: Optional original training config .yaml file path when loading a single file checkpoint.
    :param auth_token: Optional huggingface API token for accessing repositories that are restricted to your account
    :param device: Optional ``--device`` string, defaults to "cuda"
    :param extra_modules: Extra module arguments to pass directly into
        :py:meth:`diffusers.DiffusionPipeline.from_single_file` or :py:meth:`diffusers.DiffusionPipeline.from_pretrained`
    :param model_cpu_offload: This pipeline has model_cpu_offloading enabled?
    :param sequential_cpu_offload: This pipeline has sequential_cpu_offloading enabled?
    :param model_group_offload: This pipeline has group offload enabled?
    :param torch_compile: Compile repeated denoiser, ControlNet, and VAE blocks when the device
        can compile them. Quantized modules stay eager. ``DGENERATE_TORCH_COMPILE=0`` leaves
        the blocks eager. TorchDynamo, Inductor, and Triton stay quiet unless ``-v`` is set.
    :param local_files_only: Only look in the huggingface cache and do not connect to download models?
    :param missing_submodules_ok: It is okay if Text Encoders or VAE is missing from the checkpoint?

    :raises InvalidModelFileError:
    :raises InvalidModelUriError:
    :raises InvalidSchedulerNameError:
    :raises UnsupportedPipelineConfigError:
    :raises dgenerate.ModelNotFoundError:
    :raises dgenerate.ConfigNotFoundError:
    :raises dgenerate.NonHFModelDownloadError:
    :raises dgenerate.NonHFConfigDownloadError:
    :raises dgenerate.WebFileCacheOfflineModeException:

    :return: :py:class:`.TorchPipelineCreationResult`
    """
    __locals = locals()

    for name, value in __locals.items():
        if name.endswith('_uris') and isinstance(value, str):
            __locals[name] = [value]

    # dgenerate's own component loads (text encoders, VAEs) read only their
    # weights from a single file, unless the model has a UNet, which
    # UNetUri reads from the file too. The pipeline's own single file load
    # narrows this to whether dgenerate supplied the transformer.
    transformer_loaded = diffusion_transformer_class(__locals['model_type']) is not None

    with _hfhub.with_hf_errors_as_model_not_found(), \
            _safetensorscompat.single_file_components(transformer_loaded=transformer_loaded):
        return _create_diffusion_pipeline(**__locals)

class PipelineFactory:
    """
    Turns :py:func:`.create_diffusion_pipeline` into a factory that can
    repeatedly create a pipeline with the same arguments, possibly from cache.
    """

    def __init__(self,
                 model_path: str,
                 model_type: _enums.ModelType = _enums.ModelType.SD,
                 pipeline_type: _enums.PipelineType = _enums.PipelineType.TXT2IMG,
                 revision: _types.OptionalString = None,
                 variant: _types.OptionalString = None,
                 subfolder: _types.OptionalString = None,
                 dtype: _enums.DataType = _enums.DataType.AUTO,
                 unet_uri: _types.OptionalUri = None,
                 transformer_uri: _types.OptionalUri = None,
                 vae_uri: _types.OptionalUri = None,
                 lora_uris: _types.OptionalUris = None,
                 lora_fuse_scale: _types.OptionalFloat = None,
                 image_encoder_uri: _types.OptionalUri = None,
                 ip_adapter_uris: _types.OptionalUris = None,
                 textual_inversion_uris: _types.OptionalUris = None,
                 controlnet_uris: _types.OptionalUris = None,
                 t2i_adapter_uris: _types.OptionalUris = None,
                 text_encoder_uris: _types.OptionalUris = None,
                 quantizer_uri: _types.OptionalUri = None,
                 quantizer_map: _types.OptionalStrings = None,
                 pag: bool = False,
                 safety_checker: bool = False,
                 original_config: _types.OptionalString = None,
                 auth_token: _types.OptionalString = None,
                 device: str = _torchutil.default_device(),
                 extra_modules: dict[str, typing.Any] | None = None,
                 model_cpu_offload: bool = False,
                 sequential_cpu_offload: bool = False,
                 model_group_offload: bool = False,
                 torch_compile: bool = False,
                 local_files_only: bool = False):
        self._args = {k: v for k, v in
                      _types.partial_deep_copy_container(locals()).items()
                      if k not in {'self'}}

    def __call__(self) -> PipelineCreationResult:
        """
        :raises InvalidModelFileError:
        :raises ModelNotFoundError:
        :raises InvalidModelUriError:
        :raises InvalidSchedulerNameError:
        :raises UnsupportedPipelineConfigError:
        :raises dgenerate.NonHFModelDownloadError:
        :raises dgenerate.NonHFConfigDownloadError:

        :return: :py:class:`.TorchPipelineCreationResult`
        """
        return create_diffusion_pipeline(**self._args)


def _format_pipeline_creation_debug_arg(arg_name, v):
    if isinstance(v, torch.dtype):
        return str(v)

    if isinstance(v, str):
        return f'"{v}"'

    if v.__class__.__module__ != 'builtins':
        return _types.class_and_id_string(v)

    if isinstance(v, list):
        return '[' + ', '.join(_format_pipeline_creation_debug_arg(None, v) for v in v) + ']'

    if isinstance(v, (set, frozenset)):
        return '{' + ', '.join(_format_pipeline_creation_debug_arg(None, v) for v in v) + '}'

    if isinstance(v, dict):
        return '{' + ', '.join(f'"{k}"={_format_pipeline_creation_debug_arg(None, v)}' for k, v in v.items()) + '}'

    return str(v)


def _pipeline_creation_args_debug(backend, cls, method, model, **kwargs):
    _messages.debug_log(
        lambda:
        f'{backend} Pipeline Creation Call: {cls.__name__}.{method.__name__}("{model}", ' +
        _textprocessing.debug_format_args(kwargs, _format_pipeline_creation_debug_arg, as_kwargs=True) + ')')

    return method(model, **kwargs)


def _text_encoder_default(uri):
    return uri is None or uri.strip() == '+'


def _text_encoder_null(uri):
    return uri and uri.strip().lower() == 'null'


def _load_sd_family_embedded_or_builtin_vae(
        load_vae,
        *,
        vae_uri: _types.OptionalUri,
        model_type: _enums.ModelType,
        local_files_only: bool,
        dtype: _enums.DataType,
        model_path: str,
        load_embedded: typing.Callable[[], typing.Any],
) -> tuple[typing.Any, _uris.VAEUri | None]:
    """
    Load an SD-family VAE from the checkpoint or model repo, or the built-in
    Hugging Face VAE for SD 1.x / SD 2 / SDXL when none is embedded (#62).
    """
    parsed_vae_uri = None
    try:
        return load_embedded(), parsed_vae_uri
    except (_d_exceptions.ModelNotFoundError, _uris.VAEUriLoadError) as e:
        if not _singlefilecompat._sd_family_should_load_builtin_vae(
                vae_uri, model_type, local_files_only):
            raise UnsupportedPipelineConfigError(
                _singlefilecompat._sd_family_missing_embedded_vae_message(
                    local_files_only, model_type, model_path)
            ) from e
        parsed_vae_uri = _singlefilecompat._sd_family_builtin_vae_uri(
            model_type, dtype, model_path, local_files_only=local_files_only)
        _messages.warning(
            f'SD-family checkpoint has no embedded VAE; loading built-in from '
            f'"{parsed_vae_uri.model}"'
            + (f' subfolder "{parsed_vae_uri.subfolder}"'
               if parsed_vae_uri.subfolder else '')
            + '.')
        return load_vae(parsed_vae_uri), parsed_vae_uri


def _pipeline_args_hasher(args):
    def text_encoder_uri_parse(uri):
        if _text_encoder_default(uri):
            return None

        if uri.strip() == 'help':
            return 'help'

        if uri.strip() == 'null':
            return 'null'

        return _uris.TextEncoderUri.parse(uri)

    quantizer_uri = args['quantizer_uri']

    custom_hashes = {
        'unet_uri': _uris.uri_hash_with_parser(_uris.UNetUri.parse),
        'transformer_uri': _uris.uri_hash_with_parser(_uris.TransformerUri),
        'vae_uri': _uris.uri_hash_with_parser(_uris.VAEUri.parse),
        'image_encoder_uri': _uris.uri_hash_with_parser(_uris.ImageEncoderUri),
        'lora_uris': _uris.uri_list_hash_with_parser(_uris.LoRAUri.parse),
        'ip_adapter_uris': _uris.uri_list_hash_with_parser(_uris.IPAdapterUri),
        'textual_inversion_uris': _uris.uri_list_hash_with_parser(_uris.TextualInversionUri.parse),
        'text_encoder_uris': _uris.uri_list_hash_with_parser(text_encoder_uri_parse),
        'controlnet_uris': _uris.uri_list_hash_with_parser(
            lambda s: _uris.ControlNetUri.parse(s, model_type=args['model_type']),
            exclude={'scale', 'start', 'end'}),
        't2i_adapter_uris': _uris.uri_list_hash_with_parser(_uris.T2IAdapterUri.parse,
                                                            exclude={'scale'}),
        'quantizer_uri':
            _uris.uri_hash_with_parser(
                _uris.get_quantizer_uri_class(quantizer_uri).parse)
            if quantizer_uri else lambda x: None,
        'quantizer_map': lambda x: hash(tuple(sorted(x))) if x else None
    }
    return _d_memoize.args_cache_key(args, custom_hashes=custom_hashes)


def _pipeline_on_hit(key, hit):
    _d_memoize.simple_cache_hit_debug("Torch Pipeline", key, hit.pipeline)


def _pipeline_on_create(key, new):
    _d_memoize.simple_cache_miss_debug('Torch Pipeline', key, new.pipeline)


def pipeline_class_supports_textual_inversion(cls: typing.Type[diffusers.DiffusionPipeline]):
    """
    Does a pipeline class support Textual Inversions?

    :param cls: ``diffusers`` pipeline class
    :return: ``True`` or ``False``
    """
    return any('TextualInversionLoaderMixin' in x.__name__ for x in cls.__bases__)


def pipeline_class_supports_lora(cls: typing.Type[diffusers.DiffusionPipeline]):
    """
    Does a pipeline class support LoRAs?

    :param cls: ``diffusers`` pipeline class
    :return: ``True`` or ``False``
    """
    return any('LoraLoaderMixin' in x.__name__ for x in cls.__mro__)


def pipeline_class_supports_ip_adapter(cls: typing.Type[diffusers.DiffusionPipeline]):
    """
    Does a pipeline class support IP Adapters?

    :param cls: ``diffusers`` pipeline class
    :return: ``True`` or ``False``
    """
    return any('IPAdapterMixin' in x.__name__ for x in cls.__bases__)


def qwen_image_call_guidance(
        guidance_scale: float,
        negative_prompt: str | None
) -> dict[str, typing.Any]:
    """
    Map ``--guidance-scales`` onto Qwen-Image ``true_cfg_scale``.

    ``guidance_scale`` is the distilled embedded guidance and stays unset.
    True CFG runs only when ``true_cfg_scale > 1`` and a negative prompt is
    present. A single space is the negative prompt the pipeline docs use
    when the user did not give one.
    """
    kwargs: dict[str, typing.Any] = {'true_cfg_scale': float(guidance_scale)}
    if negative_prompt:
        kwargs['negative_prompt'] = negative_prompt
    elif float(guidance_scale) > 1:
        kwargs['negative_prompt'] = ' '
    return kwargs


def diffusion_transformer_class(model_type: _enums.ModelType):
    """
    Transformer class for a model type that uses one, or ``None``.
    """
    if _enums.model_type_is_sd3(model_type):
        return diffusers.SD3Transformer2DModel
    if _enums.model_type_is_flux(model_type):
        return diffusers.FluxTransformer2DModel
    if _enums.model_type_is_flux2_family(model_type):
        return diffusers.Flux2Transformer2DModel
    if _enums.model_type_is_z_image_family(model_type):
        return diffusers.ZImageTransformer2DModel
    if _enums.model_type_is_qwen_image_family(model_type):
        return diffusers.QwenImageTransformer2DModel
    if model_type == _enums.ModelType.QWEN_IMAGE_21:
        return diffusers.QwenImage21Transformer2DModel
    if model_type == _enums.ModelType.KREA2:
        return diffusers.Krea2Transformer2DModel
    if _enums.model_type_is_hydit(model_type):
        return diffusers.HunyuanDiT2DModel
    if _enums.model_type_is_hyimage_21_family(model_type):
        return diffusers.HunyuanImageTransformer2DModel
    if model_type in {
            _enums.ModelType.KANDINSKY5_IMAGE,
            _enums.ModelType.KANDINSKY5_VIDEO}:
        return diffusers.Kandinsky5Transformer3DModel
    return None


def _validate_qwen_image_controlnet_condition_channels(
        controlnet,
        pipeline_type: _enums.PipelineType,
        controlnet_uri: str):
    """
    Qwen-Image ControlNet inpaint packs VAE latents plus a downsampled mask
    (``extra_condition_channels=4`` after packing). Structural / union
    ControlNets expect packed latents only (``extra_condition_channels=0``).
    """
    if not hasattr(controlnet, 'config'):
        return

    extra = getattr(controlnet.config, 'extra_condition_channels', 0) or 0

    if pipeline_type == _enums.PipelineType.INPAINT:
        if extra == 0:
            raise UnsupportedPipelineConfigError(
                f'Qwen-Image ControlNet inpaint requires a ControlNet with '
                f'extra_condition_channels > 0 (for example '
                f'InstantX/Qwen-Image-ControlNet-Inpainting). '
                f'"{controlnet_uri}" has extra_condition_channels=0 and is for '
                f'text-to-image / structural control only '
                f'(for example InstantX/Qwen-Image-ControlNet-Union).'
            )
    elif extra != 0:
        raise UnsupportedPipelineConfigError(
            f'Qwen-Image ControlNet text-to-image requires a ControlNet with '
            f'extra_condition_channels=0 (for example '
            f'InstantX/Qwen-Image-ControlNet-Union). '
            f'"{controlnet_uri}" has extra_condition_channels={extra}; use '
            f'InstantX/Qwen-Image-ControlNet-Inpainting with '
            f'--image-seeds "image.png;mask=mask.png" for inpaint.'
        )


def _flow_image_pipeline_class(
        model_type: _enums.ModelType,
        pipeline_type: _enums.PipelineType,
        model_class_name: str | None,
        controlnet_uris: _types.OptionalUris,
        t2i_adapter_uris: _types.OptionalUris,
        ip_adapter_uris: _types.OptionalUris,
        unet_uri: _types.OptionalUri,
        help_mode: bool
):
    """
    Pipeline class for Flux.2, Z-Image, and Qwen-Image, or ``None`` for other types.

    ``model_class_name`` is ``model_index.json`` ``_class_name``. It selects
    Flux.2 Klein versus full Flux.2. Klein stays ``--model-type flux2``.
    """
    if not _enums.model_type_is_flow_image(model_type):
        return None

    model_type_string = _enums.get_model_type_string(model_type)

    if t2i_adapter_uris:
        raise UnsupportedPipelineConfigError(
            f'--model-type {model_type_string} does not support --t2i-adapters.')
    if ip_adapter_uris:
        raise UnsupportedPipelineConfigError(
            f'--model-type {model_type_string} does not support --ip-adapters.')
    if unet_uri:
        raise UnsupportedPipelineConfigError(
            f'--model-type {model_type_string} does not use a UNet. Use --transformer.')

    if model_type == _enums.ModelType.QWEN_IMAGE_21:
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                '--model-type qwen-image-21 does not support --control-nets.')
        if pipeline_type != _enums.PipelineType.TXT2IMG and not help_mode:
            raise UnsupportedPipelineConfigError(
                '--model-type qwen-image-21 accepts optional condition images, but has no '
                'img2img or inpaint pipeline.')
        return diffusers.QwenImage21Pipeline

    if model_type == _enums.ModelType.KREA2:
        if controlnet_uris:
            raise UnsupportedPipelineConfigError('--model-type krea2 does not support --control-nets.')
        if pipeline_type != _enums.PipelineType.TXT2IMG and not help_mode:
            raise UnsupportedPipelineConfigError('--model-type krea2 supports text-to-image only.')
        return diffusers.Krea2Pipeline

    if model_type == _enums.ModelType.HYIMAGE_21_REFINER:
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                '--model-type hyimage-21-refiner does not support --control-nets.')
        if pipeline_type != _enums.PipelineType.TXT2IMG and not help_mode:
            raise UnsupportedPipelineConfigError(
                '--model-type hyimage-21-refiner supports text-to-image with a required input image '
                'via --image-seeds (no img2img strength).')
        return diffusers.HunyuanImageRefinerPipeline

    if model_type == _enums.ModelType.HYIMAGE_21:
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                '--model-type hyimage-21 does not support --control-nets.')
        if pipeline_type != _enums.PipelineType.TXT2IMG and not help_mode:
            raise UnsupportedPipelineConfigError(
                '--model-type hyimage-21 supports text-to-image only.')
        return diffusers.HunyuanImagePipeline

    if model_type == _enums.ModelType.KANDINSKY5_IMAGE:
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                '--model-type kandinsky5-image does not support --control-nets.')
        if pipeline_type == _enums.PipelineType.INPAINT and not help_mode:
            raise UnsupportedPipelineConfigError(
                '--model-type kandinsky5-image has no inpaint pipeline; use it for text-to-image '
                'or image-to-image editing.')
        if pipeline_type == _enums.PipelineType.IMG2IMG:
            return diffusers.Kandinsky5I2IPipeline
        return diffusers.Kandinsky5T2IPipeline

    if _enums.model_type_is_flux2_klein_kv(model_type):
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                '--model-type flux2-klein-kv does not support --control-nets.')
        if pipeline_type != _enums.PipelineType.TXT2IMG and not help_mode:
            raise UnsupportedPipelineConfigError(
                '--model-type flux2-klein-kv has no img2img or inpaint pipeline. '
                '--image-seeds are reference images and take no mask or strength.')
        return diffusers.Flux2KleinKVPipeline

    if _enums.model_type_is_flux2(model_type):
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                '--model-type flux2 does not support --control-nets. '
                'This Diffusers version has no Flux.2 ControlNet pipeline.')
        klein = bool(model_class_name and model_class_name.startswith('Flux2Klein')
                     and not model_class_name.startswith('Flux2KleinKV'))
        if pipeline_type == _enums.PipelineType.IMG2IMG:
            raise UnsupportedPipelineConfigError(
                '--model-type flux2 has no img2img pipeline. '
                '--image-seeds without a mask are reference images. '
                'A mask is Flux.2 Klein inpaint.')
        if pipeline_type == _enums.PipelineType.INPAINT:
            if model_class_name and not klein and not help_mode:
                raise UnsupportedPipelineConfigError(
                    'Full Flux.2 does not support inpainting or --image-seeds. '
                    'Use a Flux.2 Klein checkpoint for inpaint.')
            return diffusers.Flux2KleinInpaintPipeline
        if klein:
            return diffusers.Flux2KleinPipeline
        return diffusers.Flux2Pipeline

    if _enums.model_type_is_z_image_omni(model_type):
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                '--model-type z-image-omni does not support --control-nets.')
        if pipeline_type != _enums.PipelineType.TXT2IMG and not help_mode:
            raise UnsupportedPipelineConfigError(
                '--model-type z-image-omni has no img2img or inpaint pipeline. '
                '--image-seeds are condition images and take no mask or strength.')
        return diffusers.ZImageOmniPipeline

    if _enums.model_type_is_z_image(model_type):
        if controlnet_uris and len(controlnet_uris) > 1:
            raise UnsupportedPipelineConfigError(
                'Z-Image accepts one ControlNet union model in --control-nets.')
        if controlnet_uris:
            if pipeline_type == _enums.PipelineType.IMG2IMG and not help_mode:
                raise UnsupportedPipelineConfigError(
                    'Z-Image ControlNet supports text-to-image and inpaint, not img2img.')
            if pipeline_type == _enums.PipelineType.INPAINT:
                return diffusers.ZImageControlNetInpaintPipeline
            return diffusers.ZImageControlNetPipeline
        if pipeline_type == _enums.PipelineType.IMG2IMG:
            return diffusers.ZImageImg2ImgPipeline
        if pipeline_type == _enums.PipelineType.INPAINT:
            return diffusers.ZImageInpaintPipeline
        return diffusers.ZImagePipeline

    if _enums.model_type_is_qwen_image_layered(model_type):
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                '--model-type qwen-image-layered does not support --control-nets.')
        if pipeline_type != _enums.PipelineType.TXT2IMG and not help_mode:
            raise UnsupportedPipelineConfigError(
                '--model-type qwen-image-layered takes one condition image and no mask.')
        return diffusers.QwenImageLayeredPipeline

    if _enums.model_type_is_qwen_image_edit(model_type):
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                '--model-type qwen-image-edit does not support --control-nets.')
        plus = bool(model_class_name and model_class_name.startswith('QwenImageEditPlus'))
        named_inpaint = bool(model_class_name and 'Inpaint' in model_class_name)
        if plus and pipeline_type == _enums.PipelineType.INPAINT and not help_mode:
            raise UnsupportedPipelineConfigError(
                'Qwen-Image edit-plus has no inpaint pipeline.')
        if pipeline_type == _enums.PipelineType.INPAINT or named_inpaint:
            return diffusers.QwenImageEditInpaintPipeline
        if plus:
            return diffusers.QwenImageEditPlusPipeline
        return diffusers.QwenImageEditPipeline

    if controlnet_uris:
        if pipeline_type == _enums.PipelineType.IMG2IMG and not help_mode:
            raise UnsupportedPipelineConfigError(
                'Qwen-Image ControlNet supports text-to-image and inpaint, not img2img.')
        if pipeline_type == _enums.PipelineType.INPAINT:
            return diffusers.QwenImageControlNetInpaintPipeline
        return diffusers.QwenImageControlNetPipeline
    if pipeline_type == _enums.PipelineType.IMG2IMG:
        return diffusers.QwenImageImg2ImgPipeline
    if pipeline_type == _enums.PipelineType.INPAINT:
        return diffusers.QwenImageInpaintPipeline
    return diffusers.QwenImagePipeline


_MODEL_INDEX_CHECKS = [
    (_enums.model_type_is_flux2_klein_kv, ('^Flux2Klein', 'Flux.2 Klein KV')),
    (_enums.model_type_is_flux2, ('^Flux2', 'Flux.2')),
    (_enums.model_type_is_z_image_omni, ('^ZImageOmni', 'Z-Image Omni')),
    (_enums.model_type_is_z_image, ('^ZImage', 'Z-Image')),
    (_enums.model_type_is_qwen_image_layered, ('^QwenImageLayered', 'Qwen-Image layered')),
    (_enums.model_type_is_qwen_image_edit, ('^QwenImageEdit', 'Qwen-Image edit')),
    (_enums.model_type_is_qwen_image, ('^QwenImage', 'Qwen-Image')),
    (_enums.model_type_is_qwen_image_21, ('^QwenImage21', 'Qwen-Image 2.1')),
    (_enums.model_type_is_krea2, ('^Krea2', 'Krea 2')),
    (_enums.model_type_is_hyimage_21_refiner, ('^HunyuanImageRefiner', 'HunyuanImage 2.1 Refiner')),
    (_enums.model_type_is_hyimage_21, ('^HunyuanImagePipeline', 'HunyuanImage 2.1')),
    (_enums.model_type_is_hydit, ('^HunyuanDiT', 'Hunyuan-DiT')),
    (_enums.model_type_is_kandinsky5_image, ('^Kandinsky5', 'Kandinsky 5.0 Image')),
    (_enums.model_type_is_flux, ('^Flux(?!2)', 'Flux')),
    (_enums.model_type_is_sd3, ('^StableDiffusion3.*', 'Stable Diffusion 3')),
    (_enums.model_type_is_sdxl, ('^StableDiffusionXL.*', 'Stable Diffusion XL')),
    (_enums.model_type_is_sd15, ('^StableDiffusion[^X3].*', 'Stable Diffusion')),
    (_enums.model_type_is_sd2, ('^StableDiffusion[^X3].*', 'Stable Diffusion')),
    (_enums.model_type_is_s_cascade, ('^StableCascade.*', 'Stable Cascade')),
    (_enums.model_type_is_kolors, ('^Kolors.*', 'Kolors')),
    (_enums.model_type_is_floyd, ('^IF.*', 'Deep Floyd')),
]

_MODEL_INDEX_FALLBACK_CHECKS = [
    (lambda model_type: model_type == _enums.ModelType.SD, '^LatentConsistency.*')
]

_REDIRECTED_FLOW_CLASSES = (
    ('Flux2KleinKV', _enums.model_type_is_flux2, 'flux2-klein-kv'),
    ('ZImageOmni', _enums.model_type_is_z_image, 'z-image-omni'),
    ('QwenImage21', _enums.model_type_is_qwen_image, 'qwen-image-21'),
    ('QwenImageLayered', _enums.model_type_is_qwen_image, 'qwen-image-layered'),
    ('QwenImageEdit', _enums.model_type_is_qwen_image, 'qwen-image-edit'),
    ('HunyuanImageRefiner', _enums.model_type_is_hyimage_21, 'hyimage-21-refiner'),
)


def validate_model_index_class(
        model_type: _enums.ModelType,
        model_class_name: str,
        model_path: str = 'model'
):
    """
    Raise :py:class:`UnsupportedPipelineConfigError` when ``_class_name`` does not
    belong to ``model_type``.

    Flux.2 Klein stays ``flux2``. The published Klein KV repo is indexed as
    ``Flux2KleinPipeline``, the same class name as Klein, and is loaded with
    ``Flux2KleinKVPipeline`` when ``--model-type`` is ``flux2-klein-kv``.
    Omni, edit, and layered match their own prefix, so those checkpoints are
    sent to their own ``--model-type``.
    """
    for prefix, parent_check, model_type_string in _REDIRECTED_FLOW_CLASSES:
        if model_class_name.startswith(prefix) and parent_check(model_type):
            raise UnsupportedPipelineConfigError(
                f'{model_path} uses {model_class_name}. '
                f'Use --model-type {model_type_string}.')

    for check_func, (pattern, title) in _MODEL_INDEX_CHECKS:
        if check_func(model_type) and re.match(pattern, model_class_name) is None:
            if not any(
                    check(model_type) and re.match(fallback, model_class_name) is not None
                    for check, fallback in _MODEL_INDEX_FALLBACK_CHECKS
            ):
                raise UnsupportedPipelineConfigError(
                    f'{model_path} is not a {title} model, '
                    f'incorrect --model-type value: {_enums.get_model_type_string(model_type)}'
                )


def get_pipeline_class(
        model_type: _enums.ModelType = _enums.ModelType.SD,
        pipeline_type: _enums.PipelineType = _enums.PipelineType.TXT2IMG,
        unet_uri: _types.OptionalUri = None,
        transformer_uri: _types.OptionalUri = None,
        model_class_name: str | None = None,
        vae_uri: _types.OptionalUri = None,
        lora_uris: _types.OptionalUris = None,
        image_encoder_uri: _types.OptionalUri = None,
        ip_adapter_uris: _types.OptionalUris = None,
        textual_inversion_uris: _types.OptionalUris = None,
        controlnet_uris: _types.OptionalUris = None,
        t2i_adapter_uris: _types.OptionalUris = None,
        pag: bool = False,
        help_mode: bool = False
) -> typing.Type[diffusers.DiffusionPipeline]:
    """
    Get an appropriate ``diffusers`` pipeline class for the provided arguments.

    :param model_type:  :py:class:`dgenerate.pipelinewrapper.ModelType` enum value
    :param pipeline_type: :py:class:`dgenerate.pipelinewrapper.PipelineType` enum value
    :param unet_uri: Optional ``--unet`` URI string for specifying a specific UNet
    :param transformer_uri: Optional ``--transformer`` URI string for specifying a specific Transformer.
        Supported for Stable Diffusion 3, Flux, Flux.2, Z-Image, Qwen-Image, and LTX.
    :param model_class_name: Optional ``model_index.json`` ``_class_name``. Flux.2 uses it to
        select Klein versus the full pipeline.
    :param vae_uri: Optional ``--vae`` URI string for specifying a specific VAE
    :param lora_uris: Optional ``--loras`` URI strings for specifying LoRA weights
    :param image_encoder_uri: Optional ``--image-encoder`` URI for use with IP Adapter weights or Stable Cascade
    :param ip_adapter_uris: Optional ``--ip-adapters`` URI strings for specifying IP Adapter weights
    :param textual_inversion_uris: Optional ``--textual-inversions`` URI strings for specifying Textual Inversion weights
    :param controlnet_uris: Optional ``--control-nets`` URI strings for specifying ControlNet models
    :param t2i_adapter_uris: Optional ``--t2i-adapters`` URI strings for specifying T2IAdapter models
    :param pag: Use perturbed attention guidance?
    :param help_mode: Return the class even if it does not support the selected ``pipeline_type``

    :raises UnsupportedPipelineConfigError:
    """

    # PAG check
    if pag:
        if not _enums.model_type_supports_pag(model_type):
            raise UnsupportedPipelineConfigError(
                'Perturbed attention guidance (--pag*) is only supported with '
                '--model-type sd, sdxl, kolors (txt2img), and sd3.')

        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                'Perturbed attention guidance (--pag*) is is not supported '
                'with --t2i-adapters.')

    # Flux model restrictions
    if _enums.model_type_is_flux(model_type):
        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                'Flux --model-type values are not compatible with --t2i-adapters.')
        if ip_adapter_uris and not image_encoder_uri:
            raise UnsupportedPipelineConfigError(
                'Must specify --image-encoder when using --ip-adapters with Flux.')
        if ip_adapter_uris and len(ip_adapter_uris) > 1:
            raise UnsupportedPipelineConfigError(
                'Flux --model-type values do not support multiple --ip-adapters.')

    if model_type == _enums.ModelType.FLUX_FILL:
        if pipeline_type != _enums.PipelineType.INPAINT:
            raise UnsupportedPipelineConfigError(
                'Flux fill --model-type value does not support anything but inpaint mode.'
            )

    # Deep Floyd model restrictions
    if _enums.model_type_is_floyd(model_type):
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                'Deep Floyd --model-type values are not compatible with --control-nets.')
        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                'Deep Floyd --model-type values are not compatible with --t2i-adapters.')
        if vae_uri:
            raise UnsupportedPipelineConfigError(
                'Deep Floyd --model-type values are not compatible with --vae.')
        if image_encoder_uri:
            raise UnsupportedPipelineConfigError(
                'Deep Floyd --model-type values are not compatible with --image-encoder.')

    # Stable Cascade model restrictions
    if _enums.model_type_is_s_cascade(model_type):
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                'Stable Cascade --model-type values are not compatible with --control-nets.')
        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                'Stable Cascade --model-type values are not compatible with --t2i-adapters.')
        if vae_uri:
            raise UnsupportedPipelineConfigError(
                'Stable Cascade --model-type values are not compatible with --vae.')

    # Torch SD3 restrictions
    if _enums.model_type_is_sd3(model_type):
        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                '--model-type sd3 is not compatible with --t2i-adapters.')
        if unet_uri:
            raise UnsupportedPipelineConfigError(
                '--model-type sd3 is not compatible with --unet.')

    # Torch Kolors restrictions
    if _enums.model_type_is_kolors(model_type):
        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                '--model-type kolors is not compatible with --t2i-adapters.')

    if _enums.model_type_is_hydit(model_type):
        if unet_uri:
            raise UnsupportedPipelineConfigError(
                '--model-type hydit uses a DiT transformer, not a UNet. Use --transformer instead.')
        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                '--model-type hydit is not compatible with --t2i-adapters.')
        if pag:
            raise UnsupportedPipelineConfigError(
                '--model-type hydit does not support perturbed attention guidance (--pag*).')

    if transformer_uri:
        if not _enums.model_type_is_sd3(model_type) \
                and not _enums.model_type_is_flux(model_type) \
                and not _enums.model_type_is_flow_image(model_type) \
                and not _enums.model_type_is_hydit(model_type) \
                and not _enums.model_type_is_video(model_type):
            raise UnsupportedPipelineConfigError(
                '--transformer is only supported for --model-type sd3, sd3-pix2pix, flux, flux-fill, '
                'flux-kontext, flux2, flux2-klein-kv, z-image, z-image-omni, qwen-image, qwen-image-edit, '
                'qwen-image-layered, qwen-image-21, krea2, kandinsky5-image, hydit, hyimage-21, '
                'hyimage-21-refiner, ltx2, ltx-video, wan, wan-animate, '
                'wan-animate-2, hyvideo-15, motif-video, and kandinsky5-video.')

    # Incompatible combinations
    if controlnet_uris and t2i_adapter_uris:
        raise UnsupportedPipelineConfigError(
            '--control-nets and --t2i-adapters cannot be used together.')

    if image_encoder_uri and not ip_adapter_uris and model_type != _enums.ModelType.S_CASCADE:
        raise UnsupportedPipelineConfigError(
            '--image-encoder cannot be specified without --ip-adapters if --model-type is not s-cascade.')

    # Pix2Pix model restrictions
    is_pix2pix = _enums.model_type_is_pix2pix(model_type)

    if is_pix2pix:
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                'Pix2Pix --model-type values are not compatible with --control-nets.')
        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                'Pix2Pix --model-type values are not compatible with --t2i-adapters.')
        if image_encoder_uri and model_type != _enums.ModelType.PIX2PIX:
            raise UnsupportedPipelineConfigError(
                'Only Pix2Pix --model-type pix2pix is compatible '
                'with --image-encoder. Pix2Pix SDXL is not supported.')

    is_sdxl = _enums.model_type_is_sdxl(model_type)
    is_sd3 = _enums.model_type_is_sd3(model_type)

    sdxl_controlnet_union = False
    parsed_control_net_uris = None

    try:
        if controlnet_uris and is_sdxl:
            parsed_control_net_uris = [_uris.ControlNetUri.parse(s, model_type) for s in controlnet_uris]
            sdxl_controlnet_union = controlnet_uris and is_sdxl and any(
                s.mode is not None for s in parsed_control_net_uris)
    except _uris.InvalidControlNetUriError as e:
        raise UnsupportedPipelineConfigError(str(e)) from e

    def eq_cn_uri(
            uri1: _uris.ControlNetUri,
            uri2: _uris.ControlNetUri):
        equals = True

        for name, val in _types.get_public_attributes(uri1).items():
            if name not in {'scale', 'mode', 'start', 'end'}:
                equals = (equals and val == getattr(uri2, name))

        return equals

    if sdxl_controlnet_union and \
            any(not eq_cn_uri(parsed_control_net_uris[0], u)
                for u in parsed_control_net_uris):
        raise UnsupportedPipelineConfigError(
            'SDXL ControlNet Union mode requires all ControlNet '
            'model URIs to be identical with the exception of the '
            '"scale", "mode", "start", and "end" arguments.'
        )

    # Pipeline class selection
    flow_pipeline_class = _flow_image_pipeline_class(
        model_type=model_type,
        pipeline_type=pipeline_type,
        model_class_name=model_class_name,
        controlnet_uris=controlnet_uris,
        t2i_adapter_uris=t2i_adapter_uris,
        ip_adapter_uris=ip_adapter_uris,
        unet_uri=unet_uri,
        help_mode=help_mode
    )
    if flow_pipeline_class is not None:
        pipeline_class = flow_pipeline_class
    elif _enums.model_type_is_upscaler(model_type):
        if controlnet_uris:
            raise UnsupportedPipelineConfigError(
                'Upscaler models are not compatible with --control-nets.')
        if t2i_adapter_uris:
            raise UnsupportedPipelineConfigError(
                'Upscaler models are not compatible with --t2i-adapters.')
        if image_encoder_uri:
            raise UnsupportedPipelineConfigError(
                'Upscaler models are not compatible with --image-encoder.')
        if pipeline_type != _enums.PipelineType.IMG2IMG and not help_mode:
            raise UnsupportedPipelineConfigError(
                'Upscaler models only work with img2img generation, IE: --image-seeds (with no image masks).')

        pipeline_class = (
            diffusers.StableDiffusionUpscalePipeline
            if model_type == _enums.ModelType.UPSCALER_X4
            else diffusers.StableDiffusionLatentUpscalePipeline
        )
    else:
        if pipeline_type == _enums.PipelineType.TXT2IMG:
            if is_pix2pix:
                if not help_mode:
                    raise UnsupportedPipelineConfigError(
                        'Pix2Pix models only work in img2img / inpaint mode and cannot work without --image-seeds.')
                else:
                    if is_sdxl:
                        # noinspection PyUnusedLocal
                        pipeline_class = diffusers.StableDiffusionXLInstructPix2PixPipeline
                    elif is_sd3:
                        # noinspection PyUnusedLocal
                        pipeline_class = dgenerate.extras.ultraedit.StableDiffusion3InstructPix2PixPipeline
                    else:
                        # noinspection PyUnusedLocal
                        pipeline_class = diffusers.StableDiffusionInstructPix2PixPipeline

            if model_type == _enums.ModelType.FLUX_KONTEXT:
                raise UnsupportedPipelineConfigError(
                    'Flux Kontext models only work in img2img / inpaint mode and cannot work without --image-seeds.'
                )

            if model_type == _enums.ModelType.FLUX_FILL:
                raise UnsupportedPipelineConfigError(
                    'Flux Fill models only work in inpaint mode and cannot work without --image-seeds.'
                )

            if model_type == _enums.ModelType.IF:
                pipeline_class = diffusers.IFPipeline
            elif model_type == _enums.ModelType.IFS:
                if not help_mode:
                    raise UnsupportedPipelineConfigError(
                        'Deep Floyd IF super-resolution (IFS) only works in '
                        'img2img mode and cannot work without --image-seeds.')
                else:
                    pipeline_class = diffusers.IFSuperResolutionPipeline
            elif model_type == _enums.ModelType.S_CASCADE:
                pipeline_class = diffusers.StableCascadePriorPipeline
            elif model_type == _enums.ModelType.S_CASCADE_DECODER:
                pipeline_class = diffusers.StableCascadeDecoderPipeline
            elif model_type == _enums.ModelType.FLUX:
                if controlnet_uris:
                    pipeline_class = diffusers.FluxControlNetPipeline
                else:
                    pipeline_class = diffusers.FluxPipeline
            elif model_type == _enums.ModelType.SD3:
                if pag:
                    pipeline_class = diffusers.StableDiffusion3PAGPipeline
                elif controlnet_uris:
                    if pag:
                        raise UnsupportedPipelineConfigError(
                            'Stable Diffusion 3 does not support --pag with controlnets.')

                    pipeline_class = diffusers.StableDiffusion3ControlNetPipeline
                else:
                    pipeline_class = diffusers.StableDiffusion3Pipeline
            elif _enums.model_type_is_hydit(model_type):
                if controlnet_uris:
                    pipeline_class = _hydit.HunyuanDiTControlNetPipeline
                else:
                    pipeline_class = _hydit.HunyuanDiTPipeline
            elif model_type == _enums.ModelType.KOLORS:
                if controlnet_uris:
                    if pag:
                        raise UnsupportedPipelineConfigError(
                            'Kolors ControlNet mode does not support PAG')
                    else:
                        pipeline_class = _kolors.KolorsControlNetPipeline
                else:
                    if pag:
                        pipeline_class = diffusers.KolorsPAGPipeline
                    else:
                        pipeline_class = diffusers.KolorsPipeline
            elif t2i_adapter_uris:
                # The custom type is a hack to support from_single_file for SD1.5 - 2
                # models with the associated pipeline class which does not inherit
                # the correct mixin to do so but can use the mixin just fine
                pipeline_class = (
                    diffusers.StableDiffusionXLAdapterPipeline
                    if is_sdxl
                    else diffusers.StableDiffusionAdapterPipeline
                )
            elif controlnet_uris:
                if is_sdxl:
                    if pag:
                        if sdxl_controlnet_union:
                            raise UnsupportedPipelineConfigError(
                                'SDXL ControlNet Union mode does not support PAG')
                        pipeline_class = diffusers.StableDiffusionXLControlNetPAGPipeline
                    else:
                        if sdxl_controlnet_union:
                            pipeline_class = \
                                diffusers.StableDiffusionXLControlNetUnionPipeline
                        else:
                            pipeline_class = diffusers.StableDiffusionXLControlNetPipeline
                else:
                    if pag:
                        pipeline_class = diffusers.StableDiffusionControlNetPAGPipeline
                    else:
                        pipeline_class = diffusers.StableDiffusionControlNetPipeline
            else:
                if is_sdxl:
                    if pag:
                        pipeline_class = diffusers.StableDiffusionXLPAGPipeline
                    else:
                        pipeline_class = diffusers.StableDiffusionXLPipeline
                else:
                    if pag:
                        pipeline_class = diffusers.StableDiffusionPAGPipeline
                    else:
                        pipeline_class = diffusers.StableDiffusionPipeline

        elif pipeline_type == _enums.PipelineType.IMG2IMG:
            if _enums.model_type_is_hydit(model_type) and not help_mode:
                raise UnsupportedPipelineConfigError(
                    '--model-type hydit does not support img2img in this diffusers version.')
            if _enums.model_type_is_hyimage_21_family(model_type) and not help_mode:
                raise UnsupportedPipelineConfigError(
                    f'--model-type {_enums.get_model_type_string(model_type)} does not support img2img.')

            if controlnet_uris:
                if is_pix2pix:
                    raise UnsupportedPipelineConfigError(
                        'Pix2Pix models are not compatible with --control-nets.')

            if model_type == _enums.ModelType.FLUX_FILL:
                raise UnsupportedPipelineConfigError(
                    'Flux Fill models only work in inpaint mode.'
                )

            if is_pix2pix:
                if is_sdxl:
                    # noinspection PyUnusedLocal
                    pipeline_class = diffusers.StableDiffusionXLInstructPix2PixPipeline
                elif is_sd3:
                    # noinspection PyUnusedLocal
                    pipeline_class = dgenerate.extras.ultraedit.StableDiffusion3InstructPix2PixPipeline
                else:
                    # noinspection PyUnusedLocal
                    pipeline_class = diffusers.StableDiffusionInstructPix2PixPipeline
            elif model_type == _enums.ModelType.IF:
                pipeline_class = diffusers.IFImg2ImgPipeline
            elif model_type == _enums.ModelType.IFS:
                pipeline_class = diffusers.IFSuperResolutionPipeline
            elif model_type == _enums.ModelType.IFS_IMG2IMG:
                pipeline_class = diffusers.IFImg2ImgSuperResolutionPipeline
            elif model_type == _enums.ModelType.S_CASCADE:
                pipeline_class = diffusers.StableCascadePriorPipeline
            elif model_type == _enums.ModelType.S_CASCADE_DECODER:
                raise UnsupportedPipelineConfigError(
                    'Stable Cascade decoder models do not support img2img.')
            elif model_type == _enums.ModelType.FLUX:
                if controlnet_uris:
                    pipeline_class = diffusers.FluxControlNetImg2ImgPipeline
                else:
                    pipeline_class = diffusers.FluxImg2ImgPipeline
            elif model_type == _enums.ModelType.FLUX_KONTEXT:
                if controlnet_uris:
                    raise UnsupportedPipelineConfigError(
                        '--model-type flux-kontext does not support ControlNet models.'
                    )
                pipeline_class = diffusers.FluxKontextPipeline
            elif model_type == _enums.ModelType.SD3:
                if controlnet_uris:
                    raise UnsupportedPipelineConfigError(
                        '--model-type sd3 does not support img2img mode with ControlNet models.')
                if pag:
                    pipeline_class = diffusers.StableDiffusion3PAGImg2ImgPipeline
                else:
                    pipeline_class = diffusers.StableDiffusion3Img2ImgPipeline

            elif model_type == _enums.ModelType.KOLORS:
                if controlnet_uris:
                    if pag:
                        raise UnsupportedPipelineConfigError(
                            'Kolors ControlNet does not support PAG in img2img mode'
                        )
                    pipeline_class = _kolors.KolorsControlNetImg2ImgPipeline
                else:
                    if pag:
                        raise UnsupportedPipelineConfigError(
                            'Kolors does not support PAG in img2img mode'
                        )
                    pipeline_class = diffusers.KolorsImg2ImgPipeline
            elif t2i_adapter_uris:
                raise UnsupportedPipelineConfigError(
                    'img2img mode is not supported with --t2i-adapters.')
            elif controlnet_uris:
                if is_sdxl:
                    if pag:
                        if sdxl_controlnet_union:
                            raise UnsupportedPipelineConfigError(
                                'SDXL ControlNet Union mode does not support PAG')
                        pipeline_class = diffusers.StableDiffusionXLControlNetPAGImg2ImgPipeline
                    else:
                        if sdxl_controlnet_union:
                            pipeline_class = \
                                diffusers.StableDiffusionXLControlNetUnionImg2ImgPipeline
                        else:
                            pipeline_class = diffusers.StableDiffusionXLControlNetImg2ImgPipeline
                else:
                    if pag:
                        raise UnsupportedPipelineConfigError(
                            '--model-type sd (Stable Diffusion 1.5 - 2.*) '
                            'does not support --pag in img2img mode with ControlNet models.')
                    else:
                        pipeline_class = diffusers.StableDiffusionControlNetImg2ImgPipeline
            else:
                if is_sdxl:
                    if pag:
                        pipeline_class = diffusers.StableDiffusionXLPAGImg2ImgPipeline
                    else:
                        pipeline_class = diffusers.StableDiffusionXLImg2ImgPipeline
                else:
                    if pag:
                        raise UnsupportedPipelineConfigError(
                            '--model-type sd (Stable Diffusion 1.5 - 2.*) '
                            'does not support --pag in img2img mode.')
                    else:
                        pipeline_class = diffusers.StableDiffusionImg2ImgPipeline

        elif pipeline_type == _enums.PipelineType.INPAINT:

            if _enums.model_type_is_s_cascade(model_type):
                raise UnsupportedPipelineConfigError(
                    'Stable Cascade model types do not support inpainting.')
            if _enums.model_type_is_upscaler(model_type):
                raise UnsupportedPipelineConfigError(
                    'Stable Diffusion upscaler model types do not support inpainting.')
            if _enums.model_type_is_hydit(model_type) and not help_mode:
                raise UnsupportedPipelineConfigError(
                    '--model-type hydit does not support inpainting.')
            if _enums.model_type_is_hyimage_21_family(model_type) and not help_mode:
                raise UnsupportedPipelineConfigError(
                    f'--model-type {_enums.get_model_type_string(model_type)} does not support inpainting.')

            if is_pix2pix:
                if is_sdxl:
                    # noinspection PyUnusedLocal
                    pipeline_class = dgenerate.extras.ultraedit.StableDiffusionXLInstructPix2PixPipeline
                elif is_sd3:
                    # noinspection PyUnusedLocal
                    pipeline_class = dgenerate.extras.ultraedit.StableDiffusion3InstructPix2PixPipeline
                else:
                    # noinspection PyUnusedLocal
                    pipeline_class = dgenerate.extras.ultraedit.StableDiffusionInstructPix2PixPipeline
            elif model_type == _enums.ModelType.FLUX:
                if controlnet_uris:
                    pipeline_class = diffusers.FluxControlNetInpaintPipeline
                else:
                    pipeline_class = diffusers.FluxInpaintPipeline
            elif model_type == _enums.ModelType.FLUX_FILL:
                if controlnet_uris:
                    raise UnsupportedPipelineConfigError(
                        '--model-type flux-fill does not support ControlNet models.')
                pipeline_class = diffusers.FluxFillPipeline
            elif model_type == _enums.ModelType.FLUX_KONTEXT:
                if controlnet_uris:
                    raise UnsupportedPipelineConfigError(
                        '--model-type flux-kontext does not support ControlNet models.')
                pipeline_class = diffusers.FluxKontextInpaintPipeline
            elif model_type == _enums.ModelType.IF:
                pipeline_class = diffusers.IFInpaintingPipeline
            elif model_type == _enums.ModelType.IFS:
                pipeline_class = diffusers.IFInpaintingSuperResolutionPipeline
            elif model_type == _enums.ModelType.SD3:
                if controlnet_uris:
                    return diffusers.StableDiffusion3ControlNetInpaintingPipeline
                if pag:
                    raise UnsupportedPipelineConfigError(
                        '--model-type sd3 does not support --pag in inpaint mode.'
                    )

                pipeline_class = diffusers.StableDiffusion3InpaintPipeline
            elif model_type == _enums.ModelType.KOLORS:
                if controlnet_uris:
                    if pag:
                        raise UnsupportedPipelineConfigError(
                            'Kolors ControlNet does not support PAG in inpaint mode'
                        )
                    pipeline_class = _kolors.KolorsControlNetInpaintPipeline
                else:
                    if pag:
                        raise UnsupportedPipelineConfigError(
                            'Kolors does not support PAG in inpaint mode'
                        )
                    pipeline_class = _kolors.KolorsInpaintPipeline
            elif t2i_adapter_uris:
                raise UnsupportedPipelineConfigError(
                    'inpaint mode is not supported with --t2i-adapters.')
            elif controlnet_uris:
                if is_sdxl:
                    if pag:
                        raise UnsupportedPipelineConfigError(
                            '--model-type sdxl does not support --pag '
                            'in inpaint mode with ControlNet models.'
                        )
                    else:
                        if sdxl_controlnet_union:
                            pipeline_class = \
                                diffusers.StableDiffusionXLControlNetUnionInpaintPipeline
                        else:
                            pipeline_class = diffusers.StableDiffusionXLControlNetInpaintPipeline
                else:
                    if pag:
                        pipeline_class = diffusers.StableDiffusionControlNetPAGInpaintPipeline
                    else:
                        pipeline_class = diffusers.StableDiffusionControlNetInpaintPipeline
            else:
                if is_sdxl:
                    if pag:
                        pipeline_class = diffusers.StableDiffusionXLPAGInpaintPipeline
                    else:
                        pipeline_class = diffusers.StableDiffusionXLInpaintPipeline
                else:
                    if pag:
                        raise UnsupportedPipelineConfigError(
                            '--model-type sd (Stable Diffusion 1.5 - 2.*) '
                            'does not support --pag in inpaint mode.')
                    else:
                        pipeline_class = diffusers.StableDiffusionInpaintPipeline
        else:
            # Should be impossible
            raise UnsupportedPipelineConfigError('Pipeline type not implemented.')

    if lora_uris and not pipeline_class_supports_lora(pipeline_class):
        raise UnsupportedPipelineConfigError(
            f'Given current arguments, '
            f'--model-type {_enums.get_model_type_string(model_type)} '
            f'(pipeline: {pipeline_class.__name__}) does not support LoRAs.')

    if textual_inversion_uris and not pipeline_class_supports_textual_inversion(pipeline_class):
        raise UnsupportedPipelineConfigError(
            f'Given current arguments, '
            f'--model-type {_enums.get_model_type_string(model_type)} '
            f'(pipeline: {pipeline_class.__name__}) does not support Textual Inversions.')

    if ip_adapter_uris and not pipeline_class_supports_ip_adapter(pipeline_class):
        raise UnsupportedPipelineConfigError(
            f'Given current arguments, '
            f'--model-type {_enums.get_model_type_string(model_type)} '
            f'(pipeline: {pipeline_class.__name__}) does not support IP Adapters.')

    return pipeline_class


# ``model_index.json`` entries that do not ship variant weight files.
# ``from_pretrained`` only needs ``variant`` for modules it still has to load.
_NON_WEIGHT_INDEX_CLASS_MARKERS = (
    'Scheduler',
    'Tokenizer',
    'ImageProcessor',
    'FeatureExtractor',
    'Processor',
)


def _repo_pipeline_variant(variant, model_index, creation_kwargs):
    """Drop ``variant`` once every weight module is already constructed.

    Single-file loads read ``model_index.json`` from the vendored config tree so
    a local safetensors file does not need the Hub. ``from_pretrained`` treats
    that file's directory as the repo snapshot. The tree has configs only, so a
    ``variant`` such as ``fp16`` makes diffusers reject it even though the UNet,
    VAE, and text encoder were already loaded from the real weight files.
    """
    if not variant or not isinstance(model_index, dict):
        return variant

    for name, value in model_index.items():
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        library, class_name = value[0], value[1]
        if library is None or class_name is None:
            continue
        if any(marker in str(class_name) for marker in _NON_WEIGHT_INDEX_CLASS_MARKERS):
            continue
        if name not in creation_kwargs:
            return variant

    _messages.debug_log(
        f'Not passing variant={variant!r} to from_pretrained; '
        'weight modules are already loaded and the config snapshot has no variant files.')
    return None


def _enforce_pipeline_cache_size(new_pipeline_size):
    _pipeline_cache.enforce_cpu_mem_constraints(
        _constants.PIPELINE_CACHE_MEMORY_CONSTRAINTS,
        size_var='pipeline_size',
        new_object_size=new_pipeline_size)


@_memoize(_pipeline_cache,
          exceptions={'local_files_only', 'missing_submodules_ok'},
          hasher=_pipeline_args_hasher,
          extra_identities=[lambda m: m.pipeline],
          on_hit=_pipeline_on_hit,
          on_create=_pipeline_on_create)
def _create_diffusion_pipeline(
        model_path: str,
        model_type: _enums.ModelType = _enums.ModelType.SD,
        pipeline_type: _enums.PipelineType = _enums.PipelineType.TXT2IMG,
        revision: _types.OptionalString = None,
        variant: _types.OptionalString = None,
        subfolder: _types.OptionalString = None,
        dtype: _enums.DataType = _enums.DataType.AUTO,
        unet_uri: _types.OptionalUri = None,
        transformer_uri: _types.OptionalUri = None,
        vae_uri: _types.OptionalUri = None,
        lora_uris: _types.OptionalUris = None,
        lora_fuse_scale: _types.OptionalFloat = None,
        image_encoder_uri: _types.OptionalUri = None,
        ip_adapter_uris: _types.OptionalUris = None,
        textual_inversion_uris: _types.OptionalUris = None,
        text_encoder_uris: _types.OptionalUris = None,
        controlnet_uris: _types.OptionalUris = None,
        t2i_adapter_uris: _types.OptionalUris = None,
        quantizer_uri: _types.OptionalUri = None,
        quantizer_map: _types.OptionalStrings = None,
        pag: bool = False,
        safety_checker: bool = False,
        original_config: _types.OptionalString = None,
        auth_token: _types.OptionalString = None,
        device: str = _torchutil.default_device(),
        extra_modules: dict[str, typing.Any] | None = None,
        model_cpu_offload: bool = False,
        sequential_cpu_offload: bool = False,
        model_group_offload: bool = False,
        torch_compile: bool = False,
        local_files_only: bool = False,
        missing_submodules_ok: bool = False
) -> PipelineCreationResult:
    # Ensure model path is specified
    if not model_path:
        raise ValueError('model_path must be specified.')

    # Offload checks
    _reject_mixed_offload(model_cpu_offload, sequential_cpu_offload, model_group_offload)
    submodules_on_pipeline = modules_live_on_pipeline(
        model_cpu_offload, sequential_cpu_offload, model_group_offload)

    # Device check
    if not _torchutil.is_valid_device_string(device):
        raise UnsupportedPipelineConfigError(
            f'Invalid device argument, {_torchutil.invalid_device_message(device, cap=False)}')

    # Quantizer map check
    quantizer_map_vals = [
        'unet',
        'transformer',
        'text_encoder',
        'text_encoder_2',
        'text_encoder_3',
        'controlnet',
        'connectors'
    ]

    if quantizer_map is not None:
        for map_value in quantizer_map:
            if map_value not in quantizer_map_vals:
                raise UnsupportedPipelineConfigError(
                    f'Unknown quantizer_map value: {map_value}, '
                    f'must be one of: {_textprocessing.oxford_comma(quantizer_map_vals, "or")}'
                )

    if not _hfhub.is_single_file_model_load(model_path) and original_config:
        raise UnsupportedPipelineConfigError(
            'Loading original config .yaml file is not supported '
            'when loading from a Hugging Face repo.'
        )

    if quantizer_uri and quantizer_uri.split(';')[0].strip() in {'bnb', 'bitsandbytes'}:
        if dtype is _enums.DataType.AUTO:
            # Default to all modules to float32 if no dtype is specified when using bitsandbytes
            dtype = _enums.DataType.FLOAT32

    if original_config:
        # only instance where it really makes sense to raise config not found
        # everything else is indicative of a missing model
        with _hfhub.with_hf_errors_as_config_not_found():
            original_config = _hfhub.download_non_hf_slug_config(original_config, local_files_only=local_files_only, auth_token=auth_token)

    model_path = _hfhub.download_non_hf_slug_model(model_path, local_files_only=local_files_only, auth_token=auth_token)

    if dtype is _enums.DataType.AUTO:
        detected_dtype = _util.auto_dtype(
            model_path,
            revision=revision,
            subfolder=subfolder,
            auth_token=auth_token,
            local_files_only=local_files_only)
        if detected_dtype is not None:
            _messages.debug_log(f'--dtype auto selected: {_enums.get_data_type_string(detected_dtype)}')
            dtype = detected_dtype

    if (_hfhub.is_single_file_model_load(model_path)
        and _enums.model_type_is_kolors(model_type)
    ):
        raise UnsupportedPipelineConfigError(
            'Kolors models cannot be loaded from a single file.'
        )

    model_index = _util.fetch_model_index_dict(
        model_path,
        subfolder=subfolder,
        revision=revision,
        auth_token=auth_token,
        local_files_only=local_files_only
    )

    model_class_name = model_index.get('_class_name') if isinstance(model_index, dict) else None
    if model_class_name:
        validate_model_index_class(model_type, model_class_name, model_path)

    single_file_pipeline_repo = None
    if _hfhub.is_single_file_model_load(model_path):
        single_file_pipeline_repo = _singlefilecompat.single_file_pretrained_repo_for_model_path(
            model_path,
            auth_token=auth_token,
            local_files_only=local_files_only,
        )

    pipeline_class = get_pipeline_class(
        model_type=model_type,
        pipeline_type=pipeline_type,
        unet_uri=unet_uri,
        transformer_uri=transformer_uri,
        model_class_name=model_class_name,
        vae_uri=vae_uri,
        lora_uris=lora_uris,
        image_encoder_uri=image_encoder_uri,
        ip_adapter_uris=ip_adapter_uris,
        textual_inversion_uris=textual_inversion_uris,
        controlnet_uris=controlnet_uris,
        t2i_adapter_uris=t2i_adapter_uris,
        pag=pag
    )

    text_encoder_count = len(
        [a for a in inspect.getfullargspec(pipeline_class.__init__).args
         if _is_text_encoder_component(a)])

    if not text_encoder_uris:
        text_encoder_uris = []

    if len(text_encoder_uris) > text_encoder_count:
        raise UnsupportedPipelineConfigError('To many text encoder URIs specified.')

    if extra_modules is not None:
        _messages.debug_log('Checking extra_modules for meta tensors...')
        for module in extra_modules.items():
            if module[1] is None:
                continue
            _messages.debug_log(f'Checking extra module {module[0]} = {module[1].__class__}...')
            try:
                # Processors such as CLIPImageProcessor report device=None.
                # They are not torch modules and cannot hold meta tensors.
                # Do not assign to ``device``; that is the pipeline target.
                module_device = get_torch_device(module[1])
                if module_device is not None and module_device.type == 'meta':
                    _messages.debug_log(f'"{module[0]}" has meta tensors.')
                    _disable_to(
                        module[1],
                        module[0] == 'vae'
                    )
            except ValueError:
                _messages.debug_log(
                    f'Unable to get device of {module[0]} = {module[1].__class__}')
        extra_modules = extra_modules.copy()
    else:
        extra_modules = dict()

    unet_override = 'unet' in extra_modules
    vae_override = 'vae' in extra_modules
    controlnet_override = 'controlnet' in extra_modules
    adapter_override = 'adapter' in extra_modules
    image_encoder_override = 'image_encoder' in extra_modules
    safety_checker_override = 'safety_checker' in extra_modules
    transformer_override = 'transformer' in extra_modules

    if 'text_encoder' in extra_modules and text_encoder_count == 0:
        raise UnsupportedPipelineConfigError('To many text encoders specified.')

    if 'text_encoder_2' in extra_modules and text_encoder_count < 2:
        raise UnsupportedPipelineConfigError('To many text encoders specified.')

    if 'text_encoder_3' in extra_modules and text_encoder_count < 3:
        raise UnsupportedPipelineConfigError('To many text encoders specified.')

    # noinspection PyTypeChecker
    text_encoders: list[str] = list(text_encoder_uris)

    if len(text_encoders) > 0 and _text_encoder_null(text_encoders[0]):
        extra_modules['text_encoder'] = None
    if len(text_encoders) > 1 and _text_encoder_null(text_encoders[1]):
        extra_modules['text_encoder_2'] = None
    if len(text_encoders) > 2 and _text_encoder_null(text_encoders[2]):
        extra_modules['text_encoder_3'] = None

    text_encoder_override = 'text_encoder' in extra_modules
    text_encoder_2_override = 'text_encoder_2' in extra_modules
    text_encoder_3_override = 'text_encoder_3' in extra_modules

    if len(text_encoders) > 0 and text_encoder_override:
        text_encoders[0] = None
    if len(text_encoders) > 1 and text_encoder_2_override:
        text_encoders[1] = None
    if len(text_encoders) > 2 and text_encoder_3_override:
        text_encoders[2] = None

    # If we are not auto caching UNet/Transformer, VAE, and text encoders
    # into other caches, we should estimate a cache size for the pipeline
    # that includes them

    pipeline_cached_with_submodules = \
        modules_live_on_pipeline(
            model_cpu_offload, sequential_cpu_offload, model_group_offload) or quantizer_uri

    estimated_memory_usage = estimate_pipeline_cache_footprint(
        model_type=model_type,
        model_path=model_path,
        revision=revision,
        variant=variant,
        subfolder=subfolder,
        include_unet_or_transformer=pipeline_cached_with_submodules or bool(lora_uris),
        include_vae=pipeline_cached_with_submodules,
        include_text_encoders=pipeline_cached_with_submodules or bool(lora_uris),
        lora_uris=lora_uris,
        image_encoder_uri=image_encoder_uri,
        ip_adapter_uris=ip_adapter_uris,
        textual_inversion_uris=textual_inversion_uris,
        safety_checker=safety_checker and not safety_checker_override,
        auth_token=auth_token,
        extra_args=extra_modules,
        local_files_only=local_files_only,
        include_directories=extra_pipeline_weight_directories(
            model_index if isinstance(model_index, dict) else None),
        component_repo=single_file_pipeline_repo,
    )

    if controlnet_uris and controlnet_is_owned_by_pipeline(
            model_type, model_cpu_offload, sequential_cpu_offload, model_group_offload):
        for controlnet_uri in controlnet_uris:
            parsed_controlnet = _uris.ControlNetUri.parse(controlnet_uri, model_type=model_type)
            estimated_memory_usage += _util.estimate_model_memory_use(
                repo_id=_hfhub.download_non_hf_slug_model(parsed_controlnet.model, local_files_only=local_files_only, auth_token=auth_token),
                revision=parsed_controlnet.revision,
                variant=parsed_controlnet.variant,
                subfolder=parsed_controlnet.subfolder,
                auth_token=auth_token,
                local_files_only=local_files_only
            )

    _messages.debug_log(
        f'Creating Torch Pipeline: "{pipeline_class.__name__}", '
        f'Estimated CPU Side Memory Use: {_memory.bytes_best_human_unit(estimated_memory_usage)}')

    # Helper function to determine if quantization should be applied to a module
    def should_apply_quantizer(module_name):
        if not quantizer_uri:
            return False
        if quantizer_map is None:
            # Match the create_diffusion_pipeline docstring: ControlNet stays
            # full precision unless the user puts it in --quantizer-map.
            return module_name != 'controlnet'
        return module_name in quantizer_map

    # Helper function to determine device_map - always use the selected device currently
    def get_device_map_for_quantizer(quantizer_uri):
        return device if quantizer_uri else None

    # we need to manually emulate bitsandbytes 'compute_dtype'
    # casting for sdnq as it has trouble dequanting to anything
    # but float16 on forward which messes with diffusers
    # in various ways when a model is loaded in float32
    sdnq_cast_hack = False

    if quantizer_uri and (
        _uris.get_quantizer_uri_class(quantizer_uri) is
        _uris.SDNQQuantizerUri
    ):
        sdnq_cast_hack = True

    def _reject_repeated_sdnq(component_path, component_subfolder, component_revision, component_quantizer):
        config = _sdnqload.sdnq_config_for_component(
            component_path,
            subfolder=component_subfolder,
            revision=component_revision,
            token=auth_token,
            local_files_only=local_files_only)
        if config is None:
            return False
        error = _sdnqload.sdnq_requantize_error(component_quantizer or quantizer_uri, config)
        if error:
            raise UnsupportedPipelineConfigError(error)
        return True

    if _enums.model_type_is_flow_image(model_type):
        # The text encoder has to be registered before from_pretrained.
        # The cast hack follows the transformer, which is the published SDNQ Turbo weight.
        if _reject_repeated_sdnq(model_path, 'text_encoder', revision, None):
            pass
        if _reject_repeated_sdnq(model_path, 'transformer', revision, None):
            sdnq_cast_hack = True
        if _reject_repeated_sdnq(model_path, None, revision, None):
            sdnq_cast_hack = True

    if transformer_uri:
        parsed_for_sdnq = _uris.TransformerUri.parse(transformer_uri)
        if _reject_repeated_sdnq(
                parsed_for_sdnq.model,
                parsed_for_sdnq.subfolder,
                parsed_for_sdnq.revision,
                parsed_for_sdnq.quantizer):
            sdnq_cast_hack = True

    uri_quant_check = []
    manual_quantizer_components = set()

    # Check text encoder URIs
    for idx, encoder_uri in enumerate(text_encoder_uris):
        if encoder_uri and encoder_uri.lower() not in {'+', 'help', 'null'}:
            parsed_uri = _uris.TextEncoderUri.parse(encoder_uri)
            uri_quant_check.append(parsed_uri)
            if parsed_uri.quantizer:
                encoder_name = f'text_encoder{"_2" if idx == 1 else "_3" if idx == 2 else ""}'
                manual_quantizer_components.add(encoder_name)
                quantizer_class = _uris.get_quantizer_uri_class(parsed_uri.quantizer)
                if quantizer_class is _uris.SDNQQuantizerUri:
                    sdnq_cast_hack = True


    # Check transformer URI
    if transformer_uri:
        parsed_uri = _uris.TransformerUri.parse(transformer_uri)
        uri_quant_check.append(parsed_uri)
        if parsed_uri.quantizer:
            manual_quantizer_components.add('transformer')
            quantizer_class = _uris.get_quantizer_uri_class(parsed_uri.quantizer)
            if quantizer_class is _uris.SDNQQuantizerUri:
                sdnq_cast_hack = True

    # Check unet URI
    if unet_uri:
        parsed_uri = _uris.UNetUri.parse(unet_uri)
        uri_quant_check.append(parsed_uri)
        if parsed_uri.quantizer:
            manual_quantizer_components.add('unet')
            quantizer_class = _uris.get_quantizer_uri_class(parsed_uri.quantizer)
            if quantizer_class is _uris.SDNQQuantizerUri:
                sdnq_cast_hack = True

    # Check controlnet URIs
    if controlnet_uris:
        for controlnet_uri in controlnet_uris:
            parsed_uri = _uris.ControlNetUri.parse(controlnet_uri, model_type=model_type)
            uri_quant_check.append(parsed_uri)
            if parsed_uri.quantizer:
                manual_quantizer_components.add('controlnet')
                quantizer_class = _uris.get_quantizer_uri_class(parsed_uri.quantizer)
                if quantizer_class is _uris.SDNQQuantizerUri:
                    sdnq_cast_hack = True

    if quantizer_uri or any(p.quantizer for p in uri_quant_check):
        # for now, just knock out anything cached on the gpu, such as the last pipeline
        # the quantized pipeline modules are likely going to go straight onto the GPU
        # immediately, and they are guaranteed to be of non-trivial size
        _devicecache.clear_device_cache(device)

    # Granular component caching for quantization scenarios
    cached_component_paths = {}
    manual_quantizers = {}
    components_to_cache = []

    # Determine if we need granular caching
    needs_granular_caching = (quantizer_uri and (_hfhub.is_single_file_model_load(model_path) or lora_uris)) or \
                             (manual_quantizer_components and lora_uris)

    if needs_granular_caching:

        # Determine which components to cache
        if lora_uris:
            # For LoRA + quantizer: cache only components affected by LoRA
            components_to_cache = _get_affected_components_for_lora(pipeline_class)

            # Also include any manual components with quantizers that are LoRA-affected
            lora_affected_components = set(components_to_cache)
            for component in manual_quantizer_components:
                if component in lora_affected_components:
                    if component not in components_to_cache:
                        components_to_cache.append(component)
        else:
            # For single file + quantizer: cache all quantizable components
            components_to_cache = _get_quantizable_components(model_type)

        # Filter based on quantizer_map ONLY for components that would get auto-quantized
        # Components with manual quantizers should always be cached when LoRAs are involved
        if quantizer_map is not None:
            # Keep components that either:
            # 1. Have manual quantizers (always cache these when LoRAs are involved)
            # 2. Are in the quantizer_map (for auto-quantization)
            if lora_uris:
                # When LoRAs are involved, always cache components with manual quantizers
                components_to_cache = [c for c in components_to_cache
                                       if c in manual_quantizer_components or c in quantizer_map]
            else:
                # For single file without LoRAs, only respect quantizer_map
                components_to_cache = [c for c in components_to_cache if c in quantizer_map]

        # Collect manual URIs for components that will be cached
        manual_component_uris = {}
        if unet_uri and 'unet' in components_to_cache:
            manual_component_uris['unet'] = unet_uri
        if transformer_uri and 'transformer' in components_to_cache:
            manual_component_uris['transformer'] = transformer_uri

        # Handle text encoder URIs
        if text_encoder_uris:
            for idx, encoder_uri in enumerate(text_encoder_uris):
                if not _text_encoder_default(encoder_uri):
                    encoder_name = f'text_encoder{"_2" if idx == 1 else "_3" if idx == 2 else ""}'
                    if encoder_name in components_to_cache:
                        manual_component_uris[encoder_name] = encoder_uri

        # Perform granular caching
        if components_to_cache:
            cached_component_paths, manual_quantizers = _cache_components_granular(
                model_path=model_path,
                model_type=model_type,
                pipeline_type=pipeline_type,
                components_to_cache=components_to_cache,
                lora_uris=lora_uris,
                lora_fuse_scale=lora_fuse_scale,
                revision=revision,
                variant=variant,
                subfolder=subfolder,
                dtype=dtype,
                original_config=original_config,
                auth_token=auth_token,
                manual_component_uris=manual_component_uris,
                vae_uri=vae_uri
            )

            _messages.debug_log(
                f"Granular caching complete. Cached components: {list(cached_component_paths.keys())}")
            if manual_component_uris:
                _messages.debug_log(
                    f"Manual component URIs processed: {list(manual_component_uris.keys())}")
            if manual_quantizers:
                _messages.debug_log(
                    f"Manual quantizers preserved: {list(manual_quantizers.keys())}")
        else:
            manual_quantizers = {}

    # ControlNet and VAE loading

    # Used during pipeline load
    creation_kwargs = {}

    torch_dtype = _enums.get_torch_dtype(dtype)

    parsed_controlnet_uris = []
    parsed_t2i_adapter_uris = []
    parsed_image_encoder_uri = None
    parsed_unet_uri = None
    parsed_vae_uri = None
    parsed_transformer_uri = None

    pipe_params = inspect.signature(pipeline_class.__init__).parameters

    def load_text_encoder(uri: _uris.TextEncoderUri):
        return uri.load(
            variant_fallback=variant,
            dtype_fallback=dtype,
            original_config=original_config,
            auth_token=auth_token,
            local_files_only=local_files_only,
            no_cache=bool(lora_uris) or submodules_on_pipeline,
            missing_ok=missing_submodules_ok,
            device_map=get_device_map_for_quantizer(uri.quantizer)
        )

    def load_vae(uri: _uris.VAEUri):
        vae_model = uri.load(
            dtype_fallback=dtype,
            original_config=original_config,
            auth_token=auth_token,
            local_files_only=local_files_only,
            no_cache=submodules_on_pipeline,
            missing_ok=missing_submodules_ok
        )
        if sdnq_cast_hack:
            _apply_sdnq_decode(vae_model)
        return vae_model

    def _apply_sdnq_decode(vae_model):
        if vae_model is None or getattr(vae_model, '_dgenerate_sdnq_decode', False):
            return
        og_decode = vae_model.decode

        def sdnq_decode(latents, *args, **kwargs):
            if getattr(vae_model.config, 'use_post_quant_conv', False):
                cur_dtype = vae_model.post_quant_conv.weight.dtype
            else:
                cur_dtype = _enums.get_torch_dtype(dtype)
            return og_decode(latents.to(
                dtype=vae_model.dtype if cur_dtype is None else cur_dtype), *args, **kwargs)

        vae_model.decode = sdnq_decode
        vae_model._dgenerate_sdnq_decode = True

    def sdnq_forward(og_forward, model, *args, **kwargs):
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                args[i] = arg.to(dtype=model.dtype)
        for k,v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.to(dtype=model.dtype)
        return og_forward(*args, **kwargs)

    def controlnet_quant_forward(og_forward, model, *args, **kwargs):
        """
        Forward function for quantized controlnets that casts inputs to the model's dtype.
        This is needed because diffusers doesn't handle controlnet quantization state internally.
        
        Note: Preserves specific parameters that are used as indices in embedding layers.
        Specifically, 'controlnet_mode' parameter in FLUX ControlNets is used as indices
        in the controlnet_mode_embedder embedding layer and must remain as integer tensors.
        """
        # Known parameters that should remain as integer types for embedding layer indices
        # controlnet_mode: Used in FLUX ControlNet Union models with nn.Embedding layer
        # Note: SDXL ControlNet Union uses control_type with Timesteps projection (no integer requirement)
        embedding_index_params = {'controlnet_mode'}
        
        # Cast positional arguments to model dtype (no special handling needed)
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                args[i] = arg.to(dtype=model.dtype)
        
        # Handle keyword arguments with special cases for embedding indices
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                # Preserve specific embedding index parameters or any integer tensor
                if (k in embedding_index_params or 
                    v.dtype in (torch.int64, torch.long, torch.int32, torch.int)):
                    continue  # Don't cast embedding index tensors
                kwargs[k] = v.to(dtype=model.dtype)
        return og_forward(*args, **kwargs)

    def load_unet(uri: _uris.UNetUri, unet_class):
        unet_model = uri.load(
            variant_fallback=variant,
            dtype_fallback=dtype,
            original_config=original_config,
            auth_token=auth_token,
            local_files_only=local_files_only,
            no_cache=bool(lora_uris) or
                     bool(ip_adapter_uris) or
                     submodules_on_pipeline,
            device_map=get_device_map_for_quantizer(uri.quantizer),
            unet_class=unet_class
        )

        if sdnq_cast_hack:
            unet_model.forward = functools.partial(
                sdnq_forward,
                unet_model.forward,
                unet_model
            )
        return unet_model

    def load_transformer(uri: _uris.TransformerUri, transformer_class):
        # External single-file transformer checkpoints (notably GGUF) need the
        # pipeline's transformer config to construct the module. Video
        # pipelines pass this explicitly; keep static pipelines equivalent.
        transformer_config = None if _hfhub.is_single_file_model_load(model_path) else model_path
        transformer_model = uri.load(
            variant_fallback=variant,
            dtype_fallback=dtype,
            original_config=original_config,
            auth_token=auth_token,
            local_files_only=local_files_only,
            no_cache=bool(lora_uris) or submodules_on_pipeline,
            device_map=get_device_map_for_quantizer(uri.quantizer),
            config=transformer_config,
            config_subfolder='transformer' if transformer_config else None,
            transformer_class=transformer_class
        )
        if sdnq_cast_hack or _sdnqload.module_is_sdnq(transformer_model):
            if not getattr(transformer_model, '_dgenerate_sdnq_forward', False):
                transformer_model.forward = functools.partial(
                    sdnq_forward,
                    transformer_model.forward,
                    transformer_model
                )
                transformer_model._dgenerate_sdnq_forward = True
            _apply_sdnq_decode(creation_kwargs.get('vae'))
        return transformer_model

    def load_default_text_encoder(encoder, encoder_name):
        should_quantize = should_apply_quantizer(encoder_name)

        # Check if we have a cached version from granular caching
        if encoder_name in cached_component_paths:
            # Use manual quantizer if available, otherwise use global quantizer
            text_encoder_quantizer = manual_quantizers.get(encoder_name, quantizer_uri if should_quantize else None)

            return load_text_encoder(
                _uris.TextEncoderUri(
                    encoder=encoder,
                    model=cached_component_paths[encoder_name],
                    variant=variant,
                    dtype=dtype,
                    quantizer=text_encoder_quantizer
                )
            )
        else:
            if single_file_pipeline_repo:
                encoder_model = single_file_pipeline_repo
                encoder_load_subfolder = encoder_name
            else:
                encoder_model = model_path
                encoder_load_subfolder = encoder_subfolder
            if (_hfhub.is_single_file_model_load(encoder_model) and should_quantize):
                raise UnsupportedPipelineConfigError(
                    'Cannot use global --quantizer URI when attempting to '
                    f'load default text encoder for "{encoder_name}" from a single file checkpoint. '
                    f'You must specify this text encoder manually with --text-encoders, it is likely missing '
                    f'from the model checkpoint.'
                )
            return load_text_encoder(
                _uris.TextEncoderUri(
                    encoder=encoder,
                    model=encoder_model,
                    variant=variant,
                    revision=revision,
                    subfolder=encoder_load_subfolder,
                    dtype=dtype,
                    quantizer=quantizer_uri if should_quantize else None
                )
            )

    text_encoder_override_states = [
        text_encoder_override,
        text_encoder_2_override,
        text_encoder_3_override
    ]

    # Load Text Encoders

    for idx, (name, param) in enumerate(
            [n for n in sorted(model_index.items(), key=lambda x: x[0])
             if _is_text_encoder_component(n[0])
             and isinstance(n[1], (list, tuple))
             and len(n[1]) >= 2
             and n[1][0] is not None]):
        if text_encoder_override_states[idx]:
            continue

        if _hfhub.is_single_file_model_load(model_path):
            encoder_subfolder = name
        else:
            encoder_subfolder = os.path.join(subfolder, name) if subfolder else name

        encoder_class = param[1]

        if text_encoder_uris and len(text_encoder_uris) > idx:
            encoder_uri = text_encoder_uris[idx]

            if _text_encoder_default(encoder_uri):
                creation_kwargs[name] = load_default_text_encoder(encoder_class, name)
            elif not (needs_granular_caching and name in components_to_cache):
                creation_kwargs[name] = load_text_encoder(
                    _uris.TextEncoderUri.parse(encoder_uri)
                )
            # If granular caching is needed and this encoder is in components_to_cache,
            # skip direct loading - it will be handled by granular caching
        else:
            creation_kwargs[name] = load_default_text_encoder(encoder_class, name)

    # Load VAE

    if not vae_override:
        if vae_uri:
            parsed_vae_uri = _uris.VAEUri.parse(vae_uri)

            creation_kwargs['vae'] = load_vae(parsed_vae_uri)

            _messages.debug_log(lambda:
                                f'Added Torch VAE: "{vae_uri}" to pipeline: "{pipeline_class.__name__}"')
        elif 'vae' in pipe_params:
            if _hfhub.is_single_file_model_load(model_path):
                vae_subfolder = 'vae'
            else:
                vae_subfolder = os.path.join(subfolder, 'vae') if subfolder else 'vae'

            vae_param = pipe_params['vae'].annotation

            if _types.is_union(vae_param):
                try:
                    vae_encoder_name = model_index['vae'][1]
                except (KeyError, IndexError):
                    _messages.debug_log(
                        'Skipping auto VAE caching due to model '
                        'configuration not having a VAE key.')
                    vae_encoder_name = None
                except FileNotFoundError as e:
                    raise UnsupportedPipelineConfigError(
                        'Could not find VAE configuration data.') from e

                if vae_encoder_name not in _uris.VAEUri.supported_encoder_names():
                    raise UnsupportedPipelineConfigError(
                        f'Unsupported VAE encoder type: {vae_encoder_name}'
                    )
            elif isinstance(vae_param, str):
                vae_encoder_name = vae_param
            else:
                vae_encoder_name = getattr(vae_param, '__name__', None)

            if vae_encoder_name is not None:
                if single_file_pipeline_repo:
                    vae_model = single_file_pipeline_repo
                    vae_extract_from_checkpoint = False
                else:
                    vae_model = model_path
                    vae_extract_from_checkpoint = _hfhub.is_single_file_model_load(model_path)
                def _load_embedded_vae():
                    try:
                        return load_vae(_uris.VAEUri(
                            encoder=vae_encoder_name,
                            model=vae_model,
                            variant=variant,
                            revision=revision,
                            subfolder=vae_subfolder,
                            extract=vae_extract_from_checkpoint,
                            dtype=dtype))
                    except _d_exceptions.ModelNotFoundError:
                        if vae_extract_from_checkpoint or not variant:
                            raise
                        _messages.debug_log(
                            f'VAE variant {variant!r} is not published for "{vae_model}"; '
                            'loading the default VAE weights.')
                        return load_vae(_uris.VAEUri(
                            encoder=vae_encoder_name,
                            model=vae_model,
                            revision=revision,
                            subfolder=vae_subfolder,
                            dtype=dtype))

                creation_kwargs['vae'], parsed_vae_uri = _load_sd_family_embedded_or_builtin_vae(
                    load_vae,
                    vae_uri=vae_uri,
                    model_type=model_type,
                    local_files_only=local_files_only,
                    dtype=dtype,
                    model_path=model_path,
                    load_embedded=_load_embedded_vae,
                )

    # Load UNet
    if not unet_override:
        unet_parameter = 'unet'
        if model_type == _enums.ModelType.S_CASCADE:
            unet_parameter = 'prior'
        elif model_type == _enums.ModelType.S_CASCADE_DECODER:
            unet_parameter = 'decoder'

        unet_class = diffusers.UNet2DConditionModel if unet_parameter == 'unet' \
            else diffusers.models.unets.StableCascadeUNet

        if unet_uri is not None and not (needs_granular_caching and 'unet' in components_to_cache):
            parsed_unet_uri = _uris.UNetUri.parse(unet_uri)

            if _enums.model_type_is_kolors(model_type) and parsed_unet_uri.quantizer:
                raise UnsupportedPipelineConfigError(
                    f'--model-type {_enums.get_model_type_string(model_type)} does not support '
                    f'loading a --unet with quantization applied.'
                )

            creation_kwargs[unet_parameter] = load_unet(
                parsed_unet_uri, unet_class=unet_class
            )

            _messages.debug_log(lambda:
                                f'Added Torch UNet: "{unet_uri}" to pipeline: "{pipeline_class.__name__}"')
        elif 'unet' in pipe_params:

            if _hfhub.is_single_file_model_load(model_path):
                unet_subfolder = unet_parameter
            else:
                unet_subfolder = os.path.join(subfolder, unet_parameter) if subfolder else unet_parameter

            # Check if we have a cached version from granular caching
            if 'unet' in cached_component_paths:
                unet_model_path = cached_component_paths['unet']
                unet_subfolder = None  # No subfolder for cached component
                unet_revision = None  # No revision for cached component

                # Use manual quantizer if available, otherwise use global quantizer
                unet_quantizer = manual_quantizers.get('unet',
                                                       quantizer_uri if should_apply_quantizer("unet") else None)
            else:
                unet_model_path = model_path
                unet_revision = revision
                unet_quantizer = quantizer_uri if should_apply_quantizer("unet") else None

            creation_kwargs['unet'] = \
                load_unet(
                    _uris.UNetUri(
                        model=unet_model_path,
                        variant=variant,
                        revision=unet_revision,
                        subfolder=unet_subfolder,
                        dtype=dtype,
                        quantizer=unet_quantizer
                    ), unet_class=unet_class)

    # Load Transformer

    transformer_class = diffusion_transformer_class(model_type)

    if not transformer_override:
        if transformer_uri is not None and not (
                needs_granular_caching and transformer_class is not None and 'transformer' in components_to_cache):
            assert transformer_class is not None

            parsed_transformer_uri = _uris.TransformerUri.parse(transformer_uri)

            creation_kwargs['transformer'] = load_transformer(
                parsed_transformer_uri,
                transformer_class=transformer_class
            )

            _messages.debug_log(lambda:
                                f'Added Torch Transformer: "{transformer_uri}" to '
                                f'pipeline: "{pipeline_class.__name__}"')
        elif 'transformer' in pipe_params:
            assert transformer_class is not None

            if _hfhub.is_single_file_model_load(model_path):
                transformer_subfolder = 'transformer'
            else:
                transformer_subfolder = os.path.join(subfolder, 'transformer') if subfolder else 'transformer'

            # Check if we have a cached version from granular caching
            if 'transformer' in cached_component_paths:
                transformer_model_path = cached_component_paths['transformer']
                transformer_subfolder = None  # No subfolder for cached component
                transformer_revision = None  # No revision for cached component

                # Use manual quantizer if available, otherwise use global quantizer
                transformer_quantizer = manual_quantizers.get('transformer', quantizer_uri if should_apply_quantizer(
                    "transformer") else None)
            else:
                transformer_model_path = model_path
                transformer_revision = revision
                transformer_quantizer = quantizer_uri if should_apply_quantizer("transformer") else None

            creation_kwargs['transformer'] = load_transformer(
                _uris.TransformerUri(
                    model=transformer_model_path,
                    variant=variant,
                    revision=transformer_revision,
                    subfolder=transformer_subfolder,
                    dtype=dtype,
                    quantizer=transformer_quantizer
                ), transformer_class=transformer_class)

    # load image encoder

    if image_encoder_uri is not None and not image_encoder_override:
        parsed_image_encoder_uri = _uris.ImageEncoderUri.parse(image_encoder_uri)

        if _enums.model_type_is_sd3(model_type):
            # image encoder does not participate in sequential offloading for SD3
            no_cache_image_encoder = model_cpu_offload or model_group_offload
        else:
            no_cache_image_encoder = submodules_on_pipeline

        loaded_image_encoder = parsed_image_encoder_uri.load(
            dtype_fallback=dtype,
            auth_token=auth_token,
            local_files_only=local_files_only,
            no_cache=no_cache_image_encoder,
            image_encoder_class=
            _models.SiglipImageEncoder
            if _enums.model_type_is_sd3(model_type) else
            transformers.CLIPVisionModelWithProjection
        )

        if isinstance(loaded_image_encoder, _models.SiglipImageEncoder):
            creation_kwargs['image_encoder'] = loaded_image_encoder.image_encoder
            creation_kwargs['feature_extractor'] = loaded_image_encoder.feature_extractor
        else:
            creation_kwargs['image_encoder'] = loaded_image_encoder

        _messages.debug_log(lambda:
                            f'Added Torch Image Encoder: "{image_encoder_uri}" to '
                            f'pipeline: "{pipeline_class.__name__}"')

    # Load T2I Adapters

    if t2i_adapter_uris and not adapter_override:
        t2i_adapters = None

        for t2i_adapter_uri in t2i_adapter_uris:
            parsed_t2i_adapter_uri = _uris.T2IAdapterUri.parse(t2i_adapter_uri)
            parsed_t2i_adapter_uris.append(parsed_t2i_adapter_uri)

            new_adapter = parsed_t2i_adapter_uri.load(
                auth_token=auth_token,
                dtype_fallback=dtype,
                local_files_only=local_files_only,
                no_cache=submodules_on_pipeline
            )

            _messages.debug_log(lambda:
                                f'Added Torch T2IAdapter: "{t2i_adapter_uri}" '
                                f'to pipeline: "{pipeline_class.__name__}"')

            if t2i_adapters is not None:
                if not isinstance(t2i_adapters, list):
                    t2i_adapters = [t2i_adapters, new_adapter]
                else:
                    t2i_adapters.append(new_adapter)
            else:
                t2i_adapters = new_adapter

        if isinstance(t2i_adapters, list):
            creation_kwargs['adapter'] = diffusers.MultiAdapter(t2i_adapters)
        else:
            creation_kwargs['adapter'] = t2i_adapters

    # Load ControlNets

    if controlnet_uris and not controlnet_override:
        controlnets = None
        sdxl_cn_union = None

        for controlnet_uri in controlnet_uris:
            parsed_controlnet_uri = _uris.ControlNetUri.parse(
                uri=controlnet_uri,
                model_type=model_type
            )

            parsed_controlnet_uris.append(parsed_controlnet_uri)

            # Apply global quantizer if controlnet doesn't have
            # its own quantizer and should be quantized
            controlnet_uri_to_load = parsed_controlnet_uri
            if not parsed_controlnet_uri.quantizer and should_apply_quantizer('controlnet'):
                # Create a new URI with the global quantizer
                controlnet_uri_to_load = _uris.ControlNetUri(
                    model=parsed_controlnet_uri.model,
                    revision=parsed_controlnet_uri.revision,
                    variant=parsed_controlnet_uri.variant,
                    subfolder=parsed_controlnet_uri.subfolder,
                    dtype=parsed_controlnet_uri.dtype,
                    scale=parsed_controlnet_uri.scale,
                    start=parsed_controlnet_uri.start,
                    end=parsed_controlnet_uri.end,
                    mode=parsed_controlnet_uri.mode,
                    quantizer=quantizer_uri,
                    model_type=parsed_controlnet_uri.model_type
                )

            new_net = controlnet_uri_to_load.load(
                auth_token=auth_token,
                dtype_fallback=dtype,
                local_files_only=local_files_only,
                no_cache=controlnet_is_owned_by_pipeline(
                    model_type, model_cpu_offload, sequential_cpu_offload, model_group_offload),
                device_map=get_device_map_for_quantizer(controlnet_uri_to_load.quantizer)
            )

            # Apply casting hack for quantized controlnets
            if controlnet_uri_to_load.quantizer:
                new_net.forward = functools.partial(
                    controlnet_quant_forward,
                    new_net.forward,
                    new_net
                )

            if _enums.model_type_is_qwen_image(model_type):
                _validate_qwen_image_controlnet_condition_channels(
                    new_net, pipeline_type, controlnet_uri)

            _messages.debug_log(lambda:
                                f'Added Torch ControlNet: "{controlnet_uri}" '
                                f'to pipeline: "{pipeline_class.__name__}"')

            if sdxl_cn_union is not None:
                continue
            if isinstance(new_net, diffusers.ControlNetUnionModel):
                # first model determines controlnet model,
                # the rest of the specifications just provide the mode
                sdxl_cn_union = new_net
                continue

            if controlnets is not None:
                if not isinstance(controlnets, list):
                    controlnets = [controlnets, new_net]
                else:
                    controlnets.append(new_net)
            else:
                controlnets = new_net

        if sdxl_cn_union is not None:
            controlnets = sdxl_cn_union

        if isinstance(controlnets, list):
            # not handled internally for whatever reason like the other pipelines
            if _enums.model_type_is_sd3(model_type):
                creation_kwargs['controlnet'] = diffusers.SD3MultiControlNetModel(controlnets)
            elif _enums.model_type_is_flux(model_type):
                creation_kwargs['controlnet'] = diffusers.FluxMultiControlNetModel(controlnets)
            elif _enums.model_type_is_qwen_image(model_type):
                creation_kwargs['controlnet'] = diffusers.QwenImageMultiControlNetModel(controlnets)
            elif _enums.model_type_is_hydit(model_type):
                creation_kwargs['controlnet'] = diffusers.HunyuanDiT2DMultiControlNetModel(controlnets)
            else:
                creation_kwargs['controlnet'] = controlnets
        else:
            creation_kwargs['controlnet'] = controlnets

    if _enums.model_type_is_floyd(model_type):
        creation_kwargs['watermarker'] = None

    if not safety_checker and \
            (_enums.model_type_is_sd15(model_type) or
             _enums.model_type_is_floyd(model_type)) and not safety_checker_override:
        creation_kwargs['safety_checker'] = None

    creation_kwargs.update(extra_modules)

    hydit_ip_adapter_image_encoder = None
    if _enums.model_type_is_hydit(model_type):
        hydit_ip_adapter_image_encoder = creation_kwargs.pop('image_encoder', None)
        if hydit_ip_adapter_image_encoder is not None:
            creation_kwargs.pop('feature_extractor', None)

    def _handle_generic_pipeline_load_failure(e):
        exc_msg = str(e)

        _messages.debug_log(
            f'Failed to load primary pipeline model: "{model_path}", reason: {exc_msg}')

        if model_path in exc_msg:
            if 'restricted' in exc_msg:
                # the gated repo message is far more useful to the user
                raise InvalidModelFileError(exc_msg) from e
            else:
                raise InvalidModelFileError(f'invalid model file or repo slug: {model_path}') from e

        raise InvalidModelFileError(e) from e

    if _hfhub.is_single_file_model_load(model_path):
        if subfolder is not None:
            raise UnsupportedPipelineConfigError(
                'Single file model loads do not support the subfolder option.')
        try:
            _enforce_pipeline_cache_size(estimated_memory_usage)

            # Diffusers loads the transformer out of the file itself unless
            # dgenerate already supplied it.
            with _safetensorscompat.single_file_components(
                    transformer_loaded='transformer' in creation_kwargs):
                if single_file_pipeline_repo is not None:
                    # Transformer-only checkpoints: components are loaded above
                    # from the detected Diffusers repo; the file is the transformer.
                    _messages.debug_log(
                        'Single file transformer checkpoint; loading pipeline '
                        f'components from "{single_file_pipeline_repo}".')
                    pipeline = _pipeline_creation_args_debug(
                        backend='Torch',
                        cls=pipeline_class,
                        method=pipeline_class.from_pretrained,
                        model=single_file_pipeline_repo,
                        token=auth_token,
                        revision=revision,
                        variant=_repo_pipeline_variant(variant, model_index, creation_kwargs),
                        torch_dtype=torch_dtype,
                        local_files_only=local_files_only,
                        **creation_kwargs)
                else:
                    pipeline = _pipeline_creation_args_debug(
                        backend='Torch',
                        cls=pipeline_class,
                        method=pipeline_class.from_single_file,
                        original_config=original_config,
                        model=model_path,
                        token=auth_token,
                        revision=revision,
                        variant=variant,
                        torch_dtype=torch_dtype,
                        use_safe_tensors=model_path.endswith('.safetensors'),
                        local_files_only=local_files_only,
                        **creation_kwargs)

        except diffusers.loaders.single_file.SingleFileComponentError as e:
            _handle_single_file_component_error(e)
        except (ValueError, TypeError, NameError, OSError) as e:
            _handle_generic_pipeline_load_failure(e)
    else:
        try:
            pipeline = _pipeline_creation_args_debug(
                backend='Torch',
                cls=pipeline_class,
                method=pipeline_class.from_pretrained,
                model=model_path,
                token=auth_token,
                revision=revision,
                variant=_repo_pipeline_variant(variant, model_index, creation_kwargs),
                torch_dtype=torch_dtype,
                subfolder=subfolder,
                local_files_only=local_files_only,
                **creation_kwargs)
        except (ValueError, TypeError, NameError, OSError) as e:
            _handle_generic_pipeline_load_failure(e)

    if hasattr(pipeline, 'vae') and \
            _enums.model_type_is_sd3(model_type):
        # patch to enable tiling at all resolutions
        if pipeline.vae.quant_conv is None:
            pipeline.vae.quant_conv = lambda x: x
        if pipeline.vae.post_quant_conv is None:
            pipeline.vae.post_quant_conv = lambda x: x

    if hydit_ip_adapter_image_encoder is not None:
        # Do not register_modules: HunyuanDiT is not an image_encoder pipeline component.
        pipeline.image_encoder = hydit_ip_adapter_image_encoder

    # Textual Inversions, LoRAs, IP Adapters

    parsed_textual_inversion_uris = []
    parsed_lora_uris = []
    parsed_ip_adapter_uris = []

    if textual_inversion_uris:
        for inversion_uri in textual_inversion_uris:
            parsed = _uris.TextualInversionUri.parse(inversion_uri)
            parsed_textual_inversion_uris.append(parsed)

        _uris.TextualInversionUri.load_on_pipeline(
            pipeline=pipeline,
            uris=parsed_textual_inversion_uris,
            auth_token=auth_token,
            local_files_only=local_files_only)

    if lora_uris and not quantizer_uri:
        # LoRAs are fused into the model and cached
        # to disk in the case that quantization is requested

        for lora_uri in lora_uris:
            parsed = _uris.LoRAUri.parse(lora_uri)
            parsed_lora_uris.append(parsed)

        _uris.LoRAUri.load_on_pipeline(
            pipeline=pipeline,
            uris=parsed_lora_uris,
            fuse_scale=lora_fuse_scale if lora_fuse_scale is not None else 1.0,
            auth_token=auth_token,
            local_files_only=local_files_only)

    if ip_adapter_uris:
        for ip_adapter_uri in ip_adapter_uris:
            parsed = _uris.IPAdapterUri.parse(ip_adapter_uri)
            parsed_ip_adapter_uris.append(parsed)

        _uris.IPAdapterUri.load_on_pipeline(
            pipeline=pipeline,
            uris=parsed_ip_adapter_uris,
            auth_token=auth_token,
            local_files_only=local_files_only)

    if ip_adapter_uris and (not hasattr(pipeline, 'image_encoder') or pipeline.image_encoder is None):
        raise UnsupportedPipelineConfigError(
            'Using --ip-adapters but missing required --image-encoder specification, '
            'your --ip-adapters specification did not include an image encoder model and '
            'you must specify one manually.')

    # Safety Checker

    if not safety_checker_override:
        if _enums.model_type_is_floyd(model_type):
            _set_floyd_safety_checker(pipeline, safety_checker)
        else:
            _set_sd_safety_checker(pipeline, safety_checker)

    # Compile before offload hooks wrap forward.
    if torch_compile:
        apply_torch_compile(pipeline, device)

    # Model Offloading

    # SD3 / Hunyuan-DiT IP-Adapter image_encoder must not use leaf sequential offload
    # (meta tensor errors when accelerate hooks wrap CLIP).

    if (sequential_cpu_offload or model_group_offload) and (
            (_enums.model_type_is_sd3(model_type) and 'image_encoder' in creation_kwargs)
            or hydit_ip_adapter_image_encoder is not None
            or (_enums.model_type_is_hydit(model_type) and ip_adapter_uris)):
        # The Diffusers list is a class attribute. Append would change every
        # pipeline of that class and grow on each load. Copy it onto the instance.
        excluded = list(getattr(pipeline, '_exclude_from_cpu_offload', ()) or ())
        if 'image_encoder' not in excluded:
            excluded.append('image_encoder')
        pipeline._exclude_from_cpu_offload = excluded

    if not device.startswith('cpu'):
        if sequential_cpu_offload:
            enable_sequential_cpu_offload(pipeline, device)
        elif model_cpu_offload:
            enable_model_cpu_offload(pipeline, device)
        elif model_group_offload:
            enable_group_offload(pipeline, device)

    if hasattr(pipeline, 'image_encoder') and pipeline.image_encoder is not None:
        offload_device = getattr(pipeline, '_dgenerate_offload_device', None)
        if offload_device is not None:
            from dgenerate.extras.hydit.image_encoder import HunyuanDiTCLIPImageEncoder
            if not isinstance(pipeline.image_encoder, HunyuanDiTCLIPImageEncoder):
                pipeline.image_encoder.to(offload_device)

    _messages.debug_log(f'Finished Creating Torch Pipeline: "{pipeline_class.__name__}"')

    # modules quantized in 8 bit by bitsandbytes cannot be moved off the GPU
    # there is a sequence in call_pipeline that garbage collects them out of the
    # cache before calling other pipelines, there is also a dgenerate.devicecache
    # hook to clear them out when requested
    for module in get_pipeline_modules(pipeline).values():
        if _util.is_loaded_in_8bit_bnb(module):
            _disable_to(module)

    # noinspection PyTypeChecker
    return PipelineCreationResult(
        model_path=model_path,
        pipeline=pipeline,
        parsed_unet_uri=parsed_unet_uri,
        parsed_transformer_uri=parsed_transformer_uri,
        parsed_vae_uri=parsed_vae_uri,
        parsed_lora_uris=parsed_lora_uris,
        parsed_image_encoder_uri=parsed_image_encoder_uri,
        parsed_ip_adapter_uris=parsed_ip_adapter_uris,
        parsed_textual_inversion_uris=parsed_textual_inversion_uris,
        parsed_controlnet_uris=parsed_controlnet_uris,
        parsed_t2i_adapter_uris=parsed_t2i_adapter_uris
    ), _d_memoize.CachedObjectMetadata(size=estimated_memory_usage)


def _handle_single_file_component_error(e):
    msg = str(e)
    if 'text_encoder' in msg:
        raise UnsupportedPipelineConfigError(
            f'Single file load error, missing --text-encoders / --second-model-text-encoders:\n{e}') from e
    elif 'vae =' in msg:
        raise UnsupportedPipelineConfigError(
            f'Single file load error, missing --vae:\n{e}') from e
    else:
        raise UnsupportedPipelineConfigError(
            f'Single file load error, missing component:\n{e}') from e


def get_converted_checkpoint_cache_dir():
    """
    Get cache directory where dgenerate stores any checkpoints that needed to be converted into
    diffusers directory format on disk to function, this process is used to
    support quantization on single file checkpoints.

    Or the value of the environmental variable ``DGENERATE_CACHE`` joined with ``diffusers_converted``.

    :return: string (directory path)
    """
    user_cache_path = os.environ.get('DGENERATE_CACHE')

    if user_cache_path is not None:
        path = os.path.join(user_cache_path, 'web')
    else:
        path = os.path.expanduser(os.path.join('~', '.cache', 'dgenerate', 'diffusers_converted'))

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    return path


_to_diffusers_cache_dir = get_converted_checkpoint_cache_dir()


# Component-specific caches for granular caching
def _get_component_cache_store(component_name: str):
    component_dir = os.path.join(_to_diffusers_cache_dir, component_name)
    pathlib.Path(component_dir).mkdir(parents=True, exist_ok=True)
    return _filecache.KeyValueStore(os.path.join(component_dir, 'cache.db'))


_component_cache_stores = {
    'text_encoder': _get_component_cache_store('text_encoder'),
    'text_encoder_2': _get_component_cache_store('text_encoder_2'),
    'text_encoder_3': _get_component_cache_store('text_encoder_3'),
    'transformer': _get_component_cache_store('transformer'),
    'unet': _get_component_cache_store('unet'),
}

# Records LoRA-to-component relationships as individual SQLite key/value
# entries. Older cache databases have no such records and remain readable via
# the legacy component lookup below.
_lora_component_cache_store = _filecache.KeyValueStore(
    os.path.join(_to_diffusers_cache_dir, 'lora_components', 'cache.db'))


def _get_component_cache_key(
        component_name: str,
        model_path: str,
        model_type: _enums.ModelType,
        lora_uris: _types.OptionalUris,
        lora_fuse_scale: float | None,
        revision: str | None,
        variant: str | None,
        subfolder: str | None,
        dtype: _enums.DataType,
        original_config: str | None,
        manual_component_uris: dict[str, str] | None = None
) -> str:
    manual_uri_part = ""
    if manual_component_uris and component_name in manual_component_uris:
        # Strip quantizer from manual URI for cache key since components are saved in original precision
        try:
            if component_name == 'unet':
                parsed_uri = _uris.UNetUri.parse(manual_component_uris[component_name])
                uri_without_quantizer = _uris.UNetUri(
                    model=parsed_uri.model,
                    revision=parsed_uri.revision,
                    variant=parsed_uri.variant,
                    subfolder=parsed_uri.subfolder,
                    dtype=parsed_uri.dtype,
                    quantizer=None
                )
                manual_uri_part = f"|manual:{str(uri_without_quantizer)}"
            elif component_name == 'transformer':
                parsed_uri = _uris.TransformerUri.parse(manual_component_uris[component_name])
                uri_without_quantizer = _uris.TransformerUri(
                    model=parsed_uri.model,
                    revision=parsed_uri.revision,
                    variant=parsed_uri.variant,
                    subfolder=parsed_uri.subfolder,
                    dtype=parsed_uri.dtype,
                    quantizer=None
                )
                manual_uri_part = f"|manual:{str(uri_without_quantizer)}"
            elif _is_text_encoder_component(component_name):
                parsed_uri = _uris.TextEncoderUri.parse(manual_component_uris[component_name])
                uri_without_quantizer = _uris.TextEncoderUri(
                    encoder=parsed_uri.encoder,
                    model=parsed_uri.model,
                    revision=parsed_uri.revision,
                    variant=parsed_uri.variant,
                    subfolder=parsed_uri.subfolder,
                    dtype=parsed_uri.dtype,
                    quantizer=None
                )
                manual_uri_part = f"|manual:{str(uri_without_quantizer)}"
            else:
                # Fallback for unknown component types
                manual_uri_part = f"|manual:{manual_component_uris[component_name]}"
        except Exception:
            # If parsing fails, use the original URI
            manual_uri_part = f"|manual:{manual_component_uris[component_name]}"

    # LoRA order consideration: LoRA fusion order can affect the final result since
    # LoRA operations are not necessarily commutative. We preserve the original order
    # to ensure mathematical correctness and respect user intent.
    # 
    # If you want to optimize cache efficiency by normalizing LoRA order (at the cost
    # of potentially different results), you could sort the URIs like this:
    # lora_uris_for_cache = sorted(lora_uris) if lora_uris else lora_uris
    lora_uris_for_cache = lora_uris

    # Note: quantizer_uri is NOT included in the cache key because components are saved
    # in original precision. The quantizer is only applied when loading from cache.
    return f"{component_name}|{model_path}|{str(lora_uris_for_cache)}|{str(lora_fuse_scale)}|" \
           f"{str(model_type)}|{str(revision)}|{str(variant)}|{str(subfolder)}|" \
           f"{str(dtype)}|{str(original_config)}{manual_uri_part}"


def _get_affected_components_for_lora(pipeline_class) -> list[str]:
    if hasattr(pipeline_class, '_lora_loadable_modules'):
        return list(pipeline_class._lora_loadable_modules)
    return []


def _get_lora_component_cache_identity(
        model_path: str,
        model_type: _enums.ModelType,
        pipeline_type: _enums.PipelineType,
        lora_uris: _types.OptionalUris,
        lora_fuse_scale: float | None,
        revision: str | None,
        variant: str | None,
        subfolder: str | None,
        dtype: _enums.DataType,
        original_config: str | None,
        manual_component_uris: dict[str, str] | None,
        components: list[str]
) -> str:
    """Build a database identity for a LoRA and its candidate components."""
    return repr((
        model_path, str(model_type), str(pipeline_type), tuple(lora_uris or ()),
        lora_fuse_scale, revision, variant, subfolder, str(dtype), original_config,
        tuple(sorted((manual_component_uris or {}).items())), tuple(components)))


def _get_quantizable_components(model_type: _enums.ModelType) -> list[str]:
    components = []

    # Add UNet or Transformer based on model type
    if (_enums.model_type_is_sd3(model_type) or _enums.model_type_is_flux(model_type)
            or _enums.model_type_is_flow_image(model_type)):
        # For SD3 and Flux, single file checkpoints typically only contain the transformer
        # Text encoders are usually separate standard models that don't need conversion
        components.append('transformer')
    else:
        # For older models (SD1.5, SD2, SDXL), single files often contain UNet and text encoders
        components.append('unet')
        if _enums.model_type_is_sdxl(model_type):
            components.extend(['text_encoder', 'text_encoder_2'])
        else:
            components.append('text_encoder')

    return components


def _create_minimal_pipeline_for_component_extraction(
        model_path: str,
        model_type: _enums.ModelType,
        pipeline_type: _enums.PipelineType,
        components_needed: list[str],
        revision: str | None,
        variant: str | None,
        subfolder: str | None,
        dtype: _enums.DataType,
        original_config: str | None,
        auth_token: str | None,
        lora_uris: _types.OptionalUris,
        lora_fuse_scale: float | None,
        manual_component_uris: dict[str, str] | None = None,
        vae_uri: _types.OptionalUri = None
):
    pipeline_class = get_pipeline_class(
        model_type=model_type,
        pipeline_type=pipeline_type,
        help_mode=True  # Allow creation even if pipeline_type doesn't match
    )

    torch_dtype = _enums.get_torch_dtype(dtype)
    creation_kwargs = {}
    manual_quantizers = {}  # Store quantizer info from manual URIs

    single_file_pipeline_repo = None
    if _hfhub.is_single_file_model_load(model_path):
        single_file_pipeline_repo = _singlefilecompat.single_file_pretrained_repo_for_model_path(
            model_path, auth_token=auth_token, local_files_only=False)
    component_repo = single_file_pipeline_repo or model_path
    encoder_index = None
    if single_file_pipeline_repo or not _hfhub.is_single_file_model_load(model_path):
        try:
            encoder_index = _util.fetch_model_index_dict(
                component_repo,
                revision=revision,
                subfolder=None if single_file_pipeline_repo else subfolder,
                auth_token=auth_token)
        except Exception as encoder_index_error:
            _messages.debug_log(
                f'Could not read model_index for text encoders: {encoder_index_error}')

    # Get all text encoder parameters that the pipeline expects
    pipe_params = inspect.signature(pipeline_class.__init__).parameters
    text_encoder_params = [name for name in pipe_params.keys() if _is_text_encoder_component(name)]

    # Load ALL text encoders that the pipeline expects, even if not being cached
    # This prevents SingleFileComponentError during pipeline creation
    for encoder_param in text_encoder_params:
        # Check if we have a manual URI for this encoder
        if manual_component_uris and encoder_param in manual_component_uris:
            parsed_uri = _uris.TextEncoderUri.parse(manual_component_uris[encoder_param])

            # Save quantizer info and reconstruct URI without quantizer
            if parsed_uri.quantizer:
                manual_quantizers[encoder_param] = parsed_uri.quantizer

            # Reconstruct URI without quantizer
            uri_without_quantizer = _uris.TextEncoderUri(
                encoder=parsed_uri.encoder,
                model=parsed_uri.model,
                revision=parsed_uri.revision,
                variant=parsed_uri.variant,
                subfolder=parsed_uri.subfolder,
                dtype=parsed_uri.dtype,
                quantizer=None  # Explicitly set to None
            )

            creation_kwargs[encoder_param] = uri_without_quantizer.load(
                variant_fallback=variant,
                dtype_fallback=dtype,
                original_config=original_config,
                auth_token=auth_token,
                local_files_only=False,
                no_cache=True,
                missing_ok=False
            )
        else:
            # Load default text encoder for this parameter
            # We need to load it even if we're not caching it to prevent errors
            try:
                if single_file_pipeline_repo or _hfhub.is_single_file_model_load(model_path):
                    encoder_subfolder = encoder_param
                    encoder_load_model = single_file_pipeline_repo or model_path
                else:
                    encoder_subfolder = os.path.join(subfolder, encoder_param) if subfolder else encoder_param
                    encoder_load_model = model_path

                # Infer encoder class from model_type. Do not trust
                # model_index.json here — some repos ship incorrect class
                # names. Flow-image models are the exception: their encoder
                # layout varies enough that the index is still consulted.
                # Transformer-only single files use the detected repo index
                # the same way, so Qwen and Z-Image encoders are not loaded
                # as CLIP.
                #
                # SD3 text_encoder must be CLIPTextModelWithProjection.
                # Loading it as CLIPTextModel drops text_projection; the
                # fused cache then reloads with zero matching weights under
                # bitsandbytes (Parameter without CB).
                flow_encoder_name = None
                if encoder_index and (
                        _enums.model_type_is_flow_image(model_type) or single_file_pipeline_repo):
                    flow_entry = encoder_index.get(encoder_param)
                    if flow_entry and flow_entry[1]:
                        flow_encoder_name = flow_entry[1]
                if flow_encoder_name:
                    encoder_name = flow_encoder_name
                elif encoder_param == 'text_encoder':
                    if _enums.model_type_is_sd3(model_type):
                        encoder_name = 'CLIPTextModelWithProjection'
                    else:
                        encoder_name = 'CLIPTextModel'  # Default for most SD models
                elif encoder_param == 'text_encoder_2':
                    if _enums.model_type_is_sdxl(model_type):
                        encoder_name = 'CLIPTextModelWithProjection'
                    elif _enums.model_type_is_sd3(model_type):
                        encoder_name = 'CLIPTextModelWithProjection'
                    elif _enums.model_type_is_flux(model_type):
                        encoder_name = 'CLIPTextModel'
                    else:
                        encoder_name = 'CLIPTextModel'
                elif encoder_param == 'text_encoder_3':
                    if _enums.model_type_is_sd3(model_type) or _enums.model_type_is_flux(model_type):
                        encoder_name = 'T5EncoderModel'
                    else:
                        encoder_name = 'CLIPTextModel'
                else:
                    encoder_name = 'CLIPTextModel'

                default_encoder_uri = _uris.TextEncoderUri(
                    encoder=encoder_name,
                    model=encoder_load_model,
                    variant=variant,
                    revision=revision,
                    subfolder=encoder_subfolder,
                    dtype=dtype,
                    quantizer=None
                )

                creation_kwargs[encoder_param] = default_encoder_uri.load(
                    variant_fallback=variant,
                    dtype_fallback=dtype,
                    original_config=original_config,
                    auth_token=auth_token,
                    local_files_only=False,
                    no_cache=True,
                    missing_ok=True  # Allow missing for optional encoders
                )
            except Exception as e:
                _messages.debug_log(f"Failed to load default {encoder_param}: {e}")
                # Continue without this encoder - some may be optional

    # Load manual components that are not text encoders
    if manual_component_uris:
        for component in components_needed:
            if component in manual_component_uris and not _is_text_encoder_component(component):
                if component == 'unet':
                    unet_class = diffusers.UNet2DConditionModel
                    parsed_uri = _uris.UNetUri.parse(manual_component_uris[component])

                    # Save quantizer info and reconstruct URI without quantizer
                    if parsed_uri.quantizer:
                        manual_quantizers[component] = parsed_uri.quantizer

                    # Reconstruct URI without quantizer
                    uri_without_quantizer = _uris.UNetUri(
                        model=parsed_uri.model,
                        revision=parsed_uri.revision,
                        variant=parsed_uri.variant,
                        subfolder=parsed_uri.subfolder,
                        dtype=parsed_uri.dtype,
                        quantizer=None  # Explicitly set to None
                    )

                    creation_kwargs['unet'] = uri_without_quantizer.load(
                        variant_fallback=variant,
                        dtype_fallback=dtype,
                        original_config=original_config,
                        auth_token=auth_token,
                        local_files_only=False,
                        no_cache=True,
                        unet_class=unet_class
                    )
                elif component == 'transformer':
                    transformer_class = diffusion_transformer_class(model_type)
                    if transformer_class is None:
                        continue

                    parsed_uri = _uris.TransformerUri.parse(manual_component_uris[component])

                    # Save quantizer info and reconstruct URI without quantizer
                    if parsed_uri.quantizer:
                        manual_quantizers[component] = parsed_uri.quantizer

                    # Reconstruct URI without quantizer
                    uri_without_quantizer = _uris.TransformerUri(
                        model=parsed_uri.model,
                        revision=parsed_uri.revision,
                        variant=parsed_uri.variant,
                        subfolder=parsed_uri.subfolder,
                        dtype=parsed_uri.dtype,
                        quantizer=None  # Explicitly set to None
                    )

                    creation_kwargs['transformer'] = uri_without_quantizer.load(
                        variant_fallback=variant,
                        dtype_fallback=dtype,
                        original_config=original_config,
                        auth_token=auth_token,
                        local_files_only=False,
                        no_cache=True,
                        config=None if _hfhub.is_single_file_model_load(model_path) else model_path,
                        config_subfolder='transformer' if not _hfhub.is_single_file_model_load(model_path) else None,
                        transformer_class=transformer_class
                    )

    # Load VAE if specified - this is needed for single file checkpoints that don't contain a VAE
    if vae_uri:
        parsed_vae_uri = _uris.VAEUri.parse(vae_uri)
        creation_kwargs['vae'] = parsed_vae_uri.load(
            dtype_fallback=dtype,
            original_config=original_config,
            auth_token=auth_token,
            local_files_only=False,
            no_cache=True,
            missing_ok=False
        )

    # Load the pipeline with all required components
    if _hfhub.is_single_file_model_load(model_path):
        try:
            if single_file_pipeline_repo is not None:
                pipeline = pipeline_class.from_pretrained(
                    single_file_pipeline_repo,
                    token=auth_token,
                    revision=revision,
                    variant=_repo_pipeline_variant(variant, model_index, creation_kwargs),
                    torch_dtype=torch_dtype,
                    local_files_only=local_files_only,
                    **creation_kwargs)
            else:
                pipeline = pipeline_class.from_single_file(
                    model_path,
                    original_config=original_config,
                    token=auth_token,
                    revision=revision,
                    variant=variant,
                    torch_dtype=torch_dtype,
                    use_safe_tensors=model_path.endswith('.safetensors'),
                    **creation_kwargs
                )
        except diffusers.loaders.single_file.SingleFileComponentError as e:
            _handle_single_file_component_error(e)
        except Exception as e:
            _messages.debug_log(f"Failed to create minimal pipeline from single file: {e}")
            raise
    else:
        try:
            pipeline = pipeline_class.from_pretrained(
                model_path,
                token=auth_token,
                revision=revision,
                variant=variant,
                torch_dtype=torch_dtype,
                subfolder=subfolder,
                **creation_kwargs
            )
        except Exception as e:
            _messages.debug_log(f"Failed to create minimal pipeline from pretrained: {e}")
            raise

    # Apply LoRAs if specified
    if lora_uris:
        parsed_lora_uris = [_uris.LoRAUri.parse(uri) for uri in lora_uris]

        _uris.LoRAUri.load_on_pipeline(
            pipeline=pipeline,
            uris=parsed_lora_uris,
            fuse_scale=lora_fuse_scale if lora_fuse_scale is not None else 1.0,
            auth_token=auth_token
        )

    return pipeline, manual_quantizers


def _cache_components_granular(
        model_path: str,
        model_type: _enums.ModelType,
        pipeline_type: _enums.PipelineType,
        components_to_cache: list[str],
        lora_uris: _types.OptionalUris,
        lora_fuse_scale: float | None,
        revision: str | None,
        variant: str | None,
        subfolder: str | None,
        dtype: _enums.DataType,
        original_config: str | None,
        auth_token: str | None,
        manual_component_uris: dict[str, str] | None = None,
        vae_uri: _types.OptionalUri = None
) -> tuple[dict[str, str], dict[str, str]]:
    cached_component_paths = {}
    components_to_load = []

    lora_cache_identity = None
    component_mapping_known = False
    if lora_uris:
        lora_cache_identity = _get_lora_component_cache_identity(
            model_path, model_type, pipeline_type, lora_uris, lora_fuse_scale,
            revision, variant, subfolder, dtype, original_config,
            manual_component_uris, components_to_cache)
        known_key = f'{lora_cache_identity}|complete'
        known_value = _lora_component_cache_store.get(known_key)
        if known_value == '1':
            component_mapping_known = True
            components_to_cache = [
                component for component in components_to_cache
                if _lora_component_cache_store.get(
                    f'{lora_cache_identity}|component|{component}') == '1']

    # Check which components are already cached
    for component in components_to_cache:
        cache_key = _get_component_cache_key(
            component, model_path, model_type, lora_uris, lora_fuse_scale,
            revision, variant, subfolder, dtype, original_config,
            manual_component_uris
        )

        cache_store = _component_cache_stores[component]

        with cache_store:
            cached_path = cache_store.get(cache_key)
            if cached_path and os.path.exists(cached_path):
                cached_component_paths[component] = cached_path
                _messages.debug_log(f"Using cached {component} from: {cached_path}")
            else:
                components_to_load.append(component)

    # A partial set without database metadata may come from an older broad
    # cache. Reinspect all candidates before recording the affected components.
    if lora_uris and not component_mapping_known and components_to_load and cached_component_paths:
        cached_component_paths.clear()
        components_to_load = list(components_to_cache)

    # If all components are cached, we still need to extract manual quantizer info
    manual_quantizers = {}
    if manual_component_uris:
        for component, uri in manual_component_uris.items():
            if component in components_to_cache:
                try:
                    if component == 'unet':
                        parsed_uri = _uris.UNetUri.parse(uri)
                        if parsed_uri.quantizer:
                            manual_quantizers[component] = parsed_uri.quantizer
                    elif component == 'transformer':
                        parsed_uri = _uris.TransformerUri.parse(uri)
                        if parsed_uri.quantizer:
                            manual_quantizers[component] = parsed_uri.quantizer
                    elif _is_text_encoder_component(component):
                        parsed_uri = _uris.TextEncoderUri.parse(uri)
                        if parsed_uri.quantizer:
                            manual_quantizers[component] = parsed_uri.quantizer
                except Exception as e:
                    _messages.debug_log(f"Error extracting quantizer from {component} URI: {e}")

    # If all components are cached, return early
    if not components_to_load:
        return cached_component_paths, manual_quantizers

    # Create pipeline and extract needed components
    if lora_uris:
        _messages.warning(
            f'Model "{model_path}" is having LoRAs '
            f'fused into specific components, that will then be cached on disk '
            f'prior to quantization. This is a one time task per LoRA scale value, '
            f'please be patient...'
        )
    else:
        _messages.warning(
            f'Model "{model_path}" components are being converted to '
            f'diffusers format and cached on disk prior to quantization. '
            f'This is a one time task, please be patient...'
        )

    with _d_memoize.disable_memoization_context():
        pipeline, manual_quantizers = _create_minimal_pipeline_for_component_extraction(
            model_path=model_path,
            model_type=model_type,
            pipeline_type=pipeline_type,
            components_needed=components_to_load,
            revision=revision,
            variant=variant,
            subfolder=subfolder,
            dtype=dtype,
            original_config=original_config,
            auth_token=auth_token,
            lora_uris=lora_uris,
            lora_fuse_scale=lora_fuse_scale,
            manual_component_uris=manual_component_uris,
            vae_uri=vae_uri
        )

    if lora_uris:
        affected_components = tuple(dict.fromkeys((
            *cached_component_paths,
            *(component for component in components_to_load
              if hasattr(pipeline, component)
              and getattr(getattr(pipeline, component), 'peft_config', None)))))
        pipeline.unload_lora_weights()
    else:
        affected_components = tuple(components_to_load)

    # Cache each component individually
    for component in components_to_load:
        if component not in affected_components:
            continue
        if not hasattr(pipeline, component):
            _messages.debug_log(
                f"Component {component} not found in pipeline, skipping cache")
            continue

        component_obj = getattr(pipeline, component)
        if component_obj is None:
            _messages.debug_log(
                f"Component {component} is None, skipping cache")
            continue

        # Get the cache store for the component
        cache_store = _component_cache_stores[component]

        # Generate cache key and path
        cache_key = _get_component_cache_key(
            component, model_path, model_type, lora_uris, lora_fuse_scale,
            revision, variant, subfolder, dtype, original_config,
            manual_component_uris
        )

        # Generate cache path with component/uuid structure
        component_dir = os.path.join(_to_diffusers_cache_dir, component)
        pathlib.Path(component_dir).mkdir(parents=True, exist_ok=True)

        with cache_store:
            cache_uuid = hashlib.sha256(cache_key.encode('utf-8')).hexdigest()
            cache_path = os.path.join(component_dir, cache_uuid)

            # Ensure unique path
            while os.path.exists(cache_path):
                cache_uuid = hashlib.sha256((cache_key + str(random.random())).encode('utf-8')).hexdigest()
                cache_path = os.path.join(component_dir, cache_uuid)

        # Save component
        _messages.debug_log(f"Saving cached {component} to: {cache_path}")
        from .weightsave import save_module_with_progress
        save_module_with_progress(
            component_obj, cache_path, variant=variant, desc=f'Writing {component}')

        # Store in cache
        with cache_store:
            cache_store[cache_key] = cache_path

        cached_component_paths[component] = cache_path

    if lora_uris and lora_cache_identity is not None:
        with _lora_component_cache_store:
            for component in affected_components:
                _lora_component_cache_store[
                    f'{lora_cache_identity}|component|{component}'] = '1'
            _lora_component_cache_store[f'{lora_cache_identity}|complete'] = '1'

    return cached_component_paths, manual_quantizers


__all__ = _types.module_all()
