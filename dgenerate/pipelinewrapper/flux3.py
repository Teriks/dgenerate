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

"""FLUX.3 Image Klein modular pipeline loading and placement."""

from __future__ import annotations

import contextlib
import gc
import inspect

import torch

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.enums as _enums


def declared_init_parameters(pipeline_class):
    """
    Init parameters the generic loader should treat FLUX.3 as having.

    ``Flux3ModularPipeline`` does not list its components on ``__init__``.
    The loader still builds the transformer, VAE, and text encoder the same
    way it does for a ``DiffusionPipeline``.
    """
    import diffusers

    parameters = dict(inspect.signature(pipeline_class.__init__).parameters)

    def keyword(name, annotation):
        return inspect.Parameter(
            name, inspect.Parameter.KEYWORD_ONLY, default=None, annotation=annotation)

    parameters.setdefault(
        'text_encoder', keyword('text_encoder', str))
    parameters.setdefault(
        'transformer', keyword('transformer', diffusers.Flux3Transformer2DModel))
    parameters.setdefault(
        'vae', keyword('vae', diffusers.AutoencoderKLFlux2))
    return parameters


def is_distilled(pipeline) -> bool:
    """True for the 16-step flow-map Klein checkpoint."""
    if pipeline is None:
        return False
    if 'Distilled' in type(pipeline).__name__:
        return True
    transformer = getattr(pipeline, 'transformer', None)
    config = getattr(transformer, 'config', None)
    if config is None:
        return False
    if hasattr(config, 'get'):
        return bool(config.get('flow_map', False))
    return bool(getattr(config, 'flow_map', False))


@contextlib.contextmanager
def override_upsampler_system_prompt(system_prompt):
    """Use ``system_prompt`` for one FLUX.3 upsampler call, then restore the built-ins."""
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        yield
        return
    import diffusers.modular_pipelines.flux3.prompt_upsampler as _pup

    saved = dict(_pup.SYSTEM_PROMPTS)
    try:
        for mode in _pup.SYSTEM_PROMPTS:
            _pup.SYSTEM_PROMPTS[mode] = system_prompt
        yield
    finally:
        _pup.SYSTEM_PROMPTS.clear()
        _pup.SYSTEM_PROMPTS.update(saved)


def _drop_prompt_upsampler_block(pipe) -> None:
    blocks = getattr(pipe, '_blocks', None)
    sub_blocks = getattr(blocks, 'sub_blocks', None)
    if isinstance(sub_blocks, dict):
        sub_blocks.pop('prompt_upsampler', None)


def _load_custom_prompt_upsampler(uri_string, model_dtype, token, local_files_only):
    """Load a ``--prompt-upsampler`` model and its processor."""
    import dgenerate.pipelinewrapper.uris as _uris
    import dgenerate.pipelinewrapper.videopipelines.quantization as _quantization

    uri = _uris.PromptUpsamplerUri.parse(uri_string)
    loaded = uri.load(
        dtype_fallback=model_dtype or _enums.DataType.BFLOAT16,
        auth_token=token,
        local_files_only=local_files_only,
        device_map=_quantization._quantized_device_map('cpu', False, uri.quantizer))
    return loaded.model, loaded.processor


def load_flux3_pipeline(model_path, injected, model_dtype, revision=None,
                        variant=None, subfolder=None, prompt_upsampler_uri=None,
                        token=None, local_files_only=False):
    """
    Load a FLUX.3 modular pipeline and its remaining components.

    ``from_pretrained`` only builds the block graph. Weights load in
    ``load_components``. Injected modules (a GGUF transformer, ``--vae``,
    ``--text-encoders``) are registered first so they are not loaded again.
    The image VAE stays in float32. ``prompt_upsampler_uri`` ``None`` leaves
    the prompt upsampler unloaded. ``+`` loads the checkpoint upsampler.
    Any other value is a ``--prompt-upsampler`` URI loaded in its place.
    """
    from diffusers import ModularPipeline

    injected = dict(injected or {})
    pipe = ModularPipeline.from_pretrained(
        model_path,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        **({} if not subfolder else {'subfolder': subfolder}))
    if not prompt_upsampler_uri:
        for name in ('prompt_upsampler', 'prompt_upsampler_processor'):
            if name in getattr(pipe, '_component_specs', {}):
                injected[name] = None
        _drop_prompt_upsampler_block(pipe)
    elif prompt_upsampler_uri != '+':
        model, processor = _load_custom_prompt_upsampler(
            prompt_upsampler_uri, model_dtype, token, local_files_only)
        injected['prompt_upsampler'] = model
        injected['prompt_upsampler_processor'] = processor
    modules = {
        name: value for name, value in injected.items()
        if value is not None and name in getattr(pipe, '_component_specs', {})}
    if modules:
        pipe.update_components(**modules)
    null_names = frozenset(
        name for name, value in injected.items()
        if value is None and name in getattr(pipe, '_component_specs', {}))
    model_torch = _enums.get_torch_dtype(model_dtype) or torch.bfloat16
    component_kwargs = {'torch_dtype': model_torch}
    for name in ('revision', 'variant', 'local_files_only', 'token'):
        value = {
            'revision': revision,
            'variant': variant,
            'local_files_only': local_files_only,
            'token': token,
        }[name]
        if value is not None and value is not False:
            component_kwargs[name] = value
    if subfolder:
        component_kwargs['subfolder'] = subfolder

    def unloaded(name):
        spec = pipe._component_specs.get(name)
        if spec is None or spec.default_creation_method != 'from_pretrained':
            return False
        if getattr(spec, 'pretrained_model_name_or_path', None) is None:
            return False
        if name in null_names or getattr(pipe, name, None) is not None:
            return False
        return True

    if unloaded('vae'):
        vae_kwargs = dict(component_kwargs)
        vae_kwargs['torch_dtype'] = torch.float32
        pipe.load_components(names=['vae'], **vae_kwargs)
    elif getattr(pipe, 'vae', None) is not None and not _quantized(pipe.vae):
        pipe.vae.to(dtype=torch.float32)
    load_names = [name for name in pipe._component_specs if unloaded(name)]
    if load_names:
        pipe.load_components(names=load_names, **component_kwargs)
    if null_names:
        pipe.update_components(**{name: None for name in null_names})
    if prompt_upsampler_uri:
        def reload(target=pipe):
            if prompt_upsampler_uri != '+':
                model, processor = _load_custom_prompt_upsampler(
                    prompt_upsampler_uri, model_dtype, token, local_files_only)
                target.update_components(
                    prompt_upsampler=model,
                    prompt_upsampler_processor=processor)
            else:
                target.load_components(
                    names=['prompt_upsampler', 'prompt_upsampler_processor'],
                    **component_kwargs)
            _bind_upsampler_generate(
                target.prompt_upsampler,
                getattr(target, '_dgenerate_flux3_device', 'cpu'))
        pipe._dgenerate_reload_prompt_upsampler = reload
    return pipe


def _quantized(module) -> bool:
    name = type(module).__name__.lower()
    return 'gguf' in name or 'sdnq' in name or 'bitsandbytes' in name or 'quant' in name


def _bind_upsampler_generate(module, device):
    """Run the upsampler's ``generate()`` without filling shared GPU memory."""
    import dgenerate.pipelinewrapper.promptenhance as _promptenhance
    _promptenhance.bind_rewriter_generate(module, device, park=True)


_DENOISER_NAMES = ('text_encoder', 'transformer', 'vae')


def _is_denoiser(name: str) -> bool:
    return any(name == stem or name.startswith(stem + '_') for stem in _DENOISER_NAMES)


def _module_is_hooked(module) -> bool:
    return getattr(module, '_DGENERATE_ORIGINAL_TO_DISABLED', None) is not None or getattr(
        module, '_hf_hook', None) is not None


def _place_one(module, device, pipelines):
    if module is None or not hasattr(module, 'to'):
        return
    if pipelines.module_skips_group_offload(module):
        pipelines.place_quantized_module(module, device)
    else:
        module.to(device)


def _clear_denoiser(pipe):
    """Move the denoiser off the GPU so the upsampler rewrite runs first."""
    import dgenerate.pipelinewrapper.pipelines as _pipelines
    import dgenerate.pipelinewrapper.util as _putil

    pipe._dgenerate_flux3_denoiser_placed = False
    warned = False
    for name, module in _pipelines.get_pipeline_modules(pipe).items():
        if not _is_denoiser(name) or _module_is_hooked(module):
            continue
        if _putil.is_loaded_in_8bit_bnb(module):
            if not warned:
                _messages.log(
                    f'{name}: 8-bit bitsandbytes weights stay on the GPU. '
                    'They cannot be moved for the rewrite.')
                warned = True
            continue
        try:
            current = next(module.parameters()).device
        except (StopIteration, AttributeError):
            continue
        if current.type == 'cpu':
            continue
        module.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _place_denoiser(pipe, device, offload: bool):
    """Put the text encoder, transformer, and VAE where denoising expects them."""
    import dgenerate.pipelinewrapper.pipelines as _pipelines

    if getattr(pipe, '_dgenerate_flux3_denoiser_placed', False):
        return
    heavy = ('text_encoder', 'transformer')
    if offload:
        import accelerate
        for name in heavy:
            module = getattr(pipe, name, None)
            if module is None or not hasattr(module, 'to') or _module_is_hooked(module):
                continue
            if _pipelines.module_skips_group_offload(module):
                _pipelines.place_quantized_module(module, device)
                continue
            accelerate.cpu_offload(
                module, device, offload_buffers=len(module._parameters) > 0)
            _pipelines._set_cpu_offload_flag(module, True)
        _pipelines._set_cpu_offload_flag(pipe, True)
    for name, module in _pipelines.get_pipeline_modules(pipe).items():
        if not _is_denoiser(name):
            continue
        if offload and (name in heavy or name.startswith('text_encoder') or name.startswith('transformer')):
            continue
        if _module_is_hooked(module):
            continue
        _place_one(module, device, _pipelines)
    pipe._dgenerate_flux3_denoiser_placed = True


def _release_8bit_upsampler(pipe):
    """Drop an 8-bit upsampler after its step returns so denoising can use the GPU."""
    import dgenerate.pipelinewrapper.util as _putil

    module = getattr(pipe, 'prompt_upsampler', None)
    reload = getattr(pipe, '_dgenerate_reload_prompt_upsampler', None)
    if module is None or not _putil.is_loaded_in_8bit_bnb(module) or reload is None:
        return
    pipe.prompt_upsampler = None
    del module
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _messages.log(
        'FLUX.3 prompt upsampler: 8-bit bitsandbytes weights were dropped '
        'after the rewrite. The next call reloads them.')


def _prompt_from(container):
    """Read a caption from a block-state dict or a pipeline state."""
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get('prompt')
    getter = getattr(container, 'get', None)
    if callable(getter):
        try:
            value = getter('prompt')
        except TypeError:
            value = None
        if value:
            return value
    return getattr(container, 'prompt', None)


def _upsampled_prompt(step, state):
    """Caption the upsampler step wrote back onto the pipeline state."""
    getter = getattr(step, 'get_block_state', None)
    if callable(getter) and state is not None:
        try:
            block_state = getter(state)
        except Exception:
            block_state = None
        prompt = _prompt_from(block_state)
        if prompt:
            return prompt
    return _prompt_from(state)


def _wrap_upsampler_step(pipe, device, offload: bool):
    blocks = getattr(getattr(pipe, '_blocks', None), 'sub_blocks', None)
    step = blocks.get('prompt_upsampler') if isinstance(blocks, dict) else None
    if step is None:
        _place_denoiser(pipe, device, offload)
        return
    if not getattr(step.__class__, '_dgenerate_upsampler_call_wrapped', False):
        original = step.__class__.__call__

        def wrapped(self, components, state):
            prepare = getattr(self, '_dgenerate_before_upsampler', None)
            if prepare is not None:
                prepare(components)
            result = original(self, components, state)
            import dgenerate.pipelinewrapper.promptenhance as _promptenhance
            _promptenhance.log_enhanced_prompt(_upsampled_prompt(self, state))
            finish = getattr(self, '_dgenerate_after_upsampler', None)
            if finish is not None:
                finish(components)
            return result

        step.__class__.__call__ = wrapped
        step.__class__._dgenerate_upsampler_call_wrapped = True

    def before(components):
        if getattr(components, 'prompt_upsampler', None) is None:
            reload = getattr(components, '_dgenerate_reload_prompt_upsampler', None)
            if reload is not None:
                reload()
        _clear_denoiser(components)

    def after(components):
        _release_8bit_upsampler(components)
        _place_denoiser(components, device, offload)

    step._dgenerate_before_upsampler = before
    step._dgenerate_after_upsampler = after


def place_flux3(pipe, device, model_cpu_offload, sequential_cpu_offload,
                model_group_offload):
    """
    Place FLUX.3 components.

    The modular pipeline has no ``model_cpu_offload_seq``. Any offload flag
    hooks the text encoder and transformer so each visits the GPU for its
    forward and returns to CPU. When a prompt upsampler is loaded, the text
    encoder, transformer, and VAE stay off the GPU until that rewrite returns.
    The upsampler keeps only the layer prefix that fits, then leaves. An
    8-bit bitsandbytes upsampler is dropped and reloaded on the next call.
    """
    import dgenerate.pipelinewrapper.pipelines as _pipelines
    import dgenerate.pipelinewrapper.util as _putil

    pipe._dgenerate_flux3_device = device
    offload = bool(model_cpu_offload or sequential_cpu_offload or model_group_offload)
    upsampler = getattr(pipe, 'prompt_upsampler', None)
    if offload:
        if model_cpu_offload:
            chosen = '--model-cpu-offload'
        elif sequential_cpu_offload:
            chosen = '--model-sequential-offload'
        else:
            chosen = '--model-group-offload'
        if upsampler is not None and _putil.is_loaded_in_8bit_bnb(upsampler):
            upsampler_sentence = (
                ' The 8-bit prompt upsampler is rewritten first, then dropped '
                'so its weights leave the GPU. The next call reloads it.'
                if getattr(pipe, '_dgenerate_reload_prompt_upsampler', None) is not None
                else ' The 8-bit prompt upsampler stays on the GPU. Its weights '
                     'cannot be split or moved.')
        elif upsampler is not None:
            upsampler_sentence = (
                ' The prompt upsampler is rewritten first: the layers that fit '
                'stay on the GPU and the rest run from CPU, then it leaves the GPU '
                'before the text encoder, transformer, and VAE are placed.')
        else:
            upsampler_sentence = ''
        _messages.warning(
            'FLUX.3 uses one placement for --model-cpu-offload, '
            '--model-sequential-offload, and --model-group-offload. '
            f'{chosen} hooks the text encoder and transformer so each visits '
            'the GPU for its forward pass.'
            f'{upsampler_sentence}')
    if upsampler is not None:
        _pipelines._set_cpu_offload_flag(upsampler, True)
        _bind_upsampler_generate(upsampler, device)
        _wrap_upsampler_step(pipe, device, offload)
        transformer = getattr(pipe, 'transformer', None)
        if transformer is not None and not getattr(transformer, '_dgenerate_flux3_place_hook', False):
            def _place_on_first_forward(module, inputs, _pipe=pipe, _device=device, _offload=offload):
                if not getattr(_pipe, '_dgenerate_flux3_denoiser_placed', False):
                    _place_denoiser(_pipe, _device, _offload)
                return None
            transformer.register_forward_pre_hook(_place_on_first_forward)
            transformer._dgenerate_flux3_place_hook = True
    else:
        _place_denoiser(pipe, device, offload)
    for name, module in _pipelines.get_pipeline_modules(pipe).items():
        if name == 'prompt_upsampler' or _is_denoiser(name):
            continue
        _place_one(module, device, _pipelines)


def images_from_call(result):
    """The image list a modular ``output='images'`` call returned."""
    if isinstance(result, dict):
        result = result.get('images', result.get('image'))
    if isinstance(result, (list, tuple)):
        return list(result)
    return result
