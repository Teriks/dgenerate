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

"""MiniMax-H3-Image modular pipeline loading and one-call argument checks."""

from __future__ import annotations

import contextlib
import contextvars
import inspect

import torch

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.pipelines as _pipelines


# The checkpoint was distilled on this grid. The transformer has one output
# head per step, so a different step count is not a sampling option.
INFERENCE_STEPS = 6
CANVAS_MULTIPLE = 32
MIN_GENERATION_AREA = 2048 * 2048


def declared_init_parameters(pipeline_class):
    """
    Init parameters the generic loader should treat MiniMax-H3-Image as having.

    ``MiniMaxH3ImageModularPipeline`` does not list its components on ``__init__``.
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
        'transformer', keyword('transformer', diffusers.MiniMaxH3ImageTransformer2DModel))
    parameters.setdefault(
        'vae', keyword('vae', diffusers.AutoencoderKLMiniMaxH3Image))
    return parameters


def load_pipeline(model_path, injected, model_dtype, revision=None,
                  variant=None, subfolder=None, token=None, local_files_only=False):
    """
    Load a MiniMax-H3-Image modular pipeline and its remaining components.

    ``from_pretrained`` only builds the block graph. Weights load in
    ``load_components``. Injected modules (a transformer, ``--vae``,
    ``--text-encoders``) are registered first so they are not loaded again.
    The image VAE stays in float32. The text encoder is required.
    """
    import diffusers

    injected = dict(injected or {})
    if injected.get('text_encoder', True) is None:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-image requires its Qwen3-VL text encoder. '
            'Do not pass a null --text-encoders value.')
    pipe = diffusers.MiniMaxH3ImageModularPipeline.from_pretrained(
        model_path,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        **({} if not subfolder else {'subfolder': subfolder}))
    modules = {
        name: value for name, value in injected.items()
        if value is not None and name in getattr(pipe, '_component_specs', {})}
    if modules:
        pipe.update_components(**modules)
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
        if getattr(pipe, name, None) is not None:
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
    return pipe


def _quantized(module) -> bool:
    name = type(module).__name__.lower()
    return 'gguf' in name or 'sdnq' in name or 'bitsandbytes' in name or 'quant' in name


_defer_place = contextvars.ContextVar('dgenerate_defer_minimax_image_place', default=False)


@contextlib.contextmanager
def defer_place():
    """Build the pipeline without moving the VAE onto the generation device.

    Prompt enhance then uses that device for the text encoder. Placement runs
    after the rewrite, through :func:`finish_deferred_place`.
    """
    token = _defer_place.set(True)
    try:
        yield
    finally:
        _defer_place.reset(token)


def finish_deferred_place(pipe) -> None:
    """Run placement that :func:`defer_place` postponed."""
    pending = getattr(pipe, '_dgenerate_minimax_image_place', None)
    if pending is None:
        return
    pipe._dgenerate_minimax_image_place = None
    place(pipe, *pending)


def place(pipe, device, model_cpu_offload, sequential_cpu_offload, model_group_offload):
    """
    Place MiniMax-H3-Image components.

    The modular pipeline has no ``model_cpu_offload_seq``. Any offload flag
    hooks the text encoder and transformer so each visits the GPU for its
    forward and returns to CPU. The VAE stays on the generation device.
    A prompt rewrite postpones this until the text encoder is finished, so
    the VAE is not on that device during the rewrite.
    """
    if _defer_place.get():
        pipe._dgenerate_minimax_image_place = (
            device, model_cpu_offload, sequential_cpu_offload, model_group_offload)
        return
    import dgenerate.pipelinewrapper.flux3 as _flux3
    import dgenerate.pipelinewrapper.minimaxh3 as _minimax

    offload = bool(model_cpu_offload or sequential_cpu_offload or model_group_offload)
    if offload:
        if model_cpu_offload:
            chosen = '--model-cpu-offload'
        elif sequential_cpu_offload:
            chosen = '--model-sequential-offload'
        else:
            chosen = '--model-group-offload'
        _messages.warning(
            'minimax-h3-image uses one placement for --model-cpu-offload, '
            '--model-sequential-offload, and --model-group-offload. '
            f'{chosen} hooks the text encoder and transformer so each visits '
            'the GPU for its forward pass.')
    _minimax.prepare_device_move(pipe)
    _flux3._place_denoiser(pipe, device, offload)
    for name, module in _pipelines.get_pipeline_modules(pipe).items():
        if _flux3._is_denoiser(name):
            continue
        _flux3._place_one(module, device, _pipelines)


def prepare_call(pipeline_args, user_args, prompt):
    """
    Check one MiniMax-H3-Image call and drop arguments the modular pipeline rejects.

    Guidance is distilled. Sampling is the checkpoint's six sigmas, and the
    transformer has one output head per step. ``image`` is the reference list
    when the caller already put one there. A request smaller than
    ``MIN_GENERATION_AREA`` is generated larger and resized back by the pipeline.
    """
    if user_args.sigmas is not None or user_args.timesteps is not None:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-image samples a fixed 6-step schedule. '
            'Omit --sigmas, --timesteps, and --inference-steps.')
    if user_args.inference_steps is not None and int(user_args.inference_steps) != INFERENCE_STEPS:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-image samples a fixed 6-step schedule. '
            'Omit --inference-steps, or pass 6.')
    if prompt.negative:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-image is guidance-distilled and does not accept a negative prompt.')
    guidance = user_args.guidance_scale
    if guidance is not None and float(guidance) != 1.0:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-image is guidance-distilled. Omit --guidance-scales, or pass 1.')
    if user_args.batch_size not in (None, 1):
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-image generates one image per call. Omit --batch-size, or pass 1.')
    if user_args.output_latents:
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-image returns the decoded image. Omit --output-latents.')
    positive = prompt.positive if prompt.positive else ''
    if not positive.strip():
        raise _pipelines.UnsupportedPipelineConfigError(
            'minimax-h3-image requires a prompt.')
    pipeline_args['prompt'] = positive
    # The modular call returns the whole PipelineState unless this selects the images.
    pipeline_args['output'] = 'images'
    for name in (
            'guidance_scale', 'num_inference_steps', 'negative_prompt',
            'num_images_per_prompt', 'strength', 'output_type'):
        pipeline_args.pop(name, None)
    user_args.inference_steps = INFERENCE_STEPS
    user_args.guidance_scale = 1.0
