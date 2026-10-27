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

import inspect
import typing

import diffusers
import torch

from .ip_adapter import HunyuanDiTIPAdapterMixin
from .lora import HunyuanDiTLoraLoaderMixin

# POSITIONAL_OR_KEYWORD so inspect.getfullargspec(...).args includes them (dgenerate
# checks .args, not .kwonlyargs, when wiring pipeline kwargs).
_IP_ADAPTER_CALL_PARAMS = (
    inspect.Parameter(
        'ip_adapter_image',
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        default=None,
        annotation=typing.Any),
    inspect.Parameter(
        'ip_adapter_images',
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        default=None,
        annotation=typing.Any),
)

_CONTROL_GUIDANCE_CALL_PARAMS = (
    inspect.Parameter(
        'control_guidance_start',
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        default=0.0,
        annotation=float | list[float]),
    inspect.Parameter(
        'control_guidance_end',
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        default=1.0,
        annotation=float | list[float]),
)


def _align_control_guidance(value: typing.Any, count: int, name: str) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * count
    values = [float(item) for item in value]
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(
            f'{name} has {len(values)} values for {count} Hunyuan-DiT ControlNet(s).')
    return values


def _control_guidance_keep(step: int, steps: int, start: float, end: float) -> float:
    """1 inside ``[start, end]`` as fractions of the denoising schedule, else 0."""
    position = step / steps
    next_position = (step + 1) / steps
    if position < start or next_position > end:
        return 0.0
    return 1.0


def _hydit_denoising_steps(pipeline, fallback: int) -> int:
    """
    Step count the ControlNet forward actually walks.

    ``num_inference_steps`` matches that count for a first-order scheduler.
    A custom timestep list, or a scheduler whose order is greater than one,
    makes ``scheduler.timesteps`` longer. The denoising loop calls the
    ControlNet once per entry in that list.
    """
    scheduler = getattr(pipeline, 'scheduler', None)
    timesteps = getattr(scheduler, 'timesteps', None)
    if timesteps is None:
        return fallback
    try:
        count = len(timesteps)
    except TypeError:
        return fallback
    return count if count > 0 else fallback


def align_ip_adapter_clip_embeds(
        embeds: torch.Tensor,
        prompt_batch: int,
        num_images_per_prompt: int,
        guidance_scale: float) -> torch.Tensor:
    """
    Match Hunyuan-DiT's classifier-free batch order.

    One prompt repeated ``num_images_per_prompt`` times becomes
    ``[image, image, ...]``. Several prompts become ``[p0, p0, p1, p1]``,
    the same order as the pipeline's prompt-embed repeat. Guidance then
    puts a zero block in front of that conditional block:
    ``[uncond..., cond...]``.
    """
    num_images_per_prompt = max(int(num_images_per_prompt), 1)
    prompt_batch = max(int(prompt_batch), 1)
    cond = prompt_batch * num_images_per_prompt
    rows = int(embeds.shape[0])
    if rows == prompt_batch:
        embeds = embeds.repeat_interleave(num_images_per_prompt, dim=0)
    elif rows == 1:
        embeds = embeds.repeat(cond, *([1] * (embeds.ndim - 1)))
    elif rows != cond:
        raise ValueError(
            f'Hunyuan-DiT IP-Adapter received {rows} image embedding(s) '
            f'for {cond} generation(s).')
    if guidance_scale > 1.0:
        embeds = torch.cat([torch.zeros_like(embeds), embeds], dim=0)
    return embeds


def _hydit_prompt_batch(arguments: dict[str, typing.Any]) -> int:
    for key in ('prompt', 'prompt_embeds'):
        value = arguments.get(key)
        if isinstance(value, str):
            return 1
        if isinstance(value, (list, tuple)) and value:
            return len(value)
        shape = getattr(value, 'shape', None)
        if shape is not None and len(shape) >= 1 and int(shape[0]) > 0:
            return int(shape[0])
    return 1


def _scaled_controlnet_conditioning(
        base_scale: typing.Any,
        keeps: list[float],
        *,
        multi: bool):
    if multi:
        scales = list(base_scale) if isinstance(base_scale, (list, tuple)) else [float(base_scale)] * len(keeps)
        if len(scales) != len(keeps):
            raise ValueError(
                f'controlnet_conditioning_scale has {len(scales)} values for {len(keeps)} ControlNet(s).')
        return [float(scale) * keep for scale, keep in zip(scales, keeps)]
    scale = base_scale[0] if isinstance(base_scale, (list, tuple)) else float(base_scale)
    return scale * keeps[0]


def _hydit_controlnet_call_pre(
        pipeline: diffusers.HunyuanDiTControlNetPipeline,
        arguments: dict[str, typing.Any]) -> None:
    control_image = arguments.get('control_image')
    multi = isinstance(pipeline.controlnet, diffusers.HunyuanDiT2DMultiControlNetModel)
    if isinstance(control_image, (list, tuple)) and not multi and len(control_image) == 1:
        arguments['control_image'] = control_image[0]

    starts = _align_control_guidance(
        arguments.pop('control_guidance_start', 0.0),
        len(pipeline.controlnet.nets) if multi else 1,
        'control_guidance_start')
    ends = _align_control_guidance(
        arguments.pop('control_guidance_end', 1.0),
        len(starts),
        'control_guidance_end')
    pipeline._hydit_control_guidance_restore = None
    if all(start == 0.0 for start in starts) and all(end == 1.0 for end in ends):
        return

    fallback_steps = int(arguments.get('num_inference_steps') or 1)
    explicit_timesteps = arguments.get('timesteps')
    if explicit_timesteps is not None:
        try:
            listed = len(explicit_timesteps)
        except TypeError:
            listed = 0
        if listed > 0:
            fallback_steps = listed
    base_scale = arguments.get('controlnet_conditioning_scale', 1.0)
    original_forward = pipeline.controlnet.forward
    state = {'step': 0}

    def forward(*args, **kwargs):
        steps = _hydit_denoising_steps(pipeline, fallback_steps)
        step = state['step']
        state['step'] += 1
        keeps = [
            _control_guidance_keep(step, steps, start, end)
            for start, end in zip(starts, ends)
        ]
        kwargs['conditioning_scale'] = _scaled_controlnet_conditioning(
            base_scale, keeps, multi=multi)
        return original_forward(*args, **kwargs)

    pipeline.controlnet.forward = forward

    def restore() -> None:
        pipeline.controlnet.forward = original_forward

    pipeline._hydit_control_guidance_restore = restore


def _make_hydit_call_method(
        base_pipeline_cls: type[diffusers.DiffusionPipeline],
        *,
        extra_parameters: tuple[inspect.Parameter, ...] = (),
        pre_super_hook: typing.Callable[
            [diffusers.DiffusionPipeline, dict[str, typing.Any]], None] | None = None):
    """
    Mirror the diffusers ``__call__`` signature on our subclass and add IP-Adapter kwargs.

    dgenerate inspects ``__call__`` to decide which pipeline arguments are supported.
    """
    base_call = base_pipeline_cls.__call__.__wrapped__
    base_sig = inspect.signature(base_call)
    call_sig = base_sig.replace(
        parameters=[*base_sig.parameters.values(), *extra_parameters, *_IP_ADAPTER_CALL_PARAMS])

    def __call__(self, *args, **kwargs):
        bound = call_sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        arguments = {key: value for key, value in bound.arguments.items() if key != 'self'}

        ip_image = arguments.pop('ip_adapter_image', None)
        if ip_image is None:
            ip_image = arguments.pop('ip_adapter_images', None)
        else:
            arguments.pop('ip_adapter_images', None)

        processors = getattr(self, '_ip_adapter_processors', None)
        if ip_image is not None and processors:
            guidance_scale = float(arguments.get('guidance_scale', 1.0) or 1.0)
            clip_embeds = self.encode_ip_adapter_clip_embedding(ip_image)
            clip_embeds = align_ip_adapter_clip_embeds(
                clip_embeds,
                _hydit_prompt_batch(arguments),
                int(arguments.get('num_images_per_prompt') or 1),
                guidance_scale)
            z = self.project_ip_adapter_clip_embedding(clip_embeds)
            for proc in processors:
                proc.image_prompt_embeds = z

        if pre_super_hook is not None:
            pre_super_hook(self, arguments)

        temporary_lora = (
            bool(getattr(self, '_lora_adapters', None))
            and not getattr(self, '_hydit_lora_fused', False))
        if temporary_lora:
            self._add_lora_deltas()

        try:
            return base_pipeline_cls.__call__(self, **arguments)
        finally:
            if temporary_lora:
                self._restore_lora_snapshots()
            restore = getattr(self, '_hydit_control_guidance_restore', None)
            if restore is not None:
                restore()
                self._hydit_control_guidance_restore = None
            if processors:
                for proc in processors:
                    proc.image_prompt_embeds = None

    __call__.__name__ = '__call__'
    __call__.__doc__ = base_call.__doc__
    __call__.__signature__ = call_sig
    __call__.__wrapped__ = base_call
    __call__.__qualname__ = f'{base_pipeline_cls.__name__}.__call__'
    if getattr(base_call, '__annotations__', None):
        __call__.__annotations__ = dict(base_call.__annotations__)

    # torch.no_grad() hides __signature__, which dgenerate uses to decide
    # which pipeline arguments to pass.
    wrapped = torch.no_grad()(__call__)
    wrapped.__signature__ = call_sig
    return wrapped


class HunyuanDiTPipeline(diffusers.HunyuanDiTPipeline, HunyuanDiTLoraLoaderMixin, HunyuanDiTIPAdapterMixin):
    """
    HunyuanDiT text-to-image pipeline with LoRA and IP-Adapter support.
    """

    __call__ = _make_hydit_call_method(diffusers.HunyuanDiTPipeline)


class HunyuanDiTControlNetPipeline(
        diffusers.HunyuanDiTControlNetPipeline, HunyuanDiTLoraLoaderMixin, HunyuanDiTIPAdapterMixin):
    """
    HunyuanDiT ControlNet pipeline with LoRA and IP-Adapter support.
    """

    __call__ = _make_hydit_call_method(
        diffusers.HunyuanDiTControlNetPipeline,
        extra_parameters=_CONTROL_GUIDANCE_CALL_PARAMS,
        pre_super_hook=_hydit_controlnet_call_pre)
