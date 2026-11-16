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

"""Wan 2.2 video-to-video with the high-noise and low-noise experts."""

from __future__ import annotations

import torch
from diffusers import WanVideoToVideoPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.pipelines.wan.pipeline_wan_video2video import (
    XLA_AVAILABLE,
    retrieve_timesteps,
)


def loaded_pipe_has_low_noise_expert(pipe) -> bool:
    """True when ``pipe`` is a Wan 2.2 MoE pipeline.

    Video-to-video uses :class:`WanMoEVideoToVideoPipeline` only in that case.
    A Wan 2.1 pipeline has one transformer and stays on
    ``WanVideoToVideoPipeline``.
    """
    components = getattr(pipe, 'components', None) or {}
    low = components.get('transformer_2', getattr(pipe, 'transformer_2', None))
    if low is None:
        return False
    config = getattr(pipe, 'config', None)
    if config is None:
        return False
    if hasattr(config, 'get'):
        boundary = config.get('boundary_ratio', None)
    else:
        boundary = getattr(config, 'boundary_ratio', None)
    return boundary is not None


class WanMoEVideoToVideoPipeline(WanVideoToVideoPipeline):
    """Video-to-video denoising that switches Wan 2.2 experts by timestep.

    Diffusers ships the switch on ``WanPipeline`` and
    ``WanImageToVideoPipeline``. ``WanVideoToVideoPipeline`` takes one
    transformer. This subclass keeps ``transformer_2`` and
    ``boundary_ratio``, and uses the low-noise expert once the timestep
    falls below ``boundary_ratio * num_train_timesteps``.
    """

    model_cpu_offload_seq = 'text_encoder->transformer->transformer_2->vae'
    _optional_components = ['transformer_2']

    def __init__(self, tokenizer, text_encoder, transformer, vae, scheduler,
                 transformer_2=None, boundary_ratio=None, expand_timesteps=False):
        super().__init__(tokenizer, text_encoder, transformer, vae, scheduler)
        self.register_modules(transformer_2=transformer_2)
        self.register_to_config(
            boundary_ratio=boundary_ratio, expand_timesteps=bool(expand_timesteps))

    @torch.no_grad()
    def __call__(
            self,
            video=None,
            prompt=None,
            negative_prompt=None,
            height: int = 480,
            width: int = 832,
            num_inference_steps: int = 50,
            timesteps=None,
            guidance_scale: float = 5.0,
            guidance_scale_2=None,
            strength: float = 0.8,
            num_videos_per_prompt=1,
            generator=None,
            latents=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            output_type='np',
            return_dict: bool = True,
            attention_kwargs=None,
            callback_on_step_end=None,
            callback_on_step_end_tensor_inputs=None,
            max_sequence_length: int = 512,
    ):
        boundary = getattr(self.config, 'boundary_ratio', None)
        low = getattr(self, 'transformer_2', None)
        if low is None or boundary is None:
            if guidance_scale_2 is not None and boundary is None:
                raise ValueError(
                    '`guidance_scale_2` is only supported when the pipeline\'s '
                    '`boundary_ratio` is not None.')
            return super().__call__(
                video=video,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
                strength=strength,
                num_videos_per_prompt=num_videos_per_prompt,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs
                or ['latents'],
                max_sequence_length=max_sequence_length,
            )

        if callback_on_step_end_tensor_inputs is None:
            callback_on_step_end_tensor_inputs = ['latents']
        if guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_videos_per_prompt = 1

        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            video,
            latents,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        if XLA_AVAILABLE:
            timestep_device = 'cpu'
        else:
            timestep_device = device
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, timestep_device, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, timesteps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)
        self._num_timesteps = len(timesteps)

        if latents is None:
            video = self.video_processor.preprocess_video(video, height=height, width=width).to(
                device, dtype=torch.float32)

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            video,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
            latent_timestep,
        )

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        boundary_timestep = float(boundary) * self.scheduler.config.num_train_timesteps
        expand_timesteps = bool(getattr(self.config, 'expand_timesteps', False))
        if expand_timesteps:
            mask = torch.ones(latents.shape, dtype=torch.float32, device=latents.device)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                if t >= boundary_timestep:
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    current_model = low
                    current_guidance_scale = guidance_scale_2

                latent_model_input = latents.to(current_model.dtype)
                if expand_timesteps:
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                with current_model.cache_context('cond'):
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if self.do_classifier_free_guidance:
                    with current_model.cache_context('uncond'):
                        noise_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop('latents', latents)
                    prompt_embeds = callback_outputs.pop('prompt_embeds', prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        'negative_prompt_embeds', negative_prompt_embeds)

                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    import torch_xla.core.xla_model as xm
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == 'latent':
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)
        return WanPipelineOutput(frames=video)
