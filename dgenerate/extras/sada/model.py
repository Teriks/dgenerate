import numpy as np
import torch
from typing import Type, Dict, Any, Optional, Union

from .prune import *
from . import exceptions

from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, scale_lora_layers, unscale_lora_layers
from dataclasses import dataclass


@dataclass
class UNet2DConditionOutput(BaseOutput):
    sample: torch.FloatTensor = None


@dataclass
class Transformer2DModelOutput(BaseOutput):
    sample: "torch.Tensor"  # noqa: F821


def patch_unet(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class PatchedUnet(block_class):
        def forward(
                self,
                sample: torch.FloatTensor,
                timestep: Union[torch.Tensor, float, int],
                encoder_hidden_states: torch.Tensor,
                class_labels: Optional[torch.Tensor] = None,
                timestep_cond: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
                down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
                mid_block_additional_residual: Optional[torch.Tensor] = None,
                down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                return_dict: bool = True,
        ) -> Union[UNet2DConditionOutput, Tuple]:

            skip_this_step = self._cache_bus.skip_this_step
            if skip_this_step and self._cache_bus.prev_epsilon is not None:
                self._cache_bus.last_skip_step = self._cache_bus.step
                sample = self._cache_bus.prev_epsilon
                self._cache_bus.step += 1

                return UNet2DConditionOutput(sample)

            else:
                default_overall_up_factor = 2 ** self.num_upsamplers

                # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
                forward_upsample_size = False
                upsample_size = None

                for dim in sample.shape[-2:]:
                    if dim % default_overall_up_factor != 0:
                        # Forward upsample size to force interpolation output size.
                        forward_upsample_size = True
                        break

                if attention_mask is not None:
                    attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                    attention_mask = attention_mask.unsqueeze(1)

                # convert encoder_attention_mask to a bias the same way we do for attention_mask
                if encoder_attention_mask is not None:
                    encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
                    encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

                # 0. center input if necessary
                if self.config.center_input_sample:
                    sample = 2 * sample - 1.0

                # 1. time
                t_emb = self.get_time_embed(sample=sample, timestep=timestep)
                emb = self.time_embedding(t_emb, timestep_cond)

                class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
                if class_emb is not None:
                    if self.config.class_embeddings_concat:
                        emb = torch.cat([emb, class_emb], dim=-1)
                    else:
                        emb = emb + class_emb

                aug_emb = self.get_aug_embed(
                    emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
                )
                if self.config.addition_embed_type == "image_hint":
                    aug_emb, hint = aug_emb
                    sample = torch.cat([sample, hint], dim=1)

                emb = emb + aug_emb if aug_emb is not None else emb

                if self.time_embed_act is not None:
                    emb = self.time_embed_act(emb)

                encoder_hidden_states = self.process_encoder_hidden_states(
                    encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
                )

                # 2. pre-process
                sample = self.conv_in(sample)

                # 2.5 GLIGEN position net
                if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
                    cross_attention_kwargs = cross_attention_kwargs.copy()
                    gligen_args = cross_attention_kwargs.pop("gligen")
                    cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

                # 3. down
                # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
                # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
                if cross_attention_kwargs is not None:
                    cross_attention_kwargs = cross_attention_kwargs.copy()
                    lora_scale = cross_attention_kwargs.pop("scale", 1.0)
                else:
                    lora_scale = 1.0

                if USE_PEFT_BACKEND:
                    # weight the lora layers by setting `lora_scale` for each PEFT layer
                    scale_lora_layers(self, lora_scale)

                is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
                # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
                is_adapter = down_intrablock_additional_residuals is not None

                if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
                    down_intrablock_additional_residuals = down_block_additional_residuals
                    is_adapter = True

                down_block_res_samples = (sample,)
                for downsample_block in self.down_blocks:
                    if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                        # For t2i-adapter CrossAttnDownBlock2D
                        additional_residuals = {}
                        if is_adapter and len(down_intrablock_additional_residuals) > 0:
                            additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                        sample, res_samples = downsample_block(
                            hidden_states=sample,
                            temb=emb,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask,
                            cross_attention_kwargs=cross_attention_kwargs,
                            encoder_attention_mask=encoder_attention_mask,
                            **additional_residuals,
                        )
                    else:
                        sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                        if is_adapter and len(down_intrablock_additional_residuals) > 0:
                            sample += down_intrablock_additional_residuals.pop(0)

                    down_block_res_samples += res_samples

                if is_controlnet:
                    new_down_block_res_samples = ()

                    for down_block_res_sample, down_block_additional_residual in zip(
                            down_block_res_samples, down_block_additional_residuals
                    ):
                        down_block_res_sample = down_block_res_sample + down_block_additional_residual
                        new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

                    down_block_res_samples = new_down_block_res_samples

                # 4. mid
                if self.mid_block is not None:
                    if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                        sample = self.mid_block(
                            sample,
                            emb,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask,
                            cross_attention_kwargs=cross_attention_kwargs,
                            encoder_attention_mask=encoder_attention_mask,
                        )
                    else:
                        sample = self.mid_block(sample, emb)

                    # To support T2I-Adapter-XL
                    if (
                            is_adapter
                            and len(down_intrablock_additional_residuals) > 0
                            and sample.shape == down_intrablock_additional_residuals[0].shape
                    ):
                        sample += down_intrablock_additional_residuals.pop(0)

                if is_controlnet:
                    sample = sample + mid_block_additional_residual

                # 5. up
                for i, upsample_block in enumerate(self.up_blocks):
                    is_final_block = i == len(self.up_blocks) - 1

                    res_samples = down_block_res_samples[-len(upsample_block.resnets):]
                    down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                    # if we have not reached the final block and need to forward the
                    # upsample size, we do it here
                    if not is_final_block and forward_upsample_size:
                        upsample_size = down_block_res_samples[-1].shape[2:]

                    if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                        sample = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            encoder_hidden_states=encoder_hidden_states,
                            cross_attention_kwargs=cross_attention_kwargs,
                            upsample_size=upsample_size,
                            attention_mask=attention_mask,
                            encoder_attention_mask=encoder_attention_mask,
                        )
                    else:
                        sample = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            upsample_size=upsample_size,
                        )

                # 6. post-process
                if self.conv_norm_out:
                    sample = self.conv_norm_out(sample)
                    sample = self.conv_act(sample)
                sample = self.conv_out(sample)

                # clone the sample to cache for skipping step
                self._cache_bus.prev_epsilon = sample
                self._cache_bus.step += 1

                if USE_PEFT_BACKEND:
                    unscale_lora_layers(self, lora_scale)

                if not return_dict:
                    return (sample,)

                return UNet2DConditionOutput(sample=sample)

    return PatchedUnet


def patch_transformer(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class PatchedPixArtTransformer2DModel(block_class):
        def forward(
                self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                timestep: Optional[torch.LongTensor] = None,
                added_cond_kwargs: Dict[str, torch.Tensor] = None,
                cross_attention_kwargs: Dict[str, Any] = None,
                attention_mask: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                return_dict: bool = True,
        ):
            skip_this_step = self._cache_bus.skip_this_step
            if skip_this_step and self._cache_bus.prev_epsilon[-1] is not None:
                self._cache_bus.last_skip_step = self._cache_bus.step
                self._cache_bus.skipping_path.append(self._cache_bus.step)

                output = self._cache_bus.prev_epsilon[-1].clone()
                for i in range(1):  # assume this is second order
                    self._cache_bus.prev_epsilon[i] = self._cache_bus.prev_epsilon[i + 1]
                self._cache_bus.prev_epsilon[-1] = output
                self._cache_bus.step += 1

                return UNet2DConditionOutput(output)

            if attention_mask is not None and attention_mask.ndim == 2:
                attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            # convert encoder_attention_mask to a bias the same way we do for attention_mask
            if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
                encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

            # 1. Input
            batch_size = hidden_states.shape[0]
            height, width = (
                hidden_states.shape[-2] // self.config.patch_size,
                hidden_states.shape[-1] // self.config.patch_size,
            )
            hidden_states = self.pos_embed(hidden_states)

            timestep, embedded_timestep = self.adaln_single(
                timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )

            if self.caption_projection is not None:
                encoder_hidden_states = self.caption_projection(encoder_hidden_states)
                encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

            # 2. Blocks
            for block in self.transformer_blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        attention_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        timestep,
                        cross_attention_kwargs,
                        None,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states = block(
                        hidden_states,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        timestep=timestep,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=None,
                    )

            # 3. Output
            shift, scale = (
                    self.scale_shift_table[None] + embedded_timestep[:, None].to(self.scale_shift_table.device)
            ).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states)
            # Modulation
            hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.squeeze(1)

            # unpatchify
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.config.patch_size, self.config.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.config.patch_size, width * self.config.patch_size)
            )

            # clone the sample to cache for skipping step
            for i in range(1):
                self._cache_bus.prev_epsilon[i] = self._cache_bus.prev_epsilon[i + 1]
            self._cache_bus.prev_epsilon[-1] = output
            self._cache_bus.step += 1

            if not return_dict:
                return (output,)

            return Transformer2DModelOutput(sample=output)


    class PatchedFluxTransformer2DModel(block_class):
        def forward(
                self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor = None,
                pooled_projections: torch.Tensor = None,
                timestep: torch.LongTensor = None,
                img_ids: torch.Tensor = None,
                txt_ids: torch.Tensor = None,
                guidance: torch.Tensor = None,
                joint_attention_kwargs: Optional[Dict[str, Any]] = None,
                controlnet_block_samples=None,
                controlnet_single_block_samples=None,
                return_dict: bool = True,
                controlnet_blocks_repeat: bool = False,
        ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

            skip_this_step = self._cache_bus.skip_this_step
            if len(self._cache_bus.prev_f) == 2:
                self._cache_bus.prev_f = [None, None, None]

            if skip_this_step and self._cache_bus.prev_f[-1] is not None:
                self._cache_bus.last_skip_step = self._cache_bus.step

                output = self._cache_bus.prev_f[-1].clone()
                for i in range(2):
                    self._cache_bus.prev_f[i] = self._cache_bus.prev_f[i + 1]
                self._cache_bus.prev_f[-1] = output

                self._cache_bus.step += 1

                return Transformer2DModelOutput(output)

            else:
                if joint_attention_kwargs is not None:
                    joint_attention_kwargs = joint_attention_kwargs.copy()
                    lora_scale = joint_attention_kwargs.pop("scale", 1.0)
                else:
                    lora_scale = 1.0

                if USE_PEFT_BACKEND:
                    # weight the lora layers by setting `lora_scale` for each PEFT layer
                    scale_lora_layers(self, lora_scale)
                else:
                    if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                        import warnings
                        warnings.warn(
                            "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective.",
                            UserWarning,
                            stacklevel=2
                        )

                hidden_states = self.x_embedder(hidden_states)

                timestep = timestep.to(hidden_states.dtype) * 1000
                if guidance is not None:
                    guidance = guidance.to(hidden_states.dtype) * 1000
                else:
                    guidance = None

                temb = (
                    self.time_text_embed(timestep, pooled_projections)
                    if guidance is None
                    else self.time_text_embed(timestep, guidance, pooled_projections)
                )
                encoder_hidden_states = self.context_embedder(encoder_hidden_states)

                if txt_ids.ndim == 3:
                    import warnings
                    warnings.warn(
                        "Passing `txt_ids` 3d torch.Tensor is deprecated. "
                        "Please remove the batch dimension and pass it as a 2d torch Tensor",
                        DeprecationWarning,
                        stacklevel=2
                    )
                    txt_ids = txt_ids[0]
                if img_ids.ndim == 3:
                    import warnings
                    warnings.warn(
                        "Passing `img_ids` 3d torch.Tensor is deprecated. "
                        "Please remove the batch dimension and pass it as a 2d torch Tensor",
                        DeprecationWarning,
                        stacklevel=2
                    )
                    img_ids = img_ids[0]

                ids = torch.cat((txt_ids, img_ids), dim=0)
                image_rotary_emb = self.pos_embed(ids)

                if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
                    ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
                    ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
                    joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

                encoder_len = encoder_hidden_states.shape[1]
                image_len = hidden_states.shape[1]
                pe = None


                for index_block, block in enumerate(self.transformer_blocks):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module, return_dict=None):
                            def custom_forward(*inputs):
                                if return_dict is not None:
                                    return module(*inputs, return_dict=return_dict)
                                else:
                                    return module(*inputs)

                            return custom_forward

                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                        encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )

                    else:

                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )

                    # controlnet residual
                    if controlnet_block_samples is not None:
                        interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        # For Xlabs ControlNet.
                        if controlnet_blocks_repeat:
                            hidden_states = (
                                    hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                            )
                        else:
                            hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
                # For single_transformer_blocks, we use the same image_rotary_emb as transformer_blocks
                # The original FLUX implementation doesn't concatenate hidden_states for single_transformer_blocks
                # Instead, it processes them separately with the same rotary embeddings
                for index_block, block in enumerate(self.single_transformer_blocks):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module, return_dict=None):
                            def custom_forward(*inputs):
                                if return_dict is not None:
                                    return module(*inputs, return_dict=return_dict)
                                else:
                                    return module(*inputs)

                            return custom_forward

                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                        encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )

                    else:

                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )

                    # controlnet residual
                    if controlnet_single_block_samples is not None:
                        interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        hidden_states[:, encoder_hidden_states.shape[1]:, ...] = (
                                hidden_states[:, encoder_hidden_states.shape[1]:, ...]
                                + controlnet_single_block_samples[index_block // interval_control]
                        )

                hidden_states = self.norm_out(hidden_states, temb)
                output = self.proj_out(hidden_states)

                # clone the output to cache for skipping step
                for i in range(2):
                    self._cache_bus.prev_f[i] = self._cache_bus.prev_f[i + 1]
                self._cache_bus.prev_f[-1] = output
                self._cache_bus.step += 1

                if USE_PEFT_BACKEND:
                    # remove `lora_scale` from each PEFT layer
                    unscale_lora_layers(self, lora_scale)

                if not return_dict:
                    return (output,)

                return Transformer2DModelOutput(sample=output)

    if block_class.__name__ == "FluxTransformer2DModel":
        return PatchedFluxTransformer2DModel
    elif block_class.__name__ == "PixArtTransformer2DModel":
        return PatchedPixArtTransformer2DModel
    else:
        raise exceptions.SADAUnsupportedError(
            f"Unsupported transformer model class '{block_class.__name__}' for SADA acceleration. "
            f"Only FluxTransformer2DModel and PixArtTransformer2DModel are supported."
        )