import typing
from diffusers import DiffusionPipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
import torch
import numpy as np
from contextlib import contextmanager


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _create_teacache_forward(num_inference_steps: int, rel_l1_thresh: float):
    """
    Create a TeaCache-enabled forward function with closure-based storage for settings.

    :param num_inference_steps: Number of inference steps for the pipeline
    :param rel_l1_thresh: Threshold for relative L1 distance (higher values = more speedup)
           0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup
    :return: TeaCache-enabled forward function for :py:class:`FluxTransformer2DModel`
    """
    # Variables captured by closure
    cnt = 0
    accumulated_rel_l1_distance = 0
    previous_modulated_input = None
    previous_residual = None
    
    def teacache_forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            img_ids: torch.Tensor = None,
            txt_ids: torch.Tensor = None,
            guidance: torch.Tensor = None,
            joint_attention_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
            controlnet_block_samples=None,
            controlnet_single_block_samples=None,
            return_dict: bool = True,
            controlnet_blocks_repeat: bool = False,
    ) -> typing.Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The :py:class:`FluxTransformer2DModel` forward method with TeaCache optimization.
        
        :param hidden_states: Input hidden_states of shape (batch size, channel, height, width)
        :param encoder_hidden_states: Conditional embeddings (embeddings computed from the input 
            conditions such as prompts) to use, shape (batch size, sequence_len, embed_dims)
        :param pooled_projections: Embeddings projected from the embeddings of input conditions,
            shape (batch_size, projection_dim)
        :param timestep: Used to indicate denoising step
        :param img_ids: Image IDs
        :param txt_ids: Text IDs
        :param guidance: Guidance tensor
        :param joint_attention_kwargs: A kwargs dictionary passed to the AttentionProcessor
        :param controlnet_block_samples: A list of tensors added to the residuals of transformer blocks
        :param controlnet_single_block_samples: A list of tensors for controlnet single block samples
        :param return_dict: Whether to return a :py:class:`Transformer2DModelOutput` instead of a plain tuple,
            defaults to True
        :param controlnet_blocks_repeat: Whether to repeat controlnet blocks, defaults to False
        :return: If return_dict is True, a :py:class:`Transformer2DModelOutput` is returned, 
            otherwise a tuple where the first element is the sample tensor
        """
        nonlocal cnt, accumulated_rel_l1_distance, previous_modulated_input, previous_residual
        
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
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
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
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        # TeaCache calculation
        inp = hidden_states.clone()
        temb_ = temb.clone()
        modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)
        
        if cnt == 0 or cnt == num_inference_steps - 1:
            should_calc = True
            accumulated_rel_l1_distance = 0
        else:
            coefficients = [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
            rescale_func = np.poly1d(coefficients)
            relative_diff = ((modulated_inp - previous_modulated_input).abs().mean() / 
                            previous_modulated_input.abs().mean()).cpu().item()
            accumulated_rel_l1_distance += rescale_func(relative_diff)
            
            if accumulated_rel_l1_distance < rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                accumulated_rel_l1_distance = 0
                
        previous_modulated_input = modulated_inp
        cnt += 1
        if cnt == num_inference_steps:
            cnt = 0

        # TeaCache skipping logic
        if not should_calc:
            hidden_states += previous_residual
        else:
            ori_hidden_states = hidden_states.clone()
            for index_block, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: typing.Dict[str, typing.Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
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

                    ckpt_kwargs: typing.Dict[str, typing.Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
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
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                            hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                            + controlnet_single_block_samples[index_block // interval_control]
                    )

            previous_residual = hidden_states - ori_hidden_states
        
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
    
    return teacache_forward


@contextmanager
def teacache_context(pipeline: DiffusionPipeline, num_inference_steps: int, rel_l1_thresh: float = 0.6, enable: bool = True):
    """
    Context manager for enabling TeaCache optimization on a FLUX pipeline.
    
    :param enable:
    :param pipeline: The FLUX :py:class:`DiffusionPipeline`
    :param num_inference_steps: Number of inference steps for the pipeline
    :param rel_l1_thresh: Threshold for relative L1 distance, higher values mean more speedup.
           Defaults to 0.6 (2.0x speedup). 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 
           0.6 for 2.0x speedup, 0.8 for 2.25x speedup
    :param enable: Whether to enable TeaCache optimization, defaults to True
    :yield: The pipeline with TeaCache enabled
    
    Example:
        .. code-block:: python
            
            with teacache_context(pipeline, num_inference_steps=28, rel_l1_thresh=0.6) as teacache_pipeline:
                image = teacache_pipeline(
                    prompt="An image of a squirrel in Picasso style",
                    num_inference_steps=28
                ).images[0]
    """

    if not enable:
        yield pipeline
        return

    # Save the original forward method
    original_forward = pipeline.transformer.forward
    
    # Create and apply the TeaCache-enabled forward function
    teacache_forward_fn = _create_teacache_forward(num_inference_steps, rel_l1_thresh)
    pipeline.transformer.forward = teacache_forward_fn.__get__(pipeline.transformer, pipeline.transformer.__class__)
    
    try:
        # Yield the modified pipeline
        yield pipeline
    finally:
        # Restore the original forward method
        pipeline.transformer.forward = original_forward