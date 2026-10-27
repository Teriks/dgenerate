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

import os
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.loaders import IPAdapterMixin
import huggingface_hub
import dgenerate.messages as _messages

from .image_encoder import HunyuanDiTCLIPImageEncoder, load_hunyuan_dit_clip_image_encoder


class HunyuanDiTIPAdapterAttnProcessor2_0(nn.Module):
    """
    Cross-attention processor for HunyuanDiT with IP-Adapter support.
    """

    def __init__(
        self,
        hidden_size: int = 1408,
        cross_attention_dim: int = 1024,
        num_heads: int = 16,
        head_dim: int = 88,
        scale: float = 1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale

        self.kv_proj_ip_adapter = nn.Linear(cross_attention_dim, 2 * hidden_size, bias=True)
        self.k_norm_ip_adapter = nn.LayerNorm(head_dim, elementwise_affine=True, eps=1e-6)
        self.image_prompt_embeds: torch.Tensor | None = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Hunyuan-DiT IP-Adapter cross-attention (Tencent ``FlashCrossMHAModified``):
        ``context = text_context * t_scale + ip_context * i_scale`` then one ``out_proj``.
        """
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]

        sequence_length = (
            hidden_states.shape[1]
            if encoder_hidden_states is None
            else encoder_hidden_states.shape[1]
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        text_context = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        text_context = text_context.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        text_context = text_context.to(query.dtype)

        t_scale = 1.0
        i_scale = self.scale
        combined = text_context * t_scale

        if self.image_prompt_embeds is not None and i_scale != 0.0:
            z = self.image_prompt_embeds.to(
                device=hidden_states.device, dtype=self.kv_proj_ip_adapter.weight.dtype)
            if z.shape[0] != batch_size:
                if batch_size % z.shape[0] != 0:
                    raise ValueError(
                        f'Hunyuan-DiT IP-Adapter embedding batch {z.shape[0]} '
                        f'does not divide the attention batch {batch_size}.')
                # Duplicate each row. A classifier-free pair ``[zero, image]``
                # grows to ``[zero, zero, image, image]``, which is the
                # pipeline order. Tiling the whole pair would swap those slots.
                z = z.repeat_interleave(batch_size // z.shape[0], dim=0)

            s3 = z.shape[1]
            kv_2 = self.kv_proj_ip_adapter(z).view(batch_size, s3, 2, self.num_heads, self.head_dim)
            k_2, v_2 = kv_2.unbind(dim=2)
            k_2 = self.k_norm_ip_adapter(k_2).transpose(1, 2)
            v_2 = v_2.transpose(1, 2)
            ip_context = F.scaled_dot_product_attention(query, k_2, v_2, dropout_p=0.0, is_causal=False)
            ip_context = ip_context.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ip_context = ip_context.to(query.dtype)
            combined = text_context * t_scale + ip_context * i_scale

        hidden_states = attn.to_out[0](combined)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def _unwrap_ip_adapter_load_arg(
        value: typing.Any,
        arg_name: str,
        *,
        allow_none: bool = True) -> typing.Any:
    """
    Diffusers-style ``load_ip_adapter`` passes one-element lists; Hunyuan-DiT uses scalars.
    """
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                f'Hunyuan-DiT supports a single IP-Adapter; got {len(value)} {arg_name} values.')
        value = value[0]
    if value == '.':
        return None
    if value is None and not allow_none:
        raise ValueError(f'Missing required IP-Adapter {arg_name}.')
    return value


def _normalize_ip_adapter_images(image: typing.Any) -> typing.Any:
    """
    dgenerate passes ``ip_adapter_images`` as one list per loaded IP-Adapter
    (``[[img, ...], ...]``). Hunyuan-DiT uses a single adapter and expects PIL
    images or a flat list of PIL images.
    """
    if image is None:
        return None
    if isinstance(image, (list, tuple)):
        if not image:
            return None
        if isinstance(image[0], (list, tuple)):
            if len(image) != 1:
                raise ValueError(
                    f'Hunyuan-DiT supports a single IP-Adapter; got {len(image)} adapter image groups.')
            image = image[0]
        if isinstance(image, (list, tuple)) and len(image) == 1:
            return image[0]
    return image


def _load_ipa_state_dict(model_path_or_dict, weight_name=None, subfolder=None, token=None, local_files_only=False):
    if isinstance(model_path_or_dict, dict):
        return model_path_or_dict

    model_path_or_dict = _unwrap_ip_adapter_load_arg(
        model_path_or_dict, 'model', allow_none=False)
    weight_name = _unwrap_ip_adapter_load_arg(weight_name, 'weight-name')
    subfolder = _unwrap_ip_adapter_load_arg(subfolder, 'subfolder')

    path = str(model_path_or_dict)
    if os.path.isfile(path):
        file_path = path
    elif os.path.isdir(path):
        candidates = [weight_name] if weight_name else ["ipa.pt", "ip_adapter.bin", "ip_adapter.safetensors"]
        file_path = None
        for name in candidates:
            p = os.path.join(path, name)
            if os.path.isfile(p):
                file_path = p
                break
        if file_path is None:
            for f in os.listdir(path):
                if f.endswith(".pt") or f.endswith(".bin") or f.endswith(".safetensors"):
                    file_path = os.path.join(path, f)
                    break
        if file_path is None:
            raise FileNotFoundError(f"Could not find IP-Adapter weights in directory: {path}")
    else:
        filename = weight_name if weight_name else "ipa.pt"
        try:
            file_path = huggingface_hub.hf_hub_download(
                repo_id=path,
                filename=filename,
                subfolder=subfolder,
                token=token,
                local_files_only=local_files_only,
            )
        except Exception:
            filename = "ip_adapter.bin"
            file_path = huggingface_hub.hf_hub_download(
                repo_id=path,
                filename=filename,
                subfolder=subfolder,
                token=token,
                local_files_only=local_files_only,
            )

    if file_path.endswith(".safetensors"):
        import safetensors.torch
        return safetensors.torch.load_file(file_path, device="cpu")
    else:
        return torch.load(file_path, map_location="cpu", weights_only=True)


class HunyuanDiTIPAdapterMixin(IPAdapterMixin):
    """
    IP-Adapter loading and execution mixin for HunyuanDiT pipelines.
    """

    def set_ip_adapter_scale(self, scale: float | list[float]):
        if isinstance(scale, (list, tuple)):
            scale = scale[0] if scale else 1.0
        self._ip_adapter_scale = float(scale)
        if hasattr(self, "_ip_adapter_processors"):
            for proc in self._ip_adapter_processors:
                proc.scale = self._ip_adapter_scale

    def load_ip_adapter(
        self,
        pretrained_model_name_or_path_or_dict: str | dict[str, torch.Tensor],
        subfolder: str | None = None,
        weight_name: str | None = None,
        token: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        pretrained_model_name_or_path_or_dict = _unwrap_ip_adapter_load_arg(
            pretrained_model_name_or_path_or_dict, 'model', allow_none=False)
        weight_name = _unwrap_ip_adapter_load_arg(weight_name, 'weight-name')
        subfolder = _unwrap_ip_adapter_load_arg(subfolder, 'subfolder')

        state_dict = _load_ipa_state_dict(
            pretrained_model_name_or_path_or_dict,
            weight_name=weight_name,
            subfolder=subfolder,
            token=token,
            local_files_only=local_files_only,
        )

        # Strip prefixes like 'module.'
        cleaned_sd = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            cleaned_sd[k] = v

        device = self.transformer.device
        dtype = self.transformer.dtype

        if not isinstance(getattr(self, 'image_encoder', None), HunyuanDiTCLIPImageEncoder):
            if getattr(self, 'image_encoder', None) is not None:
                _messages.debug_log(
                    'Replacing --image-encoder with Tencent Hunyuan-DiT clip_img_encoder.pt '
                    f'from "{pretrained_model_name_or_path_or_dict}".')
            # Keep the CLIP vision tower on CPU; only IP-Adapter head tensors follow the DiT.
            self.image_encoder = load_hunyuan_dit_clip_image_encoder(
                str(pretrained_model_name_or_path_or_dict),
                device='cpu',
                dtype=dtype,
                token=token,
                local_files_only=local_files_only,
            )

        # Setup projection layers
        self.ip_adapter_clip_len_trans_block = nn.Linear(1024, 8 * 1024).to(device=device, dtype=dtype)
        self.ip_adapter_img_norm_cond = nn.LayerNorm(1024).to(device=device, dtype=dtype)

        if "ip_adapter_clip_len_trans_block.weight" in cleaned_sd:
            self.ip_adapter_clip_len_trans_block.weight.data.copy_(
                cleaned_sd["ip_adapter_clip_len_trans_block.weight"].to(device=device, dtype=dtype)
            )
            self.ip_adapter_clip_len_trans_block.bias.data.copy_(
                cleaned_sd["ip_adapter_clip_len_trans_block.bias"].to(device=device, dtype=dtype)
            )

        if "ip_adapter_img_norm_cond.weight" in cleaned_sd:
            self.ip_adapter_img_norm_cond.weight.data.copy_(
                cleaned_sd["ip_adapter_img_norm_cond.weight"].to(device=device, dtype=dtype)
            )
            self.ip_adapter_img_norm_cond.bias.data.copy_(
                cleaned_sd["ip_adapter_img_norm_cond.bias"].to(device=device, dtype=dtype)
            )

        self._ip_adapter_processors = []

        # Attach processors to blocks
        for i, block in enumerate(self.transformer.blocks):
            hidden_size = block.attn2.qdim if hasattr(block.attn2, "qdim") else self.transformer.config.hidden_size
            cross_attention_dim = block.attn2.kdim if hasattr(block.attn2, "kdim") else 1024
            num_heads = block.attn2.num_heads if hasattr(block.attn2, "num_heads") else self.transformer.config.num_attention_heads
            head_dim = hidden_size // num_heads

            proc = HunyuanDiTIPAdapterAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                scale=getattr(self, "_ip_adapter_scale", 1.0),
            ).to(device=device, dtype=dtype)

            # Load block weights if available
            w_prefix = f"blocks.{i}.attn2.kv_proj_ip_adapter"
            norm_prefix = f"blocks.{i}.attn2.k_norm_ip_adapter"
            if f"{w_prefix}.weight" in cleaned_sd:
                proc.kv_proj_ip_adapter.weight.data.copy_(
                    cleaned_sd[f"{w_prefix}.weight"].to(device=device, dtype=dtype)
                )
                proc.kv_proj_ip_adapter.bias.data.copy_(
                    cleaned_sd[f"{w_prefix}.bias"].to(device=device, dtype=dtype)
                )
            if f"{norm_prefix}.weight" in cleaned_sd:
                proc.k_norm_ip_adapter.weight.data.copy_(
                    cleaned_sd[f"{norm_prefix}.weight"].to(device=device, dtype=dtype)
                )
                proc.k_norm_ip_adapter.bias.data.copy_(
                    cleaned_sd[f"{norm_prefix}.bias"].to(device=device, dtype=dtype)
                )

            block.attn2.processor = proc
            self._ip_adapter_processors.append(proc)

    def _hydit_ip_adapter_runtime_device(self) -> torch.device:
        device = self.transformer.device
        if device.type == 'meta':
            device = getattr(self, '_dgenerate_offload_device', device)
        return device

    def _sync_ip_adapter_head_modules(self, device: torch.device, dtype: torch.dtype) -> None:
        head = self.ip_adapter_clip_len_trans_block
        if head.weight.device != device or head.weight.dtype != dtype:
            self.ip_adapter_clip_len_trans_block.to(device=device, dtype=dtype)
            self.ip_adapter_img_norm_cond.to(device=device, dtype=dtype)

    def encode_ip_adapter_clip_embedding(
            self, image, device=None, dtype=None) -> torch.Tensor:
        """
        CLIP image embedding (B, 1024), L2-normalized — before IPA projection.
        """
        if dtype is None:
            dtype = self.transformer.dtype

        image = _normalize_ip_adapter_images(image)

        if hasattr(self, "image_encoder") and self.image_encoder is not None:
            if isinstance(self.image_encoder, HunyuanDiTCLIPImageEncoder):
                embeds = self.image_encoder.encode_pil_images(image)
                if device is None:
                    device = embeds.device
            else:
                encoder_device = next(self.image_encoder.parameters()).device
                if device is None:
                    device = encoder_device
                if hasattr(self, "feature_extractor") and self.feature_extractor is not None:
                    inputs = self.feature_extractor(images=image, return_tensors="pt")
                    pixel_values = inputs.pixel_values.to(
                        device=encoder_device, dtype=self.image_encoder.dtype)
                else:
                    from torchvision import transforms as T
                    transform = T.Compose([
                        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                        T.ToTensor(),
                        T.Normalize(
                            [0.48145466, 0.4578275, 0.40821073],
                            [0.26862954, 0.26130258, 0.27577711]),
                    ])
                    if isinstance(image, (list, tuple)):
                        pixel_values = torch.stack([transform(img.convert("RGB")) for img in image]).to(
                            device=encoder_device, dtype=self.image_encoder.dtype)
                    else:
                        pixel_values = transform(image.convert("RGB")).unsqueeze(0).to(
                            device=encoder_device, dtype=self.image_encoder.dtype)

                with torch.no_grad():
                    out = self.image_encoder(pixel_values)
                    if hasattr(out, "image_embeds"):
                        embeds = out.image_embeds
                    elif hasattr(out, "last_hidden_state"):
                        embeds = out.last_hidden_state[:, 0]
                    else:
                        embeds = out[0]
                    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        else:
            raise RuntimeError("Missing image_encoder required for HunyuanDiT IP-Adapter.")

        run_device = device if device is not None else self._hydit_ip_adapter_runtime_device()
        return embeds.to(device=run_device, dtype=dtype)

    def project_ip_adapter_clip_embedding(self, embeds: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "ip_adapter_clip_len_trans_block"):
            raise RuntimeError("IP-Adapter weights must be loaded before encoding images.")

        dtype = embeds.dtype
        run_device = embeds.device
        self._sync_ip_adapter_head_modules(run_device, dtype)

        z = self.ip_adapter_clip_len_trans_block(embeds)
        z = z.reshape(embeds.shape[0], -1, 8).transpose(2, 1).contiguous()
        z = self.ip_adapter_img_norm_cond(z)
        return z

    def encode_ip_adapter_image(self, image, device=None, dtype=None) -> torch.Tensor:
        """
        Encode an image into IP-Adapter conditioning tokens (B, 8, 1024).
        """
        embeds = self.encode_ip_adapter_clip_embedding(image, device=device, dtype=dtype)
        return self.project_ip_adapter_clip_embedding(embeds)
