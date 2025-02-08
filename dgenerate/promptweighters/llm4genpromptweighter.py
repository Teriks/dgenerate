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

import gc
import inspect
import re
import typing

import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.promptweighters.exceptions as _exceptions
import dgenerate.promptweighters.promptweighter as _promptweighter
import dgenerate.memory as _memory
import dgenerate.textprocessing as _textprocessing
import dgenerate.pipelinewrapper.hfutil as _hfutil
import os

import torch
import torch.nn
import huggingface_hub
import transformers


class T5EncoderWithProjection(transformers.T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.t5_encoder = transformers.T5EncoderModel(config)
        self.projection = torch.nn.Linear(config.d_model, config.d_model, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, **input_args):
        hidden_states = self.t5_encoder(**input_args).last_hidden_state
        hidden_states = hidden_states[:, 0, :]
        batch_embeddings = self.projection(hidden_states)
        return batch_embeddings


class RankGenEncoder:
    def __init__(self, model_path, model_size=None, cache_dir=None, local_files_only=False, use_auth_token=None):
        assert model_path in [
            "kalpeshk2011/rankgen-t5-xl-all",
            "kalpeshk2011/rankgen-t5-xl-pg19",
            "kalpeshk2011/rankgen-t5-base-all",
            "kalpeshk2011/rankgen-t5-large-all"
        ]

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_size is None:
            if "t5-large" in model_path or "t5_large" in model_path:
                self.model_size = "large"
            elif "t5-xl" in model_path or "t5_xl" in model_path:
                self.model_size = "xl"
            else:
                self.model_size = "base"
        else:
            self.model_size = model_size

        self.tokenizer = transformers.T5Tokenizer.from_pretrained(
            f"google/t5-v1_1-{self.model_size}",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token
        )

        self.model = T5EncoderWithProjection.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token
        )

    def to(self, *args, **kwargs):
        return self.model.to(*args, **kwargs)

    def eval(self):
        return self.model.eval()

    def encode(self, inputs, vectors_type="prefix", max_length=256):
        tokenizer = self.tokenizer
        if isinstance(inputs, str):
            inputs = [inputs]

        if vectors_type == 'prefix':
            inputs = ['pre ' + input for input in inputs]
        else:
            inputs = ['suffi ' + input for input in inputs]

        tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, max_length=max_length)
        length = tokenized_inputs['input_ids'].shape[1]
        if length > max_length:
            tokenized_inputs['input_ids'] = tokenized_inputs['input_ids'][:, :max_length]
            tokenized_inputs['attention_mask'] = tokenized_inputs['attention_mask'][:, :max_length]
        else:
            padding_length = max_length - length
            padding_tokens = torch.zeros(tokenized_inputs['input_ids'].shape[0], padding_length,
                                         dtype=tokenized_inputs['input_ids'].dtype)
            tokenized_inputs['input_ids'] = torch.cat([tokenized_inputs['input_ids'], padding_tokens], dim=1)
            tokenized_inputs['attention_mask'] = torch.cat([tokenized_inputs['attention_mask'], padding_tokens], dim=1)

        tokenized_inputs = tokenized_inputs.to(self.device)
        with torch.no_grad():
            batch_embeddings = self.model.t5_encoder(**tokenized_inputs).last_hidden_state
        return batch_embeddings


class QuickGELU(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class CrossFusion(torch.nn.Module):
    def __init__(self, llama_dim, dim, heads):
        super(CrossFusion, self).__init__()
        self.num_heads = heads
        self.dim = dim
        self.head_dim = dim // heads

        assert self.head_dim * heads == dim, "dim must be divisible"

        self.scale = self.head_dim ** -0.5
        self.llm_proj = torch.nn.Linear(llama_dim, dim)

        self.q_proj = torch.nn.Linear(dim, dim)
        self.k_proj = torch.nn.Linear(dim, dim)
        self.v_proj = torch.nn.Linear(dim, dim)
        self.out_proj = torch.nn.Linear(dim, dim)

        self.q_norm = torch.nn.LayerNorm(dim)
        self.kv_norm = torch.nn.LayerNorm(dim)
        self.norm = torch.nn.LayerNorm(dim)

        self.FFN = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            QuickGELU(),
            torch.nn.Linear(dim * 4, dim)
        )
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.q_proj.weight, std=0.02)
        torch.nn.init.trunc_normal_(self.k_proj.weight, std=0.02)
        torch.nn.init.trunc_normal_(self.v_proj.weight, std=0.02)
        torch.nn.init.trunc_normal_(self.out_proj.weight, std=0.02)

    def forward(self, clip_embed, llm_embed):
        B, _, _ = llm_embed.shape
        llm_embed = self.llm_proj(llm_embed)
        clip_embed_norm = self.q_norm(clip_embed)
        llm_embed_norm = self.kv_norm(llm_embed)
        query = self.q_proj(llm_embed_norm).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(clip_embed_norm).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(clip_embed_norm).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attention_weights = (query @ key.transpose(-2, -1)) * self.scale
        attention_weights = attention_weights.softmax(dim=-1)
        out = (attention_weights @ value).transpose(1, 2).reshape(B, -1, self.dim)
        llm_embed = self.out_proj(out) + llm_embed
        llm_embed = self.FFN(self.norm(llm_embed)) + llm_embed
        concat_embed = torch.cat((clip_embed, llm_embed), dim=1)
        return concat_embed


class LLMFusionModule(torch.nn.Module):
    def __init__(self, clip_dim, llm_dim, num_heads):
        super(LLMFusionModule, self).__init__()
        self.CrossFusionModule = torch.nn.ModuleList(
            [CrossFusion(llm_dim, clip_dim, num_heads) for _ in range(1)]
        )

    def forward(self, clip_text, llm_text):
        for module in self.CrossFusionModule:
            clip_text = module(clip_text, llm_text)
        return clip_text


class LLM4GENPromptWeighter(_promptweighter.PromptWeighter):
    r"""
    LLM4GEN prompt weighter specifically for Stable Diffusion 1.5, See: https://github.com/YUHANG-Ma/LLM4GEN

    This prompt weighter requires the use of --unet Shui-VL/LLM4GEN-models;subfolder=unet in order to function properly.

    Stable Diffusion 2.* is not supported.

    This prompt weighter supports the model types:

    NOWRAP!
    --model-type torch
    --model-type torch-pix2pix
    --model-type torch-upscaler-x4

    The "encoder" argument specifies the T5 encoder model (Rank generation model).

    The encoder specified must be one of:

    NOWRAP!
    * kalpeshk2011/rankgen-t5-xl-all
    * kalpeshk2011/rankgen-t5-xl-pg19
    * kalpeshk2011/rankgen-t5-base-all
    * kalpeshk2011/rankgen-t5-large-all

    The "projector" argument specifies a Hugging Face repo or file path to the LLM4GEN projector (CAM) model.

    The "projector_revision" argument specifies the revision of the Hugging Face projector repository, for example "main".

    The "projector_subfolder" argument specifies the subfolder for the projector file in a Hugging Face repository.

    The "projector_weight_name" argument specifies the weight name of the projector file in a Hugging Face repository.

    The "local_files_only" argument specifies that no attempt should be made to download
    models from the internet, only look for cached models on disk.

    The "token" argument allows you to explicitly specify a Hugging Face auth tokenfor downloads.

    NOWRAP!
    @misc{liu2024llm4genleveragingsemanticrepresentation,
      title={LLM4GEN: Leveraging Semantic Representation of LLMs for Text-to-Image Generation},
      author={Mushui Liu and Yuhang Ma and Xinfeng Zhang and Yang Zhen and Zeng Zhao and Zhipeng Hu and Bai Liu and Changjie Fan},
      year={2024},
    }
    """

    NAMES = ['llm4gen']

    def __init__(self,
                 encoder: str = "kalpeshk2011/rankgen-t5-xl-all",
                 projector: str = 'Shui-VL/LLM4GEN-models',
                 projector_subfolder: typing.Optional[str] = None,
                 projector_revision: typing.Optional[str] = None,
                 projector_weight_name: str = 'projector.pth',
                 local_files_only: bool = False,
                 token: typing.Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)

        supported = {
            _enums.ModelType.TORCH,
            _enums.ModelType.TORCH_PIX2PIX,
            _enums.ModelType.TORCH_UPSCALER_X4
        }

        if self.model_type not in supported:
            raise _exceptions.PromptWeightingUnsupported(
                f'Prompt weighting not supported for --model-type: {_enums.get_model_type_string(self.model_type)}')

        valid_encoders = [
            'kalpeshk2011/rankgen-t5-xl-all',
            'kalpeshk2011/rankgen-t5-xl-pg19',
            'kalpeshk2011/rankgen-t5-base-all',
            'kalpeshk2011/rankgen-t5-large-all'
        ]

        if encoder not in valid_encoders:
            raise _exceptions.PromptWeightingUnsupported(
                f'llm4gen prompt-weighter "encoder" argument must be one of: '
                f'{_textprocessing.oxford_comma(valid_encoders, "or")}')

        self.t5_model = RankGenEncoder(
            encoder,
            local_files_only=local_files_only,
            use_auth_token=token
        )

        self.llm_projector_path = self._get_projector_path(
            projector,
            weight_name=projector_weight_name,
            subfolder=projector_subfolder,
            revision=projector_revision,
            local_files_only=local_files_only,
            use_auth_token=token
        )

        self.llm_projector = LLMFusionModule(768, 2048, 8)
        self.llm_projector.load_state_dict(torch.load(self.llm_projector_path))

        self._tensors = list()

    def _get_projector_path(self,
                            model,
                            weight_name,
                            subfolder,
                            revision,
                            local_files_only: bool = False,
                            use_auth_token: str | None = None):
        try:
            if _hfutil.is_single_file_model_load(model):
                if os.path.exists(model):
                    return model
                else:
                    if local_files_only:
                        raise self.argument_error(f'Could not find projector model: {model}')
                    return _hfutil.download_non_hf_model(model)
            else:
                return huggingface_hub.hf_hub_download(
                    model,
                    filename=weight_name,
                    subfolder=subfolder,
                    token=use_auth_token,
                    revision=revision,
                    local_files_only=local_files_only)
        except Exception as e:
            raise self.argument_error(
                f'Could not find projector model: {e}')

    @torch.inference_mode()
    def translate_to_embeds(self,
                            pipeline,
                            device: str,
                            args: dict[str, any]):

        # we are responsible for generating these arguments
        # if they exist already then we cannot do our job

        forbidden_call_args = {
            'prompt_embeds',
            'negative_prompt_embeds'
        }

        if any(a in forbidden_call_args for a in args.keys()):
            raise _exceptions.PromptWeightingUnsupported(
                f'Prompt weighting not supported for --model-type: {_enums.get_model_type_string(self.model_type)}, '
                f'in mode: {_enums.get_pipeline_type_string(self.pipeline_type)}')

        if 'clip_skip' in args:
            raise _exceptions.PromptWeightingUnsupported(
                f'llm4gen prompt-weighter does not support --clip-skips.')

        pipeline_sig = set(inspect.signature(pipeline.__call__).parameters.keys())

        if 'prompt_embeds' not in pipeline_sig:
            # pipeline does not support passing prompt embeddings directly

            raise _exceptions.PromptWeightingUnsupported(
                f'Prompt weighting not supported for --model-type: {_enums.get_model_type_string(self.model_type)}, '
                f'in mode: {_enums.get_pipeline_type_string(self.pipeline_type)}')

        if not pipeline.__class__.__name__.startswith('StableDiffusion'):
            raise _exceptions.PromptWeightingUnsupported(
                f'Prompt weighting not supported for --model-type: {_enums.get_model_type_string(self.model_type)}')

        output = dict(args)

        positive = args.get('prompt')
        negative = args.get('negative_prompt')

        prompt_args = re.compile(r'^(prompt|negative_prompt)(_\d+)?$')

        for name in args.keys():
            if prompt_args.match(name):
                output.pop(name)

        positive = positive if positive else ""
        negative = negative if negative else ""

        try:
            self.llm_projector.to(device).eval()
            self.t5_model.to(device).eval()

            llm_embed = self.t5_model.encode(positive)

            if hasattr(pipeline, 'maybe_convert_prompt'):
                if positive:
                    positive = pipeline.maybe_convert_prompt(positive, tokenizer=pipeline.tokenizer)

            input_ids = pipeline.tokenizer(
                positive, return_tensors="pt", truncation=True, padding="max_length", max_length=77).input_ids

            input_ids = input_ids.to(device)
            self._tensors.append(input_ids)
            clip_embed = pipeline.text_encoder(input_ids, return_dict=False)[0]
            pos_conditioning = self.llm_projector(clip_embed, llm_embed)

            neg_llm_embed = self.t5_model.encode(negative, max_length=llm_embed.shape[1])

            if hasattr(pipeline, 'maybe_convert_prompt'):
                if negative:
                    negative = pipeline.maybe_convert_prompt(negative, tokenizer=pipeline.tokenizer)

            negative_ids = pipeline.tokenizer(
                negative, truncation=True, return_tensors="pt", padding="max_length", max_length=77).input_ids

            negative_ids = negative_ids.to(device)
            self._tensors.append(negative_ids)

            neg_clip_embeds = pipeline.text_encoder(negative_ids, return_dict=False)[0]
            neg_conditioning = self.llm_projector(neg_clip_embeds, neg_llm_embed)
        finally:
            self.t5_model.to('cpu')
            self.llm_projector.cpu()

        self._tensors.append(pos_conditioning)
        self._tensors.append(neg_conditioning)

        output.update({
            'prompt_embeds': pos_conditioning,
            'negative_prompt_embeds': neg_conditioning,
        })

        return output

    def cleanup(self):
        for tensor in self._tensors:
            tensor.to('cpu')
            del tensor
        self._tensors.clear()
        gc.collect()
        _memory.torch_gc()
