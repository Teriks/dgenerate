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

import diffusers

import dgenerate.hfhub as _hfhub
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.enums as _enums
from dgenerate.pipelinewrapper.uris import get_quantizer_uri_class as _get_quantizer_uri_class
import dgenerate.promptweighters.exceptions as _exceptions
import dgenerate.promptweighters.promptweighter as _promptweighter
import dgenerate.memory as _memory
import dgenerate.textprocessing as _textprocessing
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
    def __init__(self,
                 model_path: str,
                 model_size: str | None = None,
                 cache_dir: str | None = None,
                 local_files_only: bool = False,
                 use_auth_token: str | None = None,
                 torch_dtype: torch.dtype = torch.float32,
                 quantization_config = None,
                 device_map: str | None ="auto"):
        assert model_path in [
            "kalpeshk2011/rankgen-t5-xl-all",
            "kalpeshk2011/rankgen-t5-xl-pg19",
            'kalpeshk2011/rankgen-t5-base-all',
            'kalpeshk2011/rankgen-t5-large-all'
        ]

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
            use_auth_token=use_auth_token,
            torch_dtype=torch_dtype,
            device_map=device_map

        )

        self.model = T5EncoderWithProjection.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config
        )

    def to(self, device, **kwargs):
        if not getattr(self.model, "is_loaded_in_8bit", False):
            return self.model.to(device, **kwargs)
        return self.model

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

        tokenized_inputs = tokenized_inputs.to(self.model.device)
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


def _cam_size_estimate_32bit(llama_dim, dim, heads):
    # Parameters for each layer
    llm_proj_params = llama_dim * dim
    q_proj_params = k_proj_params = v_proj_params = out_proj_params = dim * dim
    ffn_params1 = dim * (4 * dim)
    ffn_params2 = (4 * dim) * dim
    norm_params = 3 * (2 * dim)  # For q_norm, kv_norm, norm

    # Total parameters
    total_params = (llm_proj_params +
                    4 * q_proj_params +
                    ffn_params1 +
                    ffn_params2 +
                    norm_params)

    # Memory size in bytes for float32 (4 bytes per parameter)
    memory_size_float32 = total_params * 4  # 4 bytes per parameter

    return memory_size_float32


class LLM4GENPromptWeighter(_promptweighter.PromptWeighter):
    r"""
    LLM4GEN prompt weighter specifically for Stable Diffusion 1.5, See: https://github.com/YUHANG-Ma/LLM4GEN

    Stable Diffusion 2.* is not supported.

    This prompt weighter supports the model types:

    NOWRAP!
    --model-type sd
    --model-type pix2pix
    --model-type upscaler-x4

    You may use the --second-prompts argument of dgenerate to pass a prompt
    explicitly to the T5 rankgen encoder, which uses the primary prompt by default otherwise.

    The "encoder" argument specifies the T5 rankgen encoder model variant.

    The encoder variant specified must be one of:

    NOWRAP!
    * base-all
    * large-all
    * xl-all
    * xl-pg19

    The "projector" argument specifies a Hugging Face repo or file path to the LLM4GEN projector (CAM) model.

    The "projector-revision" argument specifies the revision of the Hugging Face projector repository, for example "main".

    The "projector-subfolder" argument specifies the subfolder for the projector file in a Hugging Face repository.

    The "projector-weight-name" argument specifies the weight name of the projector file in a Hugging Face repository.

    The "weighter" argument can be used to specify a prompt weighter that will be used for CLIP embedding generation,
    this may be one of "sd-embed" or "compel". Weighting does not occur for the rankgen encoder, and if you do
    not pass --second-prompts to dgenerate while using this argument, the rankgen encoder will receive
    the primary prompt with all weighting syntax filtered out. This automatic filtering only occurs
    when you specify "weighter" without specifying --second-prompts to dgenerate.

    The "llm-dtype" argument specifies the precision for the rankgen encoder and llm4gen CAM projector model,
    changing this to 'float16' or 'bfloat16' will cut memory use in half at the possible cost of output quality.

    The "llm-quantizer" argument specifies the quantization backend to use when loading the rankgen encoder,
    this argument uses dgenerate --quantizer syntax.

    The "token" argument allows you to explicitly specify a Hugging Face auth token for downloads.

    NOWRAP!
    @misc{liu2024llm4genleveragingsemanticrepresentation,
      title={LLM4GEN: Leveraging Semantic Representation of LLMs for Text-to-Image Generation},
      author={Mushui Liu and Yuhang Ma and Xinfeng Zhang and Yang Zhen and Zeng Zhao and Zhipeng Hu and Bai Liu and Changjie Fan},
      year={2024},
    }
    """

    NAMES = ['llm4gen']

    _weighter: _promptweighter.PromptWeighter | None

    OPTION_ARGS = {
        'encoder': ['base-all', 'large-all', 'xl-all', 'xl-pg19'],
        'llm_dtype': ['float32', 'float16', 'bfloat16']
    }

    FILE_ARGS = {
        'projector': {'mode': 'dir'}
    }

    def __init__(self,
                 encoder: str = "xl-all",
                 projector: str = 'Shui-VL/LLM4GEN-models',
                 projector_subfolder: str | None = None,
                 projector_revision: str | None = None,
                 projector_weight_name: str = 'projector.pth',
                 weighter: str | None = None,
                 llm_dtype: str = 'float32',
                 llm_quantizer: str | None = None,
                 token: str | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        import dgenerate.promptweighters as _promptweighters

        supported = {
            _enums.ModelType.SD,
            _enums.ModelType.PIX2PIX,
            _enums.ModelType.UPSCALER_X4
        }

        encoder = encoder.lower()
        llm_dtype = llm_dtype.lower()

        if weighter:
            if not (weighter.startswith('sd-embed') or weighter.startswith('compel')):
                raise self.argument_error(
                    'Argument "weighter" must be one of: sd-embed, or compel'
                )

            if not _promptweighters.prompt_weighter_exists(weighter):
                raise self.argument_error(
                    'Argument "weighter" must be one of: sd-embed, or compel'
                )

        if llm_dtype not in {'bfloat16', 'float16', 'float32'}:
            raise self.argument_error(
                'Argument "llm-dtype" must '
                'be one of: float32, float16, or bfloat16')

        if llm_quantizer:
            try:
                self._llm_quantizer_class = _get_quantizer_uri_class(llm_quantizer)
                self._llm_quantization_config = \
                    self._llm_quantizer_class.parse(llm_quantizer).to_config(llm_dtype)
            except Exception as e:
                raise self.argument_error(f'Error loading "llm-quantizer" argument "{llm_quantizer}": {e}') from e
        else:
            self._llm_quantizer_class = None
            self._llm_quantization_config = None

        if self.model_type not in supported:
            raise _exceptions.PromptWeightingUnsupported(
                f'Prompt weighting not supported for --model-type: {_enums.get_model_type_string(self.model_type)}')

        valid_encoders = [
            'base-all',
            'large-all',
            'xl-all',
            'xl-pg19'
        ]

        if encoder not in valid_encoders:
            raise _exceptions.PromptWeightingUnsupported(
                f'Argument "encoder" must be one of: '
                f'{_textprocessing.oxford_comma(valid_encoders, "or")}')

        self.llm_projector_path = self._get_projector_path(
            projector,
            weight_name=projector_weight_name,
            subfolder=projector_subfolder,
            revision=projector_revision,
            local_files_only=self.local_files_only,
            use_auth_token=token
        )

        self._llm_dtype = _enums.get_torch_dtype(llm_dtype)

        # XL encoder around 12 gigs, Large encoder around 3.08 gigs
        rankgen_size_estimate = int(12 * 1024 ** 3) if 'xl' in encoder else int(3.08 * 1024 ** 3)

        if llm_dtype != 'float32':
            # transformers can actually load in 16 bit can cut the
            # memory overhead in half
            rankgen_size_estimate = rankgen_size_estimate // 2

        # Assume 32 bit, because it is being loaded in 32 bit and
        # then converted and we need space to do that
        cam_size_estimate = _cam_size_estimate_32bit(768, 2048 if 'xl' in encoder else 1024, 8)

        self.set_size_estimate(rankgen_size_estimate + cam_size_estimate)

        encoder_model_path = 'kalpeshk2011/rankgen-t5-' + encoder

        # the cached status should be linked to the
        # dtype of the model, so append that to the tag

        self.t5_model = self.load_object_cached(
            tag=encoder_model_path + (llm_quantizer if llm_quantizer else '') + llm_dtype,
            estimated_size=rankgen_size_estimate,
            method=lambda: RankGenEncoder(
                encoder_model_path,
                local_files_only=self.local_files_only,
                use_auth_token=token,
                torch_dtype=self._llm_dtype,
                quantization_config=self._llm_quantization_config,
                device_map=self.device if self._llm_quantization_config else None
            )
        )

        def load_llm_projector():
            llm_projector = LLMFusionModule(
                clip_dim=768,
                llm_dim=self.t5_model.model.config.d_model,
                num_heads=8
            )

            projector_state_dict = torch.load(self.llm_projector_path)

            # Shape usually [768, 2048] for the author provided CAM model,
            # large variants of the rank encoder models need [768, 1024]
            llm_proj_weight = projector_state_dict["CrossFusionModule.0.llm_proj.weight"]

            required_cam_model_dim = self.t5_model.model.config.d_model

            if llm_proj_weight.shape[1] > required_cam_model_dim:
                # check if we need to truncate

                new_llm_proj_weight = torch.zeros(768, required_cam_model_dim)
                new_llm_proj_weight[:, :required_cam_model_dim] = llm_proj_weight[:, :required_cam_model_dim]

                # Replace in the state_dict
                projector_state_dict["CrossFusionModule.0.llm_proj.weight"] = new_llm_proj_weight
            elif llm_proj_weight.shape[1] < required_cam_model_dim:
                raise self.argument_error(
                    f'Argument "projector" and related, bad llm_proj.weight in specified CAM model, '
                    f'cannot upscale projection weight tensor dimension [1] to: {required_cam_model_dim}')

            llm_projector.load_state_dict(projector_state_dict)
            llm_projector.to(self._llm_dtype)

            return llm_projector

        # the tag should represent that this models
        # "cached" status is linked to the type of
        # encoder model picked, because its parameters
        # depend on the encoder type, also the dtype
        # since we do not want to be repeatedly
        # upcasting and down-casting a cached model

        self.llm_projector = self.load_object_cached(
            tag=self.llm_projector_path + encoder_model_path + llm_dtype,
            estimated_size=cam_size_estimate,
            method=load_llm_projector
        )

        if weighter:
            self._weighter = _promptweighters.create_prompt_weighter(
                weighter,
                model_type=_enums.ModelType.SD,
                dtype=self.dtype,
                local_files_only=self.local_files_only,
                device=self.device
            )
        else:
            self._weighter = None

        self._tensors = list()

    def _get_projector_path(self,
                            model,
                            weight_name,
                            subfolder,
                            revision,
                            local_files_only: bool = False,
                            use_auth_token: str | None = None):
        try:
            if _hfhub.is_single_file_model_load(model):
                if os.path.exists(model):
                    return model
                else:
                    return _hfhub.download_non_hf_slug_model(model)
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
                f'Argument "projector" and related, '
                f'could not find projector model: {e}') from e

    def get_extra_supported_args(self) -> list[str]:
        # override and support passing the rankgen prompt
        # separately from the clip prompt using --second-prompts,
        # this is telling dgenerate that the pipeline can now
        # support accepting these arguments in particular due to our
        # custom embedding generation, normally, SD1.5 cannot
        # accept these as it only uses one encoder.

        # This allows dgenerate to generate these pipeline
        # arguments using --second-prompts and pass them
        # into the prompt weighter without runtime validation
        # errors occuring.
        return ['prompt_2', 'negative_prompt_2']

    def _generate_clip_embeds(self, pipeline, positive: str, negative: str):
        if not self._weighter:

            if hasattr(pipeline, 'maybe_convert_prompt'):
                if positive:
                    positive = pipeline.maybe_convert_prompt(positive, tokenizer=pipeline.tokenizer)

            input_ids = pipeline.tokenizer(
                positive, return_tensors="pt", truncation=True, padding="max_length", max_length=77).input_ids

            input_ids = input_ids.to(self.device)
            pipeline.text_encoder.to(self.device)

            positive_embeds = pipeline.text_encoder(input_ids, return_dict=False)[0]
            input_ids.to('cpu')

            if hasattr(pipeline, 'maybe_convert_prompt'):
                if negative:
                    negative = pipeline.maybe_convert_prompt(negative, tokenizer=pipeline.tokenizer)

            input_ids = pipeline.tokenizer(
                negative, return_tensors="pt", truncation=True, padding="max_length", max_length=77).input_ids

            input_ids = input_ids.to(self.device)
            negative_embeds = pipeline.text_encoder(input_ids, return_dict=False)[0]
            input_ids.to('cpu')

            return positive_embeds, negative_embeds

        else:
            result = self._weighter.translate_to_embeds(
                pipeline, {'prompt': positive, 'negative_prompt': negative})
            return result['prompt_embeds'], result['negative_prompt_embeds']

    @staticmethod
    def _remove_compel_prompting_syntax(text: str) -> str:
        import dgenerate.extras.compel.prompt_parser as _parser
        parser = _parser.PromptParser()
        conjunction = parser.parse_conjunction(text)
        plain_text_fragments = []

        def extract_plain_text(node):
            if isinstance(node, _parser.Fragment):
                plain_text_fragments.append(node.text)
            elif isinstance(node, _parser.FlattenedPrompt):
                for child in node.children:
                    extract_plain_text(child)
            elif isinstance(node, _parser.Conjunction):
                for prompt in node.prompts:
                    extract_plain_text(prompt)
            elif isinstance(node, _parser.Blend):
                for prompt in node.prompts:
                    extract_plain_text(prompt)
            elif isinstance(node, _parser.Attention):
                for child in node.children:
                    extract_plain_text(child)
            elif isinstance(node, _parser.CrossAttentionControlSubstitute):
                for child in node.original:
                    extract_plain_text(child)
                for child in node.edited:
                    extract_plain_text(child)

        extract_plain_text(conjunction)
        return ' '.join(plain_text_fragments)

    def _filter_rankgen_prompt(self, text: str, part: str):
        if self._weighter is None:
            # filtering not required, no weighting syntax expected
            return text
        elif self._weighter.loaded_by_name == 'sd-embed' or \
                (self._weighter.loaded_by_name == 'compel' and self._weighter._syntax == 'sdwui'):
            # easy
            import dgenerate.extras.sd_embed.prompt_parser as _parser
            result = ' '.join(t for t in (token.strip() for token, _ in _parser.parse_prompt_attention(text)) if t)

        elif self._weighter.loaded_by_name == 'compel':
            # hard, blends get duplicated, and its ambiguous how
            # the prompt should actually be

            # really, the user should specify --second-prompts
            # and not allow this to happen with complex weighting
            result = self._remove_compel_prompting_syntax(text)
        else:
            raise RuntimeError(
                'llm4gen prompt weighter, impossible branch, invalid weighter name.')

        if result:
            _messages.debug_log(
                f'llm4gen prompt weighter auto filtered weighting '
                f'syntax out of rankgen {part} prompt, result: "{result}"')

        return result

    @torch.inference_mode()
    def _translate_to_embeds(self,
                             pipeline: diffusers.DiffusionPipeline,
                             args: dict[str, typing.Any]):

        # we are responsible for generating these arguments
        # if they exist already then we cannot do our job

        forbidden_call_args = {
            'prompt_embeds',
            'negative_prompt_embeds'
        }

        if any(a in forbidden_call_args for a in args.keys()):
            raise _exceptions.PromptWeightingUnsupported(
                f'Prompt weighting not supported for --model-type: '
                f'{_enums.get_model_type_string(self.model_type)} with current configuration.')

        if 'clip_skip' in args:
            raise _exceptions.PromptWeightingUnsupported(
                f'llm4gen prompt-weighter does not support --clip-skips.')

        pipeline_sig = set(inspect.signature(pipeline.__call__).parameters.keys())

        if 'prompt_embeds' not in pipeline_sig:
            # pipeline does not support passing prompt embeddings directly

            raise _exceptions.PromptWeightingUnsupported(
                f'Prompt weighting not supported for --model-type: '
                f'{_enums.get_model_type_string(self.model_type)} with current configuration.')

        if not pipeline.__class__.__name__.startswith('StableDiffusion'):
            raise _exceptions.PromptWeightingUnsupported(
                f'Prompt weighting not supported for --model-type: {_enums.get_model_type_string(self.model_type)}')

        output = dict(args)

        positive = args.get('prompt')
        negative = args.get('negative_prompt')

        positive_2 = args.get('prompt_2')
        negative_2 = args.get('negative_prompt_2')

        prompt_args = re.compile(r'^(prompt|negative_prompt)(_\d+)?$')

        for name in args.keys():
            if prompt_args.match(name):
                output.pop(name)

        positive = positive if positive else ""
        negative = negative if negative else ""

        if isinstance(pipeline.unet.config.attention_head_dim, list):
            raise _exceptions.PromptWeightingUnsupported(
                'llm4gen prompt-weighter does not support Stable Diffusion 2.* models.')
        else:
            heads = pipeline.unet.config.attention_head_dim
            dim = pipeline.unet.config.cross_attention_dim
            head_dim = dim // heads

        try:
            for module in self.llm_projector.CrossFusionModule.modules():
                module.heads = heads
                module.dim = dim
                module.head_dim = head_dim

            self.memory_guard_device(self.device, self.size_estimate)
            self.move_text_encoders(pipeline, self.device)

            self.llm_projector.to(self.device).eval()
            self.t5_model.to(self.device).eval()

            embeds_dtype = _enums.get_torch_dtype(self.dtype)

            clip_embed, neg_clip_embeds = self._generate_clip_embeds(
                pipeline,
                positive,
                negative
            )

            filtered_rankgen_pos = self._filter_rankgen_prompt(positive, 'positive') if not positive_2 else positive_2
            filtered_rankgen_neg = self._filter_rankgen_prompt(negative, 'negative') if not negative_2 else negative_2

            llm_embed = self.t5_model.encode(filtered_rankgen_pos).to(self.device)
            neg_llm_embed = self.t5_model.encode(filtered_rankgen_neg, max_length=llm_embed.shape[1]).to(self.device)

            pos_conditioning = self.llm_projector(
                clip_embed.to(self._llm_dtype), llm_embed).to(embeds_dtype)

            neg_conditioning = self.llm_projector(
                neg_clip_embeds.to(self._llm_dtype), neg_llm_embed).to(embeds_dtype)
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

    def translate_to_embeds(self,
                            pipeline: diffusers.DiffusionPipeline,
                            args: dict[str, typing.Any]):
        try:
            return self._translate_to_embeds(pipeline, args)
        except RuntimeError as e:
            # catch bitsandbytes misconfigured on CPU
            if 'quant' in str(e):
                raise self.argument_error(
                    f'llm4gen prompt weighter argument "llm-quantizer": {e}') from e
            raise

    def cleanup(self):
        if self._weighter is not None:
            self._weighter.cleanup()

        for tensor in self._tensors:
            tensor.to('cpu')
            del tensor
        self._tensors.clear()
        gc.collect()
        _memory.torch_gc()
