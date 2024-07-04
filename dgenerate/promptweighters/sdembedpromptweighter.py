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

import torch

import dgenerate.extras.sd_embed as _sd_embed
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.promptweighters.exceptions as _exceptions
import dgenerate.promptweighters.promptweighter as _promptweighter


class SdEmbedPromptWeighter(_promptweighter.PromptWeighter):
    r"""
    Implements prompt weighting syntax for Stable Diffusion 1/2, Stable Diffusion XL,
    and Stable Diffusion 3 using sd_embed.

    sd_embed uses a Stable Diffusion Web UI compatible prompt syntax.

    See: https://github.com/xhinker/sd_embed

    NOWRAP!
    @misc{sd_embed_2024,
      author       = {Shudong Zhu(Andrew Zhu)},
      title        = {Long Prompt Weighted Stable Diffusion Embedding},
      howpublished = {\url{https://github.com/xhinker/sd_embed}},
      year         = {2024},
    }

    NOWRAP!
    --model-type torch
    --model-type torch-pix2pix
    --model-type torch-upscaler-x4
    --model-type torch-sdxl
    --model-type torch-sdxl-pix2pix
    --model-type torch-sd3

    The secondary prompt option for SDXL --sdxl-second-prompts is supported by this prompt weighter
    implementation. However, --sdxl-refiner-second-prompts is not supported and will be ignored
    with a warning message.
    """

    NAMES = ['sd-embed']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tensors = list()

    @torch.no_grad()
    def translate_to_embeds(self,
                            pipeline,
                            device: str,
                            args: dict[str, any]):

        # we are responsible for generating these arguments
        # if they exist already then we cannot do our job

        forbidden_call_args = {
            'prompt_embeds',
            'pooled_prompt_embeds',
            'negative_prompt_embeds',
            'negative_pooled_prompt_embeds'
        }

        if any(a in forbidden_call_args for a in args.keys()):
            raise _exceptions.PromptWeightingUnsupported(
                f'Prompt weighting not supported for --model-type: {_enums.get_model_type_string(self.model_type)}, '
                f'in mode: {_enums.get_pipeline_type_string(self.pipeline_type)}')

        pipeline_sig = set(inspect.signature(pipeline.__call__).parameters.keys())

        if 'prompt_embeds' not in pipeline_sig:
            # pipeline does not support passing prompt embeddings directly

            raise _exceptions.PromptWeightingUnsupported(
                f'Prompt weighting not supported for --model-type: {_enums.get_model_type_string(self.model_type)}, '
                f'in mode: {_enums.get_pipeline_type_string(self.pipeline_type)}')

        if not (pipeline.__class__.__name__.startswith('StableDiffusionXL')
                or pipeline.__class__.__name__.startswith('StableDiffusion')
                or pipeline.__class__.__name__.startswith('StableDiffusion3')):
            raise _exceptions.PromptWeightingUnsupported(
                f'Prompt weighting not supported for --model-type: {_enums.get_model_type_string(self.model_type)}, '
                f'in mode: {_enums.get_pipeline_type_string(self.pipeline_type)}')

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
        positive_2 = positive_2 if positive_2 else ""
        negative_2 = negative_2 if negative_2 else ""

        if hasattr(pipeline, 'maybe_convert_prompt'):
            # support refiner, which only has tokenizer_2
            tk = pipeline.tokenizer if pipeline.tokenizer is not None else pipeline.tokenizer_2

            if positive:
                positive = pipeline.maybe_convert_prompt(positive, tokenizer=tk)
            if negative:
                negative = pipeline.maybe_convert_prompt(negative, tokenizer=tk)

            if pipeline.tokenizer is not None:
                # refiner not supported for secondary prompt
                if positive_2:
                    positive_2 = pipeline.maybe_convert_prompt(positive_2, tokenizer=pipeline.tokenizer_2)
                if negative_2:
                    negative_2 = pipeline.maybe_convert_prompt(negative_2, tokenizer=pipeline.tokenizer_2)

        pos_conditioning = None
        neg_conditioning = None
        pos_pooled = None
        neg_pooled = None

        if pipeline.__class__.__name__.startswith('StableDiffusion3'):

            pos_conditioning, \
                neg_conditioning, \
                pos_pooled, \
                neg_pooled = _sd_embed.get_weighted_text_embeddings_sd3(
                pipe=pipeline,
                prompt=positive,
                neg_prompt=negative,
                pad_last_block=True,
                use_t5_encoder=pipeline.tokenizer_3 is not None,
                device=device)

        elif pipeline.__class__.__name__.startswith('StableDiffusionXL'):

            if pipeline.tokenizer is not None:

                if positive_2 or negative_2:
                    pos_conditioning, \
                        neg_conditioning, \
                        pos_pooled, \
                        neg_pooled = _sd_embed.get_weighted_text_embeddings_sdxl_2p(
                        pipe=pipeline,
                        prompt=positive,
                        prompt_2=positive_2,
                        neg_prompt=negative,
                        neg_prompt_2=negative_2,
                        device=device)
                else:
                    pos_conditioning, \
                        neg_conditioning, \
                        pos_pooled, \
                        neg_pooled = _sd_embed.get_weighted_text_embeddings_sdxl(
                        pipe=pipeline,
                        prompt=positive,
                        neg_prompt=negative,
                        device=device)
            else:
                if positive_2 or negative_2:
                    _messages.log(
                        f'Prompt weighting is not supported by --prompt-weighter '
                        f'"sd-embed" for --sdxl-refiner-second-prompts, that prompt is being ignored.',
                        level=_messages.WARNING)

                pos_conditioning, \
                    neg_conditioning, \
                    pos_pooled, \
                    neg_pooled = _sd_embed.get_weighted_text_embeddings_sdxl_refiner(
                    pipe=pipeline,
                    prompt=positive,
                    neg_prompt=negative,
                    device=device)

        elif pipeline.__class__.__name__.startswith('StableDiffusion'):

            pos_conditioning, \
                neg_conditioning = _sd_embed.get_weighted_text_embeddings_sd15(
                pipe=pipeline,
                prompt=positive,
                neg_prompt=negative,
                pad_last_block=False,
                clip_skip=args.get('clip_skip', 0),
                device=device)

        self._tensors.append(pos_conditioning)
        self._tensors.append(neg_conditioning)

        if pos_pooled is not None:
            self._tensors.append(pos_pooled)

        if neg_pooled is not None:
            self._tensors.append(neg_pooled)

        output.update({
            'prompt_embeds': pos_conditioning,
            'negative_prompt_embeds': neg_conditioning,
        })

        if pos_pooled is not None:
            output.update({
                'pooled_prompt_embeds': pos_pooled,
            })

        if neg_pooled is not None:
            output.update({
                'negative_pooled_prompt_embeds': neg_pooled,
            })

        return output

    def cleanup(self):
        for tensor in self._tensors:
            tensor.to('cpu')
            del tensor
        self._tensors.clear()
        gc.collect()
        torch.cuda.empty_cache()
