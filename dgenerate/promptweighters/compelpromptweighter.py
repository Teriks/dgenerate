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

import compel
import torch

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.promptweighters.exceptions as _exceptions
import dgenerate.promptweighters.promptweighter as _promptweighter

_Attention = typing.List[typing.Tuple[str, float]]

_reAttention = re.compile(
    r'\\\\\(|\\\\\)|\\\\\[|\\\\\]|\\\\\\\\|\||\(|\[|:([+-]?[\.\d]+)\)|\)|\]|[^\[\\\(\)\[\]:]+|:'
)

_reBreak = re.compile(r'\s*\bBREAK\b\s*', re.M)


def _parse_sdwui_attention_from_prompt(text: str) -> _Attention:
    res: _Attention = []
    round_brackets: typing.List[int] = []
    square_brackets: typing.List[int] = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position: int, multiplier: float) -> None:
        for p in range(start_position, len(res)):
            res[p] = (res[p][0], res[p][1] * multiplier)

    match_iterator = _reAttention.finditer(text)

    for m in match_iterator:
        match_text = m.group(0)
        weight = m.group(1)

        if match_text.startswith('\\'):
            res.append((match_text[1:], 1.0))
        elif match_text == '(':
            round_brackets.append(len(res))
        elif match_text == '[':
            square_brackets.append(len(res))
        elif weight and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif match_text == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif match_text == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = _reBreak.split(match_text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(('BREAK', -1))
                res.append((part, 1.0))

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if not res:
        res = [('', 1.0)]

    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i] = (res[i][0] + res[i + 1][0], res[i][1])
            del res[i + 1]
        else:
            i += 1

    return res


def _attention_to_compel_prompt(attention: _Attention) -> str:
    tokens: typing.List[str] = []
    for text, weight in attention:
        weight = round(weight, 2)
        if weight == 1.0:
            tokens.append(text)
        else:
            pad = '-' if weight < 1.0 else '+'
            sign = pad * round(abs(weight - 1.0) / 0.1)
            tokens.append(f'({text}){sign}')
    return ''.join(tokens)


def _translate_sdwui_to_compel(text: str) -> str:
    attention = _parse_sdwui_attention_from_prompt(text)
    return _attention_to_compel_prompt(attention)


class CompelPromptWeighter(_promptweighter.PromptWeighter):
    """
    Implements prompt weighting syntax for Stable Diffusion 1/2 and Stable Diffusion XL
    using compel. The default syntax is "compel" which is analogous to the syntax
    used by InvokeAI.

    Specifying the syntax "sdwui" will translate your prompt from Stable Diffusion Web UI
    syntax into compel / InvokeAI syntax before generating the prompt embeddings.

    If you wish to use prompt syntax for weighting tokens that is similar to ComfyUI,
    Automatic1111, or CivitAI for example, use: 'compel;syntax=sdwui'

    The underlying weighting behavior for tokens is not exactly the same as other software
    that uses the more common "sdwui" syntax, so your prompt may need adjusting if you are
    reusing a prompt from those other pieces of software.

    You can read about compel here: https://github.com/damian0815/compel

    And InvokeAI here: https://github.com/invoke-ai/InvokeAI

    This prompt weighter supports the model types:

    NOWRAP!
    --model-type torch
    --model-type torch-pix2pix
    --model-type torch-upscaler-x4
    --model-type torch-sdxl
    --model-type torch-sdxl-pix2pix

    The secondary prompt option for SDXL --sdxl-second-prompts is supported by this prompt weighter
    implementation. However, --sdxl-refiner-second-prompts is not supported and will be ignored
    with a warning message.
    """

    NAMES = ['compel']

    def __init__(self, syntax: str = 'compel', **kwargs):
        super().__init__(**kwargs)
        self._tensors = list()
        self._syntax = syntax

        if syntax not in {'compel', 'sdwui'}:
            raise self.argument_error(
                f'Compel prompt weighter does not support the syntax: "{syntax}", '
                f'must be one of: "compel" or "sdwui".')

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
                or pipeline.__class__.__name__.startswith('StableDiffusion')) or \
                _enums.model_type_is_sd3(self.model_type):
            raise _exceptions.PromptWeightingUnsupported(
                f'Prompt weighting not supported for --model-type: {_enums.get_model_type_string(self.model_type)}')

        output = dict(args)

        clip_skip = args.get('clip_skip', 0)

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

        if self._syntax == 'sdwui':
            if positive:
                positive = _translate_sdwui_to_compel(positive)
                _messages.debug_log(f'Positive prompt translated to compel: "{positive}"')
            if negative:
                negative = _translate_sdwui_to_compel(negative)
                _messages.debug_log(f'Negative prompt translated to compel: "{negative}"')
            if positive_2:
                positive_2 = _translate_sdwui_to_compel(positive_2)
                _messages.debug_log(f'Positive prompt 2 translated to compel: "{positive_2}"')
            if negative_2:
                negative_2 = _translate_sdwui_to_compel(negative_2)
                _messages.debug_log(f'Negative prompt 2 translated to compel: "{negative_2}"')

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

        if pipeline.__class__.__name__.startswith('StableDiffusionXL'):

            if pipeline.tokenizer is not None:

                original_clip_layers = pipeline.text_encoder.text_model.encoder.layers
                original_clip_layers_2 = pipeline.text_encoder_2.text_model.encoder.layers

                try:
                    if clip_skip > 0:
                        pipeline.text_encoder.text_model.encoder.layers = original_clip_layers[:-clip_skip]
                        pipeline.text_encoder_2.text_model.encoder.layers = original_clip_layers_2[:-clip_skip]

                    if positive_2 or negative_2:
                        compel1 = compel.Compel(
                            tokenizer=pipeline.tokenizer,
                            text_encoder=pipeline.text_encoder,
                            returned_embeddings_type=
                            compel.ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                            requires_pooled=False,
                            truncate_long_prompts=False,
                            device=device
                        )

                        torch.cuda.empty_cache()

                        compel2 = compel.Compel(
                            tokenizer=pipeline.tokenizer_2,
                            text_encoder=pipeline.text_encoder_2,
                            returned_embeddings_type=
                            compel.ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                            requires_pooled=True,
                            truncate_long_prompts=False,
                            device=device
                        )

                        conditioning1 = compel1(positive)
                        conditioning2, pos_pooled = compel2(positive_2)
                        pos_conditioning = torch.cat((conditioning1, conditioning2), dim=-1)

                        conditioning1 = compel1(negative)
                        conditioning2, neg_pooled = compel2(negative_2)
                        neg_conditioning = torch.cat((conditioning1, conditioning2), dim=-1)

                        pos_conditioning, neg_conditioning = compel1.pad_conditioning_tensors_to_same_length(
                            [pos_conditioning, neg_conditioning])
                    else:
                        compel1 = compel.Compel(
                            tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                            text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                            returned_embeddings_type=
                            compel.ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                            requires_pooled=[False, True],
                            truncate_long_prompts=False,
                            device=device
                        )

                        pos_conditioning, pos_pooled = compel1(positive)
                        neg_conditioning, neg_pooled = compel1(negative)

                        pos_conditioning, neg_conditioning = compel1.pad_conditioning_tensors_to_same_length(
                            [pos_conditioning, neg_conditioning])
                finally:
                    # leaving this modified would really
                    # screw up other stuff in dgenerate :)
                    if clip_skip > 0:
                        pipeline.text_encoder.text_model.encoder.layers = original_clip_layers
                        pipeline.text_encoder_2.text_model.encoder.layers = original_clip_layers_2

                torch.cuda.empty_cache()

            else:
                if positive_2 or negative_2:
                    _messages.log(
                        f'Prompt weighting is not supported by --prompt-weighter '
                        f'"compel" for --sdxl-refiner-second-prompts, that prompt is being ignored.',
                        level=_messages.WARNING)

                original_clip_layers_2 = pipeline.text_encoder_2.text_model.encoder.layers

                try:
                    if clip_skip > 0:
                        pipeline.text_encoder_2.text_model.encoder.layers = original_clip_layers_2[:-clip_skip]

                    compel2 = compel.Compel(
                        tokenizer=pipeline.tokenizer_2,
                        text_encoder=pipeline.text_encoder_2,
                        returned_embeddings_type=
                        compel.ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                        requires_pooled=True,
                        truncate_long_prompts=False,
                        device=device
                    )

                    pos_conditioning, pos_pooled = compel2(positive)
                    neg_conditioning, neg_pooled = compel2(negative)
                finally:
                    # leaving this modified would really
                    # screw up other stuff in dgenerate :)
                    if clip_skip > 0:
                        pipeline.text_encoder_2.text_model.encoder.layers = original_clip_layers_2

                pos_conditioning, neg_conditioning = compel2.pad_conditioning_tensors_to_same_length(
                    [pos_conditioning, neg_conditioning])

                torch.cuda.empty_cache()

        elif pipeline.__class__.__name__.startswith('StableDiffusion'):
            embedding_type = \
                compel.ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED \
                if clip_skip > 0 and pipeline.text_encoder.config.hidden_act == 'quick_gelu' \
                else compel.ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED

            _messages.debug_log('Compel Clip Skip:', args.get('clip_skip', 0))
            _messages.debug_log('Compel text_encoder.config.hidden_act:', pipeline.text_encoder.config.hidden_act)
            _messages.debug_log('Compel Embedding Type:', embedding_type)

            original_clip_layers = pipeline.text_encoder.text_model.encoder.layers

            try:
                if clip_skip > 0:
                    pipeline.text_encoder.text_model.encoder.layers = original_clip_layers[:-clip_skip]

                compel1 = compel.Compel(
                    tokenizer=pipeline.tokenizer,
                    text_encoder=pipeline.text_encoder,
                    truncate_long_prompts=False,
                    returned_embeddings_type=embedding_type,
                    device=device)

                pos_conditioning = compel1(positive)
                neg_conditioning = compel1(negative)

            finally:
                # leaving this modified would really
                # screw up other stuff in dgenerate :)
                if clip_skip > 0:
                    pipeline.text_encoder.text_model.encoder.layers = original_clip_layers

            pos_conditioning, neg_conditioning = compel1.pad_conditioning_tensors_to_same_length(
                [pos_conditioning, neg_conditioning])

            torch.cuda.empty_cache()

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
