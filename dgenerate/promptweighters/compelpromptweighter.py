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
import re
import typing

import compel
import torch

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.promptweighters.exceptions as _exceptions
import dgenerate.promptweighters.promptweighter as _promptweighter
import dgenerate.pipelinewrapper.pipelines as _pipelines

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

        clip_skip = args.get('clip_skip', 0)

        pipeline_sig = _pipelines.get_pipeline_call_args(pipeline)

        if 'prompt_embeds' not in pipeline_sig:
            # pipeline does not support passing prompt embeddings directly

            raise _exceptions.PromptWeightingUnsupported(
                f'Prompt weighting not supported for --model-type: {_enums.get_model_type_string(self.model_type)}, '
                f'in mode: {_enums.get_pipeline_type_string(self.pipeline_type)}')

        needs_pooled = False

        if pipeline.__class__.__name__.startswith('StableDiffusionXL'):

            if pipeline.text_encoder is not None:
                compel_compiler = compel.Compel(
                    tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                    text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                    returned_embeddings_type=
                    compel.ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                    device=device,
                    truncate_long_prompts=False)
                needs_pooled = True
            else:
                compel_compiler = compel.Compel(
                    tokenizer=pipeline.tokenizer_2,
                    text_encoder=pipeline.text_encoder_2,
                    truncate_long_prompts=False,
                    returned_embeddings_type=
                    compel.ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=True, device=device)
                needs_pooled = True

        elif pipeline.__class__.__name__.startswith('StableDiffusion'):
            compel_compiler = compel.Compel(
                tokenizer=pipeline.tokenizer,
                text_encoder=pipeline.text_encoder,
                truncate_long_prompts=False,
                returned_embeddings_type=
                compel.ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
                if clip_skip > 0 and pipeline.text_encoder.hidden_act == 'quick_gelu'
                else compel.ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                device=device)
        else:
            raise _exceptions.PromptWeightingUnsupported(
                f'Prompt weighting not supported for --model-type: {_enums.get_model_type_string(self.model_type)}, '
                f'in mode: {_enums.get_pipeline_type_string(self.pipeline_type)}')

        output = dict(args)

        positive = args.get('prompt')
        negative = args.get('negative_prompt')

        prompt_args = re.compile(r'^(prompt|negative_prompt)(_\d+)?$')
        extra_prompt_args = re.compile(r'^(prompt|negative_prompt)(_\d+)$')

        for name in args.keys():
            if prompt_args.match(name):
                output.pop(name)
                if extra_prompt_args.match(name):
                    _messages.log(
                        f'Diffusion argument {name} ignored by compel prompt weighting implementation.',
                        level=_messages.WARNING)

        positive = positive if positive else ""

        if positive and self._syntax == 'sdwui':
            positive = _translate_sdwui_to_compel(positive)
            _messages.log(f'Positive prompt translated to compel: "{positive}"',
                          level=_messages.DEBUG)

        if positive and hasattr(pipeline, 'maybe_convert_prompt') and pipeline.tokenizer is not None:
            positive = pipeline.maybe_convert_prompt(positive, tokenizer=pipeline.tokenizer)

        pos_pooled = None
        if needs_pooled:
            pos_conditioning, pos_pooled = compel_compiler(positive)
        else:
            pos_conditioning = compel_compiler(positive)

        self._tensors.append(pos_conditioning)

        if pos_pooled is not None:
            self._tensors.append(pos_pooled)

        if not negative:
            if pos_pooled is not None:
                output.update({
                    'prompt_embeds': pos_conditioning,
                    'pooled_prompt_embeds': pos_pooled
                })
            else:
                output.update({
                    'prompt_embeds': pos_conditioning
                })
        else:
            if self._syntax == 'sdwui':
                negative = _translate_sdwui_to_compel(negative)
                _messages.log(f'Negative prompt translated to compel: "{negative}"',
                              level=_messages.DEBUG)

            if hasattr(pipeline, 'maybe_convert_prompt') and pipeline.tokenizer is not None:
                negative = pipeline.maybe_convert_prompt(negative, tokenizer=pipeline.tokenizer)

            neg_pooled = None
            if needs_pooled:
                neg_conditioning, neg_pooled = compel_compiler(negative)
            else:
                neg_conditioning = compel_compiler(negative)

            pos_conditioning, neg_conditioning, = compel_compiler.pad_conditioning_tensors_to_same_length(
                [pos_conditioning, neg_conditioning])

            self._tensors.append(neg_conditioning)

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

        def debug_string():
            debug_args = ", ".join(
                f"{k}={v if not isinstance(v, torch.Tensor) else f'torch.Tensor({v.shape})'}" for k, v in
                output.items())
            return 'CompelPromptWeighter translated pipeline args: {' + debug_args + '}'

        _messages.debug_log(debug_string)

        return output

    def cleanup(self):
        for tensor in self._tensors:
            tensor.to('cpu')
        self._tensors.clear()
        gc.collect()
