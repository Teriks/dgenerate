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

from dgenerate.extras import compel as _compel
import dgenerate.extras.compel.convenience_wrappers as _compel_c
import torch
from dgenerate.pipelinewrapper import pipelines as _pipelines

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.promptweighters.exceptions as _exceptions
import dgenerate.promptweighters.promptweighter as _promptweighter
import dgenerate.memory as _memory

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
    --model-type sd
    --model-type pix2pix
    --model-type upscaler-x4
    --model-type sdxl
    --model-type sdxl-pix2pix
    --model-type s-cascade
    --model-type flux
    --model-type flux-fill
    --model-type flux-kontext

    The secondary prompt option for SDXL and Flux --second-prompts is supported by this prompt weighter
    implementation. However, --second-model-second-prompts is not supported and will be ignored
    with a warning message.
    
    For Flux models, the main prompt is processed by the T5 text encoder, while the secondary prompt
    (style prompt) is processed by the CLIP text encoder to generate pooled embeddings.
    """

    NAMES = ['compel']

    OPTION_ARGS = {
        'syntax': ['compel', 'sdwui']
    }

    def __init__(self, syntax: str = 'compel', **kwargs):
        super().__init__(**kwargs)

        supported = {
            _enums.ModelType.SD,
            _enums.ModelType.PIX2PIX,
            _enums.ModelType.UPSCALER_X4,
            _enums.ModelType.SDXL,
            _enums.ModelType.SDXL_PIX2PIX,
            _enums.ModelType.S_CASCADE,
            _enums.ModelType.FLUX,
            _enums.ModelType.FLUX_FILL,
            _enums.ModelType.FLUX_KONTEXT
        }

        if self.model_type not in supported:
            raise _exceptions.PromptWeightingUnsupported(
                f'Prompt weighting not supported for --model-type: {_enums.get_model_type_string(self.model_type)}')

        if syntax not in {'compel', 'sdwui'}:
            raise self.argument_error(
                f'Compel prompt weighter does not support the syntax: "{syntax}", '
                f'must be one of: "compel" or "sdwui".')

        self._tensors = list()
        self._syntax = syntax

    def get_extra_supported_args(self) -> list[str]:
        if self.model_type == _enums.ModelType.S_CASCADE:
            return ['clip_skip']
        else:
            return []

    @torch.inference_mode()
    def translate_to_embeds(self,
                            pipeline: diffusers.DiffusionPipeline,
                            args: dict[str, typing.Any]):

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
                f'Prompt weighting not supported for --model-type: '
                f'{_enums.get_model_type_string(self.model_type)} with current configuration.')

        pipeline_sig = set(inspect.signature(pipeline.__call__).parameters.keys())

        if 'prompt_embeds' not in pipeline_sig:
            # pipeline does not support passing prompt embeddings directly

            raise _exceptions.PromptWeightingUnsupported(
                f'Prompt weighting not supported for --model-type: '
                f'{_enums.get_model_type_string(self.model_type)} with current configuration.')

        if not (pipeline.__class__.__name__.startswith('StableDiffusionXL')
                or pipeline.__class__.__name__.startswith('StableDiffusion')
                or pipeline.__class__.__name__.startswith('StableCascade')
                or pipeline.__class__.__name__.startswith('Flux')) or \
                _enums.model_type_is_sd3(self.model_type):
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

        # Check if sequential offload is enabled - compel requires text encoders to be loaded
        # which is incompatible with sequential offload where models are loaded on-demand
        if _pipelines.is_sequential_cpu_offload_enabled(pipeline):
            raise _exceptions.PromptWeightingUnsupported(
                f'The compel prompt weighter is not compatible with --model-sequential-offload '
                f'because it requires text encoders to be fully loaded before pipeline execution. '
                f'Use --model-cpu-offload instead.')

        self.move_text_encoders(pipeline, self.device)

        if pipeline.__class__.__name__.startswith('StableDiffusionXL'):
            clip_skip = args.get('clip_skip', None)

            if pipeline.tokenizer is not None:
                compel_sdxl = _compel_c.CompelForSDXL(pipeline, clip_skip=clip_skip)

                # Handle secondary prompts (positive_2/negative_2) as style prompts
                result_pos = compel_sdxl(
                    main_prompt=positive,
                    style_prompt=positive_2 if positive_2 else None,
                    negative_prompt=negative,
                    negative_style_prompt=negative_2 if negative_2 else None
                )
                
                pos_conditioning = result_pos.embeds
                pos_pooled = result_pos.pooled_embeds
                neg_conditioning = result_pos.negative_embeds
                neg_pooled = result_pos.negative_pooled_embeds

                _memory.torch_gc()

            else:
                if positive_2 or negative_2:
                    _messages.warning(
                        f'Prompt weighting is not supported by --prompt-weighter '
                        f'"compel" for --second-model-second-prompts, that prompt is being ignored.'
                    )

                compel2 = _compel.Compel(
                    tokenizer=pipeline.tokenizer_2,
                    text_encoder=pipeline.text_encoder_2,
                    returned_embeddings_type=
                    _compel.ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=True,
                    truncate_long_prompts=False,
                    device=self.device,
                    clip_skip=clip_skip
                )

                pos_conditioning, pos_pooled = compel2(positive)
                neg_conditioning, neg_pooled = compel2(negative)

                pos_conditioning, neg_conditioning = compel2.pad_conditioning_tensors_to_same_length(
                    [pos_conditioning, neg_conditioning])

                _memory.torch_gc()
        elif pipeline.__class__.__name__.startswith('StableCascade'):
            # needs to be consumed as the pipeline
            # does not have this argument
            if 'clip_skip' in output:
                clip_skip = output.pop('clip_skip')
            else:
                clip_skip = None

            compel_cascade = _compel_c.CompelForStableCascade(pipeline, clip_skip=clip_skip)

            result = compel_cascade(
                prompt=positive,
                negative_prompt=negative
            )
            
            pos_conditioning = result.embeds
            pos_pooled = result.pooled_embeds
            neg_conditioning = result.negative_embeds
            neg_pooled = result.negative_pooled_embeds

            _memory.torch_gc()

        elif pipeline.__class__.__name__.startswith('StableDiffusion'):
            clip_skip = args.get('clip_skip', None)

            compel_sd = _compel_c.CompelForSD(pipeline, clip_skip=clip_skip)

            result = compel_sd(
                prompt=positive,
                negative_prompt=negative
            )
            
            pos_conditioning = result.embeds
            neg_conditioning = result.negative_embeds

            _memory.torch_gc()

        elif pipeline.__class__.__name__.startswith('Flux'):
            # Check if this Flux pipeline supports negative prompting
            pipeline_sig = inspect.signature(pipeline.__call__).parameters
            supports_negative = 'negative_prompt_embeds' in pipeline_sig
            clip_skip = args.get('clip_skip', None)
            
            if not supports_negative and (negative or negative_2):
                _messages.warning(
                    'Flux is ignoring the provided negative prompt as it '
                    'does not support negative prompting in the current configuration.'
                )
            
            # Use the modern CompelForFlux wrapper
            compel_flux = _compel_c.CompelForFlux(pipeline, clip_skip=clip_skip)
            
            # Generate embeddings using the convenience wrapper
            # Handle secondary prompts (positive_2/negative_2) as style prompts
            result = compel_flux(
                main_prompt=positive,
                style_prompt=positive_2 if positive_2 else None,
                negative_prompt=negative if supports_negative else None,
                negative_style_prompt=negative_2 if (supports_negative and negative_2) else None
            )
            
            pos_conditioning = result.embeds
            pos_pooled = result.pooled_embeds
            neg_conditioning = result.negative_embeds if supports_negative else None
            neg_pooled = result.negative_pooled_embeds if supports_negative else None
            
            _memory.torch_gc()

        if pos_conditioning is not None:
            self._tensors.append(pos_conditioning)
        if neg_conditioning is not None:
            self._tensors.append(neg_conditioning)

        if pos_pooled is not None:
            self._tensors.append(pos_pooled)

        if neg_pooled is not None:
            self._tensors.append(neg_pooled)

        output.update({
            'prompt_embeds': pos_conditioning,
        })
        
        if neg_conditioning is not None:
            output.update({
                'negative_prompt_embeds': neg_conditioning,
            })

        if pos_pooled is not None:

            if self.model_type == _enums.ModelType.S_CASCADE:
                output.update({
                    'prompt_embeds_pooled': pos_pooled,
                })
            else:
                output.update({
                    'pooled_prompt_embeds': pos_pooled,
                })

        if neg_pooled is not None:
            if self.model_type == _enums.ModelType.S_CASCADE:
                output.update({
                    'negative_prompt_embeds_pooled': neg_pooled,
                })
            else:
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
        _memory.torch_gc()
