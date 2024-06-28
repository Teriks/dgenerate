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

import compel
import torch

import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.enums as _enums


class PromptWeightingUnsupported(Exception):
    pass


class PromptWeighter:

    @property
    def model_type(self) -> _enums.ModelType:
        return self._model_type

    @property
    def pipeline_type(self) -> _enums.PipelineType:
        return self._pipeline_type

    def __init__(self, model_type: _enums.ModelType, pipeline_type: _enums.PipelineType):
        self._model_type = model_type
        self._pipeline_type = pipeline_type

    def translate_to_embeds(self,
                            pipeline,
                            device: str,
                            args: dict[str, any]):
        """
        Translate the pipeline prompt arguments to ``prompt_embeds`` and ``pooled_prompt_embeds`` as needed.
        :param pipeline: The pipeline object
        :param device: The device the pipeline modules are on
        :param args: Call arguments to the pipeline
        :return: ``args``, supplemented with prompt embedding arguments
        """
        pass

    def cleanup(self):
        """
        Preform any cleanup required after translating the pipeline arguments to embeds
        """
        pass


class CompelPromptWeighter(PromptWeighter):

    def __init__(self, model_type: _enums.ModelType, pipeline_type: _enums.PipelineType):

        super().__init__(model_type, pipeline_type)
        self._tensors = list()

    @torch.no_grad()
    def translate_to_embeds(self,
                            pipeline,
                            device: str,
                            args: dict[str, any]):

        if 'prompt_embeds' in args or \
                'pooled_prompt_embeds' in args or \
                'negative_prompt_embeds' in args or \
                'negative_pooled_prompt_embeds' in args:
            raise PromptWeightingUnsupported(
                f'Prompt weighting not supported for --model-type: {_enums.get_model_type_string(self.model_type)}, '
                f'in mode: {_enums.get_pipeline_type_string(self.pipeline_type)}')

        clip_skip = args.get('clip_skip', 0)

        if pipeline.__class__.__name__.startswith('StableDiffusionXL'):

            if pipeline.text_encoder is not None:
                c = compel.Compel(tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                                  text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                                  returned_embeddings_type=
                                  compel.ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                                  requires_pooled=[False, True],
                                  device=device,
                                  truncate_long_prompts=False)
            else:
                c = compel.Compel(tokenizer=pipeline.tokenizer_2,
                                  text_encoder=pipeline.text_encoder_2,
                                  truncate_long_prompts=False,
                                  returned_embeddings_type=
                                  compel.ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                                  requires_pooled=True, device=device)

        elif pipeline.__class__.__name__.startswith('StableDiffusion'):
            c = compel.Compel(tokenizer=pipeline.tokenizer,
                              text_encoder=pipeline.text_encoder,
                              truncate_long_prompts=False,
                              returned_embeddings_type=
                              compel.ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
                              if clip_skip > 0 and pipeline.text_encoder.hidden_act == 'quick_gelu'
                              else compel.ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
                              device=device)
        else:
            raise PromptWeightingUnsupported(
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

        if positive:
            if hasattr(pipeline, 'maybe_convert_prompt') and pipeline.tokenizer is not None:
                pos_conditioning, pos_pooled = c(pipeline.maybe_convert_prompt(positive, tokenizer=pipeline.tokenizer))
            else:
                pos_conditioning, pos_pooled = c(positive)
        else:
            pos_conditioning, pos_pooled = c("")

        self._tensors.append(pos_conditioning)
        self._tensors.append(pos_pooled)

        if not negative:
            output.update({
                'prompt_embeds': pos_conditioning,
                'pooled_prompt_embeds': pos_pooled
            })

        else:
            if hasattr(pipeline, 'maybe_convert_prompt') and pipeline.tokenizer is not None:
                neg_conditioning, neg_pooled = c(pipeline.maybe_convert_prompt(negative, tokenizer=pipeline.tokenizer))
            else:
                neg_conditioning, neg_pooled = c(negative)

            pos_conditioning, neg_conditioning, = c.pad_conditioning_tensors_to_same_length(
                [pos_conditioning, neg_conditioning])

            self._tensors.append(neg_conditioning)
            self._tensors.append(neg_pooled)

            output.update({
                'prompt_embeds': pos_conditioning,
                'pooled_prompt_embeds': pos_pooled,
                'negative_prompt_embeds': neg_conditioning,
                'negative_pooled_prompt_embeds': neg_pooled
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


def prompt_weighter_names():
    """
    Implementation names.
    :return: a list of prompt weighter implementation names.
    """
    return ['compel']


def create_prompt_weighter(name, model_type: _enums.ModelType, pipeline_type: _enums.PipelineType) -> PromptWeighter:
    if name == 'compel':
        return CompelPromptWeighter(model_type, pipeline_type)

    raise ValueError(f'Unknown prompt weighter implementation: {name}')
