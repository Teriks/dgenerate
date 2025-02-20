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
import contextlib
import gc
import os.path

import huggingface_hub
import pyparsing
import torch
import transformers

import dgenerate.promptupscalers.promptupscaler as _promptupscaler
import dgenerate.promptupscalers.exceptions as _exceptions
import dgenerate.prompt as _prompt
from dynamicprompts.generators import CombinatorialPromptGenerator as _CombinatorialPromptGenerator
from dynamicprompts.generators import RandomPromptGenerator as _RandomPromptGenerator
from dynamicprompts.generators.magicprompt import MagicPromptGenerator as _MagicPromptGenerator
from dynamicprompts.generators.magicprompt import DEFAULT_MODEL_NAME as _MAGIC_DEFAULT_MODEL
from dynamicprompts.wildcards.wildcard_manager import WildcardManager as _WildcardManager
from dynamicprompts.generators.attentiongenerator import AttentionGenerator as _AttentionGenerator
import dgenerate.types as _types
import dgenerate.memory as _memory
import dgenerate.pipelinewrapper.util as _pipelinewrapper_util
import dgenerate.messages as _messages


def _magic_load_pipeline_patch(self, model_name: str) -> transformers.pipelines.Pipeline:
    # get rid of dynamicprompts caching mechanism, and logging
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token_id = model.config.eos_token_id

    return transformers.pipeline(
        task="text-generation",
        tokenizer=tokenizer,
        model=model,
        device='cpu',
        pad_token_id=tokenizer.eos_token_id,
    )


@contextlib.contextmanager
def _with_magic_load_pipeline_patch():
    og = _MagicPromptGenerator._load_pipeline
    try:
        _MagicPromptGenerator._load_pipeline = _magic_load_pipeline_patch
        yield
    finally:
        _MagicPromptGenerator._load_pipeline = og


class DynamicPromptsPromptUpscaler(_promptupscaler.PromptUpscaler):
    """
    Upscale prompts with the dynamicprompts library.

    See: https://github.com/adieyal/dynamicprompts

    The "part" argument indicates which parts of the prompt to act on,
    possible values are: "both", "positive", and "negative"

    The "magic" argument enables magicprompt, which generates a continuation
    of your prompt using the model "Gustavosta/MagicPrompt-Stable-Diffusion".

    The "magic-model" specifies the model path for magicprompt, the
    default value is: "Gustavosta/MagicPrompt-Stable-Diffusion". This
    can be a folder on disk or a Hugging Face repository slug.
    
    The "magic-seed" argument can be used to specify a seed for just the
    "magic" prompt generation, this overrides "seed".
    
    The "magic-max-length" arguments the max prompt length for a magicprompt
    generated prompt, this value defaults to 100.

    The "magic-temperature" argument sets the sampling temperature
    to use when generating prompts with magicprompt.

    The "attention" argument enables random token attention values,
    this requires the use of the "sd-embed" prompt weighter or "compel" in
    SD Web UI syntax mode, i.e. "compel;syntax=sdwui"
    
    The "attention-min" argument sets the minimum value for random
    attention added by "attention=True". The default value is 0.1
    
    The "attention-max" argument sets the maximum value for random
    attention added by "attention=True" The Default value is 0.9

    The "random" argument specifies that instead of strictly combinatorial
    output, dynamicprompts should produce N random variations of your
    prompt given the possibilities you have provided.
    
    The "random-seed" argument can be used to specify a seed for just the
    "random" prompt generation, this overrides "seed".

    The "variations" argument specifies how many variations should
    be produced when "random" is set to true. This argument cannot
    be used without specifying "random". The default value is 1.

    The "wildcards" argument can be used to specify a wildcards directory
    for dynamicprompt's wildcard syntax.

    The "seed" argument can be used to specify a seed for the "random" prompt
    generation as well as the "magic" prompt generation simultaneously. The
    same seed will be used for both generators.
    """

    NAMES = ['dynamicprompts']

    def __init__(self,
                 part: str = 'both',
                 magic: bool = False,
                 magic_model: str = _MAGIC_DEFAULT_MODEL,
                 magic_seed: int | None = None,
                 magic_max_length: int = 100,
                 magic_temperature: float = 0.7,
                 attention: bool = False,
                 attention_min: int = 0.1,
                 attention_max: int = 0.9,
                 random: bool = False,
                 random_seed: int | None = None,
                 variations: int | None = None,
                 wildcards: str | None = None,
                 seed: int | None = None,
                 **kwargs
                 ):
        """
        :param kwargs: child class forwarded arguments
        """
        super().__init__(**kwargs)

        part = part.lower()

        if part not in {'both', 'positive', 'negative'}:
            raise self.argument_error(
                'Argument "part" must be one of: "both", "positive", or "negative"')

        if not magic and magic_seed is not None:
            raise self.argument_error(
                'Cannot specify "magic-seed" without "magic=True".'
            )

        if magic_max_length < 1:
            raise self.argument_error(
                'Cannot specify "magic-max-length" less than 1.'
            )

        if not random:
            if random_seed:
                raise self.argument_error(
                    'Cannot specify "random-seed" without "random=True".'
                )

            if variations is not None:
                raise self.argument_error(
                    'Cannot specify "variations" without "random=True".')

        if variations is not None and variations < 1:
            raise self.argument_error(
                'Argument "variations" may not be less than 1.')

        if wildcards:
            if not os.path.isdir(wildcards):
                raise self.argument_error(
                    'Argument "wildcards" must be a path to an exiting directory.'
                )

            wildcard_manager = _WildcardManager(wildcards)
        else:
            wildcard_manager = None

        if random:
            self._gen = _RandomPromptGenerator(
                wildcard_manager=wildcard_manager,
                seed=_types.default(random_seed, seed)
            )
        else:
            self._gen = _CombinatorialPromptGenerator(
                wildcard_manager=wildcard_manager
            )

        self._magic_gen = None

        if magic:
            magic_model_path = _types.default(magic_model, _MAGIC_DEFAULT_MODEL)

            self._load_magic_generator(
                magic_model_path=magic_model_path,
                magic_max_length=magic_max_length,
                magic_temperature=magic_temperature,
                magic_seed=_types.default(magic_seed, seed)
            )

            self._gen = self._magic_gen

        if attention:
            self._gen = _AttentionGenerator(
                self._gen,
                min_attention=attention_min,
                max_attention=attention_max
            )

        self._variations = variations
        self._part = part

    def _load_magic_generator(
            self,
            magic_model_path,
            magic_max_length,
            magic_temperature,
            magic_seed
    ):
        magic_generator_og_call = []
        magic_generator_model_generate_og_call = []

        def load_method():
            if self.local_files_only:
                found = huggingface_hub.try_to_load_from_cache(
                    magic_model_path, filename='model.safetensors')
                if found is None:
                    found = huggingface_hub.try_to_load_from_cache(
                        magic_model_path, filename='pytorch_model.bin')
                if not found:
                    raise self.argument_error(
                        f'Could not load "magic-model": {magic_model_path}, model not found in cache.')
            try:
                with _with_magic_load_pipeline_patch():
                    return _MagicPromptGenerator(
                        self._gen,
                        model_name=magic_model_path
                    )
            except Exception as e:
                raise self.argument_error(
                    'Could not load "magic-model": ' + str(e))

        def gen_patch(*args, **kwargs):
            kwargs.pop('max_length')
            kwargs.pop('temperature')
            return magic_generator_og_call[0](
                *args,
                max_length=magic_max_length,
                temperature=magic_temperature,
                generator=torch.Generator().manual_seed(magic_seed))

        def model_generate_patch(*args, **kwargs):
            input_ids = kwargs.pop('input_ids').to(self.device)
            attention_mask = kwargs.pop('attention_mask').to(self.device)
            result = magic_generator_model_generate_og_call[0](
                *args, input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            input_ids.to('cpu')
            attention_mask.to('cpu')
            return result

        model_files = list(
            _pipelinewrapper_util.fetch_model_files_with_size(
                magic_model_path,
                local_files_only=self.local_files_only,
                extensions={'.safetensors', '.bin'})
        )

        if len(model_files) > 1:
            model_files = [m for m in model_files if m[0].endswith('.safetensors')]

        if len(model_files) != 1:
            raise self.argument_error(f'Unable to estimate size of model: {magic_model_path}')

        estimated_size = model_files[0][1]

        _messages.debug_log(
            f'Estimated the size of MagicPromptGenerator model: '
            f'{magic_model_path}, as: {estimated_size} Bytes ({_memory.bytes_best_human_unit(estimated_size)})')

        self.set_size_estimate(estimated_size)

        self._magic_gen = self.load_object_cached(
            tag=magic_model_path,
            estimated_size=self.size_estimate,
            method=load_method
        )

        generator = self._magic_gen._generator

        if generator.__call__.__name__ != 'gen_patch':
            magic_generator_og_call.append(generator.__call__)
            generator.__call__ = gen_patch
        if generator.model.generate.__name__ != 'model_generate_patch':
            magic_generator_model_generate_og_call.append(generator.model.generate)
            generator.model.generate = model_generate_patch

    @contextlib.contextmanager
    def _with_magic_device(self):
        if self._magic_gen is not None:
            try:
                self.memory_guard_device(self.device, self.size_estimate)
                self._magic_gen._generator.model.to(self.device)
                yield
            finally:
                self._magic_gen._generator.model.to('cpu')
                gc.collect()
                _memory.torch_gc()
        else:
            yield

    def upscale(self, prompt: _prompt.Prompt) -> _prompt.PromptOrPrompts:
        """
        Upscale a prompt and return it modified.

        :param prompt: The incoming prompt.

        :return: Modified prompts, you may return multiple prompts to indicate expansion.
        """
        args = {}
        if self._variations is not None:
            args['num_images'] = self._variations

        try:
            with self._with_magic_device():
                if self._part in {'both', 'positive'}:
                    generated_pos_prompts = self._gen.generate(prompt.positive, **args)
                else:
                    generated_pos_prompts = [None]

                if prompt.negative and self._part in {'both', 'negative'}:
                    generated_neg_prompts = self._gen.generate(prompt.negative, **args)
                else:
                    generated_neg_prompts = [None]
        except pyparsing.ParseException as e:
            raise _exceptions.PromptUpscalerProcessingError(
                f'dynamicprompts prompt upscaler could '
                f'not parse prompt: "{prompt}", reason: {e}')

        output = []
        for generated_pos_prompt in generated_pos_prompts:

            for generated_neg_prompt in generated_neg_prompts:
                prompt_obj = _prompt.Prompt(
                    positive=_types.default(generated_pos_prompt, prompt.positive),
                    negative=_types.default(generated_neg_prompt, prompt.negative),
                    delimiter=prompt.delimiter
                )

                # We need to preserve the embedded diffusion
                # arguments from the original incoming prompt
                # that were parsed out by dgenerate
                prompt_obj.copy_embedded_args_from(prompt)

                # append the generated prompt to the expanded
                # output list of prompts
                output.append(prompt_obj)

        return output
