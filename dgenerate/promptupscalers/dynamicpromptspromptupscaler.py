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
import os.path
import typing

import dgenerate.promptupscalers.promptupscaler as _promptupscaler
import dgenerate.prompt as _prompt
from dynamicprompts.generators import CombinatorialPromptGenerator as _CombinatorialPromptGenerator
from dynamicprompts.generators import RandomPromptGenerator as _RandomPromptGenerator
from dynamicprompts.generators.magicprompt import MagicPromptGenerator as _MagicPromptGenerator
from dynamicprompts.generators.magicprompt import DEFAULT_MODEL_NAME as _MAGIC_DEFAULT_MODEL
from dynamicprompts.wildcards.wildcard_manager import WildcardManager as _WildcardManager
from dynamicprompts.generators.attentiongenerator import AttentionGenerator as _AttentionGenerator
import dgenerate.types as _types


class DynamicPromptsPromptUpscaler(_promptupscaler.PromptUpscaler):
    """
    Upscale prompts with the dynamicprompts library.

    See: https://github.com/adieyal/dynamicprompts

    The "magic" argument enables magicprompt, which generates a continuation
    of your prompt using the model "Gustavosta/MagicPrompt-Stable-Diffusion".

    The "magic_model" specifies the model path for magicprompt, the
    default value is: "Gustavosta/MagicPrompt-Stable-Diffusion". This
    can be a folder on disk or a Hugging Face repository slug.
    
    The "magic_seed" argument can be used to specify a seed for just the
    "magic" prompt generation, this overrides "seed".
    
    The "magic_max_length" arguments the max prompt length for a magicprompt
    generated prompt, this value defaults to 100.

    The "attention" argument enables random token attention values,
    this requires the use of the "sd-embed" prompt weighter or "compel" in
    SD Web UI syntax mode, i.e. "compel;syntax=sdwui"
    
    The "attention_min" argument sets the minimum value for random
    attention added by "attention=True". The default value is 0.1
    
    The "attention_max" argument sets the maximum value for random
    attention added by "attention=True" The Default value is 0.9

    The "random" argument specifies that instead of strictly combinatorial
    output, dynamicprompts should produce N random variations of you
    prompt given the possibilities you have provided.
    
    The "random_seed" argument can be used to specify a seed for just the
    "random" prompt generation, this overrides "seed".

    The "variations" argument specifies how many variations should
    be produced when "random" is set to true. This argument cannot
    be used without specifying "random".

    The "wildcards" argument can be used to specify a wildcards directory
    for dynamicprompt's wildcard syntax.

    The "seed" argument can be used to specify a seed for the "random" prompt
    generation as well as the "magic" prompt generation simultaneously. The
    same seed will be used for both generators.
    """

    NAMES = ['dynamicprompts']

    def __init__(self,
                 magic: bool = False,
                 magic_model: str = _MAGIC_DEFAULT_MODEL,
                 magic_seed: typing.Optional[int] = None,
                 magic_max_length: int = 100,
                 attention: bool = False,
                 attention_min: int = 0.1,
                 attention_max: int = 0.9,
                 random: bool = False,
                 random_seed: typing.Optional[int] = None,
                 variations: typing.Optional[int] = None,
                 wildcards: typing.Optional[str] = None,
                 seed: typing.Optional[int] = None,
                 **kwargs
                 ):
        """
        :param kwargs: child class forwarded arguments
        """
        super().__init__(**kwargs)

        if not magic and magic_seed is not None:
            raise self.argument_error(
                'Prompt upscaler dynamicprompts cannot specify '
                '"magic_seed" without "magic=True".'
            )

        if magic_max_length < 1:
            raise self.argument_error(
                'Prompt upscaler dynamicprompts cannot specify '
                '"magic_max_length" less than 1.'
            )

        if not random:
            if random_seed:
                raise self.argument_error(
                    'Prompt upscaler dynamicprompts cannot specify '
                    '"random_seed" without "random=True".'
                )

            if variations is not None:
                raise self.argument_error(
                    'Prompt upscaler dynamicprompts cannot specify '
                    '"variations" without "random=True".')

        if variations is not None and variations < 1:
            raise self.argument_error(
                'Prompt upscaler dynamicprompts "variations" '
                'may not be less than 1.')

        self._variations = variations

        if wildcards:
            if not os.path.isdir(wildcards):
                raise self.argument_error(
                    'Prompt upscaler dynamicprompts "wildcards" '
                    'argument must be a path to an exiting directory.'
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
            if seed is not None:
                raise self.argument_error(
                    'Prompt upscaler dynamicprompts cannot utilize '
                    'the "seed" argument in combinatorial mode.')

            self._gen = _CombinatorialPromptGenerator(
                wildcard_manager=wildcard_manager
            )

        if magic:
            self._gen = _MagicPromptGenerator(
                self._gen,
                model_name=_types.default(magic_model, _MAGIC_DEFAULT_MODEL),
                device=self.device,
                seed=_types.default(magic_seed, seed),
                max_prompt_length=magic_max_length

            )

        if attention:
            self._gen = _AttentionGenerator(
                self._gen,
                min_attention=attention_min,
                max_attention=attention_max
            )

    # noinspection PyUnboundLocalVariable
    def upscale(self, prompt: _prompt.Prompt) -> _prompt.Prompts:
        """
        Upscale a prompt and return it modified.

        :param prompt: The incoming prompt.

        :return: Modified prompts, you may return multiple prompts to indicate expansion.
        """
        args = {}
        if self._variations is not None:
            args['num_images'] = self._variations

        generated_prompts = self._gen.generate(str(prompt), **args)

        output = []
        for generated_prompt in generated_prompts:
            # Reparse the generated prompt into an object
            # dgenerate understands
            prompt_obj = _prompt.Prompt.parse(generated_prompt)

            # We need to preserve the embedded diffusion
            # arguments from the original incoming prompt
            # that were parsed out by dgenerate
            prompt_obj.copy_embedded_args_from(prompt)

            # append the generated prompt to the expanded
            # output list of prompts
            output.append(prompt_obj)

        return output
