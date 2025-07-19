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
import collections.abc

import pyparsing
import dgenerate.promptupscalers.promptupscaler as _promptupscaler
import dgenerate.promptupscalers.exceptions as _exceptions
import dgenerate.prompt as _prompt

from dynamicprompts.generators import CombinatorialPromptGenerator as _CombinatorialPromptGenerator
from dynamicprompts.generators import RandomPromptGenerator as _RandomPromptGenerator
from dynamicprompts.wildcards.wildcard_manager import WildcardManager as _WildcardManager
import dgenerate.types as _types


class DynamicPromptsUpscaler(_promptupscaler.PromptUpscaler):
    """
    Upscale prompts with the dynamicprompts library.

    This upscaler allows you to use a special syntax for
    combinatorial prompt variations.

    See: https://github.com/adieyal/dynamicprompts

    The "part" argument indicates which parts of the prompt to act on,
    possible values are: "both", "positive", and "negative"

    The "random" argument specifies that instead of strictly combinatorial
    output, dynamicprompts should produce N random variations of your
    prompt given the possibilities you have provided.

    The "seed" argument can be used to specify a seed for
    the "random" prompt generation.

    The "variations" argument specifies how many variations should
    be produced when "random" is set to true. This argument cannot
    be used without specifying "random". The default value is 1.

    The "wildcards" argument can be used to specify a wildcards directory
    for dynamicprompts wildcard syntax.
    """

    NAMES = ['dynamicprompts']

    HIDE_ARGS = ['device']

    OPTION_ARGS = {
        'part': ['both', 'positive', 'negative']
    }

    def __init__(self,
                 part: str = 'both',
                 random: bool = False,
                 seed: int | None = None,
                 variations: int | None = None,
                 wildcards: str | None = None,
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

        if not random:
            if seed:
                raise self.argument_error(
                    'Cannot specify "seed" without "random=True".'
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
                seed=seed
            )
        else:
            self._gen = _CombinatorialPromptGenerator(
                wildcard_manager=wildcard_manager
            )

        self._seed = seed
        self._variations = variations
        self._part = part

    def upscale(self, prompt: _prompt.Prompt) -> _prompt.PromptOrPrompts:

        if isinstance(prompt, collections.abc.Iterable):
            raise _exceptions.PromptUpscalerProcessingError(
                'dynamicprompts prompt upscaler cannot handle batch prompt input.'
            )

        args = {}
        if self._variations is not None:
            args['num_images'] = self._variations

        try:
            if prompt.positive and self._part in {'both', 'positive'}:
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
                f'not parse prompt: "{prompt}", reason: {e}') from e

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
