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
import random
import collections.abc

import pyparsing

import dgenerate.promptupscalers.promptupscaler as _promptupscaler
import dgenerate.promptupscalers.exceptions as _exceptions
import dgenerate.prompt as _prompt
import dgenerate.types as _types
import dgenerate.spacycache as _spacycache


class AttentionUpscaler(_promptupscaler.PromptUpscaler):
    """
    Add random attention values to your prompt tokens.

    This is ment for use with --prompt-weighter plugins such as "sd-embed" or "compel"

    The "part" argument indicates which parts of the prompt to act on,
    possible values are: "both", "positive", and "negative"

    The "min" argument sets the minimum value for random
    attention added. The default value is 0.1

    The "max" argument sets the maximum value for random
    attention added. The Default value is 0.9

    The "seed" argument can be used to specify a seed for
    the random attenuation values that are added to your prompt.

    The "lang" argument can be used to specify the prompt language,
    the default value is 'en' for english, this can be one of: 'en',
    'de', 'fr', 'es', 'it', 'nl', 'pt', 'ru', 'zh'.

    The "syntax" argument specifies the token attention value syntax,
    this can be one of "sd-embed" (SD Web UI Syntax) or "compel"
    (InvokeAI Syntax).
    """

    NAMES = ['attention']

    HIDE_ARGS = ['device']

    OPTION_ARGS = {
        'part': ['both', 'positive', 'negative'],
        'lang': ['en', 'de', 'fr', 'es', 'it', 'nl', 'pt', 'ru', 'zh'],
        'syntax': ['sd-embed', 'compel']
    }

    # these are languages where noun chunking is supported
    _langs = ['en', 'de', 'fr', 'es', 'it', 'nl', 'pt', 'ru', 'zh']

    def __init__(self,
                 part: str = 'both',
                 min: int = 0.1,
                 max: int = 0.9,
                 seed: int | None = None,
                 lang: str = 'en',
                 syntax: str = 'sd-embed',
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

        lang = lang.lower()
        if lang not in self._langs:
            raise self.argument_error(
                f'Argument "lang" must be one of: {", ".join(self._langs)}')

        syntax = syntax.lower()
        if syntax not in {'sd-embed', 'compel'}:
            raise self.argument_error(
                'Argument "syntax" must be one of: sd-embed, or compel')

        self._min, self._max = sorted(
            (min, max),
        )

        self._part = part
        self._syntax = syntax
        self._rng = random.Random(seed)

        try:
            self._nlp = _spacycache.load_spacy_model(
                f"{lang.lower()}_core_web_sm", local_files_only=self.local_files_only
            )
        except _spacycache.SpacyModelNotFoundError as e:
            raise self.argument_error(f'Could not load spaCy model: {e}') from e

    def _find_noun_chunks(self, text: str) -> list[str]:
        return [str(chunk) for chunk in self._nlp(text).noun_chunks]

    def _generate(self, prompt: str):
        keywords = self._find_noun_chunks(prompt)
        if not keywords:
            return prompt

        keyword = random.choice(keywords)
        attention = round(self._rng.uniform(self._min, self._max), 2)

        if self._syntax == 'sd-embed':
            return prompt.replace(str(keyword), f"({keyword}:{attention})")
        else:
            return prompt.replace(str(keyword), f"({keyword}){attention}")

    def upscale(self, prompt: _prompt.Prompt) -> _prompt.PromptOrPrompts:

        if isinstance(prompt, collections.abc.Iterable):
            raise _exceptions.PromptUpscalerProcessingError(
                'attention prompt upscaler cannot handle batch prompt input.'
            )

        try:
            if prompt.positive and self._part in {'both', 'positive'}:
                generated_pos_prompts = [self._generate(prompt.positive)]
            else:
                generated_pos_prompts = [None]

            if prompt.negative and self._part in {'both', 'negative'}:
                generated_neg_prompts = [self._generate(prompt.negative)]
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
