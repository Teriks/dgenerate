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

import typing
import dgenerate.prompt as _prompt


def process_prompts_batched(
        prompts: _prompt.Prompts,
        part: str,
        generator: typing.Callable[[list[str]], list[str]]
) -> list[_prompt.Prompt]:
    """
    Process a list of prompts using a text generation function which can accept a batch of strings.

    This handles processing of the positive and negative prompt selectively, and
    reconstruction of the prompt objects after processing.

    :param prompts: Input prompts.
    :param part: Prompt parts to process, "both", "positive", "negative".
    :param generator: The text processing function, should accept a list of strings, and return a list of strings
    :return: List of generated prompt objects.
    """

    generated_pos_prompts = [p.positive for p in prompts]
    generated_neg_prompts = [p.negative for p in prompts]

    if part in {'both', 'positive'}:
        non_empty_idx = [idx for idx, p in enumerate(prompts) if p.positive]

        pos_prompts = [p.positive for p in prompts if p.positive]

        if pos_prompts:
            generated = generator(pos_prompts)

            for idx, non_empty_idx in enumerate(non_empty_idx):
                generated_pos_prompts[non_empty_idx] = generated[idx]

    if part in {'both', 'negative'}:
        non_empty_idx = [idx for idx, p in enumerate(prompts) if p.negative]

        neg_prompts = [p.negative for p in prompts if p.negative]

        if neg_prompts:
            generated = generator(neg_prompts)

            for idx, non_empty_idx in enumerate(non_empty_idx):
                generated_neg_prompts[non_empty_idx] = generated[idx]

    output = []
    for idx, (generated_pos_prompt, generated_neg_prompt) in \
            enumerate(zip(generated_pos_prompts, generated_neg_prompts)):
        prompt_obj = _prompt.Prompt(
            positive=generated_pos_prompt,
            negative=generated_neg_prompt,
            delimiter=prompts[idx].delimiter
        )

        # We need to preserve the embedded diffusion
        # arguments from the original incoming prompt
        # that were parsed out by dgenerate
        prompt_obj.copy_embedded_args_from(prompts[idx])

        # append the generated prompt to the expanded
        # output list of prompts
        output.append(prompt_obj)

    return output
