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

import re


def clean_up_llm_prompt(orig_prompt: str,
                        generated_prompt: str | None,
                        remove_system: str | None,
                        remove_preamble: str | None,
                        remove_prompt: bool) -> str:
    # remove the original prompt to keep it out of the MP fixes
    removed_prompt_prefix = False
    if generated_prompt.startswith(orig_prompt):
        generated_prompt = generated_prompt[len(orig_prompt):]
        removed_prompt_prefix = True

    # get rid of special tokens
    generated_prompt = re.sub(
        r"</?[|/]?.*?[|/]?/?[>)]",
        " ", re.sub(r" </?[|/]?.*?[|/]?/?[>)] ", " ", generated_prompt)).strip()

    # half baked special token
    generated_prompt = re.sub(
        r"</?[|/]?[a-zA-Z-_]+", " ", generated_prompt
    ).strip()

    # old-style weight elevation
    generated_prompt = generated_prompt.translate(str.maketrans("{}", "()")).strip()

    # useless non-word characters at the begin/end
    generated_prompt = re.sub(r"^\W+|\W+$", "", generated_prompt)

    # clean up whitespace in weighted parens
    generated_prompt = re.sub(r"\(\s+", "(", generated_prompt)
    generated_prompt = re.sub(r"\s+\)", ")", generated_prompt)

    # clean up whitespace in hyphens between words
    generated_prompt = re.sub(r"\b\s+\-\s+\b", "-", generated_prompt)
    # other analogues to ', '
    generated_prompt = re.sub(r"\s*[,;\.]+\s*(?=[a-zA-Z(])", ", ", generated_prompt)
    # useless underscores between phrases
    generated_prompt = re.sub(r"\s+_+\s+", " ", generated_prompt)
    # empty phrases
    generated_prompt = re.sub(r"\b,\s*,\s*\b", ", ", generated_prompt)

    # Translate bangs into proper weight modifiers
    for match in re.findall(r"\b([\w\s\-]+)(\!+)", generated_prompt):
        phrase = match[0]
        full_match = match[0] + match[1]
        weight = round(pow(1.1, len(match[1])), 2)

        generated_prompt = generated_prompt.replace(full_match, f"({phrase}:{weight})")

    # Put the original prompt back in
    if removed_prompt_prefix and not remove_prompt:
        generated_prompt = f"{orig_prompt} {generated_prompt}"

    if remove_system is not None:
        generated_prompt = generated_prompt.removeprefix(remove_system).lstrip()

    if remove_preamble is not None:
        generated_prompt = generated_prompt.removeprefix(remove_preamble).lstrip()

    return ' '.join(generated_prompt.split())
