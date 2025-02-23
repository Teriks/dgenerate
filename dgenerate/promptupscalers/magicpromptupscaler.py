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
import typing

import dgenerate.pipelinewrapper.util as _pipelinewrapper_util
import dgenerate.promptupscalers.promptupscaler as _promptupscaler
import dgenerate.promptupscalers.exceptions as _exceptions
import dgenerate.prompt as _prompt
import dgenerate.types as _types
import dgenerate.memory as _memory
import dgenerate.messages as _messages

import re
import torch
import transformers


@contextlib.contextmanager
def _with_seed(seed: int | None):
    if seed is None:
        yield
    else:
        orig_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        try:
            yield
        finally:
            torch.random.set_rng_state(orig_state)


def _clean_up_magic_prompt(orig_prompt: str, prompt: str, remove_system: str, remove_preamble: str,
                           remove_prompt: bool) -> str:
    # remove the original prompt to keep it out of the MP fixes
    removed_prompt_prefix = False
    if prompt.startswith(orig_prompt):
        prompt = prompt[len(orig_prompt):]
        removed_prompt_prefix = True

    # old-style weight elevation
    prompt = prompt.translate(str.maketrans("{}", "()")).strip()

    # useless non-word characters at the begin/end
    prompt = re.sub(r"^\W+|\W+$", "", prompt)

    # clean up whitespace in weighted parens
    prompt = re.sub(r"\(\s+", "(", prompt)
    prompt = re.sub(r"\s+\)", ")", prompt)

    # clean up whitespace in hyphens between words
    prompt = re.sub(r"\b\s+\-\s+\b", "-", prompt)
    # other analogues to ', '
    prompt = re.sub(r"\s*[,;\.]+\s*(?=[a-zA-Z(])", ", ", prompt)
    # useless underscores between phrases
    prompt = re.sub(r"\s+_+\s+", " ", prompt)
    # empty phrases
    prompt = re.sub(r"\b,\s*,\s*\b", ", ", prompt)

    # Translate bangs into proper weight modifiers
    for match in re.findall(r"\b([\w\s\-]+)(\!+)", prompt):
        phrase = match[0]
        full_match = match[0] + match[1]
        weight = round(pow(1.1, len(match[1])), 2)

        prompt = prompt.replace(full_match, f"({phrase}:{weight})")

    # Put the original prompt back in
    if removed_prompt_prefix and not remove_prompt:
        prompt = f"{orig_prompt} {prompt}"

    if remove_system is not None:
        prompt = re.sub(
            r"<\|.*?\|>",
            " ", re.sub(r" <\|.*?\|> ", " ", prompt)).lstrip().removeprefix(remove_system).lstrip()

    if remove_preamble is not None:
        prompt = prompt.removeprefix(remove_preamble).lstrip()

    return prompt


class _MagicPromptGenerator:
    _blocklist_regex: re.Pattern | None = None

    def _load_pipeline(self, model_name: str) -> transformers.Pipeline:

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token_id = model.config.eos_token_id

        og_model_generate = model.generate

        def generator_patch(*args, **kwargs):
            args = list(args)

            for name, val in kwargs.items():
                if hasattr(val, 'to'):
                    kwargs[name] = val.to(self._device)

            for idx, val in enumerate(args):
                if hasattr(val, 'to'):
                    args[idx] = val.to(self._device)

            with _with_seed(self.seed):
                result = og_model_generate(*args, **kwargs)

            for val in args:
                if hasattr(val, 'to'):
                    val.to('cpu')

            for val in kwargs.values():
                if hasattr(val, 'to'):
                    val.to('cpu')

            return result

        model.generate = generator_patch

        return transformers.pipeline(
            task="text-generation",
            tokenizer=tokenizer,
            model=model,
            device='cpu',
            pad_token_id=tokenizer.eos_token_id,
        )

    def __init__(
            self,
            model_name: str = "Gustavosta/MagicPrompt-Stable-Diffusion",
            max_prompt_length: int = 100,
            temperature: float = 0.7,
            seed: int | None = None,
            blocklist_regex: str | None = None,
            batch_size: int = 1
    ) -> None:
        """
        :param model_name: The name of the model to use. Defaults to `"Gustavosta/MagicPrompt-Stable-Diffusion"`.
        :param max_prompt_length: The maximum length of the prompt to generate.
        :param temperature: The sampling temperature to use when generating prompts.
        :param seed: The seed to use when generating prompts.
        :param blocklist_regex: A regex to use to filter out prompts that match it.
        :param batch_size: The batch size to use when generating prompts.
        """
        self._model_name = model_name
        self.pipeline = None
        self.pipeline = self._load_pipeline(model_name)

        self.max_prompt_length = max_prompt_length
        self.temperature = float(temperature)
        self.seed = seed
        self.system = None
        self.preamble = None
        self.variations = batch_size
        self.remove_prompt = False

        if blocklist_regex:
            self._blocklist_regex = re.compile(blocklist_regex, re.IGNORECASE)
        else:
            self._blocklist_regex = None

        self._device = 'cpu'

    def generate(self, prompts: list[str] | str) -> list[str]:
        if not isinstance(prompts, list):
            prompts = [prompts]

        return self._generate_magic_prompts(prompts)

    def to(self, device: str | torch.device):
        self._device = device
        self.pipeline.model.to(device)

    def _regenerate_blocked_prompts(
            self,
            original_prompts: list[str],
            magic_prompts: list[str],
            max_attempts: int,
    ) -> list[str]:
        indexed_prompts_to_regenerate = []
        if self._blocklist_regex:
            for _ in range(max_attempts):
                indexed_prompts_to_regenerate = [
                    (i, prompt)
                    for i, prompt in enumerate(magic_prompts)
                    if self._blocklist_regex.search(prompt)
                ]

                if not indexed_prompts_to_regenerate:
                    break
                indexes = [x[0] for x in indexed_prompts_to_regenerate]
                prompts_to_regenerate = [original_prompts[index] for index in indexes]
                regenerated_prompts = self._generate_magic_prompts(
                    prompts_to_regenerate,
                )
                for i, prompt in zip(indexes, regenerated_prompts):
                    magic_prompts[i] = prompt

            if len(indexed_prompts_to_regenerate) > 0:
                _messages.log(
                    f"Could not generate magic prompts for "
                    f"{len(indexed_prompts_to_regenerate)} prompts after {max_attempts} attempts.",
                    level=_messages.WARNING
                )
        return magic_prompts

    def _generate_magic_prompts(self, orig_prompts: list[str]) -> list[str]:
        """
        Given a list of prompts, generate a list of magic prompts in a batch
        """

        def build_query(text):
            if self.preamble:
                return self.preamble + (' ' if not self.preamble.endswith(' ') else '') + text
            return text

        if self.system:
            orig_prompts = [
                f"<|system|> {self.system} <|user|> {build_query(query)} <|assistant|>"
                for query in orig_prompts
            ]
        else:
            orig_prompts = [build_query(query) for query in orig_prompts]

        orig_prompts *= self.variations

        prompts = self.pipeline(
            orig_prompts,
            max_length=self.max_prompt_length,
            temperature=self.temperature,
            do_sample=True,
            batch_size=len(orig_prompts),
        )

        prompts = [p[0]["generated_text"] for p in prompts]

        # Clean up the magic prompts
        prompts = [
            _clean_up_magic_prompt(
                orig_prompt,
                prompt,
                remove_system=self.system,
                remove_preamble=self.preamble,
                remove_prompt=self.remove_prompt
            )
            for orig_prompt, prompt in zip(orig_prompts, prompts)
        ]

        return prompts


class MagicPromptUpscaler(_promptupscaler.PromptUpscaler):
    """
    Upscale prompts using magicprompt or other LLMs.

    The "part" argument indicates which parts of the prompt to act on,
    possible values are: "both", "positive", and "negative"

    The "model" specifies the model path for magicprompt, the
    default value is: "Gustavosta/MagicPrompt-Stable-Diffusion". This
    can be a folder on disk or a Hugging Face repository slug.

    The "seed" argument can be used to specify a seed for prompt generation.

    The "variations" argument specifies how many variations should be produced.

    The "max-length" arguments the max prompt length for a magicprompt
    generated prompt, this value defaults to 100.

    The "temperature" argument sets the sampling temperature
    to use when generating prompts with magicprompt.

    The "system" set the system instruction for the LLM.

    The "preamble" set a text input preamble for the LLM, this
    preamble will be removed from the output generated by the LLM.

    The "remove-prompt" remove the original prompt
    from the generated text?
    """

    NAMES = ['magicprompt']

    def __init__(self,
                 part: str = 'both',
                 model: str = "Gustavosta/MagicPrompt-Stable-Diffusion",
                 seed: int | None = None,
                 variations: int = 1,
                 max_length: int = 100,
                 temperature: float = 0.7,
                 system: str | None = None,
                 preamble: str | None = None,
                 remove_prompt: bool = False,
                 **kwargs
                 ):
        """
        :param kwargs: child class forwarded arguments
        """
        super().__init__(**kwargs)

        part = part.lower()

        if part not in {'both', 'positive', 'negative'}:
            raise self.argument_error(
                'Argument "part" must be one of: "both", "positive", or "negative"'
            )

        if max_length < 1:
            raise self.argument_error(
                'Cannot specify "magic-max-length" less than 1.'
            )

        if variations < 1:
            raise self.argument_error(
                'Argument "variations" may not be less than 1.')

        model_files = list(
            _pipelinewrapper_util.fetch_model_files_with_size(
                model,
                local_files_only=self.local_files_only,
                extensions={'.safetensors', '.bin'})
        )

        if len(model_files) > 1:
            model_files = [m for m in model_files if m[0].endswith('.safetensors')]

        estimated_size = 0

        for model_entry in model_files:
            estimated_size += model_entry[1]

        _messages.debug_log(
            f'Estimated the size of LLM model: '
            f'{model}, as: {estimated_size} Bytes ({_memory.bytes_best_human_unit(estimated_size)})')

        def load_method():
            return _MagicPromptGenerator(model)

        self.set_size_estimate(estimated_size)

        self._gen = self.load_object_cached(
            tag=model,
            estimated_size=estimated_size,
            method=load_method
        )

        self._gen.system = system
        self._gen.preamble = preamble
        self._gen.remove_prompt = remove_prompt
        self._gen.seed = seed
        self._gen.max_prompt_length = max_length
        self._gen.variations = variations
        self._gen.temperature = temperature

        self._part = part

    @contextlib.contextmanager
    def _with_magic_settings(self):
        try:
            self.memory_guard_device(self.device, self.size_estimate)
            self._gen.to(self.device)
            yield
        finally:
            self._gen.to('cpu')
            gc.collect()
            _memory.torch_gc()

    @property
    def accepts_batch(self):
        """
        This prompt upscaler can accept a batch of prompts for efficient execution.
        :return: ``True``
        """
        return True

    def upscale(self, prompt: _prompt.Prompts) -> _prompt.PromptOrPrompts:
        with self._with_magic_settings():
            if self._part in {'both', 'positive'}:
                generated_pos_prompts = self._gen.generate([p.positive for p in prompt])
            else:
                generated_pos_prompts = [None] * len(prompt)

            if self._part in {'both', 'negative'}:
                generated_neg_prompts = self._gen.generate([p.negative for p in prompt])
            else:
                generated_neg_prompts = [None] * len(prompt)

        output = []
        for idx, (generated_pos_prompt, generated_neg_prompt) in enumerate(zip(generated_pos_prompts, generated_neg_prompts)):
            orig_idx = idx % len(prompt)
            orig_prompt_pos = generated_pos_prompt
            orig_prompt_neg = generated_neg_prompt

            prompt_obj = _prompt.Prompt(
                positive=_types.default(orig_prompt_pos, prompt[orig_idx].positive),
                negative=_types.default(orig_prompt_neg, prompt[orig_idx].negative),
                delimiter=prompt[orig_idx].delimiter
            )

            # We need to preserve the embedded diffusion
            # arguments from the original incoming prompt
            # that were parsed out by dgenerate
            prompt_obj.copy_embedded_args_from(prompt[orig_idx])

            # append the generated prompt to the expanded
            # output list of prompts
            output.append(prompt_obj)

        return output