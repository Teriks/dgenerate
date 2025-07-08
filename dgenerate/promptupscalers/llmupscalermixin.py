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
import abc
import importlib.util
import json
import os
import re
import typing

import spacy.pipeline
import toml
import yaml

import dgenerate.messages as _messages
import dgenerate.prompt as _prompt
import dgenerate.promptupscalers.util as _util
import dgenerate.spacycache as _spacycache


class LLMPromptUpscalerMixin(abc.ABC):
    def __init__(self,
                 part: str,
                 block_regex: str,
                 max_attempts: int,
                 cleanup_mode: str,
                 cleanup_config: str,
                 smart_truncate: bool = False,
                 **kwargs):
        """
        :param part: Prompt part to be processed, "both", "positive", "negative"
        :param block_regex: Regenerated prompts that match this regex.
        :param max_attempts: Max regeneration attempts.
        :param cleanup_mode: "magic", or anything else currently.
        :param cleanup_config: Path to .json, .toml, or .yaml cleanup config.
        :param kwargs: Mixin args.
        """

        super().__init__(**kwargs)

        self._blocklist_regex = None

        if block_regex:
            try:
                self._blocklist_regex = re.compile(block_regex, re.IGNORECASE)
            except Exception as e:
                raise self.argument_error(f'Could not compile "block-regex": {e}') from e
            else:
                self._blocklist_regex = None

        if part not in {'both', 'positive', 'negative'}:
            raise self.argument_error(
                'Argument "part" must be one of: "both", "positive", or "negative"'
            )

        if max_attempts < 1:
            raise self.argument_error(
                'Argument "max-attempts" may not be less than 1.')

        self._max_attempts = max_attempts
        self._part = part
        self._cleanup_mode = cleanup_mode
        self._smart_truncate = smart_truncate

        self._custom_cleanup_operations = None

        if cleanup_config:
            cleanup_config_operations = self._load_custom_cleanup_operations(cleanup_config)
            self._validate_custom_cleanup_operations(cleanup_config_operations)
            self._custom_cleanup_operations = cleanup_config_operations
            self._cleanup_config_path = cleanup_config

    def _validate_custom_cleanup_operations(self, operations: list[dict[str, typing.Any]]) -> bool:
        for operation in operations:
            if 'function' in operation:
                if not isinstance(operation['function'], str) or ':' not in operation['function']:
                    raise self.argument_error(
                        f'Argument "cleanup-config", config contains invalid function format: {operation["function"]}')

                module_path, function_name = operation['function'].split(':')

                if not os.path.isfile(module_path):
                    raise self.argument_error(
                        f'Argument "cleanup-config", config module file not found: {module_path}')
            elif 'pattern' in operation and 'substitution' in operation:
                if not isinstance(operation['pattern'], str) or not isinstance(operation['substitution'], str):
                    raise self.argument_error(
                        'Argument "cleanup-config", config pattern and substitution must be strings')
            else:
                raise self.argument_error(
                    'Argument "cleanup-config", each config operation must contain '
                    'either "function" or both "pattern" and "substitution"')
        return True

    def _load_custom_cleanup_module(self, module_path: str, cache: dict[str, typing.Any]):
        if module_path in cache:
            return cache[module_path]

        if not os.path.isabs(module_path):
            base_dir = os.path.dirname(os.path.abspath(self._cleanup_config_path))
            module_path = os.path.abspath(os.path.join(base_dir, module_path))

        module_name = os.path.splitext(os.path.basename(module_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)

        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cache[module_path] = module
            return module
        raise self.argument_error(
            f'Argument "cleanup-config", could not load module mentioned in config: {module_path}')

    def _execute_custom_cleanup_operations(self, text: str) -> str:
        if not self._custom_cleanup_operations:
            return text

        module_cache = {}

        for operation in self._custom_cleanup_operations:
            if 'function' in operation:
                module_path, function_name = operation['function'].split(':')
                module = self._load_custom_cleanup_module(module_path, module_cache)
                if not hasattr(module, function_name):
                    raise self.argument_error(
                        f'Argument "cleanup-config", config module '
                        f'"{module_path}" has no function "{function_name}"')

                func = getattr(module, function_name)
                if not callable(func):
                    raise self.argument_error(
                        f'Argument "cleanup-config", cleanup function '
                        f'"{function_name}" in "{module_path}" is not callable')

                text = func(text)
            elif 'pattern' in operation and 'substitution' in operation:
                flags = 0
                if (operation.get('ignore_case', False) or
                    operation.get('ignorecase', False)):
                    flags |= re.IGNORECASE
                if operation.get('multiline', False):
                    flags |= re.MULTILINE
                if operation.get('dotall', False):
                    flags |= re.DOTALL

                text = re.sub(
                    pattern=operation['pattern'],
                    repl=operation['substitution'],
                    string=text,
                    count=operation.get('count', 0),
                    flags=flags
                )

        return text

    def _load_custom_cleanup_operations(self, file_path: str) -> list[dict[str, typing.Any]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            elif file_path.endswith('.toml'):
                try:
                    return toml.load(f)['operations']
                except KeyError:
                    raise self.argument_error(
                        'Argument "cleanup-config", .toml does not contain any [[operations]]'
                    )
            elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                raise self.argument_error(
                    'Argument "cleanup-config", unsupported file format. Use .json, .toml, or .yaml/.yml')

    @abc.abstractmethod
    def _generate_prompts(self, orig_prompts: list[str]):
        pass

    def _generate(self, prompts: list[str] | str) -> list[str]:
        if not isinstance(prompts, list):
            prompts = [prompts]

        generated_prompts = self._generate_prompts(prompts)
        return self._regenerate_blocked_prompts(prompts, generated_prompts, self._max_attempts)

    def _process_prompts(self, prompts: _prompt.Prompts) -> _prompt.Prompts:
        return _util.process_prompts_batched(prompts, self._part, self._generate)

    def _regenerate_blocked_prompts(
            self,
            original_prompts: list[str],
            generated_prompts: list[str],
            max_attempts: int,
    ) -> list[str]:
        indexed_prompts_to_regenerate = []
        if self._blocklist_regex:
            for _ in range(max_attempts):
                indexed_prompts_to_regenerate = [
                    (i, prompt)
                    for i, prompt in enumerate(generated_prompts)
                    if self._blocklist_regex.search(prompt)
                ]

                if not indexed_prompts_to_regenerate:
                    break
                indexes = [x[0] for x in indexed_prompts_to_regenerate]
                prompts_to_regenerate = [original_prompts[index] for index in indexes]
                regenerated_prompts = self._generate_prompts(
                    prompts_to_regenerate,
                )
                for i, prompt in zip(indexes, regenerated_prompts):
                    generated_prompts[i] = prompt

            if len(indexed_prompts_to_regenerate) > 0:
                _messages.warning(
                    f"Could not generate prompts for "
                    f"{len(indexed_prompts_to_regenerate)} prompts after {max_attempts} attempts."
                )
        return generated_prompts

    def _load_smart_truncate_model(self):
        nlp = _spacycache.load_spacy_model(
            "xx_ent_wiki_sm", local_files_only=getattr(self, 'local_files_only', False)
        )

        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer", first=True)

        return nlp

    def _do_smart_truncate(self, text: str):

        nlp = self._load_smart_truncate_model()

        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]

        end_punctuation = tuple(spacy.pipeline.Sentencizer.default_punct_chars)

        if sentences and not sentences[-1].strip().endswith(end_punctuation):
            sentences.pop()

        result = ' '.join(sentences)

        if not result.strip():
            _messages.warning('smart-truncate resulted in empty prompt, using original text.')
            return text

        return result

    def _clean_prompt(self,
                      formatted_prompt: str,
                      generated_prompt: str | None,
                      remove_prefixes: list[str] | None,
                      remove_prompt: bool,
                      prepend: str | None) -> str:

        removed_prompt_prefix = False
        if generated_prompt.startswith(formatted_prompt):
            # if the model regurgitates our prompt, i.e. returns
            # our input text with its own additions following, we do not
            # want to clean up the regurgitated text since we know it
            # is already fine.
            generated_prompt = generated_prompt[len(formatted_prompt):]
            removed_prompt_prefix = True

        # rid the input prompt of ChatML tokens
        formatted_prompt = re.sub(
            r"</?[|/]?.*?[|/]?/?[>)]",
            " ", re.sub(r" </?[|/]?.*?[|/]?/?[>)] ", " ", formatted_prompt)).strip()

        # get rid of special tokens from the LLM output
        generated_prompt = re.sub(
            r"</?[|/]?.*?[|/]?/?[>)]",
            " ", re.sub(r" </?[|/]?.*?[|/]?/?[>)] ", " ", generated_prompt)).strip()

        # half-baked special tokens due to truncation or hallucination
        generated_prompt = re.sub(
            r"</?[|/]?[a-zA-Z-_]+", " ", generated_prompt
        ).strip()

        if self._smart_truncate:
            generated_prompt = self._do_smart_truncate(generated_prompt)

        if self._cleanup_mode == 'magic' and not self._custom_cleanup_operations:
            # MagicPrompt output specific cleanup

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

        generated_prompt = self._execute_custom_cleanup_operations(generated_prompt)

        # Put the original prompt back in if removed it from
        # the model output, and there was no explicit request
        # for it to be removed forcefully
        if removed_prompt_prefix and not remove_prompt:
            # since the model generated this continuation on its own,
            # the punctuation is probably grammatically correct, so
            # just go with it
            spacer = ' '
            if generated_prompt and generated_prompt[0] in {'.', ';', ','}:
                spacer = ''
            generated_prompt = f"{formatted_prompt}{spacer}{generated_prompt}"

        if remove_prefixes is not None:
            for prefix in remove_prefixes:
                if prefix:
                    generated_prompt = generated_prompt.removeprefix(prefix).lstrip()

        if prepend:
            prepend = prepend.strip()

            if not generated_prompt.startswith(prepend):
                spacer = ' '

                # if the generated prompt starts with a punctuation, think
                # about keeping it, there is probably a space after it
                if generated_prompt and generated_prompt[0] in {'.', ';', ','}:
                    spacer = ''

                # if the generated prompt has punctuation, but the prepended
                # text ends with punctuation, prefer the punctuation from
                # the prepended text
                if spacer == '' and prepend[-1] in {'.', ';', ','}:
                    generated_prompt = prepend + generated_prompt[1:]
                else:
                    generated_prompt = prepend + spacer + generated_prompt

        if generated_prompt and generated_prompt[0] in {'.', ';', ','}:
            # if after all that the prompt starts with a punctuation and
            # then presumably space, remove it
            generated_prompt = generated_prompt[1:].lstrip()

        # normalize whitespace between tokens to a single space
        return ' '.join(generated_prompt.split())
