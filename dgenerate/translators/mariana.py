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
import importlib.resources
import json
import huggingface_hub
import torch
import transformers

import dgenerate.messages as _messages
import dgenerate.translators.exceptions as _exceptions
import dgenerate.translators.util as _util


class MarianaTranslator:
    """
    Translate languages locally using Helsinki-NLP opus models on the CPU or GPU.

    Supports automatic pivot language selection.
    """

    _translation_map = None
    _offline_mode = False

    def __init__(self, from_lang: str, to_lang: str, local_files_only: bool = False):
        """
        :param from_lang: From language code (IETF), or language name.
        :param to_lang: To language code (IETF), or language name.
        :param local_files_only: Only use models that have been previously cached?
        :raise dgenerate.translators.TranslatorLoadError: If models cannot be loaded / found.
        """

        norm_from_lang = _util.get_language_code(from_lang)

        if norm_from_lang is None:
            raise _exceptions.TranslatorLoadError(
                f'Invalid "from" language / language code: {from_lang}')

        norm_to_lang = _util.get_language_code(to_lang)

        if norm_to_lang is None:
            raise _exceptions.TranslatorLoadError(
                f'Invalid "to" language / language code: {to_lang}')

        from_lang = norm_from_lang
        to_lang = norm_to_lang

        if MarianaTranslator._translation_map is None:
            with importlib.resources.open_text(
                    'dgenerate.translators.data',
                    'helsinki-nlp-translation-map.json') as translation_map:
                MarianaTranslator._translation_map = json.load(translation_map)

        # search by priority
        pivot_lang_codes = ['en', 'es', 'fr', 'de', 'it', 'pt']

        model_name = None
        pivot_model_name = None
        pivot_model_to = None

        self.tokenizer = None
        self.tokenizer2 = None

        self.model = None
        self.model2 = None

        try:
            model_name = MarianaTranslator._translation_map[from_lang][to_lang]
        except KeyError:
            for code in pivot_lang_codes:
                try:
                    pivot_model_name = MarianaTranslator._translation_map[from_lang][code]
                    pivot_model_to = code
                    break
                except KeyError:
                    continue

        if pivot_model_name:
            # we located a suitable pivot model

            try:
                # This model will be used to translate from the pivot language to the target language
                model_name = MarianaTranslator._translation_map[pivot_model_to][to_lang]
            except KeyError:
                # cannot translate out of the pivot language to the target language
                raise _exceptions.TranslatorLoadError(
                    f'Helsinki-NLP translation model for "{from_lang}" -> "{to_lang}" is not available.')

            _messages.debug_log(
                f'Helsinki-NLP (mariana): using pivot '
                f'{from_lang} -> {pivot_model_to} then {pivot_model_to} -> {to_lang}'
            )

            self.tokenizer, self.model = self._load_mariana(
                pivot_model_name, from_lang, to_lang, local_files_only
            )

            self.tokenizer2, self.model2 = self._load_mariana(
                model_name, from_lang, to_lang, local_files_only
            )

        elif model_name:
            self.tokenizer, self.model = self._load_mariana(
                model_name, from_lang, to_lang, local_files_only
            )
        else:
            raise _exceptions.TranslatorLoadError(
                f'Helsinki-NLP translation model for "{from_lang}" -> "{to_lang}" is not available.')

    @staticmethod
    def _load_mariana(model_name: str, from_lang: str, to_lang: str, local_files_only: bool):
        if MarianaTranslator._offline_mode:
            local_files_only = True

        if local_files_only:
            # If we are in offline mode, we need to ensure the model is cached
            if huggingface_hub.try_to_load_from_cache(
                model_name, 'config.json',
            ) is None:
                raise _exceptions.TranslatorLoadError(
                    f'Helsinki-NLP translation model for "{from_lang}" -> "{to_lang}" '
                    f'cannot be loaded in offline mode as it has not been cached.')

        try:
            tokenizer = transformers.MarianTokenizer.from_pretrained(
                model_name, local_files_only=local_files_only)
            model = transformers.MarianMTModel.from_pretrained(
                model_name, local_files_only=local_files_only)
            return tokenizer, model
        except OSError as e:
            raise _exceptions.TranslatorLoadError(
                f'Helsinki-NLP translation model for "{from_lang}" -> "{to_lang}" is not available.') from e
        except Exception as e:
            raise _exceptions.TranslatorLoadError(e) from e

    def to(self, device: str | torch.device):
        """
        Move the model(s) to a specific device.
        :param device: The device
        :return: self
        """
        self.model.to(device)
        if self.model2 is not None:
            self.model2.to(device)
        return self

    def translate(self, texts: str | list[str]) -> list[str]:
        """
        Translate a list of texts.
        :param texts: Texts to translate.
        :return: Translated texts.
        """

        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        translated = self.model.generate(**inputs)
        first_pass = self.tokenizer.batch_decode(translated, skip_special_tokens=True)

        if self.tokenizer2:
            inputs = self.tokenizer2(
                first_pass,
                return_tensors="pt",
                padding=True,
                truncation=True).to(self.model2.device)

            translated = self.model2.generate(**inputs)
            return self.tokenizer2.batch_decode(translated, skip_special_tokens=True)

        return first_pass
