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

import torch

import dgenerate.messages as _messages
import transformers
import dgenerate.translators.exceptions as _exceptions


class MarianaTranslator:
    """
    Translate languages locally using Helsinki-NLP opus models on the CPU or GPU.

    Supports automatic pivot language selection.
    """

    _translation_map = None

    def __init__(self, from_lang: str, to_lang: str, local_files_only: bool = False):
        """
        :param from_lang: From language code (IETF)
        :param to_lang: To language code (IETF)
        :param local_files_only: Only use models that have been previously cached?
        :raise dgenerate.translators.TranslatorLoadError: If models cannot be loaded / found.
        """

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

            try:
                self.tokenizer = transformers.MarianTokenizer.from_pretrained(
                    pivot_model_name, local_files_only=local_files_only)
                self.model = transformers.MarianMTModel.from_pretrained(
                    pivot_model_name, local_files_only=local_files_only)
            except OSError as e:
                raise _exceptions.TranslatorLoadError(
                    f'Helsinki-NLP translation model for "{from_lang}" -> "{to_lang}" is not available.') from e
            except Exception as e:
                raise _exceptions.TranslatorLoadError(e) from e

            try:
                self.tokenizer2 = transformers.MarianTokenizer.from_pretrained(
                    model_name, local_files_only=local_files_only)
                self.model2 = transformers.MarianMTModel.from_pretrained(
                    model_name, local_files_only=local_files_only)
            except OSError as e:
                raise _exceptions.TranslatorLoadError(
                    f'Helsinki-NLP translation model for "{from_lang}" -> "{to_lang}" is not available.') from e
            except Exception as e:
                raise _exceptions.TranslatorLoadError(e) from e

        elif model_name:
            try:
                self.tokenizer = transformers.MarianTokenizer.from_pretrained(
                    model_name, local_files_only=local_files_only)
                self.model = transformers.MarianMTModel.from_pretrained(
                    model_name, local_files_only=local_files_only)
            except OSError as e:
                raise _exceptions.TranslatorLoadError(
                    f'Helsinki-NLP translation model for "{from_lang}" -> "{to_lang}" is not available.') from e
            except Exception as e:
                raise _exceptions.TranslatorLoadError(e) from e
        else:
            raise _exceptions.TranslatorLoadError(
                f'Helsinki-NLP translation model for "{from_lang}" -> "{to_lang}" is not available.')

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

        output = []

        for text in texts:
            try:
                if self.tokenizer2:
                    # pivot model

                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.model.device)

                    translated = self.model.generate(**inputs)
                    translated = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

                    inputs.to('cpu')

                    del inputs

                    # translate out of pivot language

                    inputs = self.tokenizer2(
                        translated,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.model2.device)

                    translated = self.model2.generate(**inputs)
                    inputs.to('cpu')

                    del inputs

                    translated = self.tokenizer2.batch_decode(translated, skip_special_tokens=True)[0]

                    output.append(translated)
                else:
                    inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
                        self.model.device)

                    translated = self.model.generate(**inputs)
                    inputs.to('cpu')

                    del inputs

                    output.append(self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0])
            except Exception as e:
                raise _exceptions.TranslationError(e) from e
        return output
