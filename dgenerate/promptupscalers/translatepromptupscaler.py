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

import argostranslate.package
import argostranslate.settings
import argostranslate.translate
import transformers

import dgenerate.filelock
import dgenerate.messages as _messages
import dgenerate.prompt as _prompt
import dgenerate.promptupscalers.exceptions as _exceptions
import dgenerate.promptupscalers.promptupscaler as _promptupscaler
import dgenerate.promptupscalers.util as _util
import urllib.request


class TranslatorLoadError(Exception):
    pass


class TranslationProcessError(Exception):
    pass


class MarianaTranslator:

    def __init__(self, src_lang: str, tgt_lang: str, local_files_only: bool = False):
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

        try:
            self.tokenizer = transformers.MarianTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
            self.model = transformers.MarianMTModel.from_pretrained(model_name, local_files_only=local_files_only)
        except OSError as e:
            raise TranslatorLoadError(
                f'Helsinki-NLP translation model for "{src_lang}" -> "{tgt_lang}" is not available.') from e
        except Exception as e:
            raise TranslatorLoadError(e) from e

    def to(self, device):
        self.model.to(device)

    def translate(self, texts: str | list[str]) -> list[str]:
        if isinstance(texts, str):
            texts = [texts]
        output = []
        for text in texts:
            try:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

                translated = self.model.generate(**inputs)
                inputs.to('cpu')

                del inputs

                output.append(self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0])
            except Exception as e:
                raise TranslationProcessError(e) from e
        return output


class ArgosTranslator:

    def __init__(self, src_lang: str, tgt_lang: str, local_files_only: bool = False):

        # search by priority
        pivot_lang_codes = ['en', 'es', 'fr', 'de', 'it', 'pt']

        # argostranslate is not multiprocess safe.
        # at least make it so between dgenerate processes.

        argostranslate.settings.cache_dir.mkdir(parents=True, exist_ok=True)

        with dgenerate.filelock.temp_file_lock(
                argostranslate.settings.cache_dir / '_dgenerate.lock'
        ):

            # download package index if we do not have it, and the user allows it
            if not os.path.exists(argostranslate.settings.local_package_index):

                if not local_files_only:
                    _messages.debug_log('argostranslate, updating package index...')
                    self._argos_update_package_index()
                else:
                    raise TranslatorLoadError(
                        'argostranslate needs to download a model index '
                        'to search for available translation models, but you are in offline mode.')

            # whats available? (in the index)
            available_packages = argostranslate.package.get_available_packages()

            # Try to find this initially
            package_to_install = None

            # pivot lang package will go here if required
            pivot_lang = None

            try:
                # search for suitable model (package)
                package_to_install = next(
                    filter(
                        lambda x: x.from_code == src_lang and x.to_code == tgt_lang, available_packages
                    )
                )
            except StopIteration:
                # did not find anything, lets look for a pivot
                for code in pivot_lang_codes:
                    try:
                        pivot_lang = next(
                            filter(
                                lambda x: x.from_code == src_lang and x.to_code == code, available_packages
                            )
                        )
                        # pivot found
                        break
                    except StopIteration:
                        # keep looking
                        continue

            if pivot_lang is not None:
                # we located a suitable pivot model

                pivot_model_path = self._argos_model_path(pivot_lang)

                # if it does not exist on disk, and we are not in offline mode,
                # then download the model
                if not os.path.exists(pivot_model_path):
                    if local_files_only:
                        raise TranslatorLoadError(
                            f'argostranslate needs to download a pivot model '
                            f'for: "{pivot_lang.from_code}" -> "{pivot_lang.to_code}", '
                            f'but offline mode is active.')

                    argostranslate.package.install_from_path(
                        self._argos_download_package(pivot_lang)
                    )

                # check that we can translate from the pivot language to the desired language.
                # if we cannot, then this will throw StopIteration

                try:
                    package_to_install = next(
                        filter(
                            lambda x: x.from_code == pivot_lang.to_code and x.to_code == tgt_lang, available_packages
                        )
                    )
                except StopIteration:
                    raise TranslatorLoadError(
                        f'argostranslate translation model for "{src_lang}" -> "{tgt_lang}" is not available.')

                # This model will be used to translate from the pivot language to the target language
                model_path = self._argos_model_path(package_to_install)

                # If that model does not exist, download it if the user allows
                if not os.path.exists(model_path):
                    if local_files_only:
                        raise TranslatorLoadError(
                            f'argostranslate needs to download a model '
                            f'for: "{package_to_install.from_code}" -> "{package_to_install.to_code}", '
                            f'but offline mode is active.')

                    argostranslate.package.install_from_path(
                        self._argos_download_package(package_to_install)
                    )

            elif package_to_install is not None:
                # we do not need a pivot model and can translate directly.

                model_path = self._argos_model_path(package_to_install)

                # download the model if it does not exist
                if not os.path.exists(model_path):
                    if local_files_only:
                        raise TranslatorLoadError(
                            f'argostranslate needs to download a model '
                            f'for: "{package_to_install.from_code}" -> "{package_to_install.to_code}", '
                            f'but offline mode is active.')

                    argostranslate.package.install_from_path(
                        self._argos_download_package(package_to_install)
                    )
            else:
                # could not find anything

                raise TranslatorLoadError(
                    f'argostranslate translation model for "{src_lang}" -> "{tgt_lang}" is not available.')

        self._translation = None
        self._translation2 = None

        if pivot_lang:
            _messages.debug_log(
                f'argostranslate: using pivot {src_lang} -> {pivot_lang.to_code} then {pivot_lang.to_code} -> {tgt_lang}')

            # first step, translate to pivot language
            self._translation = argostranslate.translate.get_translation_from_codes(src_lang, pivot_lang.to_code)

            # second step, translate from pivot language to target language
            self._translation2 = argostranslate.translate.get_translation_from_codes(pivot_lang.to_code, tgt_lang)

        else:
            # directly translate
            self._translation = argostranslate.translate.get_translation_from_codes(src_lang, tgt_lang)

    @staticmethod
    def _argos_update_package_index():
        try:
            response = urllib.request.urlopen(argostranslate.settings.remote_package_index)
        except Exception:
            # They eat this exception and then log it without a re-throw, I need to handle it.
            raise TranslatorLoadError(
                f'Unable to download argostranslate package index, network error.')
        data = response.read()
        with open(argostranslate.settings.local_package_index, "wb") as f:
            f.write(data)

    @staticmethod
    def _argos_download_package(package: argostranslate.package.IPackage):
        try:
            # this actually just throws "Exception" upon download failure.
            return package.download()
        except Exception:
            raise TranslatorLoadError(
                f'Unable to download argostranslate model: {package.from_code} -> {package.to_code}, network error.')

    @staticmethod
    def _argos_model_path(package: argostranslate.package.IPackage):
        # see the source code of package.download()
        return argostranslate.settings.downloads_dir / (
                argostranslate.package.argospm_package_name(package) + '.argosmodel')

    def to(self, device):
        pass

    def translate(self, texts: str | list[str]) -> list[str]:
        if isinstance(texts, str):
            texts = [texts]
        output = []
        for text in texts:
            try:
                if self._translation2:
                    output.append(self._translation2.translate(self._translation.translate(text)))
                else:
                    output.append(self._translation.translate(text))
            except Exception as e:
                raise TranslationProcessError(e) from e
        return output


class TranslatePromptsUpscaler(_promptupscaler.PromptUpscaler):
    """
    Local language translation using argostranslate or Helsinki-NLP opus (mariana).

    Please note that translation models require a one time download,
    so run at least once with --offline-mode disabled to download the
    desired model.

    The "input" argument indicates the input language code.

    The "output" argument indicates the output language code, which defaults to english, i.e: "en".

    The "provider" argument indicates the translation provider, which may be one of "argos"
    or "mariana".  The default value is "argos", indicating argostranslate.  argos will only
    ever use the "cpu" regardless of the current --device or "device" argument value. Mariana
    will default to using the value of --device which will usually be a GPU.

    The "batch" argument enables and disables batching
    prompt text into the translator, setting this to False
    tells the plugin that you only want to ever process
    one prompt at a time, this might be useful if you are
    memory constrained and using the provider "mariana",
    but processing is much slower.
    """

    NAMES = ['translate']

    def __init__(self,
                 input: str,
                 output: str = 'en',
                 part: str = 'both',
                 provider: str = 'argos',
                 batch: bool = True,
                 **kwargs
                 ):
        """
        :param kwargs: child class forwarded arguments
        """
        super().__init__(**kwargs)

        part = part.lower()
        provider = provider.lower()

        if part not in {'both', 'positive', 'negative'}:
            raise self.argument_error(
                'Argument "part" must be one of: "both", "positive", or "negative"')

        if provider not in {'argos', 'mariana'}:
            raise self.argument_error(
                'Argument "provider" must be one of: "argos" or "mariana"')

        try:
            self._translator = self.load_object_cached(
                input + output + provider,
                estimated_size=150 * 1024 ** 2 if provider == 'argos' else 512 * 1024 ** 2,
                method=lambda: ArgosTranslator(input, output, self.local_files_only)
                if provider == 'argos' else MarianaTranslator(input, output, self.local_files_only)
            )
        except TranslatorLoadError as e:
            raise self.argument_error(str(e))

        self._accepts_batch = batch
        self._part = part

    def accepts_batch(self):
        return self._accepts_batch

    def upscale(self, prompts: _prompt.Prompts) -> _prompt.PromptOrPrompts:
        if isinstance(prompts, _prompt.Prompt):
            prompts = [prompts]

        try:
            self._translator.to(self.device)
            return _util.process_prompts_batched(prompts, self._part, self._translator.translate)
        except TranslationProcessError as e:
            raise _exceptions.PromptUpscalerProcessingError(e) from e
        finally:
            self._translator.to('cpu')
