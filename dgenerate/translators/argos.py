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
import urllib.request

import dgenerate.extras.argostranslate.package as argostranslate_package
import dgenerate.extras.argostranslate.settings as argostranslate_settings

import dgenerate.filelock
import dgenerate.messages as _messages
import dgenerate.translators.exceptions as _exceptions
import dgenerate.translators.util as _util
import dgenerate.spacycache as _spacycache


class ArgosTranslator:
    """
    Translate languages locally on the CPU using argostranslate models.

    Supports automatic pivot language selection.
    """

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

        # search by priority
        pivot_lang_codes = ['en', 'es', 'fr', 'de', 'it', 'pt']

        # argostranslate is not multiprocess safe.
        # at least make it so between dgenerate processes.

        argostranslate_settings.cache_dir.mkdir(parents=True, exist_ok=True)

        with dgenerate.filelock.temp_file_lock(
                argostranslate_settings.cache_dir / '_dgenerate.lock'
        ):

            # download package index if we do not have it, and the user allows it
            if not os.path.exists(argostranslate_settings.local_package_index):

                if not (local_files_only or self._offline_mode):
                    _messages.debug_log('argostranslate, updating package index...')
                    self._argos_update_package_index()
                else:
                    raise _exceptions.TranslatorLoadError(
                        'argostranslate needs to download a model index '
                        'to search for available translation models, but you are in offline mode.')

            # what is available? (in the index)
            available_packages = argostranslate_package.get_available_packages()

            # Try to find this initially
            package_to_install = None

            # pivot lang package will go here if required
            pivot_lang = None

            try:
                # search for suitable model (package)
                package_to_install = next(
                    filter(
                        lambda x: x.from_code == from_lang and x.to_code == to_lang, available_packages
                    )
                )
            except StopIteration:
                # did not find anything, lets look for a pivot
                for code in pivot_lang_codes:
                    try:
                        pivot_lang = next(
                            filter(
                                lambda x: x.from_code == from_lang and x.to_code == code, available_packages
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
                    if local_files_only or self._offline_mode:
                        raise _exceptions.TranslatorLoadError(
                            f'argostranslate needs to download a pivot model '
                            f'for: "{pivot_lang.from_code}" -> "{pivot_lang.to_code}", '
                            f'but offline mode is active.')

                    argostranslate_package.install_from_path(
                        self._argos_download_package(pivot_lang)
                    )

                # check that we can translate from the pivot language to the desired language.
                # if we cannot, then this will throw StopIteration

                try:
                    package_to_install = next(
                        filter(
                            lambda x: x.from_code == pivot_lang.to_code and x.to_code == to_lang, available_packages
                        )
                    )
                except StopIteration:
                    raise _exceptions.TranslatorLoadError(
                        f'argostranslate translation model for "{from_lang}" -> "{to_lang}" is not available.')

                # This model will be used to translate from the pivot language to the target language
                model_path = self._argos_model_path(package_to_install)

                # If that model does not exist, download it if the user allows
                if not os.path.exists(model_path):
                    if local_files_only or self._offline_mode:
                        raise _exceptions.TranslatorLoadError(
                            f'argostranslate needs to download a model '
                            f'for: "{package_to_install.from_code}" -> "{package_to_install.to_code}", '
                            f'but offline mode is active.')

                    argostranslate_package.install_from_path(
                        self._argos_download_package(package_to_install)
                    )

            elif package_to_install is not None:
                # we do not need a pivot model and can translate directly.

                model_path = self._argos_model_path(package_to_install)

                # download the model if it does not exist
                if not os.path.exists(model_path):
                    if local_files_only or self._offline_mode:
                        raise _exceptions.TranslatorLoadError(
                            f'argostranslate needs to download a model '
                            f'for: "{package_to_install.from_code}" -> "{package_to_install.to_code}", '
                            f'but offline mode is active.')

                    argostranslate_package.install_from_path(
                        self._argos_download_package(package_to_install)
                    )
            else:
                # could not find anything

                raise _exceptions.TranslatorLoadError(
                    f'argostranslate translation model for "{from_lang}" -> "{to_lang}" is not available.')

        self._translation = None
        self._translation2 = None

        try:
            with _spacycache.offline_mode_context(local_files_only):
                # this trys to download to the spacy cache if the models
                # do not exist, we want it to throw if we are in offline mode
                import dgenerate.extras.argostranslate.translate as argostranslate_translate
        except _spacycache.SpacyModelNotFoundError as e:
            raise _exceptions.TranslatorLoadError(
                'Unable to load argostranslate model due to being '
                'unable to download required SpaCy model with offline mode active.') from e

        if pivot_lang:
            _messages.debug_log(
                f'argostranslate: using pivot {from_lang} -> {pivot_lang.to_code} then {pivot_lang.to_code} -> {to_lang}')

            # first step, translate to pivot language
            self._translation = argostranslate_translate.get_translation_from_codes(from_lang, pivot_lang.to_code)

            # second step, translate from pivot language to target language
            self._translation2 = argostranslate_translate.get_translation_from_codes(pivot_lang.to_code, to_lang)

        else:
            # directly translate
            self._translation = argostranslate_translate.get_translation_from_codes(from_lang, to_lang)

    @staticmethod
    def _argos_update_package_index():
        try:
            response = urllib.request.urlopen(argostranslate_settings.remote_package_index)
        except Exception as e:
            # They eat this exception and then log it without a re-throw, I need to handle it.
            raise _exceptions.TranslatorLoadError(
                f'Unable to download argostranslate package index, network error.') from e
        data = response.read()
        with open(argostranslate_settings.local_package_index, "wb") as f:
            f.write(data)

    @staticmethod
    def _argos_download_package(package: argostranslate_package.AvailablePackage):
        try:
            # this actually just throws "Exception" upon download failure.
            return package.download()
        except Exception as e:
            raise _exceptions.TranslatorLoadError(
                f'Unable to download argostranslate model: {package.from_code} -> {package.to_code}, network error.') from e

    @staticmethod
    def _argos_model_path(package: argostranslate_package.IPackage):
        # see the source code of package.download()
        return argostranslate_settings.downloads_dir / (
                argostranslate_package.argospm_package_name(package) + '.argosmodel')

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
                if self._translation2:
                    output.append(self._translation2.translate(self._translation.translate(text)))
                else:
                    output.append(self._translation.translate(text))
            except Exception as e:
                raise _exceptions.TranslationError(e) from e
        return output
