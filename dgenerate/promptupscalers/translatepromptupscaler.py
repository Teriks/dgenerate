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

import dgenerate.prompt as _prompt
import dgenerate.promptupscalers.exceptions as _exceptions
import dgenerate.promptupscalers.promptupscaler as _promptupscaler
import dgenerate.promptupscalers.util as _util
import dgenerate.translators as _translators


class TranslatePromptUpscaler(_promptupscaler.PromptUpscaler):
    """
    Local language translation using argostranslate or Helsinki-NLP opus (mariana).

    Please note that translation models require a one time download,
    so run at least once with --offline-mode disabled to download the
    desired model.

    argostranslate (argos) offers lightweight translation via CPU inference.

    Helsinki-NLP (mariana) offers slightly more heavy duty (accurate) CPU or GPU inference.

    The "input" argument indicates the input language code (IETF) e.g. "en", "zh", or
    literal name of the language for example: "english", "chinese".

    The "output" argument indicates the output language code (IETF), or literal
    name of the language, this value defaults to "en" (English).

    The "provider" argument indicates the translation provider, which may be one of "argos"
    or "mariana". The default value is "argos", indicating argostranslate. argos will only
    ever use the "cpu" regardless of the current --device or "device" argument value. Mariana
    will default to using the value of --device which will usually be a GPU.

    The "batch" argument enables and disables batching
    prompt text into the translator, setting this to False
    tells the plugin that you only want to ever process
    one prompt at a time, this might be useful if you are
    memory constrained and using the provider "mariana",
    but processing is much slower.

    The "max-batch" argument allows you to adjust how
    many prompts can be processed by the model simultaneously,
    processing too many prompts at once will run your system
    out of memory (specifically for the mariana translation provider),
    processing too little prompts at once will be slow.
    Specifying "None" indicates unlimited batch size. This argument
    has no effect on argostranslate performance.
    """

    NAMES = ['translate']

    OPTION_ARGS = {
        'part': ['both', 'positive', 'negative'],
        'provider': ['argos', 'mariana']
    }

    def __init__(self,
                 input: str,
                 output: str = 'en',
                 part: str = 'both',
                 provider: str = 'argos',
                 batch: bool = True,
                 max_batch: int | None = 50,
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

        if max_batch is not None and max_batch < 1:
            raise self.argument_error(
                'Argument "max-batch" may not be less than 1.')

        try:
            self._translator = self.load_object_cached(
                input + output + provider,
                estimated_size=150 * 1024 ** 2 if provider == 'argos' else 512 * 1024 ** 2,
                method=lambda: _translators.ArgosTranslator(input, output, self.local_files_only)
                if provider == 'argos' else _translators.MarianaTranslator(input, output, self.local_files_only)
            )
        except _translators.TranslatorLoadError as e:
            raise self.argument_error(str(e)) from e

        self._accepts_batch = batch
        self._part = part
        self._max_batch = max_batch

    def _translate(self, texts: list[str]) -> list[str]:
        if self._max_batch is not None and not isinstance(self._translator, _translators.ArgosTranslator):
            translated = []
            for batch_segment in range(0, len(texts), self._max_batch):
                segment = texts[batch_segment:batch_segment + self._max_batch]
                translated.extend(self._translator.translate(segment))
            return translated
        else:
            return self._translator.translate(texts)

    def accepts_batch(self) -> bool:
        return self._accepts_batch

    def upscale(self, prompts: _prompt.Prompts) -> _prompt.PromptOrPrompts:
        if isinstance(prompts, _prompt.Prompt):
            prompts = [prompts]

        try:
            if hasattr(self._translator, 'to'):
                self._translator.to(self.device)
            return _util.process_prompts_batched(prompts, self._part, self._translate)
        except _translators.TranslationError as e:
            raise _exceptions.PromptUpscalerProcessingError(e) from e
        finally:
            if hasattr(self._translator, 'to'):
                self._translator.to('cpu')
