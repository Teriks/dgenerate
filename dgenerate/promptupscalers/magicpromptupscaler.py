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

import diffusers
import torch
import transformers

import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.util as _pipelinewrapper_util
import dgenerate.prompt as _prompt
import dgenerate.promptupscalers.exceptions as _exceptions
import dgenerate.promptupscalers.llmupscalermixin as _llmupscalermixin
import dgenerate.promptupscalers.promptupscaler as _promptupscaler
from dgenerate.pipelinewrapper.uris import get_quantizer_uri_class as _get_quantizer_uri_class
from dgenerate.pipelinewrapper.uris import BNBQuantizerUri as _BNBQuantizerUri


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


class _TextGenerationPipeline:
    def __init__(self, model, tokenizer, quantized: bool):
        self.model = model
        self.tokenizer = tokenizer
        self.quantized = quantized

    def to(self, device):
        if not self.quantized:
            self.model.to(device)

    def __call__(self,
                 prompts: list[str],
                 batch_size: int = 1,
                 max_length: int = 100,
                 **kwargs
                 ):
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i: i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length)

            model_device = self.model.device

            inputs = inputs.to(model_device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length, **kwargs)

            if model_device.type != 'cpu':
                inputs.to('cpu')

            del inputs

            results.extend(
                [self.tokenizer.decode(output) for output in outputs])

        return results


class MagicPromptUpscaler(_llmupscalermixin.LLMPromptUpscalerMixin, _promptupscaler.PromptUpscaler):
    """
    Upscale prompts using magicprompt or other LLMs via transformers.

    The "part" argument indicates which parts of the prompt to act on,
    possible values are: "both", "positive", and "negative"

    The "model" specifies the model path for magicprompt, the
    default value is: "Gustavosta/MagicPrompt-Stable-Diffusion". This
    can be a folder on disk or a Hugging Face repository slug.

    The "dtype" argument specifies the torch dtype (compute dtype) to load
    the model with, this defaults to: float32, and may be one of: float32,
    float16, or bfloat16.

    The "seed" argument can be used to specify a seed for prompt generation.

    The "variations" argument specifies how many variations should be produced.

    The "max-length" argument is the max prompt length for a
    generated prompt, this value defaults to 100.

    The "temperature" argument sets the sampling temperature
    to use when generating prompts. Larger values increase
    creativity but decrease factuality.

    The "top_k" argument sets the "top_k" generation value, i.e.
    randomly sample from the "top_k" most likely tokens at each generation step.
    Set this to 1 for greedy decoding.

    The "top_p" argument sets the "top_p" generation value, i.e.
    randomly sample at each generation step from the top most likely
    tokens whose probabilities add up to "top_p".

    The "system" argument sets the system instruction for the LLM.

    The "preamble" argument sets a text input preamble for the LLM, this
    preamble will be removed from the output generated by the LLM.

    The "remove-prompt" argument specifies whether to remove the
    original prompt from the generated text.

    The "prepend-prompt" argument specifies whether to forcefully
    prepend the original prompt to the generated prompt, this
    might be necessary if you want a continuation with some
    models, the original prompt will be prepended with a
    space at the end.

    The "batch" argument enables and disables batching
    prompt text into the LLM, setting this to False tells
    the plugin that you only want the LLM to ever process
    one prompt at a time, this might be useful if you are
    memory constrained, but processing is much slower.

    The "max-batch" argument allows you to adjust how
    many prompts can be processed by the LLM simultaneously,
    processing too many prompts at once will run your system
    out of memory, processing too little prompts at once will
    be slow. Specifying "None" indicates unlimited batch size.

    The "quantizer" argument allows you to specify a quantization
    backend for loading the LLM, this is the same syntax and supported
    backends as with the dgenerate --quantizer argument.

    The "block-regex" argument is a python syntax regex that will
    block prompts that match the regex, the prompt will be regenerated
    until the regex does not match, up to "max-attempts". This
    regex is case-insensitive.

    The "max-attempts" argument specifies how many times to reattempt
    to generate a prompt if it is blocked by "block-regex"

    The "smart-truncate" argument enables intelligent truncation
    of the prompt generated by the LLM, i.e. it will remove incomplete
    sentences from the end of the prompt utilizing spaCy NLP.

    The "cleanup-config" argument allows you to specify a
    custom LLM output cleanup configuration file in
    .json, .toml, or .yaml format. This file can be used
    to run custom pattern substitutions or python functions
    over the LLMs raw output, and overrides the built-in cleanup
    excluding "smart-truncate" which occurs before your configuration.
    """

    NAMES = ['magicprompt']

    OPTION_ARGS = {
        'part': ['both', 'positive', 'negative'],
        'dtype': ['float32', 'float16', 'bfloat16']
    }

    FILE_ARGS = {
        'model': {'mode': 'dir'},
        'cleanup-config': {'mode': 'in', 'filetypes': [('Cleanup Config', ('*.json', '*.toml', '*.yaml', '*.yml'))]}
    }

    def __init__(self,
                 part: str = 'both',
                 model: str = "Gustavosta/MagicPrompt-Stable-Diffusion",
                 dtype: str = 'float32',
                 seed: int | None = None,
                 variations: int = 1,
                 max_length: int = 100,
                 temperature: float = 0.7,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 system: str | None = None,
                 preamble: str | None = None,
                 remove_prompt: bool = False,
                 prepend_prompt: bool = False,
                 batch: bool = True,
                 max_batch: int | None = 50,
                 quantizer: str | None = None,
                 block_regex: str | None = None,
                 max_attempts: int = 10,
                 smart_truncate: bool = False,
                 cleanup_config: str | None = None,
                 **kwargs
                 ):
        """
        :param kwargs: child class forwarded arguments
        """
        super().__init__(**kwargs,
                         part=part,
                         block_regex=block_regex,
                         max_attempts=max_attempts,
                         cleanup_mode='magic' if 'magicprompt' in model.lower() else 'other',
                         smart_truncate=smart_truncate,
                         cleanup_config=cleanup_config)

        dtype = dtype.lower()
        if dtype not in {'float32', 'float16', 'bfloat16'}:
            raise self.argument_error('Argument "dtype" must be either float32, float16, or bfloat16.')

        if quantizer:
            try:
                quantizer_class = _get_quantizer_uri_class(quantizer)
                quantization_config = quantizer_class.parse(quantizer).to_config(dtype)
            except Exception as e:
                raise self.argument_error(f'Error loading "quantizer" argument "{quantizer}": {e}') from e
        else:
            quantization_config = None

        part = part.lower()
        if part not in {'both', 'positive', 'negative'}:
            raise self.argument_error(
                'Argument "part" must be one of: "both", "positive", or "negative"'
            )

        if max_length < 1:
            raise self.argument_error(
                'Cannot specify "max-length" less than 1.'
            )

        if variations < 1:
            raise self.argument_error(
                'Argument "variations" may not be less than 1.')

        if temperature < 0.0:
            raise self.argument_error(
                'Argument "temperature" may not be less than 1.')

        if top_k < 1:
            raise self.argument_error(
                'Argument "top-k" may not be less than 1.')

        if top_p < 0.0:
            raise self.argument_error(
                'Argument "top-p" may not be less than 0.')

        if top_p > 1.0:
            raise self.argument_error(
                'Argument "top-p" may not be greater than 1.')

        if max_batch is not None and max_batch < 1:
            raise self.argument_error(
                'Argument "max-batch" may not be less than 1.')

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
            if quantization_config is not None:
                self.memory_guard_device(self.device, self.size_estimate)

            torch_dtype = {'float32': torch.float32,
                           'float16': torch.float16,
                           'bfloat16': torch.bfloat16
                           }[dtype]

            if isinstance(quantization_config, diffusers.BitsAndBytesConfig):
                if quantization_config.load_in_4bit and quantization_config.bnb_4bit_compute_dtype is None:
                    quantization_config.bnb_4bit_compute_dtype = torch_dtype

            return self._load_pipeline(model, dtype=torch_dtype, quantization_config=quantization_config)

        self.set_size_estimate(estimated_size)

        self._pipeline = self.load_object_cached(
            tag=model + (quantizer if quantizer else '') + dtype,
            estimated_size=estimated_size,
            method=load_method
        )

        self._system = system
        self._preamble = preamble
        self._remove_prompt = remove_prompt
        self._seed = seed
        self._max_length = max_length
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

        self._variations = variations
        self._accepts_batch = batch
        self._max_batch = max_batch
        self._part = part
        self._quantizer = quantizer
        self._max_attempts = max_attempts
        self._prepend_prompt = prepend_prompt


    def _load_pipeline(self,
                       model_name: str,
                       dtype: torch.dtype,
                       quantization_config: typing.Optional[typing.Any] = None) -> _TextGenerationPipeline:

        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
                quantization_config=quantization_config,
                device_map=self.device if quantization_config else None,
                local_files_only=self.local_files_only
            )
        except Exception as e:
            raise self.argument_error(f'Could not load model "{model_name}": {e}')

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = model.config.eos_token_id

        return _TextGenerationPipeline(
            model, tokenizer,
            quantized=quantization_config is not None
        )

    def _to(self, device: str | torch.device):
        self._pipeline.to(device)

    def _generate_prompts(self, original_prompts: list[str]) -> list[str]:
        def build_query(text):
            if self._preamble:
                return self._preamble + (' ' if not self._preamble.endswith(' ') else '') + text
            return text

        if self._system:
            formatted_prompts = [
                f"<|system|> {self._system} <|user|> {build_query(query)} <|assistant|>"
                for query in original_prompts
            ]
        else:
            formatted_prompts = [build_query(query) for query in original_prompts]

        if not self._accepts_batch:
            generated_prompts = []
            for ptext in formatted_prompts:
                generated_prompts.extend(self._call_pipeline([ptext]))
        elif self._max_batch is not None:
            generated_prompts = []
            for batch_segment in range(0, len(formatted_prompts), self._max_batch):
                segment = formatted_prompts[batch_segment:batch_segment + self._max_batch]
                generated_prompts.extend(self._call_pipeline(segment))
        else:
            generated_prompts = self._call_pipeline(formatted_prompts)

        generated_prompts = [
            self._clean_prompt(
                formatted_prompt,
                generated_prompt,
                remove_prefixes=[self._system, self._preamble],
                remove_prompt=self._remove_prompt,
                prepend=original_prompt if self._prepend_prompt else None,
            )
            for original_prompt, formatted_prompt, generated_prompt in zip(
                original_prompts, formatted_prompts, generated_prompts
            )
        ]

        return generated_prompts

    def _call_pipeline(self, prompts: list[str]):
        return self._pipeline(
            prompts,
            max_length=self._max_length,
            temperature=self._temperature,
            top_k=self._top_k,
            top_p=self._top_p,
            do_sample=True,
            batch_size=len(prompts)
        )

    @contextlib.contextmanager
    def _with_device(self):
        if self._quantizer:
            yield
            gc.collect()
            _memory.torch_gc()
        else:
            try:
                self.memory_guard_device(self.device, self.size_estimate)
                self._to(self.device)
                yield
            finally:
                self._to('cpu')
                gc.collect()
                _memory.torch_gc()

    @property
    def accepts_batch(self) -> bool:
        """
        This prompt upscaler can accept a batch of prompts for efficient execution.
        :return: ``True``, unless the constructor argument ``batch`` was passed ``False``
        """
        return self._accepts_batch

    def upscale(self, prompts: _prompt.PromptOrPrompts) -> _prompt.PromptOrPrompts:

        if isinstance(prompts, _prompt.Prompt):
            prompts = [prompts]

        if len(prompts) > 1 and not self.accepts_batch:
            raise _exceptions.PromptUpscalerProcessingError(
                f'magicprompt prompt upscaler cannot accept batch input when '
                f'the argument "batch" is set to False.'
            )

        prompts = list(prompts) * self._variations

        try:
            with _with_seed(self._seed), self._with_device():
                return self._process_prompts(prompts)
        except torch.cuda.OutOfMemoryError as e:
            prompt_count = len(prompts)
            if prompt_count > 1:
                raise _exceptions.PromptUpscalerProcessingError(
                    f'magicprompt prompt upscaler could not '
                    f'process {len(prompts)} incoming prompt(s) due to CUDA '
                    f'out of memory error, try using the argument "batch=False" '
                    f'to process only one prompt at a time (this is slow).') from e
            raise _exceptions.PromptUpscalerProcessingError(
                f'magicprompt prompt upscaler could not '
                f'process prompt due to CUDA out of memory error: {prompts[0]}'
            ) from e
        except transformers.pipelines.PipelineException as e:
            raise _exceptions.PromptUpscalerProcessingError(
                f'magicprompt prompt upscaler could not process prompt(s) due '
                f'to transformers pipeline exception: {e}'
            ) from e
