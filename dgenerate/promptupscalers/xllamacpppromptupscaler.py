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

import json
import os
import socket

try:
    import xllamacpp
except ImportError:
    xllamacpp = None

import dgenerate.promptupscalers.promptupscaler as _promptupscaler
import dgenerate.promptupscalers.exceptions as _exceptions
import dgenerate.prompt as _prompt
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.webcache as _webcache
import dgenerate.promptupscalers.llmupscalermixin as _llmupscalermixin
import dgenerate.promptupscalers.util as _util
import dgenerate.hfhub as _hfhub

_DEFAULT_MODEL = (
    'https://huggingface.co/failspy/Phi-3-mini-128k-instruct-abliterated-v3-GGUF'
    '/resolve/main/Phi-3-mini-128k-instruct-abliterated-v3_q4.gguf'
)

# llama.cpp: -1 fits however many layers VRAM allows, and -2 or lower offloads every layer.
_GPU_LAYERS_AUTO = -1
_GPU_LAYERS_ALL = -2


def _resolve_gpu_layers(value: int | str) -> int:
    if isinstance(value, str):
        name = value.strip().lower()
        if name == 'auto':
            return _GPU_LAYERS_AUTO
        if name == 'all':
            return _GPU_LAYERS_ALL
        raise ValueError(name)
    return value

def _free_localhost_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _chat_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get('type') in (None, 'text'):
                parts.append(str(part.get('text') or ''))
        return ''.join(parts)
    if content is None:
        return ''
    return str(content)


class XllamaCppPromptUpscaler(_llmupscalermixin.LLMPromptUpscalerMixin, _promptupscaler.PromptUpscaler):
    """
    Upscale prompts with a GGUF model through llama.cpp.

    The "part" argument indicates which parts of the prompt to act on,
    possible values are: "both", "positive", and "negative".

    The "model" argument is a path or URL to a GGUF file. The default is a
    Phi-3 Mini abliterated Q4 GGUF.

    The "mmproj" argument is a path or URL to the GGUF multimodal projector
    for "model". It is required when "image" is set.

    The "image" argument is a path or URL to an image. The image is attached
    to every prompt sent to the model, and the prompt text is the instruction,
    for example a request to write an image-generation prompt for the picture.
    Animated files use the first frame. Vision models usually need
    "context-tokens" of 4096 or more so the image and the reply both fit.

    The "variations" argument specifies how many variations should be produced.

    The "max-length" argument is the max number of new tokens. It defaults to 100.

    The "temperature", "top-k", "top-p", and "min-p" arguments are the sampling
    settings passed to the model.

    The "thinking" argument enables a thought before the reply on models whose
    chat template supports it, such as Qwen and MiniCPM. It defaults to false
    so the reply is the prompt. Templates that do not use this flag ignore it.

    The "system" argument is the chat system message.

    The "preamble" argument is text placed in front of the user prompt. It is
    removed from the generated text when it is echoed back.

    The "remove-prompt" argument specifies whether to remove the original prompt
    from the generated text.

    The "prepend-prompt" argument specifies whether to prepend the original
    prompt to the generated prompt.

    The "gpu-layers" argument controls GPU offload. It accepts "auto", "all",
    or an integer, and defaults to "auto".

    NOWRAP!
    * "auto": offload as many layers as fit in VRAM.
    * "all": offload every layer onto the backend compiled into the installed
      wheel (Metal, CUDA, ROCm, or Vulkan).
    * 0: keep the model on the CPU.
    * A positive integer: offload that many layers.

    The "block-regex" argument is a case-insensitive regular expression.
    Matching prompts are regenerated up to "max-attempts" times.

    The "context-tokens" argument is the context window (``n_ctx``).

    The "smart-truncate" argument removes an incomplete trailing sentence
    using spaCy.

    The "cleanup-config" argument is a ``.json``, ``.toml``, or ``.yaml``
    file of extra substitutions applied to the model output.
    """

    NAMES = ['llama']

    HIDE_ARGS = ['device']

    OPTION_ARGS = {
        'part': ['both', 'positive', 'negative'],
    }

    FILE_ARGS = {
        'model': {'mode': 'in', 'filetypes': [('GGUF', ['*.gguf'])]},
        'mmproj': {'mode': 'in', 'filetypes': [('GGUF', ['*.gguf'])]},
        'image': _util.image_file_argument(),
        'cleanup-config': {'mode': 'in', 'filetypes': [('Cleanup Config', ('*.json', '*.toml', '*.yaml', '*.yml'))]},
    }

    def __init__(self,
                 part: str = 'both',
                 model: str = _DEFAULT_MODEL,
                 mmproj: str | None = None,
                 image: str | None = None,
                 variations: int = 1,
                 max_length: int = 100,
                 temperature: float = 0.7,
                 top_k: int = 40,
                 top_p: float = 0.4,
                 min_p: float = 0.0,
                 thinking: bool = False,
                 system: str | None = None,
                 preamble: str | None = None,
                 remove_prompt: bool = False,
                 prepend_prompt: bool = False,
                 gpu_layers: int | str = 'auto',
                 block_regex: str | None = None,
                 max_attempts: int = 10,
                 context_tokens: int = 2048,
                 smart_truncate: bool = False,
                 cleanup_config: str | None = None,
                 **kwargs
                 ):
        super().__init__(**kwargs,
                         part=part,
                         block_regex=block_regex,
                         max_attempts=max_attempts,
                         cleanup_mode='magic' if 'magicprompt' in model.lower() else 'other',
                         smart_truncate=smart_truncate,
                         cleanup_config=cleanup_config)

        if xllamacpp is None:
            raise _exceptions.PromptUpscalerError(
                'Cannot use the llama prompt upscaler without xllamacpp being installed. '
                'Install it with: pip install dgenerate[xllamacpp]'
            )

        part = part.lower()

        if max_length < 1:
            raise self.argument_error('Cannot specify "max-length" less than 1.')

        if variations < 1:
            raise self.argument_error('Argument "variations" may not be less than 1.')

        if temperature < 0.0:
            raise self.argument_error('Argument "temperature" may not be less than 0.')

        if top_k < 1:
            raise self.argument_error('Argument "top-k" may not be less than 1.')

        if top_p < 0.0:
            raise self.argument_error('Argument "top-p" may not be less than 0.')

        if top_p > 1.0:
            raise self.argument_error('Argument "top-p" may not be greater than 1.')

        if min_p < 0.0:
            raise self.argument_error('Argument "min-p" may not be less than 0.')

        if min_p > 1.0:
            raise self.argument_error('Argument "min-p" may not be greater than 1.')

        try:
            gpu_layers = _resolve_gpu_layers(gpu_layers)
        except ValueError:
            raise self.argument_error(
                'Argument "gpu-layers" must be "auto", "all", or an integer.'
            )

        image = _blank_to_none(image)
        mmproj = _blank_to_none(mmproj)
        if image and not mmproj:
            raise self.argument_error(
                'Argument "image" requires "mmproj", the GGUF multimodal projector for "model".')
        if mmproj and not image:
            raise self.argument_error(
                'Argument "mmproj" requires "image".')

        image_data_url = None
        if image:
            try:
                image_path = _util.resolve_input_image(
                    image, local_files_only=self.local_files_only)
                image_data_url = _util.png_data_url(_util.load_rgb_image(image_path))
            except ValueError as e:
                raise self.argument_error(str(e)) from e
            if context_tokens < 4096:
                _messages.warning(
                    'llama prompt upscaler: "context-tokens" is '
                    f'{context_tokens}. Vision models often need 4096 or more '
                    'tokens once an image is included.')

        if mmproj:
            mmproj = self._resolve_gguf('mmproj', mmproj)
        model = self._resolve_gguf('model', model)

        estimated_size = os.stat(model).st_size
        if mmproj:
            estimated_size += os.stat(mmproj).st_size

        _messages.debug_log(
            f'Estimated the size of LLM model: '
            f'{model}, as: {estimated_size} Bytes ({_memory.bytes_best_human_unit(estimated_size)})')

        def load_method():
            if gpu_layers != 0:
                self.memory_guard_device(self.device, estimated_size)
            params = xllamacpp.CommonParams()
            params.model.path = model
            if mmproj:
                params.mmproj.path = mmproj
                params.mmproj_use_gpu = gpu_layers != 0
            params.n_ctx = context_tokens
            params.n_predict = max_length
            params.n_gpu_layers = gpu_layers
            params.sampling.temp = temperature
            params.sampling.top_k = top_k
            params.sampling.top_p = top_p
            params.sampling.min_p = min_p
            params.hostname = '127.0.0.1'
            params.port = _free_localhost_port()
            # llama.cpp defaults to info (3), which prints model load, slot, and timing lines on stderr.
            # 1 keeps errors. Debug logging restores the info stream.
            debug = bool(_messages.LEVEL & _messages.DEBUG)
            params.verbosity = 3 if debug else 1
            params.show_timings = debug
            params.display_prompt = debug
            try:
                return xllamacpp.Server(params)
            except Exception as e:
                raise self.argument_error(f'Argument "model", could not load: {e}') from e

        self.set_size_estimate(estimated_size)

        self._server = self.load_object_cached(
            tag='|'.join((model, mmproj or '', str(context_tokens), str(gpu_layers))),
            estimated_size=estimated_size,
            method=load_method
        )

        self._image_data_url = image_data_url
        self._preamble = preamble
        self._remove_prompt = remove_prompt
        self._max_length = max_length
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._min_p = min_p
        self._thinking = thinking
        self._system = system
        self._variations = variations
        self._part = part
        self._max_attempts = max_attempts
        self._prepend_prompt = prepend_prompt

    def _resolve_gguf(self, argument: str, value: str) -> str:
        if _webcache.is_downloadable_url(value):
            try:
                return _hfhub.webcache_or_hf_blob_download(
                    value,
                    local_files_only=self.local_files_only,
                    auth_token=self.auth_token)
            except Exception as e:
                raise self.argument_error(
                    f'Could not download argument "{argument}": "{value}", error: {e}') from e
        if not os.path.exists(value):
            raise self.argument_error(
                f'Argument "{argument}" must be a path to a GGUF file on disk if not a URL.')
        return os.path.abspath(value)

    def _user_content(self, text: str):
        if not self._image_data_url:
            return text
        content = [{
            'type': 'image_url',
            'image_url': {'url': self._image_data_url},
        }]
        if text:
            content.append({'type': 'text', 'text': text})
        return content

    def _complete(self, messages: list[dict]) -> str:
        result = self._server.handle_chat_completions({
            'max_tokens': self._max_length,
            'temperature': self._temperature,
            'top_k': self._top_k,
            'top_p': self._top_p,
            'min_p': self._min_p,
            'messages': messages,
            # Templates that do not use this variable ignore it.
            'chat_template_kwargs': {'enable_thinking': self._thinking},
        })
        if isinstance(result, (bytes, bytearray)):
            result = result.decode('utf-8', errors='replace')
        if isinstance(result, str):
            result = json.loads(result)
        if not isinstance(result, dict) or not result.get('choices'):
            error = result.get('error', result) if isinstance(result, dict) else result
            if isinstance(error, dict):
                error = error.get('message', error)
            raise RuntimeError(str(error))
        message = result['choices'][0]['message']
        text = _chat_text(message.get('content')).strip()
        if not text:
            text = _chat_text(message.get('reasoning_content')).strip()
        return text

    def _generate_prompts(self, original_prompts: list[str]) -> list[str]:
        def user_text(text):
            if self._preamble:
                return self._preamble + (' ' if not self._preamble.endswith(' ') else '') + text
            return text

        generated_prompts = []
        for query in original_prompts:
            messages = []
            if self._system:
                system_content = self._system
                if self._image_data_url:
                    system_content = [{'type': 'text', 'text': self._system}]
                messages.append({'role': 'system', 'content': system_content})
            messages.append({'role': 'user', 'content': self._user_content(user_text(query))})
            generated_prompts.append(self._complete(messages))

        generated_prompts = [
            self._clean_prompt(
                user_text(original_prompt),
                generated_prompt,
                remove_prefixes=[self._system, self._preamble],
                remove_prompt=self._remove_prompt,
                prepend=original_prompt if self._prepend_prompt else None,
            )
            for original_prompt, generated_prompt in zip(original_prompts, generated_prompts)
        ]

        return generated_prompts

    @property
    def accepts_batch(self) -> bool:
        """
        Returns ``True``. This prompt upscaler can accept a batch of prompts.
        """
        return True

    def upscale(self, prompts: _prompt.PromptOrPrompts) -> _prompt.PromptOrPrompts:
        if isinstance(prompts, _prompt.Prompt):
            prompts = [prompts]

        prompts = list(prompts) * self._variations

        try:
            return self._process_prompts(prompts)
        except Exception as e:
            raise _exceptions.PromptUpscalerProcessingError(
                f'xllamacpp prompt upscaler could not process prompt(s): {e}'
            ) from e
