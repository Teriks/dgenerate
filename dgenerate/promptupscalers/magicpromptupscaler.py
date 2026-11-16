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
import copy
import gc
import os
import re
import typing

import torch
import transformers

import dgenerate.hfhub as _hfhub
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.util as _pipelinewrapper_util
import dgenerate.prompt as _prompt
import dgenerate.promptupscalers.exceptions as _exceptions
import dgenerate.promptupscalers.llmupscalermixin as _llmupscalermixin
import dgenerate.promptupscalers.promptupscaler as _promptupscaler
import dgenerate.promptupscalers.util as _util
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


def _sampling_generate_kwargs(prompt_length: int, temperature: float, top_k: int, top_p: float,
                              presence_penalty: float, max_new_tokens: int) -> dict:
    """Sampling arguments for ``model.generate``.

    Transformers does not take ``presence_penalty``. A non-zero value is applied
    by lowering each already generated token once, after the prompt.
    """
    generate_kwargs = {'max_new_tokens': max(1, int(max_new_tokens))}
    if temperature > 0:
        generate_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    else:
        generate_kwargs['do_sample'] = False
    if presence_penalty:
        from dgenerate.pipelinewrapper.promptenhance import _PresencePenaltyLogitsProcessor
        generate_kwargs['logits_processor'] = [
            _PresencePenaltyLogitsProcessor(presence_penalty, prompt_length)]
    return generate_kwargs


def _eos_token_id(config):
    """One end-of-sequence id. Gemma configs store a list of them."""
    eos = getattr(config, 'eos_token_id', None)
    if isinstance(eos, (list, tuple)):
        eos = eos[0] if eos else None
    return eos


def _model_input_device(model) -> torch.device:
    """Device that holds the token embedding after a GPU/CPU layer split."""
    from dgenerate.pipelinewrapper.promptenhance import _embedding_device
    try:
        return _embedding_device(model)
    except StopIteration:
        return torch.device('cpu')


def _attn_implementation_for_config(config) -> str | None:
    """
    Phi-3 128k (and similar) set ``sliding_window``. Without flash-attn
    window support, prefer eager attention.
    """
    if getattr(config, 'sliding_window', None) is not None:
        return 'eager'
    return None


def _legacy_rope_scaling_to_parameters(config_dict: dict) -> dict | None:
    """
    Convert checkpoint ``rope_scaling`` (``su`` / ``yarn`` / ``longrope``)
    into transformers 5 ``rope_parameters`` so native Phi-3 can load without
    ``trust_remote_code``. Remote ``modeling_phi3`` mis-shapes RoPE under
    current ``generate()`` and crashes with batch/seq size mismatches.
    """
    rope_scaling = config_dict.get('rope_scaling')
    if not isinstance(rope_scaling, dict):
        return None
    if config_dict.get('rope_parameters'):
        return None

    rope_type = rope_scaling.get('type') or rope_scaling.get('rope_type') or 'default'
    if rope_type in ('su', 'yarn'):
        rope_type = 'longrope'

    original_max = (
        rope_scaling.get('original_max_position_embeddings')
        or config_dict.get('original_max_position_embeddings')
        or 4096
    )
    max_pos = config_dict.get('max_position_embeddings') or original_max
    params = {
        'rope_type': rope_type,
        'original_max_position_embeddings': int(original_max),
        'factor': float(max_pos) / float(original_max),
    }
    if 'short_factor' in rope_scaling:
        params['short_factor'] = rope_scaling['short_factor']
    if 'long_factor' in rope_scaling:
        params['long_factor'] = rope_scaling['long_factor']
    if 'rope_theta' in config_dict:
        params['rope_theta'] = config_dict['rope_theta']
    return params


def _try_native_causal_lm_config(model_name: str, local_files_only: bool):
    """
    Build a native transformers config when the hub JSON points at outdated
    remote code (``auto_map``) that breaks on modern generate/RoPE.
    Returns ``None`` to keep ``trust_remote_code``.
    """
    try:
        config_dict, _ = transformers.PretrainedConfig.get_config_dict(
            model_name, local_files_only=local_files_only)
    except Exception:
        return None

    if config_dict.get('model_type') != 'phi3':
        return None

    config_dict = dict(config_dict)
    config_dict.pop('auto_map', None)
    rope_parameters = _legacy_rope_scaling_to_parameters(config_dict)
    if rope_parameters is not None:
        config_dict['rope_parameters'] = rope_parameters
        config_dict.pop('rope_scaling', None)

    # Drop hub-only / legacy keys that native Phi3Config rejects.
    for drop_key in (
            '_name_or_path',
            '_commit_hash',
            'transformers_version',
            'torch_dtype',
            'auto_map',
            'rope_scaling',
            'rope_theta',
    ):
        config_dict.pop(drop_key, None)

    try:
        return transformers.Phi3Config(**config_dict)
    except Exception as e:
        _messages.debug_log(
            f'magicprompt: could not build native Phi3Config for '
            f'"{model_name}": {e}')
        return None


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
        temperature = float(kwargs.pop('temperature', 0.0))
        top_k = int(kwargs.pop('top_k', 50))
        top_p = float(kwargs.pop('top_p', 1.0))
        presence_penalty = float(kwargs.pop('presence_penalty', 0.0) or 0.0)
        new_tokens_only = bool(kwargs.pop('new_tokens_only', False))
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i: i + batch_size]

            # A chat transcript is not part of the reply. max-length is the
            # reply. Truncating the transcript to that budget leaves only the
            # opening sentence.
            input_limit = max_length
            if new_tokens_only:
                model_limit = getattr(self.tokenizer, 'model_max_length', None)
                if not isinstance(model_limit, int) or model_limit > 100_000:
                    model_limit = 8192
                input_limit = model_limit
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=input_limit)

            model_device = _model_input_device(self.model)

            inputs = inputs.to(model_device)

            # Prefer max_new_tokens so total length is prompt + generation,
            # not a hard cap that can equal the already-truncated prompt.
            prompt_len = int(inputs['input_ids'].shape[-1])
            if new_tokens_only:
                reply_tokens = max(1, int(max_length))
            else:
                reply_tokens = max(1, int(max_length) - prompt_len)
            generate_kwargs = dict(kwargs)
            generate_kwargs.update(_sampling_generate_kwargs(
                prompt_len, temperature, top_k, top_p, presence_penalty,
                reply_tokens))

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generate_kwargs)

            if model_device.type != 'cpu':
                inputs.to('cpu')

            del inputs

            results.extend(
                self.tokenizer.decode(
                    output[prompt_len:] if new_tokens_only else output,
                    skip_special_tokens=new_tokens_only)
                for output in outputs)

        return results


class _ImageTextGenerationPipeline:
    def __init__(self, model, processor, quantized: bool):
        self.model = model
        self.processor = processor
        self.quantized = quantized

    def to(self, device):
        if not self.quantized:
            self.model.to(device)

    def _tokenize(self, conversations: list, processor_kwargs: dict):
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors='pt',
            processor_kwargs=processor_kwargs,
        )
        inputs = inputs.to(_model_input_device(self.model))
        dtype = getattr(self.model, 'dtype', None)
        if dtype is not None and dtype.is_floating_point and dtype != torch.float32:
            for key, value in inputs.items():
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    inputs[key] = value.to(dtype=dtype)
        return inputs

    def _without_uneven_slices(self, conversations: list, inputs, processor_kwargs: dict):
        """
        MiniCPM-V packs an image into slices and then requires every slice to
        have the same number of tokens. Uneven slices crash the vision merger,
        so those images are encoded as one slice instead.
        """
        sizes = inputs.get('target_sizes') if hasattr(inputs, 'get') else None
        if not torch.is_tensor(sizes) or sizes.ndim != 2 or sizes.shape[0] < 2 or sizes.shape[-1] < 2:
            return inputs
        areas = sizes[:, 0] * sizes[:, 1]
        if torch.all(areas == areas[0]):
            return inputs
        _messages.debug_log(
            'magicprompt: image slices have uneven sizes, encoding the image as one slice.')
        retried = dict(processor_kwargs)
        images_kwargs = dict(retried.get('images_kwargs') or {})
        images_kwargs['slice_mode'] = False
        retried['images_kwargs'] = images_kwargs
        return self._tokenize(copy.deepcopy(conversations), retried)

    def generate(self,
                 conversations: list,
                 max_new_tokens: int,
                 temperature: float,
                 top_k: int,
                 top_p: float,
                 presence_penalty: float = 0.0) -> list[str]:
        processor_kwargs = {'padding': True} if len(conversations) > 1 else {}
        # The processor rewrites message content in place. Keep a copy in case
        # the image has to be encoded again without slicing.
        original_conversations = copy.deepcopy(conversations)
        inputs = self._tokenize(conversations, processor_kwargs)
        inputs = self._without_uneven_slices(original_conversations, inputs, processor_kwargs)

        prompt_width = int(inputs['input_ids'].shape[-1])
        generate_kwargs = _sampling_generate_kwargs(
            prompt_width, temperature, top_k, top_p, presence_penalty, max_new_tokens)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)
        if not torch.is_tensor(outputs):
            outputs = outputs.sequences
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(0)

        decode = getattr(self.processor, 'batch_decode', None)
        if decode is None:
            decode = self.processor.tokenizer.batch_decode
        decoded = decode(
            outputs[:, prompt_width:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [text.strip() for text in decoded]


class MagicPromptUpscaler(_llmupscalermixin.LLMPromptUpscalerMixin, _promptupscaler.PromptUpscaler):
    """
    Upscale prompts using magicprompt or other LLMs via transformers.

    The "part" argument indicates which parts of the prompt to act on,
    possible values are: "both", "positive", and "negative"

    The "model" specifies the model path for magicprompt, the
    default value is: "Gustavosta/MagicPrompt-Stable-Diffusion". This
    can be a folder on disk, a Hugging Face repository slug, or a GGUF
    file path or URL. "weight-name" selects one .gguf file in a repository
    or folder. The class is the text encoder dgenerate would load for that
    file: the architecture and the embedding shape pick it, and the config
    is one dgenerate ships for that shape. A quant repository does not
    supply config.json, and a config.json next to the file is not read.
    The tokenizer comes from the repository, or from the vocabulary stored
    in the GGUF.

    "mmproj" is the llama.cpp vision-tower GGUF when that tower is a separate
    file. It is that file's path or URL, including a blob or resolve link.
    A file name is not looked up beside the language GGUF or in its repository.
    With "image", a language-only GGUF needs this tower.

    An unquantized model keeps as many decoder layers as fit in free GPU
    memory. The rest stay in CPU memory and run on the GPU one layer at a
    time. Quantized models, including a GGUF, stay on the single device
    given by "device". A GGUF is already quantized, so "quantizer" does not
    apply to it.

    The "image" argument is a path or URL to an image. When it is set,
    "model" must be a vision-language model that transformers can load,
    such as heretic-org/MiniCPM-V-4.6-heretic. The image is attached to
    every prompt, and the prompt text is the instruction, for example a
    request to write an image-generation prompt for the picture. Animated
    files use the first frame. With an image, "max-length" is the maximum
    number of new tokens to generate.

    The "dtype" argument specifies the torch dtype (compute dtype) to load
    the model with, this defaults to: float32, and may be one of: float32,
    float16, or bfloat16. With bitsandbytes 8-bit quantization, float32 is
    coerced to float16 (8-bit MatMul always runs in float16).

    The "seed" argument can be used to specify a seed for prompt generation.

    The "variations" argument specifies how many variations should be produced.

    The "max-length" argument is the max prompt length for a
    generated prompt, this value defaults to 100. With a chat template
    it is the length of the reply. The transcript is not counted.

    The "temperature" argument sets the sampling temperature
    to use when generating prompts. Larger values increase
    creativity but decrease factuality.

    The "top_k" argument sets the "top_k" generation value, i.e.
    randomly sample from the "top_k" most likely tokens at each generation step.
    Set this to 1 for greedy decoding.

    The "top_p" argument sets the "top_p" generation value, i.e.
    randomly sample at each generation step from the top most likely
    tokens whose probabilities add up to "top_p".

    The "presence-penalty" argument lowers the score of each token that has
    already been generated, once. 0 leaves those tokens alone. A positive
    value reduces repetition. A negative value encourages it.

    The "system" argument sets the system instruction for the LLM.
    A value of "file:path.txt" reads that instruction from a UTF-8 text file. Shell-quote
    the path when it contains spaces.     A tokenizer that has a chat template
    receives the system instruction and the preamble through that template.
    A template that does not render a system role gets that instruction
    at the start of the user message. Thinking is left off when the
    template accepts that.

    The "preamble" argument sets a text input preamble for the LLM, this
    preamble will be removed from the output generated by the LLM.
    A value of "file:path.txt" reads that preamble from a UTF-8 text file. Shell-quote
    the path when it contains spaces.

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

    The "gpu-headroom" argument is how many GiB of free GPU memory to leave
    unused when splitting an unquantized model across GPU and CPU, on top of
    10 percent of free memory. The default is 2. Raise it when generation
    runs out of memory. Lower it to keep more layers on the GPU. Quantized
    models ignore it.

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
        'model': {
            'mode': ['in', 'dir'],
            'filetypes': [('GGUF', ['*.gguf'])],
        },
        'weight-name': {'mode': 'in', 'filetypes': [('GGUF', ['*.gguf'])]},
        'mmproj': {'mode': 'in', 'filetypes': [('GGUF', ['*.gguf'])]},
        'image': _util.image_file_argument(),
        'cleanup-config': {'mode': 'in', 'filetypes': [('Cleanup Config', ('*.json', '*.toml', '*.yaml', '*.yml'))]}
    }

    def __init__(self,
                 part: str = 'both',
                 model: str = "Gustavosta/MagicPrompt-Stable-Diffusion",
                 weight_name: str | None = None,
                 mmproj: str | None = None,
                 image: str | None = None,
                 dtype: str = 'float32',
                 seed: int | None = None,
                 variations: int = 1,
                 max_length: int = 100,
                 temperature: float = 0.7,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 presence_penalty: float = 0.0,
                 system: str | None = None,
                 preamble: str | None = None,
                 remove_prompt: bool = False,
                 prepend_prompt: bool = False,
                 batch: bool = True,
                 max_batch: int | None = 50,
                 quantizer: str | None = None,
                 gpu_headroom: float | None = None,
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

        system = self._resolve_file_argument(system, 'system')
        preamble = self._resolve_file_argument(preamble, 'preamble')

        dtype = dtype.lower()
        if dtype not in {'float32', 'float16', 'bfloat16'}:
            raise self.argument_error('Argument "dtype" must be either float32, float16, or bfloat16.')

        if gpu_headroom is not None and gpu_headroom < 0:
            raise self.argument_error('Argument "gpu-headroom" must be at least 0.')

        if quantizer:
            try:
                quantizer_class = _get_quantizer_uri_class(quantizer)
                quantization_config = quantizer_class.parse(quantizer).to_transformers_config(dtype)
            except Exception as e:
                raise self.argument_error(f'Error loading "quantizer" argument "{quantizer}": {e}') from e
        else:
            quantization_config = None

        weight_name = str(weight_name).strip() if weight_name else None
        mmproj = str(mmproj).strip() if mmproj else None
        gguf = _hfhub.is_gguf_model(model) or _hfhub.is_gguf_model(weight_name or '')
        if weight_name and not _hfhub.is_gguf_model(weight_name):
            raise self.argument_error('Argument "weight-name" must name a .gguf file.')
        if gguf and quantizer:
            raise self.argument_error(
                'A GGUF model is already quantized. Do not set the "quantizer" argument.')
        if mmproj and not gguf:
            raise self.argument_error(
                'Argument "mmproj" is a GGUF vision tower and requires a GGUF "model".')

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

        if image is not None:
            image = str(image).strip() or None
        vision_image = None
        if image:
            try:
                image_path = _util.resolve_input_image(
                    image, local_files_only=self.local_files_only)
                vision_image = _util.load_rgb_image(image_path)
            except ValueError as e:
                raise self.argument_error(str(e)) from e

        mmproj_path = None
        gguf_path = None
        if gguf:
            try:
                gguf_path, _subfolder = _hfhub.materialize_named_weight(
                    model,
                    weight_name=weight_name,
                    local_files_only=self.local_files_only,
                    auth_token=self.auth_token)
            except Exception as error:
                raise self.argument_error(
                    f'Could not find GGUF "{weight_name or model}": {error}') from error
            if not os.path.isfile(gguf_path):
                raise self.argument_error(
                    f'Argument "model" must be a GGUF file on disk or a URL, got "{model}".')
            estimated_size = os.path.getsize(gguf_path)
            if mmproj:
                from dgenerate.pipelinewrapper.uris.textencoderuri import _materialize_mmproj
                from dgenerate.pipelinewrapper.uris.exceptions import TextEncoderUriLoadError
                try:
                    mmproj_path = _materialize_mmproj(
                        mmproj, self.local_files_only, self.auth_token)
                except TextEncoderUriLoadError as error:
                    raise self.argument_error(str(error)) from error
                if not _hfhub.is_gguf_model(mmproj_path):
                    raise self.argument_error(
                        f'Argument "mmproj" must be a .gguf file, got "{mmproj}".')
                estimated_size += os.path.getsize(mmproj_path)
        else:
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

            if isinstance(quantization_config, transformers.BitsAndBytesConfig):
                if quantization_config.load_in_4bit and quantization_config.bnb_4bit_compute_dtype is None:
                    quantization_config.bnb_4bit_compute_dtype = torch_dtype
                # 8-bit MatMul always runs in float16; float32 inputs spam
                # "MatMul8bitLt: inputs will be cast..." on every layer.
                if quantization_config.load_in_8bit and torch_dtype == torch.float32:
                    _messages.debug_log(
                        'magicprompt: using float16 with bitsandbytes 8-bit '
                        '(float32 inputs would be cast every MatMul).')
                    torch_dtype = torch.float16

            if gguf:
                if (self.device or 'cpu') != 'cpu':
                    self.memory_guard_device(self.device, self.size_estimate)
                return self._load_gguf_pipeline(
                    model, gguf_path, mmproj_path, dtype=torch_dtype,
                    vision=vision_image is not None)
            if vision_image is not None:
                return self._load_vision_pipeline(
                    model, dtype=torch_dtype, quantization_config=quantization_config)
            return self._load_pipeline(model, dtype=torch_dtype, quantization_config=quantization_config)

        self.set_size_estimate(estimated_size)

        self._gguf = gguf
        self._pipeline = self.load_object_cached(
            tag='|'.join((
                model, weight_name or '', mmproj or '',
                quantizer or '', dtype,
                'vision' if vision_image is not None else 'text')),
            estimated_size=estimated_size,
            method=load_method
        )

        self._vision_image = vision_image
        self._system = system
        self._preamble = preamble
        self._remove_prompt = remove_prompt
        self._seed = seed
        self._max_length = max_length
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._presence_penalty = presence_penalty

        self._variations = variations
        self._accepts_batch = batch
        self._max_batch = max_batch
        self._part = part
        self._quantizer = quantizer
        self._gpu_headroom = gpu_headroom
        self._max_attempts = max_attempts
        self._prepend_prompt = prepend_prompt


    def _gguf_repository(self, model: str):
        """Repository named by a slug or a blob link, for the config and tokenizer."""
        blob = _hfhub.HFBlobLink.parse(model)
        if blob is not None:
            return blob.repo_id, blob.revision, blob.subfolder
        if os.path.isdir(model) or _hfhub.is_single_file_model_load(model):
            return None, None, None
        return model, None, None

    def _pretrained_source(self, model: str, model_path: str) -> tuple[str, dict]:
        repo_id, revision, subfolder = self._gguf_repository(model)
        kwargs = {
            'token': self.auth_token,
            'local_files_only': self.local_files_only,
        }
        if repo_id:
            kwargs['revision'] = revision
            if subfolder:
                kwargs['subfolder'] = subfolder
            return repo_id, kwargs
        return os.path.dirname(model_path), kwargs

    def _place_gguf(self, model):
        device = self.device or 'cpu'
        if device == 'cpu':
            return model
        try:
            return model.to(device)
        except Exception as error:
            raise self.argument_error(
                f'Could not move the GGUF model to "{device}": {error}') from error

    def _vision_weights(self, model) -> bool | None:
        """True when a vision tensor holds weights, False when those tensors are empty, None when there are none."""
        from dgenerate.pipelinewrapper.ggufcompat import _is_gguf_parameter
        saw = False
        for name, param in model.named_parameters():
            if not any(token in name for token in ('vision', 'visual', 'embed_vision')):
                continue
            saw = True
            if _is_gguf_parameter(param):
                return True
            if param.device.type == 'meta':
                continue
            sample = param.detach().view(-1)[:1024]
            if int(torch.count_nonzero(sample)) > 0:
                return True
        return False if saw else None

    def _load_gguf_pipeline(self, model: str, model_path: str, mmproj_path: str | None,
                            dtype: torch.dtype, vision: bool):
        from dgenerate.pipelinewrapper.uris.exceptions import TextEncoderUriLoadError
        from dgenerate.pipelinewrapper.uris.textencoderuri import _load_gguf_text_encoder

        try:
            loaded = _load_gguf_text_encoder(
                None, model_path, dtype, mmproj_path=mmproj_path)
        except TextEncoderUriLoadError as error:
            raise self.argument_error(str(error)) from error

        if vision:
            present = self._vision_weights(loaded)
            if present is None:
                raise self.argument_error(
                    f'GGUF "{model_path}" is not a vision-language model. '
                    'Omit "image", or use a vision GGUF.')
            if not present:
                raise self.argument_error(
                    f'GGUF "{model_path}" has no vision-tower weights. '
                    'Set "mmproj" to that tower\'s path or URL.')

        source, source_kwargs = self._pretrained_source(model, model_path)
        loaded = self._place_gguf(loaded)
        if vision:
            try:
                processor = transformers.AutoProcessor.from_pretrained(source, **source_kwargs)
            except Exception as error:
                raise self.argument_error(
                    f'Could not load the image processor for "{model}": {error}') from error
            return _ImageTextGenerationPipeline(loaded, processor, quantized=True)

        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(source, **source_kwargs)
        except Exception as missing:
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    os.path.dirname(model_path),
                    gguf_file=os.path.basename(model_path),
                    local_files_only=True)
            except Exception:
                # Gemma 4 GGUF metadata stores intermediate_size as a list.
                # AutoTokenizer rejects that config. The vocabulary is still
                # readable without building the model config.
                try:
                    tokenizer = transformers.TokenizersBackend.from_pretrained(
                        os.path.dirname(model_path),
                        gguf_file=os.path.basename(model_path),
                        local_files_only=True)
                except Exception as error:
                    raise self.argument_error(
                        f'Could not load the tokenizer for "{model}": {missing}') from error
        if str(model_path).lower().endswith('.gguf'):
            from dgenerate.pipelinewrapper.uris.promptupsampleruri import (
                _mark_gguf_control_tokens,
            )
            _mark_gguf_control_tokens(tokenizer, model_path)
        tokenizer.pad_token_id = _eos_token_id(loaded.config)
        return _TextGenerationPipeline(loaded, tokenizer, quantized=True)

    def _load_pipeline(self,
                       model_name: str,
                       dtype: torch.dtype,
                       quantization_config: typing.Optional[typing.Any] = None) -> _TextGenerationPipeline:

        native_config = _try_native_causal_lm_config(
            model_name, self.local_files_only)
        trust_remote_code = native_config is None

        if native_config is not None:
            config = native_config
            _messages.debug_log(
                f'magicprompt: loading "{model_name}" with native '
                f'{type(config).__name__} (skipping outdated remote modeling).')
        else:
            try:
                config = transformers.AutoConfig.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    local_files_only=self.local_files_only,
                    token=self.auth_token
                )
            except Exception as e:
                raise self.argument_error(
                    f'Could not load config for "{model_name}": {e}')

        load_kwargs: dict[str, typing.Any] = dict(
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            quantization_config=quantization_config,
            device_map=self.device if quantization_config else None,
            local_files_only=self.local_files_only,
            token=self.auth_token,
            config=config,
        )
        attn_implementation = _attn_implementation_for_config(config)
        if attn_implementation is not None:
            load_kwargs['attn_implementation'] = attn_implementation
            _messages.debug_log(
                f'magicprompt: using attn_implementation="{attn_implementation}" '
                f'for "{model_name}" (sliding_window='
                f'{getattr(config, "sliding_window", None)}).')

        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs)
        except Exception as e:
            raise self.argument_error(f'Could not load model "{model_name}": {e}')

        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
                token=self.auth_token
            )
        except Exception as e:
            raise self.argument_error(f'Could not load tokenizer for "{model_name}": {e}')
        tokenizer.pad_token_id = _eos_token_id(model.config)

        return _TextGenerationPipeline(
            model, tokenizer,
            quantized=quantization_config is not None
        )

    def _load_vision_pipeline(self,
                              model_name: str,
                              dtype: torch.dtype,
                              quantization_config: typing.Optional[typing.Any] = None
                              ) -> _ImageTextGenerationPipeline:
        try:
            config = transformers.AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
                token=self.auth_token,
            )
        except Exception as e:
            raise self.argument_error(
                f'Could not load config for "{model_name}": {e}') from e

        load_kwargs: dict[str, typing.Any] = dict(
            trust_remote_code=True,
            dtype=dtype,
            quantization_config=quantization_config,
            device_map=self.device if quantization_config else None,
            local_files_only=self.local_files_only,
            token=self.auth_token,
            config=config,
        )
        attn_implementation = _attn_implementation_for_config(config)
        if attn_implementation is not None:
            load_kwargs['attn_implementation'] = attn_implementation

        try:
            model = transformers.AutoModelForImageTextToText.from_pretrained(
                model_name, **load_kwargs)
        except Exception as e:
            raise self.argument_error(
                f'Could not load "{model_name}" as a vision-language model for '
                f'argument "image". Use an image-text model such as '
                f'heretic-org/MiniCPM-V-4.6-heretic. {e}') from e

        try:
            processor = transformers.AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
                token=self.auth_token,
            )
        except Exception as e:
            raise self.argument_error(
                f'Could not load the image processor for "{model_name}": {e}') from e

        return _ImageTextGenerationPipeline(
            model, processor,
            quantized=quantization_config is not None,
        )

    def _generate_prompts(self, original_prompts: list[str]) -> list[str]:
        if self._vision_image is not None:
            return self._generate_vision_prompts(original_prompts)

        formatted_prompts = [
            self._format_text_prompt(query) for query in original_prompts]
        chat_template = self._text_chat_template()

        if not self._accepts_batch:
            generated_prompts = []
            for ptext in formatted_prompts:
                generated_prompts.extend(
                    self._call_pipeline([ptext], new_tokens_only=chat_template))
        elif self._max_batch is not None:
            generated_prompts = []
            for batch_segment in range(0, len(formatted_prompts), self._max_batch):
                segment = formatted_prompts[batch_segment:batch_segment + self._max_batch]
                generated_prompts.extend(
                    self._call_pipeline(segment, new_tokens_only=chat_template))
        else:
            generated_prompts = self._call_pipeline(
                formatted_prompts, new_tokens_only=chat_template)

        if chat_template:
            generated_prompts = [self._strip_thinking(text) for text in generated_prompts]

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

    def _text_chat_template(self):
        tokenizer = getattr(self._pipeline, 'tokenizer', None)
        return bool(getattr(tokenizer, 'chat_template', None))

    def _template_renders_system(self, tokenizer) -> bool:
        """Whether this chat template writes a system message into the prompt."""
        cached = getattr(self, '_chat_template_renders_system', None)
        if cached is not None:
            return cached
        probe = 'magicprompt-system-role-probe'
        try:
            rendered = tokenizer.apply_chat_template(
                [
                    {'role': 'system', 'content': probe},
                    {'role': 'user', 'content': 'prompt'},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            renders = probe in str(rendered)
        except Exception:
            renders = True
        self._chat_template_renders_system = renders
        return renders

    def _chat_messages(self, tokenizer, user: str) -> list[dict]:
        if not self._system:
            return [{'role': 'user', 'content': user}]
        if self._template_renders_system(tokenizer):
            return [
                {'role': 'system', 'content': self._system},
                {'role': 'user', 'content': user},
            ]
        # Phi-3's template skips a system role. llama.cpp pastes that text
        # onto the user message, and the instruction is then part of the prompt.
        _messages.debug_log(
            'magicprompt: chat template does not render a system role, '
            'so the system instruction is placed at the start of the user message.')
        return [{'role': 'user', 'content': f'{self._system}\n{user}'}]

    def _format_text_prompt(self, query: str) -> str:
        if self._preamble:
            user = self._preamble + (' ' if not self._preamble.endswith(' ') else '') + query
        else:
            user = query
        tokenizer = getattr(self._pipeline, 'tokenizer', None)
        if not getattr(tokenizer, 'chat_template', None):
            if self._system:
                return f"<|system|> {self._system} <|user|> {user} <|assistant|>"
            return user
        messages = self._chat_messages(tokenizer, user)
        # Qwen3's template accepts this and writes an empty think block,
        # so the reply is the prompt. A template without the variable
        # raises, and the plain template is used.
        if getattr(self, '_chat_template_no_thinking', True):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
            except Exception:
                self._chat_template_no_thinking = False
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    def _strip_thinking(self, text: str) -> str:
        return re.sub(
            r'<think>.*?</think>', '', text or '', flags=re.DOTALL | re.IGNORECASE).strip()

    def _query_text(self, text: str) -> str:
        if self._preamble:
            return self._preamble + (' ' if not self._preamble.endswith(' ') else '') + text
        return text

    def _vision_conversations(self, texts: list[str]) -> list:
        tokenizer = getattr(self._pipeline, 'tokenizer', None)
        if tokenizer is None:
            tokenizer = getattr(getattr(self._pipeline, 'processor', None), 'tokenizer', None)
        merge_system = bool(
            self._system and tokenizer is not None
            and getattr(tokenizer, 'chat_template', None)
            and not self._template_renders_system(tokenizer))
        conversations = []
        for text in texts:
            messages = []
            if merge_system:
                user_text = self._system if not text else f'{self._system}\n{text}'
            else:
                user_text = text
            if self._system and not merge_system:
                messages.append({
                    'role': 'system',
                    'content': [{'type': 'text', 'text': self._system}],
                })
            content = [{'type': 'image', 'image': self._vision_image.copy()}]
            if user_text:
                content.append({'type': 'text', 'text': user_text})
            messages.append({'role': 'user', 'content': content})
            conversations.append(messages)
        return conversations

    def _generate_vision_prompts(self, original_prompts: list[str]) -> list[str]:
        user_texts = [self._query_text(query) for query in original_prompts]
        conversations = self._vision_conversations(user_texts)

        try:
            generated_prompts = self._run_vision_batches(conversations)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:
            raise _exceptions.PromptUpscalerProcessingError(
                f'magicprompt prompt upscaler could not process the image prompt: {e}'
            ) from e

        return [
            self._clean_prompt(
                user_text,
                generated_prompt,
                remove_prefixes=[self._system, self._preamble],
                remove_prompt=self._remove_prompt,
                prepend=original_prompt if self._prepend_prompt else None,
            )
            for original_prompt, user_text, generated_prompt in zip(
                original_prompts, user_texts, generated_prompts
            )
        ]

    def _run_vision_batches(self, conversations: list) -> list[str]:
        if not self._accepts_batch:
            generated_prompts = []
            for conversation in conversations:
                generated_prompts.extend(self._pipeline.generate(
                    [conversation],
                    max_new_tokens=self._max_length,
                    temperature=self._temperature,
                    top_k=self._top_k,
                    top_p=self._top_p,
                    presence_penalty=self._presence_penalty,
                ))
        elif self._max_batch is not None:
            generated_prompts = []
            for batch_segment in range(0, len(conversations), self._max_batch):
                segment = conversations[batch_segment:batch_segment + self._max_batch]
                generated_prompts.extend(self._pipeline.generate(
                    segment,
                    max_new_tokens=self._max_length,
                    temperature=self._temperature,
                    top_k=self._top_k,
                    top_p=self._top_p,
                    presence_penalty=self._presence_penalty,
                ))
        else:
            generated_prompts = self._pipeline.generate(
                conversations,
                max_new_tokens=self._max_length,
                temperature=self._temperature,
                top_k=self._top_k,
                top_p=self._top_p,
                presence_penalty=self._presence_penalty,
            )
        return generated_prompts

    def _call_pipeline(self, prompts: list[str], new_tokens_only: bool = False):
        return self._pipeline(
            prompts,
            max_length=self._max_length,
            temperature=self._temperature,
            top_k=self._top_k,
            top_p=self._top_p,
            presence_penalty=self._presence_penalty,
            batch_size=len(prompts),
            new_tokens_only=new_tokens_only,
        )

    @contextlib.contextmanager
    def _with_device(self):
        if self._quantizer or self._gguf:
            yield
            gc.collect()
            _memory.torch_gc()
            return
        from dgenerate.pipelinewrapper.promptenhance import (
            PromptEnhanceOverrides, _fast_encoder, _headroom_bytes)
        headroom_bytes = None
        if self._gpu_headroom is not None:
            headroom_bytes = _headroom_bytes(
                PromptEnhanceOverrides(gpu_headroom=self._gpu_headroom))
        try:
            self.memory_guard_device(self.device, self.size_estimate)
            with _fast_encoder(
                    self._pipeline.model,
                    self.device,
                    offload=True,
                    headroom_bytes=headroom_bytes,
                    label='magicprompt',
                    debug=True):
                yield
        finally:
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
