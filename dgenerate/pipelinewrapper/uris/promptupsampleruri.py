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

import dgenerate.hfhub as _hfhub
import dgenerate.memoize as _d_memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.constants as _constants
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.sdnqload as _sdnqload
import dgenerate.pipelinewrapper.uris.exceptions as _exceptions
import dgenerate.pipelinewrapper.uris.util as _util
import dgenerate.pipelinewrapper.util as _pipelinewrapper_util
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.memoize import memoize as _memoize


def _mark_gguf_control_tokens(tokenizer, model_path: str) -> None:
    """Mark GGUF control tokens special so decoding skips them.

    Gemma 4 stores ``<turn|>`` as a control token and stops on it. The
    converted tokenizer leaves that piece as ordinary text, so the token
    stays on the rewritten prompt.
    """
    try:
        from gguf import GGUFReader
        from tokenizers import AddedToken
        from transformers.modeling_gguf_pytorch_utils import read_field
    except Exception:
        return
    try:
        reader = GGUFReader(model_path)
        tokens = read_field(reader, 'tokenizer.ggml.tokens')
        kinds = read_field(reader, 'tokenizer.ggml.token_type')
    except Exception as error:
        _messages.debug_log(
            f'Could not read control tokens from "{model_path}": {error}')
        return
    backend = getattr(tokenizer, '_tokenizer', None)
    if backend is None or not tokens or not kinds:
        return
    unk = getattr(tokenizer, 'unk_token_id', None)
    pieces = []
    seen = set()
    for token, kind in zip(tokens, kinds):
        if kind not in (3, 4) or not isinstance(token, str) or token in seen:
            continue
        seen.add(token)
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == unk:
            continue
        raw = tokenizer.decode([token_id], skip_special_tokens=False)
        skipped = tokenizer.decode([token_id], skip_special_tokens=True)
        if raw and raw != skipped:
            continue
        pieces.append(AddedToken(
            token, rstrip=False, lstrip=False, normalized=False, special=True))
    if not pieces:
        return
    try:
        backend.add_special_tokens(pieces)
    except Exception as error:
        _messages.debug_log(
            f'Could not mark GGUF control tokens special: {error}')


class _GGUFTokenizerProcessor:
    """A GGUF vocabulary used as the chat processor when the repository has no processor files.

    Text prompts tokenize. An image needs the image processor beside the GGUF.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        if images is not None:
            raise _exceptions.PromptUpsamplerUriLoadError(
                'This GGUF has no image processor beside it. '
                'A text prompt can be rewritten. An image needs that processor.')
        return self.tokenizer(text, return_tensors=return_tensors, **kwargs)


_prompt_upsampler_uri_parser = _textprocessing.ConceptUriParser(
    'Prompt Upsampler', [
        'model',
        'revision',
        'variant',
        'subfolder',
        'dtype',
        'quantizer',
        'compile',
        'weight-name',
        'mmproj',
    ]
)

_prompt_upsampler_cache = _d_memoize.create_object_cache(
    'prompt_upsampler', cache_type=_memory.SizedConstrainedObjectCache
)


class LoadedPromptUpsampler:
    """A cached causal or image-to-text model and the processor it was loaded with."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor


class PromptUpsamplerUri:
    """
    Representation of a ``--prompt-upsampler`` URI.

    The argument grammar matches ``--transformer``. Loading does not. This URI
    builds a Transformers causal or image-to-text model and keeps it in its own
    cache. ``--transformer`` remains the diffusion denoiser.
    """

    NAMES = ['Prompt Upsampler']

    @staticmethod
    def help():
        import dgenerate.arguments as _a
        return _a.get_raw_help_text('--prompt-upsampler')

    OPTION_ARGS = {
        'dtype': ['float16', 'bfloat16', 'float32']
    }

    FILE_ARGS = {
        'model': {'mode': ['in', 'dir'], 'filetypes': [('Models', ['*.safetensors', '*.gguf', '*.pt', '*.pth', '*.bin'])]},
        'mmproj': {'mode': ['in'], 'filetypes': [('Models', ['*.gguf'])]},
    }

    @property
    def model(self) -> str:
        """Model path or Hugging Face slug."""
        return self._model

    @property
    def revision(self) -> _types.OptionalString:
        """Model repo revision."""
        return self._revision

    @property
    def variant(self) -> _types.OptionalString:
        """Model weight variant, for example ``fp16``."""
        return self._variant

    @property
    def subfolder(self) -> _types.OptionalPath:
        """Model repo subfolder."""
        return self._subfolder

    @property
    def dtype(self) -> _enums.DataType | None:
        """Model dtype (precision)."""
        return self._dtype

    @property
    def quantizer(self) -> _types.OptionalUri:
        """``--quantizer`` URI override."""
        return self._quantizer

    @property
    def compile(self) -> bool:
        """Compile this upsampler's decoder layers."""
        return self._compile

    @property
    def weight_name(self) -> _types.OptionalName:
        """One weight file in the repository or folder, such as a GGUF."""
        return self._weight_name

    @property
    def mmproj(self) -> _types.OptionalPath:
        """
        GGUF vision tower (llama.cpp mmproj) loaded beside a GGUF upsampler.

        A local path or a URL, including a Hugging Face blob or resolve link.
        A file name is not taken from the language file or its repository.
        The LTX-2.5 Gemma enhancer and the FLUX.3 Qwen3.5 upsampler use it
        when the quant is split into a language file and a tower.
        """
        return self._mmproj

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 variant: _types.OptionalString = None,
                 subfolder: _types.OptionalString = None,
                 dtype: _enums.DataType | str | None = None,
                 quantizer: _types.OptionalUri = None,
                 compile: bool = False,
                 weight_name: _types.OptionalName = None,
                 mmproj: _types.OptionalPath = None):
        gguf_file = _hfhub.is_gguf_model(model) or bool(weight_name and _hfhub.is_gguf_model(weight_name))
        safetensors_file = (
            str(model).lower().endswith('.safetensors')
            or bool(weight_name and str(weight_name).lower().endswith('.safetensors')))
        if _hfhub.is_single_file_model_load(model) and not gguf_file and not safetensors_file:
            raise _exceptions.InvalidPromptUpsamplerUriError(
                'A prompt upsampler file is a .gguf or a .safetensors quant. '
                'Other single-file checkpoints are denoiser weights and use --transformer.')
        if quantizer and gguf_file:
            raise _exceptions.InvalidPromptUpsamplerUriError(
                'A GGUF prompt upsampler is already quantized. Do not set the quantizer argument.')
        if mmproj and not gguf_file:
            raise _exceptions.InvalidPromptUpsamplerUriError(
                'mmproj is a GGUF vision tower loaded beside a GGUF prompt upsampler.')
        self._model = model
        self._revision = revision
        self._variant = variant
        self._subfolder = subfolder
        self._quantizer = quantizer
        self._compile = bool(compile)
        self._weight_name = weight_name
        self._mmproj = mmproj
        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise _exceptions.InvalidPromptUpsamplerUriError(
                f'invalid dtype string, must be one of: '
                f'{_textprocessing.oxford_comma(_enums.supported_data_type_strings(), "or")}')

    def load(self,
             dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
             auth_token: _types.OptionalString = None,
             local_files_only: bool = False,
             no_cache: bool = False,
             device_map: str | dict | None = None) -> LoadedPromptUpsampler:
        """
        Load the upsampler and its processor.

        The class comes from the checkpoint config. An image-to-text config,
        such as Gemma or Qwen3.5, uses ``AutoModelForImageTextToText``. A
        text-only causal config uses ``AutoModelForCausalLM``.

        :param dtype_fallback: Used when the URI does not set ``dtype``.
        :param auth_token: Hugging Face token.
        :param local_files_only: Do not download missing files.
        :param no_cache: Do not keep the returned model in the upsampler cache.
        :param device_map: Placement for a quantized load. ``None`` leaves it to the loader.
        """
        def cache_all(error):
            raise _exceptions.PromptUpsamplerUriLoadError(
                f'error loading prompt upsampler "{self.model}": {error}') from error

        with _hfhub.with_hf_errors_as_model_not_found(cache_all):
            args = locals()
            args.pop('self')
            args.pop('cache_all')
            model = self._load(**args)
            _util.mark_compile_requested(model, self.compile)
            return LoadedPromptUpsampler(model, model._dgenerate_prompt_processor)

    @staticmethod
    def _enforce_cache_size(new_size):
        _prompt_upsampler_cache.enforce_cpu_mem_constraints(
            _constants.PROMPT_UPSAMPLER_CACHE_MEMORY_CONSTRAINTS,
            size_var='prompt_upsampler_size',
            new_object_size=new_size)

    def _load_weight_file(self, torch_dtype, auth_token, local_files_only, no_cache, gguf_file):
        """A GGUF or safetensors quant, with ``mmproj`` when the tower is a separate file."""
        import os

        import accelerate

        from dgenerate.pipelinewrapper.uris.textencoderuri import (
            _assign_gguf_text_encoder,
            _fit_layer_count,
            _load_gguf_text_encoder,
            _materialize_mmproj,
            _shipped_encoder_for_shape,
            _tensor_signature,
        )

        try:
            model_path, _subfolder = _hfhub.materialize_named_weight(
                self.model,
                weight_name=self.weight_name,
                revision=self.revision,
                subfolder=self.subfolder,
                local_files_only=local_files_only,
                auth_token=auth_token)
        except Exception as error:
            raise _exceptions.PromptUpsamplerUriLoadError(
                f'Failed to find prompt upsampler weights "{self.weight_name or self.model}": {error}'
            ) from error

        mmproj_path = None
        if gguf_file and self.mmproj:
            try:
                mmproj_path = _materialize_mmproj(
                    self.mmproj, local_files_only, auth_token)
            except _exceptions.TextEncoderUriLoadError as error:
                raise _exceptions.PromptUpsamplerUriLoadError(str(error)) from error
            if not _hfhub.is_gguf_model(mmproj_path):
                raise _exceptions.PromptUpsamplerUriLoadError(
                    f'mmproj "{self.mmproj}" must be a .gguf file.')

        size = os.path.getsize(model_path) if os.path.isfile(model_path) else 0
        if mmproj_path and os.path.isfile(mmproj_path):
            size += os.path.getsize(mmproj_path)
        self._enforce_cache_size(size)

        if gguf_file:
            _messages.debug_log(
                f'Loading prompt upsampler "{model_path}" from GGUF.')
            try:
                model = _load_gguf_text_encoder(
                    None, model_path, torch_dtype, mmproj_path=mmproj_path)
            except _exceptions.TextEncoderUriLoadError as error:
                raise _exceptions.PromptUpsamplerUriLoadError(str(error)) from error
        else:
            from dgenerate.pipelinewrapper import safetensorscompat
            _messages.debug_log(
                f'Loading prompt upsampler "{model_path}" from safetensors.')
            try:
                checkpoint = safetensorscompat.load_checkpoint(model_path, torch_dtype, dequantize=True)
                signature = _tensor_signature(checkpoint.checkpoint)
                if signature is None:
                    raise _exceptions.PromptUpsamplerUriLoadError(
                        f'Prompt upsampler "{model_path}" has no token embedding and decoder layers.')
                _embed, shape, layers = signature
                encoder, config_path = _shipped_encoder_for_shape(shape, layers)
                if encoder is None:
                    vocab, hidden = shape
                    raise _exceptions.PromptUpsamplerUriLoadError(
                        f'Prompt upsampler "{model_path}" (vocabulary {vocab}, hidden size {hidden}, '
                        f'{layers} layers) does not match a configuration shipped with dgenerate.')
                config = encoder.config_class.from_json_file(config_path)
                trimmed = _fit_layer_count(config, layers)
                with accelerate.init_empty_weights(include_buffers=False):
                    model = encoder(config)
                model = _assign_gguf_text_encoder(
                    model, checkpoint.checkpoint, torch_dtype, rename=None, trimmed=trimmed)
            except _exceptions.PromptUpsamplerUriLoadError:
                raise
            except _exceptions.TextEncoderUriLoadError as error:
                raise _exceptions.PromptUpsamplerUriLoadError(str(error)) from error
            except Exception as error:
                raise _exceptions.PromptUpsamplerUriLoadError(
                    f'Failed to load prompt upsampler "{model_path}": {error}') from error

        model._dgenerate_prompt_processor = self._load_processor(
            model_path, auth_token, local_files_only)
        _util._patch_module_to_for_sized_cache(_prompt_upsampler_cache, model)
        return model, _d_memoize.CachedObjectMetadata(size=size, skip=bool(no_cache))

    def _load_processor(self, model_path, auth_token, local_files_only):
        """The chat processor from the repository, or from files beside the weights."""
        import os

        from transformers import AutoProcessor

        folder = os.path.dirname(model_path)
        slug = (
            self.weight_name and not os.path.isdir(self.model)
            and not _hfhub.is_single_file_model_load(self.model)
            and _hfhub.HFBlobLink.parse(self.model) is None)
        source = self.model if slug else folder
        kwargs = {
            'token': auth_token,
            'local_files_only': local_files_only,
        }
        if slug:
            kwargs['revision'] = self.revision
            if self.subfolder:
                kwargs['subfolder'] = self.subfolder
        try:
            return AutoProcessor.from_pretrained(source, **kwargs)
        except Exception as error:
            tokenizer = self._gguf_tokenizer(model_path, error)
            if tokenizer is None:
                raise _exceptions.PromptUpsamplerUriLoadError(
                    f'No processor for prompt upsampler "{self.model}". '
                    f'The repository, or the folder beside the weights, needs the tokenizer '
                    f'and image processor: {error}') from error
            return _GGUFTokenizerProcessor(tokenizer)

    def _gguf_tokenizer(self, model_path, error):
        """The vocabulary stored in a GGUF, when the repository has no processor files."""
        import os

        from transformers import AutoTokenizer

        if not model_path or not str(model_path).lower().endswith('.gguf'):
            return None
        folder = os.path.dirname(model_path)
        gguf_file = os.path.basename(model_path)
        # AutoTokenizer builds a model config from the GGUF first. Gemma 4
        # stores intermediate_size as one integer per layer, and that config
        # class rejects the list. The vocabulary itself is still in the file.
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                folder, gguf_file=gguf_file, local_files_only=True)
        except Exception as tokenizer_error:
            try:
                from transformers import TokenizersBackend
                tokenizer = TokenizersBackend.from_pretrained(
                    folder, gguf_file=gguf_file, local_files_only=True)
            except Exception:
                _messages.debug_log(
                    f'No tokenizer in GGUF "{model_path}": {tokenizer_error}; '
                    f'processor: {error}')
                return None
        _mark_gguf_control_tokens(tokenizer, model_path)
        return tokenizer

    @_memoize(_prompt_upsampler_cache,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {
                  'self': _d_memoize.property_hasher,
              }),
              on_hit=lambda key, hit: _d_memoize.simple_cache_hit_debug("Prompt Upsampler", key, hit),
              on_create=lambda key, new: _d_memoize.simple_cache_miss_debug("Prompt Upsampler", key, new))
    def _load(self,
              dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
              auth_token: _types.OptionalString = None,
              local_files_only: bool = False,
              no_cache: bool = False,
              device_map: str | dict | None = None):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
        )

        import dgenerate.pipelinewrapper.videopipelines.quantization as _quantization

        torch_dtype = _enums.get_torch_dtype(self.dtype or dtype_fallback)
        if self.dtype is None and torch_dtype is None:
            dtype_fallback = _enums.DataType.BFLOAT16
            torch_dtype = _enums.get_torch_dtype(dtype_fallback)
        else:
            dtype_fallback = self.dtype or dtype_fallback

        if self.variant == 'null':
            variant = None
        else:
            variant = self.variant

        gguf_file = _hfhub.is_gguf_model(self.model) or bool(
            self.weight_name and _hfhub.is_gguf_model(self.weight_name))
        safetensors_file = (
            str(self.model).lower().endswith('.safetensors')
            or bool(self.weight_name and str(self.weight_name).lower().endswith('.safetensors')))
        if gguf_file or safetensors_file:
            return self._load_weight_file(
                torch_dtype, auth_token, local_files_only, no_cache, gguf_file)

        model_path = _hfhub.download_non_hf_slug_model(
            self.model, local_files_only=local_files_only, auth_token=auth_token)
        config_kwargs = {
            'revision': self.revision,
            'token': auth_token,
            'local_files_only': local_files_only,
        }
        if self.subfolder:
            config_kwargs['subfolder'] = self.subfolder
        config = AutoConfig.from_pretrained(model_path, **config_kwargs)
        model_type = getattr(config, 'model_type', None)
        if model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES:
            model_class = AutoModelForImageTextToText
        elif model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            model_class = AutoModelForCausalLM
        else:
            raise _exceptions.PromptUpsamplerUriLoadError(
                f'Prompt upsampler "{self.model}" has model type {model_type!r}. '
                'It needs to be a Transformers causal or image-to-text model.')

        prequantized_sdnq = _sdnqload.sdnq_config_for_component(
            model_path,
            subfolder=self.subfolder,
            revision=self.revision,
            token=auth_token,
            local_files_only=local_files_only)
        requantize_error = _sdnqload.sdnq_requantize_error(self.quantizer, prequantized_sdnq)
        if requantize_error:
            raise _exceptions.PromptUpsamplerUriLoadError(requantize_error)

        estimated_memory_use = _pipelinewrapper_util.estimate_model_memory_use(
            repo_id=model_path,
            revision=self.revision,
            variant=variant,
            subfolder=self.subfolder,
            local_files_only=local_files_only,
            auth_token=auth_token)
        self._enforce_cache_size(estimated_memory_use)

        _messages.debug_log(f'Loading prompt upsampler "{self.model}" as {model_class.__name__}.')
        if self.quantizer and prequantized_sdnq is None:
            model = _quantization._load_quantized_module(
                model_class, model_path, self.subfolder, self.revision,
                variant, dtype_fallback, self.quantizer, auth_token,
                local_files_only, device_map)
        else:
            load_kwargs = {
                'revision': self.revision,
                'token': auth_token,
                'local_files_only': local_files_only,
                'dtype': torch_dtype,
            }
            if self.subfolder:
                load_kwargs['subfolder'] = self.subfolder
            if variant is not None:
                load_kwargs['variant'] = variant
            if device_map is not None:
                load_kwargs['device_map'] = device_map
            model = model_class.from_pretrained(model_path, **load_kwargs)

        processor_kwargs = {
            'revision': self.revision,
            'token': auth_token,
            'local_files_only': local_files_only,
        }
        if self.subfolder:
            processor_kwargs['subfolder'] = self.subfolder
        model._dgenerate_prompt_processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)

        _messages.debug_log('Estimated Prompt Upsampler Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_use))
        _util._patch_module_to_for_sized_cache(_prompt_upsampler_cache, model)
        return model, _d_memoize.CachedObjectMetadata(
            size=estimated_memory_use,
            skip=bool(self.quantizer) or no_cache)

    @staticmethod
    def parse(uri: _types.Uri) -> 'PromptUpsamplerUri':
        """
        Parse a ``--prompt-upsampler`` URI.

        :raise InvalidPromptUpsamplerUriError:
        """
        try:
            parsed = _prompt_upsampler_uri_parser.parse(uri)
            dtype = parsed.args.get('dtype')
            supported_dtypes = _enums.supported_data_type_strings()
            if dtype is not None and dtype not in supported_dtypes:
                raise _exceptions.InvalidPromptUpsamplerUriError(
                    f'Prompt upsampler "dtype" must be {", ".join(supported_dtypes)}, '
                    f'or left undefined, received: {dtype}')
            quantizer = parsed.args.get('quantizer', None)
            return PromptUpsamplerUri(
                model=parsed.concept,
                revision=parsed.args.get('revision', None),
                variant=parsed.args.get('variant', None),
                dtype=dtype,
                subfolder=parsed.args.get('subfolder', None),
                quantizer=quantizer or None,
                compile=_util.parse_compile_argument(
                    parsed.args.get('compile'),
                    _exceptions.InvalidPromptUpsamplerUriError),
                weight_name=parsed.args.get('weight-name', None),
                mmproj=parsed.args.get('mmproj', None))
        except _textprocessing.ConceptUriParseError as error:
            raise _exceptions.InvalidPromptUpsamplerUriError(error) from error
