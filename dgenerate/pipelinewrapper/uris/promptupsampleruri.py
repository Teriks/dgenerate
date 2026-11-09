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

_prompt_upsampler_uri_parser = _textprocessing.ConceptUriParser(
    'Prompt Upsampler', [
        'model',
        'revision',
        'variant',
        'subfolder',
        'dtype',
        'quantizer',
        'compile'
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
        'model': {'mode': ['in', 'dir'], 'filetypes': [('Models', ['*.safetensors', '*.pt', '*.pth', '*.bin'])]}
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

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 variant: _types.OptionalString = None,
                 subfolder: _types.OptionalString = None,
                 dtype: _enums.DataType | str | None = None,
                 quantizer: _types.OptionalUri = None,
                 compile: bool = False):
        if _hfhub.is_single_file_model_load(model):
            raise _exceptions.InvalidPromptUpsamplerUriError(
                'A prompt upsampler URI loads a Transformers model repository or folder. '
                'Single-file checkpoints, including GGUF, are denoiser weights and use --transformer.')
        self._model = model
        self._revision = revision
        self._variant = variant
        self._subfolder = subfolder
        self._quantizer = quantizer
        self._compile = bool(compile)
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
                    _exceptions.InvalidPromptUpsamplerUriError))
        except _textprocessing.ConceptUriParseError as error:
            raise _exceptions.InvalidPromptUpsamplerUriError(error) from error
