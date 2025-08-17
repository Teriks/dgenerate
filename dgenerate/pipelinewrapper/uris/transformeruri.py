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

import diffusers

import dgenerate.hfhub as _hfhub
import dgenerate.memoize as _d_memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.util as _pipelinewrapper_util
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.memoize import memoize as _memoize
from dgenerate.pipelinewrapper import constants as _constants
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions
from dgenerate.pipelinewrapper.uris import util as _util

_transformer_uri_parser = _textprocessing.ConceptUriParser(
    'Transformer', [
        'model',
        'revision',
        'variant',
        'subfolder',
        'dtype',
        'quantizer'
    ]
)

_transformer_cache = _d_memoize.create_object_cache(
    'transformer', cache_type=_memory.SizedConstrainedObjectCache
)


class TransformerUri:
    """
    Representation of ``--transformer`` URI.
    """

    # pipelinewrapper.uris.util.get_uri_accepted_args_schema metadata

    NAMES = ['Transformer']

    @staticmethod
    def help():
        import dgenerate.arguments as _a
        return _a.get_raw_help_text('--transformer')

    OPTION_ARGS = {
        'dtype': ['float16', 'bfloat16', 'float32']
    }

    FILE_ARGS = {
        'model': {'mode': ['in', 'dir'], 'filetypes': [('Models', ['*.safetensors', '*.pt', '*.pth', '*.cpkt', '*.bin'])]}
    }

    # ===

    @property
    def model(self) -> str:
        """
        Model path, huggingface slug
        """
        return self._model

    @property
    def revision(self) -> _types.OptionalString:
        """
        Model repo revision
        """
        return self._revision

    @property
    def variant(self) -> _types.OptionalString:
        """
        Model repo revision
        """
        return self._variant

    @property
    def subfolder(self) -> _types.OptionalPath:
        """
        Model repo subfolder
        """
        return self._subfolder

    @property
    def dtype(self) -> _enums.DataType | None:
        """
        Model dtype (precision)
        """
        return self._dtype

    @property
    def quantizer(self) -> _types.OptionalUri:
        """
        --quantizer URI override
        """
        return self._quantizer

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 variant: _types.OptionalString = None,
                 subfolder: _types.OptionalString = None,
                 dtype: _enums.DataType | str | None = None,
                 quantizer: _types.OptionalUri = None):
        """
        :param model: model path
        :param revision: model revision (branch name)
        :param variant: model variant, for example ``fp16``
        :param subfolder: model subfolder
        :param dtype: model data type (precision)

        :raises InvalidTransformerUriError: If ``dtype`` is passed an invalid data type string.
        """

        if _hfhub.is_single_file_model_load(model):
            if quantizer:
                raise _exceptions.InvalidTextEncoderUriError(
                    'specifying a Transformer quantizer URI is only supported for Hugging Face '
                    'repository loads from a repo slug or disk path, single file loads are not supported.')

        self._model = model
        self._revision = revision
        self._variant = variant
        self._subfolder = subfolder
        self._quantizer = quantizer

        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise _exceptions.InvalidTransformerUriError(
                f'invalid dtype string, must be one of: '
                f'{_textprocessing.oxford_comma(_enums.supported_data_type_strings(), "or")}')

    def load(self,
             variant_fallback: _types.OptionalString = None,
             dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
             original_config: _types.OptionalPath = None,
             use_auth_token: _types.OptionalString = None,
             local_files_only: bool = False,
             no_cache: bool = False,
             device_map: str | None = None,
             transformer_class:
             type[diffusers.SD3Transformer2DModel] |
             type[
                 diffusers.FluxTransformer2DModel] = diffusers.SD3Transformer2DModel) \
            -> diffusers.SD3Transformer2DModel | diffusers.FluxTransformer2DModel:
        """
        Load a torch :py:class:`diffusers.SD3Transformer2DModel` or
        :py:class:`diffusers.FluxTransformer2DModel` from a URI.

        :param variant_fallback: If the URI does not specify a variant, use this variant.
        :param dtype_fallback: If the URI does not specify a dtype, use this dtype.
        :param original_config: Path to original model configuration for single file checkpoints, URL or `.yaml` file on disk.
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug or blob link
        :param no_cache: If True, force the returned object not to be cached by the memoize decorator.
        :param device_map: device placement strategy for quantized models, defaults to ``None``

        :param transformer_class: Transformer class type.

        :raises ModelNotFoundError: If the model could not be found.

        :return: :py:class:`diffusers.SD3Transformer2DModel` or :py:class:`diffusers.FluxTransformer2DModel`
        """
        def cache_all(e):
            raise _exceptions.TransformerUriLoadError(
                f'error loading transformer "{self.model}": {e}') from e

        with _hfhub.with_hf_errors_as_model_not_found(cache_all):
            args = locals()
            args.pop('self')
            args.pop('cache_all')
            return self._load(**args)

    @staticmethod
    def _enforce_cache_size(new_transformer_size):
        _transformer_cache.enforce_cpu_mem_constraints(
            _constants.TRANSFORMER_CACHE_MEMORY_CONSTRAINTS,
            size_var='transformer_size',
            new_object_size=new_transformer_size)

    @_memoize(_transformer_cache,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _d_memoize.property_hasher}),
              on_hit=lambda key, hit: _d_memoize.simple_cache_hit_debug("Torch Transformer", key, hit),
              on_create=lambda key, new: _d_memoize.simple_cache_miss_debug("Torch Transformer", key, new))
    def _load(self,
              variant_fallback: _types.OptionalString = None,
              dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
              original_config: _types.OptionalPath = None,
              use_auth_token: _types.OptionalString = None,
              local_files_only: bool = False,
              no_cache: bool = False,
              device_map: str | None = None,
              transformer_class:
              type[diffusers.SD3Transformer2DModel] |
              type[
                  diffusers.FluxTransformer2DModel] = diffusers.SD3Transformer2DModel) \
            -> diffusers.SD3Transformer2DModel | diffusers.FluxTransformer2DModel:

        if self.dtype is None:
            torch_dtype = _enums.get_torch_dtype(dtype_fallback)
        else:
            torch_dtype = _enums.get_torch_dtype(self.dtype)

        if self.variant is None:
            variant = variant_fallback
        elif self.variant == 'null':
            variant = None
        else:
            variant = self.variant

        model_path = _hfhub.download_non_hf_slug_model(self.model)

        if self.quantizer:
            quant_config = _util.get_quantizer_uri_class(
                self.quantizer,
                _exceptions.InvalidTransformerUriError
            ).parse(self.quantizer).to_config(torch_dtype)
        else:
            quant_config = None

        if _hfhub.is_single_file_model_load(model_path):
            try:
                original_config = _hfhub.download_non_hf_slug_config(
                    original_config) if original_config else None
            except _hfhub.NonHFConfigDownloadError as e:
                raise _exceptions.TransformerUriLoadError(
                    f'original config file "{original_config}" for Transformer could not be downloaded: {e}'
                ) from e

            estimated_memory_use = _pipelinewrapper_util.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token
            )

            self._enforce_cache_size(estimated_memory_use)

            transformer = transformer_class.from_single_file(
                model_path,
                token=use_auth_token,
                revision=self.revision,
                torch_dtype=torch_dtype,
                original_config=original_config,
                local_files_only=local_files_only,
                quantization_config=quant_config,
                device_map=device_map
            )

        else:
            if original_config:
                raise _exceptions.TransformerUriLoadError(
                    'specifying original_config file for Transformer '
                    'is only supported for single file loads.'
                )

            estimated_memory_use = _pipelinewrapper_util.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                variant=variant,
                subfolder=self.subfolder,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token
            )

            self._enforce_cache_size(estimated_memory_use)

            transformer = transformer_class.from_pretrained(
                model_path,
                revision=self.revision,
                variant=variant,
                torch_dtype=torch_dtype,
                subfolder=self.subfolder if self.subfolder else "",
                token=use_auth_token,
                local_files_only=local_files_only,
                quantization_config=quant_config,
                device_map=device_map
            )

        _messages.debug_log('Estimated Torch Transformer Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_use))

        _util._patch_module_to_for_sized_cache(_transformer_cache, transformer)

        # noinspection PyTypeChecker
        return transformer, _d_memoize.CachedObjectMetadata(
            size=estimated_memory_use,
            skip=self.quantizer or no_cache
        )

    @staticmethod
    def parse(uri: _types.Uri) -> 'TransformerUri':
        """
        Parse a ``--transformer`` uri and return an object representing its constituents

        :param uri: string with ``--transformer`` uri syntax

        :raise InvalidTransformerUriError:

        :return: :py:class:`.TransformerUri`
        """
        try:
            r = _transformer_uri_parser.parse(uri)

            dtype = r.args.get('dtype')

            supported_dtypes = _enums.supported_data_type_strings()
            if dtype is not None and dtype not in supported_dtypes:
                raise _exceptions.InvalidTransformerUriError(
                    f'Torch Transformer "dtype" must be {", ".join(supported_dtypes)}, '
                    f'or left undefined, received: {dtype}')

            return TransformerUri(model=r.concept,
                                  revision=r.args.get('revision', None),
                                  variant=r.args.get('variant', None),
                                  dtype=dtype,
                                  subfolder=r.args.get('subfolder', None),
                                  quantizer=r.args.get('quantizer', False))
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidTransformerUriError(e) from e
