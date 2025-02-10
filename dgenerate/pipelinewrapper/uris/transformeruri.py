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
import huggingface_hub
import dgenerate.memoize as _d_memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.cache as _cache
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.hfutil as _hfutil
import dgenerate.pipelinewrapper.util as _util
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.memoize import memoize as _memoize
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions

_transformer_uri_parser = _textprocessing.ConceptUriParser(
    'Transformer', ['model', 'revision', 'variant', 'subfolder', 'dtype', 'quantizer'])


class TransformerUri:
    """
    Representation of ``--transformer`` uri when ``--model-type`` torch*
    """

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
             use_auth_token: _types.OptionalString = None,
             local_files_only: bool = False,
             sequential_cpu_offload_member: bool = False,
             model_cpu_offload_member: bool = False,
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
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug or blob link

        :param sequential_cpu_offload_member: This model will be attached to
            a pipeline which will have sequential cpu offload enabled?

        :param model_cpu_offload_member: This model will be attached to a pipeline
            which will have model cpu offload enabled?
            
        :param transformer_class: Transformer class type.

        :raises ModelNotFoundError: If the model could not be found.

        :return: :py:class:`diffusers.SD3Transformer2DModel` or :py:class:`diffusers.FluxTransformer2DModel`
        """

        try:
            return self._load(variant_fallback,
                              dtype_fallback,
                              use_auth_token,
                              local_files_only,
                              sequential_cpu_offload_member,
                              model_cpu_offload_member,
                              transformer_class)

        except (huggingface_hub.utils.HFValidationError,
                huggingface_hub.utils.HfHubHTTPError) as e:
            raise _hfutil.ModelNotFoundError(e)
        except Exception as e:
            raise _exceptions.TransformerUriLoadError(
                f'error loading transformer "{self.model}": {e}')

    @_memoize(_cache._TRANSFORMER_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _d_memoize.struct_hasher}),
              on_hit=lambda key, hit: _d_memoize.simple_cache_hit_debug("Torch Transformer", key, hit),
              on_create=lambda key, new: _d_memoize.simple_cache_miss_debug("Torch Transformer", key, new))
    def _load(self,
              variant_fallback: _types.OptionalString = None,
              dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
              use_auth_token: _types.OptionalString = None,
              local_files_only: bool = False,
              sequential_cpu_offload_member: bool = False,
              model_cpu_offload_member: bool = False,
              transformer_class:
              type[diffusers.SD3Transformer2DModel] |
              type[
                  diffusers.FluxTransformer2DModel] = diffusers.SD3Transformer2DModel) \
            -> diffusers.SD3Transformer2DModel | diffusers.FluxTransformer2DModel:

        if sequential_cpu_offload_member and model_cpu_offload_member:
            # these are used for cache differentiation only
            raise ValueError('sequential_cpu_offload_member and model_cpu_offload_member cannot both be True.')

        if self.dtype is None:
            torch_dtype = _enums.get_torch_dtype(dtype_fallback)
        else:
            torch_dtype = _enums.get_torch_dtype(self.dtype)

        if self.variant is None:
            variant = variant_fallback
        else:
            variant = self.variant

        model_path = _hfutil.download_non_hf_model(self.model)

        if self.quantizer:
            quant_config = _util.get_quantizer_uri_class(
                self.quantizer,
                _exceptions.InvalidTransformerUriError
            ).parse(self.quantizer).to_config()
        else:
            quant_config = None

        if _hfutil.is_single_file_model_load(model_path):
            estimated_memory_use = _hfutil.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token
            )

            _cache.enforce_transformer_cache_constraints(
                new_transformer_size=estimated_memory_use)

            transformer = transformer_class.from_single_file(
                model_path,
                token=use_auth_token,
                revision=self.revision,
                torch_dtype=torch_dtype,
                local_files_only=local_files_only,
                quantization_config=quant_config
            )

        else:
            estimated_memory_use = _hfutil.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                variant=variant,
                subfolder=self.subfolder,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token
            )

            _cache.enforce_transformer_cache_constraints(
                new_transformer_size=estimated_memory_use)

            transformer = transformer_class.from_pretrained(
                model_path,
                revision=self.revision,
                variant=variant,
                torch_dtype=torch_dtype,
                subfolder=self.subfolder if self.subfolder else "",
                token=use_auth_token,
                local_files_only=local_files_only,
                quantization_config=quant_config
            )

        _messages.debug_log('Estimated Torch Transformer Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_use))

        _cache.transformer_create_update_cache_info(
            transformer=transformer,
            estimated_size=estimated_memory_use)

        return transformer

    @staticmethod
    def parse(uri: _types.Uri) -> 'TransformerUri':
        """
        Parse a ``--model-type`` torch* ``--transformer`` uri and return an object representing its constituents

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
            raise _exceptions.InvalidTransformerUriError(e)
