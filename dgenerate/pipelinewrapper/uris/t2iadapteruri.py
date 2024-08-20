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
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.memoize import memoize as _memoize
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions

_t2i_adapter_uri_parser = _textprocessing.ConceptUriParser(
    'T2IAdapter', ['scale', 'revision', 'variant', 'subfolder', 'dtype']
)


class T2IAdapterUri:
    """
    Representation of ``--t2i-adapters`` uri when ``--model-type`` torch*
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
    def scale(self) -> float:
        """
        T2IAdapter scale
        """
        return self._scale

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString,
                 variant: _types.OptionalString,
                 subfolder: _types.OptionalPath,
                 dtype: _enums.DataType | str | None = None,
                 scale: float = 1.0):
        """
        :param model: model path
        :param revision: model revision (branch name)
        :param variant: model variant, for example ``fp16``
        :param subfolder: model subfolder
        :param dtype: model data type (precision)
        :param scale: t2i adapter scale

        :raises InvalidT2IAdapterUriError: If ``dtype`` is passed an invalid data type string.
        """

        self._model = model
        self._revision = revision
        self._variant = variant
        self._subfolder = subfolder

        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise _exceptions.InvalidT2IAdapterUriError(
                f'invalid dtype string, must be one of: {_textprocessing.oxford_comma(_enums.supported_data_type_strings(), "or")}')

        self._scale = scale

    def load(self,
             dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
             use_auth_token: _types.OptionalString = None,
             local_files_only: bool = False,
             sequential_cpu_offload_member: bool = False,
             model_cpu_offload_member: bool = False) -> diffusers.T2IAdapter:
        """
        Load a :py:class:`diffusers.T2IAdapter` from this URI.


        :param dtype_fallback: Fallback datatype if ``dtype`` was not specified in the URI.

        :param use_auth_token: Optional huggingface API auth token, used for downloading
            restricted repos that your account has access to.

        :param local_files_only: Avoid connecting to huggingface to download models and
            only use cached models?

        :param sequential_cpu_offload_member: This model will be attached to
            a pipeline which will have sequential cpu offload enabled?

        :param model_cpu_offload_member: This model will be attached to a pipeline
            which will have model cpu offload enabled?

        :raises ModelNotFoundError: If the model could not be found.

        :return: :py:class:`diffusers.T2IAdapter`
        """
        try:
            return self._load(dtype_fallback,
                              use_auth_token,
                              local_files_only,
                              sequential_cpu_offload_member,
                              model_cpu_offload_member)
        except (huggingface_hub.utils.HFValidationError,
                huggingface_hub.utils.HfHubHTTPError) as e:
            raise _hfutil.ModelNotFoundError(e)
        except Exception as e:
            raise _exceptions.T2IAdapterUriLoadError(
                f'error loading t2i adapter "{self.model}": {e}')

    @_memoize(_cache._ADAPTER_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(
                  args, {'self': lambda o: _d_memoize.struct_hasher(
                      o, exclude={'scale'})}),
              on_hit=lambda key, hit: _d_memoize.simple_cache_hit_debug("Torch T2IAdapter", key, hit),
              on_create=lambda key, new: _d_memoize.simple_cache_miss_debug("Torch T2IAdapter", key, new))
    def _load(self,
              dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
              use_auth_token: _types.OptionalString = None,
              local_files_only: bool = False,
              sequential_cpu_offload_member: bool = False,
              model_cpu_offload_member: bool = False) -> diffusers.T2IAdapter:

        if sequential_cpu_offload_member and model_cpu_offload_member:
            # these are used for cache differentiation only
            raise ValueError('sequential_cpu_offload_member and model_cpu_offload_member cannot both be True.')

        model_path = _hfutil.download_non_hf_model(self.model)

        single_file_load_path = _hfutil.is_single_file_model_load(model_path)

        torch_dtype = _enums.get_torch_dtype(
            dtype_fallback if self.dtype is None else self.dtype)

        if single_file_load_path:

            estimated_memory_usage = _hfutil.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only
            )

            _cache.enforce_adapter_cache_constraints(
                new_adapter_size=estimated_memory_usage)

            new_adapter = diffusers.T2IAdapter.from_single_file(
                model_path,
                revision=self.revision,
                torch_dtype=torch_dtype,
                token=use_auth_token,
                local_files_only=local_files_only)
        else:

            estimated_memory_usage = _hfutil.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                variant=self.variant,
                subfolder=self.subfolder,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only
            )

            _cache.enforce_adapter_cache_constraints(
                new_adapter_size=estimated_memory_usage)

            new_adapter = diffusers.T2IAdapter.from_pretrained(
                model_path,
                revision=self.revision,
                variant=self.variant,
                subfolder=self.subfolder,
                torch_dtype=torch_dtype,
                token=use_auth_token,
                local_files_only=local_files_only)

        _messages.debug_log('Estimated Torch T2IAdapter Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_usage))

        _cache.adapter_create_update_cache_info(
            adapter=new_adapter,
            estimated_size=estimated_memory_usage)

        return new_adapter

    @staticmethod
    def parse(uri: _types.Uri) -> 'T2IAdapterUri':
        """
        Parse a ``--model-type`` torch* ``--t2i-adapters`` uri specification and
        return an object representing its constituents

        :param uri: string with ``--t2i-adapters`` uri syntax

        :raise InvalidT2IAdapterUriError:

        :return: :py:class:`.T2IAdapterUri`
        """
        try:
            r = _t2i_adapter_uri_parser.parse(uri)

            dtype = r.args.get('dtype')
            scale = r.args.get('scale', 1.0)

            supported_dtypes = _enums.supported_data_type_strings()
            if dtype is not None and dtype not in supported_dtypes:
                raise _exceptions.InvalidT2IAdapterUriError(
                    f'Torch T2IAdapter "dtype" must be {", ".join(supported_dtypes)}, '
                    f'or left undefined, received: {dtype}')

            try:
                scale = float(scale)
            except ValueError:
                raise _exceptions.InvalidT2IAdapterUriError(
                    f'Torch T2IAdapter "scale" must be a floating point number, received: {scale}')

            return T2IAdapterUri(
                model=r.concept,
                revision=r.args.get('revision', None),
                variant=r.args.get('variant', None),
                subfolder=r.args.get('subfolder', None),
                dtype=dtype,
                scale=scale)

        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidT2IAdapterUriError(e)
