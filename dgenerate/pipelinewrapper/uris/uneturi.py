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
import dgenerate.pipelinewrapper.util as _util
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.memoize import memoize as _memoize
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions

_unet_uri_parser = _textprocessing.ConceptUriParser(
    'UNet', [
        'revision',
        'variant',
        'subfolder',
        'dtype',
        'quantizer'
    ]
)


class UNetUri:
    """
    Representation of ``--unet`` uri when ``--model-type`` torch*
    """

    @property
    def model(self) -> str:
        """
        Model path, huggingface slug, file path, or blob link
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

        :raises InvalidUNetUriError: If ``model`` points to a single file,
            single file loads are not supported. Or if ``dtype`` is passed an
            invalid data type string.
        """

        if _util.is_single_file_model_load(model):
            if quantizer:
                raise _exceptions.InvalidTextEncoderUriError(
                    'specifying a UNet quantizer URI is only supported for Hugging Face '
                    'repository loads from a repo slug or disk path, single file loads are not supported.')

        self._model = model
        self._revision = revision
        self._variant = variant
        self._quantizer = quantizer

        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise _exceptions.InvalidUNetUriError(
                f'invalid dtype string, must be one of: {_textprocessing.oxford_comma(_enums.supported_data_type_strings(), "or")}')

        self._subfolder = subfolder

    def load(self,
             variant_fallback: _types.OptionalString = None,
             dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
             original_config: _types.OptionalPath = None,
             use_auth_token: _types.OptionalString = None,
             local_files_only: bool = False,
             sequential_cpu_offload_member: bool = False,
             model_cpu_offload_member: bool = False,
             unet_class=diffusers.UNet2DConditionModel):
        """
        Load a UNet of type :py:class:`diffusers.UNet2DConditionModel`

        :param variant_fallback: If the URI does not specify a variant, use this variant.
        :param dtype_fallback: If the URI does not specify a dtype, use this dtype.
        :param original_config: Path to original model configuration for single file checkpoints, URL or `.yaml` file on disk.
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug or blob link

        :param sequential_cpu_offload_member: This model will be attached to
            a pipeline which will have sequential cpu offload enabled?

        :param model_cpu_offload_member: This model will be attached to a pipeline
            which will have model cpu offload enabled?

        :param unet_class: UNet class

        :raises ModelNotFoundError: If the model could not be found.

        :return: :py:class:`diffusers.UNet2DConditionModel`
        """
        try:
            args = locals()
            args.pop('self')
            return self._load(**args)
        except (huggingface_hub.utils.HFValidationError,
                huggingface_hub.utils.HfHubHTTPError) as e:
            raise _util.ModelNotFoundError(e)
        except Exception as e:
            raise _exceptions.UNetUriLoadError(
                f'error loading unet "{self.model}": {e}')

    @_memoize(_cache._UNET_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _d_memoize.struct_hasher}),
              on_hit=lambda key, hit: _d_memoize.simple_cache_hit_debug("Torch UNet", key, hit),
              on_create=lambda key, new: _d_memoize.simple_cache_miss_debug("Torch UNet", key, new))
    def _load(self,
              variant_fallback: _types.OptionalString = None,
              dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
              original_config: _types.OptionalPath = None,
              use_auth_token: _types.OptionalString = None,
              local_files_only: bool = False,
              sequential_cpu_offload_member: bool = False,
              model_cpu_offload_member: bool = False,
              unet_class=diffusers.UNet2DConditionModel):

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

        model_path = _util.download_non_hf_model(self.model)

        if self.quantizer:
            quant_config = _util.get_quantizer_uri_class(
                self.quantizer,
                _exceptions.InvalidUNetUriError
            ).parse(self.quantizer).to_config()
        else:
            quant_config = None

        if _util.is_single_file_model_load(model_path):
            try:
                original_config = _util.download_non_hf_config(original_config) if original_config else None
            except _util.NonHFConfigDownloadError as e:
                raise _exceptions.UNetUriLoadError(
                    f'original config file "{original_config}" for UNet could not be downloaded: {e}'
                )

            estimated_memory_use = _util.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token
            )

            _cache.enforce_text_encoder_cache_constraints(
                new_text_encoder_size=estimated_memory_use)

            estimated_memory_use = _util.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token
            )

            _cache.enforce_text_encoder_cache_constraints(
                new_text_encoder_size=estimated_memory_use
            )

            try:
                unet = _util.single_file_load_sub_module(
                    path=model_path,
                    class_name=unet_class.__name__,
                    library_name='diffusers',
                    name=self.subfolder if self.subfolder else 'unet',
                    use_auth_token=use_auth_token,
                    original_config=original_config,
                    local_files_only=local_files_only,
                    revision=self.revision,
                    dtype=torch_dtype
                )
            except FileNotFoundError as e:
                # cannot find configs
                raise _exceptions.TextEncoderUriLoadError(e)

            estimated_memory_use = _util.estimate_memory_usage(unet)
        else:
            if original_config:
                raise _exceptions.UNetUriLoadError(
                    'specifying original_config file for UNet '
                    'is only supported for single file loads.')

            estimated_memory_use = _util.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                variant=variant,
                subfolder=self.subfolder,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token
            )

            _cache.enforce_unet_cache_constraints(new_unet_size=estimated_memory_use)

            unet = unet_class.from_pretrained(
                model_path,
                revision=self.revision,
                variant=variant,
                torch_dtype=torch_dtype,
                subfolder=self.subfolder,
                token=use_auth_token,
                original_config=self.original_config,
                local_files_only=local_files_only,
                quantization_config=quant_config
            )

        _messages.debug_log('Estimated Torch UNet Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_use))

        _cache.unet_create_update_cache_info(unet=unet,
                                             estimated_size=estimated_memory_use)

        return unet

    @staticmethod
    def parse(uri: _types.Uri) -> 'UNetUri':
        """
        Parse a ``--model-type`` torch* ``--unet`` uri and return an object representing its constituents

        :param uri: string with ``--unet`` uri syntax

        :raise InvalidUNetUriError:

        :return: :py:class:`.TorchUNetPath`
        """
        try:
            r = _unet_uri_parser.parse(uri)

            dtype = r.args.get('dtype')

            supported_dtypes = _enums.supported_data_type_strings()
            if dtype is not None and dtype not in supported_dtypes:
                raise _exceptions.InvalidUNetUriError(
                    f'Torch UNet "dtype" must be {", ".join(supported_dtypes)}, '
                    f'or left undefined, received: {dtype}')

            return UNetUri(
                model=r.concept,
                revision=r.args.get('revision', None),
                variant=r.args.get('variant', None),
                dtype=dtype,
                subfolder=r.args.get('subfolder', None))
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidUNetUriError(e)
