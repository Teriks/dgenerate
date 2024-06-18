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

import typing

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

_torch_vae_uri_parser = _textprocessing.ConceptUriParser('VAE',
                                                         ['model', 'revision', 'variant', 'subfolder', 'dtype'])


class TorchVAEUri:
    """
    Representation of ``--vae`` uri when ``--model-type`` torch*
    """

    @property
    def encoder(self) -> str:
        """
        Encoder class name such as "AutoencoderKL"
        """
        return self._encoder

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

    _encoders = {
        'AutoencoderKL': diffusers.AutoencoderKL,
        'AsymmetricAutoencoderKL': diffusers.AsymmetricAutoencoderKL,
        'AutoencoderTiny': diffusers.AutoencoderTiny,
        'ConsistencyDecoderVAE': diffusers.ConsistencyDecoderVAE
    }

    @staticmethod
    def supported_encoder_names() -> list[str]:
        return list(TorchVAEUri._encoders.keys())

    def __init__(self,
                 encoder: str,
                 model: str,
                 revision: _types.OptionalString = None,
                 variant: _types.OptionalString = None,
                 subfolder: _types.OptionalString = None,
                 dtype: _enums.DataType | str | None = None):
        """
        :param encoder: encoder class name, for example ``AutoencoderKL``
        :param model: model path
        :param revision: model revision (branch name)
        :param variant: model variant, for example ``fp16``
        :param subfolder: model subfolder
        :param dtype: model data type (precision)

        :raises InvalidVaeUriError: If ``dtype`` is passed an invalid data type string, or if
            ``model`` points to a single file and the specified ``encoder`` class name does not
            support loading from a single file.
        """

        if encoder not in self._encoders:
            raise _exceptions.InvalidVaeUriError(
                f'Unknown VAE encoder class {encoder}, must be one of: {_textprocessing.oxford_comma(self._encoders.keys(), "or")}')

        can_single_file_load = hasattr(self._encoders[encoder], 'from_single_file')
        single_file_load_path = _hfutil.is_single_file_model_load(model)

        if single_file_load_path and not can_single_file_load:
            raise _exceptions.InvalidVaeUriError(f'{encoder} is not capable of loading from a single file, '
                                                 f'must be loaded from a huggingface repository slug or folder on disk.')

        if single_file_load_path:
            if subfolder is not None:
                raise _exceptions.InvalidVaeUriError('Single file VAE loads do not support the subfolder option.')

        self._encoder = encoder
        self._model = model
        self._revision = revision
        self._variant = variant
        self._subfolder = subfolder

        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise _exceptions.InvalidVaeUriError(
                f'invalid dtype string, must be one of: {_textprocessing.oxford_comma(_enums.supported_data_type_strings(), "or")}')

    def load(self,
             dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
             use_auth_token: _types.OptionalString = None,
             local_files_only: bool = False,
             sequential_cpu_offload_member: bool = False,
             model_cpu_offload_member: bool = False) -> \
            typing.Union[
                diffusers.AutoencoderKL,
                diffusers.AsymmetricAutoencoderKL,
                diffusers.AutoencoderTiny,
                diffusers.ConsistencyDecoderVAE]:
        """
        Load a VAE of type :py:class:`diffusers.AutoencoderKL`, :py:class:`diffusers.AsymmetricAutoencoderKL`,
        :py:class:`diffusers.AutoencoderKLTemporalDecoder`, or :py:class:`diffusers.AutoencoderTiny` from this URI

        :param dtype_fallback: If the URI does not specify a dtype, use this dtype.
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug or blob link

        :param sequential_cpu_offload_member: This model will be attached to
            a pipeline which will have sequential cpu offload enabled?

        :param model_cpu_offload_member: This model will be attached to a pipeline
            which will have model cpu offload enabled?

        :raises ModelNotFoundError: If the model could not be found.

        :return: :py:class:`diffusers.AutoencoderKL`, :py:class:`diffusers.AsymmetricAutoencoderKL`,
            :py:class:`diffusers.AutoencoderKLTemporalDecoder`, or :py:class:`diffusers.AutoencoderTiny`
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
            raise _exceptions.VAEUriLoadError(
                f'error loading vae "{self.model}": {e}')

    @_memoize(_cache._TORCH_VAE_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _d_memoize.struct_hasher}),
              on_hit=lambda key, hit: _d_memoize.simple_cache_hit_debug("Torch VAE", key, hit),
              on_create=lambda key, new: _d_memoize.simple_cache_miss_debug("Torch VAE", key, new))
    def _load(self,
              dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
              use_auth_token: _types.OptionalString = None,
              local_files_only: bool = False,
              sequential_cpu_offload_member: bool = False,
              model_cpu_offload_member: bool = False) -> \
            typing.Union[
                diffusers.AutoencoderKL,
                diffusers.AsymmetricAutoencoderKL,
                diffusers.AutoencoderTiny,
                diffusers.ConsistencyDecoderVAE]:

        if sequential_cpu_offload_member and model_cpu_offload_member:
            # these are used for cache differentiation only
            raise ValueError('sequential_cpu_offload_member and model_cpu_offload_member cannot both be True.')

        if self.dtype is None:
            torch_dtype = _enums.get_torch_dtype(dtype_fallback)
        else:
            torch_dtype = _enums.get_torch_dtype(self.dtype)

        encoder = self._encoders[self.encoder]

        model_path = _hfutil.download_non_hf_model(self.model)

        single_file_load_path = _hfutil.is_single_file_model_load(model_path)

        if single_file_load_path:
            estimated_memory_use = _hfutil.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token
            )

            _cache.enforce_vae_cache_constraints(new_vae_size=estimated_memory_use)

            if encoder is diffusers.AutoencoderKL:
                # There is a bug in their cast
                vae = encoder.from_single_file(model_path,
                                               token=use_auth_token,
                                               revision=self.revision,
                                               local_files_only=local_files_only) \
                    .to(dtype=torch_dtype, non_blocking=False)
            else:
                vae = encoder.from_single_file(model_path,
                                               token=use_auth_token,
                                               revision=self.revision,
                                               torch_dtype=torch_dtype,
                                               local_files_only=local_files_only)

        else:

            estimated_memory_use = _hfutil.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                variant=self.variant,
                subfolder=self.subfolder,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token
            )

            _cache.enforce_vae_cache_constraints(new_vae_size=estimated_memory_use)

            vae = encoder.from_pretrained(
                model_path,
                revision=self.revision,
                variant=self.variant,
                torch_dtype=torch_dtype,
                subfolder=self.subfolder,
                token=use_auth_token,
                local_files_only=local_files_only)

        _messages.debug_log('Estimated Torch VAE Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_use))

        _cache.vae_create_update_cache_info(vae=vae,
                                            estimated_size=estimated_memory_use)

        return vae

    @staticmethod
    def parse(uri: _types.Uri) -> 'TorchVAEUri':
        """
        Parse a ``--model-type`` torch* ``--vae`` uri and return an object representing its constituents

        :param uri: string with ``--vae`` uri syntax

        :raise InvalidVaeUriError:

        :return: :py:class:`.TorchVAEPath`
        """
        try:
            r = _torch_vae_uri_parser.parse(uri)

            model = r.args.get('model')
            if model is None:
                raise _exceptions.InvalidVaeUriError('model argument for torch VAE specification must be defined.')

            dtype = r.args.get('dtype')

            supported_dtypes = _enums.supported_data_type_strings()
            if dtype is not None and dtype not in supported_dtypes:
                raise _exceptions.InvalidVaeUriError(
                    f'Torch VAE "dtype" must be {", ".join(supported_dtypes)}, '
                    f'or left undefined, received: {dtype}')

            return TorchVAEUri(encoder=r.concept,
                               model=model,
                               revision=r.args.get('revision', None),
                               variant=r.args.get('variant', None),
                               dtype=dtype,
                               subfolder=r.args.get('subfolder', None))
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidVaeUriError(e)
