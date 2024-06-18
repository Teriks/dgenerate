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

_flax_vae_uri_parser = _textprocessing.ConceptUriParser('VAE', ['model', 'revision', 'subfolder', 'dtype'])


class FlaxVAEUri:
    """
    Representation of ``--vae`` uri when ``--model-type`` flax*
    """

    @property
    def encoder(self) -> str:
        """
        Encoder class name, "FlaxAutoencoderKL"
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
        'FlaxAutoencoderKL': diffusers.FlaxAutoencoderKL
    }

    def __init__(self,
                 encoder: str,
                 model: str,
                 revision: _types.OptionalString,
                 subfolder: _types.OptionalPath,
                 dtype: _enums.DataType | None):
        """
        :param encoder: Encoder class name
        :param model: model path
        :param revision: model revision (branch name)
        :param subfolder: model subfolder
        :param dtype: model data type (precision)

        :raises InvalidVaeUriError: If ``encoder != 'FlaxAutoencoderKL'``, or if the ``model``
            path is a single file and that is not supported, or if ``dtype`` is passed an
            invalid string.
        """

        if encoder not in self._encoders:
            raise _exceptions.InvalidVaeUriError(
                f'Unknown VAE flax encoder class {encoder}, '
                f'must be one of: {_textprocessing.oxford_comma(self._encoders.keys(), "or")}')

        can_single_file_load = hasattr(self._encoders[encoder], 'from_single_file')
        single_file_load_path = _hfutil.is_single_file_model_load(model)

        if single_file_load_path and not can_single_file_load:
            raise _exceptions.InvalidVaeUriError(
                f'{encoder} is not capable of loading from a single file, '
                f'must be loaded from a huggingface repository slug or folder on disk.')

        if single_file_load_path:
            # in the future this will be supported?
            if subfolder is not None:
                raise _exceptions.InvalidVaeUriError('Single file VAE loads do not support the subfolder option.')

        self._encoder = encoder
        self._model = model
        self._revision = revision

        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise _exceptions.InvalidVaeUriError(
                f'invalid dtype string, must be one of: {_textprocessing.oxford_comma(_enums.supported_data_type_strings(), "or")}')

        self._subfolder = subfolder

    def load(self,
             dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
             use_auth_token: _types.OptionalString = None,
             local_files_only: bool = False) -> tuple[diffusers.FlaxAutoencoderKL, typing.Any]:
        """
        Load a :py:class:`diffusers.FlaxAutoencoderKL` VAE and its flax_params from this URI

        :param dtype_fallback: If the URI does not specify a dtype, use this dtype.
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug or blob link

        :raises ModelNotFoundError: If the model could not be found.

        :return: tuple (:py:class:`diffusers.FlaxAutoencoderKL`, flax_vae_params)
        """
        try:
            return self._load(dtype_fallback, use_auth_token, local_files_only)
        except (huggingface_hub.utils.HFValidationError,
                huggingface_hub.utils.HfHubHTTPError) as e:
            raise _hfutil.ModelNotFoundError(e)
        except Exception as e:
            raise _exceptions.VAEUriLoadError(
                f'error loading vae "{self.model}": {e}')

    @_memoize(_cache._FLAX_VAE_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _d_memoize.struct_hasher}),
              on_hit=lambda key, hit: _d_memoize.simple_cache_hit_debug("Flax VAE", key, hit[0]),
              on_create=lambda key, new: _d_memoize.simple_cache_miss_debug("Flax VAE", key, new[0]))
    def _load(self,
              dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
              use_auth_token: _types.OptionalString = None,
              local_files_only: bool = False) -> tuple[diffusers.FlaxAutoencoderKL, typing.Any]:

        if self.dtype is None:
            flax_dtype = _enums.get_flax_dtype(dtype_fallback)
        else:
            flax_dtype = _enums.get_flax_dtype(self.dtype)

        encoder = self._encoders[self.encoder]

        model_path = _hfutil.download_non_hf_model(self.model)

        single_file_load_path = _hfutil.is_single_file_model_load(model_path)

        if single_file_load_path:

            estimated_memory_use = _hfutil.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                flax=True
            )

            _cache.enforce_vae_cache_constraints(new_vae_size=estimated_memory_use)

            vae = encoder.from_single_file(
                model_path,
                revision=self.revision,
                dtype=flax_dtype,
                token=use_auth_token,
                local_files_only=local_files_only)
        else:

            estimated_memory_use = _hfutil.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                subfolder=self.subfolder,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                flax=True
            )

            _cache.enforce_vae_cache_constraints(new_vae_size=estimated_memory_use)

            vae = encoder.from_pretrained(
                model_path,
                revision=self.revision,
                dtype=flax_dtype,
                subfolder=self.subfolder,
                token=use_auth_token,
                local_files_only=local_files_only)

        _messages.debug_log('Estimated Flax VAE Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_use))

        _cache.vae_create_update_cache_info(vae=vae[0],
                                            estimated_size=estimated_memory_use)

        return vae

    @staticmethod
    def parse(uri: _types.Uri) -> 'FlaxVAEUri':
        """
        Parse a ``--model-type`` flax* ``--vae`` uri and return an object representing its constituents

        :param uri: string with ``--vae`` uri syntax

        :raise InvalidVaeUriError:

        :return: :py:class:`.FlaxVAEUri`
        """
        try:
            r = _flax_vae_uri_parser.parse(uri)

            model = r.args.get('model')
            if model is None:
                raise _exceptions.InvalidVaeUriError('model argument for flax VAE specification must be defined.')

            dtype = r.args.get('dtype')

            supported_dtypes = _enums.supported_data_type_strings()
            if dtype is not None and dtype not in supported_dtypes:
                raise _exceptions.InvalidVaeUriError(
                    f'Flax VAE "dtype" must be {", ".join(supported_dtypes)}, '
                    f'or left undefined, received: {dtype}')

            return FlaxVAEUri(encoder=r.concept,
                              model=model,
                              revision=r.args.get('revision', None),
                              dtype=_enums.get_flax_dtype(dtype),
                              subfolder=r.args.get('subfolder', None))
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidVaeUriError(e)
