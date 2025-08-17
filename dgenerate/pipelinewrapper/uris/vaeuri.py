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
import diffusers.loaders

import dgenerate.exceptions as _d_exceptions
import dgenerate.hfhub as _hfhub
import dgenerate.memoize as _d_memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.util as _pipelinewrapper_util
import dgenerate.textprocessing as _textprocessing
import dgenerate.torchutil as _torchutil
import dgenerate.types as _types
from dgenerate.memoize import memoize as _memoize
from dgenerate.pipelinewrapper import constants as _constants
from dgenerate.pipelinewrapper.uris import exceptions as _exceptions
from dgenerate.pipelinewrapper.uris import util as _util

_vae_uri_parser = _textprocessing.ConceptUriParser(
    'VAE', [
        'model',
        'revision',
        'variant',
        'subfolder',
        'dtype'
    ]
)

_vae_cache = _d_memoize.create_object_cache(
    'vae', cache_type=_memory.SizedConstrainedObjectCache
)


class VAEUri:
    """
    Representation of ``--vae`` URI.
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

    @property
    def extract(self) -> False:
        """
        Extract from a single file checkpoint containing multiple components?
        """
        return self._extract

    _encoders = {
        'AutoencoderKL': diffusers.AutoencoderKL,
        'AsymmetricAutoencoderKL': diffusers.AsymmetricAutoencoderKL,
        'AutoencoderTiny': diffusers.AutoencoderTiny,
        'ConsistencyDecoderVAE': diffusers.ConsistencyDecoderVAE
    }

    @staticmethod
    def supported_encoder_names() -> list[str]:
        return list(VAEUri._encoders.keys())


    # pipelinewrapper.uris.util.get_uri_accepted_args_schema metadata

    NAMES = ['VAE']

    @staticmethod
    def help():
        import dgenerate.arguments as _a

        return _a.get_raw_help_text('--vae')

    OPTION_ARGS = {
        'encoder': list(_encoders.keys()),
        'dtype': ['float16', 'bfloat16', 'float32']
    }

    FILE_ARGS = {
        'model': {'mode': ['in', 'dir'], 'filetypes': [('Models', ['*.safetensors', '*.pt', '*.pth', '*.cpkt', '*.bin'])]}
    }

    # ===

    def __init__(self,
                 encoder: str,
                 model: str,
                 revision: _types.OptionalString = None,
                 variant: _types.OptionalString = None,
                 subfolder: _types.OptionalString = None,
                 extract: bool = False,
                 dtype: _enums.DataType | str | None = None):
        """
        :param encoder: encoder class name, for example ``AutoencoderKL``
        :param model: model path
        :param revision: model revision (branch name)
        :param variant: model variant, for example ``fp16``
        :param subfolder: model subfolder
        :param extract: Extract the VAE from a single file checkpoint
            that contains other models, such as a UNet or Text Encoders.
        :param dtype: model data type (precision)

        :raises InvalidVaeUriError: If ``dtype`` is passed an invalid data type string, or if
            ``model`` points to a single file and the specified ``encoder`` class name does not
            support loading from a single file.
        """

        if encoder not in self._encoders:
            raise _exceptions.InvalidVaeUriError(
                f'Unknown VAE encoder class {encoder}, must be one of: {_textprocessing.oxford_comma(self._encoders.keys(), "or")}')

        can_single_file_load = hasattr(self._encoders[encoder], 'from_single_file')
        single_file_load_path = _hfhub.is_single_file_model_load(model)

        if single_file_load_path and not can_single_file_load and not extract:
            raise _exceptions.InvalidVaeUriError(
                f'{encoder} is not capable of loading from a single file, '
                f'must be loaded from a huggingface repository slug or folder on disk.')

        if not single_file_load_path:
            if extract:
                raise _exceptions.InvalidVaeUriError(
                    f'VAE URI cannot have "extract" set to True when "model" is not '
                    f'pointing to a single file checkpoint.')

        if single_file_load_path and not extract:
            if subfolder is not None:
                raise _exceptions.InvalidVaeUriError(
                    'Single file VAE loads do not support the subfolder option when "extract" is False.')

        self._extract = extract
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
             original_config: _types.OptionalPath = None,
             use_auth_token: _types.OptionalString = None,
             local_files_only: bool = False,
             no_cache: bool = False,
             missing_ok: bool = False) -> \
            typing.Union[
                diffusers.AutoencoderKL,
                diffusers.AsymmetricAutoencoderKL,
                diffusers.AutoencoderTiny,
                diffusers.ConsistencyDecoderVAE, None]:
        """
        Load a VAE of type :py:class:`diffusers.AutoencoderKL`, :py:class:`diffusers.AsymmetricAutoencoderKL`,
        :py:class:`diffusers.AutoencoderKLTemporalDecoder`, or :py:class:`diffusers.AutoencoderTiny` from this URI

        :param dtype_fallback: If the URI does not specify a dtype, use this dtype.
        :param original_config: Path to original model configuration for single file checkpoints, URL or `.yaml` file on disk.
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug or blob link

        :param no_cache: If ``True``, force the returned object not to be cached by the memoize decorator.
        :param missing_ok: If ``True``, when a VAE is not found inside a single file checkpoint as a sub model,
            just return ``None`` instead of throwing an error.

        :raises ModelNotFoundError: If the model could not be found.

        :return: :py:class:`diffusers.AutoencoderKL`, :py:class:`diffusers.AsymmetricAutoencoderKL`,
            :py:class:`diffusers.AutoencoderKLTemporalDecoder`, or :py:class:`diffusers.AutoencoderTiny`
        """
        def cache_all(e):
            raise _exceptions.VAEUriLoadError(
                f'error loading vae "{self.model}": {e}') from e

        with _hfhub.with_hf_errors_as_model_not_found(cache_all):
            args = locals()
            args.pop('self')
            args.pop('cache_all')
            return self._load(**args)

    @staticmethod
    def _enforce_cache_size(new_vae_size):
        _vae_cache.enforce_cpu_mem_constraints(
            _constants.VAE_CACHE_MEMORY_CONSTRAINTS,
            size_var='vae_size',
            new_object_size=new_vae_size)

    @_memoize(_vae_cache,
              exceptions={'local_files_only', 'missing_ok'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _d_memoize.property_hasher}),
              on_hit=lambda key, hit: _d_memoize.simple_cache_hit_debug("Torch VAE", key, hit),
              on_create=lambda key, new: _d_memoize.simple_cache_miss_debug("Torch VAE", key, new))
    def _load(self,
              dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
              original_config: _types.OptionalPath = None,
              use_auth_token: _types.OptionalString = None,
              local_files_only: bool = False,
              no_cache: bool = False,
              missing_ok: bool = False
              ) -> \
            typing.Union[
                diffusers.AutoencoderKL,
                diffusers.AsymmetricAutoencoderKL,
                diffusers.AutoencoderTiny,
                diffusers.ConsistencyDecoderVAE, None]:

        if self.dtype is None:
            torch_dtype = _enums.get_torch_dtype(dtype_fallback)
        else:
            torch_dtype = _enums.get_torch_dtype(self.dtype)

        encoder = self._encoders[self.encoder]

        model_path = _hfhub.download_non_hf_slug_model(self.model)

        single_file_load_path = _hfhub.is_single_file_model_load(model_path)

        if single_file_load_path:
            try:
                original_config = _hfhub.download_non_hf_slug_config(
                    original_config) if original_config else None
            except _hfhub.NonHFConfigDownloadError as e:
                raise _exceptions.VAEUriLoadError(
                    f'original config file "{original_config}" for VAE could not be downloaded: {e}'
                ) from e

            estimated_memory_use = _pipelinewrapper_util.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token
            )

            self._enforce_cache_size(estimated_memory_use)

            if not self.extract:
                if encoder is diffusers.AutoencoderKL:
                    # There is a bug in their cast
                    vae = encoder.from_single_file(
                        model_path,
                        token=use_auth_token,
                        revision=self.revision,
                        original_config=original_config,
                        local_files_only=local_files_only
                    ).to(dtype=torch_dtype, non_blocking=False)
                else:
                    vae = encoder.from_single_file(
                        model_path,
                        token=use_auth_token,
                        revision=self.revision,
                        original_config=original_config,
                        torch_dtype=torch_dtype,
                        local_files_only=local_files_only
                    )

            else:
                try:
                    vae = _pipelinewrapper_util.single_file_load_sub_module(
                        path=model_path,
                        class_name=self.encoder,
                        library_name='diffusers',
                        name=self.subfolder if self.subfolder else 'vae',
                        original_config=original_config,
                        use_auth_token=use_auth_token,
                        local_files_only=local_files_only,
                        revision=self.revision,
                        dtype=torch_dtype
                    )
                except FileNotFoundError as e:
                    # cannot find configs
                    raise _d_exceptions.ModelNotFoundError(e) from e
                except diffusers.loaders.single_file.SingleFileComponentError as e:
                    if missing_ok:
                        # noinspection PyTypeChecker
                        return None, _d_memoize.CachedObjectMetadata(
                            size=0,
                            skip=True
                        )
                    raise _exceptions.VAEUriLoadError(
                        f'Failed to load VAE from single file checkpoint {model_path}, '
                        f'make sure the file contains a VAE.') from e

                estimated_memory_use = _torchutil.estimate_module_memory_usage(vae)
        else:
            if original_config:
                raise _exceptions.VAEUriLoadError(
                    'specifying original_config file for VAE '
                    'is only supported for single file loads.')

            estimated_memory_use = _pipelinewrapper_util.estimate_model_memory_use(
                repo_id=model_path,
                revision=self.revision,
                variant=self.variant,
                subfolder=self.subfolder,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token
            )

            self._enforce_cache_size(estimated_memory_use)

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

        _util._patch_module_to_for_sized_cache(_vae_cache, vae)

        # noinspection PyTypeChecker
        return vae, _d_memoize.CachedObjectMetadata(
            size=estimated_memory_use,
            skip=no_cache
        )

    @staticmethod
    def parse(uri: _types.Uri) -> 'VAEUri':
        """
        Parse a ``--vae`` uri and return an object representing its constituents

        :param uri: string with ``--vae`` uri syntax

        :raise InvalidVaeUriError:

        :return: :py:class:`.TorchVAEPath`
        """
        try:
            r = _vae_uri_parser.parse(uri)

            model = r.args.get('model')
            if model is None:
                raise _exceptions.InvalidVaeUriError('model argument for torch VAE specification must be defined.')

            dtype = r.args.get('dtype')

            supported_dtypes = _enums.supported_data_type_strings()
            if dtype is not None and dtype not in supported_dtypes:
                raise _exceptions.InvalidVaeUriError(
                    f'Torch VAE "dtype" must be {", ".join(supported_dtypes)}, '
                    f'or left undefined, received: {dtype}')

            return VAEUri(encoder=r.concept,
                          model=model,
                          revision=r.args.get('revision', None),
                          variant=r.args.get('variant', None),
                          dtype=dtype,
                          subfolder=r.args.get('subfolder', None))
        except _textprocessing.ConceptUriParseError as e:
            raise _exceptions.InvalidVaeUriError(e) from e
