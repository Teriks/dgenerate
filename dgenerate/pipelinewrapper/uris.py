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

import dgenerate.memoize as _d_memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.cache as _cache
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.pipelinewrapper.hfutil as _hfutil
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.memoize import memoize as _memoize

_sdxl_refiner_uri_parser = _textprocessing.ConceptUriParser('SDXL Refiner',
                                                            ['revision', 'variant', 'subfolder', 'dtype'])

_torch_vae_uri_parser = _textprocessing.ConceptUriParser('VAE',
                                                         ['model', 'revision', 'variant', 'subfolder', 'dtype'])

_flax_vae_uri_parser = _textprocessing.ConceptUriParser('VAE', ['model', 'revision', 'subfolder', 'dtype'])

_torch_control_net_uri_parser = _textprocessing.ConceptUriParser('ControlNet',
                                                                 ['scale', 'start', 'end', 'revision', 'variant',
                                                                  'subfolder',
                                                                  'dtype'])

_flax_control_net_uri_parser = _textprocessing.ConceptUriParser('ControlNet',
                                                                ['scale', 'revision', 'subfolder', 'dtype',
                                                                 'from_torch'])

_lora_uri_parser = _textprocessing.ConceptUriParser('LoRA', ['scale', 'revision', 'subfolder', 'weight-name'])
_textual_inversion_uri_parser = _textprocessing.ConceptUriParser('Textual Inversion',
                                                                 ['revision', 'subfolder', 'weight-name'])


class InvalidModelUriError(Exception):
    """
    Thrown on model path syntax or logical usage error
    """
    pass


class InvalidSDXLRefinerUriError(InvalidModelUriError):
    """
    Error in ``--sdxl-refiner`` uri
    """
    pass


class InvalidVaeUriError(InvalidModelUriError):
    """
    Error in ``--vae`` uri
    """
    pass


class InvalidControlNetUriError(InvalidModelUriError):
    """
    Error in ``--control-nets`` uri
    """
    pass


class InvalidLoRAUriError(InvalidModelUriError):
    """
    Error in ``--lora`` uri
    """
    pass


class InvalidTextualInversionUriError(InvalidModelUriError):
    """
    Error in ``--textual-inversions`` uri
    """
    pass


class FlaxControlNetUri:
    """
    Representation of ``--control-nets`` uri when ``--model-type`` flax*
    """

    model: str
    """
    Model path, huggingface slug
    """

    revision: _types.OptionalString
    """
    Model repo revision
    """

    subfolder: _types.OptionalPath
    """
    Model repo subfolder
    """

    dtype: typing.Optional[_enums.DataTypes]
    """
    Model dtype (precision)
    """

    scale: float
    """
    ControlNet guidance scale
    """

    from_torch: bool
    """
    Load from a model format meant for torch?
    """

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None,
                 dtype: typing.Union[_enums.DataTypes, str, None] = None,
                 scale: float = 1.0,
                 from_torch: bool = False):

        self.model = model
        self.revision = revision
        self.subfolder = subfolder
        self.dtype = _enums.get_data_type_enum(dtype) if dtype else None
        self.scale = scale
        self.from_torch = from_torch

    def load(self,
             dtype_fallback: _enums.DataTypes = _enums.DataTypes.AUTO,
             use_auth_token: _types.OptionalString = None,
             local_files_only: bool = False) -> typing.Tuple[diffusers.FlaxControlNetModel, typing.Any]:
        """
        Load a :py:class:`diffusers.FlaxControlNetModel` from this URI.

        :param dtype_fallback: Fallback datatype if ``dtype`` was not specified in the URI.

        :param use_auth_token: Optional huggingface API auth token, used for downloading
            restricted repos that your account has access to.

        :param local_files_only: Avoid connecting to huggingface to download models and
            only use cached models?

        :return: tuple (:py:class:`diffusers.FlaxControlNetModel`, flax_control_net_params)
        """
        return self._load(dtype_fallback, use_auth_token, local_files_only)

    @_memoize(_cache._FLAX_CONTROL_NET_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _cache.struct_hasher}),
              on_hit=lambda key, hit: _cache.simple_cache_hit_debug("Flax ControlNet", key, hit),
              on_create=lambda key, new: _cache.simple_cache_miss_debug("Flax ControlNet", key, new))
    def _load(self,
              dtype_fallback: _enums.DataTypes = _enums.DataTypes.AUTO,
              use_auth_token: _types.OptionalString = None,
              local_files_only: bool = False) -> typing.Tuple[diffusers.FlaxControlNetModel, typing.Any]:

        single_file_load_path = _hfutil.is_single_file_model_load(self.model)

        if single_file_load_path:
            raise NotImplementedError('Flax --control-nets do not support single file loads from disk.')
        else:

            estimated_memory_usage = _hfutil.estimate_model_memory_use(
                repo_id=self.model,
                revision=self.revision,
                subfolder=self.subfolder,
                flax=not self.from_torch,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only
            )

            _cache.enforce_control_net_cache_constraints(
                new_control_net_size=estimated_memory_usage)

            flax_dtype = _enums.get_flax_dtype(
                dtype_fallback if self.dtype is None else self.dtype)

            new_net: diffusers.FlaxControlNetModel = \
                diffusers.FlaxControlNetModel.from_pretrained(self.model,
                                                              revision=self.revision,
                                                              subfolder=self.subfolder,
                                                              dtype=flax_dtype,
                                                              from_pt=self.from_torch,
                                                              use_auth_token=use_auth_token,
                                                              local_files_only=local_files_only)

        _messages.debug_log('Estimated Flax ControlNet Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_usage))

        _cache.controlnet_create_update_cache_info(controlnet=new_net[0],
                                                   estimated_size=estimated_memory_usage)

        return new_net


def parse_flax_control_net_uri(uri: _types.Uri) -> FlaxControlNetUri:
    """
    Parse a ``--model-type`` flax* ``--control-nets`` uri specification and return an object representing its constituents

    :param uri: string with ``--control-nets`` uri syntax

    :raise: :py:class:`.InvalidControlNetUriError`

    :return: :py:class:`.FlaxControlNetPath`
    """
    try:
        r = _flax_control_net_uri_parser.parse_concept_uri(uri)

        dtype = r.args.get('dtype')
        scale = r.args.get('scale', 1.0)
        from_torch = r.args.get('from_torch')

        if from_torch is not None:
            try:
                from_torch = _types.parse_bool(from_torch)
            except ValueError:
                raise InvalidControlNetUriError(
                    f'Flax Control Net from_torch must be undefined or boolean (true or false), received: {from_torch}')

        supported_dtypes = _enums.supported_data_type_strings()
        if dtype is not None and dtype not in supported_dtypes:
            raise InvalidControlNetUriError(
                f'Flax ControlNet "dtype" must be {", ".join(supported_dtypes)}, '
                f'or left undefined, received: {dtype}')

        try:
            scale = float(scale)
        except ValueError:
            raise InvalidControlNetUriError(
                f'Flax Control Net scale must be a floating point number, received {scale}')

        return FlaxControlNetUri(
            model=r.concept,
            revision=r.args.get('revision', None),
            subfolder=r.args.get('subfolder', None),
            scale=scale,
            dtype=dtype,
            from_torch=from_torch)

    except _textprocessing.ConceptPathParseError as e:
        raise InvalidControlNetUriError(e)


class TorchControlNetUri:
    """
    Representation of ``--control-nets`` uri when ``--model-type`` torch*
    """

    model: str
    """
    Model path, huggingface slug
    """

    revision: _types.OptionalString
    """
    Model repo revision
    """

    variant: _types.OptionalString
    """
    Model repo revision
    """

    subfolder: _types.OptionalPath
    """
    Model repo subfolder
    """

    dtype: typing.Optional[_enums.DataTypes]
    """
    Model dtype (precision)
    """

    scale: float
    """
    ControlNet guidance scale
    """

    start: float
    """
    ControlNet guidance start point, fraction of inference / timesteps.
    """

    end: float
    """
    ControlNet guidance end point, fraction of inference / timesteps.
    """

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString,
                 variant: _types.OptionalString,
                 subfolder: _types.OptionalPath,
                 dtype: typing.Union[_enums.DataTypes, str, None] = None,
                 scale: float = 1.0,
                 start: float = 0.0,
                 end: float = 1.0):

        self.model = model
        self.revision = revision
        self.variant = variant
        self.subfolder = subfolder
        self.dtype = _enums.get_data_type_enum(dtype) if dtype else None
        self.scale = scale
        self.start = start
        self.end = end

    def load(self,
             dtype_fallback: _enums.DataTypes = _enums.DataTypes.AUTO,
             use_auth_token: _types.OptionalString = None,
             local_files_only: bool = False) -> diffusers.ControlNetModel:
        """
        Load a :py:class:`diffusers.ControlNetModel` from this URI.

        :param dtype_fallback: Fallback datatype if ``dtype`` was not specified in the URI.

        :param use_auth_token: Optional huggingface API auth token, used for downloading
            restricted repos that your account has access to.

        :param local_files_only: Avoid connecting to huggingface to download models and
            only use cached models?

        :return: :py:class:`diffusers.ControlNetModel`
        """

        return self._load(dtype_fallback, use_auth_token, local_files_only)

    @_memoize(_cache._TORCH_CONTROL_NET_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _cache.struct_hasher}),
              on_hit=lambda key, hit: _cache.simple_cache_hit_debug("Torch ControlNet", key, hit),
              on_create=lambda key, new: _cache.simple_cache_miss_debug("Torch ControlNet", key, new))
    def _load(self,
              dtype_fallback: _enums.DataTypes = _enums.DataTypes.AUTO,
              use_auth_token: _types.OptionalString = None,
              local_files_only: bool = False) -> diffusers.ControlNetModel:

        single_file_load_path = _hfutil.is_single_file_model_load(self.model)

        torch_dtype = _enums.get_torch_dtype(
            dtype_fallback if self.dtype is None else self.dtype)

        if single_file_load_path:

            estimated_memory_usage = _hfutil.estimate_model_memory_use(
                repo_id=self.model,
                revision=self.revision,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only
            )

            _cache.enforce_control_net_cache_constraints(
                new_control_net_size=estimated_memory_usage)

            new_net: diffusers.ControlNetModel = \
                diffusers.ControlNetModel.from_single_file(self.model,
                                                           revision=self.revision,
                                                           torch_dtype=torch_dtype,
                                                           use_auth_token=use_auth_token,
                                                           local_files_only=local_files_only)
        else:

            estimated_memory_usage = _hfutil.estimate_model_memory_use(
                repo_id=self.model,
                revision=self.revision,
                variant=self.variant,
                subfolder=self.subfolder,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only
            )

            _cache.enforce_control_net_cache_constraints(
                new_control_net_size=estimated_memory_usage)

            new_net: diffusers.ControlNetModel = \
                diffusers.ControlNetModel.from_pretrained(self.model,
                                                          revision=self.revision,
                                                          variant=self.variant,
                                                          subfolder=self.subfolder,
                                                          torch_dtype=torch_dtype,
                                                          use_auth_token=use_auth_token,
                                                          local_files_only=local_files_only)

        _messages.debug_log('Estimated Torch ControlNet Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_usage))

        _cache.controlnet_create_update_cache_info(controlnet=new_net,
                                                   estimated_size=estimated_memory_usage)

        return new_net


def parse_torch_control_net_uri(uri: _types.Uri) -> TorchControlNetUri:
    """
    Parse a ``--model-type`` torch* ``--control-nets`` uri specification and return an object representing its constituents

    :param uri: string with ``--control-nets`` uri syntax

    :raise: :py:class:`.InvalidControlNetUriError`

    :return: :py:class:`.TorchControlNetPath`
    """
    try:
        r = _torch_control_net_uri_parser.parse_concept_uri(uri)

        dtype = r.args.get('dtype')
        scale = r.args.get('scale', 1.0)
        start = r.args.get('start', 0.0)
        end = r.args.get('end', 1.0)

        supported_dtypes = _enums.supported_data_type_strings()
        if dtype is not None and dtype not in supported_dtypes:
            raise InvalidControlNetUriError(
                f'Torch ControlNet "dtype" must be {", ".join(supported_dtypes)}, '
                f'or left undefined, received: {dtype}')

        try:
            scale = float(scale)
        except ValueError:
            raise InvalidControlNetUriError(
                f'Torch ControlNet "scale" must be a floating point number, received: {scale}')

        try:
            start = float(start)
        except ValueError:
            raise InvalidControlNetUriError(
                f'Torch ControlNet "start" must be a floating point number, received: {start}')

        try:
            end = float(end)
        except ValueError:
            raise InvalidControlNetUriError(
                f'Torch ControlNet "end" must be a floating point number, received: {end}')

        if start > end:
            raise InvalidControlNetUriError(
                f'Torch ControlNet "start" must be less than or equal to "end".')

        return TorchControlNetUri(
            model=r.concept,
            revision=r.args.get('revision', None),
            variant=r.args.get('variant', None),
            subfolder=r.args.get('subfolder', None),
            dtype=dtype,
            scale=scale,
            start=start,
            end=end)

    except _textprocessing.ConceptPathParseError as e:
        raise InvalidControlNetUriError(e)


class SDXLRefinerUri:
    """
    Representation of ``--sdxl-refiner`` uri
    """

    model: str
    """
    Model path, huggingface slug
    """

    revision: _types.OptionalString
    """
    Model repo revision
    """

    variant: _types.OptionalString
    """
    Model repo revision
    """

    subfolder: _types.OptionalPath
    """
    Model repo subfolder
    """

    dtype: typing.Optional[_enums.DataTypes]
    """
    Model dtype (precision)
    """

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 variant: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None,
                 dtype: typing.Union[_enums.DataTypes, str, None] = None):
        self.model = model
        self.revision = revision
        self.variant = variant
        self.dtype = _enums.get_data_type_enum(dtype) if dtype else None
        self.subfolder = subfolder


def parse_sdxl_refiner_uri(uri: _types.Uri) -> SDXLRefinerUri:
    """
    Parse an ``--sdxl-refiner`` uri and return an object representing its constituents

    :param uri: string with ``--sdxl-refiner`` uri syntax

    :raise: :py:class:`.InvalidSDXLRefinerUriError`

    :return: :py:class:`.SDXLRefinerPath`
    """
    try:
        r = _sdxl_refiner_uri_parser.parse_concept_uri(uri)

        supported_dtypes = _enums.supported_data_type_strings()

        dtype = r.args.get('dtype', None)
        if dtype is not None and dtype not in supported_dtypes:
            raise InvalidSDXLRefinerUriError(
                f'Torch SDXL refiner "dtype" must be {", ".join(supported_dtypes)}, '
                f'or left undefined, received: {dtype}')

        return SDXLRefinerUri(
            model=r.concept,
            revision=r.args.get('revision', None),
            variant=r.args.get('variant', None),
            dtype=dtype,
            subfolder=r.args.get('subfolder', None))
    except _textprocessing.ConceptPathParseError as e:
        raise InvalidSDXLRefinerUriError(e)


class TorchVAEUri:
    """
    Representation of ``--vae`` uri when ``--model-type`` torch*
    """

    encoder: str
    """
    Encoder class name such as "AutoencoderKL"
    """

    model: str
    """
    Model path, huggingface slug, file path, or blob link
    """

    revision: _types.OptionalString
    """
    Model repo revision
    """

    variant: _types.OptionalString
    """
    Model repo revision
    """

    subfolder: _types.OptionalPath
    """
    Model repo subfolder
    """

    dtype: typing.Optional[_enums.DataTypes]
    """
    Model dtype (precision)
    """

    def __init__(self,
                 encoder: str,
                 model: str,
                 revision: _types.OptionalString = None,
                 variant: _types.OptionalString = None,
                 subfolder: _types.OptionalString = None,
                 dtype: typing.Union[_enums.DataTypes, str, None] = None):

        self.encoder = encoder
        self.model = model
        self.revision = revision
        self.variant = variant
        self.dtype = _enums.get_data_type_enum(dtype) if dtype else None
        self.subfolder = subfolder

    def load(self,
             dtype_fallback: _enums.DataTypes = _enums.DataTypes.AUTO,
             use_auth_token: _types.OptionalString = None,
             local_files_only=False) -> typing.Union[diffusers.AutoencoderKL,
                                                     diffusers.AsymmetricAutoencoderKL,
                                                     diffusers.AutoencoderTiny]:
        """
        Load a VAE of type: :py:class:`diffusers.AutoencoderKL`, :py:class:`diffusers.AsymmetricAutoencoderKL`,
          or :py:class:`diffusers.AutoencoderTiny` from this URI

        :param dtype_fallback: If the URI does not specify a dtype, use this dtype.
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug or blob link
        :return: :py:class:`diffusers.AutoencoderKL`, :py:class:`diffusers.AsymmetricAutoencoderKL`,
          or :py:class:`diffusers.AutoencoderTiny`
        """
        return self._load(dtype_fallback, use_auth_token, local_files_only)

    @_memoize(_cache._TORCH_VAE_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _cache.struct_hasher}),
              on_hit=lambda key, hit: _cache.simple_cache_hit_debug("Torch VAE", key, hit),
              on_create=lambda key, new: _cache.simple_cache_miss_debug("Torch VAE", key, new))
    def _load(self,
              dtype_fallback: _enums.DataTypes = _enums.DataTypes.AUTO,
              use_auth_token: _types.OptionalString = None,
              local_files_only=False) -> typing.Union[diffusers.AutoencoderKL,
                                                      diffusers.AsymmetricAutoencoderKL,
                                                      diffusers.AutoencoderTiny]:

        if self.dtype is None:
            torch_dtype = _enums.get_torch_dtype(dtype_fallback)
        else:
            torch_dtype = _enums.get_torch_dtype(self.dtype)

        encoder_name = self.encoder

        if encoder_name == 'AutoencoderKL':
            encoder = diffusers.AutoencoderKL
        elif encoder_name == 'AsymmetricAutoencoderKL':
            encoder = diffusers.AsymmetricAutoencoderKL
        elif encoder_name == 'AutoencoderTiny':
            encoder = diffusers.AutoencoderTiny
        else:
            raise InvalidVaeUriError(f'Unknown VAE encoder class {encoder_name}')

        path = self.model

        can_single_file_load = hasattr(encoder, 'from_single_file')
        single_file_load_path = _hfutil.is_single_file_model_load(path)

        if single_file_load_path and not can_single_file_load:
            raise NotImplementedError(f'{encoder_name} is not capable of loading from a single file, '
                                      f'must be loaded from a huggingface repository slug or folder on disk.')

        if single_file_load_path:
            if self.subfolder is not None:
                raise NotImplementedError('Single file VAE loads do not support the subfolder option.')

            estimated_memory_use = _hfutil.estimate_model_memory_use(
                repo_id=path,
                revision=self.revision,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token
            )

            _cache.enforce_vae_cache_constraints(new_vae_size=estimated_memory_use)

            if encoder is diffusers.AutoencoderKL:
                # There is a bug in their cast
                vae = encoder.from_single_file(path,
                                               revision=self.revision,
                                               local_files_only=local_files_only) \
                    .to(dtype=torch_dtype, non_blocking=False)
            else:
                vae = encoder.from_single_file(path,
                                               revision=self.revision,
                                               torch_dtype=torch_dtype,
                                               local_files_only=local_files_only)

        else:

            estimated_memory_use = _hfutil.estimate_model_memory_use(
                repo_id=path,
                revision=self.revision,
                variant=self.variant,
                subfolder=self.subfolder,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token
            )

            _cache.enforce_vae_cache_constraints(new_vae_size=estimated_memory_use)

            vae = encoder.from_pretrained(path,
                                          revision=self.revision,
                                          variant=self.variant,
                                          torch_dtype=torch_dtype,
                                          subfolder=self.subfolder,
                                          use_auth_token=use_auth_token,
                                          local_files_only=local_files_only)

        _messages.debug_log('Estimated Torch VAE Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_use))

        _cache.vae_create_update_cache_info(vae=vae,
                                            estimated_size=estimated_memory_use)

        return vae


def parse_torch_vae_uri(uri: _types.Uri) -> TorchVAEUri:
    """
    Parse a ``--model-type`` torch* ``--vae`` uri and return an object representing its constituents

    :param uri: string with ``--vae`` uri syntax

    :raise: :py:class:`.InvalidVaeUriError`

    :return: :py:class:`.TorchVAEPath`
    """
    try:
        r = _torch_vae_uri_parser.parse_concept_uri(uri)

        model = r.args.get('model')
        if model is None:
            raise InvalidVaeUriError('model argument for torch VAE specification must be defined.')

        dtype = r.args.get('dtype')

        supported_dtypes = _enums.supported_data_type_strings()
        if dtype is not None and dtype not in supported_dtypes:
            raise InvalidVaeUriError(
                f'Torch VAE "dtype" must be {", ".join(supported_dtypes)}, '
                f'or left undefined, received: {dtype}')

        return TorchVAEUri(encoder=r.concept,
                           model=model,
                           revision=r.args.get('revision', None),
                           variant=r.args.get('variant', None),
                           dtype=dtype,
                           subfolder=r.args.get('subfolder', None))
    except _textprocessing.ConceptPathParseError as e:
        raise InvalidVaeUriError(e)


class FlaxVAEUri:
    """
    Representation of ``--vae`` uri when ``--model-type`` flax*
    """

    encoder: str
    """
    Encoder class name such as "FlaxAutoencoderKL"
    """

    model: str
    """
    Model path, huggingface slug, file path, or blob link
    """

    revision: _types.OptionalString
    """
    Model repo revision
    """

    subfolder: _types.OptionalPath
    """
    Model repo subfolder
    """

    dtype: typing.Optional[_enums.DataTypes]
    """
    Model dtype (precision)
    """

    def __init__(self,
                 encoder: str,
                 model: str,
                 revision: _types.OptionalString,
                 subfolder: _types.OptionalPath,
                 dtype: typing.Optional[_enums.DataTypes]):

        self.encoder = encoder
        self.model = model
        self.revision = revision
        self.dtype = dtype
        self.subfolder = subfolder

    def load(self,
             dtype_fallback: _enums.DataTypes = _enums.DataTypes.AUTO,
             use_auth_token: _types.OptionalString = None,
             local_files_only=False) -> typing.Tuple[diffusers.FlaxAutoencoderKL, typing.Any]:
        """
        Load a :py:class:`diffusers.FlaxAutoencoderKL` VAE and its flax_params from this URI

        :param dtype_fallback: If the URI does not specify a dtype, use this dtype.
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug or blob link
        :return: tuple (:py:class:`diffusers.FlaxAutoencoderKL`, flax_vae_params)
        """
        return self._load(dtype_fallback, use_auth_token, local_files_only)

    @_memoize(_cache._FLAX_VAE_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _cache.struct_hasher}),
              on_hit=lambda key, hit: _cache.simple_cache_hit_debug("Flax VAE", key, hit),
              on_create=lambda key, new: _cache.simple_cache_miss_debug("Flax VAE", key, new))
    def _load(self,
              dtype_fallback: _enums.DataTypes = _enums.DataTypes.AUTO,
              use_auth_token: _types.OptionalString = None,
              local_files_only=False) -> typing.Tuple[diffusers.FlaxAutoencoderKL, typing.Any]:

        if self.dtype is None:
            flax_dtype = _enums.get_flax_dtype(dtype_fallback)
        else:
            flax_dtype = _enums.get_flax_dtype(self.dtype)

        encoder_name = self.encoder

        if encoder_name == 'FlaxAutoencoderKL':
            encoder = diffusers.FlaxAutoencoderKL
        else:
            raise InvalidVaeUriError(f'Unknown VAE flax encoder class {encoder_name}')

        path = self.model

        can_single_file_load = hasattr(encoder, 'from_single_file')
        single_file_load_path = _hfutil.is_single_file_model_load(path)

        if single_file_load_path and not can_single_file_load:
            raise NotImplementedError(f'{encoder_name} is not capable of loading from a single file, '
                                      f'must be loaded from a huggingface repository slug or folder on disk.')

        if single_file_load_path:
            # in the future this will be supported?
            if self.subfolder is not None:
                raise NotImplementedError('Single file VAE loads do not support the subfolder option.')

            estimated_memory_use = _hfutil.estimate_model_memory_use(
                repo_id=path,
                revision=self.revision,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                flax=True
            )

            _cache.enforce_vae_cache_constraints(new_vae_size=estimated_memory_use)

            vae = encoder.from_single_file(path,
                                           revision=self.revision,
                                           dtype=flax_dtype,
                                           use_auth_token=use_auth_token,
                                           local_files_only=local_files_only)
        else:

            estimated_memory_use = _hfutil.estimate_model_memory_use(
                repo_id=path,
                revision=self.revision,
                subfolder=self.subfolder,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                flax=True
            )

            _cache.enforce_vae_cache_constraints(new_vae_size=estimated_memory_use)

            vae = encoder.from_pretrained(path,
                                          revision=self.revision,
                                          dtype=flax_dtype,
                                          subfolder=self.subfolder,
                                          use_auth_token=use_auth_token,
                                          local_files_only=local_files_only)

        _messages.debug_log('Estimated Flax VAE Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_use))

        _cache.vae_create_update_cache_info(vae=vae[0],
                                            estimated_size=estimated_memory_use)

        return vae


def parse_flax_vae_uri(uri: _types.Uri) -> FlaxVAEUri:
    """
    Parse a ``--model-type`` flax* ``--vae`` uri and return an object representing its constituents

    :param uri: string with ``--vae`` uri syntax

    :raise: :py:class:`.InvalidVaeUriError`

    :return: :py:class:`.FlaxVAEPath`
    """
    try:
        r = _flax_vae_uri_parser.parse_concept_uri(uri)

        model = r.args.get('model')
        if model is None:
            raise InvalidVaeUriError('model argument for flax VAE specification must be defined.')

        dtype = r.args.get('dtype')

        supported_dtypes = _enums.supported_data_type_strings()
        if dtype is not None and dtype not in supported_dtypes:
            raise InvalidVaeUriError(
                f'Flax VAE "dtype" must be {", ".join(supported_dtypes)}, '
                f'or left undefined, received: {dtype}')

        return FlaxVAEUri(encoder=r.concept,
                          model=model,
                          revision=r.args.get('revision', None),
                          dtype=_enums.get_flax_dtype(dtype),
                          subfolder=r.args.get('subfolder', None))
    except _textprocessing.ConceptPathParseError as e:
        raise InvalidVaeUriError(e)


class LoRAUri:
    """
    Representation of ``--lora`` uri
    """

    model: str
    """
    Model path, huggingface slug, file path
    """

    revision: _types.OptionalString
    """
    Model repo revision
    """

    subfolder: _types.OptionalPath
    """
    Model repo subfolder
    """

    weight_name: _types.OptionalName
    """
    Model weight-name
    """

    scale: float
    """
    LoRA scale
    """

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None,
                 weight_name: _types.OptionalName = None,
                 scale: float = 1.0):
        self.model = model
        self.scale = scale
        self.revision = revision
        self.subfolder = subfolder
        self.weight_name = weight_name

    def __str__(self):
        return f'{self.__class__.__name__}({str(_types.get_public_attributes(self))})'

    def __repr__(self):
        return str(self)

    def load_on_pipeline(self,
                         pipeline: diffusers.DiffusionPipeline,
                         use_auth_token: _types.OptionalString = None,
                         local_files_only=False):
        """
        Load LoRA weights on to a pipeline using this URI

        :param pipeline: :py:class:`diffusers.DiffusionPipeline`
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug
        """

        extra_args = {k: v for k, v in locals().items() if k not in {'self', 'pipeline'}}

        if hasattr(pipeline, 'load_lora_weights'):
            _messages.debug_log('pipeline.load_lora_weights('
                                + str(_types.get_public_attributes(self) | extra_args) + ')')

            pipeline.load_lora_weights(self.model,
                                       revision=self.revision,
                                       subfolder=self.subfolder,
                                       weight_name=self.weight_name,
                                       **extra_args)
            _messages.debug_log(f'Added LoRA: "{self}" to pipeline: "{pipeline.__class__.__name__}"')


def parse_lora_uri(uri: _types.Uri) -> LoRAUri:
    """
    Parse a ``--lora`` uri and return an object representing its constituents

    :param uri: string with ``--lora`` uri syntax

    :raise: :py:class:`.InvalidLoRAUriError`

    :return: :py:class:`.LoRAPath`
    """
    try:
        r = _lora_uri_parser.parse_concept_uri(uri)

        return LoRAUri(model=r.concept,
                       scale=float(r.args.get('scale', 1.0)),
                       weight_name=r.args.get('weight-name', None),
                       revision=r.args.get('revision', None),
                       subfolder=r.args.get('subfolder', None))
    except _textprocessing.ConceptPathParseError as e:
        raise InvalidLoRAUriError(e)


class TextualInversionUri:
    """
    Representation of ``--textual-inversions`` uri
    """

    model: str
    """
    Model path, huggingface slug, file path
    """

    revision: _types.OptionalString
    """
    Model repo revision
    """

    subfolder: _types.OptionalPath
    """
    Model repo subfolder
    """

    weight_name: _types.OptionalName
    """
    Model weight-name
    """

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None,
                 weight_name: _types.OptionalName = None):
        self.model = model
        self.revision = revision
        self.subfolder = subfolder
        self.weight_name = weight_name

    def __str__(self):
        return f'{self.__class__.__name__}({str(_types.get_public_attributes(self))})'

    def __repr__(self):
        return str(self)

    def load_on_pipeline(self,
                         pipeline: diffusers.DiffusionPipeline,
                         use_auth_token: _types.OptionalString = None,
                         local_files_only=False):
        """
        Load Textual Inversion weights on to a pipeline using this URI

        :param pipeline: :py:class:`diffusers.DiffusionPipeline`
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug
        """

        extra_args = {k: v for k, v in locals().items() if k not in {'self', 'pipeline'}}

        if hasattr(pipeline, 'load_textual_inversion'):
            _messages.debug_log('pipeline.load_textual_inversion(' +
                                str(_types.get_public_attributes(self) | extra_args) + ')')

            pipeline.load_textual_inversion(self.model,
                                            revision=self.revision,
                                            subfolder=self.subfolder,
                                            weight_name=self.weight_name,
                                            **extra_args)
            _messages.debug_log(f'Added Textual Inversion: "{self}" to pipeline: "{pipeline.__class__.__name__}"')


def parse_textual_inversion_uri(uri: _types.Uri) -> TextualInversionUri:
    """
    Parse a ``--textual-inversions`` uri and return an object representing its constituents

    :param uri: string with ``--textual-inversions`` uri syntax

    :raise: :py:class:`.InvalidTextualInversionUriError`

    :return: :py:class:`.TextualInversionPath`
    """
    try:
        r = _textual_inversion_uri_parser.parse_concept_uri(uri)

        return TextualInversionUri(model=r.concept,
                                   weight_name=r.args.get('weight-name', None),
                                   revision=r.args.get('revision', None),
                                   subfolder=r.args.get('subfolder', None))
    except _textprocessing.ConceptPathParseError as e:
        raise InvalidTextualInversionUriError(e)
