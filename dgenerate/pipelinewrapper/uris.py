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
import os.path
import typing

import diffusers
import huggingface_hub
import safetensors
import safetensors.torch
import torch

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

_s_cascade_decoder_uri_parser = _textprocessing.ConceptUriParser('Stable Cascade decoder',
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
                                                                 ['token', 'revision', 'subfolder', 'weight-name'])

_flax_unet_uri_parser = _textprocessing.ConceptUriParser('UNet',
                                                         ['revision', 'subfolder', 'dtype'])

_torch_unet_uri_parser = _textprocessing.ConceptUriParser('UNet',
                                                          ['revision', 'variant', 'subfolder', 'dtype'])


class ModelUriLoadError(Exception):
    """
    Thrown when model fails to load from a URI for a
    reason other than not being found, such as being
    unsupported.

    This exception refers to loadable sub models such as
    VAEs, LoRAs, ControlNets, Textual Inversions etc.
    """
    pass


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


class InvalidSCascadeDecoderUriError(InvalidModelUriError):
    """
    Error in ``--s-cascade-decoder`` uri
    """
    pass


class InvalidVaeUriError(InvalidModelUriError):
    """
    Error in ``--vae`` uri
    """
    pass


class InvalidUNetUriError(InvalidModelUriError):
    """
    Error in ``--unet`` uri
    """
    pass


class InvalidControlNetUriError(InvalidModelUriError):
    """
    Error in ``--control-nets`` uri
    """
    pass


class InvalidLoRAUriError(InvalidModelUriError):
    """
    Error in ``--loras`` uri
    """
    pass


class InvalidTextualInversionUriError(InvalidModelUriError):
    """
    Error in ``--textual-inversions`` uri
    """
    pass


class ControlNetUriLoadError(ModelUriLoadError):
    """
    Error while loading model file in ``--control-nets`` uri
    """
    pass


class VAEUriLoadError(ModelUriLoadError):
    """
    Error while loading model file in ``--vae`` uri
    """
    pass


class UNetUriLoadError(ModelUriLoadError):
    """
    Error while loading model file in ``--unet / --unet2`` uri
    """
    pass


class LoRAUriLoadError(ModelUriLoadError):
    """
    Error while loading model file in ``--loras`` uri
    """
    pass


class TextualInversionUriLoadError(ModelUriLoadError):
    """
    Error while loading model file in ``--textual-inversions`` uri
    """
    pass


class FlaxControlNetUri:
    """
    Representation of ``--control-nets`` uri when ``--model-type`` flax*
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
        ControlNet guidance scale
        """
        return self._scale

    @property
    def from_torch(self) -> bool:
        """
        Load from a model format meant for torch?
        """
        return self._from_torch

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None,
                 dtype: _enums.DataType | str | None = None,
                 scale: float = 1.0,
                 from_torch: bool = False):
        """
        :param model: model path
        :param revision: model revision (branch)
        :param subfolder: model subfolder
        :param dtype: data type (precision)
        :param scale: control net scale value
        :param from_torch: load from a model designed for torch?

        :raises InvalidControlNetUriError: If the ``model`` path represents a singular file (not supported),
            or if ``dtype`` is passed an invalid string
        """

        if _hfutil.is_single_file_model_load(model):
            raise InvalidControlNetUriError('Flax --control-nets do not support single file loads.')

        self._model = model
        self._revision = revision
        self._subfolder = subfolder

        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise InvalidControlNetUriError(
                f'invalid dtype string, must be one of: {_textprocessing.oxford_comma(_enums.supported_data_type_strings(), "or")}')

        self._scale = scale
        self._from_torch = from_torch

    def load(self,
             dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
             use_auth_token: _types.OptionalString = None,
             local_files_only: bool = False) -> tuple[diffusers.FlaxControlNetModel, typing.Any]:
        """
        Load a :py:class:`diffusers.FlaxControlNetModel` from this URI.

        :param dtype_fallback: Fallback datatype if ``dtype`` was not specified in the URI.

        :param use_auth_token: Optional huggingface API auth token, used for downloading
            restricted repos that your account has access to.

        :param local_files_only: Avoid connecting to huggingface to download models and
            only use cached models?

        :raises ModelNotFoundError: If the model could not be found.

        :return: tuple (:py:class:`diffusers.FlaxControlNetModel`, flax_control_net_params)
        """
        try:
            return self._load(dtype_fallback, use_auth_token, local_files_only)
        except (huggingface_hub.utils.HFValidationError,
                huggingface_hub.utils.HfHubHTTPError) as e:
            raise _hfutil.ModelNotFoundError(e)
        except Exception as e:
            raise ControlNetUriLoadError(
                f'error loading controlnet "{self.model}": {e}')

    @_memoize(_cache._FLAX_CONTROL_NET_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _d_memoize.struct_hasher}),
              on_hit=lambda key, hit: _d_memoize.simple_cache_hit_debug("Flax ControlNet", key, hit[0]),
              on_create=lambda key, new: _d_memoize.simple_cache_miss_debug("Flax ControlNet", key, new[0]))
    def _load(self,
              dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
              use_auth_token: _types.OptionalString = None,
              local_files_only: bool = False) -> tuple[diffusers.FlaxControlNetModel, typing.Any]:

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
            diffusers.FlaxControlNetModel.from_pretrained(
                self.model,
                revision=self.revision,
                subfolder=self.subfolder,
                dtype=flax_dtype,
                from_pt=self.from_torch,
                token=use_auth_token,
                local_files_only=local_files_only)

        _messages.debug_log('Estimated Flax ControlNet Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_usage))

        _cache.controlnet_create_update_cache_info(controlnet=new_net[0],
                                                   estimated_size=estimated_memory_usage)

        return new_net

    @staticmethod
    def parse(uri: _types.Uri) -> 'FlaxControlNetUri':
        """
        Parse a ``--model-type`` flax* ``--control-nets`` uri specification and return an object representing its constituents

        :param uri: string with ``--control-nets`` uri syntax

        :raise InvalidControlNetUriError:

        :return: :py:class:`.FlaxControlNetPath`
        """
        try:
            r = _flax_control_net_uri_parser.parse(uri)

            dtype = r.args.get('dtype')
            scale = r.args.get('scale', 1.0)
            from_torch = r.args.get('from_torch')

            if from_torch is not None:
                try:
                    from_torch = _types.parse_bool(from_torch)
                except ValueError:
                    raise InvalidControlNetUriError(
                        f'Flax ControlNet from_torch must be undefined or boolean (true or false), received: {from_torch}')

            supported_dtypes = _enums.supported_data_type_strings()
            if dtype is not None and dtype not in supported_dtypes:
                raise InvalidControlNetUriError(
                    f'Flax ControlNet "dtype" must be {", ".join(supported_dtypes)}, '
                    f'or left undefined, received: {dtype}')

            try:
                scale = float(scale)
            except ValueError:
                raise InvalidControlNetUriError(
                    f'Flax ControlNet scale must be a floating point number, received {scale}')

            return FlaxControlNetUri(
                model=r.concept,
                revision=r.args.get('revision', None),
                subfolder=r.args.get('subfolder', None),
                scale=scale,
                dtype=dtype,
                from_torch=from_torch)

        except _textprocessing.ConceptUriParseError as e:
            raise InvalidControlNetUriError(e)


class TorchControlNetUri:
    """
    Representation of ``--control-nets`` uri when ``--model-type`` torch*
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
        ControlNet guidance scale
        """
        return self._scale

    @property
    def start(self) -> float:
        """
        ControlNet guidance start point, fraction of inference / timesteps.
        """
        return self._start

    @property
    def end(self) -> float:
        """
        ControlNet guidance end point, fraction of inference / timesteps.
        """
        return self._end

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString,
                 variant: _types.OptionalString,
                 subfolder: _types.OptionalPath,
                 dtype: _enums.DataType | str | None = None,
                 scale: float = 1.0,
                 start: float = 0.0,
                 end: float = 1.0):
        """
        :param model: model path
        :param revision: model revision (branch name)
        :param variant: model variant, for example ``fp16``
        :param subfolder: model subfolder
        :param dtype: model data type (precision)
        :param scale: control net scale
        :param start: control net guidance start value
        :param end: control net guidance end value

        :raises InvalidControlNetUriError: If ``dtype`` is passed an invalid data type string.
        """

        self._model = model
        self._revision = revision
        self._variant = variant
        self._subfolder = subfolder

        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise InvalidControlNetUriError(
                f'invalid dtype string, must be one of: {_textprocessing.oxford_comma(_enums.supported_data_type_strings(), "or")}')

        self._scale = scale
        self._start = start
        self._end = end

    def load(self,
             dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
             use_auth_token: _types.OptionalString = None,
             local_files_only: bool = False,
             sequential_cpu_offload_member: bool = False,
             model_cpu_offload_member: bool = False) -> diffusers.ControlNetModel:
        """
        Load a :py:class:`diffusers.ControlNetModel` from this URI.



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

        :return: :py:class:`diffusers.ControlNetModel`
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
            raise ControlNetUriLoadError(
                f'error loading controlnet "{self.model}": {e}')

    @_memoize(_cache._TORCH_CONTROL_NET_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _d_memoize.struct_hasher}),
              on_hit=lambda key, hit: _d_memoize.simple_cache_hit_debug("Torch ControlNet", key, hit),
              on_create=lambda key, new: _d_memoize.simple_cache_miss_debug("Torch ControlNet", key, new))
    def _load(self,
              dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
              use_auth_token: _types.OptionalString = None,
              local_files_only: bool = False,
              sequential_cpu_offload_member: bool = False,
              model_cpu_offload_member: bool = False) -> diffusers.ControlNetModel:

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

            _cache.enforce_control_net_cache_constraints(
                new_control_net_size=estimated_memory_usage)

            new_net: diffusers.ControlNetModel = \
                diffusers.ControlNetModel.from_single_file(
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

            _cache.enforce_control_net_cache_constraints(
                new_control_net_size=estimated_memory_usage)

            new_net: diffusers.ControlNetModel = \
                diffusers.ControlNetModel.from_pretrained(
                    model_path,
                    revision=self.revision,
                    variant=self.variant,
                    subfolder=self.subfolder,
                    torch_dtype=torch_dtype,
                    token=use_auth_token,
                    local_files_only=local_files_only)

        _messages.debug_log('Estimated Torch ControlNet Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_usage))

        _cache.controlnet_create_update_cache_info(controlnet=new_net,
                                                   estimated_size=estimated_memory_usage)

        return new_net

    @staticmethod
    def parse(uri: _types.Uri) -> 'TorchControlNetUri':
        """
        Parse a ``--model-type`` torch* ``--control-nets`` uri specification and return an object representing its constituents

        :param uri: string with ``--control-nets`` uri syntax

        :raise InvalidControlNetUriError:

        :return: :py:class:`.TorchControlNetPath`
        """
        try:
            r = _torch_control_net_uri_parser.parse(uri)

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

        except _textprocessing.ConceptUriParseError as e:
            raise InvalidControlNetUriError(e)


class SDXLRefinerUri:
    """
    Representation of ``--sdxl-refiner`` uri
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

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 variant: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None,
                 dtype: _enums.DataType | str | None = None):
        """
        :param model: model path
        :param revision: model revision (branch name)
        :param variant: model variant, for example ``fp16``
        :param subfolder: model subfolder
        :param dtype: model data type (precision)

        :raises InvalidSDXLRefinerUriError: If ``dtype`` is passed an invalid data type string.
        """

        self._model = model
        self._revision = revision
        self._variant = variant

        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise InvalidSDXLRefinerUriError(
                f'invalid dtype string, must be one of: {_textprocessing.oxford_comma(_enums.supported_data_type_strings(), "or")}')

        self._subfolder = subfolder

    @staticmethod
    def parse(uri: _types.Uri) -> 'SDXLRefinerUri':
        """
        Parse an ``--sdxl-refiner`` uri and return an object representing its constituents

        :param uri: string with ``--sdxl-refiner`` uri syntax

        :raise InvalidSDXLRefinerUriError:

        :return: :py:class:`.SDXLRefinerUri`
        """
        try:
            r = _sdxl_refiner_uri_parser.parse(uri)

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
        except _textprocessing.ConceptUriParseError as e:
            raise InvalidSDXLRefinerUriError(e)


class SCascadeDecoderUri:
    """
    Representation of ``--s-cascade-decoder`` uri
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

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 variant: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None,
                 dtype: _enums.DataType | str | None = None):
        """
        :param model: model path
        :param revision: model revision (branch name)
        :param variant: model variant, for example ``fp16``
        :param subfolder: model subfolder
        :param dtype: model data type (precision)
        """

        self._model = model
        self._revision = revision
        self._variant = variant
        self._subfolder = subfolder

        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise InvalidSCascadeDecoderUriError(
                f'invalid dtype string, must be one of: '
                f'{_textprocessing.oxford_comma(_enums.supported_data_type_strings(), "or")}')

    @staticmethod
    def parse(uri: _types.Uri) -> 'SCascadeDecoderUri':
        """
        Parse an ``--s-cascade-decoder`` uri and return an object representing its constituents

        :param uri: string with ``--s-cascade-decoder`` uri syntax

        :return: :py:class:`.SDCascadeDecoderUri`
        """
        try:
            r = _s_cascade_decoder_uri_parser.parse(uri)

            supported_dtypes = _enums.supported_data_type_strings()

            dtype = r.args.get('dtype', None)
            if dtype is not None and dtype not in supported_dtypes:
                raise InvalidSCascadeDecoderUriError(
                    f'Torch Stable Cascade "dtype" must be {", ".join(supported_dtypes)}, '
                    f'or left undefined, received: {dtype}')

            return SCascadeDecoderUri(
                model=r.concept,
                revision=r.args.get('revision', None),
                variant=r.args.get('variant', None),
                dtype=dtype,
                subfolder=r.args.get('subfolder', None))
        except _textprocessing.ConceptUriParseError as e:
            raise InvalidSCascadeDecoderUriError(e)


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
            raise InvalidVaeUriError(
                f'Unknown VAE encoder class {encoder}, must be one of: {_textprocessing.oxford_comma(self._encoders.keys(), "or")}')

        can_single_file_load = hasattr(self._encoders[encoder], 'from_single_file')
        single_file_load_path = _hfutil.is_single_file_model_load(model)

        if single_file_load_path and not can_single_file_load:
            raise InvalidVaeUriError(f'{encoder} is not capable of loading from a single file, '
                                     f'must be loaded from a huggingface repository slug or folder on disk.')

        if single_file_load_path:
            if subfolder is not None:
                raise InvalidVaeUriError('Single file VAE loads do not support the subfolder option.')

        self._encoder = encoder
        self._model = model
        self._revision = revision
        self._variant = variant
        self._subfolder = subfolder

        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise InvalidVaeUriError(
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
            raise VAEUriLoadError(
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
        except _textprocessing.ConceptUriParseError as e:
            raise InvalidVaeUriError(e)


class TorchUNetUri:
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

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 variant: _types.OptionalString = None,
                 subfolder: _types.OptionalString = None,
                 dtype: _enums.DataType | str | None = None):
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

        if _hfutil.is_single_file_model_load(model):
            raise InvalidUNetUriError(
                'Loading a UNet from a single file is not supported.')

        self._model = model
        self._revision = revision
        self._variant = variant

        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise InvalidUNetUriError(
                f'invalid dtype string, must be one of: {_textprocessing.oxford_comma(_enums.supported_data_type_strings(), "or")}')

        self._subfolder = subfolder

    def load(self,
             variant_fallback: _types.OptionalString = None,
             dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
             use_auth_token: _types.OptionalString = None,
             local_files_only: bool = False,
             sequential_cpu_offload_member: bool = False,
             model_cpu_offload_member: bool = False,
             unet_class=diffusers.UNet2DConditionModel):
        """
        Load a UNet of type :py:class:`diffusers.UNet2DConditionModel`

        :param variant_fallback: If the URI does not specify a variant, use this variant.
        :param dtype_fallback: If the URI does not specify a dtype, use this dtype.
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
            return self._load(variant_fallback,
                              dtype_fallback,
                              use_auth_token,
                              local_files_only,
                              sequential_cpu_offload_member,
                              model_cpu_offload_member,
                              unet_class)
        except (huggingface_hub.utils.HFValidationError,
                huggingface_hub.utils.HfHubHTTPError) as e:
            raise _hfutil.ModelNotFoundError(e)
        except Exception as e:
            raise UNetUriLoadError(
                f'error loading unet "{self.model}": {e}')

    @_memoize(_cache._TORCH_UNET_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _d_memoize.struct_hasher}),
              on_hit=lambda key, hit: _d_memoize.simple_cache_hit_debug("Torch UNet", key, hit),
              on_create=lambda key, new: _d_memoize.simple_cache_miss_debug("Torch UNet", key, new))
    def _load(self,
              variant_fallback: _types.OptionalString = None,
              dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
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

        path = self.model

        estimated_memory_use = _hfutil.estimate_model_memory_use(
            repo_id=path,
            revision=self.revision,
            variant=variant,
            subfolder=self.subfolder,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token
        )

        _cache.enforce_unet_cache_constraints(new_unet_size=estimated_memory_use)

        unet = unet_class.from_pretrained(
            path,
            revision=self.revision,
            variant=variant,
            torch_dtype=torch_dtype,
            subfolder=self.subfolder,
            token=use_auth_token,
            local_files_only=local_files_only)

        _messages.debug_log('Estimated Torch UNet Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_use))

        _cache.unet_create_update_cache_info(unet=unet,
                                             estimated_size=estimated_memory_use)

        return unet

    @staticmethod
    def parse(uri: _types.Uri) -> 'TorchUNetUri':
        """
        Parse a ``--model-type`` torch* ``--unet`` uri and return an object representing its constituents

        :param uri: string with ``--unet`` uri syntax

        :raise InvalidUNetUriError:

        :return: :py:class:`.TorchUNetPath`
        """
        try:
            r = _torch_unet_uri_parser.parse(uri)

            dtype = r.args.get('dtype')

            supported_dtypes = _enums.supported_data_type_strings()
            if dtype is not None and dtype not in supported_dtypes:
                raise InvalidUNetUriError(
                    f'Torch UNet "dtype" must be {", ".join(supported_dtypes)}, '
                    f'or left undefined, received: {dtype}')

            return TorchUNetUri(
                model=r.concept,
                revision=r.args.get('revision', None),
                variant=r.args.get('variant', None),
                dtype=dtype,
                subfolder=r.args.get('subfolder', None))
        except _textprocessing.ConceptUriParseError as e:
            raise InvalidUNetUriError(e)


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
            raise InvalidVaeUriError(
                f'Unknown VAE flax encoder class {encoder}, '
                f'must be one of: {_textprocessing.oxford_comma(self._encoders.keys(), "or")}')

        can_single_file_load = hasattr(self._encoders[encoder], 'from_single_file')
        single_file_load_path = _hfutil.is_single_file_model_load(model)

        if single_file_load_path and not can_single_file_load:
            raise InvalidVaeUriError(
                f'{encoder} is not capable of loading from a single file, '
                f'must be loaded from a huggingface repository slug or folder on disk.')

        if single_file_load_path:
            # in the future this will be supported?
            if subfolder is not None:
                raise InvalidVaeUriError('Single file VAE loads do not support the subfolder option.')

        self._encoder = encoder
        self._model = model
        self._revision = revision

        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise InvalidVaeUriError(
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
            raise VAEUriLoadError(
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
        except _textprocessing.ConceptUriParseError as e:
            raise InvalidVaeUriError(e)


class FlaxUNetUri:
    """
    Representation of ``--unet`` uri when ``--model-type`` flax*
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

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString,
                 subfolder: _types.OptionalPath,
                 dtype: _enums.DataType | None):
        """
        :param model: model path
        :param revision: model revision (branch name)
        :param subfolder: model subfolder
        :param dtype: model data type (precision)

        :raises InvalidUNetUriError: If ``model`` points to a single file, single file loads are not supported, or
            if ``dtype`` is passed an invalid string.
        """

        single_file_load_path = _hfutil.is_single_file_model_load(model)

        if single_file_load_path:
            raise InvalidUNetUriError('Loading a UNet from a single file is not supported.')

        self._model = model
        self._revision = revision

        try:
            self._dtype = _enums.get_data_type_enum(dtype) if dtype else None
        except ValueError:
            raise InvalidUNetUriError(
                f'invalid dtype string, must be one of: {_textprocessing.oxford_comma(_enums.supported_data_type_strings(), "or")}')

        self._subfolder = subfolder

    def load(self,
             dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
             use_auth_token: _types.OptionalString = None,
             local_files_only: bool = False) -> tuple[diffusers.FlaxUNet2DConditionModel, typing.Any]:
        """
        Load a :py:class:`diffusers.FlaxUNet2DConditionModel` UNet and its flax_params from this URI

        :param dtype_fallback: If the URI does not specify a dtype, use this dtype.
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug or blob link

        :raises ModelNotFoundError: If the model could not be found.

        :return: tuple (:py:class:`diffusers.FlaxUNet2DConditionModel`, flax_unet_params)
        """
        try:
            return self._load(dtype_fallback, use_auth_token, local_files_only)
        except (huggingface_hub.utils.HFValidationError,
                huggingface_hub.utils.HfHubHTTPError) as e:
            raise _hfutil.ModelNotFoundError(e)
        except Exception as e:
            raise UNetUriLoadError(
                f'error loading unet "{self.model}": {e}')

    @_memoize(_cache._FLAX_UNET_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _d_memoize.struct_hasher}),
              on_hit=lambda key, hit: _d_memoize.simple_cache_hit_debug("Flax UNet", key, hit[0]),
              on_create=lambda key, new: _d_memoize.simple_cache_miss_debug("Flax UNet", key, new[0]))
    def _load(self,
              dtype_fallback: _enums.DataType = _enums.DataType.AUTO,
              use_auth_token: _types.OptionalString = None,
              local_files_only: bool = False) -> tuple[diffusers.FlaxUNet2DConditionModel, typing.Any]:

        if self.dtype is None:
            flax_dtype = _enums.get_flax_dtype(dtype_fallback)
        else:
            flax_dtype = _enums.get_flax_dtype(self.dtype)

        estimated_memory_use = _hfutil.estimate_model_memory_use(
            repo_id=self.model,
            revision=self.revision,
            subfolder=self.subfolder,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            flax=True
        )

        _cache.enforce_unet_cache_constraints(new_unet_size=estimated_memory_use)

        unet = diffusers.FlaxUNet2DConditionModel.from_pretrained(
            self.model,
            revision=self.revision,
            dtype=flax_dtype,
            subfolder=self.subfolder,
            token=use_auth_token,
            local_files_only=local_files_only)

        _messages.debug_log('Estimated Flax UNet Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_use))

        _cache.unet_create_update_cache_info(unet=unet[0],
                                             estimated_size=estimated_memory_use)

        return unet

    @staticmethod
    def parse(uri: _types.Uri) -> 'FlaxUNetUri':
        """
        Parse a ``--model-type`` flax* ``--unet`` uri and return an object representing its constituents

        :param uri: string with ``--unet`` uri syntax

        :raise InvalidUNetUriError:

        :return: :py:class:`.FlaxUNetUri`
        """
        try:
            r = _flax_unet_uri_parser.parse(uri)

            dtype = r.args.get('dtype')

            supported_dtypes = _enums.supported_data_type_strings()
            if dtype is not None and dtype not in supported_dtypes:
                raise InvalidUNetUriError(
                    f'Flax UNet "dtype" must be {", ".join(supported_dtypes)}, '
                    f'or left undefined, received: {dtype}')

            return FlaxUNetUri(
                model=r.concept,
                revision=r.args.get('revision', None),
                dtype=_enums.get_flax_dtype(dtype),
                subfolder=r.args.get('subfolder', None))
        except _textprocessing.ConceptUriParseError as e:
            raise InvalidUNetUriError(e)


class LoRAUri:
    """
    Representation of a ``--loras`` uri
    """

    @property
    def model(self) -> str:
        """
        Model path, huggingface slug, file path
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
    def weight_name(self) -> _types.OptionalName:
        """
        Model weight-name
        """
        return self._weight_name

    @property
    def scale(self) -> float:
        """
        LoRA scale
        """
        return self._scale

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None,
                 weight_name: _types.OptionalName = None,
                 scale: float = 1.0):
        self._model = model
        self._scale = scale
        self._revision = revision
        self._subfolder = subfolder
        self._weight_name = weight_name

    def __str__(self):
        return f'{self.__class__.__name__}({str(_types.get_public_attributes(self))})'

    def __repr__(self):
        return str(self)

    def load_on_pipeline(self,
                         pipeline: diffusers.DiffusionPipeline,
                         use_auth_token: _types.OptionalString = None,
                         local_files_only: bool = False):
        """
        Load LoRA weights on to a pipeline using this URI

        :param pipeline: :py:class:`diffusers.DiffusionPipeline`
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug

        :raises ModelNotFoundError: If the model could not be found.
        """
        try:
            self._load_on_pipeline(pipeline=pipeline,
                                   use_auth_token=use_auth_token,
                                   local_files_only=local_files_only)
        except (huggingface_hub.utils.HFValidationError,
                huggingface_hub.utils.HfHubHTTPError) as e:
            raise _hfutil.ModelNotFoundError(e)
        except Exception as e:
            raise LoRAUriLoadError(
                f'error loading lora "{self.model}": {e}')

    def _load_on_pipeline(self,
                          pipeline: diffusers.DiffusionPipeline,
                          use_auth_token: _types.OptionalString = None,
                          local_files_only: bool = False):

        if hasattr(pipeline, 'load_lora_weights'):
            debug_args = {k: v for k, v in locals().items() if k not in {'self', 'pipeline'}}
            _messages.debug_log('pipeline.load_lora_weights('
                                + str(_types.get_public_attributes(self) | debug_args) + ')')

            model_path = _hfutil.download_non_hf_model(self.model)

            if local_files_only and not os.path.exists(model_path):
                # Temporary fix for diffusers bug

                subfolder = self.subfolder if self.subfolder else ''

                probable_path_1 = os.path.join(
                    subfolder, 'pytorch_lora_weights.safetensors' if
                    self.weight_name is None else self.weight_name)

                probable_path_2 = os.path.join(
                    subfolder, 'pytorch_lora_weights.bin')

                file_path = huggingface_hub.try_to_load_from_cache(self.model,
                                                                   filename=probable_path_1,
                                                                   revision=self.revision)

                if not isinstance(file_path, str):
                    file_path = huggingface_hub.try_to_load_from_cache(self.model,
                                                                       filename=probable_path_2,
                                                                       revision=self.revision)

                if not isinstance(file_path, str):
                    raise RuntimeError(
                        f'LoRA model "{self.model}" '
                        'was not available in the local huggingface cache.')

                model_path = os.path.dirname(file_path)

            pipeline.load_lora_weights(model_path,
                                       revision=self.revision,
                                       subfolder=self.subfolder,
                                       weight_name=self.weight_name,
                                       local_files_only=local_files_only,
                                       token=use_auth_token)

            pipeline.fuse_lora(lora_scale=self.scale)

            _messages.debug_log(f'Added LoRA: "{self}" to pipeline: "{pipeline.__class__.__name__}"')
        else:
            raise RuntimeError(f'Pipeline: {pipeline.__class__.__name__} '
                               f'does not support loading LoRAs.')

    @staticmethod
    def parse(uri: _types.Uri) -> 'LoRAUri':
        """
        Parse a ``--loras`` uri and return an object representing its constituents

        :param uri: string with ``--loras`` uri syntax

        :raise InvalidLoRAUriError:

        :return: :py:class:`.LoRAPath`
        """
        try:
            r = _lora_uri_parser.parse(uri)

            return LoRAUri(model=r.concept,
                           scale=float(r.args.get('scale', 1.0)),
                           weight_name=r.args.get('weight-name', None),
                           revision=r.args.get('revision', None),
                           subfolder=r.args.get('subfolder', None))
        except _textprocessing.ConceptUriParseError as e:
            raise InvalidLoRAUriError(e)


def _load_textual_inversion_state_dict(pretrained_model_name_or_path, **kwargs):
    from diffusers.utils.hub_utils import _get_model_file

    text_inversion_name = "learned_embeds.bin"
    text_inversion_name_safe = "learned_embeds.safetensors"

    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", None)
    token = kwargs.pop("token", None)
    revision = kwargs.pop("revision", None)
    subfolder = kwargs.pop("subfolder", None)
    weight_name = kwargs.pop("weight_name", None)
    use_safetensors = kwargs.pop("use_safetensors", None)

    allow_pickle = False
    if use_safetensors is None:
        use_safetensors = True
        allow_pickle = True

    user_agent = {
        "file_type": "text_inversion",
        "framework": "pytorch",
    }

    # 3.1. Load textual inversion file
    state_dict = None
    model_file = None

    # Let's first try to load .safetensors weights
    if (use_safetensors and weight_name is None) or (
            weight_name is not None and weight_name.endswith(".safetensors")
    ):
        try:
            model_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=weight_name or text_inversion_name_safe,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            state_dict = safetensors.torch.load_file(model_file, device="cpu")
        except Exception as e:
            if not allow_pickle:
                raise e

            model_file = None

    if model_file is None:
        model_file = _get_model_file(
            pretrained_model_name_or_path,
            weights_name=weight_name or text_inversion_name,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
        )
        state_dict = torch.load(model_file, map_location="cpu")
    return model_file, state_dict


class TextualInversionUri:
    """
    Representation of ``--textual-inversions`` uri
    """

    @property
    def model(self) -> str:
        """
        Model path, huggingface slug, file path
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
    def weight_name(self) -> _types.OptionalName:
        """
        Model weight-name
        """
        return self._weight_name

    @property
    def token(self) -> _types.OptionalString:
        """
        Prompt keyword
        """
        return self._token

    def __init__(self,
                 model: str,
                 token: str | None = None,
                 revision: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None,
                 weight_name: _types.OptionalName = None):
        self._token = token
        self._model = model
        self._revision = revision
        self._subfolder = subfolder
        self._weight_name = weight_name

    def __str__(self):
        return f'{self.__class__.__name__}({str(_types.get_public_attributes(self))})'

    def __repr__(self):
        return str(self)

    def load_on_pipeline(self,
                         pipeline: diffusers.DiffusionPipeline,
                         use_auth_token: _types.OptionalString = None,
                         local_files_only: bool = False):
        """
        Load Textual Inversion weights on to a pipeline using this URI

        :param pipeline: :py:class:`diffusers.DiffusionPipeline`
        :param use_auth_token: optional huggingface auth token.
        :param local_files_only: avoid downloading files and only look for cached files
            when the model path is a huggingface slug

        :raises ModelNotFoundError: If the model could not be found.
        """
        try:
            self._load_on_pipeline(pipeline=pipeline,
                                   use_auth_token=use_auth_token,
                                   local_files_only=local_files_only)
        except (huggingface_hub.utils.HFValidationError,
                huggingface_hub.utils.HfHubHTTPError) as e:
            raise _hfutil.ModelNotFoundError(e)
        except Exception as e:
            raise TextualInversionUriLoadError(
                f'error loading textual inversion "{self.model}": {e}')

    def _load_on_pipeline(self,
                          pipeline: diffusers.DiffusionPipeline,
                          use_auth_token: _types.OptionalString = None,
                          local_files_only: bool = False):

        if hasattr(pipeline, 'load_textual_inversion'):
            debug_args = {k: v for k, v in locals().items() if k not in {'self', 'pipeline'}}

            _messages.debug_log('pipeline.load_textual_inversion(' +
                                str(_types.get_public_attributes(self) | debug_args) + ')')

            model_path = _hfutil.download_non_hf_model(self.model)

            # this is tricky because there is stupidly a positional argument named 'token'
            # as well as an accepted kwargs value with the key 'token'

            old_token = os.environ.get('HF_TOKEN', None)
            if use_auth_token is not None:
                os.environ['HF_TOKEN'] = use_auth_token

            try:
                if pipeline.__class__.__name__.startswith('StableDiffusionXL'):
                    filename, dicts = _load_textual_inversion_state_dict(
                        model_path,
                        revision=self.revision,
                        subfolder=self.subfolder,
                        weight_name=self.weight_name,
                        local_files_only=local_files_only
                    )

                    if 'clip_l' not in dicts or 'clip_g' not in dicts:
                        raise RuntimeError(
                            'clip_l or clip_g not found in SDXL textual '
                            f'inversion model "{self.model}" state dict, '
                            'unsupported model format.')

                    # token is the file name (no extension) with spaces
                    # replaced by underscores when the user does not provide
                    # a prompt token
                    token = os.path.splitext(
                        os.path.basename(filename))[0].replace(' ', '_') \
                        if self.token is None else self.token

                    pipeline.load_textual_inversion(dicts['clip_l'],
                                                    token=token,
                                                    text_encoder=pipeline.text_encoder,
                                                    tokenizer=pipeline.tokenizer)

                    pipeline.load_textual_inversion(dicts['clip_g'],
                                                    token=token,
                                                    text_encoder=pipeline.text_encoder_2,
                                                    tokenizer=pipeline.tokenizer_2)
                else:
                    pipeline.load_textual_inversion(model_path,
                                                    token=self.token,
                                                    revision=self.revision,
                                                    subfolder=self.subfolder,
                                                    weight_name=self.weight_name,
                                                    local_files_only=local_files_only)
            finally:
                if old_token is not None:
                    os.environ['HF_TOKEN'] = old_token

            _messages.debug_log(f'Added Textual Inversion: "{self}" to pipeline: "{pipeline.__class__.__name__}"')
        else:
            raise RuntimeError(f'Pipeline: {pipeline.__class__.__name__} '
                               f'does not support loading textual inversions.')

    @staticmethod
    def parse(uri: _types.Uri) -> 'TextualInversionUri':
        """
        Parse a ``--textual-inversions`` uri and return an object representing its constituents

        :param uri: string with ``--textual-inversions`` uri syntax

        :raise InvalidTextualInversionUriError:

        :return: :py:class:`.TextualInversionPath`
        """
        try:
            r = _textual_inversion_uri_parser.parse(uri)

            return TextualInversionUri(model=r.concept,
                                       token=r.args.get('token', None),
                                       weight_name=r.args.get('weight-name', None),
                                       revision=r.args.get('revision', None),
                                       subfolder=r.args.get('subfolder', None))
        except _textprocessing.ConceptUriParseError as e:
            raise InvalidTextualInversionUriError(e)


__all__ = _types.module_all()
