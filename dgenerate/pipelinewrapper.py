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
import decimal
import inspect
import os
import re
import textwrap
import typing

try:
    import jax
    import jaxlib
    import jax.numpy as jnp
    from flax.jax_utils import replicate as _flax_replicate
    from flax.training.common_utils import shard as _flax_shard

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
except ImportError:
    jnp = None
    _flax_replicate = None
    _flax_shard = None
    jax = None
    flax = None

import enum
import torch
import PIL.Image
import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.memoize as _d_memoize
from dgenerate.memoize import memoize as _memoize
import dgenerate.prompt as _prompt
import dgenerate.types as _types
import diffusers

TORCH_MODEL_CACHE = dict()
"""Global in memory cache for torch diffusers pipelines"""

FLAX_MODEL_CACHE = dict()
"""Global in memory cache for flax diffusers pipelines"""

TORCH_CONTROL_NET_CACHE = dict()
"""Global in memory cache for torch ControlNet models"""

FLAX_CONTROL_NET_CACHE = dict()
"""Global in memory cache for flax ControlNet models"""

TORCH_VAE_CACHE = dict()
"""Global in memory cache for torch VAE models"""

FLAX_VAE_CACHE = dict()
"""Global in memory cache for flax VAE models"""

DEFAULT_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 5
DEFAULT_IMAGE_SEED_STRENGTH = 0.8
DEFAULT_IMAGE_GUIDANCE_SCALE = 1.5
DEFAULT_SDXL_HIGH_NOISE_FRACTION = 0.8
DEFAULT_X4_UPSCALER_NOISE_LEVEL = 20
DEFAULT_OUTPUT_WIDTH = 512
DEFAULT_OUTPUT_HEIGHT = 512


class OutOfMemoryError(Exception):
    """
    Raised when a GPU or processing device runs out of memory.
    """

    def __init__(self, message):
        super().__init__(f'Device Out Of Memory: {message}')


class InvalidModelPathError(Exception):
    """
    Thrown on model path syntax or logical usage error
    """
    pass


class InvalidSDXLRefinerUriError(InvalidModelPathError):
    """
    Error in --sdxl-refiner path
    """
    pass


class InvalidVaeUriError(InvalidModelPathError):
    """
    Error in --vae path
    """
    pass


class InvalidControlNetUriError(InvalidModelPathError):
    """
    Error in --control-nets path
    """
    pass


class InvalidLoRAUriError(InvalidModelPathError):
    """
    Error in --lora path
    """
    pass


class InvalidTextualInversionUriError(InvalidModelPathError):
    """
    Error in --textual-inversions path
    """
    pass


class InvalidSchedulerName(Exception):
    """
    Unknown scheduler name used
    """
    pass


class SchedulerHelpException(Exception):
    """
    Not an error, runtime scheduler help was requested, info printed, then this exception raised to get out
    """
    pass


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


def _simple_cache_hit_debug(title: str, cache_key: str, cache_hit: typing.Any):
    _messages.debug_log(f'Cache Hit, Loaded {title}: "{cache_hit.__class__.__name__}",',
                        f'Cache Key: "{cache_key}"')


def _simple_cache_miss_debug(title: str, cache_key: str, new: typing.Any):
    _messages.debug_log(f'Cache Miss, Created {title}: "{new.__class__.__name__}",',
                        f'Cache Key: "{cache_key}"')


def _struct_hasher(obj) -> str:
    return _textprocessing.quote(
        _d_memoize.args_cache_key(_types.get_public_attributes(obj)))


class InvalidDeviceOrdinalException(Exception):
    """
    GPU in device specification (cuda:N) does not exist
    """
    pass


def is_valid_device_string(device, raise_ordinal=True):
    match = re.match(r'^(?:cpu|cuda(?::([0-9]+))?)$', device)
    if match:
        if match.lastindex:
            ordinal = int(match[1])
            valid_ordinal = ordinal < torch.cuda.device_count()
            if raise_ordinal and not valid_ordinal:
                raise InvalidDeviceOrdinalException(f'CUDA device ordinal {ordinal} is invalid, no such device exists.')
            return valid_ordinal
        return True
    return False


class DataTypes(enum.Enum):
    AUTO = 0
    FLOAT16 = 1
    FLOAT32 = 2


def supported_data_type_strings():
    """
    Return a list of supported --dtype strings
    """
    return ['auto', 'float16', 'float32']


def supported_data_type_enums() -> typing.List[DataTypes]:
    """
    Return a list of supported :py:class:`.DataTypes` enum values
    """
    return [get_data_type_enum(i) for i in supported_data_type_strings()]


def get_data_type_enum(id_str: typing.Union[DataTypes, str]) -> DataTypes:
    """
    Convert a --dtype string to its :py:class:`.DataTypes` enum value

    :param id_str: --dtype string
    :return: :py:class:`.DataTypes`
    """

    if isinstance(id_str, DataTypes):
        return id_str

    return {'auto': DataTypes.AUTO,
            'float16': DataTypes.FLOAT16,
            'float32': DataTypes.FLOAT32}[id_str.strip().lower()]


def get_data_type_string(data_type_enum: DataTypes) -> str:
    """
    Convert a :py:class:`.DataTypes` enum value to its --dtype string

    :param data_type_enum: :py:class:`.DataTypes` value
    :return: --dtype string
    """

    model_type = get_data_type_enum(data_type_enum)

    return {DataTypes.AUTO: 'auto',
            DataTypes.FLOAT16: 'float16',
            DataTypes.FLOAT32: 'float32'}[model_type]


class ModelTypes(enum.Enum):
    """
    Enum representation of --model-type
    """
    TORCH = 0
    TORCH_PIX2PIX = 1
    TORCH_SDXL = 2
    TORCH_SDXL_PIX2PIX = 3
    TORCH_UPSCALER_X2 = 4
    TORCH_UPSCALER_X4 = 5
    FLAX = 6


def supported_model_type_strings():
    """
    Return a list of supported --model-type strings
    """
    base_set = ['torch', 'torch-pix2pix', 'torch-sdxl', 'torch-sdxl-pix2pix', 'torch-upscaler-x2', 'torch-upscaler-x4']
    if have_jax_flax():
        return base_set + ['flax']
    else:
        return base_set


def supported_model_type_enums() -> typing.List[ModelTypes]:
    """
    Return a list of supported :py:class:`.ModelTypes` enum values
    """
    return [get_model_type_enum(i) for i in supported_model_type_strings()]


def get_model_type_enum(id_str: typing.Union[ModelTypes, str]) -> ModelTypes:
    """
    Convert a --model-type string to its :py:class:`.ModelTypes` enum value

    :param id_str: --model-type string
    :return: :py:class:`.ModelTypes`
    """

    if isinstance(id_str, ModelTypes):
        return id_str

    return {'torch': ModelTypes.TORCH,
            'torch-pix2pix': ModelTypes.TORCH_PIX2PIX,
            'torch-sdxl': ModelTypes.TORCH_SDXL,
            'torch-sdxl-pix2pix': ModelTypes.TORCH_SDXL_PIX2PIX,
            'torch-upscaler-x2': ModelTypes.TORCH_UPSCALER_X2,
            'torch-upscaler-x4': ModelTypes.TORCH_UPSCALER_X4,
            'flax': ModelTypes.FLAX}[id_str.strip().lower()]


def get_model_type_string(model_type_enum: ModelTypes) -> str:
    """
    Convert a :py:class:`.ModelTypes` enum value to its --model-type string

    :param model_type_enum: :py:class:`.ModelTypes` value
    :return: --model-type string
    """

    model_type = get_model_type_enum(model_type_enum)

    return {ModelTypes.TORCH: 'torch',
            ModelTypes.TORCH_PIX2PIX: 'torch-pix2pix',
            ModelTypes.TORCH_SDXL: 'torch-sdxl',
            ModelTypes.TORCH_SDXL_PIX2PIX: 'torch-sdxl-pix2pix',
            ModelTypes.TORCH_UPSCALER_X2: 'torch-upscaler-x2',
            ModelTypes.TORCH_UPSCALER_X4: 'torch-upscaler-x4',
            ModelTypes.FLAX: 'flax'}[model_type]


def model_type_is_upscaler(model_type: typing.Union[ModelTypes, str]) -> bool:
    """
    Does a --model-type string or :py:class:`.ModelTypes` enum value represent an upscaler model?

    :param model_type: --model-type string or :py:class:`.ModelTypes` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 'upscaler' in model_type


def model_type_is_sdxl(model_type: typing.Union[ModelTypes, str]) -> bool:
    """
    Does a --model-type string or :py:class:`.ModelTypes` enum value represent an SDXL model?

    :param model_type: --model-type string or :py:class:`.ModelTypes` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 'sdxl' in model_type


def model_type_is_torch(model_type: typing.Union[ModelTypes, str]) -> bool:
    """
    Does a --model-type string or :py:class:`.ModelTypes` enum value represent an Torch model?

    :param model_type: --model-type string or :py:class:`.ModelTypes` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 'torch' in model_type


def model_type_is_flax(model_type: typing.Union[ModelTypes, str]) -> bool:
    """
    Does a --model-type string or :py:class:`.ModelTypes` enum value represent an Flax model?

    :param model_type: --model-type string or :py:class:`.ModelTypes` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 'flax' in model_type


def model_type_is_pix2pix(model_type: typing.Union[ModelTypes, str]) -> bool:
    """
    Does a --model-type string or :py:class:`.ModelTypes` enum value represent an pix2pix type model?

    :param model_type: --model-type string or :py:class:`.ModelTypes` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 'pix2pix' in model_type


def have_jax_flax():
    """
    Do we have jax/flax support?

    :return: bool
    """
    return jax is not None


def _get_flax_dtype(dtype):
    if dtype is None:
        return None

    if isinstance(dtype, jnp.dtype):
        return dtype

    if isinstance(dtype, DataTypes):
        dtype = get_data_type_string(dtype)

    return {'float16': jnp.bfloat16,
            'float32': jnp.float32,
            'float64': jnp.float64,
            'auto': None}[dtype.lower()]


def _get_torch_dtype(dtype: typing.Union[DataTypes, torch.dtype, str, None]) -> typing.Union[torch.dtype, None]:
    if dtype is None:
        return None

    if isinstance(dtype, torch.dtype):
        return dtype

    if isinstance(dtype, DataTypes):
        dtype = get_data_type_string(dtype)

    return {'float16': torch.float16,
            'float32': torch.float32,
            'float64': torch.float64,
            'auto': None}[dtype.lower()]


class FlaxControlNetPath:
    """
    Representation of --control-nets path when --model-type flax*
    """

    def __init__(self, model, scale, revision, subfolder, dtype, from_torch):
        self.model = model
        self.revision = revision
        self.subfolder = subfolder
        self.dtype = dtype
        self.from_torch = from_torch
        self.scale = scale

    @_memoize(FLAX_CONTROL_NET_CACHE,
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _struct_hasher}),
              on_hit=lambda key, hit: _simple_cache_hit_debug("Flax ControlNet", key, hit),
              on_create=lambda key, new: _simple_cache_miss_debug("Flax ControlNet", key, new))
    def load(self, flax_dtype_fallback, **kwargs) -> diffusers.FlaxControlNetModel:
        single_file_load_path = _is_single_file_model_load(self.model)

        if single_file_load_path:
            raise NotImplementedError('Flax --control-nets do not support single file loads from disk.')
        else:
            new_net: diffusers.FlaxControlNetModel = \
                diffusers.FlaxControlNetModel.from_pretrained(self.model,
                                                              revision=self.revision,
                                                              subfolder=self.subfolder,
                                                              dtype=flax_dtype_fallback if self.dtype is None else self.dtype,
                                                              from_pt=self.from_torch,
                                                              **kwargs)
        return new_net


def parse_flax_control_net_uri(uri: _types.Uri) -> FlaxControlNetPath:
    """
    Parse a --model-type flax* --control-nets uri specification and return an object representing its constituents

    :param uri: string with --control-nets uri syntax

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
                from_torch = bool(from_torch)
            except ValueError:
                raise InvalidControlNetUriError(
                    f'Flax Control Net from_torch must be undefined or boolean (true or false), received: {from_torch}')

        supported_dtypes = supported_data_type_strings()
        if dtype is not None and dtype not in supported_dtypes:
            raise InvalidControlNetUriError(
                f'Flax ControlNet "dtype" must be {", ".join(supported_dtypes)}, '
                f'or left undefined, received: {dtype}')

        try:
            scale = float(scale)
        except ValueError:
            raise InvalidControlNetUriError(
                f'Flax Control Net scale must be a floating point number, received {scale}')

        return FlaxControlNetPath(
            model=r.concept,
            revision=r.args.get('revision', None),
            subfolder=r.args.get('subfolder', None),
            scale=scale,
            dtype=_get_flax_dtype(dtype),
            from_torch=from_torch)

    except _textprocessing.ConceptPathParseError as e:
        raise InvalidControlNetUriError(e)


class TorchControlNetPath:
    """
    Representation of --control-nets path when --model-type torch*
    """

    def __init__(self, model, scale, start, end, revision, variant, subfolder, dtype):
        self.model = model
        self.revision = revision
        self.variant = variant
        self.subfolder = subfolder
        self.dtype = dtype
        self.scale = scale
        self.start = start
        self.end = end

    @_memoize(TORCH_CONTROL_NET_CACHE,
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _struct_hasher}),
              on_hit=lambda key, hit: _simple_cache_hit_debug("Torch ControlNet", key, hit),
              on_create=lambda key, new: _simple_cache_miss_debug("Torch ControlNet", key, new))
    def load(self, torch_dtype_fallback, **kwargs) -> diffusers.ControlNetModel:

        single_file_load_path = _is_single_file_model_load(self.model)

        if single_file_load_path:
            new_net: diffusers.ControlNetModel = \
                diffusers.ControlNetModel.from_single_file(self.model,
                                                           revision=self.revision,
                                                           torch_dtype=torch_dtype_fallback if self.dtype is None else self.dtype,
                                                           **kwargs)
        else:
            new_net: diffusers.ControlNetModel = \
                diffusers.ControlNetModel.from_pretrained(self.model,
                                                          revision=self.revision,
                                                          variant=self.variant,
                                                          subfolder=self.subfolder,
                                                          torch_dtype=torch_dtype_fallback if self.dtype is None else self.dtype,
                                                          **kwargs)
        return new_net


def parse_torch_control_net_uri(uri: _types.Uri) -> TorchControlNetPath:
    """
    Parse a --model-type torch* --control-nets uri specification and return an object representing its constituents

    :param path: string with --control-nets uri syntax

    :raise: :py:class:`.InvalidControlNetUriError`

    :return: :py:class:`.TorchControlNetPath`
    """
    try:
        r = _torch_control_net_uri_parser.parse_concept_uri(uri)

        dtype = r.args.get('dtype')
        scale = r.args.get('scale', 1.0)
        start = r.args.get('start', 0.0)
        end = r.args.get('end', 1.0)

        supported_dtypes = supported_data_type_strings()
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

        return TorchControlNetPath(
            model=r.concept,
            revision=r.args.get('revision', None),
            variant=r.args.get('variant', None),
            subfolder=r.args.get('subfolder', None),
            dtype=_get_torch_dtype(dtype),
            scale=scale,
            start=start,
            end=end)

    except _textprocessing.ConceptPathParseError as e:
        raise InvalidControlNetUriError(e)


class SDXLRefinerPath:
    """
    Representation of --sdxl-refiner path
    """

    def __init__(self, model, revision, variant, dtype, subfolder):
        self.model = model
        self.revision = revision
        self.variant = variant
        self.dtype = dtype
        self.subfolder = subfolder


def parse_sdxl_refiner_uri(uri: _types.Uri) -> SDXLRefinerPath:
    """
    Parse an --sdxl-refiner uri and return an object representing its constituents

    :param path: string with --sdxl-refiner uri syntax

    :raise: :py:class:`.InvalidSDXLRefinerUriError`

    :return: :py:class:`.SDXLRefinerPath`
    """
    try:
        r = _sdxl_refiner_uri_parser.parse_concept_uri(uri)

        supported_dtypes = supported_data_type_strings()

        dtype = r.args.get('dtype', None)
        if dtype is not None and dtype not in supported_dtypes:
            raise InvalidSDXLRefinerUriError(
                f'Torch SDXL refiner "dtype" must be {", ".join(supported_dtypes)}, '
                f'or left undefined, received: {dtype}')

        return SDXLRefinerPath(
            model=r.concept,
            revision=r.args.get('revision', None),
            variant=r.args.get('variant', None),
            dtype=_get_torch_dtype(dtype),
            subfolder=r.args.get('subfolder', None))
    except _textprocessing.ConceptPathParseError as e:
        raise InvalidSDXLRefinerUriError(e)


class TorchVAEPath:
    """
    Representation of --vae path when --model-type torch*
    """

    def __init__(self, encoder, model, revision, variant, subfolder, dtype):
        self.encoder = encoder
        self.model = model
        self.revision = revision
        self.variant = variant
        self.dtype = dtype
        self.subfolder = subfolder


def parse_torch_vae_uri(uri: _types.Uri) -> TorchVAEPath:
    """
    Parse a --model-type torch* --vae uri and return an object representing its constituents

    :param path: string with --vae uri syntax

    :raise: :py:class:`.InvalidVaeUriError`

    :return: :py:class:`.TorchVAEPath`
    """
    try:
        r = _torch_vae_uri_parser.parse_concept_uri(uri)

        model = r.args.get('model')
        if model is None:
            raise InvalidVaeUriError('model argument for torch VAE specification must be defined.')

        dtype = r.args.get('dtype')

        supported_dtypes = supported_data_type_strings()
        if dtype is not None and dtype not in supported_dtypes:
            raise InvalidVaeUriError(
                f'Torch VAE "dtype" must be {", ".join(supported_dtypes)}, '
                f'or left undefined, received: {dtype}')

        return TorchVAEPath(encoder=r.concept,
                            model=model,
                            revision=r.args.get('revision', None),
                            variant=r.args.get('variant', None),
                            dtype=_get_torch_dtype(dtype),
                            subfolder=r.args.get('subfolder', None))
    except _textprocessing.ConceptPathParseError as e:
        raise InvalidVaeUriError(e)


class FlaxVAEPath:
    """
    Representation of --vae path when --model-type flax*
    """

    def __init__(self, encoder, model, revision, dtype, subfolder):
        self.encoder = encoder
        self.model = model
        self.revision = revision
        self.dtype = dtype
        self.subfolder = subfolder


def parse_flax_vae_uri(uri: _types.Uri) -> FlaxVAEPath:
    """
    Parse a --model-type flax* --vae uri and return an object representing its constituents

    :param path: string with --vae uri syntax

    :raise: :py:class:`.InvalidVaeUriError`

    :return: :py:class:`.FlaxVAEPath`
    """
    try:
        r = _flax_vae_uri_parser.parse_concept_uri(uri)

        model = r.args.get('model')
        if model is None:
            raise InvalidVaeUriError('model argument for flax VAE specification must be defined.')

        dtype = r.args.get('dtype')

        supported_dtypes = supported_data_type_strings()
        if dtype is not None and dtype not in supported_dtypes:
            raise InvalidVaeUriError(
                f'Flax VAE "dtype" must be {", ".join(supported_dtypes)}, '
                f'or left undefined, received: {dtype}')

        return FlaxVAEPath(encoder=r.concept,
                           model=model,
                           revision=r.args.get('revision', None),
                           dtype=_get_flax_dtype(dtype),
                           subfolder=r.args.get('subfolder', None))
    except _textprocessing.ConceptPathParseError as e:
        raise InvalidVaeUriError(e)


class LoRAPath:
    """
    Representation of --lora path
    """

    def __init__(self, model, scale, revision, subfolder, weight_name):
        self.model = model
        self.scale = scale
        self.revision = revision
        self.subfolder = subfolder
        self.weight_name = weight_name

    def __str__(self):
        return f'{self.__class__.__name__}({str(_types.get_public_attributes(self))})'

    def __repr__(self):
        return str(self)

    def load_on_pipeline(self, pipeline, **kwargs):
        if hasattr(pipeline, 'load_lora_weights'):
            _messages.debug_log(f'Added LoRA: "{self}" to pipeline: "{pipeline.__class__.__name__}"')
            pipeline.load_lora_weights(self.model,
                                       revision=self.revision,
                                       subfolder=self.subfolder,
                                       weight_name=self.weight_name,
                                       **kwargs)


def parse_lora_uri(uri: _types.Uri) -> LoRAPath:
    """
    Parse a --lora uri and return an object representing its constituents

    :param path: string with --lora uri syntax

    :raise: :py:class:`.InvalidLoRAUriError`

    :return: :py:class:`.LoRAPath`
    """
    try:
        r = _lora_uri_parser.parse_concept_uri(uri)

        return LoRAPath(model=r.concept,
                        scale=float(r.args.get('scale', 1.0)),
                        weight_name=r.args.get('weight-name', None),
                        revision=r.args.get('revision', None),
                        subfolder=r.args.get('subfolder', None))
    except _textprocessing.ConceptPathParseError as e:
        raise InvalidLoRAUriError(e)


class TextualInversionPath:
    """
    Representation of --textual-inversions path
    """

    def __init__(self, model, revision, subfolder, weight_name):
        self.model = model
        self.revision = revision
        self.subfolder = subfolder
        self.weight_name = weight_name

    def __str__(self):
        return f'{self.__class__.__name__}({str(_types.get_public_attributes(self))})'

    def __repr__(self):
        return str(self)

    def load_on_pipeline(self, pipeline, **kwargs):
        if hasattr(pipeline, 'load_textual_inversion'):
            _messages.debug_log(f'Added Textual Inversion: "{self}" to pipeline: "{pipeline.__class__.__name__}"')

            pipeline.load_textual_inversion(self.model,
                                            revision=self.revision,
                                            subfolder=self.subfolder,
                                            weight_name=self.weight_name,
                                            **kwargs)


def parse_textual_inversion_uri(uri: _types.Uri) -> TextualInversionPath:
    """
    Parse a --textual-inversions uri and return an object representing its constituents

    :param path: string with --textual-inversions uri syntax

    :raise: :py:class:`.InvalidTextualInversionUriError`

    :return: :py:class:`.TextualInversionPath`
    """
    try:
        r = _textual_inversion_uri_parser.parse_concept_uri(uri)

        return TextualInversionPath(model=r.concept,
                                    weight_name=r.args.get('weight-name', None),
                                    revision=r.args.get('revision', None),
                                    subfolder=r.args.get('subfolder', None))
    except _textprocessing.ConceptPathParseError as e:
        raise InvalidTextualInversionUriError(e)


def _is_single_file_model_load(path):
    path, ext = os.path.splitext(path)

    if path.startswith('http://') or path.startswith('https://'):
        return True

    if os.path.isdir(path):
        return True

    if not ext:
        return False

    if ext in {'.pt', '.pth', '.bin', '.msgpack', '.ckpt', '.safetensors'}:
        return True

    return False


def _uri_hash_with_parser(parser):
    def hasher(path):
        if not path:
            return path

        return _struct_hasher(parser(path))

    return hasher


@_memoize(TORCH_VAE_CACHE,
          hasher=lambda args: _d_memoize.args_cache_key(args,
                                                        {'uri': _uri_hash_with_parser(parse_torch_vae_uri)}),
          on_hit=lambda key, hit: _simple_cache_hit_debug("Torch VAE", key, hit),
          on_create=lambda key, new: _simple_cache_miss_debug("Torch VAE", key, new))
def _load_torch_vae(uri: _types.Uri,
                    torch_dtype_fallback: torch.dtype,
                    use_auth_token: bool) -> typing.Union[
    diffusers.AutoencoderKL, diffusers.AsymmetricAutoencoderKL, diffusers.AutoencoderTiny]:
    parsed_concept = parse_torch_vae_uri(uri)

    if parsed_concept.dtype is None:
        parsed_concept.dtype = torch_dtype_fallback

    encoder_name = parsed_concept.encoder

    if encoder_name == 'AutoencoderKL':
        encoder = diffusers.AutoencoderKL
    elif encoder_name == 'AsymmetricAutoencoderKL':
        encoder = diffusers.AsymmetricAutoencoderKL
    elif encoder_name == 'AutoencoderTiny':
        encoder = diffusers.AutoencoderTiny
    else:
        raise InvalidVaeUriError(f'Unknown VAE encoder class {encoder_name}')

    path = parsed_concept.model

    can_single_file_load = hasattr(encoder, 'from_single_file')
    single_file_load_path = _is_single_file_model_load(path)

    if single_file_load_path and not can_single_file_load:
        raise NotImplementedError(f'{encoder_name} is not capable of loading from a single file, '
                                  f'must be loaded from a huggingface repository slug or folder on disk.')

    if single_file_load_path:
        if parsed_concept.subfolder is not None:
            raise NotImplementedError('Single file VAE loads do not support the subfolder option.')

        if encoder is diffusers.AutoencoderKL:
            # There is a bug in their cast
            vae = encoder.from_single_file(path, revision=parsed_concept.revision). \
                to(dtype=parsed_concept.dtype, non_blocking=False)
        else:
            vae = encoder.from_single_file(path,
                                           revision=parsed_concept.revision,
                                           torch_dtype=parsed_concept.dtype)

    else:
        vae = encoder.from_pretrained(path,
                                      revision=parsed_concept.revision,
                                      variant=parsed_concept.variant,
                                      torch_dtype=parsed_concept.dtype,
                                      subfolder=parsed_concept.subfolder,
                                      use_auth_token=use_auth_token)
    return vae


@_memoize(FLAX_VAE_CACHE,
          hasher=lambda args: _d_memoize.args_cache_key(args,
                                                        {'uri': _uri_hash_with_parser(parse_flax_vae_uri)}),
          on_hit=lambda key, hit: _simple_cache_hit_debug("Flax VAE", key, hit),
          on_create=lambda key, new: _simple_cache_miss_debug("Flax VAE", key, new))
def _load_flax_vae(uri: _types.Uri,
                   flax_dtype_fallback,
                   use_auth_token: bool):
    parsed_concept = parse_flax_vae_uri(uri)

    if parsed_concept.dtype is None:
        parsed_concept.dtype = flax_dtype_fallback

    encoder_name = parsed_concept.encoder

    if encoder_name == 'FlaxAutoencoderKL':
        encoder = diffusers.FlaxAutoencoderKL
    else:
        raise InvalidVaeUriError(f'Unknown VAE flax encoder class {encoder_name}')

    path = parsed_concept.model

    can_single_file_load = hasattr(encoder, 'from_single_file')
    single_file_load_path = _is_single_file_model_load(path)

    if single_file_load_path and not can_single_file_load:
        raise NotImplementedError(f'{encoder_name} is not capable of loading from a single file, '
                                  f'must be loaded from a huggingface repository slug or folder on disk.')

    if single_file_load_path:
        # in the future this will be supported?
        if parsed_concept.subfolder is not None:
            raise NotImplementedError('Single file VAE loads do not support the subfolder option.')
        vae = encoder.from_single_file(path,
                                       revision=parsed_concept.revision,
                                       dtype=parsed_concept.dtype)
    else:
        vae = encoder.from_pretrained(path,
                                      revision=parsed_concept.revision,
                                      dtype=parsed_concept.dtype,
                                      subfolder=parsed_concept.subfolder,
                                      use_auth_token=use_auth_token)
    return vae


def _load_scheduler(pipeline, model_path, scheduler_name=None):
    if scheduler_name is None:
        return

    compatibles = pipeline.scheduler.compatibles

    if isinstance(pipeline, diffusers.StableDiffusionLatentUpscalePipeline):
        # Seems to only work with this scheduler
        compatibles = [c for c in compatibles if c.__name__ == 'EulerDiscreteScheduler']

    if _scheduler_is_help(scheduler_name):
        help_string = _textprocessing.underline(f'Compatible schedulers for "{model_path}" are:') + '\n\n'
        help_string += '\n'.join((" " * 4) + _textprocessing.quote(i.__name__) for i in compatibles) + '\n'
        _messages.log(help_string, underline=True)
        raise SchedulerHelpException(help_string)

    for i in compatibles:
        if i.__name__.endswith(scheduler_name):
            pipeline.scheduler = i.from_config(pipeline.scheduler.config)
            return

    raise InvalidSchedulerName(
        f'Scheduler named "{scheduler_name}" is not a valid compatible scheduler, '
        f'options are:\n\n{chr(10).join(sorted(" " * 4 + _textprocessing.quote(i.__name__.split(".")[-1]) for i in compatibles))}')


def clear_model_cache():
    """
    Clear all in memory model caches.

        * TORCH_MODEL_CACHE
        * FLAX_MODEL_CACHE
        * TORCH_CONTROL_NET_CACHE
        * FLAX_CONTROL_NET_CACHE
        * TORCH_VAE_CACHE
        * FLAX_VAE_CACHE

    """
    TORCH_MODEL_CACHE.clear()
    TORCH_CONTROL_NET_CACHE.clear()
    FLAX_CONTROL_NET_CACHE.clear()
    TORCH_VAE_CACHE.clear()
    FLAX_VAE_CACHE.clear()
    FLAX_MODEL_CACHE.clear()


def _disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False] * num_images
    else:
        return images, False


class _PipelineTypes:
    BASIC = 1
    IMG2IMG = 2
    INPAINT = 3


def _describe_pipeline_type(type_enum):
    return {_PipelineTypes.BASIC: 'txt2img', _PipelineTypes.IMG2IMG: 'img2img', _PipelineTypes.INPAINT: 'inpaint'}[
        type_enum]


def _args_except(args, *exceptions):
    return {k: v for k, v in args.items() if k not in exceptions}


def _scheduler_is_help(name):
    if name is None:
        return False
    return name.strip().lower() == 'help'


def _set_vae_slicing_tiling(pipeline, vae_tiling, vae_slicing):
    has_vae = hasattr(pipeline, 'vae') and pipeline.vae is not None
    pipeline_class = pipeline.__class__

    if vae_tiling:
        if has_vae:
            if hasattr(pipeline.vae, 'enable_tiling'):
                _messages.debug_log(f'Enabling VAE tiling on Pipeline: "{pipeline_class.__name__}",',
                                    f'VAE: "{pipeline.vae.__class__.__name__}"')
                pipeline.vae.enable_tiling()
            else:
                raise NotImplementedError(
                    '--vae-tiling not supported as loaded VAE does not support it.'
                )
        else:
            raise NotImplementedError(
                '--vae-tiling not supported as no VAE is present for the specified model.')
    elif has_vae:
        if hasattr(pipeline.vae, 'disable_tiling'):
            _messages.debug_log(f'Disabling VAE tiling on Pipeline: "{pipeline_class.__name__}",',
                                f'VAE: "{pipeline.vae.__class__.__name__}"')
            pipeline.vae.disable_tiling()

    if vae_slicing:
        if has_vae:
            if hasattr(pipeline.vae, 'enable_slicing'):
                _messages.debug_log(f'Enabling VAE slicing on Pipeline: "{pipeline_class.__name__}",',
                                    f'VAE: "{pipeline.vae.__class__.__name__}"')
                pipeline.vae.enable_slicing()
            else:
                raise NotImplementedError(
                    '--vae-slicing not supported as loaded VAE does not support it.'
                )
        else:
            raise NotImplementedError(
                '--vae-slicing not supported as no VAE is present for the specified model.')
    elif has_vae:
        if hasattr(pipeline.vae, 'disable_slicing'):
            _messages.debug_log(f'Disabling VAE slicing on Pipeline: "{pipeline_class.__name__}",',
                                f'VAE: "{pipeline.vae.__class__.__name__}"')
            pipeline.vae.disable_slicing()


def _set_torch_safety_checker(pipeline, safety_checker_bool):
    if not safety_checker_bool:
        if hasattr(pipeline, 'safety_checker') and pipeline.safety_checker is not None:
            # If it's already None for some reason you'll get a call
            # to an unassigned feature_extractor by assigning it a value

            # The attribute will not exist for SDXL pipelines currently

            pipeline.safety_checker = _disabled_safety_checker


def _uri_list_hash_with_parser(parser):
    def hasher(paths):
        if not paths:
            return paths

        if isinstance(paths, str):
            return _uri_hash_with_parser(parser)(paths)

        return '[' + ','.join(_uri_hash_with_parser(parser)(path) for path in paths) + ']'

    return hasher


@_memoize(TORCH_MODEL_CACHE,
          hasher=lambda args: _d_memoize.args_cache_key(args,
                                                        {'vae_uri': _uri_hash_with_parser(parse_torch_vae_uri),
                                                         'lora_uris':
                                                             _uri_list_hash_with_parser(parse_lora_uri),
                                                         'textual_inversion_uris':
                                                             _uri_list_hash_with_parser(
                                                                 parse_textual_inversion_uri),
                                                         'control_net_uris':
                                                             _uri_list_hash_with_parser(
                                                                 parse_torch_control_net_uri)}),
          on_hit=lambda key, hit: _simple_cache_hit_debug("Torch Pipeline", key, hit[0]),
          on_create=lambda key, new: _simple_cache_miss_debug('Torch Pipeline', key, new[0]))
def _create_torch_diffusion_pipeline(pipeline_type,
                                     model_type,
                                     model_path,
                                     revision,
                                     variant,
                                     dtype,
                                     model_subfolder=None,
                                     vae_uri=None,
                                     lora_uris=None,
                                     textual_inversion_uris=None,
                                     control_net_uris=None,
                                     scheduler=None,
                                     safety_checker=False,
                                     auth_token=None,
                                     device='cuda',
                                     extra_args=None,
                                     model_cpu_offload=False,
                                     sequential_cpu_offload=False):
    # Pipeline class selection

    if model_type_is_upscaler(model_type):
        if pipeline_type != _PipelineTypes.IMG2IMG and not _scheduler_is_help(scheduler):
            raise NotImplementedError(
                'Upscaler models only work with img2img generation, IE: --image-seeds (with no image masks).')

        if model_type == ModelTypes.TORCH_UPSCALER_X2:
            if lora_uris or textual_inversion_uris:
                raise NotImplementedError(
                    '--model-type torch-upscaler-x2 is not compatible with --lora or --textual-inversions.')

        pipeline_class = (diffusers.StableDiffusionUpscalePipeline if model_type == ModelTypes.TORCH_UPSCALER_X4
                          else diffusers.StableDiffusionLatentUpscalePipeline)
    else:
        sdxl = model_type_is_sdxl(model_type)
        pix2pix = model_type_is_pix2pix(model_type)

        if pipeline_type == _PipelineTypes.BASIC:
            if pix2pix:
                raise NotImplementedError(
                    'pix2pix models only work in img2img mode and cannot work without --image-seeds.')

            if control_net_uris:
                pipeline_class = diffusers.StableDiffusionXLControlNetPipeline if sdxl else diffusers.StableDiffusionControlNetPipeline
            else:
                pipeline_class = diffusers.StableDiffusionXLPipeline if sdxl else diffusers.StableDiffusionPipeline
        elif pipeline_type == _PipelineTypes.IMG2IMG:

            if pix2pix:
                if control_net_uris:
                    raise NotImplementedError('pix2pix models are not compatible with --control-nets.')

                pipeline_class = diffusers.StableDiffusionXLInstructPix2PixPipeline if sdxl else diffusers.StableDiffusionInstructPix2PixPipeline
            else:
                if control_net_uris:
                    if sdxl:
                        pipeline_class = diffusers.StableDiffusionXLControlNetImg2ImgPipeline
                    else:
                        pipeline_class = diffusers.StableDiffusionControlNetImg2ImgPipeline
                else:
                    pipeline_class = diffusers.StableDiffusionXLImg2ImgPipeline if sdxl else diffusers.StableDiffusionImg2ImgPipeline

        elif pipeline_type == _PipelineTypes.INPAINT:
            if pix2pix:
                raise NotImplementedError(
                    'pix2pix models only work in img2img mode and cannot work in inpaint mode (with a mask).')

            if control_net_uris:
                if sdxl:
                    pipeline_class = diffusers.StableDiffusionXLControlNetInpaintPipeline
                else:
                    pipeline_class = diffusers.StableDiffusionControlNetInpaintPipeline
            else:
                pipeline_class = diffusers.StableDiffusionXLInpaintPipeline if sdxl else diffusers.StableDiffusionInpaintPipeline
        else:
            # Should be impossible
            raise NotImplementedError('Pipeline type not implemented.')

    _messages.debug_log(f'Creating Torch Pipeline: "{pipeline_class.__name__}"')

    # Block invalid Textual Inversion and LoRA usage

    if textual_inversion_uris:
        if model_type == ModelTypes.TORCH_UPSCALER_X2:
            raise NotImplementedError(
                'Model type torch-upscaler-x2 cannot be used with textual inversion models.')

        if isinstance(textual_inversion_uris, str):
            textual_inversion_uris = [textual_inversion_uris]

    if lora_uris is not None:
        if not isinstance(lora_uris, str):
            raise NotImplementedError('Using multiple LoRA models is currently not supported.')

        if model_type_is_upscaler(model_type):
            raise NotImplementedError(
                'LoRA models cannot be used with upscaler models.')

        lora_uris = [lora_uris]

    # ControlNet and VAE loading

    # Used during pipeline load
    creation_kwargs = {}

    torch_dtype = _get_torch_dtype(dtype)

    parsed_control_net_uris = []

    if scheduler is None or not _scheduler_is_help(scheduler):
        # prevent waiting on VAE load just to get the scheduler
        # help message for the main model

        if vae_uri is not None:
            creation_kwargs['vae'] = _load_torch_vae(vae_uri,
                                                     torch_dtype_fallback=torch_dtype,
                                                     use_auth_token=auth_token)
            _messages.debug_log(lambda:
                                f'Added Torch VAE: "{vae_uri}" to pipeline: "{pipeline_class.__name__}"')

    if control_net_uris:
        if model_type_is_pix2pix(model_type):
            raise NotImplementedError(
                'Using ControlNets with pix2pix models is not supported.'
            )

        control_nets = None

        for control_net_uri in control_net_uris:
            parsed_control_net_uri = parse_torch_control_net_uri(control_net_uri)

            parsed_control_net_uris.append(parsed_control_net_uri)

            new_net = parsed_control_net_uri.load(use_auth_token=auth_token,
                                                  torch_dtype_fallback=torch_dtype)

            _messages.debug_log(lambda:
                                f'Added Torch ControlNet: "{control_net_uri}" '
                                f'to pipeline: "{pipeline_class.__name__}"')

            if control_nets is not None:
                if not isinstance(control_nets, list):
                    control_nets = [control_nets, new_net]
                else:
                    control_nets.append(new_net)
            else:
                control_nets = new_net

        creation_kwargs['controlnet'] = control_nets

    if extra_args is not None:
        creation_kwargs.update(extra_args)

    # Create Pipeline

    if _is_single_file_model_load(model_path):
        if model_subfolder is not None:
            raise NotImplementedError('Single file model loads do not support the subfolder option.')
        pipeline = pipeline_class.from_single_file(model_path,
                                                   revision=revision,
                                                   variant=variant,
                                                   torch_dtype=torch_dtype,
                                                   use_safe_tensors=model_path.endswith('.safetensors'),
                                                   **creation_kwargs)
    else:
        pipeline = pipeline_class.from_pretrained(model_path,
                                                  revision=revision,
                                                  variant=variant,
                                                  torch_dtype=torch_dtype,
                                                  subfolder=model_subfolder,
                                                  use_auth_token=auth_token,
                                                  **creation_kwargs)

    # Select Scheduler

    _load_scheduler(pipeline=pipeline,
                    model_path=model_path,
                    scheduler_name=scheduler)

    # Textual Inversions and LoRAs

    if textual_inversion_uris is not None:
        for inversion_uri in textual_inversion_uris:
            parse_textual_inversion_uri(inversion_uri). \
                load_on_pipeline(pipeline, use_auth_token=auth_token)

    if lora_uris is not None:
        for lora_uri in lora_uris:
            parse_lora_uri(lora_uri). \
                load_on_pipeline(pipeline, use_auth_token=auth_token)

    # Safety Checker

    _set_torch_safety_checker(pipeline, safety_checker)

    # Model Offloading

    pipeline._dgenerate_sequential_offload = sequential_cpu_offload
    pipeline._dgenerate_cpu_offload = model_cpu_offload

    if sequential_cpu_offload and 'cuda' in device:
        pipeline.enable_sequential_cpu_offload(device=device)
    elif model_cpu_offload and 'cuda' in device:
        pipeline.enable_model_cpu_offload(device=device)

    _messages.debug_log(f'Finished Creating Torch Pipeline: "{pipeline_class.__name__}"')
    return pipeline, parsed_control_net_uris


@_memoize(FLAX_MODEL_CACHE,
          hasher=lambda args: _d_memoize.args_cache_key(args,
                                                        {'vae_uri': _uri_hash_with_parser(parse_flax_vae_uri),
                                                         'control_net_uris':
                                                             _uri_list_hash_with_parser(
                                                                 parse_flax_control_net_uri)}),
          on_hit=lambda key, hit: _simple_cache_hit_debug("Flax Pipeline", key, hit[0]),
          on_create=lambda key, new: _simple_cache_miss_debug('Flax Pipeline', key, new[0]))
def _create_flax_diffusion_pipeline(pipeline_type,
                                    model_path,
                                    revision,
                                    dtype,
                                    model_subfolder=None,
                                    vae_uri=None,
                                    control_net_uris=None,
                                    scheduler=None,
                                    safety_checker=False,
                                    auth_token=None,
                                    extra_args=None):
    has_control_nets = False
    if control_net_uris is not None:
        if len(control_net_uris) > 1:
            raise NotImplementedError('Flax does not support multiple --control-nets.')
        if len(control_net_uris) == 1:
            has_control_nets = True

    if pipeline_type == _PipelineTypes.BASIC:
        if has_control_nets:
            pipeline_class = diffusers.FlaxStableDiffusionControlNetPipeline
        else:
            pipeline_class = diffusers.FlaxStableDiffusionPipeline
    elif pipeline_type == _PipelineTypes.IMG2IMG:
        if has_control_nets:
            raise NotImplementedError('Flax does not support img2img mode with --control-nets.')
        pipeline_class = diffusers.FlaxStableDiffusionImg2ImgPipeline
    elif pipeline_type == _PipelineTypes.INPAINT:
        if has_control_nets:
            raise NotImplementedError('Flax does not support inpaint mode with --control-nets.')
        pipeline_class = diffusers.FlaxStableDiffusionInpaintPipeline
    else:
        raise NotImplementedError('Pipeline type not implemented.')

    _messages.debug_log(f'Creating Flax Pipeline: "{pipeline_class.__name__}"')

    kwargs = {}
    vae_params = None
    control_net_params = None

    flax_dtype = _get_flax_dtype(dtype)

    parsed_control_net_uris = []

    if scheduler is None or not _scheduler_is_help(scheduler):
        # prevent waiting on VAE load just get the scheduler
        # help message for the main model

        if vae_uri is not None:
            kwargs['vae'], vae_params = _load_flax_vae(vae_uri,
                                                       flax_dtype_fallback=flax_dtype,
                                                       use_auth_token=auth_token)
            _messages.debug_log(lambda:
                                f'Added Flax VAE: "{vae_uri}" to pipeline: "{pipeline_class.__name__}"')

    if control_net_uris is not None:
        control_net_uri = control_net_uris[0]

        parsed_flax_control_net_uri = parse_flax_control_net_uri(control_net_uri)

        parsed_control_net_uris.append(parsed_flax_control_net_uri)

        control_net, control_net_params = parse_flax_control_net_uri(control_net_uri) \
            .load(use_auth_token=auth_token, flax_dtype_fallback=flax_dtype)

        _messages.debug_log(lambda:
                            f'Added Flax ControlNet: "{control_net_uri}" '
                            f'to pipeline: "{pipeline_class.__name__}"')

        kwargs['controlnet'] = control_net

    if extra_args is not None:
        kwargs.update(extra_args)

    pipeline, params = pipeline_class.from_pretrained(model_path,
                                                      revision=revision,
                                                      dtype=flax_dtype,
                                                      subfolder=model_subfolder,
                                                      use_auth_token=auth_token,
                                                      **kwargs)

    if vae_params is not None:
        params['vae'] = vae_params

    if control_net_params is not None:
        params['controlnet'] = control_net_params

    _load_scheduler(pipeline=pipeline,
                    model_path=model_path,
                    scheduler_name=scheduler)

    if not safety_checker:
        pipeline.safety_checker = None

    _messages.debug_log(f'Finished Creating Flax Pipeline: "{pipeline_class.__name__}"')
    return pipeline, params, parsed_control_net_uris


class PipelineWrapperResult:
    """
    The result of calling :py:class:`.DiffusionPipelineWrapper`
    """
    images: typing.Optional[typing.List[PIL.Image.Image]]
    dgenerate_opts: typing.List[typing.Tuple[str, typing.Any]]

    @property
    def image_count(self):
        """
        The number of images produced.

        :return: int
        """
        return len(self.images)

    @property
    def image(self):
        """
        The first image in the batch of requested batch size.

        :return: :py:class:`PIL.Image.Image`
        """
        return self.images[0] if self.images else None

    def image_grid(self, cols_rows: _types.Size):
        """
        Render an image grid from the images in this result.

        :param cols_rows: columns and rows (WxH) desired as a tuple
        :return: :py:class:`PIL.Image.Image`
        """
        if not self.images:
            raise ValueError('No images present.')

        if len(self.images) == 1:
            return self.images[0]

        cols, rows = cols_rows

        w, h = self.images[0].size
        grid = PIL.Image.new('RGB', size=(cols * w, rows * h))
        for i, img in enumerate(self.images):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

    def __init__(self, images: typing.Optional[typing.List[PIL.Image.Image]]):
        self.images = images
        self.dgenerate_opts = dict()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.images is not None:
            for i in self.images:
                if i is not None:
                    i.close()
                    self.images = None

    def add_dgenerate_opt(self, name: str, value: typing.Any):
        """
        Add an option value to be used by :py:meth:`.PipelineWrapperResult.gen_dgenerate_config`

        :param name: The option name
        :param value: The option value
        """
        self.dgenerate_opts.append((name, value))



    @staticmethod
    def _set_opt_value_syntax(val):
        if isinstance(val, tuple):
            return _textprocessing.format_size(val)
        if isinstance(val, list):
            return ' '.join(_textprocessing.quote_spaces(v) for v in val)
        return val

    @staticmethod
    def _format_option_pair(val):
        if len(val) > 1:
            return f'{val[0]} {PipelineWrapperResult._set_opt_value_syntax(val[1])}'
        return val[0]

    def gen_dgenerate_config(self,
                             extra_args: typing.Optional[typing.Sequence[typing.Tuple[str, typing.Any]]] = None,
                             extra_comments: typing.Optional[typing.Sequence[str]] = None):
        """
        Generate a valid dgenerate config file with a single invocation that reproduces this result.

        :param extra_comments: Extra strings to use as comments after the
            version check directive
        :param extra_args: Extra invocation arguments to add to the config file.
        :return: The configuration as a string
        """

        from .__init__ import __version__

        config = f'#! dgenerate {__version__}\n\n'

        if extra_comments:
            wrote_comments = False
            for comment in extra_comments:
                wrote_comments = True
                for part in comment.split('\n'):
                    config += '# ' + part.rstrip()

            if wrote_comments:
                config += '\n\n'

        opts = _textprocessing.quote_spaces(
            self.dgenerate_opts + (extra_args if extra_args else []))

        for opt in opts[:-1]:
            config += f'{self._format_option_pair(opt)} \\\n'

        last = opts[-1]

        if len(last) == 2:
            config += self._format_option_pair(last)

        return config

    def gen_dgenerate_command(self,
                              extra_args: typing.Optional[typing.Sequence[typing.Tuple[str, typing.Any]]] = None):
        """
        Generate a valid dgenerate command line invocation that reproduces this result.

        :param extra_args: Extra arguments to add to the end of the command line.
        :return: A string containing the dgenerate command line needed to reproduce this result.
        """
        opts = _textprocessing.quote_spaces(self.dgenerate_opts)

        return f'dgenerate {" ".join(f"{self._format_option_pair(opt)}" for opt in opts)}'


class DiffusionArguments:
    """
    Represents all possible arguments for a :py:class:`.DiffusionPipelineWrapper` call.
    """
    prompt: _types.OptionalPrompt = None
    sdxl_second_prompt: _types.OptionalPrompt = None
    sdxl_refiner_prompt: _types.OptionalPrompt = None
    sdxl_refiner_second_prompt: _types.OptionalPrompt = None
    seed: _types.OptionalInteger = None
    image_seed_strength: _types.OptionalFloat = None
    upscaler_noise_level: _types.OptionalInteger = None
    sdxl_high_noise_fraction: _types.OptionalFloat = None
    sdxl_refiner_inference_steps: _types.OptionalInteger = None
    sdxl_refiner_guidance_scale: _types.OptionalFloat = None
    sdxl_refiner_guidance_rescale: _types.OptionalFloat = None
    sdxl_aesthetic_score: _types.OptionalFloat = None
    sdxl_original_size: _types.OptionalSize = None
    sdxl_target_size: _types.OptionalSize = None
    sdxl_crops_coords_top_left: _types.OptionalCoordinate = None
    sdxl_negative_aesthetic_score: _types.OptionalFloat = None
    sdxl_negative_original_size: _types.OptionalSize = None
    sdxl_negative_target_size: _types.OptionalSize = None
    sdxl_negative_crops_coords_top_left: _types.OptionalCoordinate = None
    sdxl_refiner_aesthetic_score: _types.OptionalFloat = None
    sdxl_refiner_original_size: _types.OptionalSize = None
    sdxl_refiner_target_size: _types.OptionalSize = None
    sdxl_refiner_crops_coords_top_left: _types.OptionalCoordinate = None
    sdxl_refiner_negative_aesthetic_score: _types.OptionalFloat = None
    sdxl_refiner_negative_original_size: _types.OptionalSize = None
    sdxl_refiner_negative_target_size: _types.OptionalSize = None
    sdxl_refiner_negative_crops_coords_top_left: _types.OptionalCoordinate = None
    guidance_scale: _types.OptionalFloat = None
    image_guidance_scale: _types.OptionalFloat = None
    guidance_rescale: _types.OptionalFloat = None
    inference_steps: _types.OptionalInteger = None

    def get_pipeline_wrapper_args(self):
        """
        Get the arguments dictionary needed to call :py:class:`.DiffusionPipelineWrapper`

        :return: dictionary of argument names with values
        """
        pipeline_args = {}
        for attr, hint in typing.get_type_hints(self).items():
            val = getattr(self, attr)
            if not attr.startswith('_') and not (callable(val) or val is None):
                pipeline_args[attr] = val
        return pipeline_args

    @staticmethod
    def _describe_prompt(prompt_format, prompt: _prompt.Prompt, pos_title, neg_title):
        if prompt is None:
            return

        prompt_wrap_width = _textprocessing.long_text_wrap_width()
        prompt_val = prompt.positive
        if prompt_val:
            header = f'{pos_title}: '
            prompt_val = textwrap.fill(prompt_val,
                                       width=prompt_wrap_width - len(header),
                                       break_long_words=False,
                                       break_on_hyphens=False,
                                       subsequent_indent=' ' * len(header))
            prompt_format.append(f'{header}"{prompt_val}"')

        prompt_val = prompt.negative
        if prompt_val:
            header = f'{neg_title}: '
            prompt_val = textwrap.fill(prompt_val,
                                       width=prompt_wrap_width - len(header),
                                       break_long_words=False,
                                       break_on_hyphens=False,
                                       subsequent_indent=' ' * len(header))
            prompt_format.append(f'{header}"{prompt_val}"')

    def describe_pipeline_wrapper_args(self) -> str:
        """
        Describe the pipeline wrapper arguments in a pretty, human-readable way, with word wrapping
        depending on console size or a maximum length depending on what stdout currently is.

        :return: description string.
        """
        prompt_format = []
        DiffusionArguments._describe_prompt(
            prompt_format, self.prompt,
            "Prompt",
            "Negative Prompt")

        DiffusionArguments._describe_prompt(
            prompt_format, self.sdxl_second_prompt,
            "SDXL Second Prompt",
            "SDXL Second Negative Prompt")

        DiffusionArguments._describe_prompt(
            prompt_format, self.sdxl_refiner_prompt,
            "SDXL Refiner Prompt",
            "SDXL Refiner Negative Prompt")

        DiffusionArguments._describe_prompt(
            prompt_format, self.sdxl_refiner_second_prompt,
            "SDXL Refiner Second Prompt",
            "SDXL Refiner Second Negative Prompt")

        prompt_format = '\n'.join(prompt_format)
        if prompt_format:
            prompt_format = '\n' + prompt_format

        inputs = [f'Seed: {self.seed}']

        descriptions = [
            (self.image_seed_strength, "Image Seed Strength:"),
            (self.upscaler_noise_level, "Upscaler Noise Level:"),
            (self.sdxl_high_noise_fraction, "SDXL High Noise Fraction:"),
            (self.sdxl_refiner_inference_steps, "SDXL Refiner Inference Steps:"),
            (self.sdxl_refiner_guidance_scale, "SDXL Refiner Guidance Scale:"),
            (self.sdxl_refiner_guidance_rescale, "SDXL Refiner Guidance Rescale:"),
            (self.sdxl_aesthetic_score, "SDXL Aesthetic Score:"),
            (self.sdxl_original_size, "SDXL Original Size:"),
            (self.sdxl_target_size, "SDXL Target Size:"),
            (self.sdxl_crops_coords_top_left, "SDXL Top Left Crop Coords:"),
            (self.sdxl_negative_aesthetic_score, "SDXL Negative Aesthetic Score:"),
            (self.sdxl_negative_original_size, "SDXL Negative Original Size:"),
            (self.sdxl_negative_target_size, "SDXL Negative Target Size:"),
            (self.sdxl_negative_crops_coords_top_left, "SDXL Negative Top Left Crop Coords:"),
            (self.sdxl_refiner_aesthetic_score, "SDXL Refiner Aesthetic Score:"),
            (self.sdxl_refiner_original_size, "SDXL Refiner Original Size:"),
            (self.sdxl_refiner_target_size, "SDXL Refiner Target Size:"),
            (self.sdxl_refiner_crops_coords_top_left, "SDXL Refiner Top Left Crop Coords:"),
            (self.sdxl_refiner_negative_aesthetic_score, "SDXL Refiner Negative Aesthetic Score:"),
            (self.sdxl_refiner_negative_original_size, "SDXL Refiner Negative Original Size:"),
            (self.sdxl_refiner_negative_target_size, "SDXL Refiner Negative Target Size:"),
            (self.sdxl_refiner_negative_crops_coords_top_left, "SDXL Refiner Negative Top Left Crop Coords:"),
            (self.guidance_scale, "Guidance Scale:"),
            (self.image_guidance_scale, "Image Guidance Scale:"),
            (self.guidance_rescale, "Guidance Rescale:"),
            (self.inference_steps, "Inference Steps:")
        ]

        for prompt_val, desc in descriptions:
            if prompt_val is not None:
                inputs.append(desc + ' ' + str(prompt_val))

        inputs = '\n'.join(inputs)

        return inputs + prompt_format


class DiffusionPipelineWrapper:
    """
    Monolithic diffusion pipelines wrapper.
    """

    def __str__(self):
        return f'{self.__class__.__name__}({str(_types.get_public_attributes(self))})'

    def __repr__(self):
        return str(self)

    def __init__(self,
                 model_path: _types.Path,
                 dtype: typing.Union[DataTypes, str] = DataTypes.AUTO,
                 device: str = 'cuda',
                 model_type: typing.Union[ModelTypes, str] = ModelTypes.TORCH,
                 revision: _types.OptionalName = None,
                 variant: _types.OptionalName = None,
                 model_subfolder: _types.OptionalName = None,
                 vae_uri: _types.OptionalUri = None,
                 vae_tiling: bool = False,
                 vae_slicing: bool = False,
                 lora_uris: typing.Union[str, _types.OptionalUris] = None,
                 textual_inversion_uris: typing.Union[str, _types.OptionalUris] = None,
                 control_net_uris: typing.Union[str, _types.OptionalUris] = None,
                 sdxl_refiner_uri: _types.OptionalUri = None,
                 scheduler: _types.OptionalName = None,
                 sdxl_refiner_scheduler: _types.OptionalName = None,
                 safety_checker: bool = False,
                 auth_token: _types.OptionalString = None):

        self._model_subfolder = model_subfolder
        self._device = device
        self._model_type = get_model_type_enum(model_type)
        self._model_path = model_path
        self._pipeline = None
        self._flax_params = None
        self._revision = revision
        self._variant = variant
        self._dtype = get_data_type_enum(dtype)
        self._device = device
        self._vae_uri = vae_uri
        self._vae_tiling = vae_tiling
        self._vae_slicing = vae_slicing
        self._safety_checker = safety_checker
        self._scheduler = scheduler
        self._sdxl_refiner_scheduler = sdxl_refiner_scheduler
        self._lora_uris = lora_uris
        self._lora_scale = None
        self._textual_inversion_uris = textual_inversion_uris
        self._control_net_uris = control_net_uris
        self._parsed_control_net_uris = []
        self._sdxl_refiner_uri = sdxl_refiner_uri
        self._sdxl_refiner_pipeline = None
        self._auth_token = auth_token
        self._pipeline_type = None

        if sdxl_refiner_uri is not None:
            parsed_uri = parse_sdxl_refiner_uri(sdxl_refiner_uri)
            self._sdxl_refiner_uri = parsed_uri.model
            self._sdxl_refiner_revision = parsed_uri.revision
            self._sdxl_refiner_variant = parsed_uri.variant
            self._sdxl_refiner_dtype = parsed_uri.dtype
            self._sdxl_refiner_subfolder = parsed_uri.subfolder

        if lora_uris is not None:
            if model_type == 'flax':
                raise NotImplementedError('LoRA loading is not implemented for flax.')

            if not isinstance(lora_uris, str):
                raise NotImplementedError('Using multiple LoRA models is currently not supported.')

            self._lora_scale = parse_lora_uri(lora_uris).scale

    @staticmethod
    def _pipeline_to(pipeline, device):
        if hasattr(pipeline, 'to'):
            if not pipeline._dgenerate_cpu_offload and \
                    not pipeline._dgenerate_sequential_offload:
                return pipeline.to(device)
            else:
                return pipeline
        return pipeline

    _LAST_CALLED_PIPE = None

    @staticmethod
    def _call_pipeline(pipeline, device, **kwargs):
        _messages.debug_log(f'Calling Pipeline: "{pipeline.__class__.__name__}",',
                            f'Device: "{device}",',
                            'Args:',
                            lambda: _textprocessing.debug_format_args(kwargs,
                                                                      value_transformer=lambda key, value:
                                                                      f'torch.Generator(seed={value.initial_seed()})'
                                                                      if isinstance(value, torch.Generator) else value))

        if pipeline is DiffusionPipelineWrapper._LAST_CALLED_PIPE:
            return pipeline(**kwargs)
        else:
            DiffusionPipelineWrapper._pipeline_to(
                DiffusionPipelineWrapper._LAST_CALLED_PIPE, 'cpu')

        DiffusionPipelineWrapper._pipeline_to(pipeline, device)
        r = pipeline(**kwargs)

        DiffusionPipelineWrapper._LAST_CALLED_PIPE = pipeline
        return r

    @property
    def revision(self) -> _types.OptionalName:
        """
        Currently set revision for the main model or None
        """
        return self._revision

    @property
    def safety_checker(self) -> bool:
        """
        Safety checker enabled status
        """
        return self._safety_checker

    @property
    def variant(self) -> _types.OptionalName:
        """
        Currently set variant for the main model or None
        """
        return self._variant

    @property
    def dtype(self) -> DataTypes:
        """
        Currently set dtype for the main model
        """
        return self._dtype

    @property
    def textual_inversion_uris(self) -> _types.OptionalUris:
        """
        List of supplied --textual-inversions path strings or None
        """
        return [self._textual_inversion_uris] if \
            isinstance(self._textual_inversion_uris, str) else self._textual_inversion_uris

    @property
    def control_net_uris(self) -> _types.OptionalUris:
        """
        List of supplied --control-nets path strings or None
        """
        return [self._control_net_uris] if \
            isinstance(self._control_net_uris, str) else self._control_net_uris

    @property
    def device(self) -> _types.Name:
        """
        Currently set --device string
        """
        return self._device

    @property
    def model_path(self) -> _types.Path:
        """
        Model path for the main model
        """
        return self._model_path

    @property
    def scheduler(self) -> _types.OptionalName:
        """
        Selected scheduler name for the main model or None
        """
        return self._scheduler

    @property
    def sdxl_refiner_scheduler(self) -> _types.OptionalName:
        """
        Selected scheduler name for the SDXL refiner or None
        """
        return self._sdxl_refiner_scheduler

    @property
    def sdxl_refiner_uri(self) -> _types.OptionalUri:
        """
        Model path for the SDXL refiner or None
        """
        return self._sdxl_refiner_uri

    @property
    def model_type_enum(self) -> ModelTypes:
        """
        Currently set --model-type enum value
        """
        return self._model_type

    @property
    def model_type_string(self) -> str:
        """
        Currently set --model-type string value
        """
        return get_model_type_string(self._model_type)

    @property
    def dtype_enum(self) -> DataTypes:
        """
        Currently set --dtype enum value
        """
        return self._dtype

    @property
    def dtype_string(self) -> str:
        """
        Currently set --dtype string value
        """
        return get_data_type_string(self._dtype)

    @property
    def model_subfolder(self) -> _types.OptionalName:
        """
        Selected model subfolder for the main model, (remote repo subfolder or local) or None
        """
        return self._model_subfolder

    @property
    def vae_uri(self) -> _types.OptionalUri:
        """
        Selected --vae path for the main model or None
        """
        return self._vae_uri

    @property
    def vae_tiling(self) -> bool:
        """
        Current --vae-tiling status
        """
        return self._vae_tiling

    @property
    def vae_slicing(self) -> bool:
        """
        Current --vae-slicing status
        """
        return self._vae_slicing

    @property
    def lora_uris(self) -> _types.OptionalUris:
        """
        List of supplied --lora path strings or None
        """
        return [self._lora_uris] if \
            isinstance(self._lora_uris, str) else self._lora_uris

    @property
    def auth_token(self) -> _types.OptionalString:
        """
        Current --auth-token value or None
        """
        return self._auth_token

    def reconstruct_dgenerate_opts(self, **args) -> \
            typing.List[typing.Union[typing.Tuple[str], typing.Tuple[str, typing.Any]]]:
        """
        Reconstruct dgenerates command line arguments from a particular set of pipeline call arguments.

        :param args: arguments to :py:class:`.DiffusionArguments`

        :return: List of tuples of length 1 or 2 representing the option
        """

        def _format_size(val):
            if val is None:
                return None

            return f'{val[0]}x{val[1]}'

        batch_size: int = args.get('batch_size', None)
        prompt: _prompt.Prompt = args.get('prompt', None)
        sdxl_second_prompt: _prompt.Prompt = args.get('sdxl_second_prompt', None)
        sdxl_refiner_prompt: _prompt.Prompt = args.get('sdxl_refiner_prompt', None)
        sdxl_refiner_second_prompt: _prompt.Prompt = args.get('sdxl_refiner_second_prompt', None)

        image = args.get('image', None)
        control_image = args.get('control_image', None)
        image_seed_strength = args.get('image_seed_strength', None)
        upscaler_noise_level = args.get('upscaler_noise_level', None)
        mask_image = args.get('mask_image', None)
        seed = args.get('seed')
        width = args.get('width', None)
        height = args.get('height', None)
        inference_steps = args.get('inference_steps')
        guidance_scale = args.get('guidance_scale')
        guidance_rescale = args.get('guidance_rescale')
        image_guidance_scale = args.get('image_guidance_scale')

        sdxl_refiner_inference_steps = args.get('sdxl_refiner_inference_steps')
        sdxl_refiner_guidance_scale = args.get('sdxl_refiner_guidance_scale')
        sdxl_refiner_guidance_rescale = args.get('sdxl_refiner_guidance_rescale')

        sdxl_high_noise_fraction = args.get('sdxl_high_noise_fraction', None)
        sdxl_aesthetic_score = args.get('sdxl_aesthetic_score', None)

        sdxl_original_size = \
            _format_size(args.get('sdxl_original_size', None))
        sdxl_target_size = \
            _format_size(args.get('sdxl_target_size', None))
        sdxl_crops_coords_top_left = \
            _format_size(args.get('sdxl_crops_coords_top_left', None))

        sdxl_negative_aesthetic_score = args.get('sdxl_negative_aesthetic_score', None)

        sdxl_negative_original_size = \
            _format_size(args.get('sdxl_negative_original_size', None))
        sdxl_negative_target_size = \
            _format_size(args.get('sdxl_negative_target_size', None))
        sdxl_negative_crops_coords_top_left = \
            _format_size(args.get('sdxl_negative_crops_coords_top_left', None))

        sdxl_refiner_aesthetic_score = args.get('sdxl_refiner_aesthetic_score', None)

        sdxl_refiner_original_size = \
            _format_size(args.get('sdxl_refiner_original_size', None))
        sdxl_refiner_target_size = \
            _format_size(args.get('sdxl_refiner_target_size', None))
        sdxl_refiner_crops_coords_top_left = \
            _format_size(args.get('sdxl_refiner_crops_coords_top_left', None))

        sdxl_refiner_negative_aesthetic_score = args.get('sdxl_refiner_negative_aesthetic_score', None)

        sdxl_refiner_negative_original_size = \
            _format_size(args.get('sdxl_refiner_negative_original_size', None))
        sdxl_refiner_negative_target_size = \
            _format_size(args.get('sdxl_refiner_negative_target_size', None))
        sdxl_refiner_negative_crops_coords_top_left = \
            _format_size(args.get('sdxl_refiner_negative_crops_coords_top_left', None))

        opts = [(self.model_path,),
                ('--model-type', self.model_type_string),
                ('--dtype', self.dtype_string),
                ('--device', self._device),
                ('--inference-steps', inference_steps),
                ('--guidance-scales', guidance_scale),
                ('--seeds', seed)]

        if batch_size is not None:
            opts.append(('--batch-size', batch_size))

        if guidance_rescale is not None:
            opts.append(('--guidance-rescales', guidance_rescale))

        if image_guidance_scale is not None:
            opts.append(('--image-guidance-scales', image_guidance_scale))

        if prompt is not None:
            opts.append(('--prompts', prompt))

        if sdxl_second_prompt is not None:
            opts.append(('--sdxl-second-prompt', sdxl_second_prompt))

        if sdxl_refiner_prompt is not None:
            opts.append(('--sdxl-refiner-prompt', sdxl_refiner_prompt))

        if sdxl_refiner_second_prompt is not None:
            opts.append(('--sdxl-refiner-second-prompt', sdxl_refiner_second_prompt))

        if self._revision is not None:
            opts.append(('--revision', self._revision))

        if self._variant is not None:
            opts.append(('--variant', self._variant))

        if self._model_subfolder is not None:
            opts.append(('--subfolder', self._model_subfolder))

        if self._vae_uri is not None:
            opts.append(('--vae', self._vae_uri))

        if self._vae_tiling:
            opts.append(('--vae-tiling',))

        if self._vae_slicing:
            opts.append(('--vae-slicing',))

        if self._sdxl_refiner_uri is not None:
            opts.append(('--sdxl-refiner', self._sdxl_refiner_uri))

        if self._lora_uris is not None:
            opts.append(('--lora', self._lora_uris))

        if self._textual_inversion_uris is not None:
            opts.append(('--textual-inversions', self._textual_inversion_uris))

        if self._control_net_uris is not None:
            opts.append(('--control-nets', self._control_net_uris))

        if self._scheduler is not None:
            opts.append(('--scheduler', self._scheduler))

        if self._sdxl_refiner_scheduler is not None:
            if self._sdxl_refiner_scheduler != self._scheduler:
                opts.append(('--sdxl-refiner-scheduler', self._sdxl_refiner_scheduler))

        if sdxl_high_noise_fraction is not None:
            opts.append(('--sdxl-high-noise-fractions', sdxl_high_noise_fraction))

        if sdxl_refiner_inference_steps is not None:
            opts.append(('--sdxl-refiner-inference-steps', sdxl_refiner_inference_steps))

        if sdxl_refiner_guidance_scale is not None:
            opts.append(('--sdxl-refiner-guidance-scales', sdxl_refiner_guidance_scale))

        if sdxl_refiner_guidance_rescale is not None:
            opts.append(('--sdxl-refiner-guidance-rescales', sdxl_refiner_guidance_rescale))

        if sdxl_aesthetic_score is not None:
            opts.append(('--sdxl-aesthetic-scores', sdxl_aesthetic_score))

        if sdxl_original_size is not None:
            opts.append(('--sdxl-original-size', sdxl_original_size))

        if sdxl_target_size is not None:
            opts.append(('--sdxl-target-size', sdxl_target_size))

        if sdxl_crops_coords_top_left is not None:
            opts.append(('--sdxl-crops-coords-top-left', sdxl_crops_coords_top_left))

        if sdxl_negative_aesthetic_score is not None:
            opts.append(('--sdxl-negative-aesthetic-scores', sdxl_negative_aesthetic_score))

        if sdxl_negative_original_size is not None:
            opts.append(('--sdxl-negative-original-sizes', sdxl_negative_original_size))

        if sdxl_negative_target_size is not None:
            opts.append(('--sdxl-negative-target-sizes', sdxl_negative_target_size))

        if sdxl_negative_crops_coords_top_left is not None:
            opts.append(('--sdxl-negative-crops-coords-top-left', sdxl_negative_crops_coords_top_left))

        if sdxl_refiner_aesthetic_score is not None:
            opts.append(('--sdxl-refiner-aesthetic-scores', sdxl_refiner_aesthetic_score))

        if sdxl_refiner_original_size is not None:
            opts.append(('--sdxl-refiner-original-sizes', sdxl_refiner_original_size))

        if sdxl_refiner_target_size is not None:
            opts.append(('--sdxl-refiner-target-sizes', sdxl_refiner_target_size))

        if sdxl_refiner_crops_coords_top_left is not None:
            opts.append(('--sdxl-refiner-crops-coords-top-left', sdxl_refiner_crops_coords_top_left))

        if sdxl_refiner_negative_aesthetic_score is not None:
            opts.append(('--sdxl-refiner-negative-aesthetic-scores', sdxl_refiner_negative_aesthetic_score))

        if sdxl_refiner_negative_original_size is not None:
            opts.append(('--sdxl-refiner-negative-original-sizes', sdxl_refiner_negative_original_size))

        if sdxl_refiner_negative_target_size is not None:
            opts.append(('--sdxl-refiner-negative-target-sizes', sdxl_refiner_negative_target_size))

        if sdxl_refiner_negative_crops_coords_top_left is not None:
            opts.append(('--sdxl-refiner-negative-crops-coords-top-left', sdxl_refiner_negative_crops_coords_top_left))

        if width is not None and height is not None:
            opts.append(('--output-size', f'{width}x{height}'))
        elif width is not None:
            opts.append(('--output-size', f'{width}'))

        if image is not None:
            if hasattr(image, 'filename'):
                seed_args = []

                if mask_image is not None and hasattr(mask_image, 'filename'):
                    seed_args.append(f'mask={mask_image.filename}')
                if control_image is not None and hasattr(control_image, 'filename'):
                    seed_args.append(f'control={control_image.filename}')

                if not seed_args:
                    opts.append(('--image-seeds', image.filename))
                else:
                    opts.append(('--image-seeds',
                                 _textprocessing.quote(image.filename + ';' + ';'.join(seed_args))))

                if image_seed_strength is not None:
                    opts.append(('--image-seed-strengths', image_seed_strength))

                if upscaler_noise_level is not None:
                    opts.append(('--upscaler-noise-levels', upscaler_noise_level))
        elif control_image is not None:
            if hasattr(control_image, 'filename'):
                opts.append(('--image-seeds', control_image.filename))

        return opts

    def _pipeline_defaults(self, user_args):
        args = dict()
        args['guidance_scale'] = float(user_args.get('guidance_scale', DEFAULT_GUIDANCE_SCALE))
        args['num_inference_steps'] = int(user_args.get('inference_steps', DEFAULT_INFERENCE_STEPS))

        def set_strength():
            strength = float(user_args.get('image_seed_strength', DEFAULT_IMAGE_SEED_STRENGTH))
            inference_steps = args.get('num_inference_steps')

            if (strength * inference_steps) < 1.0:
                strength = 1.0 / inference_steps
                _messages.log(
                    f'image-seed-strength * inference-steps '
                    f'was calculated at < 1, image-seed-strength defaulting to (1.0 / inference-steps): {strength}',
                    level=_messages.WARNING)

            args['strength'] = strength

        if self._control_net_uris is not None:
            control_image = user_args['control_image']
            if self._pipeline_type == _PipelineTypes.BASIC:
                args['image'] = control_image
            elif self._pipeline_type == _PipelineTypes.IMG2IMG or \
                    self._pipeline_type == _PipelineTypes.INPAINT:
                args['image'] = user_args['image']
                args['control_image'] = control_image
                set_strength()

            mask_image = user_args.get('mask_image')
            if mask_image is not None:
                args['mask_image'] = mask_image

            args['width'] = user_args.get('width', control_image.width)
            args['height'] = user_args.get('height', control_image.height)

        elif 'image' in user_args:
            image = user_args['image']
            args['image'] = image

            if model_type_is_upscaler(self._model_type):
                if self._model_type == ModelTypes.TORCH_UPSCALER_X4:
                    args['noise_level'] = int(user_args.get('upscaler_noise_level', DEFAULT_X4_UPSCALER_NOISE_LEVEL))
            elif not model_type_is_pix2pix(self._model_type):
                set_strength()

            mask_image = user_args.get('mask_image')
            if mask_image is not None:
                args['mask_image'] = mask_image
                args['width'] = image.size[0]
                args['height'] = image.size[1]

            if self._model_type == ModelTypes.TORCH_SDXL_PIX2PIX:
                # Required
                args['width'] = image.size[0]
                args['height'] = image.size[1]
        else:
            args['height'] = user_args.get('height', DEFAULT_OUTPUT_HEIGHT)
            args['width'] = user_args.get('width', DEFAULT_OUTPUT_WIDTH)

        if self._lora_scale is not None:
            args['cross_attention_kwargs'] = {'scale': self._lora_scale}

        return args

    def _get_control_net_conditioning_scale(self):
        if not self._parsed_control_net_uris:
            return 1.0
        return [p.scale for p in self._parsed_control_net_uris] if \
            len(self._parsed_control_net_uris) > 1 else self._parsed_control_net_uris[0].scale

    def _get_control_net_guidance_start(self):
        if not self._parsed_control_net_uris:
            return 0.0
        return [p.start for p in self._parsed_control_net_uris] if \
            len(self._parsed_control_net_uris) > 1 else self._parsed_control_net_uris[0].start

    def _get_control_net_guidance_end(self):
        if not self._parsed_control_net_uris:
            return 1.0
        return [p.end for p in self._parsed_control_net_uris] if \
            len(self._parsed_control_net_uris) > 1 else self._parsed_control_net_uris[0].end

    def _call_flax_control_net(self, positive_prompt, negative_prompt, default_args, user_args):
        device_count = jax.device_count()

        pipe: diffusers.FlaxStableDiffusionControlNetPipeline = self._pipeline

        default_args['prng_seed'] = jax.random.split(jax.random.PRNGKey(user_args.get('seed', 0)), device_count)
        prompt_ids = pipe.prepare_text_inputs([positive_prompt] * device_count)

        if negative_prompt is not None:
            negative_prompt_ids = pipe.prepare_text_inputs([negative_prompt] * device_count)
        else:
            negative_prompt_ids = None

        processed_image = pipe.prepare_image_inputs([default_args.get('image')] * device_count)
        default_args.pop('image')

        p_params = _flax_replicate(self._flax_params)
        prompt_ids = _flax_shard(prompt_ids)
        negative_prompt_ids = _flax_shard(negative_prompt_ids)
        processed_image = _flax_shard(processed_image)

        default_args.pop('width', None)
        default_args.pop('height', None)

        images = DiffusionPipelineWrapper._call_pipeline(
            pipeline=self._pipeline,
            device=self.device,
            prompt_ids=prompt_ids,
            image=processed_image,
            params=p_params,
            neg_prompt_ids=negative_prompt_ids,
            controlnet_conditioning_scale=self._get_control_net_conditioning_scale(),
            jit=True, **default_args)[0]

        return PipelineWrapperResult(
            self._pipeline.numpy_to_pil(images.reshape((images.shape[0],) + images.shape[-3:])))

    def _flax_prepare_text_input(self, text):
        tokenizer = self._pipeline.tokenizer
        text_input = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        return text_input.input_ids

    def _call_flax(self, default_args, user_args):
        for arg, val in user_args.items():
            if arg.startswith('sdxl') and val is not None:
                raise NotImplementedError(
                    f'{arg.replace("_", "-")}s may only be used with SDXL models.')

        if user_args.get('guidance_rescale') is not None:
            raise NotImplementedError('--guidance-rescales is not supported when using --model-type flax.')

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in
                                                      range(0, user_args.get('batch_size', 1)))

        prompt: _prompt.Prompt() = user_args.get('prompt', _prompt.Prompt())
        positive_prompt = prompt.positive if prompt.positive else ''
        negative_prompt = prompt.negative

        if hasattr(self._pipeline, 'controlnet'):
            return self._call_flax_control_net(positive_prompt, negative_prompt,
                                               default_args, user_args)

        device_count = jax.device_count()

        default_args['prng_seed'] = jax.random.split(jax.random.PRNGKey(user_args.get('seed', 0)), device_count)

        if negative_prompt is not None:
            negative_prompt_ids = _flax_shard(
                self._flax_prepare_text_input([negative_prompt] * device_count))
        else:
            negative_prompt_ids = None

        if 'image' in default_args:
            if 'mask_image' in default_args:

                prompt_ids, processed_images, processed_masks = \
                    self._pipeline.prepare_inputs(prompt=[positive_prompt] * device_count,
                                                  image=[default_args['image']] * device_count,
                                                  mask=[default_args['mask_image']] * device_count)

                default_args['masked_image'] = _flax_shard(processed_images)
                default_args['mask'] = _flax_shard(processed_masks)

                # inpainting pipeline does not have a strength argument, simply ignore it
                default_args.pop('strength')

                default_args.pop('image')
                default_args.pop('mask_image')
            else:
                prompt_ids, processed_images = self._pipeline.prepare_inputs(
                    prompt=[positive_prompt] * device_count,
                    image=[default_args['image']] * device_count)
                default_args['image'] = _flax_shard(processed_images)

            default_args['width'] = processed_images[0].shape[2]
            default_args['height'] = processed_images[0].shape[1]
        else:
            prompt_ids = self._pipeline.prepare_inputs([positive_prompt] * device_count)

        images = DiffusionPipelineWrapper._call_pipeline(
            pipeline=self._pipeline,
            device=self._device,
            prompt_ids=_flax_shard(prompt_ids),
            neg_prompt_ids=negative_prompt_ids,
            params=_flax_replicate(self._flax_params),
            **default_args, jit=True)[0]

        return PipelineWrapperResult(self._pipeline.numpy_to_pil(
            images.reshape((images.shape[0],) + images.shape[-3:])))

    def _get_non_universal_pipeline_arg(self,
                                        pipeline,
                                        default_args,
                                        user_args,
                                        pipeline_arg_name,
                                        user_arg_name,
                                        option_name,
                                        default,
                                        transform=None):
        if pipeline.__call__.__wrapped__ is not None:
            # torch.no_grad()
            func = pipeline.__call__.__wrapped__
        else:
            func = pipeline.__call__

        if pipeline_arg_name in inspect.getfullargspec(func).args:
            if user_arg_name in user_args:
                # Only provide a default if the user
                # provided the option, and it's value was None
                val = user_args.get(user_arg_name, default)
                val = val if not transform else transform(val)
                default_args[pipeline_arg_name] = val
                return val
            return None
        else:
            val = user_args.get(user_arg_name, None)
            if val is not None:
                raise NotImplementedError(
                    f'{option_name} cannot be used with --model-type "{self.model_type_string}" in '
                    f'{_describe_pipeline_type(self._pipeline_type)} mode with the current '
                    f'combination of arguments and model.')
            return None

    def _get_sdxl_conditioning_args(self, pipeline, default_args, user_args, user_prefix=None):
        if user_prefix:
            user_prefix += '_'
            option_prefix = _textprocessing.dashup(user_prefix)
        else:
            user_prefix = ''
            option_prefix = ''

        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'aesthetic_score', f'sdxl_{user_prefix}aesthetic_score',
                                             f'--sdxl-{option_prefix}aesthetic-scores', None)
        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'original_size', f'sdxl_{user_prefix}original_size',
                                             f'--sdxl-{option_prefix}original-sizes', None)
        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'target_size', f'sdxl_{user_prefix}target_size',
                                             f'--sdxl-{option_prefix}target-sizes', None)

        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'crops_coords_top_left',
                                             f'sdxl_{user_prefix}crops_coords_top_left',
                                             f'--sdxl-{option_prefix}crops-coords-top-left', (0, 0))

        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'negative_aesthetic_score',
                                             f'sdxl_{user_prefix}negative_aesthetic_score',
                                             f'--sdxl-{option_prefix}negative-aesthetic-scores', None)

        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'negative_original_size',
                                             f'sdxl_{user_prefix}negative_original_size',
                                             f'--sdxl-{option_prefix}negative-original-sizes', None)

        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'negative_target_size',
                                             f'sdxl_{user_prefix}negative_target_size',
                                             f'--sdxl-{option_prefix}negative-target-sizes', None)

        self._get_non_universal_pipeline_arg(pipeline, default_args, user_args,
                                             'negative_crops_coords_top_left',
                                             f'sdxl_{user_prefix}negative_crops_coords_top_left',
                                             f'--sdxl-{option_prefix}negative-crops-coords-top-left', (0, 0))

    @staticmethod
    def _pop_sdxl_conditioning_args(default_args):
        default_args.pop('aesthetic_score', None)
        default_args.pop('target_size', None)
        default_args.pop('original_size', None)
        default_args.pop('crops_coords_top_left', None)
        default_args.pop('negative_aesthetic_score', None)
        default_args.pop('negative_target_size', None)
        default_args.pop('negative_original_size', None)
        default_args.pop('negative_crops_coords_top_left', None)

    def _call_torch(self, default_args, user_args):
        prompt: _prompt.Prompt() = user_args.get('prompt', _prompt.Prompt())
        default_args['prompt'] = prompt.positive if prompt.positive else ''
        default_args['negative_prompt'] = prompt.negative

        self._get_sdxl_conditioning_args(self._pipeline, default_args, user_args)

        self._get_non_universal_pipeline_arg(self._pipeline,
                                             default_args, user_args,
                                             'prompt_2', 'sdxl_second_prompt',
                                             '--sdxl-second-prompts', _prompt.Prompt(),
                                             transform=lambda p: p.positive)

        self._get_non_universal_pipeline_arg(self._pipeline,
                                             default_args, user_args,
                                             'negative_prompt_2', 'sdxl_second_prompt',
                                             '--sdxl-second-prompts', _prompt.Prompt(),
                                             transform=lambda p: p.negative)

        self._get_non_universal_pipeline_arg(self._pipeline,
                                             default_args, user_args,
                                             'guidance_rescale', 'guidance_rescale',
                                             '--guidance-rescales', 0.0)

        self._get_non_universal_pipeline_arg(self._pipeline,
                                             default_args, user_args,
                                             'image_guidance_scale', 'image_guidance_scale',
                                             '--image-guidance-scales', 1.5)

        batch_size = user_args.get('batch_size', 1)
        mock_batching = False

        if self._model_type != ModelTypes.TORCH_UPSCALER_X2:
            # Upscaler does not take this argument, can only produce one image
            default_args['num_images_per_prompt'] = batch_size
        else:
            mock_batching = batch_size > 1

        def generate_images(*args, **kwargs):
            if mock_batching:
                images = []
                for i in range(0, batch_size):
                    images.append(DiffusionPipelineWrapper._call_pipeline(
                        *args, **kwargs).images[0])
                return images
            else:
                return DiffusionPipelineWrapper._call_pipeline(
                    *args, **kwargs).images

        default_args['generator'] = torch.Generator(device=self._device).manual_seed(user_args.get('seed', 0))

        if isinstance(self._pipeline, diffusers.StableDiffusionInpaintPipelineLegacy):
            # Not necessary, will cause an error
            default_args.pop('width')
            default_args.pop('height')

        has_control_net = hasattr(self._pipeline, 'controlnet')
        sd_edit = has_control_net or isinstance(self._pipeline,
                                                diffusers.StableDiffusionXLInpaintPipeline)

        if has_control_net:
            default_args['controlnet_conditioning_scale'] = \
                self._get_control_net_conditioning_scale()

            default_args['control_guidance_start'] = \
                self._get_control_net_guidance_start()

            default_args['control_guidance_end'] = \
                self._get_control_net_guidance_end()

        if self._sdxl_refiner_pipeline is None:
            return PipelineWrapperResult(generate_images(
                pipeline=self._pipeline,
                device=self._device, **default_args))

        high_noise_fraction = user_args.get('sdxl_high_noise_fraction',
                                            DEFAULT_SDXL_HIGH_NOISE_FRACTION)

        if sd_edit:
            i_start = dict()
            i_end = dict()
        else:
            i_start = {'denoising_start': high_noise_fraction}
            i_end = {'denoising_end': high_noise_fraction}

        image = DiffusionPipelineWrapper._call_pipeline(pipeline=self._pipeline,
                                                        device=self._device,
                                                        **default_args,
                                                        **i_end,
                                                        output_type='latent').images

        default_args['image'] = image

        if not isinstance(self._sdxl_refiner_pipeline, diffusers.StableDiffusionXLInpaintPipeline):
            # Width / Height not necessary for any other refiner
            if not (isinstance(self._pipeline, diffusers.StableDiffusionXLImg2ImgPipeline) and
                    isinstance(self._sdxl_refiner_pipeline, diffusers.StableDiffusionXLImg2ImgPipeline)):
                # Width / Height does not get passed to img2img
                default_args.pop('width')
                default_args.pop('height')

        # refiner does not use LoRA
        default_args.pop('cross_attention_kwargs', None)

        # Or any of these
        self._pop_sdxl_conditioning_args(default_args)
        default_args.pop('guidance_rescale', None)
        default_args.pop('controlnet_conditioning_scale', None)
        default_args.pop('control_guidance_start', None)
        default_args.pop('control_guidance_end', None)
        default_args.pop('image_guidance_scale', None)
        default_args.pop('control_image', None)

        # we will handle the strength parameter if it is necessary below
        default_args.pop('strength', None)

        # We do not want to override the refiner secondary prompt
        # with that of --sdxl-second-prompts by default
        default_args.pop('prompt_2', None)
        default_args.pop('negative_prompt_2', None)

        self._get_sdxl_conditioning_args(self._sdxl_refiner_pipeline,
                                         default_args, user_args,
                                         user_prefix='refiner')

        self._get_non_universal_pipeline_arg(self._pipeline,
                                             default_args, user_args,
                                             'prompt', 'sdxl_refiner_prompt',
                                             '--sdxl-refiner-prompts', _prompt.Prompt(),
                                             transform=lambda p: p.positive)

        self._get_non_universal_pipeline_arg(self._pipeline,
                                             default_args, user_args,
                                             'negative_prompt', 'sdxl_refiner_prompt',
                                             '--sdxl-refiner-prompts', _prompt.Prompt(),
                                             transform=lambda p: p.negative)

        self._get_non_universal_pipeline_arg(self._pipeline,
                                             default_args, user_args,
                                             'prompt_2', 'sdxl_refiner_second_prompt',
                                             '--sdxl-refiner-second-prompts', _prompt.Prompt(),
                                             transform=lambda p: p.positive)

        self._get_non_universal_pipeline_arg(self._pipeline,
                                             default_args, user_args,
                                             'negative_prompt_2', 'sdxl_refiner_second_prompt',
                                             '--sdxl-refiner-second-prompts', _prompt.Prompt(),
                                             transform=lambda p: p.negative)

        self._get_non_universal_pipeline_arg(self._pipeline,
                                             default_args, user_args,
                                             'guidance_rescale', 'sdxl_refiner_guidance_rescale',
                                             '--sdxl-refiner-guidance-rescales', 0.0)

        sdxl_refiner_inference_steps = user_args.get('sdxl_refiner_inference_steps')
        if sdxl_refiner_inference_steps is not None:
            default_args['num_inference_steps'] = sdxl_refiner_inference_steps

        sdxl_refiner_guidance_scale = user_args.get('sdxl_refiner_guidance_scale')
        if sdxl_refiner_guidance_scale is not None:
            default_args['guidance_scale'] = sdxl_refiner_guidance_scale

        sdxl_refiner_guidance_rescale = user_args.get('sdxl_refiner_guidance_rescale')
        if sdxl_refiner_guidance_rescale is not None:
            default_args['guidance_rescale'] = sdxl_refiner_guidance_rescale

        if sd_edit:
            strength = float(decimal.Decimal('1.0') - decimal.Decimal(str(high_noise_fraction)))

            if strength <= 0.0:
                strength = 0.2
                _messages.log(f'Refiner edit mode image seed strength (1.0 - high-noise-fraction) '
                              f'was calculated at <= 0.0, defaulting to {strength}',
                              level=_messages.WARNING)
            else:
                _messages.log(f'Running refiner in edit mode with '
                              f'refiner image seed strength = {strength}, IE: (1.0 - high-noise-fraction)')

            inference_steps = default_args.get('num_inference_steps')

            if (strength * inference_steps) < 1.0:
                strength = 1.0 / inference_steps
                _messages.log(
                    f'Refiner edit mode image seed strength (1.0 - high-noise-fraction) * inference-steps '
                    f'was calculated at < 1, defaulting to (1.0 / inference-steps): {strength}',
                    level=_messages.WARNING)

            default_args['strength'] = strength

        return PipelineWrapperResult(
            DiffusionPipelineWrapper._call_pipeline(
                pipeline=self._sdxl_refiner_pipeline,
                device=self._device,
                **default_args, **i_start).images)

    def _lazy_init_pipeline(self, pipeline_type):
        if self._pipeline is not None:
            if self._pipeline_type == pipeline_type:
                return

        self._pipeline_type = pipeline_type

        if model_type_is_sdxl(self._model_type) and self._textual_inversion_uris is not None:
            raise NotImplementedError('Textual inversion not supported for SDXL.')

        if self._model_type == ModelTypes.FLAX:
            if not have_jax_flax():
                raise NotImplementedError('flax and jax are not installed.')

            if self._textual_inversion_uris is not None:
                raise NotImplementedError('Textual inversion not supported for flax.')

            if self._pipeline_type != _PipelineTypes.BASIC and self._control_net_uris:
                raise NotImplementedError('Inpaint and Img2Img not supported for flax with ControlNet.')

            if self._vae_tiling or self._vae_slicing:
                raise NotImplementedError('--vae-tiling/--vae-slicing not supported for flax.')

            self._pipeline, self._flax_params, self._parsed_control_net_uris = \
                _create_flax_diffusion_pipeline(pipeline_type,
                                                self._model_path,
                                                revision=self._revision,
                                                dtype=self._dtype,
                                                vae_uri=self._vae_uri,
                                                control_net_uris=self._control_net_uris,
                                                scheduler=self._scheduler,
                                                safety_checker=self._safety_checker,
                                                auth_token=self._auth_token)

        elif self._sdxl_refiner_uri is not None:
            if not model_type_is_sdxl(self._model_type):
                raise NotImplementedError('Only Stable Diffusion XL models support refiners, '
                                          'please use --model-type torch-sdxl if you are trying to load an sdxl model.')

            if not _scheduler_is_help(self._sdxl_refiner_scheduler):
                # Don't load this up if were just going to be getting
                # information about compatible schedulers for the refiner
                self._pipeline, self._parsed_control_net_uris = \
                    _create_torch_diffusion_pipeline(pipeline_type,
                                                     self._model_type,
                                                     self._model_path,
                                                     model_subfolder=self._model_subfolder,
                                                     revision=self._revision,
                                                     variant=self._variant,
                                                     dtype=self._dtype,
                                                     vae_uri=self._vae_uri,
                                                     lora_uris=self._lora_uris,
                                                     control_net_uris=self._control_net_uris,
                                                     scheduler=self._scheduler,
                                                     safety_checker=self._safety_checker,
                                                     auth_token=self._auth_token,
                                                     device=self._device)

            refiner_pipeline_type = _PipelineTypes.IMG2IMG if pipeline_type is _PipelineTypes.BASIC else pipeline_type

            if self._pipeline is not None:
                refiner_extra_args = {'vae': self._pipeline.vae,
                                      'text_encoder_2': self._pipeline.text_encoder_2}
            else:
                refiner_extra_args = None

            self._sdxl_refiner_pipeline, discarded = \
                _create_torch_diffusion_pipeline(refiner_pipeline_type,
                                                 ModelTypes.TORCH_SDXL,
                                                 self._sdxl_refiner_uri,
                                                 model_subfolder=self._sdxl_refiner_subfolder,
                                                 revision=self._sdxl_refiner_revision,

                                                 variant=self._sdxl_refiner_variant if
                                                 self._sdxl_refiner_variant is not None else self._variant,

                                                 dtype=self._sdxl_refiner_dtype if
                                                 self._sdxl_refiner_dtype is not None else self._dtype,

                                                 scheduler=self._scheduler if
                                                 self._sdxl_refiner_scheduler is None else self._sdxl_refiner_scheduler,

                                                 safety_checker=self._safety_checker,
                                                 auth_token=self._auth_token,
                                                 extra_args=refiner_extra_args)
        else:
            offload = self._control_net_uris and self._model_type == ModelTypes.TORCH_SDXL

            self._pipeline, self._parsed_control_net_uris = \
                _create_torch_diffusion_pipeline(pipeline_type,
                                                 self._model_type,
                                                 self._model_path,
                                                 model_subfolder=self._model_subfolder,
                                                 revision=self._revision,
                                                 variant=self._variant,
                                                 dtype=self._dtype,
                                                 vae_uri=self._vae_uri,
                                                 lora_uris=self._lora_uris,
                                                 textual_inversion_uris=self._textual_inversion_uris,
                                                 control_net_uris=self._control_net_uris,
                                                 scheduler=self._scheduler,
                                                 safety_checker=self._safety_checker,
                                                 auth_token=self._auth_token,
                                                 device=self._device,
                                                 sequential_cpu_offload=offload)

        _set_vae_slicing_tiling(pipeline=self._pipeline,
                                vae_tiling=self._vae_tiling,
                                vae_slicing=self._vae_slicing)

        if self._sdxl_refiner_pipeline is not None:
            _set_vae_slicing_tiling(pipeline=self._sdxl_refiner_pipeline,
                                    vae_tiling=self._vae_tiling,
                                    vae_slicing=self._vae_slicing)

    @staticmethod
    def _determine_pipeline_type(kwargs):
        if 'image' in kwargs and 'mask_image' in kwargs:
            # Inpainting is handled by INPAINT type
            return _PipelineTypes.INPAINT

        if 'image' in kwargs:
            # Image only is handled by IMG2IMG type
            return _PipelineTypes.IMG2IMG

        # All other situations handled by BASIC type
        return _PipelineTypes.BASIC

    def __call__(self, **kwargs) -> PipelineWrapperResult:
        """
        Call the pipeline and generate a result.

        :param kwargs: See :py:meth:`.DiffusionArguments.get_pipeline_wrapper_args`

        :raises: :py:class:`dgenerate.pipelinewrapper.InvalidModelPathError`
            :py:class:`dgenerate.pipelinewrapper.InvalidSDXLRefinerUriError`
            :py:class:`dgenerate.pipelinewrapper.InvalidVaeUriError`
            :py:class:`dgenerate.pipelinewrapper.InvalidLoRAUriError`
            :py:class:`dgenerate.pipelinewrapper.InvalidControlNetUriError`
            :py:class:`dgenerate.pipelinewrapper.InvalidTextualInversionUriError`
            :py:class:`dgenerate.pipelinewrapper.InvalidSchedulerName`
            :py:class:`dgenerate.pipelinewrapper.OutOfMemoryError`
            :py:class:`NotImplementedError`

        :return: :py:class:`.PipelineWrapperResult`
        """
        self._lazy_init_pipeline(DiffusionPipelineWrapper._determine_pipeline_type(kwargs))

        default_args = self._pipeline_defaults(kwargs)

        _messages.debug_log(f'Calling Pipeline Wrapper: "{self}",'
                            '\nCalled with User Args: ',
                            lambda: _textprocessing.debug_format_args(kwargs))

        if self._model_type == ModelTypes.FLAX:
            try:
                result = self._call_flax(default_args, kwargs)
            except jaxlib.xla_extension.XlaRuntimeError as e:
                raise OutOfMemoryError(e)
        else:
            try:
                result = self._call_torch(default_args, kwargs)
            except torch.cuda.OutOfMemoryError as e:
                raise OutOfMemoryError(e)

        result.dgenerate_opts = self.reconstruct_dgenerate_opts(**kwargs)
        return result
