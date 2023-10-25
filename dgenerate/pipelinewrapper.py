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
import gc
import inspect
import os
import re
import shlex
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
import dgenerate.memory as _memory
import dgenerate.hfutil as _hfutil
import dgenerate.image as _image
import diffusers

DEFAULT_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 5
DEFAULT_IMAGE_SEED_STRENGTH = 0.8
DEFAULT_IMAGE_GUIDANCE_SCALE = 1.5
DEFAULT_SDXL_HIGH_NOISE_FRACTION = 0.8
DEFAULT_X4_UPSCALER_NOISE_LEVEL = 20
DEFAULT_OUTPUT_WIDTH = 512
DEFAULT_OUTPUT_HEIGHT = 512

_TORCH_PIPELINE_CACHE = dict()
"""Global in memory cache for torch diffusers pipelines"""

_FLAX_PIPELINE_CACHE = dict()
"""Global in memory cache for flax diffusers pipelines"""

_PIPELINE_CACHE_SIZE = 0
"""Estimated memory consumption in bytes of all pipelines cached in memory"""

_TORCH_CONTROL_NET_CACHE = dict()
"""Global in memory cache for torch ControlNet models"""

_FLAX_CONTROL_NET_CACHE = dict()
"""Global in memory cache for flax ControlNet models"""

_CONTROL_NET_CACHE_SIZE = 0
"""Estimated memory consumption in bytes of all ControlNet models cached in memory"""

_TORCH_VAE_CACHE = dict()
"""Global in memory cache for torch VAE models"""

_FLAX_VAE_CACHE = dict()
"""Global in memory cache for flax VAE models"""

_VAE_CACHE_SIZE = 0
"""Estimated memory consumption in bytes of all VAE models cached in memory"""


def pipeline_cache_size() -> int:
    """
    Return the estimated memory usage in bytes of all diffusers pipelines currently cached in memory.

    :return: memory usage in bytes.
    """
    return _PIPELINE_CACHE_SIZE


def vae_cache_size() -> int:
    """
    Return the estimated memory usage in bytes of all user specified VAEs currently cached in memory.

    :return:  memory usage in bytes.
    """
    return _VAE_CACHE_SIZE


def control_net_cache_size() -> int:
    """
    Return the estimated memory usage in bytes of all user specified ControlNet models currently cached in memory.

    :return:  memory usage in bytes.
    """
    return _CONTROL_NET_CACHE_SIZE


CACHE_MEMORY_CONSTRAINTS: typing.List[str] = ['used_percent > 90']
"""
Cache constraint expressions for when to clear all model caches (DiffusionPipeline, VAE, and ControlNet), 
syntax provided via :py:meth:`dgenerate.util.memory_constraints`

If any of these constraints are met, a call to :py:meth:`.enforce_cache_constraints` will call
:py:meth:`.clear_model_cache` and force a garbage collection.
"""

PIPELINE_CACHE_MEMORY_CONSTRAINTS: typing.List[str] = ['pipeline_size > (available * 0.75)']
"""
Cache constraint expressions for when to clear the DiffusionPipeline cache, 
syntax provided via :py:meth:`dgenerate.util.memory_constraints`

If any of these constraints are met, a call to :py:meth:`.enforce_cache_constraints` will call
:py:meth:`.clear_pipeline_cache` and force a garbage collection.

Extra variables include: *cache_size* (the current estimated cache size in bytes), 
and *pipeline_size* (the estimated size of the new pipeline before it is brought into memory, in bytes)

"""

CONTROL_NET_CACHE_MEMORY_CONSTRAINTS: typing.List[str] = ['control_net_size > (available * 0.75)']
"""
Cache constraint expressions for when to clear the ControlNet cache, 
syntax provided via :py:meth:`dgenerate.util.memory_constraints`

If any of these constraints are met, a call to :py:meth:`.enforce_cache_constraints` will call
:py:meth:`.clear_control_net_cache` and force a garbage collection.

Extra variables include: *cache_size* (the current estimated cache size in bytes), 
and *control_net_size* (the estimated size of the new ControlNet before it is brought into memory, in bytes)
"""

VAE_CACHE_MEMORY_CONSTRAINTS: typing.List[str] = ['vae_size > (available * 0.75)']
"""
Cache constraint expressions for when to clear VAE cache, 
syntax provided via :py:meth:`dgenerate.util.memory_constraints`

If any of these constraints are met, a call to :py:meth:`.enforce_cache_constraints` will call
:py:meth:`.clear_vae_cache` and force a garbage collection.

Extra variables include: *cache_size* (the current estimated cache size in bytes), 
and *vae_size* (the estimated size of the new VAE before it is brought into memory, in bytes)
"""


def clear_pipeline_cache(collect=True):
    """
    Clear DiffusionPipeline cache and then garbage collect.

    :param collect: Call :py:meth:`gc.collect` ?
    """
    global _TORCH_PIPELINE_CACHE, \
        _FLAX_PIPELINE_CACHE, \
        _PIPELINE_CACHE_SIZE

    _TORCH_PIPELINE_CACHE.clear()
    _FLAX_PIPELINE_CACHE.clear()
    _PIPELINE_CACHE_SIZE = 0

    if collect:
        _messages.debug_log(
            f'{_types.fullname(clear_pipeline_cache)} calling gc.collect() by request')

        gc.collect()


def clear_control_net_cache(collect=True):
    """
    Clear ControlNet cache and then garbage collect.

    :param collect: Call :py:meth:`gc.collect` ?
    """
    global _TORCH_CONTROL_NET_CACHE, \
        _FLAX_CONTROL_NET_CACHE, \
        _CONTROL_NET_CACHE_SIZE

    _TORCH_CONTROL_NET_CACHE.clear()
    _FLAX_CONTROL_NET_CACHE.clear()

    _CONTROL_NET_CACHE_SIZE = 0

    if collect:
        _messages.debug_log(
            f'{_types.fullname(clear_control_net_cache)} calling gc.collect() by request')

        gc.collect()


def clear_vae_cache(collect=True):
    """
    Clear VAE cache and then garbage collect.

    :param collect: Call :py:meth:`gc.collect` ?
    """
    global _TORCH_VAE_CACHE, \
        _FLAX_VAE_CACHE, \
        _VAE_CACHE_SIZE

    _TORCH_VAE_CACHE.clear()
    _FLAX_VAE_CACHE.clear()

    _VAE_CACHE_SIZE = 0

    if collect:
        _messages.debug_log(
            f'{_types.fullname(clear_vae_cache)} calling gc.collect() by request')

        gc.collect()


def clear_model_cache(collect=True):
    """
    Clear all in memory model caches and garbage collect.

    :param collect: Call :py:meth:`gc.collect` ?

    """
    global _TORCH_PIPELINE_CACHE, \
        _FLAX_PIPELINE_CACHE, \
        _TORCH_CONTROL_NET_CACHE, \
        _FLAX_CONTROL_NET_CACHE, \
        _TORCH_VAE_CACHE, \
        _FLAX_VAE_CACHE, \
        _PIPELINE_CACHE_SIZE, \
        _CONTROL_NET_CACHE_SIZE, \
        _VAE_CACHE_SIZE

    _TORCH_PIPELINE_CACHE.clear()
    _FLAX_PIPELINE_CACHE.clear()
    _TORCH_CONTROL_NET_CACHE.clear()
    _FLAX_CONTROL_NET_CACHE.clear()
    _TORCH_VAE_CACHE.clear()
    _FLAX_VAE_CACHE.clear()

    _PIPELINE_CACHE_SIZE = 0
    _CONTROL_NET_CACHE_SIZE = 0
    _VAE_CACHE_SIZE = 0

    if collect:
        _messages.debug_log(
            f'{_types.fullname(clear_model_cache)} calling gc.collect() by request')

        gc.collect()


def enforce_cache_constraints(collect=True):
    """
    Enforce :py:attr:`CACHE_MEMORY_CONSTRAINTS` and clear caches accordingly

    :param collect: Call :py:meth:`gc.collect` after a cache clear ?
    :return: Whether any caches were cleared due to constraint expressions.
    """

    m_name = __name__

    _messages.debug_log(f'Enforcing {m_name}.CACHE_MEMORY_CONSTRAINTS =',
                        CACHE_MEMORY_CONSTRAINTS)

    _messages.debug_log(_memory.memory_use_debug_string())

    if _memory.memory_constraints(CACHE_MEMORY_CONSTRAINTS):
        _messages.debug_log(f'{m_name}.CACHE_MEMORY_CONSTRAINTS '
                            f'{CACHE_MEMORY_CONSTRAINTS} met, '
                            f'calling {_types.fullname(clear_model_cache)}.')

        clear_model_cache(collect=collect)
        return True

    return False


def enforce_pipeline_cache_constraints(new_pipeline_size, collect=True):
    """
    Enforce :py:attr:`.PIPELINE_CACHE_MEMORY_CONSTRAINTS` and clear the
    :py:class:`diffusers.DiffusionPipeline` cache if needed.

    :param new_pipeline_size: estimated size in bytes of any new pipeline that is about to enter memory
    :param collect: Call :py:meth:`gc.collect` after a cache clear ?
    :return: Whether the cache was cleared due to constraint expressions.
    """

    m_name = __name__

    _messages.debug_log(f'Enforcing {m_name}.PIPELINE_CACHE_MEMORY_CONSTRAINTS =',
                        PIPELINE_CACHE_MEMORY_CONSTRAINTS,
                        f'(cache_size = {_memory.bytes_best_human_unit(pipeline_cache_size())},',
                        f'pipeline_size = {_memory.bytes_best_human_unit(new_pipeline_size)})')

    _messages.debug_log(_memory.memory_use_debug_string())

    if _memory.memory_constraints(PIPELINE_CACHE_MEMORY_CONSTRAINTS,
                                  extra_vars={'cache_size': pipeline_cache_size(),
                                              'pipeline_size': new_pipeline_size}):
        _messages.debug_log(f'{m_name}.PIPELINE_CACHE_MEMORY_CONSTRAINTS '
                            f'{PIPELINE_CACHE_MEMORY_CONSTRAINTS} met, '
                            f'calling {_types.fullname(clear_pipeline_cache)}.')
        clear_pipeline_cache(collect=collect)
        return True
    return False


def enforce_vae_cache_constraints(new_vae_size, collect=True):
    """
    Enforce :py:attr:`.VAE_CACHE_MEMORY_CONSTRAINTS` and clear the
    VAE cache if needed.

    :param new_vae_size: estimated size in bytes of any new vae that is about to enter memory
    :param collect: Call :py:meth:`gc.collect` after a cache clear ?
    :return: Whether the cache was cleared due to constraint expressions.
    """

    m_name = __name__

    _messages.debug_log(f'Enforcing {m_name}.VAE_CACHE_MEMORY_CONSTRAINTS =',
                        VAE_CACHE_MEMORY_CONSTRAINTS,
                        f'(cache_size = {_memory.bytes_best_human_unit(vae_cache_size())},',
                        f'vae_size = {_memory.bytes_best_human_unit(new_vae_size)})')

    _messages.debug_log(_memory.memory_use_debug_string())

    if _memory.memory_constraints(VAE_CACHE_MEMORY_CONSTRAINTS,
                                  extra_vars={'cache_size': vae_cache_size(),
                                              'vae_size': new_vae_size}):
        _messages.debug_log(f'{m_name}.VAE_CACHE_MEMORY_CONSTRAINTS '
                            f'{VAE_CACHE_MEMORY_CONSTRAINTS} met, '
                            f'calling {_types.fullname(clear_vae_cache)}.')

        clear_vae_cache(collect=collect)
        return True
    return False


def enforce_control_net_cache_constraints(new_control_net_size, collect=True):
    """
    Enforce :py:attr:`.CONTROL_NET_CACHE_MEMORY_CONSTRAINTS` and clear the
    ControlNet cache if needed.

    :param new_control_net_size: estimated size in bytes of any new control net that is about to enter memory
    :param collect: Call :py:meth:`gc.collect` after a cache clear ?
    :return: Whether the cache was cleared due to constraint expressions.
    """

    m_name = __name__

    _messages.debug_log(f'Enforcing {m_name}.CONTROL_NET_CACHE_MEMORY_CONSTRAINTS =',
                        CONTROL_NET_CACHE_MEMORY_CONSTRAINTS,
                        f'(cache_size = {_memory.bytes_best_human_unit(control_net_cache_size())},',
                        f'control_net_size = {_memory.bytes_best_human_unit(new_control_net_size)})')

    _messages.debug_log(_memory.memory_use_debug_string())

    if _memory.memory_constraints(CONTROL_NET_CACHE_MEMORY_CONSTRAINTS,
                                  extra_vars={'cache_size': control_net_cache_size(),
                                              'control_net_size': new_control_net_size}):
        _messages.debug_log(f'{m_name}.CONTROL_NET_CACHE_MEMORY_CONSTRAINTS '
                            f'{CONTROL_NET_CACHE_MEMORY_CONSTRAINTS} met, '
                            f'calling {_types.fullname(clear_control_net_cache)}.')

        clear_control_net_cache(collect=collect)
        return True
    return False


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
    Error in ``--sdxl-refiner`` uri
    """
    pass


class InvalidVaeUriError(InvalidModelPathError):
    """
    Error in ``--vae`` uri
    """
    pass


class InvalidControlNetUriError(InvalidModelPathError):
    """
    Error in ``--control-nets`` uri
    """
    pass


class InvalidLoRAUriError(InvalidModelPathError):
    """
    Error in ``--lora`` uri
    """
    pass


class InvalidTextualInversionUriError(InvalidModelPathError):
    """
    Error in ``--textual-inversions`` uri
    """
    pass


class InvalidSchedulerName(Exception):
    """
    Unknown scheduler name used
    """
    pass


class SchedulerHelpException(Exception):
    """
    Not an error, runtime scheduler help was requested by passing "help" to a scheduler name
    argument of :py:meth:`.DiffusionPipelineWrapper.__init__` such as ``scheduler`` or ``sdxl_refiner_scheduler``.
    Upon calling :py:meth:`.DiffusionPipelineWrapper.__call__` info was printed using :py:meth:`dgenerate.messages.log`,
    then this exception raised to get out of the call stack.
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


class PipelineTypes(enum.Enum):
    TXT2IMG = 1
    IMG2IMG = 2
    INPAINT = 3


def get_pipeline_type_enum(id_str: typing.Union[PipelineTypes, str, None]) -> PipelineTypes:
    """
    Get a :py:class:`.PipelineTypes` enum value from a string.

    :param id_str: one of: "txt2img", "img2img", or "inpaint"
    :return: :py:class:`.PipelineTypes`
    """

    if isinstance(id_str, PipelineTypes):
        return id_str

    return {'txt2img': PipelineTypes.TXT2IMG,
            'img2img': PipelineTypes.IMG2IMG,
            'inpaint': PipelineTypes.INPAINT}[id_str.strip().lower()]


def get_pipeline_type_string(pipeline_type_enum: PipelineTypes):
    """
    Convert a :py:class:`.PipelineTypes` enum value to a string.

    :param pipeline_type_enum: :py:class:`.PipelineTypes` value

    :return: one of: "txt2img", "img2img", or "inpaint"
    """
    pipeline_type = get_pipeline_type_enum(pipeline_type_enum)

    return {PipelineTypes.TXT2IMG: 'txt2img',
            PipelineTypes.IMG2IMG: 'img2img',
            PipelineTypes.INPAINT: 'inpaint'}[pipeline_type]


class DataTypes(enum.Enum):
    AUTO = 0
    FLOAT16 = 1
    FLOAT32 = 2


def supported_data_type_strings():
    """
    Return a list of supported ``--dtype`` strings
    """
    return ['auto', 'float16', 'float32']


def supported_data_type_enums() -> typing.List[DataTypes]:
    """
    Return a list of supported :py:class:`.DataTypes` enum values
    """
    return [get_data_type_enum(i) for i in supported_data_type_strings()]


def get_data_type_enum(id_str: typing.Union[DataTypes, str, None]) -> DataTypes:
    """
    Convert a ``--dtype`` string to its :py:class:`.DataTypes` enum value

    :param id_str: ``--dtype`` string
    :return: :py:class:`.DataTypes`
    """

    if isinstance(id_str, DataTypes):
        return id_str

    return {'auto': DataTypes.AUTO,
            'float16': DataTypes.FLOAT16,
            'float32': DataTypes.FLOAT32}[id_str.strip().lower()]


def get_data_type_string(data_type_enum: DataTypes) -> str:
    """
    Convert a :py:class:`.DataTypes` enum value to its ``--dtype`` string

    :param data_type_enum: :py:class:`.DataTypes` value
    :return: ``--dtype`` string
    """

    model_type = get_data_type_enum(data_type_enum)

    return {DataTypes.AUTO: 'auto',
            DataTypes.FLOAT16: 'float16',
            DataTypes.FLOAT32: 'float32'}[model_type]


class ModelTypes(enum.Enum):
    """
    Enum representation of ``--model-type``
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
    Return a list of supported ``--model-type`` strings
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
    Convert a ``--model-type`` string to its :py:class:`.ModelTypes` enum value

    :param id_str: ``--model-type`` string
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
    Convert a :py:class:`.ModelTypes` enum value to its ``--model-type`` string

    :param model_type_enum: :py:class:`.ModelTypes` value
    :return: ``--model-type`` string
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
    Does a ``--model-type`` string or :py:class:`.ModelTypes` enum value represent an upscaler model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelTypes` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 'upscaler' in model_type


def model_type_is_sdxl(model_type: typing.Union[ModelTypes, str]) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelTypes` enum value represent an SDXL model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelTypes` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 'sdxl' in model_type


def model_type_is_torch(model_type: typing.Union[ModelTypes, str]) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelTypes` enum value represent an Torch model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelTypes` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 'torch' in model_type


def model_type_is_flax(model_type: typing.Union[ModelTypes, str]) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelTypes` enum value represent an Flax model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelTypes` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 'flax' in model_type


def model_type_is_pix2pix(model_type: typing.Union[ModelTypes, str]) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelTypes` enum value represent an pix2pix type model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelTypes` enum value
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


def _get_flax_dtype(dtype: typing.Union[DataTypes, typing.Any, None]):
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


# noinspection HttpUrlsUsage
def _is_single_file_model_load(path):
    """
    Should we use diffusers.loaders.FromSingleFileMixin.from_single_file on this path?
    :param path: The path
    :return: true or false
    """
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

    dtype: typing.Optional[DataTypes]
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
                 dtype: typing.Union[DataTypes, str, None] = None,
                 scale: float = 1.0,
                 from_torch: bool = False):

        self.model = model
        self.revision = revision
        self.subfolder = subfolder
        self.dtype = get_data_type_enum(dtype) if dtype else None
        self.scale = scale
        self.from_torch = from_torch

    @_memoize(_FLAX_CONTROL_NET_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _struct_hasher}),
              on_hit=lambda key, hit: _simple_cache_hit_debug("Flax ControlNet", key, hit),
              on_create=lambda key, new: _simple_cache_miss_debug("Flax ControlNet", key, new))
    def load(self,
             dtype_fallback: DataTypes = DataTypes.AUTO,
             use_auth_token: _types.OptionalString = None,
             local_files_only: bool = False) -> typing.Tuple[diffusers.FlaxControlNetModel, typing.Any]:
        """
        Load a :py:class:`diffusers.FlaxControlNetModel` from this URI.

        :param dtype_fallback: Fallback datatype if ``dtype`` was not specified in the URI.

        :param use_auth_token: Optional huggingface API auth token, used for downloading
            restricted repos that your account has access to.

        :param local_files_only: Avoid connecting to huggingface to download models and
            only use cached models?

        :return: :py:class:`diffusers.FlaxControlNetModel`
        """

        single_file_load_path = _is_single_file_model_load(self.model)

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

            enforce_control_net_cache_constraints(
                new_control_net_size=estimated_memory_usage)

            flax_dtype = _get_flax_dtype(
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

        global _CONTROL_NET_CACHE_SIZE
        _CONTROL_NET_CACHE_SIZE += estimated_memory_usage

        # Tag for internal use
        new_net[0].DGENERATE_SIZE_ESTIMATE = estimated_memory_usage

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

    dtype: typing.Optional[DataTypes]
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
                 dtype: typing.Union[DataTypes, str, None] = None,
                 scale: float = 1.0,
                 start: float = 0.0,
                 end: float = 1.0):

        self.model = model
        self.revision = revision
        self.variant = variant
        self.subfolder = subfolder
        self.dtype = get_data_type_enum(dtype) if dtype else None
        self.scale = scale
        self.start = start
        self.end = end

    @_memoize(_TORCH_CONTROL_NET_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _struct_hasher}),
              on_hit=lambda key, hit: _simple_cache_hit_debug("Torch ControlNet", key, hit),
              on_create=lambda key, new: _simple_cache_miss_debug("Torch ControlNet", key, new))
    def load(self,
             dtype_fallback: DataTypes = DataTypes.AUTO,
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

        single_file_load_path = _is_single_file_model_load(self.model)

        torch_dtype = _get_torch_dtype(
            dtype_fallback if self.dtype is None else self.dtype)

        if single_file_load_path:

            estimated_memory_usage = _hfutil.estimate_model_memory_use(
                repo_id=self.model,
                revision=self.revision,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only
            )

            enforce_control_net_cache_constraints(
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

            enforce_control_net_cache_constraints(
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

        global _CONTROL_NET_CACHE_SIZE
        _CONTROL_NET_CACHE_SIZE += estimated_memory_usage

        # Tag for internal use
        new_net.DGENERATE_SIZE_ESTIMATE = estimated_memory_usage

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

    dtype: typing.Optional[DataTypes]
    """
    Model dtype (precision)
    """

    def __init__(self,
                 model: str,
                 revision: _types.OptionalString = None,
                 variant: _types.OptionalString = None,
                 subfolder: _types.OptionalPath = None,
                 dtype: typing.Union[DataTypes, str, None] = None):
        self.model = model
        self.revision = revision
        self.variant = variant
        self.dtype = get_data_type_enum(dtype) if dtype else None
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

        supported_dtypes = supported_data_type_strings()

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

    dtype: typing.Optional[DataTypes]
    """
    Model dtype (precision)
    """

    def __init__(self,
                 encoder: str,
                 model: str,
                 revision: _types.OptionalString = None,
                 variant: _types.OptionalString = None,
                 subfolder: _types.OptionalString = None,
                 dtype: typing.Union[DataTypes, str, None] = None):

        self.encoder = encoder
        self.model = model
        self.revision = revision
        self.variant = variant
        self.dtype = get_data_type_enum(dtype) if dtype else None
        self.subfolder = subfolder

    @_memoize(_TORCH_VAE_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _struct_hasher}),
              on_hit=lambda key, hit: _simple_cache_hit_debug("Torch VAE", key, hit),
              on_create=lambda key, new: _simple_cache_miss_debug("Torch VAE", key, new))
    def load(self,
             dtype_fallback: DataTypes = DataTypes.AUTO,
             use_auth_token: _types.OptionalString = None,
             local_files_only=False) -> typing.Union[diffusers.AutoencoderKL,
                                                     diffusers.AsymmetricAutoencoderKL,
                                                     diffusers.AutoencoderTiny]:

        if self.dtype is None:
            torch_dtype = _get_torch_dtype(dtype_fallback)
        else:
            torch_dtype = _get_torch_dtype(self.dtype)

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
        single_file_load_path = _is_single_file_model_load(path)

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

            enforce_vae_cache_constraints(new_vae_size=estimated_memory_use)

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

            enforce_vae_cache_constraints(new_vae_size=estimated_memory_use)

            vae = encoder.from_pretrained(path,
                                          revision=self.revision,
                                          variant=self.variant,
                                          torch_dtype=torch_dtype,
                                          subfolder=self.subfolder,
                                          use_auth_token=use_auth_token,
                                          local_files_only=local_files_only)

        _messages.debug_log('Estimated Torch VAE Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_use))

        global _VAE_CACHE_SIZE
        _VAE_CACHE_SIZE += estimated_memory_use

        # Tag for internal use

        vae.DGENERATE_SIZE_ESTIMATE = estimated_memory_use

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

        supported_dtypes = supported_data_type_strings()
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

    dtype: typing.Optional[DataTypes]
    """
    Model dtype (precision)
    """

    def __init__(self,
                 encoder: str,
                 model: str,
                 revision: _types.OptionalString,
                 subfolder: _types.OptionalPath,
                 dtype: typing.Optional[DataTypes]):

        self.encoder = encoder
        self.model = model
        self.revision = revision
        self.dtype = dtype
        self.subfolder = subfolder

    @_memoize(_FLAX_VAE_CACHE,
              exceptions={'local_files_only'},
              hasher=lambda args: _d_memoize.args_cache_key(args, {'self': _struct_hasher}),
              on_hit=lambda key, hit: _simple_cache_hit_debug("Flax VAE", key, hit),
              on_create=lambda key, new: _simple_cache_miss_debug("Flax VAE", key, new))
    def load(self,
             dtype_fallback: DataTypes = DataTypes.AUTO,
             use_auth_token: _types.OptionalString = None,
             local_files_only=False) -> typing.Tuple[diffusers.FlaxAutoencoderKL, typing.Any]:

        if self.dtype is None:
            flax_dtype = _get_flax_dtype(dtype_fallback)
        else:
            flax_dtype = _get_flax_dtype(self.dtype)

        encoder_name = self.encoder

        if encoder_name == 'FlaxAutoencoderKL':
            encoder = diffusers.FlaxAutoencoderKL
        else:
            raise InvalidVaeUriError(f'Unknown VAE flax encoder class {encoder_name}')

        path = self.model

        can_single_file_load = hasattr(encoder, 'from_single_file')
        single_file_load_path = _is_single_file_model_load(path)

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

            enforce_vae_cache_constraints(new_vae_size=estimated_memory_use)

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

            enforce_vae_cache_constraints(new_vae_size=estimated_memory_use)

            vae = encoder.from_pretrained(path,
                                          revision=self.revision,
                                          dtype=flax_dtype,
                                          subfolder=self.subfolder,
                                          use_auth_token=use_auth_token,
                                          local_files_only=local_files_only)

        _messages.debug_log('Estimated Flax VAE Memory Use:',
                            _memory.bytes_best_human_unit(estimated_memory_use))

        global _VAE_CACHE_SIZE
        _VAE_CACHE_SIZE += estimated_memory_use

        # Tag for internal use

        vae[0].DGENERATE_SIZE_ESTIMATE = estimated_memory_use

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

        supported_dtypes = supported_data_type_strings()
        if dtype is not None and dtype not in supported_dtypes:
            raise InvalidVaeUriError(
                f'Flax VAE "dtype" must be {", ".join(supported_dtypes)}, '
                f'or left undefined, received: {dtype}')

        return FlaxVAEUri(encoder=r.concept,
                          model=model,
                          revision=r.args.get('revision', None),
                          dtype=_get_flax_dtype(dtype),
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
        extra_args = {k: v for k, v in locals() if k not in {'self', 'pipeline'}}

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
        extra_args = {k: v for k, v in locals() if k not in {'self', 'pipeline'}}

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


def _uri_hash_with_parser(parser):
    def hasher(path):
        if not path:
            return path

        return _struct_hasher(parser(path))

    return hasher


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


def _disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False] * num_images
    else:
        return images, False


def _args_except(args, *exceptions):
    return {k: v for k, v in args.items() if k not in exceptions}


def _scheduler_is_help(name):
    if name is None:
        return False
    return name.strip().lower() == 'help'


def set_vae_slicing_tiling(pipeline: typing.Union[diffusers.DiffusionPipeline,
                                                  diffusers.FlaxDiffusionPipeline],
                           vae_tiling: bool,
                           vae_slicing: bool):
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
            return '[]'

        if isinstance(paths, str):
            return '[' + _uri_hash_with_parser(parser)(paths) + ']'

        return '[' + ','.join(_uri_hash_with_parser(parser)(path) for path in paths) + ']'

    return hasher


def _estimate_pipeline_memory_use(
        model_path,
        revision,
        variant=None,
        subfolder=None,
        vae_uri=None,
        lora_uris=None,
        textual_inversion_uris=None,
        safety_checker=False,
        auth_token=None,
        extra_args=None,
        local_files_only=False,
        flax=False):
    usage = _hfutil.estimate_model_memory_use(
        repo_id=model_path,
        revision=revision,
        variant=variant,
        subfolder=subfolder,
        include_vae=not vae_uri,
        safety_checker=safety_checker,
        include_text_encoder=not extra_args or 'text_encoder' not in extra_args,
        include_text_encoder_2=not extra_args or 'text_encoder_2' not in extra_args,
        use_auth_token=auth_token,
        local_files_only=local_files_only,
        flax=flax
    )

    if lora_uris:
        if isinstance(lora_uris, str):
            lora_uris = [lora_uris]

        for lora_uri in lora_uris:
            parsed = parse_lora_uri(lora_uri)

            usage += _hfutil.estimate_model_memory_use(
                repo_id=parsed.model,
                revision=parsed.revision,
                subfolder=parsed.subfolder,
                weight_name=parsed.weight_name,
                use_auth_token=auth_token,
                local_files_only=local_files_only,
                flax=flax
            )

    if textual_inversion_uris:
        if isinstance(textual_inversion_uris, str):
            textual_inversion_uris = [textual_inversion_uris]

        for textual_inversion_uri in textual_inversion_uris:
            parsed = parse_textual_inversion_uri(textual_inversion_uri)

            usage += _hfutil.estimate_model_memory_use(
                repo_id=parsed.model,
                revision=parsed.revision,
                subfolder=parsed.subfolder,
                weight_name=parsed.weight_name,
                use_auth_token=auth_token,
                local_files_only=local_files_only,
                flax=flax
            )

    return usage


class TorchPipelineCreationResult:
    pipeline: diffusers.DiffusionPipeline
    parsed_vae_uri: typing.Optional[TorchVAEUri]
    parsed_lora_uris: typing.List[LoRAUri]
    parsed_textual_inversion_uris: typing.List[TextualInversionUri]
    parsed_control_net_uris: typing.List[TorchControlNetUri]

    def __init__(self,
                 pipeline: diffusers.DiffusionPipeline,
                 parsed_vae_uri: typing.Optional[TorchVAEUri],
                 parsed_lora_uris: typing.List[LoRAUri],
                 parsed_textual_inversion_uris: typing.List[TextualInversionUri],
                 parsed_control_net_uris: typing.List[TorchControlNetUri]):
        self.pipeline = pipeline
        self.parsed_vae_uri = parsed_vae_uri
        self.parsed_lora_uris = parsed_lora_uris
        self.parsed_textual_inversion_uris = parsed_textual_inversion_uris
        self.parsed_control_net_uris = parsed_control_net_uris


@_memoize(_TORCH_PIPELINE_CACHE,
          exceptions={'local_files_only'},
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
          on_hit=lambda key, hit: _simple_cache_hit_debug("Torch Pipeline", key, hit.pipeline),
          on_create=lambda key, new: _simple_cache_miss_debug('Torch Pipeline', key, new.pipeline))
def create_torch_diffusion_pipeline(pipeline_type: PipelineTypes,
                                    model_type: ModelTypes,
                                    model_path: str,
                                    revision: _types.OptionalString = None,
                                    variant: _types.OptionalString = None,
                                    subfolder: _types.OptionalString = None,
                                    dtype: DataTypes = DataTypes.AUTO,
                                    vae_uri: _types.OptionalUri = None,
                                    lora_uris: _types.OptionalUris = None,
                                    textual_inversion_uris: _types.OptionalUris = None,
                                    control_net_uris: _types.OptionalUris = None,
                                    scheduler: _types.OptionalString = None,
                                    safety_checker: bool = False,
                                    auth_token: _types.OptionalString = None,
                                    device: str = 'cuda',
                                    extra_args: typing.Optional[typing.Dict[str, typing.Any]] = None,
                                    model_cpu_offload: bool = False,
                                    sequential_cpu_offload: bool = False,
                                    local_files_only: bool = False) -> TorchPipelineCreationResult:
    # Pipeline class selection

    if model_type_is_upscaler(model_type):
        if pipeline_type != PipelineTypes.IMG2IMG and not _scheduler_is_help(scheduler):
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

        if pipeline_type == PipelineTypes.TXT2IMG:
            if pix2pix:
                raise NotImplementedError(
                    'pix2pix models only work in img2img mode and cannot work without --image-seeds.')

            if control_net_uris:
                pipeline_class = diffusers.StableDiffusionXLControlNetPipeline if sdxl else diffusers.StableDiffusionControlNetPipeline
            else:
                pipeline_class = diffusers.StableDiffusionXLPipeline if sdxl else diffusers.StableDiffusionPipeline
        elif pipeline_type == PipelineTypes.IMG2IMG:

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

        elif pipeline_type == PipelineTypes.INPAINT:
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

    estimated_memory_usage = _estimate_pipeline_memory_use(
        model_path=model_path,
        revision=revision,
        variant=variant,
        subfolder=subfolder,
        vae_uri=vae_uri,
        lora_uris=lora_uris,
        textual_inversion_uris=textual_inversion_uris,
        safety_checker=True,  # it is always going to get loaded, monkey patched out currently
        auth_token=auth_token,
        extra_args=extra_args,
        local_files_only=local_files_only
    )

    _messages.debug_log(
        f'Creating Torch Pipeline: "{pipeline_class.__name__}", '
        f'Estimated CPU Side Memory Use: {_memory.bytes_best_human_unit(estimated_memory_usage)}')

    enforce_pipeline_cache_constraints(
        new_pipeline_size=estimated_memory_usage)

    # Block invalid Textual Inversion and LoRA usage

    if textual_inversion_uris:
        if model_type == ModelTypes.TORCH_UPSCALER_X2:
            raise NotImplementedError(
                'Model type torch-upscaler-x2 cannot be used with textual inversion models.')

        if isinstance(textual_inversion_uris, str):
            textual_inversion_uris = [textual_inversion_uris]

    if lora_uris:
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
    parsed_vae_uri = None

    if scheduler is None or not _scheduler_is_help(scheduler):
        # prevent waiting on VAE load just to get the scheduler
        # help message for the main model

        if vae_uri is not None:
            parsed_vae_uri = parse_torch_vae_uri(vae_uri)

            creation_kwargs['vae'] = \
                parsed_vae_uri.load(
                    dtype_fallback=dtype,
                    use_auth_token=auth_token,
                    local_files_only=local_files_only)

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
                                                  dtype_fallback=dtype,
                                                  local_files_only=local_files_only)

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
        if subfolder is not None:
            raise NotImplementedError('Single file model loads do not support the subfolder option.')
        pipeline = pipeline_class.from_single_file(model_path,
                                                   revision=revision,
                                                   variant=variant,
                                                   torch_dtype=torch_dtype,
                                                   use_safe_tensors=model_path.endswith('.safetensors'),
                                                   local_files_only=local_files_only,
                                                   **creation_kwargs)
    else:
        pipeline = pipeline_class.from_pretrained(model_path,
                                                  revision=revision,
                                                  variant=variant,
                                                  torch_dtype=torch_dtype,
                                                  subfolder=subfolder,
                                                  use_auth_token=auth_token,
                                                  local_files_only=local_files_only,
                                                  **creation_kwargs)

    # Select Scheduler

    _load_scheduler(pipeline=pipeline,
                    model_path=model_path,
                    scheduler_name=scheduler)

    # Textual Inversions and LoRAs

    parsed_textual_inversion_uris = []
    parsed_lora_uris = []

    if textual_inversion_uris:
        for inversion_uri in textual_inversion_uris:
            parsed = parse_textual_inversion_uri(inversion_uri)
            parsed_textual_inversion_uris.append(parsed)
            parsed.load_on_pipeline(pipeline,
                                    use_auth_token=auth_token,
                                    local_files_only=local_files_only)

    if lora_uris:
        for lora_uri in lora_uris:
            parsed = parse_lora_uri(lora_uri)
            parsed_lora_uris.append(parsed)
            parsed.load_on_pipeline(pipeline,
                                    use_auth_token=auth_token,
                                    local_files_only=local_files_only)

    # Safety Checker

    _set_torch_safety_checker(pipeline, safety_checker)

    # Model Offloading

    # Tag the pipeline with our own attributes
    pipeline.DGENERATE_SEQUENTIAL_OFFLOAD = sequential_cpu_offload
    pipeline.DGENERATE_CPU_OFFLOAD = model_cpu_offload

    if sequential_cpu_offload and 'cuda' in device:
        pipeline.enable_sequential_cpu_offload(device=device)
    elif model_cpu_offload and 'cuda' in device:
        pipeline.enable_model_cpu_offload(device=device)

    global _PIPELINE_CACHE_SIZE
    _PIPELINE_CACHE_SIZE += estimated_memory_usage

    # Tag for internal use

    pipeline.DGENERATE_SIZE_ESTIMATE = estimated_memory_usage

    _messages.debug_log(f'Finished Creating Torch Pipeline: "{pipeline_class.__name__}"')

    return TorchPipelineCreationResult(
        pipeline=pipeline,
        parsed_vae_uri=parsed_vae_uri,
        parsed_lora_uris=parsed_lora_uris,
        parsed_textual_inversion_uris=parsed_textual_inversion_uris,
        parsed_control_net_uris=parsed_control_net_uris
    )


class FlaxPipelineCreationResult:
    pipeline: diffusers.FlaxDiffusionPipeline
    flax_params: typing.Dict[str, typing.Any]
    parsed_vae_uri: typing.Optional[FlaxVAEUri]
    flax_vae_params: typing.Optional[typing.Dict[str, typing.Any]]
    parsed_control_net_uris: typing.List[FlaxControlNetUri]
    flax_control_net_params: typing.Optional[typing.Dict[str, typing.Any]]

    def __init__(self,
                 pipeline: diffusers.FlaxDiffusionPipeline,
                 flax_params: typing.Dict[str, typing.Any],
                 parsed_vae_uri: typing.Optional[FlaxVAEUri],
                 flax_vae_params: typing.Optional[typing.Dict[str, typing.Any]],
                 parsed_control_net_uris: typing.List[FlaxControlNetUri],
                 flax_control_net_params: typing.Optional[typing.Dict[str, typing.Any]]):
        self.pipeline = pipeline
        self.flax_params = flax_params
        self.parsed_control_net_uris = parsed_control_net_uris
        self.parsed_vae_uri = parsed_vae_uri
        self.flax_vae_params = flax_vae_params
        self.flax_control_net_params = flax_control_net_params


@_memoize(_FLAX_PIPELINE_CACHE,
          exceptions={'local_files_only'},
          hasher=lambda args: _d_memoize.args_cache_key(args,
                                                        {'vae_uri': _uri_hash_with_parser(parse_flax_vae_uri),
                                                         'control_net_uris':
                                                             _uri_list_hash_with_parser(
                                                                 parse_flax_control_net_uri)}),
          on_hit=lambda key, hit: _simple_cache_hit_debug("Flax Pipeline", key, hit.pipeline),
          on_create=lambda key, new: _simple_cache_miss_debug('Flax Pipeline', key, new.pipeline))
def create_flax_diffusion_pipeline(pipeline_type: PipelineTypes,
                                   model_path: str,
                                   revision: _types.OptionalString = None,
                                   subfolder: _types.OptionalString = None,
                                   dtype: DataTypes = DataTypes.AUTO,
                                   vae_uri: _types.OptionalString = None,
                                   control_net_uris: _types.OptionalString = None,
                                   scheduler: _types.OptionalString = None,
                                   safety_checker: bool = False,
                                   auth_token: _types.OptionalString = None,
                                   extra_args: typing.Optional[typing.Dict[str, typing.Any]] = None,
                                   local_files_only: bool = False) -> FlaxPipelineCreationResult:
    has_control_nets = False
    if control_net_uris:
        if len(control_net_uris) > 1:
            raise NotImplementedError('Flax does not support multiple --control-nets.')
        if len(control_net_uris) == 1:
            has_control_nets = True

    if pipeline_type == PipelineTypes.TXT2IMG:
        if has_control_nets:
            pipeline_class = diffusers.FlaxStableDiffusionControlNetPipeline
        else:
            pipeline_class = diffusers.FlaxStableDiffusionPipeline
    elif pipeline_type == PipelineTypes.IMG2IMG:
        if has_control_nets:
            raise NotImplementedError('Flax does not support img2img mode with --control-nets.')
        pipeline_class = diffusers.FlaxStableDiffusionImg2ImgPipeline
    elif pipeline_type == PipelineTypes.INPAINT:
        if has_control_nets:
            raise NotImplementedError('Flax does not support inpaint mode with --control-nets.')
        pipeline_class = diffusers.FlaxStableDiffusionInpaintPipeline
    else:
        raise NotImplementedError('Pipeline type not implemented.')

    estimated_memory_usage = _estimate_pipeline_memory_use(
        model_path=model_path,
        revision=revision,
        subfolder=subfolder,
        vae_uri=vae_uri,
        safety_checker=True,  # it is always going to get loaded, monkey patched out currently
        auth_token=auth_token,
        extra_args=extra_args,
        local_files_only=local_files_only,
        flax=True
    )

    _messages.debug_log(
        f'Creating Flax Pipeline: "{pipeline_class.__name__}", '
        f'Estimated CPU Side Memory Use: {_memory.bytes_best_human_unit(estimated_memory_usage)}')

    enforce_pipeline_cache_constraints(
        new_pipeline_size=estimated_memory_usage)

    kwargs = {}
    vae_params = None
    control_net_params = None

    flax_dtype = _get_flax_dtype(dtype)

    parsed_control_net_uris = []
    parsed_flax_vae_uri = None

    if scheduler is None or not _scheduler_is_help(scheduler):
        # prevent waiting on VAE load just get the scheduler
        # help message for the main model

        if vae_uri is not None:
            parsed_flax_vae_uri = parse_flax_vae_uri(vae_uri)

            kwargs['vae'], vae_params = parsed_flax_vae_uri.load(
                dtype_fallback=dtype,
                use_auth_token=auth_token,
                local_files_only=local_files_only)
            _messages.debug_log(lambda:
                                f'Added Flax VAE: "{vae_uri}" to pipeline: "{pipeline_class.__name__}"')

    if control_net_uris:
        control_net_uri = control_net_uris[0]

        parsed_flax_control_net_uri = parse_flax_control_net_uri(control_net_uri)

        parsed_control_net_uris.append(parsed_flax_control_net_uri)

        control_net, control_net_params = parsed_flax_control_net_uri \
            .load(use_auth_token=auth_token,
                  dtype_fallback=dtype,
                  local_files_only=local_files_only)

        _messages.debug_log(lambda:
                            f'Added Flax ControlNet: "{control_net_uri}" '
                            f'to pipeline: "{pipeline_class.__name__}"')

        kwargs['controlnet'] = control_net

    if extra_args is not None:
        kwargs.update(extra_args)

    pipeline, params = pipeline_class.from_pretrained(model_path,
                                                      revision=revision,
                                                      dtype=flax_dtype,
                                                      subfolder=subfolder,
                                                      use_auth_token=auth_token,
                                                      local_files_only=local_files_only,
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

    global _PIPELINE_CACHE_SIZE
    _PIPELINE_CACHE_SIZE += estimated_memory_usage

    # Tag for internal use

    pipeline.DGENERATE_SIZE_ESTIMATE = estimated_memory_usage

    _messages.debug_log(f'Finished Creating Flax Pipeline: "{pipeline_class.__name__}"')

    return FlaxPipelineCreationResult(
        pipeline=pipeline,
        flax_params=params,
        parsed_vae_uri=parsed_flax_vae_uri,
        flax_vae_params=vae_params,
        parsed_control_net_uris=parsed_control_net_uris,
        flax_control_net_params=control_net_params
    )


class PipelineWrapperResult:
    """
    The result of calling :py:class:`.DiffusionPipelineWrapper`
    """
    images: typing.Optional[typing.List[PIL.Image.Image]]

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
        self.dgenerate_opts = list()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.images is not None:
            for i in self.images:
                if i is not None:
                    i.close()
                    self.images = None


class DiffusionArguments:
    """
    Represents all possible arguments for a :py:class:`.DiffusionPipelineWrapper` call.
    """
    prompt: _types.OptionalPrompt = None
    image: typing.Optional[PIL.Image.Image] = None
    mask_image: typing.Optional[PIL.Image.Image] = None
    control_images: typing.Optional[typing.List[PIL.Image.Image]] = None
    width: _types.OptionalSize = None
    height: _types.OptionalSize = None
    batch_size: _types.OptionalInteger = None
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

    def get_pipeline_wrapper_kwargs(self):
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
            prompt_val = \
                _textprocessing.wrap(
                    prompt_val,
                    width=prompt_wrap_width - len(header),
                    subsequent_indent=' ' * len(header))
            prompt_format.append(f'{header}"{prompt_val}"')

        prompt_val = prompt.negative
        if prompt_val:
            header = f'{neg_title}: '
            prompt_val = \
                _textprocessing.wrap(
                    prompt_val,
                    width=prompt_wrap_width - len(header),
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
                 auth_token: _types.OptionalString = None,
                 local_files_only: bool = False):

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
        self._local_files_only = local_files_only

        if sdxl_refiner_uri is not None:
            parsed_uri = parse_sdxl_refiner_uri(sdxl_refiner_uri)
            self._sdxl_refiner_uri = parsed_uri.model
            self._sdxl_refiner_revision = parsed_uri.revision
            self._sdxl_refiner_variant = parsed_uri.variant
            self._sdxl_refiner_dtype = parsed_uri.dtype
            self._sdxl_refiner_subfolder = parsed_uri.subfolder

        if lora_uris:
            if model_type == 'flax':
                raise NotImplementedError('LoRA loading is not implemented for flax.')

            if not isinstance(lora_uris, str):
                raise NotImplementedError('Using multiple LoRA models is currently not supported.')

            self._lora_scale = parse_lora_uri(lora_uris).scale

    @staticmethod
    def _pipeline_to_cpu_update_cache_info(pipeline):

        # Update CPU side memory overhead estimates when
        # a pipeline gets casted back to the CPU

        global _PIPELINE_CACHE_SIZE, \
            _VAE_CACHE_SIZE, \
            _CONTROL_NET_CACHE_SIZE

        enforce_pipeline_cache_constraints(pipeline.DGENERATE_SIZE_ESTIMATE)
        _PIPELINE_CACHE_SIZE += pipeline.DGENERATE_SIZE_ESTIMATE

        _messages.debug_log(f'{_types.class_and_id_string(pipeline)} '
                            f'Size = {pipeline.DGENERATE_SIZE_ESTIMATE} Bytes '
                            f'is entering CPU side memory, {_types.fullname(pipeline_cache_size)}() '
                            f'is now {pipeline_cache_size()} Bytes')

        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            if hasattr(pipeline.vae, 'DGENERATE_SIZE_ESTIMATE'):
                # VAE returning to CPU side memory
                enforce_vae_cache_constraints(pipeline.vae.DGENERATE_SIZE_ESTIMATE)
                _VAE_CACHE_SIZE += pipeline.vae.DGENERATE_SIZE_ESTIMATE

                _messages.debug_log(f'{_types.class_and_id_string(pipeline.vae)} '
                                    f'Size = {pipeline.vae.DGENERATE_SIZE_ESTIMATE} Bytes '
                                    f'is entering CPU side memory, {_types.fullname(vae_cache_size)}() '
                                    f'is now {vae_cache_size()} Bytes')

        if hasattr(pipeline, 'controlnet') and pipeline.controlnet is not None:
            if isinstance(pipeline.controlnet,
                          diffusers.pipelines.controlnet.MultiControlNetModel):
                total_size = 0
                for control_net in pipeline.controlnet.nets:
                    total_size += control_net.DGENERATE_SIZE_ESTIMATE

                    _messages.debug_log(f'{_types.class_and_id_string(control_net)} '
                                        f'Size = {control_net.DGENERATE_SIZE_ESTIMATE} Bytes '
                                        f'from "MultiControlNetModel" is entering CPU side memory.')

                enforce_control_net_cache_constraints(total_size)
                _CONTROL_NET_CACHE_SIZE += total_size

                _messages.debug_log(f'"MultiControlNetModel" size fully estimated, '
                                    f'{_types.fullname(control_net_cache_size)}() '
                                    f'is now {control_net_cache_size()} Bytes')

            else:
                # ControlNet returning to CPU side memory
                enforce_control_net_cache_constraints(pipeline.controlnet.DGENERATE_SIZE_ESTIMATE)
                _CONTROL_NET_CACHE_SIZE += pipeline.controlnet.DGENERATE_SIZE_ESTIMATE

                _messages.debug_log(f'{_types.class_and_id_string(pipeline.controlnet)} '
                                    f'Size = {pipeline.controlnet.DGENERATE_SIZE_ESTIMATE} Bytes '
                                    f'is entering CPU side memory, {_types.fullname(control_net_cache_size)}() '
                                    f'is now {control_net_cache_size()} Bytes')

    @staticmethod
    def _pipeline_off_cpu_update_cache_info(pipeline):

        # Update CPU side memory overhead estimates when a
        # pipeline gets casted to the GPU or some other
        # processing device

        # It is not the case that once the model is casted off the CPU
        # that there is no CPU side memory overhead for that model,
        # in fact there is usually a non insignificant amount of system
        # ram still used when the model is casted to another device. This
        # function does not account for that and assumes that once the model is
        # casted off of the CPU, that all of its memory overhead is on the
        # other device IE: VRAM or some other tensor processing unit.
        # This is not entirely correct but probably close enough for the
        # CPU side memory overhead estimates, and for use as a heuristic
        # value for garbage collection of the model cache that exists
        # in CPU side memory

        global _PIPELINE_CACHE_SIZE, \
            _VAE_CACHE_SIZE, \
            _CONTROL_NET_CACHE_SIZE

        _PIPELINE_CACHE_SIZE -= pipeline.DGENERATE_SIZE_ESTIMATE

        _messages.debug_log(f'{_types.class_and_id_string(pipeline)} '
                            f'Size = {pipeline.DGENERATE_SIZE_ESTIMATE} Bytes '
                            f'is leaving CPU side memory, {_types.fullname(pipeline_cache_size)}() '
                            f'is now {pipeline_cache_size()} Bytes')

        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            if hasattr(pipeline.vae, 'DGENERATE_SIZE_ESTIMATE'):
                _VAE_CACHE_SIZE -= pipeline.vae.DGENERATE_SIZE_ESTIMATE

                _messages.debug_log(f'{_types.class_and_id_string(pipeline.vae)} '
                                    f'Size = {pipeline.vae.DGENERATE_SIZE_ESTIMATE} Bytes '
                                    f'is leaving CPU side memory, {_types.fullname(vae_cache_size)}() '
                                    f'is now {vae_cache_size()} Bytes')

        if hasattr(pipeline, 'controlnet') and pipeline.controlnet is not None:
            if isinstance(pipeline.controlnet,
                          diffusers.pipelines.controlnet.MultiControlNetModel):
                for control_net in pipeline.controlnet.nets:
                    _CONTROL_NET_CACHE_SIZE -= control_net.DGENERATE_SIZE_ESTIMATE

                    _messages.debug_log(f'{_types.class_and_id_string(control_net)} Size = '
                                        f'{control_net.DGENERATE_SIZE_ESTIMATE} Bytes '
                                        f'from "MultiControlNetModel" is leaving CPU side memory, '
                                        f'{_types.fullname(control_net_cache_size)}() is now '
                                        f'{control_net_cache_size()} Bytes')

            elif isinstance(pipeline.controlnet, diffusers.models.ControlNetModel):
                _CONTROL_NET_CACHE_SIZE -= pipeline.controlnet.DGENERATE_SIZE_ESTIMATE

                _messages.debug_log(f'{_types.class_and_id_string(pipeline.controlnet)} '
                                    f'Size = {pipeline.controlnet.DGENERATE_SIZE_ESTIMATE} Bytes '
                                    f'is leaving CPU side memory, {_types.fullname(control_net_cache_size)}() '
                                    f'is now {control_net_cache_size()} Bytes')

    @staticmethod
    def _pipeline_to(pipeline, device):
        if hasattr(pipeline, 'to'):
            if not pipeline.DGENERATE_CPU_OFFLOAD and \
                    not pipeline.DGENERATE_SEQUENTIAL_OFFLOAD:

                if device == 'cpu':
                    DiffusionPipelineWrapper._pipeline_to_cpu_update_cache_info(pipeline)
                else:
                    DiffusionPipelineWrapper._pipeline_off_cpu_update_cache_info(pipeline)
                try:
                    return pipeline.to(device)
                except RuntimeError as e:
                    if 'memory' in str(e).lower():
                        raise OutOfMemoryError(e)
                    raise e
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
    def local_files_only(self) -> bool:
        """
        Currently set value for **local_files_only**

        :return:
        """
        return self._local_files_only

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
        List of supplied --textual-inversions uri strings or None
        """
        return [self._textual_inversion_uris] if \
            isinstance(self._textual_inversion_uris, str) else self._textual_inversion_uris

    @property
    def control_net_uris(self) -> _types.OptionalUris:
        """
        List of supplied --control-nets uri strings or None
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
        Currently set ``--model-type`` enum value
        """
        return self._model_type

    @property
    def model_type_string(self) -> str:
        """
        Currently set ``--model-type`` string value
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
        Selected --vae uri for the main model or None
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
        List of supplied --lora uri strings or None
        """
        return [self._lora_uris] if \
            isinstance(self._lora_uris, str) else self._lora_uris

    @property
    def auth_token(self) -> _types.OptionalString:
        """
        Current --auth-token value or None
        """
        return self._auth_token

    def reconstruct_dgenerate_opts(self,
                                   args: typing.Optional[DiffusionArguments] = None,
                                   extra_opts: typing.Optional[
                                       typing.List[
                                           typing.Union[typing.Tuple[str], typing.Tuple[str, typing.Any]]]] = None,
                                   shell_quote=True,
                                   **kwargs) -> \
            typing.List[typing.Union[typing.Tuple[str], typing.Tuple[str, typing.Any]]]:
        """
        Reconstruct dgenerates command line arguments from a particular set of pipeline wrapper call arguments.

        :param args: :py:class:`.DiffusionArguments` object to take values from

        :param extra_opts: Extra option pairs to be added to the end of reconstructed options,
            this should be a list of tuples of length 1 (switch only) or length 2 (switch with args)

        :param shell_quote: Shell quote and format the argument values? or return them raw.

        :param kwargs: pipeline wrapper keyword arguments, these will override values derived from
            any :py:class:`.DiffusionArguments` object given to the *args* argument. See:
            :py:class:`.DiffusionArguments.get_pipeline_wrapper_kwargs`

        :return: List of tuples of length 1 or 2 representing the option
        """

        if args is not None:
            kwargs = args.get_pipeline_wrapper_kwargs() | (kwargs if kwargs else dict())

        def _format_size(val):
            if val is None:
                return None

            return f'{val[0]}x{val[1]}'

        batch_size: int = kwargs.get('batch_size', None)
        prompt: _prompt.Prompt = kwargs.get('prompt', None)
        sdxl_second_prompt: _prompt.Prompt = kwargs.get('sdxl_second_prompt', None)
        sdxl_refiner_prompt: _prompt.Prompt = kwargs.get('sdxl_refiner_prompt', None)
        sdxl_refiner_second_prompt: _prompt.Prompt = kwargs.get('sdxl_refiner_second_prompt', None)

        image = kwargs.get('image', None)
        control_images = kwargs.get('control_images', None)
        image_seed_strength = kwargs.get('image_seed_strength', None)
        upscaler_noise_level = kwargs.get('upscaler_noise_level', None)
        mask_image = kwargs.get('mask_image', None)
        seed = kwargs.get('seed')
        width = kwargs.get('width', None)
        height = kwargs.get('height', None)
        inference_steps = kwargs.get('inference_steps')
        guidance_scale = kwargs.get('guidance_scale')
        guidance_rescale = kwargs.get('guidance_rescale')
        image_guidance_scale = kwargs.get('image_guidance_scale')

        sdxl_refiner_inference_steps = kwargs.get('sdxl_refiner_inference_steps')
        sdxl_refiner_guidance_scale = kwargs.get('sdxl_refiner_guidance_scale')
        sdxl_refiner_guidance_rescale = kwargs.get('sdxl_refiner_guidance_rescale')

        sdxl_high_noise_fraction = kwargs.get('sdxl_high_noise_fraction', None)
        sdxl_aesthetic_score = kwargs.get('sdxl_aesthetic_score', None)

        sdxl_original_size = \
            _format_size(kwargs.get('sdxl_original_size', None))
        sdxl_target_size = \
            _format_size(kwargs.get('sdxl_target_size', None))
        sdxl_crops_coords_top_left = \
            _format_size(kwargs.get('sdxl_crops_coords_top_left', None))

        sdxl_negative_aesthetic_score = kwargs.get('sdxl_negative_aesthetic_score', None)

        sdxl_negative_original_size = \
            _format_size(kwargs.get('sdxl_negative_original_size', None))
        sdxl_negative_target_size = \
            _format_size(kwargs.get('sdxl_negative_target_size', None))
        sdxl_negative_crops_coords_top_left = \
            _format_size(kwargs.get('sdxl_negative_crops_coords_top_left', None))

        sdxl_refiner_aesthetic_score = kwargs.get('sdxl_refiner_aesthetic_score', None)

        sdxl_refiner_original_size = \
            _format_size(kwargs.get('sdxl_refiner_original_size', None))
        sdxl_refiner_target_size = \
            _format_size(kwargs.get('sdxl_refiner_target_size', None))
        sdxl_refiner_crops_coords_top_left = \
            _format_size(kwargs.get('sdxl_refiner_crops_coords_top_left', None))

        sdxl_refiner_negative_aesthetic_score = kwargs.get('sdxl_refiner_negative_aesthetic_score', None)

        sdxl_refiner_negative_original_size = \
            _format_size(kwargs.get('sdxl_refiner_negative_original_size', None))
        sdxl_refiner_negative_target_size = \
            _format_size(kwargs.get('sdxl_refiner_negative_target_size', None))
        sdxl_refiner_negative_crops_coords_top_left = \
            _format_size(kwargs.get('sdxl_refiner_negative_crops_coords_top_left', None))

        opts = [(self.model_path,),
                ('--model-type', self.model_type_string),
                ('--dtype', self.dtype_string),
                ('--device', self._device),
                ('--inference-steps', inference_steps),
                ('--guidance-scales', guidance_scale),
                ('--seeds', seed)]

        if batch_size is not None and batch_size > 1:
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

        if self._lora_uris:
            opts.append(('--lora', self._lora_uris))

        if self._textual_inversion_uris:
            opts.append(('--textual-inversions', self._textual_inversion_uris))

        if self._control_net_uris:
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
            seed_args = []

            if mask_image is not None:
                seed_args.append(f'mask={_image.get_filename(mask_image)}')
            if control_images:
                seed_args.append(f'control={", ".join(_image.get_filename(c) for c in control_images)}')

            if isinstance(image, list):
                opts.append(('--image-seeds',
                             ','.join(_image.get_filename(i) for i in image)))
            elif image:
                if not seed_args:
                    opts.append(('--image-seeds',
                                 _image.get_filename(image)))
                else:
                    opts.append(('--image-seeds',
                                 _image.get_filename(image) + ';' + ';'.join(seed_args)))

            if upscaler_noise_level is not None:
                opts.append(('--upscaler-noise-levels', upscaler_noise_level))

            if image_seed_strength is not None:
                opts.append(('--image-seed-strengths', image_seed_strength))

        elif control_images:
            opts.append(('--image-seeds',
                         ', '.join(_image.get_filename(c) for c in control_images)))

        if extra_opts:
            opts += extra_opts

        if shell_quote:
            for idx, option in enumerate(opts):
                if len(option) > 1:
                    name, value = option
                    if isinstance(value, (str, _prompt.Prompt)):
                        opts[idx] = (name, shlex.quote(str(value)))
                    elif isinstance(value, tuple):
                        opts[idx] = (name, _textprocessing.format_size(value))
                    else:
                        opts[idx] = (name, str(value))
                else:
                    solo_val = str(option[0])
                    if not solo_val.startswith('-'):
                        # not a solo switch option, some value
                        opts[idx] = (shlex.quote(solo_val),)

        return opts

    @staticmethod
    def _set_opt_value_syntax(val):
        if isinstance(val, tuple):
            return _textprocessing.format_size(val)
        if isinstance(val, list):
            return ' '.join(DiffusionPipelineWrapper._set_opt_value_syntax(v) for v in val)
        return shlex.quote(str(val))

    @staticmethod
    def _format_option_pair(val):
        if len(val) > 1:
            opt_name, opt_value = val

            if isinstance(opt_value, _prompt.Prompt):
                header_len = len(opt_name) + 2
                prompt_text = \
                    _textprocessing.wrap(
                        shlex.quote(str(opt_value)),
                        subsequent_indent=' ' * header_len,
                        width=75)

                prompt_text = ' \\\n'.join(prompt_text.split('\n'))
                return f'{opt_name} {prompt_text}'

            return f'{opt_name} {DiffusionPipelineWrapper._set_opt_value_syntax(opt_value)}'

        solo_val = str(val[0])

        if solo_val.startswith('-'):
            return solo_val

        # Not a switch option, some value
        return shlex.quote(solo_val)

    def gen_dgenerate_config(self,
                             args: typing.Optional[DiffusionArguments] = None,
                             extra_opts: typing.Optional[
                                 typing.List[typing.Union[typing.Tuple[str], typing.Tuple[str, typing.Any]]]] = None,
                             extra_comments: typing.Optional[typing.Sequence[str]] = None,
                             **kwargs):
        """
        Generate a valid dgenerate config file with a single invocation that reproduces this result.

        :param args: :py:class:`.DiffusionArguments` object to take values from
        :param extra_comments: Extra strings to use as comments after the initial
            version check directive
        :param extra_opts: Extra option pairs to be added to the end of reconstructed options
            of the dgenerate invocation, this should be a list of tuples of length 1 (switch only)
            or length 2 (switch with args)
        :param kwargs: pipeline wrapper keyword arguments, these will override values derived from
            any :py:class:`.DiffusionArguments` object given to the *args* argument. See:
            :py:class:`.DiffusionArguments.get_pipeline_wrapper_kwargs`
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

        opts = \
            self.reconstruct_dgenerate_opts(args, **kwargs, shell_quote=False) + \
            (extra_opts if extra_opts else [])

        for opt in opts[:-1]:
            config += f'{self._format_option_pair(opt)} \\\n'

        last = opts[-1]

        if len(last) == 2:
            config += self._format_option_pair(last)

        return config

    def gen_dgenerate_command(self,
                              args: typing.Optional[DiffusionArguments] = None,
                              extra_opts: typing.Optional[
                                  typing.List[typing.Union[typing.Tuple[str], typing.Tuple[str, typing.Any]]]] = None,
                              **kwargs):
        """
        Generate a valid dgenerate command line invocation that reproduces this result.

        :param args: :py:class:`.DiffusionArguments` object to take values from
        :param extra_opts: Extra option pairs to be added to the end of reconstructed options
            of the dgenerate invocation, this should be a list of tuples of length 1 (switch only)
            or length 2 (switch with args)
        :param kwargs: pipeline wrapper keyword arguments, these will override values derived from
            any :py:class:`.DiffusionArguments` object given to the *args* argument. See:
            :py:class:`.DiffusionArguments.get_pipeline_wrapper_kwargs`
        :return: A string containing the dgenerate command line needed to reproduce this result.
        """

        opt_string = \
            ' '.join(f"{self._format_option_pair(opt)}"
                     for opt in self.reconstruct_dgenerate_opts(args, **kwargs,
                                                                shell_quote=False) + extra_opts)

        return f'dgenerate {opt_string}'

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

        if self._control_net_uris:
            control_images = user_args.get('control_images')

            if not control_images:
                raise ValueError(
                    'Must provide control_images argument when using ControlNet models.')

            control_images_cnt = len(control_images)
            control_net_uris_cnt = len(self._control_net_uris)

            if control_images_cnt < control_net_uris_cnt:
                # Pad it out so that the last image mentioned is used
                # for the rest of the controlnets specified

                for i in range(0, control_net_uris_cnt - control_images_cnt):
                    control_images.append(control_images[-1])

            elif control_images_cnt > control_net_uris_cnt:
                # User provided too many control_images, behavior is undefined.

                raise ValueError(
                    f'You specified {control_images_cnt} control image sources and '
                    f'only {control_net_uris_cnt} ControlNet URIs. The amount of '
                    f'control images must be less than or equal to the amount of ControlNet URIs.')

            # They should always be of equal dimension, anything
            # else results in an error down the line.
            args['width'] = user_args.get('width', control_images[0].width)
            args['height'] = user_args.get('height', control_images[0].height)

            if self._pipeline_type == PipelineTypes.TXT2IMG:
                args['image'] = control_images
            elif self._pipeline_type == PipelineTypes.IMG2IMG or \
                    self._pipeline_type == PipelineTypes.INPAINT:
                args['image'] = user_args['image']
                args['control_image'] = control_images
                set_strength()

            mask_image = user_args.get('mask_image')
            if mask_image is not None:
                args['mask_image'] = mask_image

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
        # Only works with txt2image

        device_count = jax.device_count()

        pipe: diffusers.FlaxStableDiffusionControlNetPipeline = self._pipeline

        default_args['prng_seed'] = jax.random.split(jax.random.PRNGKey(user_args.get('seed', 0)), device_count)
        prompt_ids = pipe.prepare_text_inputs([positive_prompt] * device_count)

        if negative_prompt is not None:
            negative_prompt_ids = pipe.prepare_text_inputs([negative_prompt] * device_count)
        else:
            negative_prompt_ids = None

        control_net_image = default_args.get('image')
        if isinstance(control_net_image, list):
            control_net_image = control_net_image[0]

        processed_image = pipe.prepare_image_inputs([control_net_image] * device_count)
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
                    f'{get_pipeline_type_string(self._pipeline_type)} mode with the current '
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
                return False

        self._pipeline_type = pipeline_type

        if model_type_is_sdxl(self._model_type) and self._textual_inversion_uris:
            raise NotImplementedError('Textual inversion not supported for SDXL.')

        if self._model_type == ModelTypes.FLAX:
            if not have_jax_flax():
                raise NotImplementedError('flax and jax are not installed.')

            if self._textual_inversion_uris:
                raise NotImplementedError('Textual inversion not supported for flax.')

            if self._pipeline_type != PipelineTypes.TXT2IMG and self._control_net_uris:
                raise NotImplementedError('Inpaint and Img2Img not supported for flax with ControlNet.')

            if self._vae_tiling or self._vae_slicing:
                raise NotImplementedError('--vae-tiling/--vae-slicing not supported for flax.')

            creation_result = \
                create_flax_diffusion_pipeline(pipeline_type,
                                               self._model_path,
                                               revision=self._revision,
                                               dtype=self._dtype,
                                               vae_uri=self._vae_uri,
                                               control_net_uris=self._control_net_uris,
                                               scheduler=self._scheduler,
                                               safety_checker=self._safety_checker,
                                               auth_token=self._auth_token,
                                               local_files_only=self._local_files_only)

            self._pipeline = creation_result.pipeline
            self._flax_params = creation_result.flax_params
            self._parsed_control_net_uris = creation_result.parsed_control_net_uris

        elif self._sdxl_refiner_uri is not None:
            if not model_type_is_sdxl(self._model_type):
                raise NotImplementedError('Only Stable Diffusion XL models support refiners, '
                                          'please use --model-type torch-sdxl if you are trying to load an sdxl model.')

            if not _scheduler_is_help(self._sdxl_refiner_scheduler):
                # Don't load this up if were just going to be getting
                # information about compatible schedulers for the refiner
                creation_result = \
                    create_torch_diffusion_pipeline(pipeline_type,
                                                    self._model_type,
                                                    self._model_path,
                                                    subfolder=self._model_subfolder,
                                                    revision=self._revision,
                                                    variant=self._variant,
                                                    dtype=self._dtype,
                                                    vae_uri=self._vae_uri,
                                                    lora_uris=self._lora_uris,
                                                    control_net_uris=self._control_net_uris,
                                                    scheduler=self._scheduler,
                                                    safety_checker=self._safety_checker,
                                                    auth_token=self._auth_token,
                                                    device=self._device,
                                                    local_files_only=self._local_files_only)
                self._pipeline = creation_result.pipeline
                self._parsed_control_net_uris = creation_result.parsed_control_net_uris

            refiner_pipeline_type = PipelineTypes.IMG2IMG if pipeline_type is PipelineTypes.TXT2IMG else pipeline_type

            if self._pipeline is not None:
                refiner_extra_args = {'vae': self._pipeline.vae,
                                      'text_encoder_2': self._pipeline.text_encoder_2}
            else:
                refiner_extra_args = None

            creation_result = \
                create_torch_diffusion_pipeline(refiner_pipeline_type,
                                                ModelTypes.TORCH_SDXL,
                                                self._sdxl_refiner_uri,
                                                subfolder=self._sdxl_refiner_subfolder,
                                                revision=self._sdxl_refiner_revision,

                                                variant=self._sdxl_refiner_variant if
                                                self._sdxl_refiner_variant is not None else self._variant,

                                                dtype=self._sdxl_refiner_dtype if
                                                self._sdxl_refiner_dtype is not None else self._dtype,

                                                scheduler=self._scheduler if
                                                self._sdxl_refiner_scheduler is None else self._sdxl_refiner_scheduler,

                                                safety_checker=self._safety_checker,
                                                auth_token=self._auth_token,
                                                extra_args=refiner_extra_args,
                                                local_files_only=self._local_files_only)
            self._sdxl_refiner_pipeline = creation_result.pipeline
        else:
            offload = self._control_net_uris and self._model_type == ModelTypes.TORCH_SDXL

            creation_result = \
                create_torch_diffusion_pipeline(pipeline_type,
                                                self._model_type,
                                                self._model_path,
                                                subfolder=self._model_subfolder,
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
                                                sequential_cpu_offload=offload,
                                                local_files_only=self._local_files_only)
            self._pipeline = creation_result.pipeline
            self._parsed_control_net_uris = creation_result.parsed_control_net_uris

        set_vae_slicing_tiling(pipeline=self._pipeline,
                               vae_tiling=self._vae_tiling,
                               vae_slicing=self._vae_slicing)

        if self._sdxl_refiner_pipeline is not None:
            set_vae_slicing_tiling(pipeline=self._sdxl_refiner_pipeline,
                                   vae_tiling=self._vae_tiling,
                                   vae_slicing=self._vae_slicing)

        return True

    @staticmethod
    def _determine_pipeline_type(kwargs):
        if 'image' in kwargs and 'mask_image' in kwargs:
            # Inpainting is handled by INPAINT type
            return PipelineTypes.INPAINT

        if 'image' in kwargs:
            # Image only is handled by IMG2IMG type
            return PipelineTypes.IMG2IMG

        # All other situations handled by BASIC type
        return PipelineTypes.TXT2IMG

    def __call__(self, args: typing.Optional[DiffusionArguments] = None, **kwargs) -> PipelineWrapperResult:
        """
        Call the pipeline and generate a result.

        :param args: Optional :py:class:`.DiffusionArguments`

        :param kwargs: See :py:meth:`.DiffusionArguments.get_pipeline_wrapper_kwargs`,
            any keyword arguments given here will override values derived from the
            :py:class:`.DiffusionArguments` object given to the *args* parameter.

        :raises: :py:class:`.InvalidModelPathError`
            :py:class:`.InvalidSDXLRefinerUriError`
            :py:class:`.InvalidVaeUriError`
            :py:class:`.InvalidLoRAUriError`
            :py:class:`.InvalidControlNetUriError`
            :py:class:`.InvalidTextualInversionUriError`
            :py:class:`.InvalidSchedulerName`
            :py:class:`.OutOfMemoryError`
            :py:class:`NotImplementedError`

        :return: :py:class:`.PipelineWrapperResult`
        """

        if args is not None:
            kwargs = args.get_pipeline_wrapper_kwargs() | (kwargs if kwargs else dict())

        _messages.debug_log(f'Calling Pipeline Wrapper: "{self}"')
        _messages.debug_log(f'Pipeline Wrapper Args: ',
                            lambda: _textprocessing.debug_format_args(kwargs))

        enforce_cache_constraints()

        loaded_new = self._lazy_init_pipeline(
            DiffusionPipelineWrapper._determine_pipeline_type(kwargs))

        if loaded_new:
            enforce_cache_constraints()

        default_args = self._pipeline_defaults(kwargs)

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

        return result
