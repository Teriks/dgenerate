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
import enum
import typing

import torch

import dgenerate.types as _types

try:
    import jax
    import jaxlib
    import jax.numpy as jnp
except ImportError:
    jax = None
    flax = None
    jnp = None


class PipelineType(enum.Enum):
    """
    Represents possible diffusers pipeline types.
    """

    TXT2IMG = 1
    """
    Text to image mode. Prompt only generation.
    """

    IMG2IMG = 2
    """
    Image to image mode. 
    Generation seeded / controlled with an image in some fashion.
    """

    INPAINT = 3
    """
    Inpainting mode. 
    Generation seeded / controlled with an image and a mask in some fashion.
    """


def get_pipeline_type_enum(id_str: typing.Union[PipelineType, str, None]) -> PipelineType:
    """
    Get a :py:class:`.PipelineType` enum value from a string.

    :param id_str: one of: "txt2img", "img2img", or "inpaint"

    :raises ValueError: if an invalid string value (name) is passed

    :return: :py:class:`.PipelineType`
    """

    if isinstance(id_str, PipelineType):
        return id_str

    try:
        return {'txt2img': PipelineType.TXT2IMG,
                'img2img': PipelineType.IMG2IMG,
                'inpaint': PipelineType.INPAINT}[id_str.strip().lower()]
    except KeyError:
        raise ValueError('invalid PipelineType string')


def get_pipeline_type_string(pipeline_type_enum: PipelineType):
    """
    Convert a :py:class:`.PipelineType` enum value to a string.

    :param pipeline_type_enum: :py:class:`.PipelineType` value

    :return: one of: "txt2img", "img2img", or "inpaint"
    """
    pipeline_type = get_pipeline_type_enum(pipeline_type_enum)

    return {PipelineType.TXT2IMG: 'txt2img',
            PipelineType.IMG2IMG: 'img2img',
            PipelineType.INPAINT: 'inpaint'}[pipeline_type]


class DataType(enum.Enum):
    """
    Represents model precision
    """

    AUTO = 0
    """Auto selection."""

    FLOAT16 = 1
    """16 bit floating point."""

    FLOAT32 = 2
    """32 bit floating point."""


def supported_data_type_strings():
    """
    Return a list of supported ``--dtype`` strings
    """
    return ['auto', 'float16', 'float32']


def supported_data_type_enums() -> list[DataType]:
    """
    Return a list of supported :py:class:`.DataType` enum values
    """
    return [get_data_type_enum(i) for i in supported_data_type_strings()]


def get_data_type_enum(id_str: typing.Union[DataType, str, None]) -> DataType:
    """
    Convert a ``--dtype`` string to its :py:class:`.DataType` enum value

    :param id_str: ``--dtype`` string

    :raises ValueError: if an invalid string value (name) is passed

    :return: :py:class:`.DataType`
    """

    if isinstance(id_str, DataType):
        return id_str

    try:
        return {'auto': DataType.AUTO,
                'float16': DataType.FLOAT16,
                'float32': DataType.FLOAT32}[id_str.strip().lower()]
    except KeyError:
        raise ValueError('invalid DataType string')


def get_data_type_string(data_type_enum: DataType) -> str:
    """
    Convert a :py:class:`.DataType` enum value to its ``--dtype`` string

    :param data_type_enum: :py:class:`.DataType` value
    :return: ``--dtype`` string
    """

    model_type = get_data_type_enum(data_type_enum)

    return {DataType.AUTO: 'auto',
            DataType.FLOAT16: 'float16',
            DataType.FLOAT32: 'float32'}[model_type]


class ModelType(enum.Enum):
    """
    Enum representation of ``--model-type``
    """

    TORCH = 0
    """Stable Diffusion, such as SD 1.0 - 2.x"""

    TORCH_PIX2PIX = 1
    """Stable Diffusion pix2pix prompt guided editing."""

    TORCH_SDXL = 2
    """Stable Diffusion XL"""

    TORCH_IF = 3
    """Deep Floyd IF stage 1"""

    TORCH_IFS = 4
    """Deep Floyd IF superscaler (stage 2)"""

    TORCH_IFS_IMG2IMG = 5
    """Deep Floyd IF superscaler (stage 2) image to image / variation mode."""

    TORCH_SDXL_PIX2PIX = 6
    """Stable Diffusion XL pix2pix prompt guided editing."""

    TORCH_UPSCALER_X2 = 7
    """Stable Diffusion X2 upscaler"""

    TORCH_UPSCALER_X4 = 8
    """Stable Diffusion X4 upscaler"""

    FLAX = 9
    """
    Stable Diffusion, such as SD 1.0 - 2.x, with Flax / Jax parallelization.
    """


def supported_model_type_strings():
    """
    Return a list of supported ``--model-type`` strings
    """
    base_set = ['torch',
                'torch-pix2pix',
                'torch-sdxl',
                'torch-sdxl-pix2pix',
                'torch-upscaler-x2',
                'torch-upscaler-x4',
                'torch-if',
                'torch-ifs',
                'torch-ifs-img2img']

    if have_jax_flax():
        return base_set + ['flax']
    else:
        return base_set


def supported_model_type_enums() -> list[ModelType]:
    """
    Return a list of supported :py:class:`.ModelType` enum values
    """
    return [get_model_type_enum(i) for i in supported_model_type_strings()]


def get_model_type_enum(id_str: typing.Union[ModelType, str]) -> ModelType:
    """
    Convert a ``--model-type`` string to its :py:class:`.ModelType` enum value

    :param id_str: ``--model-type`` string

    :raises ValueError: if an invalid string value (name) is passed

    :return: :py:class:`.ModelType`
    """

    if isinstance(id_str, ModelType):
        return id_str

    try:
        return {'torch': ModelType.TORCH,
                'torch-pix2pix': ModelType.TORCH_PIX2PIX,
                'torch-sdxl': ModelType.TORCH_SDXL,
                'torch-if': ModelType.TORCH_IF,
                'torch-ifs': ModelType.TORCH_IFS,
                'torch-ifs-img2img': ModelType.TORCH_IFS_IMG2IMG,
                'torch-sdxl-pix2pix': ModelType.TORCH_SDXL_PIX2PIX,
                'torch-upscaler-x2': ModelType.TORCH_UPSCALER_X2,
                'torch-upscaler-x4': ModelType.TORCH_UPSCALER_X4,
                'flax': ModelType.FLAX}[id_str.strip().lower()]
    except KeyError:
        raise ValueError('invalid ModelType string')


def get_model_type_string(model_type_enum: ModelType) -> str:
    """
    Convert a :py:class:`.ModelType` enum value to its ``--model-type`` string

    :param model_type_enum: :py:class:`.ModelType` value
    :return: ``--model-type`` string
    """

    model_type = get_model_type_enum(model_type_enum)

    return {ModelType.TORCH: 'torch',
            ModelType.TORCH_PIX2PIX: 'torch-pix2pix',
            ModelType.TORCH_SDXL: 'torch-sdxl',
            ModelType.TORCH_IF: 'torch-if',
            ModelType.TORCH_IFS: 'torch-ifs',
            ModelType.TORCH_IFS_IMG2IMG: 'torch-ifs-img2img',
            ModelType.TORCH_SDXL_PIX2PIX: 'torch-sdxl-pix2pix',
            ModelType.TORCH_UPSCALER_X2: 'torch-upscaler-x2',
            ModelType.TORCH_UPSCALER_X4: 'torch-upscaler-x4',
            ModelType.FLAX: 'flax'}[model_type]


def model_type_is_upscaler(model_type: typing.Union[ModelType, str]) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelType` enum value represent an upscaler model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelType` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 'upscaler' in model_type


def model_type_is_sdxl(model_type: typing.Union[ModelType, str]) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelType` enum value represent an SDXL model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelType` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 'sdxl' in model_type


def model_type_is_torch(model_type: typing.Union[ModelType, str]) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelType` enum value represent an Torch model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelType` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 'torch' in model_type


def model_type_is_flax(model_type: typing.Union[ModelType, str]) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelType` enum value represent an Flax model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelType` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 'flax' in model_type


def model_type_is_pix2pix(model_type: typing.Union[ModelType, str]) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelType` enum value represent an pix2pix type model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelType` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 'pix2pix' in model_type


def model_type_is_floyd(model_type: typing.Union[ModelType, str]) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelType` enum value represent an floyd "if" of "ifs" type model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelType` enum value
    :return: bool
    """
    model_type = get_model_type_enum(model_type)

    return model_type == ModelType.TORCH_IF or \
           model_type == ModelType.TORCH_IFS or \
           model_type == ModelType.TORCH_IFS_IMG2IMG


def model_type_is_floyd_if(model_type: typing.Union[ModelType, str]) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelType` enum value represent an floyd "if" type model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelType` enum value
    :return: bool
    """
    model_type = get_model_type_enum(model_type)

    return model_type == ModelType.TORCH_IF


def model_type_is_floyd_ifs(model_type: typing.Union[ModelType, str]) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelType` enum value represent an floyd "ifs" type model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelType` enum value
    :return: bool
    """
    model_type = get_model_type_enum(model_type)

    return model_type == ModelType.TORCH_IFS or model_type == ModelType.TORCH_IFS_IMG2IMG


def have_jax_flax():
    """
    Do we have jax/flax support?

    :return: bool
    """
    return jax is not None


def get_flax_dtype(dtype: typing.Union[DataType, str, typing.Any, None]):
    """
    Return a :py:class:`jax.numpy.dtype` datatype from a :py:class:`.DataType` value,
    or a string, or a :py:class:`jax.numpy.dtype` datatype itself.

    Passing ``None`` results in ``None`` being returned.

    Passing 'auto' or :py:attr:`DataType.AUTO` results in ``None`` being returned.

    :param dtype: :py:class:`.DataType`, string, :py:class:`jax.numpy.dtype`, ``None``

    :raises ValueError: if an invalid string value (name) is passed

    :return: :py:class:`jax.numpy.dtype`
    """

    if dtype is None:
        return None

    if isinstance(dtype, jnp.dtype):
        return dtype

    if isinstance(dtype, DataType):
        dtype = get_data_type_string(dtype)

    try:
        return {'float16': jnp.bfloat16,
                'float32': jnp.float32,
                'float64': jnp.float64,
                'auto': None}[dtype.lower()]
    except KeyError:
        raise ValueError('invalid DataType string')


def get_torch_dtype(dtype: typing.Union[DataType, torch.dtype, str, None]) -> typing.Union[torch.dtype, None]:
    """
    Return a :py:class:`torch.dtype` datatype from a :py:class:`.DataType` value, or a string,
    or a :py:class:`torch.dtype` datatype itself.

    Passing ``None`` results in ``None`` being returned.

    Passing 'auto' or :py:attr:`DataType.AUTO` results in ``None`` being returned.

    :param dtype: :py:class:`.DataType`, string, :py:class:`torch.dtype`, None

    :raises ValueError: if an invalid string value (name) is passed

    :return: :py:class:`torch.dtype`
    """

    if dtype is None:
        return None

    if isinstance(dtype, torch.dtype):
        return dtype

    if isinstance(dtype, DataType):
        dtype = get_data_type_string(dtype)

    try:
        return {'float16': torch.float16,
                'float32': torch.float32,
                'float64': torch.float64,
                'auto': None}[dtype.lower()]
    except KeyError:
        raise ValueError('invalid DataType string')


__all__ = _types.module_all()
