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

try:
    import jax
    import jaxlib
    import jax.numpy as jnp
except ImportError:
    jax = None
    flax = None
    jnp = None


class PipelineTypes(enum.Enum):
    """
    Represents possible diffusers pipeline types.
    """
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
    """
    Represents model precision
    """
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
    TORCH_IF = 3,
    TORCH_IFS = 4,
    TORCH_IFS_IMG2IMG = 9,
    TORCH_SDXL_PIX2PIX = 5
    TORCH_UPSCALER_X2 = 6
    TORCH_UPSCALER_X4 = 7
    FLAX = 8


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
            'torch-if': ModelTypes.TORCH_IF,
            'torch-ifs': ModelTypes.TORCH_IFS,
            'torch-ifs-img2img': ModelTypes.TORCH_IFS_IMG2IMG,
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
            ModelTypes.TORCH_IF: 'torch-if',
            ModelTypes.TORCH_IFS: 'torch-ifs',
            ModelTypes.TORCH_IFS_IMG2IMG: 'torch-ifs-img2img',
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


def model_type_is_floyd(model_type: typing.Union[ModelTypes, str]) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelTypes` enum value represent an floyd "if" of "ifs" type model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelTypes` enum value
    :return: bool
    """
    model_type = get_model_type_enum(model_type)

    return model_type == ModelTypes.TORCH_IF or \
           model_type == ModelTypes.TORCH_IFS or \
           model_type == ModelTypes.TORCH_IFS_IMG2IMG


def model_type_is_floyd_if(model_type: typing.Union[ModelTypes, str]) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelTypes` enum value represent an floyd "if" type model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelTypes` enum value
    :return: bool
    """
    model_type = get_model_type_enum(model_type)

    return model_type == ModelTypes.TORCH_IF


def model_type_is_floyd_ifs(model_type: typing.Union[ModelTypes, str]) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelTypes` enum value represent an floyd "ifs" type model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelTypes` enum value
    :return: bool
    """
    model_type = get_model_type_enum(model_type)

    return model_type == ModelTypes.TORCH_IFS or model_type == ModelTypes.TORCH_IFS_IMG2IMG


def have_jax_flax():
    """
    Do we have jax/flax support?

    :return: bool
    """
    return jax is not None


def get_flax_dtype(dtype: typing.Union[DataTypes, str, typing.Any, None]):
    """
    Return a jax.numpy datatype from a :py:class:`.DataTypes` value, or a string,
    or a jax.numpy datatype itself.

    Passing None results in None being returned.

    Passing 'auto' or :py:attr:`DataTypes.AUTO` results in None being returned.

    :param dtype: :py:class:`.DataTypes`, string, jax.numpy dtype, None
    :return: jax.numpy dtype
    """

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


def get_torch_dtype(dtype: typing.Union[DataTypes, torch.dtype, str, None]) -> typing.Union[torch.dtype, None]:
    """
    Return a torch.dtype datatype from a :py:class:`.DataTypes` value, or a string,
    or a torch.dtype datatype itself.

    Passing None results in None being returned.

    Passing 'auto' or :py:attr:`DataTypes.AUTO` results in None being returned.

    :param dtype: :py:class:`.DataTypes`, string, torch.dtype, None
    :return: torch.dtype
    """

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
