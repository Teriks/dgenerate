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

import torch

import dgenerate.types as _types


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


def get_pipeline_type_enum(id_str: PipelineType | str | None) -> PipelineType:
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

    BFLOAT16 = 3
    """16 bit brain floating point."""


def supported_data_type_strings():
    """
    Return a list of supported ``--dtype`` strings
    """
    return ['auto', 'bfloat16', 'float16', 'float32']


def supported_data_type_enums() -> list[DataType]:
    """
    Return a list of supported :py:class:`.DataType` enum values
    """
    return [get_data_type_enum(i) for i in supported_data_type_strings()]


def get_data_type_enum(id_str: DataType | str | None) -> DataType:
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
                'float32': DataType.FLOAT32,
                'bfloat16': DataType.BFLOAT16}[id_str.strip().lower()]
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
            DataType.FLOAT32: 'float32',
            DataType.BFLOAT16: 'bfloat16'}[model_type]


class ModelType(enum.Enum):
    """
    Enum representation of ``--model-type``
    """

    SD = 0
    """Stable Diffusion, such as SD 1.0 - 2.x"""

    PIX2PIX = 1
    """Stable Diffusion pix2pix prompt guided editing."""

    SDXL = 2
    """Stable Diffusion XL"""

    IF = 3
    """Deep Floyd IF stage 1"""

    IFS = 4
    """Deep Floyd IF superscaler (stage 2)"""

    IFS_IMG2IMG = 5
    """Deep Floyd IF superscaler (stage 2) image to image / variation mode."""

    SDXL_PIX2PIX = 6
    """Stable Diffusion XL pix2pix prompt guided editing."""

    UPSCALER_X2 = 7
    """Stable Diffusion X2 upscaler"""

    UPSCALER_X4 = 8
    """Stable Diffusion X4 upscaler"""

    S_CASCADE = 9
    """
    Stable Cascade prior
    """

    S_CASCADE_DECODER = 10
    """
    Stable Cascade decoder
    """

    SD3 = 11
    """
    Stable Diffusion 3
    """

    SD3_PIX2PIX = 12
    """
    Stable Diffusion 3 pix2pix prompt guided editing.
    """

    FLUX = 13
    """
    Flux pipeline
    """

    FLUX_FILL = 14
    """
    Flux infill / outfill pipeline
    """

    FLUX_KONTEXT = 15
    """
    Flux Kontext pipeline
    """

    KOLORS = 16
    """Kolors (SDXL + ChatGLM)"""


def supported_model_type_strings():
    """
    Return a list of supported ``--model-type`` strings
    """
    return ['sd',
            'pix2pix',
            'sdxl',
            'sdxl-pix2pix',
            'kolors',
            'upscaler-x2',
            'upscaler-x4',
            'if',
            'ifs',
            'ifs-img2img',
            's-cascade',
            'sd3',
            'sd3-pix2pix',
            'flux',
            'flux-fill',
            'flux-kontext']


def supported_model_type_enums() -> list[ModelType]:
    """
    Return a list of supported :py:class:`.ModelType` enum values
    """
    return [get_model_type_enum(i) for i in supported_model_type_strings()]


def get_model_type_enum(id_str: ModelType | str) -> ModelType:
    """
    Convert a ``--model-type`` string to its :py:class:`.ModelType` enum value

    :param id_str: ``--model-type`` string

    :raises ValueError: if an invalid string value (name) is passed

    :return: :py:class:`.ModelType`
    """

    if isinstance(id_str, ModelType):
        return id_str

    try:
        return {'sd': ModelType.SD,
                'pix2pix': ModelType.PIX2PIX,
                'sdxl': ModelType.SDXL,
                'kolors': ModelType.KOLORS,
                'if': ModelType.IF,
                'ifs': ModelType.IFS,
                'ifs-img2img': ModelType.IFS_IMG2IMG,
                'sdxl-pix2pix': ModelType.SDXL_PIX2PIX,
                'upscaler-x2': ModelType.UPSCALER_X2,
                'upscaler-x4': ModelType.UPSCALER_X4,
                's-cascade': ModelType.S_CASCADE,
                'sd3': ModelType.SD3,
                'sd3-pix2pix': ModelType.SD3_PIX2PIX,
                'flux': ModelType.FLUX,
                'flux-fill': ModelType.FLUX_FILL,
                'flux-kontext': ModelType.FLUX_KONTEXT}[id_str.strip().lower()]
    except KeyError:
        raise ValueError('invalid ModelType string')


def get_model_type_string(model_type_enum: ModelType) -> str:
    """
    Convert a :py:class:`.ModelType` enum value to its ``--model-type`` string

    :param model_type_enum: :py:class:`.ModelType` value
    :return: ``--model-type`` string
    """

    model_type = get_model_type_enum(model_type_enum)

    return {ModelType.SD: 'sd',
            ModelType.PIX2PIX: 'pix2pix',
            ModelType.SDXL: 'sdxl',
            ModelType.KOLORS: 'kolors',
            ModelType.IF: 'if',
            ModelType.IFS: 'ifs',
            ModelType.IFS_IMG2IMG: 'ifs-img2img',
            ModelType.SDXL_PIX2PIX: 'sdxl-pix2pix',
            ModelType.UPSCALER_X2: 'upscaler-x2',
            ModelType.UPSCALER_X4: 'upscaler-x4',
            ModelType.S_CASCADE: 's-cascade',
            ModelType.S_CASCADE_DECODER: 's-cascade-decoder',
            ModelType.SD3: 'sd3',
            ModelType.SD3_PIX2PIX: 'sd3-pix2pix',
            ModelType.FLUX: 'flux',
            ModelType.FLUX_FILL: 'flux-fill',
            ModelType.FLUX_KONTEXT: 'flux-kontext'}[model_type]


def model_type_is_sd15(model_type: ModelType | str) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelType` enum value represent an SD1.5 model?

    These model types may also be able to load SD2 checkpoints, specifically: :py:attr:`.ModelType.SD` can.

    :param model_type: ``--model-type`` string or :py:class:`.ModelType` enum value
    :return: bool
    """
    model_type = get_model_type_enum(model_type)

    return model_type in {
        ModelType.SD,
        ModelType.PIX2PIX,
        ModelType.UPSCALER_X2
    }


def model_type_is_sd2(model_type: ModelType | str) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelType` enum value represent an SD 2.X compatible model?

    These model types may also be able to load SD1.5 checkpoints, specifically: :py:attr:`.ModelType.SD` can.

    :param model_type: ``--model-type`` string or :py:class:`.ModelType` enum value
    :return: bool
    """
    model_type = get_model_type_enum(model_type)

    return model_type in {
        ModelType.SD,
        ModelType.UPSCALER_X4
    }


def model_type_is_upscaler(model_type: ModelType | str) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelType` enum value represent an upscaler model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelType` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 'upscaler' in model_type


def model_type_is_sdxl(model_type: ModelType | str) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelType` enum value represent an SDXL model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelType` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 'sdxl' in model_type


def model_type_is_kolors(model_type: ModelType | str) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelType` enum value represent a Kolors model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelType` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 'kolors' in model_type


def model_type_is_sd3(model_type: ModelType | str) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelType` enum value represent an SD3 model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelType` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 'sd3' in model_type


def model_type_is_flux(model_type: ModelType | str) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelType` enum value represent a Flux model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelType` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 'flux' in model_type


def model_type_is_s_cascade(model_type: ModelType | str) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelType` enum value represent a Stable Cascade related model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelType` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 's-cascade' in model_type


def model_type_is_pix2pix(model_type: ModelType | str) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelType` enum value represent an pix2pix type model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelType` enum value
    :return: bool
    """
    model_type = get_model_type_string(model_type)

    return 'pix2pix' in model_type


def model_type_is_floyd(model_type: ModelType | str) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelType` enum value represent an floyd "if" of "ifs" type model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelType` enum value
    :return: bool
    """
    model_type = get_model_type_enum(model_type)

    return model_type == ModelType.IF or \
        model_type == ModelType.IFS or \
        model_type == ModelType.IFS_IMG2IMG


def model_type_is_floyd_if(model_type: ModelType | str) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelType` enum value represent an floyd "if" type model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelType` enum value
    :return: bool
    """
    model_type = get_model_type_enum(model_type)

    return model_type == ModelType.IF


def model_type_is_floyd_ifs(model_type: ModelType | str) -> bool:
    """
    Does a ``--model-type`` string or :py:class:`.ModelType` enum value represent an floyd "ifs" type model?

    :param model_type: ``--model-type`` string or :py:class:`.ModelType` enum value
    :return: bool
    """
    model_type = get_model_type_enum(model_type)

    return model_type == ModelType.IFS or model_type == ModelType.IFS_IMG2IMG


def get_torch_dtype(dtype: DataType | torch.dtype | str | None) -> torch.dtype | None:
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
        return {'bfloat16': torch.bfloat16,
                'float16': torch.float16,
                'float32': torch.float32,
                'float64': torch.float64,
                'auto': None}[dtype.lower()]
    except KeyError:
        raise ValueError('invalid DataType string')


__all__ = _types.module_all()
