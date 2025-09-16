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

import argparse
import collections.abc
import sys
import typing
from argparse import Action

import torch

import dgenerate.imageprocessors as _imageprocessors
import dgenerate.latentsprocessors as _latentsprocessors
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.memoize as _memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.prompt as _prompt
import dgenerate.promptupscalers as _promptupscalers
import dgenerate.promptweighters as _promptweighters
import dgenerate.renderloopconfig as _renderloopconfig
import dgenerate.resources as _resources
import dgenerate.textprocessing as _textprocessing
import dgenerate.torchutil as _torchutil
import dgenerate.types as _types

__doc__ = """
Argument parsing for the dgenerate command line tool.
"""

_SUPPORTED_MODEL_TYPES_PRETTY = \
    _textprocessing.oxford_comma(_pipelinewrapper.supported_model_type_strings(), 'or')

_SUPPORTED_ANIMATION_OUTPUT_FORMATS_PRETTY = \
    _textprocessing.oxford_comma(_mediaoutput.get_supported_animation_writer_formats(), 'or')

_SUPPORTED_ALL_OUTPUT_FORMATS_PRETTY = \
    _textprocessing.oxford_comma(
        _mediaoutput.get_supported_static_image_formats() + _mediaoutput.get_supported_tensor_formats(), 'or')

_SUPPORTED_DATA_TYPES_PRETTY = \
    _textprocessing.oxford_comma(_pipelinewrapper.supported_data_type_strings(), 'or')


class DgenerateHelpException(Exception):
    """
    Raised by :py:func:`.parse_args` and :py:func:`.parse_known_args`
    when ``--help`` is encountered and ``help_raises=True``
    """
    pass


def _model_type(val):
    val = val.lower()
    if val not in _pipelinewrapper.supported_model_type_strings():
        raise argparse.ArgumentTypeError(
            f'Must be one of: {_SUPPORTED_MODEL_TYPES_PRETTY}. Unknown value: {val}')
    return _pipelinewrapper.get_model_type_enum(val)


def _type_dtype(dtype):
    dtype = dtype.lower()
    supported_dtypes = _pipelinewrapper.supported_data_type_strings()
    if dtype not in supported_dtypes:
        raise argparse.ArgumentTypeError(f'Must be {_textprocessing.oxford_comma(supported_dtypes, "or")}')
    else:
        return _pipelinewrapper.get_data_type_enum(dtype)


def _type_main_prompts(prompt):
    try:
        prompt = _prompt.Prompt.parse(prompt)
        prompt.set_embedded_args_on(
            _pipelinewrapper.DiffusionArguments,
            forbidden_checker=_pipelinewrapper.DiffusionArguments.prompt_embedded_arg_checker,
            validate_only=True)
        return prompt
    except (ValueError, _prompt.PromptEmbeddedArgumentError) as e:
        raise argparse.ArgumentTypeError(
            f'Prompt parse error: {str(e).strip()}')


def _type_secondary_prompts(prompt):
    try:
        return _prompt.Prompt.parse(prompt, parse_embedded_args=False)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f'Prompt parse error: {str(e).strip()}')


def _type_prompt_weighter(uri):
    uri = str(uri)
    if not _promptweighters.prompt_weighter_exists(uri):
        raise argparse.ArgumentTypeError(
            f'Unknown prompt weighter implementation: {_promptweighters.prompt_weighter_name_from_uri(uri)}, '
            f'must be one of: {_textprocessing.oxford_comma(_promptweighters.prompt_weighter_names(), "or")}')
    return uri


def _type_prompt_upscaler(uri):
    uri = str(uri)
    if not _promptupscalers.prompt_upscaler_exists(uri):
        raise argparse.ArgumentTypeError(
            f'Unknown prompt upscaler implementation: {_promptupscalers.prompt_upscaler_name_from_uri(uri)}, '
            f'must be one of: {_textprocessing.oxford_comma(_promptupscalers.prompt_upscaler_names(), "or")}')
    return uri


def _type_latents_processor(uri):
    uri = str(uri)
    if uri != _pipelinewrapper.constants.LATENTS_PROCESSOR_SEP and not _latentsprocessors.latents_processor_exists(uri):
        raise argparse.ArgumentTypeError(
            f'Unknown latents processor implementation: {_latentsprocessors.latents_processor_name_from_uri(uri)}, '
            f'must be one of: {_textprocessing.oxford_comma(_latentsprocessors.latents_processor_names(), "or")}')
    return uri


def _type_image_processor(uri):
    uri = str(uri)
    if uri != _renderloopconfig.IMAGE_PROCESSOR_SEP and not _imageprocessors.image_processor_exists(uri):
        raise argparse.ArgumentTypeError(
            f'Unknown image processor implementation: {_imageprocessors.image_processor_name_from_uri(uri)}, '
            f'must be one of: {_textprocessing.oxford_comma(_imageprocessors.image_processor_names(), "or")}')
    return uri


def _max_sequence_length(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val < 1:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 1')

    if val > 512:
        raise argparse.ArgumentTypeError('Must be less than or equal to 512')

    return val


def _type_clip_skip(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


def _type_adapter_factor(val):
    try:
        val = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be a float')

    if val < 0.0 or val > 1.0:
        raise argparse.ArgumentTypeError(
            'Must be greater than or equal to 0.0 and less than or equal to 1.0')
    return val


def _type_inference_steps(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val <= 0:
        raise argparse.ArgumentTypeError('Must be greater than 0')
    return val


def _type_guidance_scale(val):
    try:
        val = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be a floating point number')

    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


def _type_size(size):
    if size is None:
        return None
    try:
        return _textprocessing.parse_image_size(size)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e).strip())


def _type_output_size(size):
    x, y = _type_size(size)

    if x % 8 != 0:
        raise argparse.ArgumentTypeError('Output X dimension must be divisible by 8')

    if y % 8 != 0:
        raise argparse.ArgumentTypeError('Output Y dimension must be divisible by 8')

    return x, y


def _type_image_coordinate(coord):
    if coord is None:
        return 0, 0

    r = coord.split(',')

    try:
        return int(r[0]), int(r[1])
    except ValueError:
        raise argparse.ArgumentTypeError('Coordinates must be integer values')


def _type_sdxl_high_noise_fractions(val):
    try:
        val = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be a floating point number')

    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


def _type_device(device):
    if not _torchutil.is_valid_device_string(device):
        raise argparse.ArgumentTypeError(_torchutil.invalid_device_message(device))

    return device


def _type_seeds(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    return val


def _type_gen_seeds(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    return _renderloopconfig.gen_seeds(val)


def _type_animation_format(val):
    val = val.lower()
    if val not in _mediaoutput.get_supported_animation_writer_formats() + ['frames']:
        raise argparse.ArgumentTypeError(
            f'Must be {_SUPPORTED_ANIMATION_OUTPUT_FORMATS_PRETTY}. Unknown value: {val}')
    return val


def _type_image_format(val):
    val = val.lower()
    supported_image_formats = _mediaoutput.get_supported_static_image_formats()
    supported_tensor_formats = _mediaoutput.get_supported_tensor_formats()
    all_supported_formats = supported_image_formats + supported_tensor_formats

    if val not in all_supported_formats:
        raise argparse.ArgumentTypeError(
            f'Must be one of {_textprocessing.oxford_comma(all_supported_formats, "or")}')
    return val


def _type_frame_start(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


def _type_frame_end(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


def _type_image_seed_strengths(val):
    try:
        val = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be a floating point number')

    if val < 0 or val > 1:
        raise argparse.ArgumentTypeError(
            'Must be greater than or equal to zero, and less than or equal to one')
    return val


def _type_tea_cache_rel_l1_thresh(val):
    try:
        val = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be a floating point number')

    if val < 0 or val > 1:
        raise argparse.ArgumentTypeError(
            'Must be greater than or equal to zero, and less than or equal to one')
    return val


def _type_upscaler_noise_levels(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')

    return val


def _type_image_guidance_scale(val):
    try:
        val = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be a floating point number')

    if val < 1:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 1')
    return val


def _type_batch_size(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val < 1:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 1')
    return val


def _type_expression(arg):
    try:
        _memory.memory_constraint_syntax_check(arg)
    except _memory.MemoryConstraintSyntaxError as e:
        raise argparse.ArgumentTypeError(f'Syntax error: {str(e).strip()}')


def _type_text_encoder(val):
    if val == '+':
        return None
    return val


def _type_adetailer_index_filter_value(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError(
            'Must be an integer value'
        )

    if val < 0:
        raise argparse.ArgumentTypeError(
            'Must be greater than or equal to zero'
        )
    return val


def _type_adetailer_class_filter_value(val):
    """
    Parse a class filter value which can be either an integer (class ID) or a string (class name).
    """
    try:
        # Try to convert to integer if it's a digit
        if val.isdigit():
            return int(val)
    except (ValueError, AttributeError):
        pass
    # Return as is if it's a string
    return val


def _type_adetailer_mask_shape(val):
    val = val.lower()

    try:
        parsed_shape = _textprocessing.parse_basic_mask_shape(val)
    except ValueError:
        parsed_shape = None

    if parsed_shape is None or parsed_shape not in {
        _textprocessing.BasicMaskShape.RECTANGLE,
        _textprocessing.BasicMaskShape.ELLIPSE
    }:
        raise argparse.ArgumentTypeError(
            'Must be one of: "r", "rect", "rectangle", or "c", "circle", "ellipse"'
        )

    return val


def _type_adetailer_mask_padding(val):
    try:
        val = _textprocessing.parse_dimensions(val)

        if len(val) not in {1, 2, 4}:
            raise ValueError()

    except ValueError:
        raise argparse.ArgumentTypeError(
            'Must be an integer value, WIDTHxHEIGHT, or LEFTxTOPxRIGHTxBOTTOM')

    if len(val) == 1:
        return val[0]

    return val


def _type_adetailer_mask_blur(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


def _type_adetailer_size(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val <= 1:
        raise argparse.ArgumentTypeError('Must be greater than 1')
    return val


def _type_adetailer_mask_dilation(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


def _type_inpaint_crop_padding(val):
    try:
        val = _textprocessing.parse_dimensions(val)

        if len(val) not in {1, 2, 4}:
            raise ValueError()

    except ValueError:
        raise argparse.ArgumentTypeError(
            'Must be an integer value, WIDTHxHEIGHT, or LEFTxTOPxRIGHTxBOTTOM')

    if len(val) == 1:
        return val[0]

    return val


def _type_inpaint_crop_feather(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


def _type_ras_sample_ratio(val: str) -> float:
    try:
        val = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be a floating point number')
    if not 0 <= val <= 1:
        raise argparse.ArgumentTypeError('Must be between 0.0 and 1.0')
    return val


def _type_ras_high_ratio(val: str) -> float:
    try:
        val = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be a floating point number')
    if not 0 <= val <= 1:
        raise argparse.ArgumentTypeError('Must be between 0.0 and 1.0')
    return val


def _type_ras_starvation_scale(val: str) -> float:
    try:
        val = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be a floating point number')
    if not 0 <= val <= 1:
        raise argparse.ArgumentTypeError('Must be between 0.0 and 1.0')
    return val


def _type_ras_error_reset_steps(val: str) -> list[int]:
    try:
        if ',' in val:
            steps = [int(x.strip()) for x in val.split(',')]
        else:
            steps = [int(val.strip())]
        if not all(x >= 0 for x in steps):
            raise argparse.ArgumentTypeError(
                'All RAS step numbers must be positive'
            )
    except ValueError:
        raise argparse.ArgumentTypeError(
            'RAS steps must be a comma-separated list of '
            'positive integers, or a single integer'
        )
    return steps


def _type_ras_metric(val: str) -> str:
    val = val.lower()
    if val not in {'std', 'l2norm'}:
        raise argparse.ArgumentTypeError('Must be one of: std or l2norm')
    return val


def _type_ras_start_steps(val: str) -> int:
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')
    if val < 1:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 1')
    return val


def _type_ras_end_steps(val: str) -> int:
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')
    if val < 1:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 1')
    return val


def _type_ras_skip_num_step(val: str) -> int:
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')
    return val


def _type_ras_skip_num_step_length(val: str) -> int:
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')
    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


def _type_deep_cache_interval(val: str) -> int:
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')
    if val <= 0:
        raise argparse.ArgumentTypeError('Must be greater than 0')
    return val


def _type_deep_cache_branch_id(val: str) -> int:
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')
    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


def _type_sada_max_downsamples(val: str) -> int:
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')
    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


def _type_sada_sxs(val: str) -> int:
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')
    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


def _type_sada_sys(val: str) -> int:
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')
    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


def _type_sada_acc_ranges(val: str) -> list[int]:
    try:
        if ',' in val:
            ranges = [int(x.strip()) for x in val.split(',')]
            if len(ranges) == 2:
                start, end = ranges
                if start > end:
                    raise argparse.ArgumentTypeError(
                        f'SADA acceleration range start value ({start}) must be less than or equal to end value ({end})'
                    )
        else:
            raise argparse.ArgumentTypeError(
                'SADA acceleration ranges must possess at least two values'
            )
        if not all(x >= 3 for x in ranges):
            raise argparse.ArgumentTypeError(
                'All SADA acceleration range values must be at least 3'
            )
    except ValueError:
        raise argparse.ArgumentTypeError(
            'SADA acceleration ranges must be a comma-separated list of '
            'integers >= 3, or a single integer >= 3'
        )
    return ranges


def _type_sada_lagrange_terms(val: str) -> int:
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')
    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


def _type_sada_lagrange_ints(val: str) -> int:
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')
    if val <= 0:
        raise argparse.ArgumentTypeError('Must be greater than 0')
    return val


def _type_sada_lagrange_steps(val: str) -> int:
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')
    if val <= 0:
        raise argparse.ArgumentTypeError('Must be greater than 0')
    return val


def _type_sada_max_fixes(val: str) -> int:
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')
    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


def _type_sada_max_intervals(val: str) -> int:
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')
    if val <= 0:
        raise argparse.ArgumentTypeError('Must be greater than 0')
    return val


def _type_sigmas(val: str) -> list[float] | str:
    try:
        if val.startswith('expr:'):
            sigmas = val.removeprefix('expr:').strip()
        elif ',' in val:
            sigmas = [float(x.strip()) for x in val.split(',')]
        else:
            sigmas = [float(val.strip())]
    except ValueError:
        raise argparse.ArgumentTypeError(
            'Sigmas must be a comma-separated list of '
            'floating point numbers, a single float, or an expression '
            'involving "sigmas" denoted by "expr: ... "'
        )
    return sigmas


def _type_freeu_params(val: str) -> tuple[float, float, float, float]:
    try:
        steps = [float(x.strip()) for x in val.split(',')]
        if len(steps) != 4:
            raise argparse.ArgumentTypeError(
                f'Not enough FreeU parameters supplied, requires 4 floats in CSV format, got: {val}'
            )
    except ValueError:
        raise argparse.ArgumentTypeError(
            'FreeU parameters must be a comma-separated list of four floats'
        )
    # noinspection PyTypeChecker
    return tuple(steps)


def _type_quantizer_map(val: str):
    vals = [
        'unet',
        'transformer',
        'text_encoder',
        'text_encoder_2',
        'text_encoder_3',
        'controlnet'
    ]

    if val not in vals:
        raise argparse.ArgumentTypeError(
            f'Quantizer map values must be one of: '
            f'{_textprocessing.oxford_comma(vals, "or")}, received invalid value: {val}'
        )

    return val


def _type_latents(val: str) -> torch.Tensor:
    """
    Load and validate a tensor file for the --latents argument.
    
    :param val: The tensor file path
    :return: The loaded tensor
    :raises ArgumentTypeError: If the file doesn't exist, isn't a valid tensor format, or can't be loaded
    """
    import os
    import dgenerate.mediainput as _mediainput

    # Check if file exists
    if not os.path.exists(val):
        raise argparse.ArgumentTypeError(f'Tensor file not found: {val}')

    # Check if it's a valid tensor file format
    if not _mediainput.is_tensor_file(val):
        supported_formats = _mediainput.get_supported_tensor_formats()
        raise argparse.ArgumentTypeError(
            f'File "{val}" is not a supported tensor format. '
            f'Supported formats: {", ".join(supported_formats)}'
        )

    # Load the tensor
    try:
        tensor = _mediainput.load_tensor_file(val, val)
        return tensor
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f'Failed to load tensor from "{val}": {str(e)}'
        )


def _type_denoising_fraction(val: str) -> float:
    """
    Validate a denoising fraction value (0.0 to 1.0).
    
    :param val: The denoising fraction as a string
    :return: The validated float value
    :raises ArgumentTypeError: If the value is not a valid float between 0.0 and 1.0
    """
    try:
        val = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be a floating point number')

    if val < 0.0 or val > 1.0:
        raise argparse.ArgumentTypeError(
            'Must be between 0.0 and 1.0 (inclusive)')

    return val


class _SetAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, set(values))


_ARG_PARSER_CACHE = dict()


@_memoize.memoize(cache=_ARG_PARSER_CACHE)
def _create_parser(add_model=True, add_help=True, prints_usage=True):
    parser = argparse.ArgumentParser(
        prog='dgenerate',
        exit_on_error=False,
        allow_abbrev=False,
        add_help=add_help,
        formatter_class=_textprocessing.ArgparseParagraphFormatter,
        description="""Batch image generation and manipulation tool supporting Stable Diffusion
        and related techniques / algorithms, with support for video and animated image processing."""
    )

    def _exit(status=0, message=None):
        if status == 0:
            # help
            raise DgenerateHelpException('dgenerate --help used.')
        raise argparse.ArgumentError(None, message)

    def _usage(file):
        if prints_usage:
            _messages.log(parser.format_usage().rstrip())

    parser.exit = _exit
    parser.print_usage = _usage

    actions: list[Action] = []

    if add_model:
        actions.append(
            parser.add_argument(
                'model_path', action='store',
                help="""Hugging Face model repository slug, Hugging Face blob link to a model file,
                path to folder on disk, or path to a .pt, .pth, .bin, .ckpt, or .safetensors file."""
            )
        )

    actions.append(
        parser.add_argument(
            '-v', '--verbose', action='store_true', default=False,
            help="""Output information useful for debugging, such as pipeline
                    call and model load parameters."""
        )
    )

    actions.append(
        parser.add_argument(
            '--version', action='version', version=f"dgenerate v{_resources.version()}",
            help="Show dgenerate's version and exit"
        )
    )

    popen_group = parser.add_mutually_exclusive_group()

    actions.append(
        popen_group.add_argument(
            '--file', dest=None, action='store_true', default=False,
            help="""Convenience argument for reading a configuration script from a file instead
                    of using a pipe. This is a meta argument which can not be used within a
                    configuration script and is only valid from the command line or during a
                    popen invocation of dgenerate. This argument understands glob syntax, 
                    even on windows, and can accept multiple config file names, which will be
                    executed in sequence."""
        )
    )

    actions.append(
        popen_group.add_argument(
            '--shell', dest=None, action='store_true', default=False,
            help="""When reading configuration from STDIN (a pipe), read forever, even when
                    configuration errors occur. This allows dgenerate to run in the background and
                    be controlled by another process sending commands. Launching
                    dgenerate with this option and not piping it input will attach it to the
                    terminal like a shell. Entering configuration into this shell requires
                    two newlines to submit a command due to parsing lookahead. IE: two presses
                    of the enter key. This is a meta argument which can not be used within a
                    configuration script and is only valid from the command line or during a
                    popen invocation of dgenerate."""
        )
    )

    actions.append(
        popen_group.add_argument(
            '--no-stdin', dest=None, action='store_true', default=False,
            help="""Can be used to indicate to dgenerate that it will not receive any
                    piped in input. This is useful for running dgenerate via popen from Python
                    or another application using normal arguments, where it would otherwise
                    try to read from STDIN and block forever because it is not attached to
                    a terminal. This is a meta argument which can not be used within a
                    configuration script and is only valid from the command line or during
                    a popen invocation of dgenerate."""
        )
    )

    actions.append(
        popen_group.add_argument(
            '--console', action='store_true', default=False,
            help="""Launch a terminal-like Tkinter GUI that interacts with an instance
                    of dgenerate running in the background. This allows you to interactively write
                    dgenerate config scripts as if dgenerate were a shell / REPL. This is a meta argument
                    which can not be used within a configuration script and is only valid from the command
                    line or during a popen invocation of dgenerate."""
        )
    )

    actions.append(
        parser.add_argument(
            '--set', dest=None, default=False, metavar='VARIABLE=VALUE',
            help="""Set template variables that will be applied before config execution.
                    Mirrors the functionality of the \\set config directive. Must use the syntax
                    --set variable=value [variable2=value2 ...]. Can accept multiple variable=value 
                    pairs and can be used multiple times. All --set and --setp arguments are processed 
                    in the order they appear on the command line. This is a meta argument which can not 
                    be used within a configuration script and is only valid from the command line or 
                    during a popen invocation of dgenerate."""
        )
    )

    actions.append(
        parser.add_argument(
            '--setp', dest=None, default=False, metavar='VARIABLE=VALUE',
            help="""Set template variables to the result of evaluating python expressions
                    that will be applied before config execution. Mirrors the functionality of 
                    the \\setp config directive. Must use the syntax --setp variable=expression 
                    [variable2=expression2 ...]. Can accept multiple variable=expression pairs and 
                    can be used multiple times. All --set and --setp arguments are processed in the 
                    order they appear on the command line. This is a meta argument which can not be 
                    used within a configuration script and is only valid from the command line or 
                    during a popen invocation of dgenerate."""
        )
    )

    actions.append(
        parser.add_argument(
            '--plugin-modules', action='store', default=[], nargs="+", dest='plugin_module_paths', metavar="PATH",
            help="""Specify one or more plugin module folder paths (folder containing __init__.py) or
                    Python .py file paths, or Python module names to load as plugins. Plugin modules can
                    currently implement image processors, config directives, config template functions,
                    prompt weighters, and sub-commands."""
        )
    )

    actions.append(
        parser.add_argument(
            '--sub-command', action='store', default=None, metavar="SUB_COMMAND",
            help="""Specify the name a sub-command to invoke. dgenerate exposes some extra image processing
                    functionality through the use of sub-commands. Sub commands essentially replace the entire set
                    of accepted arguments with those of a sub-command which implements additional functionality.
                    See --sub-command-help for a list of sub-commands and help."""
        )
    )

    actions.append(
        parser.add_argument(
            '--sub-command-help', action='store', nargs='*', default=None, metavar='SUB_COMMAND',
            help="""Use this option alone (or with --plugin-modules) and no model specification
                    in order to list available sub-command names. Calling a sub-command with
                    "--sub-command name --help" will produce argument help output for that sub-command.
                    When used with --plugin-modules, sub-commands implemented by the specified plugins
                    will also be listed."""
        )
    )

    actions.append(
        parser.add_argument(
            '-ofm', '--offline-mode', action='store_true',
            help="""Prevent dgenerate from downloading resources that do not already exist on disk. 
                    Referencing a model on Hugging Face hub that has not been cached because it was not 
                    previously downloaded will result in a failure when using this option, as well as
                    attempting to download any new content into dgenerates web cache.  This will 
                    prevent dgenerate from downloading anything, it will only look for cached 
                    resources when processing URLs or Hugging Face slugs. It will not be able to download 
                    any default models that have been baked into the code as well. This option is fed to 
                    sub-commands when using the --sub-command argument, meaning that all sub-commands can 
                    parse this argument by default, though they may complain if it is not supported, 
                    such as with the "civitai-links" sub-command."""
        )
    )

    actions.append(
        parser.add_argument(
            '--templates-help', nargs='*', dest=None, default=None, metavar='VARIABLE_NAME',
            help="""Print a list of template variables available in the interpreter environment
                    used for dgenerate config scripts, particularly the variables set after a dgenerate
                    invocation occurs. When used as a command line option, their values are not presented,
                    just their names and types. Specifying names will print type information for
                    those variable names."""
        )
    )

    actions.append(
        parser.add_argument(
            '--directives-help', nargs='*', dest=None, default=None, metavar='DIRECTIVE_NAME',
            help="""Use this option alone (or with --plugin-modules) and no model specification
                    in order to list available config directive names. Providing names will print documentation
                    for the specified directive names. When used with --plugin-modules, directives implemented
                    by the specified plugins will also be listed."""
        )
    )

    actions.append(
        parser.add_argument(
            '--functions-help', nargs='*', dest=None, default=None, metavar='FUNCTION_NAME',
            help="""Use this option alone (or with --plugin-modules) and no model specification
                    in order to list available config template function names. Providing names will print
                    documentation for the specified function names. When used with --plugin-modules,
                    functions implemented by the specified plugins will also be listed."""
        )
    )

    actions.append(
        parser.add_argument(
            '-gc', '--global-config', action='store', default=None,
            metavar="FILE",
            help="""Provide a json, yaml, or toml file to configure dgenerate's global settings.
                    These settings include various default values for generation and garbage
                    collection settings for the in memory caches."""
        )
    )

    actions.append(
        parser.add_argument(
            '-mt', '--model-type', action='store', default='sd', type=_model_type,
            help=f"""Use when loading different model types.
                     Currently supported: {_SUPPORTED_MODEL_TYPES_PRETTY}. (default: sd)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-rev', '--revision', action='store', default="main", metavar="BRANCH",
            help="""The model revision to use when loading from a Hugging Face repository,
                    (The Git branch / tag, default is "main")"""
        )
    )

    actions.append(
        parser.add_argument(
            '-var', '--variant', action='store', default=None,
            help="""If specified when loading from a Hugging Face repository or folder, load weights
                    from "variant" filename, e.g. "pytorch_model.<variant>.safetensors".
                    Defaults to automatic selection."""
        )
    )

    actions.append(
        parser.add_argument(
            '-sbf', '--subfolder', action='store', default=None,
            help="""Main model subfolder.
                    If specified when loading from a Hugging Face repository or folder,
                    load weights from the specified subfolder."""
        )
    )

    actions.append(
        parser.add_argument(
            '-olc', '--original-config', default=None, metavar="FILE", dest='original_config',
            help="""This argument can be used to supply an original LDM config .yaml file 
            that was provided with a single file checkpoint.""")
    )

    actions.append(
        parser.add_argument(
            '-olc2', '--second-model-original-config', default=None, metavar="FILE",
            dest='second_model_original_config',
            help="""This argument can be used to supply an original LDM config .yaml file 
            that was provided with a single file checkpoint for the secondary model, 
            i.e. the SDXL Refiner or Stable Cascade Decoder.""")
    )

    actions.append(
        parser.add_argument(
            '-atk', '--auth-token', action='store', default=None, metavar="TOKEN",
            help="""Huggingface auth token.
                    Required to download restricted repositories that have access permissions
                    granted to your Hugging Face account."""
        )
    )

    actions.append(
        parser.add_argument(
            '-bs', '--batch-size', action='store', default=None, metavar="INTEGER", type=_type_batch_size,
            help="""The number of image variations to produce per set of individual diffusion parameters
                    in one rendering step simultaneously on a single GPU.
                    
                    When generating animations with a --batch-size greater than one, a separate animation
                    (with the filename suffix "animation_N") will be written to for each image in the batch.
                    
                    If --batch-grid-size is specified when producing an animation then the image grid is used
                    for the output frames.
                    
                    During animation rendering each image in the batch will still be written to the output directory
                    along side the produced animation as either suffixed files or image grids depending on the
                    options you choose. (Default: 1)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-bgs', '--batch-grid-size', action='store', default=None, metavar="SIZE", type=_type_size,
            help="""Produce a single image containing a grid of images with the number of COLUMNSxROWS
                    given to this argument when --batch-size is greater than 1. If not specified with a
                    --batch-size greater than 1, images will be written individually with an image number suffix
                    (image_N) in the filename signifying which image in the batch they are."""
        )
    )

    actions.append(
        parser.add_argument(
            '-ad', '--adetailer-detectors',
            nargs='+',
            action='store',
            default=None,
            metavar='ADETAILER_DETECTOR_URIS',
            dest='adetailer_detector_uris',
            help="""Specify one or more adetailer YOLO detector model URIs. When specifying this option, 
                    you must provide an image to --image-seeds, inpaint masks will be auto generated 
                    based on what is detected by the provided detector models.
                    
                    The models will be used in sequence to detect and then inpaint your image within
                    the detection areas. This can be used for face detailing, face swapping, hand detailing, 
                    etc. on any arbitrary image provided using an image generation model of your choice.
                    
                    This option supports: --model-type sd, sdxl, kolors, sd3, flux, and flux-fill
                    
                    NOWRAP!
                    Example: --adetailer-detectors Bingsu/adetailer;weight-name=face_yolov8n.pt
                    
                    The "revision" argument specifies the model revision to use for the adetailer model when loading from
                    Hugging Face repository, (The Git branch / tag, default is "main").
                    
                    The "subfolder" argument specifies the adetailer model subfolder, if specified when loading from a
                    Hugging Face repository or folder, weights from the specified subfolder.
                    
                    The "weight-name" argument indicates the name of the weights file to be loaded when
                    loading from a Hugging Face repository or folder on disk.
                    
                    The "class-filter" (overrides --adetailer-class-filter) argument is a list of class IDs or 
                    class names that indicates what YOLO detection classes to keep. This filter is applied first, 
                    before index-filter. Detections that don't match any of the specified classes will be ignored.
                    
                    Example "class-filter" values:
                    
                        NOWRAP!
                        * Only keep detection class ID 0:
                        class-filter=0
                        
                        NOWRAP!
                        * Only keep detection class "hand":
                        class-filter=hand
                        
                        NOWRAP!
                        * Keep class IDs 2 and 3:
                        class-filter=2,3
                        
                        NOWRAP!
                        * Keep class ID 0 and class name "hand":
                        class-filter=0,hand
                        
                        NOWRAP!
                        * String digits are interpreted as integers:
                        class-filter="0" (interpreted as class name "0", not likely useful)
                        
                        NOWRAP!
                        * List syntax is also supported:
                        class-filter=[0, "hand"]
                        
                    The "index-filter" (overrides --adetailer-index-filter) argument is a list values or a
                    single value that indicates what YOLO detection indices to keep, the index values start
                    at zero. Detections are sorted by their top left bounding box coordinate from left to right, 
                    top to bottom, by (confidence descending). The order of detections in the image is identical to
                    the reading order of words on a page (english). Inpainting will only be performed on the 
                    specified detection indices, if no indices are specified, then inpainting 
                    will be performed on all detections. This filter is applied after class-filter.
                
                    Example "index-filter" values:
                    
                        NOWRAP!
                        * keep the first, leftmost, topmost detection:
                        index-filter=0
                        
                        NOWRAP!
                        * keep detections 1 and 3:
                        index-filter=[1, 3]
                        
                        NOWRAP!
                        * CSV syntax is supported (tuple):
                        index-filter=1,3
                
                    The "detector-padding" (overrides --adetailer-detector-paddings)
                    argument specifies the amount of padding that will be added to the detection 
                    rectangle which is used to generate a masked area. The default is 0, you can 
                    make the mask area around the detected feature larger with positive padding
                    and smaller with negative padding.
                
                    Padding examples:
                    
                        NOWRAP!
                        32 (32px Uniform, all sides)
                        
                        NOWRAP!
                        10x20 (10px Horizontal, 20px Vertical)
                        
                        NOWRAP!
                        10x20x30x40 (10px Left, 20px Top, 30px Right, 40px Bottom)
                
                    The "mask-padding" (overrides --adetailer-mask-paddings) argument indicates how 
                    much padding to place around the masked area when cropping out the image to be 
                    inpainted. This value must be large enough to accommodate any feathering on the 
                    edge of the mask caused by "mask-blur" or "mask-dilation" for the best result, 
                    the default value is 32. The syntax for specifying this value is identical 
                    to "detector-padding".
                    
                    The "mask-shape" (overrides --adetailer-mask-shapes) argument indicates 
                    what mask shape adetailer should attempt to draw around a detected feature, 
                    the default value is "rectangle". You may also specify "circle" to generate 
                    an ellipsoid shaped mask, which might be helpful for achieving better blending.
                    Valid values are: ("r", "rect", "rectangle"), or ("c", "circle", "ellipse").
                
                    The "mask-blur" (overrides --adetailer-mask-blurs) argument indicates the 
                    level of gaussian blur to apply to the generated inpaint mask, which 
                    can help with smooth blending in of the inpainted feature
                    
                    The "model-masks" (overrides --adetailer-model-masks) argument indicates 
                    that masks generated by the model itself should be preferred over masks 
                    generated from the detection bounding box. If this is True, and the 
                    model itself returns mask data, "mask-shape", "mask-padding", and 
                    "detector-padding" will all be ignored.
                
                    The "mask-dilation" (overrides --adetailer-mask-dilations) argument 
                    indicates the amount of dilation applied to the inpaint mask, see: cv2.dilate
                    
                    The "confidence" argument indicates the confidence value to use with the YOLO 
                    detector model, this value defaults to 0.3 if not specified.
                    
                    The "prompt" (overrides --prompt positive) argument overrides the positive
                    inpainting prompt for detections by this detector.
                    
                    The "negative-prompt" (overrides --prompt negative) argument overrides the
                    negative inpainting prompt for detections by this detector.
                    
                    The "device" argument indicates a device override for the YOLO detector model, the 
                    detector model can be set to run on a different device if desired, for example: 
                    cuda:0, cuda:1, cpu, etc. It runs on the same device as --device by default.
                    
                    If you wish to load a weights file directly from disk, use: --adetailer-detectors "yolo_model.pt"
                    
                    You may also load a YOLO model directly from a URL or Hugging Face blob link.
                    
                    Example: --adetailer-detectors https://modelsite.com/yolo-model.pt
                    """)
    )

    actions.append(
        parser.add_argument(
            '-adm', '--adetailer-model-masks',
            dest='adetailer_model_masks',
            action='store_true',
            help="""Indicates that masks generated by the model itself should be preferred over 
                    masks generated from the detection bounding box. If this is specified, and the model itself
                    returns mask data, --adetailer-mask-shapes, --adetailer-mask-paddings, 
                    and --adetailer-detector-paddings will all be ignored."""
        ))

    actions.append(
        parser.add_argument(
            '-adf', '--adetailer-class-filter',
            nargs='+',
            action=_SetAction,
            type=_type_adetailer_class_filter_value,
            default=None,
            metavar='CLASS_FILTER',
            dest='adetailer_class_filter',
            help="""A list of class IDs or class names that indicates what YOLO detection classes to keep.
                    This filter is applied before index-filter. Detections that don't match any of the 
                    specified classes will be ignored. This filtering occurs before --adetailer-index-filter.
                    
                    NOWRAP!
                    Examples:
                    --adetailer-class-filter 0 2        # Keep only class IDs 0 and 2
                    --adetailer-class-filter person car # Keep only "person" and "car" classes
                    --adetailer-class-filter 0 person   # Keep class ID 0 and class name "person"
                    """)
    )

    actions.append(
        parser.add_argument(
            '-adi', '--adetailer-index-filter',
            nargs='+',
            action=_SetAction,
            type=_type_adetailer_index_filter_value,
            default=None,
            metavar='INTEGER',
            dest='adetailer_index_filter',
            help="""A list index values that indicates what adetailer YOLO detection indices to keep, 
                    the index values start at zero. Detections are sorted by their top left bounding box 
                    coordinate from left to right, top to bottom, by (confidence descending). The order of 
                    detections in the image is identical to the reading order of words on a page (english). 
                    Inpainting will only be performed on the specified detection indices, if no indices 
                    are specified, then inpainting will be performed on all detections.
                    This filter is applied after class-filter.
                    """)
    )

    actions.append(
        parser.add_argument(
            '-ads', '--adetailer-mask-shapes',
            nargs='+',
            action='store',
            type=_type_adetailer_mask_shape,
            default=None,
            metavar='ADETAILER_MASK_SHAPE',
            dest='adetailer_mask_shapes',
            help="""One or more adetailer mask shapes to try. This indicates what mask shape 
                    adetailer should attempt to draw around a detected feature, the default value is 
                    "rectangle". You may also specify "circle" to generate an ellipsoid shaped mask, 
                    which might be helpful for achieving better blending. 
                    
                    Valid values are: ("r", "rect", "rectangle"), or ("c", "circle", "ellipse")
                    
                    (default: rectangle).
                    """)
    )

    actions.append(
        parser.add_argument(
            '-addp', '--adetailer-detector-paddings',
            nargs='+',
            action='store',
            type=_type_adetailer_mask_padding,
            default=None,
            metavar='ADETAILER_DETECTOR_PADDING',
            dest='adetailer_detector_paddings',
            help="""One or more adetailer detector padding values to try. This value specifies the 
                    amount of padding that will be added to the detection rectangle which is used to
                    generate a masked area. The default is 0, you can make the mask area around the 
                    detected feature larger with positive padding and smaller with negative padding.
                    
                    Example:
                    
                    32 (32px Uniform, all sides)
                    
                    10x20 (10px Horizontal, 20px Vertical)
                    
                    10x20x30x40 (10px Left, 20px Top, 30px Right, 40px Bottom)

                    (default: 0).""")
    )

    actions.append(
        parser.add_argument(
            '-admp', '--adetailer-mask-paddings',
            nargs='+',
            action='store',
            type=_type_adetailer_mask_padding,
            default=None,
            metavar='ADETAILER_MASK_PADDING',
            dest='adetailer_mask_paddings',
            help="""One or more adetailer mask padding values to try. This value
                    indicates how much padding to place around the masked area when 
                    cropping out the image to be inpainted, this value must be large enough 
                    to accommodate any feathering on the edge of the mask caused
                    by "--adetailer-mask-blurs" or "--adetailer-mask-dilations" 
                    for the best result.
                    
                    Example:
                    
                    32 (32px Uniform, all sides)
                    
                    10x20 (10px Horizontal, 20px Vertical)
                    
                    10x20x30x40 (10px Left, 20px Top, 30px Right, 40px Bottom)

                    (default: 32).""")
    )

    actions.append(
        parser.add_argument(
            '-adb', '--adetailer-mask-blurs',
            nargs='+',
            action='store',
            type=_type_adetailer_mask_blur,
            default=None,
            metavar='ADETAILER_MASK_BLUR',
            dest='adetailer_mask_blurs',
            help="""The level of gaussian blur to apply
                    to the generated adetailer inpaint mask, which can help with 
                    smooth blending in of the inpainted feature. (default: 4)""")
    )

    actions.append(
        parser.add_argument(
            '-add', '--adetailer-mask-dilations',
            nargs='+',
            action='store',
            type=_type_adetailer_mask_dilation,
            default=None,
            metavar='ADETAILER_MASK_DILATION',
            dest='adetailer_mask_dilations',
            help="The amount of dilation applied to the adetailer inpaint mask, see: cv2.dilate. (default: 4)")
    )

    actions.append(
        parser.add_argument(
            '-adsz', '--adetailer-sizes',
            nargs='+',
            action='store',
            type=_type_adetailer_size,
            default=None,
            metavar='ADETAILER_SIZE',
            dest='adetailer_sizes',
            help="""One or more target sizes for processing detected areas.
                    When specified, detected areas will always be scaled to this target size (with aspect ratio preserved)
                    for processing, then scaled back to the original size for compositing.
                    This can significantly improve detail quality for small detected features like faces or hands,
                    or reduce processing time for overly large detected areas.
                    The scaling is based on the larger dimension (width or height) of the detected area.
                    The optimal resampling method is automatically selected for both upscaling and downscaling.
                    Each value must be an integer greater than 1. (default: none - process at native resolution)""")
    )

    actions.append(
        parser.add_argument(
            '-adc', '--adetailer-crop-control-image',
            action='store_true',
            default=None,
            dest='adetailer_crop_control_image',
            help="""Should adetailer crop ControlNet control images to the feature detection area? 
                    Your input image and control image should be the the same dimension, otherwise 
                    this argument is ignored with a warning. When this argument is not specified, 
                    the control image provided is simply resized to the same size as the 
                    detection area."""))

    actions.append(
        parser.add_argument(
            '-te', '--text-encoders', nargs='+', type=_type_text_encoder, action='store', default=None,
            metavar='TEXT_ENCODER_URIS', dest='text_encoder_uris',
            help=f'''Specify Text Encoders for the main model using URIs, main models
                    may use one or more text encoders depending on the --model-type value and other
                    dgenerate arguments. See: --text-encoders help for information
                    about what text encoders are needed for your invocation.
                    
                    Examples: 
                    
                    NOWRAP!
                    "CLIPTextModel;model=huggingface/text_encoder"
                    "CLIPTextModelWithProjection;model=huggingface/text_encoder;revision=main"
                    "T5EncoderModel;model=text_encoder_folder_on_disk"
                    "DistillT5EncoderModel;model=text_encoder_folder_on_disk"
                    
                    For main models which require multiple text encoders, the + symbol may be used
                    to indicate that a default value should be used for a particular text encoder,
                    for example: --text-encoders + + huggingface/encoder3. Any trailing text
                    encoders which are not specified are given their default value.
                    
                    The value "null" may be used to indicate that a specific text
                    encoder should not be loaded.
                    
                    The "revision" argument specifies the model revision to use for the Text Encoder
                    when loading from Hugging Face repository, (The Git branch / tag, default is "main").
                    
                    The "variant" argument specifies the Text Encoder model variant. If "variant" is specified
                    when loading from a Hugging Face repository or folder, weights will be loaded from "variant" filename,
                    e.g. "pytorch_model.<variant>.safetensors".
                    For this argument, "variant" defaults to the value of --variant if it is not specified in the URI.
                    
                    The "subfolder" argument specifies the Text Encoder model subfolder, if specified when loading 
                    from a Hugging Face repository or folder, weights from the specified subfolder.
                    
                    The "dtype" argument specifies the Text Encoder model precision, it defaults to the value of -t/--dtype
                    and should be one of: {_SUPPORTED_DATA_TYPES_PRETTY}.
                    
                    The "quantizer" URI argument can be used to specify a quantization backend
                    for the text encoder using the same URI syntax as --quantizer. This is supported
                    when loading from Hugging Face repo slugs / folders on disk, and when using the "mode"
                    argument with monolithic (non-sharded) checkpoints. This is not supported when
                    loading a submodule out of a combined checkpoint file with "subfolder".
                    If working from the command line you may need to nested quote this URI, i.e:
                    
                    NOWRAP!
                    --text-encoders 'CLIPTextModel;model=huggingface/text_encoder;quantizer="bnb;bits=8"'
                    
                    The "mode" argument can be used to load monolithic single file checkpoints with specific
                    architecture configurations. Available modes are:
                    
                    Flux & T5 universal modes:
                    
                    NOWRAP!
                    * "clip-l" for monolithic Flux CLIP-L checkpoints
                    * "t5-xxl" for monolithic Flux T5 checkpoints
                    
                    SD3 and SD3.5 specific modes:
                    
                    NOWRAP!
                    * "clip-l-sd3" for SD3/SD3.5 medium CLIP-L checkpoints
                    * "clip-g-sd3" for SD3/SD3.5 medium CLIP-G checkpoints
                    * "t5-xxl-sd3" for SD3/SD3.5 T5-XXL checkpoints
                    * "clip-l-sd35-large" for SD3.5 large variant CLIP-L checkpoints
                    * "clip-g-sd35-large" for SD3.5 large variant CLIP-G checkpoints
                    
                    
                    The "mode" option is mutually exclusive with "subfolder".
                    
                    Available encoder classes are:
                    
                    NOWRAP!
                    * CLIPTextModel
                    * CLIPTextModelWithProjection
                    * T5EncoderModel
                    * DistillT5EncoderModel (see: LifuWang/DistillT5)
                    * ChatGLMModel (for Kolors models)
                    
                    If you wish to load weights directly from a path on disk, you must point this argument at the folder
                    they exist in, which should also contain the config.json file for the Text Encoder.
                    For example, a downloaded repository folder from Hugging Face.'''
        )
    )

    actions.append(
        parser.add_argument(
            '-te2', '--second-model-text-encoders', nargs='+', type=_type_text_encoder, action='store', default=None,
            metavar='TEXT_ENCODER_URIS', dest='second_model_text_encoder_uris',
            help="""--text-encoders but for the SDXL refiner or Stable Cascade decoder model."""
        )
    )

    actions.append(
        parser.add_argument(
            '-un', '--unet', action='store', default=None, metavar="UNET_URI", dest='unet_uri',
            help=f"""Specify a UNet using a URI.
                    
                    Examples: 
                    
                    NOWRAP!
                    "huggingface/unet", "huggingface/unet;revision=main", "unet_folder_on_disk"
                    
                    The "revision" argument specifies the model revision to use for the UNet when loading from
                    Hugging Face repository, (The Git branch / tag, default is "main").
                    
                    The "variant" argument specifies the UNet model variant. If "variant" is specified
                    when loading from a Hugging Face repository or folder, weights will be loaded from "variant" filename,
                    e.g. "pytorch_model.<variant>.safetensors.
                    For this argument, "variant" defaults to the value of --variant if it is not specified in the URI.
                    
                    The "subfolder" argument specifies the UNet model subfolder, if specified when loading from a
                    Hugging Face repository or folder, weights from the specified subfolder. If you are loading
                    from a combined single file checkpoint containing multiple components, this value will be 
                    used to determine the key in the checkpoint that contains the unet, by default "unet" is 
                    used if subfolder is not provided.
                    
                    The "dtype" argument specifies the UNet model precision, it defaults to the value of -t/--dtype
                    and should be one of: {_SUPPORTED_DATA_TYPES_PRETTY}.
                    
                    The "quantizer" argument specifies a quantization backend and configuration for the
                    UNet model individually, and uses the same URI syntax as --quantizer. 
                    If working from the command line you may need to nested quote this URI, i.e:
                    
                    NOWRAP!
                    --unet 'huggingface/unet;quantizer="bnb;bits=8"'
                    
                    If you wish to load weights directly from a path on disk, you must point this argument at the folder
                    they exist in, which should also contain the config.json file for the UNet.
                    For example, a downloaded repository folder from Hugging Face."""
        )
    )

    actions.append(
        parser.add_argument(
            '-un2', '--second-model-unet', action='store', default=None, metavar="UNET_URI",
            dest='second_model_unet_uri',
            help=f"""Specify a second UNet, this is only valid when using SDXL or Stable Cascade
                    model types. This UNet will be used for the SDXL refiner, or Stable Cascade decoder model."""
        )
    )

    actions.append(
        parser.add_argument(
            '-tf', '--transformer', action='store', default=None, metavar="TRANSFORMER_URI", dest='transformer_uri',
            help=f"""Specify a Stable Diffusion 3 or Flux Transformer model using a URI.
                    
                    Examples: 
                    
                    NOWRAP!
                    "huggingface/transformer"
                    "huggingface/transformer;revision=main"
                    "transformer_folder_on_disk"
                    
                    Blob links / single file loads are supported for SD3 Transformers.
                    
                    The "revision" argument specifies the model revision to use for the Transformer when loading from
                    Hugging Face repository or blob link, (The Git branch / tag, default is "main").
                    
                    The "variant" argument specifies the Transformer model variant. If "variant" is specified
                    when loading from a Hugging Face repository or folder, weights will be loaded from "variant" filename,
                    e.g. "pytorch_model.<variant>.safetensors.
                    For this argument, "variant" defaults to the value of --variant if it is not specified in the URI.
                    
                    The "subfolder" argument specifies the Transformer model subfolder, if specified when loading from a
                    Hugging Face repository or folder, weights from the specified subfolder.
                    
                    The "dtype" argument specifies the Transformer model precision, it defaults to the value of -t/--dtype
                    and should be one of: {_SUPPORTED_DATA_TYPES_PRETTY}.
                    
                    The "quantizer" argument specifies a quantization backend and configuration for the
                    Transformer model individually, and uses the same URI syntax as --quantizer. 
                    If working from the command line you may need to nested quote this URI, i.e:
                    
                    NOWRAP!
                    --transformer 'huggingface/transformer;quantizer="bnb;bits=8"'
                    
                    If you wish to load a weights file directly from disk, the simplest
                    way is: --transformer "transformer.safetensors", or with a dtype "transformer.safetensors;dtype=float16".
                    All loading arguments except "dtype" and "quantizer" are unused in this case and may produce an
                    error message if used.
                    
                    If you wish to load a specific weight file from a Hugging Face repository, use the blob link
                    loading syntax: --transformer
                    "AutoencoderKL;https://huggingface.co/UserName/repository-name/blob/main/transformer.safetensors",
                    the "revision" argument may be used with this syntax."""
        )
    )

    actions.append(
        parser.add_argument(
            '-vae', '--vae', action='store', default=None, metavar="VAE_URI", dest='vae_uri',
            help=f"""Specify a VAE using a URI, the URI syntax is: 
                    "AutoEncoderClass;model=(Hugging Face repository slug/blob link or file/folder path)".
                    
                    Examples: 
                    
                    NOWRAP!
                    "AutoencoderKL;model=vae.pt"
                    "AsymmetricAutoencoderKL;model=huggingface/vae"
                    "AutoencoderTiny;model=huggingface/vae"
                    "ConsistencyDecoderVAE;model=huggingface/vae"
                    
                    The AutoencoderKL encoder class accepts Hugging Face repository slugs/blob links,
                    .pt, .pth, .bin, .ckpt, and .safetensors files.
                    
                    Other encoders can only accept Hugging Face repository slugs/blob links, or a path to
                    a folder on disk with the model configuration and model file(s).
                    
                    If an AutoencoderKL VAE model file exists at a URL which serves the file as
                    a raw download, you may provide an http/https link to it and it will be
                    downloaded to dgenerate's web cache.
                    
                    Aside from the "model" argument, there are four other optional arguments that can be specified,
                    these are: "revision", "variant", "subfolder", "dtype".
                    
                    They can be specified as so in any order, they are not positional:
                    
                    NOWRAP!
                    "AutoencoderKL;model=huggingface/vae;revision=main;variant=fp16;subfolder=sub_folder;dtype=float16"
                    
                    The "revision" argument specifies the model revision to use for the VAE when loading from
                    Hugging Face repository or blob link, (The Git branch / tag, default is "main").
                    
                    The "variant" argument specifies the VAE model variant. If "variant" is specified
                    when loading from a Hugging Face repository or folder, weights will be loaded from
                    "variant" filename, e.g. "pytorch_model.<variant>.safetensors. "variant" in the case
                    of --vae does not default to the value of --variant to prevent failures during
                    common use cases.
                    
                    The "subfolder" argument specifies the VAE model subfolder, if specified when loading from a
                    Hugging Face repository or folder, weights from the specified subfolder.
                    
                    The "extract" argument specifies that "model" points at a combind single file 
                    checkpoint containing multiple components such as the UNet and Text Encoders, and 
                    that we should extract the VAE. When using this argument you can use "subfolder" to
                    indicate the key in the checkpoint containing the model, this defaults to "vae".
                    
                    The "dtype" argument specifies the VAE model precision, it defaults to the value of -t/--dtype
                    and should be one of: {_SUPPORTED_DATA_TYPES_PRETTY}.
                    
                    If you wish to load a weights file directly from disk, the simplest
                    way is: --vae "AutoencoderKL;my_vae.safetensors", or with a dtype "AutoencoderKL;my_vae.safetensors;dtype=float16".
                    All loading arguments except "dtype" are unused in this case and may produce an error message if used.
                    
                    If you wish to load a specific weight file from a Hugging Face repository, use the blob link
                    loading syntax: --vae "AutoencoderKL;https://huggingface.co/UserName/repository-name/blob/main/vae_model.safetensors",
                    the "revision" argument may be used with this syntax."""
        )
    )

    actions.append(
        parser.add_argument(
            '-vt', '--vae-tiling', action='store_true', default=False,
            help="""Enable VAE tiling. Assists in the generation of
                    large images with lower memory overhead. The VAE will split the input tensor
                    into tiles to compute decoding and encoding in several steps. This is
                    useful for saving a large amount of memory and to allow processing larger images.
                    Note that if you are using --control-nets you may still run into memory
                    issues generating large images, or with --batch-size greater than 1."""
        )
    )

    actions.append(
        parser.add_argument(
            '-vs', '--vae-slicing', action='store_true', default=False,
            help="""Enable VAE slicing. Assists in the generation
                    of large images with lower memory overhead. The VAE will split the input tensor
                    in slices to compute decoding in several steps. This is useful to save some memory,
                    especially when --batch-size is greater than 1. Note that if you are using --control-nets
                    you may still run into memory issues generating large images."""
        )
    )

    actions.append(
        parser.add_argument(
            '-lra', '--loras', nargs='+', action='store', default=None, metavar="LORA_URI", dest='lora_uris',
            help="""Specify one or more LoRA models using URIs. These should be a
                    Hugging Face repository slug / blob link, path to model file on disk (for example, a .pt, .pth, .bin,
                    .ckpt, or .safetensors file), or model folder containing model files.
                    
                    If a LoRA model file exists at a URL which serves the file as
                    a raw download, you may provide an http/https link to it and it will be
                    downloaded to dgenerate's web cache.
                    
                    Optional arguments can be provided after a LoRA model specification,
                    these are: "scale", "revision", "subfolder", and "weight-name".
                    
                    They can be specified as so in any order, they are not positional:
                    
                    NOWRAP!
                    "huggingface/lora;scale=1.0;revision=main;subfolder=repo_subfolder;weight-name=lora.safetensors"
                    
                    The "scale" argument indicates the scale factor of the LoRA.
                    
                    The "revision" argument specifies the model revision to use for the LoRA when loading from
                    Hugging Face repository, (The Git branch / tag, default is "main").
                    
                    The "subfolder" argument specifies the LoRA model subfolder, if specified when loading from a
                    Hugging Face repository or folder, weights from the specified subfolder.
                    
                    The "weight-name" argument indicates the name of the weights file to be loaded when
                    loading from a Hugging Face repository or folder on disk.
                    
                    If you wish to load a weights file directly from disk, the simplest
                    way is: --loras "my_lora.safetensors", or with a scale "my_lora.safetensors;scale=1.0",
                    all other loading arguments are unused in this case and may produce an error message if used."""
        )
    )

    actions.append(
        parser.add_argument(
            '-lrfs', '--lora-fuse-scale', default=None, type=float, metavar="LORA_FUSE_SCALE",
            help="""LoRA weights are merged into the main model at this scale. When specifying multiple
                    LoRA models, they are fused together into one set of weights using their individual scale values,
                    after which they are fused into the main model at this scale value. (default: 1.0)."""
        )
    )

    actions.append(
        parser.add_argument(
            '-ie', '--image-encoder', action='store', default=None, metavar="IMAGE_ENCODER_URI",
            dest='image_encoder_uri',
            help=f"""Specify an Image Encoder using a URI. 
                    
                    Image Encoders are used with --ip-adapters models, and must be specified if none of the
                    loaded --ip-adapters contain one. An error will be produced in this situation, which
                    requires you to use this argument.
                    
                    An image encoder can also be manually specified for Stable Cascade models.
                    
                    Examples: 
                    
                    NOWRAP!
                    "huggingface/image_encoder"
                    "huggingface/image_encoder;revision=main"
                    "image_encoder_folder_on_disk"
                    
                    Blob links / single file loads are not supported for Image Encoders.
                    
                    The "revision" argument specifies the model revision to use for the Image Encoder when loading from
                    Hugging Face repository or blob link, (The Git branch / tag, default is "main").
                    
                    The "variant" argument specifies the Image Encoder model variant. If "variant" is specified when
                    loading from a Hugging Face repository or folder, weights will be loaded from "variant" filename,
                    e.g. "pytorch_model.<variant>.safetensors.
                    
                    Similar to --vae, "variant" does not default to the value of --variant in order to prevent
                    errors with common use cases. If you specify multiple IP Adapters, they must all
                    have the same "variant" value or you will receive a usage error.
                    
                    The "subfolder" argument specifies the Image Encoder model subfolder, if specified when loading from a
                    Hugging Face repository or folder, weights from the specified subfolder.
                    
                    The "dtype" argument specifies the Image Encoder model precision, it defaults to the value of -t/--dtype
                    and should be one of: {_SUPPORTED_DATA_TYPES_PRETTY}.
                    
                    If you wish to load weights directly from a path on disk, you must point this argument at the folder
                    they exist in, which should also contain the config.json file for the Image Encoder. For example, a downloaded
                    repository folder from Hugging Face."""
        )
    )

    actions.append(
        parser.add_argument(
            '-ipa', '--ip-adapters', nargs='+', action='store', default=None, metavar="IP_ADAPTER_URI",
            dest='ip_adapter_uris',
            help="""Specify one or more IP Adapter models using URIs. These should be a
                    Hugging Face repository slug / blob link, path to model file on disk (for example, a .pt, .pth, .bin,
                    .ckpt, or .safetensors file), or model folder containing model files.
                    
                    If an IP Adapter model file exists at a URL which serves the file as
                    a raw download, you may provide an http/https link to it and it will be
                    downloaded to dgenerate's web cache.
                    
                    Optional arguments can be provided after an IP Adapter model specification,
                    these are: "scale", "revision", "subfolder", and "weight-name".
                    
                    They can be specified as so in any order, they are not positional:
                    
                    NOWRAP!
                    "huggingface/ip-adapter;scale=1.0;revision=main;subfolder=repo_subfolder;weight-name=ip_adapter.safetensors".
                    
                    The "scale" argument indicates the scale factor of the IP Adapter.
                    
                    The "revision" argument specifies the model revision to use for the IP Adapter
                    when loading from Hugging Face repository, (The Git branch / tag, default is "main").
                    
                    The "subfolder" argument specifies the IP Adapter model subfolder, if specified when
                    loading from a Hugging Face repository or folder, weights from the specified subfolder.
                    
                    The "weight-name" argument indicates the name of the weights file to be loaded when
                    loading from a Hugging Face repository or folder on disk.
                    
                    If you wish to load a weights file directly from disk, the simplest
                    way is: --ip-adapters "ip_adapter.safetensors", or with a scale "ip_adapter.safetensors;scale=1.0",
                    all other loading arguments are unused in this case and may produce an error message if used."""
        )
    )

    actions.append(
        parser.add_argument(
            '-ti', '--textual-inversions', nargs='+', action='store', default=None, metavar="URI",
            dest='textual_inversion_uris',
            help="""Specify one or more Textual Inversion models using URIs.
                    These should be a Hugging Face repository slug / blob link, path to model file on disk
                    (for example, a .pt, .pth, .bin, .ckpt, or .safetensors file), or model folder
                    containing model files.
                    
                    If a Textual Inversion model file exists at a URL which serves the file as
                    a raw download, you may provide an http/https link to it and it will be
                    downloaded to dgenerate's web cache.
                    
                    Optional arguments can be provided after the Textual Inversion model specification,
                    these are: "token", "revision", "subfolder", and "weight-name".
                    
                    They can be specified as so in any order, they are not positional:
                    
                    NOWRAP!
                    "huggingface/ti_model;revision=main;subfolder=repo_subfolder;weight-name=ti_model.safetensors".
                    
                    The "token" argument can be used to override the prompt token used for the
                    textual inversion prompt embedding. For normal Stable Diffusion the default
                    token value is provided by the model itself, but for Stable Diffusion XL and Flux
                    the default token value is equal to the model file name with no extension and all
                    spaces replaced by underscores.
                    
                    The "revision" argument specifies the model revision to use for the Textual Inversion model
                    when loading from Hugging Face repository, (The Git branch / tag, default is "main").
                    
                    The "subfolder" argument specifies the Textual Inversion model subfolder, if specified
                    when loading from a Hugging Face repository or folder, weights from the specified subfolder.
                    
                    The "weight-name" argument indicates the name of the weights file to be loaded when
                    loading from a Hugging Face repository or folder on disk.
                    
                    If you wish to load a weights file directly from disk, the simplest way is: 
                    --textual-inversions "my_ti_model.safetensors", all other loading arguments
                    are unused in this case and may produce an error message if used."""
        )
    )

    image_guidance_group = parser.add_mutually_exclusive_group()

    actions.append(
        image_guidance_group.add_argument(
            '-cn', '--control-nets', nargs='+', action='store', default=None, metavar="CONTROLNET_URI",
            dest='controlnet_uris',
            help=f"""Specify one or more ControlNet models using URIs. This should be a
                    Hugging Face repository slug / blob link, path to model file on disk
                    (for example, a .pt, .pth, .bin, .ckpt, or .safetensors file), or model
                    folder containing model files.
                    
                    If a ControlNet model file exists at a URL which serves the file as
                    a raw download, you may provide an http/https link to it and it will be
                    downloaded to dgenerate's web cache.
                    
                    Optional arguments can be provided after the ControlNet model specification,
                    these are: "scale", "start", "end", "mode", "revision", "variant", "subfolder", and "dtype".
                    
                    They can be specified as so in any order, they are not positional:
                    
                    NOWRAP!
                    "huggingface/controlnet;scale=1.0;start=0.0;end=1.0;revision=main;variant=fp16;subfolder=repo_subfolder;dtype=float16".
                    
                    The "scale" argument specifies the scaling factor applied to the ControlNet model,
                    the default value is 1.0.
                    
                    The "start" argument specifies at what fraction of the total inference steps to begin applying
                    the ControlNet, defaults to 0.0, IE: the very beginning.
                    
                    The "end" argument specifies at what fraction of the total inference steps to stop applying
                    the ControlNet, defaults to 1.0, IE: the very end.
                    
                    The "mode" argument can be used when using --model-type sdxl / flux 
                    and a ControlNet Union model to specify the ControlNet mode. This may be a 
                    string or an integer.
                    
                    For --model-type sdxl Acceptable "mode" values are: 
                    
                    NOWRAP!
                        "openpose" = 0
                        "depth" = 1
                        "hed" = 2
                        "pidi" = 2
                        "scribble" = 2
                        "ted" = 2
                        "canny" = 3
                        "lineart" = 3
                        "anime_lineart" = 3
                        "mlsd" = 3
                        "normal" = 4
                        "segment" = 5
                        
                    
                    For --model-type flux Acceptable "mode" values are: 
                    
                    NOWRAP!
                        "canny" = 0
                        "tile" = 1
                        "depth" = 2
                        "blur" = 3
                        "pose" = 4
                        "gray" = 5
                        "lq" = 6
                    
                    The "revision" argument specifies the model revision to use for the ControlNet model
                    when loading from Hugging Face repository, (The Git branch / tag, default is "main").
                    
                    The "variant" argument specifies the ControlNet model variant, if "variant" is specified
                    when loading from a Hugging Face repository or folder, weights will be loaded from "variant" 
                    filename, e.g. "pytorch_model.<variant>.safetensors. "variant" defaults to automatic selection.
                    "variant" in the case of --control-nets does not default to the value of --variant to prevent
                    failures during common use cases.
                    
                    The "subfolder" argument specifies the ControlNet model subfolder, if specified
                    when loading from a Hugging Face repository or folder, weights from the specified subfolder.
                    
                    The "dtype" argument specifies the ControlNet model precision, it defaults to the value of -t/--dtype
                    and should be one of: {_SUPPORTED_DATA_TYPES_PRETTY}.
                    
                    If you wish to load a weights file directly from disk, the simplest way is: 
                    --control-nets "my_controlnet.safetensors" or --control-nets "my_controlnet.safetensors;scale=1.0;dtype=float16",
                    all other loading arguments aside from "scale", "start", "end", and "dtype" are unused in this case and may produce
                    an error message if used.
                    
                    If you wish to load a specific weight file from a Hugging Face repository, use the blob link
                    loading syntax: --control-nets
                    "https://huggingface.co/UserName/repository-name/blob/main/controlnet.safetensors",
                    the "revision" argument may be used with this syntax."""
        )
    )

    actions.append(
        image_guidance_group.add_argument(
            '-t2i', '--t2i-adapters', nargs='+', action='store', default=None, metavar="T2I_ADAPTER_URI",
            dest='t2i_adapter_uris',
            help=f"""Specify one or more T2IAdapter models using URIs. This should be a
                    Hugging Face repository slug / blob link, path to model file on disk
                    (for example, a .pt, .pth, .bin, .ckpt, or .safetensors file), or model
                    folder containing model files.
                    
                    If a T2IAdapter model file exists at a URL which serves the file as
                    a raw download, you may provide an http/https link to it and it will be
                    downloaded to dgenerate's web cache.
                    
                    Optional arguments can be provided after the T2IAdapter model specification,
                    these are: "scale", "revision", "variant", "subfolder", and "dtype".
                    
                    They can be specified as so in any order, they are not positional:
                    
                    NOWRAP!
                    "huggingface/t2iadapter;scale=1.0;revision=main;variant=fp16;subfolder=repo_subfolder;dtype=float16".
                    
                    The "scale" argument specifies the scaling factor applied to the T2IAdapter model,
                    the default value is 1.0.
                    
                    The "revision" argument specifies the model revision to use for the T2IAdapter model
                    when loading from Hugging Face repository, (The Git branch / tag, default is "main").
                    
                    The "variant" argument specifies the T2IAdapter model variant, if "variant" is specified when
                    loading from a Hugging Face repository or folder, weights will be loaded from "variant" filename,
                    e.g. "pytorch_model.<variant>.safetensors. "variant"  defaults to automatic selection.
                    "variant" in the case of --t2i-adapters does not default to the value of --variant to
                    prevent failures during common use cases.
                    
                    The "subfolder" argument specifies the ControlNet model subfolder, if specified
                    when loading from a Hugging Face repository or folder, weights from the specified subfolder.
                    
                    The "dtype" argument specifies the T2IAdapter model precision, it defaults to the value of -t/--dtype
                    and should be one of: {_SUPPORTED_DATA_TYPES_PRETTY}.
                    
                    If you wish to load a weights file directly from disk, the simplest way is: 
                    --t2i-adapters "my_t2i_adapter.safetensors" or --t2i-adapters "my_t2i_adapter.safetensors;scale=1.0;dtype=float16",
                    all other loading arguments aside from "scale" and "dtype" are unused in this case and may produce
                    an error message if used.
                    
                    If you wish to load a specific weight file from a Hugging Face repository, use the blob link
                    loading syntax: --t2i-adapters
                    "https://huggingface.co/UserName/repository-name/blob/main/t2i_adapter.safetensors",
                    the "revision" argument may be used with this syntax."""
        )
    )

    actions.append(
        parser.add_argument(
            '-q', '--quantizer',
            action='store', default=None, metavar="QUANTIZER_URI", dest='quantizer_uri',
            help=f"""Global quantization configuration via URI.
            
            This URI specifies the quantization backend and its configuration.
            
            Quantization will be applied to all text encoders, and unet / transformer 
            models with the provided settings when using this argument. ControlNet models
            are NOT quantized by default and must be explicitly included via --quantizer-map.
            
            If you wish to specify different quantization types per encoder, unet / transformer,
            or controlnet, you should use the "quantizer" URI argument of --text-encoders, 
            --unet / --transformer, or --control-nets to specify the quantization settings 
            on a per model basis.
            
            Available backends are: (bnb / bitsandbytes, sdnq)
            
            bitsandbytes can be specified with "bnb" or "bitsandbytes"
            
            Example: 
            
            NOWRAP!
            --quantizer bnb;bits=4
            
            or: 
            
            NOWRAP!
            --quantizer bitsandbytes;bits=4
            
            The bitsandbytes backend URI possesses these arguments and defaults:
            
            NOWRAP!
            * bits: int = 8 (must be 4 or 8)
            * bits4-compute-dtype: str = None (auto set when not specified)
            * bits4-quant-type: str = "fp4"
            * bits4-use-double-quant = False
            * bits4-quant-storage: str = None
            
            SDNQ (SD.Next Quantization) backend can be specified with "sdnq"
            
            Example:
            
            NOWRAP!
            --quantizer sdnq;type=int4
            
            The SDNQ backend URI possesses these arguments and defaults:
            
            NOWRAP!
            * type: str = "int8"
            * group-size: int = 0 (how many tensor elements will share a quantization group, must be >= 0)
            * quant-conv: bool = False (quantize convolutional layers)
            * quantized-matmul: bool = False (use quantized matrix multiplication)
            * quantized-matmul-conv: bool = False (use quantized matrix multiplication for convolutional layers)
            
            SDNQ supports the quantization types:
            
            NOWRAP!
            * bool 
            * int8, int7, int6, int5, int4, int3, int2
            * uint8, uint7, uint6, uint5, uint4, uint3, uint2, uint1, 
            * float8_e4m3fn, float8_e4m3fnuz, float8_e5m2, float8_e5m2fnuz
            """
        )
    )

    actions.append(
        parser.add_argument(
            '--quantizer-help', action='store', nargs='*', default=None, metavar="QUANTIZER_NAME", dest=None,
            help="""
            Use this option alone with no model specification in order to list quantizer 
            (quantization backend) names.
                    
            Specifying one or more quantizer names after this option will cause usage
            documentation for the specified quantization backend to be printed."""
        )
    )

    actions.append(
        parser.add_argument(
            '-qm', '--quantizer-map', type=_type_quantizer_map,
            nargs='+', action='store', default=None, metavar="SUBMODULE", dest='quantizer_map',
            help=f"""Global quantization map, used with --quantizer.
            
            This argument can be used to specify which sub-modules have the quantization pre-process
            performed on them.
            
            By default when a --quantizer URI is specified, the UNet / Transformer, and all Text Encoders
            are processed. ControlNet models are NOT processed by default.
            
            When using --quantizer, you can use this argument to specify exactly which sub-modules undergo
            quantization.
            
            Accepted values are: "unet", "transformer", "text_encoder", "text_encoder_2", "text_encoder_3", "controlnet"
            """
        )
    )

    actions.append(
        parser.add_argument(
            '-q2', '--second-model-quantizer',
            action='store', default=None, metavar="QUANTIZER_URI", dest='second_model_quantizer_uri',
            help=f"""
            Global quantization configuration via URI for the secondary model, 
            such as the SDXL Refiner or Stable Cascade decoder. See: --quantizer for syntax examples.
            """
        )
    )

    actions.append(
        parser.add_argument(
            '-qm2', '--second-model-quantizer-map', type=_type_quantizer_map,
            nargs='+', action='store', default=None, metavar="SUBMODULE", dest='second_model_quantizer_map',
            help=f"""
            Global quantization map for the secondary model, used with --second-model-quantizer.
            This affects the SDXL Refiner or Stable Cascade decoder, See: --quantizer-map for syntax examples.
            """
        )
    )

    actions.append(
        parser.add_argument(
            '-sch',
            '--scheduler',
            '--schedulers',
            dest='scheduler_uri',
            action='store', nargs='+', default=None, metavar="SCHEDULER_URI",
            help=f"""Specify a scheduler (sampler) by URI. 
                    
                    NOWRAP!
                    Passing "help" to this argument will print the compatible schedulers for a model without generating any images. 
                    
                    NOWRAP!
                    Passing "helpargs" will yield a help message with a list of overridable arguments for each scheduler and their typical defaults.
                    
                    Arguments listed by "helpargs" can be overridden using the URI syntax typical to other dgenerate URI arguments.
                    
                    You may pass multiple scheduler URIs to this argument, each URI will be tried in turn.
                    """
        )
    )

    actions.append(
        parser.add_argument(
            '-sch2',
            '--second-model-scheduler',
            '--second-model-schedulers',
            dest='second_model_scheduler_uri',
            nargs='+', action='store', default=None, metavar="SCHEDULER_URI",
            help="""Specify a scheduler (sampler) by URI for the SDXL Refiner or Stable Cascade Decoder pass. 
                 Operates the exact same way as --scheduler including the "help" option. Passing 'helpargs' will 
                 yield a help message with a list of overridable arguments for each scheduler and their 
                 typical defaults. Defaults to the value of --scheduler.
                 
                 You may pass multiple scheduler URIs to this argument, each URI will be tried in turn.
                 """
        )
    )

    actions.append(
        parser.add_argument(
            '-fu', '--freeu-params',
            default=None,
            nargs='+',
            dest='freeu_params',
            metavar='CSV_FLOAT',
            type=_type_freeu_params,
            help=f"""FreeU is a technique for improving image quality by re-balancing the contributions from 
                     the UNet's skip connections and backbone feature maps.
                     
                     This can be used with no cost to performance, to potentially improve image quality.
                     
                     This argument can be used to specify The FreeU parameters: s1, s2, b1, and b2 in that order.
                     
                     It accepts CSV, for example: --freeu-params "0.9,0.2,1.1,1.2"
                     
                     If you supply multiple CSV strings, they will be tried in turn.
                     
                     This argument only applies to models that utilize a UNet: SD1.5/2, SDXL, and Kolors
                     
                     See: https://huggingface.co/docs/diffusers/main/en/using-diffusers/freeu
                     
                     And: https://github.com/ChenyangSi/FreeU
                     """
        )
    )

    actions.append(
        parser.add_argument(
            '-hd', '--hi-diffusion',
            action='store_true', default=False, dest='hi_diffusion',
            help=f"""Activate HiDiffusion for the primary model? 
            
                     This can increase the resolution at which the model can
                     output images while retaining quality with no overhead, and 
                     possibly improved performance.
                     
                     NOWRAP!
                     See: https://github.com/megvii-research/HiDiffusion
                     
                     This is supported for --model-type sd, sdxl, kolors, pix2pix, and sdxl-pix2pix.
                     """
        )
    )

    actions.append(
        parser.add_argument(
            '--hi-diffusion-no-win-attn',
            action='store_true', default=False, dest='hi_diffusion_no_win_attn',
            help=f"""Disable window attention when using HiDiffusion for the primary model?
            
                     This disables the MSW-MSA (Multi-Scale Window Multi-Head Self-Attention) component of HiDiffusion.
                     
                     NOWRAP!
                     See: https://github.com/megvii-research/HiDiffusion
                     
                     This is supported for: --model-type sd, sdxl, and --kolors.
                     """
        )
    )

    actions.append(
        parser.add_argument(
            '--hi-diffusion-no-raunet',
            action='store_true', default=False, dest='hi_diffusion_no_raunet',
            help=f"""Disable RAU-Net when using HiDiffusion for the primary model?
            
                     This disables the Resolution-Aware U-Net component of HiDiffusion.
                     
                     NOWRAP!
                     See: https://github.com/megvii-research/HiDiffusion
                     
                     This is supported for: --model-type sd, sdxl, and --kolors.
                     """
        )
    )

    # SADA (Stability-guided Adaptive Diffusion Acceleration) arguments
    actions.append(
        parser.add_argument(
            '--sada',
            action='store_true', default=False, dest='sada',
            help="""Enable SADA (Stability-guided Adaptive Diffusion Acceleration) with model-specific default parameters for the primary model.
            
            
            This is equivalent to setting all SADA parameters to their model-specific default values:
            
            NOWRAP!
            - SD/SD2: 
                * --sada-max-downsamples 1 
                * --sada-sxs 3 
                * --sada-sys 3 
                * --sada-lagrange-terms 4 
                * --sada-lagrange-ints 4 
                * --sada-lagrange-steps 24 
                * --sada-max-fixes 5120
            - SDXL/Kolors: 
                * --sada-max-downsamples 2 
                * --sada-sxs 3 
                * --sada-sys 3 
                * --sada-lagrange-terms 4 
                * --sada-lagrange-ints 4 
                * --sada-lagrange-steps 24 
                * --sada-max-fixes 10240
            - Flux: 
                * --sada-max-downsamples 0 
                * --sada-lagrange-terms 3 
                * --sada-lagrange-ints 4 
                * --sada-lagrange-steps 20 
                * --sada-max-fixes 0
            
            NOWRAP!
            See: https://github.com/Ting-Justin-Jiang/sada-icml
            
            This is supported for: --model-type sd, sdxl, kolors, flux*.
            
            SADA is not compatible with HiDiffusion, DeepCache, or TeaCache.
            """
        )
    )

    actions.append(
        parser.add_argument(
            '--sada-max-downsamples',
            metavar='INTEGER',
            nargs='+', dest='sada_max_downsamples', type=_type_sada_max_downsamples,
            help="""SADA maximum downsample factors for the primary model.
            
            Controls the maximum downsample factor in the SADA algorithm. 
            Lower values can improve quality but may reduce speedup.
            
            
            Model-specific defaults:
            
            NOWRAP!
            - SD/SD2: 1
            - SDXL/Kolors: 2
            - Flux: 0
            
            Supplying any SADA parameter implies that SADA is enabled.
            
            This is supported for: --model-type sd, sdxl, kolors, flux*.
            
            Each value supplied will be tried in turn.
            """
        )
    )

    actions.append(
        parser.add_argument(
            '--sada-sxs',
            metavar='INTEGER',
            nargs='+', dest='sada_sxs', type=_type_sada_sxs,
            help="""SADA spatial downsample factors X for the primary model.
            
            Controls the spatial downsample factor in the X dimension.
            Higher values can increase speedup but may affect quality.
            
            
            Model-specific defaults:
            
            NOWRAP!
            - SD/SD2: 3
            - SDXL/Kolors: 3
            - Flux: 0 (not used)
            
            Supplying any SADA parameter implies that SADA is enabled.
            
            This is supported for: --model-type sd, sdxl, kolors, flux*.
            
            Each value supplied will be tried in turn.
            """
        )
    )

    actions.append(
        parser.add_argument(
            '--sada-sys',
            metavar='INTEGER',
            nargs='+', dest='sada_sys', type=_type_sada_sys,
            help="""SADA spatial downsample factors Y for the primary model.
            
            Controls the spatial downsample factor in the Y dimension.
            Higher values can increase speedup but may affect quality.
            
            
            Model-specific defaults:
            
            NOWRAP!
            - SD/SD2: 3
            - SDXL/Kolors: 3
            - Flux: 0 (not used)
            
            Supplying any SADA parameter implies that SADA is enabled.
            
            This is supported for: --model-type sd, sdxl, kolors, flux*.
            
            Each value supplied will be tried in turn.
            """
        )
    )

    actions.append(
        parser.add_argument(
            '--sada-acc-ranges',
            metavar='INTEGER',
            nargs='+', dest='sada_acc_ranges', type=_type_sada_acc_ranges,
            help="""SADA acceleration range start / end steps for the primary model.
            
            Defines the start / end step for SADA acceleration. 
            
            Starting step must be at least 3 as SADA leverages third-order dynamics.
            
            Defaults to "10,47".
            
            Supply ranges as comma seperated values, for example: --sada-acc-ranges "10,47" "12,40"
            
            Supplying any SADA parameter implies that SADA is enabled.
            
            This is supported for: --model-type sd, sdxl, kolors, flux*.
            
            Each value supplied will be tried in turn.
            """
        )
    )

    actions.append(
        parser.add_argument(
            '--sada-lagrange-terms',
            metavar='INTEGER',
            nargs='+', dest='sada_lagrange_terms', type=_type_sada_lagrange_terms,
            help="""SADA Lagrangian interpolation terms for the primary model.
            
            Number of terms to use in Lagrangian interpolation. 
            Set to 0 to disable Lagrangian interpolation.
            
            Model-specific defaults:
            
            NOWRAP!
            - SD/SD2: 4
            - SDXL/Kolors: 4
            - Flux: 3
            
            Supplying any SADA parameter implies that SADA is enabled.
            
            This is supported for: --model-type sd, sdxl, kolors, flux*.
            
            Each value supplied will be tried in turn.
            """
        )
    )

    actions.append(
        parser.add_argument(
            '--sada-lagrange-ints',
            metavar='INTEGER',
            nargs='+', dest='sada_lagrange_ints', type=_type_sada_lagrange_ints,
            help="""SADA Lagrangian interpolation intervals for the primary model.
            
            Interval for Lagrangian interpolation. Must be compatible with 
            sada-lagrange-steps (lagrange-step %% lagrange-int == 0).
            
            Model-specific defaults:
            
            NOWRAP!
            - SD/SD2: 4
            - SDXL/Kolors: 4
            - Flux: 4
            
            Supplying any SADA parameter implies that SADA is enabled.
            
            This is supported for: --model-type sd, sdxl, kolors, flux*.
            
            Each value supplied will be tried in turn.
            """
        )
    )

    actions.append(
        parser.add_argument(
            '--sada-lagrange-steps',
            metavar='INTEGER',
            nargs='+', dest='sada_lagrange_steps', type=_type_sada_lagrange_steps,
            help="""SADA Lagrangian interpolation steps for the primary model.
            
            Step value for Lagrangian interpolation. Must be compatible with 
            sada-lagrange-ints (lagrange-step %% lagrange-int == 0).
            
            Model-specific defaults:
            
            NOWRAP!
            - SD/SD2: 24
            - SDXL/Kolors: 24
            - Flux: 20
            
            Supplying any SADA parameter implies that SADA is enabled.
            
            This is supported for: --model-type sd, sdxl, kolors, flux*.
            
            Each value supplied will be tried in turn.
            """
        )
    )

    actions.append(
        parser.add_argument(
            '--sada-max-fixes',
            metavar='INTEGER',
            nargs='+', dest='sada_max_fixes', type=_type_sada_max_fixes,
            help="""SADA maximum fixed memories for the primary model.
            
            Maximum amount of fixed memory to use in SADA optimization.
            
           
            Model-specific defaults:
            
             NOWRAP!
            - SD/SD2: 5120 (5 * 1024)
            - SDXL/Kolors: 10240 (10 * 1024)
            - Flux: 0
            
            Supplying any SADA parameter implies that SADA is enabled.
            
            This is supported for: --model-type sd, sdxl, kolors, flux*.
            
            Each value supplied will be tried in turn.
            """
        )
    )

    actions.append(
        parser.add_argument(
            '--sada-max-intervals',
            metavar='INTEGER',
            nargs='+', dest='sada_max_intervals', type=_type_sada_max_intervals,
            help="""SADA maximum intervals for optimization for the primary model.
            
            Maximum interval between optimizations in the SADA algorithm.
            
            Defaults to 4.
            
            Supplying any SADA parameter implies that SADA is enabled.
            
            This is supported for: --model-type sd, sdxl, kolors, flux*.
            
            Each value supplied will be tried in turn.
            """
        )
    )

    actions.append(
        parser.add_argument(
            '-rfu', '--sdxl-refiner-freeu-params',
            default=None,
            nargs='+',
            dest='sdxl_refiner_freeu_params',
            metavar='CSV_FLOAT',
            type=_type_freeu_params,
            help=f"""FreeU parameters for the SDXL refiner, see: --freeu-params"""
        )
    )

    actions.append(
        parser.add_argument(
            '-dc', '--deep-cache',
            action='store_true', default=False, dest='deep_cache',
            help=f"""Activate DeepCache for the main model?

                  DeepCache caches intermediate attention layer outputs to speed up
                  the diffusion process. Recommended for higher inference steps.
                  
                  NOWRAP!
                  See: https://github.com/horseee/DeepCache
                  
                  This is supported for Stable Diffusion, Stable Diffusion XL,
                  Stable Diffusion Upscaler X4, Kolors, and Pix2Pix variants.
                  """
        )
    )

    actions.append(
        parser.add_argument(
            '-dci', '--deep-cache-intervals',
            metavar='INTEGER',
            nargs='+', dest='deep_cache_intervals', type=_type_deep_cache_interval,
            help="""Cache interval for DeepCache for the main model.
            
            Controls how frequently the attention layers are cached during
            the diffusion process. Lower values cache more frequently, potentially
            resulting in more speedup but using more memory.
            
            This value must be greater than zero.
            
            Each value supplied will be tried in turn.
            
            Supplying any values implies --deep-cache.
            
            This is supported for Stable Diffusion, Stable Diffusion XL,
            Stable Diffusion Upscaler X4, Kolors, and Pix2Pix variants.
            
            (default: 5)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-dcb', '--deep-cache-branch-ids',
            metavar='INTEGER',
            nargs='+', dest='deep_cache_branch_ids', type=_type_deep_cache_branch_id,
            help="""Branch ID for DeepCache for the main model.
            
            Controls which branches of the UNet attention blocks the caching
            is applied to. Advanced usage only.
            
            This value must be greater than or equal to 0.
            
            Each value supplied will be tried in turn.
            
            Supplying any values implies --deep-cache.
            
            This is supported for Stable Diffusion, Stable Diffusion XL,
            Stable Diffusion Upscaler X4, Kolors, and Pix2Pix variants.
            
            (default: 1)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-rdc', '--sdxl-refiner-deep-cache',
            action='store_true', default=None, dest='sdxl_refiner_deep_cache',
            help=f"""Activate DeepCache for the SDXL Refiner?
            
                  See: --deep-cache
                  
                  This is supported for Stable Diffusion XL and Kolors based models.
                  """
        )
    )

    actions.append(
        parser.add_argument(
            '-rdci', '--sdxl-refiner-deep-cache-intervals',
            metavar='INTEGER',
            nargs='+', dest='sdxl_refiner_deep_cache_intervals', type=_type_deep_cache_interval,
            help="""Cache interval for DeepCache for the SDXL Refiner.
            
            Controls how frequently the attention layers are cached during
            the diffusion process. Lower values cache more frequently, potentially
            resulting in more speedup but using more memory.
            
            This value must be greater than zero.
            
            Each value supplied will be tried in turn.
            
            Supplying any values implies --sdxl-refiner-deep-cache.
            
            This is supported for Stable Diffusion XL and Kolors based models.
            
            (default: 5)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-rdcb', '--sdxl-refiner-deep-cache-branch-ids',
            metavar='INTEGER',
            nargs='+', dest='sdxl_refiner_deep_cache_branch_ids', type=_type_deep_cache_branch_id,
            help="""Branch ID for DeepCache for the SDXL Refiner.
            
            Controls which branches of the UNet attention blocks the caching
            is applied to. Advanced usage only.
            
            This value must be greater than or equal to 0.
            
            Each value supplied will be tried in turn.
            
            Supplying any values implies --sdxl-refiner-deep-cache.
            
            This is supported for Stable Diffusion XL and Kolors based models.
            
            (default: 1)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-tc', '--tea-cache',
            action='store_true', default=False, dest='tea_cache',
            help=f"""Activate TeaCache for the primary model?
    
                     This is supported for Flux, TeaCache uses a novel caching mechanism 
                     in the forward pass of the flux transformer to reduce the amount of
                     computation needed to generate an image, this can speed up inference
                     with small amounts of quality loss.
                     
                     NOWRAP!
                     See: https://github.com/ali-vilab/TeaCache
                     
                     Also see: --tea-cache-rel-l1-thresholds
                     
                     This is supported for: --model-type flux*.
                     """
        )
    )

    actions.append(
        parser.add_argument(
            '-tcr', '--tea-cache-rel-l1-thresholds', metavar='FLOAT',
            nargs='*', type=_type_tea_cache_rel_l1_thresh, default=None, dest='tea_cache_rel_l1_thresholds',
            help=f"""TeaCache relative L1 thresholds to try when --tea-cache is enabled.
            
                     This should be one or more float values between 0.0 and 1.0, each value will be tried in turn.
    
                     Higher values mean more speedup.
                    
                     Defaults to 0.6 (2.0x speedup). 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 
                     0.6 for 2.0x speedup, 0.8 for 2.25x speedup
                     
                     NOWRAP!
                     See: https://github.com/ali-vilab/TeaCache
                     
                     Supplying any values implies --tea-cache.
                    
                     This is supported for: --model-type flux*.
                     
                     (default: 0.6)
                     """
        )
    )

    actions.append(
        parser.add_argument(
            '-ra', '--ras',
            action='store_true', default=False, dest='ras',
            help=f"""Activate RAS (Region-Adaptive Sampling) for the primary model?
            
                     This can increase inference speed with SD3.
                    
                     NOWRAP!
                     See: https://github.com/microsoft/ras
                    
                     This is supported for: --model-type sd3.
                     """
        )
    )

    actions.append(
        parser.add_argument(
            '-rif', '--ras-index-fusion',
            action='store_true', dest='ras_index_fusion', default=None,
            help="""Enable index fusion in RAS (Reinforcement Attention System) for the primary model?

            This can improve attention computation in RAS for SD3 models.
            
            Supplying this flag implies --ras.

            This is supported for: --model-type sd3, (but not for SD3.5 models)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-rsr', '--ras-sample-ratios',
            metavar='FLOAT',
            nargs='+', dest='ras_sample_ratios', type=_type_ras_sample_ratio,
            help="""Average sample ratios for each RAS step.

            For instance, setting this to 0.5 on a sequence of 4096 tokens will result in
            the noise of averagely 2048 tokens to be updated during each RAS step.

            Must be between 0.0 and 1.0 (non-inclusive)
            
            Each value will be tried in turn.
            
            Supplying any values implies --ras.

            This is supported for: --model-type sd3.

            (default: 0.5)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-rhr', '--ras-high-ratios',
            metavar='FLOAT',
            nargs='+', dest='ras_high_ratios', type=_type_ras_high_ratio,
            help="""Ratios of high value tokens to be cached in RAS.

            Based on the metric selected, the ratio of the high value chosen to be cached.

            Must be between 0.0 and 1.0 (non-inclusive) to balance the sample
            ratio between the main subject and the background.
            
            Each value will be tried in turn.
            
            Supplying any values implies --ras.

            This is supported for: --model-type sd3.

            (default: 1.0)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-rss', '--ras-starvation-scales',
            metavar='FLOAT',
            nargs='+', dest='ras_starvation_scales', type=_type_ras_starvation_scale,
            help="""Starvation scales for RAS patch selection.

            RAS tracks how often a token is dropped and incorporates this count as a scaling
            factor in the metric for selecting tokens. This scale factor prevents excessive blurring
            or noise in the final generated image.

            Larger scaling factor will result in more uniform sampling.

            Must be between 0.0 and 1.0 (non-inclusive)
            
            Each value will be tried in turn.
            
            Supplying any values implies --ras.

            This is supported for: --model-type sd3.

            (default: 0.1)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-rer', '--ras-error-reset-steps',
            metavar='CSV_INT',
            nargs='+', dest='ras_error_reset_steps', type=_type_ras_error_reset_steps,
            help="""Dense sampling steps to reset accumulated error in RAS.

            The dense sampling steps inserted between the RAS steps to reset the accumulated error.
            Each argument should be either a single integer or a comma-separated list of integers, 
            e.g. 12 or "12,22".
            
            Multiple values or comma-separated lists can be provided, and each will be tried in turn.
            
            NOWRAP!
            Example: --ras-error-reset-steps 12 "5,10,15"
            
            Supplying any values implies --ras.

            This is supported for: --model-type sd3.

            (default: "12,22")"""
        )
    )

    actions.append(
        parser.add_argument(
            '-rme', '--ras-metrics',
            metavar='RAS_METRIC',
            nargs='+', dest='ras_metrics', type=_type_ras_metric,
            help="""Metrics to try for RAS (Region-Adaptive Sampling).
            
            This controls how RAS measures the importance of tokens for caching.
            Valid values are "std" (standard deviation) or "l2norm" (L2 norm).
            
            Each value will be tried in turn.
            
            Supplying any values implies --ras.
            
            This is supported for: --model-type sd3.
            
            (default: "std")"""
        )
    )

    actions.append(
        parser.add_argument(
            '-rst', '--ras-start-steps',
            metavar='INTEGER',
            nargs='+', dest='ras_start_steps', type=_type_ras_start_steps,
            help="""Starting steps to try for RAS (Region-Adaptive Sampling).

            This controls when RAS begins applying its sampling strategy. 
            Must be greater than or equal to 1.
            
            Each value will be tried in turn.
            
            Supplying any values implies --ras.

            This is supported for: --model-type sd3.

            (default: 4)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-res', '--ras-end-steps',
            metavar='INTEGER',
            nargs='+', dest='ras_end_steps', type=_type_ras_end_steps,
            help="""Ending steps to try for RAS (Region-Adaptive Sampling).

            This controls when RAS stops applying its sampling strategy.
            Must be greater than or equal to 1.
            
            Each value will be tried in turn.
            
            Supplying any values implies --ras.

            This is supported for: --model-type sd3.

            (default: --inference-steps)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-rsn', '--ras-skip-num-steps',
            metavar='INTEGER',
            nargs='+', dest='ras_skip_num_steps', type=_type_ras_skip_num_step,
            help="""Skip steps for RAS (Region-Adaptive Sampling).

            This controls the number of steps to skip between RAS steps.
            
            The actual number of tokens skipped will be rounded down to the nearest multiple of 64
            to ensure efficient memory access patterns for attention computation.
            
            When used with --ras-skip-num-step-lengths greater than 0, this value will determine
            how the number of skipped tokens changes over time. Positive values will increase
            the number of skipped tokens over time, while negative values will decrease it.
            
            Each value will be tried in turn.
            
            Supplying any values implies --ras.

            This is supported for: --model-type sd3.

            (default: 0)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-rsl', '--ras-skip-num-step-lengths',
            metavar='INTEGER',
            nargs='+', dest='ras_skip_num_step_lengths', type=_type_ras_skip_num_step_length,
            help="""Skip step lengths for RAS (Region-Adaptive Sampling).

            This controls the length of steps to skip between RAS steps.
            Must be greater than or equal to 0.
            
            When set to 0, static dropping is used where the number of skipped tokens remains
            constant throughout the generation process.
            
            When greater than 0, dynamic dropping is enabled where the number of skipped tokens
            varies over time based on --ras-skip-num-steps. The pattern of skipping will repeat
            every --ras-skip-num-step-lengths steps.
            
            Each value will be tried in turn.
            
            Supplying any values implies --ras.

            This is supported for: --model-type sd3.

            (default: 0)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-pag', '--pag', action='store_true', default=False,
            help=f"""Use perturbed attention guidance? This is supported
            for --model-type sd, sdxl, and sd3 for most use cases.
            This enables PAG for the main model using default scale values."""
        )
    )

    actions.append(
        parser.add_argument(
            '-pags', '--pag-scales', nargs='+', action='store',
            type=_type_guidance_scale, default=None, metavar="FLOAT",
            help=f"""One or more perturbed attention guidance scales to try.
            Specifying values enables PAG for the main model.
            (default: [3.0])"""
        )
    )

    actions.append(
        parser.add_argument(
            '-pagas', '--pag-adaptive-scales', nargs='+', action='store',
            type=_type_guidance_scale, default=None, metavar="FLOAT",
            help=f"""One or more adaptive perturbed attention guidance scales to try.
            Specifying values enables PAG for the main model.
            (default: [0.0])"""
        )
    )

    actions.append(
        parser.add_argument(
            '-rpag', '--sdxl-refiner-pag', action='store_true', default=None,
            help=f"""Use perturbed attention guidance in the SDXL refiner? 
            This is supported for --model-type sdxl for most use cases.
            This enables PAG for the SDXL refiner model using default scale
            values."""
        )
    )

    actions.append(
        parser.add_argument(
            '-rpags', '--sdxl-refiner-pag-scales', nargs='+', action='store',
            type=_type_guidance_scale, default=None, metavar="FLOAT",
            help=f"""One or more perturbed attention guidance scales to try
            with the SDXL refiner pass. Specifying values enables PAG for the refiner.
            (default: [3.0])"""
        )
    )

    actions.append(
        parser.add_argument(
            '-rpagas', '--sdxl-refiner-pag-adaptive-scales', nargs='+', action='store',
            type=_type_guidance_scale, default=None, metavar="FLOAT",
            help=f"""One or more adaptive perturbed attention guidance scales to try
            with the SDXL refiner pass. Specifying values enables PAG for the refiner.
            (default: [0.0])"""
        )
    )

    _model_offload_group = parser.add_mutually_exclusive_group()

    actions.append(
        _model_offload_group.add_argument(
            '-mqo', '--model-sequential-offload', action='store_true', default=False,
            help="""Force sequential model offloading for the main pipeline, this may drastically reduce memory consumption
                    and allow large models to run when they would otherwise not fit in your GPUs VRAM.
                    Inference will be much slower. Mutually exclusive with --model-cpu-offload"""
        )
    )

    actions.append(
        _model_offload_group.add_argument(
            '-mco', '--model-cpu-offload', action='store_true', default=False,
            help="""Force model cpu offloading for the main pipeline, this may reduce memory consumption
                    and allow large models to run when they would otherwise not fit in your GPUs VRAM.
                    Inference will be slower. Mutually exclusive with --model-sequential-offload"""
        )
    )

    _model_offload_group2 = parser.add_mutually_exclusive_group()

    actions.append(
        _model_offload_group2.add_argument(
            '-mqo2', '--second-model-sequential-offload',
            dest='second_model_sequential_offload',
            action='store_true', default=None,
            help="""Force sequential model offloading for the SDXL Refiner or Stable Cascade Decoder pipeline, 
                    this may drastically reduce memory consumption and allow large models to run when they would 
                    otherwise not fit in your GPUs VRAM. Inference will be much slower. 
                    Mutually exclusive with --second-model-cpu-offload"""
        )
    )

    actions.append(
        _model_offload_group2.add_argument(
            '-mco2', '--second-model-cpu-offload',
            dest='second_model_cpu_offload',
            action='store_true', default=None,
            help="""Force model cpu offloading for the SDXL Refiner or Stable Cascade Decoder pipeline,
                    this may reduce memory consumption and allow large models to run when they would 
                    otherwise not fit in your GPUs VRAM. Inference will be slower. Mutually 
                    exclusive with --second-model-sequential-offload"""
        )
    )

    actions.append(
        parser.add_argument(
            '--s-cascade-decoder', action='store', default=None, metavar="MODEL_URI", dest='s_cascade_decoder_uri',
            help=f"""Specify a Stable Cascade (s-cascade) decoder model path using a URI.
                    This should be a Hugging Face repository slug / blob link, path to model file
                    on disk (for example, a .pt, .pth, .bin, .ckpt, or .safetensors file), or model
                    folder containing model files.
                    
                    Optional arguments can be provided after the decoder model specification,
                    these are: "revision", "variant", "subfolder", and "dtype".
                    
                    They can be specified as so in any order, they are not positional:
                    
                    NOWRAP!
                    "huggingface/decoder_model;revision=main;variant=fp16;subfolder=repo_subfolder;dtype=float16".
                    
                    The "revision" argument specifies the model revision to use for the decoder model
                    when loading from Hugging Face repository, (The Git branch / tag, default is "main").
                    
                    The "variant" argument specifies the decoder model variant and defaults to the value of
                    --variant. When "variant" is specified when loading from a Hugging Face repository or folder,
                    weights will be loaded from "variant" filename, e.g. "pytorch_model.<variant>.safetensors.
                    
                    The "subfolder" argument specifies the decoder model subfolder, if specified
                    when loading from a Hugging Face repository or folder, weights from the specified subfolder.
                    
                    The "dtype" argument specifies the Stable Cascade decoder model precision, it defaults to
                    the value of -t/--dtype and should be one of: {_SUPPORTED_DATA_TYPES_PRETTY}.
                    
                    If you wish to load a weights file directly from disk, the simplest way is: 
                    --sdxl-refiner "my_decoder.safetensors" or --sdxl-refiner "my_decoder.safetensors;dtype=float16",
                    all other loading arguments aside from "dtype" are unused in this case and may produce
                    an error message if used.
                    
                    If you wish to load a specific weight file from a Hugging Face repository, use the blob link
                    loading syntax: --s-cascade-decoder
                    "https://huggingface.co/UserName/repository-name/blob/main/decoder.safetensors",
                    the "revision" argument may be used with this syntax."""
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-refiner', action='store', default=None, metavar="MODEL_URI", dest='sdxl_refiner_uri',
            help=f"""Specify a Stable Diffusion XL (sdxl) refiner model path using a URI.
                    This should be a Hugging Face repository slug / blob link, path to model file
                    on disk (for example, a .pt, .pth, .bin, .ckpt, or .safetensors file), or model
                    folder containing model files.
                    
                    Optional arguments can be provided after the SDXL refiner model specification,
                    these are: "revision", "variant", "subfolder", and "dtype".
                    
                    They can be specified as so in any order, they are not positional:
                    
                    NOWRAP!
                    "huggingface/refiner_model_xl;revision=main;variant=fp16;subfolder=repo_subfolder;dtype=float16".
                    
                    The "revision" argument specifies the model revision to use for the refiner model
                    when loading from Hugging Face repository, (The Git branch / tag, default is "main").
                    
                    The "variant" argument specifies the SDXL refiner model variant and defaults to the value of
                    --variant. When "variant" is specified when loading from a Hugging Face repository or folder,
                    weights will be loaded from "variant" filename, e.g. "pytorch_model.<variant>.safetensors.
                    
                    The "subfolder" argument specifies the SDXL refiner model subfolder, if specified
                    when loading from a Hugging Face repository or folder, weights from the specified subfolder.
                    
                    The "dtype" argument specifies the SDXL refiner model precision, it defaults to the value of -t/--dtype
                    and should be one of: {_SUPPORTED_DATA_TYPES_PRETTY}.
                    
                    If you wish to load a weights file directly from disk, the simplest way is: 
                    --sdxl-refiner "my_sdxl_refiner.safetensors" or --sdxl-refiner "my_sdxl_refiner.safetensors;dtype=float16",
                    all other loading arguments aside from "dtype" are unused in this case and may produce
                    an error message if used.
                    
                    If you wish to load a specific weight file from a Hugging Face repository, use the blob link
                    loading syntax: --sdxl-refiner
                    "https://huggingface.co/UserName/repository-name/blob/main/refiner_model.safetensors",
                    the "revision" argument may be used with this syntax."""

        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-refiner-edit', action='store_true', default=None,
            help="""Force the SDXL refiner to operate in edit mode instead of cooperative denoising mode
                    as it would normally do for inpainting and ControlNet usage. The main model will perform
                    the full amount of inference steps requested by --inference-steps. The output of the main model
                    will be passed to the refiner model and processed with an image seed strength in img2img mode
                    determined by (1.0 - high-noise-fraction)"""
        )
    )

    # SDXL Main pipeline

    actions.append(
        parser.add_argument(
            '--sdxl-t2i-adapter-factors', nargs='+', action='store', metavar="FLOAT", default=None,
            type=_type_adapter_factor,
            help="""One or more SDXL specific T2I adapter factors to try, this controls the amount of
                    time-steps for which a T2I adapter applies guidance to an image, this is a value between
                    0.0 and 1.0. A value of 0.5 for example indicates that the T2I adapter is only active for
                    half the amount of time-steps it takes to completely render an image."""
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-aesthetic-scores', metavar="FLOAT", action='store', nargs='+', default=[], type=float,
            help="""One or more Stable Diffusion XL (sdxl) "aesthetic-score" micro-conditioning parameters.
                    Used to simulate an aesthetic score of the generated image by influencing the positive text
                    condition. Part of SDXL's micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952]."""
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-crops-coords-top-left', metavar="COORD", action='store', nargs='+', default=[],
            type=_type_image_coordinate,
            help="""One or more Stable Diffusion XL (sdxl) "negative-crops-coords-top-left" micro-conditioning
                    parameters in the format "0,0". --sdxl-crops-coords-top-left can be used to generate an image that
                    appears to be "cropped" from the position --sdxl-crops-coords-top-left downwards. Favorable,
                    well-centered images are usually achieved by setting --sdxl-crops-coords-top-left to "0,0".
                    Part of SDXL's micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952]."""
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-original-sizes', dest='sdxl_original_sizes', metavar="SIZE", action='store',
            nargs='+', default=[], type=_type_size,
            help="""One or more Stable Diffusion XL (sdxl) "original-size" micro-conditioning parameters in
                    the format (WIDTH)x(HEIGHT). If not the same as --sdxl-target-sizes the image will appear to be
                    down or up-sampled. --sdxl-original-sizes defaults to --output-size or the size of any input
                    images if not specified. Part of SDXL's micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952]"""
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-target-sizes', dest='sdxl_target_sizes', metavar="SIZE", action='store',
            nargs='+', default=[], type=_type_size,
            help="""One or more Stable Diffusion XL (sdxl) "target-size" micro-conditioning parameters in
                    the format (WIDTH)x(HEIGHT). For most cases, --sdxl-target-sizes should be set to the desired
                    height and width of the generated image. If not specified it will default to --output-size or
                    the size of any input images. Part of SDXL\'s micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952]"""
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-negative-aesthetic-scores', metavar="FLOAT", action='store', nargs='+', default=[], type=float,
            help="""One or more Stable Diffusion XL (sdxl) "negative-aesthetic-score" micro-conditioning parameters.
                    Part of SDXL's micro-conditioning as explained in section 2.2 of [https://huggingface.co/papers/2307.01952].
                    Can be used to simulate an aesthetic score of the generated image by influencing the negative text condition."""
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-negative-original-sizes', metavar="SIZE", action='store', nargs='+', default=[], type=_type_size,
            help="""One or more Stable Diffusion XL (sdxl) "negative-original-sizes" micro-conditioning parameters.
                    Negatively condition the generation process based on a specific image resolution. Part of SDXL's
                    micro-conditioning as explained in section 2.2 of [https://huggingface.co/papers/2307.01952].
                    For more information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208"""
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-negative-target-sizes', metavar="SIZE", action='store', nargs='+', default=[], type=_type_size,
            help="""One or more Stable Diffusion XL (sdxl) "negative-original-sizes" micro-conditioning parameters.
                    To negatively condition the generation process based on a target image resolution. It should be as same
                    as the "--sdxl-target-sizes" for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952]. For more information, refer to this issue thread:
                    https://github.com/huggingface/diffusers/issues/4208."""
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-negative-crops-coords-top-left', metavar="COORD", action='store', nargs='+', default=[],
            type=_type_image_coordinate,
            help="""One or more Stable Diffusion XL (sdxl) "negative-crops-coords-top-left" micro-conditioning
                    parameters in the format "0,0". Negatively condition the generation process based on a specific
                    crop coordinates. Part of SDXL's micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952]. For more information, refer
                    to this issue thread: https://github.com/huggingface/diffusers/issues/4208."""
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-refiner-aesthetic-scores', metavar="FLOAT", action='store', nargs='+', default=[], type=float,
            help="See: --sdxl-aesthetic-scores, applied to SDXL refiner pass."
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-refiner-crops-coords-top-left', metavar="COORD", action='store', nargs='+', default=[],
            type=_type_image_coordinate,
            help="See: --sdxl-crops-coords-top-left, applied to SDXL refiner pass."
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-refiner-original-sizes', metavar="SIZE", action='store', nargs='+', default=[], type=_type_size,
            help="See: --sdxl-refiner-original-sizes, applied to SDXL refiner pass."
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-refiner-target-sizes', metavar="SIZE", action='store', nargs='+', default=[], type=_type_size,
            help="See: --sdxl-refiner-target-sizes, applied to SDXL refiner pass."
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-refiner-negative-aesthetic-scores', metavar="FLOAT", action='store', nargs='+', default=[],
            type=float,
            help="See: --sdxl-negative-aesthetic-scores, applied to SDXL refiner pass."
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-refiner-negative-original-sizes', metavar="SIZE", action='store', nargs='+', default=[],
            type=_type_size,
            help="See: --sdxl-negative-original-sizes, applied to SDXL refiner pass."
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-refiner-negative-target-sizes', metavar="SIZE", action='store', nargs='+', default=[],
            type=_type_size,
            help="See: --sdxl-negative-target-sizes, applied to SDXL refiner pass."
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-refiner-negative-crops-coords-top-left', metavar="COORD", action='store', nargs='+', default=[],
            type=_type_image_coordinate,
            help="See: --sdxl-negative-crops-coords-top-left, applied to SDXL refiner pass."
        )
    )

    actions.append(
        parser.add_argument(
            '-hnf', '--sdxl-high-noise-fractions', action='store', nargs='+', default=None, metavar="FLOAT",
            type=_type_sdxl_high_noise_fractions,
            help="""One or more high-noise-fraction values for Stable Diffusion XL (sdxl),
                    this fraction of inference steps will be processed by the base model, while the rest
                    will be processed by the refiner model. Multiple values to this argument will result in
                    additional generation steps for each value. In certain situations when collaborative denoising
                    is not supported, such as when using --control-nets and inpainting with SDXL, the inverse
                    proportion of this value IE: (1.0 - high-noise-fraction) becomes the --image-seed-strengths
                    input to the SDXL refiner in plain img2img mode. Edit mode may be forced with the 
                    option --sdxl-refiner-edit (default: [0.8])"""
        )
    )

    actions.append(
        parser.add_argument(
            '-rgr', '--sdxl-refiner-guidance-rescales', action='store', nargs='+', default=None, metavar="FLOAT",
            type=_type_guidance_scale,
            help="""One or more guidance rescale values for the SDXL refiner when in use.
                    Override the guidance rescale value used by the SDXL refiner,
                    which defaults to the value taken from --guidance-rescales."""
        )
    )

    actions.append(
        parser.add_argument(
            '-sc', '--safety-checker', action='store_true', default=False,
            help="""Enable safety checker loading, this is off by default.
                    When turned on images with NSFW content detected may result in solid black output.
                    Some pretrained models have no safety checker model present, in that case this
                    option has no effect."""
        )
    )

    actions.append(
        parser.add_argument(
            '-d', '--device', action='store', type=_type_device,
            default=_torchutil.default_device(),
            help="""cuda / cpu, or other device supported by torch.
             
            For example mps on MacOS, and xpu for intel GPUs. 
            
            default: cuda [prioritize when available] then xpu. And only mps on MacOS. 
            
            Use: cuda:0, cuda:1, cuda:2, etc. to specify a specific cuda supporting GPU.
            
            Device indices are also supported for xpu, but not for mps.
            """
        )
    )

    actions.append(
        parser.add_argument(
            '-t', '--dtype', action='store', default='auto', type=_type_dtype,
            help=f'Model precision: {_SUPPORTED_DATA_TYPES_PRETTY}. (default: auto)'
        )
    )

    actions.append(
        parser.add_argument(
            '-s', '--output-size', action='store', default=None, type=_type_output_size, metavar="SIZE",
            help="""Image output size, for txt2img generation this is the exact output size.
                    The dimensions specified for this value must be aligned by 8 or you will receive an error message.
                    If an --image-seeds URI is used its Seed, Mask, and/or Control component image sources will be
                    resized to this dimension with aspect ratio maintained before being used for generation by default,
                    except in the case of Stable Cascade where the images are used as a style prompt (not a noised seed),
                    and can be of varying dimensions.
                    
                    If --no-aspect is not specified, width will be fixed and a new height
                    (aligned by 8) will be calculated for the input images. In most cases resizing the image inputs
                    will result in an image output of an equal size to the inputs, except for upscalers and Deep Floyd
                    --model-type values (if*).
                    
                    If only one integer value is provided, that is the value for both dimensions.
                    X/Y dimension values should be separated by "x". 
                    
                    This value defaults to 512x512 for Stable Diffusion when no --image-seeds are
                    specified (IE txt2img mode), 1024x1024 for Stable Cascade and Stable Diffusion 3/XL or
                    Flux model types, and 64x64 for --model-type if (Deep Floyd stage 1).
                    
                    Deep Floyd stage 1 images passed to superscaler models (--model-type ifs*) 
                    that are specified  with the 'floyd' keyword argument in an --image-seeds definition are
                    never resized or processed in any way."""
        )
    )

    actions.append(
        parser.add_argument(
            '-na', '--no-aspect', action='store_true',
            help="""This option disables aspect correct resizing of images provided to --image-seeds globally.
                    Seed, Mask, and Control guidance images will be resized to the closest dimension specified by --output-size
                    that is aligned by 8 pixels with no consideration of the source aspect ratio. This can be
                    overriden at the --image-seeds level with the image seed keyword argument 'aspect=true/false'."""
        )
    )

    actions.append(
        parser.add_argument(
            '-o', '--output-path', action='store', default='output', metavar="PATH",
            help="""Output path for generated images and files.
                    This directory will be created if it does not exist. 
                    
                    NOWRAP!
                    (default: ./output)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-op', '--output-prefix', action='store', default=None, type=str, metavar="PREFIX",
            help="""Name prefix for generated images and files.
                    This prefix will be added to the beginning of every generated file,
                    followed by an underscore."""
        )
    )

    actions.append(
        parser.add_argument(
            '-ox', '--output-overwrite', action='store_true', default=False,
            help="""Enable overwrites of files in the output directory that already exists.
                    The default behavior is not to do this, and instead append a filename suffix:
                    "_duplicate_(number)" when it is detected that the generated file name already exists."""
        )
    )

    actions.append(
        parser.add_argument(
            '-oc', '--output-configs', action='store_true', default=False,
            help="""Write a configuration text file for every output image or animation.
                    The text file can be used reproduce that particular output image or animation by piping
                    it to dgenerate STDIN or by using the --file option, for example "dgenerate < config.dgen" 
                    or "dgenerate --file config.dgen". These files will be written to --output-path and are
                    affected by --output-prefix and --output-overwrite as well. The files will be named
                    after their corresponding image or animation file. Configuration files produced for
                    animation frame images will utilize --frame-start and --frame-end to specify the
                    frame number."""
        )
    )

    output_metadata_group = parser.add_mutually_exclusive_group()

    actions.append(
        output_metadata_group.add_argument(
            '-om', '--output-metadata', action='store_true', default=False,
            help="""Write the information produced by --output-configs to the image metadata of each image.
                    Metadata will not be written to animated files. For PNGs, the data is written to a
                    PNG metadata property named "DgenerateConfig" and can be read using ImageMagick like so: 
                    "magick identify -format "%%[Property:DgenerateConfig] generated_file.png". For JPEGs,
                    the data is written to the EXIF UserComment on the image. Only PNGs and JPEGs are
                    supported for metadata writing, see: --image-format"""
        )
    )

    actions.append(
        output_metadata_group.add_argument(
            '-oam', '--output-auto1111-metadata', action='store_true', default=False,
            help="""Write Automatic1111 compatible metadata to the image metadata of each image,
                    this includes hashes for single file model checkpoints. Metadata will not be written 
                    to animated files. For PNGs, the data is written to a PNG metadata property named 
                    "parameters". For JPEGs, the data is written to the EXIF UserComment on the image. 
                    Only PNGs and JPEGs are supported for metadata writing, see: --image-format"""
        )
    )

    actions.append(
        parser.add_argument(
            '-pw', '--prompt-weighter', metavar='PROMPT_WEIGHTER_URI', dest='prompt_weighter_uri', action='store',
            default=None, type=_type_prompt_weighter,
            help="""Specify a prompt weighter implementation by URI, for example: 
                 
                 NOWRAP!
                 --prompt-weighter compel, or --prompt-weighter sd-embed. 
                  
                 By default, no prompt weighting syntax is enabled, 
                 meaning that you cannot adjust token weights as you may be able to do in software such as 
                 ComfyUI, Automatic1111, CivitAI etc. And in some cases the length of your prompt is limited. 
                 Prompt weighters support these special token weighting syntaxes and long prompts, 
                 currently there are two implementations "compel" and "sd-embed". See: --prompt-weighter-help 
                 for a list of implementation names. You may also use --prompt-weighter-help "name" to 
                 see comprehensive documentation for a specific prompt weighter implementation."""
        )
    )

    actions.append(
        parser.add_argument(
            '-pw2', '--second-model-prompt-weighter',
            metavar='PROMPT_WEIGHTER_URI',
            dest='second_model_prompt_weighter',
            action='store',
            default=None, type=_type_prompt_weighter,
            help='--prompt-weighter URI value that that applies to to --sdxl-refiner or --s-cascade-decoder.'
        )
    )

    actions.append(
        parser.add_argument(
            '--prompt-weighter-help', metavar='PROMPT_WEIGHTER_NAMES', dest=None, nargs='*',
            help="""Use this option alone (or with --plugin-modules) and no model specification
                 in order to list available prompt weighter names. Specifying one or more
                 prompt weighter names after this option will cause usage documentation for the specified
                 prompt weighters to be printed. When used with --plugin-modules, prompt weighters
                 implemented by the specified plugins will also be listed."""
        )
    )

    actions.append(
        parser.add_argument(
            '-lp',
            '--latents-processors', nargs='+', action='store', default=None,
            metavar="LATENTS_PROCESSOR_URI", dest='latents_processors',
            type=_type_latents_processor,
            help="""Specify one or more latents processor URIs for processing raw input latents before pipeline execution.
                    These processors are applied to latents provided through --image-seeds when using latents syntax
                    such as "latents: file.pt", "img2img.png;latents=file.pt", or directly "file.pt" (raw latents 
                    used as noise initialization). The processors are applied in sequence before the latents 
                    are passed to the diffusion pipeline.
                    
                    You may specify multiple processor URIs and they will be chained together sequentially.
                    
                    If you have multiple latents specified for batching, for example
                    
                    NOWRAP!
                    (--image-seeds "latents: latents-1.pt, latents-2.pt"), 
                    
                    you may use the delimiter "+" to separate
                    latents processor chains, so that a certain chain affects a certain latents input, 
                    the plus symbol may also be used to represent a null processor.
                    
                    For example: 
                    
                    NOWRAP!
                    (--latents-processors affect-1 + affect-2)
                    
                    NOWRAP!
                    (--latents-processors + affect-2)
                    
                    NOWRAP!
                    (--latents-processors affect-1 +)
                    
                    See: --latents-processor-help for a list of available implementations."""
        )
    )

    actions.append(
        parser.add_argument(
            '-ilp',
            '--img2img-latents-processors', nargs='+', action='store', default=None,
            metavar="LATENTS_PROCESSOR_URI", dest='img2img_latents_processors',
            type=_type_latents_processor,
            help="""Specify one or more latents processor URIs for processing img2img latents before pipeline execution.
                    These processors are applied to latent tensors provided through the --image-seeds argument when 
                    doing img2img with tensor inputs. The processors are applied in sequence and may occur 
                    before VAE decoding (for models that decode img2img latents) or before direct pipeline usage.
                    
                    You may specify multiple processor URIs and they will be chained together sequentially.
                    
                    If you have multiple img2img latents specified for batching, for example
                    
                    NOWRAP!
                    (--image-seeds "images: latents-1.pt, latents-2.pt"), 
                    
                    you may use the delimiter "+" to separate
                    latents processor chains, so that a certain chain affects a certain latents input, 
                    the plus symbol may also be used to represent a null processor.
                    
                    For example: 
                    
                    NOWRAP!
                    (--img2img-latents-processors affect-1 + affect-2)
                    
                    NOWRAP!
                    (--img2img-latents-processors + affect-2)
                    
                    NOWRAP!
                    (--img2img-latents-processors affect-1 +)
                    
                    See: --latents-processor-help for a list of available implementations."""
        )
    )

    actions.append(
        parser.add_argument(
            '-lpp',
            '--latents-post-processors', nargs='+', action='store', default=None,
            metavar="LATENTS_PROCESSOR_URI", dest='latents_post_processors',
            type=_type_latents_processor,
            help="""Specify one or more latents processor URIs for processing output latents when outputting to latents.
                    These processors are applied to latents when --image-format is set to a tensor format (pt, pth, safetensors). 
                    The processors are applied in sequence after the diffusion pipeline generates the latents 
                    but before they are returned in the result.
                    
                    You may specify multiple processor URIs and they will be chained together sequentially.
                    
                    See: --latents-processor-help for a list of available implementations."""
        )
    )

    actions.append(
        parser.add_argument(
            '--latents-processor-help', metavar='LATENTS_PROCESSOR_NAMES', dest=None, nargs='*',
            help="""Use this option alone (or with --plugin-modules) and no model specification
                    in order to list available latents processor names. Specifying one or more
                    latents processor names after this option will cause usage documentation for the specified
                    latents processors to be printed. When used with --plugin-modules, latents processors
                    implemented by the specified plugins will also be listed."""
        )
    )

    actions.append(
        parser.add_argument(
            '-pu',
            '--prompt-upscaler',
            '--prompt-upscalers',
            metavar='PROMPT_UPSCALER_URI', dest='prompt_upscaler_uri', action='store', nargs='+',
            default=None, type=_type_prompt_upscaler,
            help="""Specify a prompt upscaler implementation by URI, for example: --prompt-weighter dynamicprompts.
                    Prompt upscaler plugins can perform pure text processing and expansion on incoming prompt text, 
                    possibly resulting in more generation steps (variations) if the prompt upscaler returns multiple 
                    prompts per input prompt.
                    
                    NOWRAP!
                    For example: --prompt-upscaler "dynamicprompts;scale=1.5"
                    
                    You may specify multiple upscaler URIs and they will be chained together sequentially.
                    """
        )
    )

    actions.append(
        parser.add_argument(
            '-pu2',
            '--second-model-prompt-upscaler',
            '--second-model-prompt-upscalers',
            metavar='PROMPT_UPSCALER_URI', dest='second_model_prompt_upscaler_uri', action='store', nargs='+',
            default=None, type=_type_prompt_upscaler,
            help='Specify a --prompt-upscaler URI that will affect --second-model-prompts only, by default '
                 'the prompt upscaler specified by --prompt-upscaler will be used.'
        )
    )

    actions.append(
        parser.add_argument(
            '--second-model-second-prompt-upscaler',
            '--second-model-second-prompt-upscalers',
            metavar='PROMPT_UPSCALER_URI', dest='second_model_second_prompt_upscaler_uri', action='store', nargs='+',
            default=None, type=_type_prompt_upscaler,
            help='Specify a --prompt-upscaler URI that will affect --second-model-second-prompts only, by default '
                 'the prompt upscaler specified by --prompt-upscaler will be used.'
        )
    )

    actions.append(
        parser.add_argument(
            '--second-prompt-upscaler',
            '--second-prompt-upscalers',
            metavar='PROMPT_UPSCALER_URI', dest='second_prompt_upscaler_uri', action='store', nargs='+',
            default=None, type=_type_prompt_upscaler,
            help='Specify a --prompt-upscaler URI that will affect --second-prompts only, by default '
                 'the prompt upscaler specified by --prompt-upscaler will be used.'
        )
    )

    actions.append(
        parser.add_argument(
            '--third-prompt-upscaler',
            '--third-prompt-upscalers',
            metavar='PROMPT_UPSCALER_URI', dest='third_prompt_upscaler_uri', action='store', nargs='+',
            default=None, type=_type_prompt_upscaler,
            help='Specify a --prompt-upscaler URI that will affect --third-prompts only, by default '
                 'the prompt upscaler specified by --prompt-upscaler will be used.'
        )
    )

    actions.append(
        parser.add_argument(
            '--prompt-upscaler-help', metavar='PROMPT_UPSCALER_NAMES', dest=None, nargs='*',
            help="""Use this option alone (or with --plugin-modules) and no model specification
                 in order to list available prompt upscaler names. Specifying one or more
                 prompt upscaler names after this option will cause usage documentation for the specified
                 prompt upscalers to be printed. When used with --plugin-modules, prompt upscalers
                 implemented by the specified plugins will also be listed."""
        )
    )

    actions.append(
        parser.add_argument(
            '-p', '--prompts', nargs='+', action='store', metavar="PROMPT", default=[_prompt.Prompt()],
            type=_type_main_prompts,
            help="""One or more prompts to try, an image group is generated for each prompt,
                    prompt data is split by ; (semi-colon). The first value is the positive
                    text influence, things you want to see. The Second value is negative
                    influence IE. things you don't want to see.
                    
                    NOWRAP!
                    Example: --prompts "photo of a horse in a field; artwork, painting, rain".
                    
                    (default: [(empty string)])"""
        )
    )

    actions.append(
        parser.add_argument(
            '--second-prompts', nargs='+', action='store', metavar="PROMPT", default=None,
            type=_type_secondary_prompts,
            help="""One or more secondary prompts to try using the sdxl (SDXL), sd3 
                    (Stable Diffusion 3) or flux (Flux) secondary text encoder. By default 
                    the model is passed the primary prompt for this value, this option allows you 
                    to choose a different prompt. The negative prompt component can be specified 
                    with the same syntax as --prompts"""
        )
    )

    actions.append(
        parser.add_argument(
            '--third-prompts', nargs='+', action='store', metavar="PROMPT", default=None,
            type=_type_secondary_prompts,
            help="""One or more tertiary prompts to try using the sd3 (Stable Diffusion 3) 
                    tertiary (T5) text encoder, Flux does not support this argument. By default the 
                    model is passed the primary prompt for this value, this option allows you to choose 
                    a different prompt. The negative prompt component can be specified with the 
                    same syntax as --prompts"""
        )
    )

    actions.append(
        parser.add_argument(
            '--second-model-prompts', nargs='+', action='store', metavar="PROMPT", default=None,
            type=_type_secondary_prompts,
            help="""One or more prompts to try with the SDXL Refiner or Stable Cascade decoder model,
                    by default the decoder model gets the primary prompt, this argument
                    overrides that with a prompt of your choosing. The negative prompt
                    component can be specified with the same syntax as --prompts"""
        )
    )

    actions.append(
        parser.add_argument(
            '--second-model-second-prompts', nargs='+', action='store', metavar="PROMPT", default=None,
            type=_type_secondary_prompts,
            help="""One or more prompts to try with the SDXL refiner models secondary
                    text encoder (Stable Cascade Decoder is not supported), by default the 
                    SDXL refiner model gets the primary prompt passed to its second text encoder, 
                    this argument overrides that with a prompt of your choosing. The negative prompt 
                    component can be specified with the same syntax as --prompts
                    """
        )
    )

    actions.append(
        parser.add_argument(
            '--max-sequence-length', action='store', metavar='INTEGER', default=None, type=_max_sequence_length,
            help="""The maximum amount of prompt tokens that the T5EncoderModel
                    (third text encoder) of Stable Diffusion 3 or Flux can handle. This should be
                    an integer value between 1 and 512 inclusive. The higher the value
                    the more resources and time are required for processing. 
                    (default: 256 for SD3, 512 for Flux)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-cs', '--clip-skips', nargs='+', action='store', metavar="INTEGER", default=None, type=_type_clip_skip,
            help="""One or more clip skip values to try. Clip skip is the number of layers to be skipped from CLIP
                    while computing the prompt embeddings, it must be a value greater than or equal to zero. A value of 1 means
                    that the output of the pre-final layer will be used for computing the prompt embeddings. This is only
                    supported for --model-type values "sd", "sdxl", and "sd3"."""
        )
    )

    seed_options = parser.add_mutually_exclusive_group()

    actions.append(
        seed_options.add_argument(
            '-se', '--seeds', nargs='+', action='store', metavar="SEED", type=_type_seeds,
            help="""One or more seeds to try, define fixed seeds to achieve deterministic output.
                    This argument may not be used when --gse/--gen-seeds is used.
                    (default: [randint(0, 99999999999999)])"""
        )
    )

    actions.append(
        parser.add_argument(
            '-sei', '--seeds-to-images', action='store_true',
            help="""When this option is enabled, each provided --seeds value or value generated by --gen-seeds
                    is used for the corresponding image input given by --image-seeds. If the amount of --seeds given
                    is not identical to that of the amount of --image-seeds given, the seed is determined as:
                    seed = seeds[image_seed_index %% len(seeds)], IE: it wraps around."""
        )
    )

    actions.append(
        seed_options.add_argument(
            '-gse', '--gen-seeds', action='store', type=_type_gen_seeds, metavar="COUNT", dest='seeds',
            help="""Auto generate N random seeds to try. This argument may not
                    be used when -se/--seeds is used."""
        )
    )

    actions.append(
        parser.add_argument(
            '-af', '--animation-format', action='store', default='mp4', type=_type_animation_format, metavar="FORMAT",
            help=f"""Output format when generating an animation from an input video / gif / webp etc.
                    Value must be one of: {_SUPPORTED_ANIMATION_OUTPUT_FORMATS_PRETTY}. You may also specify "frames"
                    to indicate that only frames should be output and no coalesced animation file should be rendered.
                    (default: mp4)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-if', '--image-format', action='store', default='png', type=_type_image_format, metavar="FORMAT",
            help=f"""Output format when writing static images or tensors. For image formats, any selection other than "png", "jpg", or "jpeg"
                    is not compatible with --output-metadata. For tensor formats (pt, pth, safetensors), raw latent tensors will be saved
                    instead of decoded images. Value must be one of: {_SUPPORTED_ALL_OUTPUT_FORMATS_PRETTY}. (default: png)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-nf', '--no-frames', action='store_true',
            help=f"""Do not write frame images individually when rendering an animation,
                    only write the animation file. This option is incompatible with --animation-format frames."""
        )
    )

    actions.append(
        parser.add_argument(
            '-fs', '--frame-start', default=0, type=_type_frame_start, metavar="FRAME_NUMBER",
            help="""Starting frame slice point for animated files (zero-indexed), the specified frame
                    will be included. (default: 0)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-fe', '--frame-end', default=None, type=_type_frame_end, metavar="FRAME_NUMBER",
            help="""Ending frame slice point for animated files (zero-indexed), the specified frame
                    will be included."""
        )
    )

    actions.append(
        parser.add_argument(
            '-is', '--image-seeds', action='store', nargs='+', default=[], metavar="SEED",
            help="""One or more image seed URIs to process, these may consist of URLs or file paths.
                    Videos / GIFs / WEBP files will result in frames being rendered as well as an animated
                    output file being generated if more than one frame is available in the input file.
                    Inpainting for static images can be achieved by specifying a black and white mask image in each
                    image seed string using a semicolon as the separating character, like so: 
                    
                    NOWRAP!
                    "my-seed-image.png;my-image-mask.png", white areas of the mask indicate where
                    
                    generated content is to be placed in your seed image.
                    
                    Output dimensions specific to the image seed can be specified by placing the dimension
                    at the end of the string following a semicolon like so: 
                    
                    NOWRAP!
                    "my-seed-image.png;512x512" or "my-seed-image.png;my-image-mask.png;512x512". 
                    
                    When using --control-nets, a singular
                    image specification is interpreted as the control guidance image, and you can specify
                    multiple control image sources by separating them with commas in the case where multiple
                    ControlNets are specified, IE: 
                    
                    NOWRAP!
                    (--image-seeds "control-image1.png, control-image2.png") OR (--image-seeds "seed.png;control=control-image1.png, control-image2.png").
                     
                    Using --control-nets with img2img or inpainting can be accomplished with the syntax: 
                    
                    NOWRAP!
                    "my-seed-image.png;mask=my-image-mask.png;control=my-control-image.png;resize=512x512".
                    
                    The "mask" and "resize" arguments are optional when using --control-nets. Videos, GIFs,
                    and WEBP are also supported as inputs when using --control-nets, even for the "control"
                    argument.
                    
                    --image-seeds is capable of reading from multiple animated files at once or any
                    combination of animated files and images, the animated file with the least amount of frames
                    dictates how many frames are generated and static images are duplicated over the total amount
                    of frames. The keyword argument "aspect" can be used to determine resizing behavior when
                    the global argument --output-size or the local keyword argument "resize" is specified,
                    it is a boolean argument indicating whether aspect ratio of the input image should be
                    respected or ignored. 
                    
                    The keyword argument "floyd" can be used to specify images from
                    a previous deep floyd stage when using --model-type ifs*. When keyword arguments
                    are present, all applicable images such as "mask", "control", etc. must also be defined
                    with keyword arguments instead of with the short syntax.
                    
                    In place of static images, you may pass a latents file generated by dgenerate
                    containing the raw un-decoded latents from a previous generation, latents can
                    be generated with --image-format pt, pth, or safetensors. Latents may be passed 
                    for img2img input only. Latents will first be decoded back into pixel space 
                    (into a normal image) by the receiving models VAE. Except in the case of
                    --model-type upscaler-x2, which can handle the denoised latents directly.
                    
                    Latent img2img input is not supported for --model-type s-cascade as Stable Cascade
                    cannot perform traditional img2img, and will result in an error if attempted. Latent input 
                    is also not supported for ControlNet/T2I Adapter guidance images, or IP Adapter images, as 
                    these guidance models operate on images in pixel space.
                    """
        )
    )

    image_seed_noise_opts = parser.add_mutually_exclusive_group()

    actions.append(
        parser.add_argument(
            '-sip', '--seed-image-processors',
            type=_type_image_processor,
            action='store', nargs='+', default=None, metavar="PROCESSOR_URI",
            help="""Specify one or more image processor actions to perform on the primary
                    img2img image(s) specified by --image-seeds.
                    
                    When specifying latents as img2img input, these processors will run 
                    on the image after the latents are decoded by the VAE.
                    
                    NOWRAP!
                    For example: --seed-image-processors "flip" "mirror" "grayscale".
                    
                    To obtain more information about what image processors are available and how to use them,
                    see: --image-processor-help.
                    
                    If you have multiple images specified for batching, for example
                    
                    NOWRAP!
                    (--image-seeds "images: img2img-1.png, img2img-2.png"), 
                    
                    you may use the delimiter "+" to separate
                    image processor chains, so that a certain chain affects a certain seed image, the plus symbol
                    may also be used to represent a null processor.
                    
                    For example: 
                    
                    NOWRAP!
                    (--seed-image-processors affect-img-1 + affect-img-2)
                    
                    NOWRAP!
                    (--seed-image-processors + affect-img-2)
                    
                    NOWRAP!
                    (--seed-image-processors affect-img-1 +)
                    
                    The amount of processors / processor chains must not exceed the amount of input images,
                    or you will receive a syntax error message. To obtain more information about what image
                    processors are available and how to use them, see: --image-processor-help."""
        )
    )

    actions.append(
        parser.add_argument(
            '-mip', '--mask-image-processors',
            type=_type_image_processor,
            action='store', nargs='+', default=None, metavar="PROCESSOR_URI",
            help="""Specify one or more image processor actions to perform on the inpaint mask
                    image(s) specified by --image-seeds.
                    
                    NOWRAP!
                    For example: --mask-image-processors "invert".
                    
                    To obtain more information about what image processors are available and how to use them,
                    see: --image-processor-help.
                    
                    If you have multiple masks specified for batching, for example
                    --image-seeds ("images: img2img-1.png, img2img-2.png; mask-1.png, mask-2.png"), you may use
                    the delimiter "+" to separate image processor chains, so that a certain chain affects a certain
                    mask image, the plus symbol may also be used to represent a null processor.
                    
                    For example: 
                    (--mask-image-processors affect-mask-1 + affect-mask-2), or (--mask-image-processors + affect-mask-2),
                    or (--mask-image-processors affect-mask-1 +).
                    
                    The amount of processors / processor chains must not
                    exceed the amount of input mask images, or you will receive a syntax error message. To obtain
                    more information about what image processors are available and how to use them,
                    see: --image-processor-help."""
        )
    )

    actions.append(
        parser.add_argument(
            '-cip', '--control-image-processors',
            type=_type_image_processor,
            action='store', nargs='+', default=None, metavar="PROCESSOR_URI",
            help="""Specify one or more image processor actions to perform on the control
                    image specified by --image-seeds, this option is meant to be used with --control-nets.
                    
                    NOWRAP!
                    Example: --control-image-processors "canny;lower=50;upper=100".
                    
                    The delimiter "+" can be used to specify a different processor group for each image when using
                    multiple control images with --control-nets.
                    
                    For example if you have 
                    
                    NOWRAP!
                    --image-seeds "img1.png, img2.png" 
                    
                    or 
                    
                    NOWRAP!
                    --image-seeds "...;control=img1.png, img2.png" 
                    
                    specified and multiple ControlNet models specified with --control-nets, you can specify processors for
                    those control images with the syntax: 
                    
                    NOWRAP!
                    (--control-image-processors "processes-img1" + "processes-img2").
                    
                    This syntax also supports chaining of processors, for example: 
                    
                    NOWRAP!
                    (--control-image-processors "first-process-img1" "second-process-img1" + "process-img2").
                     
                    The amount of specified processors must not exceed the amount of specified control images, or you
                    will receive a syntax error message.
                    
                    Images which do not have a processor defined for them will not be processed, and the plus character can
                    be used to indicate an image is not to be processed and instead skipped over when that image is a
                    leading element, for example 
                    
                    NOWRAP!
                    (--control-image-processors + "process-second") 
                    
                    would indicate that
                    the first control guidance image is not to be processed, only the second.
                    
                    To obtain more information about what image processors
                    are available and how to use them, see: --image-processor-help."""
        )
    )

    actions.append(
        parser.add_argument(
            '--image-processor-help', action='store', nargs='*', default=None, metavar="PROCESSOR_NAME", dest=None,
            help="""Use this option alone (or with --plugin-modules) and no model
                    specification in order to list available image processor names.
                    Specifying one or more image processor names after this option will cause usage
                    documentation for the specified image processors to be printed. When used with
                    --plugin-modules, image processors implemented by the specified plugins
                    will also be listed."""
        )
    )

    actions.append(
        parser.add_argument(
            '-pp', '--post-processors', action='store', nargs='+', default=None, metavar="PROCESSOR_URI",
            help="""Specify one or more image processor actions to perform on generated
                    output before it is saved. 
                    
                    NOWRAP!
                    For example: --post-processors "upcaler;model=4x_ESRGAN.pth".
                    
                    To obtain more information about what processors are available and how to use them,
                    see: --image-processor-help."""
        )
    )

    actions.append(
        image_seed_noise_opts.add_argument(
            '-iss', '--image-seed-strengths', action='store', nargs='+', default=None, metavar="FLOAT",
            type=_type_image_seed_strengths,
            help=f"""One or more image strength values to try when using --image-seeds for
                    img2img or inpaint mode. Closer to 0 means high usage of the seed image (less noise convolution),
                    1 effectively means no usage (high noise convolution). Low values will produce something closer
                    or more relevant to the input image, high values will give the AI more creative freedom. This
                    value must be greater than 0 and less than or equal to 1. (default: [0.8])"""
        )
    )

    actions.append(
        image_seed_noise_opts.add_argument(
            '-uns', '--upscaler-noise-levels', action='store', nargs='+', default=None, metavar="INTEGER",
            type=_type_upscaler_noise_levels,
            help=f"""One or more upscaler noise level values to try when using the super
                    resolution upscaler --model-type upscaler-x4 or ifs. Specifying
                    this option for --model-type upscaler-x2 will produce an error message.
                    The higher this value the more noise is added to the image before upscaling
                    (similar to --image-seed-strengths). (default: [20 for x4, 250 for
                    ifs/ifs-img2img, 0 for ifs inpainting mode])
                    """
        )
    )

    # Inpaint crop arguments
    actions.append(
        parser.add_argument(
            '-ic', '--inpaint-crop', action='store_true', default=False,
            dest='inpaint_crop',
            help="""Enable cropping to mask bounds for inpainting. When enabled, input images will be
                    automatically cropped to the bounds of their masks (plus any padding) before processing, 
                    then the generated result will be pasted back onto the original uncropped image. This 
                    allows inpainting at higher effective resolutions for better quality results.
                    
                    Cannot be used with image seed batching (--image-seeds with multiple images/masks in the definition).
                    
                    Each image/mask pair must be processed individually as different masks may have different
                    crop bounds. However, --batch-size > 1 is supported for generating multiple variations of
                    a single crop."""
        )
    )

    actions.append(
        parser.add_argument(
            '-icp', '--inpaint-crop-paddings', action='store', nargs='+', default=None, metavar="PADDING",
            type=_type_inpaint_crop_padding,
            dest='inpaint_crop_paddings',
            help="""One or more padding values to use around mask bounds for inpaint cropping. 
                    Automatically enables --inpaint-crop. Each value will be tried in turn (combinatorial).
                    
                    Example:
                    
                    32 (32px Uniform, all sides)
                    
                    10x20 (10px Horizontal, 20px Vertical)
                    
                    10x20x30x40 (10px Left, 20px Top, 30px Right, 40px Bottom)
                    
                    Note: Inpaint crop cannot be used with multiple input images. See --inpaint-crop for details.
                    
                    (default: [32])"""
        )
    )

    actions.append(
        parser.add_argument(
            '-icm', '--inpaint-crop-masked', action='store_true', default=False,
            dest='inpaint_crop_masked',
            help="""Use the mask when pasting the generated result back onto the original image for 
                    inpaint cropping. Automatically enables --inpaint-crop. This means only the masked 
                    areas will be replaced. Cannot be used together with --inpaint-crop-feathers.
                    
                    Note: Inpaint crop cannot be used with individual --image-seeds batching. See --inpaint-crop for details."""
        )
    )

    actions.append(
        parser.add_argument(
            '-icf', '--inpaint-crop-feathers', action='store', nargs='+', default=None, metavar="FEATHER",
            type=_type_inpaint_crop_feather,
            dest='inpaint_crop_feathers',
            help="""One or more feather values to use when pasting the generated result back onto the 
                    original image for inpaint cropping. Automatically enables --inpaint-crop. Each value 
                    will be tried in turn (combinatorial). Feathering creates smooth transitions from opaque 
                    to transparent. Cannot be used together with --inpaint-crop-masked.
                    
                    Note: Inpaint crop cannot be used with individual --image-seeds batching. See --inpaint-crop for details.
                    
                    (default: none - simple paste without feathering)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-gs', '--guidance-scales', action='store', nargs='+',
            default=[_pipelinewrapper.constants.DEFAULT_GUIDANCE_SCALE],
            metavar="FLOAT", type=_type_guidance_scale,
            help="""One or more guidance scale values to try. Guidance scale effects how much your
                    text prompt is considered. Low values draw more data from images unrelated
                    to text prompt. 
                    
                    NOWRAP!
                    (default: [5])"""
        )
    )

    actions.append(
        parser.add_argument(
            '-si', '--sigmas', action='store', nargs='+',
            default=None,
            metavar="CSV_FLOAT_OR_EXPRESSION", type=_type_sigmas,
            help="""One or more comma-separated lists (or singular values) of floating
                    point sigmas to try. This is supported when using a --scheduler
                    that supports setting sigmas. Sigma values control the noise schedule
                    in the diffusion process, allowing for fine-grained control over
                    how noise is added and removed during image generation.
                    
                    NOWRAP!
                    Example: --sigmas "1.0,0.8,0.6,0.4,0.2" 
                    
                    Or expressions: 
                    
                    NOWRAP!
                    "expr: sigmas * .95"
                    
                    sigmas from --scheduler are 
                    represented as a numpy array in an interpreted expression, numpy
                    is available through the namespace "np", this uses asteval.
                    
                    Or singular values: 
                    
                    NOWRAP!
                    --sigmas 0.4
                    
                    NOWRAP!
                    Expressions and CSV lists can be intermixed: --sigmas "1.0,..." "expr: sigmas * 0.95"
                    
                    Each provided value (each quoted string in the example above) will be tried in turn.
                    """
        )
    )

    actions.append(
        parser.add_argument(
            '-igs', '--image-guidance-scales', action='store', nargs='+', default=None, metavar="FLOAT",
            type=_type_image_guidance_scale,
            help="""One or more image guidance scale values to try. This can push the generated image towards the
                    initial image when using --model-type *-pix2pix models, it is unsupported for other model types.
                    Use in conjunction with --image-seeds, inpainting (masks) and --control-nets are not supported.
                    Image guidance scale is enabled by setting image-guidance-scale > 1. Higher image guidance scale
                    encourages generated images that are closely linked to the source image, usually at the expense of
                    lower image quality. Requires a value of at least 1. (default: [1.5])"""
        )
    )

    actions.append(
        parser.add_argument(
            '-gr', '--guidance-rescales', action='store', nargs='+', default=[], metavar="FLOAT",
            type=_type_guidance_scale,
            help="""One or more guidance rescale factors to try. Proposed by 
                    
                    NOWRAP!
                    [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf) 
                    
                    "guidance_scale" is defined as "φ" in equation 16. of 
                    
                    NOWRAP!
                    [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). 
                    
                    Guidance rescale factor should fix overexposure
                    when using zero terminal SNR. This is supported for basic text to image generation
                    when using --model-type "sd" but not inpainting, img2img, or --control-nets.
                    When using --model-type "sdxl" it is supported for basic generation, inpainting,
                    and img2img, unless --control-nets is specified in which case only inpainting is supported.
                    It is supported for --model-type "sdxl-pix2pix" but not --model-type "pix2pix".
                    
                    NOWRAP!
                    (default: [0.0])"""
        )
    )

    actions.append(
        parser.add_argument(
            '-ifs', '--inference-steps', action='store', nargs='+',
            default=[_pipelinewrapper.constants.DEFAULT_INFERENCE_STEPS],
            type=_type_inference_steps, metavar="INTEGER",
            help="""One or more inference steps values to try. The amount of inference (de-noising) steps
                    effects image clarity to a degree, higher values bring the image closer to what
                    the AI is targeting for the content of the image. Values between 30-40
                    produce good results, higher values may improve image quality and or
                    change image content. 
                    
                    NOWRAP!
                    (default: [30])"""
        )
    )

    actions.append(
        parser.add_argument(
            '-ifs2', '--second-model-inference-steps', action='store', nargs='+', default=None, metavar="INTEGER",
            type=_type_inference_steps,
            help=f"""One or more inference steps values for the SDXL refiner or Stable Cascade decoder
                     when in use. Override the number of inference steps used by the second model,
                     which defaults to the value taken from --inference-steps for SDXL and 
                     {_pipelinewrapper.constants.DEFAULT_S_CASCADE_DECODER_INFERENCE_STEPS} for Stable Cascade.
                     """
        )
    )

    actions.append(
        parser.add_argument(
            '-gs2', '--second-model-guidance-scales', action='store', nargs='+', default=None, metavar="FLOAT",
            type=_type_guidance_scale,
            help=f"""One or more inference steps values for the SDXL refiner or Stable Cascade decoder
                     when in use. Override the guidance scale value used by the second model,
                     which defaults to the value taken from --guidance-scales for SDXL and 
                     {_pipelinewrapper.constants.DEFAULT_S_CASCADE_DECODER_GUIDANCE_SCALE} for Stable Cascade.
                     """
        )
    )

    actions.append(
        parser.add_argument(
            '-sir', '--sdxl-refiner-sigmas', action='store', nargs='+',
            default=None,
            metavar="CSV_FLOAT_OR_EXPRESSION", type=_type_sigmas,
            help="""See: --sigmas, but for the SDXL Refiner."""
        )
    )

    actions.append(
        parser.add_argument(
            '-ds', '--denoising-start', action='store', default=None, metavar="FLOAT",
            dest='denoising_start', type=_type_denoising_fraction,
            help="""Fraction of total timesteps at which denoising should start (0.0 to 1.0). 
                    This allows you to skip the early noising steps and start denoising from 
                    a specific point in the noise schedule. Useful for cooperative denoising 
                    workflows where one model handles the initial denoising and another model 
                    refines the result.
                    
                    Scheduler Compatibility:
                    
                    For SD 1.5 models, only stateless schedulers are supported:
                    
                    NOWRAP!
                    * EulerDiscreteScheduler
                    * LMSDiscreteScheduler
                    * EDMEulerScheduler, 
                    * DPMSolverMultistepScheduler
                    * DDIMScheduler
                    * DDPMScheduler
                    * PNDMScheduler
                    
                    For SDXL models, all schedulers are supported via native denoising_start/denoising_end.
                    
                    For SD3/Flux models, FlowMatchEulerDiscreteScheduler is supported.
                    
                    NOWRAP!
                    Example: --denoising-start 0.8
                    
                    A value of 0.8 means denoising will start at 80 percent through the total timesteps, 
                    effectively skipping the first 20 percent of the normal denoising process."""
        )
    )

    actions.append(
        parser.add_argument(
            '-de', '--denoising-end', action='store', default=None, metavar="FLOAT",
            dest='denoising_end', type=_type_denoising_fraction,
            help="""Fraction of total timesteps at which denoising should end (0.0 to 1.0). 
                    This allows you to stop denoising early, leaving the output in a partially 
                    noisy state. Useful for generating noisy latents that can be saved with 
                    --image-format pt/pth/safetensors and passed to another model or generation 
                    stage using the "latents: ..." or "img2img.png;latents= ..." syntax of 
                    --image-seeds.
                    
                    Scheduler Compatibility:
                    
                    For SD 1.5 models, only stateless schedulers are supported:
                    
                    NOWRAP!
                    * EulerDiscreteScheduler
                    * LMSDiscreteScheduler
                    * EDMEulerScheduler, 
                    * DPMSolverMultistepScheduler
                    * DDIMScheduler
                    * DDPMScheduler
                    * PNDMScheduler
                    
                    For SDXL models, all schedulers are supported via native denoising_start/denoising_end.
                    
                    For SD3/Flux models, FlowMatchEulerDiscreteScheduler is supported.
                    
                    NOWRAP!
                    Example: --denoising-end 0.5
                    
                    A value of 0.5 means denoising will stop at 50 percent through the total timesteps, 
                    leaving the result partially noisy for further processing by another model."""
        )
    )

    return parser, actions


class DgenerateUsageError(Exception):
    """
    Raised by :py:func:`.parse_args` and :py:func:`.parse_known_args` on argument usage errors.
    """
    pass


class DgenerateArguments(_renderloopconfig.RenderLoopConfig):
    """
    Represents dgenerate's parsed command line arguments, can be used
    as a configuration object for :py:class:`dgenerate.renderloop.RenderLoop`.
    """

    plugin_module_paths: _types.Paths
    """
    Plugin module paths ``--plugin-modules``
    """

    verbose: bool = False
    """
    Enable debug output? ``-v/--verbose``
    """

    global_config: _types.OptionalPath = None
    """
    global config file path.
    """

    def __init__(self):
        super().__init__()
        self.plugin_module_paths = []


_, _actions = _create_parser()

_attr_name_to_option = {a.dest: a.option_strings[-1] if a.option_strings else a.dest for a in _actions}

_all_valid_options = set()

for action in _actions:
    if action.option_strings:
        _all_valid_options.update(action.option_strings)


def is_valid_option(option: str):
    """
    Check if an option string is a valid option name in the parser.

    :param option: The option name, short or long opt.
    :return: ``True`` or ``False``
    """
    return option in _all_valid_options


def config_attribute_name_to_option(name):
    """
    Convert an attribute name of :py:class:`.DgenerateArguments` into its command line option name.

    :param name: the attribute name
    :return: the command line argument name as a string
    """
    return _attr_name_to_option[name]


def get_raw_help_text(option: str) -> str:
    """
    Get the raw help text for a given command line option.

    This text will not be formatted in any way, and may be
    indented as is defined in source code.

    You should utilize :py:func:`inspect.cleandoc` and
    :py:func:`dgenerate.textprocessing.wrap_paragraphs` to
    format the text if displaying it to the user is intended.

    :param option: The command line option name, short or long opt.
    :return: The help text for the option.
    :raises ValueError: If the option is not valid.
    """
    if not is_valid_option(option):
        raise ValueError(f"Unknown option: {option}")

    for a in _actions:
        if option in a.option_strings:
            return a.help

    raise ValueError(f"Option {option} not found in actions.")


def _parse_args(args=None, print_usage=True, overrides: dict[str, typing.Any] | None = None) -> DgenerateArguments:
    parser = _create_parser(prints_usage=print_usage)[0]
    args = parser.parse_args(args, namespace=DgenerateArguments())
    if overrides:
        args.set_from(overrides, missing_value_throws=False)
    args.check(config_attribute_name_to_option)
    return args


def _check_unknown_args(args: typing.Sequence[str], log_error: bool):
    # this treats the model argument as optional

    parser = _create_parser(add_model=True, add_help=False, prints_usage=False)[0]
    try:
        # try first to parse without adding a fake model argument
        parser.parse_args(args)
    except argparse.ArgumentTypeError as e:
        if log_error:
            _messages.log(parser.format_usage().rstrip())
            _messages.error(str(e).strip())

        raise DgenerateUsageError(str(e)) from e
    except argparse.ArgumentError:
        try:
            # try again one more time with a fake model argument
            parser.parse_args(['fake_model'] + list(args))
        except (argparse.ArgumentTypeError,
                argparse.ArgumentError) as e:

            # truly erroneous command line
            if log_error:
                _messages.log(parser.format_usage().rstrip())
                _messages.error(str(e).strip())

            raise DgenerateUsageError(str(e)) from e


def parse_templates_help(
        args: collections.abc.Sequence[str] | None = None,
        throw_unknown: bool = False,
        log_error: bool = False) -> tuple[list[str] | None, list[str]]:
    """
    Retrieve the ``--templates-help`` argument value

    :param args: command line arguments

    :param throw_unknown: Raise :py:class:`DgenerateUsageError` if any other
     specified argument is not a valid dgenerate argument? This treats the
     primary model argument as optional, and only goes into effect if the
     specific argument is detected.

    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?

    :return: (values | ``None``, unknown_args_list)
    """
    parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
    parser.add_argument('--templates-help', nargs='*', default=None)
    parsed, unknown = parser.parse_known_args(args)

    if parsed.templates_help is not None and throw_unknown:
        _check_unknown_args(unknown, log_error)

    return parsed.templates_help, unknown


def parse_directives_help(
        args: collections.abc.Sequence[str] | None = None,
        throw_unknown: bool = False,
        log_error: bool = False) -> tuple[list[str] | None, list[str]]:
    """
    Retrieve the ``--directives-help`` argument value

    :param args: command line arguments

    :param throw_unknown: Raise :py:class:`DgenerateUsageError` if any other
     specified argument is not a valid dgenerate argument? This treats the
     primary model argument as optional, and only goes into effect if the
     specific argument is detected.

    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?

    :return: (values | ``None``, unknown_args_list)
    """
    parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
    parser.add_argument('--directives-help', nargs='*', default=None)
    parsed, unknown = parser.parse_known_args(args)

    if parsed.directives_help is not None and throw_unknown:
        _check_unknown_args(unknown, log_error)

    return parsed.directives_help, unknown


def parse_functions_help(
        args: collections.abc.Sequence[str] | None = None,
        throw_unknown: bool = False,
        log_error: bool = False) -> tuple[list[str] | None, list[str]]:
    """
    Retrieve the ``--functions-help`` argument value

    :param args: command line arguments

    :param throw_unknown: Raise :py:class:`DgenerateUsageError` if any other
     specified argument is not a valid dgenerate argument? This treats the
     primary model argument as optional, and only goes into effect if the specific
     argument is detected.

    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?

    :return: (values | ``None``, unknown_args_list)
    """
    parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
    parser.add_argument('--functions-help', nargs='*', default=None)
    parsed, unknown = parser.parse_known_args(args)

    if parsed.functions_help is not None and throw_unknown:
        _check_unknown_args(unknown, log_error)

    return parsed.functions_help, unknown


def parse_plugin_modules(
        args: collections.abc.Sequence[str] | None = None,
        throw_unknown: bool = False,
        log_error: bool = False) -> tuple[list[str] | None, list[str]]:
    """
    Retrieve the ``--plugin-modules`` argument value

    :param args: command line arguments

    :param throw_unknown: Raise :py:class:`DgenerateUsageError` if any other
     specified argument is not a valid dgenerate argument? This treats the
     primary model argument as optional, and only goes into effect if the
     specific argument is detected.

    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?

    :raise DgenerateUsageError: If no argument values were provided.

    :return: (values | ``None``, unknown_args_list)
    """

    try:
        parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
        parser.add_argument('--plugin-modules', action='store', default=None, nargs="+")
        parsed, unknown = parser.parse_known_args(args)
    except argparse.ArgumentError as e:
        raise DgenerateUsageError(e) from e

    if parsed.plugin_modules is not None and throw_unknown:
        _check_unknown_args(unknown, log_error)

    return parsed.plugin_modules, unknown


def parse_quantizer_help(
        args: collections.abc.Sequence[str] | None = None,
        throw_unknown: bool = False,
        log_error: bool = False) -> tuple[list[str] | None, list[str]]:
    """
    Retrieve the ``--quantizer-help`` argument value

    :param args: command line arguments

    :param throw_unknown: Raise :py:class:`DgenerateUsageError` if any other
     specified argument is not a valid dgenerate argument? This treats the
     primary model argument as optional, and only goes into effect if the specific
     argument is detected.

    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?

    :return: (values | ``None``, unknown_args_list)
    """

    parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
    parser.add_argument('--quantizer-help', action='store', nargs='*', default=None)
    parsed, unknown = parser.parse_known_args(args)

    if parsed.quantizer_help is not None and throw_unknown:
        _check_unknown_args(unknown, log_error)

    return parsed.quantizer_help, unknown


def parse_image_processor_help(
        args: collections.abc.Sequence[str] | None = None,
        throw_unknown: bool = False,
        log_error: bool = False) -> tuple[list[str] | None, list[str]]:
    """
    Retrieve the ``--image-processor-help`` argument value

    :param args: command line arguments

    :param throw_unknown: Raise :py:class:`DgenerateUsageError` if any other
     specified argument is not a valid dgenerate argument? This treats the
     primary model argument as optional, and only goes into effect if the specific
     argument is detected.

    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?

    :return: (values | ``None``, unknown_args_list)
    """

    parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
    parser.add_argument('--image-processor-help', action='store', nargs='*', default=None)
    parsed, unknown = parser.parse_known_args(args)

    if parsed.image_processor_help is not None and throw_unknown:
        _check_unknown_args(unknown, log_error)

    return parsed.image_processor_help, unknown


def parse_prompt_weighter_help(
        args: collections.abc.Sequence[str] | None = None,
        throw_unknown: bool = False,
        log_error: bool = False) -> tuple[list[str] | None, list[str]]:
    """
    Retrieve the ``--prompt-weighter-help`` argument value

    :param args: command line arguments

    :param throw_unknown: Raise :py:class:`DgenerateUsageError` if any other
     specified argument is not a valid dgenerate argument? This treats the
     primary model argument as optional, and only goes into effect if the
     specific argument is detected.

    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?

    :return: (values | ``None``, unknown_args_list)
    """

    parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
    parser.add_argument('--prompt-weighter-help', action='store', nargs='*', default=None)
    parsed, unknown = parser.parse_known_args(args)

    if parsed.prompt_weighter_help is not None and throw_unknown:
        _check_unknown_args(unknown, log_error)

    return parsed.prompt_weighter_help, unknown


def parse_latents_processor_help(
        args: collections.abc.Sequence[str] | None = None,
        throw_unknown: bool = False,
        log_error: bool = False) -> tuple[list[str] | None, list[str]]:
    """
    Retrieve the ``--latents-processor-help`` argument value

    :param args: command line arguments

    :param throw_unknown: Raise :py:class:`DgenerateUsageError` if any other
     specified argument is not a valid dgenerate argument? This treats the
     primary model argument as optional, and only goes into effect if the
     specific argument is detected.

    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?

    :return: (values | ``None``, unknown_args_list)
    """

    parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
    parser.add_argument('--latents-processor-help', action='store', nargs='*', default=None)
    parsed, unknown = parser.parse_known_args(args)

    if parsed.latents_processor_help is not None and throw_unknown:
        _check_unknown_args(unknown, log_error)

    return parsed.latents_processor_help, unknown


def parse_prompt_upscaler_help(
        args: collections.abc.Sequence[str] | None = None,
        throw_unknown: bool = False,
        log_error: bool = False) -> tuple[list[str] | None, list[str]]:
    """
    Retrieve the ``--prompt-upscaler-help`` argument value

    :param args: command line arguments

    :param throw_unknown: Raise :py:class:`DgenerateUsageError` if any other
     specified argument is not a valid dgenerate argument? This treats the
     primary model argument as optional, and only goes into effect if the
     specific argument is detected.

    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?

    :return: (values | ``None``, unknown_args_list)
    """

    parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
    parser.add_argument('--prompt-upscaler-help', action='store', nargs='*', default=None)
    parsed, unknown = parser.parse_known_args(args)

    if parsed.prompt_upscaler_help is not None and throw_unknown:
        _check_unknown_args(unknown, log_error)

    return parsed.prompt_upscaler_help, unknown


def parse_sub_command(
        args: collections.abc.Sequence[str] | None = None) -> tuple[str | None, list[str]]:
    """
    Retrieve the ``--sub-command`` argument value

    :param args: command line arguments

    :raise DgenerateUsageError: If no argument value was provided.

    :return: (value | ``None``, unknown_args_list)
    """

    try:
        parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
        parser.add_argument('--sub-command', action='store', default=None)
        parsed, unknown = parser.parse_known_args(args)
    except argparse.ArgumentError as e:
        raise DgenerateUsageError(e) from e

    return parsed.sub_command, unknown


def parse_offline_mode(
        args: collections.abc.Sequence[str] | None = None) -> tuple[bool, list[str]]:
    """
    Parse out ``-ofm/--offline-mode``

    :param args: command line arguments

    :raise DgenerateUsageError: If no argument value was provided.

    :return: (value | ``None``, unknown_args_list)
    """

    try:
        parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
        parser.add_argument('-ofm', '--offline-mode', action='store_true', default=False)
        parsed, unknown = parser.parse_known_args(args)
    except argparse.ArgumentError as e:
        raise DgenerateUsageError(e) from e

    return parsed.offline_mode, unknown


def parse_sub_command_help(
        args: collections.abc.Sequence[str] | None = None,
        throw_unknown: bool = False,
        log_error: bool = False) -> tuple[list[str] | None, list[str]]:
    """
    Retrieve the ``--sub-command-help`` argument value

    :param args: command line arguments

    :param throw_unknown: Raise :py:class:`DgenerateUsageError` if any other
     specified argument is not a valid dgenerate argument? This treats the
     primary model argument as optional, and only goes into effect if the
     specific argument is detected.

    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?

    :return: (values | ``None``, unknown_args_list)
    """

    parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
    parser.add_argument('--sub-command-help', action='store', nargs='*', default=None)
    parsed, unknown = parser.parse_known_args(args)

    if parsed.sub_command_help is not None and throw_unknown:
        _check_unknown_args(unknown, log_error)

    return parsed.sub_command_help, unknown


def parse_device(
        args: collections.abc.Sequence[str] | None = None,
        throw_unknown: bool = False,
        log_error: bool = False) -> tuple[str | None, list[str]]:
    """
    Retrieve the ``-d/--device`` argument value

    :param args: command line arguments

    :param throw_unknown: Raise :py:class:`DgenerateUsageError` if any other
     specified argument is not a valid dgenerate argument? This treats the
     primary model argument as optional, and only goes into effect if the
     specific argument is detected.

    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?

    :raise DgenerateUsageError: If no argument value was provided.

    :return: (value | ``None``, unknown_args_list)
    """

    try:
        parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
        parser.add_argument('-d', '--device', type=_type_device, default=None)
        parsed, unknown = parser.parse_known_args(args)
    except argparse.ArgumentError as e:
        raise DgenerateUsageError(e) from e

    if parsed.device is not None and throw_unknown:
        _check_unknown_args(unknown, log_error)

    return parsed.device, unknown


def parse_verbose(
        args: collections.abc.Sequence[str] | None = None,
        throw_unknown: bool = False,
        log_error: bool = False) -> tuple[bool, list[str]]:
    """
    Retrieve the ``-v/--verbose`` argument value

    :param args: command line arguments

    :param throw_unknown: Raise :py:class:`DgenerateUsageError` if any other
     specified argument is not a valid dgenerate argument? This treats the
     primary model argument as optional, and only goes into effect if the
     specific argument is detected.

    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?

    :return: (value, unknown_args_list)
    """

    parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
    parser.add_argument('-v', '--verbose', action='store_true')
    parsed, unknown = parser.parse_known_args(args)

    if parsed.verbose and throw_unknown:
        _check_unknown_args(unknown, log_error)

    return parsed.verbose, unknown


def parse_known_args(args: collections.abc.Sequence[str] | None = None,
                     throw: bool = True,
                     log_error: bool = True,
                     no_model: bool = True,
                     no_help: bool = True,
                     help_raises: bool = False) -> tuple[DgenerateArguments, list[str]] | None:
    """
    Parse only known arguments off the command line.

    Ignores dgenerate's only required argument ``model_path`` by default.

    No logical validation is performed, :py:meth:`DgenerateArguments.check()` is not called by this function,
    only argument parsing and simple type validation is performed by this function.

    :param args: arguments list, as in args taken from sys.argv, or in that format
    :param throw: throw :py:exc:`.DgenerateUsageError` on error? defaults to ``True``
    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?
    :param no_model: Remove the ``model_path`` argument from the parser.
    :param no_help: Remove the ``--help`` argument from the parser.
    :param help_raises: ``--help`` raises :py:exc:`dgenerate.arguments.DgenerateHelpException` ?
        When ``True``, this will occur even if ``throw=False``

    :raises DgenerateUsageError: on argument error (simple type validation only)
    :raises DgenerateHelpException:

    :return: (:py:class:`.DgenerateArguments`, unknown_args_list).
        If ``throw=False`` then ``None`` will be returned on errors.
    """

    if args is None:
        args = list(sys.argv[1:])
    else:
        args = list(args)

    try:
        _custom_parser, _ = _create_parser(
            add_model=not no_model,
            add_help=not no_help)

        # noinspection PyTypeChecker
        known, unknown = _custom_parser.parse_known_args(args, namespace=DgenerateArguments())

        return typing.cast(DgenerateArguments, known), unknown

    except DgenerateHelpException:
        if help_raises:
            raise
    except (argparse.ArgumentTypeError,
            argparse.ArgumentError) as e:
        if log_error:
            pass
            _messages.error(f'dgenerate: error: {str(e).strip()}')
        if throw:
            raise DgenerateUsageError(e) from e
        return None


def parse_args(args: collections.abc.Sequence[str] | None = None,
               overrides: dict[str, str] | None = None,
               throw: bool = True,
               log_error: bool = True,
               help_raises: bool = False) -> DgenerateArguments | None:
    """
    Parse dgenerate's command line arguments and return a configuration object.

    :param args: arguments list, as in args taken from sys.argv, or in that format
    :param overrides: Optional dictionary of overrides to apply to the
        :py:class:`.DgenerateArguments` object after parsing but before validation,
        this should consist of attribute names with values.
    :param throw: throw :py:exc:`.DgenerateUsageError` on error? defaults to ``True``
    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?
    :param help_raises: ``--help`` raises :py:exc:`dgenerate.arguments.DgenerateHelpException` ?
        When ``True``, this will occur even if ``throw=False``

    :raise DgenerateUsageError:
    :raises DgenerateHelpException:

    :return: :py:class:`.DgenerateArguments`. If ``throw=False`` then
        ``None`` will be returned on errors.
    """

    try:
        return _parse_args(args, print_usage=log_error, overrides=overrides)
    except DgenerateHelpException:
        if help_raises:
            raise
    except (_renderloopconfig.RenderLoopConfigError,
            argparse.ArgumentTypeError,
            argparse.ArgumentError) as e:
        if log_error:
            _messages.error(f'dgenerate: error: {str(e).strip()}')
        if throw:
            raise DgenerateUsageError(e) from e
        return None
