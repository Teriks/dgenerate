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

import optimum.quanto

import dgenerate
import dgenerate.imageprocessors.constants as _imgp_constants
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.memoize as _memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.pipelinewrapper.util as _pipelinewrapper_util
import dgenerate.prompt as _prompt
import dgenerate.promptweighters as _promptweighters
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types

__doc__ = """
Argument parsing for the dgenerate command line tool.
"""

_SUPPORTED_MODEL_TYPES_PRETTY = \
    _textprocessing.oxford_comma(_pipelinewrapper.supported_model_type_strings(), 'or')

_SUPPORTED_ANIMATION_OUTPUT_FORMATS_PRETTY = \
    _textprocessing.oxford_comma(_mediaoutput.get_supported_animation_writer_formats(), 'or')

_SUPPORTED_STATIC_IMAGE_OUTPUT_FORMATS_PRETTY = \
    _textprocessing.oxford_comma(_mediaoutput.get_supported_static_image_formats(), 'or')

_SUPPORTED_DATA_TYPES_PRETTY = \
    _textprocessing.oxford_comma(_pipelinewrapper.supported_data_type_strings(), 'or')


class DgenerateHelpException(Exception):
    """
    Raised by :py:func:`.parse_args` and :py:func:`.parse_known_args`
    when ``--help`` is encountered and ``help_raises=True``
    """
    pass


class _DgenerateUnknownArgumentError(Exception):
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
        raise argparse.ArgumentTypeError(f'Must be {_textprocessing.oxford_comma(supported_dtypes, "or")}.')
    else:
        return _pipelinewrapper.get_data_type_enum(dtype)


def _type_prompts(prompt):
    try:
        return _prompt.Prompt.parse(prompt)
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
        raise argparse.ArgumentTypeError('Output X dimension must be divisible by 8.')

    if y % 8 != 0:
        raise argparse.ArgumentTypeError('Output Y dimension must be divisible by 8.')

    return x, y


def _type_image_coordinate(coord):
    if coord is None:
        return 0, 0

    r = coord.split(',')

    try:
        return int(r[0]), int(r[1])
    except ValueError:
        raise argparse.ArgumentTypeError('Coordinates must be integer values.')


def _type_sdxl_high_noise_fractions(val):
    try:
        val = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be a floating point number')

    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


def _type_device(device):
    try:
        if not _pipelinewrapper.is_valid_device_string(device):
            raise argparse.ArgumentTypeError(
                f'Must be cuda or cpu, or other device supported by torch. Unknown value: {device}')
    except _pipelinewrapper.InvalidDeviceOrdinalException as e:
        raise argparse.ArgumentTypeError(e)

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

    return dgenerate.gen_seeds(val)


def _type_animation_format(val):
    val = val.lower()
    if val not in _mediaoutput.get_supported_animation_writer_formats() + ['frames']:
        raise argparse.ArgumentTypeError(
            f'Must be {_SUPPORTED_ANIMATION_OUTPUT_FORMATS_PRETTY}. Unknown value: {val}')
    return val


def _type_image_format(val):
    val = val.lower()
    if val not in _mediaoutput.get_supported_static_image_formats():
        raise argparse.ArgumentTypeError(
            f'Must be one of {_textprocessing.oxford_comma(_mediaoutput.get_supported_static_image_formats(), "or")}')
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
            'Must be greater than or equal to zero, and less than or equal to one.')
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


def _type_adetailer_mask_padding(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


def _type_adetailer_mask_blur(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


def _type_adetailer_mask_dilation(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val < 0:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 0')
    return val


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
        raise _DgenerateUnknownArgumentError(message)

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
            '--version', action='version', version=f"dgenerate v{dgenerate.__version__}",
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
                    popen invocation of dgenerate."""
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
            '-ofm', '--offline-mode', action='store_true', default=False,
            help="""Whether dgenerate should try to download Hugging Face models that do not
                    exist in the disk cache, or only use what is available in the cache. Referencing
                    a model on Hugging Face that has not been cached because it was not previously
                    downloaded will result in a failure when using this option."""
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
            '-mt', '--model-type', action='store', default='torch', type=_model_type,
            help=f"""Use when loading different model types.
                     Currently supported: {_SUPPORTED_MODEL_TYPES_PRETTY}. (default: torch)"""
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
                    
                    During animation rendering each image in the batch will still be  written to the output directory
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
                    
                    This option supports: --model-type torch, torch-sdxl, torch-sd3, torch-flux, and torch-flux-fill
                    
                    Example: --adetailer-detectors Bingsu/adetailer;weight-name=face_yolov8n.pt
                    
                    The "revision" argument specifies the model revision to use for the adetailer model when loading from
                    Hugging Face repository, (The Git branch / tag, default is "main").
                    
                    The "subfolder" argument specifies the adetailer model subfolder, if specified when loading from a
                    Hugging Face repository or folder, weights from the specified subfolder.
                    
                    The "weight-name" argument indicates the name of the weights file to be loaded when
                    loading from a Hugging Face repository or folder on disk.
                    
                    The "confidence" argument indicates the confidence value to use with the YOLO 
                    detector model, this value defaults to 0.3 if not specified.
                    
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
            '-adp', '--adetailer-mask-paddings',
            nargs='+',
            action='store',
            type=_type_adetailer_mask_padding,
            default=None,
            metavar='ADETAILER_MASK_PADDING',
            dest='adetailer_mask_paddings',
            help="""One or more adetailer mask padding values to try. This specifies how much padding 
                    should be between the adetailer detected feature and the boundary of the mask area. (default: 32).""")
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
            '-adc', '--adetailer-crop-control-image',
            action='store_true',
            default=False,
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
            help=f"""Specify Text Encoders for the main model using URIs, main models
                    may use one or more text encoders depending on the --model-type value and other
                    dgenerate arguments. See: --text-encoders help for information
                    about what text encoders are needed for your invocation.
                    
                    Examples: "CLIPTextModel;model=huggingface/text_encoder",
                    "CLIPTextModelWithProjection;model=huggingface/text_encoder;revision=main",
                    "T5EncoderModel;model=text_encoder_folder_on_disk".
                    
                    For main models which require multiple text encoders, the + symbol may be used
                    to indicate that a default value should be used for a particular text encoder,
                    for example: --text-encoders + + huggingface/encoder3.  Any trailing text
                    encoders which are not specified are given their default value.
                    
                    The value "null" may be used to indicate that a specific text
                    encoder should not be loaded.
                    
                    Blob links / single file loads are not supported for Text Encoders.
                    
                    The "revision" argument specifies the model revision to use for the Text Encoder
                    when loading from Hugging Face repository, (The Git branch / tag, default is "main").
                    
                    The "variant" argument specifies the Text Encoder model variant. If "variant" is specified
                    when loading from a Hugging Face repository or folder, weights will be loaded from "variant" filename,
                    e.g. "pytorch_model.<variant>.safetensors".
                    For this argument, "variant" defaults to the value of --variant if it is not specified in the URI.
                    
                    The "subfolder" argument specifies the UNet model subfolder, if specified when loading from a
                    Hugging Face repository or folder, weights from the specified subfolder.
                    
                    The "dtype" argument specifies the Text Encoder model precision, it defaults to the value of -t/--dtype
                    and should be one of: {_SUPPORTED_DATA_TYPES_PRETTY}.
                    
                    The "quantize" argument specifies whether or not to use optimum-quanto to quantize the text encoder weights,
                    and may be passed the values {_textprocessing.oxford_comma(list(_textprocessing.quote(q) for q in optimum.quanto.qtypes.keys()), 'or')} to
                    specify the quantization datatype, this can be utilized to run Flux models with much less GPU memory.
                    
                    If you wish to load weights directly from a path on disk, you must point this argument at the folder
                    they exist in, which should also contain the config.json file for the Text Encoder.
                    For example, a downloaded repository folder from Hugging Face."""
        )
    )

    actions.append(
        parser.add_argument(
            '-te2', '--text-encoders2', nargs='+', type=_type_text_encoder, action='store', default=None,
            metavar='TEXT_ENCODER_URIS', dest='second_text_encoder_uris',
            help="""--text-encoders but for the SDXL refiner or Stable Cascade decoder model."""
        )
    )

    actions.append(
        parser.add_argument(
            '-un', '--unet', action='store', default=None, metavar="UNET_URI", dest='unet_uri',
            help=f"""Specify a UNet using a URI.
                    
                    Examples: "huggingface/unet", "huggingface/unet;revision=main", "unet_folder_on_disk".
                    
                    Blob links / single file loads are not supported for UNets.
                    
                    The "revision" argument specifies the model revision to use for the UNet when loading from
                    Hugging Face repository, (The Git branch / tag, default is "main").
                    
                    The "variant" argument specifies the UNet model variant. If "variant" is specified
                    when loading from a Hugging Face repository or folder, weights will be loaded from "variant" filename,
                    e.g. "pytorch_model.<variant>.safetensors.
                    For this argument, "variant" defaults to the value of --variant if it is not specified in the URI.
                    
                    The "subfolder" argument specifies the UNet model subfolder, if specified when loading from a
                    Hugging Face repository or folder, weights from the specified subfolder.
                    
                    The "dtype" argument specifies the UNet model precision, it defaults to the value of -t/--dtype
                    and should be one of: {_SUPPORTED_DATA_TYPES_PRETTY}.
                    
                    If you wish to load weights directly from a path on disk, you must point this argument at the folder
                    they exist in, which should also contain the config.json file for the UNet.
                    For example, a downloaded repository folder from Hugging Face."""
        )
    )

    actions.append(
        parser.add_argument(
            '-un2', '--unet2', action='store', default=None, metavar="UNET_URI", dest='second_unet_uri',
            help=f"""Specify a second UNet, this is only valid when using SDXL or Stable Cascade
                    model types. This UNet will be used for the SDXL refiner, or Stable Cascade decoder model."""
        )
    )

    actions.append(
        parser.add_argument(
            '-tf', '--transformer', action='store', default=None, metavar="TRANSFORMER_URI", dest='transformer_uri',
            help=f"""Specify a Stable Diffusion 3 or Flux Transformer model using a URI.
                    
                    Examples: "huggingface/transformer", "huggingface/transformer;revision=main", "transformer_folder_on_disk".
                    
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
                    
                    The "quantize" argument specifies whether or not to use optimum-quanto to quantize the transformer weights,
                    and may be passed the values {_textprocessing.oxford_comma(list(_textprocessing.quote(q) for q in optimum.quanto.qtypes.keys()), 'or')} to
                    specify the quantization datatype, this can be utilized to run Flux models with much less GPU memory.
                    
                    If you wish to load a weights file directly from disk, the simplest
                    way is: --transformer "transformer.safetensors", or with a dtype "transformer.safetensors;dtype=float16".
                    All loading arguments except "dtype" and "quantize" are unused in this case and may produce an
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
                    
                    Examples: "AutoencoderKL;model=vae.pt", "AsymmetricAutoencoderKL;model=huggingface/vae",
                    "AutoencoderTiny;model=huggingface/vae", "ConsistencyDecoderVAE;model=huggingface/vae".
                    
                    The AutoencoderKL encoder class accepts Hugging Face repository slugs/blob links,
                    .pt, .pth, .bin, .ckpt, and .safetensors files.
                    
                    Other encoders can only accept Hugging Face repository slugs/blob links, or a path to
                    a folder on disk with the model configuration and model file(s).
                    
                    If an AutoencoderKL VAE model file exists at a URL which serves the file as
                    a raw download, you may provide an http/https link to it and it will be
                    downloaded to dgenerates web cache.
                    
                    Aside from the "model" argument, there are four other optional arguments that can be specified,
                    these are: "revision", "variant", "subfolder", "dtype".
                    
                    They can be specified as so in any order, they are not positional:
                    "AutoencoderKL;model=huggingface/vae;revision=main;variant=fp16;subfolder=sub_folder;dtype=float16".
                    
                    The "revision" argument specifies the model revision to use for the VAE when loading from
                    Hugging Face repository or blob link, (The Git branch / tag, default is "main").
                    
                    The "variant" argument specifies the VAE model variant. If "variant" is specified
                    when loading from a Hugging Face repository or folder, weights will be loaded from
                    "variant" filename, e.g. "pytorch_model.<variant>.safetensors. "variant" in the case
                    of --vae does not default to the value of --variant to prevent failures during
                    common use cases.
                    
                    The "subfolder" argument specifies the VAE model subfolder, if specified when loading from a
                    Hugging Face repository or folder, weights from the specified subfolder.
                    
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
                    Hugging Face repository slug, path to model file on disk (for example, a .pt, .pth, .bin,
                    .ckpt, or .safetensors file), or model folder containing model files.
                    
                    If a LoRA model file exists at a URL which serves the file as
                    a raw download, you may provide an http/https link to it and it will be
                    downloaded to dgenerates web cache.
                    
                    Hugging Face blob links are not supported, see "subfolder" and "weight-name" below instead.
                    
                    Optional arguments can be provided after a LoRA model specification,
                    these are: "scale", "revision", "subfolder", and "weight-name".
                    
                    They can be specified as so in any order, they are not positional:
                    "huggingface/lora;scale=1.0;revision=main;subfolder=repo_subfolder;weight-name=lora.safetensors".
                    
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
            help="""LoRA weights are merged into the main model at this scale.  When specifying multiple
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
                    loaded --ip-adapters contain one.  An error will be produced in this situation, which
                    requires you to use this argument.
                    
                    An image encoder can also be manually specified for Stable Cascade models.
                    
                    Examples: "huggingface/image_encoder", "huggingface/image_encoder;revision=main", "image_encoder_folder_on_disk".
                    
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
                    Hugging Face repository slug, path to model file on disk (for example, a .pt, .pth, .bin,
                    .ckpt, or .safetensors file), or model folder containing model files.
                    
                    If an IP Adapter model file exists at a URL which serves the file as
                    a raw download, you may provide an http/https link to it and it will be
                    downloaded to dgenerates web cache.
                    
                    Hugging Face blob links are not supported, see "subfolder" and "weight-name" below instead.
                    
                    Optional arguments can be provided after an IP Adapter model specification,
                    these are: "scale", "revision", "subfolder", and "weight-name".
                    
                    They can be specified as so in any order, they are not positional:
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
                    These should be a Hugging Face repository slug, path to model file on disk
                    (for example, a .pt, .pth, .bin, .ckpt, or .safetensors file), or model folder
                    containing model files.
                    
                    If a Textual Inversion model file exists at a URL which serves the file as
                    a raw download, you may provide an http/https link to it and it will be
                    downloaded to dgenerates web cache.
                    
                    Hugging Face blob links are not supported, see "subfolder" and "weight-name" below instead.
                    
                    Optional arguments can be provided after the Textual Inversion model specification,
                    these are: "token", "revision", "subfolder", and "weight-name".
                    
                    They can be specified as so in any order, they are not positional:
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
                    downloaded to dgenerates web cache.
                    
                    Optional arguments can be provided after the ControlNet model specification,
                    these are: "scale", "start", "end", "revision", "variant", "subfolder", and "dtype".
                    
                    They can be specified as so in any order, they are not positional:
                    "huggingface/controlnet;scale=1.0;start=0.0;end=1.0;revision=main;variant=fp16;subfolder=repo_subfolder;dtype=float16".
                    
                    The "scale" argument specifies the scaling factor applied to the ControlNet model,
                    the default value is 1.0.
                    
                    The "start" argument specifies at what fraction of the total inference steps to begin applying
                    the ControlNet, defaults to 0.0, IE: the very beginning.
                    
                    The "end"  argument specifies at what fraction of the total inference steps to stop applying
                    the ControlNet, defaults to 1.0, IE: the very end.
                    
                    The "mode" argument can be used when using --model-type torch-flux and ControlNet Union
                    to specify the ControlNet mode.  Acceptable values are: "canny", "tile", "depth", "blur",
                    "pose", "gray", "lq". This value may also be an integer between 0 and 6, inclusive.
                    
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
                    downloaded to dgenerates web cache.
                    
                    Optional arguments can be provided after the T2IAdapter model specification,
                    these are: "scale", "revision", "variant", "subfolder", and "dtype".
                    
                    They can be specified as so in any order, they are not positional:
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
            '-sch',
            '--scheduler',
            '--schedulers',
            dest='scheduler',
            action='store', nargs='+', default=None, metavar="SCHEDULER_URI",
            help=f"""Specify a scheduler (sampler) by URI. Passing "help" to this argument
                    will print the compatible schedulers for a model without generating any images. Passing "helpargs" 
                    will yield a help message with a list of overridable arguments for each scheduler and their typical defaults.
                    Arguments listed by "helpargs" can be overridden using the URI syntax typical to other dgenerate URI arguments.
                    
                    You may pass multiple scheduler URIs to this argument, each URI will be tried in turn.
                    """
        )
    )

    actions.append(
        parser.add_argument(
            '-pag', '--pag', action='store_true', default=False,
            help=f"""Use perturbed attention guidance? This is supported
            for --model-type torch, torch-sdxl, and torch-sd3 for most use cases.
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
            '-rpag', '--sdxl-refiner-pag', action='store_true', default=False,
            help=f"""Use perturbed attention guidance in the SDXL refiner? 
            This is supported for --model-type torch-sdxl for most use cases.
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

    actions.append(
        parser.add_argument(
            '--s-cascade-decoder', action='store', default=None, metavar="MODEL_URI", dest='s_cascade_decoder_uri',
            help=f"""Specify a Stable Cascade (torch-s-cascade) decoder model path using a URI.
                    This should be a Hugging Face repository slug / blob link, path to model file
                    on disk (for example, a .pt, .pth, .bin, .ckpt, or .safetensors file), or model
                    folder containing model files.
                    
                    Optional arguments can be provided after the decoder model specification,
                    these are: "revision", "variant", "subfolder", and "dtype".
                    
                    They can be specified as so in any order, they are not positional:
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

    _second_pass_offload_group = parser.add_mutually_exclusive_group()

    actions.append(
        _second_pass_offload_group.add_argument(
            '-dqo', '--s-cascade-decoder-sequential-offload', action='store_true', default=False,
            help="""Force sequential model offloading for the Stable Cascade decoder pipeline, this may drastically
                    reduce memory consumption and allow large models to run when they would otherwise not fit in
                    your GPUs VRAM. Inference will be much slower. Mutually exclusive with --s-cascade-decoder-cpu-offload"""
        )
    )

    actions.append(
        _second_pass_offload_group.add_argument(
            '-dco', '--s-cascade-decoder-cpu-offload', action='store_true', default=False,
            help="""Force model cpu offloading for the Stable Cascade decoder pipeline, this may reduce memory consumption
                    and allow large models to run when they would otherwise not fit in your GPUs VRAM.
                    Inference will be slower. Mutually exclusive with --s-cascade-decoder-sequential-offload"""
        )
    )

    actions.append(
        parser.add_argument(
            '--s-cascade-decoder-prompts', nargs='+', action='store', metavar="PROMPT", default=None,
            type=_type_prompts,
            help="""One or more prompts to try with the Stable Cascade decoder model,
                    by default the decoder model gets the primary prompt, this argument
                    overrides that with a prompt of your choosing. The negative prompt
                    component can be specified with the same syntax as --prompts"""
        )
    )

    actions.append(
        parser.add_argument(
            '--s-cascade-decoder-inference-steps', action='store', nargs='+',
            default=[_pipelinewrapper.DEFAULT_S_CASCADE_DECODER_INFERENCE_STEPS], type=_type_inference_steps,
            metavar="INTEGER",
            help=f"""One or more inference steps values to try with the Stable Cascade decoder.
                    (default: [{_pipelinewrapper.DEFAULT_S_CASCADE_DECODER_INFERENCE_STEPS}])"""
        )
    )

    actions.append(
        parser.add_argument(
            '--s-cascade-decoder-guidance-scales', action='store', nargs='+',
            default=[_pipelinewrapper.DEFAULT_S_CASCADE_DECODER_GUIDANCE_SCALE],
            type=_type_guidance_scale, metavar="INTEGER",
            help=f"""One or more guidance scale values to try with the Stable Cascade decoder.
                     (default: [{_pipelinewrapper.DEFAULT_S_CASCADE_DECODER_GUIDANCE_SCALE}])"""
        )
    )

    actions.append(
        parser.add_argument(
            '--s-cascade-decoder-scheduler',
            '--s-cascade-decoder-schedulers',
            dest='s_cascade_decoder_scheduler',
            nargs='+', action='store', default=None, metavar="SCHEDULER_URI",
            help="""Specify a scheduler (sampler) by URI for the Stable Cascade decoder pass.
                    Operates the exact same way as --scheduler including the "help" option. Passing 'helpargs' 
                    will yield a help message with a list of overridable arguments for each scheduler and
                    their typical defaults. Defaults to the value of --scheduler.
                    
                    You may pass multiple scheduler URIs to this argument, each URI will be tried in turn.
                    """
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-refiner', action='store', default=None, metavar="MODEL_URI", dest='sdxl_refiner_uri',
            help=f"""Specify a Stable Diffusion XL (torch-sdxl) refiner model path using a URI.
                    This should be a Hugging Face repository slug / blob link, path to model file
                    on disk (for example, a .pt, .pth, .bin, .ckpt, or .safetensors file), or model
                    folder containing model files.
                    
                    Optional arguments can be provided after the SDXL refiner model specification,
                    these are: "revision", "variant", "subfolder", and "dtype".
                    
                    They can be specified as so in any order, they are not positional:
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

    _second_pass_offload_group.add_argument(
        '-rqo', '--sdxl-refiner-sequential-offload', action='store_true', default=False,
        help="""Force sequential model offloading for the SDXL refiner pipeline, this may drastically
                reduce memory consumption and allow large models to run when they would otherwise not fit in
                your GPUs VRAM. Inference will be much slower. Mutually exclusive with --refiner-cpu-offload"""
    )

    actions.append(
        _second_pass_offload_group.add_argument(
            '-rco', '--sdxl-refiner-cpu-offload', action='store_true', default=False,
            help="""Force model cpu offloading for the SDXL refiner pipeline, this may reduce memory consumption
                    and allow large models to run when they would otherwise not fit in your GPUs VRAM.
                    Inference will be slower. Mutually exclusive with --refiner-sequential-offload"""
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-refiner-scheduler',
            '--sdxl-refiner-schedulers',
            dest='sdxl_refiner_scheduler',
            nargs='+', action='store', default=None, metavar="SCHEDULER_URI",
            help="""Specify a scheduler (sampler) by URI for the SDXL refiner pass. Operates the exact
                 same way as --scheduler including the "help" option. Passing 'helpargs' will yield a help
                 message with a list of overridable arguments for each scheduler and their typical defaults.
                 Defaults to the value of --scheduler.
                 
                 You may pass multiple scheduler URIs to this argument, each URI will be tried in turn.
                 """
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
            '--sdxl-second-prompts', nargs='+', action='store', metavar="PROMPT", default=None, type=_type_prompts,
            help="""One or more secondary prompts to try using SDXL's secondary text encoder.
                    By default the model is passed the primary prompt for this value, this option
                    allows you to choose a different prompt. The negative prompt component can be
                    specified with the same syntax as --prompts"""
        )
    )

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
            help="""One or more Stable Diffusion XL (torch-sdxl) "aesthetic-score" micro-conditioning parameters.
                    Used to simulate an aesthetic score of the generated image by influencing the positive text
                    condition. Part of SDXL's micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952]."""
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-crops-coords-top-left', metavar="COORD", action='store', nargs='+', default=[],
            type=_type_image_coordinate,
            help="""One or more Stable Diffusion XL (torch-sdxl) "negative-crops-coords-top-left" micro-conditioning
                    parameters in the format "0,0". --sdxl-crops-coords-top-left can be used to generate an image that
                    appears to be "cropped" from the position --sdxl-crops-coords-top-left downwards. Favorable,
                    well-centered images are usually achieved by setting --sdxl-crops-coords-top-left to "0,0".
                    Part of SDXL's micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952]."""
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-original-size', '--sdxl-original-sizes', dest='sdxl_original_sizes', metavar="SIZE", action='store',
            nargs='+', default=[], type=_type_size,
            help="""One or more Stable Diffusion XL (torch-sdxl) "original-size" micro-conditioning parameters in
                    the format (WIDTH)x(HEIGHT). If not the same as --sdxl-target-size the image will appear to be
                    down or up-sampled. --sdxl-original-size defaults to --output-size or the size of any input
                    images if not specified. Part of SDXL's micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952]"""
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-target-size', '--sdxl-target-sizes', dest='sdxl_target_sizes', metavar="SIZE", action='store',
            nargs='+', default=[], type=_type_size,
            help="""One or more Stable Diffusion XL (torch-sdxl) "target-size" micro-conditioning parameters in
                    the format (WIDTH)x(HEIGHT). For most cases, --sdxl-target-size should be set to the desired
                    height and width of the generated image. If not specified it will default to --output-size or
                    the size of any input images. Part of SDXL\'s micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952]"""
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-negative-aesthetic-scores', metavar="FLOAT", action='store', nargs='+', default=[], type=float,
            help="""One or more Stable Diffusion XL (torch-sdxl) "negative-aesthetic-score" micro-conditioning parameters.
                    Part of SDXL's micro-conditioning as explained in section 2.2 of [https://huggingface.co/papers/2307.01952].
                    Can be used to simulate an aesthetic score of the generated image by influencing the negative text condition."""
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-negative-original-sizes', metavar="SIZE", action='store', nargs='+', default=[], type=_type_size,
            help="""One or more Stable Diffusion XL (torch-sdxl) "negative-original-sizes" micro-conditioning parameters.
                    Negatively condition the generation process based on a specific image resolution. Part of SDXL's
                    micro-conditioning as explained in section 2.2 of [https://huggingface.co/papers/2307.01952].
                    For more information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208"""
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-negative-target-sizes', metavar="SIZE", action='store', nargs='+', default=[], type=_type_size,
            help="""One or more Stable Diffusion XL (torch-sdxl) "negative-original-sizes" micro-conditioning parameters.
                    To negatively condition the generation process based on a target image resolution. It should be as same
                    as the "--sdxl-target-size" for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952]. For more information, refer to this issue thread:
                    https://github.com/huggingface/diffusers/issues/4208."""
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-negative-crops-coords-top-left', metavar="COORD", action='store', nargs='+', default=[],
            type=_type_image_coordinate,
            help="""One or more Stable Diffusion XL (torch-sdxl) "negative-crops-coords-top-left" micro-conditioning
                    parameters in the format "0,0". Negatively condition the generation process based on a specific
                    crop coordinates. Part of SDXL's micro-conditioning as explained in section 2.2 of
                    [https://huggingface.co/papers/2307.01952]. For more information, refer
                    to this issue thread: https://github.com/huggingface/diffusers/issues/4208."""
        )
    )

    # SDXL Refiner pipeline

    actions.append(
        parser.add_argument(
            '--sdxl-refiner-prompts', nargs='+', action='store', metavar="PROMPT", default=None, type=_type_prompts,
            help="""One or more prompts to try with the SDXL refiner model,
                    by default the refiner model gets the primary prompt, this argument
                    overrides that with a prompt of your choosing. The negative prompt
                    component can be specified with the same syntax as --prompts"""
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-refiner-clip-skips',

            nargs='+', action='store', metavar="INTEGER", default=None, type=_type_clip_skip,
            help="""One or more clip skip override values to try for the SDXL refiner,
                    which normally uses the clip skip value for the main model when it is
                    defined by --clip-skips."""
        )
    )

    actions.append(
        parser.add_argument(
            '--sdxl-refiner-second-prompts', nargs='+', action='store', metavar="PROMPT", default=None,
            type=_type_prompts,
            help="""One or more prompts to try with the SDXL refiner models secondary
                    text encoder, by default the refiner model gets the primary prompt passed
                    to its second text encoder, this argument overrides that with a prompt
                    of your choosing. The negative prompt component can be specified with the
                    same syntax as --prompts"""
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
            help="""One or more high-noise-fraction values for Stable Diffusion XL (torch-sdxl),
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
            '-ri', '--sdxl-refiner-inference-steps', action='store', nargs='+', default=None, metavar="INT",
            type=_type_inference_steps,
            help="""One or more inference steps values for the SDXL refiner when in use.
                    Override the number of inference steps used by the SDXL refiner,
                    which defaults to the value taken from --inference-steps."""
        )
    )

    actions.append(
        parser.add_argument(
            '-rg', '--sdxl-refiner-guidance-scales', action='store', nargs='+', default=None, metavar="FLOAT",
            type=_type_guidance_scale,
            help="""One or more guidance scale values for the SDXL refiner when in use.
                    Override the guidance scale value used by the SDXL refiner,
                    which defaults to the value taken from --guidance-scales."""
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
            '-d', '--device', action='store',
            default=_pipelinewrapper_util.default_device(),
            help="""cuda / cpu, or other device supported by torch, for example mps on MacOS.
            (default: cuda, mps on MacOS). Use: cuda:0, cuda:1, cuda:2, etc. to specify a specific
            cuda supporting GPU."""
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
                    --model-type values (torch-if*).
                    
                    If only one integer value is provided, that is the value for both dimensions.
                    X/Y dimension values should be separated by "x".  
                    
                    This value defaults to 512x512 for Stable Diffusion when no --image-seeds are
                    specified (IE txt2img mode), 1024x1024 for Stable Cascade and Stable Diffusion 3/XL or
                    Flux model types, and 64x64 for --model-type torch-if (Deep Floyd stage 1).
                    
                    Deep Floyd stage 1 images passed to superscaler models (--model-type torch-ifs*) 
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
                    This directory will be created if it does not exist. (default: ./output)"""
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
                    or "dgenerate --file config.dgen".  These files will be written to --output-path and are
                    affected by --output-prefix and --output-overwrite as well. The files will be named
                    after their corresponding image or animation file. Configuration files produced for
                    animation frame images will utilize --frame-start and --frame-end to specify the
                    frame number."""
        )
    )

    actions.append(
        parser.add_argument(
            '-om', '--output-metadata', action='store_true', default=False,
            help="""Write the information produced by --output-configs to the PNG metadata of each image.
                    Metadata will not be written to animated files (yet). The data is written to a
                    PNG metadata property named DgenerateConfig and can be read using ImageMagick like so: 
                    "magick identify -format "%%[Property:DgenerateConfig] generated_file.png"."""
        )
    )

    actions.append(
        parser.add_argument(
            '-pw', '--prompt-weighter', metavar='PROMPT_WEIGHTER_URI', dest='prompt_weighter_uri', action='store',
            default=None, type=_type_prompt_weighter,
            help='Specify a prompt weighter implementation by URI, for example: --prompt-weighter compel, or '
                 '--prompt-weighter sd-embed. By default, no prompt weighting syntax is enabled, '
                 'meaning that you cannot adjust token weights as you may be able to do in software such as '
                 'ComfyUI, Automatic1111, CivitAI etc. And in some cases the length of your prompt is limited. '
                 'Prompt weighters support these special token weighting syntaxes and long prompts, '
                 'currently there are two implementations "compel" and "sd-embed". See: --prompt-weighter-help '
                 'for a list of implementation names. You may also use --prompt-weighter-help "name" to '
                 'see comprehensive documentation for a specific prompt weighter implementation.'
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
            '-p', '--prompts', nargs='+', action='store', metavar="PROMPT", default=[_prompt.Prompt()],
            type=_type_prompts,
            help="""One or more prompts to try, an image group is generated for each prompt,
                    prompt data is split by ; (semi-colon). The first value is the positive
                    text influence, things you want to see. The Second value is negative
                    influence IE. things you don't want to see.
                    Example: --prompts "photo of a horse in a field; artwork, painting, rain".
                    (default: [(empty string)])"""
        )
    )

    actions.append(
        parser.add_argument(
            '--sd3-max-sequence-length', action='store', metavar='INTEGER', default=None, type=_max_sequence_length,
            help="""The maximum amount of prompt tokens that the T5EncoderModel
                    (third text encoder) of Stable Diffusion 3 can handle. This should be
                    an integer value between 1 and 512 inclusive. The higher the value
                    the more resources and time are required for processing. (default: 256)"""
        )
    )

    actions.append(
        parser.add_argument(
            '--sd3-second-prompts', nargs='+', action='store', metavar="PROMPT", default=None, type=_type_prompts,
            help="""One or more secondary prompts to try using the torch-sd3 (Stable Diffusion 3) 
                    secondary text encoder. By default the model is passed the primary prompt for this value,
                    this option allows you to choose a different prompt. The negative prompt component can be
                    specified with the same syntax as --prompts"""
        )
    )

    actions.append(
        parser.add_argument(
            '--sd3-third-prompts', nargs='+', action='store', metavar="PROMPT", default=None, type=_type_prompts,
            help="""One or more tertiary prompts to try using the torch-sd3 (Stable Diffusion 3) 
                    tertiary (T5) text encoder. By default the model is passed the primary prompt for this value,
                    this option allows you to choose a different prompt. The negative prompt component can be
                    specified with the same syntax as --prompts"""
        )
    )

    actions.append(
        parser.add_argument(
            '--flux-second-prompts', nargs='+', action='store', metavar="PROMPT", default=None, type=_type_prompts,
            help="""One or more secondary prompts to try using the torch-flux (Flux) 
                    secondary (T5) text encoder. By default the model is passed the primary prompt for this value,
                    this option allows you to choose a different prompt."""
        )
    )

    actions.append(
        parser.add_argument(
            '--flux-max-sequence-length', action='store', metavar='INTEGER', default=None, type=_max_sequence_length,
            help="""The maximum amount of prompt tokens that the T5EncoderModel
                    (second text encoder) of Flux can handle. This should be
                    an integer value between 1 and 512 inclusive. The higher the value
                    the more resources and time are required for processing. (default: 512)"""
        )
    )

    actions.append(
        parser.add_argument(
            '-cs', '--clip-skips', nargs='+', action='store', metavar="INTEGER", default=None, type=_type_clip_skip,
            help="""One or more clip skip values to try. Clip skip is the number of layers to be skipped from CLIP
                    while computing the prompt embeddings, it must be a value greater than or equal to zero. A value of 1 means
                    that the output of the pre-final layer will be used for computing the prompt embeddings. This is only
                    supported for --model-type values "torch", "torch-sdxl", and "torch-sd3"."""
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
            help=f"""Output format when writing static images. Any selection other than "png" is not
                    compatible with --output-metadata. Value must be one of: {_SUPPORTED_STATIC_IMAGE_OUTPUT_FORMATS_PRETTY}. (default: png)"""
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
                    "my-seed-image.png;my-image-mask.png", white areas of the mask indicate where
                    generated content is to be placed in your seed image.
                    
                    Output dimensions specific to the image seed can be specified by placing the dimension
                    at the end of the string following a semicolon like so: "my-seed-image.png;512x512" or
                    "my-seed-image.png;my-image-mask.png;512x512". When using --control-nets, a singular
                    image specification is interpreted as the control guidance image, and you can specify
                    multiple control image sources by separating them with commas in the case where multiple
                    ControlNets are specified, IE: (--image-seeds "control-image1.png, control-image2.png") OR
                    (--image-seeds "seed.png;control=control-image1.png, control-image2.png").
                     
                    Using --control-nets with img2img or inpainting can be accomplished with the syntax: 
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
                    a previous deep floyd stage when using --model-type torch-ifs*. When keyword arguments
                    are present, all applicable images such as "mask", "control", etc. must also be defined
                    with keyword arguments instead of with the short syntax."""
        )
    )

    image_seed_noise_opts = parser.add_mutually_exclusive_group()

    actions.append(
        parser.add_argument(
            '-sip', '--seed-image-processors', action='store', nargs='+', default=None, metavar="PROCESSOR_URI",
            help="""Specify one or more image processor actions to perform on the primary
                    image(s) specified by --image-seeds.
                    
                    For example: --seed-image-processors "flip" "mirror" "grayscale".
                    
                    To obtain more information about what image processors are available and how to use them,
                    see: --image-processor-help.
                    
                    If you have multiple images specified for batching, for example
                    (--image-seeds "images: img2img-1.png, img2img-2.png"), you may use the delimiter "+" to separate
                    image processor chains, so that a certain chain affects a certain seed image, the plus symbol
                    may also be used to represent a null processor.
                    
                    For example: (--seed-image-processors affect-img-1 + affect-img-2), or
                    (--seed-image-processors + affect-img-2), or (--seed-image-processors affect-img-1 +).
                    
                    The amount of processors / processor chains must not exceed the amount of input images,
                    or you will receive a syntax error message. To obtain more information about what image
                    processors  are available and how to use them, see: --image-processor-help."""
        )
    )

    actions.append(
        parser.add_argument(
            '-mip', '--mask-image-processors', action='store', nargs='+', default=None, metavar="PROCESSOR_URI",
            help="""Specify one or more image processor actions to perform on the inpaint mask
                    image(s) specified by --image-seeds.
                    
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
            '-cip', '--control-image-processors', action='store', nargs='+', default=None, metavar="PROCESSOR_URI",
            help="""Specify one or more image processor actions to perform on the control
                    image specified by --image-seeds, this option is meant to be used with --control-nets.
                    
                    Example: --control-image-processors "canny;lower=50;upper=100".
                    
                    The delimiter "+" can be used to specify a different processor group for each image when using
                    multiple control images with --control-nets.
                    
                    For example if you have --image-seeds "img1.png, img2.png" or --image-seeds "...;control=img1.png, img2.png" 
                    specified and multiple ControlNet models specified with --control-nets, you can specify processors for
                    those control images with the syntax: (--control-image-processors "processes-img1" + "processes-img2").
                    
                    This syntax also supports chaining of processors, for example: 
                    (--control-image-processors "first-process-img1" "second-process-img1" + "process-img2").
                     
                    The amount of specified processors must not exceed the amount of specified control images, or you
                    will receive a syntax error message.
                    
                    Images which do not have a processor defined for them will not be processed, and the plus character can
                    be used to indicate an image is not to be processed and instead skipped over when that image is a
                    leading element, for example (--control-image-processors + "process-second") would indicate that
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
                    output before it is saved. For example: --post-processors "upcaler;model=4x_ESRGAN.pth".
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
                    resolution upscaler --model-type torch-upscaler-x4 or torch-ifs. Specifying
                    this option for --model-type torch-upscaler-x2 will produce an error message.
                    The higher this value the more noise is added to the image before upscaling
                    (similar to --image-seed-strengths). (default: [20 for x4, 250 for
                    torch-ifs/torch-ifs-img2img, 0 for torch-ifs inpainting mode])"""
        )
    )

    actions.append(
        parser.add_argument(
            '-gs', '--guidance-scales', action='store', nargs='+', default=[_pipelinewrapper.DEFAULT_GUIDANCE_SCALE],
            metavar="FLOAT", type=_type_guidance_scale,
            help="""One or more guidance scale values to try. Guidance scale effects how much your
                    text prompt is considered. Low values draw more data from images unrelated
                    to text prompt. (default: [5])"""
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
            help="""One or more guidance rescale factors to try. Proposed by [Common Diffusion Noise Schedules and
                    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf) "guidance_scale" is defined
                    as "φ" in equation 16. of [Common Diffusion Noise Schedules and Sample Steps are Flawed]
                    (https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure
                    when using zero terminal SNR. This is supported for basic text to image generation
                    when using --model-type "torch" but not inpainting, img2img, or --control-nets.
                    When using --model-type "torch-sdxl" it is supported for basic generation, inpainting,
                    and img2img, unless --control-nets is specified in which case only inpainting is supported.
                    It is supported for --model-type "torch-sdxl-pix2pix" but not --model-type "torch-pix2pix".
                    (default: [0.0])"""
        )
    )

    actions.append(
        parser.add_argument(
            '-ifs', '--inference-steps', action='store', nargs='+', default=[_pipelinewrapper.DEFAULT_INFERENCE_STEPS],
            type=_type_inference_steps, metavar="INTEGER",
            help="""One or more inference steps values to try. The amount of inference (de-noising) steps
                    effects image clarity to a degree, higher values bring the image closer to what
                    the AI is targeting for the content of the image. Values between 30-40
                    produce good results, higher values may improve image quality and or
                    change image content. (default: [30])"""
        )
    )

    actions.append(
        parser.add_argument(
            '-mc', '--cache-memory-constraints', action='store', nargs='+', default=None, type=_type_expression,
            metavar="EXPR",
            help=f"""Cache constraint expressions describing when to clear all model caches
                    automatically (DiffusionPipeline, UNet, VAE, ControlNet, and Text Encoder) considering current memory
                    usage. If any of these constraint expressions are met all models cached in memory will be cleared.
                    Example, and default value: {' '.join(_textprocessing.quote_spaces(_pipelinewrapper.CACHE_MEMORY_CONSTRAINTS))}"""
                 f' For Syntax See: [https://dgenerate.readthedocs.io/en/v{dgenerate.__version__}/'
                 f'dgenerate_submodules.html#dgenerate.pipelinewrapper.CACHE_MEMORY_CONSTRAINTS]'
        )
    )

    actions.append(
        parser.add_argument(
            '-pmc', '--pipeline-cache-memory-constraints', action='store', nargs='+', default=None,
            type=_type_expression, metavar="EXPR",
            help=f"""Cache constraint expressions describing when to automatically clear the in memory
                    DiffusionPipeline cache considering current memory usage, and estimated memory usage of
                    new models that are about to enter memory. If any of these constraint expressions are
                    met all DiffusionPipeline objects cached in memory will be cleared. Example, and default
                    value: {' '.join(_textprocessing.quote_spaces(_pipelinewrapper.PIPELINE_CACHE_MEMORY_CONSTRAINTS))}"""
                 f' For Syntax See: [https://dgenerate.readthedocs.io/en/v{dgenerate.__version__}/'
                 f'dgenerate_submodules.html#dgenerate.pipelinewrapper.PIPELINE_CACHE_MEMORY_CONSTRAINTS]'
        )
    )

    actions.append(
        parser.add_argument(
            '-umc', '--unet-cache-memory-constraints', action='store', nargs='+', default=None, type=_type_expression,
            metavar="EXPR",
            help=f"""Cache constraint expressions describing when to automatically clear the in memory UNet
                    cache considering current memory usage, and estimated memory usage of new UNet models that
                    are about to enter memory. If any of these constraint expressions are met all UNet
                    models cached in memory will be cleared. Example, and default
                    value: {' '.join(_textprocessing.quote_spaces(_pipelinewrapper.UNET_CACHE_MEMORY_CONSTRAINTS))}"""
                 f' For Syntax See: [https://dgenerate.readthedocs.io/en/v{dgenerate.__version__}/'
                 f'dgenerate_submodules.html#dgenerate.pipelinewrapper.UNET_CACHE_MEMORY_CONSTRAINTS]'
        )
    )

    actions.append(
        parser.add_argument(
            '-vmc', '--vae-cache-memory-constraints', action='store', nargs='+', default=None, type=_type_expression,
            metavar="EXPR",
            help=f"""Cache constraint expressions describing when to automatically clear the in memory VAE
                    cache considering current memory usage, and estimated memory usage of new VAE models that
                    are about to enter memory. If any of these constraint expressions are met all VAE
                    models cached in memory will be cleared. Example, and default
                    value: {' '.join(_textprocessing.quote_spaces(_pipelinewrapper.VAE_CACHE_MEMORY_CONSTRAINTS))}"""
                 f' For Syntax See: [https://dgenerate.readthedocs.io/en/v{dgenerate.__version__}/'
                 f'dgenerate_submodules.html#dgenerate.pipelinewrapper.VAE_CACHE_MEMORY_CONSTRAINTS]'
        )
    )

    actions.append(
        parser.add_argument(
            '-cmc', '--control-net-cache-memory-constraints', action='store', nargs='+', default=None,
            type=_type_expression, metavar="EXPR",
            help=f"""Cache constraint expressions describing when to automatically clear the in memory ControlNet
                    cache considering current memory usage, and estimated memory usage of new ControlNet models that
                    are about to enter memory. If any of these constraint expressions are met all ControlNet
                    models cached in memory will be cleared. Example, and default
                    value: {' '.join(_textprocessing.quote_spaces(_pipelinewrapper.CONTROLNET_CACHE_MEMORY_CONSTRAINTS))}"""
                 f' For Syntax See: [https://dgenerate.readthedocs.io/en/v{dgenerate.__version__}/'
                 f'dgenerate_submodules.html#dgenerate.pipelinewrapper.CONTROLNET_CACHE_MEMORY_CONSTRAINTS]'
        )
    )

    actions.append(
        parser.add_argument(
            '-tmc', '--text-encoder-cache-memory-constraints', action='store', nargs='+', default=None,
            type=_type_expression, metavar="EXPR",
            help=f"""Cache constraint expressions describing when to automatically clear the in memory Text Encoder
                    cache considering current memory usage, and estimated memory usage of new Text Encoder models that
                    are about to enter memory. If any of these constraint expressions are met all Text Encoder
                    models cached in memory will be cleared. Example, and default
                    value: {' '.join(_textprocessing.quote_spaces(_pipelinewrapper.TEXT_ENCODER_CACHE_MEMORY_CONSTRAINTS))}"""
                 f' For Syntax See: [https://dgenerate.readthedocs.io/en/v{dgenerate.__version__}/'
                 f'dgenerate_submodules.html#dgenerate.pipelinewrapper.TEXT_ENCODER_CACHE_MEMORY_CONSTRAINTS]'
        )
    )

    actions.append(
        parser.add_argument(
            '-iemc', '--image-encoder-cache-memory-constraints', action='store', nargs

            ='+', default=None, type=_type_expression, metavar="EXPR",
            help=f"""Cache constraint expressions describing when to automatically clear the in memory Image Encoder
                    cache considering current memory usage, and estimated memory usage of new Image Encoder models that
                    are about to enter memory. If any of these constraint expressions are met all Image Encoder
                    models cached in memory will be cleared. Example, and default
                    value: {' '.join(_textprocessing.quote_spaces(_pipelinewrapper.IMAGE_ENCODER_CACHE_MEMORY_CONSTRAINTS))}"""
                 f' For Syntax See: [https://dgenerate.readthedocs.io/en/v{dgenerate.__version__}/'
                 f'dgenerate_submodules.html#dgenerate.pipelinewrapper.IMAGE_ENCODER_CACHE_MEMORY_CONSTRAINTS]'
        )
    )

    actions.append(
        parser.add_argument(
            '-amc', '--adapter-cache-memory-constraints', action='store', nargs='+', default=None,
            type=_type_expression, metavar="EXPR",
            help=f"""Cache constraint expressions describing when to automatically clear the in memory T2I Adapter
                    cache considering current memory usage, and estimated memory usage of new T2I Adapter models that
                    are about to enter memory. If any of these constraint expressions are met all T2I Adapter
                    models cached in memory will be cleared. Example, and default
                    value: {' '.join(_textprocessing.quote_spaces(_pipelinewrapper.ADAPTER_CACHE_MEMORY_CONSTRAINTS))}"""
                 f' For Syntax See: [https://dgenerate.readthedocs.io/en/v{dgenerate.__version__}/'
                 f'dgenerate_submodules.html#dgenerate.pipelinewrapper.ADAPTER_CACHE_MEMORY_CONSTRAINTS]'
        )
    )

    actions.append(
        parser.add_argument(
            '-tfmc', '--transformer-cache-memory-constraints', action='store', nargs='+', default=None,
            type=_type_expression, metavar="EXPR",
            help=f"""Cache constraint expressions describing when to automatically clear the in memory Transformer
                    cache considering current memory usage, and estimated memory usage of new Transformer models that
                    are about to enter memory. If any of these constraint expressions are met all Transformer
                    models cached in memory will be cleared. Example, and default
                    value: {' '.join(_textprocessing.quote_spaces(_pipelinewrapper.TRANSFORMER_CACHE_MEMORY_CONSTRAINTS))}"""
                 f' For Syntax See: [https://dgenerate.readthedocs.io/en/v{dgenerate.__version__}/'
                 f'dgenerate_submodules.html#dgenerate.pipelinewrapper.TRANSFORMER_CACHE_MEMORY_CONSTRAINTS]'
        )
    )

    actions.append(
        parser.add_argument(
            '-ipmc', '--image-processor-memory-constraints', action='store', nargs='+', default=None,
            type=_type_expression, metavar="EXPR",
            help=f"""Cache constraint expressions describing when to automatically clear the entire in memory
                    diffusion model cache considering current memory usage, and estimated memory usage of new
                    image processor models that are about to enter memory. If any of these constraint expressions
                    are met all diffusion related models cached in memory will be cleared. Example, and default
                    value: {' '.join(_textprocessing.quote_spaces(_imgp_constants.IMAGE_PROCESSOR_MEMORY_CONSTRAINTS))}"""
                 f' For Syntax See: [https://dgenerate.readthedocs.io/en/v{dgenerate.__version__}/'
                 f'dgenerate_submodules.html#dgenerate.imageprocessors.IMAGE_PROCESSOR_MEMORY_CONSTRAINTS]'
        )
    )

    actions.append(
        parser.add_argument(
            '-ipcc', '--image-processor-cuda-memory-constraints', action='store', nargs='+', default=None,
            type=_type_expression, metavar="EXPR",
            help=f"""Cache constraint expressions describing when to automatically clear the last active
                    diffusion model from VRAM considering current GPU memory usage, and estimated GPU memory
                    usage of new image processor models that are about to enter VRAM. If any of these
                    constraint expressions are met the last active diffusion model in VRAM will be destroyed.
                    Example, and default value: {' '.join(_textprocessing.quote_spaces(_imgp_constants.IMAGE_PROCESSOR_CUDA_MEMORY_CONSTRAINTS))}"""
                 f' For Syntax See: [https://dgenerate.readthedocs.io/en/v{dgenerate.__version__}/'
                 f'dgenerate_submodules.html#dgenerate.imageprocessors.IMAGE_PROCESSOR_CUDA_MEMORY_CONSTRAINTS]'
        )
    )

    return parser, actions


class DgenerateUsageError(Exception):
    """
    Raised by :py:func:`.parse_args` and :py:func:`.parse_known_args` on argument usage errors.
    """
    pass


class DgenerateArguments(dgenerate.RenderLoopConfig):
    """
    Represents dgenerates parsed command line arguments, can be used
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

    cache_memory_constraints: typing.Optional[collections.abc.Sequence[str]] = None
    """
    See: :py:attr:`dgenerate.pipelinewrapper.CACHE_MEMORY_CONSTRAINTS`
    """

    pipeline_cache_memory_constraints: typing.Optional[collections.abc.Sequence[str]] = None
    """
    See: :py:attr:`dgenerate.pipelinewrapper.PIPELINE_CACHE_MEMORY_CONSTRAINTS`
    """

    unet_cache_memory_constraints: typing.Optional[collections.abc.Sequence[str]] = None
    """
    See: :py:attr:`dgenerate.pipelinewrapper.UNET_CACHE_MEMORY_CONSTRAINTS`
    """

    vae_cache_memory_constraints: typing.Optional[collections.abc.Sequence[str]] = None
    """
    See: :py:attr:`dgenerate.pipelinewrapper.VAE_CACHE_MEMORY_CONSTRAINTS`
    """

    controlnet_cache_memory_constraints: typing.Optional[collections.abc.Sequence[str]] = None
    """
    See: :py:attr:`dgenerate.pipelinewrapper.CONTROLNET_CACHE_MEMORY_CONSTRAINTS`
    """

    adapter_cache_memory_constraints: typing.Optional[collections.abc.Sequence[str]] = None
    """
    See: :py:attr:`dgenerate.pipelinewrapper.ADAPTER_CACHE_MEMORY_CONSTRAINTS`
    """

    transformer_cache_memory_constraints: typing.Optional[collections.abc.Sequence[str]] = None
    """
    See: :py:attr:`dgenerate.pipelinewrapper.TRANSFORMER_CACHE_MEMORY_CONSTRAINTS`
    """

    text_encoder_cache_memory_constraints: typing.Optional[collections.abc.Sequence[str]] = None
    """
    See: :py:attr:`dgenerate.pipelinewrapper.TEXT_ENCODER_CACHE_MEMORY_CONSTRAINTS`
    """

    image_encoder_cache_memory_constraints: typing.Optional[collections.abc.Sequence[str]] = None
    """
    See: :py:attr:`dgenerate.pipelinewrapper.IMAGE_ENCODER_CACHE_MEMORY_CONSTRAINTS`
    """

    image_processor_memory_constraints: typing.Optional[collections.abc.Sequence[str]] = None
    """
    See: :py:attr:`dgenerate.imageprocessors.IMAGE_PROCESSOR_MEMORY_CONSTRAINTS`
    """

    image_processor_cuda_memory_constraints: typing.Optional[collections.abc.Sequence[str]] = None
    """
    See: :py:attr:`dgenerate.imageprocessors.IMAGE_PROCESSOR_CUDA_MEMORY_CONSTRAINTS`
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


def _parse_args(args=None, print_usage=True) -> DgenerateArguments:
    parser = _create_parser(prints_usage=print_usage)[0]
    args = parser.parse_args(args, namespace=DgenerateArguments())
    args.check(config_attribute_name_to_option)
    return args


def _check_unknown_args(args: typing.Sequence[str], log_error: bool):
    # this treats the model argument as optional

    parser = _create_parser(add_model=True, add_help=False, prints_usage=False)[0]
    try:
        # try first to parse without adding a fake model argument
        parser.parse_args(args)
    except (argparse.ArgumentTypeError,
            argparse.ArgumentError,
            _DgenerateUnknownArgumentError) as e:

        if isinstance(e, _DgenerateUnknownArgumentError):
            # an argument is missing?

            try:
                # try again one more time with a fake model argument
                parser.parse_args(['fake_model'] + list(args))
            except (argparse.ArgumentTypeError,
                    argparse.ArgumentError,
                    _DgenerateUnknownArgumentError) as e:

                # truly erroneous command line
                if log_error:
                    _messages.log(parser.format_usage().rstrip())
                    _messages.log(str(e).strip(), level=_messages.ERROR)

                raise DgenerateUsageError(str(e))
        else:
            # something other than a missing argument
            if log_error:
                _messages.log(parser.format_usage().rstrip())
                _messages.log(str(e).strip(), level=_messages.ERROR)

            raise DgenerateUsageError(str(e))


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
        raise DgenerateUsageError(e)

    if parsed.plugin_modules is not None and throw_unknown:
        _check_unknown_args(unknown, log_error)

    return parsed.plugin_modules, unknown


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
        raise DgenerateUsageError(e)

    return parsed.sub_command, unknown


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
        raise DgenerateUsageError(e)

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

    Ignores dgenerates only required argument ``model_path`` by default.

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
            argparse.ArgumentError,
            _DgenerateUnknownArgumentError) as e:
        if log_error:
            pass
            _messages.log(f'dgenerate: error: {str(e).strip()}', level=_messages.ERROR)
        if throw:
            raise DgenerateUsageError(e)
        return None


def parse_args(args: collections.abc.Sequence[str] | None = None,
               throw: bool = True,
               log_error: bool = True,
               help_raises: bool = False) -> DgenerateArguments | None:
    """
    Parse dgenerates command line arguments and return a configuration object.

    :param args: arguments list, as in args taken from sys.argv, or in that format
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
        return _parse_args(args, print_usage=log_error)
    except DgenerateHelpException:
        if help_raises:
            raise
    except (dgenerate.RenderLoopConfigError,
            argparse.ArgumentTypeError,
            argparse.ArgumentError,
            _DgenerateUnknownArgumentError) as e:
        if log_error:
            _messages.log(f'dgenerate: error: {str(e).strip()}', level=_messages.ERROR)
        if throw:
            raise DgenerateUsageError(e)
        return None
