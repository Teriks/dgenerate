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
import typing

import dgenerate.arguments as _arguments
import dgenerate.image_process.renderloopconfig as _renderloopconfig
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
import dgenerate.torchutil as _torchutil


class ImageProcessUsageError(Exception):
    """
    Thrown by :py:func:`.parse_args` on usage errors.
    """
    pass


class ImageProcessHelpException(Exception):
    """
    Raised by :py:func:`.parse_args` when ``--help`` is encountered and ``help_raises=True``
    """
    pass


class _ImageProcessUnknownArgumentError(Exception):
    pass


def _type_align(val):
    try:
        val = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')

    if val < 1:
        raise argparse.ArgumentTypeError('Must be greater than or equal to 1')
    return val


class ImageProcessArgs(_renderloopconfig.ImageProcessRenderLoopConfig):
    """
    Configuration object for :py:class:`.ImageProcessRenderLoop`
    """

    plugin_module_paths: _types.Paths

    def __init__(self):
        super().__init__()
        self.plugin_module_paths = []


def _create_arg_parser(prog, description):
    if description is None:
        description = 'This command allows you to use dgenerate image processors directly on files of your choosing.'

    parser = argparse.ArgumentParser(
        prog,
        description=description,
        formatter_class=_textprocessing.ArgparseParagraphFormatter,
        exit_on_error=False,
        allow_abbrev=False)

    def _exit(status=0, message=None):
        if status == 0:
            # help
            raise ImageProcessHelpException('image-process --help used.')
        raise _ImageProcessUnknownArgumentError(message)

    parser.exit = _exit

    actions = []

    actions.append(parser.add_argument(
        'input', nargs='+',
        help='Input file paths, may be a static images or animated files supported by dgenerate. '
             'URLs will be downloaded.'))

    actions.append(parser.add_argument(
        '-p', '--processors', nargs='+',
        help="""One or more image processor URIs, specifying multiple will chain them together.
             See: dgenerate --image-processor-help"""))

    actions.append(
        parser.add_argument('--plugin-modules', action='store', default=[], nargs="+",
                            dest='plugin_module_paths',
                            metavar="PATH",
                            help="""Specify one or more plugin module folder paths (folder containing __init__.py) or 
                            python .py file paths to load as plugins. Plugin modules can implement image processors."""))

    actions.append(parser.add_argument(
        '-o', '--output', nargs='+', default=None,
        help="""Output files, parent directories mentioned in output paths will be created for you if
        they do not exist. If you do not specify output files, the output file will be placed next to the 
        input file with the added suffix '_processed_N' unless --output-overwrite is specified, in that case 
        it will be overwritten. If you specify multiple input files and output files, you must specify an output
        file for every input file, or a directory (indicated with a trailing directory seperator character, 
        for example "my_dir/" or "my_dir\\" if the directory does not exist yet). Failure to specify an 
        output file with a URL as an input is considered an error. Supported file extensions for image 
        output are equal to those listed under --frame-format."""))

    actions.append(parser.add_argument(
        '-ff', '--frame-format', default='png', type=_arguments._type_image_format,
        help=f'Image format for animation frames. '
             f'Must be one of: {_textprocessing.oxford_comma(_mediaoutput.get_supported_static_image_formats(), "or")}.'))

    actions.append(parser.add_argument(
        '-ox', '--output-overwrite', action='store_true',
        help='Indicate that it is okay to overwrite files, instead of appending a duplicate suffix.'))

    actions.append(parser.add_argument(
        '-r', '--resize', default=None, type=_arguments._type_size,
        help='Perform naive image resizing, the best resampling algorithm is auto selected.'))

    actions.append(parser.add_argument(
        '-na', '--no-aspect', action='store_true',
        help='Make --resize ignore aspect ratio.'))

    actions.append(parser.add_argument(
        '-al', '--align', default=1, type=_type_align,
        help="""Align images / videos dimensions to this value in pixels. 
                Default is 1, meaning no particular alignment."""))

    actions.append(parser.add_argument(
        '-d', '--device',
        default=_torchutil.default_device(),
        type=_arguments._type_device,
        help='Processing device, for example "cuda", "cuda:1". '
             'Or "mps" on MacOS. "xpu" is available for intel '
             'devices and also supports device indices. (default: cuda, mps on MacOS)'))

    actions.append(parser.add_argument(
        '-fs', '--frame-start', default=0, type=_arguments._type_frame_start,
        metavar="FRAME_NUMBER",
        help="""Starting frame slice point for animated files (zero-indexed), the specified
                frame will be included. (default: 0)"""))

    actions.append(parser.add_argument(
        '-fe', '--frame-end', default=None, type=_arguments._type_frame_end,
        metavar="FRAME_NUMBER",
        help="""Ending frame slice point for animated files (zero-indexed), the specified 
                frame will be included."""))

    write_types = parser.add_mutually_exclusive_group()

    actions.append(write_types.add_argument(
        '-nf', '--no-frames', action='store_true',
        help='Do not write frames, only an animation file. Cannot be used with --no-animation-file.'))

    actions.append(write_types.add_argument(
        '-naf', '--no-animation-file', action='store_true',
        help='Do not write an animation file, only frames. Cannot be used with --no-frames.'))

    actions.append(parser.add_argument(
        '-ofm', '--offline-mode', action='store_true',
        help="""Prevent downloads of resources that do not exist on disk already."""))

    return parser, actions


def config_attribute_name_to_option(name):
    """
    Convert an attribute name of :py:class:`.ImageProcessArgs` into its command line option name.

    :param name: the attribute name
    :return: the command line argument name as a string
    """
    _, actions = _create_arg_parser('image-process', None)

    return {a.dest: a.option_strings[-1] if a.option_strings else a.dest for a in actions}[name]


def parse_args(args: collections.abc.Sequence[str] | None = None,
               overrides: dict[str, typing.Any] | None = None,
               help_name: str = 'image-process',
               help_desc: str = None,
               throw: bool = True,
               log_error: bool = True,
               help_raises: bool = False) -> ImageProcessArgs | None:
    """
    Parse and validate the arguments used for ``image-process``, which is a dgenerate
    sub-command as well as config directive.

    :param args: command line arguments
    :param overrides: Optional dictionary of overrides to apply to the
        :py:class:`.ImageProcessArgs` object after parsing but before validation,
        this should consist of attribute names with values.
    :param help_name: program name displayed in ``--help`` output.
    :param help_desc: program description displayed in ``--help`` output.
    :param throw: throw :py:exc:`.ImageProcessUsageError` on error? defaults to ``True``
    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?
    :param help_raises: ``--help`` raises :py:exc:`.ImageProcessHelpException` ?
        When ``True``, this will occur even if ``throw=False``

    :raises ImageProcessUsageError:
    :raises ImageProcessHelpException:

    :return: parsed arguments object
    """

    parser = _create_arg_parser(help_name, help_desc)[0]
    parsed = None
    try:
        parsed = typing.cast(ImageProcessArgs, parser.parse_args(args, namespace=ImageProcessArgs()))
        if overrides:
            parsed.set_from(overrides, missing_value_throws=False)
        parsed.check()
    except ImageProcessHelpException as e:
        if help_raises:
            raise e
    except (_renderloopconfig.ImageProcessRenderLoopConfigError,
            argparse.ArgumentTypeError,
            argparse.ArgumentError,
            _ImageProcessUnknownArgumentError) as e:
        if log_error:
            _messages.error(f'{help_name}: error: {str(e).strip()}')
        if throw:
            raise ImageProcessUsageError(e) from e
        return None

    if parsed is not None:
        try:
            parsed.check(config_attribute_name_to_option)
        except _renderloopconfig.ImageProcessRenderLoopConfigError as e:
            raise ImageProcessUsageError(e) from e
    return parsed


__all__ = _types.module_all()
