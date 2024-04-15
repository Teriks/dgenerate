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

import dgenerate
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types

__doc__ = """
Argument parsing for the dgenerate command line tool.
"""


_SUPPORTED_ANIMATION_OUTPUT_FORMATS_PRETTY = \
    _textprocessing.oxford_comma(_mediaoutput.supported_animation_writer_formats(), 'or')

_SUPPORTED_STATIC_IMAGE_OUTPUT_FORMATS_PRETTY = \
    _textprocessing.oxford_comma(_mediaoutput.supported_static_image_formats(), 'or')


class DgenerateHelpException(Exception):
    """
    Raised by :py:func:`.parse_args` and :py:func:`.parse_known_args`
    when ``--help`` is encountered and ``help_raises=True``
    """
    pass


class _DgenerateUnknownArgumentError(Exception):
    pass





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



def _type_device(device):

    return device



def _type_animation_format(val):
    val = val.lower()
    if val not in _mediaoutput.supported_animation_writer_formats() + ['frames']:
        raise argparse.ArgumentTypeError(
            f'Must be {_SUPPORTED_ANIMATION_OUTPUT_FORMATS_PRETTY}. Unknown value: {val}')
    return val


def _type_image_format(val):
    if val not in _mediaoutput.supported_static_image_formats():
        raise argparse.ArgumentTypeError(
            f'Must be one of {_textprocessing.oxford_comma(_mediaoutput.supported_static_image_formats(), "or")}')
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


def _create_parser(add_model=True, add_help=True):
    parser = argparse.ArgumentParser(
        prog='dgenerate', exit_on_error=False, allow_abbrev=False, add_help=add_help,
        description="""Batch image generation and manipulation tool supporting Stable Diffusion 
        and related techniques / algorithms, with support for video and animated image processing.""")

    def _exit(status=0, message=None):
        if status == 0:
            # help
            raise DgenerateHelpException('dgenerate --help used.')
        raise _DgenerateUnknownArgumentError(message)

    parser.exit = _exit

    actions: list[Action] = []

    if add_model:
        actions.append(
            parser.add_argument('model_path', action='store',
                                help="""huggingface model repository slug, huggingface blob link to a model file, 
                                    path to folder on disk, or path to a .pt, .pth, .bin, .ckpt, or .safetensors file."""))

    actions.append(
        parser.add_argument('-v', '--verbose', action='store_true', default=False,
                            help="""Output information useful for debugging, such as pipeline 
                            call and model load parameters."""))

    actions.append(
        parser.add_argument('--version', action='version', version=f"dgenerate v{dgenerate.__version__}",
                            help="Show dgenerate's version and exit"))

    actions.append(
        parser.add_argument('--plugin-modules', action='store', default=[], nargs="+", dest='plugin_module_paths',
                            metavar="PATH",
                            help="""Specify one or more plugin module folder paths (folder containing __init__.py) or 
                            python .py file paths to load as plugins. Plugin modules can currently implement 
                            image processors and config directives."""))

    actions.append(
        parser.add_argument('--sub-command', action='store', default=None,
                            metavar="SUB_COMMAND",
                            help="""Specify the name a sub-command to invoke. dgenerate exposes some extra image processing
                            functionality through the use of sub-commands. Sub commands essentially replace the entire set
                            of accepted arguments with those of a sub-command which implements additional functionality.
                            See --sub-command-help for a list of sub-commands and help."""))

    actions.append(
        parser.add_argument('--sub-command-help', action='store', nargs='*', default=None,
                            metavar="SUB_COMMAND",
                            help="""List available sub-commands, providing sub-command names will 
                                    produce their documentation. Calling a subcommand with "--sub-command name --help" 
                                    will produce argument help output for that subcommand."""))

    # This argument is handled in dgenerate.invoker.invoke_dgenerate
    actions.append(
        parser.add_argument('--templates-help', nargs='*', dest=None, default=None, metavar='VARIABLE_NAME',
                            help="""Print a list of template variables available in dgenerate configs 
                            during batch processing from STDIN. When used as a command option, their values
                            are not presented, just their names and types. Specifying names will print 
                            type information for those variable names."""))

    actions.append(
        parser.add_argument('--directives-help', nargs='*', dest=None, default=None, metavar='DIRECTIVE_NAME',
                            help="""Print a list of directives available in dgenerate configs 
                            during batch processing from STDIN. Providing names will print 
                            documentation for the specified directive names. When used with 
                            --plugin-modules, directives implemented by the specified plugins
                             will also be listed."""))

    return parser, actions


class DgenerateUsageError(Exception):
    """
    Raised by :py:func:`.parse_args` and :py:func:`.parse_known_args` on argument usage errors.
    """
    pass


class DgenerateArguments:
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

    def __init__(self):
        super().__init__()
        self.plugin_module_paths = []


_parser, _actions = _create_parser()

_attr_name_to_option = {a.dest: a.option_strings[-1] if a.option_strings else a.dest for a in _actions}


def config_attribute_name_to_option(name):
    """
    Convert an attribute name of :py:class:`.DgenerateArguments` into its command line option name.

    :param name: the attribute name
    :return: the command line argument name as a string
    """
    return _attr_name_to_option[name]


def _parse_args(args=None) -> DgenerateArguments:
    args = _parser.parse_args(args, namespace=DgenerateArguments())
    args.check(config_attribute_name_to_option)
    return args


def parse_templates_help(
        args: typing.Optional[collections.abc.Sequence[str]] = None) -> tuple[list[str], list[str]]:
    """
    Retrieve the ``--templates-help`` argument value

    :param args: command line arguments
    :return: (value, unknown_args_list)
    """
    parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
    parser.add_argument('--templates-help', nargs='*', default=None)
    parsed, unknown = parser.parse_known_args(args)

    return parsed.templates_help, unknown


def parse_directives_help(
        args: typing.Optional[collections.abc.Sequence[str]] = None) -> tuple[list[str], list[str]]:
    """
    Retrieve the ``--directives-help`` argument value

    :param args: command line arguments
    :return: (value, unknown_args_list)
    """
    parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
    parser.add_argument('--directives-help', nargs='*', default=None)
    parsed, unknown = parser.parse_known_args(args)

    return parsed.directives_help, unknown


def parse_plugin_modules(
        args: typing.Optional[collections.abc.Sequence[str]] = None) -> tuple[list[str], list[str]]:
    """
    Retrieve the ``--plugin-modules`` argument value

    :param args: command line arguments

    :raise DgenerateUsageError: If no argument values were provided.

    :return: (values, unknown_args_list)
    """

    try:
        parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
        parser.add_argument('--plugin-modules', action='store', default=[], nargs="+")
        parsed, unknown = parser.parse_known_args(args)
    except argparse.ArgumentError as e:
        raise DgenerateUsageError(e)

    return parsed.plugin_modules, unknown


def parse_image_processor_help(
        args: typing.Optional[collections.abc.Sequence[str]] = None) -> tuple[list[str], list[str]]:
    """
    Retrieve the ``--image-processor-help`` argument value

    :param args: command line arguments
    :return: (values, unknown_args_list)
    """

    parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
    parser.add_argument('--image-processor-help', action='store', nargs='*', default=None)
    parsed, unknown = parser.parse_known_args(args)

    return parsed.image_processor_help, unknown


def parse_sub_command(
        args: typing.Optional[collections.abc.Sequence[str]] = None) -> tuple[str, list[str]]:
    """
    Retrieve the ``--sub-command`` argument value

    :param args: command line arguments

    :raise DgenerateUsageError: If no argument value was provided.

    :return: (value, unknown_args_list)
    """

    try:
        parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
        parser.add_argument('--sub-command', action='store', default=None)
        parsed, unknown = parser.parse_known_args(args)
    except argparse.ArgumentError as e:
        raise DgenerateUsageError(e)

    return parsed.sub_command, unknown


def parse_sub_command_help(
        args: typing.Optional[collections.abc.Sequence[str]] = None) -> tuple[list[str], list[str]]:
    """
    Retrieve the ``--sub-command-help`` argument value

    :param args: command line arguments
    :return: (values, unknown_args_list)
    """

    parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
    parser.add_argument('--sub-command-help', action='store', nargs='*', default=None)
    parsed, unknown = parser.parse_known_args(args)

    return parsed.sub_command_help, unknown


def parse_device(
        args: typing.Optional[collections.abc.Sequence[str]] = None) -> tuple[str, list[str]]:
    """
    Retrieve the ``-d/--device`` argument value

    :param args: command line arguments

    :raise DgenerateUsageError: If no argument value was provided.

    :return: (value, unknown_args_list)
    """

    try:
        parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
        parser.add_argument('-d', '--device', type=_type_device)
        parsed, unknown = parser.parse_known_args(args)
    except argparse.ArgumentError as e:
        raise DgenerateUsageError(e)

    return parsed.device, unknown


def parse_verbose(
        args: typing.Optional[collections.abc.Sequence[str]] = None) -> tuple[bool, list[str]]:
    """
    Retrieve the ``-v/--verbose`` argument value

    :param args: command line arguments
    :return: (value, unknown_args_list)
    """

    parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False, add_help=False)
    parser.add_argument('-v', '--verbose', action='store_true')
    parsed, unknown = parser.parse_known_args(args)
    return parsed.verbose, unknown


def parse_known_args(args: typing.Optional[collections.abc.Sequence[str]] = None,
                     throw: bool = True,
                     log_error: bool = True,
                     no_model: bool = True,
                     no_help: bool = True,
                     help_raises: bool = False) -> typing.Optional[tuple[DgenerateArguments, list[str]]]:
    """
    Parse only known arguments off the command line.

    Ignores dgenerates only required argument ``model_path`` by default.

    No logical validation is preformed, :py:meth:`DgenerateArguments.check()` is not called by this function,
    only argument parsing and simple type validation is preformed by this function.

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


def parse_args(args: typing.Optional[collections.abc.Sequence[str]] = None,
               throw: bool = True,
               log_error: bool = True,
               help_raises: bool = False) -> typing.Optional[DgenerateArguments]:
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
        return _parse_args(args)
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
