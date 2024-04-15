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
import collections.abc
import typing

import dgenerate.arguments as _arguments
import dgenerate.events as _event
import dgenerate.imageprocessors as _imageprocessors
import dgenerate.messages as _messages
import dgenerate.plugin as _plugin
import dgenerate.subcommands as _subcommands

__doc__ = """
Functions to invoke dgenerate inside the current process using its command line arguments.
"""


class DgenerateExitEvent(_event.Event):
    """
    Generated in the event stream created by :py:func:`.invoke_dgenerate_events`

    Exit with return code event for :py:func:`.invoke_dgenerate_events`
    """

    return_code: int

    def __init__(self, origin, return_code: int):
        super().__init__(origin)
        self.return_code = return_code


InvokeDgenerateEvents = typing.Union[DgenerateExitEvent]
"""
Events yield-able by :py:func:`.invoke_dgenerate_events`
"""

InvokeDgenerateEventStream = typing.Generator[InvokeDgenerateEvents, None, None]
"""
Event stream produced by :py:func:`.invoke_dgenerate_events`
"""


def invoke_dgenerate(args: collections.abc.Sequence[str],
                     throw: bool = False,
                     log_error: bool = True,
                     help_raises: bool = False) -> int:
    """
    Invoke dgenerate using its command line arguments and return a return code.

    dgenerate is invoked in the current process, this method does not spawn a subprocess.

    :param args: dgenerate command line arguments in the form of a list, see: shlex module, or sys.argv
    :param render_loop: :py:class:`dgenerate.renderloop.RenderLoop` instance,
        if ``None`` is provided one will be created.
    :param throw: Whether to throw known exceptions or handle them.
    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?
    :param help_raises: ``--help`` raises :py:exc:`dgenerate.arguments.DgenerateHelpException` ?
        When ``True``, this will occur even if ``throw=False``

    :raises dgenerate.DgenerateUsageError:
    :raises dgenerate.DgenerateHelpException:
    :raises dgenerate.ImageSeedError:
    :raises dgenerate.UnknownMimetypeError:
    :raises dgenerate.FrameStartOutOfBounds:
    :raises dgenerate.ImageProcessorArgumentError:
    :raises dgenerate.ImageProcessorNotFoundError:
    :raises dgenerate.InvalidModelUriError:
    :raises dgenerate.InvalidSchedulerNameError:
    :raises dgenerate.OutOfMemoryError:
    :raises dgenerate.ModelNotFoundError:
    :raises dgenerate.ModuleFileNotFoundError:
    :raises dgenerate.UnsupportedPipelineConfigError:
    :raises EnvironmentError:

    :return: integer return-code, anything other than 0 is failure
    """

    for event in invoke_dgenerate_events(**locals()):
        if isinstance(event, DgenerateExitEvent):
            return event.return_code
    return 0


def invoke_dgenerate_events(
        args: collections.abc.Sequence[str],
        throw: bool = False,
        log_error: bool = True,
        help_raises: bool = False) -> InvokeDgenerateEventStream:
    """
    Invoke dgenerate using its command line arguments and return a stream of events,
    you must iterate over this stream of events to progress through the execution of
    dgenerate.

    dgenerate is invoked in the current process, this method does not spawn a subprocess.

    The exceptions mentioned here are those you may encounter upon iterating,
    they will not occur upon simple acquisition of the event stream iterator.

    :param args: dgenerate command line arguments in the form of a list, see: shlex module, or sys.argv
    :param render_loop: :py:class:`dgenerate.renderloop.RenderLoop` instance,
        if ``None`` is provided one will be created.
    :param throw: Whether to throw known exceptions or handle them.
    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?
    :param help_raises: ``--help`` raises :py:exc:`dgenerate.arguments.DgenerateHelpException` ?
        When ``True``, this will occur even if ``throw=False``

    :raises dgenerate.DgenerateUsageError:
    :raises dgenerate.DgenerateHelpException:
    :raises dgenerate.ImageSeedError:
    :raises dgenerate.UnknownMimetypeError:
    :raises dgenerate.FrameStartOutOfBounds:
    :raises dgenerate.ImageProcessorArgumentError:
    :raises dgenerate.ImageProcessorNotFoundError:
    :raises dgenerate.InvalidModelUriError:
    :raises dgenerate.InvalidSchedulerNameError:
    :raises dgenerate.OutOfMemoryError:
    :raises dgenerate.ModelNotFoundError:
    :raises dgenerate.ModuleFileNotFoundError:
    :raises dgenerate.UnsupportedPipelineConfigError:
    :raises EnvironmentError:


    :return: :py:data:`.InvokeDgenerateEventStream`
    """

    def rethrow_with_message(e):
        if log_error:
            _messages.log(f'dgenerate: error: {str(e).strip()}',
                          level=_messages.ERROR)
        if throw:
            raise _arguments.DgenerateUsageError(e)
        return DgenerateExitEvent(invoke_dgenerate_events, 1)

    try:
        plugin_module_paths, plugin_module_paths_rest = _arguments.parse_plugin_modules(args)
    except _arguments.DgenerateUsageError as e:
        yield rethrow_with_message(e)
        return

    image_processor_help, _ = _arguments.parse_image_processor_help(args)

    if image_processor_help is not None:
        try:
            yield DgenerateExitEvent(
                origin=invoke_dgenerate_events,
                return_code=_imageprocessors.image_processor_help(
                    names=image_processor_help,
                    plugin_module_paths=plugin_module_paths,
                    log_error=False,
                    throw=True))

        except _imageprocessors.ImageProcessorHelpUsageError as e:
            yield rethrow_with_message(e)
        return

    sub_command_help, _ = _arguments.parse_sub_command_help(args)
    if sub_command_help is not None:
        try:
            yield DgenerateExitEvent(
                origin=invoke_dgenerate_events,
                return_code=_subcommands.sub_command_help(
                    names=sub_command_help,
                    plugin_module_paths=plugin_module_paths,
                    log_error=False,
                    throw=True))
        except _subcommands.SubCommandHelpUsageError as e:
            yield rethrow_with_message(e)
        return

    try:
        sub_command_name, sub_command_name_rest = _arguments.parse_sub_command(plugin_module_paths_rest)
    except _arguments.DgenerateUsageError as e:
        yield rethrow_with_message(e)
        return

    sub_command_name = "image-process"

    if sub_command_name is not None:
        verbose, verbose_rest = _arguments.parse_verbose(sub_command_name_rest)

        if verbose:
            _messages.push_level(_messages.DEBUG)
        else:
            _messages.push_level(_messages.INFO)
        try:
            yield DgenerateExitEvent(
                origin=invoke_dgenerate_events,
                return_code=_subcommands.SubCommandLoader().load(uri=sub_command_name,
                                                                 plugin_module_paths=plugin_module_paths,
                                                                 args=verbose_rest)())
        except (_plugin.PluginNotFoundError,
                _plugin.PluginArgumentError) as e:
            yield rethrow_with_message(e)
        finally:
            _messages.pop_level()
        return

    exit(1)
    return
