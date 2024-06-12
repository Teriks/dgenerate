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
import dgenerate.mediainput as _mediainput
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.plugin as _plugin
import dgenerate.renderloop as _renderloop
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


InvokeDgenerateEvents = _renderloop.RenderLoopEvent | DgenerateExitEvent
"""
Events yield-able by :py:func:`.invoke_dgenerate_events`
"""

InvokeDgenerateEventStream = typing.Generator[InvokeDgenerateEvents, None, None]
"""
Event stream produced by :py:func:`.invoke_dgenerate_events`
"""


def invoke_dgenerate(args: collections.abc.Sequence[str],
                     render_loop: _renderloop.RenderLoop | None = None,
                     throw: bool = False,
                     log_error: bool = True,
                     help_raises: bool = False) -> int:
    """
    Invoke dgenerate using its command line arguments and return a return code.

    dgenerate is invoked in the current process, this method does not spawn a subprocess.

    Meta arguments such as ``--file``, ``--shell``, ``--no-stdin``, and ``--console`` are not supported

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
    :raises dgenerate.MediaIdentificationError:
    :raises dgenerate.FrameStartOutOfBounds:
    :raises dgenerate.ImageProcessorArgumentError:
    :raises dgenerate.ImageProcessorNotFoundError:
    :raises dgenerate.InvalidModelFileError:
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
        render_loop: _renderloop.RenderLoop | None = None,
        throw: bool = False,
        log_error: bool = True,
        help_raises: bool = False) -> InvokeDgenerateEventStream:
    """
    Invoke dgenerate using its command line arguments and return a stream of events,
    you must iterate over this stream of events to progress through the execution of
    dgenerate.

    dgenerate is invoked in the current process, this method does not spawn a subprocess.

    Meta arguments such as ``--file``, ``--shell``, ``--no-stdin``, and ``--console`` are not supported

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
    :raises dgenerate.MediaIdentificationError:
    :raises dgenerate.FrameStartOutOfBounds:
    :raises dgenerate.ImageProcessorArgumentError:
    :raises dgenerate.ImageProcessorNotFoundError:
    :raises dgenerate.InvalidModelFileError:
    :raises dgenerate.InvalidModelUriError:
    :raises dgenerate.ModelUriLoadError:
    :raises dgenerate.InvalidSchedulerNameError:
    :raises dgenerate.OutOfMemoryError:
    :raises dgenerate.ModelNotFoundError:
    :raises dgenerate.ModuleFileNotFoundError:
    :raises dgenerate.UnsupportedPipelineConfigError:
    :raises EnvironmentError:


    :return: :py:data:`.InvokeDgenerateEventStream`
    """
    if render_loop is None:
        render_loop = _renderloop.RenderLoop()

    def rethrow_with_message(error):
        if log_error:
            _messages.log(f'dgenerate: error: {str(error).strip()}',
                          level=_messages.ERROR)
        if throw:
            raise _arguments.DgenerateUsageError(error)
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

    template_help_variable_names, _ = _arguments.parse_templates_help(args)
    directives_help_variable_names, _ = _arguments.parse_directives_help(args)
    functions_help_variable_names, _ = _arguments.parse_functions_help(args)

    if template_help_variable_names is not None \
            or directives_help_variable_names is not None \
            or functions_help_variable_names is not None:

        import dgenerate.batchprocess

        config_runner_args = dict()
        if plugin_module_paths:
            config_runner_args['injected_args'] = ['--plugin-modules'] + plugin_module_paths

        config_runner = dgenerate.batchprocess.ConfigRunner(**config_runner_args)
        if template_help_variable_names is not None:
            try:
                _messages.log(
                    config_runner.generate_template_variables_help(
                        template_help_variable_names,
                        show_values=False))
            except ValueError as e:
                yield rethrow_with_message(e)
                return
        if directives_help_variable_names is not None:
            try:
                _messages.log(
                    config_runner.generate_directives_help(
                        directives_help_variable_names))
            except ValueError as e:
                yield rethrow_with_message(e)
                return
        if functions_help_variable_names is not None:
            try:
                _messages.log(
                    config_runner.generate_functions_help(
                        functions_help_variable_names))
            except ValueError as e:
                yield rethrow_with_message(e)
                return

        yield DgenerateExitEvent(invoke_dgenerate_events, 0)
        return

    constraint_lists = []

    try:
        arguments = _arguments.parse_args(args, log_error=log_error, help_raises=True)
    except _arguments.DgenerateHelpException:
        if help_raises:
            raise
        yield DgenerateExitEvent(invoke_dgenerate_events, 0)
        return
    except _arguments.DgenerateUsageError as e:
        if throw:
            raise e
        yield DgenerateExitEvent(invoke_dgenerate_events, 1)
        return

    try:
        if arguments.cache_memory_constraints:
            constraint_lists.append(_pipelinewrapper.CACHE_MEMORY_CONSTRAINTS)
            _pipelinewrapper.CACHE_MEMORY_CONSTRAINTS = arguments.cache_memory_constraints

        if arguments.pipeline_cache_memory_constraints:
            constraint_lists.append(_pipelinewrapper.PIPELINE_CACHE_MEMORY_CONSTRAINTS)
            _pipelinewrapper.PIPELINE_CACHE_MEMORY_CONSTRAINTS = arguments.pipeline_cache_memory_constraints

        if arguments.unet_cache_memory_constraints:
            constraint_lists.append(_pipelinewrapper.UNET_CACHE_MEMORY_CONSTRAINTS)
            _pipelinewrapper.UNET_CACHE_MEMORY_CONSTRAINTS = arguments.unet_cache_memory_constraints

        if arguments.vae_cache_memory_constraints:
            constraint_lists.append(_pipelinewrapper.VAE_CACHE_MEMORY_CONSTRAINTS)
            _pipelinewrapper.VAE_CACHE_MEMORY_CONSTRAINTS = arguments.vae_cache_memory_constraints

        if arguments.control_net_cache_memory_constraints:
            constraint_lists.append(_pipelinewrapper.CONTROL_NET_CACHE_MEMORY_CONSTRAINTS)
            _pipelinewrapper.CONTROL_NET_CACHE_MEMORY_CONSTRAINTS = arguments.control_net_cache_memory_constraints

        render_loop.config = arguments
        render_loop.image_processor_loader.load_plugin_modules(arguments.plugin_module_paths)

        if arguments.verbose:
            _messages.push_level(_messages.DEBUG)
        else:
            # enable setting and unsetting in batch processing
            _messages.push_level(_messages.INFO)

        yield from render_loop.events()

    except (_mediainput.ImageSeedError,
            _mediainput.UnknownMimetypeError,
            _mediainput.MediaIdentificationError,
            _mediainput.FrameStartOutOfBounds,
            _pipelinewrapper.ModelNotFoundError,
            _pipelinewrapper.NonHFModelDownloadError,
            _pipelinewrapper.InvalidModelFileError,
            _pipelinewrapper.InvalidModelUriError,
            _pipelinewrapper.ModelUriLoadError,
            _pipelinewrapper.InvalidSchedulerNameError,
            _pipelinewrapper.UnsupportedPipelineConfigError,
            _pipelinewrapper.OutOfMemoryError,
            _plugin.PluginNotFoundError,
            _plugin.PluginArgumentError,
            EnvironmentError) as e:
        yield rethrow_with_message(e)
        return
    finally:
        _messages.pop_level()

        if arguments is not None:
            if arguments.control_net_cache_memory_constraints:
                _pipelinewrapper.CONTROL_NET_CACHE_MEMORY_CONSTRAINTS = constraint_lists.pop()

            if arguments.unet_cache_memory_constraints:
                _pipelinewrapper.UNET_CACHE_MEMORY_CONSTRAINTS = constraint_lists.pop()

            if arguments.vae_cache_memory_constraints:
                _pipelinewrapper.VAE_CACHE_MEMORY_CONSTRAINTS = constraint_lists.pop()

            if arguments.pipeline_cache_memory_constraints:
                _pipelinewrapper.PIPELINE_CACHE_MEMORY_CONSTRAINTS = constraint_lists.pop()

            if arguments.cache_memory_constraints:
                _pipelinewrapper.CACHE_MEMORY_CONSTRAINTS = constraint_lists.pop()

    # Return the template environment for pipelining
    yield DgenerateExitEvent(invoke_dgenerate_events, 0)
    return
