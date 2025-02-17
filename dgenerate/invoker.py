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

import dgenerate.exceptions as _d_exceptions
import dgenerate.arguments as _arguments
import dgenerate.events as _event
import dgenerate.imageprocessors as _imageprocessors
import dgenerate.mediainput as _mediainput
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.promptweighters as _promptweighters
import dgenerate.promptupscalers as _promptupscalers
import dgenerate.plugin as _plugin
import dgenerate.renderloop as _renderloop
import dgenerate.subcommands as _subcommands
import dgenerate.prompt as _prompt

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
    :raises dgenerate.InvalidModelFileError:
    :raises dgenerate.InvalidModelUriError:
    :raises dgenerate.ModelUriLoadError:
    :raises dgenerate.InvalidSchedulerNameError:
    :raises dgenerate.OutOfMemoryError:
    :raises dgenerate.ModelNotFoundError:
    :raises dgenerate.NonHFModelDownloadError:
    :raises dgenerate.UnsupportedPipelineConfigError:
    :raises dgenerate.PromptWeightingUnsupported:
    :raises dgenerate.PluginNotFoundError:
    :raises dgenerate.PluginArgumentError:
    :raises dgenerate.ModuleFileNotFoundError:
    :raises OSError:

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
    :raises dgenerate.InvalidModelFileError:
    :raises dgenerate.InvalidModelUriError:
    :raises dgenerate.ModelUriLoadError:
    :raises dgenerate.InvalidSchedulerNameError:
    :raises dgenerate.OutOfMemoryError:
    :raises dgenerate.ModelNotFoundError:
    :raises dgenerate.NonHFModelDownloadError:
    :raises dgenerate.UnsupportedPipelineConfigError:
    :raises dgenerate.PromptWeightingUnsupported:
    :raises dgenerate.PromptEmbeddedArgumentError:
    :raises dgenerate.PluginNotFoundError:
    :raises dgenerate.PluginArgumentError:
    :raises dgenerate.ModuleFileNotFoundError:
    :raises OSError:

    :return: :py:data:`.InvokeDgenerateEventStream`
    """
    if render_loop is None:
        render_loop = _renderloop.RenderLoop()

    def rethrow_with_message(error, usage_error=False):
        if log_error:
            _messages.log(f'dgenerate: error: {str(error).strip()}',
                          level=_messages.ERROR)
        if throw:
            if usage_error:
                raise _arguments.DgenerateUsageError(error)
            else:
                raise error
        return DgenerateExitEvent(invoke_dgenerate_events, 1)

    try:
        plugin_module_paths, plugin_module_paths_rest = _arguments.parse_plugin_modules(args)
    except _arguments.DgenerateUsageError as e:
        yield rethrow_with_message(e)
        return

    if plugin_module_paths:
        _plugin.import_plugins(plugin_module_paths)

    try:
        prompt_weighter_help, _ = _arguments.parse_prompt_weighter_help(
            args, throw_unknown=True, log_error=log_error)
    except _arguments.DgenerateUsageError:
        if throw:
            raise
        yield DgenerateExitEvent(invoke_dgenerate_events, 1)
        return

    if prompt_weighter_help is not None:
        try:
            yield DgenerateExitEvent(
                origin=invoke_dgenerate_events,
                return_code=_promptweighters.prompt_weighter_help(
                    names=prompt_weighter_help,
                    log_error=False,
                    throw=True))

        except (_promptweighters.PromptWeighterNotFoundError,
                _plugin.ModuleFileNotFoundError) as e:
            yield rethrow_with_message(e)
        return

    try:
        prompt_upscaler_help, _ = _arguments.parse_prompt_upscaler_help(
            args, throw_unknown=True, log_error=log_error)
    except _arguments.DgenerateUsageError:
        if throw:
            raise
        yield DgenerateExitEvent(invoke_dgenerate_events, 1)
        return

    if prompt_upscaler_help is not None:
        try:
            yield DgenerateExitEvent(
                origin=invoke_dgenerate_events,
                return_code=_promptupscalers.prompt_upscaler_help(
                    names=prompt_upscaler_help,
                    log_error=False,
                    throw=True))

        except (_promptupscalers.PromptUpscalerNotFoundError,
                _plugin.ModuleFileNotFoundError) as e:
            yield rethrow_with_message(e)
        return

    try:
        image_processor_help, _ = _arguments.parse_image_processor_help(
            args, throw_unknown=True, log_error=log_error)
    except _arguments.DgenerateUsageError:
        if throw:
            raise
        yield DgenerateExitEvent(invoke_dgenerate_events, 1)
        return

    if image_processor_help is not None:
        try:
            yield DgenerateExitEvent(
                origin=invoke_dgenerate_events,
                return_code=_imageprocessors.image_processor_help(
                    names=image_processor_help,
                    log_error=False,
                    throw=True))

        except (_imageprocessors.ImageProcessorNotFoundError,
                _plugin.ModuleFileNotFoundError) as e:
            yield rethrow_with_message(e)
        return

    try:
        sub_command_help, _ = _arguments.parse_sub_command_help(
            args, throw_unknown=True, log_error=log_error)
    except _arguments.DgenerateUsageError:
        if throw:
            raise
        yield DgenerateExitEvent(invoke_dgenerate_events, 1)
        return

    if sub_command_help is not None:
        try:
            yield DgenerateExitEvent(
                origin=invoke_dgenerate_events,
                return_code=_subcommands.sub_command_help(
                    names=sub_command_help,
                    log_error=False,
                    throw=True))
        except (_subcommands.SubCommandNotFoundError,
                _plugin.ModuleFileNotFoundError) as e:
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
                return_code=_subcommands.SubCommandLoader().load(
                    plugin_module_paths=plugin_module_paths,
                    uri=sub_command_name,
                    args=verbose_rest)())
        except (_plugin.PluginNotFoundError,
                _plugin.PluginArgumentError) as e:
            yield rethrow_with_message(e)
        finally:
            _messages.pop_level()
        return

    try:
        template_help_variable_names, _ = _arguments.parse_templates_help(
            args, throw_unknown=True, log_error=log_error)
        directives_help_variable_names, _ = _arguments.parse_directives_help(
            args, throw_unknown=True, log_error=log_error)
        functions_help_variable_names, _ = _arguments.parse_functions_help(
            args, throw_unknown=True, log_error=log_error)
    except _arguments.DgenerateUsageError:
        if throw:
            raise
        yield DgenerateExitEvent(invoke_dgenerate_events, 1)
        return

    if template_help_variable_names is not None \
            or directives_help_variable_names is not None \
            or functions_help_variable_names is not None:

        import dgenerate.batchprocess

        config_runner_args = dict()
        if plugin_module_paths:
            config_runner_args['injected_args'] = ['--plugin-modules'] + plugin_module_paths

        try:
            config_runner = dgenerate.batchprocess.ConfigRunner(**config_runner_args)
        except _plugin.ModuleFileNotFoundError as e:
            yield rethrow_with_message(e)
            return

        if template_help_variable_names is not None:
            try:
                _messages.log(
                    config_runner.generate_template_variables_help(
                        template_help_variable_names,
                        show_values=False))
            except ValueError as e:
                yield rethrow_with_message(e, usage_error=True)
                return
        if directives_help_variable_names is not None:
            try:
                _messages.log(
                    config_runner.generate_directives_help(
                        directives_help_variable_names))
            except ValueError as e:
                yield rethrow_with_message(e, usage_error=True)
                return
        if functions_help_variable_names is not None:
            try:
                _messages.log(
                    config_runner.generate_functions_help(
                        functions_help_variable_names))
            except ValueError as e:
                yield rethrow_with_message(e, usage_error=True)
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
    except _arguments.DgenerateUsageError:
        if throw:
            raise
        yield DgenerateExitEvent(invoke_dgenerate_events, 1)
        return

    cache_memory_constraints = {
        "cache_memory_constraints": "_pipelinewrapper.CACHE_MEMORY_CONSTRAINTS",
        "pipeline_cache_memory_constraints": "_pipelinewrapper.PIPELINE_CACHE_MEMORY_CONSTRAINTS",
        "unet_cache_memory_constraints": "_pipelinewrapper.UNET_CACHE_MEMORY_CONSTRAINTS",
        "vae_cache_memory_constraints": "_pipelinewrapper.VAE_CACHE_MEMORY_CONSTRAINTS",
        "controlnet_cache_memory_constraints": "_pipelinewrapper.CONTROLNET_CACHE_MEMORY_CONSTRAINTS",
        "adapter_cache_memory_constraints": "_pipelinewrapper.ADAPTER_CACHE_MEMORY_CONSTRAINTS",
        "transformer_cache_memory_constraints": "_pipelinewrapper.TRANSFORMER_CACHE_MEMORY_CONSTRAINTS",
        "text_encoder_cache_memory_constraints": "_pipelinewrapper.TEXT_ENCODER_CACHE_MEMORY_CONSTRAINTS",
        "image_encoder_cache_memory_constraints": "_pipelinewrapper.IMAGE_ENCODER_CACHE_MEMORY_CONSTRAINTS",
        "image_processor_memory_constraints": "_imageprocessors.IMAGE_PROCESSOR_MEMORY_CONSTRAINTS",
        "image_processor_cuda_memory_constraints": "_imageprocessors.IMAGE_PROCESSOR_CUDA_MEMORY_CONSTRAINTS"
    }

    try:
        for arg_name, global_var in cache_memory_constraints.items():
            arg_value = getattr(arguments, arg_name, None)
            if arg_value is not None:
                constraint_lists.append(eval(global_var))
                exec(f"{global_var} = arg_value")

        render_loop.config = arguments

        render_loop.config.apply_prompt_upscalers()

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
            _pipelinewrapper.NonHFConfigDownloadError,
            _pipelinewrapper.InvalidModelFileError,
            _pipelinewrapper.InvalidModelUriError,
            _pipelinewrapper.ModelUriLoadError,
            _pipelinewrapper.SchedulerLoadError,
            _pipelinewrapper.UnsupportedPipelineConfigError,
            _promptweighters.PromptWeightingUnsupported,
            _prompt.PromptEmbeddedArgumentError,
            _plugin.ModuleFileNotFoundError,
            _plugin.PluginNotFoundError,
            _plugin.PluginArgumentError,
            _d_exceptions.OutOfMemoryError,
            OSError) as e:
        yield rethrow_with_message(e)
        return
    finally:
        _messages.pop_level()

        if arguments is not None:
            for arg_name, global_var in reversed(cache_memory_constraints.items()):
                if getattr(arguments, arg_name, None) is not None:
                    exec(f"{global_var} = constraint_lists.pop()")

    # Return the template environment for pipelining
    yield DgenerateExitEvent(invoke_dgenerate_events, 0)
    return
