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
import dgenerate.imageprocessors as _imageprocessors
import dgenerate.mediainput as _mediainput
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.renderloop as _renderloop
import dgenerate.subcommands as _subcommands


def invoke_dgenerate(
        args: collections.abc.Sequence[str],
        render_loop: typing.Optional[_renderloop.RenderLoop] = None,
        throw: bool = False,
        log_error: bool = True,
        help_exits: bool = False):
    """
    Invoke dgenerate using its command line arguments and return a return code.

    dgenerate is invoked in the current process, this method does not spawn a subprocess.

    :param args: dgenerate command line arguments in the form of a list, see: shlex module, or sys.argv
    :param render_loop: :py:class:`dgenerate.renderloop.RenderLoop` instance,
        if ``None`` is provided one will be created.
    :param throw: Whether to throw known exceptions or handle them.
    :param log_error: Write ERROR diagnostics with :py:mod:`dgenerate.messages`?
    :param help_exits: ``--help`` raises :py:exc:`dgenerate.arguments.DgenerateHelpException` ?

    :raises dgenerate.arguments.DgenerateUsageError:
    :raises dgenerate.arguments.DgenerateHelpException:
    :raises dgenerate.mediainput.ImageSeedError:
    :raises dgenerate.mediainput.UnknownMimetypeError:
    :raises dgenerate.mediainput.FrameStartOutOfBounds:
    :raises dgenerate.imageprocessors.ImageProcessorArgumentError:
    :raises dgenerate.imageprocessors.ImageProcessorNotFoundError:
    :raises dgenerate.pipelinewrapper.InvalidModelUriError:
    :raises dgenerate.pipelinewrapper.InvalidSchedulerName:
    :raises dgenerate.pipelinewrapper.OutOfMemoryError:
    :raises dgenerate.pipelinewrapper.ModelNotFoundError:
    :raises dgenerate.plugin.ModuleFileNotFoundError:
    :raises NotImplementedError:
    :raises EnvironmentError:


    :return: integer return-code, anything other than 0 is failure
    """
    if render_loop is None:
        render_loop = _renderloop.RenderLoop()

    def rethrow_with_message(e):
        if log_error:
            _messages.log(f'dgenerate: error: {str(e).strip()}',
                          level=_messages.ERROR)
        if throw:
            raise _arguments.DgenerateUsageError(e)
        return 1

    try:
        plugin_module_paths, plugin_module_paths_rest = _arguments.parse_plugin_modules(args)
    except _arguments.DgenerateUsageError as e:
        return rethrow_with_message(e)

    image_processor_help, _ = _arguments.parse_image_processor_help(args)

    if image_processor_help is not None:
        try:
            return _imageprocessors.image_processor_help(
                names=image_processor_help,
                plugin_module_paths=plugin_module_paths,
                log_error=False,
                throw=True)
        except (_imageprocessors.ImageProcessorHelpUsageError,
                _imageprocessors.ImageProcessorNotFoundError) as e:
            return rethrow_with_message(e)

    sub_command_help, _ = _arguments.parse_sub_command_help(args)
    if sub_command_help is not None:
        try:
            return _subcommands.sub_command_help(
                names=sub_command_help,
                plugin_module_paths=plugin_module_paths,
                log_error=False,
                throw=True)
        except (_subcommands.SubCommandHelpUsageError,
                _subcommands.SubCommandNotFoundError) as e:
            return rethrow_with_message(e)

    try:
        sub_command_name, sub_command_name_rest = _arguments.parse_sub_command(plugin_module_paths_rest)
    except _arguments.DgenerateUsageError as e:
        return rethrow_with_message(e)

    if sub_command_name is not None:
        verbose, verbose_rest = _arguments.parse_verbose(sub_command_name_rest)

        if verbose:
            _messages.push_level(_messages.DEBUG)
        else:
            _messages.push_level(_messages.INFO)
        try:
            return _subcommands.SubCommandLoader().load(uri=sub_command_name,
                                                        plugin_module_paths=plugin_module_paths,
                                                        args=verbose_rest)()
        except (_subcommands.SubCommandNotFoundError,
                _subcommands.SubCommandArgumentError) as e:
            return rethrow_with_message(e)
        finally:
            _messages.pop_level()

    template_help_variable_names, _ = _arguments.parse_templates_help(args)
    directives_help_variable_names, _ = _arguments.parse_directives_help(args)

    if template_help_variable_names is not None \
            or directives_help_variable_names is not None:
        import dgenerate.batchprocess

        config_runner_args = dict()
        if plugin_module_paths:
            config_runner_args['injected_args'] = ['--plugin-modules'] + plugin_module_paths

        config_runner = dgenerate.batchprocess.ConfigRunner(**config_runner_args)
        if template_help_variable_names is not None:
            _messages.log(
                config_runner.generate_template_variables_help(
                    template_help_variable_names,
                    show_values=False) + '\n', underline=True)
        if directives_help_variable_names is not None:
            _messages.log(
                config_runner.generate_directives_help(
                    directives_help_variable_names))
        return 0

    constraint_lists = []

    try:
        arguments = _arguments.parse_args(args, log_error=log_error, help_raises=True)
    except _arguments.DgenerateHelpException:
        if help_exits:
            raise
        return 0
    except _arguments.DgenerateUsageError as e:
        if throw:
            raise e
        return 1

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

        render_loop.run()

    except (_mediainput.ImageSeedError,
            _mediainput.UnknownMimetypeError,
            _mediainput.FrameStartOutOfBounds,
            _pipelinewrapper.ModelNotFoundError,
            _pipelinewrapper.InvalidModelUriError,
            _pipelinewrapper.InvalidSchedulerName,
            _pipelinewrapper.OutOfMemoryError,
            _imageprocessors.ImageProcessorArgumentError,
            _imageprocessors.ImageProcessorNotFoundError,
            NotImplementedError,
            EnvironmentError) as e:
        return rethrow_with_message(e)
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
    return 0
