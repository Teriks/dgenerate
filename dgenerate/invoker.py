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
import typing

import dgenerate.arguments as _arguments
import dgenerate.mediainput as _mediainput
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.preprocessors as _preprocessors
import dgenerate.postprocessors as _postprocessors
import dgenerate.renderloop as _renderloop


def invoke_dgenerate(
        args: typing.Sequence[str],
        render_loop: typing.Optional[_renderloop.RenderLoop] = None,
        throw: bool = False):
    """
    Invoke dgenerate using its command line arguments and return a return code.

    dgenerate is invoked in the current process, this method does not spawn a subprocess.

    :param args: dgenerate command line arguments in the form of a list, see: shlex module, or sys.argv
    :param render_loop: :py:class:`dgenerate.renderloop.RenderLoop` instance,
        if None is provided one will be created.
    :param throw: Whether to throw exceptions or handle them.


    :raises dgenerate.arguments.DgenerateUsageError:
    :raises dgenerate.mediainput.ImageSeedError:
    :raises dgenerate.mediainput.UnknownMimetypeError:
    :raises dgenerate.preprocessors.ImagePreprocessorArgumentError:
    :raises dgenerate.preprocessors.ImagePreprocessorNotFoundError:
    :raises dgenerate.postprocessors.ImagePostprocessorArgumentError:
    :raises dgenerate.postprocessors.ImagePostprocessorNotFoundError:
    :raises dgenerate.pipelinewrapper.InvalidModelUriError:
    :raises dgenerate.pipelinewrapper.InvalidSchedulerName:
    :raises dgenerate.pipelinewrapper.OutOfMemoryError:
    :raises dgenerate.pipelinewrapper.ModelNotFoundError:
    :raises NotImplementedError:
    :raises EnvironmentError:


    :return: integer return-code, anything other than 0 is failure
    """
    if render_loop is None:
        render_loop = _renderloop.RenderLoop()

    if '-iph' in args or '--image-preprocessor-help' in args:
        try:
            return _preprocessors.image_preprocessor_help(args, throw=throw)
        except _preprocessors.PreprocessorHelpUsageError as e:
            raise _arguments.DgenerateUsageError(e)

    if '-pph' in args or '--postprocessor-help' in args:
        try:
            return _postprocessors.postprocessor_help(args, throw=throw)
        except _postprocessors.PostprocessorHelpUsageError as e:
            raise _arguments.DgenerateUsageError(e)

    if '--templates-help' in args:
        _messages.log(render_loop.generate_template_variables_help(show_values=False) + '\n', underline=True)
        return 0

    arguments = None
    constraint_lists = []

    try:
        arguments = _arguments.parse_args(args)
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

        if arguments.vae_cache_memory_constraints:
            constraint_lists.append(_pipelinewrapper.VAE_CACHE_MEMORY_CONSTRAINTS)
            _pipelinewrapper.VAE_CACHE_MEMORY_CONSTRAINTS = arguments.vae_cache_memory_constraints

        if arguments.control_net_cache_memory_constraints:
            constraint_lists.append(_pipelinewrapper.CONTROL_NET_CACHE_MEMORY_CONSTRAINTS)
            _pipelinewrapper.CONTROL_NET_CACHE_MEMORY_CONSTRAINTS = arguments.control_net_cache_memory_constraints

        render_loop.config = arguments

        render_loop.preprocessor_loader. \
            load_plugin_modules(arguments.plugin_module_paths)

        render_loop.postprocessor_loader. \
            load_plugin_modules(arguments.plugin_module_paths)

        if arguments.verbose:
            _messages.LEVEL = _messages.DEBUG
        else:
            # enable setting and unsetting in batch processing
            _messages.LEVEL = _messages.INFO

        render_loop.run()

    except (_mediainput.ImageSeedError,
            _mediainput.UnknownMimetypeError,
            _pipelinewrapper.ModelNotFoundError,
            _pipelinewrapper.InvalidModelUriError,
            _pipelinewrapper.InvalidSchedulerName,
            _pipelinewrapper.OutOfMemoryError,
            _preprocessors.ImagePreprocessorArgumentError,
            _preprocessors.ImagePreprocessorNotFoundError,
            _postprocessors.ImagePostprocessorArgumentError,
            _postprocessors.ImagePostprocessorNotFoundError,
            NotImplementedError,
            EnvironmentError) as e:

        _messages.log(f'Error: {e}', level=_messages.ERROR)

        if throw:
            raise e
        return 1
    finally:
        if arguments is not None:
            if arguments.control_net_cache_memory_constraints:
                _pipelinewrapper.CONTROL_NET_CACHE_MEMORY_CONSTRAINTS = constraint_lists.pop()

            if arguments.vae_cache_memory_constraints:
                _pipelinewrapper.VAE_CACHE_MEMORY_CONSTRAINTS = constraint_lists.pop()

            if arguments.pipeline_cache_memory_constraints:
                _pipelinewrapper.PIPELINE_CACHE_MEMORY_CONSTRAINTS = constraint_lists.pop()

            if arguments.cache_memory_constraints:
                _pipelinewrapper.CACHE_MEMORY_CONSTRAINTS = constraint_lists.pop()

    # Return the template environment for pipelining
    return 0
