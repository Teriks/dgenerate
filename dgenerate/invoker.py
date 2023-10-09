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

import torch

import dgenerate.arguments as _arguments
import dgenerate.diffusionloop as _diffusionloop
import dgenerate.mediainput as _mediainput
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.plugin as _plugin
import dgenerate.preprocessors as _preprocessors


def invoke_dgenerate(
        render_loop: _diffusionloop.DiffusionRenderLoop,
        args: typing.List[typing.Union[str, float, int]],
        throw: bool = False):
    if '--image-preprocessor-help' in args:
        try:
            return _preprocessors.image_preprocessor_help(args, throw=throw)
        except _preprocessors.PreprocessorHelpUsageError as e:
            raise _arguments.DgenerateUsageError(e)

    try:
        arguments = _arguments.parse_args(args, throw=True)
    except _arguments.DgenerateUsageError as e:
        if throw:
            raise e
        return 1

    render_loop.config.set_from(arguments)

    render_loop.image_preprocessor_loader. \
        search_modules.update(_plugin.load_modules(arguments.plugin_module_paths))

    if arguments.verbose:
        _messages.LEVEL = _messages.DEBUG
    else:
        # enable setting and unsetting in batch processing
        _messages.LEVEL = _messages.INFO

    try:
        render_loop.run()
    except (_mediainput.ImageSeedParseError,
            _mediainput.ImageSeedSizeMismatchError,
            _pipelinewrapper.InvalidSDXLRefinerPathError,
            _pipelinewrapper.InvalidVaePathError,
            _pipelinewrapper.InvalidLoRAPathError,
            _pipelinewrapper.InvalidTextualInversionPathError,
            _pipelinewrapper.InvalidSchedulerName,
            _preprocessors.ImagePreprocessorArgumentError,
            _preprocessors.ImagePreprocessorNotFoundError,
            torch.cuda.OutOfMemoryError,
            NotImplementedError,
            EnvironmentError) as e:
        _messages.log(f'Error: {e}', level=_messages.ERROR)
        if throw:
            raise e
        return 1
    # Return the template environment for pipelining
    return 0
