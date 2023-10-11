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
import dgenerate.preprocessors as _preprocessors
import dgenerate.textprocessing as _textprocessing


def invoke_dgenerate(
        render_loop: _diffusionloop.DiffusionRenderLoop,
        args: typing.Sequence[str],
        throw: bool = False):
    if '--image-preprocessor-help' in args:
        try:
            return _preprocessors.image_preprocessor_help(args, throw=throw)
        except _preprocessors.PreprocessorHelpUsageError as e:
            raise _arguments.DgenerateUsageError(e)

    if '--templates-help' in args:
        _messages.log(_diffusionloop.DiffusionRenderLoop(
                config=_arguments.DgenerateArguments()).generate_template_variables_help())
        return 0

    try:
        arguments = _arguments.parse_args(args, throw=True)
    except _arguments.DgenerateUsageError as e:
        if throw:
            raise e
        return 1

    render_loop.config = arguments

    render_loop.preprocessor_loader. \
        load_modules(arguments.plugin_module_paths)

    if arguments.verbose:
        _messages.LEVEL = _messages.DEBUG
    else:
        # enable setting and unsetting in batch processing
        _messages.LEVEL = _messages.INFO

    try:
        render_loop.run()
    except (_mediainput.ImageSeedError,
            _mediainput.UnknownMimetypeError,
            _pipelinewrapper.InvalidModelPathError,
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
