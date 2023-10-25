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

__version__ = '2.0.0'

import sys
import warnings

warnings.filterwarnings('ignore', module='controlnet_aux')
warnings.filterwarnings('ignore', module='diffusers')
warnings.filterwarnings('ignore', module='transformers')
warnings.filterwarnings('ignore', module='huggingface_hub')

try:
    import diffusers
    import transformers
    from dgenerate.diffusionloop import \
        DiffusionRenderLoop, \
        DiffusionRenderLoopConfig, \
        DiffusionRenderLoopConfigError, \
        ImageGeneratedCallbackArgument, \
        gen_seeds

    from dgenerate.pipelinewrapper import \
        InvalidModelUriError, \
        InvalidSchedulerName, \
        ModelTypes, \
        DataTypes, \
        DiffusionArguments, \
        OutOfMemoryError

    from dgenerate.prompt import Prompt
    from dgenerate.batchprocess import BatchProcessError, create_config_runner
    from dgenerate.invoker import invoke_dgenerate
    from dgenerate.arguments import parse_args, DgenerateUsageError
    from dgenerate.pipelinewrapper import ModelTypes, DiffusionArguments
    from dgenerate.mediainput import ImageSeedError, UnknownMimetypeError, ImageSeed

    from dgenerate.preprocessors import ImagePreprocessorArgumentError, ImagePreprocessorNotFoundError
    from dgenerate import messages

    transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    diffusers.logging.set_verbosity(diffusers.logging.CRITICAL)
except KeyboardInterrupt:
    print('Aborting due to keyboard interrupt!')
    sys.exit(1)


def main():
    """
    Entry point for the dgenerate command line tool.
    """
    try:
        render_loop = DiffusionRenderLoop()
        if not sys.stdin.isatty():
            # Not a terminal, batch process STDIN
            try:
                create_config_runner(render_loop=render_loop,
                                     version=__version__,
                                     injected_args=sys.argv[1:], throw=True).run_file(sys.stdin)
            except BatchProcessError as e:
                messages.log(f'Config Error: {e}', level=messages.ERROR)
                sys.exit(1)
        else:
            sys.exit(invoke_dgenerate(sys.argv[1:], render_loop=render_loop))
    except KeyboardInterrupt:
        print('Aborting due to keyboard interrupt!', file=sys.stderr)
        sys.exit(1)
