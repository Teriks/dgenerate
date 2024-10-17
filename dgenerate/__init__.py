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

__version__ = '4.2.3'

import os
import sys


# Set the maximum split size for the CUDA memory allocator
# to handle large allocations efficiently
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = os.environ.get(
    'PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')

# Keep CUDA launch blocking disabled for better performance
os.environ['PYTORCH_CUDA_LAUNCH_BLOCKING'] = os.environ.get(
    'PYTORCH_CUDA_LAUNCH_BLOCKING', '0')


__am_dgenerate_app = \
    os.path.splitext(
        os.path.basename(os.path.realpath(sys.argv[0])))[0] in {'dgenerate', 'dgenerate_windowed'}

__stderr_null = False
__stdout_null = False
__dev_null = None

if __am_dgenerate_app:
    if sys.stdout is None or sys.stderr is None:
        __dev_null = open(os.devnull, 'w', encoding='utf-8')

    if sys.stdout is None:
        sys.stdout = __dev_null
        __stdout_null = True

    if sys.stderr is None:
        sys.stderr = __dev_null
        __stderr_null = True

# handle --console meta argument

if __am_dgenerate_app and '--console' in sys.argv:
    # avoid a slow UI startup time

    import dgenerate.console as _console

    args = sys.argv[1:]
    while '--console' in args:
        args.remove('--console')
    _console.main(args)
    sys.exit(0)

import collections.abc
import warnings

if os.environ.get('DGENERATE_BACKEND_WARNINGS', '0') == '0':
    warnings.filterwarnings('ignore', module='dgenerate.extras.controlnet_aux')
    warnings.filterwarnings('ignore', module='timm')
    warnings.filterwarnings('ignore', module='peft')
    warnings.filterwarnings('ignore', module='diffusers')
    warnings.filterwarnings('ignore', module='transformers')
    warnings.filterwarnings('ignore', module='huggingface_hub')
    warnings.filterwarnings('ignore', module='torch')
    warnings.filterwarnings('ignore', module='controlnet_aux')

try:
    import diffusers
    import transformers

    from dgenerate.renderloop import (
        RenderLoop,
        RenderLoopConfig,
        RenderLoopConfigError,
        RenderLoopEvent,
        RenderLoopEventStream,
        ImageGeneratedEvent,
        ImageFileSavedEvent,
        StartingAnimationFileEvent,
        StartingAnimationEvent,
        AnimationFinishedEvent,
        AnimationFileFinishedEvent,
        AnimationETAEvent,
        StartingGenerationStepEvent,
        gen_seeds,
    )

    from dgenerate.pipelinewrapper import (
        InvalidModelFileError,
        InvalidModelUriError,
        ModelUriLoadError,
        NonHFModelDownloadError,
        InvalidSchedulerNameError,
        UnsupportedPipelineConfigError,
        ModelType,
        DataType,
        ModelNotFoundError,
        PipelineType,
    )

    from dgenerate.pipelinewrapper.util import (
        default_device
    )

    from dgenerate.exceptions import OutOfMemoryError

    from dgenerate.promptweighters import PromptWeightingUnsupported

    from dgenerate.prompt import Prompt

    from dgenerate.batchprocess import (
        BatchProcessError,
        ConfigRunner,
        ConfigRunnerPlugin,
        ConfigRunnerPluginLoader,
    )

    from dgenerate.invoker import (
        invoke_dgenerate,
        invoke_dgenerate_events,
    )

    from dgenerate.arguments import (
        parse_args,
        DgenerateUsageError,
        DgenerateArguments,
        DgenerateHelpException,
    )

    from dgenerate.mediainput import (
        ImageSeedError,
        UnknownMimetypeError,
        FrameStartOutOfBounds,
        MediaIdentificationError,
    )

    from dgenerate.imageprocessors import (
        ImageProcessorArgumentError,
        ImageProcessorNotFoundError,
        ImageProcessorImageModeError,
        ImageProcessorError,
    )

    from dgenerate.plugin import (
        ModuleFileNotFoundError,
        PluginNotFoundError,
        PluginArgumentError,
    )

    from dgenerate.textprocessing import format_image_seed_uri

    import dgenerate.messages
    import dgenerate.types
    import dgenerate.files

    transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    diffusers.logging.set_verbosity(diffusers.logging.CRITICAL)
except KeyboardInterrupt:
    print('Exiting dgenerate due to keyboard interrupt!', file=sys.stderr)
    sys.exit(1)


def main(args: collections.abc.Sequence[str] | None = None):
    """
    Entry point for the dgenerate command line tool.

    :param args: program arguments, if ``None`` is provided they will be taken from ``sys.argv``
    """

    # pyinstaller bundled apps do not
    # respect this automatically
    unbuffered_io = os.environ.get('PYTHONUNBUFFERED', '0').strip() != '0'
    encoding = 'utf-8'

    if not __stdout_null and sys.stdout.encoding.lower() != encoding:
        sys.stdout.reconfigure(encoding=encoding)

    if not __stderr_null and sys.stderr.encoding.lower() != encoding:
        sys.stderr.reconfigure(encoding=encoding)

    if not __stdout_null and unbuffered_io:
        sys.stdout = dgenerate.files.Unbuffered(sys.stdout)
        dgenerate.messages.set_message_file(sys.stdout)

    if not __stderr_null and unbuffered_io:
        sys.stderr = dgenerate.files.Unbuffered(sys.stderr)
        dgenerate.messages.set_error_file(sys.stderr)

    if args is None:
        args = sys.argv[1:]

    # handle meta arguments

    input_file = None
    shell_mode = '--shell' in args
    nostdin_mode = '--no-stdin' in args

    while '--file' in args:
        try:
            pos = args.index('--file')
        except ValueError:
            break

        try:
            input_file = args[pos + 1]
            if input_file.startswith('-'):
                raise IndexError
            args = args[:pos] + args[pos + 2:]
        except IndexError:
            dgenerate.messages.log(
                'dgenerate: error: --file missing argument.')
            sys.exit(1)

    while '--shell' in args:
        args.remove('--shell')

    while '--no-stdin' in args:
        args.remove('--no-stdin')

    if input_file and shell_mode:
        dgenerate.messages.log(
            'dgenerate: error: --shell cannot be used with --file.')
        sys.exit(1)

    try:
        render_loop = RenderLoop()
        render_loop.config = DgenerateArguments()
        # ^ this is necessary for --templates-help to
        # render all the correct values
        if input_file:
            runner = ConfigRunner(render_loop=render_loop,
                                  version=__version__,
                                  injected_args=args)
            try:
                with open(input_file, 'rt') as file:
                    runner.run_file(file)
            except (ModuleFileNotFoundError, FileNotFoundError) as e:
                dgenerate.messages.log(f'dgenerate: error: {str(e).strip()}',
                                       level=dgenerate.messages.ERROR)
                sys.exit(1)

            except BatchProcessError as e:
                dgenerate.messages.log(f'Config Error: {str(e).strip()}',
                                       level=dgenerate.messages.ERROR)
                sys.exit(1)

        elif sys.stdin is not None and (not dgenerate.files.stdin_is_tty() or shell_mode) and not nostdin_mode:
            # Not a terminal, batch process STDIN
            runner = ConfigRunner(render_loop=render_loop,
                                  version=__version__,
                                  injected_args=args)
            while True:
                try:
                    runner.run_file(sys.stdin)
                    if not shell_mode:
                        sys.exit(0)
                except ModuleFileNotFoundError as e:
                    # missing plugin file parsed by ConfigRunner out of injected args
                    dgenerate.messages.log(f'dgenerate: error: {str(e).strip()}',
                                           level=dgenerate.messages.ERROR)
                    if not shell_mode:
                        sys.exit(1)

                except BatchProcessError as e:
                    dgenerate.messages.log(f'Config Error: {str(e).strip()}',
                                           level=dgenerate.messages.ERROR)
                    if not shell_mode:
                        sys.exit(1)
        else:
            sys.exit(invoke_dgenerate(args, render_loop=render_loop))
    except KeyboardInterrupt:
        print('Exiting dgenerate due to keyboard interrupt!', file=sys.stderr)
        sys.exit(1)


__all__ = dgenerate.types.module_all()
