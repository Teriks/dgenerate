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

__version__ = '3.5.0'

import sys
import os

__am_dgenerate_app = \
    os.path.splitext(
        os.path.basename(os.path.realpath(sys.argv[0])))[0] == 'dgenerate'

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

if __am_dgenerate_app and '--console' in sys.argv:
    # avoid a slow UI startup time

    import dgenerate.console as _console

    args = sys.argv[1:]
    while '--console' in args:
        args.remove('--console')
    _console.main(args)
    sys.exit(0)

import collections.abc
import typing
import warnings

warnings.filterwarnings('ignore', module='dgenerate.extras.controlnet_aux')
warnings.filterwarnings('ignore', module='timm')
warnings.filterwarnings('ignore', module='diffusers')
warnings.filterwarnings('ignore', module='transformers')
warnings.filterwarnings('ignore', module='huggingface_hub')
warnings.filterwarnings('ignore', module='torch')

try:
    import diffusers
    import transformers

    from dgenerate.renderloop import \
        RenderLoop, \
        RenderLoopConfig, \
        RenderLoopConfigError, \
        RenderLoopEvent, \
        RenderLoopEventStream, \
        ImageGeneratedEvent, \
        ImageFileSavedEvent, \
        StartingAnimationFileEvent, \
        StartingAnimationEvent, \
        AnimationFinishedEvent, \
        AnimationFileFinishedEvent, \
        AnimationETAEvent, \
        StartingGenerationStepEvent, \
        gen_seeds

    from dgenerate.pipelinewrapper import \
        InvalidModelFileError, \
        InvalidModelUriError, \
        InvalidSchedulerNameError, \
        UnsupportedPipelineConfigError, \
        ModelType, \
        DataType, \
        OutOfMemoryError, \
        ModelNotFoundError, \
        PipelineType

    from dgenerate.prompt import Prompt

    from dgenerate.batchprocess import \
        BatchProcessError, \
        ConfigRunner, \
        ConfigRunnerPlugin, \
        ConfigRunnerPluginLoader

    from dgenerate.invoker import \
        invoke_dgenerate, \
        invoke_dgenerate_events

    from dgenerate.arguments import \
        parse_args, \
        DgenerateUsageError, \
        DgenerateArguments, \
        DgenerateHelpException

    from dgenerate.mediainput import \
        ImageSeedError, \
        UnknownMimetypeError, \
        FrameStartOutOfBounds

    from dgenerate.imageprocessors import \
        ImageProcessorArgumentError, \
        ImageProcessorNotFoundError

    from dgenerate.plugin import \
        ModuleFileNotFoundError, \
        PluginNotFoundError, \
        PluginArgumentError

    import dgenerate.messages
    import dgenerate.types

    transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    diffusers.logging.set_verbosity(diffusers.logging.CRITICAL)
except KeyboardInterrupt:
    print('Exiting dgenerate due to keyboard interrupt!', file=sys.stderr)
    sys.exit(1)


class __Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def main(args: typing.Optional[collections.abc.Sequence[str]] = None):
    """
    Entry point for the dgenerate command line tool.

    :param args: program arguments, if ``None`` is provided they will be taken from ``sys.argv``
    """

    unbuffered_io = os.environ.get('PYTHONUNBUFFERED', '0').strip() != '0'
    encoding = 'utf-8'

    if not __stdout_null and sys.stdout.encoding.lower() != encoding:
        sys.stdout.reconfigure(encoding=encoding)

    if not __stderr_null and sys.stderr.encoding.lower() != encoding:
        sys.stderr.reconfigure(encoding=encoding)

    if not __stdout_null and unbuffered_io:
        sys.stdout = __Unbuffered(sys.stdout)

    if not __stderr_null and unbuffered_io:
        sys.stderr = __Unbuffered(sys.stderr)

    if args is None:
        args = sys.argv[1:]

    server_mode = '--shell' in args
    nostdin_mode = '--no-stdin' in args

    while '--shell' in args:
        args.remove('--shell')

    while '--no-stdin' in args:
        args.remove('--no-stdin')

    if server_mode and nostdin_mode:
        dgenerate.messages.log(
            'dgenerate: error: --no-stdin cannot be used with --shell.')
        sys.exit(1)

    if sys.stdin.isatty() and nostdin_mode:
        dgenerate.messages.log(
            'dgenerate: error: --no-stdin is not valid when stdin is a terminal (tty).')
        sys.exit(1)

    try:
        render_loop = RenderLoop()
        render_loop.config = DgenerateArguments()
        # ^ this is necessary for --templates-help to
        # render all the correct values

        if (not sys.stdin.isatty() or server_mode) and not nostdin_mode:
            # Not a terminal, batch process STDIN
            runner = ConfigRunner(render_loop=render_loop,
                                  version=__version__,
                                  injected_args=args)
            while True:
                try:
                    runner.run_file(sys.stdin)
                    if not server_mode:
                        sys.exit(0)
                except ModuleFileNotFoundError as e:
                    # missing plugin file parsed by ConfigRunner out of injected args
                    dgenerate.messages.log(f'dgenerate: error: {str(e).strip()}',
                                           level=dgenerate.messages.ERROR)
                    if not server_mode:
                        sys.exit(1)

                except BatchProcessError as e:
                    dgenerate.messages.log(f'Config Error: {str(e).strip()}',
                                           level=dgenerate.messages.ERROR)
                    if not server_mode:
                        sys.exit(1)
        else:
            sys.exit(invoke_dgenerate(args, render_loop=render_loop))
    except KeyboardInterrupt:
        print('Exiting dgenerate due to keyboard interrupt!', file=sys.stderr)
        sys.exit(1)


__all__ = dgenerate.types.module_all()
