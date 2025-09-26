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

import argparse
import contextlib
import glob
import inspect
import itertools
import os
import pathlib
import sys
import logging

# Set the maximum split size for the CUDA memory allocator
# and GC threshold to handle large allocations efficiently
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = os.environ.get(
    'PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.8,max_split_size_mb:512')

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
    warnings.filterwarnings('ignore', module='ctranslate2')
    logging.getLogger("diffusers.modular_pipelines").setLevel(logging.CRITICAL)

try:
    from dgenerate.resources import __version__

    # Import accelerate patch BEFORE any ML libraries that might use accelerate
    import dgenerate._patches.accelerate_device_fallback_patch

    import diffusers
    import transformers

    import dgenerate._patches.transformers_dynamiccache_patch
    import dgenerate._patches.tqdm_huggingface_hub_patch
    import dgenerate._patches.hfhub_local_entry_missing_message_patch
    import dgenerate._patches.diffusers_local_files_only_patch
    import dgenerate._patches.diffusers_single_file_config_patch


    from dgenerate.hfhub import (
        NonHFDownloadError,
        NonHFModelDownloadError,
        NonHFConfigDownloadError
    )

    from dgenerate.spacycache import (
        SpacyModelNotFoundError
    )

    from dgenerate.webcache import (
        WebFileCacheOfflineModeException
    )

    from dgenerate.pipelinewrapper import (
        InvalidModelFileError,
        InvalidModelUriError,
        ModelUriLoadError,
        SchedulerLoadError,
        SchedulerArgumentError,
        InvalidSchedulerNameError,
        UnsupportedPipelineConfigError,
        ModelType,
        DataType,
        PipelineType,
        DiffusionPipelineWrapper,
        DiffusionArguments
    )

    from dgenerate.torchutil import (
        default_device
    )

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

    from dgenerate.promptupscalers import PromptUpscalerProcessingError
    from dgenerate.promptweighters import PromptWeightingUnsupported

    from dgenerate.exceptions import OutOfMemoryError, ModelNotFoundError

    from dgenerate.prompt import Prompt, PromptEmbeddedArgumentError

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

    from dgenerate.latentsprocessors import (
        LatentsProcessorArgumentError,
        LatentsProcessorNotFoundError,
        LatentsProcessorError
    )

    from dgenerate.plugin import (
        ModuleFileNotFoundError,
        PluginNotFoundError,
        PluginArgumentError,
    )

    from dgenerate.devicecache import (
        clear_device_cache
    )

    from dgenerate.textprocessing import format_image_seed_uri

    import dgenerate.messages
    import dgenerate.types
    import dgenerate.files
    import dgenerate.translators

    import logging
    import dgenerate.extras.argostranslate.utils

    if os.environ.get('DGENERATE_BACKEND_WARNINGS', '0') == '0':
        transformers.logging.set_verbosity(transformers.logging.CRITICAL)
        diffusers.logging.set_verbosity(diffusers.logging.CRITICAL)
        dgenerate.extras.argostranslate.utils.logger.setLevel(logging.CRITICAL)
except KeyboardInterrupt:
    if __am_dgenerate_app:
        print('Exiting dgenerate due to keyboard interrupt!', file=sys.stderr)
    sys.exit(1)

_offline_mode = False


def is_offline_mode() -> bool:
    """
    Check if dgenerate is in global offline mode.

    :return: ``True`` if dgenerate is in offline mode, ``False`` otherwise.
    """
    global _offline_mode
    return _offline_mode


def enable_offline_mode():
    """
    Enable global offline mode for dgenerate.

    This will prevent any network requests from being made.
    """
    global _offline_mode
    _offline_mode = True
    dgenerate.hfhub.enable_offline_mode()
    dgenerate.webcache.enable_offline_mode()
    dgenerate.spacycache.enable_offline_mode()
    dgenerate.translators.enable_offline_mode()


def disable_offline_mode():
    """
    Disable offline mode for dgenerate.

    This will allow network requests to be made again.
    """
    global _offline_mode
    _offline_mode = False
    dgenerate.hfhub.disable_offline_mode()
    dgenerate.webcache.disable_offline_mode()
    dgenerate.spacycache.disable_offline_mode()
    dgenerate.translators.disable_offline_mode()


@contextlib.contextmanager
def offline_mode_context(enabled=True):
    """
    Context manager to temporarily enable or disable offline mode for dgenerate.

    :param enabled: If `True`, enables offline mode. If `False`, disables it.
    """
    global _offline_mode
    original_mode = _offline_mode

    if enabled:
        enable_offline_mode()
    else:
        disable_offline_mode()
    try:
        yield
    finally:
        if original_mode:
            enable_offline_mode()
        else:
            disable_offline_mode()

def _parse_set_args(set_args):
    """Parse --set or --setp meta arguments into (variable, value) pairs."""
    if not set_args:
        return []

    pairs = []
    for arg in set_args:
        if '=' not in arg:
            raise ValueError(f'Invalid argument: "{arg}". Must use variable=value syntax.')
        
        # Handle variable=value syntax (allow spaces around =)
        var, value = arg.split('=', 1)
        var = var.strip()
        value = value.strip()

        # Validate variable name (basic check)
        if not var:
            raise ValueError(f'Invalid argument: empty variable name in "{arg}"')

        pairs.append((var, value))

    return pairs


# Custom action to preserve order of --set and --setp meta arguments
class _OrderedSetAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs='+', **kwargs):
        # Use nargs='+' to ensure at least one argument is consumed
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if not hasattr(namespace, 'ordered_sets'):
            namespace.ordered_sets = []

        # Determine argument type from option_string
        arg_type = 'set' if option_string == '--set' else 'setp'
        namespace.ordered_sets.append((arg_type, values))


# Default content for init.dgen file
_DEFAULT_INIT_DGEN_CONTENT = inspect.cleandoc("""
    # dgenerate startup configuration
    # This file is executed automatically when dgenerate starts
    # Use it to set environment variables and other initialization

    # Example environment variable settings:

    # Cache directories
    # \\env DGENERATE_CACHE=/path/to/my/cache
    # \\env HF_HOME=/path/to/hf/cache

    # Authentication tokens
    # \\env HF_TOKEN=your_huggingface_token_here
    # \\env CIVIT_AI_TOKEN=your_civitai_token_here

    # Performance and behavior
    # \\env DGENERATE_TORCH_COMPILE=0
    # \\env DGENERATE_OFFLINE_MODE=1

    # Cache expiry (e.g. "days=7;hours=12" or "forever")
    # \\env DGENERATE_WEB_CACHE_EXPIRY_DELTA=days=7

    # Add your initialization commands below:

""") + '\n'


def _reset_hf_constants():
    """
    Reset HF_HOME-dependent constants in huggingface_hub, transformers, and diffusers
    after HF_HOME environment variable changes during init.dgen processing.

    This ensures that all libraries use the updated cache paths.
    """
    try:
        import sys

        # Reset huggingface_hub constants
        if 'huggingface_hub.constants' in sys.modules:
            import huggingface_hub.constants as hf_constants

            # Recalculate HF_HOME-dependent paths
            default_home = os.path.join(os.path.expanduser("~"), ".cache")
            hf_constants.HF_HOME = os.path.expandvars(
                os.path.expanduser(
                    os.getenv(
                        "HF_HOME",
                        os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "huggingface"),
                    )
                )
            )
            hf_constants.hf_cache_home = hf_constants.HF_HOME  # for backward compatibility

            default_cache_path = os.path.join(hf_constants.HF_HOME, "hub")
            default_assets_cache_path = os.path.join(hf_constants.HF_HOME, "assets")

            # Legacy env variables
            hf_constants.HUGGINGFACE_HUB_CACHE = os.getenv("HUGGINGFACE_HUB_CACHE", default_cache_path)
            hf_constants.HUGGINGFACE_ASSETS_CACHE = os.getenv("HUGGINGFACE_ASSETS_CACHE", default_assets_cache_path)

            # New env variables
            hf_constants.HF_HUB_CACHE = os.path.expandvars(
                os.path.expanduser(
                    os.getenv(
                        "HF_HUB_CACHE",
                        hf_constants.HUGGINGFACE_HUB_CACHE,
                    )
                )
            )
            hf_constants.HF_ASSETS_CACHE = os.path.expandvars(
                os.path.expanduser(
                    os.getenv(
                        "HF_ASSETS_CACHE",
                        hf_constants.HUGGINGFACE_ASSETS_CACHE,
                    )
                )
            )

            hf_constants.HF_TOKEN_PATH = os.path.expandvars(
                os.path.expanduser(
                    os.getenv(
                        "HF_TOKEN_PATH",
                        os.path.join(hf_constants.HF_HOME, "token"),
                    )
                )
            )
            hf_constants.HF_STORED_TOKENS_PATH = os.path.join(os.path.dirname(hf_constants.HF_TOKEN_PATH), "stored_tokens")

            # Reset XET cache path
            default_xet_cache_path = os.path.join(hf_constants.HF_HOME, "xet")
            hf_constants.HF_XET_CACHE = os.getenv("HF_XET_CACHE", default_xet_cache_path)

            dgenerate.messages.log(
                'Reset huggingface_hub constants after HF_HOME change',
                level=dgenerate.messages.DEBUG
            )

        # Reset transformers constants
        if 'transformers.utils.hub' in sys.modules:
            import transformers.utils.hub as tf_hub

            # Update transformers cache path
            tf_hub.default_cache_path = os.path.join(
                os.getenv("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")),
                "hub"
            )
            tf_hub.TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", tf_hub.default_cache_path)

            dgenerate.messages.log(
                'Reset transformers constants after HF_HOME change',
                level=dgenerate.messages.DEBUG
            )

        # Reset diffusers constants
        if 'diffusers.utils.constants' in sys.modules:
            import diffusers.utils.constants as df_constants

            # Re-import HF_HOME from huggingface_hub to get updated value
            from huggingface_hub.constants import HF_HOME
            df_constants.HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(HF_HOME, "modules"))

            dgenerate.messages.log(
                'Reset diffusers constants after HF_HOME change',
                level=dgenerate.messages.DEBUG
            )

    except Exception as e:
        # Don't fail if constant reset fails, just log a warning
        dgenerate.messages.log(
            f'Warning: Failed to reset HF constants: {str(e).strip()}',
            level=dgenerate.messages.WARNING
        )


def _run_init_dgen(runner):
    """
    Execute init.dgen config file from ~/.dgenerate/ if it exists.
    Creates a default init.dgen file if it doesn't exist.

    :param runner: ConfigRunner instance to execute the init config with
    """
    try:
        # Create ~/.dgenerate directory if it doesn't exist
        dgenerate_dir = pathlib.Path(pathlib.Path.home(), '.dgenerate')
        dgenerate_dir.mkdir(exist_ok=True)

        init_dgen_path = dgenerate_dir / 'init.dgen'

        # Create default init.dgen if it doesn't exist
        if not init_dgen_path.exists():
            try:
                with open(init_dgen_path, 'w', encoding='utf-8') as file:
                    file.write(_DEFAULT_INIT_DGEN_CONTENT)
                dgenerate.messages.log(f'Created default init config: {init_dgen_path}',
                                     level=dgenerate.messages.DEBUG)
            except Exception as e:
                dgenerate.messages.log(f'Error creating default init config: {str(e).strip()}',
                                     level=dgenerate.messages.DEBUG)

        if init_dgen_path.exists():
            try:
                with open(init_dgen_path, 'rt', encoding='utf-8') as file:
                    runner.run_file(file)
                    dgenerate.messages.log(f'Executed init config: {init_dgen_path}',
                                         level=dgenerate.messages.DEBUG)

                # Reset HF constants after processing init.dgen in case HF_HOME was changed
                _reset_hf_constants()

            except Exception as e:
                dgenerate.messages.log(f'Error executing init config {init_dgen_path}: {str(e).strip()}',
                                     level=dgenerate.messages.WARNING)
    except Exception as e:
        # Don't fail startup if init.dgen processing fails
        dgenerate.messages.log(f'Error processing init config: {str(e).strip()}',
                             level=dgenerate.messages.DEBUG)


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

    # Create a parser that knows about meta arguments to properly consume them
    meta_args_parser = argparse.ArgumentParser(prog='dgenerate', add_help=False)

    def _exit(status=0, message=None):
        if status == 0:
            pass
        dgenerate.messages.log(message.strip())
        sys.exit(1)

    meta_args_parser.exit = _exit
    meta_args_parser.print_usage = lambda x: None

    meta_input_group = meta_args_parser.add_mutually_exclusive_group()
    meta_input_group.add_argument('--shell', action='store_true')
    meta_input_group.add_argument('--file', nargs='+')

    meta_args_parser.add_argument('--no-stdin', action='store_true')

    meta_args_parser.add_argument('--set', action=_OrderedSetAction, metavar='VARIABLE=VALUE')
    meta_args_parser.add_argument('--setp', action=_OrderedSetAction, metavar='VARIABLE=VALUE')

    # Parse meta arguments first, ensuring they're completely consumed
    meta_args, args = meta_args_parser.parse_known_args(args)

    # Parse meta set/setp arguments in the order they appeared
    ordered_variable_ops = []
    try:
        if hasattr(meta_args, 'ordered_sets') and meta_args.ordered_sets:
            for arg_type, values in meta_args.ordered_sets:
                pairs = _parse_set_args(values)
                for var, value in pairs:
                    ordered_variable_ops.append((arg_type, var, value))
    except ValueError as e:
        dgenerate.messages.log(f'dgenerate: error: {str(e).strip()}',
                               level=dgenerate.messages.ERROR)
        sys.exit(1)

    if '-ofm' in args or '--offline-mode' in args:
        enable_offline_mode()

    try:
        render_loop = RenderLoop()
        render_loop.config = DgenerateArguments()

        # ^ this is necessary for --templates-help to
        # render all the correct values
        if meta_args.file:
            runner = ConfigRunner(render_loop=render_loop,
                                  version=__version__,
                                  injected_args=args)

            # Execute init.dgen if it exists
            _run_init_dgen(runner)

            # Apply --set and --setp meta arguments directly to the runner in order
            try:
                for arg_type, var, value in ordered_variable_ops:
                    if arg_type == 'set':
                        runner.user_set(var, value)
                    else:  # setp
                        runner.user_setp(var, value)
            except Exception as e:
                dgenerate.messages.log(f'dgenerate: error: {str(e).strip()}',
                                       level=dgenerate.messages.ERROR)
                sys.exit(1)

            input_files = itertools.chain.from_iterable(
                [glob.glob(input_file) if '*' in input_file else [input_file] for input_file in meta_args.file])

            for input_file in input_files:
                try:
                    with open(input_file, 'rt', encoding='utf-8') as file:
                        runner.run_file(file)
                except (ModuleFileNotFoundError, FileNotFoundError) as e:
                    dgenerate.messages.log(f'dgenerate: error: {str(e).strip()}',
                                           level=dgenerate.messages.ERROR)
                    sys.exit(1)

                except BatchProcessError as e:
                    dgenerate.messages.log(f'Config Error: {str(e).strip()}',
                                           level=dgenerate.messages.ERROR)
                    sys.exit(1)

        elif sys.stdin is not None and (
                not dgenerate.files.stdin_is_tty() or meta_args.shell) and not meta_args.no_stdin:
            # Not a terminal, batch process STDIN
            runner = ConfigRunner(render_loop=render_loop,
                                  version=__version__,
                                  injected_args=args)

            # Execute init.dgen if it exists
            _run_init_dgen(runner)

            # Apply --set and --setp meta arguments directly to the runner in order
            try:
                for argType, var, value in ordered_variable_ops:
                    if argType == 'set':
                        runner.user_set(var, value)
                    else:  # setp
                        runner.user_setp(var, value)
            except Exception as e:
                dgenerate.messages.log(f'dgenerate: error: {str(e).strip()}',
                                       level=dgenerate.messages.ERROR)
                sys.exit(1)

            while True:
                try:
                    runner.run_file(sys.stdin)
                    if not meta_args.shell:
                        sys.exit(0)
                except ModuleFileNotFoundError as e:
                    # missing plugin file parsed by ConfigRunner out of injected args
                    dgenerate.messages.log(f'dgenerate: error: {str(e).strip()}',
                                           level=dgenerate.messages.ERROR)
                    if not meta_args.shell:
                        sys.exit(1)

                except BatchProcessError as e:
                    dgenerate.messages.log(f'Config Error: {str(e).strip()}',
                                           level=dgenerate.messages.ERROR)
                    if not meta_args.shell:
                        sys.exit(1)
        else:
            # CLI usage - create a temporary ConfigRunner just to execute init.dgen
            init_runner = ConfigRunner(
                render_loop=render_loop,
                version=__version__,
                injected_args=args
            )
            
            # Execute init.dgen if it exists
            _run_init_dgen(init_runner)
            
            sys.exit(invoke_dgenerate(args, render_loop=render_loop))
    except KeyboardInterrupt:
        print('Exiting dgenerate due to keyboard interrupt!', file=sys.stderr)
        sys.exit(1)


__all__ = dgenerate.types.module_all()
