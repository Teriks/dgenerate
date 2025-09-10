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
import gc
import glob
import importlib
import inspect
import os
import pathlib
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
import types
import typing

import dgenerate
import dgenerate.arguments as _arguments
import dgenerate.batchprocess.batchprocessor as _batchprocessor
import dgenerate.batchprocess.configrunnerbuiltins as _configrunnerbuiltins
import dgenerate.batchprocess.configrunnerpluginloader as _configrunnerpluginloader
import dgenerate.batchprocess.util as _util
import dgenerate.devicecache as _devicecache
import dgenerate.files as _files
import dgenerate.invoker as _invoker
import dgenerate.memoize as _memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.plugin as _plugin
import dgenerate.renderloop as _renderloop
import dgenerate.textprocessing as _textprocessing
import dgenerate.torchutil as _torchutil
import dgenerate.types as _types


class ConfigRunner(_batchprocessor.BatchProcessor):
    """
    A :py:class:`.BatchProcessor` that can run dgenerate batch processing configs from a string or file.
    """

    @property
    def plugin_module_paths(self) -> frozenset[str]:
        """
        Set of plugin module paths if they were injected into the config runner by ``--plugin-modules``
        or used in a ``\\import_plugins`` statement in a config.

        :return: a set of paths, may be empty but not ``None``
        """
        return frozenset(self._plugin_module_paths)

    @property
    def local_files_only(self) -> bool:
        """
        Is this config runner only going to look for resources such as models in cache / on disk?

        This will be ``True`` if ``-ofm/--offline-mode`` was parsed from ``injected_args``
        """
        return self._local_files_only

    def __init__(self,
                 injected_args: collections.abc.Sequence[str] | None = None,
                 render_loop: _renderloop.RenderLoop | None = None,
                 plugin_loader: _configrunnerpluginloader.ConfigRunnerPluginLoader = None,
                 version: _types.Version | str = dgenerate.__version__,
                 throw: bool = False):
        """

        :raises dgenerate.plugin.ModuleFileNotFoundError: If a module path parsed from
            ``--plugin-modules`` in ``injected_args`` could not be found on disk.


        :param injected_args: dgenerate command line arguments in the form of a list, see: shlex module, or sys.argv.
            These arguments will be injected at the end of every dgenerate invocation in the config. ``--plugin-modules``
            are parsed from ``injected_args`` and added to ``plugin_loader``. If ``-v/--verbose`` is present in ``injected_args``
            debugging output will be enabled globally while the config runs, and not just for invocations. Passing ``-v/--verbose``
            also enables printing stack traces for all unhandled directive exceptions to ``stderr``. If ``-ofm/--offline-mode``
            is present in ``injected_args``, plugins will be instructed to only look for resources such as models in cache / on disk
            and never attempt to download from the internet.

        :param render_loop: RenderLoop instance, if ``None`` is provided one will be created.
        :param plugin_loader: Batch processor plugin loader, if one is not provided one will be created.
        :param version: Config version for ``#! dgenerate x.x.x`` version checks, defaults to ``dgenerate.__version__``
        :param throw: Whether to throw exceptions from :py:func:`dgenerate.invoker.invoke_dgenerate` or handle them.
            If you set this to ``True`` exceptions will propagate out of dgenerate invocations instead of a
            :py:exc:`dgenerate.batchprocess.BatchProcessError` being raised by the created
            :py:class:`dgenerate.batchprocess.BatchProcessor`. A line number where the error occurred can be
            obtained using :py:attr:`dgenerate.batchprocess.BatchProcessor.current_line`.
        """

        def invoker(command_line, parsed_args):
            header = 'Processing Arguments:\n\n'
            args_wrapped = \
                _textprocessing.wrap(
                    command_line,
                    width=_textprocessing.long_text_wrap_width()) + '\n'

            _messages.log(header + args_wrapped, underline=True)

            try:
                return_code = \
                    _invoker.invoke_dgenerate(parsed_args,
                                              render_loop=self.render_loop,
                                              throw=throw)
                if return_code == 0:
                    self.template_variables.update(self._generate_template_variables())
                return return_code
            finally:
                self.render_loop.model_extra_modules = None

        super().__init__(
            invoker=invoker,
            name='dgenerate',
            version=version,
            injected_args=injected_args if injected_args else [])

        if render_loop is None:
            render_loop = _renderloop.RenderLoop()

        self._plugin_module_paths = set()

        injected_plugin_modules = []

        if injected_args:
            injected_plugin_modules = _arguments.parse_plugin_modules(injected_args)[0]
            if injected_plugin_modules:
                self._plugin_module_paths.update(injected_plugin_modules)
                _plugin.import_plugins(self._plugin_module_paths)
            else:
                injected_plugin_modules = []

        # Parse local_files_only from injected_args
        if injected_args:
            self._local_files_only = _arguments.parse_offline_mode(injected_args)[0]
        else:
            self._local_files_only = False

        self.render_loop = render_loop

        self.template_variables = {
            'injected_args': self.injected_args,
            'injected_device': _arguments.parse_device(self.injected_args)[0],
            'injected_verbose': _arguments.parse_verbose(self.injected_args)[0],
            'injected_plugin_modules': injected_plugin_modules,
            'saved_modules': dict(),
            'glob': glob,
            'os': os,
        }

        self.template_variables = self._generate_template_variables()

        self.reserved_template_variables = set(self.template_variables.keys())

        self.template_functions = {
            'import_module': _configrunnerbuiltins.import_module,
            'unquote': _configrunnerbuiltins.unquote,
            'quote': _configrunnerbuiltins.quote,
            'format_prompt': _configrunnerbuiltins.format_prompt,
            'format_size': _configrunnerbuiltins.format_size,
            'align_size': _configrunnerbuiltins.align_size,
            'pow2_size': _configrunnerbuiltins.pow2_size,
            'image_size': _configrunnerbuiltins.image_size,
            'size_is_aligned': _configrunnerbuiltins.size_is_aligned,
            'size_is_pow2': _configrunnerbuiltins.size_is_pow2,
            'format_model_type': _configrunnerbuiltins.format_model_type,
            'format_dtype': _configrunnerbuiltins.format_dtype,
            'last': _configrunnerbuiltins.last,
            'first': _configrunnerbuiltins.first,
            'gen_seeds': _configrunnerbuiltins.gen_seeds,
            'cwd': _configrunnerbuiltins.cwd,
            'download': _configrunnerbuiltins.download,
            'have_feature': _configrunnerbuiltins.have_feature,
            'platform': _configrunnerbuiltins.platform,
            'frange': _configrunnerbuiltins.frange,
            'have_cuda': _configrunnerbuiltins.have_cuda,
            'have_xpu': _configrunnerbuiltins.have_xpu,
            'have_mps': _configrunnerbuiltins.have_mps,
            'total_memory': _configrunnerbuiltins.total_memory,
            'default_device': _configrunnerbuiltins.default_device,
            'csv': _configrunnerbuiltins.csv
        }

        self.directives = {
            'help': self._help_directive,
            'templates_help': self._templates_help_directive,
            'directives_help': self._directives_help_directive,
            'functions_help': self._functions_help_directive,
            'quantizer_help': self._quantizer_help,
            'image_processor_help': self._image_processor_help_directive,
            'latents_processor_help': self._latents_processor_help_directive,
            'prompt_weighter_help': self._prompt_weighter_help_directive,
            'prompt_upscaler_help': self._prompt_upscaler_help_directive,
            'clear_object_cache': self._clear_object_cache,
            'list_object_caches': self._list_object_caches,
            'clear_device_cache': self._clear_device_cache,
            # 'gc_dump': self._gc_dump,
            'save_modules': self._save_modules_directive,
            'use_modules': self._use_modules_directive,
            'clear_modules': self._clear_modules_directive,
            'gen_seeds': self._gen_seeds_directive,
            'download': self._download_directive,
            'pwd': self._pwd_directive,
            'ls': self._ls_directive,
            'cd': self._cd_directive,
            'pushd': self._pushd_directive,
            'popd': self._popd_directive,
            'exec': self._exec_directive,
            'mv': self._mv_directive,
            'cp': self._cp_directive,
            'mkdir': self._mkdir_directive,
            'rmdir': self._rmdir_directive,
            'rm': self._rm_directive,
            'exit': self._exit_directive,
            'import': self._import_directive
        }

        self.plugin_loader = \
            _configrunnerpluginloader.ConfigRunnerPluginLoader() if \
                plugin_loader is None else plugin_loader

        for plugin_class in self.plugin_loader.get_available_classes():
            self.plugin_loader.load(plugin_class.get_names()[0],
                                    config_runner=self,
                                    render_loop=self.render_loop,
                                    local_files_only=self._local_files_only)

        self.directives['import_plugins'] = self._import_plugins_directive

        self._directory_stack = []

    def _import_plugins_directive(self, plugin_paths: collections.abc.Sequence[str]):
        """
        Imports plugins from within a config, this imports config plugins as well as image processor plugins.
        This has an identical effect to the --plugin-modules argument. You may specify multiple plugin
        module directories or python files containing plugin implementations, you may also specify
        modules that are installed in the python environment by name.
        """

        if len(plugin_paths) == 0:
            raise _batchprocessor.BatchProcessError(
                '\\import_plugins must be used with at least one argument.')

        self._plugin_module_paths.update(plugin_paths)

        _plugin.import_plugins(self._plugin_module_paths)

        new_classes = self.plugin_loader.load_plugin_modules(plugin_paths)
        for cls in new_classes:
            self.plugin_loader.load(cls.get_names()[0],
                                    config_runner=self,
                                    render_loop=self.render_loop,
                                    local_files_only=self._local_files_only)

        return 0

    def _exit_directive(self, args: collections.abc.Sequence[str]):
        """
        Causes the dgenerate process to exit with a specific return code.
        This directive accepts one argument, the return code, which is optional
        and 0 by default. It must be an integer value.
        """
        if (len(args)) == 0:
            sys.exit(0)

        try:
            return_code = int(args[0])
        except ValueError as e:
            raise _batchprocessor.BatchProcessError(
                f'\\exit return code must be an integer value, received: {args[0]}') from e

        sys.exit(return_code)

    @staticmethod
    def _clear_device_cache(args: collections.abc.Sequence[str]):
        """
        Clear any objects cached by dgenerate that may actively exist on an accelerator device.

        You must specify the device or devices as arguments.
        """

        if len(args) < 1:
            raise _batchprocessor.BatchProcessError(
                f'\\clear_device_cache must receive at least one argument, for example: cuda:0')

        for arg in args:
            if not _torchutil.is_valid_device_string(arg):
                raise _batchprocessor.BatchProcessError(
                    f'\\clear_device_cache invalid device, {_torchutil.invalid_device_message(arg, cap=False)}')

        for arg in set(args):
            _devicecache.clear_device_cache(arg)

        _memory.torch_gc()

        return 0

    # @staticmethod
    # def _gc_dump(args: collections.abc.Sequence[str]):
    #     """
    #     :param args:
    #     :return:
    #     """
    #     import objgraph
    #     objects = gc.get_objects()
    #     for arg in args:
    #         for object in objects:
    #             if arg in object.__class__.__name__:
    #                 objgraph.show_backrefs([object], max_depth=10, filename=f'gc_dump_{arg}.png')
    #
    @staticmethod
    def _clear_object_cache(args: collections.abc.Sequence[str]):
        """
        Clear a specific dgenerate object cache, such as: unet, vae, text_encoder, etc.

        Calling with no arguments clears all object caches.

        See: \\list_object_caches, for a list of valid object cache names.
        """

        if len(args) == 0:
            _memoize.clear_object_caches()
        else:

            valid_names = _memoize.get_object_cache_names()
            for arg in args:
                if arg not in valid_names:
                    raise _batchprocessor.BatchProcessError(
                        f'\\clear_object_cache, object cache "{arg}" does not exist.')

            for arg in args:
                _memoize.get_object_cache(arg).clear(collect=False)

            gc.collect()

        return 0

    @staticmethod
    def _list_object_caches(args: collections.abc.Sequence[str]):
        """
        List object cache names (and memory footprint if applicable) that may be cleared with \\clear_object_cache.
        """

        _messages.log('Object caches:\n')

        for object_cache in _memoize.get_object_cache_names():
            bin = _memoize.get_object_cache(object_cache)
            if isinstance(bin, _memory.SizedConstrainedObjectCache):
                _messages.log(
                    ' ' * 4 + '"' + object_cache +
                    f'": {len(bin)} objects, cpu side RAM - {_memory.bytes_best_human_unit(bin.size)}'
                )
            else:
                _messages.log(' ' * 4 + '"' + object_cache + f'": {len(bin)} objects')

        return 0

    def _save_modules_directive(self, args: collections.abc.Sequence[str]):
        """
        Save a set of pipeline modules off the last diffusers pipeline used for the
        main model of a dgenerate invocation. The first argument is a variable name
        that the modules will be saved to, which can be reference later with \\use_modules.
        The rest of the arguments are names of pipeline modules that you want to save to this
        variable as a set of modules that are kept together, usable names are: unet, vae,
        transformer, text_encoder, text_encoder_2, text_encoder_3, tokenizer, tokenizer_2,
        tokenizer_3, safety_checker, feature_extractor, controlnet, scheduler, unet
        """
        saved_modules = self.template_variables.get('saved_modules')

        if len(args) < 2:
            raise _batchprocessor.BatchProcessError(
                '\\save_modules directive must have at least 2 arguments, '
                'a variable name and one or more module names.')

        if self.render_loop.pipeline_wrapper is None:
            raise _batchprocessor.BatchProcessError(
                '\\save_modules directive cannot be used until a '
                'dgenerate invocation has occurred.')

        name = args[0]

        if not name.isidentifier():
            raise _batchprocessor.BatchProcessError(
                f'Cannot save modules to "{name}" on line {self.current_line}, '
                f'invalid identifier/name token, must be a valid python variable name / identifier.')

        creation_result = self.render_loop.pipeline_wrapper.recall_main_pipeline()
        saved_modules[name] = creation_result.get_pipeline_modules(args[1:])
        return 0

    def _use_modules_directive(self, args: collections.abc.Sequence[str]):
        """
        Use a set of pipeline modules saved with \\save_modules, accepts one argument,
        the name that set of modules was saved to.
        """
        saved_modules = self.template_variables.get('saved_modules')

        if not saved_modules:
            raise _batchprocessor.BatchProcessError(
                '\\use_modules error, no modules are currently saved that can be referenced.')

        if len(args) != 1:
            raise _batchprocessor.BatchProcessError(
                '\\use_modules accepts one argument and one argument only, '
                'the name that the modules were previously saved to with \\save_modules'
            )

        saved_name = args[0]

        if not saved_name.isidentifier():
            raise _batchprocessor.BatchProcessError(
                f'Cannot use modules from "{saved_name}" on line {self.current_line}, '
                f'invalid identifier/name token, must be a valid python variable name / identifier.')

        if saved_name not in saved_modules:
            raise _batchprocessor.BatchProcessError(
                f'Cannot use modules from "{saved_name}" on line {self.current_line}, '
                f'there are no modules saved to this variable name.')

        self.render_loop.model_extra_modules = saved_modules[saved_name]
        return 0

    def _clear_modules_directive(self, args: collections.abc.Sequence[str]):
        """
        Clears a named set of pipeline modules saved with \\save_modules, accepts one argument, the name
        that the set of modules was saved to. When no argument is provided, all modules ever
        saved are cleared.
        """
        saved_modules = self.template_variables.get('saved_modules')

        if len(args) > 0:
            for arg in args:
                if not arg.isidentifier():
                    raise _batchprocessor.BatchProcessError(
                        f'Cannot clear modules from "{arg}" on line {self.current_line}, '
                        f'invalid identifier/name token, must be a valid python variable name / identifier.')

            for arg in args:
                try:
                    del saved_modules[arg]
                except KeyError as e:
                    raise _batchprocessor.BatchProcessError(
                        f'No pipeline modules were saved to the variable name "{arg}", '
                        f'that name could not be found.') from e
        else:
            saved_modules.clear()
        return 0

    def _gen_seeds_directive(self, args: collections.abc.Sequence[str]):
        """
        Generate N random integer seeds (as strings) and store them as a list to a template variable name.

        The first argument is the variable name, the second argument is the number of seeds to generate.
        """
        if len(args) == 2:
            name = args[0]

            if not name.isidentifier():
                raise _batchprocessor.BatchProcessError(
                    f'Cannot generate seeds into "{name}" on line {self.current_line}, '
                    f'invalid identifier/name token, must be a valid python variable name / identifier.')

            try:
                self.template_variables[name] = \
                    [str(s) for s in _renderloop.gen_seeds(int(args[1]))]
            except ValueError as e:
                raise _batchprocessor.BatchProcessError(
                    'The second argument of \\gen_seeds must be an integer value.') from e
        else:
            raise _batchprocessor.BatchProcessError(
                '\\gen_seeds directive takes 2 arguments, template variable '
                'name (to store value at), and number of seeds to generate.')
        return 0

    def _pwd_directive(self, args: collections.abc.Sequence[str]):
        """
        Print the current working directory.
        """
        _messages.log(os.getcwd())
        return 0

    def _mv_directive(self, args: collections.abc.Sequence[str]):
        """
        Move a file or directory to a new location on disk.

        First argument is the source, second argument is the destination.
        """
        if len(args) != 2:
            raise _batchprocessor.BatchProcessError(
                '\\mv directive takes two arguments, source and destination.')
        shutil.move(args[0], args[1])
        return 0

    def _cp_directive(self, args: collections.abc.Sequence[str]):
        """
        Copy a file or directory to a new location on disk.

        First argument is the source, second argument is the destination.
        """
        if len(args) != 2:
            raise _batchprocessor.BatchProcessError(
                '\\cp directive takes two arguments, source and destination.')
        shutil.copy2(args[0], args[1])
        return 0

    def _mkdir_directive(self, command_line_args: collections.abc.Sequence[str]):
        """
        Create a directory or directory tree.

        Basic implementation of the Unix 'mkdir' command.

        Supports the -p / --parents argument.

        See: \\mkdir --help
        """
        parser = _util.DirectiveArgumentParser(
            prog='\\mkdir', description='Create directories.')

        parser.add_argument('-p', '--parents',
                            action='store_true',
                            help='No error if existing, make parent directories as needed.')

        parser.add_argument('directories',
                            nargs='+',
                            help='List of directories to create')

        args = parser.parse_args(command_line_args)

        if parser.return_code is not None:
            return parser.return_code

        for directory in args.directories:
            try:
                if args.parents:
                    os.makedirs(directory, exist_ok=True)
                else:
                    os.mkdir(directory)
            except OSError as e:
                raise _batchprocessor.BatchProcessError(f'Error creating directory "{directory}": {e.strerror}') from e
        return 0

    def _download_directive(self, command_line_args: collections.abc.Sequence[str]):
        """
        Download a file from a URL to the web cache or a specified path,
        and assign the file path to a template variable.

        NOWRAP!
        \\download my_variable https://modelhost.com/model.safetensors

        NOWRAP!
        \\download my_variable https://modelhost.com/model.safetensors -o model.safetensors

        NOWRAP!
        \\download my_variable https://modelhost.com/model.safetensors -o directory/

        When an --output path is specified, if the file already exists it will
        be reused by default (simple caching behavior), this can be disabled with
        -x/--overwrite indicating that the file should always be downloaded.

        -x/--overwrite can also be used to overwrite cached
        files in the dgenerate web cache.

        An error will be raised by default if a text mimetype is encountered,
        this can be overridden with -t/--text

        Be weary that if you have a long-running loop in your config using
        a top level jinja template, which refers to your template variable,
        cache expiry may invalidate the file stored in your variable.

        You can rectify this by putting the download directive inside of
        your processing loop so that the file is simply re-downloaded.

        Or you may be better off using the download
        template function which provides this functionality as a template function.

        For usage see: \\download --help
        """
        parser = _util.DirectiveArgumentParser(
            prog='\\download', description='Download a file.')

        parser.add_argument('variable',
                            help='Assign the path of the downloaded '
                                 'file to this template variable name.')
        parser.add_argument('url', help='URL of the file to download.')
        parser.add_argument('-o', '--output', default=None, metavar='PATH',
                            help="Path to download the file to. "
                                 "If none is provided the file is placed in dgenerate's "
                                 "web cache. If this path ends with a forward slash or "
                                 "backslash it is considered to be a directory, the file "
                                 "name will be determined by the URL or content disposition "
                                 "of the http/https request if available.")
        parser.add_argument('-x', '--overwrite',
                            action='store_true',
                            default=False,
                            help='Always overwrite existing files instead of reusing them.')
        parser.add_argument('-t', '--text',
                            action='store_true',
                            default=False,
                            help='Allow for downloading text/* mimetypes without raising an error. '
                                 'This is not typically what this directive should be used for. '
                                 'It should be used for binary files such as images and models. '
                                 'By default it will error when it encounters a text mimetype because '
                                 'that likely indicates you have hit a login page while attempting '
                                 'to download a model or an image.')

        args = parser.parse_args(command_line_args)

        if parser.return_code is not None:
            return parser.return_code

        self.user_define_check(args.variable)

        try:
            file_path = _configrunnerbuiltins.download(args.url, args.output, args.overwrite, args.text)
        except _batchprocessor.BatchProcessError as e:
            _messages.error(str(e))
            return 1

        self.template_variables[args.variable] = file_path

        return 0

    def _exec_directive(self, args: collections.abc.Sequence[str]):
        """
        Execute a shell command line as a new process or processes.

        The pipe | operator is supported for piping to standard input, as well as bash file redirection syntax.

        The scripts / bin directory of dgenerate's python environment is prioritized in the PATH
        for processes launched by this directive only, so ``python`` can be used to run python scripts
        using the environment dgenerate is installed into. You can also use tools such as
        ``accelerate``. This PATH modification only applies to \\exec commands, not globally.

        The following redirection operators are supported:

            NOWRAP!
            '<'   : Read into stdin
            '>'   : stdout to file
            '1>'  : stdout to file
            '2>'  : stderr to file
            '&>'  : stdout & stderr to file
            '>>'  : append stdout to file
            '1>>' : append stdout to file
            '2>>' : append stderr to file
            '&>>' : append stout & stderr to file
            '2>&1': redirect stderr to stdout
            '1>&2': redirect stdout to stderr

        Examples:

            NOWRAP!
            \\exec dgenerate < my_config.dgen &> log.txt
            \\exec dgenerate < my_config.dgen > log.txt 2>&1
            \\exec dgenerate < my_config.dgen > stdout.txt 2> stderr.txt

        Using python to access dgenerate's Python environment:

            \\exec python -c "import torch; print(torch.__version__)"
            \\exec python my_script.py
            \\exec accelerate launch my_training_script.py

        Windows cat pipe:

            \\exec cmd /c "type my_config.dgen" | dgenerate &> test.log

        Linux cat pipe:

            \\exec cat my_config.dgen | dgenerate &> test.log
        """

        if len(args) == 0:
            raise _batchprocessor.BatchProcessError(
                '\\exec directive must be passed at least one argument.')

        _messages.log(self.executing_text,
                      underline=True)

        args = list(args)

        open_files = []
        open_processes = []

        process = None

        try:
            stdin = None
            if '<' in args:
                index = args.index('<')
                if index + 1 < len(args):
                    stdin = open(args[index + 1], 'r')
                    open_files.append(stdin)
                    args = args[:index] + args[index + 2:]
                else:
                    raise _batchprocessor.BatchProcessError(
                        'No input file specified for redirection.')

            if '|' in args:
                commands = []
                current_command = []
                for arg in args:
                    if arg == '|':
                        commands.append(current_command)
                        current_command = []
                    else:
                        current_command.append(arg)
                commands.append(current_command)
            else:
                commands = [args]

            previous_process = None

            def stdout_handler(line):
                _messages.get_message_file().buffer.write(line)
                _messages.get_message_file().flush()

            def stderr_handler(line):
                _messages.get_error_file().buffer.write(line)
                _messages.get_error_file().flush()

            # Set up environment with prioritized PATH before processing any commands
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['PYTHONIOENCODING'] = 'utf-8'

            # Prioritize dgenerate's Python environment Scripts directory in PATH
            # This allows users to use 'python' and other tools from dgenerate's environment
            scripts_dirs = []

            # First, check if we're in a virtual environment
            venv_path = os.environ.get('VIRTUAL_ENV')
            if venv_path:
                # We're in a virtual environment, use its Scripts/bin directory
                venv_scripts = os.path.join(venv_path, 'Scripts')  # Windows
                venv_bin = os.path.join(venv_path, 'bin')  # Unix

                if os.path.isdir(venv_scripts):
                    scripts_dirs.append(venv_scripts)
                if os.path.isdir(venv_bin):
                    scripts_dirs.append(venv_bin)

            # Also check the directory containing sys.executable
            python_dir = os.path.dirname(sys.executable)

            # Check for Scripts directory (Windows)
            scripts_path = os.path.join(python_dir, 'Scripts')
            if os.path.isdir(scripts_path) and scripts_path not in scripts_dirs:
                scripts_dirs.append(scripts_path)

            # Check for bin directory (Unix)
            bin_path = os.path.join(python_dir, 'bin')
            if os.path.isdir(bin_path) and bin_path not in scripts_dirs:
                scripts_dirs.append(bin_path)

            # Always include the directory containing the Python executable
            if python_dir not in scripts_dirs:
                scripts_dirs.append(python_dir)

            # Prepend all relevant directories to PATH in subprocess environment
            if 'PATH' in env:
                env['PATH'] = os.pathsep.join(scripts_dirs) + os.pathsep + env['PATH']
            else:
                env['PATH'] = os.pathsep.join(scripts_dirs)

            for command in commands:
                if not command:
                    raise _batchprocessor.BatchProcessError(
                        f'no command specified to pipe to / from.')

                redirects = {'>', '1>', '2>', '&>', '>>', '1>>', '2>>', '&>>', '2>&1', '1>&2'}
                _i = 0
                while any(i in command for i in redirects):
                    if command[_i] in redirects:
                        remove_cnt = 1
                        mode = 'ab' if '>>' in command[_i] else 'wb'
                        if command[_i] == '2>&1':
                            stderr_handler = stdout_handler
                        elif command[_i] == '1>&2':
                            stdout_handler = stderr_handler
                        else:
                            remove_cnt = 2
                            try:
                                file = open(command[_i + 1], mode)
                                open_files.append(file)
                            except IndexError as e:
                                raise _batchprocessor.BatchProcessError(
                                    f'{command[_i]} no output file specified.') from e
                            if command[_i][0] != '2':
                                def stdout_handler(line):
                                    file.write(line)
                                    file.flush()
                            if command[_i][0] != '1':
                                def stderr_handler(line):
                                    file.write(line)
                                    file.flush()
                        command = command[:_i] + command[_i + remove_cnt:]
                        _i -= remove_cnt
                    _i += 1

                stdin = stdin if previous_process is None else previous_process.stdout

                if command[0] == 'dgenerate' and stdin is None and '--shell' not in command:
                    command = list(command) + ['--no-stdin']

                # Resolve the executable path using the modified PATH
                resolved_command = command.copy()
                resolved_executable = shutil.which(command[0], path=env['PATH'])
                if resolved_executable:
                    resolved_command[0] = resolved_executable

                try:
                    executable = \
                        os.path.splitext(
                            os.path.basename(os.path.realpath(sys.argv[0])))[0]

                    # This is a ridiculous hack for the windowed variant of the
                    # dgenerate executable on windows that allows proper behavior
                    # for sub-shells executed in a config from a terminal using
                    # the standard dgenerate executable compiled for console mode
                    # as well as the executable compiled for windowed mode

                    if platform.system() == 'Windows' and executable == 'dgenerate_windowed':
                        extra_kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW}
                    else:
                        extra_kwargs = dict()

                    process = subprocess.Popen(resolved_command,
                                               stdin=sys.stdin if stdin is None else stdin,
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE,
                                               env=env,
                                               **extra_kwargs)

                    open_processes.append(process)
                except FileNotFoundError as e:
                    raise _batchprocessor.BatchProcessError(
                        f'Command "{command[0]}" not found on system.') from e

                previous_process = process

            stop_threads = threading.Event()

            def handle_stream(stream, handler):
                line_reader = _files.TerminalLineReader(stream)
                line = True
                while line:
                    line = line_reader.readline()
                    if stop_threads.is_set():
                        break
                    if line is not None:
                        handler(line)

            thread1 = threading.Thread(
                target=handle_stream,
                args=(process.stdout, stdout_handler))
            thread1.daemon = True
            thread1.start()

            thread2 = threading.Thread(
                target=handle_stream,
                args=(process.stderr, stderr_handler))
            thread2.daemon = True
            thread2.start()

            return_code = process.wait()

            stop_threads.set()
            thread1.join()
            thread2.join()

        finally:

            for f in open_files:
                f.close()

            for p in open_processes:
                if p.poll() is None:
                    p.terminate()
                    return_code = p.wait()

        return return_code

    def _ls_directive(self, args: collections.abc.Sequence[str]):
        """
        List directory contents.

        Basic implementation of the Unix 'ls' command, accepts the argument -l and -a

        Also accepts -la or -al
        """

        parser = _util.DirectiveArgumentParser(
            prog='\\ls',
            description='List directory contents.')

        parser.add_argument('paths', metavar='PATH', type=str, nargs='*', default=['.'],
                            help='the path(s) to list')
        parser.add_argument('-l', action='store_true',
                            help='use a long listing format')
        parser.add_argument('-a', '--all', action='store_true',
                            help='do not ignore entries starting with .')

        args = parser.parse_args(args)

        if parser.return_code is not None:
            return parser.return_code

        _messages.log(self.executing_text,
                      underline=True)

        paths = args.paths
        long_opt = args.l
        all_opt = args.all

        directories = []
        files = []

        for path in map(pathlib.Path, paths):
            if not path.exists():
                _messages.log(f'\\ls: {path}: No such file or directory')
                continue

            if path.is_file():
                files.append(path)
            if path.is_dir():
                directories.append(path)

        if long_opt and files:
            max_size_length = max(len(str(file.stat().st_size)) for file in files)

        for path in files:
            if long_opt:
                file_stat = path.stat()
                file_permissions = stat.filemode(file_stat.st_mode)
                num_links = file_stat.st_nlink
                file_size = file_stat.st_size
                mod_time = time.ctime(file_stat.st_mtime)
                _messages.log(
                    f'{file_permissions:<11} {num_links:<3} '
                    f'{file_size:<{max_size_length}} {mod_time:<25} {path.name}')
            else:
                _messages.log(f'{path.name}')

        if files and directories:
            _messages.log()

        for idx, path in enumerate(directories):
            if len(directories) > 1 or len(files) > 0:
                _messages.log(f'{path}:')
            if long_opt:
                lens = [len(str(p.stat().st_size))
                        for p in path.iterdir() if all_opt or not
                        p.name.startswith('.')]
                max_size_length = max(lens) if lens else 0

            for sub_path in sorted(path.iterdir()):
                if not all_opt and sub_path.name.startswith('.'):
                    continue
                if long_opt:
                    file_stat = sub_path.stat()
                    file_permissions = stat.filemode(file_stat.st_mode)
                    num_links = file_stat.st_nlink
                    file_size = file_stat.st_size
                    mod_time = time.ctime(file_stat.st_mtime)
                    _messages.log(
                        f'{file_permissions:<11} {num_links:<3} '
                        f'{file_size:<{max_size_length}} {mod_time:<25} '
                        f'{sub_path.name}{"/" if sub_path.is_dir() else ""}')
                else:
                    _messages.log(f'{sub_path.name}{"/" if sub_path.is_dir() else ""}')

            if len(directories) > 1 and idx < len(directories) - 1:
                _messages.log()

        return 0

    def _cd_directive(self, args: collections.abc.Sequence[str]):
        """
        Change the current working directory.

        Takes one argument, the directory to change to.
        """
        if len(args) == 1:
            try:
                path = os.path.abspath(args[0])
                os.chdir(path)
                _messages.log(f'Working Directory Changed To: "{path}"')
            except OSError as e:
                raise _batchprocessor.BatchProcessError(e) from e
        else:
            raise _batchprocessor.BatchProcessError(
                '\\cd directive takes 1 argument, the directory name.')
        return 0

    def _pushd_directive(self, args: collections.abc.Sequence[str]):
        """
        Push the current working directory on to the directory stack and change to the specified directory.

        Takes one argument, the directory to change to.
        """
        if len(args) == 1:
            try:
                path = os.path.abspath(args[0])
                old_dir = os.getcwd()
                os.chdir(path)
                _messages.log(f'Working Directory Changed To: "{path}"')
                self._directory_stack.append(old_dir)
            except OSError as e:
                raise _batchprocessor.BatchProcessError(e) from e
        else:
            raise _batchprocessor.BatchProcessError(
                '\\pushd directive takes 1 argument, the directory name.')
        return 0

    def _popd_directive(self, args: collections.abc.Sequence[str]):
        """
        Pop the last directory of the directory stack and change to that directory.
        """
        try:
            directory = self._directory_stack.pop()
            os.chdir(directory)
            _messages.log(f'Working Directory Changed To: "{directory}"')
        except IndexError as e:
            raise _batchprocessor.BatchProcessError('\\popd failed, no directory on the stack.') from e
        except OSError as e:
            self._directory_stack.append(directory)
            raise _batchprocessor.BatchProcessError(e) from e
        return 0

    def _rmdir_directive(self, args: collections.abc.Sequence[str]):
        """
        Remove one or more directories.

        Basic implementation of the Unix 'rmdir' command.

        Supports basic POSIX arguments.

        See: \\rmdir --help
        """
        parser = _util.DirectiveArgumentParser(prog='\\rmdir')

        parser.add_argument('-p', '--parents', action='store_true',
                            help='Remove directory and its parents')
        parser.add_argument('directories', nargs='+')
        parsed_args = parser.parse_args(args)

        if parser.return_code is not None:
            return parser.return_code

        for d in parsed_args.directories:
            if parsed_args.parents:
                shutil.rmtree(d, ignore_errors=True)
            else:
                try:
                    os.rmdir(d)
                except OSError as e:
                    raise _batchprocessor.BatchProcessError(
                        f"Failed to remove directory {d}: {e.strerror}") from e
        return 0

    def _rm_directive(self, args: collections.abc.Sequence[str]):
        """
        Remove Files.

        Basic implementation of the Unix 'rm' command.

        Supports basic POSIX arguments.

        See: \\rm --help
        """
        parser = _util.DirectiveArgumentParser(prog='\\rm')

        parser.add_argument('-r', '--recursive', action='store_true',
                            help='Remove directories and their contents recursively')
        parser.add_argument('-f', '--force', action='store_true',
                            help='Ignore nonexistent files and arguments, never prompt')
        parser.add_argument('files', nargs='+')
        parsed_args = parser.parse_args(args)

        if parser.return_code is not None:
            return parser.return_code

        for f in parsed_args.files:
            try:
                if parsed_args.recursive:
                    if os.path.isdir(f):
                        shutil.rmtree(f, ignore_errors=parsed_args.force)
                    else:
                        if parsed_args.force:
                            try:
                                os.remove(f)
                            except FileNotFoundError:
                                pass
                        else:
                            os.remove(f)
                else:
                    if parsed_args.force:
                        try:
                            os.remove(f)
                        except FileNotFoundError:
                            pass
                    else:
                        os.remove(f)
            except OSError as e:
                raise _batchprocessor.BatchProcessError(
                    f"Failed to remove {('directory' if os.path.isdir(f) else 'file')} {f}: {e.strerror}") from e
        return 0

    def _generate_template_variables_with_types(self) -> dict[str, tuple[type, typing.Any]]:
        template_variables = dict()

        variable_prefix = 'last_'

        for attr, hint in typing.get_type_hints(self.render_loop.config.__class__).items():
            value = getattr(self.render_loop.config, attr)
            if variable_prefix:
                prefix = variable_prefix if not attr.startswith(variable_prefix) else ''
            else:
                prefix = ''
            gen_name = prefix + attr
            if gen_name not in template_variables:
                if _types.is_type_or_optional(hint, collections.abc.Sequence):
                    t_val = value if value is not None else []
                    template_variables[gen_name] = (hint, t_val)
                else:
                    template_variables[gen_name] = (hint, value)

        template_variables.update({
            'last_images': (collections.abc.Iterable[str], self.render_loop.written_images),
            'last_animations': (collections.abc.Iterable[str], self.render_loop.written_animations),
            'injected_args': (collections.abc.Sequence[str], self.template_variables.get('injected_args')),
            'injected_device': (_types.OptionalString, self.template_variables.get('injected_device')),
            'injected_verbose': (_types.OptionalBoolean, self.template_variables.get('injected_verbose')),
            'injected_plugin_modules': (_types.OptionalPaths, self.template_variables.get('injected_plugin_modules')),
            'saved_modules': (dict[str, dict[str, typing.Any]], self.template_variables.get('saved_modules')),
            'glob': (types.ModuleType, self.template_variables.get('glob')),
            'os': (types.ModuleType, self.template_variables.get('os'))
        })

        return template_variables

    def _generate_template_variables(self) -> dict[str, typing.Any]:
        return {k: v[1] for k, v in self._generate_template_variables_with_types().items()}

    def generate_directives_help(self, directive_names: typing.Collection[str] | None = None,
                                 help_wrap_width: int | None = None):
        """
        Generate the help string for ``--directives-help``

        :param directive_names: Display help for specific directives, if ``None`` or ``[]`` is specified, display all.
        :param help_wrap_width: Wrap documentation strings by this amount,
            if ``None`` use :py:func:`dgenerate.textprocessing.long_text_wrap_width()`

        :raise ValueError: if given directive names could not be found

        :return: help string
        """

        if directive_names is None:
            directive_names = []

        directives: dict[str, str | typing.Callable] = self.directives.copy()

        directives.update(self.directives_builtins_help)

        if len(directive_names) == 0:
            help_string = f'Available config directives:' + '\n\n'
            help_string += '\n'.join((' ' * 4) + _textprocessing.quote('\\' + i) for i in sorted(directives.keys()))
        else:
            help_string = ''

            directive_names = [n.lstrip('\\') for n in directive_names]

            if directive_names is not None and len(directive_names) > 0:
                found = dict()
                not_found = []
                for n in directive_names:
                    if n not in directives:
                        not_found.append(n)
                        continue
                    found[n] = directives[n]
                if not_found:
                    raise ValueError(
                        f'No directives named: {_textprocessing.oxford_comma(not_found, "or")}')
                directives = found

            def docs():
                for name, impl in directives.items():
                    if isinstance(impl, str):
                        doc = impl
                    else:
                        doc = inspect.cleandoc(impl.__doc__).strip() \
                            if impl.__doc__ is not None else 'No documentation provided.'
                    doc = \
                        _textprocessing.wrap_paragraphs(
                            doc,
                            initial_indent=' ' * 4,
                            subsequent_indent=' ' * 4,
                            width=_types.default(help_wrap_width, _textprocessing.long_text_wrap_width()))
                    yield name + _textprocessing.underline(':\n\n' + doc + '\n')

            help_string += '\n'.join(docs())

        return help_string

    def generate_functions_help(self, function_names: typing.Collection[str] | None = None,
                                help_wrap_width: int | None = None):
        """
        Generate the help string for ``--functions-help``


        :param function_names: Display help for specific functions, if ``None`` or ``[]`` is specified, display all.
        :param help_wrap_width: Wrap documentation strings by this amount,
            if ``None`` use :py:func:`dgenerate.textprocessing.long_text_wrap_width()`

        :raise ValueError: if given directive names could not be found

        :return: help string
        """

        if function_names is None:
            function_names = []

        functions: dict[str, str | typing.Callable] = self.builtins.copy()
        functions.update(self.template_functions)

        functions = dict(sorted(functions.items(), key=lambda f: f[0]))

        if len(function_names) == 0:

            help_string = f'Available config template functions:' + '\n\n'
            help_string += '\n'.join(
                (' ' * 4) +
                (_types.format_function_signature(v, alternate_name=n)
                 if inspect.isfunction(v) else _types.format_function_signature(
                    v.__init__, alternate_name=n, omit_params={'self'}))
                for n, v in functions.items())

        else:
            help_string = ''

            if function_names is not None and len(function_names) > 0:
                found = dict()
                not_found = []
                for n in function_names:
                    if n not in functions:
                        not_found.append(n)
                        continue
                    found[n] = functions[n]
                if not_found:
                    raise ValueError(
                        f'No template functions named: {_textprocessing.oxford_comma(not_found, "or")}')
                functions = found

            def docs():
                for name, impl in functions.items():
                    doc = inspect.cleandoc(impl.__doc__).strip() \
                        if impl.__doc__ is not None else 'No documentation provided.'

                    doc = \
                        _textprocessing.wrap_paragraphs(
                            doc,
                            initial_indent=' ' * 4,
                            subsequent_indent=' ' * 4,
                            width=_types.default(help_wrap_width, _textprocessing.long_text_wrap_width()))
                    yield (_types.format_function_signature(impl, alternate_name=name) if inspect.isfunction(impl)
                           else _types.format_function_signature(impl.__init__,
                                                                 alternate_name=name,
                                                                 omit_params={'self'})) + \
                        _textprocessing.underline(':\n\n' + doc + '\n')

            help_string += '\n'.join(docs())

        return help_string

    def generate_template_variables_help(self,
                                         variable_names: typing.Collection[str] | None = None,
                                         show_values: bool = True):
        """
        Generate a help string describing available template variables, their types, and values for use in batch processing.

        This is used for ``--templates-help``

        :param variable_names: Display help for specific variables, if ``None`` or ``[]`` is specified, display all.

        :param show_values: Show the value of the template variable or just the name?

        :raise ValueError: if given variable names could not be found

        :return: a human-readable description of all template variables
        """

        values = self._generate_template_variables_with_types()

        for k, v in self.template_variables.items():
            if k not in values:
                values[k] = (v.__class__, v)

        if variable_names is not None and len(variable_names) > 0:
            found = dict()
            not_found = []
            for n in variable_names:
                if n not in values:
                    not_found.append(n)
                    continue
                found[n] = values[n]
            if not_found:
                raise ValueError(
                    f'No template variables named: {_textprocessing.oxford_comma(not_found, "or")}')
            values = found
        else:
            values = dict(sorted(values.items(), key=lambda x: x[0]))

        if len(values) > 1:
            header = 'Config template variables are'
        else:
            header = 'Config template variable is'

        help_string = f'{header}:' + '\n\n'

        def wrap(val):
            return _textprocessing.wrap(
                repr(val),
                width=_textprocessing.long_text_wrap_width(),
                subsequent_indent=' ' * 17)

        return help_string + '\n'.join(
            ' ' * 4 + f'Name: {_textprocessing.quote(i[0])}\n{" " * 8}'
                      f'Type: {i[1][0]}' + (f'\n{" " * 8}Value: {wrap(i[1][1])}' if show_values else '') for i in
            values.items())

    def _templates_help_directive(self, args: collections.abc.Sequence[str]):
        """
        Prints all template variables in the global scope, with their types and values.

        Providing template variable names as arguments prints just information about those template variables.

        This does not cause the config to exit.
        """
        _messages.log(self.generate_template_variables_help(args) + '\n')
        return 0

    def _directives_help_directive(self, args: collections.abc.Sequence[str]):
        """
        Prints all directives.

        Providing directive names as arguments prints documentation for those directives.

        This does not cause the config to exit.
        """
        _messages.log(self.generate_directives_help(args) + '\n')
        return 0

    def _functions_help_directive(self, args: collections.abc.Sequence[str]):
        """
        Prints all template functions.

        Providing function names as arguments prints documentation for those functions.

        This does not cause the config to exit.
        """
        _messages.log(self.generate_functions_help(args) + '\n')
        return 0

    def _quantizer_help(self, args: collections.abc.Sequence[str]):
        """
        Prints all quantizer names. Alias for --quantizer-help

        Providing quantizer names as arguments prints documentation for those quantizers.

        This does not cause the config to exit.
        """

        self.run_string(shlex.join(['--quantizer-help'] + list(args)))
        return 0

    def _image_processor_help_directive(self, args: collections.abc.Sequence[str]):
        """
        Prints all image processor names. Alias for --image-processor-help

        Providing processor names as arguments prints documentation for those processors.

        This does not cause the config to exit.
        """

        self.run_string(shlex.join(['--image-processor-help'] + list(args)))
        return 0

    def _latents_processor_help_directive(self, args: collections.abc.Sequence[str]):
        """
        Prints all image processor names. Alias for --latents-processor-help

        Providing processor names as arguments prints documentation for those processors.

        This does not cause the config to exit.
        """

        self.run_string(shlex.join(['--latents-processor-help'] + list(args)))
        return 0

    def _prompt_weighter_help_directive(self, args: collections.abc.Sequence[str]):
        """
        Prints all prompt weighter names. Alias for --prompt-weighter-help

        Providing prompt weighter names as arguments prints documentation for those prompt weighters.

        This does not cause the config to exit.
        """

        self.run_string(shlex.join(['--prompt-weighter-help'] + list(args)))
        return 0

    def _prompt_upscaler_help_directive(self, args: collections.abc.Sequence[str]):
        """
        Prints all prompt upscaler names. Alias for --prompt-upscaler-help

        Providing prompt upscaler names as arguments prints documentation for those prompt upscalers.

        This does not cause the config to exit.
        """

        self.run_string(shlex.join(['--prompt-upscaler-help'] + list(args)))
        return 0

    def _help_directive(self, args: collections.abc.Sequence[str]):
        """
        dgenerate --help

        Alias for --help
        """
        self.run_string('--help')
        return 0

    def _import_directive(self, args: collections.abc.Sequence[str]):
        """
        Import Python modules into the template variables.
        
        This implements Python's import syntax, including the 'as' keyword.
        
        Examples:
        
            \\import torch
            \\import torch.nn
            \\import torch.nn.functional as f
            
        The imported modules are available as template variables with the
        same name as the imported module, or the alias if 'as' is used.
        """

        if not args:
            raise _batchprocessor.BatchProcessError(
                '\\import directive must be used with at least one argument.')

        # Check if we have an 'as' alias (should be exactly 3 args: module, 'as', alias)
        if len(args) == 3 and args[1] == 'as':
            module_path = args[0]
            variable_name = args[2]

            if not variable_name.isidentifier():
                raise _batchprocessor.BatchProcessError(
                    f'Invalid identifier "{variable_name}" in import statement.')

            self.user_define_check(variable_name)

            try:
                # Import the full module
                module = importlib.import_module(module_path)

                # Add it to template variables with the alias
                self.template_variables[variable_name] = module

            except ImportError as e:
                raise _batchprocessor.BatchProcessError(
                    f'Failed to import module "{module_path}": {str(e)}') from e

        elif len(args) == 1:
            # Regular import without 'as'
            module_path = args[0]

            # Split the path into parts
            parts = module_path.split('.')

            # Validate module name parts
            if not all(part.isidentifier() for part in parts):
                raise _batchprocessor.BatchProcessError(
                    f'Invalid module name component in "{module_path}".')

            # The base module name is the first part
            base_module_name = parts[0]

            if base_module_name != 'os':
                # harmless to allow, even though we
                # already import os as a builtin module
                self.user_define_check(base_module_name)

            try:
                # First import the base module
                base_module = importlib.import_module(base_module_name)

                # Add the base module to template variables
                self.template_variables[base_module_name] = base_module

                # If it's a nested import (e.g., os.path), import the full module too
                if len(parts) > 1:
                    # Import the full module to ensure all submodules are properly loaded
                    importlib.import_module(module_path)

            except ImportError as e:
                raise _batchprocessor.BatchProcessError(
                    f'Failed to import module "{module_path}": {str(e)}') from e

        else:
            raise _batchprocessor.BatchProcessError(
                f'Invalid import syntax. Use "\\import module" or "\\import module as alias".')

        return 0


__all__ = _types.module_all()
