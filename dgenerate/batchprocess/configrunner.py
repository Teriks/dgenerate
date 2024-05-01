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
import glob
import inspect
import os
import shlex
import shutil
import subprocess
import threading
import types
import typing

import dgenerate
import dgenerate.arguments as _arguments
import dgenerate.batchprocess.batchprocessor as _batchprocessor
import dgenerate.batchprocess.configrunnerpluginloader as _configrunnerpluginloader
import dgenerate.invoker as _invoker
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.prompt as _prompt
import dgenerate.renderloop as _renderloop
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
import time
import stat


def _format_prompt_single(prompt):
    pos = prompt.positive
    neg = prompt.negative

    if pos is None:
        raise _batchprocessor.BatchProcessError('Attempt to format a prompt with no positive prompt value.')

    if pos and neg:
        return shlex.quote(f"{pos}; {neg}")
    return shlex.quote(pos)


def _format_prompt(
        prompts: typing.Union[_prompt.Prompt,
        collections.abc.Iterable[_prompt.Prompt]]):
    """
    Format a prompt object, or a list of prompt objects, into quoted string(s)
    """
    if isinstance(prompts, _prompt.Prompt):
        return _format_prompt_single(prompts)
    return ' '.join(_format_prompt_single(p) for p in prompts)


def _format_size(size: collections.abc.Iterable[int]):
    """
    Join an iterable of integers into a string seperated by the character 'x', for example (512, 512) -> "512x512"
    """
    return _textprocessing.format_size(size)


def _quote(strings: typing.Union[str, collections.abc.Iterable[typing.Any]]):
    """
    Shell quote a string or iterable of strings
    """
    if isinstance(strings, str):
        return shlex.quote(str(strings))
    return ' '.join(shlex.quote(str(s)) for s in strings)


def _unquote(strings: typing.Union[str, collections.abc.Iterable[typing.Any]]):
    """
    Un-Shell quote a string or iterable of strings
    """
    if isinstance(strings, str):
        return shlex.split(str(strings))
    return [shlex.split(str(s)) for s in strings]


def _last(iterable: typing.Union[list, collections.abc.Iterable[typing.Any]]):
    """
    Return the last element in an iterable collection.
    """
    if isinstance(iterable, list):
        return iterable[-1]
    try:
        *_, last_item = iterable
    except ValueError:
        raise _batchprocessor.BatchProcessError(
            'Usage of template function "last" on an empty iterable.')
    return last_item


def _first(iterable: collections.abc.Iterable[typing.Any]):
    """
    Return the first element in an iterable collection.
    """
    try:
        v = next(iter(iterable))
    except StopIteration:
        raise _batchprocessor.BatchProcessError(
            'Usage of template function "first" on an empty iterable.')
    return v


def _cwd():
    """
    Return the current working directory as a string.
    """
    return os.getcwd()


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

    def __init__(self,
                 injected_args: typing.Optional[collections.abc.Sequence[str]] = None,
                 render_loop: typing.Optional[_renderloop.RenderLoop] = None,
                 plugin_loader: _configrunnerpluginloader.ConfigRunnerPluginLoader = None,
                 version: typing.Union[_types.Version, str] = dgenerate.__version__,
                 throw: bool = False):
        """

        :raises dgenerate.plugin.ModuleFileNotFoundError: If a module path parsed from
            ``--plugin-modules`` in ``injected_args`` could not be found on disk.


        :param injected_args: dgenerate command line arguments in the form of a list, see: shlex module, or sys.argv.
            These arguments will be injected at the end of every dgenerate invocation in the config. ``--plugin-modules``
            are parsed from ``injected_args`` and added to ``plugin_loader``. If ``-v/--verbose`` is present in ``injected_args``
            debugging output will be enabled globally while the config runs, and not just for invocations.

        :param render_loop: RenderLoop instance, if ``None`` is provided one will be created.
        :param plugin_loader: Batch processor plugin loader, if one is not provided one will be created.
        :param version: Config version for ``#! dgenerate x.x.x`` version checks, defaults to ``dgenerate.__version__``
        :param throw: Whether to throw exceptions from :py:func:`dgenerate.invoker.invoke_dgenerate` or handle them.
            If you set this to ``True`` exceptions will propagate out of dgenerate invocations instead of a
            :py:exc:`dgenerate.batchprocess.BatchProcessError` being raised by the created
            :py:class:`dgenerate.batchprocess.BatchProcessor`. A line number where the error occurred can be
             obtained using :py:attr:`dgenerate.batchprocess.BatchProcessor.current_line`.
        """

        def invoker(args):
            try:
                return_code = \
                    _invoker.invoke_dgenerate(args,
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

        self.render_loop = render_loop

        self.template_variables = {
            'injected_args': self.injected_args,
            'injected_device': _arguments.parse_device(self.injected_args)[0],
            'injected_verbose': _arguments.parse_verbose(self.injected_args)[0],
            'injected_plugin_modules': _arguments.parse_plugin_modules(self.injected_args)[0],
            'saved_modules': dict(),
            'glob': glob,
            'path': os.path,
        }

        self.template_variables = self._generate_template_variables()

        self.reserved_template_variables = set(self.template_variables.keys())

        self.template_functions = {
            'unquote': _unquote,
            'quote': _quote,
            'format_prompt': _format_prompt,
            'format_size': _format_size,
            'last': _last,
            'first': _first,
            'cwd': _cwd
        }

        def return_zero(func, help):
            def wrap(args):
                func()
                return 0

            wrap.__doc__ = help

            return wrap

        self.directives = {
            'help': self._help_directive,
            'templates_help': self._templates_help_directive,
            'directives_help': self._directives_help_directive,
            'functions_help': self._functions_help_directive,
            'image_processor_help': self._image_processor_help_directive,
            'clear_model_cache': return_zero(
                _pipelinewrapper.clear_model_cache,
                help='Clear all user specified models from the in memory cache.'),
            'clear_pipeline_cache': return_zero(
                _pipelinewrapper.clear_pipeline_cache,
                help='Clear all diffusers pipelines from the in memory cache, '
                     'this will not clear user specified VAEs, UNets, and ControlNet models, '
                     'just pipeline objects which may or may not have automatically loaded those for you.'),
            'clear_unet_cache': return_zero(
                _pipelinewrapper.clear_unet_cache,
                help='Clear all user specified UNet models from the in memory cache.'),
            'clear_vae_cache': return_zero(
                _pipelinewrapper.clear_vae_cache,
                help='Clear all user specified VAE models from the in memory cache.'),
            'clear_control_net_cache': return_zero(
                _pipelinewrapper.clear_control_net_cache,
                help='Clear all user specified ControlNet models from the in memory cache.'),
            'save_modules': self._save_modules_directive,
            'use_modules': self._use_modules_directive,
            'clear_modules': self._clear_modules_directive,
            'gen_seeds': self._gen_seeds_directive,
            'pwd': self._pwd_directive,
            'ls': self._ls_directive,
            'cd': self._cd_directive,
            'pushd': self._pushd_directive,
            'popd': self._popd_directive,
            'exec': self._exec_directive,
            'mv': self._mv_directive,
            'cp': self._cp_directive,
            'mkdir': self._mkdir_directive,
            'exit': self._exit_directive
        }

        self.plugin_loader = \
            _configrunnerpluginloader.ConfigRunnerPluginLoader() if \
                plugin_loader is None else plugin_loader

        self._plugin_module_paths = set()

        if injected_args:
            self._plugin_module_paths.update(_arguments.parse_plugin_modules(injected_args)[0])
            self.plugin_loader.load_plugin_modules(self._plugin_module_paths)
            self.render_loop.image_processor_loader.load_plugin_modules(self._plugin_module_paths)

        for plugin_class in self.plugin_loader.get_available_classes():
            self.plugin_loader.load(plugin_class.get_names()[0],
                                    config_runner=self,
                                    render_loop=self.render_loop)

        self.directives['import_plugins'] = self._import_plugins_directive

        self._directory_stack = []

    def _import_plugins_directive(self, plugin_paths: collections.abc.Sequence[str]):
        """
        Imports plugins from within a config, this imports config plugins as well as image processor plugins.
        This has an identical effect to the --plugin-modules argument. You may specify multiple plugin
        module directories or python files containing plugin implementations.
        """

        if len(plugin_paths) == 0:
            raise _batchprocessor.BatchProcessError(
                '\\import_plugins must be used with at least one argument.')

        self._plugin_module_paths.update(plugin_paths)
        self.render_loop.image_processor_loader.load_plugin_modules(plugin_paths)
        new_classes = self.plugin_loader.load_plugin_modules(plugin_paths)
        for cls in new_classes:
            self.plugin_loader.load(cls.get_names()[0],
                                    config_runner=self,
                                    render_loop=self.render_loop)

        return 0

    def _exit_directive(self, args: collections.abc.Sequence[str]):
        """
        Causes the dgenerate process to exit with a specific return code.
        This directive accepts one argument, the return code, which is optional
        and 0 by default. It must be an integer value.
        """
        if (len(args)) == 0:
            exit(0)

        try:
            return_code = int(args[0])
        except ValueError:
            raise _batchprocessor.BatchProcessError(
                f'\\exit return code must be an integer value, received: {args[0]}')

        exit(return_code)

    def _save_modules_directive(self, args: collections.abc.Sequence[str]):
        """
        Save a set of pipeline modules off the last diffusers pipeline used for the
        main model of a dgenerate invocation. The first argument is a variable name
        that the modules will be saved to, which can be reference later with \\use_modules.
        The rest of the arguments are names of pipeline modules that you want to save to this
        variable as a set of modules that are kept together, usable names are: unet, vae, text_encoder,
        text_encoder_2, tokenizer, tokenizer_2, safety_checker, feature_extractor, controlnet,
        scheduler, unet
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

        creation_result = self.render_loop.pipeline_wrapper.recall_main_pipeline()
        saved_modules[args[0]] = creation_result.get_pipeline_modules(args[1:])
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
                try:
                    del saved_modules[arg]
                except KeyError:
                    raise _batchprocessor.BatchProcessError(
                        f'No pipeline modules were saved to the variable name "{arg}", '
                        f'that name could not be found.')
        else:
            saved_modules.clear()
        return 0

    def _gen_seeds_directive(self, args: collections.abc.Sequence[str]):
        """
        Generate N random integer seeds and store them as a list to a template variable name.

        The first argument is the variable name, the second argument is the number of seeds to generate.
        """
        if len(args) == 2:
            try:
                self.template_variables[args[0]] = \
                    [str(s) for s in _renderloop.gen_seeds(int(args[1]))]
            except ValueError:
                raise _batchprocessor.BatchProcessError(
                    'The second argument of \\gen_seeds must be an integer value.')
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

    def _mkdir_directive(self, args: collections.abc.Sequence[str]):
        """
        Make one or more directories, parent directories will be created if they do not exist.
        """
        if len(args) < 1:
            raise _batchprocessor.BatchProcessError(
                '\\mkdir directive must be provided at least one argument.')
        for d in args:
            os.makedirs(d, exist_ok=True)
        return 0

    def _exec_directive(self, args: collections.abc.Sequence[str]):
        """
        Execute a shell command line as a new process or processes.

        The pipe | operator is supported for piping to standard input, as well as bash file redirection syntax.

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
            \exec dgenerate < my_config.txt &> log.txt
            \exec dgenerate < my_config.txt > log.txt 2>&1
            \exec dgenerate < my_config.txt > stdout.txt 2> stderr.txt

        Windows cat pipe:

            \exec cmd /c "type my_config.txt" | dgenerate &> test.log

        Linux cat pipe:

            \exec cat my_config.txt | dgenerate &> test.log
        """

        if len(args) == 0:
            raise _batchprocessor.BatchProcessError(
                '\\exec directive must be passed at least one argument.')

        args = list(args)

        open_files = []
        open_processes = []

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

            for command in commands:
                if not command:
                    raise _batchprocessor.BatchProcessError(
                        f'no command specified to pipe to.')

                stdout = _messages.get_message_file()
                stderr = _messages.get_error_file()

                redirects = {'>', '1>', '2>', '&>', '>>', '1>>', '2>>', '&>>', '2>&1', '1>&2'}
                _i = 0
                while any(i in command for i in redirects):
                    if command[_i] in redirects:
                        remove_cnt = 1
                        mode = 'a' if '>>' in command[_i] else 'w'
                        if command[_i] == '2>&1':
                            stderr = stdout
                        elif command[_i] == '1>&2':
                            stdout = stderr
                        else:
                            remove_cnt = 2
                            try:
                                file = open(command[_i + 1], mode, encoding='utf-8')
                                open_files.append(file)
                            except IndexError:
                                raise _batchprocessor.BatchProcessError(
                                    f'{command[_i]} no output file specified.')
                            if command[_i][0] != '2':
                                stdout = file
                            if command[_i][0] != '1':
                                stderr = file
                        command = command[:_i] + command[_i + remove_cnt:]
                        _i -= remove_cnt
                    _i += 1

                stdin = stdin if previous_process is None else previous_process.stdout

                if command[0] == 'dgenerate' and stdin is None:
                    command = list(command) + ['--no-stdin']

                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'
                env['PYTHONIOENCODING'] = 'utf-8'
                try:
                    process = subprocess.Popen(command,
                                               stdin=stdin,
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE,
                                               env=env)
                    open_processes.append(process)
                except FileNotFoundError:
                    raise _batchprocessor.BatchProcessError(
                        f'Command "{command[0]}" not found on system.')

                previous_process = process

            stop_threads = threading.Event()

            def readlines_unbuffered(file):
                line = []
                while byte := file.read(1):
                    line.append(byte)
                    if byte in {b'\n', b'\r'}:
                        yield b''.join(line).decode('utf-8')
                        line = []
                if line:
                    yield b''.join(line).decode('utf-8')

            def handle_stream(stream, out_stream):
                for line in readlines_unbuffered(stream):
                    if stop_threads.is_set():
                        break
                    text = line.rstrip()
                    if text:
                        print(text, file=out_stream)
                        out_stream.flush()

            thread1 = threading.Thread(
                target=handle_stream,
                args=(process.stdout, stdout))
            thread1.daemon = True
            thread1.start()

            thread2 = threading.Thread(
                target=handle_stream,
                args=(process.stderr, stderr))
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

        Basic implementation of the Unix 'ls' command, accepts the argument -l
        """

        long = '-l' in args
        if long:
            args = [a for a in args if a != '-l']

        if len(args) == 1:
            paths = [args[0]]
        elif len(args) > 1:
            paths = args
        else:
            paths = ['.']

        _messages.log('')

        for path in paths:
            if not os.path.exists(path):
                _messages.log(f'ls: {path}: No such file or directory')
                continue
            if len(paths) > 1:
                _messages.log(f'{path}:')
            if long:
                max_size_length = \
                    max(len(str(os.stat(os.path.join(path, filename)).st_size)) for filename in os.listdir(path)
                        if all or not filename.startswith('.'))

            for filename in sorted(os.listdir(path)):
                if not all and filename.startswith('.'):
                    continue

                if long:
                    file_stat = os.stat(os.path.join(path, filename))
                    file_permissions = stat.filemode(file_stat.st_mode)
                    num_links = file_stat.st_nlink
                    file_size = file_stat.st_size
                    mod_time = time.ctime(file_stat.st_mtime)
                    _messages.log(
                        f'{file_permissions:<11} {num_links:<3} '
                        f'{file_size:<{max_size_length}} {mod_time:<25} {filename}')
                else:
                    _messages.log(
                        f'{filename}')
            if len(paths) > 1:
                _messages.log('')

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
                raise _batchprocessor.BatchProcessError(e)
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
                raise _batchprocessor.BatchProcessError(e)
        else:
            raise _batchprocessor.BatchProcessError(
                '\\pushd directive takes 1 argument, the directory name.')
        return 0

    def _popd_directive(self, args: collections.abc.Sequence[str]):
        """
        Pop the last directory of the directory stack and change to that directory.
        """
        try:
            dir = self._directory_stack.pop()
            os.chdir(dir)
            _messages.log(f'Working Directory Changed To: "{dir}"')
        except IndexError:
            raise _batchprocessor.BatchProcessError('\\popd failed, no directory on the stack.')
        except OSError as e:
            self._directory_stack.append(dir)
            raise _batchprocessor.BatchProcessError(e)
        return 0

    def _config_generate_template_variables_with_types(self) -> dict[str, tuple[type, typing.Any]]:
        template_variables = {}

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
        })

        return template_variables

    def _generate_template_variables_with_types(self) -> dict[str, tuple[type, typing.Any]]:
        template_variables = self._config_generate_template_variables_with_types()

        template_variables['injected_args'] = (collections.abc.Sequence[str],
                                               self.template_variables.get('injected_args'))

        template_variables['injected_device'] = (_types.OptionalString,
                                                 self.template_variables.get('injected_device'))

        template_variables['injected_verbose'] = (_types.OptionalBoolean,
                                                  self.template_variables.get('injected_verbose'))

        template_variables['injected_plugin_modules'] = (_types.OptionalPaths,
                                                         self.template_variables.get('injected_plugin_modules'))

        template_variables['saved_modules'] = (dict[str, dict[str, typing.Any]],
                                               self.template_variables.get('saved_modules'))

        template_variables['glob'] = (types.ModuleType, self.template_variables.get('glob'))

        template_variables['path'] = (types.ModuleType, self.template_variables.get('path'))

        return template_variables

    def _generate_template_variables(self) -> dict[str, typing.Any]:
        return {k: v[1] for k, v in self._generate_template_variables_with_types().items()}

    def generate_directives_help(self, directive_names: typing.Optional[typing.Collection[str]] = None):
        """
        Generate the help string for ``--directives-help``


        :param directive_names: Display help for specific directives, if ``None`` or ``[]`` is specified, display all.

        :raise ValueError: if given directive names could not be found

        :return: help string
        """

        if directive_names is None:
            directive_names = []

        directives: dict[str, typing.Union[str, typing.Callable]] = self.directives.copy()

        directives.update(self.directives_builtins_help)

        if len(directive_names) == 0:
            help_string = f'Available config directives:' + '\n\n'
            help_string += '\n'.join((' ' * 4) + _textprocessing.quote('\\' + i) for i in directives.keys())
        else:
            help_string = ''

            directive_names = {n.lstrip('\\') for n in directive_names}

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
                            width=_textprocessing.long_text_wrap_width())
                    yield name + _textprocessing.underline(':\n\n' + doc + '\n')

            help_string += '\n'.join(docs())

        return help_string

    def generate_functions_help(self, function_names: typing.Optional[typing.Collection[str]] = None):
        """
        Generate the help string for ``--functions-help``


        :param function_names: Display help for specific functions, if ``None`` or ``[]`` is specified, display all.

        :raise ValueError: if given directive names could not be found

        :return: help string
        """

        if function_names is None:
            function_names = []

        functions: dict[str, typing.Union[str, typing.Callable]] = self.builtins.copy()
        functions.update(self.template_functions)

        functions = dict(sorted(functions.items()))

        if len(function_names) == 0:

            help_string = f'Available config template functions:' + '\n\n'
            help_string += '\n\n'.join(
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
                            width=_textprocessing.long_text_wrap_width())
                    yield (_types.format_function_signature(impl, alternate_name=name) if inspect.isfunction(impl)
                           else _types.format_function_signature(impl.__init__,
                                                                 alternate_name=name,
                                                                 omit_params={'self'})) + \
                        _textprocessing.underline(':\n\n' + doc + '\n')

            help_string += '\n'.join(docs())

        return help_string

    def generate_template_variables_help(self,
                                         variable_names: typing.Optional[typing.Collection[str]] = None,
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

    def _image_processor_help_directive(self, args: collections.abc.Sequence[str]):
        """
        Prints all image processor names. Alias for --image-processor-help

        Providing processor names as arguments prints documentation for those processors.

        This does not cause the config to exit.
        """

        self.run_string(shlex.join(['--image-processor-help'] + list(args)))
        return 0

    def _help_directive(self, args: collections.abc.Sequence[str]):
        """
        dgenerate --help

        Alias for --help
        """
        self.run_string('--help')
        return 0


__all__ = _types.module_all()
