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

import dgenerate
import dgenerate.batchprocess.configrunnerpluginloader as _configrunnerpluginloader
import dgenerate.plugin as _plugin
import dgenerate.renderloop as _renderloop
import dgenerate.types as _types


class ConfigRunnerPlugin(_plugin.Plugin):
    """
    Abstract base class for config runner plugin implementations.
    """

    def __init__(self,
                 loaded_by_name: str,
                 config_runner: typing.Optional['dgenerate.batchprocess.ConfigRunner'] = None,
                 render_loop: typing.Optional[_renderloop.RenderLoop] = None,
                 local_files_only: bool = False,
                 **kwargs):

        super().__init__(loaded_by_name=loaded_by_name,
                         argument_error_type=_configrunnerpluginloader.ConfigRunnerPluginArgumentError,
                         **kwargs)

        self.__config_runner = config_runner
        self.__render_loop = render_loop
        self.__local_files_only = local_files_only

    def set_template_variable(self, name, value):
        """
        Set a template variable on the :py:class:`dgenerate.batchprocess.ConfigRunner` instance.

        :param name: variable name
        :param value: variable value
        """
        if self.config_runner is not None:
            self.config_runner.template_variables[name] = value

    def update_template_variables(self, values):
        """
        Update multiple template variable values on the :py:class:`dgenerate.batchprocess.ConfigRunner` instance.

        :param values: variable values, dictionary of names to values
        """
        if self.config_runner is not None:
            self.config_runner.template_variables.update(values)

    def register_directive(self, name, implementation: typing.Callable[[collections.abc.Sequence[str]], int]):
        """
        Register a config directive implementation on the :py:class:`dgenerate.batchprocess.ConfigRunner` instance.

        Your directive should return a return code, 0 for success and anything else for failure.

        Returning non zero will cause :py:class:`BatchProcessError` to be raised from the runner, halting execution of the config.

        Any non-exiting exception will be eaten and rethrown as :py:class:`BatchProcessError`, also halting execution of the config.

        :raise RuntimeError: if a config directive with the same name already exists

        :param name: directive name
        :param implementation: implementation callable
        """
        if self.config_runner is not None:

            if name in self.config_runner.directives:
                raise RuntimeError(
                    f'directive name "{name}" cannot be registered by plugin '
                    f'"{self.loaded_by_name}" because that directive name already exists.')

            self.config_runner.directives[name] = implementation

    def register_template_function(self, name, implementation: typing.Callable):
        """
        Register a config template function implementation on the :py:class:`dgenerate.batchprocess.ConfigRunner` instance.

        :raise RuntimeError: if a template function with the same name already exists

        :param name: function name
        :param implementation: implementation callable
        """
        if self.config_runner is not None:

            if name in self.config_runner.template_functions:
                raise RuntimeError(
                    f'template function name "{name}" cannot be registered by plugin '
                    f'"{self.loaded_by_name}" because that template function name already exists.')

            self.config_runner.template_functions[name] = implementation

    @property
    def injected_args(self) -> collections.abc.Sequence[str]:
        """
        Return any arguments injected into the config from the command line.

        If none were injected an empty sequence will be returned.

        :return: command line arguments
        """
        if self.config_runner is not None:
            return self.config_runner.injected_args
        return []

    @property
    def render_loop(self) -> _renderloop.RenderLoop | None:
        """
        Provides access to the currently instantiated :py:class:`dgenerate.renderloop.RenderLoop` object.

        This object will have been used for any previous invocation of dgenerate in a config file.
        """
        return self.__render_loop

    @property
    def config_runner(self) -> typing.Optional['dgenerate.batchprocess.ConfigRunner']:
        """
        Provides access to the currently instantiated :py:class:`dgenerate.batchprocess.ConfigRunner` object
        running the config file that this directive is being invoked in.
        """
        return self.__config_runner

    @property
    def plugin_module_paths(self) -> frozenset[str]:
        """
        Set of plugin module paths if they were injected into the config runner by ``--plugin-modules``
        or used in a ``\\import_plugins`` statement in a config.

        :return: a set of paths, may be empty but not ``None``
        """
        if self.config_runner is not None:
            return self.config_runner.plugin_module_paths
        return frozenset()

    @property
    def local_files_only(self) -> bool:
        """
        Is this plugin only going to look for resources such as models in cache / on disk?
        """
        return self.__local_files_only


__all__ = _types.module_all()
