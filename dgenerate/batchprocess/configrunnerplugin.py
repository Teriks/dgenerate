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

import dgenerate.batchprocess.batchprocessor as _batchprocessor
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
                 batch_processor: typing.Optional[_batchprocessor.BatchProcessor] = None,
                 render_loop: typing.Optional[_renderloop.RenderLoop] = None,
                 plugin_module_paths: typing.Optional[typing.List[str]] = None,
                 **kwargs):

        super().__init__(loaded_by_name=loaded_by_name,
                         argument_error_type=_configrunnerpluginloader.ConfigRunnerPluginArgumentError,
                         **kwargs)

        self.__batch_processor = batch_processor
        self.__render_loop = render_loop

        self.__plugin_module_paths = \
            plugin_module_paths if \
                plugin_module_paths is not None else []

    def set_template_variable(self, name, value):
        if self.batch_processor is not None:
            self.batch_processor.template_variables[name] = value

    def update_template_variables(self, values):
        if self.batch_processor is not None:
            self.batch_processor.template_variables.update(values)

    def register_directive(self, name, implementation: typing.Callable[[typing.List[str]], None]):
        if self.batch_processor is not None:

            if name in self.batch_processor.directives:
                raise RuntimeError(
                    f'directive name "{name}" cannot be registered by plugin '
                    f'"{self.loaded_by_name}" because that directive name already exists.')

            self.batch_processor.directives[name] = implementation

    @property
    def render_loop(self) -> typing.Optional[_renderloop.RenderLoop]:
        """
        Provides access to the currently instantiated :py:class:`dgenerate.renderloop.RenderLoop` object.

        This object will have been used for any previous invocation of dgenerate in a config file.
        """
        return self.__render_loop

    @property
    def batch_processor(self) -> typing.Optional[_batchprocessor.BatchProcessor]:
        """
        Provides access to the currently instantiated :py:class:`dgenerate.batchprocess.BatchProcessor` object
        running the config file that this directive is being invoked in.
        """
        return self.__batch_processor

    @property
    def plugin_module_paths(self) -> typing.List[str]:
        """
        List of plugin module paths if they were injected into the batch process by ``-pm/--plugin-modules``
        :return: a list of strings, may be empty but not ``None``
        """

        return self.__plugin_module_paths


__all__ = _types.module_all()
