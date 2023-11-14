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

import dgenerate.batchprocess.batchprocessordirective as _batchprocessordirective
import dgenerate.plugin as _plugin
import dgenerate.types as _types


class BatchProcessorDirectivePluginNotFoundError(Exception):
    """
    Thrown when :py:class:`.DirectiveLoader` cannot find any :py:class:`dgenerate.batchprocess.BatchProcessorDirective`
    implementation for a specified name.
    """
    pass


class BatchProcessorDirectivePluginArgumentError(Exception):
    """
    Thrown when a :py:class:`dgenerate.batchprocess.BatchProcessorDirective` plugin is
    not instantiated with correct arguments.
    """
    pass


class DirectiveLoader(_plugin.PluginLoader):
    """
    Loads :py:class:`dgenerate.batchprocess.BatchProcessorDirective` plugins.
    """

    def __init__(self):
        """"""

        # The empty string above disables sphinx inherited doc

        super().__init__(base_class=_batchprocessordirective.BatchProcessorDirective,
                         description='directive',
                         reserved_args=[('render-loop',), ('batch-processor',)],
                         argument_error_type=BatchProcessorDirectivePluginArgumentError,
                         not_found_error_type=BatchProcessorDirectivePluginNotFoundError)

        self.add_search_module_string('dgenerate.batchprocess')

    def load(self, uri: _types.Uri, **kwargs) -> _batchprocessordirective.BatchProcessorDirective:
        return typing.cast(_batchprocessordirective.BatchProcessorDirective, super().load(uri, **kwargs))


__all__ = _types.module_all()
