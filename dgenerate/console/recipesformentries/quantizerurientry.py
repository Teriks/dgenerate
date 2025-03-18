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

import dgenerate.console.recipesformentries.pluginschemaentry as _schemaentry
import dgenerate.console.resources as _resources


class _QuantizerEntry(_schemaentry._PluginSchemaEntry):
    NAME = 'quantizer'

    def __init__(self, *args, **kwargs):

        schema = _resources.get_schema('quantizers')

        super().__init__(
            *args,
            label='Quantizer',
            help_button=False,
            schema=schema,
            **kwargs)

        if self.arg is None:
            self.arg = '--quantizer'

    def _create_entry_single_type(self,
                                  param_name: str,
                                  param_type: str,
                                  default_value: typing.Any,
                                  optional: bool,
                                  row: int) -> _schemaentry._PluginArgEntry:
        if param_name == 'bits' and self.plugin_name_var.get() == 'bnb':
            values = ['8', '4']
            return self._create_dropdown_entry(values, default_value, optional, row)
        if 'dtype' in param_name or 'storage' in param_name and self.plugin_name_var.get() == 'bnb':
            values = ['float16', 'bfloat16', 'float32']
            return self._create_dropdown_entry(values, default_value, optional, row)
        if 'quant-type' in param_name and self.plugin_name_var.get() == 'bnb':
            values = ['fp4', 'nf4']
            return self._create_dropdown_entry(values, default_value, optional, row)
        else:
            created_simple_type, entry = self._create_int_float_bool_entries(param_type, default_value, optional, row)
            if created_simple_type:
                return entry
            return self._create_raw_type_entry(param_type, default_value, optional, row)
