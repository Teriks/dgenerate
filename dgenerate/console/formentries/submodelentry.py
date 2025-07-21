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

import tkinter as tk
import dgenerate.console.formentries.pluginschemaentry as _schemaentry
import dgenerate.console.resources as _resources
import dgenerate.console.formentries.entry as _entry
import dgenerate.console.formentries.quantizerurientry as _quantizerurientry


class _SubModelEntry(_schemaentry._PluginSchemaEntry):
    NAME = 'submodel'

    def __init__(self, *args, **kwargs):

        self._file_in_arguments = {
            'model': _resources.get_file_dialog_args(['models'])
        }

        self._dir_in_arguments = {
            'model'
        }

        # models that cannot load from a single file
        self._models_no_file = {
            'Image Encoder',
            'UNet'
        }

        # models that cannot load from a directory
        self._models_no_dir = {
            'Adetailer Detector'
        }

        super().__init__(
            *args,
            label='Sub Model',
            help_button=True,
            schema_help_node='HELP',
            schema=_resources.get_schema('submodels'),
            **kwargs)

    def _format_uri(self):
        model_name = self.plugin_name_var.get()
        if not model_name:
            return ''

        uri_parts = []
        first_item = True

        for param_name, (_, variable, default_value, _) in self.entries.items():
            current_value = self._normalize_value(variable.get())
            if current_value != self._normalize_value(default_value) and current_value.strip():
                if not first_item:
                    uri_parts.append(f"{param_name}={_entry.shell_quote_if(current_value, strict=True)}")
                else:
                    uri_parts.append(_entry.shell_quote_if(current_value, strict=False))
                    first_item = False

        return ';'.join(uri_parts)


    def _create_entry_single_type(self,
                                  param_name: str,
                                  param_type: str,
                                  default_value: typing.Any,
                                  optional: bool,
                                  options: list | None,
                                  row: int) -> _schemaentry._PluginArgEntry:

        if options:
            return self._create_dropdown_entry(options, default_value, optional, row)

        created_simple_type, entry = self._create_int_float_bool_entries(
            param_type, default_value, optional, row
        )

        if created_simple_type:
            return entry
        elif 'quantizer' in param_name:
            return self._create_quantizer_entry(row)
        else:
            return self._create_raw_type_entry(
                    param_type, default_value, optional, options, row
                )