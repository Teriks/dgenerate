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

import tkinter as tk
import typing

import dgenerate.console.recipesformentries.pluginschemaentry as _schemaentry
import dgenerate.console.recipesformentries.quantizerurientry as _quantizerurientry
import dgenerate.console.resources as _resources


class _PromptUpscalerEntry(_schemaentry._PluginSchemaEntry):
    NAME = 'promptupscaler'

    def __init__(self, *args, **kwargs):

        schema = _resources.get_schema('promptupscalers')
        quantizers = _resources.get_schema('quantizers')

        config = kwargs.get('config', {})

        hidden_args = set()
        if config.get('hide-device', False):
            hidden_args.add('device')

        # we want to be able to pop open a quantizer URI select with dynamic widgets
        # and still be compatible with the recipe form, so grid rows have to be reserved
        # for it, +2, one for the selection dropdown, and one for the divider
        # that comes at the end of the widgets, we need the max amount of rows
        # that will ever be needed when switching through quantizers

        if len(quantizers) != 0:
            max_additional_rows = max(len(args) for args in quantizers.values()) + 2
        else:
            max_additional_rows = 0

        super().__init__(*args,
                         label='Prompt Upscaler',
                         hidden_args=hidden_args,
                         help_button=True,
                         schema_help_node='PROMPT_UPSCALER_HELP',
                         schema=schema,
                         max_additional_rows=max_additional_rows,
                         **kwargs)

        if self.arg is None:
            self.arg = '--prompt-upscaler'

    def _apply_file_dir_selects(self, param_name: str, entry: _schemaentry._PluginArgEntry):
        if param_name == 'model':
            if self.plugin_name_var.get() == 'gpt4all':
                entry.file_types = {'filetypes': [('gguf', ('*.gguf'))]}
            elif self.plugin_name_var.get() == 'magicprompt':
                entry.directory = True
            entry.raw = False
        if param_name == 'cleanup-config':
            if self.plugin_name_var.get() in {'gpt4all', 'magicprompt'}:
                entry.file_types = {'filetypes': [('Cleanup Config', ('*.json', '*.toml', '*.yaml', '*.yml'))]}
                entry.raw = False
        return entry

    def _create_entry_single_type(self,
                                  param_name: str,
                                  param_type: str,
                                  default_value: typing.Any,
                                  optional: bool,
                                  row: int) -> _schemaentry._PluginArgEntry:

        created_simple_type, entry = self._create_int_float_bool_entries(param_type, default_value, optional, row)
        if created_simple_type:
            return entry
        elif 'device' == param_name:
            return self._create_dropdown_entry(
                _resources.get_torch_devices(),
                default_value,
                optional,
                row
            )
        elif 'compute' == param_name and self.plugin_name_var.get() == 'gpt4all':
            return self._create_dropdown_entry(
                _resources.get_gpt4all_compute_devices(),
                default_value,
                optional,
                row
            )
        elif 'part' == param_name:
            values = ['both', 'positive', 'negative']
            return self._create_dropdown_entry(values, default_value, optional, row)
        elif 'quantizer' in param_name:
            return self._create_quantizer_entry(row)
        elif 'dtype' in param_name:
            values = ['float32', 'float16', 'bfloat16']
            return self._create_dropdown_entry(values, default_value, optional, row)
        else:
            return self._apply_file_dir_selects(
                param_name, self._create_raw_type_entry(param_type, default_value, optional, row))

    def _create_quantizer_entry(self, row):
        entry = _quantizerurientry._QuantizerEntry(
            master=self.master,
            row=row,
            recipe_form=self.master,
            placeholder='URI',
            config={'optional': True, 'default': ''}
        )

        entry.arg = None

        class _Var(tk.Variable):
            def get(self) -> str:
                uri_value = entry.template('URI')
                if uri_value:
                    return f"'{uri_value}'"
                else:
                    return ''

            def set(self, value) -> None:
                entry.plugin_name_var.set(value)

        return _schemaentry._PluginArgEntry(
            raw=False,
            widgets=entry.primary_widgets(),
            variable=_Var(),
            widget_rows=entry.widget_rows,
            widgets_delete=entry.destroy_dynamic_widgets
        )
