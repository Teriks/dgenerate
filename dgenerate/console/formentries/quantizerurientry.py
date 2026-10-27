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
import tkinter.ttk as ttk

import dgenerate.console.appearance as _appearance
import dgenerate.console.formentries.entry as _entry
import dgenerate.console.formentries.pluginschemaentry as _schemaentry
import dgenerate.console.quantizermapselect as _quantizer_map_select
import dgenerate.console.resources as _resources
import dgenerate.console.textentry as _t_entry


class _QuantizerEntry(_schemaentry._PluginSchemaEntry):
    NAME = 'quantizer'

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            label='Quantizer',
            help_button=True,
            schema_help_node='QUANTIZER_HELP',
            schema=_resources.get_schema('quantizers'),
            max_additional_rows=1,
            **kwargs)

        if self.arg is None:
            self.arg = '--quantizer'

        self._map_arg = self.config.get('map-arg')
        if not self._map_arg:
            if self.arg == '--second-model-quantizer':
                self._map_arg = '--second-model-quantizer-map'
            else:
                self._map_arg = '--quantizer-map'

        map_default = self.config.get('map-default', '')
        if not map_default:
            map_default = self.config.get('quantizer-map-default', '')

        map_options = self.config.get('map-options')
        self._map_option_filter = list(map_options) if map_options else None

        self._map_var = tk.StringVar(
            value=_quantizer_map_select.format_quantizer_map_text(
                _quantizer_map_select.parse_quantizer_map_text(map_default)))

        self._map_row = self.row + self.widget_rows - 1

        self._map_label = ttk.Label(
            self.master,
            text='(Optional) Quantizer map',
            anchor='e')
        self._map_entry = _t_entry.TextEntry(self.master, textvariable=self._map_var)
        self._map_button_frame = ttk.Frame(self.master)
        self._map_build_button = ttk.Button(
            self._map_button_frame,
            text='Build',
            command=self._open_map_builder)

        self._map_label.grid(row=self._map_row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self._map_entry.grid(row=self._map_row, column=1, padx=(5, 2), sticky='ew')
        self._map_button_frame.grid(row=self._map_row, column=2, padx=(2, 5), sticky='w')
        self._map_build_button.pack(side='left')
        _appearance.apply(self._map_label)
        _appearance.apply(self._map_entry)
        _appearance.apply(self._map_build_button)

        self.plugin_name_var.trace_add('write', lambda *_: self._sync_map_enabled())
        self._sync_map_enabled()

    def sync_widget_states(self) -> None:
        self._sync_map_enabled()

    def _sync_map_enabled(self) -> None:
        if self._row_inactive:
            return
        state = tk.NORMAL if self.plugin_name_var.get().strip() else tk.DISABLED
        self._map_entry.configure(state=state)
        self._map_build_button.configure(state=state)

    def _map_options(self) -> list[str]:
        options = _quantizer_map_select.quantizer_map_options_for_model_type(
            self.config.get('model-type'))
        if self._map_option_filter:
            allowed = set(self._map_option_filter)
            options = [name for name in options if name in allowed]
        return options

    def _open_map_builder(self):
        def apply_map(value: str):
            self._map_var.set(value)

        model_type = self.config.get('model-type')
        if model_type:
            model_type = str(model_type).strip()
        _quantizer_map_select.open_quantizer_map_select(
            self.form.winfo_toplevel(),
            options=self._map_options(),
            initial_text=self._map_var.get(),
            apply=apply_map,
            model_type=model_type,
        )

    def template(self, content):
        uri = self._format_uri()
        if not uri:
            return self._template(content, '')
        map_value = self._map_var.get().strip()
        extra = ''
        if map_value:
            extra = f' {self._map_arg} {map_value}'
        return self._template(content, uri + extra)
