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
import typing

import dgenerate.console.filedialog as _filedialog
import dgenerate.console.recipesformentries.entry as _entry
import dgenerate.console.resources as _resources
from dgenerate.console.spinbox import IntSpinbox, FloatSpinbox


class _KarrasSchedulerEntry(_entry._Entry):
    NAME = 'karrasscheduler'

    def __init__(self, *args, **kwargs):
        self.schema = _resources.get_schema('karrasschedulers')

        config = kwargs.get('config', {})

        self.internal_divider = config.get(
            'internal-divider', not config.get('divider-after', False))

        max_rows = max(len(args) for args in self.schema.values()) + (1 if not self.internal_divider else 2)

        super().__init__(*args, **kwargs, widget_rows=max_rows)

        if self.arg is None:
            self.arg = '--scheduler'

        self.declared_optional = self.config.get('optional', True)

        label = f'(Optional) {self.config.get("label", "Scheduler")}' \
            if self.declared_optional else self.config.get("label", "Scheduler")

        self.optional = False
        self.scheduler_name_var = tk.StringVar(self.master)
        values = list(self.schema.keys())

        self.dropdown_label = tk.Label(self.master, text=label)

        if self.declared_optional:
            self.algorithm_dropdown = tk.OptionMenu(self.master, self.scheduler_name_var, '', *values,
                                                    command=self._on_scheduler_change)
        else:
            self.scheduler_name_var.set(values[0])
            self.algorithm_dropdown = tk.OptionMenu(self.master, self.scheduler_name_var, *values,
                                                    command=self._on_scheduler_change)

        self.dropdown_label.grid(row=self.row, column=0, padx=_entry.ROW_XPAD, sticky="e")
        self.algorithm_dropdown.grid(row=self.row, column=1, padx=_entry.ROW_XPAD, sticky="ew")

        self.entries = {}
        self.dynamic_widgets = []

        self.on_updated_callback = None

        if not self.declared_optional:
            self._on_scheduler_change(None)

    def _on_scheduler_change(self, event):
        self._clear_dynamic_widgets()

        algorithm_name = self.scheduler_name_var.get()
        if not algorithm_name:
            return

        parameters = self.schema.get(algorithm_name, {})

        i = 0

        for i, (param_name, param_info) in enumerate(parameters.items()):
            self._create_widget_for_param(param_name, param_info, self.row + i + 1)

        if self.internal_divider:
            separator = ttk.Separator(self.master, orient='horizontal')
            separator.grid(row=self.row + i + 2, column=0, sticky='ew', columnspan=100, pady=_entry.DIVIDER_YPAD)
            self.dynamic_widgets.append(separator)

        if self.on_updated_callback is not None:
            self.on_updated_callback()

    def _clear_dynamic_widgets(self):
        for widget in self.dynamic_widgets:
            widget.destroy()
        self.dynamic_widgets.clear()
        self.entries.clear()

    def _create_widget_for_param(self, param_name, param_info, row):
        optional: bool = param_info.get('optional', False)
        default_value: typing.Any = param_info.get('default', "")
        param_types: list[str] = param_info['types']

        raw, widgets, variable = self._create_entry(param_name, param_types, default_value, optional, row)

        entry = widgets[0]

        optional_label = '(Optional) ' if optional else ''

        label_text = \
            f"{optional_label}{param_name} ({', '.join(param_info['types'])})" if raw else \
            f"{optional_label}{param_name}"

        label = tk.Label(self.master, text=label_text)
        self.dynamic_widgets.append(label)
        label.grid(row=row, column=0, padx=_entry.ROW_XPAD, sticky="e")

        for widget in widgets:
            self.dynamic_widgets.append(widget)

        # editing resets invalid border
        variable.trace_add('write', lambda *a, e=entry: e.config(highlightthickness=0))

        self.entries[param_name] = (entry, variable, default_value, optional)

    def _create_entry(self, param_name, param_types, default_value, optional, row) -> \
            tuple[bool, list[tk.Widget], tk.Variable]:
        """
        :param param_name: parameter name
        :param param_types: parameter accepted types, as strings
        :param default_value: default value, possibly empty string
        :param optional: can accept None?
        :return:
            (raw? label requires type info,
            dynamic widgets list first widget should be the entry field,
            entry data variable)
        """

        if len(param_types) == 1:
            param_type = param_types[0]
            return self._create_single_type_entry(param_name, param_type, default_value, optional, row)
        else:
            variable = tk.StringVar(value=str(default_value))
            entry = tk.Entry(self.master, textvariable=variable)
            entry.grid(row=row, column=1, sticky='we', padx=_entry.ROW_XPAD)
            return True, [entry], variable

    def _create_single_type_entry(self, param_name, param_type, default_value, optional, row) -> \
            tuple[bool, list[tk.Widget], tk.Variable]:
        """
        :param param_name: parameter name
        :param param_type: parameter accepted type, as string
        :param default_value: default value, possibly empty string
        :param optional: can accept None?
        :return:
            (raw? label requires type info,
            dynamic widgets list first widget should be the entry field,
            entry data variable)
        """

        if param_type in ['int', 'float']:

            variable = tk.StringVar(value=default_value)
            increment = 1 if param_type == 'int' else 0.01
            entry_class = IntSpinbox if param_type == 'int' else FloatSpinbox

            entry = entry_class(self.master,
                                textvariable=variable,
                                increment=increment)

            spin_buttons = entry.create_spin_buttons(self.master)

            entry.grid(row=row, column=1, sticky='we', padx=_entry.ROW_XPAD)
            spin_buttons.grid(row=row, column=2, sticky='w')

            return True, [entry, spin_buttons], variable

        elif param_type == 'bool' and default_value != "":
            if optional:
                variable = tk.StringVar(value=str(default_value))
                values = ['True', 'False', 'None']

                entry = tk.OptionMenu(self.master, variable, *values)

                entry.grid(row=row, column=1, sticky='we', padx=_entry.ROW_XPAD)
                return False, [entry], variable
            else:
                variable = tk.BooleanVar(value=default_value)
                entry = tk.Checkbutton(self.master, variable=variable)
                entry.grid(row=row, column=1, sticky='w')
                return False, [entry], variable

        elif param_name == 'prediction-type':
            default_value = str(default_value)
            variable = tk.StringVar(value=default_value)
            entry = tk.OptionMenu(
                self.master, variable, *_resources.get_karras_scheduler_prediction_types())
            entry.grid(row=row, column=1, sticky='we', padx=_entry.ROW_XPAD)
            return True, [entry], variable
        else:
            default_value = str(default_value)
            variable = tk.StringVar(
                value=default_value if default_value != 'None' else '')
            entry = tk.Entry(self.master, textvariable=variable)
            entry.grid(row=row, column=1, sticky='we', padx=_entry.ROW_XPAD)
            return True, [entry], variable

    def invalid(self):
        for entry, variable, _, optional in self.entries.values():
            if not optional and not str(variable.get()):
                _entry.invalid_colors(entry)

            if isinstance(entry, FloatSpinbox) and not entry.is_valid():
                _entry.invalid_colors(entry)

    def is_empty(self):
        for _, variable, _, optional in self.entries.values():
            if not optional and not str(variable.get()):
                return True
        return False

    def is_valid(self):
        for entry, variable, _, optional in self.entries.values():
            if not optional and not str(variable.get()):
                return False
            if isinstance(entry, FloatSpinbox) and not entry.is_valid():
                return False
        return True

    def valid(self):
        for entry, variable, _, optional in self.entries.values():
            if not optional and str(variable.get()):
                _entry.valid_colors(entry)

            if isinstance(entry, FloatSpinbox) and entry.is_valid():
                _entry.valid_colors(entry)

    @staticmethod
    def _select_in_file_command(entry, dialog_args):
        file_path = _filedialog.open_file_dialog(**dialog_args)
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    @staticmethod
    def _select_out_file_command(entry, dialog_args):
        file_path = _filedialog.open_file_save_dialog(**dialog_args)
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    @staticmethod
    def _normalize_value(value):
        str_val = str(value)
        return {'none': 'None', 'true': 'True', 'false': 'False'}.get(
            str_val.lower(), str_val)

    def _format_uri(self):
        algorithm_name = self.scheduler_name_var.get()
        if not algorithm_name:
            return ''
        uri_parts = [algorithm_name]
        for param_name, (_, variable, default_value, _) in self.entries.items():
            current_value = self._normalize_value(variable.get())
            if current_value != self._normalize_value(default_value) and current_value.strip():
                uri_parts.append(f"{param_name}={_entry.shell_quote_if(current_value)}")
        return ';'.join(uri_parts)

    def template(self, content):
        return self._template(content, self._format_uri())
