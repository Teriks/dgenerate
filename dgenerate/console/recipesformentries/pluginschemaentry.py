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
from dgenerate.console.mousewheelbind import bind_mousewheel, un_bind_mousewheel
from dgenerate.console.spinbox import FloatSpinbox, IntSpinbox


class _PluginArgEntry:
    def __init__(self,
                 raw: bool,
                 widgets: list[tk.Widget],
                 variable: tk.Variable,
                 file_types: dict | None = None,
                 file_out: bool = False,
                 directory: bool = False,
                 widget_rows: int = 1,
                 widgets_delete: typing.Optional[typing.Callable] = None):
        """
        :param raw: Display acceptable types to the user in the label?
        :param widgets: Associated controls.
        :param variable: Tk.Variable associated with the entry
        :param file_types: File select dialog file types
        :param file_out: Output file dialog?
        :param directory: Directory select?
        :param widget_rows: The entry uses up this many grid rows.
        :param widgets_delete: dynamic widgets deletion callback, use this
            if the entry creates dynamic widgets via interaction that are
            not in "widgets" initially
        """
        self.raw = raw
        self.widgets = widgets
        self.variable = variable
        self.file_types = file_types
        self.file_out = file_out
        self.directory = directory
        self.widget_rows = widget_rows
        self.widgets_delete = widgets_delete


class _PluginSchemaEntry(_entry._Entry):
    def __init__(self,
                 *args,
                 label: str,
                 help_button: bool,
                 schema: dict,
                 schema_help_node: str = 'HELP',
                 hidden_args: set | None = None,
                 max_additional_rows: int | None = None,
                 dropdown_parent: tk.Widget | None = None,
                 dropdown_row: int | None = None,
                 **kwargs):
        self.schema = schema

        config = kwargs.get('config', {})

        self.internal_divider = config.get(
            'internal-divider', not config.get('divider-after', False))

        self.plugin_filter = config.get(
            'filter', None
        )

        if len(self.schema) == 0:
            max_rows = 1
        else:
            max_rows = (max(len(args) for args in self.schema.values()) +
                        (1 if not self.internal_divider else 2) +
                        (max_additional_rows if max_additional_rows is not None else 0))

        super().__init__(*args, **kwargs, widget_rows=max_rows)

        self.declared_optional = self.config.get('optional', True)

        self._label = label

        self._has_help_button = help_button
        self._schema_help_node = schema_help_node
        self._hidden_args = hidden_args if hidden_args is not None else set()

        label = f'(Optional) {self.config.get("label", label)}' \
            if self.declared_optional else self.config.get("label", label)

        self.optional = False
        self.plugin_name_var = tk.StringVar(self.master)

        if self.plugin_filter:
            values = list(k for k in self.schema.keys() if k in self.plugin_filter)
        else:
            values = list(self.schema.keys())

        self.dropdown_label = None
        self.plugin_dropdown = None
        self.plugin_help_button = None
        self.current_help_text = ''

        self._dropdown_row = self.row if not dropdown_parent else dropdown_row
        self._dropdown_parent = dropdown_parent if dropdown_parent else self.master

        self._create_dropdown(label, values)

        self.entries = {}
        self.dynamic_widgets = []
        self._widget_delete_callbacks = []

        self.on_updated_callback = None

        self._last_known_dropdown_value = None

        if not self.declared_optional:
            self._on_plugin_change(self.plugin_name_var.get())

    def _create_dropdown(self, label: str, values: list[str]):
        self.dropdown_label = tk.Label(self._dropdown_parent, text=label)

        if self.declared_optional:
            self.plugin_dropdown = tk.OptionMenu(
                self._dropdown_parent,
                self.plugin_name_var, '', *values,
                command=lambda s: self._on_plugin_change(str(s)))
        else:
            self.plugin_name_var.set(values[0])

            if self._has_help_button:
                self.current_help_text = self.schema[values[0]][self._schema_help_node]

            self.plugin_dropdown = tk.OptionMenu(
                self._dropdown_parent, self.plugin_name_var, *values,
                command=lambda s: self._on_plugin_change(str(s)))

        if len(self.schema) == 0:
            self.plugin_dropdown.config(bg='darkgray')
            self.plugin_dropdown.configure(state=tk.DISABLED)

        if self._has_help_button:
            self.plugin_help_button = tk.Button(self._dropdown_parent, text='Help', command=self._show_help)

        self.dropdown_label.grid(row=self._dropdown_row, column=0, padx=_entry.ROW_XPAD, sticky="e")
        self.plugin_dropdown.grid(row=self._dropdown_row, column=1, padx=_entry.ROW_XPAD, sticky="ew")

        if not self.declared_optional:
            self._show_help_button()

    def primary_widgets(self):
        """
        Dropdown label, dropdown, help button if active
        :return: list of widgets
        """
        return [self.dropdown_label,
                self.plugin_dropdown] + (
            [self.plugin_help_button] if self.plugin_help_button else [])

    def destroy_dynamic_widgets(self):
        """
        Destroy any dynamically created widgets
        """
        for widget in self.dynamic_widgets:
            widget.destroy()
        for callback in self._widget_delete_callbacks:
            callback()
        self._widget_delete_callbacks.clear()
        self.dynamic_widgets.clear()
        self.entries.clear()

    def _on_plugin_change(self, selected_value: str):
        if self._last_known_dropdown_value == selected_value:
            return
        else:
            self._last_known_dropdown_value = selected_value

        self.destroy_dynamic_widgets()

        algorithm_name = self.plugin_name_var.get()
        if not algorithm_name:
            self._hide_help_button()
            return

        self._show_help_button()
        parameters = self.schema.get(algorithm_name, {})

        row_offset = 1

        for param_name, param_info in parameters.items():
            if param_name == self._schema_help_node:
                self.current_help_text = param_info
                continue

            if param_name in self._hidden_args:
                continue

            entry = self._create_widget_for_param(
                param_name, param_info, self.row + row_offset)

            row_offset += entry.widget_rows

        if self.internal_divider:
            separator = ttk.Separator(self.master, orient='horizontal')
            separator.grid(row=self.row + row_offset, column=0, sticky='ew', columnspan=100,
                           pady=_entry.DIVIDER_YPAD)
            self.dynamic_widgets.append(separator)

        if self.on_updated_callback is not None:
            self.on_updated_callback()

    def _add_file_in_button(self, row, entry, dialog_args):

        file_button = tk.Button(self.master, text='Select File',
                                command=lambda e=entry, d=dialog_args: self._select_in_file_command(e, d))
        file_button.grid(row=row, column=2, padx=_entry.ROW_XPAD, sticky='w')
        self.dynamic_widgets.append(file_button)

    def _add_file_out_button(self, row, entry, dialog_args):

        file_button = tk.Button(self.master, text='Save File',
                                command=lambda e=entry, d=dialog_args: self._select_out_file_command(e, d))
        file_button.grid(row=row, column=2, padx=_entry.ROW_XPAD, sticky='w')
        self.dynamic_widgets.append(file_button)

    def _add_directory_in_button(self, row, entry):

        file_button = tk.Button(self.master, text='Select Directory',
                                command=lambda e=entry: self._select_in_directory_command(e))
        file_button.grid(row=row, column=2, padx=_entry.ROW_XPAD, sticky='w')
        self.dynamic_widgets.append(file_button)

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
    def _select_in_directory_command(entry):
        file_path = _filedialog.open_directory_dialog()
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    def _create_dropdown_entry(self, values: list[str], default_value: typing.Any, optional: bool, row: int) -> _PluginArgEntry:
        if default_value:
            variable = tk.StringVar(value=str(default_value))
        else:
            variable = tk.StringVar(value='')
        if optional:
            values = [''] + values
        entry = tk.OptionMenu(self.master, variable, *values)
        entry.grid(row=row, column=1, sticky='we', padx=_entry.ROW_XPAD)
        return _PluginArgEntry(raw=False, widgets=[entry], variable=variable)

    def _create_int_float_bool_entries(self, param_type: str, default_value: typing.Any, optional: bool, row: int):
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

            return True, _PluginArgEntry(raw=True, widgets=[entry, spin_buttons], variable=variable)

        elif param_type == 'bool' and default_value != "":
            if optional:
                variable = tk.StringVar(value=str(default_value))
                values = ['True', 'False', 'None']

                entry = tk.OptionMenu(self.master, variable, *values)

                entry.grid(row=row, column=1, sticky='we', padx=_entry.ROW_XPAD)
                return True, _PluginArgEntry(raw=False, widgets=[entry], variable=variable)
            else:
                variable = tk.BooleanVar(value=default_value)
                entry = tk.Checkbutton(self.master, variable=variable)
                entry.grid(row=row, column=1, sticky='w')
                return True, _PluginArgEntry(raw=False, widgets=[entry], variable=variable)

        return False, None

    def _create_raw_type_entry(self, param_type: str, default_value: typing.Any, optional: bool, row: int):
        default_value = str(default_value)
        variable = tk.StringVar(
            value=default_value if default_value != 'None' else '')
        entry = tk.Entry(self.master, textvariable=variable)
        entry.grid(row=row, column=1, sticky='we', padx=_entry.ROW_XPAD)
        return _PluginArgEntry(raw=True, widgets=[entry], variable=variable)

    def _create_entry_multi_type(self,
                                 param_name: str,
                                 param_types: list[str],
                                 default_value: typing.Any,
                                 optional: bool,
                                 row: int
                                 ) -> _PluginArgEntry:
        if len(param_types) == 1:
            return self._create_entry_single_type(param_name, param_types[0], default_value, optional, row)
        else:
            return self._create_raw_type_entry(param_types[0], default_value, optional, row)

    def _create_entry_single_type(self,
                                  param_name: str,
                                  param_type: str,
                                  default_value: typing.Any,
                                  optional: bool,
                                  row: int
                                  ) -> _PluginArgEntry:

        created_simple_type, entry = self._create_int_float_bool_entries(param_type, default_value, optional, row)
        if created_simple_type:
            return entry
        else:
            return self._create_raw_type_entry(param_type, default_value, optional, row)

    def _create_widget_for_param(self, param_name: str, param_info: dict, row: int) -> _PluginArgEntry:
        optional: bool = param_info.get('optional', False)
        default_value: typing.Any = param_info.get('default', "")
        param_types: list[str] = param_info['types']

        arg_entry = self._create_entry_multi_type(param_name, param_types, default_value, optional, row)

        if arg_entry.widgets_delete is not None:
            self._widget_delete_callbacks.append(arg_entry.widgets_delete)

        entry = arg_entry.widgets[0]

        optional_label = '(Optional) ' if optional else ''

        label_text = \
            f"{optional_label}{param_name} ({', '.join(param_info['types'])})" if arg_entry.raw else \
                f"{optional_label}{param_name}"

        label = tk.Label(self.master, text=label_text)
        self.dynamic_widgets.append(label)
        label.grid(row=row, column=0, padx=_entry.ROW_XPAD, sticky="e")

        for widget in arg_entry.widgets:
            self.dynamic_widgets.append(widget)

        if arg_entry.file_types:
            if arg_entry.file_out:
                self._add_file_out_button(row, entry, arg_entry.file_types)
            else:
                self._add_file_in_button(row, entry, arg_entry.file_types)
        elif arg_entry.directory:
            self._add_directory_in_button(row, entry)

        # editing resets invalid border
        arg_entry.variable.trace_add('write', lambda *a, e=entry: e.config(highlightthickness=0))

        self.entries[param_name] = (entry, arg_entry.variable, default_value, optional)

        return arg_entry

    def _show_help(self):
        top = tk.Toplevel(
            self.recipe_form.master if self.recipe_form.master else self.recipe_form
        )

        top.title(f'{self._label} Help: {self.plugin_name_var.get()}')
        top.attributes('-topmost', 1)
        top.attributes('-topmost', 0)

        frame = tk.Frame(top)
        frame.pack(expand=True, fill='both')

        v_scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        h_scrollbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        text_widget = tk.Text(frame, wrap=tk.NONE, state='disabled', yscrollcommand=v_scrollbar.set,
                              xscrollcommand=h_scrollbar.set)

        text_widget.configure(**_resources.get_textbox_theme())

        text_widget.config(state='normal')
        text_widget.insert(tk.END, self.current_help_text + '\n\n')
        text_widget.config(state='disabled')

        v_scrollbar.config(command=text_widget.yview)
        h_scrollbar.config(command=text_widget.xview)
        text_widget.pack(expand=True, fill='both')

        width = 850
        height = 600

        # Centering logic

        # master_x = self.recipe_form.winfo_rootx()
        # master_y = self.recipe_form.winfo_rooty()
        # master_width = self.recipe_form.winfo_width()
        # master_height = self.recipe_form.winfo_height()

        # # Calculate position x, y
        # x = master_x + (master_width // 2) - (width // 2)
        # y = master_y + (master_height // 2) - (height // 2)

        # Dock right
        master_x = self.recipe_form.winfo_rootx()
        master_y = self.recipe_form.winfo_rooty()
        master_width = self.recipe_form.winfo_width()

        # Calculate position x, y to dock the help window to the right of recipe_form
        x = master_x + master_width
        y = master_y

        top.geometry(f'{width}x{height}+{x}+{y}')

        bind_mousewheel(top.bind, self._on_help_mouse_wheel)

        og_destroy = top.destroy

        def new_destroy():
            un_bind_mousewheel(top.bind)
            og_destroy()

        top.destroy = new_destroy

    @staticmethod
    def _on_help_mouse_wheel(event):
        return "break"

    def _show_help_button(self):
        if self.plugin_help_button is not None:
            self.plugin_help_button.grid(row=self.row, column=2, padx=_entry.ROW_XPAD, sticky="w")

    def _hide_help_button(self):
        if self.plugin_help_button is not None:
            self.plugin_help_button.grid_forget()

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
    def _normalize_value(value):
        str_val = str(value)
        return {'none': 'None', 'true': 'True', 'false': 'False'}.get(
            str_val.lower(), str_val)

    def _format_uri(self):
        algorithm_name = self.plugin_name_var.get()
        if not algorithm_name:
            return ''
        uri_parts = [algorithm_name]
        for param_name, (_, variable, default_value, _) in self.entries.items():
            current_value = self._normalize_value(variable.get())
            if current_value != self._normalize_value(default_value) and current_value.strip():
                uri_parts.append(f"{param_name}={_entry.shell_quote_if(current_value, strict=True)}")
        return ';'.join(uri_parts)

    def template(self, content):
        return self._template(content, self._format_uri())
