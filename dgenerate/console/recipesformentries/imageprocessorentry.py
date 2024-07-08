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

import dgenerate.console.filedialog as _filedialog
import dgenerate.console.recipesformentries.entry as _entry
import dgenerate.console.resources as _resources
from dgenerate.console.mousewheelbind import bind_mousewheel
from dgenerate.console.spinbox import IntSpinbox, FloatSpinbox


class _ImageProcessorEntry(_entry._Entry):
    NAME = 'imageprocessor'

    def __init__(self, *args, **kwargs):
        self.schema = _resources.get_schema('imageprocessors')
        max_rows = max(len(args) for args in self.schema.values()) + 1
        super().__init__(*args, **kwargs, widget_rows=max_rows)

        self.declared_optional = self.config.get('optional', True)
        label = f'(Optional) {self.config.get("label", "Image Processor")}' if self.declared_optional else self.config.get(
            "label", "Image Processor")

        self.optional = False
        self.processor_name_var = tk.StringVar(self.master)
        values = list(self.schema.keys())

        self.dropdown_label = tk.Label(self.master, text=label)

        if self.declared_optional:
            self.algorithm_dropdown = tk.OptionMenu(self.master, self.processor_name_var, '', *values,
                                                    command=self._on_processor_change)
        else:
            self.processor_name_var.set(values[0])
            self.algorithm_dropdown = tk.OptionMenu(self.master, self.processor_name_var, *values,
                                                    command=self._on_processor_change)

        self.processor_help_button = tk.Button(self.master, text='Help', command=self._show_help)
        self.dropdown_label.grid(row=self.row, column=0, padx=_entry.ROW_XPAD, sticky="e")
        self.algorithm_dropdown.grid(row=self.row, column=1, padx=_entry.ROW_XPAD, sticky="ew")

        if not self.declared_optional:
            self._show_help_button()

        self.entries = {}
        self.dynamic_widgets = []
        self.file_arguments = {
            'model': [('Models', ' *.'.join(['*.' + ext for ext in _resources.supported_torch_model_formats_open()]))]}

    def _show_help(self):
        top = tk.Toplevel(self.master)
        top.title(f'Image Processor Help: {self.processor_name_var.get()}')
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

        if self.recipe_form.master and hasattr(self.recipe_form.master, '_input_text'):
            # get the colors from the main editor text box
            bg = self.recipe_form.master._input_text.text.cget('bg')
            fg = self.recipe_form.master._input_text.text.cget('fg')
            text_widget.configure(bg=bg, fg=fg)

        text_widget.config(state='normal')
        text_widget.insert(tk.END, self.current_help_text)
        text_widget.config(state='disabled')

        v_scrollbar.config(command=text_widget.yview)
        h_scrollbar.config(command=text_widget.xview)
        text_widget.pack(expand=True, fill='both')

        top.geometry("850x600")

        bind_mousewheel(top.bind, self._on_help_mouse_wheel)

    @staticmethod
    def _on_help_mouse_wheel(event):
        return "break"

    def _show_help_button(self):
        self.processor_help_button.grid(row=self.row, column=2, padx=_entry.ROW_XPAD, sticky="w")

    def _hide_help_button(self):
        self.processor_help_button.grid_forget()

    def _on_processor_change(self, event):
        self._clear_dynamic_widgets()

        algorithm_name = self.processor_name_var.get()
        if not algorithm_name:
            self._hide_help_button()
            return

        self._show_help_button()
        parameters = self.schema.get(algorithm_name, {})

        for i, (param_name, param_info) in enumerate(parameters.items()):
            if param_name == 'PROCESSOR_HELP':
                self.current_help_text = param_info
                continue

            self._create_widget_for_param(param_name, param_info, self.row + i + 1)

    def _clear_dynamic_widgets(self):
        for widget in self.dynamic_widgets:
            widget.destroy()
        self.dynamic_widgets.clear()
        self.entries.clear()

    def _create_widget_for_param(self, param_name, param_info, row):
        optional: bool = param_info.get('optional', False)
        default_value: typing.Any = param_info.get('default', "")
        param_types: list[str] = param_info['types']

        raw, entry, variable, sticky, padx = self._create_entry(param_types, default_value, optional)

        optional_label = '(Optional) ' if optional else ''

        label_text = \
            f"{optional_label}{param_name} ({', '.join(param_info['types'])})" if raw else \
            f"{optional_label}{param_name}"

        label = tk.Label(self.master, text=label_text)
        self.dynamic_widgets.append(label)
        label.grid(row=row, column=0, padx=_entry.ROW_XPAD, sticky="e")

        self.dynamic_widgets.append(entry)
        entry.grid(row=row, column=1, padx=padx, sticky=sticky)

        if param_name in self.file_arguments:
            self._add_file_button(row, entry, self.file_arguments[param_name])

        # editing resets invalid border
        variable.trace_add('write', lambda *a, e=entry: e.config(highlightthickness=0))

        self.entries[param_name] = (entry, variable, default_value, optional)

    def _create_entry(self, param_types, default_value, optional) -> \
            tuple[bool, tk.Widget, tk.Variable, str, tuple[int, int] | None]:
        """
        :param param_types: parameter accepted types, as strings
        :param default_value: default value, possibly empty string
        :param optional: can accept None?
        :return: (raw? label requires type info, entry widget, entry data variable, widget sticky coords, xpad)
        """

        if len(param_types) == 1:
            param_type = param_types[0]
            return self._create_single_type_entry(param_type, default_value, optional)
        else:
            variable = tk.StringVar(value=str(default_value))
            entry = tk.Entry(self.master, textvariable=variable)
            return True, entry, variable, 'we', _entry.ROW_XPAD

    def _create_single_type_entry(self, param_type, default_value, optional) -> \
            tuple[bool, tk.Widget, tk.Variable, str, tuple[int, int] | None]:
        """
        :param param_type: parameter accepted type, as string
        :param default_value: default value, possibly empty string
        :param optional: can accept None?
        :return: (entry widget, entry text variable, widget sticky coords, xpad)
        """

        if param_type in ['int', 'float'] and default_value != "":

            variable = tk.StringVar(value=default_value)
            increment = 1 if param_type == 'int' else 0.01
            entry_class = IntSpinbox if param_type == 'int' else FloatSpinbox
            entry = entry_class(self.master, textvariable=variable, increment=increment,
                                format='%.15f' if param_type == 'float' else None)

            return False, entry, variable, 'we', _entry.ROW_XPAD

        elif param_type == 'bool' and default_value != "":

            if optional:
                variable = tk.StringVar(value=default_value)
                values = ['True', 'False', 'None']
                values.remove(str(default_value))
                entry = tk.OptionMenu(self.master, variable, str(default_value), *values)
                return False, entry, variable, 'we', _entry.ROW_XPAD
            else:
                variable = tk.BooleanVar(value=default_value)
                entry = tk.Checkbutton(self.master, variable=variable)
                return False, entry, variable, 'w', None

        else:

            variable = tk.StringVar(value=str(default_value))
            entry = tk.Entry(self.master, textvariable=variable)
            return True, entry, variable, 'we', _entry.ROW_XPAD

    def _add_file_button(self, row, entry, file_types):

        file_button = tk.Button(self.master, text='Select File',
                                command=lambda e=entry, f=file_types: self._select_model_command(e, f))
        file_button.grid(row=row, column=2, padx=_entry.ROW_XPAD, sticky='w')
        self.dynamic_widgets.append(file_button)

    def invalid(self):
        for entry, variable, default_value, optional in self.entries.values():
            if optional is False and not str(variable.get()):
                entry.config(
                    highlightbackground="red",
                    highlightcolor="red",
                    highlightthickness=2)

    def is_empty(self):
        for entry, variable, default_value, optional in self.entries.values():
            if optional is False and not str(variable.get()):
                return True
        return False

    def valid(self):
        for entry, variable, default_value, optional in self.entries.values():
            if optional:
                entry.config(highlightthickness=0)

    @staticmethod
    def _select_model_command(entry, file_types):
        file_path = _filedialog.open_file_dialog(filetypes=file_types)
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    @staticmethod
    def _normalize_value(value):
        value = str(value).lower()
        return {'none': 'None', 'true': 'True', 'false': 'False'}.get(value, value)

    def _format_uri(self):
        algorithm_name = self.processor_name_var.get()
        if not algorithm_name:
            return ''
        uri_parts = [algorithm_name]
        for param_name, (entry, variable, default_value, _) in self.entries.items():
            current_value = self._normalize_value(variable.get())
            if current_value != self._normalize_value(default_value) and current_value.strip():
                uri_parts.append(f"{param_name}={current_value}")
        return ';'.join(uri_parts)

    def template(self, content):
        return self._template(content, self._format_uri())
