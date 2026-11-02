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

import sys
import tkinter as tk
import tkinter.ttk as ttk

import dgenerate.console.combobox as _combobox
import dgenerate.console.formentries.entry as _entry
import dgenerate.console.textentry as _t_entry
from dgenerate.console.spinbox import FloatSpinbox, IntSpinbox


class _FeatureToggleEntry(_entry._Entry):
    """Checkbox that reveals option rows and writes only values the user changed."""

    def _build(self, label, option):
        self._feature_option = option
        self.enabled_var = tk.BooleanVar(value=self.config.get('default', False))
        self.bool_var = self.enabled_var
        self._next_row = self.row + 1
        self._entries = []
        self._spin_buttons = []
        self._child_labels = []
        self._child_checks = []
        self._readonly = []
        self._help_labels = []

        self.label_widget = ttk.Label(
            self.master, text=self.get_label(label), anchor='e')
        self.checkbox = ttk.Checkbutton(self.master, variable=self.enabled_var)
        self.label_widget.grid(row=self.row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.checkbox.grid(row=self.row, column=1, sticky='w')

    def _take_row(self):
        row = self._next_row
        self._next_row += 1
        return row

    def _place(self, row, label, entry, option, spin_buttons=None, check=None, readonly=False):
        label_widget = ttk.Label(self.master, text=label, anchor='e')
        label_widget.grid(row=row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        entry.grid(row=row, column=1, padx=_entry.ROW_XPAD, sticky='ew')
        self._child_labels.append(label_widget)
        self._entries.append(entry)
        if spin_buttons is not None:
            spin_buttons.grid(row=row, column=2, sticky='w')
            self._spin_buttons.append(spin_buttons)
        if check is not None:
            self._child_checks.append(check)
        if readonly:
            self._readonly.append(entry)
        if option:
            self._help_labels.append((label_widget, option))
        return entry

    def _add_int_row(self, label, text_var, option, from_=0, to=sys.maxsize):
        row = self._take_row()
        entry = IntSpinbox(self.master, textvariable=text_var, from_=from_, to=to)
        spin_buttons = entry.create_spin_buttons(self.master)
        text_var.trace_add('write', lambda *_args: self.valid())
        return self._place(row, label + ' (int)', entry, option, spin_buttons=spin_buttons)

    def _add_float_row(self, label, text_var, option, from_=0.0, to=float('inf')):
        row = self._take_row()
        entry = FloatSpinbox(
            self.master, from_=from_, to=to, increment=0.01, textvariable=text_var)
        spin_buttons = entry.create_spin_buttons(self.master)
        text_var.trace_add('write', lambda *_args: self.valid())
        return self._place(row, label + ' (float)', entry, option, spin_buttons=spin_buttons)

    def _add_string_row(self, label, text_var, option):
        row = self._take_row()
        entry = _t_entry.TextEntry(self.master, textvariable=text_var)
        return self._place(row, label, entry, option)

    def _add_dropdown_row(self, label, text_var, option, values):
        row = self._take_row()
        entry = _combobox.ComboBox(self.master, textvariable=text_var, values=values)
        return self._place(row, label, entry, option, readonly=True)

    def _add_check_row(self, label, variable, option):
        row = self._take_row()
        check = ttk.Checkbutton(self.master, variable=variable)
        return self._place(row, label, check, option, check=check)

    def _finish(self):
        if self._next_row - self.row != self.widget_rows:
            raise RuntimeError(
                f'{type(self).__name__} reserved {self.widget_rows} rows '
                f'and used {self._next_row - self.row}.')
        self.enabled_var.trace_add('write', self._update_controls)
        self._update_controls()

        if hasattr(self.form, 'get_entry_by_id'):
            exclude = self.config.get('exclude', None)
            if isinstance(exclude, str):
                exclude = [exclude]
            for other_id in exclude or []:
                other_entry = self.form.get_entry_by_id(other_id)
                if other_entry is not None:
                    self.exclude(other_entry)

    def argument_help_targets(self):
        return [(self.label_widget, self._feature_option), *self._help_labels]

    def sync_widget_states(self):
        self._update_controls()

    def _update_controls(self, *_args):
        enabled = self.enabled_var.get()
        state = tk.NORMAL if enabled else tk.DISABLED
        for entry in self._entries:
            if entry in self._readonly:
                entry.config(state='readonly' if enabled else tk.DISABLED)
            elif entry not in self._child_checks:
                entry.config(state=state)
        for spin_buttons in self._spin_buttons:
            for child in spin_buttons.children.values():
                child.config(state=state)
        mark = ['!disabled'] if enabled else ['disabled']
        for label in self._child_labels:
            label.state(mark)
        for check in self._child_checks:
            check.state(mark)
        if not enabled:
            self.valid()

    def is_valid(self):
        if not self.enabled_var.get():
            return True
        return all(self._entry_valid(entry) for entry in self._entries)

    @staticmethod
    def _entry_valid(entry):
        check = getattr(entry, 'is_valid', None)
        return True if check is None else check()

    def is_active(self):
        return self.enabled_var.get()

    def invalid(self):
        for entry in self._entries:
            if not self._entry_valid(entry):
                _entry.invalid_colors(entry)

    def valid(self):
        for entry in self._entries:
            _entry.valid_colors(entry)

    def exclude(self, other_entry):
        other_var = getattr(other_entry, 'bool_var', None)
        if other_var is None:
            other_vars = getattr(other_entry, 'bool_vars', None)
            if other_vars:
                other_var = other_vars[0]
        if other_var is None:
            other_var = getattr(other_entry, 'enabled_var', None)
        if other_var is None:
            return

        self.enabled_var.trace_add(
            'write', lambda *_args: other_var.set(False) if self.enabled_var.get() else None)
        other_var.trace_add(
            'write', lambda *_args: self.enabled_var.set(False) if other_var.get() else None)

    def _append_int(self, arguments, option, variable, default):
        self._append_number(arguments, option, variable, default, int)

    def _append_float(self, arguments, option, variable, default):
        self._append_number(arguments, option, variable, default, float)

    @staticmethod
    def _append_number(arguments, option, variable, default, cast):
        value = variable.get().strip()
        if value == '' or value == '-':
            return
        if default is not None and cast(value) == default:
            return
        arguments.append(option + ' ' + value)

    @staticmethod
    def _append_text(arguments, option, variable, default):
        value = variable.get().strip()
        if not value or value == default:
            return
        arguments.append(option + ' ' + _entry.shell_quote_if(value))

    def _feature_arguments(self):
        return []

    def template(self, content):
        if not self.enabled_var.get():
            return _entry.replace_first(content, self.placeholder, '')
        # Any related option enables the feature, so the parent flag is only
        # needed when every option is still at its default.
        arguments = list(self._feature_arguments())
        if not arguments:
            arguments = [self._feature_option]
        return _entry.replace_first(content, self.placeholder, '\n'.join(arguments))
