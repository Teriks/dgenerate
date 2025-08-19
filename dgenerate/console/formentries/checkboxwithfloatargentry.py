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

import dgenerate.console.formentries.entry as _entry
from dgenerate.console.spinbox import FloatSpinbox


class _CheckboxWithFloatArgEntry(_entry._Entry):
    NAME = 'checkboxwithfloatarg'

    def __init__(self, *args, **kwargs):
        if 'widget_rows' not in kwargs:
            kwargs['widget_rows'] = 2

        super().__init__(*args, **kwargs)

        # initialize BooleanVar with a default value from config
        self.bool_var = tk.BooleanVar(value=self.config.get('default', False))

        # create label widget
        self.label_widget = tk.Label(self.master,
                                     text=self.get_label(
                                         self.arg if self.arg else 'Switch Option'), anchor='e')

        # create checkbox widget with boolean variable
        self.checkbox = tk.Checkbutton(self.master, variable=self.bool_var)

        # place the widgets in a grid layout
        self.label_widget.grid(row=self.row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.checkbox.grid(row=self.row, column=1, sticky='w')

        self.default_float = str(self.config.get('float', '1.0'))

        self.float_arg = self.config.get('float-arg', 'float')

        self.float_var = tk.StringVar(value=self.default_float)

        self.float_label = tk.Label(
            self.master,
            text=self.get_label(
                'Float Value', key='float-label') + ' (float)',
            anchor='e')

        self.min = self.config.get('min', float('-inf'))
        self.max = self.config.get('max', float('inf'))

        self.float_entry = FloatSpinbox(
            self.master,
            self.float_var,
            from_=self.min,
            to=self.max)

        self.float_var.trace_add('write', lambda *e: self.valid())

        self.spin_buttons = self.float_entry.create_spin_buttons(self.master)
        self.spin_buttons.grid(row=self.row + 1, column=2, sticky='w')

        self.float_label.grid(row=self.row + 1, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.float_entry.grid(row=self.row + 1, column=1, padx=_entry.ROW_XPAD, sticky='ew')

        self.bool_var.trace_add('write', lambda *a: self._check_float_active())

        if not self.bool_var.get():
            self.float_entry.config(state=tk.DISABLED)
            for c in self.spin_buttons.children.values():
                c.config(state=tk.DISABLED)

        # Handle exclude configuration
        if hasattr(self.form, 'get_entry_by_id'):
            exclude: str | list | None = self.config.get('exclude', None)
            if exclude is not None:
                if isinstance(exclude, str):
                    other_entry = self.form.get_entry_by_id(exclude)
                    if hasattr(other_entry, 'exclude'):
                        other_entry.exclude(self)
                else:
                    for other_id in exclude:
                        other_entry = self.form.get_entry_by_id(other_id)
                        if hasattr(other_entry, 'exclude'):
                            other_entry.exclude(self)

    def is_valid(self):
        return self.float_entry.is_valid()

    def invalid(self):
        if not self.float_entry.is_valid():
            _entry.invalid_colors(self.float_entry)

    def valid(self):
        if self.float_entry.is_valid():
            _entry.valid_colors(self.float_entry)

    def _check_float_active(self):
        if self.bool_var.get():
            self.float_entry.config(state=tk.NORMAL)
            for c in self.spin_buttons.children.values():
                c.config(state=tk.NORMAL)
        else:
            self.float_entry.config(state=tk.DISABLED)
            for c in self.spin_buttons.children.values():
                c.config(state=tk.DISABLED)

    def template(self, content):
        new_content = ''

        if self.bool_var.get():
            float_val = self.float_var.get().strip()
            new_content = self.arg if self.arg else ''
            if float_val:
                arg_part = (self.float_arg if self.float_arg else '')
                new_content += ('\n' if self.arg else '') + arg_part + (' ' if arg_part else '') + float_val

        return _entry.replace_first(content, self.placeholder, new_content)

    def exclude(self, other_entry):
        """Set up mutual exclusion with another checkbox entry.
        When this checkbox is checked, the other will be unchecked and vice versa."""
        from dgenerate.console.formentries.argswitchcheckbox import _ArgSwitchCheckbox
        from dgenerate.console.formentries.argswitchconditionalcheckboxes import _ArgSwitchConditionalCheckboxes
        from dgenerate.console.formentries.checkboxwithtwointargsentry import _CheckboxWithTwoIntArgsEntry
        
        if isinstance(other_entry, _ArgSwitchCheckbox):
            # This checkbox excludes simple checkbox
            self.bool_var.trace_add('write', lambda *args: other_entry.bool_var.set(False) if self.bool_var.get() else None)
            other_entry.bool_var.trace_add('write', lambda *args: self.bool_var.set(False) if other_entry.bool_var.get() else None)
            
        elif isinstance(other_entry, _ArgSwitchConditionalCheckboxes):
            # This checkbox excludes the master checkbox of conditional checkboxes
            self.bool_var.trace_add('write', lambda *args: other_entry.bool_vars[0].set(False) if self.bool_var.get() else None)
            other_entry.bool_vars[0].trace_add('write', lambda *args: self.bool_var.set(False) if other_entry.bool_vars[0].get() else None)
            
        elif isinstance(other_entry, _CheckboxWithFloatArgEntry):
            # Both are float arg checkboxes - mutual exclusion
            self.bool_var.trace_add('write', lambda *args: other_entry.bool_var.set(False) if self.bool_var.get() else None)
            other_entry.bool_var.trace_add('write', lambda *args: self.bool_var.set(False) if other_entry.bool_var.get() else None)
            
        elif isinstance(other_entry, _CheckboxWithTwoIntArgsEntry):
            # This checkbox excludes checkbox with two int args
            self.bool_var.trace_add('write', lambda *args: other_entry.bool_var.set(False) if self.bool_var.get() else None)
            other_entry.bool_var.trace_add('write', lambda *args: self.bool_var.set(False) if other_entry.bool_var.get() else None)
