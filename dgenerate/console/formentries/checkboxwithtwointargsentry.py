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

import dgenerate.console.formentries.entry as _entry
from dgenerate.console.spinbox import IntSpinbox


class _CheckboxWithTwoIntArgsEntry(_entry._Entry):
    NAME = 'checkboxwithtwointargs'

    def __init__(self, *args, **kwargs):
        if 'widget_rows' not in kwargs:
            kwargs['widget_rows'] = 3

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

        # First integer argument
        self.default_int1 = str(self.config.get('int1', '0'))
        self.int1_arg = self.config.get('int1-arg', 'int1')
        self.int1_var = tk.StringVar(value=self.default_int1)
        self.int1_min = self.config.get('int1-min', -sys.maxsize)
        self.int1_max = self.config.get('int1-max', sys.maxsize)

        self.int1_label = tk.Label(
            self.master,
            text=self.get_label(
                'First Integer', key='int1-label') + ' (int)',
            anchor='e')

        self.int1_entry = IntSpinbox(
            self.master,
            self.int1_var,
            from_=self.int1_min,
            to=self.int1_max)

        self.int1_var.trace_add('write', lambda *e: self.valid())

        self.int1_spin_buttons = self.int1_entry.create_spin_buttons(self.master)
        self.int1_spin_buttons.grid(row=self.row + 1, column=2, sticky='w')

        self.int1_label.grid(row=self.row + 1, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.int1_entry.grid(row=self.row + 1, column=1, padx=_entry.ROW_XPAD, sticky='ew')

        # Second integer argument
        self.default_int2 = str(self.config.get('int2', '0'))
        self.int2_arg = self.config.get('int2-arg', 'int2')
        self.int2_var = tk.StringVar(value=self.default_int2)
        self.int2_min = self.config.get('int2-min', -sys.maxsize)
        self.int2_max = self.config.get('int2-max', sys.maxsize)

        self.int2_label = tk.Label(
            self.master,
            text=self.get_label(
                'Second Integer', key='int2-label') + ' (int)',
            anchor='e')

        self.int2_entry = IntSpinbox(
            self.master,
            self.int2_var,
            from_=self.int2_min,
            to=self.int2_max)

        self.int2_var.trace_add('write', lambda *e: self.valid())

        self.int2_spin_buttons = self.int2_entry.create_spin_buttons(self.master)
        self.int2_spin_buttons.grid(row=self.row + 2, column=2, sticky='w')

        self.int2_label.grid(row=self.row + 2, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.int2_entry.grid(row=self.row + 2, column=1, padx=_entry.ROW_XPAD, sticky='ew')

        self.bool_var.trace_add('write', lambda *a: self._check_int_active())

        if not self.bool_var.get():
            self.int1_entry.config(state=tk.DISABLED)
            self.int2_entry.config(state=tk.DISABLED)
            for c in self.int1_spin_buttons.children.values():
                c.config(state=tk.DISABLED)
            for c in self.int2_spin_buttons.children.values():
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
        return self.int1_entry.is_valid() and self.int2_entry.is_valid()

    def invalid(self):
        if not self.int1_entry.is_valid():
            _entry.invalid_colors(self.int1_entry)
        if not self.int2_entry.is_valid():
            _entry.invalid_colors(self.int2_entry)

    def valid(self):
        if self.int1_entry.is_valid():
            _entry.valid_colors(self.int1_entry)
        if self.int2_entry.is_valid():
            _entry.valid_colors(self.int2_entry)

    def _check_int_active(self):
        if self.bool_var.get():
            self.int1_entry.config(state=tk.NORMAL)
            self.int2_entry.config(state=tk.NORMAL)
            for c in self.int1_spin_buttons.children.values():
                c.config(state=tk.NORMAL)
            for c in self.int2_spin_buttons.children.values():
                c.config(state=tk.NORMAL)
        else:
            self.int1_entry.config(state=tk.DISABLED)
            self.int2_entry.config(state=tk.DISABLED)
            for c in self.int1_spin_buttons.children.values():
                c.config(state=tk.DISABLED)
            for c in self.int2_spin_buttons.children.values():
                c.config(state=tk.DISABLED)

    def template(self, content):
        new_content = ''

        if self.bool_var.get():
            int1_val = self.int1_var.get().strip()
            int2_val = self.int2_var.get().strip()
            new_content = self.arg if self.arg else ''
            
            if int1_val:
                arg_part1 = (self.int1_arg if self.int1_arg else '')
                new_content += ('\n' if self.arg else '') + arg_part1 + (' ' if arg_part1 else '') + int1_val
                
            if int2_val:
                arg_part2 = (self.int2_arg if self.int2_arg else '')
                new_content += ('\n' if new_content else '') + arg_part2 + (' ' if arg_part2 else '') + int2_val

        return _entry.replace_first(content, self.placeholder, new_content)

    def exclude(self, other_entry):
        """Set up mutual exclusion with another checkbox entry.
        When this checkbox is checked, the other will be unchecked and vice versa."""
        from dgenerate.console.formentries.argswitchcheckbox import _ArgSwitchCheckbox
        from dgenerate.console.formentries.argswitchconditionalcheckboxes import _ArgSwitchConditionalCheckboxes
        from dgenerate.console.formentries.checkboxwithfloatargentry import _CheckboxWithFloatArgEntry
        
        if isinstance(other_entry, _ArgSwitchCheckbox):
            # This checkbox excludes simple checkbox
            self.bool_var.trace_add('write', lambda *args: other_entry.bool_var.set(False) if self.bool_var.get() else None)
            other_entry.bool_var.trace_add('write', lambda *args: self.bool_var.set(False) if other_entry.bool_var.get() else None)
            
        elif isinstance(other_entry, _ArgSwitchConditionalCheckboxes):
            # This checkbox excludes the master checkbox of conditional checkboxes
            self.bool_var.trace_add('write', lambda *args: other_entry.bool_vars[0].set(False) if self.bool_var.get() else None)
            other_entry.bool_vars[0].trace_add('write', lambda *args: self.bool_var.set(False) if other_entry.bool_vars[0].get() else None)
            
        elif isinstance(other_entry, _CheckboxWithFloatArgEntry):
            # This checkbox excludes checkbox with float arg
            self.bool_var.trace_add('write', lambda *args: other_entry.bool_var.set(False) if self.bool_var.get() else None)
            other_entry.bool_var.trace_add('write', lambda *args: self.bool_var.set(False) if other_entry.bool_var.get() else None)
            
        elif isinstance(other_entry, _CheckboxWithTwoIntArgsEntry):
            # Both are two int args checkboxes - mutual exclusion
            self.bool_var.trace_add('write', lambda *args: other_entry.bool_var.set(False) if self.bool_var.get() else None)
            other_entry.bool_var.trace_add('write', lambda *args: self.bool_var.set(False) if other_entry.bool_var.get() else None)