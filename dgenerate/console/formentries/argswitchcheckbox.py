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


class _ArgSwitchCheckbox(_entry._Entry):
    NAME = 'switch'

    def __init__(self, *args, **kwargs):
        if 'widget_rows' not in kwargs:
            kwargs['widget_rows'] = 1

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

    def exclude(self, other_entry):
        """Set up mutual exclusion with another checkbox entry.
        When this checkbox is checked, the other will be unchecked and vice versa."""
        from dgenerate.console.formentries.argswitchconditionalcheckboxes import _ArgSwitchConditionalCheckboxes
        from dgenerate.console.formentries.checkboxwithfloatargentry import _CheckboxWithFloatArgEntry
        from dgenerate.console.formentries.checkboxwithtwointargsentry import _CheckboxWithTwoIntArgsEntry
        
        if isinstance(other_entry, _ArgSwitchCheckbox):
            # Both are simple checkboxes - mutual exclusion
            self.bool_var.trace_add('write', lambda *args: other_entry.bool_var.set(False) if self.bool_var.get() else None)
            other_entry.bool_var.trace_add('write', lambda *args: self.bool_var.set(False) if other_entry.bool_var.get() else None)
            
        elif isinstance(other_entry, _ArgSwitchConditionalCheckboxes):
            # This checkbox excludes the master checkbox of conditional checkboxes
            self.bool_var.trace_add('write', lambda *args: other_entry.bool_vars[0].set(False) if self.bool_var.get() else None)
            other_entry.bool_vars[0].trace_add('write', lambda *args: self.bool_var.set(False) if other_entry.bool_vars[0].get() else None)
            
        elif isinstance(other_entry, _CheckboxWithFloatArgEntry):
            # This checkbox excludes the checkbox with float arg
            self.bool_var.trace_add('write', lambda *args: other_entry.bool_var.set(False) if self.bool_var.get() else None)
            other_entry.bool_var.trace_add('write', lambda *args: self.bool_var.set(False) if other_entry.bool_var.get() else None)
            
        elif isinstance(other_entry, _CheckboxWithTwoIntArgsEntry):
            # This checkbox excludes the checkbox with two int args
            self.bool_var.trace_add('write', lambda *args: other_entry.bool_var.set(False) if self.bool_var.get() else None)
            other_entry.bool_var.trace_add('write', lambda *args: self.bool_var.set(False) if other_entry.bool_var.get() else None)

    def _is_checked(self):
        return self.bool_var.get()

    def template(self, content):
        return _entry.replace_first(content, self.placeholder, self.arg if self._is_checked() else '')
