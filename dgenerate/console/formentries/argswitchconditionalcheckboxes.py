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


class _ArgSwitchConditionalCheckboxes(_entry._Entry):
    NAME = 'conditionalcheckboxes'

    def __init__(self, *args, **kwargs):
        config = kwargs.get('config')

        labels = config.get('labels', ['Master Checkbox', 'Missing Labels'])

        super().__init__(*args, **kwargs, widget_rows=len(labels))

        self.args = self.config.get('args', ['MISSING ARGS'])

        if len(labels) != len(self.args):
            raise RuntimeError('conditionalcheckbox, not as many labels as there are args.')

        self.defaults = self.config.get('defaults', [False] * len(labels))
        
        if len(self.defaults) != len(labels):
            self.defaults = [False] * len(labels)

        self.bool_vars = []
        self.label_widgets = []
        self.check_boxes = []

        for idx, label in enumerate(labels):
            # initialize BooleanVar with a default value from config
            self.bool_vars.append(tk.BooleanVar(value=self.defaults[idx]))

            # create label widget
            label_text = labels[idx]
            label_text = label_text if not self.optional else f"(Optional) {label_text}"
                
            self.label_widgets.append(
                tk.Label(self.master,
                         text=label_text,
                         anchor='e'))

            # create checkbox widget with boolean variable
            checkbox = tk.Checkbutton(self.master, variable=self.bool_vars[-1])
            self.check_boxes.append(checkbox)

            # place the widgets in a grid layout
            self.label_widgets[-1].grid(
                row=self.row + idx, column=0, padx=_entry.ROW_XPAD, sticky='e')
            self.check_boxes[-1].grid(
                row=self.row + idx, column=1, sticky='w')

        # Set up the master checkbox behavior
        def master_changed():
            master_enabled = self.bool_vars[0].get()
            for idx in range(1, len(self.check_boxes)):
                if master_enabled:
                    self.check_boxes[idx].config(state='normal')
                else:
                    self.check_boxes[idx].config(state='disabled')
                    self.bool_vars[idx].set(False)

        # Bind the master checkbox change event
        self.bool_vars[0].trace_add('write', lambda *args: master_changed())
        
        # Initialize the state
        master_changed()

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
        When this master checkbox is checked, the other will be unchecked and vice versa."""
        from dgenerate.console.formentries.argswitchcheckbox import _ArgSwitchCheckbox
        from dgenerate.console.formentries.checkboxwithfloatargentry import _CheckboxWithFloatArgEntry
        from dgenerate.console.formentries.checkboxwithtwointargsentry import _CheckboxWithTwoIntArgsEntry
        
        if isinstance(other_entry, _ArgSwitchCheckbox):
            # Master checkbox excludes simple checkbox
            self.bool_vars[0].trace_add('write', lambda *args: other_entry.bool_var.set(False) if self.bool_vars[0].get() else None)
            other_entry.bool_var.trace_add('write', lambda *args: self.bool_vars[0].set(False) if other_entry.bool_var.get() else None)

        elif isinstance(other_entry, _ArgSwitchConditionalCheckboxes):
            # Both are conditional checkboxes - master checkboxes exclude each other
            self.bool_vars[0].trace_add('write', lambda *args: other_entry.bool_vars[0].set(False) if self.bool_vars[0].get() else None)
            other_entry.bool_vars[0].trace_add('write', lambda *args: self.bool_vars[0].set(False) if other_entry.bool_vars[0].get() else None)
            
        elif isinstance(other_entry, _CheckboxWithFloatArgEntry):
            # Master checkbox excludes checkbox with float arg
            self.bool_vars[0].trace_add('write', lambda *args: other_entry.bool_var.set(False) if self.bool_vars[0].get() else None)
            other_entry.bool_var.trace_add('write', lambda *args: self.bool_vars[0].set(False) if other_entry.bool_var.get() else None)
            
        elif isinstance(other_entry, _CheckboxWithTwoIntArgsEntry):
            # Master checkbox excludes checkbox with two int args
            self.bool_vars[0].trace_add('write', lambda *args: other_entry.bool_var.set(False) if self.bool_vars[0].get() else None)
            other_entry.bool_var.trace_add('write', lambda *args: self.bool_vars[0].set(False) if other_entry.bool_var.get() else None)

    def template(self, content):
        args = []
        
        # Only include args if master checkbox is checked
        if self.bool_vars[0].get():
            args.append(self.args[0])  # Always include master arg if checked
            
            # Include dependent args if they are checked
            for idx in range(1, len(self.bool_vars)):
                if self.bool_vars[idx].get():
                    args.append(self.args[idx])
        
        arg_string = '\n'.join(args)
        return _entry.replace_first(content, self.placeholder, arg_string)
