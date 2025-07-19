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


class _ArgSwitchRadio(_entry._Entry):
    NAME = 'switchradio'

    def __init__(self, *args, **kwargs):
        config = kwargs.get('config')

        labels = config.get('labels', ['Missing Labels'])

        super().__init__(*args, **kwargs, widget_rows=len(labels))

        self.args = self.config.get('args', ['MISSING ARGS'])

        if len(labels) != len(self.args):
            raise RuntimeError('switchradio, not as many labels as there are args.')

        self.default = self.config.get(
            'default', -1 if self.optional else 0)

        self.bool_vars = []
        self.label_widgets = []
        self.check_boxes = []

        for idx, label in enumerate(labels):

            # initialize BooleanVar with a default value from config
            self.bool_vars.append(tk.BooleanVar(value=idx == self.default))

            def var_changed(var=self.bool_vars[-1]):
                for other_var in self.bool_vars:
                    if other_var is not var:
                        if var.get():
                            other_var.set(False)
                if not self.optional:
                    if not any(i.get() for i in self.bool_vars):
                        var.set(True)

            # create label widget
            self.label_widgets.append(
                tk.Label(self.master,
                         text=labels[idx] if not
                         self.optional else f"(Optional) {labels[idx]}",
                         anchor='e'))

            # create checkbox widget with boolean variable
            self.check_boxes.append(
                tk.Checkbutton(self.master,
                               variable=self.bool_vars[-1],
                               command=var_changed))

            # place the widgets in a grid layout
            self.label_widgets[-1].grid(
                row=self.row + idx, column=0, padx=_entry.ROW_XPAD, sticky='e')
            self.check_boxes[-1].grid(
                row=self.row + idx, column=1, sticky='w')

    def template(self, content):
        arg = ''
        for idx, val in enumerate(self.bool_vars):
            if val.get():
                arg = self.args[idx]

        return _entry.replace_first(content, self.placeholder, arg)
