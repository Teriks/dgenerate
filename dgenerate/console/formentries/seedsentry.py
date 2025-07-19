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


class _SeedsEntry(_entry._Entry):
    NAME = 'seeds'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=1)

        self.text_var = tk.StringVar(
            value=self.config.get('default', '1'))

        self.label_widget = tk.Label(
            self.master,
            text=self.get_label('Int') + ' (int)', anchor='e')

        entry_frame = tk.Frame(self.master)

        self.is_count_label_wiget = tk.Label(entry_frame, text='Is Count?')
        self.is_count_label_wiget.grid(column=0, row=0)

        self.is_count_var = tk.BooleanVar(value=True)

        self.is_count_check = tk.Checkbutton(entry_frame, variable=self.is_count_var, command=self._is_count_changed)
        self.is_count_check.grid(column=1, row=0)

        self.entry = IntSpinbox(entry_frame,
                                from_=1,
                                to=sys.maxsize,
                                textvariable=self.text_var)

        self.entry.grid(column=2, row=0, sticky='ew')

        entry_frame.grid_columnconfigure(2, weight=1)

        self.label_widget.grid(row=self.row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        entry_frame.grid(row=self.row, column=1, padx=_entry.ROW_XPAD, sticky='ew')

        self.text_var.trace_add('write', lambda *e: self.valid())

        spin_buttons = self.entry.create_spin_buttons(self.master)
        spin_buttons.grid(row=self.row, column=2, sticky='w')

    def _is_count_changed(self):
        if self.is_count_var.get():
            if int(self.text_var.get()) < 1:
                self.text_var.set('1')
            self.entry.from_ = 1
        else:
            self.entry.from_ = -sys.maxsize

    def invalid(self):
        _entry.invalid_colors(self.entry)

    def valid(self):
        _entry.valid_colors(self.entry)

    def is_empty(self):
        return self.text_var.get().strip() == ''

    def template(self, content):
        if self.is_count_var.get():
            return self._template(content, '--gen-seeds ' + self.text_var.get().strip())
        else:
            return self._template(content, '--seeds ' + self.text_var.get().strip())
