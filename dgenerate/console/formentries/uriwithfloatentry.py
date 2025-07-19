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
import dgenerate.console.formentries.urientry as _urientry
from dgenerate.console.spinbox import FloatSpinbox


class _UriWithFloatEntry(_urientry._UriEntry):
    NAME = 'uriwithfloat'

    def __init__(self, *args, **kwargs):

        if 'widget_rows' not in kwargs:
            kwargs['widget_rows'] = 2

        super().__init__(*args, **kwargs)

        self.default_float = str(self.config.get('float', '1.0'))

        self.float_arg = self.config.get('float-arg', 'float')

        self.float_var = \
            tk.StringVar(value=self.default_float)

        self.float_label = tk.Label(
            self.master,
            text=self.get_label(
                'URI Float', key='float-label') + ' (float)',
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

        self.uri_entry.bind('<Key>', lambda e: self.valid())

        self.uri_var.trace_add('write', lambda *a: self._check_float_active())

        if not self.uri_entry.get():
            self.float_entry.config(state=tk.DISABLED)
            for c in self.spin_buttons.children.values():
                c.config(state=tk.DISABLED)

    def is_valid(self):
        return self.float_entry.is_valid()

    def invalid(self):
        if not self.optional and not self.uri_entry():
            _entry.invalid_colors(self.uri_entry)

        if not self.float_entry.is_valid():
            _entry.invalid_colors(self.float_entry)

    def valid(self):
        if not self.optional and self.uri_entry():
            _entry.valid_colors(self.uri_entry)

        if self.float_entry.is_valid():
            _entry.valid_colors(self.float_entry)

    def _check_float_active(self):
        if self.uri_var.get():
            self.float_entry.config(state=tk.NORMAL)
            for c in self.spin_buttons.children.values():
                c.config(state=tk.NORMAL)
        else:
            self.float_entry.config(state=tk.DISABLED)
            for c in self.spin_buttons.children.values():
                c.config(state=tk.DISABLED)

    def template(self, content):
        uri = _entry.shell_quote_if(self.uri_var.get().strip())
        float_val = self.float_var.get().strip()

        value = ''
        if uri and float_val and float_val != self.default_float:
            value = f'{uri};{self.float_arg}={float_val}'
        elif uri:
            value = uri

        return self._template(content, value)
