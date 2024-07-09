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

import dgenerate.console.recipesformentries.entry as _entry


class _StringEntry(_entry._Entry):
    NAME = 'string'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=1)

        self.text_var = tk.StringVar(
            value=self.config.get('default', ''))

        self.label_widget = tk.Label(
            self.master,
            text=self.get_label('String'), anchor='e')

        self.entry = tk.Entry(self.master, textvariable=self.text_var)

        self.label_widget.grid(row=self.row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.entry.grid(row=self.row, column=1, padx=_entry.ROW_XPAD, sticky='ew')
        self.entry.bind('<Key>', lambda e: self.valid())

    def invalid(self):
        _entry.invalid_colors(self.entry)

    def valid(self):
        _entry.valid_colors(self.entry)

    def is_empty(self):
        return self.text_var.get().strip() == ''

    def template(self, content):
        return self._template(content, _entry.shell_quote_if(self.text_var.get().strip()))
