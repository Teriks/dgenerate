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

import dgenerate.console.filedialog as _filedialog
import dgenerate.console.formentries.entry as _entry
import dgenerate.console.resources as _resources
import dgenerate.console.textentry as _t_entry


class _FileEntry(_entry._Entry):
    NAME = 'file'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=1)

        self.text_var = tk.StringVar(value=self.config.get('default', ''))

        self.label_widget = tk.Label(self.master,
                                     text=self.get_label('File'), anchor='e')

        self.entry = _t_entry.TextEntry(self.master, textvariable=self.text_var)
        self.select_mode = self.config.get('mode', 'input')
        self.file_types = self.config.get('file-types', [])

        dialog_args = _resources.get_file_dialog_args(self.file_types)

        def select_command():
            if 'output' in self.select_mode:
                r = _filedialog.open_file_save_dialog(**dialog_args)
            else:
                r = _filedialog.open_file_dialog(**dialog_args)
            if r is not None:
                self.text_var.set(r)

        if 'output' in self.select_mode:
            self.button = tk.Button(self.master, text="Save File",
                                    command=select_command)
        else:
            self.button = tk.Button(self.master, text="File",
                                    command=select_command)

        self.label_widget.grid(row=self.row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.entry.grid(row=self.row, column=1, padx=_entry.ROW_XPAD, sticky='ew')
        self.button.grid(row=self.row, column=2, padx=(2, 5), sticky='w')
        self.entry.bind('<Key>', lambda e: self.valid())

    def invalid(self):
        _entry.invalid_colors(self.entry)

    def valid(self):
        _entry.valid_colors(self.entry)

    def is_empty(self):
        return self.text_var.get().strip() == ''

    def template(self, content):
        return self._template(content, _entry.shell_quote_if(self.text_var.get().strip()))
