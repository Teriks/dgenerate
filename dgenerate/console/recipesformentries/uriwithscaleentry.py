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
import dgenerate.console.recipesformentries.entry as _entry
import dgenerate.console.resources as _resources
from dgenerate.console.spinbox import FloatSpinbox


class _UriWithScaleEntry(_entry._Entry):
    NAME = 'uriwithscale'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=2)

        self.uri_var = \
            tk.StringVar(value=self.config.get('default', ''))

        uri_label_text = self.get_label('URI')

        self.uri_label = tk.Label(
            self.master,
            text=uri_label_text,
            anchor='e')

        self.uri_entry = \
            tk.Entry(self.master, textvariable=self.uri_var)

        self.default_scale = str(self.config.get('scale', '1.0'))

        self.scale_var = \
            tk.StringVar(value=self.default_scale)

        self.scale_label = tk.Label(
            self.master,
            text=self.get_label('URI', key='scale_label'),
            anchor='e')

        self.min = self.config.get('min', float('-inf'))
        self.max = self.config.get('max', float('inf'))

        self.scale_entry = FloatSpinbox(
            self.master,
            self.scale_var,
            from_=self.min,
            to=self.max)

        self.spin_buttons = self.scale_entry.create_spin_buttons(self.master)
        self.spin_buttons.grid(row=self.row + 1, column=2, sticky='w')

        is_file = self.config.get('file', False)
        is_dir = self.config.get('dir', False)

        if is_file and is_dir:
            raise RuntimeError('URI cannot be both file and dir selectable.')

        if is_file or is_dir:

            def select_command():
                if is_file:
                    file_types = self.config.get('file-types', [])
                    dialog_args = _resources.get_file_dialog_args(file_types)
                    r = _filedialog.open_file_dialog(**dialog_args)
                else:
                    r = _filedialog.open_directory_dialog()

                if r is not None:
                    self.uri_var.set(r)

            self.button = tk.Button(self.master,
                                    text="Select File" if is_file else "Select Directory",
                                    command=select_command)

            self.button.grid(row=self.row, column=2, padx=(2, 5), sticky='w')

        self.uri_label.grid(row=self.row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.uri_entry.grid(row=self.row, column=1, padx=_entry.ROW_XPAD, sticky='ew')

        self.scale_label.grid(row=self.row + 1, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.scale_entry.grid(row=self.row + 1, column=1, padx=_entry.ROW_XPAD, sticky='ew')

        self.uri_entry.bind('<Key>', lambda e: self.valid())

        self.uri_var.trace_add('write', lambda *a: self._check_scale_active())

        if not self.uri_entry.get():
            self.scale_entry.config(state=tk.DISABLED)
            for c in self.spin_buttons.children.values():
                c.config(state=tk.DISABLED)

    def _check_scale_active(self):
        if self.uri_var.get():
            self.scale_entry.config(state=tk.NORMAL)
            for c in self.spin_buttons.children.values():
                c.config(state=tk.NORMAL)
        else:
            self.scale_entry.config(state=tk.DISABLED)
            for c in self.spin_buttons.children.values():
                c.config(state=tk.DISABLED)

    def invalid(self):
        _entry.invalid_colors(self.uri_entry)

    def valid(self):
        _entry.valid_colors(self.uri_entry)

    def is_empty(self):
        return self.uri_entry.get().strip() == ''

    def template(self, content):
        uri = self.uri_var.get().strip()
        scale = self.scale_var.get().strip()

        value = ''
        if uri and scale and scale != self.default_scale:
            value = f'{uri};scale={scale}'
        elif uri:
            value = uri

        return self._template(content, _entry.shell_quote_if(value))
