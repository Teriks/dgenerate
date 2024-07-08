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
import shlex
import tkinter as tk

import dgenerate.console.filedialog as _filedialog
import dgenerate.console.recipesformentries.entry as _entry
import dgenerate.console.resources as _resources


class _TorchVaeEntry(_entry._Entry):
    NAME = 'torchvae'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=2)

        if self.arg is None:
            self.arg = '--vae'

        self.vae_uri_var = \
            tk.StringVar(value=self.config.get('default', ''))

        vae_uri_label_text = self.get_label('VAE')

        self.vae_uri_label = tk.Label(
            self.master, text=vae_uri_label_text, anchor='e')

        self.vae_uri_entry = \
            tk.Entry(self.master, textvariable=self.vae_uri_var)

        vae_types = _resources.get_torch_vae_types()

        self.vae_type_var = \
            tk.StringVar(value=self.config.get('vae-type', vae_types[0]))

        self.vae_type_label = tk.Label(
            self.master, text='VAE Type', anchor='e')

        self.vae_type_entry = tk.OptionMenu(
            self.master, self.vae_type_var, *vae_types)

        def select_command():
            file_globs = ['*.' + ext for ext in _resources.supported_torch_model_formats_open()]
            r = _filedialog.open_file_dialog(
                filetypes=[('Models', ' *.'.join(file_globs))])
            if r is not None:
                self.vae_uri_var.set(r)

        self.button = tk.Button(self.master, text="Select File",
                                command=select_command)

        self.vae_uri_label.grid(row=self.row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.vae_uri_entry.grid(row=self.row, column=1, padx=_entry.ROW_XPAD, sticky='ew')
        self.button.grid(row=self.row, column=2, padx=(2, 5), sticky='w')

        self.vae_type_label.grid(row=self.row + 1, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.vae_type_entry.grid(row=self.row + 1, column=1, padx=_entry.ROW_XPAD, sticky='ew')

        self.vae_uri_entry.bind('<Key>', lambda e: self.valid())

    def invalid(self):
        self.vae_uri_entry.config(
            highlightbackground="red",
            highlightcolor="red",
            highlightthickness=2)

    def is_empty(self):
        return self.vae_uri_entry.get().strip() == ''

    def valid(self):
        self.vae_uri_entry.config(highlightthickness=0)

    def template(self, content):
        vae_uri = self.vae_uri_var.get().strip()
        vae_type = self.vae_type_var.get().strip()

        value = ''
        if vae_uri and vae_type:
            value = f'{vae_type};model={shlex.quote(vae_uri)}'

        return self._template(content, value)
