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

import dgenerate.console.recipesformentries.entry as _entry
import dgenerate.console.recipesformentries.urientry as _urientry
import dgenerate.console.resources as _resources


class _TorchVaeEntry(_urientry._UriEntry):
    NAME = 'torchvae'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=2)

        if self.arg is None:
            self.arg = '--vae'

        vae_types = _resources.get_torch_vae_types()

        self.vae_type_var = \
            tk.StringVar(value=self.config.get('vae-type', vae_types[0]))

        self.vae_type_label = tk.Label(
            self.master, text='VAE Type', anchor='e')

        self.vae_type_entry = tk.OptionMenu(
            self.master, self.vae_type_var, *vae_types)

        self.vae_type_label.grid(row=self.row + 1, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.vae_type_entry.grid(row=self.row + 1, column=1, padx=_entry.ROW_XPAD, sticky='ew')

        self._vae_type_enabled_color = self.vae_type_entry.cget('bg')

        if not self.uri_var.get():
            self.vae_type_entry.config(bg='darkgray')
            self.vae_type_entry.config(state=tk.DISABLED)

        self.uri_var.trace_add('write', lambda *a: self._check_vae_type_active())

    def _check_vae_type_active(self):
        if self.uri_var.get():
            self.vae_type_entry.config(bg=self._vae_type_enabled_color)
            self.vae_type_entry.config(state=tk.NORMAL)
        else:
            self.vae_type_entry.config(bg='darkgray')
            self.vae_type_entry.config(state=tk.DISABLED)

    def template(self, content):
        vae_uri = self.uri_var.get().strip()
        vae_type = self.vae_type_var.get().strip()

        value = ''
        if vae_uri and vae_type:
            value = f'{vae_type};model={shlex.quote(vae_uri)}'

        return self._template(content, value)
