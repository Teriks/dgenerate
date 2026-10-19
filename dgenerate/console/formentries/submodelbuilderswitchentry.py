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
import tkinter.ttk as ttk

import dgenerate.console.formentries.entry as _entry
from dgenerate.console.formentries.submodelbuilderentry import _SubModelBuilderEntry


class _SubModelBuilderSwitchEntry(_SubModelBuilderEntry):
    NAME = 'submodelbuilderswitch'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bool_var = tk.BooleanVar(value=self.config.get('default', False))
        self.uri_label.grid_remove()
        self.switch_frame = ttk.Frame(self.master)
        self.switch_label = ttk.Label(
            self.switch_frame,
            text=self.get_label('Latent Upscale'),
            anchor='e')
        self.checkbox = ttk.Checkbutton(
            self.switch_frame,
            variable=self.bool_var)
        self.switch_label.pack(side='left')
        self.checkbox.pack(side='left')
        self.switch_frame.grid(
            row=self.row, column=0, padx=_entry.ROW_XPAD, sticky='e')

    def template(self, content):
        if not self.bool_var.get():
            value = ''
        else:
            uri = self.uri_var.get().strip()
            value = self.arg if not uri else f'{self.arg} {_entry.shell_quote_if(uri)}'
        return _entry.replace_first(content, self.placeholder, value)
