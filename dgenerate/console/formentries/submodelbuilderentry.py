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
import dgenerate.console.textentry as _t_entry

class _SubModelBuilderEntry(_entry._Entry):
    NAME = 'submodelbuilder'

    def __init__(self, *args, **kwargs):

        config = kwargs['config']

        # required
        self.model = config['model']

        if 'widget_rows' not in kwargs:
            kwargs['widget_rows'] = 1

        super().__init__(*args, **kwargs)

        uri_label_text = self.get_label('URI')

        self.uri_label = tk.Label(
            self.master,
            text=uri_label_text,
            anchor='e')

        self.uri_var = \
            tk.StringVar(value=self.config.get('default', ''))

        self.uri_entry = \
            _t_entry.TextEntry(self.master, textvariable=self.uri_var)

        self.build_uri_button = tk.Button(
            self.master,
            text='Build URI',
            command=self._build_uri)


        self.uri_label.grid(row=self.row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.uri_entry.grid(row=self.row, column=1, padx=_entry.ROW_XPAD, sticky='ew')
        self.build_uri_button.grid(row=self.row, column=2, padx=(2, 5), sticky='w')

    def _insert(self, value: str):
        if value is not None and value.strip():
            self.uri_var.set(value)

    def _build_uri(self):
        import dgenerate.console.submodelselect as _submodelselect
        dialog = _submodelselect.request_uri(self.form, self._insert, filter=[self.model])

        og_destroy = dialog.destroy
        def _destroy():
            og_destroy()
            if hasattr(self.form, 'bind_mousewheel'):
                self.form.bind_mousewheel()

        dialog.destroy = _destroy

    def template(self, content):
        return self._template(content, _entry.shell_quote_if(
            self.uri_var.get().strip()))