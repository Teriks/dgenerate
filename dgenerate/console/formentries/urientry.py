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


class _UriEntry(_entry._Entry):
    NAME = 'uri'

    def __init__(self, *args, **kwargs):

        if 'widget_rows' not in kwargs:
            kwargs['widget_rows'] = 1

        super().__init__(*args, **kwargs)

        self.uri_var = \
            tk.StringVar(value=self.config.get('default', ''))

        uri_label_text = self.get_label('URI')

        self.uri_label = tk.Label(
            self.master,
            text=uri_label_text,
            anchor='e')

        self.uri_entry = \
            _t_entry.TextEntry(self.master, textvariable=self.uri_var)

        file_types = self.config.get('file-types', [])
        is_dir = self.config.get('dir', False)

        if file_types and is_dir:

            def select_file_command():
                dialog_args = _resources.get_file_dialog_args(file_types)
                r = _filedialog.open_file_dialog(**dialog_args)

                if r is not None:
                    self.uri_var.set(r)

            def select_dir_command():
                r = _filedialog.open_directory_dialog()
                if r is not None:
                    self.uri_var.set(r)

            self.button_frame = tk.Frame(self.master)

            file_button = tk.Button(self.button_frame,
                                    text="File",
                                    command=select_file_command)
            file_button.pack(side=tk.LEFT)

            directory_button = tk.Button(self.button_frame,
                                         text="Directory",
                                         command=select_dir_command)
            directory_button.pack(side=tk.LEFT)

            self.button_frame.grid(row=self.row, column=2, padx=(2, 5), sticky='w')

        elif file_types or is_dir:

            def select_command():
                if file_types:
                    dialog_args = _resources.get_file_dialog_args(file_types)
                    r = _filedialog.open_file_dialog(**dialog_args)
                else:
                    r = _filedialog.open_directory_dialog()

                if r is not None:
                    self.uri_var.set(r)

            self.button_frame = tk.Frame(self.master)

            button = tk.Button(self.button_frame,
                               text="File" if file_types else "Directory",
                               command=select_command)
            button.pack(side=tk.LEFT)

            self.button_frame.grid(row=self.row, column=2, padx=(2, 5), sticky='w')

        self.uri_label.grid(row=self.row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.uri_entry.grid(row=self.row, column=1, padx=_entry.ROW_XPAD, sticky='ew')

        self.uri_entry.bind('<Key>', lambda e: self.valid())

    def invalid(self):
        _entry.invalid_colors(self.uri_entry)

    def valid(self):
        _entry.valid_colors(self.uri_entry)

    def is_empty(self):
        return self.uri_entry.get().strip() == ''

    def template(self, content):
        return self._template(content, _entry.shell_quote_if(
            self.uri_var.get().strip()))
