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
import typing

import dgenerate.console.recipesformentries.pluginschemaentry as _pluginschemaentry
import dgenerate.console.util as _util


class _PluginUriSelect(tk.Toplevel):
    def __init__(self,
                 title: str,
                 plugin_entry_class: type[_pluginschemaentry._PluginSchemaEntry],
                 insert: typing.Callable[[str], None],
                 master=None,
                 position: tuple[int, int] = None
                 ):
        super().__init__(master)
        self.title(title)

        self.transient(master)
        self.resizable(True, False)

        self._insert = insert

        self.plugin_frame = tk.Frame(self)

        self.plugin_entry = plugin_entry_class(
            recipe_form=self,
            master=self.plugin_frame,
            config={"optional": False, "internal-divider": False},
            placeholder='URI',
            row=1
        )

        self.plugin_entry.arg = None

        self.plugin_frame.grid(row=0, pady=(5, 5), padx=(5, 5), sticky='nsew')

        self.button = tk.Button(self, text='Insert', command=self._insert_action)

        self.button.grid(row=1, pady=(0, 5))

        self.plugin_frame.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        def plugin_updated():
            self.update_idletasks()
            self.minsize(self.winfo_reqwidth(), self.winfo_reqheight())

        self.plugin_entry.on_updated_callback = plugin_updated

        _util.position_toplevel(master, self, position=position)

    def _insert_action(self):
        if not self.plugin_entry.is_valid():
            self.plugin_entry.invalid()
            return
        else:
            self.plugin_entry.valid()

        value = self.plugin_entry.template('URI')
        if value.strip():
            self._insert(value)
        self.destroy()
