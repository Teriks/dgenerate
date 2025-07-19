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
import collections.abc
import platform
import tkinter as tk
import typing

import dgenerate.console.formentries.pluginschemaentry as _pluginschemaentry
import dgenerate.console.util as _util
from dgenerate.console.mousewheelbind import bind_mousewheel, handle_canvas_scroll, un_bind_mousewheel


class _PluginUriSelect(tk.Toplevel):
    def __init__(self,
                 title: str,
                 plugin_entry_class: type[_pluginschemaentry._PluginSchemaEntry],
                 insert: typing.Callable[[str], None],
                 master=None,
                 filter: collections.abc.Collection[str] = None,
                 position: tuple[int, int] = None,
                 size: tuple[int, int] = None
                 ):
        super().__init__(master)
        self.title(title)

        self.transient(master)
        self.resizable(True, True)

        self._insert = insert

        self.plugin_dropdown_master = tk.Frame(self)
        self.plugin_dropdown_master.grid(row=0, column=0, sticky='ew', padx=5, pady=(5, 0))
        self.plugin_dropdown_master.grid_columnconfigure(1, weight=1)
        self.plugin_dropdown_row = 0

        outer_frame = tk.Frame(self, bd=3, relief="sunken")
        outer_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)

        self.canvas = tk.Canvas(outer_frame, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(outer_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, padx=10, pady=10, highlightthickness=0)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.scrollbar.grid(row=0, column=1, sticky='ns')

        outer_frame.grid_rowconfigure(0, weight=1)
        outer_frame.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.plugin_frame = tk.Frame(self.scrollable_frame)
        self.plugin_frame.grid(row=0, column=0, sticky='nsew')

        config: dict[str, typing.Any] = {
            "optional": False,
            "internal-divider": False
        }

        if filter:
            config['filter'] = filter

        self.plugin_entry = plugin_entry_class(
            form=self,
            master=self.plugin_frame,
            config=config,
            placeholder='URI',
            row=0,
            dropdown_parent=self.plugin_dropdown_master,
            dropdown_row=0
        )

        self.plugin_entry.arg = None

        self.button_frame = tk.Frame(self)
        self.button_frame.grid(row=2, column=0, pady=(0, 5))

        self.button = tk.Button(self.button_frame, text='Insert', command=self._insert_action)
        self.button.pack()

        self.plugin_frame.grid_columnconfigure(1, weight=1)
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        self.bind("<Configure>", self._on_resize)

        self.bind_mousewheel()

        if platform.system() == 'Windows':
            default_size = (500, 400)
        else:
            default_size = (600, 400)

        _util.position_toplevel(
            master,
            self,
            position=position,
            size=size if size else default_size
        )

        self.plugin_entry.on_updated_callback = self._on_plugin_change

    def bind_mousewheel(self):
        bind_mousewheel(self.canvas.bind_all, self._on_mouse_wheel)

    def unbind_mousewheel(self):
        un_bind_mousewheel(self.canvas.unbind_all)

    def _on_plugin_change(self, *args):
        self.canvas.yview_moveto(0)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_resize(self, event):
        canvas_width = self.winfo_width() - self.scrollbar.winfo_width() - 30
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

    def _on_mouse_wheel(self, event):
        if 'popdown' in str(event.widget).lower():
            return "break"

        # Notify entry of scroll event
        if hasattr(self, 'plugin_entry'):
            self.plugin_entry.on_form_scroll()
        handle_canvas_scroll(self.canvas, event)

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

    def destroy(self) -> None:
        self.unbind_mousewheel()
        super().destroy()
