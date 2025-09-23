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
import typing

import dgenerate.console.util as _util
import dgenerate.console.helpdialog as _helpdialog


def _adjust_combobox_width(combo, options):
    f = tk.font.Font(combo, combo.cget("font"))
    # Estimate width based on average character width
    avg_char_width = f.measure("0")  # Get the average width of a character
    max_width = max(f.measure(option) for option in options) // avg_char_width  # Convert pixel width to character width
    combo.config(width=max_width + 2)  # Add a couple of characters to avoid tight fit


class _DropdownSelectWithHelp(tk.Toplevel):
    def __init__(self,
                 item_name: str,
                 values_to_help: dict[str, str],
                 insert: typing.Callable[[str], None],
                 master=None,
                 position: tuple[int, int] = None
                 ):
        super().__init__(master)
        self.title(f'Insert {item_name}')
        self._item_name = item_name
        self._values_dict = values_to_help
        self._templates = None
        self._dropdown = None
        self._values = list(sorted(self._values_dict.keys()))
        self._current_value = tk.StringVar()
        self._entries = []
        self._content = None
        self._ok = False
        self.transient(master)
        self.resizable(False, False)
        self._insert = insert

        self._frame = tk.Frame(self)

        # Create dropdown
        self._value_dropdown = ttk.Combobox(
            self._frame, textvariable=self._current_value, values=self._values)
        self._value_dropdown.bind('<KeyRelease>', self._filter_values)
        self._value_dropdown.bind('<<ComboboxSelected>>', self._on_select)
        self._value_dropdown.bind('<<Paste>>', lambda e: self.after(1, self._filter_values, None))

        _adjust_combobox_width(self._value_dropdown, self._values)

        # Create buttons
        self._insert_button = tk.Button(self._frame, text="Insert", command=self._insert_action)
        self._help_button = tk.Button(self._frame, text='Help', command=self._show_help)
        self._insert_button.config(state=tk.DISABLED)
        self._help_button.config(state=tk.DISABLED)

        self._current_value.trace_add('write',
                                  lambda *a:
                                  (self._help_button.config(state=tk.ACTIVE),
                                   self._insert_button.config(state=tk.ACTIVE))
                                  if self._current_value.get() else
                                  (self._help_button.config(state=tk.DISABLED),
                                   self._insert_button.config(state=tk.DISABLED)))

        # Grid layout
        self._value_dropdown.grid(row=0, column=1, sticky="we")
        self._help_button.grid(row=0, column=2, padx=(5, 5))
        self._insert_button.grid(row=1, column=0, columnspan=2, pady=(5, 0))

        self._frame.pack(pady=(5, 5), padx=(5, 5))

        _util.position_toplevel(master, self, position=position)

    def _filter_values(self, event):
        if event is not None and event.keysym in ('Up', 'Down', 'Return', 'Escape'):
            return
        
        current_text = self._current_value.get().lower()
        if current_text:
            filtered_values = [v for v in self._values if current_text in v.lower()]
        else:
            filtered_values = self._values
        self._value_dropdown['values'] = filtered_values

    def _on_select(self, event):
        self._value_dropdown['values'] = self._values

    def _show_help(self):
        if self._current_value.get() not in self._values_dict:
            return

        title = f'{self._item_name} Help: {self._current_value.get()}'
        help_text = self._values_dict[self._current_value.get()]
        _helpdialog.show_help_dialog(tk._default_root, title, help_text, position_widget=self)

    def _insert_action(self):
        value = self._current_value.get()
        if value.strip():
            self._insert(value)
        self.destroy()
