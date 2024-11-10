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
import dgenerate.console.resources as _resources
import dgenerate.console.util as _util
from dgenerate.console.mousewheelbind import bind_mousewheel


def _adjust_combobox_width(combo, options):
    f = tk.font.Font(combo, combo.cget("font"))
    # Estimate width based on average character width
    avg_char_width = f.measure("0")  # Get the average width of a character
    max_width = max(f.measure(option) for option in options) // avg_char_width  # Convert pixel width to character width
    combo.config(width=max_width + 2)  # Add a couple of characters to avoid tight fit


class _DgenerateArgumentSelect(tk.Toplevel):
    def __init__(self, master=None, position: tuple[int, int] = None):
        super().__init__(master)
        self.title('Insert Argument')
        self._arguments_dict = _resources.get_dgenerate_arguments()
        self._templates = None
        self._dropdown = None
        self._arguments = list(sorted(self._arguments_dict.keys()))
        self._current_argument = tk.StringVar()
        self._entries = []
        self._content = None
        self._ok = False
        self.transient(master)
        self.grab_set()
        self.resizable(False, False)
        self._insert = False

        self._frame = tk.Frame(self)

        # Create dropdown
        self._argument_dropdown = ttk.Combobox(
            self._frame, textvariable=self._current_argument, values=self._arguments)

        _adjust_combobox_width(self._argument_dropdown, self._arguments)

        # Create buttons
        self._insert_button = tk.Button(self._frame, text="Insert", command=self._insert_action)
        self._help_button = tk.Button(self._frame, text='Help', command=self._show_help)

        # Grid layout
        self._argument_dropdown.grid(row=0, column=1, sticky="we")
        self._help_button.grid(row=0, column=2, padx=(5, 5))
        self._insert_button.grid(row=2, column=0, columnspan=2, pady=(5, 0))

        self._frame.pack(pady=(5, 5), padx=(5, 5))

        _util.position_toplevel(master, self, position=position)

    def _show_help(self):
        if self._current_argument.get() not in self._arguments_dict:
            return

        top = tk.Toplevel(self)
        top.title(f'Argument Help: {self._current_argument.get()}')
        top.attributes('-topmost', 1)
        top.attributes('-topmost', 0)

        frame = tk.Frame(top)
        frame.pack(expand=True, fill='both')

        v_scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        h_scrollbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        text_widget = tk.Text(frame, wrap=tk.NONE, state='disabled', yscrollcommand=v_scrollbar.set,
                              xscrollcommand=h_scrollbar.set)

        text_widget.configure(**_resources.get_textbox_theme())

        text_widget.config(state='normal')

        text_widget.insert(
            tk.END, self._arguments_dict[self._current_argument.get()] + '\n\n')

        text_widget.config(state='disabled')

        v_scrollbar.config(command=text_widget.yview)
        h_scrollbar.config(command=text_widget.xview)
        text_widget.pack(expand=True, fill='both')

        width = 850
        height = 600

        # Dock right
        master_x = self.winfo_rootx()
        master_y = self.winfo_rooty()
        master_width = self.winfo_width()

        # Calculate position x, y to dock the help window to the right of recipe_form
        x = master_x + master_width
        y = master_y

        top.geometry(f'{width}x{height}+{x}+{y}')

        bind_mousewheel(top.bind, self._on_help_mouse_wheel)

    @staticmethod
    def _on_help_mouse_wheel(event):
        return "break"

    def _insert_action(self):
        self._insert = True
        self.destroy()

    def get_argument(self):
        self.wait_window(self)
        if not self._insert:
            return None

        return self._current_argument.get()


_last_pos = None


def request_argument(master):
    global _last_pos

    window = _DgenerateArgumentSelect(master, position=_last_pos)

    og_destroy = window.destroy

    def destroy():
        global _last_pos
        _last_pos = _last_size = (window.winfo_x(), window.winfo_y())
        og_destroy()

    window.destroy = destroy

    def on_closing():
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", on_closing)

    return window.get_argument()
