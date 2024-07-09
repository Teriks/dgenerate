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

import dgenerate.console.recipesformentries as _entries


class _ImageProcessorSelect(tk.Toplevel):
    def __init__(self, master=None, position: tuple[int, int] = None, size: tuple[int, int] = None):
        super().__init__(master)
        self.title('Insert Image Processor URI')

        self.transient(master)
        self.grab_set()
        self.resizable(True, False)

        self._insert = False

        self.processor_frame = tk.Frame(self)

        self.processor = _entries._ImageProcessorEntry(
            recipe_form=self,
            master=self.processor_frame,
            config={"optional": False},
            placeholder='URI',
            row=1)

        self.processor_frame.grid(row=0, pady=(5, 5), padx=(5, 5), sticky='nsew')

        self.button = tk.Button(self, text='Insert', command=self._insert_action)

        self.button.grid(row=1, pady=(0, 5))

        self.processor_frame.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.withdraw()
        self.update_idletasks()

        if position is None:
            window_width = self.master.winfo_width()
            window_height = self.master.winfo_height()
            top_level_width = self.winfo_width()
            top_level_height = self.winfo_height()

            position_top = self.master.winfo_y() + window_height // 2 - top_level_height // 2
            position_left = self.master.winfo_x() + window_width // 2 - top_level_width // 2

            self.geometry(f"+{position_left}+{position_top}")

        else:
            self.geometry("+{}+{}".format(*position))

        self.deiconify()

    def _insert_action(self):
        if not self.processor.is_valid():
            self.processor.invalid()
            return
        else:
            self.processor.valid()

        self._insert = True
        self.destroy()

    def get_uri(self):
        self.wait_window(self)
        if not self._insert:
            return None
        return self.processor.template('URI')


_last_pos = None


def request_uri(master):
    global _last_pos

    window = _ImageProcessorSelect(master, position=_last_pos)

    og_destroy = window.destroy

    def destroy():
        global _last_pos
        _last_pos = _last_size = (window.winfo_x(), window.winfo_y())
        og_destroy()

    window.destroy = destroy

    def on_closing():
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", on_closing)

    return window.get_uri()
