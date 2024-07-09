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

import dgenerate.console.resources as _resources


class _KarrasSchedulerSelect(tk.Toplevel):
    def __init__(self, master=None, position: tuple[int, int] = None):
        super().__init__(master)
        self.title('Insert Karras Scheduler')
        self._templates = None
        self._dropdown = None
        self._scheduler_names = _resources.get_karras_schedulers()
        self._current_scheduler = tk.StringVar(value=self._scheduler_names[0])
        self._prediction_types = [''] + _resources.get_karras_scheduler_prediction_types()
        self._current_prediction_type = tk.StringVar(value=self._prediction_types[0])
        self._entries = []
        self._content = None
        self._ok = False
        self.transient(master)
        self.grab_set()
        self.resizable(False, False)
        self._insert = False

        self._frame = tk.Frame(self)

        # Create labels
        self._scheduler_label = tk.Label(self._frame, text="Scheduler Name")
        self._prediction_type_label = tk.Label(self._frame, text="Prediction Type")

        # Create dropdowns
        self._scheduler_dropdown = tk.OptionMenu(self._frame, self._current_scheduler, *self._scheduler_names)
        self._prediction_type_dropdown = tk.OptionMenu(self._frame, self._current_prediction_type,
                                                       *self._prediction_types)

        # Create Insert button
        self._insert_button = tk.Button(self._frame, text="Insert", command=self._insert_action)

        # Grid layout
        self._scheduler_label.grid(row=0, column=0)
        self._scheduler_dropdown.grid(row=0, column=1, sticky="we")
        self._prediction_type_label.grid(row=1, column=0)
        self._prediction_type_dropdown.grid(row=1, column=1, sticky="we")
        self._insert_button.grid(row=2, column=0, columnspan=2, pady=(5, 0))

        self._frame.pack(pady=(5, 5), padx=(5, 5))

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
        self._insert = True
        self.destroy()

    def get_scheduler(self):
        self.wait_window(self)
        if not self._insert:
            return None
        prediction_type = self._current_prediction_type.get()
        if prediction_type:
            return self._current_scheduler.get() + ';prediction-type=' + self._current_prediction_type.get()
        else:
            return self._current_scheduler.get()


_last_pos = None


def request_scheduler(master):
    global _last_pos

    window = _KarrasSchedulerSelect(master, position=_last_pos)

    og_destroy = window.destroy

    def destroy():
        global _last_pos
        _last_pos = _last_size = (window.winfo_x(), window.winfo_y())
        og_destroy()

    window.destroy = destroy

    def on_closing():
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", on_closing)

    return window.get_scheduler()
