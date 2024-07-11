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
import dgenerate.console.recipesformentries.entry as _entry
import dgenerate.console.resources as _resources
import dgenerate.textprocessing as _textprocessing


class _ImageSeedSelect(tk.Toplevel):
    def __init__(self, parent, position=None):
        super().__init__(parent)
        self.title("Insert Image Seed URI")

        self.grid_columnconfigure(1, weight=1)

        self.configure(pady=5, padx=5)

        self.transient(parent)
        self.grab_set()
        self.resizable(True, False)

        self.entries = []
        self._create_widgets()

        self.withdraw()
        self.update_idletasks()

        self.minsize(500, self.winfo_height())

        self._set_position(position)

        self.deiconify()
        self.result = None

    def _set_position(self, position):
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

    def _create_widgets(self):

        labels = ["(Required If Inpainting) Seed Image", "(Optional) Inpaint Image", "(Optional) Control Image"]

        for i, label in enumerate(labels):
            tk.Label(self, text=label).grid(row=i, column=0, sticky=tk.E)

            entry = tk.Entry(self)
            entry.grid(row=i, column=1, padx=(2, 5), sticky=tk.EW)
            entry.bind("<Key>", self._valid)
            self.entries.append(entry)

            button = tk.Button(self, text="Open File", command=lambda e=entry: self._open_file(e))
            button.grid(row=i, column=2, padx=(0, 5), sticky=tk.W)

        self.seed_image_entry = self.entries[0]
        self.inpaint_image_entry = self.entries[1]
        self.control_image_entry = self.entries[2]

        tk.Label(self, text='(Optional) Resize Dimension (WxH)').grid(
            row=3, column=0, sticky=tk.E)

        self.resize_entry = tk.Entry(self)
        self.resize_entry.grid(row=3, column=1, padx=(2, 5), sticky=tk.EW)
        self.resize_entry.bind("<Key>", self._valid)

        tk.Label(self, text='Resize Preserves Aspect?').grid(
            row=4, column=0, sticky=tk.E)

        self.aspect_entry_var = tk.BooleanVar(value=True)

        self.aspect_entry = tk.Checkbutton(self, variable=self.aspect_entry_var)
        self.aspect_entry.grid(row=4, column=1, sticky=tk.W)

        self.insert_button = tk.Button(self, text="Insert", command=self._insert_click)
        self.insert_button.grid(row=5, column=0, columnspan=3, pady=5)

    def _open_file(self, entry):
        file_path = _filedialog.open_file_dialog(
            **_resources.get_file_dialog_args(['images-in']))
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    def _insert_click(self):

        image_seed = _entry.shell_quote_if(self.seed_image_entry.get().strip())
        inpaint_image = _entry.shell_quote_if(self.inpaint_image_entry.get().strip())
        control_image = _entry.shell_quote_if(self.control_image_entry.get().strip())

        # Validate that image seed is specified if inpaint image is provided
        if inpaint_image and not image_seed:
            _entry.invalid_colors(self.entries[0])
            return

        # Validate resize entry
        if not self._resize_entry_valid():
            _entry.invalid_colors(self.resize_entry)
            return

        resize_value = self.resize_entry.get().strip()
        aspect_value = self.aspect_entry_var.get()

        self.result = _textprocessing.format_image_seed_uri(
            seed_image=image_seed,
            inpaint_image=inpaint_image,
            control_image=control_image,
            resize=resize_value,
            aspect=aspect_value
        )

        self.destroy()

    def _resize_entry_valid(self):
        value = self.resize_entry.get().strip()
        if value == '':
            return True
        try:
            _textprocessing.parse_image_size(value)
            return True
        except ValueError:
            return False

    def _valid(self, event):
        for entry in self.entries:
            if entry.get().strip():
                _entry.valid_colors(entry)

        if self._resize_entry_valid():
            _entry.valid_colors(self.resize_entry)

    def get_uri(self):
        self.wait_window(self)
        return self.result


_last_pos = None


def request_uri(master):
    global _last_pos

    window = _ImageSeedSelect(master, position=_last_pos)

    og_destroy = window.destroy

    def destroy():
        global _last_pos
        _last_pos = (window.winfo_x(), window.winfo_y())
        og_destroy()

    window.destroy = destroy

    def on_closing():
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", on_closing)

    return window.get_uri()
