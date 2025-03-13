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

import dgenerate.console.filedialog as _filedialog
import dgenerate.console.recipesformentries.entry as _entry
import dgenerate.console.resources as _resources
import dgenerate.console.spinbox as _spinbox
import dgenerate.console.util as _util
import dgenerate.textprocessing as _textprocessing


class _ImageSeedSelect(tk.Toplevel):
    def __init__(self, insert: typing.Callable[[str], None], master=None, position: tuple[int, int] = None):
        super().__init__(master)
        self.title("Insert Image Seed URI")

        self.grid_columnconfigure(1, weight=1)

        self.configure(pady=5, padx=5)

        self.transient(master)
        self.resizable(True, False)

        self.entries = []
        self._create_widgets()

        self.minsize(500, self.winfo_height())

        self._insert = insert

        _util.position_toplevel(master, self, position=position)

    def _create_widgets(self):

        labels = ["(Required If Inpainting) Seed Image", "(Optional) Inpaint Mask Image", "(Optional) Control Image"]

        for i, label in enumerate(labels):
            tk.Label(self, text=label).grid(row=i, column=0, sticky=tk.E)

            entry = tk.Entry(self)
            entry.grid(row=i, column=1, padx=(2, 5), sticky=tk.EW)
            entry.bind("<Key>", self._valid)
            self.entries.append(entry)

            button = tk.Button(self, text="Open File", command=lambda e=entry: self._open_file(e))
            button.grid(row=i, column=2, padx=(0, 5), sticky=tk.W)

        self.seed_image_entry = self.entries[0]
        self.mask_image_entry = self.entries[1]
        self.control_image_entry = self.entries[2]

        tk.Label(self, text='(Optional) Resize Dimension (WxH)').grid(
            row=3, column=0, sticky=tk.E)

        self.resize_entry = tk.Entry(self)
        self.resize_entry.grid(row=3, column=1, padx=(2, 5), sticky=tk.EW)
        self.resize_entry.bind("<Key>", self._valid)
        self.entries.append(self.resize_entry)

        tk.Label(self, text='Resize Preserves Aspect?').grid(
            row=4, column=0, sticky=tk.E)

        self.aspect_entry_var = tk.BooleanVar(value=True)

        self.aspect_entry = tk.Checkbutton(self, variable=self.aspect_entry_var)
        self.aspect_entry.grid(row=4, column=1, sticky=tk.W)

        tk.Label(self, text='(Optional) Frame Start').grid(
            row=5, column=0, sticky=tk.E)

        self.frame_start_entry = _spinbox.IntSpinbox(self, from_=0, textvariable=tk.StringVar(value=''))
        self.frame_start_entry.create_spin_buttons(self).grid(row=5, column=2, sticky=tk.W)
        self.frame_start_entry.grid(row=5, column=1, padx=(2, 5), sticky=tk.EW)
        self.frame_start_entry.bind("<Key>", self._valid)
        self.entries.append(self.frame_start_entry)

        tk.Label(self, text='(Optional) Frame End').grid(
            row=6, column=0, sticky=tk.E)

        self.frame_end_entry = _spinbox.IntSpinbox(self, from_=0, textvariable=tk.StringVar(value=''))
        self.frame_end_entry.create_spin_buttons(self).grid(row=6, column=2, sticky=tk.W)
        self.frame_end_entry.grid(row=6, column=1, padx=(2, 5), sticky=tk.EW)
        self.frame_end_entry.bind("<Key>", self._valid)
        self.entries.append(self.frame_end_entry)

        self.insert_button = tk.Button(self, text="Insert", command=self._insert_click)
        self.insert_button.grid(row=7, column=0, columnspan=3, pady=5)

    @staticmethod
    def _open_file(entry):
        file_path = _filedialog.open_file_dialog(
            **_resources.get_file_dialog_args(['images-in']))
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    def _insert_click(self):

        image_seed = _entry.shell_quote_if(self.seed_image_entry.get().strip(), strict=True)
        mask_image = _entry.shell_quote_if(self.mask_image_entry.get().strip(), strict=True)
        control_image = _entry.shell_quote_if(self.control_image_entry.get().strip(), strict=True)

        # Validate that image seed is specified if inpaint image is provided
        if mask_image and not image_seed:
            _entry.invalid_colors(self.entries[0])
            return

        # Validate resize entry
        if not self._resize_entry_valid():
            _entry.invalid_colors(self.resize_entry)
            return

        resize_value = self.resize_entry.get().strip()
        aspect_value = self.aspect_entry_var.get()

        frame_start_value = self.frame_start_entry.get().strip()
        frame_end_value = self.frame_end_entry.get().strip()

        frame_start = int(frame_start_value) if frame_start_value else None
        frame_end = int(frame_end_value) if frame_end_value else None

        # validate that seed_image or control_image is specified if any keyword arguments are provided
        if (frame_start is not None or
            frame_end is not None or
            aspect_value is False or resize_value) and not (
                image_seed or control_image):
            _entry.invalid_colors(self.entries[0])
            _entry.invalid_colors(self.entries[2])
            return

        value = _textprocessing.format_image_seed_uri(
            seed_images=image_seed,
            mask_images=mask_image,
            control_images=control_image,
            resize=resize_value,
            aspect=aspect_value,
            frame_start=frame_start,
            frame_end=frame_end
        )

        if value:
            self._insert(value)

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
            _entry.valid_colors(entry)


_last_pos = None
_cur_window = None


def request_uri(master, insert: typing.Callable[[str], None]):
    global _last_pos, _cur_window

    if _cur_window is not None:
        _cur_window.focus_set()
        return

    _cur_window = _ImageSeedSelect(
        insert=insert,
        master=master,
        position=_last_pos
    )

    og_destroy = _cur_window.destroy

    # noinspection PyUnresolvedReferences
    def destroy():
        global _last_pos, _cur_window
        _last_pos = (_cur_window.winfo_x(), _cur_window.winfo_y())
        _cur_window = None
        og_destroy()

    _cur_window.destroy = destroy

    def on_closing():
        _cur_window.destroy()

    _cur_window.protocol("WM_DELETE_WINDOW", on_closing)
