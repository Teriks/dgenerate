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
import dgenerate.console.formentries.entry as _entry
import dgenerate.console.resources as _resources
import dgenerate.console.spinbox as _spinbox
import dgenerate.console.util as _util
import dgenerate.textprocessing as _textprocessing
import dgenerate.console.textentry as _t_entry

_dialog_state = _util.DialogState(save_position=True, save_size=False)


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

        labels = [
            "(Required If Inpainting) Seed Image",
            "(Optional) Inpaint Mask Image",
            "(Optional) Control Image",
            "(Optional) IP Adapter Image",
            "(Optional) Latents (.pt/.pth/.safetensors)"
        ]

        for i, label in enumerate(labels):
            tk.Label(self, text=label).grid(row=i, column=0, sticky=tk.E)

            entry = _t_entry.TextEntry(self)
            entry.grid(row=i, column=1, padx=(2, 5), sticky=tk.EW)
            entry.bind("<Key>", self._valid)
            self.entries.append(entry)

            if i == 3:  # Adapter image entry
                button = tk.Button(self, text="File", command=lambda e=entry: self._open_adapter_file(e))
            elif i == 4:  # Latents entry
                button = tk.Button(self, text="File", command=lambda e=entry: self._open_latents_file(e))
            else:
                button = tk.Button(self, text="File", command=lambda e=entry: self._open_file(e))
            button.grid(row=i, column=2, padx=(0, 5), sticky=tk.W)

        self.seed_image_entry = self.entries[0]
        self.mask_image_entry = self.entries[1]
        self.control_image_entry = self.entries[2]
        self.adapter_image_entry = self.entries[3]
        self.latents_entry = self.entries[4]

        tk.Label(self, text='(Optional) Resize Dimension (WxH)').grid(
            row=5, column=0, sticky=tk.E)

        self.resize_entry = _t_entry.TextEntry(self)
        self.resize_entry.grid(row=5, column=1, padx=(2, 5), sticky=tk.EW)
        self.resize_entry.bind("<Key>", self._valid)
        self.entries.append(self.resize_entry)

        tk.Label(self, text='Resize Preserves Aspect?').grid(
            row=6, column=0, sticky=tk.E)

        self.aspect_entry_var = tk.BooleanVar(value=True)

        self.aspect_entry = tk.Checkbutton(self, variable=self.aspect_entry_var)
        self.aspect_entry.grid(row=6, column=1, sticky=tk.W)

        tk.Label(self, text='(Optional) Frame Start').grid(
            row=7, column=0, sticky=tk.E)

        self.frame_start_entry = _spinbox.IntSpinbox(self, from_=0, textvariable=tk.StringVar(value=''))
        self.frame_start_entry.create_spin_buttons(self).grid(row=7, column=2, sticky=tk.W)
        self.frame_start_entry.grid(row=7, column=1, padx=(2, 5), sticky=tk.EW)
        self.frame_start_entry.bind("<Key>", self._valid)
        self.entries.append(self.frame_start_entry)

        tk.Label(self, text='(Optional) Frame End').grid(
            row=8, column=0, sticky=tk.E)

        self.frame_end_entry = _spinbox.IntSpinbox(self, from_=0, textvariable=tk.StringVar(value=''))
        self.frame_end_entry.create_spin_buttons(self).grid(row=8, column=2, sticky=tk.W)
        self.frame_end_entry.grid(row=8, column=1, padx=(2, 5), sticky=tk.EW)
        self.frame_end_entry.bind("<Key>", self._valid)
        self.entries.append(self.frame_end_entry)

        self.insert_button = tk.Button(self, text="Insert", command=self._insert_click)
        self.insert_button.grid(row=9, column=0, columnspan=3, pady=5)

    @staticmethod
    def _open_file(entry):
        file_path = _filedialog.open_file_dialog(
            **_resources.get_file_dialog_args(['images-in']))
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    @staticmethod
    def _open_latents_file(entry):
        file_path = _filedialog.open_file_dialog(
            title="Select Latents File",
            filetypes=[
                ("Tensor files", "*.pt *.pth *.safetensors"),
                ("PyTorch files", "*.pt *.pth"),
                ("SafeTensors files", "*.safetensors"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    @staticmethod
    def _open_adapter_file(entry):
        file_path = _filedialog.open_file_dialog(
            **_resources.get_file_dialog_args(['images-in']))
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    def _insert_click(self):

        image_seed = _entry.shell_quote_if(self.seed_image_entry.get().strip(), strict=True)
        mask_image = _entry.shell_quote_if(self.mask_image_entry.get().strip(), strict=True)
        control_image = _entry.shell_quote_if(self.control_image_entry.get().strip(), strict=True)
        latents = _entry.shell_quote_if(self.latents_entry.get().strip(), strict=True)
        adapter_image = _entry.shell_quote_if(self.adapter_image_entry.get().strip(), strict=True)

        # Convert empty strings to None
        image_seed = image_seed if image_seed else None
        mask_image = mask_image if mask_image else None
        control_image = control_image if control_image else None
        latents = latents if latents else None

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

        # Check if only latents are specified (no images that can be resized)
        if latents and not (image_seed or control_image or adapter_image):
            if resize_value:
                _entry.invalid_colors(self.resize_entry)
                return
            # Also check aspect ratio - only matters for image resizing
            if aspect_value is False:  # Non-default value
                _entry.invalid_colors(self.aspect_entry)
                return
            # Check frame parameters - latents don't have animation frames
            frame_start_value = self.frame_start_entry.get().strip()
            frame_end_value = self.frame_end_entry.get().strip()
            if frame_start_value or frame_end_value:
                _entry.invalid_colors(self.frame_start_entry)
                _entry.invalid_colors(self.frame_end_entry)
                return

        frame_start_value = self.frame_start_entry.get().strip()
        frame_end_value = self.frame_end_entry.get().strip()

        frame_start = int(frame_start_value) if frame_start_value else None
        frame_end = int(frame_end_value) if frame_end_value else None

        # validate that at least one input source is specified if any keyword arguments are provided
        if (frame_start is not None or
            frame_end is not None or
            aspect_value is False or resize_value) and not (
                image_seed or control_image or latents or adapter_image):
            _entry.invalid_colors(self.entries[0])
            _entry.invalid_colors(self.entries[2])
            _entry.invalid_colors(self.entries[3])
            _entry.invalid_colors(self.entries[4])
            return

        # validate that at least one input source is specified
        if not (image_seed or control_image or latents or adapter_image):
            _entry.invalid_colors(self.entries[0])
            _entry.invalid_colors(self.entries[2])
            _entry.invalid_colors(self.entries[3])
            _entry.invalid_colors(self.entries[4])
            return

        value = _textprocessing.format_image_seed_uri(
            seed_images=image_seed,
            mask_images=mask_image,
            control_images=control_image,
            latents=latents,
            adapter_images=adapter_image,
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


def request_uri(master, insert: typing.Callable[[str], None], dialog_state: _util.DialogState | None = None):
    return _util.create_singleton_dialog(
        master=master,
        dialog_class=_ImageSeedSelect,
        state=_dialog_state if dialog_state is None else dialog_state,
        dialog_kwargs={'insert': insert}
    )
