import tkinter as tk

import dgenerate.console.filedialog as _filedialog
import dgenerate.console.recipesformentries.entry as _entry
import dgenerate.console.resources as _resources


class _ImageSeedSelect(tk.Toplevel):
    def __init__(self, parent, position=None):
        super().__init__(parent)
        self.title("Insert Image Seed URI")

        self.grid_columnconfigure(1, weight=1)

        self.resizable(True, False)
        self.transient(parent)

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
            tk.Label(self, text=label).grid(row=i, column=0, padx=5, pady=5, sticky=tk.E)

            entry = tk.Entry(self)
            entry.grid(row=i, column=1, padx=(2, 5), sticky=tk.EW)
            entry.bind("<KeyRelease>", self._valid)
            self.entries.append(entry)

            button = tk.Button(self, text="Open File", command=lambda e=entry: self._open_file(e))
            button.grid(row=i, column=2, padx=(0, 5), sticky=tk.W)

        self.insert_button = tk.Button(self, text="Insert", command=self._insert_click)
        self.insert_button.grid(row=3, column=0, columnspan=3, pady=5)

    def _open_file(self, entry):
        file_path = _filedialog.open_file_dialog(
            **_resources.get_file_dialog_args(['images-in']))
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    def _insert_click(self):
        image_seed = _entry.shell_quote_if(self.entries[0].get().strip())
        inpaint_image = _entry.shell_quote_if(self.entries[1].get().strip())
        control_image = _entry.shell_quote_if(self.entries[2].get().strip())

        # Validate that image seed is specified if inpaint image is provided
        if inpaint_image and not image_seed:
            _entry.invalid_colors(self.entries[0])
            return

        components = []
        if image_seed:
            components.append(image_seed)
        if inpaint_image:
            if control_image:
                components.append(f"mask={inpaint_image}")
            else:
                components.append(inpaint_image)
        if control_image:
            if image_seed:
                components.append(f"control={control_image}")
            else:
                components.append(control_image)

        self.result = ";".join(components)

        self.destroy()

    def _valid(self, event):
        for entry in self.entries:
            if entry.get().strip():
                _entry.valid_colors(entry)

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
