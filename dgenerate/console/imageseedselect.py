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
import tkinter.font as tkfont
import typing

import dgenerate.console.filedialog as _filedialog
import dgenerate.console.formentries.entry as _entry
import dgenerate.console.resources as _resources
import dgenerate.console.spinbox as _spinbox
import dgenerate.console.textentry as _t_entry
import dgenerate.console.util as _util
import dgenerate.textprocessing as _textprocessing

_dialog_state = _util.DialogState(save_position=True, save_size=False)

_CARD = '#f7f7f7'
_LINE = '#d0d0d0'
_HINT = '#666666'


class _ImageSeedSelect(tk.Toplevel):
    def __init__(self, insert: typing.Callable[[str], None], master=None, position: tuple[int, int] = None):
        super().__init__(master)
        self.title('Insert Image Seed URI')
        self.configure(padx=10, pady=8)
        self.transient(master)
        self.resizable(True, True)

        base = tkfont.nametofont('TkDefaultFont')
        self._title_font = base.copy()
        self._title_font.configure(weight='bold')
        self._hint_font = base.copy()
        self._hint_font.configure(size=max(base.cget('size') - 1, 8))

        self.entries = []
        self._insert = insert
        self._row = 0

        self._canvas = tk.Canvas(self, width=740, height=560, highlightthickness=0, borderwidth=0)
        self._scroll = tk.Scrollbar(self, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._scroll.set)
        self._inner = tk.Frame(self._canvas, padx=12, pady=10)
        self._inner.grid_columnconfigure(0, weight=1)
        self._window = self._canvas.create_window((0, 0), window=self._inner, anchor='nw')
        self._inner.bind('<Configure>', self._fit_scroll)
        self._canvas.bind('<Configure>', self._fit_inner_width)

        self._canvas.grid(row=0, column=0, sticky=tk.NSEW)
        self._scroll.grid(row=0, column=1, sticky=tk.NS, padx=(6, 0))
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        bar = tk.Frame(self)
        bar.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(8, 0))
        tk.Frame(bar, height=1, bg=_LINE).pack(fill=tk.X, pady=(0, 8))
        self._insert_button = tk.Button(bar, text='Insert', width=16, command=self._insert_click)
        self._insert_button.pack()

        self._build_form()
        self._hook_wheel(self._inner)
        self.minsize(680, 420)
        _util.position_toplevel(master, self, position=position)

    def _fit_scroll(self, _event):
        self._canvas.configure(scrollregion=self._canvas.bbox('all'))

    def _fit_inner_width(self, event):
        self._canvas.itemconfigure(self._window, width=event.width)

    def _hook_wheel(self, widget):
        if isinstance(widget, (_spinbox.IntSpinbox, _spinbox.FloatSpinbox)):
            return
        widget.bind('<MouseWheel>', self._on_wheel)
        widget.bind('<Button-4>', self._on_wheel_up)
        widget.bind('<Button-5>', self._on_wheel_down)
        for child in widget.winfo_children():
            self._hook_wheel(child)

    def _on_wheel(self, event):
        self._canvas.yview_scroll(-1 if event.delta > 0 else 1, 'units')
        return 'break'

    def _on_wheel_up(self, _event):
        self._canvas.yview_scroll(-1, 'units')
        return 'break'

    def _on_wheel_down(self, _event):
        self._canvas.yview_scroll(1, 'units')
        return 'break'

    def _show_if_hidden(self, widget):
        self.update_idletasks()
        canvas = self._canvas
        if not self.winfo_viewable() or canvas.winfo_height() <= 1:
            return
        margin = 8
        visible_top = canvas.winfo_rooty()
        visible_bottom = visible_top + canvas.winfo_height()
        widget_top = widget.winfo_rooty()
        widget_bottom = widget_top + max(widget.winfo_height(), 1)
        if widget_top >= visible_top + margin and widget_bottom <= visible_bottom - margin:
            return
        bbox = canvas.bbox('all')
        if not bbox:
            return
        span = max(bbox[3] - bbox[1], 1)
        view_height = canvas.winfo_height()
        if widget_bottom - widget_top >= view_height or widget_top < visible_top + margin:
            delta = widget_top - visible_top - margin
        else:
            delta = widget_bottom - (visible_bottom - margin)
        top = canvas.yview()[0] + delta / span
        canvas.yview_moveto(min(1, max(0, top)))

    def _track(self, widget):
        widget.bind('<Key>', self._valid)
        self.entries.append(widget)
        return widget

    def _section(self, title, hint):
        header = tk.Frame(self._inner)
        header.grid(row=self._row, column=0, sticky=tk.EW, pady=(12, 2))
        self._row += 1
        tk.Label(header, text=title, font=self._title_font).pack(side=tk.LEFT)
        tk.Frame(header, height=1, bg=_LINE).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        if hint:
            tk.Label(
                self._inner, text=hint, font=self._hint_font, fg=_HINT, anchor=tk.W
            ).grid(row=self._row, column=0, sticky=tk.W)
            self._row += 1
        return header

    def _build_form(self):
        self._section('Pictures', 'Add a row for each file. One mask covers every seed, or use one mask per seed.')
        self._seeds = _FileRows(self, self._open_image, 'Seed image', start=1)
        self._masks = _FileRows(self, self._open_image, 'Inpaint mask')
        self._controls = _FileRows(self, self._open_image, 'Control image')

        self._section('Latents', 'Tensor files (.pt, .pth, .safetensors), one per seed when seeds are set.')
        self._latents = _FileRows(self, self._open_latents, 'Latents')

        self._section('IP Adapter', 'Each row is one image. Rows are combined with +.')
        self._adapters = _AdapterRows(self)

        self._section('Size and frames', None)
        self._resize_entry = self._labeled_entry('Resize (WxH)')
        aspect_row = tk.Frame(self._inner)
        aspect_row.grid(row=self._row, column=0, sticky=tk.W, pady=2)
        self._row += 1
        self._aspect_var = tk.BooleanVar(value=True)
        tk.Checkbutton(aspect_row, text='Keep aspect ratio', variable=self._aspect_var).pack(side=tk.LEFT)
        self._frame_start = self._labeled_spin('Frame start', 0, None)
        self._frame_end = self._labeled_spin('Frame end', 0, None)

        self._section('LTX', 'The last frame stays at full strength. Index is a latent frame, strength is 0 to 1.')
        end_row = tk.Frame(self._inner)
        end_row.grid(row=self._row, column=0, sticky=tk.EW, pady=2)
        self._row += 1
        end_row.grid_columnconfigure(1, weight=1)
        tk.Label(end_row, text='Last frame').grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        self._end_entry = self._track(_t_entry.TextEntry(end_row))
        self._end_entry.grid(row=0, column=1, sticky=tk.EW)
        tk.Button(end_row, text='File', command=lambda: self._open_image(self._end_entry)).grid(
            row=0, column=2, padx=(4, 0))

        self._ltx_index = self._labeled_spin('Latent frame index', -100000, 100000)
        self._ltx_strength = self._labeled_float('Condition strength')
        self._extras = _ExtraRows(self)

    def _labeled_entry(self, label):
        row = tk.Frame(self._inner)
        row.grid(row=self._row, column=0, sticky=tk.EW, pady=2)
        self._row += 1
        row.grid_columnconfigure(1, weight=1)
        tk.Label(row, text=label).grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        entry = self._track(_t_entry.TextEntry(row))
        entry.grid(row=0, column=1, sticky=tk.EW)
        return entry

    def _labeled_spin(self, label, low, high):
        row = tk.Frame(self._inner)
        row.grid(row=self._row, column=0, sticky=tk.EW, pady=2)
        self._row += 1
        row.grid_columnconfigure(1, weight=1)
        tk.Label(row, text=label).grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        spin = _spinbox.IntSpinbox(row, from_=low, to=high, textvariable=tk.StringVar(value=''))
        spin.grid(row=0, column=1, sticky=tk.EW)
        spin.create_spin_buttons(row).grid(row=0, column=2, padx=(4, 0))
        self._track(spin)
        return spin

    def _labeled_float(self, label):
        row = tk.Frame(self._inner)
        row.grid(row=self._row, column=0, sticky=tk.EW, pady=2)
        self._row += 1
        row.grid_columnconfigure(1, weight=1)
        tk.Label(row, text=label).grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        spin = _spinbox.FloatSpinbox(
            row, from_=0, to=1, increment=0.1, textvariable=tk.StringVar(value=''))
        spin.grid(row=0, column=1, sticky=tk.EW)
        spin.create_spin_buttons(row).grid(row=0, column=2, padx=(4, 0))
        self._track(spin)
        return spin

    def _place(self, widget):
        widget.grid(row=self._row, column=0, sticky=tk.EW, pady=(0, 4))
        self._row += 1

    @staticmethod
    def _fill(entry, file_path):
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    def _open_image(self, entry):
        self._fill(entry, _filedialog.open_file_dialog(
            **_resources.get_file_dialog_args(['images-in', 'videos-in'])))

    def _open_latents(self, entry):
        self._fill(entry, _filedialog.open_file_dialog(
            title='Select Latents File',
            filetypes=[
                ('Tensor files', '*.pt *.pth *.safetensors'),
                ('PyTorch files', '*.pt *.pth'),
                ('SafeTensors files', '*.safetensors'),
                ('All files', '*.*')
            ]))

    def _paths_argument(self, paths):
        if not paths:
            return None
        if len(paths) == 1:
            return paths[0]
        return paths

    def _insert_click(self):
        if not self._ltx_index.is_valid():
            _entry.invalid_colors(self._ltx_index)
            return
        if not self._ltx_strength.is_valid():
            _entry.invalid_colors(self._ltx_strength)
            return
        if not self._frame_start.is_valid() or not self._frame_end.is_valid():
            _entry.invalid_colors(self._frame_start)
            _entry.invalid_colors(self._frame_end)
            return

        seeds = self._seeds.paths()
        masks = self._masks.paths()
        controls = self._controls.paths()
        latents = self._latents.paths()
        end_image = _entry.shell_quote_if(self._end_entry.get().strip(), strict=True) or None

        if masks and not seeds:
            self._seeds.mark()
            return
        if len(masks) > 1 and len(masks) != len(seeds):
            self._masks.mark()
            return
        if seeds and latents and len(seeds) != len(latents):
            self._latents.mark()
            self._seeds.mark()
            return

        resize_value = self._resize_entry.get().strip()
        if resize_value:
            try:
                _textprocessing.parse_image_size(resize_value)
            except ValueError:
                _entry.invalid_colors(self._resize_entry)
                return
        aspect_value = self._aspect_var.get()

        frame_start_value = self._frame_start.get().strip()
        frame_end_value = self._frame_end.get().strip()
        frame_start = int(frame_start_value) if frame_start_value else None
        frame_end = int(frame_end_value) if frame_end_value else None
        ltx_index_value = self._ltx_index.get().strip()
        ltx_strength_value = self._ltx_strength.get().strip()
        ltx_index = int(ltx_index_value) if ltx_index_value else None
        ltx_strength = float(ltx_strength_value) if ltx_strength_value else None

        try:
            adapters = self._adapters.values()
            extras = self._extras.values()
        except _RowError as error:
            _entry.invalid_colors(error.widget)
            return

        latents_only = latents and not (seeds or controls or adapters or end_image)
        if latents_only and (resize_value or aspect_value is False or
                             frame_start is not None or frame_end is not None or
                             ltx_index is not None or ltx_strength is not None or extras):
            _entry.invalid_colors(self._resize_entry)
            return
        if (ltx_index is not None or ltx_strength is not None or extras) and not seeds:
            self._seeds.mark()
            return
        if (frame_start is not None or frame_end is not None or aspect_value is False or resize_value) and not (
                seeds or controls or latents or adapters or end_image):
            self._seeds.mark()
            return
        if not (seeds or controls or latents or adapters or end_image):
            self._seeds.mark()
            return

        try:
            value = _textprocessing.format_image_seed_uri(
                seed_images=self._paths_argument(seeds),
                mask_images=self._paths_argument(masks),
                control_images=self._paths_argument(controls),
                latents=self._paths_argument(latents),
                adapter_images=self._paths_argument(adapters),
                resize=resize_value or None,
                aspect=aspect_value,
                frame_start=frame_start,
                frame_end=frame_end,
                end_image=end_image,
                ltx_index=ltx_index,
                ltx_strength=ltx_strength,
                ltx_extra_conditions=extras or None
            )
        except ValueError:
            self._seeds.mark()
            return

        if value:
            self._insert(value)
        self.destroy()

    def _valid(self, _event):
        for entry in self.entries:
            _entry.valid_colors(entry)


class _RowError(Exception):
    def __init__(self, widget):
        self.widget = widget


class _FileRows:
    def __init__(self, dialog: _ImageSeedSelect, opener, title, start=0):
        self.dialog = dialog
        self.opener = opener
        self.entries = []
        self._frames = []

        shell = tk.Frame(dialog._inner)
        dialog._place(shell)
        shell.grid_columnconfigure(0, weight=1)
        header = tk.Frame(shell)
        header.grid(row=0, column=0, sticky=tk.EW)
        tk.Label(header, text=title).pack(side=tk.LEFT)
        tk.Button(header, text='Add', command=self.add).pack(side=tk.RIGHT)
        self.body = tk.Frame(shell)
        self.body.grid(row=1, column=0, sticky=tk.EW)
        self.body.grid_columnconfigure(0, weight=1)
        for _ in range(start):
            self.add()

    def add(self):
        row = tk.Frame(self.body)
        row.grid(row=len(self._frames), column=0, sticky=tk.EW, pady=1)
        row.grid_columnconfigure(0, weight=1)
        entry = self.dialog._track(_t_entry.TextEntry(row))
        entry.grid(row=0, column=0, sticky=tk.EW)
        tk.Button(row, text='File', command=lambda e=entry: self.opener(e)).grid(row=0, column=1, padx=(4, 0))
        tk.Button(row, text='Remove', command=lambda r=row, e=entry: self.remove(r, e)).grid(
            row=0, column=2, padx=(4, 0))
        self._frames.append(row)
        self.entries.append(entry)
        self.dialog._hook_wheel(row)
        self.dialog._show_if_hidden(row)

    def remove(self, row, entry):
        if entry in self.entries:
            self.entries.remove(entry)
        if entry in self.dialog.entries:
            self.dialog.entries.remove(entry)
        if row in self._frames:
            self._frames.remove(row)
        row.destroy()
        for index, frame in enumerate(self._frames):
            frame.grid(row=index, column=0, sticky=tk.EW, pady=1)
        self.body.configure(height=1)
        self.dialog.update_idletasks()

    def paths(self):
        values = []
        for entry in self.entries:
            text = entry.get().strip()
            if text:
                values.append(_entry.shell_quote_if(text, strict=True))
        return values

    def mark(self):
        if not self.entries:
            self.add()
        for entry in self.entries:
            _entry.invalid_colors(entry)


class _AdapterRows:
    def __init__(self, dialog: _ImageSeedSelect):
        self.dialog = dialog
        self.rows = []
        shell = tk.Frame(dialog._inner)
        dialog._place(shell)
        shell.grid_columnconfigure(0, weight=1)
        header = tk.Frame(shell)
        header.grid(row=0, column=0, sticky=tk.EW)
        tk.Label(header, text='Adapter images').pack(side=tk.LEFT)
        tk.Button(header, text='Add', command=self.add).pack(side=tk.RIGHT)
        self.body = tk.Frame(shell)
        self.body.grid(row=1, column=0, sticky=tk.EW)
        self.body.grid_columnconfigure(0, weight=1)

    def add(self):
        card = tk.Frame(self.body, bg=_CARD, highlightbackground=_LINE, highlightthickness=1, padx=6, pady=4)
        card.grid(row=len(self.rows), column=0, sticky=tk.EW, pady=3)
        card.grid_columnconfigure(1, weight=1)
        tk.Label(card, text='Image', bg=_CARD).grid(row=0, column=0, sticky=tk.W)
        path = self.dialog._track(_t_entry.TextEntry(card))
        path.grid(row=0, column=1, sticky=tk.EW, padx=4)
        tk.Button(card, text='File', command=lambda e=path: self.dialog._open_image(e)).grid(row=0, column=2)
        tk.Button(card, text='Remove', command=lambda c=card: self.remove(c)).grid(row=0, column=3, padx=(4, 0))

        tk.Label(card, text='Resize', bg=_CARD).grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
        options = tk.Frame(card, bg=_CARD)
        options.grid(row=1, column=1, columnspan=3, sticky=tk.EW, pady=(4, 0))
        resize = self.dialog._track(_t_entry.TextEntry(options, width=12))
        resize.pack(side=tk.LEFT)
        tk.Label(options, text='Align', bg=_CARD).pack(side=tk.LEFT, padx=(8, 4))
        align = _spinbox.IntSpinbox(options, from_=1, to=256, width=6, textvariable=tk.StringVar(value=''))
        align.pack(side=tk.LEFT)
        align.create_spin_buttons(options).pack(side=tk.LEFT, padx=(2, 0))
        self.dialog._track(align)
        aspect = tk.BooleanVar(value=True)
        tk.Checkbutton(options, text='Aspect', variable=aspect, bg=_CARD).pack(side=tk.LEFT, padx=(8, 0))

        self.rows.append({'card': card, 'path': path, 'resize': resize, 'align': align, 'aspect': aspect})
        self.dialog._hook_wheel(card)
        self.dialog._show_if_hidden(card)

    def remove(self, card):
        kept = []
        for row in self.rows:
            if row['card'] is card:
                for widget in (row['path'], row['resize'], row['align']):
                    if widget in self.dialog.entries:
                        self.dialog.entries.remove(widget)
                card.destroy()
            else:
                kept.append(row)
        self.rows = kept
        for index, row in enumerate(self.rows):
            row['card'].grid(row=index, column=0, sticky=tk.EW, pady=3)
        self.body.configure(height=1)
        self.dialog.update_idletasks()

    def values(self):
        uris = []
        for row in self.rows:
            path = row['path'].get().strip()
            resize = row['resize'].get().strip()
            align = row['align'].get().strip()
            aspect = row['aspect'].get()
            if not path and not resize and not align and aspect:
                continue
            if not path:
                raise _RowError(row['path'])
            if not row['align'].is_valid():
                raise _RowError(row['align'])
            if resize:
                try:
                    _textprocessing.parse_image_size(resize)
                except ValueError:
                    raise _RowError(row['resize'])
            uri = _entry.shell_quote_if(path, strict=True)
            if resize:
                uri += f'|resize={resize}'
            if align and align != '1':
                uri += f'|align={align}'
            if not aspect:
                uri += '|aspect=false'
            uris.append(uri)
        return uris


class _ExtraRows:
    def __init__(self, dialog: _ImageSeedSelect):
        self.dialog = dialog
        self.rows = []
        shell = tk.Frame(dialog._inner)
        dialog._place(shell)
        shell.grid_columnconfigure(0, weight=1)
        header = tk.Frame(shell)
        header.grid(row=0, column=0, sticky=tk.EW, pady=(6, 0))
        tk.Label(header, text='Extra conditions').pack(side=tk.LEFT)
        tk.Button(header, text='Add', command=self.add).pack(side=tk.RIGHT)
        self.body = tk.Frame(shell)
        self.body.grid(row=1, column=0, sticky=tk.EW)
        self.body.grid_columnconfigure(0, weight=1)

    def add(self):
        card = tk.Frame(self.body, bg=_CARD, highlightbackground=_LINE, highlightthickness=1, padx=6, pady=4)
        card.grid(row=len(self.rows), column=0, sticky=tk.EW, pady=3)
        card.grid_columnconfigure(1, weight=1)
        tk.Label(card, text='Image', bg=_CARD).grid(row=0, column=0, sticky=tk.W)
        path = self.dialog._track(_t_entry.TextEntry(card))
        path.grid(row=0, column=1, sticky=tk.EW, padx=4)
        tk.Button(card, text='File', command=lambda e=path: self.dialog._open_image(e)).grid(row=0, column=2)
        tk.Button(card, text='Remove', command=lambda c=card: self.remove(c)).grid(row=0, column=3, padx=(4, 0))

        tk.Label(card, text='Index', bg=_CARD).grid(row=1, column=0, sticky=tk.W, pady=(4, 0))
        index = _spinbox.IntSpinbox(card, from_=-100000, to=100000, textvariable=tk.StringVar(value=''))
        index.grid(row=1, column=1, sticky=tk.EW, padx=4, pady=(4, 0))
        index.create_spin_buttons(card).grid(row=1, column=2, pady=(4, 0))
        self.dialog._track(index)

        strength_row = tk.Frame(card, bg=_CARD)
        strength_row.grid(row=2, column=0, columnspan=4, sticky=tk.EW, pady=(4, 0))
        strength_row.grid_columnconfigure(1, weight=1)
        tk.Label(strength_row, text='Strength', bg=_CARD).grid(row=0, column=0, sticky=tk.W)
        strength = _spinbox.FloatSpinbox(
            strength_row, from_=0, to=1, increment=0.1, textvariable=tk.StringVar(value=''))
        strength.grid(row=0, column=1, sticky=tk.EW, padx=4)
        strength.create_spin_buttons(strength_row).grid(row=0, column=2)
        self.dialog._track(strength)

        self.rows.append({'card': card, 'path': path, 'index': index, 'strength': strength})
        self.dialog._hook_wheel(card)
        self.dialog._show_if_hidden(card)

    def remove(self, card):
        kept = []
        for row in self.rows:
            if row['card'] is card:
                for widget in (row['path'], row['index'], row['strength']):
                    if widget in self.dialog.entries:
                        self.dialog.entries.remove(widget)
                card.destroy()
            else:
                kept.append(row)
        self.rows = kept
        for index, row in enumerate(self.rows):
            row['card'].grid(row=index, column=0, sticky=tk.EW, pady=3)
        self.body.configure(height=1)
        self.dialog.update_idletasks()

    def values(self):
        extras = []
        for row in self.rows:
            path = _entry.shell_quote_if(row['path'].get().strip(), strict=True)
            index_value = row['index'].get().strip()
            strength_value = row['strength'].get().strip()
            if not path and not index_value and not strength_value:
                continue
            if not row['index'].is_valid():
                raise _RowError(row['index'])
            if not row['strength'].is_valid():
                raise _RowError(row['strength'])
            if not path or index_value == '':
                raise _RowError(row['path'] if not path else row['index'])
            extras.append((
                path,
                int(index_value),
                float(strength_value) if strength_value else None
            ))
        return extras


def request_uri(master, insert: typing.Callable[[str], None], dialog_state: _util.DialogState | None = None):
    return _util.create_singleton_dialog(
        master=master,
        dialog_class=_ImageSeedSelect,
        state=_dialog_state if dialog_state is None else dialog_state,
        dialog_kwargs={'insert': insert}
    )
