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
import tkinter.ttk as ttk
import typing

import dgenerate.console.appearance as _appearance
import dgenerate.console.filedialog as _filedialog
import dgenerate.console.formentries.entry as _entry
import dgenerate.console.resources as _resources
import dgenerate.console.textentry as _t_entry
import dgenerate.console.util as _util
import dgenerate.textprocessing as _textprocessing
from dgenerate.console.combobox import ComboBox

_dialog_state = _util.DialogState(save_position=True, save_size=True)

_URI_PARSER = _textprocessing.ConceptUriParser(
    'TextEncoder', [
        'model', 'revision', 'variant', 'subfolder', 'dtype', 'quantizer', 'mode'])

_DTYPES = ('', 'auto', 'float16', 'bfloat16', 'float32')
_FIELD_WIDTH = 48


def _files_metadata() -> dict:
    return _resources.get_schema('submodels')['Text Encoder']['model'].get('files', {})


class _TextEncoderUriDialog(tk.Toplevel):
    def __init__(self,
                 slot: dict,
                 initial: str,
                 insert: typing.Callable[[str], None],
                 master=None,
                 position: tuple[int, int] | None = None,
                 size: tuple[int, int] | None = None):
        super().__init__(master)
        self.title('Build Text Encoder URI')
        self.transient(master)
        self.resizable(True, True)
        self._insert = insert
        self._slot = slot
        self._encoders = list(slot['encoders'])
        self._modes = list(slot.get('modes') or [])

        frame = ttk.Frame(self, padding=(10, 10, 10, 0))
        frame.grid(row=0, column=0, sticky='nsew')
        frame.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        parsed = _parse_initial(initial, self._encoders)
        row = 0

        self._encoder_var = tk.StringVar(value=parsed['encoder'] or self._encoders[0])
        row = self._add_labeled(frame, row, 'Encoder', self._encoder_combo(frame))

        self._model_var = tk.StringVar(value=parsed['model'])
        self._model_entry = _t_entry.TextEntry(
            frame, textvariable=self._model_var, width=_FIELD_WIDTH)
        model_entry = self._model_entry
        model_buttons = ttk.Frame(frame)
        ttk.Button(model_buttons, text='File', command=self._pick_file).pack(side='left')
        ttk.Button(model_buttons, text='Directory', command=self._pick_directory).pack(side='left')
        row = self._add_labeled(frame, row, 'Model', model_entry, model_buttons)

        self._revision_var = tk.StringVar(value=parsed['revision'])
        self._variant_var = tk.StringVar(value=parsed['variant'])
        self._subfolder_var = tk.StringVar(value=parsed['subfolder'])
        self._revision_entry = _t_entry.TextEntry(
            frame, textvariable=self._revision_var, width=_FIELD_WIDTH)
        self._variant_entry = _t_entry.TextEntry(
            frame, textvariable=self._variant_var, width=_FIELD_WIDTH)
        self._subfolder_entry = _t_entry.TextEntry(
            frame, textvariable=self._subfolder_var, width=_FIELD_WIDTH)
        row = self._add_labeled(frame, row, 'Revision', self._revision_entry)
        row = self._add_labeled(frame, row, 'Variant', self._variant_entry)
        row = self._add_labeled(frame, row, 'Subfolder', self._subfolder_entry)

        self._dtype_var = tk.StringVar(value=parsed['dtype'] if parsed['dtype'] in _DTYPES else '')
        dtype = ComboBox(
            frame, textvariable=self._dtype_var, values=_DTYPES,
            state='readonly', width=_text_width(_DTYPES))
        row = self._add_labeled(frame, row, 'dtype', dtype)

        self._quantizer_var = tk.StringVar(value=parsed['quantizer'])
        quantizer_entry = _t_entry.TextEntry(
            frame, textvariable=self._quantizer_var, width=_FIELD_WIDTH)
        quantizer_buttons = ttk.Frame(frame)
        ttk.Button(quantizer_buttons, text='Build', command=self._build_quantizer).pack(side='left')
        row = self._add_labeled(frame, row, 'Quantizer', quantizer_entry, quantizer_buttons)

        self._mode_var = tk.StringVar(value='')
        self._mode_help = tk.StringVar(value='')
        help_label = None
        if self._modes:
            allowed = {mode['value'] for mode in self._modes}
            if parsed['mode'] in allowed:
                self._mode_var.set(parsed['mode'])
            mode_values = ('',) + tuple(mode['value'] for mode in self._modes)
            mode_box = ComboBox(
                frame,
                textvariable=self._mode_var,
                values=mode_values,
                state='readonly',
                width=_text_width(mode_values))
            mode_box.bind('<<ComboboxSelected>>', lambda _event: self._sync_mode())
            row = self._add_labeled(frame, row, 'Monolithic mode', mode_box)
            help_label = ttk.Label(frame, textvariable=self._mode_help, justify='left')
            help_label.grid(row=row, column=1, columnspan=2, sticky='w', pady=(0, 6))
            row += 1
            self._mode_var.trace_add('write', lambda *_: self._sync_mode())

        buttons = ttk.Frame(self, padding=(10, 8, 10, 10))
        buttons.grid(row=1, column=0, sticky='sew')
        ttk.Button(buttons, text='Insert', command=self._insert_uri).pack()

        self._sync_mode()
        _appearance.install(self)
        if help_label is not None:
            help_label.configure(wraplength=max(
                _field_pixels(self._encoders, mode_values, _DTYPES),
                tkfont.nametofont('TkDefaultFont').measure('n' * _FIELD_WIDTH)))
        self.update_idletasks()
        self.minsize(self.winfo_reqwidth(), self.winfo_reqheight())
        _util.position_toplevel(master, self, position=position, size=size)

    def _encoder_combo(self, frame):
        if len(self._encoders) == 1:
            return ttk.Label(frame, text=self._encoders[0], anchor='w')
        return ComboBox(
            frame,
            textvariable=self._encoder_var,
            values=self._encoders,
            state='readonly',
            width=_text_width(self._encoders))

    def _add_labeled(self, frame, row, text, widget, extra=None) -> int:
        label = ttk.Label(frame, text=text, anchor='e')
        label.grid(row=row, column=0, padx=(0, 6), pady=3, sticky='e')
        widget.grid(row=row, column=1, pady=3, sticky='ew')
        if extra is not None:
            extra.grid(row=row, column=2, padx=(4, 0), sticky='w')
        return row + 1

    def _sync_mode(self):
        selected = self._mode_var.get().strip() if self._modes else ''
        monolithic = bool(selected)
        state = 'disabled' if monolithic else 'normal'
        for widget in (self._revision_entry, self._variant_entry, self._subfolder_entry):
            widget.configure(state=state)
        if not self._modes:
            return
        if not selected:
            self._mode_help.set(
                'Leave blank for a folder, GGUF, or pre-quantized file.')
            return
        for mode in self._modes:
            if mode['value'] == selected:
                self._mode_help.set(mode['label'] + '. This replaces revision, variant, and subfolder.')
                return
        self._mode_help.set('')

    def _pick_file(self):
        metadata = _files_metadata()
        chosen = _filedialog.open_file_dialog(filetypes=metadata.get('filetypes', []))
        if chosen:
            self._model_var.set(chosen)

    def _pick_directory(self):
        chosen = _filedialog.open_directory_dialog()
        if chosen:
            self._model_var.set(chosen)

    def _build_quantizer(self):
        import dgenerate.console.quantizerselect as _quantizerselect

        def insert(value: str):
            self._quantizer_var.set(value)

        _quantizerselect.request_uri(self, insert)

    def _insert_uri(self):
        model = self._model_var.get().strip()
        if not model:
            _entry.invalid_colors(self._model_entry)
            return
        _entry.valid_colors(self._model_entry)
        parts = [self._encoder_var.get().strip(), f'model={model}']
        mode = self._mode_var.get().strip() if self._modes else ''
        if mode:
            parts.append(f'mode={mode}')
        else:
            for name, variable in (
                    ('revision', self._revision_var),
                    ('variant', self._variant_var),
                    ('subfolder', self._subfolder_var)):
                value = variable.get().strip()
                if value:
                    parts.append(f'{name}={value}')
        dtype = self._dtype_var.get().strip()
        if dtype:
            parts.append(f'dtype={dtype}')
        quantizer = self._quantizer_var.get().strip()
        if quantizer:
            if any(character in quantizer for character in ' ;"\''):
                quoted = _textprocessing.shell_quote(quantizer, double=True, strict=True)
                if quoted == quantizer:
                    quoted = '"' + quantizer.replace('"', '\\"') + '"'
                quantizer = quoted
            parts.append(f'quantizer={quantizer}')
        self._insert(';'.join(parts))
        self.destroy()


def _text_width(values) -> int:
    return max((len(str(value)) for value in values if value), default=0)


def _field_pixels(*groups) -> int:
    chars = max(_text_width(group) for group in groups)
    return tkfont.nametofont('TkDefaultFont').measure('n' * chars)


def _parse_initial(uri: str, encoders: list[str]) -> dict[str, str]:
    empty = {
        'encoder': '', 'model': '', 'revision': '', 'variant': '',
        'subfolder': '', 'dtype': '', 'quantizer': '', 'mode': ''}
    text = (uri or '').strip()
    if not text or text in ('+', 'null'):
        return empty
    try:
        parsed = _URI_PARSER.parse(text)
    except _textprocessing.ConceptUriParseError:
        empty['model'] = text
        return empty
    encoder = parsed.concept if parsed.concept in encoders else ''
    if not encoder and parsed.concept not in encoders:
        # A raw path was parsed as the concept. Keep it as the model.
        empty['model'] = text
        return empty
    values = dict(empty)
    values['encoder'] = encoder
    for name in ('model', 'revision', 'variant', 'subfolder', 'dtype', 'quantizer', 'mode'):
        value = parsed.args.get(name)
        if value is None:
            continue
        values[name] = value if isinstance(value, str) else str(value)
    return values


def request_uri(master,
                slot: dict,
                initial: str,
                insert: typing.Callable[[str], None]):
    return _util.create_singleton_dialog(
        master=master,
        dialog_class=_TextEncoderUriDialog,
        state=_dialog_state,
        dialog_kwargs={'slot': slot, 'initial': initial, 'insert': insert})
