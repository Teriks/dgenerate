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

import dgenerate.console.filedialog as _filedialog
import dgenerate.console.formentries.entry as _entry
import dgenerate.console.resources as _resources
import dgenerate.console.textentry as _t_entry


def format_text_encoder_values(values: list[str | None]) -> str:
    """
    Join per-slot values into a ``--text-encoders`` argument.

    ``None`` is a slot left at its default. Those slots are written as ``+``
    when a later slot is set, and trailing defaults are omitted.
    """
    tokens = list(values)
    while tokens and tokens[-1] is None:
        tokens.pop()
    if not tokens:
        return ''
    return ' '.join('+' if token is None else token for token in tokens)


def _with_encoder_class(text: str, encoders: list[str]) -> str:
    """A bare path from the file button has no encoder class. The slot's class is filled in."""
    head = text.split(';', 1)[0]
    if head in encoders:
        return text
    return f'{encoders[0]};model={text}'


class _TextEncodersEntry(_entry._Entry):
    """
    One optional URI field per text encoder slot of a ``--model-type``.

    The encoder class is chosen in the URI builder. ``mode`` is offered only
    for slots that have a monolithic CLIP or T5 checkpoint mode, and it is
    left blank unless the user picks one. A ``null?`` checkbox is offered only
    for a slot the pipeline still runs without, which is the SD3 T5 encoder.
    It sits beside that slot's label so the file buttons stay the same width
    as every other row.
    ``defaults`` is a list aligned to those slots. ``null`` leaves a slot
    blank, which is written as ``+`` when a later slot is set.
    """

    NAME = 'textencoders'

    def __init__(self, *args, **kwargs):
        config = kwargs.get('config', {})
        model_type = config.get('model-type')
        if not model_type:
            raise RuntimeError('textencoders recipe entry requires "model-type".')
        schema = _resources.get_schema('textencoders')
        if model_type not in schema:
            raise RuntimeError(
                f'textencoders has no slots for model type {model_type!r}.')
        self._slots = schema[model_type]
        kwargs['widget_rows'] = len(self._slots)
        super().__init__(*args, **kwargs)

        if self.arg is None:
            self.arg = '--text-encoders'

        files = _resources.get_schema('submodels')['Text Encoder']['model'].get('files', {})
        self._filetypes = files.get('filetypes', [])
        self.uri_vars: list[tk.StringVar] = []
        self.bool_vars: list[tk.BooleanVar] = []
        self.label_widgets = []
        self._entries = []
        self._buttons = []

        prefix = self.config.get('label', 'Text Encoder')
        defaults = self.config.get('defaults') or []
        for index, slot in enumerate(self._slots):
            number = '' if len(self._slots) == 1 else f' {index + 1}'
            label_text = f'{prefix}{number} ({slot["label"]})'
            if self.optional:
                label_text = f'(Optional) {label_text}'
            label_box = ttk.Frame(self.master)
            label = ttk.Label(label_box, text=label_text, anchor='e')
            label.pack(side='left')
            initial = ''
            if index < len(defaults) and defaults[index] not in (None, ''):
                initial = str(defaults[index])
            uri_var = tk.StringVar(value=initial)
            skip_var = tk.BooleanVar(value=False)
            entry = _t_entry.TextEntry(self.master, textvariable=uri_var)
            buttons = ttk.Frame(self.master)

            def pick_file(variable=uri_var):
                chosen = _filedialog.open_file_dialog(filetypes=self._filetypes)
                if chosen:
                    variable.set(chosen)

            def pick_directory(variable=uri_var):
                chosen = _filedialog.open_directory_dialog()
                if chosen:
                    variable.set(chosen)

            def build(variable=uri_var, slot=slot, skip=skip_var):
                import dgenerate.console.textencoderselect as _textencoderselect

                def insert(value: str):
                    variable.set(value)
                    skip.set(False)

                dialog = _textencoderselect.request_uri(
                    self.form, slot, variable.get(), insert)
                original = dialog.destroy

                def destroy():
                    original()
                    if hasattr(self.form, 'bind_mousewheel'):
                        self.form.bind_mousewheel()

                dialog.destroy = destroy
                if hasattr(self.form, 'unbind_mousewheel'):
                    self.form.unbind_mousewheel()

            file_button = ttk.Button(buttons, text='File', command=pick_file)
            dir_button = ttk.Button(buttons, text='Directory', command=pick_directory)
            build_button = ttk.Button(buttons, text='Build URI', command=build)
            file_button.pack(side='left')
            dir_button.pack(side='left')
            build_button.pack(side='left')
            if slot.get('nullable'):
                ttk.Checkbutton(
                    label_box, text='null?', variable=skip_var, padding=0
                ).pack(side='left', padx=(2, 0))

            row = self.row + index
            label_box.grid(row=row, column=0, padx=_entry.ROW_XPAD, sticky='e')
            entry.grid(row=row, column=1, padx=_entry.ROW_XPAD, sticky='ew')
            buttons.grid(row=row, column=2, padx=(2, 5), sticky='w')

            self.uri_vars.append(uri_var)
            self.bool_vars.append(skip_var)
            self.label_widgets.append(label)
            self._entries.append(entry)
            self._buttons.append((file_button, dir_button, build_button, entry))
            skip_var.trace_add('write', lambda *_: self.sync_widget_states())

        self.args = [self.arg] * len(self.label_widgets)

    def sync_widget_states(self) -> None:
        if self._row_inactive:
            return
        for skip, buttons in zip(self.bool_vars, self._buttons):
            state = 'disabled' if skip.get() else 'normal'
            for widget in buttons:
                widget.configure(state=state)

    def is_empty(self) -> bool:
        if any(skip.get() for skip in self.bool_vars):
            return False
        return not any(variable.get().strip() for variable in self.uri_vars)

    def template(self, content):
        values = []
        for uri_var, skip, slot in zip(self.uri_vars, self.bool_vars, self._slots):
            if skip.get():
                values.append('null')
                continue
            text = uri_var.get().strip()
            if not text or text == '+':
                values.append(None)
            elif text == 'null':
                values.append('null')
            else:
                values.append(_entry.shell_quote_if(_with_encoder_class(text, slot['encoders'])))
        return self._template(content, format_text_encoder_values(values))
