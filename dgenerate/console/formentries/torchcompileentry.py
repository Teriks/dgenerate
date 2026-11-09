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

import collections.abc

import tkinter as tk
import tkinter.ttk as ttk

import dgenerate.console.appearance as _appearance
import dgenerate.console.formentries.entry as _entry
import dgenerate.console.quantizermapselect as _quantizer_map_select
import dgenerate.console.textentry as _t_entry

# Keep in sync with TORCH_COMPILE_MAP_NAMES in pipelinewrapper/pipelines.py.
# prompt_enhancer is an alias of prompt_upsampler, so it is not a checkbox.
# pipelines.py imports torch and diffusers, so the console copies the names.
TORCH_COMPILE_SUBMODULES = (
    'unet',
    'transformer',
    'transformer_2',
    'controlnet',
    'vae',
    'text_encoder',
    'text_encoder_2',
    'text_encoder_3',
    'image_encoder',
    'prompt_upsampler',
    'connectors',
)

# Models that have a separate prompt upsampler or Gemma enhancer.
_PROMPT_UPSAMPLER_MODEL_TYPES = frozenset({'flux3', 'ltx2'})


def torch_compile_options_for_model_type(model_type: str | None) -> list[str]:
    """Checkbox names for one ``--model-type``. Unknown types keep the full list."""
    if not model_type or not str(model_type).strip():
        return list(TORCH_COMPILE_SUBMODULES)
    key = str(model_type).strip().lower()
    quantizer_names = set(_quantizer_map_select.quantizer_map_options_for_model_type(key))
    quantizer_names.add('vae')
    if key in _PROMPT_UPSAMPLER_MODEL_TYPES:
        quantizer_names.add('prompt_upsampler')
    return [name for name in TORCH_COMPILE_SUBMODULES if name in quantizer_names]


def parse_torch_compile_text(text: str) -> list[str]:
    selected = []
    for part in str(text).replace(',', ' ').split():
        name = part.strip()
        if name == 'prompt_enhancer':
            name = 'prompt_upsampler'
        if name and name in TORCH_COMPILE_SUBMODULES and name not in selected:
            selected.append(name)
    return selected


def format_torch_compile_text(selected: collections.abc.Collection[str]) -> str:
    chosen = set(selected)
    if 'prompt_enhancer' in chosen:
        chosen.add('prompt_upsampler')
        chosen.discard('prompt_enhancer')
    return ' '.join(name for name in TORCH_COMPILE_SUBMODULES if name in chosen)


class _TorchCompileEntry(_entry._Entry):
    NAME = 'torchcompile'

    def __init__(self, *args, **kwargs):
        if 'widget_rows' not in kwargs:
            kwargs['widget_rows'] = 2
        super().__init__(*args, **kwargs)
        if self.arg is None:
            self.arg = '--torch-compile'
        # The shared help binder reads these names for the second label.
        self._map_arg = self.arg

        label = self.config.get('label', 'Torch Compile')
        if label == 'Torch Compile':
            components_label = 'Components'
        else:
            components_label = label.replace('Torch Compile', 'Components')
        self.config.setdefault('components-label', components_label)

        self.bool_var = tk.BooleanVar(value=self.config.get('default', False))
        self.text_var = tk.StringVar(
            value=format_torch_compile_text(parse_torch_compile_text(
                self.config.get('components-default', ''))))

        self.label_widget = ttk.Label(
            self.master, text=self.get_label(label), anchor='e')
        self.checkbox = ttk.Checkbutton(self.master, variable=self.bool_var)

        self._map_label = ttk.Label(
            self.master,
            text=self.get_label('Components', key='components-label'),
            anchor='e')
        self.entry = _t_entry.TextEntry(self.master, textvariable=self.text_var)
        button_frame = ttk.Frame(self.master)
        self.build_button = ttk.Button(
            button_frame, text='Build', command=self._open_builder)

        self.label_widget.grid(row=self.row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.checkbox.grid(row=self.row, column=1, sticky='w')
        self._map_label.grid(row=self.row + 1, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.entry.grid(row=self.row + 1, column=1, padx=(5, 2), sticky='ew')
        button_frame.grid(row=self.row + 1, column=2, padx=(2, 5), sticky='w')
        self.build_button.pack(side='left')

        for widget in (self.label_widget, self.checkbox, self._map_label,
                       self.entry, self.build_button):
            _appearance.apply(widget)

        self.bool_var.trace_add('write', lambda *_args: self.sync_widget_states())
        self.sync_widget_states()

    def is_active(self) -> bool:
        return self.bool_var.get()

    def sync_widget_states(self) -> None:
        if self._row_inactive:
            return
        state = tk.NORMAL if self.bool_var.get() else tk.DISABLED
        self.entry.configure(state=state)
        self.build_button.configure(state=state)

    def _options(self) -> list[str]:
        return torch_compile_options_for_model_type(self.config.get('model-type'))

    def _open_builder(self):
        def apply_modules(value: str):
            self.text_var.set(value)

        model_type = self.config.get('model-type')
        if model_type:
            model_type = str(model_type).strip()
        _quantizer_map_select.open_quantizer_map_select(
            self.form.winfo_toplevel(),
            options=self._options(),
            initial_text=self.text_var.get(),
            apply=apply_modules,
            model_type=model_type,
            title='Torch compile' + (f' ({model_type})' if model_type else ''),
            prompt='Select components passed to --torch-compile. '
                   'An empty list keeps the default set.',
            parse=parse_torch_compile_text,
        )

    def template(self, content):
        if not self.bool_var.get():
            return _entry.replace_first(content, self.placeholder, '')
        value = self.text_var.get().strip()
        emitted = self.arg if not value else f'{self.arg} {value}'
        return _entry.replace_first(content, self.placeholder, emitted)
