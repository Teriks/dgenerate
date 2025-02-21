import importlib.util
import shutil

import dgenerate.console.recipesformentries.entry as _entry
import tkinter as tk


class _QuantizerEntry(_entry._Entry):
    NAME = 'quantizer'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=1)

        if self.arg is None:
            self.arg = '--quantizer'

        opts = []

        if importlib.util.find_spec('bitsandbytes') is not None:
            opts += [
                'bnb;bits=8',
                'bnb;bits=4;bits4-compute-dtype=float16',
                'bnb;bits=4;bits4-compute-dtype=bloat16',
                'bnb;bits=4;bits4-compute-dtype=float32'
            ]

        if importlib.util.find_spec('torchao') is not None:
            opts += [
                'torchao;type=int8wo',
                'torchao;type=int4wo',
            ]

        options = self.config.get(
            'options', opts
        )

        if self.optional:
            options = [''] + options

        self.text_var = tk.StringVar(
            value=self.config.get('default', options[0]))

        self.label_widget = tk.Label(
            self.master,
            text=self.get_label('Quantizer'), anchor='e')

        self.entry = tk.OptionMenu(self.master,
                                   self.text_var,
                                   *options)

        if len(opts) <= 1:
            self.entry.configure(state="disabled")

        self.label_widget.grid(row=self.row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.entry.grid(row=self.row, column=1, padx=_entry.ROW_XPAD, sticky='ew')

    def invalid(self):
        _entry.invalid_colors(self.entry)

    def valid(self):
        _entry.valid_colors(self.entry)

    def is_empty(self):
        return self.text_var.get().strip() == ''

    def template(self, content):
        return self._template(content, self.text_var.get().strip())
