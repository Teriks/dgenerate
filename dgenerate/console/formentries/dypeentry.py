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

import sys
import tkinter as tk
import tkinter.ttk as ttk

import dgenerate.console.combobox as _combobox
import dgenerate.console.formentries.entry as _entry
from dgenerate.console.spinbox import IntSpinbox


class _DyPEEntry(_entry._Entry):
    NAME = 'dype'

    _METHODS = ('yarn', 'ntk', 'base')

    _MSCALES = ('constant', 'decay', 'off')

    _AXIS_SCALINGS = ('independent', 'shared')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=5)

        self.enabled_var = tk.BooleanVar(value=self.config.get('default', False))
        self.label_widget = ttk.Label(
            self.master, text=self.get_label('Use DyPE'), anchor='e')
        self.checkbox = ttk.Checkbutton(self.master, variable=self.enabled_var)
        self.label_widget.grid(row=self.row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.checkbox.grid(row=self.row, column=1, sticky='w')

        self.method_var = tk.StringVar(value=self.config.get('default-method', 'yarn'))
        self.method_label = ttk.Label(self.master, text='Method', anchor='e')
        self.method_entry = _combobox.ComboBox(
            self.master, textvariable=self.method_var, values=self._METHODS)
        self.method_label.grid(
            row=self.row + 1, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.method_entry.grid(
            row=self.row + 1, column=1, padx=_entry.ROW_XPAD, sticky='ew')

        self.resolution_var = tk.StringVar(value=self.config.get('base-resolution', ''))
        self.resolution_label = ttk.Label(
            self.master, text='Base Resolution (pixels)', anchor='e')
        self.resolution_entry = IntSpinbox(
            self.master, textvariable=self.resolution_var, from_=16, to=sys.maxsize)
        self.resolution_label.grid(
            row=self.row + 2,
            column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.resolution_entry.grid(
            row=self.row + 2,
            column=1, padx=_entry.ROW_XPAD, sticky='ew')
        self.resolution_var.trace_add('write', lambda *_args: self.valid())

        self.mscale_var = tk.StringVar(value=self.config.get('default-mscale', 'constant'))
        self.mscale_label = ttk.Label(self.master, text='YARN Magnitude Scale', anchor='e')
        self.mscale_entry = _combobox.ComboBox(
            self.master, textvariable=self.mscale_var, values=self._MSCALES)
        self.mscale_label.grid(
            row=self.row + 3, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.mscale_entry.grid(
            row=self.row + 3, column=1, padx=_entry.ROW_XPAD, sticky='ew')

        self.axis_scaling_var = tk.StringVar(
            value=self.config.get('default-axis-scaling', 'shared'))
        self.axis_scaling_label = ttk.Label(self.master, text='Axis Scaling', anchor='e')
        self.axis_scaling_entry = _combobox.ComboBox(
            self.master, textvariable=self.axis_scaling_var, values=self._AXIS_SCALINGS)
        self.axis_scaling_label.grid(
            row=self.row + 4, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.axis_scaling_entry.grid(
            row=self.row + 4, column=1, padx=_entry.ROW_XPAD, sticky='ew')

        self.enabled_var.trace_add('write', self._update_controls)
        self._update_controls()

        if hasattr(self.form, 'get_entry_by_id'):
            exclude = self.config.get('exclude', None)
            if isinstance(exclude, str):
                exclude = [exclude]
            for other_id in exclude or []:
                other_entry = self.form.get_entry_by_id(other_id)
                if other_entry is not None:
                    self.exclude(other_entry)

    def sync_widget_states(self):
        self._update_controls()

    def _update_controls(self, *_args):
        enabled = self.enabled_var.get()
        state = tk.NORMAL if enabled else tk.DISABLED
        self.method_label.state(['!disabled' if enabled else 'disabled'])
        self.method_entry.config(state=state)
        self.resolution_label.state(['!disabled' if enabled else 'disabled'])
        self.resolution_entry.config(state=state)
        self.mscale_label.state(['!disabled' if enabled else 'disabled'])
        self.mscale_entry.config(state=state)
        self.axis_scaling_label.state(['!disabled' if enabled else 'disabled'])
        self.axis_scaling_entry.config(state=state)
        if not enabled:
            self.valid()

    def is_valid(self):
        return not self.enabled_var.get() or self.resolution_entry.is_valid()

    def is_active(self):
        return self.enabled_var.get()

    def invalid(self):
        if self.enabled_var.get() and not self.resolution_entry.is_valid():
            _entry.invalid_colors(self.resolution_entry)

    def valid(self):
        _entry.valid_colors(self.resolution_entry)

    def exclude(self, other_entry):
        other_var = getattr(other_entry, 'bool_var', None)
        if other_var is None:
            other_vars = getattr(other_entry, 'bool_vars', None)
            if other_vars:
                other_var = other_vars[0]
        if other_var is None:
            return

        self.enabled_var.trace_add(
            'write', lambda *_args: other_var.set(False) if self.enabled_var.get() else None)
        other_var.trace_add(
            'write', lambda *_args: self.enabled_var.set(False) if other_var.get() else None)

    def template(self, content):
        if not self.enabled_var.get():
            return _entry.replace_first(content, self.placeholder, '')

        arguments = ['--dype-methods ' + self.method_var.get()]
        resolution = self.resolution_var.get().strip()
        if resolution:
            arguments.append('--dype-base-resolutions ' + resolution)
        mscale = self.mscale_var.get().strip()
        if mscale and mscale != 'constant':
            arguments.append('--dype-mscales ' + mscale)
        axis_scaling = self.axis_scaling_var.get().strip()
        if axis_scaling and axis_scaling != 'shared':
            arguments.append('--dype-axis-scalings ' + axis_scaling)
        return _entry.replace_first(content, self.placeholder, '\n'.join(arguments))
