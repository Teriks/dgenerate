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

import dgenerate.console.formentries.entry as _entry
from dgenerate.console.spinbox import FloatSpinbox, IntSpinbox

# Keep in sync with dgenerate.pipelinewrapper.constants.
_DEFAULT_WARMUP_STEP = 6
_DEFAULT_POST_STEP = 2
_DEFAULT_REFRESH_STEP = 16
_DEFAULT_THRESHOLD = 0.93
_DEFAULT_CACHE_THRESHOLD = 0.04
_DEFAULT_EROSION_DILATION = True


class _RegionEEntry(_entry._Entry):
    NAME = 'regione'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=7)

        self.enabled_var = tk.BooleanVar(value=self.config.get('default', False))
        self.bool_var = self.enabled_var

        self.label_widget = ttk.Label(
            self.master, text=self.get_label('Use RegionE'), anchor='e')
        self.checkbox = ttk.Checkbutton(self.master, variable=self.enabled_var)
        self.label_widget.grid(row=self.row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.checkbox.grid(row=self.row, column=1, sticky='w')

        self._int_entries = []
        self._float_entries = []
        self._int_spin_buttons = []
        self._float_spin_buttons = []

        self.warmup_var = tk.StringVar(
            value=str(self.config.get('warmup-step', _DEFAULT_WARMUP_STEP)))
        self._add_int_row(
            row=self.row + 1,
            label='Warmup Steps',
            text_var=self.warmup_var,
            from_=0,
            to=sys.maxsize)

        self.post_var = tk.StringVar(
            value=str(self.config.get('post-step', _DEFAULT_POST_STEP)))
        self._add_int_row(
            row=self.row + 2,
            label='Post Steps',
            text_var=self.post_var,
            from_=0,
            to=sys.maxsize)

        self.refresh_var = tk.StringVar(
            value=str(self.config.get('refresh-step', _DEFAULT_REFRESH_STEP)))
        self._add_int_row(
            row=self.row + 3,
            label='Refresh Step',
            text_var=self.refresh_var,
            from_=0,
            to=sys.maxsize)

        self.threshold_var = tk.StringVar(
            value=str(self.config.get('threshold', _DEFAULT_THRESHOLD)))
        self._add_float_row(
            row=self.row + 4,
            label='Region Threshold',
            text_var=self.threshold_var,
            from_=0.0,
            to=1.0)

        self.cache_threshold_var = tk.StringVar(
            value=str(self.config.get('cache-threshold', _DEFAULT_CACHE_THRESHOLD)))
        self._add_float_row(
            row=self.row + 5,
            label='Cache Threshold',
            text_var=self.cache_threshold_var,
            from_=0.0,
            to=sys.maxsize)

        self.erosion_var = tk.BooleanVar(
            value=self.config.get('erosion-dilation', _DEFAULT_EROSION_DILATION))
        self.erosion_label = ttk.Label(
            self.master, text='Mask Erosion/Dilation', anchor='e')
        self.erosion_checkbox = ttk.Checkbutton(self.master, variable=self.erosion_var)
        self.erosion_label.grid(
            row=self.row + 6, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.erosion_checkbox.grid(row=self.row + 6, column=1, sticky='w')

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

    def _add_int_row(self, row, label, text_var, from_, to):
        label_widget = ttk.Label(self.master, text=label + ' (int)', anchor='e')
        entry = IntSpinbox(self.master, textvariable=text_var, from_=from_, to=to)
        spin_buttons = entry.create_spin_buttons(self.master)
        label_widget.grid(row=row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        entry.grid(row=row, column=1, padx=_entry.ROW_XPAD, sticky='ew')
        spin_buttons.grid(row=row, column=2, sticky='w')
        text_var.trace_add('write', lambda *_args: self.valid())
        self._int_entries.append(entry)
        self._int_spin_buttons.append(spin_buttons)

    def _add_float_row(self, row, label, text_var, from_, to):
        label_widget = ttk.Label(self.master, text=label + ' (float)', anchor='e')
        entry = FloatSpinbox(
            self.master, from_=from_, to=to, increment=0.01, textvariable=text_var)
        spin_buttons = entry.create_spin_buttons(self.master)
        label_widget.grid(row=row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        entry.grid(row=row, column=1, padx=_entry.ROW_XPAD, sticky='ew')
        spin_buttons.grid(row=row, column=2, sticky='w')
        text_var.trace_add('write', lambda *_args: self.valid())
        self._float_entries.append(entry)
        self._float_spin_buttons.append(spin_buttons)

    def sync_widget_states(self):
        self._update_controls()

    def _update_controls(self, *_args):
        enabled = self.enabled_var.get()
        state = tk.NORMAL if enabled else tk.DISABLED
        for entry in self._int_entries + self._float_entries:
            entry.config(state=state)
        for spin_buttons in self._int_spin_buttons + self._float_spin_buttons:
            for child in spin_buttons.children.values():
                child.config(state=state)
        self.erosion_label.state(['!disabled' if enabled else 'disabled'])
        self.erosion_checkbox.state(['!disabled' if enabled else 'disabled'])
        if not enabled:
            self.valid()

    def is_valid(self):
        if not self.enabled_var.get():
            return True
        return all(e.is_valid() for e in self._int_entries + self._float_entries)

    def is_active(self):
        return self.enabled_var.get()

    def invalid(self):
        for entry in self._int_entries + self._float_entries:
            if not entry.is_valid():
                _entry.invalid_colors(entry)

    def valid(self):
        for entry in self._int_entries + self._float_entries:
            _entry.valid_colors(entry)

    def exclude(self, other_entry):
        other_var = getattr(other_entry, 'bool_var', None)
        if other_var is None:
            other_vars = getattr(other_entry, 'bool_vars', None)
            if other_vars:
                other_var = other_vars[0]
        if other_var is None:
            other_var = getattr(other_entry, 'enabled_var', None)
        if other_var is None:
            return

        self.enabled_var.trace_add(
            'write', lambda *_args: other_var.set(False) if self.enabled_var.get() else None)
        other_var.trace_add(
            'write', lambda *_args: self.enabled_var.set(False) if other_var.get() else None)

    def template(self, content):
        if not self.enabled_var.get():
            return _entry.replace_first(content, self.placeholder, '')

        arguments = ['--regione']

        warmup = self.warmup_var.get().strip()
        if warmup and int(warmup) != _DEFAULT_WARMUP_STEP:
            arguments.append('--regione-warmup-steps ' + warmup)

        post = self.post_var.get().strip()
        if post and int(post) != _DEFAULT_POST_STEP:
            arguments.append('--regione-post-steps ' + post)

        refresh = self.refresh_var.get().strip()
        if refresh and int(refresh) != _DEFAULT_REFRESH_STEP:
            arguments.append('--regione-refresh-steps ' + refresh)

        threshold = self.threshold_var.get().strip()
        if threshold and float(threshold) != _DEFAULT_THRESHOLD:
            arguments.append('--regione-thresholds ' + threshold)

        cache_threshold = self.cache_threshold_var.get().strip()
        if cache_threshold and float(cache_threshold) != _DEFAULT_CACHE_THRESHOLD:
            arguments.append('--regione-cache-thresholds ' + cache_threshold)

        if self.erosion_var.get() != _DEFAULT_EROSION_DILATION:
            if self.erosion_var.get():
                arguments.append('--regione-erosion-dilation')
            else:
                arguments.append('--regione-no-erosion-dilation')

        return _entry.replace_first(content, self.placeholder, '\n'.join(arguments))
