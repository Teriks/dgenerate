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

import dgenerate.console.recipesformentries.entry as _entry
import dgenerate.console.resources as _resources


class _KarrasSchedulerEntry(_entry._Entry):
    NAME = 'karrasscheduler'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=2)

        if self.arg is None:
            self.arg = '--scheduler'

        schedulers = _resources.get_karras_schedulers()

        if self.optional:
            schedulers = [''] + schedulers

        self.scheduler_name_text_var = \
            tk.StringVar(value=self.config.get('default', schedulers[0]))

        prediction_types = [''] + _resources.get_karras_scheduler_prediction_types()

        self.prediction_type_entry_text_var = \
            tk.StringVar(value=self.config.get('prediction-type', ''))

        scheduler_name_label_text = self.get_label('Scheduler')

        self.scheduler_name_label = tk.Label(
            self.master, text=scheduler_name_label_text, anchor='e')

        self.scheduler_name_entry = tk.OptionMenu(
            self.master, self.scheduler_name_text_var, *schedulers)

        self.prediction_type_label = tk.Label(
            self.master, text=scheduler_name_label_text + ' Prediction Type', anchor='e'
        )

        self.prediction_type_entry = tk.OptionMenu(
            self.master, self.prediction_type_entry_text_var, *prediction_types)

        self.scheduler_name_label.grid(row=self.row, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.scheduler_name_entry.grid(row=self.row, column=1, padx=_entry.ROW_XPAD, sticky='ew')

        self.prediction_type_label.grid(row=self.row + 1, column=0, padx=_entry.ROW_XPAD, sticky='e')
        self.prediction_type_entry.grid(row=self.row + 1, column=1, padx=_entry.ROW_XPAD, sticky='ew')

    def invalid(self):
        _entry.invalid_colors(self.scheduler_name_entry)

    def valid(self):
        _entry.valid_colors(self.scheduler_name_entry)

    def is_empty(self):
        return self.scheduler_name_text_var.get().strip() == ''

    def template(self, content):
        value = self.scheduler_name_text_var.get().strip()
        pred_type = self.prediction_type_entry_text_var.get().strip()

        if value and pred_type:
            value += f';prediction-type={pred_type}'

        return self._template(content, value)
