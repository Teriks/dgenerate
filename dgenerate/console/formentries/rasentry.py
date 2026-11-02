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

from dgenerate.console.formentries.featuretoggleentry import _FeatureToggleEntry

_DEFAULT_SAMPLE_RATIO = 0.5
_DEFAULT_HIGH_RATIO = 1.0
_DEFAULT_STARVATION_SCALE = 0.1
_DEFAULT_ERROR_RESET = '12,22'
_DEFAULT_METRIC = 'std'
_METRICS = ('std', 'l2norm')
_DEFAULT_START_STEP = 4
_DEFAULT_SKIP_NUM_STEP = 0
_DEFAULT_SKIP_LENGTH = 0


class _RASEntry(_FeatureToggleEntry):
    NAME = 'ras'

    def __init__(self, *args, **kwargs):
        config = kwargs.get('config')
        if config is None:
            config = args[3]
        self._index_fusion = bool(config.get('index-fusion', False))
        super().__init__(*args, **kwargs, widget_rows=11 if self._index_fusion else 10)
        self._build('Use RAS', '--ras')

        self.index_fusion_var = None
        if self._index_fusion:
            self.index_fusion_var = tk.BooleanVar(value=False)
            self._add_check_row(
                'Index Fusion', self.index_fusion_var, '--ras-index-fusion')

        self.sample_var = tk.StringVar(value=str(_DEFAULT_SAMPLE_RATIO))
        self._add_float_row(
            'Sample Ratio', self.sample_var, '--ras-sample-ratios', from_=0.0, to=1.0)

        self.high_var = tk.StringVar(value=str(_DEFAULT_HIGH_RATIO))
        self._add_float_row(
            'High-Value Ratio', self.high_var, '--ras-high-ratios', from_=0.0, to=1.0)

        self.starvation_var = tk.StringVar(value=str(_DEFAULT_STARVATION_SCALE))
        self._add_float_row(
            'Starvation Scale', self.starvation_var, '--ras-starvation-scales',
            from_=0.0, to=1.0)

        self.error_var = tk.StringVar(value=_DEFAULT_ERROR_RESET)
        self._add_string_row(
            'Error Reset Steps', self.error_var, '--ras-error-reset-steps')

        self.metric_var = tk.StringVar(value=_DEFAULT_METRIC)
        self._add_dropdown_row(
            'Metric', self.metric_var, '--ras-metrics', _METRICS)

        self.start_var = tk.StringVar(value=str(_DEFAULT_START_STEP))
        self._add_int_row(
            'Start Step', self.start_var, '--ras-start-steps', from_=1)

        self.end_var = tk.StringVar(value='')
        self._add_int_row(
            'End Step', self.end_var, '--ras-end-steps', from_=1)

        self.skip_var = tk.StringVar(value=str(_DEFAULT_SKIP_NUM_STEP))
        self._add_int_row(
            'Skip Num Step', self.skip_var, '--ras-skip-num-steps',
            from_=-sys.maxsize, to=sys.maxsize)

        self.skip_length_var = tk.StringVar(value=str(_DEFAULT_SKIP_LENGTH))
        self._add_int_row(
            'Skip Num Step Length', self.skip_length_var, '--ras-skip-num-step-lengths',
            from_=0)

        self._finish()

    def _feature_arguments(self):
        arguments = []
        if self.index_fusion_var is not None and self.index_fusion_var.get():
            arguments.append('--ras-index-fusion')
        self._append_float(
            arguments, '--ras-sample-ratios', self.sample_var, _DEFAULT_SAMPLE_RATIO)
        self._append_float(
            arguments, '--ras-high-ratios', self.high_var, _DEFAULT_HIGH_RATIO)
        self._append_float(
            arguments, '--ras-starvation-scales', self.starvation_var,
            _DEFAULT_STARVATION_SCALE)
        self._append_text(
            arguments, '--ras-error-reset-steps', self.error_var, _DEFAULT_ERROR_RESET)
        self._append_text(
            arguments, '--ras-metrics', self.metric_var, _DEFAULT_METRIC)
        self._append_int(
            arguments, '--ras-start-steps', self.start_var, _DEFAULT_START_STEP)
        self._append_int(arguments, '--ras-end-steps', self.end_var, None)
        self._append_int(
            arguments, '--ras-skip-num-steps', self.skip_var, _DEFAULT_SKIP_NUM_STEP)
        self._append_int(
            arguments, '--ras-skip-num-step-lengths', self.skip_length_var,
            _DEFAULT_SKIP_LENGTH)
        return arguments
