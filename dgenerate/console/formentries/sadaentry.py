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

# Model-specific SADA defaults from the --sada help text.
_PRESETS = {
    'sd': {
        'max_downsample': 1,
        'sx': 3,
        'sy': 3,
        'acc_range': '10,47',
        'terms': 4,
        'interval': 4,
        'step': 24,
        'max_fix': 5120,
        'max_interval': 4,
    },
    'sdxl': {
        'max_downsample': 2,
        'sx': 3,
        'sy': 3,
        'acc_range': '10,47',
        'terms': 4,
        'interval': 4,
        'step': 24,
        'max_fix': 10240,
        'max_interval': 4,
    },
    'flux': {
        'max_downsample': 0,
        'sx': 0,
        'sy': 0,
        'acc_range': '10,47',
        'terms': 3,
        'interval': 4,
        'step': 20,
        'max_fix': 0,
        'max_interval': 4,
    },
}


class _SADAEntry(_FeatureToggleEntry):
    NAME = 'sada'

    def __init__(self, *args, **kwargs):
        config = kwargs.get('config')
        if config is None:
            config = args[3]
        preset_name = config.get('preset')
        preset = _PRESETS.get(preset_name)
        if preset is None:
            raise RuntimeError(
                'SADA recipe control requires "preset" to be one of '
                + ', '.join(sorted(_PRESETS)) + '.')
        self._preset = preset

        super().__init__(*args, **kwargs, widget_rows=10)
        self._build('Use SADA', '--sada')

        self.downsample_var = tk.StringVar(value=str(preset['max_downsample']))
        self._add_int_row(
            'Max Downsample', self.downsample_var, '--sada-max-downsamples', from_=0)

        self.sx_var = tk.StringVar(value=str(preset['sx']))
        self._add_int_row(
            'Spatial Downsample X', self.sx_var, '--sada-sxs', from_=0)

        self.sy_var = tk.StringVar(value=str(preset['sy']))
        self._add_int_row(
            'Spatial Downsample Y', self.sy_var, '--sada-sys', from_=0)

        self.acc_var = tk.StringVar(value=preset['acc_range'])
        self._add_string_row(
            'Acceleration Range', self.acc_var, '--sada-acc-ranges')

        self.terms_var = tk.StringVar(value=str(preset['terms']))
        self._add_int_row(
            'Lagrange Terms', self.terms_var, '--sada-lagrange-terms', from_=0)

        self.lagrange_interval_var = tk.StringVar(value=str(preset['interval']))
        self._add_int_row(
            'Lagrange Interval', self.lagrange_interval_var, '--sada-lagrange-ints', from_=0)

        self.lagrange_step_var = tk.StringVar(value=str(preset['step']))
        self._add_int_row(
            'Lagrange Step', self.lagrange_step_var, '--sada-lagrange-steps', from_=0)

        self.max_fix_var = tk.StringVar(value=str(preset['max_fix']))
        self._add_int_row(
            'Max Fixed Memory', self.max_fix_var, '--sada-max-fixes', from_=0)

        self.max_interval_var = tk.StringVar(value=str(preset['max_interval']))
        self._add_int_row(
            'Max Interval', self.max_interval_var, '--sada-max-intervals',
            from_=0, to=sys.maxsize)

        self._finish()

    def _feature_arguments(self):
        preset = self._preset
        arguments = []
        self._append_int(
            arguments, '--sada-max-downsamples', self.downsample_var, preset['max_downsample'])
        self._append_int(arguments, '--sada-sxs', self.sx_var, preset['sx'])
        self._append_int(arguments, '--sada-sys', self.sy_var, preset['sy'])
        self._append_text(
            arguments, '--sada-acc-ranges', self.acc_var, preset['acc_range'])
        self._append_int(
            arguments, '--sada-lagrange-terms', self.terms_var, preset['terms'])
        self._append_int(
            arguments, '--sada-lagrange-ints', self.lagrange_interval_var, preset['interval'])
        self._append_int(
            arguments, '--sada-lagrange-steps', self.lagrange_step_var, preset['step'])
        self._append_int(
            arguments, '--sada-max-fixes', self.max_fix_var, preset['max_fix'])
        self._append_int(
            arguments, '--sada-max-intervals', self.max_interval_var, preset['max_interval'])
        return arguments
