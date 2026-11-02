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

# Keep in sync with dgenerate.pipelinewrapper.constants.
_DEFAULT_WEIGHT = 0.5
_DEFAULT_ORDER = 4
_DEFAULT_LAMBDA = 0.1
_DEFAULT_WARMUP_STEPS = 2
_DEFAULT_WINDOW_SIZE = 2.0
_DEFAULT_FLEX_WINDOW = 0.75


class _SpectrumEntry(_FeatureToggleEntry):
    NAME = 'spectrum'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, widget_rows=7)
        self._build('Use Spectrum', '--spectrum')

        self.weight_var = tk.StringVar(value=str(_DEFAULT_WEIGHT))
        self._add_float_row(
            'Blend Weight', self.weight_var, '--spectrum-weights', from_=0.0, to=1.0)

        self.order_var = tk.StringVar(value=str(_DEFAULT_ORDER))
        self._add_int_row(
            'Chebyshev Order', self.order_var, '--spectrum-orders', from_=1, to=sys.maxsize)

        self.lambda_var = tk.StringVar(value=str(_DEFAULT_LAMBDA))
        self._add_float_row(
            'Ridge Penalty', self.lambda_var, '--spectrum-lambdas', from_=0.0)

        self.warmup_var = tk.StringVar(value=str(_DEFAULT_WARMUP_STEPS))
        self._add_int_row(
            'Warmup Steps', self.warmup_var, '--spectrum-warmup-steps', from_=0)

        self.window_var = tk.StringVar(value=str(_DEFAULT_WINDOW_SIZE))
        self._add_float_row(
            'Window Size', self.window_var, '--spectrum-window-sizes', from_=1.0)

        self.flex_var = tk.StringVar(value=str(_DEFAULT_FLEX_WINDOW))
        self._add_float_row(
            'Flex Window', self.flex_var, '--spectrum-flex-windows', from_=0.0)

        self._finish()

    def _feature_arguments(self):
        arguments = []
        self._append_float(
            arguments, '--spectrum-weights', self.weight_var, _DEFAULT_WEIGHT)
        self._append_int(
            arguments, '--spectrum-orders', self.order_var, _DEFAULT_ORDER)
        self._append_float(
            arguments, '--spectrum-lambdas', self.lambda_var, _DEFAULT_LAMBDA)
        self._append_int(
            arguments, '--spectrum-warmup-steps', self.warmup_var, _DEFAULT_WARMUP_STEPS)
        self._append_float(
            arguments, '--spectrum-window-sizes', self.window_var, _DEFAULT_WINDOW_SIZE)
        self._append_float(
            arguments, '--spectrum-flex-windows', self.flex_var, _DEFAULT_FLEX_WINDOW)
        return arguments
