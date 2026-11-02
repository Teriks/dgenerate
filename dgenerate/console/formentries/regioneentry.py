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

# Keep in sync with dgenerate.extras.regione.runtime pipeline defaults.
_PRESETS = {
    'flux-kontext': {
        'warmup_step': 6,
        'post_step': 2,
        'refresh_step': 16,
        'threshold': 0.93,
        'cache_threshold': 0.04,
        'erosion_dilation': True,
    },
    'qwen-edit': {
        'warmup_step': 6,
        'post_step': 2,
        'refresh_step': 16,
        'threshold': 0.80,
        'cache_threshold': 0.03,
        'erosion_dilation': True,
    },
}


class _RegionEEntry(_FeatureToggleEntry):
    NAME = 'regione'

    def __init__(self, *args, **kwargs):
        config = kwargs.get('config')
        if config is None:
            config = args[3]
        preset_name = config.get('preset')
        preset = _PRESETS.get(preset_name)
        if preset is None:
            raise RuntimeError(
                'RegionE recipe control requires "preset" to be one of '
                + ', '.join(sorted(_PRESETS)) + '.')
        self._preset = preset

        super().__init__(*args, **kwargs, widget_rows=7)
        self._build('Use RegionE', '--regione')

        self.warmup_var = tk.StringVar(value=str(preset['warmup_step']))
        self._add_int_row(
            'Warmup Steps', self.warmup_var, '--regione-warmup-steps', from_=0)

        self.post_var = tk.StringVar(value=str(preset['post_step']))
        self._add_int_row(
            'Post Steps', self.post_var, '--regione-post-steps', from_=0)

        self.refresh_var = tk.StringVar(value=str(preset['refresh_step']))
        self._add_int_row(
            'Refresh Step', self.refresh_var, '--regione-refresh-steps', from_=0)

        self.threshold_var = tk.StringVar(value=str(preset['threshold']))
        self._add_float_row(
            'Region Threshold', self.threshold_var, '--regione-thresholds',
            from_=0.0, to=1.0)

        self.cache_threshold_var = tk.StringVar(value=str(preset['cache_threshold']))
        self._add_float_row(
            'Cache Threshold', self.cache_threshold_var, '--regione-cache-thresholds',
            from_=0.0)

        self.erosion_var = tk.BooleanVar(value=preset['erosion_dilation'])
        self._add_check_row(
            'Mask Erosion/Dilation', self.erosion_var, '--regione-erosion-dilation')

        self._finish()

    def _feature_arguments(self):
        preset = self._preset
        arguments = []
        self._append_int(
            arguments, '--regione-warmup-steps', self.warmup_var, preset['warmup_step'])
        self._append_int(
            arguments, '--regione-post-steps', self.post_var, preset['post_step'])
        self._append_int(
            arguments, '--regione-refresh-steps', self.refresh_var, preset['refresh_step'])
        self._append_float(
            arguments, '--regione-thresholds', self.threshold_var, preset['threshold'])
        self._append_float(
            arguments, '--regione-cache-thresholds', self.cache_threshold_var,
            preset['cache_threshold'])
        if self.erosion_var.get() != preset['erosion_dilation']:
            if self.erosion_var.get():
                arguments.append('--regione-erosion-dilation')
            else:
                arguments.append('--regione-no-erosion-dilation')
        return arguments
