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
import typing

import dgenerate.console.appearance as _appearance
import dgenerate.console.util as _util

# Names accepted by --quantizer-map. Order is the checkbox and text order.
QUANTIZER_MAP_SUBMODULES = (
    'unet',
    'transformer',
    'transformer_2',
    'text_encoder',
    'text_encoder_2',
    'text_encoder_3',
    'image_encoder',
    'controlnet',
    'connectors',
)

# Submodules that exist on the pipeline for that --model-type and that
# --quantizer-map will accept. The VAE is never a map target.
_QUANTIZER_MAP_BY_MODEL_TYPE = {
    'sd': ('unet', 'text_encoder', 'controlnet'),
    'pix2pix': ('unet', 'text_encoder', 'controlnet'),
    'upscaler-x2': ('unet', 'text_encoder'),
    'upscaler-x4': ('unet', 'text_encoder'),
    'if': ('unet', 'text_encoder'),
    'ifs': ('unet', 'text_encoder'),
    'ifs-img2img': ('unet', 'text_encoder'),
    'sdxl': ('unet', 'text_encoder', 'text_encoder_2', 'controlnet'),
    'sdxl-pix2pix': ('unet', 'text_encoder', 'text_encoder_2', 'controlnet'),
    'kolors': ('unet', 'text_encoder', 'controlnet'),
    's-cascade': ('text_encoder',),
    's-cascade-decoder': ('text_encoder',),
    'sd3': ('transformer', 'text_encoder', 'text_encoder_2', 'text_encoder_3', 'controlnet'),
    'sd3-pix2pix': ('transformer', 'text_encoder', 'text_encoder_2', 'text_encoder_3'),
    'flux': ('transformer', 'text_encoder', 'text_encoder_2', 'controlnet'),
    'flux-fill': ('transformer', 'text_encoder', 'text_encoder_2'),
    'flux-kontext': ('transformer', 'text_encoder', 'text_encoder_2'),
    'flux2': ('transformer', 'text_encoder'),
    'flux2-klein-kv': ('transformer', 'text_encoder'),
    'z-image': ('transformer', 'text_encoder', 'controlnet'),
    'z-image-omni': ('transformer', 'text_encoder'),
    'qwen-image': ('transformer', 'text_encoder', 'controlnet'),
    'qwen-image-edit': ('transformer', 'text_encoder'),
    'qwen-image-layered': ('transformer', 'text_encoder'),
    'qwen-image-21': ('transformer', 'text_encoder'),
    'krea2': ('transformer', 'text_encoder'),
    'kandinsky5-image': ('transformer', 'text_encoder', 'text_encoder_2'),
    'hydit': ('transformer', 'text_encoder', 'text_encoder_2', 'controlnet'),
    'hyimage-21': ('transformer', 'text_encoder', 'text_encoder_2'),
    'hyimage-21-refiner': ('transformer', 'text_encoder'),
    'ltx2': ('transformer', 'text_encoder', 'connectors'),
    'ltx-video': ('transformer', 'text_encoder'),
    'wan': ('transformer', 'transformer_2', 'text_encoder', 'image_encoder'),
    'wan-animate': ('transformer', 'text_encoder', 'image_encoder'),
    'wan-animate-2': ('transformer', 'text_encoder', 'image_encoder'),
    'hyvideo-15': ('transformer', 'text_encoder'),
    'motif-video': ('transformer', 'text_encoder'),
    'kandinsky5-video': ('transformer', 'text_encoder', 'text_encoder_2'),
}


def quantizer_map_options_for_model_type(model_type: str | None) -> list[str]:
    """Checkbox names for one ``--model-type``. Unknown types keep the full list."""
    if not model_type or not str(model_type).strip():
        return list(QUANTIZER_MAP_SUBMODULES)
    names = _QUANTIZER_MAP_BY_MODEL_TYPE.get(str(model_type).strip().lower())
    if names is None:
        return list(QUANTIZER_MAP_SUBMODULES)
    allowed = set(names)
    return [name for name in QUANTIZER_MAP_SUBMODULES if name in allowed]


def parse_quantizer_map_text(text: str) -> list[str]:
    selected = []
    for part in str(text).replace(',', ' ').split():
        name = part.strip()
        if name and name in QUANTIZER_MAP_SUBMODULES and name not in selected:
            selected.append(name)
    return selected


def format_quantizer_map_text(selected: collections.abc.Collection[str]) -> str:
    ordered = [name for name in QUANTIZER_MAP_SUBMODULES if name in selected]
    return ' '.join(ordered)


class _QuantizerMapSelect(tk.Toplevel):
    def __init__(self,
                 options: list[str],
                 initial: collections.abc.Collection[str],
                 apply: typing.Callable[[str], None],
                 master=None,
                 position: tuple[int, int] = None,
                 model_type: str | None = None):
        super().__init__(master)
        self.title('Quantizer map' + (f' ({model_type})' if model_type else ''))
        self.transient(master)
        self.resizable(False, False)
        self._apply = apply
        self._ok = False

        frame = ttk.Frame(self, padding=(10, 10))
        frame.grid(row=0, column=0, sticky='nsew')

        ttk.Label(
            frame,
            text='Select pipeline submodules to quantize with --quantizer:',
            wraplength=360,
        ).grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 8))

        self._vars: dict[str, tk.BooleanVar] = {}
        initial_set = set(initial)
        row = 1
        for name in options:
            var = tk.BooleanVar(value=name in initial_set)
            self._vars[name] = var
            ttk.Checkbutton(frame, text=name, variable=var).grid(
                row=row, column=0, sticky='w')
            row += 1

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=(10, 0), sticky='e')
        ttk.Button(button_frame, text='OK', command=self._ok_action).pack(side='left', padx=(0, 5))
        ttk.Button(button_frame, text='Cancel', command=self.destroy).pack(side='left')

        _appearance.install(self)
        _util.position_toplevel(master, self, position=position)
        self.grab_set()
        self.protocol('WM_DELETE_WINDOW', self.destroy)

    def _ok_action(self):
        selected = [name for name, var in self._vars.items() if var.get()]
        self._apply(format_quantizer_map_text(selected))
        self._ok = True
        self.destroy()


def open_quantizer_map_select(master,
                              options: list[str],
                              initial_text: str,
                              apply: typing.Callable[[str], None],
                              position: tuple[int, int] = None,
                              model_type: str | None = None) -> None:
    initial = parse_quantizer_map_text(initial_text)
    _QuantizerMapSelect(
        options=options,
        initial=initial,
        apply=apply,
        master=master,
        position=position,
        model_type=model_type,
    )
