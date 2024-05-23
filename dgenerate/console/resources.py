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

import importlib.resources
import io
import platform
import subprocess
import tkinter
import tkinter as tk
import typing
import webbrowser

import PIL.Image
import PIL.ImageTk


def set_window_icon(window: typing.Union[tkinter.Tk, tkinter.Toplevel]):
    if platform.system() == 'Windows':
        with importlib.resources.path('dgenerate', 'icon.ico') as path:
            window.iconbitmap(default=path)
    else:
        # noinspection PyTypeChecker
        window.iconphoto(
            True,
            PIL.ImageTk.PhotoImage(
                PIL.Image.open(
                    io.BytesIO(importlib.resources.read_binary('dgenerate', 'icon.ico')))))


def get_karras_schedulers():
    return [
        "EulerDiscreteScheduler",
        "HeunDiscreteScheduler",
        "UniPCMultistepScheduler",
        "DDPMScheduler",
        "EulerDiscreteScheduler",
        "DDIMScheduler",
        "DEISMultistepScheduler",
        "LMSDiscreteScheduler",
        "DPMSolverMultistepScheduler",
        "EulerAncestralDiscreteScheduler",
        "DPMSolverSinglestepScheduler",
        "DPMSolverSDEScheduler",
        "KDPM2DiscreteScheduler",
        "PNDMScheduler",
        "KDPM2AncestralDiscreteScheduler",
        "LCMScheduler"
    ]


def get_torch_vae_types():
    return ["AutoencoderKL",
            "AsymmetricAutoencoderKL",
            "AutoencoderTiny",
            "ConsistencyDecoderVAE"]


def get_cuda_devices():
    try:
        extra_kwargs = dict()
        if platform.system() == 'Windows':
            extra_kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW}

        result = subprocess.run(['nvidia-smi',
                                 '--query-gpu=index',
                                 '--format=csv,noheader'],
                                stdin=None,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                **extra_kwargs)
        devices = result.stdout.decode().strip().split('\n')
        return ['cuda:' + device for device in devices]
    except FileNotFoundError:
        return ['cpu']


def get_karras_scheduler_prediction_types():
    return ['epsilon', 'v_prediction']


def supported_torch_model_formats_open():
    return ['safetensors', 'pt', 'pth', 'cpkt', 'bin']


def add_help_menu_links(menu: tk.Menu):
    import dgenerate
    ver = dgenerate.__version__

    menu.add_command(
        label='Homepage',
        command=lambda:
        webbrowser.open(
            f'https://github.com/Teriks/dgenerate/tree/v{ver}'))

    menu.add_separator()

    menu.add_command(
        label='Config Examples',
        command=lambda:
        webbrowser.open(
            f'https://github.com/Teriks/dgenerate/tree/v{ver}/examples'))

    menu.add_command(
        label='Config Documentation',
        command=lambda:
        webbrowser.open(
            f'https://dgenerate.readthedocs.io/en/v{ver}/readme.html#writing-and-running-configs'))

    menu.add_command(
        label='Project Documentation',
        command=lambda:
        webbrowser.open(
            f'https://dgenerate.readthedocs.io/en/v{ver}/readme.html'))
