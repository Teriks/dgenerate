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
import bisect
import importlib.resources
import importlib.util
import io
import os
import platform
import shutil
import subprocess
import tkinter
import tkinter as tk
import webbrowser
import json

import PIL.Image
import PIL.ImageTk
import toml
import dgenerate.resources as _d_resources


def get_file_dialog_args(file_types: list):
    dialog_args = dict()
    schema = get_schema('mediaformats')

    if file_types:
        type_desc = []
        dialog_args['filetypes'] = type_desc
        file_type_mappings = {
            'models': ('Models', supported_torch_model_formats_open),
            'config': ('Config', lambda: ['toml', 'yaml', 'yml', 'json']),
            'toml': ('Config', lambda: ['toml']),
            'yaml': ('Config', lambda: ['yaml', 'yml']),
            'json': ('Config', lambda: ['json']),
            'images-in': ('Images', lambda: schema['images-in']),
            'videos-in': ('Videos', lambda: schema['videos-in']),
            'images-out': ('Images', lambda: schema['images-out']),
            'videos-out': ('Videos', lambda: schema['videos-out']),
        }

        for file_type in file_types:
            if file_type in file_type_mappings:
                label, get_extensions = file_type_mappings[file_type]
                file_globs = ['*.' + ext for ext in get_extensions()]
                type_desc.append((label, ' '.join(file_globs)))

    return dialog_args


_RECIPES = dict()


def get_recipes():
    if _RECIPES:
        return _RECIPES

    recipes = []

    for file in importlib.resources.files('dgenerate.console').joinpath('recipes').iterdir():
        text = file.read_text()
        title, order, rest = text.split('\n', 2)
        title = title.split(':', 1)[1].strip()
        order = int(order.split(':', 1)[1].strip())
        bisect.insort(recipes, (order, title, rest), key=lambda x: x[0])

    for recipe in recipes:
        _RECIPES[recipe[1]] = recipe[2]

    return _RECIPES


def set_window_icon(window: tkinter.Tk | tkinter.Toplevel):
    if platform.system() == 'Windows':
        window.iconbitmap(default=_d_resources.get_icon_path())
    else:
        # noinspection PyTypeChecker
        window.iconphoto(
            True,
            PIL.ImageTk.PhotoImage(
                PIL.Image.open(
                    io.BytesIO(_d_resources.get_icon_data()))))


def get_themes():
    for file in importlib.resources.files('dgenerate.console').joinpath('themes').iterdir():
        yield os.path.splitext(file.name)[0], toml.loads(file.read_text())


_LOADED_SCHEMAS = dict()


def _schema_filter_quantizers(schema: dict):
    if importlib.util.find_spec('bitsandbytes') is None:
        schema.pop('bnb', None)
    return schema


def _schema_filter_imageprocessors(schema: dict):
    if importlib.util.find_spec('ncnn') is None:
        schema.pop('upscaler-ncnn', None)
    return schema


def _schema_filter_promptupscalers(schema: dict):
    if importlib.util.find_spec('gpt4all') is None:
        schema.pop('gpt4all', None)
    return schema


def get_schema(name):
    if name in _LOADED_SCHEMAS:
        return _LOADED_SCHEMAS.get(name)
    try:
        filter = globals()[f'_schema_filter_{name}']
    except KeyError:
        filter = lambda x: x
    with importlib.resources.open_text('dgenerate.console.schemas', f'{name}.json') as file:
        schema = filter(json.load(file))
        _LOADED_SCHEMAS[name] = schema
        return schema


def get_karras_schedulers() -> list[str]:
    return get_schema('karrasschedulers')['names']


def get_dgenerate_arguments() -> dict[str, str]:
    return get_schema('arguments')


def get_dgenerate_directives() -> dict[str, str]:
    return get_schema('directives')


def get_dgenerate_functions() -> dict[str, str]:
    return get_schema('functions')


def get_torch_vae_types() -> list[str]:
    return ["AutoencoderKL",
            "AsymmetricAutoencoderKL",
            "AutoencoderTiny",
            "ConsistencyDecoderVAE"]


def get_gpt4all_compute_devices() -> list[str]:
    """
    - "cpu": Model will run on the central processing unit.
    - "gpu": Use Metal on ARM64 macOS, otherwise the same as "kompute".
    - "kompute": Use the best GPU provided by the Kompute backend.
    - "cuda": Use the best GPU provided by the CUDA backend.
    - "amd", "nvidia": Use the best GPU provided by the Kompute backend from this vendor.
    """

    opts = ['cpu', 'gpu', 'kompute']

    if shutil.which('nvidia-smi'):
        opts.append('cuda')

    if shutil.which('rocm-smi'):
        opts.append('amd')

    return opts


def get_torch_devices() -> list[str]:
    if platform.system() == 'Darwin':
        # Assume MPS is available without importing torch
        return ['mps', 'cpu']
    else:
        try:
            extra_kwargs = dict()
            if platform.system() == 'Windows':
                extra_kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW}

            # Detect CUDA devices using nvidia-smi
            if shutil.which('nvidia-smi') is not None:
                result = subprocess.run(['nvidia-smi',
                                         '--query-gpu=index',
                                         '--format=csv,noheader'],
                                        stdin=None,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        **extra_kwargs)

                if result.returncode != 0:
                    return ['cpu']

                devices = result.stdout.decode().strip().split('\n')
                return ['cuda:' + device for device in devices] + ['cpu']

            # Detect ROCm devices using rocm-smi
            elif shutil.which('rocm-smi') is not None:
                result = subprocess.run(['rocm-smi', '-l'],
                                        stdin=None,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        **extra_kwargs)

                if result.returncode != 0:
                    return ['cpu']

                gpu_count = sum(1 for line in result.stdout.decode().split('\n')
                                if line.strip().startswith("GPU"))

                if gpu_count > 0:
                    devices = [str(i) for i in range(gpu_count)]
                    return ['cuda:' + device for device in devices] + ['cpu']

            # Detect XPU devices using xpu-smi
            elif shutil.which('xpu-smi') is not None:
                result = subprocess.run(['xpu-smi', 'discovery', '-j'],
                                        stdin=None,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        **extra_kwargs)

                if result.returncode != 0:
                    return ['cpu']

                # Parse xpu-smi JSON output to get device IDs
                try:
                    data = json.loads(result.stdout.decode())
                    device_list = data.get('device_list', [])
                    device_ids = [str(device['device_id']) for device in device_list 
                                  if 'device_id' in device]
                    
                    if device_ids:
                        return ['xpu:' + device_id for device_id in device_ids] + ['cpu']
                except (json.JSONDecodeError, KeyError, TypeError):
                    # Fallback to CPU if JSON parsing fails
                    return ['cpu']

            # No GPUs found, fallback to CPU
            return ['cpu']
        except:
            return ['cpu']


def get_karras_scheduler_prediction_types() -> list[str]:
    return ['epsilon', 'v_prediction']


def supported_torch_model_formats_open() -> list[str]:
    return ['safetensors', 'pt', 'pth', 'cpkt', 'bin']


def add_help_menu_links(menu: tk.Menu):
    release_info = _d_resources.get_release_info()

    if release_info.pre_release:
        git_tree = release_info.commit
        read_the_docs = release_info.branch
    else:
        git_tree = 'v' + _d_resources.version()
        read_the_docs = git_tree

    menu.add_command(
        label=f'Homepage (pre-release, commit: {git_tree})' if release_info.pre_release else f'Homepage ({git_tree})',
        command=lambda:
        webbrowser.open(
            f'https://github.com/Teriks/dgenerate/tree/{git_tree}'))

    menu.add_separator()

    menu.add_command(
        label='Config Examples',
        command=lambda:
        webbrowser.open(
            f'https://github.com/Teriks/dgenerate/tree/{git_tree}/examples'))

    menu.add_command(
        label='Config Documentation',
        command=lambda:
        webbrowser.open(
            f'https://dgenerate.readthedocs.io/en/{read_the_docs}/manual.html#writing-and-running-configs'))

    menu.add_command(
        label='Project Documentation',
        command=lambda:
        webbrowser.open(
            f'https://dgenerate.readthedocs.io/en/{read_the_docs}/manual.html'))

    if release_info.pre_release:
        return

    release_info = _d_resources.check_latest_release()

    if release_info is not None:
        try:
            compare_result = _d_resources.compare_versions(release_info.tag_name, git_tree)
        except:
            # do not kill the UI just for this.
            compare_result = _d_resources.VersionComparison.SAME

        if compare_result == _d_resources.VersionComparison.V1_NEWER:
            menu.add_separator()

            menu.add_command(
                label=f'Newer Release Available! ({release_info.tag_name})',
                command=lambda:
                webbrowser.open(
                    release_info.release_url))
