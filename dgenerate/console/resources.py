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
import enum
import importlib.resources
import importlib.util
import io
import json
import os
import platform
import shutil
import subprocess
import tkinter
import tkinter as tk
import typing
import webbrowser

import PIL.Image
import PIL.ImageTk
import packaging.version
import requests
import toml


class ReleaseInfo:
    tag_name: str
    release_name: str
    release_url: str

    def __init__(self,
                 tag_name: str,
                 release_name: str,
                 release_url: str):
        self.tag_name = tag_name
        self.release_name = release_name
        self.release_url = release_url

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)


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


def get_quantizer_recipes():
    opts = []
    if importlib.util.find_spec('bitsandbytes') is not None:
        opts += [
            'bnb;bits=8',
            'bnb;bits=4;bits4-compute-dtype=float16',
            'bnb;bits=4;bits4-compute-dtype=bfloat16',
            'bnb;bits=4;bits4-compute-dtype=float32'
        ]

    if importlib.util.find_spec('torchao') is not None:
        opts += [
            'torchao;type=int8wo',
            'torchao;type=int4wo',
        ]
    return opts


def check_latest_release() -> ReleaseInfo | None:
    """
    Get the latest software release for this software.

    :return: :py:class:`ReleaseInfo`
    """

    url = f"https://api.github.com/repos/Teriks/dgenerate/releases/latest"

    headers = {
        "Accept": "application/vnd.github.v3+json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=2)
        response.raise_for_status()
        latest_release = response.json()

        tag_name = latest_release['tag_name']
        release_name = latest_release['name']
        release_url = latest_release['html_url']

        return ReleaseInfo(tag_name, release_name, release_url)

    except requests.exceptions.RequestException:
        return None


def set_window_icon(window: tkinter.Tk | tkinter.Toplevel):
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


def get_themes():
    for file in importlib.resources.files('dgenerate.console').joinpath('themes').iterdir():
        yield os.path.splitext(file.name)[0], toml.loads(file.read_text())


_LOADED_SCHEMAS = dict()


def get_schema(name):
    if name in _LOADED_SCHEMAS:
        return _LOADED_SCHEMAS.get(name)

    with importlib.resources.open_text('dgenerate.console.schemas', f'{name}.json') as file:
        schema = json.load(file)
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

            # No GPUs found, fallback to CPU
            return ['cpu']
        except:
            return ['cpu']


def get_karras_scheduler_prediction_types() -> list[str]:
    return ['epsilon', 'v_prediction']


def supported_torch_model_formats_open() -> list[str]:
    return ['safetensors', 'pt', 'pth', 'cpkt', 'bin']


def release_version():
    from dgenerate.console import __version__
    value = __version__
    if value[0] != 'v':
        return 'v' + value
    else:
        return value


_textbox_theme = dict()


def get_textbox_theme() -> dict[str, typing.Any]:
    """
    Get textbox configuration arguments that are applied
    To all text boxes which wish to appear the same as the
    editor / console main input textbox.

    Usage: text_widget.configure(**resources.get_textbox_theme())

    :return: config argument dict
    """
    return dict(_textbox_theme)


def set_textbox_theme(**kwargs):
    """
    Set textbox configuration arguments that are applied
    To all text boxes which wish to appear the same as the
    editor / console main input textbox.

    Usage: resources.set_textbox_theme(bg='white', fg='black')

    :param kwargs: tk.config arguments
    """
    global _textbox_theme
    _textbox_theme = dict(kwargs)


class VersionComparison(enum.Enum):
    V1_NEWER = 0
    V2_NEWER = 1
    SAME = 2


def _ver_extra(ver: packaging.version.Version) -> tuple[str, int] | None:
    if ver.is_postrelease:
        return 'post', ver.post
    if ver.is_devrelease:
        return 'dev', ver.dev
    if ver.is_prerelease:
        return ver.pre
    return None


def compare_versions(version1, version2) -> VersionComparison:
    try:
        version1 = packaging.version.Version(version1)
        version2 = packaging.version.Version(version2)
    except packaging.version.InvalidVersion as e:
        raise ValueError(str(e))

    if version1 > version2:
        return VersionComparison.V1_NEWER
    elif version1 < version2:
        return VersionComparison.V2_NEWER
    else:
        extra1 = _ver_extra(version1)
        extra2 = _ver_extra(version2)

        if extra1 != extra2:
            # fallback, probably not required
            if extra1 is not None and extra2 is None:
                return VersionComparison.V1_NEWER
            elif extra1 is None and extra2 is not None:
                return VersionComparison.V2_NEWER
            elif extra1[1] > extra2[1]:
                return VersionComparison.V1_NEWER
            elif extra1[1] < extra2[1]:
                return VersionComparison.V2_NEWER

        return VersionComparison.SAME


def add_help_menu_links(menu: tk.Menu):
    ver = release_version()

    menu.add_command(
        label=f'Homepage ({ver})',
        command=lambda:
        webbrowser.open(
            f'https://github.com/Teriks/dgenerate/tree/{ver}'))

    menu.add_separator()

    menu.add_command(
        label='Config Examples',
        command=lambda:
        webbrowser.open(
            f'https://github.com/Teriks/dgenerate/tree/{ver}/examples'))

    menu.add_command(
        label='Config Documentation',
        command=lambda:
        webbrowser.open(
            f'https://dgenerate.readthedocs.io/en/{ver}/readme.html#writing-and-running-configs'))

    menu.add_command(
        label='Project Documentation',
        command=lambda:
        webbrowser.open(
            f'https://dgenerate.readthedocs.io/en/{ver}/readme.html'))

    release_info = check_latest_release()
    cur_ver = release_version()

    if release_info is not None:
        try:
            compare_result = compare_versions(release_info.tag_name, cur_ver)
        except:
            # do not kill the UI just for this.
            compare_result = VersionComparison.SAME

        if compare_result == VersionComparison.V1_NEWER:
            menu.add_separator()

            menu.add_command(
                label=f'Newer Release Available! ({release_info.tag_name})',
                command=lambda:
                webbrowser.open(
                    release_info.release_url))
