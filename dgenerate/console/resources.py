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
import enum
import importlib.resources
import io
import json
import os
import platform
import re
import subprocess
import tkinter
import tkinter as tk
import webbrowser

import PIL.Image
import PIL.ImageTk
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
        response = requests.get(url, headers=headers)
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


def get_schema(name):
    with importlib.resources.open_text('dgenerate.console.schemas', f'{name}.json') as file:
        return json.load(file)


def get_karras_schedulers():
    return get_schema('karrasschedulers')['names']


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


def release_version():
    from dgenerate.console import __version__
    value = __version__
    if value[0] != 'v':
        return 'v' + value
    else:
        return value


class VersionComparison(enum.Enum):
    V1_NEWER = 0
    V2_NEWER = 1
    SAME = 2


def compare_versions(version1, version2) -> VersionComparison:
    def parse_version(version):
        # Remove any non-alphanumeric characters except dots
        version = re.sub(r'[^0-9a-zA-Z.]+', '', version)
        # Remove leading 'v' if present
        if version.startswith('v'):
            version = version[1:]
        # Remove any suffix
        if '-' in version:
            version = version.split('-')[0]
        major, minor, patch = version.split('.')
        return int(major), int(minor), int(patch)

    v1_major, v1_minor, v1_patch = parse_version(version1)
    v2_major, v2_minor, v2_patch = parse_version(version2)

    if (v1_major, v1_minor, v1_patch) > (v2_major, v2_minor, v2_patch):
        return VersionComparison.V1_NEWER
    elif (v1_major, v1_minor, v1_patch) < (v2_major, v2_minor, v2_patch):
        return VersionComparison.V2_NEWER
    else:
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

    if release_info is not None:

        if compare_versions(release_info.tag_name, release_version()) == VersionComparison.V1_NEWER:
            menu.add_separator()

            menu.add_command(
                label=f'Newer Release Available! ({release_info.tag_name})',
                command=lambda:
                webbrowser.open(
                    release_info.release_url))
