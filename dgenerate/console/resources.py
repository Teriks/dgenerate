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
import importlib.util
import io
import os
import platform
import shutil
import subprocess
import tempfile
import tkinter
import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.ttk as ttk
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

    recipes_dir = importlib.resources.files('dgenerate.console').joinpath('recipes')
    index_path = recipes_dir.joinpath('index.json')
    try:
        index = json.loads(index_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as e:
        raise RuntimeError(f'recipes/index.json is not valid JSON: {e}') from e
    if not isinstance(index, list) or not all(isinstance(name, str) for name in index):
        raise RuntimeError('recipes/index.json must be a list of recipe filenames.')

    duplicates = sorted({name for name in index if index.count(name) > 1})
    if duplicates:
        raise RuntimeError(
            'recipes/index.json lists the same file more than once: '
            + ', '.join(duplicates))

    on_disk = {
        entry.name for entry in recipes_dir.iterdir()
        if entry.is_file() and entry.name.endswith('.recipe')}
    indexed = set(index)
    missing = [name for name in index if name not in on_disk]
    unlisted = sorted(on_disk - indexed)
    if missing or unlisted:
        parts = []
        if missing:
            parts.append(
                'recipes/index.json names files that are not in recipes/: '
                + ', '.join(missing))
        if unlisted:
            parts.append(
                'recipe files missing from recipes/index.json: '
                + ', '.join(unlisted))
        raise RuntimeError('; '.join(parts) + '.')

    for name in index:
        text = recipes_dir.joinpath(name).read_text(encoding='utf-8')
        title, rest = text.split('\n', 1)
        if not title.startswith('title:'):
            raise RuntimeError(f'{name} must start with a title: line.')
        title = title.split(':', 1)[1].strip()
        if title in _RECIPES:
            raise RuntimeError(f'recipe title {title!r} is used by more than one file.')
        _RECIPES[title] = rest

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
    if importlib.util.find_spec('xllamacpp') is None:
        schema.pop('llama', None)
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
            "ConsistencyDecoderVAE",
            "AutoencoderKLFlux2",
            "AutoencoderKLQwenImage",
            "AutoencoderKLWan",
            "AutoencoderKLLTXVideo",
            "AutoencoderKLLTX2Video"]


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


def _prepare_installer(installer_path: str):
    if platform.system() == 'Darwin':
        subprocess.run(
            ['xattr', '-d', 'com.apple.quarantine', installer_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False)
    if platform.system() != 'Windows':
        os.chmod(installer_path, os.stat(installer_path).st_mode | 0o111)


def _choose_update_action(parent, message: str) -> str:
    """
    Ask whether to download the installer, open the release page, or cancel.

    :return: ``install``, ``view``, or ``cancel``.
    """
    choice = {'value': 'cancel'}
    dialog = tk.Toplevel(parent)
    dialog.title('Update')
    dialog.resizable(False, False)
    dialog.transient(parent)
    dialog.grab_set()

    frame = ttk.Frame(dialog, padding=16)
    frame.pack(fill=tk.BOTH, expand=True)
    ttk.Label(frame, text=message, justify=tk.LEFT, wraplength=440).pack(anchor=tk.W)

    buttons = ttk.Frame(frame)
    buttons.pack(anchor=tk.E, pady=(12, 0))

    def choose(value):
        choice['value'] = value
        dialog.destroy()

    ttk.Button(
        buttons, text='Download and Install',
        command=lambda: choose('install')).pack(side=tk.LEFT, padx=(0, 6))
    ttk.Button(
        buttons, text='View Release',
        command=lambda: choose('view')).pack(side=tk.LEFT, padx=(0, 6))
    ttk.Button(
        buttons, text='Cancel',
        command=lambda: choose('cancel')).pack(side=tk.LEFT)

    dialog.protocol('WM_DELETE_WINDOW', lambda: choose('cancel'))
    dialog.update_idletasks()
    parent.update_idletasks()
    x = parent.winfo_rootx() + max(0, (parent.winfo_width() - dialog.winfo_width()) // 2)
    y = parent.winfo_rooty() + max(0, (parent.winfo_height() - dialog.winfo_height()) // 2)
    dialog.geometry(f'+{x}+{y}')
    parent.wait_window(dialog)
    return choice['value']


def start_network_installer(console, tag: str, page_url: str, message: str,
                            installer_args: list[str] | None = None):
    """
    Offer a download of the platform installer for ``tag``, or the release page.

    Download and Install launches the installer GUI and closes the Console UI.
    A self-managed install opens ``page_url`` instead of this dialog.
    """
    if not _d_resources.get_release_info().network_install:
        webbrowser.open(page_url)
        return

    action = _choose_update_action(console, message)
    if action == 'view':
        webbrowser.open(page_url)
        return
    if action != 'install':
        return

    download = _d_resources.fetch_release_installer_download(tag)
    if download is None:
        messagebox.showerror(
            'Update',
            'Could not find an installer for this platform, '
            'or the release could not be reached.\n\n'
            f'See {page_url}')
        return

    asset_name, download_url = download
    install_dir = tempfile.mkdtemp(prefix='dgenerate_install_')
    installer_path = os.path.join(install_dir, asset_name)

    try:
        console.config(cursor='watch')
        console.update_idletasks()
        if not _d_resources.download_release_installer(download_url, installer_path):
            messagebox.showerror(
                'Update',
                f'Download failed. Try again later or install manually from {page_url}')
            return
    finally:
        try:
            console.config(cursor='')
        except tk.TclError:
            pass

    _prepare_installer(installer_path)

    popen_kwargs = {}
    if platform.system() == 'Windows':
        popen_kwargs['cwd'] = os.path.expanduser('~')

    if not installer_args:
        installer_args = []

    subprocess.Popen([installer_path, *installer_args], **popen_kwargs)
    console.destroy()


def add_help_menu_links(menu: tk.Menu, console=None):
    release_info = _d_resources.get_release_info()
    installed_version = release_info.version or _d_resources.version()

    if release_info.pre_release:
        git_tree = release_info.commit
        read_the_docs = release_info.branch
    else:
        git_tree = 'v' + installed_version.lstrip('vV')
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

    def add_newer_release_item(tag_name: str, page_url: str):
        menu.add_separator()
        if release_info.network_install and console is not None:
            menu.add_command(
                label=f'Newer Release Available! ({tag_name})',
                command=lambda tag=tag_name, url=page_url: start_network_installer(
                    console,
                    tag,
                    url,
                    f'A newer release is available ({tag}).\n\n'
                    'Download and Install closes the Console and opens '
                    f'the network installer with release {tag} selected.\n\n'
                    'View Release opens the release page.',
                    ['-v', tag]))
        else:
            menu.add_command(
                label=f'Newer Release Available! ({tag_name})',
                command=lambda url=page_url: webbrowser.open(url))

    if release_info.pre_release:
        superseded = None
        if release_info.branch:
            superseded = _d_resources.check_master_superseding_release(
                release_info.branch, installed_version)
        if superseded is not None:
            add_newer_release_item(superseded.tag_name, superseded.release_url)
            return

        if release_info.commit and release_info.branch:
            branch_update = _d_resources.check_pre_release_branch_update(
                release_info.commit, release_info.branch)
            if branch_update is not None and branch_update.ahead_by > 0:
                menu.add_separator()
                page_url = branch_update.pre_release_release_url
                if release_info.network_install and console is not None:
                    menu.add_command(
                        label='Pre-Release Update Available!',
                        command=lambda url=page_url, branch=release_info.branch:
                        start_network_installer(
                            console,
                            'pre-release',
                            url,
                            'A newer pre-release build is available.\n\n'
                            'Download and Install closes the Console and opens '
                            'the network installer with Development Branch and '
                            f'{branch} selected.\n\n'
                            'View Release opens the pre-release page.',
                            ['-b', branch]))
                else:
                    menu.add_command(
                        label='Pre-Release Update Available!',
                        command=lambda url=page_url: webbrowser.open(url))
        return

    latest_release = _d_resources.check_latest_release()

    if latest_release is not None:
        try:
            compare_result = _d_resources.compare_versions(
                latest_release.tag_name, installed_version)
        except Exception:
            # do not kill the UI just for this.
            compare_result = _d_resources.VersionComparison.SAME

        if compare_result == _d_resources.VersionComparison.V1_NEWER:
            add_newer_release_item(latest_release.tag_name, latest_release.release_url)
