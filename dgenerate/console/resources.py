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
import re
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


def _tool_output(command: list[str],
                 stderr: int = subprocess.DEVNULL) -> subprocess.CompletedProcess | None:
    """Run a device-query tool. ``None`` when it cannot be started or times out."""
    extra_kwargs = {}
    if platform.system() == 'Windows':
        # Missing on non-Windows interpreters. Tests patch the platform to
        # Windows and would otherwise fail while importing this flag.
        extra_kwargs['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    try:
        return subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr,
            timeout=10,
            check=False,
            **extra_kwargs)
    except (OSError, subprocess.SubprocessError):
        return None


def _output_text(output: str | bytes | None) -> str:
    if not output:
        return ''
    if isinstance(output, bytes):
        output = output.decode(errors='replace')
    return output


def _index_lines(text: str) -> list[str]:
    """Integer lines from a tool listing, with ``\\r\\n`` line endings removed."""
    return [line.strip() for line in text.splitlines() if line.strip().isdigit()]


def _visible_ordinals(physical: list[str], spec: str | None) -> list[str]:
    """
    Process-local device ordinals after a visibility mask.

    PyTorch renumbers the selected devices from zero in the order given.
    An empty mask or ``-1`` hides every device. Integer tokens must name a
    discovered device. Any other token, such as a GPU UUID, counts as one
    visible device. A subdevice token (``0.0``) is left uninterpreted and
    the discovered list is kept.
    """
    if spec is None:
        return physical
    spec = spec.strip()
    if spec in ('', '-1'):
        return []
    tokens = [token.strip() for token in spec.split(',') if token.strip()]
    if not tokens:
        return []
    if any('.' in token for token in tokens):
        return physical
    if any(not token.isdigit() for token in tokens):
        return [str(index) for index in range(len(tokens))]
    known = set(physical)
    visible = [token for token in tokens if token in known]
    return [str(index) for index in range(len(visible))]


def _cuda_names(physical: list[str], spec: str | None) -> list[str]:
    return [f'cuda:{ordinal}' for ordinal in _visible_ordinals(physical, spec)] + ['cpu']


def _nvidia_ordinals() -> list[str] | None:
    """Physical NVIDIA indexes, ``[]`` when the tool ran and saw none, or ``None`` if it is absent."""
    if shutil.which('nvidia-smi') is None:
        return None
    result = _tool_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'])
    if result is None or result.returncode != 0:
        return []
    return _index_lines(_output_text(result.stdout))


def _unique_in_order(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _rocm_ordinals_from_text(text: str) -> tuple[list[str], bool]:
    """
    GPU ordinals from installer-recognized AMD tool text.

    The second value is true when the ordinals came from ``hipinfo``.
    ``hipinfo`` asks the HIP runtime, so ``device#`` is already the
    process-local ordinal: ``HIP_VISIBLE_DEVICES`` has been applied, and
    ``CUDA_VISIBLE_DEVICES`` has been applied when the HIP variable is
    unset. Those numbers must not be masked again.

    ``rocminfo`` and ``rocm-smi`` list every GPU. ``rocminfo`` prints one
    ``Device Type: GPU`` line per GPU agent and does not print
    ``gcnArchName``; a ``gcnArchName`` line is still accepted. ``rocm-smi
    --showproductname`` repeats ``GPU[n]`` once per property, so those
    indexes are de-duplicated.
    """
    hip_devices = re.findall(r'(?im)^device#\s+(\d+)', text)
    if hip_devices:
        return _unique_in_order(hip_devices), True
    product_indexes = re.findall(r'GPU\[(\d+)\]', text)
    if product_indexes:
        return _unique_in_order(product_indexes), False
    named_agents = re.findall(r'gcnArchName:\s*gfx\d+', text, re.IGNORECASE)
    if named_agents:
        return [str(index) for index in range(len(named_agents))], False
    gpu_agents = re.findall(r'Device Type:\s+GPU\b', text, re.IGNORECASE)
    return [str(index) for index in range(len(gpu_agents))], False


def _rocm_commands() -> list[list[str]]:
    """Tools the network installer already uses to identify an AMD GPU."""
    commands = []
    for name in ('hipinfo', 'hipInfo', 'rocminfo'):
        if shutil.which(name) is not None:
            commands.append([name])
    if shutil.which('rocm-smi') is not None:
        commands.append(['rocm-smi', '--showproductname'])
    return commands


def _rocm_ordinals() -> tuple[list[str] | None, bool]:
    """
    ROCm GPU indexes, and whether those indexes are already process-local.

    The list is ``None`` when ``hipinfo``, ``rocminfo``, and ``rocm-smi``
    are all absent. An empty list means one of those tools ran and no GPU
    could be identified. A Windows display-adapter name can tell the
    installer that an AMD GPU is present, but it is not a torch device
    ordinal, so it is not listed here.
    """
    commands = _rocm_commands()
    if not commands:
        return None, False
    saw_output = False
    for command in commands:
        result = _tool_output(command, stderr=subprocess.PIPE)
        if result is None or result.returncode != 0:
            continue
        saw_output = True
        text = _output_text(result.stdout) + '\n' + _output_text(result.stderr)
        ordinals, hip_local = _rocm_ordinals_from_text(text)
        if ordinals:
            return ordinals, hip_local
    return ([], False) if saw_output else (None, False)


# Client integrated graphics sit at PCI 00:02.0. Discrete parts such as
# Arc use another bus, and xpu-smi's "Intel(R) Graphics [id]" name is used
# for those as well, so the name check stays specific to iGPU brands.
_XPU_INTEGRATED_BDF = re.compile(r'(?i)(?:^|:)00:02\.0$')
_XPU_INTEGRATED_NAME = re.compile(
    r'(?i)\b(?:UHD Graphics|Iris(?:\(R\))?\s+(?:Xe|Plus)|HD Graphics)\b')


def _xpu_is_integrated(device: dict) -> bool:
    bdf = device.get('pci_bdf_address')
    if isinstance(bdf, str) and _XPU_INTEGRATED_BDF.search(bdf.strip()):
        return True
    name = device.get('device_name')
    return isinstance(name, str) and _XPU_INTEGRATED_NAME.search(name) is not None


def _xpu_discovered() -> list[tuple[str, bool]] | None:
    """
    ``(device_id, integrated)`` pairs from ``xpu-smi discovery -j``.

    ``None`` when ``xpu-smi`` is absent. An empty list means the tool ran
    and no device id could be read.
    """
    if shutil.which('xpu-smi') is None:
        return None
    result = _tool_output(['xpu-smi', 'discovery', '-j'])
    if result is None or result.returncode != 0:
        return []
    try:
        data = json.loads(_output_text(result.stdout))
        device_list = data.get('device_list', [])
    except (json.JSONDecodeError, AttributeError, TypeError):
        return []
    discovered = []
    for device in device_list:
        if not isinstance(device, dict) or 'device_id' not in device:
            continue
        discovered.append((str(device['device_id']), _xpu_is_integrated(device)))
    return discovered


def _xpu_visible_ordinals(devices: list[tuple[str, bool]],
                          spec: str | None) -> list[str]:
    """
    Process-local ``xpu`` ordinals.

    Level Zero treats an unset mask, an empty mask, and ``default`` as no
    mask. ``-1`` selects nothing. Integer tokens are physical device ids,
    kept in the order written, with duplicates and unknown ids dropped.
    When any selected device is discrete, integrated GPUs are dropped, which
    is what ``torch.xpu`` counts. A subdevice token (``0.0``) is left
    uninterpreted and the discovered list is kept.
    """
    if spec is None or spec in ('', 'default'):
        selected = list(devices)
    else:
        if spec.strip() in ('', '-1'):
            return []
        tokens = [token.strip() for token in spec.split(',') if token.strip()]
        if not tokens:
            return []
        if any('.' in token for token in tokens):
            selected = list(devices)
        elif any(not token.isdigit() for token in tokens):
            return [str(index) for index in range(len(tokens))]
        else:
            known = {device_id: integrated for device_id, integrated in devices}
            selected = []
            seen = set()
            for token in tokens:
                if token not in known or token in seen:
                    continue
                selected.append((token, known[token]))
                seen.add(token)
    if any(not integrated for _, integrated in selected):
        selected = [item for item in selected if not item[1]]
    return [str(index) for index in range(len(selected))]


def _macos_has_mps() -> bool:
    """Apple Silicon has MPS. Intel Macs, including under Rosetta, do not."""
    if platform.machine().lower() in {'arm64', 'aarch64'}:
        return True
    result = _tool_output(['sysctl', '-n', 'hw.optional.arm64'])
    if result is None or result.returncode != 0:
        return False
    return _output_text(result.stdout).strip() == '1'


def get_torch_devices() -> list[str]:
    """
    Devices the console may offer for ``--device``.

    Names match what this process can pass to torch. ``CUDA_VISIBLE_DEVICES``
    and ``HIP_VISIBLE_DEVICES`` renumber ``cuda:`` ordinals from zero. The
    query uses vendor tools and does not import torch, so the console can
    open before torch is loaded.
    """
    try:
        if platform.system() == 'Darwin':
            if _macos_has_mps():
                return ['mps', 'cpu']
            return ['cpu']

        nvidia = _nvidia_ordinals()
        if nvidia is not None:
            return _cuda_names(nvidia, os.environ.get('CUDA_VISIBLE_DEVICES'))

        rocm, hip_local = _rocm_ordinals()
        if rocm is not None:
            if hip_local:
                return [f'cuda:{ordinal}' for ordinal in rocm] + ['cpu']
            spec = os.environ.get('HIP_VISIBLE_DEVICES')
            if spec is None:
                spec = os.environ.get('CUDA_VISIBLE_DEVICES')
            return _cuda_names(rocm, spec)

        xpu = _xpu_discovered()
        if xpu is not None:
            ordinals = _xpu_visible_ordinals(xpu, os.environ.get('ZE_AFFINITY_MASK'))
            return [f'xpu:{ordinal}' for ordinal in ordinals] + ['cpu']

        return ['cpu']
    except Exception:
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

    # The installer download runs on the UI thread, and the window is not
    # destroyed until it finishes. Stop the shell first, including anything
    # it spawned, so a live download cannot hold the window open.
    # restart=True only kills now. If the download fails and this call
    # returns to the main loop, the shell is started again. A successful
    # install destroys the window before that can happen.
    console.kill_shell_process(restart=True)

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
