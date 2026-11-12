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

"""Install the MSVC C++ tools by running Microsoft's Build Tools installer.

dgenerate does not ship the compiler. ``vswhere`` detection stays in sync
with ``dgenerate/msvcenv.py``.
"""

import os
import ssl
import subprocess
import tempfile
import urllib.request

import certifi
from packaging.version import InvalidVersion, Version

from network_installer.subprocess_utils import run_silent


# Visual Studio 2026 stable channel. vs/17 is the 2022 product and stays on that line.
BUILD_TOOLS_URL = 'https://aka.ms/vs/stable/vs_BuildTools.exe'
VC_TOOLS_COMPONENT = 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64'
VC_TOOLS_WORKLOAD = 'Microsoft.VisualStudio.Workload.VCTools'
# 0 is success. 3010 is success with a reboot requested.
SUCCESS_EXIT_CODES = {0, 3010}
MSVC_TOOLCHAIN_MIN_VERSION = Version('6.0.0')


def msvc_option_checked(*, preset: bool, installed: bool, intel_xpu: bool) -> bool:
    """Whether the extras checkbox starts selected.

    Intel XPU selects it, and ``--msvc`` selects it on any Windows machine.
    An existing MSVC toolset stays unselected so the installer does not run again.
    """
    if installed:
        return False
    return preset or intel_xpu


def supports_msvc_toolchain(version: str | None) -> bool:
    """True when this dgenerate version can offer the Build Tools option.

    The compiler option starts at 6.0.0. Older releases install from wheels
    and do not need it for ``torch.compile``.
    """
    if not version:
        return False
    text = version.strip()
    if text[:1] in {'v', 'V'}:
        text = text[1:]
    try:
        return Version(text) >= MSVC_TOOLCHAIN_MIN_VERSION
    except InvalidVersion:
        return False


def vswhere_path() -> str | None:
    """Path to ``vswhere.exe``, when the Visual Studio installer is present."""
    for root in (
            os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'),
            os.environ.get('ProgramFiles', r'C:\Program Files'),
    ):
        path = os.path.join(root, 'Microsoft Visual Studio', 'Installer', 'vswhere.exe')
        if os.path.isfile(path):
            return path
    return None


def vc_tools_installed() -> bool:
    """True when a Visual Studio install already has the MSVC x64 toolset."""
    if os.name != 'nt':
        return False
    vswhere = vswhere_path()
    if vswhere is None:
        return False
    completed = run_silent(
        [
            vswhere,
            '-latest',
            '-prerelease',
            '-products', '*',
            '-requires', VC_TOOLS_COMPONENT,
            '-property', 'installationPath',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        return False
    install = completed.stdout.decode('utf-8', errors='replace').strip()
    return bool(install)


def bootstrapper_arguments() -> list[str]:
    """Arguments for the current ``vs_BuildTools.exe``.

    ``--quiet`` accepts Microsoft's license and draws no installer window.
    The bootstrapper's manifest still asks Windows for administrator approval,
    which is the consent dialog. ``--wait`` keeps this process until setup
    finishes so the exit code belongs to the install.
    """
    return [
        '--quiet',
        '--wait',
        '--norestart',
        '--nocache',
        '--add', VC_TOOLS_WORKLOAD,
        '--includeRecommended',
    ]


def _ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.load_verify_locations(certifi.where())
    return context


def _download_bootstrapper(destination: str, log) -> str | None:
    log(f'Downloading Visual Studio Build Tools from {BUILD_TOOLS_URL}')
    request = urllib.request.Request(
        BUILD_TOOLS_URL,
        headers={'User-Agent': 'dgenerate-network-installer/1.0'},
    )
    with urllib.request.urlopen(request, context=_ssl_context()) as response:
        with open(destination, 'wb') as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
    with open(destination, 'rb') as handle:
        if handle.read(2) != b'MZ':
            return 'The download was not the Visual Studio installer.'
    return None


def install_vc_tools(log) -> tuple[bool, str | None]:
    """Download and run the current Visual Studio Build Tools installer.

    Microsoft's window stays hidden. Windows still shows the administrator
    consent dialog because the bootstrapper requires elevation.

    :param log: Callable that receives progress lines.
    :return: Success, and an error string when installation did not finish.
    """
    if os.name != 'nt':
        return False, 'Visual Studio Build Tools can only be installed on Windows.'
    if vc_tools_installed():
        log('MSVC C++ tools are already installed.')
        return True, None

    log(
        'Approve the administrator prompt. Microsoft\'s installer then runs '
        'with no window. Continuing accepts the Visual Studio license. '
        'The C++ workload is several GB.'
    )
    bootstrapper = None
    try:
        fd, bootstrapper = tempfile.mkstemp(prefix='vs_BuildTools_', suffix='.exe')
        os.close(fd)
        error = _download_bootstrapper(bootstrapper, log)
        if error:
            return False, error
        command = [bootstrapper, *bootstrapper_arguments()]
        log('Running Visual Studio Build Tools. This can take a while.')
        completed = run_silent(command, check=False)
    except Exception as error:
        return False, str(error)
    finally:
        if bootstrapper:
            try:
                os.remove(bootstrapper)
            except OSError:
                pass

    if vc_tools_installed():
        if completed.returncode == 3010:
            log('Visual Studio C++ tools are installed. Reboot before using cl.')
        else:
            log('Visual Studio C++ tools are installed.')
        return True, None
    if completed.returncode in SUCCESS_EXIT_CODES:
        return False, 'The Build Tools installer finished, but the MSVC toolset was not found.'
    return False, f'Visual Studio Build Tools installer exited with code {completed.returncode}.'
