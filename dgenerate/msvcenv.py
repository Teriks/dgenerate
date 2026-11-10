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

"""Load the MSVC environment when ``cl`` is installed but not on ``PATH``."""

import os
import platform
import shutil
import subprocess


def _vswhere_path() -> str | None:
    for root in (
            os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'),
            os.environ.get('ProgramFiles', r'C:\Program Files'),
    ):
        path = os.path.join(root, 'Microsoft Visual Studio', 'Installer', 'vswhere.exe')
        if os.path.isfile(path):
            return path
    return None


def _vcvars_script() -> str:
    machine = platform.machine().lower()
    if machine in {'amd64', 'x86_64'}:
        return 'vcvars64.bat'
    if machine in {'arm64', 'aarch64'}:
        return 'vcvarsarm64.bat'
    return 'vcvars32.bat'


def _vs_install_path() -> str | None:
    vswhere = _vswhere_path()
    if vswhere is None:
        return None
    completed = subprocess.run(
        [
            vswhere,
            '-latest',
            '-prerelease',
            '-products', '*',
            '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
            '-property', 'installationPath',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        return None
    install = completed.stdout.decode('utf-8', errors='replace').strip()
    return install or None


def _vcvars_environment(script: str) -> dict[str, str] | None:
    # A single string, not an argument list. list2cmdline escapes the quotes
    # around a path that contains spaces, and cmd then cannot find the script.
    completed = subprocess.run(
        f'cmd /u /c call "{script}" >nul && set',
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout:
        return None
    environment = {}
    text = completed.stdout.decode('utf-16le', errors='replace')
    for line in text.splitlines():
        if not line or '=' not in line:
            continue
        key, value = line.split('=', 1)
        if key and '\0' not in key and '\0' not in value:
            environment[key] = value
    return environment or None


def activate_msvc_environment() -> None:
    """Put ``cl`` and the Windows SDK on this process's environment.

    The Build Tools installer leaves ``cl`` off ``PATH``. Torch Inductor looks
    up that bare name, and a compile also needs ``INCLUDE`` and ``LIB``.
    ``vcvars`` sets those for this process only.
    """
    if os.name != 'nt':
        return
    if os.environ.get('CXX') or shutil.which('cl'):
        return
    try:
        install = _vs_install_path()
        if not install:
            return
        script = os.path.join(install, 'VC', 'Auxiliary', 'Build', _vcvars_script())
        if not os.path.isfile(script):
            return
        environment = _vcvars_environment(script)
        if environment:
            os.environ.update(environment)
    except Exception:
        return
