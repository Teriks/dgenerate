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

"""Install the xllamacpp wheel that matches this machine.

``pip install dgenerate[xllamacpp]`` always gets the PyPI build: CPU on Linux
and Windows, Metal on macOS. CUDA, ROCm, and Vulkan builds use the same
package name on separate indexes, so a setuptools extra cannot select them.

After the extra is installed, run this with the same interpreter::

    python installer/network_installer/xllamacppinstall.py

The network installer runs it after the extra is installed. It reinstalls the
already-installed version from the matching index. When no GPU build applies,
the PyPI wheel is left in place.
"""

import argparse
import importlib.metadata
import platform
import re
import sys

try:
    from network_installer.platform_detection import detect_gpu
    from network_installer.subprocess_utils import run_silent
except ImportError:
    from platform_detection import detect_gpu
    from subprocess_utils import run_silent

_INDEX_ROOT = 'https://xorbitsai.github.io/xllamacpp/whl'

XLLAMACPP_EXTRAS = frozenset({'xllamacpp'})


def xllamacpp_index_url(
        system: str,
        cuda: tuple[int, int] | None = None,
        rocm: tuple[int, int] | None = None,
        has_nvidia: bool = False,
        has_amd: bool = False,
        has_intel: bool = False,
        nvidia_legacy: bool = False,
) -> str | None:
    """
    Return the xllamacpp package index for this machine, or ``None``.

    ``None`` means the PyPI wheel is already the right build (CPU, or Metal
    on macOS).
    """
    if system == 'Darwin':
        return None

    if has_nvidia and system in {'Linux', 'Windows'}:
        # Maxwell, Pascal, Volta, and pre-12.8 drivers use Vulkan.
        # A detected NVIDIA GPU with no parsed driver version gets cu128,
        # which still runs on current CUDA 12 and 13 drivers.
        if nvidia_legacy or (cuda is not None and cuda < (12, 8)):
            return f'{_INDEX_ROOT}/vulkan'
        if cuda is not None and cuda >= (13, 2):
            return f'{_INDEX_ROOT}/cu132'
        return f'{_INDEX_ROOT}/cu128'

    if system == 'Linux' and rocm is not None:
        if rocm >= (7, 2):
            return f'{_INDEX_ROOT}/rocm-7.2.4'
        if rocm >= (6, 4):
            return f'{_INDEX_ROOT}/rocm-6.4.1'

    # ROCm wheels are Linux-only. Windows AMD, Intel, and Linux AMD without
    # a matching ROCm runtime use the Vulkan wheel.
    if system in {'Linux', 'Windows'} and (has_amd or has_intel):
        return f'{_INDEX_ROOT}/vulkan'

    return None


def _parse_version_pair(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    match = re.search(r'(\d+)\.(\d+)', text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def detect_xllamacpp_index_url(system: str | None = None) -> str | None:
    """Detect the xllamacpp index URL using the installer's GPU detection."""
    gpu = detect_gpu()
    system = system or platform.system()
    return xllamacpp_index_url(
        system,
        cuda=_parse_version_pair(gpu.cuda_version),
        rocm=_parse_version_pair(gpu.rocm_version),
        has_nvidia=gpu.has_nvidia,
        has_amd=gpu.has_amd,
        has_intel=gpu.has_intel,
        nvidia_legacy=gpu.nvidia_is_mpv_legacy,
    )


def installed_version(python: str | None = None) -> str | None:
    """Return the xllamacpp version installed for ``python``, or this interpreter."""
    if python is None:
        try:
            return importlib.metadata.version('xllamacpp')
        except importlib.metadata.PackageNotFoundError:
            return None
    result = run_silent(
        [python, '-c', 'import importlib.metadata; print(importlib.metadata.version("xllamacpp"))'],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    version = result.stdout.strip()
    return version or None


def install_xllamacpp_wheel(
        python: str | None = None,
        dry_run: bool = False,
        index_url: str | None = None,
) -> int:
    """
    Reinstall the installed xllamacpp version from the GPU index when one applies.

    ``python`` is the interpreter whose environment should be updated. The
    network installer passes the venv interpreter because this module runs
    in the installer process.

    :return: process exit code
    """
    version = installed_version(python)
    if version is None:
        print(
            'xllamacpp is not installed. Install the extra first:\n'
            '  pip install dgenerate[xllamacpp]\n'
            'Then run: python installer/network_installer/xllamacppinstall.py',
            file=sys.stderr,
        )
        return 1

    if index_url is None:
        index_url = detect_xllamacpp_index_url()
    if index_url is None:
        print(
            f'xllamacpp {version} from PyPI is the build for this machine '
            '(CPU, or Metal on macOS).'
        )
        return 0

    cmd = [
        python or sys.executable, '-m', 'pip', 'install',
        f'xllamacpp=={version}',
        '--force-reinstall',
        '--no-deps',
        '--index-url', index_url,
    ]
    print('Installing xllamacpp from', index_url)
    print(' ', ' '.join(cmd))
    if dry_run:
        return 0

    completed = run_silent(cmd, check=False)
    if completed.returncode != 0:
        print(
            'The GPU xllamacpp wheel could not be installed. '
            'The PyPI build is still installed.',
            file=sys.stderr,
        )
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Replace the PyPI xllamacpp wheel with the CUDA, ROCm, or Vulkan build when this machine can use one.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the pip command without running it.',
    )
    args = parser.parse_args(argv)
    return install_xllamacpp_wheel(dry_run=args.dry_run)


if __name__ == '__main__':
    sys.exit(main())
