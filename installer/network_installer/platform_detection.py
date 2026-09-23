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

"""
Platform detection utilities for the network installer.
"""

import platform
import re
import subprocess
import sys
from dataclasses import dataclass

try:
    from network_installer.subprocess_utils import run_silent
except ImportError:
    from subprocess_utils import run_silent
from packaging import version as pkg_version


def _get_system_python_version() -> str:
    """
    Get the system Python version by running python --version.

    :return: Python version string in format "major.minor" or "unknown" if not found
    """
    # Try different Python commands
    for cmd in ['python3', 'python', 'py']:
        try:
            result = run_silent([cmd, '--version'],
                                capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Parse version from output like "Python 3.11.5"
                version_match = re.search(r'Python\s+(\d+\.\d+)', result.stdout)
                if version_match:
                    return version_match.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    return "unknown"


@dataclass
class PlatformInfo:
    """Platform information and capabilities."""
    system: str
    architecture: str
    machine: str
    python_version: str
    python_bits: str


@dataclass
class GPUInfo:
    """GPU information and capabilities."""
    has_nvidia: bool = False
    has_amd: bool = False
    has_intel: bool = False
    gpu_name: str | None = None
    cuda_version: str | None = None
    rocm_version: str | None = None
    xpu_version: str | None = None
    nvidia_compute_cap: float | None = None
    nvidia_is_mpv_legacy: bool = False


def get_platform_info() -> PlatformInfo:
    """
    Get comprehensive platform information.
    
    :return: PlatformInfo object containing platform information
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Normalize architecture names
    if machine in ('x86_64', 'amd64'):
        arch = 'x64'
    elif machine in ('i386', 'i686'):
        arch = 'x86'
    elif machine.startswith('arm'):
        arch = 'arm64' if '64' in machine else 'arm'
    else:
        arch = machine

    return PlatformInfo(
        system=system,
        architecture=arch,
        machine=machine,
        python_version=_get_system_python_version(),
        python_bits='64' if sys.maxsize > 2 ** 32 else '32'
    )


def _parse_nvidia_smi_cuda_version(text: str) -> str | None:
    """Read the CUDA version from ``nvidia-smi`` output.

    Current drivers print ``CUDA UMD Version:``; older ones print ``CUDA Version:``.
    """
    match = re.search(r'CUDA(?:\s+UMD)?\s+Version\s*:\s*(\d+\.\d+)', text, re.IGNORECASE)
    return match.group(1) if match else None


def _query_nvidia_cuda_version() -> str | None:
    """Return the installed CUDA UMD version.

    ``nvidia-smi --version`` avoids the process table. The full table is the
    fallback for drivers that only print the version there.
    """
    for cmd in (['nvidia-smi', '--version'], ['nvidia-smi']):
        try:
            result = run_silent(cmd, capture_output=True, text=True, timeout=20)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue
        if result.returncode != 0:
            continue
        version = _parse_nvidia_smi_cuda_version((result.stdout or '') + (result.stderr or ''))
        if version:
            return version
    return None


def detect_gpu() -> GPUInfo:
    """
    Detect GPU information and capabilities.
    
    :return: GPUInfo object containing detected GPU information
    """
    gpu_info = GPUInfo()

    system = platform.system().lower()

    if system == 'windows':
        try:
            # Try to get GPU info using nvidia-smi
            result = run_silent(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                gpu_info.has_nvidia = True
                gpu_info.gpu_name = result.stdout.strip()

                gpu_info.cuda_version = _query_nvidia_cuda_version()
                # Try to get NVIDIA compute capability (e.g., 5.2, 6.1, 7.0)
                try:
                    cc_result = run_silent(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader,nounits'],
                                           capture_output=True, text=True, timeout=10)
                    if cc_result.returncode == 0 and cc_result.stdout.strip():
                        try:
                            compute_cap = float(cc_result.stdout.strip().split('\n')[0])
                            gpu_info.nvidia_compute_cap = compute_cap
                            # Maxwell (5.x), Pascal (6.x), Volta (7.0) are legacy for CUDA 12.8/12.9 wheels
                            major = int(compute_cap)
                            minor = int(round((compute_cap - major) * 10))
                            gpu_info.nvidia_is_mpv_legacy = (major == 5) or (major == 6) or (major == 7 and minor == 0)
                        except ValueError:
                            pass
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Check for Intel XPU on Windows
        try:
            xpu_result = run_silent(['xpu-smi', '--version'], capture_output=True, text=True, timeout=10)
            if xpu_result.returncode == 0:
                gpu_info.has_intel = True
                xpu_match = re.search(r'(\d+\.\d+\.\d+)', xpu_result.stdout)
                if xpu_match:
                    gpu_info.xpu_version = xpu_match.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Check for AMD GPU on Windows. rocm-smi is not installed by the
        # Windows HIP SDK or AMD's torch wheels, so prefer the display adapter
        # list and hipinfo, then fall back to rocm-smi if it happens to exist.
        amd_name, amd_rocm = _detect_windows_amd()
        if amd_name:
            gpu_info.has_amd = True
            if not gpu_info.gpu_name:
                gpu_info.gpu_name = amd_name
            gpu_info.rocm_version = amd_rocm

    elif system == 'linux':
        try:
            # Check for NVIDIA GPU
            result = run_silent(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                gpu_info.has_nvidia = True
                gpu_info.gpu_name = result.stdout.strip()

                gpu_info.cuda_version = _query_nvidia_cuda_version()
                # Get NVIDIA compute capability
                try:
                    cc_result = run_silent(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader,nounits'],
                                           capture_output=True, text=True, timeout=10)
                    if cc_result.returncode == 0 and cc_result.stdout.strip():
                        try:
                            compute_cap = float(cc_result.stdout.strip().split('\n')[0])
                            gpu_info.nvidia_compute_cap = compute_cap
                            major = int(compute_cap)
                            minor = int(round((compute_cap - major) * 10))
                            gpu_info.nvidia_is_mpv_legacy = (major == 5) or (major == 6) or (major == 7 and minor == 0)
                        except ValueError:
                            pass
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass

            # Check for AMD GPU (ROCm)
            try:
                rocm_result = run_silent(['rocm-smi', '--version'], capture_output=True, text=True, timeout=10)
                if rocm_result.returncode == 0:
                    gpu_info.has_amd = True
                    rocm_match = re.search(r'ROCm\s+(\d+\.\d+\.\d+)', rocm_result.stdout)
                    if rocm_match:
                        gpu_info.rocm_version = rocm_match.group(1)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

            # Check for Intel XPU
            try:
                xpu_result = run_silent(['xpu-smi', '--version'], capture_output=True, text=True, timeout=10)
                if xpu_result.returncode == 0:
                    gpu_info.has_intel = True
                    xpu_match = re.search(r'(\d+\.\d+\.\d+)', xpu_result.stdout)
                    if xpu_match:
                        gpu_info.xpu_version = xpu_match.group(1)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    elif system == 'darwin':
        # macOS - check for Metal support
        try:
            result = run_silent(['system_profiler', 'SPDisplaysDataType'],
                                capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                if 'Metal' in result.stdout:
                    # Note: We don't have has_metal in GPUInfo, but we can set gpu_name
                    # Extract GPU name
                    gpu_match = re.search(r'Chipset Model:\s*(.+)', result.stdout)
                    if gpu_match:
                        gpu_info.gpu_name = gpu_match.group(1).strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return gpu_info


def detect_opengl_support() -> bool:
    """
    Detect if the system supports OpenGL.
    """
    system = platform.system().lower()

    if system == 'linux':
        try:
            # Check for OpenGL libraries
            result = run_silent(['ldconfig', '-p'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Look for OpenGL libraries
                if 'libGL.so' in result.stdout or 'libOpenGL.so' in result.stdout:
                    return True

            # Alternative check using pkg-config
            try:
                result = run_silent(['pkg-config', '--exists', 'gl'], capture_output=True, timeout=10)
                if result.returncode == 0:
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # If we can't detect, assume it's NOT available to be safe
        return False

    elif system == 'windows':
        try:
            # On Windows, OpenGL is typically available
            # We could check for specific DLLs, but it's usually present
            return True
        except:
            return True

    elif system == 'darwin':
        # macOS has OpenGL support built-in
        return True

    return False


AMD_WINDOWS_MULTIARCH_INDEX = "https://repo.amd.com/rocm/whl-multi-arch/"

# CUDA variants published per torch version, highest first.
# Each entry is (min_cuda_major, min_cuda_minor, index_suffix).
# Patch-specific keys are (major, minor, patch); otherwise (major, minor).
_CUDA_INDEX_TABLE: dict[tuple, list[tuple[int, int, str]]] = {
    (2, 14): [(13, 2, "cu132"), (13, 0, "cu130"), (12, 6, "cu126")],
    (2, 13): [(13, 2, "cu132"), (13, 0, "cu130"), (12, 9, "cu129"), (12, 6, "cu126")],
    (2, 12, 1): [(13, 2, "cu132"), (13, 0, "cu130"), (12, 9, "cu129"), (12, 6, "cu126")],
    (2, 12): [(13, 2, "cu132"), (13, 0, "cu130"), (12, 6, "cu126")],
    (2, 11): [(13, 0, "cu130"), (12, 9, "cu129"), (12, 8, "cu128"), (12, 6, "cu126")],
    (2, 10): [(13, 0, "cu130"), (12, 9, "cu129"), (12, 8, "cu128"), (12, 6, "cu126")],
    (2, 9): [(13, 0, "cu130"), (12, 9, "cu129"), (12, 8, "cu128"), (12, 6, "cu126")],
    (2, 8): [(12, 9, "cu129"), (12, 8, "cu128"), (12, 6, "cu126")],
    (2, 7): [(12, 8, "cu128"), (12, 6, "cu126"), (11, 8, "cu118")],
    (2, 6): [(12, 6, "cu126"), (12, 4, "cu124"), (11, 8, "cu118")],
    (2, 5): [(12, 4, "cu124"), (12, 1, "cu121"), (11, 8, "cu118")],
    (2, 4): [(12, 4, "cu124"), (12, 1, "cu121"), (11, 8, "cu118")],
    (2, 3): [(12, 1, "cu121"), (11, 8, "cu118")],
    (2, 2): [(12, 1, "cu121"), (11, 8, "cu118")],
    (2, 1): [(12, 1, "cu121"), (11, 8, "cu118")],
    (2, 0): [(11, 8, "cu118")],
}

# ROCm variants published per torch version, highest first.
_ROCM_INDEX_TABLE: dict[tuple, list[tuple[int, int, str]]] = {
    (2, 14): [(7, 14, "rocm7.14"), (7, 2, "rocm7.2")],
    (2, 13): [(7, 2, "rocm7.2"), (7, 1, "rocm7.1")],
    (2, 12): [(7, 2, "rocm7.2"), (7, 1, "rocm7.1")],
    (2, 11): [(7, 2, "rocm7.2"), (7, 1, "rocm7.1")],
    (2, 10): [(7, 1, "rocm7.1"), (7, 0, "rocm7.0")],
    (2, 9): [(6, 4, "rocm6.4"), (6, 3, "rocm6.3")],
    (2, 8): [(6, 4, "rocm6.4"), (6, 3, "rocm6.3")],
    (2, 7): [(6, 3, "rocm6.3")],
}


# xFormers 0.0.35 is built for torch 2.10. Newer torch wheels do not load it.
# torchvision 0.25.0 is the build published with torch 2.10.0.
XFORMERS_MAX_TORCH = (2, 10)
XFORMERS_TORCH_VERSION = "2.10.0"
XFORMERS_TORCHVISION_VERSION = "0.25.0"
XFORMERS_TORCHAUDIO_VERSION = "2.10.0"


def _torch_release(torch_version: str | None) -> tuple[int, ...] | None:
    if not torch_version:
        return None
    spec = torch_version.strip()
    for prefix in ('==', '>=', '~=', '^'):
        if spec.startswith(prefix):
            spec = spec[len(prefix):]
            break
    spec = spec.split(',')[0].split(';')[0].strip()
    if not spec:
        return None
    try:
        return pkg_version.parse(spec).release
    except Exception:
        return None


def cap_torch_version_for_xformers(torch_version: str | None) -> str | None:
    """
    Torch version to install when the xformers extra is selected.

    Pins newer than 2.10 are capped at 2.10.0. Older pins are left unchanged,
    because those releases ship an xformers build for that torch.
    """
    release = _torch_release(torch_version)
    if release is None:
        return None
    major = release[0]
    minor = release[1] if len(release) > 1 else 0
    if (major, minor) > XFORMERS_MAX_TORCH:
        return XFORMERS_TORCH_VERSION
    spec = torch_version.strip()
    for prefix in ('==', '>=', '~=', '^'):
        if spec.startswith(prefix):
            spec = spec[len(prefix):]
            break
    return spec.split(',')[0].split(';')[0].strip() or None


def xformers_version_overrides(
        torch_version: str | None,
        torchvision_version: str | None = None,
        torchaudio_version: str | None = None,
) -> dict[str, str] | None:
    """
    Exact pins to pass to ``uv pip install --overrides`` when xformers is selected.

    ``None`` means the release's torch pin is already 2.10 or older.
    """
    capped = cap_torch_version_for_xformers(torch_version)
    if not capped or capped == _bare_version(torch_version):
        return None
    overrides = {'torch': capped}
    if torchvision_version is None or _version_newer_than(torchvision_version, XFORMERS_TORCHVISION_VERSION):
        overrides['torchvision'] = XFORMERS_TORCHVISION_VERSION
    if torchaudio_version and _version_newer_than(torchaudio_version, XFORMERS_TORCHAUDIO_VERSION):
        overrides['torchaudio'] = XFORMERS_TORCHAUDIO_VERSION
    return overrides


def _bare_version(spec: str | None) -> str | None:
    if not spec:
        return None
    text = spec.strip()
    for prefix in ('==', '>=', '~=', '^'):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text.split(',')[0].split(';')[0].strip() or None


def _version_newer_than(left: str | None, right: str) -> bool:
    left_release = _torch_release(left)
    right_release = _torch_release(right)
    if left_release is None or right_release is None:
        return False
    return left_release > right_release


def is_amd_windows_multiarch_index(url: str | None) -> bool:
    """Return True if url is AMD's Windows multi-arch torch index."""
    return bool(url) and "repo.amd.com/rocm/whl-multi-arch" in url


def _windows_display_adapter_names() -> list[str]:
    """Read display adapter names from the Windows registry."""
    names: list[str] = []
    try:
        import winreg
    except ImportError:
        return names

    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as class_key:
            i = 0
            while True:
                try:
                    sub = winreg.EnumKey(class_key, i)
                except OSError:
                    break
                i += 1
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(class_key, sub) as adapter:
                        name, _ = winreg.QueryValueEx(adapter, "DriverDesc")
                        if name:
                            names.append(str(name))
                except OSError:
                    continue
    except OSError:
        pass
    return names


def _parse_rocm_version_text(text: str) -> str | None:
    match = re.search(r'(?:ROCm|HIP(?:\s+version)?)\s*[:=]?\s*(\d+\.\d+(?:\.\d+)?)', text, re.IGNORECASE)
    return match.group(1) if match else None


def _detect_windows_amd() -> tuple[str | None, str | None]:
    """
    Detect an AMD GPU on Windows without relying on rocm-smi.

    :return: (adapter_name, rocm_or_hip_version) — either may be None.
    """
    name = None
    for adapter in _windows_display_adapter_names():
        lowered = adapter.lower()
        if any(token in lowered for token in ('radeon', 'amd instinct', 'amd radeon')) or (
            'amd' in lowered and 'radeon' in lowered
        ):
            name = adapter
            break
        if re.search(r'\bAMD\b', adapter) and not any(
            skip in lowered for skip in ('processor', 'chipset', 'audio', 'capture')
        ):
            name = adapter
            break

    version = None
    for cmd in (['hipinfo'], ['hipInfo'], ['rocm-smi', '--version']):
        try:
            result = run_silent(cmd, capture_output=True, text=True, timeout=10)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue
        if result.returncode == 0:
            parsed = _parse_rocm_version_text(result.stdout + '\n' + result.stderr)
            if parsed:
                version = parsed
                break
            if cmd[0].startswith('rocm') or cmd[0].lower().startswith('hip'):
                # The tool exists; treat that as AMD even if the version line is unfamiliar
                if name is None:
                    name = name or 'AMD GPU'
                break

    return name, version


def _lookup_version_table(table: dict[tuple, list], major: int, minor: int, patch: int | None = None):
    if patch is not None and (major, minor, patch) in table:
        return table[(major, minor, patch)]
    if (major, minor) in table:
        return table[(major, minor)]
    newest_key = max(
        (k for k in table if len(k) == 2 and k[0] == major),
        default=None
    )
    if newest_key and (major, minor) > newest_key:
        return table[newest_key]
    return None


def _select_index_suffix(
        installed_major: int,
        installed_minor: int,
        variants: list[tuple[int, int, str]],
        prefer_suffix: str | None = None,
) -> str | None:
    """
    Return the newest published suffix this toolkit can run.

    ``None`` means every published build needs a newer CUDA or ROCm than
    the one installed. Callers use the CPU index in that case. A newer
    wheel does not load on an older toolkit.
    """
    if prefer_suffix:
        for _maj, _min, suffix in variants:
            if suffix == prefer_suffix:
                return suffix
    for maj, mino, suffix in variants:
        if (installed_major, installed_minor) >= (maj, mino):
            return suffix
    return None


def get_torch_index_url(torch_version: str | None = None) -> str | None:
    """
    Get the appropriate PyTorch index URL based on platform and GPU detection.
    For dgenerate 5.0.0+, torch_version is clean (e.g., "2.7.1") without CUDA prefixes.

    :param torch_version: The torch version required (e.g., "2.7.1"). Used for compatibility checking.
    :return: The appropriate PyTorch index URL, or None if no special index is needed.
    """
    platform_info = get_platform_info()
    gpu_info = detect_gpu()

    system = platform_info.system

    # Parse torch version for compatibility checking (no CUDA suffix extraction needed)
    torch_major = None
    torch_minor = None
    torch_patch = None

    if torch_version:
        try:
            parsed_version = pkg_version.parse(torch_version)
            torch_major = parsed_version.release[0]
            torch_minor = parsed_version.release[1] if len(parsed_version.release) > 1 else 0
            torch_patch = parsed_version.release[2] if len(parsed_version.release) > 2 else 0
        except Exception:
            # If we can't parse the version, fall back to default behavior
            pass

    if system == 'windows':
        if gpu_info.has_nvidia and gpu_info.cuda_version:
            # Use system CUDA version detection
            cuda_version = gpu_info.cuda_version
            cuda_major = int(cuda_version.split('.')[0])
            cuda_minor = int(cuda_version.split('.')[1]) if len(cuda_version.split('.')) > 1 else 0

            # Map based on PyTorch version and CUDA version
            return _get_torch_cuda_url(torch_major, torch_minor, torch_patch, cuda_major, cuda_minor,
                                       gpu_info.nvidia_is_mpv_legacy)
        elif gpu_info.has_amd:
            # Official pytorch.org ROCm wheels are Linux-only. Windows AMD
            # installs torch[device-all] from AMD's multi-arch index (#86).
            return AMD_WINDOWS_MULTIARCH_INDEX
        elif gpu_info.has_intel and gpu_info.xpu_version:
            xpu_url = _get_torch_xpu_url(torch_major, torch_minor, torch_patch)
            if xpu_url:
                return xpu_url
            # If XPU is not supported for this torch version, fall back to CPU
            return "https://download.pytorch.org/whl/cpu"
        else:
            return "https://download.pytorch.org/whl/cpu"

    elif system == 'linux':
        if gpu_info.has_nvidia and gpu_info.cuda_version:
            # Use system CUDA version detection
            cuda_version = gpu_info.cuda_version
            cuda_major = int(cuda_version.split('.')[0])
            cuda_minor = int(cuda_version.split('.')[1]) if len(cuda_version.split('.')) > 1 else 0

            # Map based on PyTorch version and CUDA version
            return _get_torch_cuda_url(torch_major, torch_minor, torch_patch, cuda_major, cuda_minor,
                                       gpu_info.nvidia_is_mpv_legacy)
        elif gpu_info.has_amd and gpu_info.rocm_version:
            # Use system ROCm version detection
            return _get_torch_rocm_url(torch_major, torch_minor, torch_patch, gpu_info.rocm_version)
        elif gpu_info.has_intel and gpu_info.xpu_version:
            xpu_url = _get_torch_xpu_url(torch_major, torch_minor, torch_patch)
            if xpu_url:
                return xpu_url
            # If XPU is not supported for this torch version, fall back to CPU
            return "https://download.pytorch.org/whl/cpu"
        else:
            return "https://download.pytorch.org/whl/cpu"

    elif system == 'darwin':
        # macOS doesn't need --index-url according to manual
        return None

    return None


def _get_torch_cuda_url(torch_major: int | None, torch_minor: int | None, torch_patch: int | None,
                        cuda_major: int, cuda_minor: int, nvidia_is_mpv_legacy: bool = False) -> str:
    """
    Get the appropriate PyTorch CUDA URL based on torch and CUDA versions.

    :param torch_major: Major version of torch (e.g., 2 for torch 2.x.x)
    :param torch_minor: Minor version of torch (e.g., 7 for torch 2.7.x)
    :param torch_patch: Patch version of torch (e.g., 1 for torch 2.7.1)
    :param cuda_major: Major version of CUDA (e.g., 12 for CUDA 12.x)
    :param cuda_minor: Minor version of CUDA (e.g., 8 for CUDA 12.8)
    :return: The appropriate PyTorch CUDA index URL.
    """
    variants = None
    if torch_major is not None and torch_minor is not None:
        variants = _lookup_version_table(_CUDA_INDEX_TABLE, torch_major, torch_minor, torch_patch)

    if not variants:
        if cuda_major >= 13:
            return "https://download.pytorch.org/whl/cu130"
        if cuda_major >= 12:
            return "https://download.pytorch.org/whl/cu128"
        return "https://download.pytorch.org/whl/cu118"

    prefer = "cu126" if nvidia_is_mpv_legacy else None
    suffix = _select_index_suffix(cuda_major, cuda_minor, variants, prefer_suffix=prefer)
    if suffix is None:
        return "https://download.pytorch.org/whl/cpu"
    return f"https://download.pytorch.org/whl/{suffix}"


def _get_torch_rocm_url(torch_major: int | None, torch_minor: int | None, torch_patch: int | None,
                        rocm_version: str) -> str:
    """
    Get the appropriate PyTorch ROCm URL based on torch and ROCm versions.

    :param torch_major: Major version of torch (e.g., 2 for torch 2.x.x)
    :param torch_minor: Minor version of torch (e.g., 7 for torch 2.7.x)
    :param torch_patch: Patch version of torch (e.g., 1 for torch 2.7.1)
    :param rocm_version: ROCm version string (e.g., "6.3", "5.7")
    :return: The appropriate PyTorch ROCm index URL.
    """
    try:
        rocm_parts = rocm_version.split('.')
        rocm_major = int(rocm_parts[0])
        rocm_minor = int(rocm_parts[1]) if len(rocm_parts) > 1 else 0
    except (ValueError, IndexError):
        rocm_major, rocm_minor = 0, 0

    variants = None
    if torch_major is not None and torch_minor is not None:
        variants = _lookup_version_table(_ROCM_INDEX_TABLE, torch_major, torch_minor, torch_patch)

    if variants:
        suffix = _select_index_suffix(rocm_major, rocm_minor, variants)
        if suffix is None:
            return "https://download.pytorch.org/whl/cpu"
        return f"https://download.pytorch.org/whl/{suffix}"

    # Older torch versions that are no longer in the table keep their last known mapping
    if torch_major == 2 and torch_minor == 6:
        if rocm_major == 6 and rocm_minor >= 2:
            return "https://download.pytorch.org/whl/rocm6.2.4" if rocm_minor > 2 or (
                len(rocm_version.split('.')) > 2 and int(rocm_version.split('.')[2]) >= 4
            ) else "https://download.pytorch.org/whl/rocm6.2"
        return "https://download.pytorch.org/whl/rocm6.1"
    if torch_major == 2 and torch_minor == 5:
        return "https://download.pytorch.org/whl/rocm6.2" if rocm_major == 6 and rocm_minor >= 2 \
            else "https://download.pytorch.org/whl/rocm6.1"
    if torch_major == 2 and torch_minor == 4:
        return "https://download.pytorch.org/whl/rocm6.1"
    if torch_major == 2 and torch_minor == 3:
        return "https://download.pytorch.org/whl/rocm6.0"
    if torch_major == 2 and torch_minor == 2:
        return "https://download.pytorch.org/whl/rocm5.7" if rocm_major >= 5 and rocm_minor >= 7 \
            else "https://download.pytorch.org/whl/rocm5.6"
    if torch_major == 2 and torch_minor == 1:
        return "https://download.pytorch.org/whl/rocm5.6"
    if torch_major == 2 and torch_minor == 0:
        return "https://download.pytorch.org/whl/rocm5.4.2"

    if rocm_major >= 7:
        return "https://download.pytorch.org/whl/rocm7.2"
    if rocm_major >= 6:
        return "https://download.pytorch.org/whl/rocm6.3"
    return "https://download.pytorch.org/whl/rocm5.7"


def _get_torch_xpu_url(torch_major: int | None, torch_minor: int | None, torch_patch: int | None) -> str | None:
    """
    Get the appropriate PyTorch XPU URL based on torch version.

    :param torch_major: Major version of torch (e.g., 2 for torch 2.x.x)
    :param torch_minor: Minor version of torch (e.g., 5 for torch 2.5.x)
    :param torch_patch: Patch version of torch (e.g., 1 for torch 2.5.1)
    :return: The appropriate PyTorch XPU URL, or None if XPU is not supported for this torch version.
    """
    # If we can't determine torch version, assume not supported
    if torch_major is None or torch_minor is None:
        return None

    # PyTorch 2.5+ supports XPU
    if torch_major == 2 and torch_minor >= 5:
        return "https://download.pytorch.org/whl/xpu"

    # PyTorch 2.4 and earlier do not have official XPU support
    # Users should use Intel Extension for PyTorch instead
    return None
