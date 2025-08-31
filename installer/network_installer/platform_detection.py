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
from network_installer.subprocess_utils import run_silent
from packaging import version as pkg_version
from typing import Optional


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
    gpu_name: Optional[str] = None
    cuda_version: Optional[str] = None
    rocm_version: Optional[str] = None
    xpu_version: Optional[str] = None
    nvidia_compute_cap: Optional[float] = None
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

                # Try to get CUDA version
                cuda_result = run_silent(['nvidia-smi'], capture_output=True, text=True, timeout=10)
                if cuda_result.returncode == 0:
                    cuda_match = re.search(r'CUDA Version:\s*(\d+\.\d+)', cuda_result.stdout)
                    if cuda_match:
                        gpu_info.cuda_version = cuda_match.group(1)
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

        # Check for AMD GPU (ROCm) on Windows
        try:
            rocm_result = run_silent(['rocm-smi', '--version'], capture_output=True, text=True, timeout=10)
            if rocm_result.returncode == 0:
                gpu_info.has_amd = True
                rocm_match = re.search(r'ROCm\s+(\d+\.\d+\.\d+)', rocm_result.stdout)
                if rocm_match:
                    gpu_info.rocm_version = rocm_match.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    elif system == 'linux':
        try:
            # Check for NVIDIA GPU
            result = run_silent(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                gpu_info.has_nvidia = True
                gpu_info.gpu_name = result.stdout.strip()

                # Get CUDA version
                cuda_result = run_silent(['nvidia-smi'], capture_output=True, text=True, timeout=10)
                if cuda_result.returncode == 0:
                    cuda_match = re.search(r'CUDA Version:\s*(\d+\.\d+)', cuda_result.stdout)
                    if cuda_match:
                        gpu_info.cuda_version = cuda_match.group(1)
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


def get_torch_index_url(torch_version: Optional[str] = None) -> Optional[str]:
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


def _get_torch_cuda_url(torch_major: Optional[int], torch_minor: Optional[int], torch_patch: Optional[int],
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
    # If we can't determine torch version, use conservative defaults
    if torch_major is None or torch_minor is None:
        if cuda_major >= 12:
            return "https://download.pytorch.org/whl/cu128"
        elif cuda_major >= 11:
            return "https://download.pytorch.org/whl/cu118"
        else:
            return "https://download.pytorch.org/whl/cu118"

    # PyTorch 2.8+ special handling based on NVIDIA architecture classes
    # Maxwell (5.x), Pascal (6.x), Volta (7.0) should use cu126 for 2.8/2.9 wheels
    # Newer architectures can use cu128/cu129 when available
    # See: https://github.com/pytorch/pytorch/issues/157517
    if torch_major == 2 and torch_minor >= 8:
        if nvidia_is_mpv_legacy:
            return "https://download.pytorch.org/whl/cu126"
        if cuda_major == 12:
            if cuda_minor >= 9:
                return "https://download.pytorch.org/whl/cu129"
            elif cuda_minor >= 8:
                return "https://download.pytorch.org/whl/cu128"
            elif cuda_minor >= 6:
                return "https://download.pytorch.org/whl/cu126"
            else:
                return "https://download.pytorch.org/whl/cu118"
        elif cuda_major >= 11:
            return "https://download.pytorch.org/whl/cu118"
        else:
            return "https://download.pytorch.org/whl/cu118"

    # PyTorch 2.7.x
    if torch_major == 2 and torch_minor == 7:
        if cuda_major == 12:
            if cuda_minor >= 8:
                return "https://download.pytorch.org/whl/cu128"
            elif cuda_minor >= 6:
                return "https://download.pytorch.org/whl/cu126"
            else:
                return "https://download.pytorch.org/whl/cu118"  # Fallback
        elif cuda_major >= 11:
            return "https://download.pytorch.org/whl/cu118"
        else:
            return "https://download.pytorch.org/whl/cu118"

    # PyTorch 2.6.x
    elif torch_major == 2 and torch_minor == 6:
        if cuda_major == 12:
            if cuda_minor >= 6:
                return "https://download.pytorch.org/whl/cu126"
            elif cuda_minor >= 4:
                return "https://download.pytorch.org/whl/cu124"
            else:
                return "https://download.pytorch.org/whl/cu118"  # Fallback
        elif cuda_major >= 11:
            return "https://download.pytorch.org/whl/cu118"
        else:
            return "https://download.pytorch.org/whl/cu118"

    # PyTorch 2.5.x
    elif torch_major == 2 and torch_minor == 5:
        if cuda_major == 12:
            if cuda_minor >= 4:
                return "https://download.pytorch.org/whl/cu124"
            elif cuda_minor >= 1:
                return "https://download.pytorch.org/whl/cu121"
            else:
                return "https://download.pytorch.org/whl/cu118"  # Fallback
        elif cuda_major >= 11:
            return "https://download.pytorch.org/whl/cu118"
        else:
            return "https://download.pytorch.org/whl/cu118"

    # PyTorch 2.4.x
    elif torch_major == 2 and torch_minor == 4:
        if cuda_major == 12:
            if cuda_minor >= 4:
                return "https://download.pytorch.org/whl/cu124"
            elif cuda_minor >= 1:
                return "https://download.pytorch.org/whl/cu121"
            else:
                return "https://download.pytorch.org/whl/cu118"  # Fallback
        elif cuda_major >= 11:
            return "https://download.pytorch.org/whl/cu118"
        else:
            return "https://download.pytorch.org/whl/cu118"

    # PyTorch 2.3.x
    elif torch_major == 2 and torch_minor == 3:
        if cuda_major == 12 and cuda_minor >= 1:
            return "https://download.pytorch.org/whl/cu121"
        elif cuda_major >= 11:
            return "https://download.pytorch.org/whl/cu118"
        else:
            return "https://download.pytorch.org/whl/cu118"

    # PyTorch 2.2.x
    elif torch_major == 2 and torch_minor == 2:
        if cuda_major == 12 and cuda_minor >= 1:
            return "https://download.pytorch.org/whl/cu121"
        elif cuda_major >= 11:
            return "https://download.pytorch.org/whl/cu118"
        else:
            return "https://download.pytorch.org/whl/cu118"

    # PyTorch 2.1.x
    elif torch_major == 2 and torch_minor == 1:
        if cuda_major == 12 and cuda_minor >= 1:
            return "https://download.pytorch.org/whl/cu121"
        elif cuda_major >= 11:
            return "https://download.pytorch.org/whl/cu118"
        else:
            return "https://download.pytorch.org/whl/cu118"

    # PyTorch 2.0.x
    elif torch_major == 2 and torch_minor == 0:
        if cuda_major == 11:
            if cuda_minor >= 8:
                return "https://download.pytorch.org/whl/cu118"
            elif cuda_minor >= 7:
                return None  # No special index URL needed for CUDA 11.7
            else:
                return "https://download.pytorch.org/whl/cu118"  # Fallback
        else:
            return "https://download.pytorch.org/whl/cu118"

    # PyTorch 1.x (uses --extra-index-url, but we'll use --index-url for consistency)
    elif torch_major == 1:
        if cuda_major >= 11:
            return "https://download.pytorch.org/whl/cu118"
        else:
            return "https://download.pytorch.org/whl/cu118"

    # Unknown torch version, use conservative defaults
    else:
        if cuda_major >= 12:
            return "https://download.pytorch.org/whl/cu128"
        elif cuda_major >= 11:
            return "https://download.pytorch.org/whl/cu118"
        else:
            return "https://download.pytorch.org/whl/cu118"


def _get_torch_rocm_url(torch_major: Optional[int], torch_minor: Optional[int], torch_patch: Optional[int],
                        rocm_version: str) -> str:
    """
    Get the appropriate PyTorch ROCm URL based on torch and ROCm versions.

    :param torch_major: Major version of torch (e.g., 2 for torch 2.x.x)
    :param torch_minor: Minor version of torch (e.g., 7 for torch 2.7.x)
    :param torch_patch: Patch version of torch (e.g., 1 for torch 2.7.1)
    :param rocm_version: ROCm version string (e.g., "6.3", "5.7")
    :return: The appropriate PyTorch ROCm index URL.
    """
    # Parse ROCm version
    try:
        rocm_parts = rocm_version.split('.')
        rocm_major = int(rocm_parts[0])
        rocm_minor = int(rocm_parts[1]) if len(rocm_parts) > 1 else 0
        rocm_patch = int(rocm_parts[2]) if len(rocm_parts) > 2 else 0
    except (ValueError, IndexError):
        # If we can't parse ROCm version, use conservative defaults
        return "https://download.pytorch.org/whl/rocm6.3"

    # If we can't determine torch version, use conservative defaults
    if torch_major is None or torch_minor is None:
        if rocm_major >= 6:
            return "https://download.pytorch.org/whl/rocm6.3"
        elif rocm_major >= 5:
            return "https://download.pytorch.org/whl/rocm5.7"
        else:
            return "https://download.pytorch.org/whl/rocm5.7"

    # PyTorch 2.7.x
    if torch_major == 2 and torch_minor >= 7:
        if rocm_major == 6 and rocm_minor >= 3:
            return "https://download.pytorch.org/whl/rocm6.3"
        else:
            return "https://download.pytorch.org/whl/rocm6.3"  # Fallback

    # PyTorch 2.6.x
    elif torch_major == 2 and torch_minor == 6:
        if rocm_major == 6:
            if rocm_minor >= 2:
                if rocm_minor == 2 and rocm_patch >= 4:
                    return "https://download.pytorch.org/whl/rocm6.2.4"
                elif rocm_minor >= 2:
                    return "https://download.pytorch.org/whl/rocm6.2"
                else:
                    return "https://download.pytorch.org/whl/rocm6.1"
            elif rocm_minor >= 1:
                return "https://download.pytorch.org/whl/rocm6.1"
            else:
                return "https://download.pytorch.org/whl/rocm6.1"  # Fallback
        else:
            return "https://download.pytorch.org/whl/rocm6.1"  # Fallback

    # PyTorch 2.5.x
    elif torch_major == 2 and torch_minor == 5:
        if rocm_major == 6:
            if rocm_minor >= 2:
                return "https://download.pytorch.org/whl/rocm6.2"
            elif rocm_minor >= 1:
                return "https://download.pytorch.org/whl/rocm6.1"
            else:
                return "https://download.pytorch.org/whl/rocm6.1"  # Fallback
        else:
            return "https://download.pytorch.org/whl/rocm6.1"  # Fallback

    # PyTorch 2.4.x
    elif torch_major == 2 and torch_minor == 4:
        if rocm_major == 6 and rocm_minor >= 1:
            return "https://download.pytorch.org/whl/rocm6.1"
        else:
            return "https://download.pytorch.org/whl/rocm6.1"  # Fallback

    # PyTorch 2.3.x
    elif torch_major == 2 and torch_minor == 3:
        if rocm_major == 6 and rocm_minor >= 0:
            return "https://download.pytorch.org/whl/rocm6.0"
        else:
            return "https://download.pytorch.org/whl/rocm6.0"  # Fallback

    # PyTorch 2.2.x
    elif torch_major == 2 and torch_minor == 2:
        if rocm_major == 5:
            if rocm_minor >= 7:
                return "https://download.pytorch.org/whl/rocm5.7"
            elif rocm_minor >= 6:
                return "https://download.pytorch.org/whl/rocm5.6"
            else:
                return "https://download.pytorch.org/whl/rocm5.6"  # Fallback
        else:
            return "https://download.pytorch.org/whl/rocm5.7"  # Fallback

    # PyTorch 2.1.x
    elif torch_major == 2 and torch_minor == 1:
        if rocm_major == 5 and rocm_minor >= 6:
            return "https://download.pytorch.org/whl/rocm5.6"
        else:
            return "https://download.pytorch.org/whl/rocm5.6"  # Fallback

    # PyTorch 2.0.x
    elif torch_major == 2 and torch_minor == 0:
        if rocm_major == 5:
            if rocm_minor >= 4:
                if rocm_minor == 4 and rocm_patch >= 2:
                    return "https://download.pytorch.org/whl/rocm5.4.2"
                else:
                    return "https://download.pytorch.org/whl/rocm5.4.2"  # Fallback
            else:
                return "https://download.pytorch.org/whl/rocm5.4.2"  # Fallback
        else:
            return "https://download.pytorch.org/whl/rocm5.4.2"  # Fallback

    # PyTorch 1.x (uses --extra-index-url, but we'll use --index-url for consistency)
    elif torch_major == 1:
        if rocm_major >= 5:
            return "https://download.pytorch.org/whl/rocm5.7"
        else:
            return "https://download.pytorch.org/whl/rocm5.7"

    # Unknown torch version, use conservative defaults
    else:
        if rocm_major >= 6:
            return "https://download.pytorch.org/whl/rocm6.3"
        elif rocm_major >= 5:
            return "https://download.pytorch.org/whl/rocm5.7"
        else:
            return "https://download.pytorch.org/whl/rocm5.7"


def _get_torch_xpu_url(torch_major: Optional[int], torch_minor: Optional[int], torch_patch: Optional[int]) -> Optional[
    str]:
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
