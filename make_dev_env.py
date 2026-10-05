#!/usr/bin/env python3

import subprocess
import shutil
import platform
import sys
from pathlib import Path


def run_command(cmd, check=True, capture_output=False):
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=capture_output,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {cmd}")
        print(f"Error: {e}")
        if check:
            sys.exit(1)
        return e


ROCM_INDEX = "https://download.pytorch.org/whl/rocm7.2/"
CUDA_INDEX = "https://download.pytorch.org/whl/cu130/"
XPU_INDEX = "https://download.pytorch.org/whl/xpu/"


def detect_gpu():
    # Check for ROCm
    if shutil.which("rocminfo"):
        return ROCM_INDEX

    # Check for NVIDIA
    if shutil.which("nvidia-smi"):
        return CUDA_INDEX

    # Check for Intel XPU
    if shutil.which("xpu-smi"):
        return XPU_INDEX

    # No GPU detected, use CPU-only
    return None


def get_install_extras():
    gpu_url = detect_gpu()

    # Match the network installer: Vulkan preview by default, OpenGL kept for
    # DGENERATE_CONSOLE_UI_VULKAN=0. bitsandbytes 0.50 covers CUDA, ROCm, XPU,
    # and Apple MPS (Metal kernels come from the hard ``kernels`` dependency).
    # xllamacpp's PyPI wheel is replaced after install when a CUDA / ROCm /
    # Vulkan build applies.
    base_extras = [
        "dev",
        "ncnn",
        "xllamacpp",
        "bitsandbytes",
        "console_ui_vulkan",
        "console_ui_opengl",
    ]

    if platform.system() == "Windows" and _triton_windows_wanted():
        base_extras.append("triton_windows")

    return base_extras, gpu_url


def _triton_windows_wanted() -> bool:
    """NVIDIA, or AMD RDNA 3+ (same rule as the network installer)."""
    installer_dir = Path(__file__).parent / 'installer'
    if str(installer_dir) not in sys.path:
        sys.path.insert(0, str(installer_dir))
    try:
        from network_installer.platform_detection import detect_gpu as detect_gpu_info
        from network_installer.platform_detection import triton_windows_compatible
    except ImportError:
        from platform_detection import detect_gpu as detect_gpu_info
        from platform_detection import triton_windows_compatible
    return triton_windows_compatible(detect_gpu_info())


def venv_paths(venv_path: Path):
    if platform.system() == "Windows":
        return (
            venv_path / "Scripts" / "activate.bat",
            venv_path / "Scripts" / "python.exe",
        )
    return (
        venv_path / "bin" / "activate",
        venv_path / "bin" / "python",
    )


def package_importable(python_exe: Path, module: str) -> bool:
    result = subprocess.run(
        [str(python_exe), '-c', f'import {module}'],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def install_xllamacpp_for_machine(script_dir: Path, python_exe: Path) -> None:
    """Ensure the xllamacpp extra is present, then swap in a GPU wheel when one applies."""
    if not package_importable(python_exe, 'xllamacpp'):
        print("xllamacpp missing after the editable install; installing the extra...")
        result = run_command(
            f'"{python_exe}" -m pip install --editable "{script_dir}[xllamacpp]"',
            check=False,
        )
        if result.returncode != 0 or not package_importable(python_exe, 'xllamacpp'):
            print("\nFailed to install xllamacpp.")
            sys.exit(1)

    print("Selecting the xllamacpp wheel for this machine...")
    selector = script_dir / 'installer' / 'network_installer' / 'xllamacppinstall.py'
    # Run with the venv interpreter so detection and pip target the new env.
    # Pass the same interpreter explicitly so a broken PATH cannot retarget pip.
    wheel = subprocess.run(
        [str(python_exe), str(selector)],
        cwd=str(script_dir),
        check=False,
        text=True,
    )
    if wheel.returncode != 0:
        # Same policy as the network installer: keep the PyPI wheel.
        print(
            "Warning: could not install a GPU xllamacpp wheel. "
            "The PyPI build from the extra is still installed."
        )
    elif not package_importable(python_exe, 'xllamacpp'):
        print("\nxllamacpp is not importable after wheel selection.")
        sys.exit(1)


def main():
    script_dir = Path(__file__).parent.absolute()
    venv_path = script_dir / "venv"

    print("Setting up dgenerate development environment...")
    print(f"Script directory: {script_dir}")
    print(f"Virtual environment will be created at: {venv_path}")

    # Remove existing venv if it exists
    if venv_path.exists():
        print("Removing existing virtual environment...")
        shutil.rmtree(venv_path)

    # Create virtual environment
    print("Creating virtual environment...")
    run_command(f'"{sys.executable}" -m venv "{venv_path}"')

    activate_script, python_exe = venv_paths(venv_path)

    # Verify virtual environment was created
    if not python_exe.exists():
        print(f"Error: Virtual environment was not created properly. {python_exe} not found.")
        sys.exit(1)

    print(f"Virtual environment created successfully at: {venv_path}")
    print(f"Python executable: {python_exe}")

    # Get installation extras and PyTorch index URL
    extras, pytorch_url = get_install_extras()
    extras_str = ",".join(extras)

    print(f"Detected extras: {extras_str}")
    if pytorch_url:
        print(f"PyTorch index URL: {pytorch_url}")
    else:
        print("No GPU detected, using CPU-only PyTorch")

    # Use python -m pip so the venv pip module is what runs, not a shadowed pip.exe
    install_cmd = f'"{python_exe}" -m pip install --upgrade pip'
    print(f"Upgrading pip: {install_cmd}")
    run_command(install_cmd, check=False)

    install_cmd = f'"{python_exe}" -m pip install --editable "{script_dir}[{extras_str}]"'
    if pytorch_url:
        install_cmd += f' --extra-index-url {pytorch_url}'

    print("Installing dgenerate in development mode...")
    print(f"Command: {install_cmd}")

    result = run_command(install_cmd, check=False)

    if result.returncode != 0 and 'console_ui_vulkan' in extras:
        # Vulkan bindings failed on this platform/pip; keep OpenGL and retry.
        print(
            "Editable install failed with console_ui_vulkan; "
            "retrying without that extra..."
        )
        extras = [extra for extra in extras if extra != 'console_ui_vulkan']
        extras_str = ",".join(extras)
        install_cmd = f'"{python_exe}" -m pip install --editable "{script_dir}[{extras_str}]"'
        if pytorch_url:
            install_cmd += f' --extra-index-url {pytorch_url}'
        print(f"Command: {install_cmd}")
        result = run_command(install_cmd, check=False)

    if result.returncode == 0:
        install_xllamacpp_for_machine(script_dir, python_exe)
        if 'console_ui_vulkan' in extras and not package_importable(python_exe, 'vulkan'):
            print(
                "Warning: console_ui_vulkan was requested but vulkan did not import; "
                "the OpenGL preview will be used."
            )
        print("\nDevelopment environment setup completed successfully!")
        print(f"\nTo activate the virtual environment:")
        if platform.system() == "Windows":
            print(f'  {activate_script}')
        else:
            print(f'  source "{activate_script}"')
        print(f"\nTo run dgenerate:")
        print(f'dgenerate --help')
    else:
        print("\nInstallation failed!")
        print("Please check the error messages above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
