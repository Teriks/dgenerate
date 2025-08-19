#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil
import platform
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


def detect_gpu():
    # Check for ROCm
    if shutil.which("rocminfo"):
        return "https://download.pytorch.org/whl/rocm6.4/"
    
    # Check for NVIDIA
    if shutil.which("nvidia-smi"):
        return "https://download.pytorch.org/whl/cu128/"
    
    # Check for Intel XPU
    if shutil.which("xpu-smi"):
        return "https://download.pytorch.org/whl/xpu/"
    
    # No GPU detected, use CPU-only
    return None


def get_install_extras():
    gpu_url = detect_gpu()
    
    base_extras = ["dev", "ncnn", "gpt4all", "console_ui_opengl"]
    
    if gpu_url == "https://download.pytorch.org/whl/cu128/":
        # NVIDIA GPU
        base_extras.extend(["gpt4all_cuda", "bitsandbytes"])
        if platform.system() == "Windows":
            base_extras.append("triton_windows")
    elif gpu_url == "https://download.pytorch.org/whl/xpu/":
        # Intel XPU
        base_extras.extend(["bitsandbytes"])
    elif gpu_url == "https://download.pytorch.org/whl/rocm6.4/":
        # AMD ROCm - no additional extras needed
        pass
    
    return base_extras, gpu_url


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
    
    # Determine the activation script based on platform
    if platform.system() == "Windows":
        activate_script = venv_path / "Scripts" / "activate.bat"
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:
        activate_script = venv_path / "bin" / "activate"
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
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
    
    # Build the pip install command
    install_cmd = f'"{pip_exe}" install --editable "{script_dir}[{extras_str}]"'
    if pytorch_url:
        install_cmd += f' --extra-index-url {pytorch_url}'
    
    print("Installing dgenerate in development mode...")
    print(f"Command: {install_cmd}")
    
    # Install the package
    result = run_command(install_cmd, check=False)
    
    if result.returncode == 0:
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
