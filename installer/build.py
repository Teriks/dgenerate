#!/usr/bin/env python3

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
Main build script for the dgenerate network installer.
This script orchestrates the entire build process:
1. Build windowed stubs for all platforms
2. Build the network installer
"""

import os
import sys
import subprocess
import platform
import shutil
import tempfile
import zipfile
from pathlib import Path

def run_command(cmd, description, cwd=None):
    """Run a command and handle errors."""

    cmd = [str(i) for i in cmd]

    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        print(f"SUCCESS: {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed with return code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False



def build_windowed_stubs(network_installer_dir, venv_python):
    """Build windowed stubs for platforms that need them."""
    print("\n=== Building Windowed Stubs ===")
    
    # Get the current platform
    current_platform = platform.system().lower()
    
    # Skip windowed stub building on macOS - AppleScript approach is used instead
    if current_platform == "darwin":
        print("SUCCESS: Skipping windowed stub build on macOS (AppleScript approach used instead)")
        return True
    
    # Skip windowed stub building on Linux - Terminal=false in .desktop files is used instead
    if current_platform == "linux":
        print("SUCCESS: Skipping windowed stub build on Linux (Terminal=false in .desktop files used instead)")
        return True
    
    # Build for current platform using PyInstaller directly (Windows only)
    spec_file = network_installer_dir / "dgenerate_windowed.spec"
    if not spec_file.exists():
        print("ERROR: PyInstaller spec file not found")
        return False
    
    if not run_command(
        [venv_python, "-m", "PyInstaller", str(spec_file), 
         '--distpath', network_installer_dir / 'network_installer' / 'resources'], 
         f"Building windowed stub for {current_platform}",
         cwd=network_installer_dir
    ):
        return False
    
    # For cross-platform builds, we would need to build on each platform
    # For now, we'll build for the current platform only
    print(f"SUCCESS: Built windowed stub for {current_platform}")

    # Look for the built executable (Windows only at this point)
    if current_platform == "windows":
        stub_name = "dgenerate_windowed.exe"
    else:
        stub_name = "dgenerate_windowed"
    
    # Check if the stub was created
    stub_path = (
        network_installer_dir / 
        'network_installer' / 
        'resources' / stub_name
    )

    if not stub_path.exists():
        print(f"ERROR: Windowed stub not found at {stub_path}")
        return False
    
    print(f"SUCCESS: Windowed stub created: {stub_path}")
    
    return True

def build_shortcut_stubs(network_installer_dir):
    """Build shortcut stubs for all platforms."""
    print("\n=== Building Shortcut Stubs ===")
    
    # The shortcut stub is already in the shortcut_stub directory
    # We just need to ensure it's properly packaged
    shortcut_stub_dir = network_installer_dir / "dgenerate_windowed"
    if not shortcut_stub_dir.exists():
        print("ERROR: shortcut_stub directory not found")
        return False
    
    # Check that the main stub file exists
    stub_file = shortcut_stub_dir / "dgenerate_windowed.py"
    if not stub_file.exists():
        print("ERROR: dgenerate_launcher.py not found in shortcut_stub directory")
        return False
    
    print("SUCCESS: Shortcut stub files ready")
    return True

def create_build_venv(network_installer_dir):
    """Create a virtual environment for building."""
    print("\n=== Creating Build Virtual Environment ===")
    
    venv_dir = network_installer_dir / "build_venv"
    
    # Remove existing venv if it exists
    if venv_dir.exists():
        print("SUCCESS: Removing existing virtual environment...")
        import shutil
        shutil.rmtree(venv_dir)
    
    # Create new virtual environment
    if not run_command([sys.executable, "-m", "venv", str(venv_dir)], 
                      "Creating virtual environment",
                      cwd=network_installer_dir):
        return False, None
    
    # Get the Python executable in the venv
    if platform.system().lower() == "windows":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"
    
    if not venv_python.exists():
        print(f"ERROR: Virtual environment Python not found at {venv_python}")
        return False, None
    
    print(f"SUCCESS: Virtual environment created at {venv_dir}")
    return True, str(venv_python)

def install_dependencies(network_installer_dir, venv_python):
    """Install the network installer package and its dependencies."""
    print("\n=== Installing Dependencies ===")
    
    # Upgrade pip first
    if not run_command([venv_python, "-m", "pip", "install", "--upgrade", "pip"], 
                      "Upgrading pip",
                      cwd=network_installer_dir):
        return False

    # Install the package
    install_cmd = [venv_python, "-m", "pip", "install", "-e", "."]
    
    if not run_command(install_cmd, 
                      "Installing network installer with build dependencies",
                      cwd=network_installer_dir):
        return False
    
    print("SUCCESS: Dependencies installed successfully")
    return True

def build_installer(network_installer_dir, venv_python):
    """Build the network installer executable."""
    print("\n=== Building Network Installer ===")
    
    # Check if we have the required files
    required_files = [
        network_installer_dir / "network_installer.spec"
    ]
    
    for file_path in required_files:
        if not file_path.exists():
            print(f"ERROR: Required file not found: {file_path}")
            return False
    
    # Build the installer using PyInstaller
    spec_file = network_installer_dir / "network_installer.spec"
    if not run_command([venv_python, "-m", "PyInstaller", str(spec_file)], 
                      "Building network installer executable",
                      cwd=network_installer_dir):
        return False
    
    # Check if the installer was built
    dist_dir = network_installer_dir / "dist"
    current_platform = platform.system().lower()
    if current_platform == "windows":
        installer_name = "dgenerate-network-installer.exe"
    else:
        installer_name = "dgenerate-network-installer"
    
    installer_path = dist_dir / installer_name
    if not installer_path.exists():
        print(f"ERROR: Installer executable not found at {installer_path}")
        return False
    
    print(f"SUCCESS: Network installer built successfully: {installer_path}")
    return True



def convert_ico_to_png_subprocess(venv_python, source_path, dest_path):
    """Convert ICO to PNG using subprocess with venv Python."""
    import inspect
    conversion_script = inspect.cleandoc(f'''
        import sys
        try:
            from PIL import Image
            with Image.open("{source_path}") as img:
                img.save("{dest_path}", "PNG")
            print("SUCCESS")
        except Exception as e:
            print(f"ERROR: {{e}}")
            sys.exit(1)
    ''')
    
    try:
        result = subprocess.run(
            [venv_python, "-c", conversion_script],
            capture_output=True, text=True, timeout=60
        )
        
        if result.returncode == 0 and "SUCCESS" in result.stdout:
            return True
        else:
            print(f"Icon conversion failed: {result.stderr or result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        print("Icon conversion timed out")
        return False
    except Exception as e:
        print(f"Error running icon conversion: {e}")
        return False

def copy_resources_to_installer(network_installer_dir, venv_python=None):
    """Ensure resources are available in installer."""
    print("\n=== Verifying Resources in Installer ===")
    
    # Create resources directory in the installer package
    resources_dir = network_installer_dir / "network_installer" / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)

    # Get platform information
    current_platform = platform.system().lower()
    
    # Handle icons based on platform
    icon_files = [
        ("dgenerate/icon.ico", "icon"),
        ("dgenerate/config_icon.ico", "config_icon")
    ]
    
    for source_file, base_name in icon_files:
        source_path = network_installer_dir.parent / source_file
        
        if current_platform == "windows":
            # On Windows, copy ICO files directly
            dest_path = resources_dir / f"{base_name}.ico"
            try:
                shutil.copy2(source_path, dest_path)
                print(f"SUCCESS: Copied {source_file} to {dest_path}")
            except Exception as e:
                print(f"ERROR: Failed to copy {source_file} to resources directory: {e}")
                return False
        else:
            # On Mac/Linux, convert ICO to PNG using venv Python
            dest_path = resources_dir / f"{base_name}.png"
            if not venv_python:
                print("ERROR: venv_python required for icon conversion on Mac/Linux")
                return False
                
            if not convert_ico_to_png_subprocess(venv_python, source_path, dest_path):
                print(f"ERROR: Failed to convert {source_file} to PNG")
                return False
            else:
                print(f"SUCCESS: Converted {source_file} to PNG at {dest_path}")
    
    # Copy LICENSE file
    try:
        shutil.copy2(network_installer_dir.parent / "LICENSE", resources_dir / "LICENSE")
    except Exception as e:
        print(f"ERROR: Failed to copy LICENSE to resources directory: {e}")
        return False
    

    print("SUCCESS: Resources directory created")
    return True


def main():
    """Main build orchestration."""
    print("=== dgenerate Network Installer Build ===")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")
    
    # Get the network_installer directory path (script's location)
    network_installer_dir = Path(__file__).parent.absolute()
    
    # Step 1: Create build virtual environment
    success, venv_python = create_build_venv(network_installer_dir)
    if not success:
        print("\nERROR: Virtual environment creation failed. Aborting installer build.")
        return 1
    
    # Step 2: Install dependencies
    if not install_dependencies(network_installer_dir, venv_python):
        print("\nERROR: Dependency installation failed. Aborting installer build.")
        return 1
    
    # Step 3: Copy resources to installer (must be done before windowed stub build)
    if not copy_resources_to_installer(network_installer_dir, venv_python):
        print("\nERROR: Failed to copy resources to installer. Aborting installer build.")
        return 1
    
    # Step 4: Build shortcut stubs
    if not build_shortcut_stubs(network_installer_dir):
        print("\nERROR: Shortcut stub build failed. Aborting installer build.")
        return 1
    
    # Step 5: Build windowed stubs
    if not build_windowed_stubs(network_installer_dir, venv_python):
        print("\nERROR: Windowed stub build failed. Aborting installer build.")
        return 1
    
    # Step 6: Build the installer
    if not build_installer(network_installer_dir, venv_python):
        print("\nERROR: Installer build failed.")
        return 1
    
    print("\n=== Build Complete ===")
    print("SUCCESS: Shortcut stubs ready")
    print("SUCCESS: Windowed stubs built and copied to installer")
    print("SUCCESS: Network installer built successfully")
    print("\nDistribution files created in:")
    print("  - dist/ (installer packages)")
    print("  - network_installer/network_installer/resources/ (stubs)")
    print("  - shortcut_stub/ (shortcut stubs)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())