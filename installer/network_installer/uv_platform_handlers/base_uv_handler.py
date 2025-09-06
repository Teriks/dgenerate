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
Base platform handler for the dgenerate installer.
Contains common functionality shared across all platforms.
"""

import importlib.resources as resources
import inspect
import os
import platform
import shutil
import ssl
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

import certifi
from network_installer.subprocess_utils import run_silent, popen_silent


class BasePlatformHandler(ABC):
    """
    Abstract base class for platform-specific handlers.
    Contains common functionality and defines the interface for platform-specific operations.
    """

    def __init__(self, installer_instance):
        """
        Initialize the platform handler.
        
        :param installer_instance: Reference to the main UvInstaller instance
        """
        self.installer = installer_instance
        self.log_callback = installer_instance.log_callback
        self.source_dir = installer_instance.source_dir
        self.system = installer_instance.system

        # Set up SSL context for secure downloads
        self._ssl_context = self._create_ssl_context()

        # Common paths
        self.install_base = installer_instance.install_base
        self.uv_dir = installer_instance.uv_dir
        self.venv_dir = installer_instance.venv_dir

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create an SSL context using certifi certificates."""
        context = ssl.create_default_context()
        context.load_verify_locations(certifi.where())
        return context

    @property
    def scripts_dir(self) -> Path:
        """Get the scripts directory in the virtual environment."""
        return self.get_venv_scripts_dir()

    @property
    def bin_dir(self) -> Path:
        """Get the bin directory where stub scripts are placed (for PATH)."""
        return self.install_base / 'bin'

    @property
    def venv_bin_path(self) -> Path:
        """Get the actual venv bin/Scripts directory path."""
        return self.get_venv_scripts_dir()

    def get_uv_download_url(self) -> str:
        """Get the appropriate uv download URL for the current platform."""
        machine = platform.machine().lower()

        # Map platform.machine() to uv's arch names
        arch_map = {
            'x86_64': 'x86_64',
            'amd64': 'x86_64',
            'arm64': 'aarch64',
            'aarch64': 'aarch64',
            'armv7l': 'armv7',
        }

        arch = arch_map.get(machine, 'x86_64')  # Default to x86_64

        if self.system == 'windows':
            if arch == 'x86_64':
                return "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
            else:
                # Fallback to x86_64 for unsupported Windows architectures
                return "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
        elif self.system == 'linux':
            return f"https://github.com/astral-sh/uv/releases/latest/download/uv-{arch}-unknown-linux-gnu.tar.gz"
        elif self.system == 'darwin':  # macOS
            return f"https://github.com/astral-sh/uv/releases/latest/download/uv-{arch}-apple-darwin.tar.gz"
        else:
            raise ValueError(f"Unsupported platform: {self.system}")

    def find_or_install_uv(self) -> Path | None:
        """
        Download and install uv for the current platform.
        Always downloads a fresh uv, never reuses existing installations.
        
        :return: Path to the uv executable or None if failed
        """
        # Always download and install a fresh uv
        self.log_callback("Downloading fresh uv installation...")
        return self.download_and_extract_uv()

    def download_and_extract_uv(self) -> Path | None:
        """
        Download and extract uv for the current platform.
        
        :return: Path to the uv executable or None if failed
        """
        try:
            # Get download URL
            url = self.get_uv_download_url()
            self.log_callback(f"Downloading uv from: {url}")

            # Create temporary file for download
            with tempfile.NamedTemporaryFile(delete=False,
                                             suffix='.zip' if self.system == 'windows' else '.tar.gz') as tmp_file:
                # Download uv
                with urllib.request.urlopen(url, context=self._ssl_context) as response:
                    shutil.copyfileobj(response, tmp_file)
                tmp_path = tmp_file.name

            # Extract uv
            self.uv_dir.mkdir(parents=True, exist_ok=True)

            if self.system == 'windows':
                # Extract zip file
                with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                    zip_ref.extractall(self.uv_dir)

                # Find the uv.exe file
                uv_exe = next(self.uv_dir.rglob('uv.exe'), None)
            else:
                # Extract tar.gz file
                with tarfile.open(tmp_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(self.uv_dir)

                # Find the uv executable
                uv_exe = next(self.uv_dir.rglob('uv'), None)

            # Clean up temporary file
            os.unlink(tmp_path)

            if uv_exe and uv_exe.exists():
                # Make executable on Unix systems
                if self.system != 'windows':
                    uv_exe.chmod(0o755)

                self.log_callback(f"✓ uv installed successfully: {uv_exe}")
                return uv_exe
            else:
                self.log_callback("Error: Could not find uv executable after extraction")
                return None

        except Exception as e:
            self.log_callback(f"Error downloading/extracting uv: {e}")
            return None

    def create_venv(self, uv_exe: Path, python_version: str = None) -> bool:
        """
        Create a Python virtual environment using uv.
        
        :param uv_exe: Path to uv executable
        :param python_version: Python version to use (e.g., "3.11", "3.12")
        :return: True if successful, False otherwise
        """
        try:
            # Clean up existing venv if it exists
            if self.venv_dir.exists():
                shutil.rmtree(self.venv_dir)

            # Create new venv
            cmd = [str(uv_exe), 'venv', str(self.venv_dir)]
            if python_version:
                cmd.extend(['--python', python_version])

            self.log_callback(f"Creating virtual environment: {' '.join(cmd)}")

            result = run_silent(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                self.log_callback("✓ Virtual environment created successfully")
                return True
            else:
                # Check if it's a permission error with managed Python installations
                if python_version and (
                        "Permission denied" in result.stderr or "Failed to inspect Python interpreter" in result.stderr):
                    self.log_callback(f"Permission error with managed Python {python_version} installation")
                    self.log_callback("Clearing corrupted managed Python installations...")

                    # Clear uv's managed Python installations
                    if self._clear_uv_python_installations(uv_exe):
                        self.log_callback("Retrying virtual environment creation...")

                        # Retry the same command
                        result = run_silent(cmd, capture_output=True, text=True, timeout=300)

                        if result.returncode == 0:
                            self.log_callback(
                                "✓ Virtual environment created successfully after clearing corrupted installations")
                            return True
                        else:
                            self.log_callback(f"Still failed after clearing: {result.stderr}")
                            return False
                    else:
                        self.log_callback("Failed to clear managed Python installations")
                        return False
                else:
                    self.log_callback(f"Failed to create virtual environment: {result.stderr}")
                    return False

        except Exception as e:
            self.log_callback(f"Error creating virtual environment: {e}")
            return False

    def _clear_uv_python_installations(self, uv_exe: Path) -> bool:
        """
        Clear uv's managed Python installations to resolve permission issues.
        
        :param uv_exe: Path to uv executable
        :return: True if successful, False otherwise
        """
        try:
            # Get the uv Python cache directory
            result = run_silent([str(uv_exe), 'python', 'dir'], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                cache_dir = Path(result.stdout.strip())
                if cache_dir.exists():
                    self.log_callback(f"Clearing Python cache directory: {cache_dir}")
                    shutil.rmtree(cache_dir)
                    self.log_callback("✓ Python cache directory cleared")
                    return True
                else:
                    self.log_callback("Python cache directory does not exist")
                    return True
            else:
                self.log_callback(f"Failed to get Python cache directory: {result.stderr}")

                # Fallback: try to find and clear the standard cache location
                if self.system == 'darwin':  # macOS
                    cache_path = Path.home() / '.local' / 'share' / 'uv' / 'python'
                elif self.system == 'linux':
                    cache_path = Path.home() / '.local' / 'share' / 'uv' / 'python'
                elif self.system == 'windows':
                    cache_path = Path.home() / 'AppData' / 'Local' / 'uv' / 'python'
                else:
                    return False

                if cache_path.exists():
                    self.log_callback(f"Clearing fallback Python cache directory: {cache_path}")
                    shutil.rmtree(cache_path)
                    self.log_callback("✓ Fallback Python cache directory cleared")
                    return True
                else:
                    self.log_callback("No Python cache directory found to clear")
                    return True

        except Exception as e:
            self.log_callback(f"Error clearing Python installations: {e}")
            return False

    def install_dgenerate(self, uv_exe: Path, source_dir: str, selected_extras: list[str],
                          torch_index_url: str | None = None) -> bool:
        """
        Install dgenerate using uv pip (fast Rust-based pip implementation).
        
        :param uv_exe: Path to uv executable
        :param source_dir: Directory containing dgenerate source code
        :param selected_extras: List of extras to install
        :param torch_index_url: Optional PyTorch index URL
        :return: True if successful, False otherwise
        """
        try:
            # Build the install command with uv pip targeting the virtual environment
            # Use regular install (not editable) to avoid dependency on source directory
            # uv will automatically copy the source to site-packages during installation
            cmd = [str(uv_exe), 'pip', 'install', '--python', str(self.get_venv_python()), source_dir]

            # Add extras if specified
            if selected_extras:
                extras_str = '[' + ','.join(selected_extras) + ']'
                cmd[cmd.index(source_dir)] = f"{source_dir}{extras_str}"

            # Add torch index URL if specified
            if torch_index_url:
                cmd.extend(['--extra-index-url', torch_index_url, '--index-strategy', 'unsafe-best-match'])

            self.log_callback(f"Installing dgenerate: {' '.join(cmd)}")

            # Run the installation
            result = run_silent(cmd, capture_output=True, text=True, timeout=1800)

            if result.returncode == 0:
                self.log_callback("dgenerate installed successfully")

                # Copy windowed stub to bin directory
                self.log_callback("Copying windowed stub to bin directory...")
                if not self._copy_windowed_stub():
                    self.log_callback("Warning: Failed to copy windowed stub, but installation is still functional")
                else:
                    self.log_callback("✓ Windowed stub copied to Scripts directory")

                # Copy dgenerate executable to bin directory
                self.log_callback("Copying dgenerate executable to bin directory...")
                if not self._copy_dgenerate_executable():
                    self.log_callback("Warning: Failed to copy dgenerate executable to bin directory")
                else:
                    self.log_callback("✓ dgenerate executable copied to bin directory")

                # Precompile Python bytecode for faster startup
                self.log_callback("Precompiling Python bytecode for faster startup...")
                if not self._compile_bytecode(uv_exe):
                    self.log_callback("Warning: Bytecode compilation failed, but installation is still functional")
                else:
                    self.log_callback("✓ Python bytecode compilation completed")

                return True
            else:
                self.log_callback(f"Failed to install dgenerate: {result.stderr}")
                return False

        except Exception as e:
            self.log_callback(f"Error installing dgenerate: {e}")
            return False

    def _compile_bytecode(self, uv_exe: Path) -> bool:
        """
        Precompile Python bytecode for faster startup.
        
        :param uv_exe: Path to uv executable
        :return: True if successful, False otherwise
        """
        try:
            # Get the site-packages directory
            if self.system == 'windows':
                site_packages = self.venv_dir / "Lib" / "site-packages"
            else:
                # Dynamically detect the Python version in the venv instead of using installer's version
                lib_dir = self.venv_dir / "lib"
                if lib_dir.exists():
                    # Find python* directories (e.g., python3.13, python3.12, etc.)
                    python_dirs = list(lib_dir.glob('python*'))
                    if python_dirs:
                        # Use the first (and usually only) python directory found
                        python_dir = python_dirs[0]
                        site_packages = python_dir / "site-packages"
                    else:
                        # Fallback to sys.version_info if no python dirs found
                        site_packages = lib_dir / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
                else:
                    # Fallback if lib directory doesn't exist
                    site_packages = self.venv_dir / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

            if not site_packages.exists():
                self.log_callback(f"Warning: site-packages directory not found at {site_packages}")
                return False

            # Use Python's compileall module to compile all Python files
            # Run the compilation directly with the venv python to avoid uv run
            compile_cmd = [
                str(self.get_venv_python()),
                '-m', 'compileall', str(site_packages), '-q'
            ]

            self.log_callback(f"Running bytecode compilation: {' '.join(compile_cmd)}")

            # Run compilation with a timeout
            result = run_silent(compile_cmd, capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                self.log_callback("Bytecode compilation completed successfully")
                return True
            else:
                self.log_callback(f"Bytecode compilation failed with return code {result.returncode}")
                if result.stderr:
                    self.log_callback(f"Compilation error: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.log_callback("Bytecode compilation timed out after 10 minutes")
            return False
        except Exception as e:
            self.log_callback(f"Error during bytecode compilation: {e}")
            return False

    def get_venv_python(self) -> Path:
        """Get the path to the Python executable in the virtual environment."""
        if self.system == 'windows':
            return self.venv_dir / 'Scripts' / 'python.exe'
        else:
            return self.venv_dir / 'bin' / 'python'

    def _copy_windowed_stub(self) -> bool:
        """
        Copy the windowed stub executable to both the Scripts directory and bin directory.
        The bin directory copy is what shortcuts/file associations will use.
        
        :return: True if successful, False otherwise
        """
        try:
            # Determine the stub name based on platform
            if self.system == 'windows':
                stub_name = 'dgenerate_windowed.exe'
            else:
                stub_name = 'dgenerate_windowed'

            # Try to find the stub in the installer resources
            stub_source = None

            # Method 1: Look in the installer package resources
            try:
                with resources.path('network_installer.resources', stub_name) as resource_path:
                    if resource_path.exists():
                        stub_source = resource_path
            except (ImportError, FileNotFoundError):
                pass

            # Method 2: Look in PyInstaller bundle
            if not stub_source:
                try:
                    if hasattr(sys, '_MEIPASS'):
                        # PyInstaller bundle
                        bundle_path = Path(sys._MEIPASS) / 'network_installer' / 'resources' / stub_name
                        if bundle_path.exists():
                            stub_source = bundle_path
                except Exception:
                    pass

            # Method 3: Look in development resources directory
            if not stub_source:
                try:
                    # Development mode - look for resources directory
                    current_dir = Path(__file__).parent.parent
                    dev_path = current_dir / 'resources' / stub_name
                    if dev_path.exists():
                        stub_source = dev_path
                except Exception:
                    pass

            if not stub_source:
                self.log_callback(f"Warning: Could not find windowed stub {stub_name} in any expected location")
                return False

            # Copy dgenerate_windowed directly to bin directory only
            # This is our custom executable, not a standard Python package
            self.bin_dir.mkdir(parents=True, exist_ok=True)
            bin_target_path = self.bin_dir / stub_name
            shutil.copy2(stub_source, bin_target_path)

            # Make executable on Unix systems
            if self.system != 'windows':
                bin_target_path.chmod(0o755)

            self.log_callback(f"✓ Copied {stub_name} to {bin_target_path}")

            # Note: dgenerate_windowed is NOT copied to Scripts directory
            # as it's a custom executable designed specifically for bin directory

            # Copy icon files for desktop shortcuts
            if not self._copy_icons():
                self.log_callback("Warning: Failed to copy desktop shortcut icons")

            # Copy icon files for file associations
            if not self._copy_file_association_icon():
                self.log_callback("Warning: Failed to copy file association icons")

            return True

        except Exception as e:
            self.log_callback(f"Error copying windowed stub: {e}")
            return False

    def _copy_dgenerate_executable(self) -> bool:
        """
        Copy and modify the dgenerate executable to work from bin directory.
        This modifies the Python script stub to find the venv from the new location.
        
        :return: True if successful, False otherwise
        """
        try:
            # Determine the executable name based on platform
            if self.system == 'windows':
                exe_name = 'dgenerate.exe'
            else:
                exe_name = 'dgenerate'

            # Source executable in Scripts/bin directory
            source_exe = self.scripts_dir / exe_name
            if not source_exe.exists():
                self.log_callback(f"Warning: {exe_name} not found at {source_exe}")
                return False

            # Create bin directory and target path
            self.bin_dir.mkdir(parents=True, exist_ok=True)
            target_exe = self.bin_dir / exe_name

            # Copy the executable to bin directory
            shutil.copy2(source_exe, target_exe)

            # Make executable on Unix systems
            if self.system != 'windows':
                target_exe.chmod(0o755)

            # Now modify the executable to handle the new environment location
            if not self._modify_dgenerate_executable(target_exe):
                self.log_callback(f"Warning: Failed to modify {exe_name} for new location")
                return False

            self.log_callback(f"✓ Copied and modified {exe_name} to {target_exe}")

            # Also keep an unmodified copy in the original Scripts/bin directory
            # in case other tools expect to find it there
            original_backup = self.scripts_dir / f"{exe_name}.original"
            if not original_backup.exists():
                shutil.copy2(source_exe, original_backup)
                self.log_callback(f"✓ Kept backup copy at {original_backup}")

            return True

        except Exception as e:
            self.log_callback(f"Error copying dgenerate executable: {e}")
            return False

    def _modify_dgenerate_executable(self, exe_path: Path) -> bool:
        """
        Modify the Python script stub to find the venv from bin directory location.
        
        :param exe_path: Path to the executable to modify
        :return: True if successful, False otherwise
        """
        try:
            # Read the executable content
            with open(exe_path, 'rb') as f:
                content = f.read()

            # For Windows, we need to modify the Python script launcher
            if self.system == 'windows':
                return self._modify_windows_script_launcher(exe_path, content)
            else:
                return self._modify_unix_script_stub(exe_path, content)

        except Exception as e:
            self.log_callback(f"Error modifying executable {exe_path}: {e}")
            return False

    def _modify_windows_script_launcher(self, exe_path: Path, content: bytes) -> bool:
        """
        Modify Windows Python script launcher to add venv to PATH.
        
        :param exe_path: Path to the executable
        :param content: Original executable content
        :return: True if successful, False otherwise
        """
        try:
            # Windows script launchers are complex binaries with embedded Python code
            # We need to find and patch the embedded script portion

            # Look for the embedded Python script pattern in the launcher
            # This is a simplified approach - may need refinement based on Python version
            script_marker = b'#!python'

            if script_marker in content:
                # Find the script section and inject PATH modification
                marker_pos = content.find(script_marker)

                # Create PATH modification code
                path_setup_code = inspect.cleandoc(f'''
                    import os
                    import sys
                    from pathlib import Path

                    # Add venv Scripts directory to PATH if we're in bin directory
                    exe_dir = Path(sys.executable).parent
                    if exe_dir.name == 'bin':
                        venv_scripts = exe_dir.parent / 'venv' / 'Scripts'
                        if venv_scripts.exists():
                            current_path = os.environ.get('PATH', '')
                            if str(venv_scripts) not in current_path:
                                os.environ['PATH'] = f"{{venv_scripts}};{{current_path}}"
                ''').encode('utf-8')

                # Insert the code after the shebang line
                newline_pos = content.find(b'\n', marker_pos)
                if newline_pos != -1:
                    new_content = content[:newline_pos + 1] + path_setup_code + content[newline_pos + 1:]

                    with open(exe_path, 'wb') as f:
                        f.write(new_content)

                    return True

            # Fallback: If we can't modify the launcher, just copy it
            # The user will need to ensure PATH is set correctly
            self.log_callback(f"Warning: Could not modify Windows launcher {exe_path}, copied as-is")
            return True

        except Exception as e:
            self.log_callback(f"Error modifying Windows launcher {exe_path}: {e}")
            return False

    def _modify_unix_script_stub(self, exe_path: Path, content: bytes) -> bool:
        """
        Modify Unix Python script stub to add venv to PATH.
        
        :param exe_path: Path to the executable
        :param content: Original executable content
        :return: True if successful, False otherwise
        """
        try:
            # Unix scripts are usually text files with shebang
            content_str = content.decode('utf-8')

            # Find the shebang line
            lines = content_str.split('\n')
            if lines and lines[0].startswith('#!'):
                # Insert PATH setup after shebang
                path_setup = inspect.cleandoc(f'''
                    import os
                    import sys
                    from pathlib import Path

                    # Add venv bin directory to PATH if we're in bin directory
                    exe_dir = Path(sys.executable).parent  
                    if exe_dir.name == 'bin':
                        venv_bin = exe_dir.parent / 'venv' / 'bin'
                        if venv_bin.exists():
                            current_path = os.environ.get('PATH', '')
                            if str(venv_bin) not in current_path:
                                os.environ['PATH'] = f"{{venv_bin}}:{{current_path}}"
                ''')

                # Insert after shebang
                modified_lines = [lines[0], path_setup] + lines[1:]
                modified_content = '\n'.join(modified_lines)

                with open(exe_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)

                return True
            else:
                # Not a script file, copy as-is
                self.log_callback(f"Warning: {exe_path} doesn't appear to be a script, copied as-is")
                return True

        except UnicodeDecodeError:
            # Binary file, can't modify
            self.log_callback(f"Warning: {exe_path} is binary, copied as-is")
            return True
        except Exception as e:
            self.log_callback(f"Error modifying Unix script {exe_path}: {e}")
            return False

    def _copy_icons(self) -> bool:
        """
        Copy appropriate icon format to install directory.
        Icons are pre-converted by the build process.
        
        :return: True if successful, False otherwise
        """
        try:
            # Determine the icon filename based on platform
            if self.system == 'windows':
                icon_filename = 'icon.ico'
            else:  # Linux and macOS
                icon_filename = 'icon.png'

            # Find icon file in installer resources
            icon_source = None

            # Method 1: Look in the installer package resources
            try:
                with resources.path('network_installer.resources', icon_filename) as resource_path:
                    if resource_path.exists():
                        icon_source = resource_path
            except (ImportError, FileNotFoundError):
                pass

            # Method 2: Look in PyInstaller bundle
            if not icon_source:
                try:
                    if hasattr(sys, '_MEIPASS'):
                        # PyInstaller bundle
                        bundle_path = Path(sys._MEIPASS) / 'network_installer' / 'resources' / icon_filename
                        if bundle_path.exists():
                            icon_source = bundle_path
                except Exception:
                    pass

            # Method 3: Look in development resources directory
            if not icon_source:
                try:
                    # Development mode - look for resources directory
                    current_dir = Path(__file__).parent.parent
                    dev_path = current_dir / 'resources' / icon_filename
                    if dev_path.exists():
                        icon_source = dev_path
                except Exception:
                    pass

            if not icon_source:
                self.log_callback(f"Warning: Could not find {icon_filename} in any expected location")
                return False

            # Copy icon file to install directory with appropriate name
            if self.system == 'windows':
                target_path = self.install_base / 'icon.ico'
            else:
                target_path = self.install_base / 'icon.png'

            shutil.copy2(icon_source, target_path)
            self.log_callback(f"✓ Copied {icon_filename} to {target_path}")
            return True

        except Exception as e:
            self.log_callback(f"Error copying icons: {e}")
            return False

    def _copy_file_association_icon(self) -> bool:
        """
        Copy appropriate config icon format to install directory for file associations.
        Icons are pre-converted by the build process.
        
        :return: True if successful, False otherwise
        """
        try:
            # Determine the config icon filename based on platform
            if self.system == 'windows':
                config_icon_filename = 'config_icon.ico'
            else:  # Linux and macOS
                config_icon_filename = 'config_icon.png'

            # Find config icon file in installer resources
            config_icon_source = None

            # Method 1: Look in the installer package resources
            try:
                with resources.path('network_installer.resources', config_icon_filename) as resource_path:
                    if resource_path.exists():
                        config_icon_source = resource_path
            except (ImportError, FileNotFoundError):
                pass

            # Method 2: Look in PyInstaller bundle
            if not config_icon_source:
                try:
                    if hasattr(sys, '_MEIPASS'):
                        # PyInstaller bundle
                        bundle_path = Path(sys._MEIPASS) / 'network_installer' / 'resources' / config_icon_filename
                        if bundle_path.exists():
                            config_icon_source = bundle_path
                except Exception:
                    pass

            # Method 3: Look in development resources directory
            if not config_icon_source:
                try:
                    # Development mode - look for resources directory
                    current_dir = Path(__file__).parent.parent
                    dev_path = current_dir / 'resources' / config_icon_filename
                    if dev_path.exists():
                        config_icon_source = dev_path
                except Exception:
                    pass

            if not config_icon_source:
                self.log_callback(f"Warning: Could not find {config_icon_filename} for file associations")
                return False

            # Copy config icon file to install directory with appropriate name
            if self.system == 'windows':
                target_path = self.install_base / 'config_icon.ico'
            else:
                target_path = self.install_base / 'config_icon.png'

            shutil.copy2(config_icon_source, target_path)
            self.log_callback(f"✓ Copied {config_icon_filename} to {target_path}")
            return True

        except Exception as e:
            self.log_callback(f"Error copying file association icons: {e}")
            return False

    def get_venv_scripts_dir(self) -> Path:
        """Get the scripts directory in the virtual environment."""
        if self.system == 'windows':
            return self.venv_dir / 'Scripts'
        else:
            return self.venv_dir / 'bin'

    @abstractmethod
    def create_stub_scripts(self) -> bool:
        """
        Create platform-specific stub scripts in the bin directory.
        Each platform must implement this with proper argument forwarding.
        
        :return: True if successful, False otherwise
        """
        pass

    def detect_existing_installation(self) -> 'ExistingInstallation':
        """
        Detect if dgenerate is already installed.
        
        :return: ExistingInstallation object with detection results
        """
        from network_installer.common_types import ExistingInstallation, InstallationInfo
        
        try:
            # Check if our installation directory exists
            if not self.install_base.exists():
                return ExistingInstallation(exists=False, installer_type='uv')

            # Check if virtual environment exists
            if not self.venv_dir.exists():
                return ExistingInstallation(
                    exists=True,
                    installer_type='uv',
                    path=str(self.install_base),
                    version='Incomplete (no virtual environment)'
                )

            # Check if dgenerate executable exists
            dgenerate_exe = self.scripts_dir / ('dgenerate.exe' if self.system == 'windows' else 'dgenerate')
            if not dgenerate_exe.exists():
                return ExistingInstallation(
                    exists=True,
                    installer_type='uv',
                    path=str(self.install_base),
                    version='Incomplete (no dgenerate executable)'
                )

            # Fast check - just verify the executable exists and is executable
            if not dgenerate_exe.is_file():
                return ExistingInstallation(
                    exists=True,
                    installer_type='uv',
                    path=str(self.install_base),
                    version='Broken (invalid executable)'
                )

            # Complete installation found - create installation info
            installation_info = InstallationInfo(
                install_base=str(self.install_base),
                venv_dir=str(self.venv_dir),
                scripts_dir=str(self.scripts_dir),
                dgenerate_exe=str(dgenerate_exe),
                installer_type='uv'
            )

            return ExistingInstallation(
                exists=True,
                installer_type='uv',
                path=str(self.install_base),
                version='Unknown (fast check)',
                installation_info=installation_info
            )

        except Exception as e:
            self.log_callback(f"Error detecting existing installation: {e}")
            return ExistingInstallation(exists=False, installer_type='uv')

    def cleanup_existing_installations(self) -> bool:
        """
        Clean up existing dgenerate installations.
        
        :return: True if successful, False otherwise
        """
        try:
            # Check if dgenerate is running
            if self._is_dgenerate_running():
                self.log_callback("Warning: dgenerate is currently running. Please close it and try again.")
                return False

            # Remove installation directory
            if self.install_base.exists():
                shutil.rmtree(self.install_base)
                self.log_callback(f"Removed existing installation: {self.install_base}")

            return True

        except Exception as e:
            self.log_callback(f"Error cleaning up existing installation: {e}")
            return False

    def _is_dgenerate_running(self) -> bool:
        """Check if dgenerate is currently running."""
        try:
            if self.system == 'windows':
                result = run_silent(['tasklist', '/FI', 'IMAGENAME eq dgenerate.exe'],
                                    capture_output=True, text=True, timeout=10)
            else:
                result = run_silent(['pgrep', '-f', 'dgenerate'],
                                    capture_output=True, text=True, timeout=10)

            return result.returncode == 0 and 'dgenerate' in result.stdout

        except Exception:
            return False

    def uninstall(self) -> bool:
        """
        Completely uninstall dgenerate.
        
        :return: True if successful, False otherwise
        """
        try:
            # Remove from PATH
            self._remove_from_path()

            # Remove desktop shortcuts
            self._remove_desktop_shortcuts()

            # Remove file associations
            self._remove_file_associations()

            # Remove installation directory
            if self.install_base.exists():
                if not self._remove_install_directory():
                    self.log_callback("Uninstallation failed: Could not remove installation directory")
                    return False
            else:
                self.log_callback(f"Installation directory does not exist: {self.install_base}")

            # Clean up environment variables
            self.cleanup_environment()

            # Remove uv that was installed by this installer
            self._remove_installed_uv()

            return True

        except Exception as e:
            self.log_callback(f"Error during uninstallation: {e}")
            return False

    def _remove_installed_uv(self):
        """
        Remove the uv executable that was installed by this installer.
        Only removes uv if it was installed in our uv_dir.
        """
        try:
            if self.uv_dir.exists():
                # Check if uv is in our installation directory
                uv_files = list(self.uv_dir.rglob('uv*'))
                if uv_files:
                    self.log_callback(f"Removing uv installation: {self.uv_dir}")
                    shutil.rmtree(self.uv_dir)
                    self.log_callback("✓ Removed uv installation")
                else:
                    self.log_callback("No uv installation found to remove")
            else:
                self.log_callback("No uv installation directory found")

        except Exception as e:
            self.log_callback(f"Warning: Could not remove uv installation: {e}")

    def _remove_install_directory(self) -> bool:
        """
        Remove the installation directory with robust error handling.
        
        :return: True if successful, False otherwise
        """
        try:
            self.log_callback(f"Removing installation directory: {self.install_base}")

            # First, try to remove any read-only attributes that might prevent deletion
            try:
                self._remove_readonly_attributes(self.install_base)
            except Exception as e:
                self.log_callback(f"Warning: Could not remove read-only attributes: {e}")

            # Try to remove the directory multiple times with delays
            for attempt in range(3):
                try:
                    if self.install_base.exists():
                        shutil.rmtree(self.install_base)
                        self.log_callback(f"✓ Removed installation directory: {self.install_base}")
                        return True
                    else:
                        self.log_callback(f"Installation directory already removed: {self.install_base}")
                        return True

                except PermissionError as e:
                    self.log_callback(f"Attempt {attempt + 1}: Permission denied removing directory: {e}")
                    if attempt < 2:  # Not the last attempt
                        time.sleep(1)  # Wait a bit and try again
                        continue
                    else:
                        self.log_callback("Failed to remove directory due to permission errors")
                        self.log_callback("This may be due to files being in use or insufficient permissions")
                        return False

                except OSError as e:
                    self.log_callback(f"Attempt {attempt + 1}: OS error removing directory: {e}")
                    if attempt < 2:  # Not the last attempt
                        time.sleep(1)  # Wait a bit and try again
                        continue
                    else:
                        self.log_callback("Failed to remove directory due to OS errors")
                        self.log_callback("Some files may still be in use")
                        return False

            return False

        except Exception as e:
            self.log_callback(f"Error removing installation directory: {e}")
            return False

    def _remove_readonly_attributes(self, path: Path) -> None:
        """
        Recursively remove read-only attributes from files and directories.
        
        :param path: Path to process
        """
        try:
            if not path.exists():
                return

            # Remove read-only attribute from the current path
            if path.is_file():
                path.chmod(stat.S_IWRITE | stat.S_IREAD)
            elif path.is_dir():
                path.chmod(stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)

                # Recursively process directory contents
                try:
                    for item in path.iterdir():
                        self._remove_readonly_attributes(item)
                except PermissionError:
                    # Skip if we can't read the directory
                    pass

        except Exception:
            # Ignore errors in this helper function
            pass

    def cleanup_environment(self) -> bool:
        """
        Clean up environment variables that might have been set during installation.
        
        :return: True if successful, False otherwise
        """
        try:
            # Clear virtual environment variables
            env_vars_to_clear = [
                'VIRTUAL_ENV',
                'VIRTUAL_ENV_PROMPT',
                '_OLD_VIRTUAL_PROMPT',
                'VSCODE_ENV_REPLACE',
                '_OLD_VIRTUAL_PATH',
                '_OLD_VIRTUAL_PYTHONHOME'
            ]

            for var in env_vars_to_clear:
                if var in os.environ:
                    del os.environ[var]

            self.log_callback("Environment variables cleaned up")
            return True

        except Exception as e:
            self.log_callback(f"Error cleaning up environment: {e}")
            return False

    def get_installation_info(self) -> 'InstallationInfo':
        """
        Get information about the current installation.
        
        :return: InstallationInfo object containing installation information
        """
        from network_installer.common_types import InstallationInfo
        
        return InstallationInfo(
            install_base=str(self.install_base),
            venv_dir=str(self.venv_dir),
            scripts_dir=str(self.scripts_dir),
            dgenerate_exe=str(self.scripts_dir / ('dgenerate.exe' if self.system == 'windows' else 'dgenerate')),
            installer_type='uv'
        )

    # Abstract methods that must be implemented by platform-specific handlers
    @abstractmethod
    def add_scripts_to_path(self) -> bool:
        """Add scripts directory to system PATH."""
        pass

    @abstractmethod
    def create_desktop_shortcut(self) -> bool:
        """Create desktop shortcut for dgenerate."""
        pass

    @abstractmethod
    def create_file_associations(self) -> bool:
        """Create file associations for .dgen files."""
        pass

    @abstractmethod
    def _remove_from_path(self):
        """Remove scripts directory from system PATH."""
        pass

    @abstractmethod
    def _remove_desktop_shortcuts(self):
        """Remove desktop shortcuts."""
        pass

    @abstractmethod
    def _remove_file_associations(self):
        """Remove file associations."""
        pass

    def apply_source_patches(self, source_dir: str) -> bool:
        """
        Apply platform-specific patches to the source code before installation.
        
        :param source_dir: Path to the dgenerate source directory
        :return: True if successful or no patches needed, False if failed
        """
        # Default implementation: no patches needed
        return True
