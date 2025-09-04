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
Linux-specific platform handler for the dgenerate installer.
"""

import inspect
import os
import subprocess
import shutil
import tkinter as tk
from pathlib import Path
from typing import Optional, List, Tuple

from .base_uv_handler import BasePlatformHandler


class LinuxPlatformHandler(BasePlatformHandler):
    """
    Linux-specific implementation of platform operations.
    """

    def _detect_package_manager(self) -> Tuple[str, str]:
        """
        Detect the package manager and return install command template.
        
        :return: Tuple of (package_manager_name, install_command_template)
        """
        # Check for package managers in order of preference
        package_managers = [
            ('apt', 'sudo apt install -y {packages}', ['apt', 'apt-get']),
            ('dnf', 'sudo dnf install -y {packages}', ['dnf']),
            ('yum', 'sudo yum install -y {packages}', ['yum']),
            ('zypper', 'sudo zypper install -y {packages}', ['zypper']),
            ('pacman', 'sudo pacman -S --needed {packages}', ['pacman']),
            ('apk', 'sudo apk add {packages}', ['apk']),
            ('emerge', 'sudo emerge {packages}', ['emerge']),
            ('pkg', 'sudo pkg install {packages}', ['pkg']),
        ]
        
        for pm_name, cmd_template, executables in package_managers:
            for executable in executables:
                try:
                    result = subprocess.run(['which', executable], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        return (pm_name, cmd_template)
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
        
        return ('unknown', 'sudo <package-manager> install tcl tk')

    def _get_tcl_tk_packages_for_distro(self, package_manager: str) -> str:
        """
        Get the appropriate Tcl/Tk package names for the detected package manager.
        
        :param package_manager: Detected package manager name
        :return: Space-separated package names
        """
        package_map = {
            'apt': 'libtcl8.6 libtk8.6',  # Debian/Ubuntu
            'dnf': 'tcl tk',               # Fedora/RHEL/CentOS/Rocky/Alma
            'yum': 'tcl tk',               # Older RHEL/CentOS
            'zypper': 'tcl tk',            # openSUSE/SLES
            'pacman': 'tcl tk',            # Arch/Manjaro
            'apk': 'tcl tk',               # Alpine Linux
            'emerge': 'dev-lang/tcl dev-lang/tk',  # Gentoo
            'pkg': 'tcl86 tk86',           # FreeBSD/OpenBSD/NetBSD
        }
        
        return package_map.get(package_manager, 'tcl tk')

    def _check_tcl_tk_installed(self) -> bool:
        """
        Check if Tcl/Tk runtime libraries are installed on the system.
        
        :return: True if Tcl/Tk libraries are found, False otherwise
        """
        system_lib_paths = [
            Path('/usr/lib/x86_64-linux-gnu'),  # Debian/Ubuntu x86_64
            Path('/usr/lib/aarch64-linux-gnu'),  # Debian/Ubuntu ARM64
            Path('/usr/lib64'),                  # RedHat/CentOS/Fedora
            Path('/usr/lib'),                    # Generic fallback
        ]
        
        # Look for any Tcl/Tk libraries
        for system_lib_path in system_lib_paths:
            if not system_lib_path.exists():
                continue
                
            tcl_libs = list(system_lib_path.rglob('libtcl*.so*'))
            tk_libs = list(system_lib_path.rglob('libtk*.so*'))
            
            if tcl_libs and tk_libs:
                self.log_callback(f"Found Tcl/Tk libraries in {system_lib_path}")
                return True
        
        return False

    def _prompt_for_tcl_tk_installation(self, is_gui: bool = False) -> bool:
        """
        Prompt user to install Tcl/Tk runtime libraries.
        
        :param is_gui: Whether this is being called from GUI or CLI
        :return: True to continue, False to abort
        """
        try:
            package_manager, cmd_template = self._detect_package_manager()
            packages = self._get_tcl_tk_packages_for_distro(package_manager)
            install_cmd = cmd_template.format(packages=packages)
            
            if package_manager == 'unknown':
                message = inspect.cleandoc(f"""
                    Tcl/Tk Runtime Libraries Not Found

                    For optimal tkinter font support, please install Tcl/Tk runtime libraries.

                    Could not detect your package manager. Please use the appropriate command for your distribution:

                    # Debian / Ubuntu / derivatives
                    sudo apt install -y libtcl8.6 libtk8.6

                    # Fedora / RHEL / CentOS / Rocky / Alma  
                    sudo dnf install -y tcl tk

                    # openSUSE / SLES
                    sudo zypper install -y tcl tk

                    # Arch / Manjaro
                    sudo pacman -S --needed tcl tk

                    # Alpine Linux
                    sudo apk add tcl tk

                    # Gentoo
                    sudo emerge dev-lang/tcl dev-lang/tk

                    # FreeBSD / OpenBSD / NetBSD
                    sudo pkg install tcl86 tk86

                    You can install these packages in another terminal and then continue.
                """)
            else:
                message = inspect.cleandoc(f"""
                    Tcl/Tk Runtime Libraries Not Found

                    For optimal tkinter font support, please install Tcl/Tk runtime libraries.

                    Detected system: {package_manager}
                    Install command:

                    {install_cmd}

                    You can run this command in another terminal and then continue the installation.
                """)

            if is_gui:
                # Show modal dialog for GUI
                return self._show_tcl_tk_dialog(message)
            else:
                # Show CLI prompt
                print("\n" + "="*60)
                print(message)
                print("="*60)
                
                while True:
                    response = input("\nContinue installation anyway? (y/N): ").strip().lower()
                    if response in ['y', 'yes']:
                        return True
                    elif response in ['n', 'no', '']:
                        return False
                    else:
                        print("Please enter 'y' for yes or 'n' for no.")
                        
        except Exception as e:
            self.log_callback(f"Error prompting for Tcl/Tk installation: {e}")
            return True  # Continue on error

    def _show_tcl_tk_dialog(self, message: str) -> bool:
        """
        Show a modal dialog for Tcl/Tk installation prompt.
        
        :param message: Message to display
        :return: True to continue, False to abort
        """
        try:
            # Create a modal dialog
            dialog = tk.Toplevel()
            dialog.title("Tcl/Tk Runtime Libraries Required")
            dialog.resizable(False, False)
            dialog.grab_set()  # Make it modal
            
            # Center the dialog
            dialog.geometry("600x500")
            
            # Add the message
            text_widget = tk.Text(dialog, wrap=tk.WORD, padx=10, pady=10, 
                                 height=25, width=70, state=tk.DISABLED)
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Enable text widget to insert content
            text_widget.config(state=tk.NORMAL)
            text_widget.insert(tk.END, message)
            text_widget.config(state=tk.DISABLED)
            
            # Add buttons
            button_frame = tk.Frame(dialog)
            button_frame.pack(pady=10)
            
            result = {'continue': False}
            
            def on_continue():
                result['continue'] = True
                dialog.destroy()
                
            def on_cancel():
                result['continue'] = False
                dialog.destroy()
            
            tk.Button(button_frame, text="Continue Anyway", 
                     command=on_continue, width=15).pack(side=tk.LEFT, padx=5)
            tk.Button(button_frame, text="Cancel Installation", 
                     command=on_cancel, width=15).pack(side=tk.LEFT, padx=5)
            
            # Wait for user response
            dialog.wait_window()
            
            return result['continue']
            
        except Exception as e:
            self.log_callback(f"Error showing Tcl/Tk dialog: {e}")
            return True  # Continue on error

    def _symlink_system_tcl_tk_libraries(self) -> bool:
        """
        Symlink system Tcl/Tk libraries to replace UV's isolated libraries.
        This provides better font support by using system-integrated Tcl/Tk.
        
        :return: True if successful, False otherwise
        """
        try:
            # UV installs Python in ~/.local/share/uv/python/cpython-VERSION-linux-ARCH-gnu/lib/
            # We need to find the actual UV Python installation directory
            uv_python_base = Path.home() / '.local' / 'share' / 'uv' / 'python'
            
            if not uv_python_base.exists():
                self.log_callback("UV Python base directory not found, skipping Tcl/Tk symlinking")
                return True
            
            # Find the cpython directory (e.g., cpython-3.13.7-linux-x86_64-gnu)
            cpython_dirs = list(uv_python_base.rglob('cpython-*-linux-*-gnu'))
            
            if not cpython_dirs:
                self.log_callback("No UV Python installations found, skipping Tcl/Tk symlinking")
                return True
            
            # Use the most recent cpython installation
            cpython_dir = max(cpython_dirs, key=lambda p: p.stat().st_mtime)
            uv_lib_dir = cpython_dir / 'lib'
            
            self.log_callback(f"Found UV Python installation: {cpython_dir}")
            self.log_callback(f"UV library directory: {uv_lib_dir}")

            if not uv_lib_dir.exists():
                self.log_callback("UV Python lib directory not found")
                return True

            # Common system library locations to check
            system_lib_paths = [
                Path('/usr/lib/x86_64-linux-gnu'),  # Debian/Ubuntu x86_64
                Path('/usr/lib/aarch64-linux-gnu'), # Debian/Ubuntu ARM64
                Path('/usr/lib64'),                 # RedHat/CentOS/Fedora
                Path('/usr/lib'),                   # Generic fallback
            ]

            symlinks_created = False

            # Find all Tcl/Tk libraries using pattern matching
            self.log_callback("Searching for Tcl/Tk libraries to symlink...")
            
            # Find all UV Tcl/Tk libraries using recursive glob patterns
            uv_tcl_libs = list(uv_lib_dir.rglob('libtcl*.so*'))
            uv_tk_libs = list(uv_lib_dir.rglob('libtk*.so*'))
            uv_libs = uv_tcl_libs + uv_tk_libs
            
            if not uv_libs:
                self.log_callback("No Tcl/Tk libraries found in UV Python installation")
                return True
                
            self.log_callback(f"Found {len(uv_libs)} UV Tcl/Tk libraries to potentially replace")
            
            for uv_lib_file in uv_libs:
                lib_name = uv_lib_file.name
                
                # Skip backup files from previous runs
                if '.uv_backup' in lib_name:
                    continue
                    
                self.log_callback(f"Processing UV library: {lib_name}")
                
                # Find corresponding system library
                system_lib_found = None
                for system_lib_path in system_lib_paths:
                    if not system_lib_path.exists():
                        continue
                        
                    # Try exact name match first
                    system_lib_file = system_lib_path / lib_name
                    if system_lib_file.exists():
                        system_lib_found = system_lib_file
                        break
                    
                    # Try pattern matching for different versions
                    # Extract base name (e.g., libtcl8.6.so -> libtcl, libtk8.7.so.1 -> libtk)
                    if lib_name.startswith('libtcl'):
                        pattern = 'libtcl*.so*'
                    elif lib_name.startswith('libtk'):
                        pattern = 'libtk*.so*'
                    else:
                        continue
                        
                    # Find any matching system library recursively
                    matching_system_libs = list(system_lib_path.rglob(pattern))
                    if matching_system_libs:
                        # Use the first match (they should be compatible)
                        system_lib_found = matching_system_libs[0]
                        self.log_callback(f"  Found compatible system library: {system_lib_found}")
                        break
                
                if system_lib_found:
                    try:
                        # Backup the original UV library
                        backup_path = uv_lib_file.with_suffix(uv_lib_file.suffix + '.uv_backup')
                        if not backup_path.exists():
                            shutil.move(str(uv_lib_file), str(backup_path))
                            self.log_callback(f"✓ Backed up UV library: {lib_name}")

                        # Create symlink to system library
                        if uv_lib_file.exists() or uv_lib_file.is_symlink():
                            # Remove existing file/symlink first
                            uv_lib_file.unlink()
                        
                        uv_lib_file.symlink_to(system_lib_found)
                        self.log_callback(f"✓ Symlinked {lib_name} -> {system_lib_found}")
                        symlinks_created = True
                        
                    except OSError as e:
                        self.log_callback(f"Warning: Could not symlink {lib_name}: {e}")
                else:
                    self.log_callback(f"  No compatible system library found for {lib_name}")

            if symlinks_created:
                self.log_callback("✓ System Tcl/Tk libraries symlinked for better font support")
            else:
                self.log_callback("Note: No Tcl/Tk library pairs found for symlinking")

            return True

        except Exception as e:
            self.log_callback(f"Error symlinking Tcl/Tk libraries: {e}")
            return False

    def _find_system_tcl_tk_libraries(self) -> List[tuple[Path, Path]]:
        """
        Find system Tcl/Tk libraries that can be symlinked.
        
        :return: List of (system_lib_path, venv_lib_path) tuples
        """
        system_lib_paths = [
            Path('/usr/lib/x86_64-linux-gnu'),  # Debian/Ubuntu x86_64
            Path('/usr/lib/aarch64-linux-gnu'),  # Debian/Ubuntu ARM64  
            Path('/usr/lib64'),                  # RedHat/CentOS/Fedora
            Path('/usr/lib'),                    # Generic fallback
        ]

        library_pairs = []
        venv_lib_dir = self.venv_dir / 'lib'

        for system_lib_path in system_lib_paths:
            if not system_lib_path.exists():
                continue

            # Check for Tcl/Tk libraries
            tcl_tk_files = [
                'libtcl8.6.so', 'libtk8.6.so',
                'libtcl8.7.so', 'libtk8.7.so', 
                'libtcl8.8.so', 'libtk8.8.so',
            ]

            for lib_file in tcl_tk_files:
                system_lib = system_lib_path / lib_file
                venv_lib = venv_lib_dir / lib_file

                if system_lib.exists() and venv_lib.exists():
                    library_pairs.append((system_lib, venv_lib))

        return library_pairs

    def apply_source_patches(self, source_dir: str) -> bool:
        """
        Apply Linux-specific patches including Tcl/Tk library symlinking for better font support.
        
        :param source_dir: Path to the dgenerate source directory
        :return: True if successful, False if failed
        """
        try:
            # Check if Tcl/Tk runtime libraries are installed
            self.log_callback("Checking for system Tcl/Tk runtime libraries...")
            if not self._check_tcl_tk_installed():
                self.log_callback("Tcl/Tk runtime libraries not found on system")
                
                # Determine if this is a GUI installation by checking if we have a tkinter root
                is_gui = False
                try:
                    is_gui = tk._default_root is not None
                except:
                    pass
                
                # Prompt user to install Tcl/Tk libraries
                if not self._prompt_for_tcl_tk_installation(is_gui=is_gui):
                    self.log_callback("User cancelled installation due to missing Tcl/Tk libraries")
                    return False
                    
                self.log_callback("Continuing installation without Tcl/Tk symlinking...")
                return True
            
            # Apply Tcl/Tk library symlinking for better font support
            self.log_callback("Setting up system Tcl/Tk integration for better font support...")
            if not self._symlink_system_tcl_tk_libraries():
                self.log_callback("Warning: Failed to symlink system Tcl/Tk libraries")
                # Don't fail the installation for this - it's an enhancement
            
            return True
            
        except Exception as e:
            self.log_callback(f"Error applying Linux patches: {e}")
            return False

    def add_scripts_to_path(self) -> bool:
        """
        Add only the bin directory (with stub scripts) to the system PATH.
        This avoids exposing the entire venv and potential Python conflicts.
        
        :return: True if successful, False otherwise
        """
        try:
            # First create stub scripts in the bin directory
            if not self.create_stub_scripts():
                self.log_callback("Failed to create stub scripts")
                return False

            bin_dir_str = str(self.bin_dir)
            self.log_callback(f"Adding {bin_dir_str} to system PATH...")

            # Get the shell profile file
            shell_profile = self._get_shell_profile()
            if not shell_profile:
                self.log_callback("Could not determine shell profile file")
                return False

            # Check if already in PATH
            current_path = os.environ.get('PATH', '')
            if bin_dir_str in current_path:
                self.log_callback(f"✓ Directory {bin_dir_str} already in PATH")
                return True

            # Add to shell profile with comment for easy identification
            comment_line = '# Added by dgenerate network installer'
            export_line = f'export PATH="{bin_dir_str}:$PATH"'

            with open(shell_profile, 'a', encoding='utf-8') as f:
                f.write(f'\n{comment_line}\n{export_line}\n')

            self.log_callback(f"✓ Added {bin_dir_str} to {shell_profile}")
            self.log_callback("  Please restart your terminal or run 'source ~/.bashrc' to use dgenerate")

            return True

        except Exception as e:
            self.log_callback(f"Error modifying Linux PATH: {e}")
            return False

    def create_stub_scripts(self) -> bool:
        """
        Create Linux-specific stub scripts.
        Note: dgenerate and dgenerate_windowed are copied directly, not stubbed.
        """
        try:
            # Create bin directory
            self.bin_dir.mkdir(parents=True, exist_ok=True)

            # No stub scripts needed - dgenerate and dgenerate_windowed 
            # are copied directly and modified to handle environment setup
            self.log_callback("✓ Linux bin directory created (no stub scripts needed)")
            return True

        except Exception as e:
            self.log_callback(f"Error creating Linux bin directory: {e}")
            return False

    def create_desktop_shortcut(self) -> bool:
        """
        Create desktop shortcut for dgenerate using the windowed stub.
        
        :return: True if successful, False otherwise
        """
        try:
            # Get desktop path
            desktop = Path.home() / 'Desktop'
            
            # Check if Desktop directory exists (may not exist in Docker containers)
            if not desktop.exists():
                self.log_callback("Desktop directory not found, skipping desktop shortcut creation")
                return True

            # Get windowed executable path from bin directory
            windowed_exe = self.bin_dir / "dgenerate_windowed"
            if not windowed_exe.exists():
                self.log_callback("dgenerate_windowed executable not found in bin directory, cannot create shortcut")
                return False

            # Get icon path
            icon_path = self.install_base / "icon.png"
            if not icon_path.exists():
                self.log_callback("icon.png not found, using default icon")
                icon_line = ""
            else:
                icon_line = f"Icon={icon_path}"

            # Create .desktop file
            shortcut_path = desktop / "Dgenerate Console.desktop"

            # Use the stub script from bin directory instead of the venv executable
            windowed_stub = windowed_exe

            # Create the desktop entry content
            desktop_entry = inspect.cleandoc(f"""
                [Desktop Entry]
                Version=1.0
                Type=Application
                Name=Dgenerate Console
                Comment=Launch Dgenerate Console
                Exec={windowed_stub} --console
                Path={Path.home()}
                Terminal=false
                Categories=Development;
                {icon_line}
            """)

            # Write the desktop entry file
            with open(shortcut_path, 'w', encoding='utf-8') as f:
                f.write(desktop_entry)

            # Make it executable
            shortcut_path.chmod(0o755)

            self.log_callback(f"✓ Created Linux desktop shortcut: {shortcut_path}")
            return True

        except Exception as e:
            self.log_callback(f"Error creating Linux desktop shortcut: {e}")
            return False

    def create_file_associations(self) -> bool:
        """
        Create file associations for .dgen files.
        
        :return: True if successful, False otherwise
        """
        try:
            # Get windowed executable path for console UI
            windowed_exe = self.bin_dir / "dgenerate_windowed"
            if not windowed_exe.exists():
                self.log_callback(
                    "dgenerate_windowed executable not found in bin directory, cannot create file associations")
                return False

            self.log_callback("Creating Linux file associations for .dgen files...")

            # Create MIME type association
            mime_dir = Path.home() / '.local' / 'share' / 'mime' / 'packages'
            mime_dir.mkdir(parents=True, exist_ok=True)

            mime_file = mime_dir / 'dgenerate.xml'

            mime_content = inspect.cleandoc(f"""
                <?xml version="1.0" encoding="UTF-8"?>
                <mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
                  <mime-type type="application/x-dgenerate-config">
                    <comment>dgenerate Configuration File</comment>
                    <glob pattern="*.dgen"/>
                  </mime-type>
                </mime-info>
            """)

            with open(mime_file, 'w', encoding='utf-8') as f:
                f.write(mime_content)

            # Create desktop entry for the application
            apps_dir = Path.home() / '.local' / 'share' / 'applications'
            apps_dir.mkdir(parents=True, exist_ok=True)

            app_file = apps_dir / 'dgenerate.desktop'

            # Check for config icon
            config_icon_path = self.install_base / 'config_icon.png'
            icon_line = f"Icon={config_icon_path}\n" if config_icon_path.exists() else ""

            app_content = inspect.cleandoc(f"""
                [Desktop Entry]
                Version=1.0
                Type=Application
                Name=dgenerate
                Comment=dgenerate Configuration Editor
                Exec={windowed_exe} --console %f
                {icon_line}Terminal=false
                Categories=Development;
                MimeType=application/x-dgenerate-config;
            """)

            with open(app_file, 'w', encoding='utf-8') as f:
                f.write(app_content)

            # Update MIME database
            try:
                subprocess.run(['update-mime-database', str(Path.home() / '.local' / 'share' / 'mime')],
                               capture_output=True, text=True, timeout=30)
                self.log_callback("Updated MIME database")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.log_callback("Warning: Could not update MIME database (update-mime-database not found)")

            # Update desktop database
            try:
                subprocess.run(['update-desktop-database', str(Path.home() / '.local' / 'share' / 'applications')],
                               capture_output=True, text=True, timeout=30)
                self.log_callback("Updated desktop database")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.log_callback("Warning: Could not update desktop database (update-desktop-database not found)")

            self.log_callback("✓ Linux file associations created successfully")
            return True

        except Exception as e:
            self.log_callback(f"Error creating Linux file associations: {e}")
            return False

    def _remove_from_path(self):
        """Remove from Linux PATH by modifying shell profile."""
        try:
            bin_dir_str = str(self.bin_dir)

            # Common shell profile files
            profile_files = [
                Path.home() / '.bashrc',
                Path.home() / '.zshrc',
                Path.home() / '.profile',
            ]

            comment_line = '# Added by dgenerate network installer'
            export_line = f'export PATH="{bin_dir_str}:$PATH"'

            for profile_file in profile_files:
                if profile_file.exists():
                    try:
                        # Read current content
                        content = profile_file.read_text()

                        # Remove our lines
                        lines = content.split('\n')
                        new_lines = []
                        skip_next = False

                        for line in lines:
                            if line.strip() == comment_line:
                                skip_next = True
                                continue
                            elif skip_next and line.strip() == export_line:
                                skip_next = False
                                continue
                            elif line.strip() == export_line:
                                continue
                            else:
                                new_lines.append(line)
                                skip_next = False

                        # Write back the content
                        profile_file.write_text('\n'.join(new_lines))
                        self.log_callback(f"Removed PATH export from {profile_file}")

                    except Exception as e:
                        self.log_callback(f"Warning: Could not modify {profile_file}: {e}")

        except Exception as e:
            self.log_callback(f"Error removing from Linux PATH: {e}")

    def _remove_desktop_shortcuts(self):
        """Remove desktop shortcuts."""
        try:
            # Get desktop path
            desktop = Path.home() / 'Desktop'

            # Remove .desktop file
            shortcut_path = desktop / "Dgenerate Console.desktop"
            if shortcut_path.exists():
                shortcut_path.unlink()
                self.log_callback(f"Removed Linux desktop shortcut: {shortcut_path}")

            # Also check for old shortcut name
            old_shortcut_path = desktop / "dgenerate.desktop"
            if old_shortcut_path.exists():
                old_shortcut_path.unlink()
                self.log_callback(f"Removed old Linux desktop shortcut: {old_shortcut_path}")

            # Remove windowed executable on Unix systems
            windowed_exe = self.scripts_dir / "dgenerate_windowed"
            if windowed_exe.exists():
                try:
                    windowed_exe.unlink()
                    self.log_callback(f"Removed windowed executable: {windowed_exe}")
                except Exception as e:
                    self.log_callback(f"Could not remove windowed executable {windowed_exe}: {e}")

        except Exception as e:
            self.log_callback(f"Error removing Linux desktop shortcuts: {e}")

    def _remove_file_associations(self):
        """Remove Linux file associations."""
        try:
            self.log_callback("Removing Linux file associations...")

            # Remove MIME type file
            mime_file = Path.home() / '.local' / 'share' / 'mime' / 'packages' / 'dgenerate.xml'
            if mime_file.exists():
                mime_file.unlink()
                self.log_callback("Removed MIME type file")

            # Remove desktop entry
            app_file = Path.home() / '.local' / 'share' / 'applications' / 'dgenerate.desktop'
            if app_file.exists():
                app_file.unlink()
                self.log_callback("Removed desktop entry")

            # Update MIME database
            try:
                subprocess.run(['update-mime-database', str(Path.home() / '.local' / 'share' / 'mime')],
                               capture_output=True, text=True, timeout=30)
                self.log_callback("Updated MIME database")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.log_callback("Warning: Could not update MIME database")

            # Update desktop database
            try:
                subprocess.run(['update-desktop-database', str(Path.home() / '.local' / 'share' / 'applications')],
                               capture_output=True, text=True, timeout=30)
                self.log_callback("Updated desktop database")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.log_callback("Warning: Could not update desktop database")

            self.log_callback("✓ Linux file associations removed successfully")

        except Exception as e:
            self.log_callback(f"Error removing Linux file associations: {e}")

    def _get_shell_profile(self) -> Optional[Path]:
        """Get the appropriate shell profile file for Linux."""
        home = Path.home()

        # Check for different shell profiles
        profiles = [
            home / '.bashrc',  # Most common on Linux
            home / '.bash_profile',
            home / '.zshrc',
            home / '.profile'
        ]

        for profile in profiles:
            if profile.exists():
                return profile

        # Default to .bashrc if none exist (most common on Linux)
        return home / '.bashrc'

    def _verify_path_update(self):
        """Verify that the PATH was actually updated."""
        try:
            # For Linux, check if the export line exists in the shell profile
            shell_profile = self._get_shell_profile()
            if shell_profile and shell_profile.exists():
                content = shell_profile.read_text()
                bin_dir_str = str(self.bin_dir)
                export_line = f'export PATH="{bin_dir_str}:$PATH"'

                if export_line in content:
                    self.log_callback("✓ PATH verification successful - dgenerate is now available globally")
                    self.log_callback("  Note: You may need to restart your terminal or run 'source ~/.bashrc'")
                else:
                    self.log_callback("Warning: PATH verification failed - dgenerate may not be available globally")
            else:
                self.log_callback("Warning: Could not verify PATH update")

        except Exception as e:
            self.log_callback(f"Warning: Could not verify PATH update: {e}")
