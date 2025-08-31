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
from pathlib import Path
from typing import Optional

from .base_uv_installer import BasePlatformHandler


class LinuxPlatformHandler(BasePlatformHandler):
    """
    Linux-specific implementation of platform operations.
    """

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
