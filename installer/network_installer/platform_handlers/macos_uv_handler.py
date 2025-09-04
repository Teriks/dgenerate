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
macOS-specific platform handler for the dgenerate installer.
"""

import inspect
import inspect
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from .base_uv_handler import BasePlatformHandler


class MacOSPlatformHandler(BasePlatformHandler):
    """
    macOS-specific implementation of platform operations.
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
            self.log_callback(f"Error modifying macOS PATH: {e}")
            return False

    def create_stub_scripts(self) -> bool:
        """
        Create macOS-specific stub scripts.
        Note: dgenerate and dgenerate_windowed are copied directly, not stubbed.
        """
        try:
            # Create bin directory
            self.bin_dir.mkdir(parents=True, exist_ok=True)

            # No stub scripts needed - dgenerate and dgenerate_windowed 
            # are copied directly and modified to handle environment setup
            self.log_callback("✓ macOS bin directory created (no stub scripts needed)")
            return True

        except Exception as e:
            self.log_callback(f"Error creating macOS bin directory: {e}")
            return False

    def create_desktop_shortcut(self) -> bool:
        """
        Create desktop shortcut for dgenerate using AppleScript app.
        
        :return: True if successful, False otherwise
        """
        try:
            # Get desktop path
            desktop = Path.home() / 'Desktop'

            # Get dgenerate executable path from bin directory
            dgenerate_exe = self.bin_dir / "dgenerate"
            if not dgenerate_exe.exists():
                self.log_callback("dgenerate executable not found in bin directory, cannot create shortcut")
                return False

            # Get the uv managed Python directory for Tcl/Tk libraries
            uv_python_base = Path.home() / '.local' / 'share' / 'uv' / 'python'

            applescript_content = inspect.cleandoc(f'''
                try
                    set venvBin to "{self.venv_dir}/bin"
                    set envPath to venvBin & ":{self.bin_dir}:$PATH"
                    set homeDir to "{Path.home()}"
                    set venvDir to "{self.venv_dir}"
                    set winExe to "{dgenerate_exe}"
                    set uvPythonBase to "{uv_python_base}"
                    
                    -- Set up proper environment including Tcl/Tk paths from uv managed Python
                    -- Use shell expansion to find the correct cpython directory
                    set envCommand to "cd \\"" & homeDir & "\\" && " & ¬
                        "PATH=\\"" & envPath & "\\" " & ¬
                        "VIRTUAL_ENV=\\"" & venvDir & "\\" " & ¬
                        "TCL_LIBRARY=$(ls -d \\"" & uvPythonBase & "\\"/cpython-*-macos-aarch64-none/lib/tcl* 2>/dev/null | head -1) " & ¬
                        "TK_LIBRARY=$(ls -d \\"" & uvPythonBase & "\\"/cpython-*-macos-aarch64-none/lib/tk* 2>/dev/null | head -1) " & ¬
                        "\\"" & winExe & "\\" --console"
                    
                    do shell script envCommand
                on error errMsg
                    display dialog "Error launching dgenerate: " & errMsg buttons {{"OK"}} default button "OK"
                end try
            ''')

            # Create temporary AppleScript file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.scpt', delete=False) as temp_script:
                temp_script.write(applescript_content)
                temp_script_path = temp_script.name

            try:
                # Use osacompile to create the app
                app_path = desktop / "Dgenerate Console.app"

                # Remove existing app if it exists
                if app_path.exists():
                    shutil.rmtree(app_path)

                # Compile AppleScript to app
                result = subprocess.run([
                    'osacompile', '-o', str(app_path), temp_script_path
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    # Add icon to the app bundle
                    self._add_icon_to_app_bundle(app_path)

                    # Modify Info.plist to hide the launcher app from dock
                    self._hide_app_from_dock(app_path)

                    self.log_callback(f"✓ Created macOS desktop shortcut: {app_path}")
                    return True
                else:
                    self.log_callback(f"Error compiling AppleScript: {result.stderr}")
                    # Fallback to .command file
                    return self._create_command_file_shortcut(desktop, dgenerate_exe)

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_script_path)
                except:
                    pass

        except Exception as e:
            self.log_callback(f"Error creating macOS desktop shortcut: {e}")
            # Fallback to .command file
            return self._create_command_file_shortcut(desktop, dgenerate_exe)

    def _create_command_file_shortcut(self, desktop: Path, dgenerate_exe: Path) -> bool:
        """Fallback method to create .command file shortcut."""
        try:
            # Create .command file (macOS equivalent of .bat)
            shortcut_path = desktop / "Dgenerate Console.command"

            # Create the command file content
            command_content = inspect.cleandoc(f'''
                #!/bin/bash
                cd "{Path.home()}"
                "{dgenerate_exe}" --console
            ''')

            # Write the command file
            with open(shortcut_path, 'w', encoding='utf-8') as f:
                f.write(command_content)

            # Make it executable
            shortcut_path.chmod(0o755)

            self.log_callback(f"✓ Created fallback macOS desktop shortcut: {shortcut_path}")
            return True

        except Exception as e:
            self.log_callback(f"Error creating fallback macOS desktop shortcut: {e}")
            return False

    def _add_icon_to_app_bundle(self, app_path: Path) -> bool:
        """
        Add icon to the AppleScript app bundle.
        
        :param app_path: Path to the app bundle
        :return: True if successful, False otherwise
        """
        try:
            # Find the icon file in resources
            icon_source = None

            # Method 1: Look in the installer package resources
            try:
                import importlib.resources as resources
                with resources.path('network_installer.resources', 'icon.png') as resource_path:
                    if resource_path.exists():
                        icon_source = resource_path
            except (ImportError, FileNotFoundError):
                pass

            # Method 2: Look in PyInstaller bundle
            if not icon_source:
                try:
                    if hasattr(sys, '_MEIPASS'):
                        # PyInstaller bundle
                        bundle_path = Path(sys._MEIPASS) / 'network_installer' / 'resources' / 'icon.png'
                        if bundle_path.exists():
                            icon_source = bundle_path
                except Exception:
                    pass

            # Method 3: Look in development resources directory
            if not icon_source:
                try:
                    dev_icon_path = Path(__file__).parent.parent.parent / 'resources' / 'icon.png'
                    if dev_icon_path.exists():
                        icon_source = dev_icon_path
                except Exception:
                    pass

            if not icon_source:
                self.log_callback("Warning: Could not find icon.png for app bundle")
                return False

            # Copy icon to app bundle resources
            resources_dir = app_path / 'Contents' / 'Resources'
            icon_dest = resources_dir / 'icon.png'

            import shutil
            shutil.copy2(icon_source, icon_dest)

            # Try multiple methods to set the icon
            icon_set = False

            # Method 1: Try fileicon command (if available)
            try:
                import subprocess
                result = subprocess.run([
                    'fileicon', 'set', str(app_path), str(icon_dest)
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    self.log_callback("✓ Added icon to app bundle using fileicon")
                    icon_set = True
            except FileNotFoundError:
                pass

            # Method 2: Set icon directly in Info.plist with PNG reference
            if not icon_set:
                try:
                    info_plist_path = app_path / 'Contents' / 'Info.plist'
                    if info_plist_path.exists():
                        with open(info_plist_path, 'rb') as f:
                            plist_data = plistlib.load(f)

                        # Add icon reference
                        plist_data['CFBundleIconFile'] = 'icon.png'

                        # Write back
                        with open(info_plist_path, 'wb') as f:
                            plistlib.dump(plist_data, f)

                        self.log_callback("✓ Added icon reference to app bundle")
                        icon_set = True
                except Exception as e:
                    self.log_callback(f"Warning: Could not update Info.plist: {e}")

            if not icon_set:
                self.log_callback("Warning: Icon copied to resources but could not be set as app icon")
                self.log_callback("  You may need to manually set the icon or install fileicon")

            return True

        except Exception as e:
            self.log_callback(f"Error adding icon to app bundle: {e}")
            return False

    def _hide_app_from_dock(self, app_path: Path) -> bool:
        """
        Modify the app bundle's Info.plist to hide it from the dock.
        This prevents the AppleScript launcher from showing as a separate dock icon.
        
        :param app_path: Path to the app bundle
        :return: True if successful, False otherwise
        """
        try:
            info_plist_path = app_path / 'Contents' / 'Info.plist'
            if not info_plist_path.exists():
                self.log_callback("Warning: Info.plist not found, cannot hide app from dock")
                return False

            # Read the current plist
            with open(info_plist_path, 'rb') as f:
                plist_data = plistlib.load(f)

            # Add LSUIElement to hide from dock
            # LSUIElement = true means the app runs as a background agent without dock icon
            plist_data['LSUIElement'] = True

            # Write back the modified plist
            with open(info_plist_path, 'wb') as f:
                plistlib.dump(plist_data, f)

            self.log_callback("✓ Configured app to run without dock icon")
            return True

        except Exception as e:
            self.log_callback(f"Warning: Could not hide app from dock: {e}")
            return False

    def create_file_associations(self) -> bool:
        """
        Create file associations for .dgen files on macOS.
        On macOS, we skip app bundle creation entirely and provide manual instructions.
        
        :return: True if successful, False otherwise
        """
        try:
            # Check if dgenerate executable exists (we don't need dgenerate_windowed on macOS)
            dgenerate_exe = self.bin_dir / "dgenerate"
            if not dgenerate_exe.exists():
                self.log_callback("dgenerate executable not found in bin directory, cannot create file associations")
                return False

            self.log_callback("Setting up macOS file associations for .dgen files...")
            self.log_callback("Note: Using AppleScript approach - no windowed executable needed")

            # Skip app bundle creation entirely and go straight to manual instructions
            return self._provide_manual_association_instructions()

        except Exception as e:
            self.log_callback(f"Error setting up macOS file associations: {e}")
            return False

    def _provide_manual_association_instructions(self) -> bool:
        """Provide manual file association instructions for macOS."""
        self.log_callback("✓ File association setup complete")
        self.log_callback("  Note: File associations on macOS are limited")
        self.log_callback("  You can use dgenerate from the terminal with:")
        self.log_callback("    dgenerate --console file.dgen")
        self.log_callback("  Or double-click the desktop shortcut to open dgenerate")
        self.log_callback("  (App bundle creation skipped on macOS for simplicity)")
        return True

    def _copy_windowed_stub(self) -> bool:
        """
        Override base class method to skip windowed stub copying on macOS.
        The AppleScript approach eliminates the need for a windowed executable.
        
        :return: Always True since we're skipping windowed stub operations
        """
        self.log_callback("✓ Skipping windowed stub copying on macOS (AppleScript approach used instead)")

        # Still copy icon files for desktop shortcuts and file associations
        if not self._copy_icons():
            self.log_callback("Warning: Failed to copy desktop shortcut icons")

        if not self._copy_file_association_icon():
            self.log_callback("Warning: Failed to copy file association icons")

        return True

    def _copy_file_association_icon(self) -> bool:
        """
        Override base class method to skip icon copying on macOS.
        Icons are intentionally not used on macOS for simplicity.
        
        :return: Always True since we're skipping icon operations
        """
        self.log_callback("✓ Skipping file association icon copying on macOS (not needed)")
        return True

    def _remove_from_path(self):
        """Remove from macOS PATH by modifying shell profile."""
        try:
            scripts_dir_str = str(self.scripts_dir)

            # Common shell profile files
            profile_files = [
                Path.home() / '.bashrc',
                Path.home() / '.zshrc',
                Path.home() / '.profile',
            ]

            comment_line = '# Added by dgenerate network installer'
            export_line = f'export PATH="{scripts_dir_str}:$PATH"'

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
            self.log_callback(f"Error removing from macOS PATH: {e}")

    def _remove_desktop_shortcuts(self):
        """Remove desktop shortcuts."""
        try:
            # Get desktop path
            desktop = Path.home() / 'Desktop'

            # Remove .app file (AppleScript app)
            app_path = desktop / "Dgenerate Console.app"
            if app_path.exists():
                shutil.rmtree(app_path)
                self.log_callback(f"Removed macOS desktop shortcut: {app_path}")

            # Remove .command file (fallback)
            shortcut_path = desktop / "Dgenerate Console.command"
            if shortcut_path.exists():
                shortcut_path.unlink()
                self.log_callback(f"Removed fallback macOS desktop shortcut: {shortcut_path}")

            # Also check for old shortcut name
            old_shortcut_path = desktop / "dgenerate.command"
            if old_shortcut_path.exists():
                old_shortcut_path.unlink()
                self.log_callback(f"Removed old macOS desktop shortcut: {old_shortcut_path}")

            # Remove windowed executable on Unix systems
            windowed_exe = self.scripts_dir / "dgenerate_windowed"
            if windowed_exe.exists():
                try:
                    windowed_exe.unlink()
                    self.log_callback(f"Removed windowed executable: {windowed_exe}")
                except Exception as e:
                    self.log_callback(f"Could not remove windowed executable {windowed_exe}: {e}")

        except Exception as e:
            self.log_callback(f"Error removing macOS desktop shortcuts: {e}")

    def _remove_file_associations(self):
        """Remove macOS file associations (no app bundle to remove)."""
        try:
            self.log_callback("Removing macOS file associations...")
            self.log_callback("Note: No app bundle to remove (app bundles are not created on macOS)")

            # Note about potential remaining associations
            self.log_callback("Note: If file associations persist, you can manually remove them in:")
            self.log_callback("  System Preferences > General > Default Apps")

        except Exception as e:
            self.log_callback(f"Error removing macOS file associations: {e}")

    def _get_shell_profile(self) -> Optional[Path]:
        """Get the appropriate shell profile file for macOS."""
        home = Path.home()

        # Check for different shell profiles
        profiles = [
            home / '.zshrc',  # Default shell on modern macOS
            home / '.bash_profile',
            home / '.bashrc',
            home / '.profile'
        ]

        for profile in profiles:
            if profile.exists():
                return profile

        # Default to .zshrc if none exist (modern macOS default)
        return home / '.zshrc'

    def _verify_path_update(self):
        """Verify that the PATH was actually updated."""
        try:
            # For macOS, check if the export line exists in the shell profile
            shell_profile = self._get_shell_profile()
            if shell_profile and shell_profile.exists():
                content = shell_profile.read_text()
                scripts_dir_str = str(self.scripts_dir)
                export_line = f'export PATH="{scripts_dir_str}:$PATH"'

                if export_line in content:
                    self.log_callback("✓ PATH verification successful - dgenerate is now available globally")
                    self.log_callback("  Note: You may need to restart your terminal or run 'source ~/.zshrc'")
                else:
                    self.log_callback("Warning: PATH verification failed - dgenerate may not be available globally")
            else:
                self.log_callback("Warning: Could not verify PATH update")

        except Exception as e:
            self.log_callback(f"Warning: Could not verify PATH update: {e}")
