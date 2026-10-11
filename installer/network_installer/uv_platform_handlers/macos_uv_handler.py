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
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .base_uv_handler import BasePlatformHandler


class MacOSPlatformHandler(BasePlatformHandler):
    """
    macOS-specific implementation of platform operations.
    """

    def add_scripts_to_path(self) -> bool:
        """
        Add only the bin directory to the user shell PATH.

        Writes an ``export PATH=...`` line into the shell profile(s) that
        Terminal will actually source. On modern macOS that is ``~/.zshrc``;
        a leftover ``~/.bash_profile`` must not win over a missing zshrc.

        :return: True if successful, False otherwise
        """
        try:
            # First create stub scripts in the bin directory
            if not self.create_stub_scripts():
                self.log_callback("Failed to create stub scripts")
                return False

            bin_dir_str = str(self.bin_dir)
            self.log_callback(f"Adding {bin_dir_str} to system PATH...")

            profiles = self._shell_profiles_to_update()
            if not profiles:
                self.log_callback("Could not determine shell profile file")
                return False

            comment_line = '# Added by dgenerate network installer'
            export_line = f'export PATH="{bin_dir_str}:$PATH"'
            updated = []

            for shell_profile in profiles:
                if self._profile_has_export(shell_profile, export_line):
                    self.log_callback(
                        f"✓ Directory {bin_dir_str} already configured in {shell_profile}")
                    continue

                shell_profile.parent.mkdir(parents=True, exist_ok=True)
                with open(shell_profile, 'a', encoding='utf-8') as f:
                    f.write(f'\n{comment_line}\n{export_line}\n')
                updated.append(shell_profile)
                self.log_callback(f"✓ Added {bin_dir_str} to {shell_profile}")

            if updated:
                # Prefer telling the user about the profile their login shell uses.
                primary = profiles[0]
                self.log_callback(
                    f"  Please open a new terminal or run 'source {primary}' to use dgenerate")
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
                    # Icon, dock hiding, and re-sign happen together. A second
                    # plist write after signing puts the generic script icon back.
                    self._add_icon_to_app_bundle(app_path)

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

            icns_path = shortcut_path.with_suffix('.icns')
            try:
                if self._write_icns(icns_path) and self._set_finder_icon(shortcut_path, icns_path):
                    self.log_callback("✓ Set the dgenerate icon on the .command shortcut")
            finally:
                icns_path.unlink(missing_ok=True)

            self.log_callback(f"✓ Created fallback macOS desktop shortcut: {shortcut_path}")
            return True

        except Exception as e:
            self.log_callback(f"Error creating fallback macOS desktop shortcut: {e}")
            return False

    def _copy_icon_png(self, dest: Path) -> bool:
        """Copy packaged ``icon.png`` to ``dest`` (build converts ``dgenerate/icon.ico``)."""
        # importlib.resources may extract to a temp path that vanishes when the
        # context exits, so copy inside the with-block.
        try:
            import importlib.resources as resources
            with resources.path('network_installer.resources', 'icon.png') as resource_path:
                if resource_path.exists():
                    shutil.copy2(resource_path, dest)
                    return True
        except (ImportError, FileNotFoundError, TypeError, OSError):
            pass

        candidates = []
        if hasattr(sys, '_MEIPASS'):
            candidates.append(
                Path(sys._MEIPASS) / 'network_installer' / 'resources' / 'icon.png')
        candidates.append(self.install_base / 'icon.png')
        candidates.append(Path(__file__).parent.parent / 'resources' / 'icon.png')
        for candidate in candidates:
            if candidate.exists():
                shutil.copy2(candidate, dest)
                return True
        return False

    def _write_icns(self, icns_path: Path) -> bool:
        """Build ``icns_path`` from the packaged PNG."""
        scratch = Path(tempfile.mkdtemp())
        try:
            png_path = scratch / 'icon.png'
            if not self._copy_icon_png(png_path):
                self.log_callback("Warning: Could not find icon.png for the desktop shortcut")
                return False
            return self._png_to_icns(png_path, icns_path)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def _png_to_icns(self, png_path: Path, icns_path: Path) -> bool:
        """Build an ``.icns`` from a PNG with stock ``sips`` + ``iconutil``."""
        iconset = Path(tempfile.mkdtemp(suffix='.iconset'))
        try:
            # Names required by iconutil.
            variants = (
                (16, 'icon_16x16.png'),
                (32, 'diana.k@example.org'),
                (32, 'icon_32x32.png'),
                (64, 'ivan.p@example.net'),
                (128, 'icon_128x128.png'),
                (256, 'wendy.h@example.net'),
                (256, 'icon_256x256.png'),
                (512, 'wendy.h@example.net'),
                (512, 'icon_512x512.png'),
                (1024, 'walt.e@example.net'),
            )
            iconset_ok = True
            for size, name in variants:
                out = iconset / name
                result = subprocess.run(
                    ['sips', '-z', str(size), str(size), str(png_path), '--out', str(out)],
                    capture_output=True, text=True)
                if result.returncode != 0 or not out.exists():
                    self.log_callback(
                        f"Warning: sips failed for {name}: {result.stderr or result.stdout}")
                    iconset_ok = False
                    break
            if iconset_ok:
                result = subprocess.run(
                    ['iconutil', '-c', 'icns', str(iconset), '-o', str(icns_path)],
                    capture_output=True, text=True)
                if result.returncode == 0 and icns_path.exists():
                    return True
                self.log_callback(
                    f"Warning: iconutil failed: {result.stderr or result.stdout}")
            result = subprocess.run(
                ['sips', '-s', 'format', 'icns', str(png_path), '--out', str(icns_path)],
                capture_output=True, text=True)
            if result.returncode == 0 and icns_path.exists():
                return True
            self.log_callback(
                f"Warning: sips icns conversion failed: {result.stderr or result.stdout}")
            return False
        finally:
            shutil.rmtree(iconset, ignore_errors=True)

    def _add_icon_to_app_bundle(self, app_path: Path) -> bool:
        """
        Give the AppleScript launcher the dgenerate icon.

        ``osacompile`` ships ``applet.icns`` and points ``CFBundleIconFile``
        at it. Replacing that file, then signing the bundle again, is what
        makes Finder drop the generic white script icon. ``LSUIElement`` is
        written in the same plist edit so a later rewrite does not unsign it.
        """
        try:
            resources_dir = app_path / 'Contents' / 'Resources'
            resources_dir.mkdir(parents=True, exist_ok=True)
            icns_dest = resources_dir / 'applet.icns'
            if not self._write_icns(icns_dest):
                return False

            info_plist_path = app_path / 'Contents' / 'Info.plist'
            if not info_plist_path.exists():
                self.log_callback("Warning: Info.plist not found, cannot set the shortcut icon")
                return False
            with open(info_plist_path, 'rb') as f:
                plist_data = plistlib.load(f)
            # Name without extension. This is the file osacompile already registered.
            plist_data['CFBundleIconFile'] = 'applet'
            plist_data['LSUIElement'] = True
            with open(info_plist_path, 'wb') as f:
                plistlib.dump(plist_data, f)

            # Sign the edited bundle. Finder keeps the generic script icon
            # when the applet's signature no longer matches its resources.
            subprocess.run(
                ['xattr', '-cr', str(app_path)],
                capture_output=True, text=True)
            if not self._codesign_adhoc(app_path):
                self.log_callback("Warning: Could not re-sign the desktop shortcut")
            # Finder's desktop icon is a custom-icon resource, separate from
            # CFBundleIconFile. Apply it after signing.
            self._set_finder_icon(app_path, icns_dest)
            self._register_app(app_path)
            self.log_callback("✓ Set the dgenerate icon on the desktop shortcut")
            return True
        except Exception as e:
            self.log_callback(f"Error adding icon to app bundle: {e}")
            return False

    def _set_finder_icon(self, target: Path, image_path: Path) -> bool:
        """Ask AppKit to attach ``image_path`` as the Finder icon for ``target``."""
        script = inspect.cleandoc('''
            use framework "AppKit"
            on run argv
                set imagePath to item 1 of argv
                set targetPath to item 2 of argv
                set theImage to current application's NSImage's alloc()'s initWithContentsOfFile:imagePath
                if theImage is missing value then error "Could not load icon"
                set applied to current application's NSWorkspace's sharedWorkspace()'s setIcon:theImage forFile:targetPath options:0
                if applied is false then error "setIcon returned false"
                return "ok"
            end run
        ''')
        script_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.applescript', delete=False) as handle:
                handle.write(script)
                script_path = handle.name
            result = subprocess.run(
                ['osascript', script_path, str(image_path), str(target)],
                capture_output=True, text=True)
            if result.returncode != 0:
                self.log_callback(
                    f"Warning: Could not set Finder icon: {result.stderr or result.stdout}")
                return False
            return True
        except Exception as e:
            self.log_callback(f"Warning: Could not set Finder icon: {e}")
            return False
        finally:
            if script_path:
                try:
                    os.unlink(script_path)
                except OSError:
                    pass

    def _codesign_adhoc(self, app_path: Path) -> bool:
        """Re-sign after editing the bundle so Finder accepts the new icon."""
        result = subprocess.run(
            ['codesign', '--force', '--sign', '-', str(app_path)],
            capture_output=True, text=True)
        if result.returncode != 0:
            self.log_callback(
                f"Warning: codesign failed: {result.stderr or result.stdout}")
            return False
        return True

    def _register_app(self, app_path: Path) -> None:
        """Tell Launch Services the bundle changed so the Desktop icon updates."""
        app_path.touch()
        lsregister = Path(
            '/System/Library/Frameworks/CoreServices.framework/'
            'Frameworks/LaunchServices.framework/Support/lsregister')
        if not lsregister.exists():
            return
        subprocess.run(
            [str(lsregister), '-f', str(app_path)],
            capture_output=True, text=True)

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
            # bin_dir is what add_scripts_to_path writes; also strip the older
            # scripts_dir (venv/bin) export left by previous installer versions.
            export_paths = {str(self.bin_dir), str(self.scripts_dir)}
            export_lines = {f'export PATH="{path}:$PATH"' for path in export_paths}

            profile_files = [
                Path.home() / '.zshrc',
                Path.home() / '.zprofile',
                Path.home() / '.bash_profile',
                Path.home() / '.bashrc',
                Path.home() / '.profile',
            ]

            comment_line = '# Added by dgenerate network installer'

            for profile_file in profile_files:
                if not profile_file.exists():
                    continue
                try:
                    content = profile_file.read_text(encoding='utf-8')
                    lines = content.split('\n')
                    new_lines = []
                    skip_next = False
                    changed = False

                    for line in lines:
                        stripped = line.strip()
                        if stripped == comment_line:
                            skip_next = True
                            changed = True
                            continue
                        if skip_next and stripped in export_lines:
                            skip_next = False
                            changed = True
                            continue
                        if stripped in export_lines:
                            changed = True
                            continue
                        new_lines.append(line)
                        skip_next = False

                    if changed:
                        profile_file.write_text('\n'.join(new_lines), encoding='utf-8')
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

    def _shell_name(self) -> str:
        """Return the user's login shell basename (e.g. ``zsh``, ``bash``)."""
        shell = os.environ.get('SHELL') or ''
        if shell:
            return Path(shell).name
        # GUI launches often omit SHELL; fall back to the directory service.
        try:
            result = subprocess.run(
                ['dscl', '.', '-read', f'/Users/{Path.home().name}', 'UserShell'],
                capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for part in result.stdout.split():
                    if part.startswith('/'):
                        return Path(part).name
        except (OSError, subprocess.TimeoutExpired):
            pass
        return 'zsh'

    def _shell_profiles_to_update(self) -> list[Path]:
        """Shell profiles that should receive the dgenerate PATH export.

        Always include ``~/.zshrc`` on macOS (default since Catalina). Prefer
        that over an existing leftover ``~/.bash_profile``, which zsh never
        sources. Also update the bash profile when the login shell is bash.
        """
        home = Path.home()
        profiles: list[Path] = [home / '.zshrc']
        if self._shell_name() == 'bash':
            for candidate in (home / '.bash_profile', home / '.bashrc'):
                if candidate.exists():
                    profiles.append(candidate)
                    break
            else:
                profiles.append(home / '.bash_profile')
        return profiles

    def _get_shell_profile(self) -> Path | None:
        """Get the primary shell profile file for macOS."""
        profiles = self._shell_profiles_to_update()
        return profiles[0] if profiles else None

    @staticmethod
    def _profile_has_export(profile: Path, export_line: str) -> bool:
        """True when ``profile`` already contains the given PATH export."""
        if not profile.exists():
            return False
        try:
            return export_line in profile.read_text(encoding='utf-8')
        except OSError:
            return False

    def _verify_path_update(self):
        """Verify that the PATH was actually updated."""
        try:
            export_line = f'export PATH="{self.bin_dir}:$PATH"'
            profiles = self._shell_profiles_to_update()
            configured = [
                profile for profile in profiles
                if self._profile_has_export(profile, export_line)
            ]

            if configured:
                self.log_callback(
                    "✓ PATH verification successful - dgenerate is now available globally")
                self.log_callback(
                    f"  Note: You may need to restart your terminal or run "
                    f"'source {configured[0]}'")
            else:
                self.log_callback(
                    "Warning: PATH verification failed - dgenerate may not be available globally")

        except Exception as e:
            self.log_callback(f"Warning: Could not verify PATH update: {e}")
