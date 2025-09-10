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
Windows-specific platform handler for the dgenerate installer.
"""

import inspect
import os
import sys
import tempfile
import time
from network_installer.subprocess_utils import run_silent
from pathlib import Path

try:
    import winreg
except ImportError:
    winreg = None

import ctypes

try:
    from ctypes import wintypes
except ImportError:
    wintypes = None

from .base_uv_handler import BasePlatformHandler


class WindowsPlatformHandler(BasePlatformHandler):
    """
    Windows-specific implementation of platform operations.
    """
    
    def check_and_enable_long_paths(self) -> bool:
        """
        Check if Windows long paths are enabled and enable them if necessary.
        Automatically detects GUI mode by checking if a tkinter root exists.
        
        :return: True if long paths are enabled or successfully enabled, False otherwise
        """
        try:
            # Check if long paths are already enabled
            if self.is_long_path_enabled():
                self.log_callback("✓ Windows long paths are already enabled")
                return True
            
            self.log_callback("Windows long paths are currently disabled")
            self.log_callback("Some Python packages require paths longer than 260 characters")
            
            # Detect if we're in GUI mode by checking for tkinter root
            gui_mode = self._is_gui_mode()
            self.log_callback(f"Detected mode: {'GUI' if gui_mode else 'Silent'}")
            
            # If in GUI mode, ask for user consent first
            if gui_mode:
                if not self.prompt_for_long_path_consent():
                    self.log_callback("User declined to enable Windows long path support")
                    self.log_callback("Installation cannot proceed without long path support")
                    return False
            
            # Try to enable long paths with elevation
            if self.enable_long_paths_with_elevation():
                self.log_callback("✓ Windows long paths have been enabled successfully")
                return True
            else:
                self.log_callback("ERROR: Failed to enable Windows long paths")
                self.log_callback("Installation cannot proceed without long path support")
                return False
                
        except Exception as e:
            self.log_callback(f"ERROR: Failed to check/enable long paths: {e}")
            return False
    
    def _is_gui_mode(self) -> bool:
        """
        Detect if we're running in GUI mode by checking if a tkinter root exists.
        
        :return: True if GUI mode, False if silent mode
        """
        try:
            import tkinter as tk
            
            # Check if there's an active tkinter root window
            root = tk._default_root
            if root is not None and root.winfo_exists():
                return True
                
            # Alternative check: try to get all tkinter windows
            try:
                if tk._default_root or len(tk.Tk._instances) > 0:
                    return True
            except (AttributeError, TypeError):
                pass
                
            return False
            
        except ImportError:
            # tkinter not available, assume silent mode
            return False
        except Exception:
            # Any other error, assume silent mode for safety
            return False
    
    def is_long_path_enabled(self) -> bool:
        """
        Check if Windows long path support is enabled.
        
        :return: True if enabled, False otherwise
        """
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                               r'SYSTEM\CurrentControlSet\Control\FileSystem',
                               0, winreg.KEY_READ) as key:
                try:
                    value, _ = winreg.QueryValueEx(key, 'LongPathsEnabled')
                    return bool(value)
                except FileNotFoundError:
                    # Registry key doesn't exist, long paths are disabled
                    return False
        except Exception as e:
            self.log_callback(f"Warning: Could not check long path status: {e}")
            return False
    
    def is_admin(self) -> bool:
        """
        Check if the current process is running with administrator privileges.
        
        :return: True if admin, False otherwise
        """
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except Exception:
            return False
    
    def enable_long_paths_with_elevation(self) -> bool:
        """
        Enable Windows long paths by requesting elevation if needed.
        
        :return: True if successful, False otherwise
        """
        try:
            # If we're already admin, try to enable directly
            if self.is_admin():
                return self._enable_long_paths_registry()
            
            # Otherwise, request elevation
            return self._request_elevation_for_long_paths()
            
        except Exception as e:
            self.log_callback(f"Error enabling long paths with elevation: {e}")
            return False
    
    def _enable_long_paths_registry(self) -> bool:
        """
        Enable long paths by modifying the registry (requires admin privileges).
        
        :return: True if successful, False otherwise
        """
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r'SYSTEM\CurrentControlSet\Control\FileSystem',
                               0, winreg.KEY_WRITE) as key:
                winreg.SetValueEx(key, 'LongPathsEnabled', 0, winreg.REG_DWORD, 1)
            
            self.log_callback("Long paths enabled in registry")
            return True
            
        except PermissionError:
            self.log_callback("Permission denied - administrator privileges required")
            return False
        except Exception as e:
            self.log_callback(f"Error modifying registry: {e}")
            return False
    
    def _request_elevation_for_long_paths(self) -> bool:
        """
        Request UAC elevation to enable long paths using VBScript with proper async handling.
        
        :return: True if successful, False otherwise
        """
        try:
            # Create a simple VBScript that enables long paths
            vbs_script = inspect.cleandoc('''
                On Error Resume Next
                
                ' Create WScript.Shell object for registry access
                Set objWScript = CreateObject("WScript.Shell")
                
                ' Registry path and value
                regPath = "HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem\\LongPathsEnabled"
                
                ' Try to write to registry
                objWScript.RegWrite regPath, 1, "REG_DWORD"
                
                ' Check if write was successful
                If Err.Number = 0 Then
                    WScript.Echo "SUCCESS: Long paths enabled"
                    WScript.Quit 0
                Else
                    WScript.Echo "FAILED: " & Err.Description & " (Error: " & Err.Number & ")"
                    WScript.Quit 1
                End If
            ''')
            
            # Write VBScript to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.vbs', delete=False, encoding='utf-8') as f:
                f.write(vbs_script)
                vbs_file = f.name
            
            try:
                self.log_callback("Requesting administrator privileges to enable long paths...")
                self.log_callback("Please click 'Yes' in the UAC dialog that appears")
                
                # Build the command to run cscript with the VBScript
                cscript_path = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'System32', 'cscript.exe')
                vbs_args = f'//NoLogo "{vbs_file}"'
                
                # Use ShellExecute to run VBScript with elevation
                result_code = ctypes.windll.shell32.ShellExecuteW(
                    None,                    # hwnd
                    "runas",                 # lpVerb (elevation)
                    cscript_path,            # lpFile
                    vbs_args,                # lpParameters
                    None,                    # lpDirectory
                    0                        # nShowCmd (SW_HIDE - hidden window)
                )
                
                # ShellExecute returns > 32 for success, <= 32 for error
                if result_code <= 32:
                    self.log_callback(f"ShellExecute failed with code: {result_code}")
                    
                    # Map common ShellExecute error codes
                    error_messages = {
                        0: "Out of memory or resources",
                        2: "File not found", 
                        3: "Path not found",
                        5: "Access denied",
                        8: "Out of memory",
                        26: "Cannot share an open file",
                        27: "File association incomplete or invalid",
                        28: "DDE timeout",
                        29: "DDE transaction failed",
                        30: "DDE busy",
                        31: "No association for file extension",
                        32: "DLL not found"
                    }
                    
                    error_msg = error_messages.get(result_code, f"Unknown error code {result_code}")
                    self.log_callback(f"ShellExecute error: {error_msg}")
                    return False
                
                self.log_callback(f"Elevation request launched successfully (code: {result_code})")
                
                # Wait for the elevated VBScript process to complete
                # Since ShellExecute is asynchronous, we poll the registry to detect completion
                self.log_callback("Waiting for elevation process to complete...")
                
                max_wait = 60  # Maximum 60 seconds (UAC can take time)
                poll_interval = 0.5  # Check every 500ms for faster response
                
                for i in range(int(max_wait / poll_interval)):
                    time.sleep(poll_interval)
                    current_status = self.is_long_path_enabled()
                    
                    if current_status:
                        elapsed = (i + 1) * poll_interval
                        self.log_callback(f"✓ Long paths enabled successfully (took {elapsed:.1f} seconds)")
                        return True
                    
                    # Show progress every 5 seconds to avoid spam
                    elapsed = (i + 1) * poll_interval
                    if elapsed % 5.0 == 0:
                        self.log_callback(f"Still waiting... ({elapsed:.0f}/{max_wait} seconds)")
                
                # If we get here, either the user cancelled or it failed
                self.log_callback("Timeout waiting for long paths to be enabled")
                self.log_callback("This usually means:")
                self.log_callback("  1. The UAC dialog was cancelled by the user")
                self.log_callback("  2. The elevated VBScript process failed") 
                self.log_callback("  3. Insufficient permissions even with elevation")
                self.log_callback("  4. VBScript execution is disabled on this system")
                
                return False
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(vbs_file)
                except:
                    pass
                    
        except Exception as e:
            self.log_callback(f"Error requesting elevation: {e}")
            return False
    
    def prompt_for_long_path_consent(self) -> bool:
        """
        Show a GUI dialog asking for user consent to enable long paths.
        
        :return: True if user consents, False otherwise
        """
        try:
            import tkinter as tk
            from tkinter import messagebox
            
            # Create a hidden root window
            root = tk.Tk()
            root.withdraw()
            
            # Show the consent dialog
            title = "Windows Long Path Support Required"
            message = inspect.cleandoc('''
                dgenerate requires Windows long path support to be enabled.
                
                Some Python packages install files with paths longer than 260 characters,
                which Windows blocks by default.
                
                The installer needs to:
                • Request administrator privileges
                • Enable long path support in the Windows registry
                
                This is a one-time system change that will benefit other applications too.
                
                Do you want to enable Windows long path support now?
            ''')
            
            result = messagebox.askyesno(title, message, icon='question')
            
            # Clean up
            root.destroy()
            
            return result
            
        except Exception as e:
            self.log_callback(f"Error showing long path consent dialog: {e}")
            # If GUI fails, assume consent (will fail later if elevation is denied)
            return True

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

            # Open the user environment variables registry key
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r'Environment',
                                0,
                                winreg.KEY_READ | winreg.KEY_WRITE) as key:

                # Get current PATH value
                try:
                    current_path, _ = winreg.QueryValueEx(key, 'PATH')
                except FileNotFoundError:
                    current_path = ''

                # Check if our directory is already in PATH
                path_parts = [p.strip() for p in current_path.split(os.pathsep) if p.strip()]

                if bin_dir_str not in path_parts:
                    # Add our directory to the beginning of PATH
                    new_path = bin_dir_str + os.pathsep + current_path if current_path else bin_dir_str

                    # Set the new PATH value
                    winreg.SetValueEx(key, 'PATH', 0, winreg.REG_EXPAND_SZ, new_path)

                    self.log_callback(f"✓ Added {bin_dir_str} to system PATH")

                    # Notify the system of the environment change
                    self._broadcast_environment_change()

                else:
                    self.log_callback(f"✓ Directory {bin_dir_str} already in PATH")

            return True

        except Exception as e:
            self.log_callback(f"Error modifying Windows PATH: {e}")
            return False

    def create_stub_scripts(self) -> bool:
        """
        Create Windows-specific stub scripts.
        Note: dgenerate.exe and dgenerate_windowed.exe are copied directly, not stubbed.
        """
        try:
            # Create bin directory
            self.bin_dir.mkdir(parents=True, exist_ok=True)

            # No stub scripts needed - dgenerate.exe and dgenerate_windowed.exe 
            # are copied directly and modified to handle environment setup
            self.log_callback("✓ Windows bin directory created (no stub scripts needed)")
            return True

        except Exception as e:
            self.log_callback(f"Error creating Windows bin directory: {e}")
            return False

    def create_desktop_shortcut(self) -> bool:
        """
        Create desktop shortcut for dgenerate using the windowed stub.
        
        :return: True if successful, False otherwise
        """
        try:
            # Get desktop path
            desktop = Path(os.environ.get('USERPROFILE', '')) / 'Desktop'
            if not desktop.exists():
                desktop = Path.home() / 'Desktop'

            # Create shortcut
            shortcut_path = desktop / "Dgenerate Console.lnk"

            # Get windowed executable path from bin directory
            windowed_exe = self.bin_dir / "dgenerate_windowed.exe"
            if not windowed_exe.exists():
                self.log_callback("dgenerate_windowed.exe not found in bin directory, cannot create shortcut")
                return False

            # Get icon path
            icon_path = self.install_base / "icon.ico"
            if not icon_path.exists():
                self.log_callback("icon.ico not found, using default icon")
                icon_location = f"{windowed_exe},0"
            else:
                icon_location = str(icon_path)

            # Create the shortcut using VBScript
            if self._create_shortcut_vbs(shortcut_path, windowed_exe, str(icon_path)):
                self.log_callback(f"✓ Created desktop shortcut: {shortcut_path}")

                # Also create Start Menu shortcut
                if self._create_start_menu_shortcut(windowed_exe, str(icon_path)):
                    self.log_callback("✓ Created Start Menu shortcut")
                else:
                    self.log_callback("Warning: Failed to create Start Menu shortcut")

                return True
            else:
                self.log_callback("Failed to create desktop shortcut")
                return False

        except Exception as e:
            self.log_callback(f"Error creating Windows desktop shortcut: {e}")
            return False

    def _create_shortcut_vbs(self, shortcut_path: Path, target_exe: Path, icon_path: str) -> bool:
        """
        Create Windows shortcut using VBScript.
        
        :param shortcut_path: Path where shortcut will be created
        :param target_exe: Path to executable to run
        :param icon_path: Path to icon file
        :return: True if successful, False otherwise
        """
        try:
            # Create a temporary VBScript file
            vbs_content = inspect.cleandoc(f'''
                Set oWS = WScript.CreateObject("WScript.Shell")
                sLinkFile = "{shortcut_path}"
                Set oLink = oWS.CreateShortcut(sLinkFile)
                oLink.TargetPath = "{target_exe}"
                oLink.Arguments = "--console"
                oLink.WorkingDirectory = "{Path.home()}"
                oLink.IconLocation = "{icon_path}"
                oLink.Description = "Launch dgenerate Console UI"
                oLink.Save
            ''')

            # Write VBScript to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.vbs', delete=False) as f:
                f.write(vbs_content)
                vbs_file = f.name

            try:
                # Execute VBScript using subprocess_utils to prevent terminal flash
                run_silent(['cscript', '//NoLogo', vbs_file],
                           capture_output=True, text=True, check=True)
                return True
            finally:
                # Clean up temporary file
                try:
                    os.unlink(vbs_file)
                except:
                    pass

        except Exception as e:
            self.log_callback(f"Error creating shortcut with VBScript: {e}")
            return False

    def _create_start_menu_shortcut(self, windowed_exe: Path, icon_path: str) -> bool:
        """
        Create Start Menu shortcut for dgenerate console.
        
        :param windowed_exe: Path to windowed executable
        :param icon_path: Path to icon file
        :return: True if successful, False otherwise
        """
        try:
            # Get Start Menu Programs path
            start_menu = Path(os.environ.get('APPDATA', '')) / 'Microsoft' / 'Windows' / 'Start Menu' / 'Programs'
            if not start_menu.exists():
                # Fallback to user profile
                start_menu = Path(os.environ.get('USERPROFILE',
                                                 '')) / 'AppData' / 'Roaming' / 'Microsoft' / 'Windows' / 'Start Menu' / 'Programs'

            if not start_menu.exists():
                self.log_callback("Could not find Start Menu Programs directory")
                return False

            # Create shortcut
            shortcut_path = start_menu / "Dgenerate Console.lnk"

            # Create the shortcut using VBScript
            if self._create_shortcut_vbs(shortcut_path, windowed_exe, str(icon_path)):
                self.log_callback(f"✓ Created Start Menu shortcut: {shortcut_path}")
                return True
            else:
                self.log_callback("Failed to create Start Menu shortcut")
                return False

        except Exception as e:
            self.log_callback(f"Error creating Start Menu shortcut: {e}")
            return False

    def create_file_associations(self) -> bool:
        """
        Create file associations for .dgen files.
        
        :return: True if successful, False otherwise
        """
        try:
            # Get windowed executable path for console UI
            windowed_exe = self.bin_dir / "dgenerate_windowed.exe"
            if not windowed_exe.exists():
                self.log_callback("dgenerate_windowed.exe not found in bin directory, cannot create file associations")
                return False

            self.log_callback("Creating Windows file associations for .dgen files...")

            # Create file association
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r'Software\Classes\.dgen') as key:
                winreg.SetValue(key, '', winreg.REG_SZ, 'dgenerate.config')
                winreg.SetValueEx(key, 'Content Type', 0, winreg.REG_SZ, 'application/x-dgenerate-config')

            # Create application association
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r'Software\Classes\dgenerate.config') as key:
                winreg.SetValue(key, '', winreg.REG_SZ, 'dgenerate Configuration File')

                # Set icon for file association
                config_icon_path = self.install_base / 'config_icon.ico'
                if config_icon_path.exists():
                    with winreg.CreateKey(key, r'DefaultIcon') as icon_key:
                        winreg.SetValue(icon_key, '', winreg.REG_SZ, str(config_icon_path))

                # Create shell/open/command
                with winreg.CreateKey(key, r'shell\open\command') as cmd_key:
                    winreg.SetValue(cmd_key, '', winreg.REG_SZ, f'"{windowed_exe}" --console "%1"')

                # Create shell/openwithprogids
                with winreg.CreateKey(key, r'shell\openwithprogids') as progids_key:
                    winreg.SetValueEx(progids_key, 'dgenerate.config', 0, winreg.REG_NONE, b'')

            # Add to OpenWithProgids for .dgen files
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r'Software\Classes\.dgen\OpenWithProgids') as key:
                winreg.SetValueEx(key, 'dgenerate.config', 0, winreg.REG_NONE, b'')

            # Register dgenerate_windowed.exe as an application
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER,
                                  r'Software\Classes\Applications\dgenerate_windowed.exe') as key:
                winreg.SetValueEx(key, 'FriendlyAppName', 0, winreg.REG_SZ, 'dgenerate')

                # Create shell/open/command
                with winreg.CreateKey(key, r'shell\open\command') as cmd_key:
                    winreg.SetValue(cmd_key, '', winreg.REG_SZ, f'"{windowed_exe}" --console "%1"')

            # Register icon
            self._register_dgenerate_icon()

            self.log_callback("✓ Windows file associations created successfully")
            return True

        except Exception as e:
            self.log_callback(f"Error creating Windows file associations: {e}")
            return False

    def _remove_from_path(self):
        """Remove from Windows PATH using registry."""
        try:
            scripts_dir_str = str(self.scripts_dir)
            self.log_callback(f"Removing {scripts_dir_str} from system PATH...")

            # Open the user environment variables registry key
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r'Environment',
                                0,
                                winreg.KEY_READ | winreg.KEY_WRITE) as key:

                # Get current PATH value
                try:
                    current_path, _ = winreg.QueryValueEx(key, 'PATH')
                except FileNotFoundError:
                    self.log_callback("PATH not found in registry")
                    return

                # Remove our directory from PATH
                path_parts = [p.strip() for p in current_path.split(os.pathsep) if p.strip()]

                if scripts_dir_str in path_parts:
                    path_parts.remove(scripts_dir_str)
                    new_path = os.pathsep.join(path_parts)

                    # Set the new PATH value
                    winreg.SetValueEx(key, 'PATH', 0, winreg.REG_EXPAND_SZ, new_path)

                    self.log_callback(f"✓ Removed {scripts_dir_str} from system PATH")

                    # Notify the system of the environment change
                    self._broadcast_environment_change()

                else:
                    self.log_callback(f"Directory {scripts_dir_str} not found in PATH")

        except Exception as e:
            self.log_callback(f"Error removing from Windows PATH: {e}")

    def _remove_desktop_shortcuts(self):
        """Remove desktop shortcuts."""
        try:
            # Get desktop path
            desktop = Path(os.environ.get('USERPROFILE', '')) / 'Desktop'
            if not desktop.exists():
                desktop = Path.home() / 'Desktop'

            # Remove desktop shortcut
            shortcut_path = desktop / "Dgenerate Console.lnk"
            if shortcut_path.exists():
                shortcut_path.unlink()
                self.log_callback(f"Removed desktop shortcut: {shortcut_path}")

            # Remove Start Menu shortcut
            start_menu = Path(os.environ.get('APPDATA', '')) / 'Microsoft' / 'Windows' / 'Start Menu' / 'Programs'
            if not start_menu.exists():
                start_menu = Path(os.environ.get('USERPROFILE',
                                                 '')) / 'AppData' / 'Roaming' / 'Microsoft' / 'Windows' / 'Start Menu' / 'Programs'

            if start_menu.exists():
                start_menu_shortcut = start_menu / "Dgenerate Console.lnk"
                if start_menu_shortcut.exists():
                    start_menu_shortcut.unlink()
                    self.log_callback(f"Removed Start Menu shortcut: {start_menu_shortcut}")

            # Also remove the Python stub launcher and windowed executable (if they exist)
            # Remove Python stub
            python_stub = desktop / "dgenerate_launcher.py"
            if python_stub.exists():
                try:
                    python_stub.unlink()
                    self.log_callback(f"Removed Python stub launcher: {python_stub}")
                except Exception as e:
                    self.log_callback(f"Could not remove Python stub {python_stub}: {e}")

            # Remove windowed executable
            windowed_exe = self.scripts_dir / "dgenerate_windowed.exe"
            if windowed_exe.exists():
                try:
                    windowed_exe.unlink()
                    self.log_callback(f"Removed windowed executable: {windowed_exe}")
                except Exception as e:
                    self.log_callback(f"Could not remove windowed executable {windowed_exe}: {e}")

        except Exception as e:
            self.log_callback(f"Error removing Windows desktop shortcuts: {e}")

    def _remove_file_associations(self):
        """Remove Windows file associations."""
        try:
            # Helper function to safely delete a registry key and its subkeys
            def delete_registry_key(hkey, key_path):
                try:
                    # First, try to delete all subkeys recursively
                    with winreg.OpenKey(hkey, key_path, 0, winreg.KEY_ALL_ACCESS) as key:
                        subkeys = []
                        i = 0
                        while True:
                            try:
                                subkey_name = winreg.EnumKey(key, i)
                                subkeys.append(subkey_name)
                                i += 1
                            except OSError:
                                break

                        # Delete subkeys first
                        for subkey_name in subkeys:
                            delete_registry_key(hkey, f"{key_path}\\{subkey_name}")

                    # Then delete the key itself
                    winreg.DeleteKey(hkey, key_path)
                    self.log_callback(f"Deleted registry key: {key_path}")

                except FileNotFoundError:
                    # Key doesn't exist, which is fine
                    pass
                except Exception as e:
                    self.log_callback(f"Could not delete registry key {key_path}: {e}")

            # Helper function to safely delete a registry value
            def delete_registry_value(hkey, key_path, value_name):
                try:
                    with winreg.OpenKey(hkey, key_path, 0, winreg.KEY_SET_VALUE) as key:
                        winreg.DeleteValue(key, value_name)
                    self.log_callback(f"Deleted registry value: {key_path}\\{value_name}")
                except FileNotFoundError:
                    # Key or value doesn't exist, which is fine
                    pass
                except Exception as e:
                    self.log_callback(f"Could not delete registry value {key_path}\\{value_name}: {e}")

            # Remove file associations - delete in reverse order of creation
            self.log_callback("Removing Windows registry entries for file associations...")

            # Remove individual values first
            delete_registry_value(winreg.HKEY_CURRENT_USER, r'Software\Classes\.dgen', 'Content Type')
            delete_registry_value(winreg.HKEY_CURRENT_USER, r'Software\Classes\Applications\dgenerate_windowed.exe',
                                  'FriendlyAppName')

            # Remove OpenWithProgids entries
            delete_registry_value(winreg.HKEY_CURRENT_USER, r'Software\Classes\.dgen\OpenWithProgids',
                                  'dgenerate.config')
            delete_registry_key(winreg.HKEY_CURRENT_USER, r'Software\Classes\.dgen\OpenWithProgids')

            # Remove shell commands
            delete_registry_key(winreg.HKEY_CURRENT_USER,
                                r'Software\Classes\Applications\dgenerate_windowed.exe\shell\open\command')
            delete_registry_key(winreg.HKEY_CURRENT_USER,
                                r'Software\Classes\Applications\dgenerate_windowed.exe\shell\open')
            delete_registry_key(winreg.HKEY_CURRENT_USER, r'Software\Classes\Applications\dgenerate_windowed.exe\shell')
            delete_registry_key(winreg.HKEY_CURRENT_USER, r'Software\Classes\Applications\dgenerate_windowed.exe')

            delete_registry_key(winreg.HKEY_CURRENT_USER, r'Software\Classes\dgenerate.config\shell\openwithprogids')
            delete_registry_key(winreg.HKEY_CURRENT_USER, r'Software\Classes\dgenerate.config\shell\open\command')
            delete_registry_key(winreg.HKEY_CURRENT_USER, r'Software\Classes\dgenerate.config\shell\open')
            delete_registry_key(winreg.HKEY_CURRENT_USER, r'Software\Classes\dgenerate.config\shell')
            delete_registry_key(winreg.HKEY_CURRENT_USER, r'Software\Classes\dgenerate.config')

            # Remove the .dgen file association
            delete_registry_key(winreg.HKEY_CURRENT_USER, r'Software\Classes\.dgen')

            self.log_callback("✓ Windows file associations removed successfully")

        except Exception as e:
            self.log_callback(f"Error removing Windows file associations: {e}")

    def _broadcast_environment_change(self):
        """Broadcast environment change to Windows."""
        try:
            # Constants for SendMessageTimeout
            HWND_BROADCAST = 0xFFFF
            WM_SETTINGCHANGE = 0x1A
            SMTO_ABORTIFHUNG = 0x0002

            # Broadcast the change to all windows
            result = ctypes.windll.user32.SendMessageTimeoutW(
                HWND_BROADCAST,
                WM_SETTINGCHANGE,
                0,
                "Environment",
                SMTO_ABORTIFHUNG,
                5000,  # 5 second timeout
                ctypes.byref(wintypes.DWORD())
            )

            if result:
                self.log_callback("Environment change broadcasted to Windows")
            else:
                self.log_callback("Warning: Could not broadcast environment change")

        except Exception as e:
            self.log_callback(f"Warning: Could not broadcast environment change: {e}")

    def _register_dgenerate_icon(self):
        """Register dgenerate icon in Windows registry."""
        try:
            # Get windowed executable path
            windowed_exe = self.scripts_dir / "dgenerate_windowed.exe"
            if not windowed_exe.exists():
                return

            # Register icon for dgenerate_windowed.exe
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER,
                                  r'Software\Classes\Applications\dgenerate_windowed.exe\DefaultIcon') as key:
                winreg.SetValue(key, '', winreg.REG_SZ, f'"{windowed_exe},0"')

            self.log_callback("Registered dgenerate icon")

        except Exception as e:
            self.log_callback(f"Warning: Could not register dgenerate icon: {e}")

    def _verify_path_update(self):
        """Verify that the PATH was actually updated."""
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Environment', 0, winreg.KEY_READ) as key:
                try:
                    current_path, _ = winreg.QueryValueEx(key, 'PATH')
                    scripts_dir_str = str(self.scripts_dir)

                    if scripts_dir_str in current_path:
                        self.log_callback("✓ PATH verification successful - dgenerate is now available globally")
                        self.log_callback("  You can run 'dgenerate' from any new command prompt")
                    else:
                        self.log_callback("✓ dgenerate added to system PATH")
                        self.log_callback("  You can now run 'dgenerate' from any new command prompt")
                        self.log_callback(
                            "  Note: Existing terminals may need to be restarted to pick up the PATH change")
                except FileNotFoundError:
                    self.log_callback("Warning: Could not verify PATH update")

        except Exception as e:
            self.log_callback(f"Warning: Could not verify PATH update: {e}")

    def apply_source_patches(self, source_dir: str, version: str | None = None) -> bool:
        """
        Apply Windows-specific patches to the source code before installation.
        
        :param source_dir: Path to the dgenerate source directory
        :param version: Optional version string from SetupAnalyzer
        :return: True if successful or no patches needed, False if failed
        """
        # Apply base patches first (including CUDA specifier patching)
        if not super().apply_source_patches(source_dir, version):
            return False
            
        # Apply Windows-specific patches
        return self._patch_dgenerate_console(source_dir)

    def _patch_dgenerate_console(self, source_dir: str) -> bool:
        """
        Apply Windows AppUserModelID patch to dgenerate console __init__.py.
        This fixes taskbar icon grouping for the console UI on Windows.
        
        :param source_dir: Path to the dgenerate source directory
        :return: True if successful or not needed, False if failed
        """
        # Path to dgenerate console __init__.py
        console_init_path = Path(source_dir) / 'dgenerate' / 'console' / '__init__.py'

        if not console_init_path.exists():
            self.log_callback(f"Console __init__.py not found: {console_init_path}")
            self.log_callback("Skipping AppUserModelID patch (console module not present)")
            return True  # Not an error - some versions might not have console

        try:
            # Read the current file
            with open(console_init_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check if patch is already applied
            if 'SetCurrentProcessExplicitAppUserModelID' in content:
                self.log_callback("AppUserModelID patch already applied")
                return True

            # Prepare the patch
            patch_lines = [
                "import ctypes",
                "ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('dgenerate.python.exe')",
                ""  # Empty line after patch
            ]
            patch_text = '\n'.join(patch_lines)

            # Apply patch at the very top of the file
            patched_content = patch_text + content

            # Write the patched file
            with open(console_init_path, 'w', encoding='utf-8') as f:
                f.write(patched_content)

            self.log_callback(f"✓ Applied Windows AppUserModelID patch to {console_init_path}")
            return True

        except Exception as e:
            self.log_callback(f"Error applying AppUserModelID patch: {e}")
            return False
