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
UV-based Python environment setup and dgenerate installer.
Unified installer that handles both low-level uv operations and high-level installation orchestration.
"""

import json
import os
import platform
import re
from network_installer.platform_handlers import (
    WindowsPlatformHandler,
    MacOSPlatformHandler,
    LinuxPlatformHandler
)
from network_installer.setup_analyzer import SetupAnalyzer
from pathlib import Path
from typing import Optional, List, Dict


class InstallationResult:
    """Result object for installation operations."""

    def __init__(
            self,
            success: bool,
            desktop_shortcut_created: bool = False,
            installation_info: Optional[Dict] = None,
            error: Optional[str] = None
    ):
        self.success = success
        self.desktop_shortcut_created = desktop_shortcut_created
        self.installation_info = installation_info or {}
        self.error = error


class UvInstaller:
    """
    Unified dgenerate installer using uv.
    Handles both low-level uv operations and high-level installation orchestration.
    """

    def __init__(self, log_callback=None, source_dir=None):
        """
        Initialize the UvInstaller.

        :param log_callback: Optional callback function for logging
        :param source_dir: Optional path to dgenerate source directory
        """
        self.log_callback = log_callback or print
        self.system = platform.system().lower()
        self.source_dir = source_dir

        # Set up installation directories
        if self.system == 'windows':
            # Use AppData/Roaming for Windows (consistent with Python pip --user)
            appdata_local = os.environ.get('APPDATA', os.path.expanduser('~/AppData/Local'))
            self.install_base = Path(appdata_local) / 'dgenerate'
        else:
            # Use ~/.local for Unix systems
            self.install_base = Path.home() / '.local' / 'dgenerate'

        self.uv_dir = self.install_base / 'uv'
        self.venv_dir = self.install_base / 'venv'

        # Note: Don't create directories here - only create them when needed during installation
        # Creating them here causes false positive detection of existing installations

        # Initialize platform-specific handler
        self.platform_handler = self._create_platform_handler()

        # Initialize setup analyzer if source directory is provided
        self.setup_analyzer = None
        self._setup_analyzed = False
        if source_dir:
            setup_py_path = os.path.join(source_dir, 'setup.py')
            if os.path.exists(setup_py_path):
                self.setup_analyzer = SetupAnalyzer(setup_py_path, log_callback=self.log_callback)

    def _create_platform_handler(self):
        """Create the appropriate platform handler."""
        if self.system == 'windows':
            return WindowsPlatformHandler(self)
        elif self.system == 'darwin':
            return MacOSPlatformHandler(self)
        else:
            return LinuxPlatformHandler(self)

    @property
    def scripts_dir(self) -> Path:
        """Get the scripts directory in the virtual environment."""
        return self.platform_handler.get_venv_scripts_dir()

    # ===========================================
    # High-level installation orchestration
    # ===========================================

    def install(
            self,
            selected_extras: List[str],
            torch_index_url: Optional[str] = None,
            commit_hash: str = None,
            branch: str = None,
            is_pre_release: bool = False,
            skip_existing_check: bool = False
    ) -> InstallationResult:
        """
        Install dgenerate with the specified extras.

        :param selected_extras: List of extras to install
        :param torch_index_url: Optional PyTorch index URL
        :param commit_hash: Git commit hash for release.json
        :param branch: Git branch name for release.json
        :param is_pre_release: Whether this is a pre-release build
        :return: InstallationResult object
        """
        try:
            # Step 1: Check for existing installations (unless already handled by caller)
            if not skip_existing_check:
                self.log_callback("Checking for existing dgenerate installations...")
                existing_install = self.detect_existing_installation()

                if existing_install['exists']:
                    # Clean up existing installation
                    self.log_callback("Cleaning up existing installation...")
                    if not self.cleanup_existing_installations():
                        self.log_callback(
                            "Failed to clean up existing installation. Please close any running dgenerate processes and try again.")
                        return InstallationResult(success=False, error="Failed to clean up existing installation")
                else:
                    self.log_callback("No existing dgenerate network installation found")
            else:
                self.log_callback("Skipping existing installation check (already handled)")

            # Step 2: Load setup.py to get dependencies (if not already done)
            if self.setup_analyzer and not self._setup_analyzed:
                self.log_callback("Analyzing dgenerate setup...")
                if not self.setup_analyzer.load_setup_as_library():
                    self.log_callback("Failed to load setup.py")
                    return InstallationResult(success=False, error="Failed to load setup.py")
                self._setup_analyzed = True

            # Step 3: Skip Python version compatibility check since uv will handle Python installation
            self.log_callback(
                "Skipping Python version compatibility check (uv will install required Python version)...")

            # Step 4: Find or install uv
            self.log_callback("Setting up uv...")
            uv_exe = self.find_or_install_uv()
            if not uv_exe:
                self.log_callback("Failed to set up uv")
                return InstallationResult(success=False, error="Failed to set up uv")

            # Step 5: Create virtual environment
            if self.setup_analyzer:
                # Get the recommended Python version from the source analysis
                python_req = self.setup_analyzer.get_python_requirement()
                recommended_python = self.setup_analyzer.get_recommended_python_version()
                if python_req:
                    self.log_callback(f"Detected Python requirement: {python_req}")
                self.log_callback(f"Using recommended Python version: {recommended_python}")
            else:
                recommended_python = "3.12"  # Default fallback
                self.log_callback(f"Using default Python version: {recommended_python}")

            self.log_callback("Creating Python virtual environment...")

            # Try to create venv with recommended version first, then fallbacks if needed
            if not self._create_venv_with_fallbacks(uv_exe, recommended_python):
                self.log_callback("Failed to create virtual environment with any available Python version")
                return InstallationResult(success=False, error="Failed to create virtual environment")

            # Step 6: Generate release.json if commit and branch info provided
            if commit_hash and branch:
                self.log_callback("Generating release.json...")
                if not self.generate_release_json(commit_hash, branch, is_pre_release):
                    self.log_callback("Warning: Failed to generate release.json, continuing with installation")

            # Step 6.5: Apply platform-specific patches
            self.log_callback("Applying platform-specific patches...")
            if not self.platform_handler.apply_source_patches(self.source_dir):
                self.log_callback("Warning: Failed to apply some platform-specific patches")

            # Step 7: Install dgenerate
            self.log_callback("Installing dgenerate...")
            if not self.install_dgenerate(uv_exe, self.source_dir, selected_extras, torch_index_url):
                self.log_callback("Failed to install dgenerate")
                return InstallationResult(success=False, error="Failed to install dgenerate")

            # Step 8: Add dgenerate Scripts to PATH (without activating venv)
            self.log_callback("Adding dgenerate to system PATH...")
            if not self.add_scripts_to_path():
                self.log_callback("Warning: Failed to add dgenerate to PATH. You may need to add it manually.")

            # Step 9: Create desktop shortcut (only for dgenerate v3.5.0+)
            desktop_shortcut_created = False
            if self._should_create_desktop_shortcut():
                self.log_callback("Creating desktop shortcut...")
                if self.create_desktop_shortcut():
                    desktop_shortcut_created = True
                else:
                    self.log_callback("Warning: Failed to create desktop shortcut.")
            else:
                self.log_callback("Skipping desktop shortcut creation (not needed for this dgenerate version)")

            # Step 10: Create file associations for .dgen files
            self.log_callback("Creating file associations for .dgen files...")
            if not self.create_file_associations():
                self.log_callback("Warning: Failed to create file associations.")

            self.log_callback("Installation completed successfully!")

            # Display installation info
            info = self.get_installation_info()
            self.log_callback("\nInstallation Details:")
            self.log_callback(f"  Installation directory: {info['install_base']}")
            self.log_callback(f"  Virtual environment: {info['venv_dir']}")
            self.log_callback(f"  dgenerate executable: {info['dgenerate_exe']}")
            self.log_callback(f"  dgenerate is now available globally - you can run 'dgenerate' from any terminal")

            # Clean up setup analysis modifications
            if self.setup_analyzer:
                self.setup_analyzer.cleanup()

            # Clean up environment variables to prevent global contamination
            self.log_callback("Cleaning up environment variables...")
            self.cleanup_environment()

            # Return installation results
            return InstallationResult(
                success=True,
                desktop_shortcut_created=desktop_shortcut_created,
                installation_info=info
            )

        except Exception as e:
            self.log_callback(f"Installation failed: {e}")
            # Clean up setup analysis modifications even on failure
            if self.setup_analyzer:
                self.setup_analyzer.cleanup()
            # Clean up environment variables even on failure
            self.cleanup_environment()
            return InstallationResult(
                success=False,
                desktop_shortcut_created=False,
                error=str(e)
            )

    def mark_setup_analyzed(self):
        """Mark that setup analysis has been completed."""
        self._setup_analyzed = True

    def generate_release_json(
            self,
            commit_hash: str,
            branch: str,
            is_pre_release: bool = False
    ) -> bool:
        """
        Generate release.json file with version, commit, branch, and pre_release information.

        :param commit_hash: Git commit hash (short)
        :param branch: Git branch name
        :param is_pre_release: Whether this is a pre-release build
        :return: True if successful, False otherwise
        """
        try:
            if not self.setup_analyzer:
                self.log_callback("Setup analyzer not available for release.json generation")
                return False

            # Use version from SetupAnalyzer
            version = self.setup_analyzer.version
            if not version:
                self.log_callback("Could not get version from SetupAnalyzer")
                return False

            # Create release.json content
            release_data = {
                "version": version,
                "commit": commit_hash,
                "branch": branch,
                "pre_release": is_pre_release
            }

            # Create the release.json file in the dgenerate/dgenerate directory
            dgenerate_dir = os.path.join(self.source_dir, 'dgenerate')
            os.makedirs(dgenerate_dir, exist_ok=True)
            release_json_path = os.path.join(dgenerate_dir, 'release.json')

            with open(release_json_path, 'w', encoding='utf-8') as f:
                json.dump(release_data, f, indent=2, ensure_ascii=False)

            self.log_callback(f"Generated release.json: {release_json_path}")
            self.log_callback(f"  Version: {version}")
            self.log_callback(f"  Commit: {commit_hash}")
            self.log_callback(f"  Branch: {branch}")
            self.log_callback(f"  Pre-release: {is_pre_release}")

            return True

        except Exception as e:
            self.log_callback(f"Error generating release.json: {e}")
            return False

    def _should_create_desktop_shortcut(self) -> bool:
        """
        Check if desktop shortcut should be created based on dgenerate version and console directory existence.
        
        Desktop shortcuts are only needed for dgenerate v3.5.0+ which introduced the console UI,
        and only if the dgenerate/console directory exists in the source tree.
        
        :return: True if desktop shortcut should be created, False otherwise
        """
        try:
            if not self.setup_analyzer:
                self.log_callback("Setup analyzer not available, skipping desktop shortcut")
                return False

            # First check if the console directory exists in the source tree
            console_dir = os.path.join(self.source_dir, "dgenerate", "console")
            if not os.path.exists(console_dir):
                self.log_callback(f"Console directory not found at {console_dir}, skipping desktop shortcut")
                return False

            # Then check version compatibility
            version = self.setup_analyzer.version
            if not version:
                self.log_callback("Warning: Could not determine dgenerate version, skipping desktop shortcut")
                return False

            # Parse version string (e.g., "3.5.0", "4.0.0", "5.1.2")
            version_match = re.match(r'^(\d+)\.(\d+)\.(\d+)', version)
            if not version_match:
                self.log_callback(f"Warning: Could not parse version '{version}', skipping desktop shortcut")
                return False

            major, minor, patch = map(int, version_match.groups())

            # Desktop shortcuts are only needed for v3.5.0+
            if major > 3 or (major == 3 and minor >= 5):
                self.log_callback(
                    f"dgenerate version {version} supports console UI and console directory exists, will create desktop shortcut")
                return True
            else:
                self.log_callback(f"dgenerate version {version} does not support console UI, skipping desktop shortcut")
                return False

        except Exception as e:
            self.log_callback(f"Error checking version for desktop shortcut: {e}")
            return False

    def _create_venv_with_fallbacks(
            self,
            uv_exe,
            primary_version: str
    ) -> bool:
        """
        Try to create venv with primary version, then fallback versions if needed.
        
        :param uv_exe: Path to uv executable
        :param primary_version: Primary Python version to try first
        :return: True if venv was created successfully with any version
        """
        # Try primary version first
        if self.create_venv(uv_exe, python_version=primary_version):
            return True

        # If primary failed, try fallback versions
        self.log_callback(f"Primary Python version {primary_version} failed, trying fallback versions...")

        # For dgenerate 5.0.0+, try 3.13, 3.12, 3.11
        fallback_versions = ["3.13", "3.12", "3.11"]

        # Remove the primary version that already failed
        fallback_versions = [v for v in fallback_versions if v != primary_version]

        # Try fallback versions
        for fallback_version in fallback_versions:
            self.log_callback(f"Trying Python {fallback_version}...")
            if self.create_venv(uv_exe, python_version=fallback_version):
                self.log_callback(f"Successfully created venv with Python {fallback_version}")
                return True
            else:
                self.log_callback(f"Python {fallback_version} also failed")

        return False

    # ===========================================
    # Low-level uv operations (delegated to platform handlers)
    # ===========================================

    def get_uv_download_url(self) -> str:
        """Get the appropriate uv download URL for the current platform."""
        return self.platform_handler.get_uv_download_url()

    def find_or_install_uv(self) -> Optional[Path]:
        """
        Find uv in PATH or download and install it.
        
        :return: Path to the uv executable or None if failed
        """
        return self.platform_handler.find_or_install_uv()

    def download_and_extract_uv(self) -> Optional[Path]:
        """
        Download and extract uv for the current platform.
        
        :return: Path to the uv executable or None if failed
        """
        return self.platform_handler.download_and_extract_uv()

    def create_venv(self, uv_exe: Path, python_version: str = None) -> bool:
        """
        Create a Python virtual environment using uv.
        
        :param uv_exe: Path to uv executable
        :param python_version: Python version to use (e.g., "3.11", "3.12")
        :return: True if successful, False otherwise
        """
        return self.platform_handler.create_venv(uv_exe, python_version)

    def install_dgenerate(self, uv_exe: Path, source_dir: str, selected_extras: List[str],
                          torch_index_url: Optional[str] = None) -> bool:
        """
        Install dgenerate using uv.
        
        :param uv_exe: Path to uv executable
        :param source_dir: Directory containing dgenerate source code
        :param selected_extras: List of extras to install
        :param torch_index_url: Optional PyTorch index URL
        :return: True if successful, False otherwise
        """
        return self.platform_handler.install_dgenerate(uv_exe, source_dir, selected_extras, torch_index_url)

    def detect_existing_installation(self) -> dict:
        """
        Detect if dgenerate is already installed.
        
        :return: Dictionary with detection results
        """
        return self.platform_handler.detect_existing_installation()

    def cleanup_existing_installations(self) -> bool:
        """
        Clean up existing dgenerate installations.
        
        :return: True if successful, False otherwise
        """
        return self.platform_handler.cleanup_existing_installations()

    def uninstall_completely(self) -> bool:
        """
        Completely uninstall dgenerate.
        
        :return: True if successful, False otherwise
        """
        return self.platform_handler.uninstall_completely()

    def cleanup_environment(self) -> bool:
        """
        Clean up environment variables that might have been set during installation.
        
        :return: True if successful, False otherwise
        """
        return self.platform_handler.cleanup_environment()

    def get_installation_info(self) -> dict:
        """
        Get information about the current installation.
        
        :return: Dictionary containing installation information
        """
        return self.platform_handler.get_installation_info()

    def add_scripts_to_path(self) -> bool:
        """
        Add only the Scripts directory to the system PATH (without activating venv).
        
        :return: True if successful, False otherwise
        """
        return self.platform_handler.add_scripts_to_path()

    def create_desktop_shortcut(self) -> bool:
        """
        Create desktop shortcut for dgenerate console.
        
        :return: True if successful, False otherwise
        """
        return self.platform_handler.create_desktop_shortcut()

    def create_file_associations(self) -> bool:
        """
        Create file associations for .dgen files.
        
        :return: True if successful, False otherwise
        """
        return self.platform_handler.create_file_associations()

    def get_venv_python(self) -> Path:
        """Get the path to the Python executable in the virtual environment."""
        return self.platform_handler.get_venv_python()
