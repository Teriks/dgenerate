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
Main entry point for the dgenerate network installer.
"""

import argparse
import platform
import sys
import tempfile
import traceback


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="dgenerate Network Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Start GUI installer
  %(prog)s -s -v 3.5.0       # Install version 3.5.0 silently (no GUI)
  %(prog)s -s -b master      # Install master branch silently
  %(prog)s -u                # Uninstall dgenerate silently
        """
    )

    parser.add_argument('-s', '--silent', action='store_true',
                        help='Silent mode - no GUI, always overwrite existing installation')
    parser.add_argument('-v', '--version', type=str,
                        help='Install specific version (e.g., 3.5.0)')
    parser.add_argument('-b', '--branch', type=str,
                        help='Install specific branch (e.g., master)')
    parser.add_argument('-e', '--extras', type=str, nargs='+',
                        help='Specify extras to install (e.g. --extras bitsandbytes gpt4all')
    parser.add_argument('-u', '--uninstall', action='store_true',
                        help='Uninstall dgenerate (no GUI)')

    return parser.parse_args()


def run_silent_install(version=None, branch=None, extras=None):
    """Run silent installation without GUI."""
    try:
        print("dgenerate Network Installer - Silent Mode")
        print("=" * 50)

        # Import non-GUI components
        from network_installer.github_client import GitHubClient
        
        # Use UV installer for all platforms with enhanced Linux font support
        print("Using UV installer with enhanced Linux font support")
        from network_installer.uv_installer import UvInstaller
        installer_class = UvInstaller

        # Create temporary directory for source
        temp_dir = tempfile.mkdtemp(prefix="dgenerate_install_")

        # Initialize GitHub client
        github_client = GitHubClient()

        # Determine source reference
        if version:
            ref = version
            print(f"Installing version: {version}")
        elif branch:
            ref = branch
            print(f"Installing branch: {branch}")
        else:
            ref = "master"
            print("Installing latest release (master)")

        # Download source
        print("Downloading source code...")
        source_dir = github_client.download_source_archive(ref, temp_dir,
                                                           lambda downloaded, total: None)

        if not source_dir:
            print("ERROR: Failed to download source code")
            return False

        print(f"Source downloaded to: {source_dir}")

        # Create installer
        installer = installer_class(log_callback=print, source_dir=source_dir)

        # Check for existing installation
        existing_install = installer.detect_existing_installation()
        if existing_install.exists:
            print("Existing installation detected. Overwriting...")
            # Uninstall existing installation
            if not installer.uninstall_completely():
                print("ERROR: Failed to uninstall existing installation")
                return False
            print("Existing installation removed successfully")

        # Install with default extras
        print("Installing dgenerate...")
        # Use default extras for silent installation
        result = installer.install(
            selected_extras=extras or [],
            skip_existing_check=True  # Already handled above
        )
        if not result.success:
            print(f"ERROR: Installation failed: {result.error}")
            return False

        print("Installation completed successfully!")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.format_exc())
        return False


def run_silent_uninstall():
    """Run silent uninstallation without GUI."""
    try:
        print("dgenerate Network Installer - Silent Uninstall")
        print("=" * 50)

        # Use UV installer for all platforms
        print("Using UV uninstaller")
        from network_installer.uv_installer import UvInstaller
        installer_class = UvInstaller

        # Create temporary installer to access uninstall functionality
        with tempfile.TemporaryDirectory() as temp_dir:
            installer = installer_class(log_callback=print, source_dir=temp_dir)

            # Check for existing installation
            existing_install = installer.detect_existing_installation()
            if not existing_install.exists:
                print("No existing dgenerate installation found")
                return True

            print("Existing installation detected. Uninstalling...")

            # Uninstall
            if not installer.uninstall():
                print("ERROR: Failed to uninstall")
                return False

            print("Uninstallation completed successfully!")
            return True

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.format_exc())
        return False


def main():
    """Main entry point."""
    args = parse_arguments()

    # Handle command line arguments first
    if args.uninstall:
        success = run_silent_uninstall()
        sys.exit(0 if success else 1)

    if args.silent:
        # Validate arguments
        if args.version and args.branch:
            print("ERROR: Cannot specify both version (-v) and branch (-b)")
            sys.exit(1)

        success = run_silent_install(
            version=args.version,
            branch=args.branch,
            extras=args.extras or []
        )

        sys.exit(0 if success else 1)

    # If no command line arguments, run GUI
    try:
        # Import and run the GUI
        from network_installer.gui import main as gui_main
        gui_main()

    except Exception as e:
        # Show error dialog if GUI fails to start
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()  # Hide the main window

            error_msg = f"Failed to start dgenerate Network Installer:\n\n{str(e)}\n\n{traceback.format_exc()}"
            messagebox.showerror("Startup Error", error_msg)

        except:
            # Fallback to console output
            print(f"Failed to start dgenerate Network Installer: {e}")
            print(traceback.format_exc())
            input("Press Enter to exit...")

        sys.exit(1)


if __name__ == "__main__":
    main()
