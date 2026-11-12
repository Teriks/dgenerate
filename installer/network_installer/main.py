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
import os
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
  %(prog)s -b version_6.0.0  # GUI with that development branch selected
  %(prog)s -v v5.0.0         # GUI with that release selected
  %(prog)s -s -v 3.5.0       # Install version 3.5.0 silently (no GUI)
  %(prog)s -s -b master      # Install master branch silently
  %(prog)s -s --msvc         # Also install Visual Studio Build Tools (Windows)
  %(prog)s -u                # Uninstall dgenerate silently
        """
    )

    parser.add_argument('-s', '--silent', action='store_true',
                        help='Silent mode - no GUI, always overwrite existing installation')
    parser.add_argument('-v', '--version', type=str,
                        help='Release tag to install. With -s this installs silently; '
                             'without -s the GUI opens with that release selected')
    parser.add_argument('-b', '--branch', type=str,
                        help='Git branch to install. With -s this installs silently; '
                             'without -s the GUI opens with Development Branch selected')
    parser.add_argument('-e', '--extras', type=str, nargs='+',
                        help='Specify extras to install (e.g. --extras bitsandbytes xllamacpp')
    parser.add_argument('--msvc', action='store_true',
                        help='Install Visual Studio Build Tools for torch.compile (Windows, dgenerate >= 6.0.0). '
                             'With -s this runs Microsoft\'s installer; without -s the GUI '
                             'opens with that option selected')
    parser.add_argument('-u', '--uninstall', action='store_true',
                        help='Uninstall dgenerate (no GUI)')

    return parser.parse_args()


def run_silent_install(version=None, branch=None, extras=None, install_msvc=False):
    """Run silent installation without GUI."""
    try:
        print("dgenerate Network Installer - Silent Mode")
        print("=" * 50)

        if install_msvc and os.name != 'nt':
            print("ERROR: --msvc installs Visual Studio Build Tools and is Windows-only")
            return False

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

        commit_hash, release_branch, is_pre_release = _silent_release_info(
            github_client, version=version, branch=branch)

        # Create installer
        installer = installer_class(log_callback=print, source_dir=source_dir)

        # Check for existing installation
        existing_install = installer.detect_existing_installation()
        if existing_install.exists:
            print("Existing installation detected. Overwriting...")
            # Uninstall existing installation
            if not installer.uninstall():
                print("ERROR: Failed to uninstall existing installation")
                return False
            print("Existing installation removed successfully")

        # Install with default extras
        print("Installing dgenerate...")
        if extras is None:
            extras = _default_extras(source_dir)
        result = installer.install(
            selected_extras=extras,
            commit_hash=commit_hash,
            branch=release_branch,
            is_pre_release=is_pre_release,
            skip_existing_check=True  # Already handled above
        )
        if not result.success:
            print(f"ERROR: Installation failed: {result.error}")
            return False

        if install_msvc:
            from network_installer.msvc_toolchain import install_vc_tools, supports_msvc_toolchain
            package_version = installer.setup_analyzer.version if installer.setup_analyzer else None
            if not supports_msvc_toolchain(package_version):
                print(
                    "Skipping Visual Studio Build Tools. "
                    "That option is for dgenerate 6.0.0 and newer "
                    f"(this source is {package_version})."
                )
            else:
                print("Installing Visual Studio Build Tools...")
                toolchain_ok, toolchain_error = install_vc_tools(print)
                if not toolchain_ok:
                    print(f"ERROR: {toolchain_error}")
                    print("dgenerate itself installed. The C++ compiler installation did not finish.")
                    return False

        print("Installation completed successfully!")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.format_exc())
        return False


def _short_commit(github_client, ref: str) -> str | None:
    info = github_client.get_commit_info(ref)
    sha = info.get('sha') if isinstance(info, dict) else None
    if isinstance(sha, str) and sha:
        return sha[:7]
    print(f"Warning: could not resolve commit for {ref}")
    return None


def _silent_release_info(github_client, version: str | None, branch: str | None):
    """
    Commit, branch, and pre-release flag written into release.json.

    Matches the GUI installer: a branch install is a pre-release, a version
    install follows that release's prerelease flag, and the default install
    tracks master as a stable release.
    """
    if branch:
        print(f"Resolving commit for branch {branch}...")
        return _short_commit(github_client, branch), branch, True

    if version:
        print(f"Resolving commit for release {version}...")
        release = github_client.get_release_by_tag(version)
        is_pre_release = bool(release.get('prerelease', False)) if release else False
        return _short_commit(github_client, version), 'master', is_pre_release

    print("Resolving commit for master...")
    return _short_commit(github_client, 'master'), 'master', False


def _default_extras(source_dir: str) -> list[str]:
    """Extras selected when a silent install does not pass ``--extras``.

    Vulkan is included whenever the downloaded source still defines that
    extra. It is the default preview on Windows and Linux; on macOS the
    OpenGL viewer is used unless ``DGENERATE_CONSOLE_UI_VULKAN=1``.
    """
    setup_py = os.path.join(source_dir, 'setup.py')
    try:
        from network_installer.platform_detection import detect_gpu
        from network_installer.setup_analyzer import SetupAnalyzer
    except ImportError:
        from platform_detection import detect_gpu
        from setup_analyzer import SetupAnalyzer
    analyzer = SetupAnalyzer(setup_py, log_callback=print)
    if not analyzer.load_setup_as_library():
        print('Could not read setup.py extras. Installing the Vulkan preview only.')
        return ['console_ui_vulkan']
    chosen = analyzer.get_recommended_extras(detect_gpu())
    if 'console_ui_vulkan' in analyzer.extras and 'console_ui_vulkan' not in chosen:
        chosen.append('console_ui_vulkan')
    print('Default extras: ' + ', '.join(chosen))
    return chosen


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

    # Force setup.py to use lockfile requires for exact dependency versions
    os.environ['DGENERATE_FORCE_LOCKFILE_REQUIRES'] = '1'

    # Handle command line arguments first
    if args.uninstall:
        success = run_silent_uninstall()
        sys.exit(0 if success else 1)

    if args.version and args.branch:
        print("ERROR: Cannot specify both version (-v) and branch (-b)")
        sys.exit(1)

    if args.silent:
        success = run_silent_install(
            version=args.version,
            branch=args.branch,
            extras=args.extras,
            install_msvc=args.msvc,
        )

        sys.exit(0 if success else 1)

    # GUI, with optional source preselection from -v / -b
    try:
        # Import and run the GUI
        from network_installer.gui import main as gui_main
        gui_main(version=args.version, branch=args.branch, install_msvc=args.msvc)

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
