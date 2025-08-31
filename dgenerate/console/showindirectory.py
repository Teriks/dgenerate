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

import os
import platform
import shutil
import subprocess
import sys


def is_supported() -> bool:
    """Check if file explorer operations are supported on this platform."""
    return platform.system() in {'Windows', 'Darwin'} or shutil.which('nautilus')


def show_in_directory(file_path: str) -> bool:
    """
    Open file explorer and select/highlight the specified file.
    
    :param file_path: Path to the file to highlight
        
    :return: ``True`` if operation succeeded, ``False`` otherwise
    """
    if not file_path or not os.path.exists(file_path):
        return False
    
    try:
        if platform.system() == 'Windows':
            return _open_windows_explorer(file_path)
        elif platform.system() == 'Darwin':
            return _open_macos_finder(file_path)
        else:
            return _open_linux_nautilus(file_path)
    except Exception as e:
        print(f"Failed to open file explorer: {e}", file=sys.stderr)
        return False


def _open_windows_explorer(file_path: str) -> bool:
    """Open Windows Explorer and select the file."""
    try:
        # Correct Windows explorer syntax: /select,filepath (no space after comma)
        norm_path = os.path.normpath(file_path).replace('/', '\\')
        subprocess.Popen(
            ['explorer', f'/select,{norm_path}'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        # Fallback to just opening the directory
        try:
            directory = os.path.dirname(file_path)
            subprocess.Popen(
                ['explorer', directory],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            print(f"Windows explorer file selection failed, opened directory instead: {e}", file=sys.stderr)
            return True
        except Exception as fallback_e:
            print(f"Failed to open directory in explorer: {e}, fallback failed: {fallback_e}", file=sys.stderr)
            return False


def _open_macos_finder(file_path: str) -> bool:
    """Open macOS Finder and select the file."""
    try:
        # Use AppleScript to open Finder and highlight the file
        # Properly escape the file path for AppleScript
        escaped_path = file_path.replace('"', '\\"')
        subprocess.Popen(
            ['osascript', '-e', f'tell application "Finder" to reveal POSIX file "{escaped_path}"'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        subprocess.Popen(
            ['osascript', '-e', 'tell application "Finder" to activate'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        # Fallback to just opening the directory
        try:
            directory = os.path.dirname(file_path)
            subprocess.Popen(
                ['open', directory],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            print(f"macOS Finder file selection failed, opened directory instead: {e}", file=sys.stderr)
            return True
        except Exception as fallback_e:
            print(f"Failed to open directory in Finder: {e}, fallback failed: {fallback_e}", file=sys.stderr)
            return False


def _open_linux_nautilus(file_path: str) -> bool:
    """Open Linux Nautilus and select the file."""
    try:
        # Try to use nautilus to open and select the file
        subprocess.Popen(
            ['nautilus', '--select', file_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        # Fallback to just opening the directory
        try:
            directory = os.path.dirname(file_path)
            subprocess.Popen(
                ['nautilus', directory],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            print(f"Linux nautilus file selection failed, opened directory instead: {e}", file=sys.stderr)
            return True
        except Exception as fallback_e:
            print(f"Failed to open directory in nautilus: {e}, fallback failed: {fallback_e}", file=sys.stderr)
            return False 