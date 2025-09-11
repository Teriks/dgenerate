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
Windows-only PyInstaller launcher for dgenerate.
Prevents console windows from appearing when launching dgenerate from desktop shortcuts.
"""

import sys
import subprocess
from pathlib import Path
from tkinter import messagebox

def find_venv_from_exe():
    """Auto-detect Windows virtual environment from the executable's location."""
    if getattr(sys, 'frozen', False):
        exe_dir = Path(sys.executable).parent
    else:
        exe_dir = Path(__file__).parent
    
    # Check if we're in a bin directory (network installer layout)
    if exe_dir.name == 'bin':
        # Look for venv as a sibling directory
        potential_venv = exe_dir.parent / 'venv'
        if (potential_venv / "Scripts" / "python.exe").exists():
            return potential_venv
    
    # Fallback: original behavior - walk up from executable location
    current_dir = exe_dir
    while current_dir.parent != current_dir:
        if (current_dir / "Scripts" / "python.exe").exists():
            return current_dir
        current_dir = current_dir.parent
    return None

def main():
    # Auto-detect virtual environment
    venv_dir = find_venv_from_exe()
    if not venv_dir:
        messagebox.showerror(
            "dgenerate Error",
            "Could not find dgenerate virtual environment.\n\n"
            "Please reinstall dgenerate or check your installation."
        )
        sys.exit(1)

    # Windows-only: Scripts directory and executable
    scripts_dir = venv_dir / "Scripts"
    dgenerate_exe = scripts_dir / "dgenerate.exe"

    if not dgenerate_exe.exists():
        messagebox.showerror(
            "dgenerate Error",
            f"dgenerate executable not found at: {dgenerate_exe}\n\n"
            "Please reinstall dgenerate or check your installation."
        )
        sys.exit(1)

    # Prepare subprocess arguments
    args = sys.argv[1:]
    cmd = [str(dgenerate_exe)] + args

    # Windows-specific: Hide console window while allowing GUI windows to display
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE

    # Set working directory to user home for a sane default
    user_home = Path.home()
    
    process = subprocess.Popen(
        cmd,
        cwd=str(user_home),
        startupinfo=startupinfo
    )

    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()

if __name__ == "__main__":
    main()
