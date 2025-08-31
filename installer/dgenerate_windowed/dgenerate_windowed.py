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
PyInstaller launcher for dgenerate.
Creates a hidden Tkinter window to ensure the taskbar uses the EXE icon.
"""

import sys
import os
import subprocess
import platform
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

def find_venv_from_exe():
    """Auto-detect virtual environment from the executable's location."""
    if getattr(sys, 'frozen', False):
        exe_dir = Path(sys.executable).parent
    else:
        exe_dir = Path(__file__).parent
    
    # Check if we're in a bin directory (network installer layout)
    if exe_dir.name == 'bin':
        # Look for venv as a sibling directory
        potential_venv = exe_dir.parent / 'venv'
        if (potential_venv / "Scripts" / "python.exe").exists() or (potential_venv / "bin" / "python").exists():
            return potential_venv
    
    # Fallback: original behavior - walk up from executable location
    current_dir = exe_dir
    while current_dir.parent != current_dir:
        if (current_dir / "Scripts" / "python.exe").exists() or (current_dir / "bin" / "python").exists():
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

    # Determine scripts directory and executable
    if platform.system().lower() == 'windows':
        scripts_dir = venv_dir / "Scripts"
        dgenerate_exe = scripts_dir / "dgenerate.exe"
    else:
        scripts_dir = venv_dir / "bin"
        dgenerate_exe = scripts_dir / "dgenerate"

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

    # On Windows, use the proper combination for GUI applications
    # This prevents console window while allowing GUI windows to display
    if platform.system().lower() == 'windows':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        # Use 0 for creationflags to allow normal GUI window creation
        creationflags = 0
    else:
        startupinfo = None
        creationflags = 0

    # Set up clean environment for the subprocess
    env = os.environ.copy()
    
    # Add venv Scripts/bin directory to PATH
    # This ensures all dependencies can be found when running dgenerate
    if str(scripts_dir) not in env.get('PATH', ''):
        if platform.system().lower() == 'windows':
            env['PATH'] = f"{scripts_dir};{env.get('PATH', '')}"
        else:
            env['PATH'] = f"{scripts_dir}:{env.get('PATH', '')}"
    
    # Remove PyInstaller-specific environment variables that can interfere with Tcl/Tk
    # These variables point to PyInstaller's bundled Tcl/Tk which conflicts with the system's
    pyinstaller_vars = [
        '_MEIPASS',
        '_MEIPASS2', 
        'TCL_LIBRARY',
        'TK_LIBRARY',
        'TKPATH',
        'TCLPATH'
    ]
    
    for var in pyinstaller_vars:
        env.pop(var, None)

    # Set working directory to user home for a sane default
    user_home = Path.home()
    
    process = subprocess.Popen(
        cmd,
        cwd=str(user_home),
        startupinfo=startupinfo,
        creationflags=creationflags,
        env=env
    )

    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()

if __name__ == "__main__":
    main()
