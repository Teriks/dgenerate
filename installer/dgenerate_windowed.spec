# -*- mode: python ; coding: utf-8 -*-

import os
import sys
import platform
from pathlib import Path

# Get the project root directory (directory containing this spec file)
# Use SPECPATH which is available in PyInstaller context
project_root = Path(SPECPATH).resolve()

# Define resources directory
resources_dir = project_root / 'network_installer' / 'resources'

def get_icon_path():
    """Get platform-appropriate icon path."""
    is_windows = platform.system().lower() == 'windows'
    
    if is_windows:
        # Windows: use .ico directly
        base_icon_path = resources_dir / 'icon.ico'
        if base_icon_path.exists():
            return str(base_icon_path)
    else:
        # macOS/Linux: use .png (converted by build process)
        png_path = resources_dir / 'icon.png'
        if png_path.exists():
            return str(png_path)
        
        # Fallback to .ico if PNG not available
        base_icon_path = resources_dir / 'icon.ico'
        if base_icon_path.exists():
            print("Warning: PNG icon not found, falling back to ICO")
            return str(base_icon_path)
    
    print("Warning: No icon found")
    return None

# Get the icon path
icon_path = get_icon_path()

# Analysis
a = Analysis(
    ['dgenerate_windowed/dgenerate_windowed.py'],
    pathex=[str(project_root / 'network_installer')],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['setuptools'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# PyInstaller
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='dgenerate_windowed',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # This makes it a windowed application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path,
)
