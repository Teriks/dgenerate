# -*- mode: python ; coding: utf-8 -*-

import platform
import sys
from pathlib import Path
import os

# Platform detection
current_platform = platform.system().lower()
is_windows = current_platform == 'windows'
is_linux = current_platform == 'linux'
is_macos = current_platform == 'darwin'

# Version info for Windows executable
version_info = None
if is_windows:

    from PyInstaller.utils.win32.versioninfo import (
        VSVersionInfo, FixedFileInfo, StringFileInfo, StringTable, 
        StringStruct, VarFileInfo, VarStruct
    )

    version_info = VSVersionInfo(
        ffi=FixedFileInfo(
            filevers=(0, 1, 0, 0),
            prodvers=(0, 1, 0, 0),
            mask=0x3f,
            flags=0x0,
            OS=0x40004,
            fileType=0x1,
            subtype=0x0,
            date=(0, 0)
        ),
        kids=[
            StringFileInfo([
                StringTable(
                    u'040904B0',
                    [StringStruct(u'CompanyName', u'dgenerate'),
                     StringStruct(u'FileDescription', u'Network installer for dgenerate AI image generation tool'),
                     StringStruct(u'FileVersion', u'0.1.0.0'),
                     StringStruct(u'InternalName', u'dgenerate-network-installer'),
                     StringStruct(u'LegalCopyright', u'Copyright (C) 2024 Teriks'),
                     StringStruct(u'OriginalFilename', u'dgenerate-installer-win.exe'),
                     StringStruct(u'ProductName', u'dgenerate Network Installer'),
                     StringStruct(u'ProductVersion', u'0.1.0.0')])
            ]), 
            VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
        ]
    )

# Determine executable name and icon
if is_windows:
    exe_name = 'dgenerate-network-installer.exe'
else:
    exe_name = 'dgenerate-network-installer'

# Get the project root directory (directory containing this spec file)
# Use SPECPATH which is available in PyInstaller context
project_root = Path(SPECPATH).resolve()

# Set up resources data
resources_dir = project_root / 'network_installer' / 'resources'
datas = []

# Add resources directory if it exists
if resources_dir.exists():
    datas.append((str(resources_dir), 'network_installer/resources'))

# Platform-specific icon handling with conversion
def get_icon_path():
    """Get the appropriate icon path for the current platform."""
    base_icon_path = resources_dir / 'icon.ico'
    
    if not base_icon_path.exists():
        print(f"Warning: Icon file not found at {base_icon_path}")
        return None
    
    if is_windows:
        # Windows: use .ico directly
        return str(base_icon_path)
    else:
        # macOS/Linux: use .png (converted by build process)
        png_path = resources_dir / 'icon.png'
        if png_path.exists():
            return str(png_path)
        
        # Fallback to .ico if PNG not available
        print("Warning: PNG icon not found, falling back to ICO")
        return str(base_icon_path)

# Get the icon path
icon_path = get_icon_path()

block_cipher = None

a = Analysis(
    ['network_installer/main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        # Standard Windows modules
        'winreg',
        'ctypes',
        'ctypes.wintypes'
    ] if is_windows else [],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['setuptools'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=exe_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path,
    version=version_info if is_windows else None,
)
