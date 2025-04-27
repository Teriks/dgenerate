# -*- mode: python ; coding: utf-8 -*-
import glob
import os
import re
from importlib.machinery import SourceFileLoader
from importlib.metadata import PackageNotFoundError

from PyInstaller.building.api import PYZ, EXE, COLLECT
from PyInstaller.building.build_main import Analysis
from PyInstaller.utils.hooks import copy_metadata, collect_data_files, collect_dynamic_libs

setup = SourceFileLoader('setup_as_library', '../setup.py').load_module()

wix_version = re.compile("Name=\"dgenerate\" Version=\".*?\"")

with open('Product.wix', 'r') as f:
    content = f.read()
    content_new = wix_version.sub(f"Name=\"dgenerate\" Version=\"{setup.VERSION.lstrip('v')}\"", content)
    with open('Product.wix', 'w') as f2:
        f2.write(content_new)

block_cipher = None

# the cv2 hook works properly, other mentioned deps just are not needed
exclude_from_collection = {
    'cmake',
    'lit',
    'opencv-python',
    'opencv-contrib-python',
    'tzdata'
}

# these packages have names that do not align with their folder name
# or have needed source files
requires_extra_data_forced = [
    'dgenerate',
    'bitsandbytes',
    'triton',
    'gpt4all',
    'fake_useragent',
    'fontTools',
    'antlr4',
    'bs4',
    'PIL',
    'dateutil',
    'pkg_resources',
    'win32',
    'win32com',
    'win32comext',
    'win32ctypes']

# these packages require source code to work properly
require_source = {
    'dgenerate',
    'transformers',
    'diffusers',
    'triton',
    'bitsandbytes',
    'pandas',
    'ultralytics',
    'spandrel',
    'spandrel_extra_arches',
    'timm',
    'torch',
    'torchvision',
    'torchsde',
    'numpy',
    'scipy',
    'matplotlib',
    'pandas'
}

datas = []
binaries = []
module_collection_mode = {}

required_package_names = \
    list(i for i in setup.get_poetry_lockfile_as_pip_requires().keys() if i not in exclude_from_collection) \
    + requires_extra_data_forced

for package_name in required_package_names:

    if package_name not in requires_extra_data_forced:
        try:
            datas += copy_metadata(package_name)
        except PackageNotFoundError:
            print(f'copy_metadata failed for {package_name}')

    if '-' in package_name:
        # collect data files cannot find this by the
        # actual package name
        package_name = package_name.replace('-', '_')

    print(f'Data Collection For: {package_name}')
    include_source = package_name in require_source
    module_collection_mode[package_name] = 'pyz+py' if include_source else 'pyz'
    datas += collect_data_files(
        package_name,
        include_py_files=include_source,
        includes=['**/*.info',
                  '**/*.recipe',
                  '**/*.csv',
                  '**/*.marisa',
                  '**/*.txt',
                  '**/*.c',
                  '**/*.cpp',
                  '**/*.cu',
                  '**/*.cuh',
                  '**/*.h',
                  '**/*.json',
                  '**/*.jsonl',
                  '**/*.toml',
                  '**/*.yaml',
                  '**/*.yml',
                  '**/*.exe'] +
                 (['**/*.py', '**/*.pyi']
                  if include_source else []))
    binaries += collect_dynamic_libs(package_name, search_patterns=['*.dll', '*.pyd'])

a = Analysis(
    ['../dgenerate/__winentry__.py'],
    pathex=[],
    binaries=binaries,
    datas=datas + [('../dgenerate/icon.ico', './dgenerate'),
                   ('../dgenerate/config_icon.ico', './dgenerate')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tzdata'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    module_collection_mode=module_collection_mode
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='dgenerate',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='../dgenerate/icon.ico'
)

exe2 = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='dgenerate_windowed',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='../dgenerate/icon.ico'
)

coll = COLLECT(
    exe,
    exe2,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='dgenerate',
)
