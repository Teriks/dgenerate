# -*- mode: python ; coding: utf-8 -*-

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

exclude_from_forced_collection = {'triton', 'cmake', 'lit', 'opencv-python', 'opencv-contrib-python'}
# the cv2 hook works properly, other mentioned deps just are not needed

requires_extra_data_forced = [
    'dgenerate',
    'skimage',
    'fontTools',
    'antlr4',
    'bs4',
    'PIL',
    'dateutil',
    'win32',
    'win32com',
    'win32comext',
    'win32ctypes']
# these packages have names that do not align with their folder name
# or have needed source files

datas = []
binaries = []
module_collection_mode = {}

required_package_names = \
    list(setup.get_poetry_lockfile_as_pip_requires(
        exclude=exclude_from_forced_collection).keys()) + requires_extra_data_forced

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
    module_collection_mode[package_name] = 'pyz+py'
    datas += collect_data_files(package_name, include_py_files=True,
                                includes=['**/*.py',
                                          '**/*.pyi',
                                          '**/*.info',
                                          '**/*.c',
                                          '**/*.cpp',
                                          '**/*.cu',
                                          '**/*.cuh',
                                          '**/*.h'])
    binaries += collect_dynamic_libs(package_name, search_patterns=['*.dll', '*.pyd'])

# need the browser data
datas += collect_data_files('fake_useragent', include_py_files=False, includes=['**/*.json'])

a = Analysis(
    ['../dgenerate/dgenerate.py'],
    pathex=[],
    binaries=binaries,
    datas=datas + [('../dgenerate/icon.ico', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='../dgenerate/icon.ico'
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='dgenerate',
)
