# -*- mode: python ; coding: utf-8 -*-
import io
import os
import re
from ast import literal_eval

from PyInstaller.building.api import PYZ, EXE, COLLECT
from PyInstaller.building.build_main import Analysis
from PyInstaller.utils.hooks import copy_metadata, collect_data_files, collect_dynamic_libs

dgenerate_init = os.path.join('..', 'dgenerate', '__init__.py')

with io.open(dgenerate_init) as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)


def lockfile_deps():
    poetry_lock_packages = re.compile(r"\[\[package\]\].*?optional = .*?python-versions.*?\n", re.MULTILINE | re.DOTALL)
    with open('../poetry/poetry.lock') as f:
        contents = f.read()
        for match in poetry_lock_packages.findall(contents):
            vals = match.strip().split('\n')[1:]
            d = dict()
            for val in vals:
                left, right = val.split('=', 1)
                right = right.strip()
                if right == 'true':
                    right = True
                elif right == 'false':
                    right = False
                else:
                    right = literal_eval(right)
                d[left.strip()] = right
            yield d


def get_requires(optional=False, exclude: set = None):
    if exclude is None:
        exclude = set()
    return list(f'{dep["name"]}=={dep["version"]}'
                for dep in lockfile_deps()
                if dep['optional'] == optional and dep['name'] not in exclude)


wix_version = re.compile("Name=\"dgenerate\" Version=\".*?\"")

with open('Product.wix', 'r') as f:
    content = f.read()
    content_new = wix_version.sub(f"Name=\"dgenerate\" Version=\"{version.lstrip('v')}\"", content)
    with open('Product.wix', 'w') as f2:
        f2.write(content_new)

block_cipher = None

exclude = {'triton', 'cmake', 'lit', 'opencv-python', 'opencv-contrib-python', 'controlnet-aux'}
# cv2 hook automatic, controlnet-aux has a package name to folder mismatch

requires_extra_data = ['skimage', 'controlnet_aux']
datas = []
binaries = []
module_collection_mode = {}

for package in get_requires(exclude=exclude) + requires_extra_data:
    name = package.split('=')[0]

    if name not in requires_extra_data:
        datas += copy_metadata(name)

    print(f'Data Collection For: {name}')
    module_collection_mode[name] = 'pyz+py'
    datas += collect_data_files(name, include_py_files=True,
                                includes=['**/*.py',
                                          '**/*.pyi',
                                          '**/*.info',
                                          '**/*.c',
                                          '**/*.cpp',
                                          '**/*.cu',
                                          '**/*.cuh',
                                          '**/*.h'])
    binaries += collect_dynamic_libs(name, search_patterns=['*.dll', '*.pyd'])


a = Analysis(
    ['../dgenerate/dgenerate.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
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
