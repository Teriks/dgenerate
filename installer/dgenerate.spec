# -*- mode: python ; coding: utf-8 -*-
import io
import re
from ast import literal_eval

from PyInstaller.building.api import PYZ, EXE, COLLECT
from PyInstaller.building.build_main import Analysis
from PyInstaller.utils.hooks import copy_metadata, collect_data_files

version = ''
with io.open('../dgenerate/__init__.py') as f:
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


def get_requires(optional=False, exclude={}):
    return list(f'{dep["name"]}=={dep["version"]}' for dep in lockfile_deps()
                if dep['optional'] == optional and dep['name'] not in exclude)


wix_version = re.compile("Name=\"dgenerate\" Version=\".*?\"")

with open('Product.wix', 'r') as f:
    content = f.read()
    content_new = wix_version.sub(f"Name=\"dgenerate\" Version=\"{version.lstrip('v')}\"", content)
    with open('Product.wix', 'w') as f2:
        f2.write(content_new)

block_cipher = None

datas = []

for package in get_requires(exclude={'triton', 'cmake', 'lit'} if os.name == 'nt' else {}):
    print(package)
    datas += copy_metadata(package.split('=')[0])

need_data = ['accelerate',
             'transformers',
             'pytorch_lightning',
             'aiosignal',
             'async_timeout',
             'attrs',
             'colorama',
             'diffusers',
             'fake_useragent',
             'filelock',
             'fsspec',
             'huggingface_hub',
             'idna',
             'importlib_metadata',
             'lightning_fabric',
             'lightning_utilities',
             'mpmath',
             'networkx',
             'omegaconf',
             'packaging',
             'requests',
             'sympy',
             'torchmetrics',
             'tqdm',
             'urllib3',
             'wheel',
             'zipp']

module_collection_mode = {}

for i in need_data:
    module_collection_mode[i] = 'pyz+py'
    datas += collect_data_files(i, include_py_files=True,
                                includes=['**/*.py', '**/*.info', '**/*.c', '**/*.cpp', '**/*.cu', '**/*.cuh',
                                          '**/*.h'])

a = Analysis(
    ['../dgenerate/dgenerate.py'],
    pathex=[],
    binaries=[],
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
