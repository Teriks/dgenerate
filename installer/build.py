#! /usr/bin/env python3
import glob
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
import re
import shutil
import subprocess
import sys
import platform
import argparse

if platform.system() != 'Windows':
    print('Windows installer build is not supported on non windows systems.')
    sys.exit(1)

# Argument parsing
parser = argparse.ArgumentParser(prog='build')
parser.add_argument('--wix-only', nargs='*',
                    help='Rebuild MSI, using build cache instead of a clean build.')
parser.add_argument('--custom-diffusers',
                    help='Custom diffusers pip install package.')
parser.add_argument('--max-archive-size',
                    help='Maximum archive size, split archives larger than this, -v argument of 7zip.', default='1900m')

args = parser.parse_args()


script_dir = os.path.dirname(os.path.abspath(__file__))

# Change to script directory

os.chdir(script_dir)

with open(os.path.join('..', 'dgenerate', '__init__.py')) as _f:
    VERSION = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', _f.read(), re.MULTILINE).group(1)


def create_portable_environment():
    # Create and activate VENV

    subprocess.run([sys.executable, '-m', 'venv', 'venv'])
    venv_path = os.path.join(os.getcwd(), 'venv')

    env = os.environ.copy()
    env['VIRTUAL_ENV'] = venv_path
    env['PATH'] = os.path.join(venv_path, 'Scripts') + os.pathsep + env['PATH']
    env['PYTHONPATH'] = os.path.join(venv_path, 'Lib', 'site-packages')
    env['DGENERATE_FORCE_LOCKFILE_REQUIRES'] = '1'
    python_exe = os.path.join(venv_path, 'Scripts', 'python.exe')

    # Install a dgenerate into VENV

    os.chdir('..')

    if os.path.exists('build'):
        shutil.rmtree('build')

    subprocess.run([python_exe,
                    '-m', 'pip', 'install', '.[win-installer]',
                    '--extra-index-url', 'https://download.pytorch.org/whl/cu124/'], env=env)

    if args.custom_diffusers:
        subprocess.run([python_exe,
                        '-m', 'pip', 'install', args.custom_diffusers, '--force'], env=env)

    os.chdir(script_dir)

    # Build a executable and distributable environment
    subprocess.run([python_exe, '-m', 'PyInstaller', 'dgenerate.spec', '--clean'], env=env)

    gpt4all_lib_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'dist', 'dgenerate', '_internal', 'gpt4all', 'llmodel_DO_NOT_MODIFY', 'build')

    for dll in glob.glob(os.path.join(gpt4all_lib_dir, '*.dll')):

        name = os.path.basename(dll)
        new_path = os.path.join(gpt4all_lib_dir, 'lib'+name)
        print(dll, '->', new_path)
        os.rename(dll, new_path)


# Remove old artifacts

if not args.wix_only:
    for directory in ['venv', 'build', 'dist', 'obj', 'bin']:
        shutil.rmtree(directory, ignore_errors=True)

    create_portable_environment()

    # Create a multi-part zip archive
    os.chdir('dist')
    subprocess.run(['C:\\Program Files\\7-Zip\\7z.exe',
                    f'-v{args.max_archive_size}',
                    'a', f'dgenerate_{VERSION}_portable.zip', 'dgenerate'])
    os.chdir(script_dir)
else:
    if not os.path.exists('venv'):
        create_portable_environment()

# Build WiX installer
subprocess.run(['dotnet', 'build', 'dgenerate.wixproj', '--configuration', 'Release'])

if not args.wix_only:
    # Create a multi-part zip archive
    os.chdir('bin\\Release')
    cab_files = [f for f in os.listdir() if f.endswith('.cab')]
    subprocess.run(
        ['C:\\Program Files\\7-Zip\\7z.exe',
         f'-v{args.max_archive_size}',
         'a', f'dgenerate_{VERSION}_installer.zip', 'dgenerate.msi'] + cab_files)
    os.chdir(script_dir)

    # Move all parts of the multi-part zip archive to 'bin\\Release'
    for root, dirs, files in os.walk('dist'):
        for file in files:
            if file.startswith(f'dgenerate_{VERSION}_portable.zip'):
                shutil.move(os.path.join(root, file), 'bin\\Release')
