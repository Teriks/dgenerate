#! /usr/bin/env python3

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
import shutil
import subprocess
import sys
import platform

if platform.system() != 'Windows':
    print('Windows installer build is not supported on non windows systems.')
    sys.exit(1)

wix_only = '--wix-only' in sys.argv

script_dir = os.path.dirname(os.path.abspath(__file__))

# Change to script directory

os.chdir(script_dir)


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
    subprocess.run([python_exe,
                    '-m', 'pip', 'install', '.[win-installer, ncnn]',
                    '--extra-index-url', 'https://download.pytorch.org/whl/cu121/'], env=env)
    os.chdir(script_dir)

    # Build a executable and distributable environment
    subprocess.run([python_exe, '-m', 'PyInstaller', 'dgenerate.spec', '--clean'], env=env)


# Remove old artifacts

if not wix_only:
    for directory in ['venv', 'build', 'dist', 'obj', 'bin']:
        shutil.rmtree(directory, ignore_errors=True)

    create_portable_environment()

    # Create a multi-part zip archive
    os.chdir('dist')
    subprocess.run(['C:\\Program Files\\7-Zip\\7z.exe', '-v1500m', 'a', 'dgenerate_portable.zip', 'dgenerate'])
    os.chdir(script_dir)
else:
    if not os.path.exists('venv'):
        create_portable_environment()

# Build WiX installer
subprocess.run(['dotnet', 'build', 'dgenerate.wixproj', '--configuration', 'Release'])

if not wix_only:
    # Create a multi-part zip archive
    os.chdir('bin\\Release')
    cab_files = [f for f in os.listdir() if f.endswith('.cab')]
    subprocess.run(
        ['C:\\Program Files\\7-Zip\\7z.exe', '-v1500m', 'a', 'dgenerate_installer.zip', 'dgenerate.msi'] + cab_files)
    os.chdir(script_dir)

    # Move all parts of the multi-part zip archive to 'bin\\Release'
    for root, dirs, files in os.walk('dist'):
        for file in files:
            if file.startswith('dgenerate_portable.zip'):
                shutil.move(os.path.join(root, file), 'bin\\Release')
