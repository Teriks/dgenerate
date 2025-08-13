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
import platform
import subprocess
import sys
from importlib.machinery import SourceFileLoader

args = sys.argv[1:]

script_path = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_path, '..'))
hf_cache_local = os.path.join(project_dir, 'docker_cache', 'huggingface')
dgenerate_cache_local = os.path.join(project_dir, 'docker_cache', 'dgenerate')
pip_cache_local = os.path.join(project_dir, 'docker_cache', 'pip')

setup = SourceFileLoader(
    'setup_as_library', os.path.join(project_dir, 'setup.py')).load_module()

print('hf_cache_local:', hf_cache_local)
print('dgenerate_cache_local:', dgenerate_cache_local)
print('pip_cache_local:', pip_cache_local)
print('project_dir:', project_dir)

os.makedirs(hf_cache_local, exist_ok=True)
os.makedirs(dgenerate_cache_local, exist_ok=True)
os.makedirs(pip_cache_local, exist_ok=True)

env_defs = []

while '-e' in args:
    try:
        pos = args.index('-e')
    except ValueError:
        break

    try:
        env_def = args[pos + 1]
        env_defs += ['-e', env_def]
        if env_def.startswith('-'):
            raise IndexError
        args = args[:pos] + args[pos + 2:]
    except IndexError:
        print(
            'run: error: -e missing argument.')
        sys.exit(1)

dev_mode = False
amd_mode = False

while '--dev' in args:
    args.remove('--dev')
    dev_mode = True

while '--amd' in args:
    args.remove('--amd')
    amd_mode = True

if len(args) == 0:
    args = ['bash']

if amd_mode:
    extra_index = "https://download.pytorch.org/whl/rocm6.4/"
    docker_file = 'dockerfile-rocm'
    image = 'dgenerate-rocm'

    # just need to be able to test this build process
    # on windows even though GPU passthrough does
    # not function
    gpu_opts = ['--device', '/dev/kfd', '--device', '/dev/dri', '--security-opt', 'seccomp=unconfined'] \
        if platform.system() != 'Windows' else []
else:
    extra_index = "https://download.pytorch.org/whl/cu128/"
    docker_file = 'dockerfile-cuda'
    image = 'dgenerate-cuda'
    gpu_opts = ['--gpus', 'all']

subprocess.check_call(['docker', 'image', 'build', '-f', docker_file, '-t', f'teriks/{image}:{setup.VERSION}', '.'])

subprocess.call(['docker', 'rm', '-f', image])

subprocess.call(['docker', 'run', *env_defs,
                 *gpu_opts, '--name', image,
                 '-e', f"DGENERATE_INSTALL_DEV={1 if dev_mode else 0}",
                 '-e', f"DGENERATE_INSTALL_INDEX={extra_index}",
                 '-v', f"{project_dir}:/opt/dgenerate",
                 '-v', f"{hf_cache_local}:/home/dgenerate/.cache/huggingface",
                 '-v', f"{dgenerate_cache_local}:/home/dgenerate/.cache/dgenerate",
                 '-v', f"{pip_cache_local}:/home/dgenerate/.cache/pip",
                 '-it', f'teriks/{image}:{setup.VERSION}',
                 'bash', '-c', f"source docker/install.sh; {' '.join(args)}"])
