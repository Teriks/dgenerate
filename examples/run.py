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

import glob
import os.path
import subprocess
import sys

import argparse

try:
    import dgenerate.batchprocess as _batchprocess
except ImportError:
    _batchprocess = None

cwd = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(prog='run')

parser.add_argument('--paths', nargs='*',
                    help='example paths, do not include the working directory (examples parent directory).')
parser.add_argument('--subprocess-only', action='store_true', default=False,
                    help='Use a different subprocess for every example.')
parser.add_argument('--skip-animations', action='store_true', default=False, help='Entirely skip rendering animations.')
parser.add_argument('--skip-library', action='store_true', default=False, help='Entirely skip library usage examples.')
parser.add_argument('--skip-flax', action='store_true', default=False, help='Entirely skip flax examples on linux.')

parser.add_argument('--skip-deepfloyd', action='store_true', default=False,
                    help='Entirely skip deep floyd examples '
                         '(they use a lot of memory and may cause problems occasionally).')

parser.add_argument('--short-animations', action='store_true', default=False,
                    help='Reduce animation examples to rendering only 3 frames.')

known_args, injected_args = parser.parse_known_args()

library_installed = _batchprocess is not None and not known_args.skip_library

if known_args.paths:
    configs = []

    for path in known_args.paths:
        _, ext = os.path.splitext(path)
        if ext:
            configs += [path]
        else:
            if library_installed:
                configs += glob.glob(
                    os.path.join(cwd, *os.path.split(path), '**', '*main.py'),
                    recursive=True)

            configs += glob.glob(
                os.path.join(cwd, *os.path.split(path), '**', '*config.dgen'),
                recursive=True)

else:
    configs = []

    if library_installed:
        configs = glob.glob(
            os.path.join(cwd, '**', '*main.py'),
            recursive=True)

    configs += glob.glob(
        os.path.join(cwd, '**', '*config.dgen'),
        recursive=True)


def log(*args):
    print(*args, flush=True)


for config in configs:
    c = os.path.relpath(config, cwd)

    if known_args.skip_animations and 'animation' in c:
        log(f'SKIPPING ANIMATION: {config}')
        continue

    if 'flax' in c:
        if os.name == 'nt':
            log(f'SKIPPING FLAX ON WINDOWS: {config}')
            continue
        if known_args.skip_flax:
            log(f'SKIPPING FLAX: {config}')
            continue

    if 'deepfloyd' in c:
        if known_args.skip_deepfloyd:
            log(f'SKIPPING DEEPFLOYD: {config}')
        continue

    extra_args = []
    if known_args.short_animations and 'animation' in c:
        log(f'SHORTENING ANIMATION TO 3 FRAMES MAX: {config}')
        extra_args = ['--frame-end', '2']

    log(f'RUNNING: {config}')

    with open(config, mode='rt' if _batchprocess else 'rb') as f:
        dirname = os.path.dirname(config)
        _, ext = os.path.splitext(config)
        if ext == '.dgen':
            try:
                if _batchprocess is not None and not known_args.subprocess_only:
                    log('ENTERING DIRECTORY:', dirname)
                    os.chdir(dirname)
                    content = f.read()
                    try:
                        _batchprocess.ConfigRunner(injected_args + extra_args).run_string(content)
                    except SystemExit as e:
                        if e.code != 0:
                            raise
                    except _batchprocess.BatchProcessError as e:
                        log(e)
                        sys.exit(1)
                else:
                    subprocess.run(["dgenerate"] + injected_args + extra_args, stdin=f, cwd=dirname, check=True)
            except KeyboardInterrupt:
                sys.exit(1)
        elif ext == '.py':
            try:
                subprocess.run([sys.executable] + [config] + injected_args, stdin=f, cwd=dirname, check=True)
            except KeyboardInterrupt:
                sys.exit(1)
