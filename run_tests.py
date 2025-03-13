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

import argparse
import os
import shlex
import subprocess
import sys
import unittest

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()

parser.add_argument('-c', '--clean', default=False, action='store_true')
parser.add_argument('-e', '--examples', default=False, action='store_true')
parser.add_argument('--examples-log', default='examples/examples.log')

args, unknown_args = parser.parse_known_args()

runner = unittest.TextTestRunner()


def join_with_globs(args):
    return ' '.join(arg if '*' in arg else shlex.quote(arg) for arg in args)


if runner.run(unittest.defaultTestLoader.discover("tests", pattern='*_test.py')).wasSuccessful():

    if not args.examples:
        exit(0)

    print('unit tests passed, running examples..')

    if args.clean:
        os.chdir('examples')
        print('running: git clean -f -d -x in examples folder...')
        subprocess.run('git clean -f -d -x', shell=True)
        os.chdir('..')

    run_string = f'{sys.executable} examples/run.py {join_with_globs(unknown_args)} ' \
                 f'--short-animations --output-configs --output-metadata -v > {args.examples_log} 2>&1'

    print('running:', run_string)

    subprocess.run(run_string, shell=True)
else:
    exit(1)
