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

import io
import re

from setuptools import setup, find_packages

version = ''
with io.open('dgenerate/__init__.py') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

if not version:
    raise RuntimeError('version is not set')

with open('README.rst', 'r', encoding='utf-8') as f:
    readme = f.read()

setup(name='dgenerate',
      python_requires='>=3.10',
      author='Teriks',
      author_email='Teriks@users.noreply.github.com',
      url='https://github.com/Teriks/dgenerate',
      version=version,
      packages=find_packages(),
      license='BSD 3-Clause',
      description='Stable diffusion batch image generation tool with support for '
                  'video / gif / webp animation transcoding.',
      long_description=readme,
      include_package_data=True,
      install_requires=['diffusers~=0.20.0',
                        'transformers~=4.31.0',
                        'pytorch_lightning~=2.0.7',
                        'omegaconf~=2.3.0',
                        'numpy~=1.24.3',
                        'Requests~=2.31.0',
                        'av~=10.0.0',
                        'Pillow~=9.5.0',
                        'torch~=2.0.1',
                        'fake-useragent~=1.1.3'],
      entry_points={
          'console_scripts': [
              'dgenerate = dgenerate:main',
          ]
      },
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'License :: OSI Approved :: BSD License',
          'Intended Audience :: Other Audience',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Topic :: Utilities',
      ])
