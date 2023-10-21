#! /usr/bin/env python3

import io
import os
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
import re
import sys
from ast import literal_eval

from setuptools import setup, find_packages

setup_path = os.path.dirname(os.path.abspath(__file__))

with io.open(os.path.join(setup_path, 'dgenerate/__init__.py')) as _f:
    VERSION = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', _f.read(), re.MULTILINE).group(1)

if not VERSION:
    raise RuntimeError('version is not set')

with open(os.path.join(setup_path, 'README.rst'), 'r', encoding='utf-8') as _f:
    README = _f.read()

if not README:
    raise RuntimeError('readme is not set')


def poetry_lockfile_deps():
    poetry_lock_packages = re.compile(r"\[\[package\]\].*?optional = .*?python-versions.*?\n", re.MULTILINE | re.DOTALL)
    with open(os.path.join(setup_path, 'poetry/poetry.lock')) as f:
        for match in poetry_lock_packages.findall(f.read()):
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


def get_poetry_lockfile_as_pip_requires(optionals=False, exclude=None):
    exclude = set() if exclude is None else exclude
    return {dep["name"]: '==' + dep["version"] for dep in poetry_lockfile_deps()
            if dep['optional'] == optionals and dep['name'] not in exclude}


def poetry_pyproject_deps(exclude=None, include_optional=False):
    exclude = set() if exclude is None else exclude
    start = False
    with open(os.path.join(setup_path, 'poetry/pyproject.toml')) as f:
        for line in f:
            line = line.strip()

            if line == '[tool.poetry.dependencies]':
                start = True
            elif start and line.startswith('['):
                break
            elif start and line and not line.startswith('#'):
                name, spec = (i.strip() for i in line.split('=', 1))
                if spec.startswith('{'):
                    spec = \
                        spec.replace('=', ':'). \
                            replace('true', 'True'). \
                            replace('false', 'False'). \
                            replace('version', '"version"'). \
                            replace('optional', '"optional"'). \
                            replace('extras', '"extras"'). \
                            replace('source', '"source"')

                    version = literal_eval(spec)
                    is_optional = version.get('optional')
                    if not include_optional and is_optional:
                        continue
                else:
                    version = {'version': spec.strip('"\'')}

                if name in exclude:
                    continue
                yield name, version


def _pad_version(parts):
    if len(parts) < 3:
        for i in range(0, 3 - len(parts)):
            parts.append(0)


def _to_version_str(parts):
    return '.'.join(str(p) for p in parts)


def _version_to_parts(string, cast=True):
    parts = string.split('+')

    extra = ''
    if len(parts) == 2:
        extra = '+' + parts[1]

    version = parts[0]

    return [int(i) if cast else i for i in version.split('.')], extra


def poetry_caret_to_pip(version):
    v, _ = _version_to_parts(version)
    v2 = []
    bumped = False
    for idx, p in enumerate(v):
        if (p != 0 or idx == len(v) - 1) and not bumped:
            bumped = True
            v2.append(p + 1)
        else:
            v2.append(0)

    _pad_version(v)
    _pad_version(v2)

    return f">={_to_version_str(v)},<{_to_version_str(v2)}"


def _bump_version_rest(parts):
    v2 = []
    for idx, p in enumerate(parts):
        if idx == len(parts) - 1:
            v2.append(p + 1)
        else:
            v2.append(p)
    return v2


def poetry_tilde_to_pip(version):
    v, _ = _version_to_parts(version)
    if len(v) > 2:
        v = v[:2]

    v2 = _bump_version_rest(v)

    _pad_version(v)
    _pad_version(v2)

    return f">={_to_version_str(v)},<{_to_version_str(v2)}"


def poetry_star_to_pip(version):
    v, _ = _version_to_parts(version, cast=False)
    v = v[:v.index('*')]

    if not v:
        return '>=0.0.0'

    v2 = _bump_version_rest([int(i) for i in v])

    _pad_version(v)
    _pad_version(v2)

    return f">={_to_version_str(v)},<{_to_version_str(v2)}"


def poetry_version_to_pip_requirement(version):
    if version.startswith('^'):
        return poetry_caret_to_pip(version.lstrip('^'))
    if version.startswith('~'):
        return poetry_tilde_to_pip(version.lstrip('~'))
    if '*' in version:
        return poetry_star_to_pip(version)

    if version.startswith('=', 1):
        return version
    else:
        return f"=={version}"


def get_poetry_pyproject_as_pip_requires(include_optional=False, exclude=None):
    return {name: poetry_version_to_pip_requirement(version.get("version")) for name, version
            in poetry_pyproject_deps(include_optional=include_optional, exclude=exclude)}


if __name__ != 'setup_as_library':
    requires = get_poetry_pyproject_as_pip_requires()

    python_requirement = requires.get('python')

    exclude = {'triton'} if 'linux' not in sys.platform else set()

    pyproject_requirements = [name + spec for name, spec in
                              get_poetry_pyproject_as_pip_requires(
                                  exclude=exclude.union({'python'})).items()]

    lockfile_flax_requirements = [name + spec for name, spec in
                                  get_poetry_lockfile_as_pip_requires(optionals=True).items()]

    setup(name='dgenerate',
          python_requires=python_requirement,
          author='Teriks',
          author_email='Teriks@users.noreply.github.com',
          url='https://github.com/Teriks/dgenerate',
          version=VERSION,
          packages=find_packages(),
          license='BSD 3-Clause',
          description='Stable diffusion batch image generation tool with support for '
                      'video / gif / webp animation transcoding.',
          long_description=README,
          include_package_data=True,
          install_requires=pyproject_requirements,
          extras_require={
              'flax': lockfile_flax_requirements,
              'dev': ['pyinstaller==6.0.0',
                      'sphinx==7.2.6',
                      'sphinx_rtd_theme==1.3.0'],
              'docs': ['sphinx_rtd_theme==1.3.0']
          },
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
