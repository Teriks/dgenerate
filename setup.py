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
import re
from ast import literal_eval

from setuptools import setup, find_packages

setup_path = os.path.dirname(os.path.abspath(__file__))

poetry_lockfile_path = \
    os.environ.get('DGENERATE_POETRY_LOCKFILE_PATH',
                   os.path.join(setup_path, 'poetry', 'poetry.lock')).strip('"').strip("'")

poetry_pyproject_path = \
    os.environ.get('DGENERATE_POETRY_PYPROJECT_PATH',
                   os.path.join(setup_path, 'poetry', 'pyproject.toml')).strip('"').strip("'")

dgenerate_platform = \
    os.environ.get('DGENERATE_PLATFORM', platform.system()).lower()

dgenerate_platform_tag = \
    os.environ.get('DGENERATE_PLATFORM_TAG', 'any')

force_lockfile_requires = \
    os.environ.get('DGENERATE_FORCE_LOCKFILE_REQUIRES')


def version_from_file(path: str):
    with open(path, 'r') as f:
        version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)
    return version


VERSION = version_from_file(os.path.join(setup_path, 'dgenerate', 'resources.py'))

if not VERSION:
    raise RuntimeError('version is not set')

with open(os.path.join(setup_path, 'README.rst'), 'r', encoding='utf-8') as _f:
    README = _f.read()

if not README:
    raise RuntimeError('readme is not set')


def poetry_lockfile_deps():
    poetry_lock_packages = re.compile(r"\[\[package\]\].*?optional = .*?python-versions.*?\n", re.MULTILINE | re.DOTALL)
    with open(poetry_lockfile_path) as f:
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


def get_poetry_lockfile_as_pip_requires(optionals=False) -> dict[str, str]:
    return {dep["name"]: '==' + dep["version"] for dep in poetry_lockfile_deps()
            if dep['optional'] == optionals and dep['name']}


def poetry_pyproject_deps(include_optional=False):
    start = False
    with open(poetry_pyproject_path) as f:
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
                            replace('source', '"source"'). \
                            replace('platform', '"platform"')

                    version = literal_eval(spec)
                    is_optional = version.get('optional')
                    if not include_optional and is_optional:
                        continue
                else:
                    version = {'version': spec.strip('"\'')}

                yield name, version


def _pad_version(parts):
    if len(parts) < 3:
        for i in range(0, 3 - len(parts)):
            parts.append(0)


def _to_version_str(parts, suffix='') -> str:
    return '.'.join(str(p) for p in parts) + suffix


def _version_to_parts(string, cast=True) -> tuple[list[int], str]:
    # Match the base version (major.minor.patch) and any suffix
    match = re.match(r'^(\d+(?:\.\d+)*)(.*)', string)
    if not match:
        # Fallback for edge cases
        return [0], string

    base_version, suffix = match.groups()

    if cast:
        try:
            version_parts = [int(i) for i in base_version.split('.')]
        except ValueError:
            # If casting fails, return as strings
            version_parts = base_version.split('.')
    else:
        version_parts = base_version.split('.')

    return version_parts, suffix


def poetry_caret_to_pip(version) -> str:
    v, suffix = _version_to_parts(version)
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

    return f">={_to_version_str(v, suffix)},<{_to_version_str(v2)}"


def _bump_version_rest(parts) -> list[int]:
    v2 = []
    for idx, p in enumerate(parts):
        if idx == len(parts) - 1:
            v2.append(p + 1)
        else:
            v2.append(p)
    return v2


def poetry_tilde_to_pip(version) -> str:
    v, suffix = _version_to_parts(version)
    if len(v) > 2:
        v = v[:2]

    v2 = _bump_version_rest(v)

    _pad_version(v)
    _pad_version(v2)

    return f">={_to_version_str(v, suffix)},<{_to_version_str(v2)}"


def poetry_star_to_pip(version) -> str:
    # Handle wildcard versions like "1.2.*"
    # Split on the first '*' to separate the base version from the wildcard
    if '*' not in version:
        return f"=={version}"

    base_version = version.split('*')[0].rstrip('.')

    if not base_version:
        return '>=0.0.0'

    # Parse the base version parts
    v = [int(i) for i in base_version.split('.')]
    v2 = _bump_version_rest(v)

    _pad_version(v)
    _pad_version(v2)

    return f">={_to_version_str(v)},<{_to_version_str(v2)}"


def poetry_version_to_pip_requirement(version) -> str:
    if version.startswith('^'):
        return poetry_caret_to_pip(version.lstrip('^'))
    if version.startswith('~'):
        return poetry_tilde_to_pip(version.lstrip('~'))
    if '*' in version:
        return poetry_star_to_pip(version)

    if version.startswith('=', 1):
        return version
    elif version.startswith('<'):
        return version
    elif version.startswith('>'):
        return version
    else:
        return f"=={version}"


def get_poetry_pyproject_as_pip_requires(include_optional=False) -> dict[str, str]:
    return {name: poetry_version_to_pip_requirement(version.get("version")) for name, version
            in poetry_pyproject_deps(include_optional=include_optional)}


requires = get_poetry_pyproject_as_pip_requires() \
    if not force_lockfile_requires else get_poetry_lockfile_as_pip_requires()


def _exclude_requires(name):
    if name in requires:
        requires.pop(name)


_pyinstaller_requires = 'pyinstaller' + requires.pop('pyinstaller')
_sphinx_requires = 'sphinx-rtd-theme' + requires.pop('sphinx-rtd-theme')
_poetry_requires = 'poetry' + requires.pop('poetry')
_graphviz_requires = 'graphviz' + requires.pop('graphviz')
_pyopengltk_requires = 'pyopengltk' + requires.pop('pyopengltk')
_PyOpenGL_requires = 'PyOpenGL' + requires.pop('PyOpenGL')
_PyOpenGL_accelerate_requires = 'PyOpenGL-accelerate' + requires.pop('PyOpenGL-accelerate')
_ncnn_requires = 'ncnn' + requires.pop('ncnn')
_gpt4all_requires_spec = requires.pop('gpt4all')

extras: dict[str, list[str]] = {
    'ncnn': [_ncnn_requires],
    'console_ui_opengl': [
        _pyopengltk_requires,
        _PyOpenGL_requires,
        _PyOpenGL_accelerate_requires
    ],
    'gpt4all': ['gpt4all' + _gpt4all_requires_spec],
    'dev': [_pyinstaller_requires,
            _sphinx_requires,
            _poetry_requires,
            _graphviz_requires],
    'readthedocs': _sphinx_requires
}

python_requirement = requires.get('python')

if python_requirement:
    requires.pop('python')

if dgenerate_platform != 'linux':
    _exclude_requires('triton')

if dgenerate_platform == 'darwin':
    _exclude_requires('bitsandbytes')

if dgenerate_platform == 'windows':
    for name in list(requires.keys()):
        if name.startswith('nvidia-'):
            requires.pop(name)
else:
    requires.pop('triton-windows')

if 'bitsandbytes' in requires:
    extras['bitsandbytes'] = ['bitsandbytes' + requires.pop('bitsandbytes')]

if dgenerate_platform in {'linux', 'windows'}:
    extras['gpt4all_cuda'] = ['gpt4all[cuda]' + _gpt4all_requires_spec]

if dgenerate_platform == 'windows':
    extras['triton_windows'] = ['triton-windows' + requires.pop('triton-windows')]

    extras['win-installer'] = (
            [_pyinstaller_requires] +
            extras['ncnn'] +
            extras['gpt4all_cuda'] +
            extras['triton_windows'] +
            extras['console_ui_opengl'] +
            extras['bitsandbytes']
    )

if __name__ != 'setup_as_library':
    setup(name='dgenerate',
          python_requires=python_requirement,
          author='Teriks',
          author_email='Teriks@users.noreply.github.com',
          url='https://github.com/Teriks/dgenerate',
          version=VERSION,
          packages=find_packages() +
                   ['dgenerate.console.themes',
                    'dgenerate.console.schemas',
                    'dgenerate.console.recipes',
                    'dgenerate.extras.hidiffusion.sd_module_key',
                    'dgenerate.translators.data'],
          package_data={
              'dgenerate': ['icon.ico', 'config_icon.ico', '*.json'],
              'dgenerate.console.themes': ['*.toml'],
              'dgenerate.console.schemas': ['*.json'],
              'dgenerate.console.recipes': ['*.recipe'],
              'dgenerate.translators.data': ['*.json'],
              'dgenerate.pipelinewrapper.hub_configs': ['**/*.json', '**/*.txt', '**/*.model'],
              'dgenerate.extras.hidiffusion.sd_module_key': ['*.txt'],
              'dgenerate.extras': ['**/LICENSE', '**/NOTICE']
          },
          include_package_data=True,
          license='BSD 3-Clause',
          description='Batch image generation and manipulation tool supporting Stable Diffusion and related techniques / '
                      'algorithms, with support for video and animated image processing.',
          long_description=README,
          long_description_content_type="text/x-rst",
          install_requires=[name + spec for name, spec in requires.items()],
          extras_require=extras,
          entry_points={
              'console_scripts': [
                  'dgenerate = dgenerate:main'
              ]
          },
          classifiers=[
              'Intended Audience :: Other Audience',
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'Natural Language :: English',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX :: Linux',
              'Environment :: Console',
              'Environment :: GPU :: NVIDIA CUDA',
              'Topic :: Utilities',
              'Topic :: Artistic Software',
              'Topic :: Multimedia :: Graphics',
              'Topic :: Multimedia :: Video',
              'Topic :: Scientific/Engineering :: Artificial Intelligence',
              'Topic :: Scientific/Engineering :: Image Processing',
          ])
