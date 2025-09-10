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
import sys
from ast import literal_eval
import tomllib

# Only import setuptools when actually running setup, not when loading as library
if __name__ != 'setup_as_library':
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
    with open(poetry_lockfile_path, 'rb') as f:
        lockfile_data = tomllib.load(f)
    
    packages = lockfile_data.get('package', [])
    for package in packages:
        yield package


def _should_include_dependency(marker_expression: str) -> bool:
    """Check if dependency should be included based on marker expression."""
    if not marker_expression or marker_expression == "sys_platform == \"not_darwin\"":
        return True
    
    # Define the environment variables that markers can use
    marker_env = {
        'platform_system': platform.system(),  # 'Windows', 'Linux', 'Darwin'
        'platform_machine': platform.machine(),  # 'x86_64', 'amd64', 'arm64', etc.
        'platform_python_implementation': platform.python_implementation(),  # 'CPython', 'PyPy'
        'sys_platform': {
            'Windows': 'win32',
            'Linux': 'linux', 
            'Darwin': 'darwin'
        }.get(platform.system(), platform.system().lower()),
        'extra': None,  # This would be set during extras processing
    }
    
    try:
        # Replace double quotes with single quotes to avoid issues
        safe_expression = marker_expression.replace('"', "'")
        
        # Replace python_version comparisons with True since we can't properly eval them
        # This assumes that if the dependency is in the lockfile, python_version requirements are satisfied
        safe_expression = re.sub(r'python_version\s*[><=!]+\s*[\'"][^\'\"]*[\'"]', 'True', safe_expression)
        
        return eval(safe_expression, {"__builtins__": {}}, marker_env)
    except Exception:
        # If evaluation fails, include the dependency by default
        return True


def get_poetry_lockfile_as_pip_requires(optionals=False) -> dict[str, str]:
    requirements = {}
    
    for dep in poetry_lockfile_deps():
        if dep['optional'] != optionals or not dep.get('name'):
            continue
            
        # Handle markers from lockfile
        markers = dep.get('markers')
        if markers and not _should_include_dependency(markers):
            continue
            
        requirements[dep["name"]] = '==' + dep["version"]
    
    return requirements


def poetry_pyproject_deps(include_optional=False):
    with open(poetry_pyproject_path, 'rb') as f:
        pyproject_data = tomllib.load(f)
    
    dependencies = pyproject_data.get('tool', {}).get('poetry', {}).get('dependencies', {})
    
    for name, spec in dependencies.items():
        if name == 'python':
            # Handle python version separately
            yield name, {'version': spec}
            continue
            
        if isinstance(spec, str):
            # Simple version specification
            version = {'version': spec}
        elif isinstance(spec, dict):
            # Complex specification with version, optional, platform, etc.
            version = spec.copy()
            is_optional = version.get('optional', False)
            if not include_optional and is_optional:
                continue
        else:
            # Skip invalid specifications
            continue
            
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
    requirements = {}
    
    for name, version_info in poetry_pyproject_deps(include_optional=include_optional):
        if name == 'python':
            continue  # Handle python separately
            
        version_spec = poetry_version_to_pip_requirement(version_info.get("version", ""))
        
        # Handle platform markers from pyproject.toml (uses "platform" field)
        platform_field = version_info.get("platform")
        if platform_field:
            if not _should_include_dependency(platform_field):
                continue
        
        requirements[name] = version_spec
    
    return requirements


requires = get_poetry_pyproject_as_pip_requires() \
    if not force_lockfile_requires else get_poetry_lockfile_as_pip_requires()


def _exclude_requires(name):
    if name in requires:
        requires.pop(name)


_pyinstaller_requires = 'pyinstaller==6.15.0'
_sphinx_requires = 'sphinx-rtd-theme==3.0.2'
_poetry_requires = 'poetry~=2.1.4'

_pyopengltk_requires = 'pyopengltk' + requires.pop('pyopengltk')
_PyOpenGL_requires = 'pyopengl' + requires.pop('pyopengl')
_PyOpenGL_accelerate_requires = 'pyopengl-accelerate' + requires.pop('pyopengl-accelerate')
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
    'dev': [_sphinx_requires,
            _poetry_requires],
    'readthedocs': _sphinx_requires
}

# Get python requirement from pyproject.toml (not affected by lockfile/pyproject choice)
python_requirement = None
try:
    python_deps = list(poetry_pyproject_deps())
    for name, version in python_deps:
        if name == 'python':
            python_requirement = poetry_version_to_pip_requirement(version.get("version"))
            break
except Exception:
    pass

# Fallback: try to get from requires dict (in case it was included)
if python_requirement is None:
    python_requirement = requires.get('python')

if python_requirement:
    requires.pop('python', None)  # Use None as default to avoid KeyError

if dgenerate_platform != 'linux':
    _exclude_requires('triton')

if dgenerate_platform == 'darwin':
    _exclude_requires('bitsandbytes')
    _exclude_requires('xformers')  # xFormers doesn't support macOS

if dgenerate_platform != 'windows':
    requires.pop('triton-windows')

if 'bitsandbytes' in requires:
    extras['bitsandbytes'] = ['bitsandbytes' + requires.pop('bitsandbytes')]

# xFormers support - only for NVIDIA CUDA on Linux/Windows
if 'xformers' in requires:
    _xformers_requires_spec = requires.pop('xformers')
    if dgenerate_platform in {'linux', 'windows'}:
        extras['xformers'] = ['xformers' + _xformers_requires_spec]

if dgenerate_platform in {'linux', 'windows'}:
    extras['gpt4all_cuda'] = ['gpt4all[cuda]' + _gpt4all_requires_spec]

if dgenerate_platform == 'windows':
    extras['triton_windows'] = ['triton-windows' + requires.pop('triton-windows')]

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
