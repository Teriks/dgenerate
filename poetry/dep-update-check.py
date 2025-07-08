# helper script for listing the latest version of all dependencies
import os
from importlib.machinery import SourceFileLoader
import platform
import pip
import sys
import requests

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

setup = SourceFileLoader('setup_as_library', 'setup.py').load_module()


pip_version = pip.__version__
python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
system_name = platform.system()

headers = {
    'User-Agent': f'pip/{pip_version} {{"python":"{python_version}","system":"{system_name}"}}'
}


def get_latest_version(package_name):
    response = requests.get(f'https://pypi.org/pypi/{package_name}/json', headers=headers)
    latest_version = response.json()['info']['version']
    return latest_version


for i in setup.poetry_pyproject_deps():
    if i[0] == 'python':
        continue
    t = get_latest_version(i[0])
    cur_ver = i[1]['version'].lstrip('^')
    if cur_ver != t:
        print(i[0] + f' = "{t}"')
