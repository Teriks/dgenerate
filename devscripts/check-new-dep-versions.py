# helper script for listing the latest version of all dependencies
import os
from importlib.machinery import SourceFileLoader

import requests

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

setup = SourceFileLoader('setup_as_library', 'setup.py').load_module()


def get_latest_version(package_name):
    response = requests.get(f'https://pypi.org/pypi/{package_name}/json')
    latest_version = response.json()['info']['version']
    return latest_version


for i in setup.poetry_pyproject_deps():
    if i[0] == 'python':
        continue
    t = get_latest_version(i[0])
    cur_ver = i[1]['version'].lstrip('^')
    if cur_ver != t:
        print(i[0] + f' = "{t}"')
