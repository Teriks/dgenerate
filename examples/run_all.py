import glob
import os.path
import subprocess
import sys

pwd = os.path.dirname(__file__)
search = os.path.join(pwd, '**', 'config.txt')
configs = glob.glob(search, recursive=True)


for config in configs:
    c = os.path.relpath(config, pwd)

    if 'flax' in c:
        if os.name == 'nt':
            continue

    print(f'RUNNING: {config}')
    with open(config) as f:
        subprocess.run(["dgenerate"] + sys.argv[1:], shell=True, stdin=f, cwd=os.path.dirname(config), check=True)