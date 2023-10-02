import glob
import os.path
import subprocess
import sys

pwd = os.path.dirname(__file__)


if 'flax_only' in sys.argv:
    search = os.path.join(pwd, 'flax', '**', 'config.txt')
    sys.argv.remove('flax_only')
else:
    search = os.path.join(pwd, '**', 'config.txt')

configs = glob.glob(search, recursive=True)

for config in configs:
    c = os.path.relpath(config, pwd)

    if 'flax' in c:
        if os.name == 'nt':
            continue

    print(f'RUNNING: {config}')
    proc = ["dgenerate"] + sys.argv[1:]
    with open(config) as f:
        subprocess.run(proc, stdin=f, cwd=os.path.dirname(config), check=True)