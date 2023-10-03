import glob
import os.path
import subprocess
import sys

pwd = os.path.dirname(__file__)

args = sys.argv[1:]

if len(args) > 0:
    first_arg = args[0]
else:
    first_arg = None

if first_arg and not first_arg.startswith('-'):
    _, ext = os.path.splitext(first_arg)
    if ext:
        configs = [first_arg]
    else:
        search = os.path.join(pwd, *os.path.split(first_arg), '**', '*config.txt')
        configs = glob.glob(search, recursive=True)
    args = args[1:]
else:
    search = os.path.join(pwd, '**', '*config.txt')
    configs = glob.glob(search, recursive=True)

for config in configs:
    c = os.path.relpath(config, pwd)

    if 'flax' in c:
        if os.name == 'nt':
            print(f'SKIPPING FLAX CONFIG ON WINDOWS: {config}')
            sys.stdout.flush()
            continue

    print(f'RUNNING CONFIG: {config}')
    sys.stdout.flush()
    proc = ["dgenerate"] + args
    with open(config) as f:
        subprocess.run(proc, stdin=f, cwd=os.path.dirname(config), check=True)
