import glob
import os.path
import subprocess
import sys

try:
    import dgenerate.batchprocess as _batchprocess
except ImportError:
    _batchprocess = None

pwd = os.path.dirname(__file__)

args = sys.argv[1:]

_skip_animations = False
_short_animations = False

if 'skip_animations' in args:
    _skip_animations = True
    args.remove('skip_animations')

if 'short_animations' in args:
    _skip_animations = False
    _short_animations = True
    args.remove('short_animations')

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

    if _skip_animations and 'animation' in c:
        print(f'SKIPPING ANIMATION CONFIG: {config}')
        sys.stdout.flush()
        continue

    if 'flax' in c:
        if os.name == 'nt':
            print(f'SKIPPING FLAX CONFIG ON WINDOWS: {config}')
            sys.stdout.flush()
            continue

    extra_args = []
    if _short_animations and 'animation' in c:
        print(f'SHORTENING ANIMATION TO 3 FRAMES MAX: {config}')
        sys.stdout.flush()
        extra_args = ['--frame-end', '2']

    print(f'RUNNING CONFIG: {config}')
    sys.stdout.flush()
    proc = ["dgenerate"] + args + extra_args
    with open(config, mode='rt' if _batchprocess else 'rb') as f:
        try:
            if _batchprocess:
                os.chdir(os.path.dirname(config))
                _batchprocess.create_config_runner(args + extra_args, throw=True).run_file(f)
            else:
                subprocess.run(proc, stdin=f, cwd=os.path.dirname(config), check=True)
        except KeyboardInterrupt:
            sys.exit(1)
