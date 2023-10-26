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

        configs = glob.glob(
            os.path.join(pwd, *os.path.split(first_arg), '**', '*main.py'),
            recursive=True)

        configs += glob.glob(
            os.path.join(pwd, *os.path.split(first_arg), '**', '*config.txt'),
            recursive=True)

    args = args[1:]
else:
    configs = glob.glob(
        os.path.join(pwd, *os.path.split(first_arg), '**', '*main.py'),
        recursive=True)

    configs += glob.glob(os.path.join(pwd, '**', '*config.txt'),
                         recursive=True)

for config in configs:
    c = os.path.relpath(config, pwd)

    if _skip_animations and 'animation' in c:
        print(f'SKIPPING ANIMATION: {config}')
        sys.stdout.flush()
        continue

    if 'flax' in c:
        if os.name == 'nt':
            print(f'SKIPPING FLAX ON WINDOWS: {config}')
            sys.stdout.flush()
            continue

    extra_args = []
    if _short_animations and 'animation' in c:
        print(f'SHORTENING ANIMATION TO 3 FRAMES MAX: {config}')
        sys.stdout.flush()
        extra_args = ['--frame-end', '2']

    print(f'RUNNING: {config}')
    sys.stdout.flush()

    with open(config, mode='rt' if _batchprocess else 'rb') as f:
        dirname = os.path.dirname(config)
        _, ext = os.path.splitext(config)
        if ext == '.config':
            try:
                if _batchprocess:
                    print('ENTERING DIRECTORY:', dirname)
                    os.chdir(dirname)
                    _batchprocess.create_config_runner(args + extra_args, throw=True).run_file(f)
                else:
                    subprocess.run(["dgenerate"] + args + extra_args, stdin=f, cwd=dirname, check=True)
            except KeyboardInterrupt:
                sys.exit(1)
        else:
            try:
                subprocess.run([sys.executable] + [config], stdin=f, cwd=dirname, check=True)
            except KeyboardInterrupt:
                sys.exit(1)
