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
_skip_library = False
_short_animations = False
_skip_flax = False

if '--subprocess-only' in args:
    _batchprocess = None
    args.remove('--subprocess-only')

if '--skip-animations' in args:
    _skip_animations = True
    args.remove('--skip-animations')

if '--skip-library' in args:
    _skip_library = True
    args.remove('--skip-library')

if '--skip-flax' in args:
    _skip_flax = True
    args.remove('--skip-flax')

if '--short-animations' in args:
    _skip_animations = False
    _short_animations = True
    args.remove('--short-animations')

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
        os.path.join(pwd, '**', '*main.py'),
        recursive=True)

    configs += glob.glob(os.path.join(pwd, '**', '*config.txt'),
                         recursive=True)


def log(*args):
    print(*args, flush=True)


for config in configs:
    c = os.path.relpath(config, pwd)

    if _skip_animations and 'animation' in c:
        log(f'SKIPPING ANIMATION: {config}')
        continue

    if 'flax' in c:
        if os.name == 'nt':
            log(f'SKIPPING FLAX ON WINDOWS: {config}')
            continue
        if _skip_flax:
            log(f'SKIPPING FLAX: {config}')
            continue

    extra_args = []
    if _short_animations and 'animation' in c:
        log(f'SHORTENING ANIMATION TO 3 FRAMES MAX: {config}')
        extra_args = ['--frame-end', '2']

    log(f'RUNNING: {config}')

    with open(config, mode='rt' if _batchprocess else 'rb') as f:
        dirname = os.path.dirname(config)
        _, ext = os.path.splitext(config)
        if ext == '.txt':
            try:
                if _batchprocess is not None:
                    log('ENTERING DIRECTORY:', dirname)
                    os.chdir(dirname)
                    content = f.read()
                    try:
                        _batchprocess.ConfigRunner(args + extra_args).run_string(content)
                    except _batchprocess.BatchProcessError as e:
                        log(e)
                        sys.exit(1)
                else:
                    subprocess.run(["dgenerate"] + args + extra_args, stdin=f, cwd=dirname, check=True)
            except KeyboardInterrupt:
                sys.exit(1)
        elif _batchprocess is not None and not _skip_library:
            # library is installed
            try:
                subprocess.run([sys.executable] + [config] + args, stdin=f, cwd=dirname, check=True)
            except KeyboardInterrupt:
                sys.exit(1)
