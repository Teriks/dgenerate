import glob
import os.path
import subprocess
import sys

import argparse

try:
    import dgenerate.batchprocess as _batchprocess
except ImportError:
    _batchprocess = None

pwd = os.path.dirname(__file__)

parser = argparse.ArgumentParser(prog='run')

parser.add_argument('--paths', nargs='*',
                    help='example paths, do not include the working directory (examples parent directory).')
parser.add_argument('--subprocess-only', action='store_true', default=False,
                    help='Use a different subprocess for every example.')
parser.add_argument('--skip-animations', action='store_true', default=False, help='Entirely skip rendering animations.')
parser.add_argument('--skip-library', action='store_true', default=False, help='Entirely skip library usage examples.')
parser.add_argument('--skip-flax', action='store_true', default=False, help='Entirely skip flax examples on linux.')
parser.add_argument('--short-animations', action='store_true', default=False,
                    help='Reduce animation examples to rendering only 3 frames.')

known_args, injected_args = parser.parse_known_args()

library_installed = _batchprocess is not None and not known_args.skip_library

if known_args.paths:
    configs = []

    for path in known_args.paths:
        _, ext = os.path.splitext(path)
        if ext:
            configs += [path]
        else:
            if library_installed:
                configs += glob.glob(
                    os.path.join(pwd, *os.path.split(path), '**', '*main.py'),
                    recursive=True)

            configs += glob.glob(
                os.path.join(pwd, *os.path.split(path), '**', '*config.txt'),
                recursive=True)

else:
    configs = []

    if library_installed:
        configs = glob.glob(
            os.path.join(pwd, '**', '*main.py'),
            recursive=True)

    configs += glob.glob(
        os.path.join(pwd, '**', '*config.txt'),
        recursive=True)


def log(*args):
    print(*args, flush=True)


for config in configs:
    c = os.path.relpath(config, pwd)

    if known_args.skip_animations and 'animation' in c:
        log(f'SKIPPING ANIMATION: {config}')
        continue

    if 'flax' in c:
        if os.name == 'nt':
            log(f'SKIPPING FLAX ON WINDOWS: {config}')
            continue
        if known_args.skip_flax:
            log(f'SKIPPING FLAX: {config}')
            continue

    extra_args = []
    if known_args.short_animations and 'animation' in c:
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
                        _batchprocess.ConfigRunner(injected_args + extra_args).run_string(content)
                    except _batchprocess.BatchProcessError as e:
                        log(e)
                        sys.exit(1)
                else:
                    subprocess.run(["dgenerate"] + injected_args + extra_args, stdin=f, cwd=dirname, check=True)
            except KeyboardInterrupt:
                sys.exit(1)
        elif ext == '.py':
            try:
                subprocess.run([sys.executable] + [config] + injected_args, stdin=f, cwd=dirname, check=True)
            except KeyboardInterrupt:
                sys.exit(1)
