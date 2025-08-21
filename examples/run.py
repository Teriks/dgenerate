#! /usr/bin/env python3

# Copyright (c) 2023, Teriks
# dgenerate is distributed under the following BSD 3-Clause License

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import glob
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

mp.set_start_method('spawn', force=True)

try:
    import dgenerate
    import dgenerate.batchprocess as _batchprocess
except ImportError:
    _batchprocess = None



cwd = os.path.dirname(os.path.abspath(__file__))

LIB_EXAMPLES_GLOB = '*main.py'
CONFIG_GLOB = '*config.dgen'

# Argument parsing
parser = argparse.ArgumentParser(prog='run')
parser.add_argument('--paths', nargs='*', help='Example paths, relative to the working directory.')
parser.add_argument('--subprocess-only', action='store_true', default=False,
                    help='Use a different subprocess for every example.')
parser.add_argument('--skip', nargs='*', default=None, help='Skip paths containing these strings.')
parser.add_argument('--short-animations', action='store_true', default=False,
                    help='Render only 3 frames for animations.')
parser.add_argument('--checkpoint',
                    help='Checkpoint file to save/load progress. Use to resume after failures.')


def log(*args):
    print(*args, flush=True)


def load_checkpoint(checkpoint_file):
    """Load checkpoint data from file."""
    if not os.path.exists(checkpoint_file):
        return None
    
    try:
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            
        # Validate checkpoint data structure
        if not isinstance(data, dict):
            log(f"Warning: Invalid checkpoint file {checkpoint_file}: not a dictionary")
            return None
            
        if 'completed_configs' not in data or 'total_configs' not in data:
            log(f"Warning: Invalid checkpoint file {checkpoint_file}: missing required fields")
            return None
            
        return data
    except (json.JSONDecodeError, IOError) as e:
        log(f"Warning: Could not load checkpoint file {checkpoint_file}: {e}")
        return None


def save_checkpoint(checkpoint_file, data):
    """Save checkpoint data to file."""
    try:
        # Ensure the directory exists
        checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_file))
        if checkpoint_dir:  # Only create directory if there is one
            os.makedirs(checkpoint_dir, exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        log(f"Warning: Could not save checkpoint file {checkpoint_file}: {e}")


def update_checkpoint(checkpoint_file, completed_configs, total_configs, args_info):
    """Update checkpoint with completed configurations."""
    checkpoint_data = {
        'completed_configs': completed_configs,
        'total_configs': total_configs,
        'args_info': args_info,
        'timestamp': time.time()
    }
    save_checkpoint(checkpoint_file, checkpoint_data)


def create_initial_checkpoint(checkpoint_file, configs, args_info):
    """Create initial checkpoint file with planned work."""
    checkpoint_data = {
        'completed_configs': [],
        'total_configs': configs,
        'args_info': args_info,
        'timestamp': None,
        'status': 'initialized'
    }
    save_checkpoint(checkpoint_file, checkpoint_data)
    log(f"Created initial checkpoint file: {checkpoint_file}")
    log(f"Planned to run {len(configs)} configurations")

def should_skip_config(config, known_args):
    c = os.path.relpath(config, cwd)
    skips = known_args.skip

    if skips is not None:
        for skip in skips:
            if skip in c:
                log(f'SKIPPING "{skip}": {config}')
                return True

    return False

def check_return_code(configs, exitcode):
    # ncnn often exits with a segfault because
    # the python binding is not that comprehensive,
    # and it cannot clean itself up properly after doing work.
    # We do not want to stop all the tests just for that
    if exitcode != 0:
        if not any('ncnn' in os.path.relpath(c, cwd).lower() for c in configs):
            log(f"Process exited with error code {exitcode}")
            sys.exit(exitcode)


def run_config(config, injected_args, extra_args, use_subprocess=False):
    config = os.path.abspath(config)

    log(f'RUNNING{" IN SUBPROCESS" if use_subprocess else ""}: {config}')

    dirname = os.path.dirname(config)
    _, ext = os.path.splitext(config)

    try:
        if use_subprocess:
            result = None
            if ext == '.dgen':
                result = subprocess.run(["dgenerate", '--file', config] + injected_args + extra_args, cwd=dirname)
            elif ext == '.py':
                result = subprocess.run([sys.executable] + [config] + injected_args, cwd=dirname)

            if result is not None:
                check_return_code([config], result.returncode)
                return True
        else:
            with open(config, mode='rt' if _batchprocess else 'rb') as f:
                if ext == '.dgen':
                    try:
                        if _batchprocess is not None:
                            log('ENTERING DIRECTORY:', dirname)
                            original_dir = os.getcwd()
                            try:
                                os.chdir(dirname)
                                content = f.read()
                                try:
                                    _batchprocess.ConfigRunner(injected_args + extra_args, throw=False).run_string(
                                        content)
                                    return True
                                except dgenerate.OutOfMemoryError as e:
                                    log(e)
                                    sys.exit(1)
                                except SystemExit as e:
                                    if e.code != 0:
                                        raise
                                except _batchprocess.BatchProcessError as e:
                                    log(e)
                                    sys.exit(1)
                            finally:
                                # Always restore the original directory
                                os.chdir(original_dir)
                        else:
                            log(
                                'Cannot run example in example runner process, '
                                'dgenerate library installation not found, '
                                'running in subprocess.')
                            result = subprocess.run(
                                ["dgenerate", '--file', config] + injected_args + extra_args, cwd=dirname)
                            if result is not None:
                                check_return_code([config], result.returncode)
                                return True
                    except KeyboardInterrupt:
                        sys.exit(1)
                elif ext == '.py':
                    try:
                        result = subprocess.run([sys.executable] + [config] + injected_args, stdin=f, cwd=dirname)
                        check_return_code([config], result.returncode)
                        return True
                    except KeyboardInterrupt:
                        sys.exit(1)
    except Exception as e:
        log(f"Error running {config}: {e}")
        return False
    
    return False


def run_directory_subprocess(configs, injected_args, extra_args, known_args, checkpoint_file=None, completed_configs=None):
    if completed_configs is None:
        completed_configs = set()
    
    for config in configs:
        if should_skip_config(config, known_args):
            continue
        
        # Skip if already completed
        if config in completed_configs:
            log(f'SKIPPING (already completed): {config}')
            continue
            
        success = run_config(config, injected_args, extra_args)
        if success:
            completed_configs.add(config)
            if checkpoint_file:
                log(f"Progress: {len(completed_configs)}/{len(configs)} configurations completed")
                update_checkpoint(checkpoint_file, list(completed_configs), configs, {
                    'injected_args': injected_args,
                    'extra_args': extra_args,
                    'known_args': {k: v for k, v in vars(known_args).items() if k != 'checkpoint'}
                })


def filter_to_directories_under_top_level(directories, top_level_directory):
    top_level_directory = os.path.abspath(top_level_directory) + os.path.sep
    return [d for d in directories if os.path.abspath(d).startswith(top_level_directory)]


def main():
    known_args, injected_args = parser.parse_known_args()
    library_installed = _batchprocess is not None


    # Handle checkpoint loading
    completed_configs = set()
    if known_args.checkpoint:
        checkpoint_data = load_checkpoint(known_args.checkpoint)
        if checkpoint_data:
            log(f"Loaded checkpoint from {known_args.checkpoint}")
            log(f"Resuming from {len(checkpoint_data.get('completed_configs', []))} completed configurations")
            completed_configs = set(checkpoint_data.get('completed_configs', []))
        else:
            log(f"Starting fresh run with checkpoint file: {known_args.checkpoint}")
            # Create initial checkpoint after configs are determined

    if known_args.paths:
        configs = []
        for path in [p for path in known_args.paths for p in glob.glob(path, recursive=True)]:
            path = os.path.abspath(path)
            _, ext = os.path.splitext(path)
            if ext:
                configs.append(path)
            else:
                if library_installed:
                    configs.extend(
                        glob.glob(os.path.join(path, '**', LIB_EXAMPLES_GLOB), recursive=True))
                configs.extend(
                    glob.glob(os.path.join(path, '**', CONFIG_GLOB), recursive=True))
    else:
        configs = []
        if library_installed:
            configs.extend(
                glob.glob(os.path.join(cwd, '**', LIB_EXAMPLES_GLOB), recursive=True))
        configs.extend(
            glob.glob(os.path.join(cwd, '**', CONFIG_GLOB), recursive=True))

    for i in configs:
        missing_file = False
        if not os.path.exists(i):
            log(f'Config path "{i}" does not exist.')
            missing_file = True
        if missing_file:
            sys.exit(1)

    # Create initial checkpoint if starting fresh
    if known_args.checkpoint and not completed_configs:
        create_initial_checkpoint(known_args.checkpoint, configs, {
            'injected_args': injected_args,
            'known_args': {k: v for k, v in vars(known_args).items() if k != 'checkpoint'}
        })

    top_level_dirs = sorted(set(os.path.join(cwd, os.path.relpath(config, cwd).split(os.sep)[0]) for config in configs))

    if known_args.subprocess_only:
        for config in sorted(configs):
            if should_skip_config(config, known_args):
                continue

            # Skip if already completed
            if config in completed_configs:
                log(f'SKIPPING (already completed): {config}')
                continue

            extra_args = []
            if known_args.short_animations and 'animation' in config:
                log(f'SHORTENING ANIMATION TO 3 FRAMES MAX: {config}')
                extra_args = ['--frame-end', '2']

            success = run_config(config, injected_args, extra_args, use_subprocess=True)
            if success:
                completed_configs.add(config)
                log(f"Progress: {len(completed_configs)}/{len(configs)} configurations completed")
                if known_args.checkpoint:
                    update_checkpoint(known_args.checkpoint, list(completed_configs), configs, {
                        'injected_args': injected_args,
                        'extra_args': extra_args,
                        'known_args': {k: v for k, v in vars(known_args).items() if k != 'checkpoint'}
                    })
    
    # Final summary for subprocess-only mode
    if known_args.subprocess_only:
        if known_args.checkpoint:
            log(f"Run completed. Total configurations: {len(configs)}, Completed: {len(completed_configs)}")
            if len(completed_configs) == len(configs):
                log("All configurations completed successfully!")
            else:
                log(f"Remaining configurations: {len(configs) - len(completed_configs)}")
                log("You can resume using the same checkpoint file.")
        else:
            log(f"Run completed. Total configurations: {len(configs)}")
    else:
        for top_dir in top_level_dirs:
            log(f'RUNNING CONFIGURATIONS IN DIRECTORY IN SUBPROCESS: {top_dir}')
            extra_args = []
            if known_args.short_animations:
                log(f'SHORTENING ANIMATIONS IN DIRECTORY TO 3 FRAMES MAX: {top_dir}')
                extra_args = ['--frame-end', '2']

            directory_configs = filter_to_directories_under_top_level(configs, top_dir)

            # Create a copy of completed_configs to avoid race conditions
            process_completed_configs = completed_configs.copy()
            p = mp.Process(target=run_directory_subprocess,
                           args=(directory_configs, injected_args, extra_args, known_args, 
                                 known_args.checkpoint, process_completed_configs))
            p.start()
            p.join()
            # Merge completed configs back into the main set
            if known_args.checkpoint:
                completed_configs.update(process_completed_configs)
            check_return_code(directory_configs, p.exitcode)
    
    # Final summary for non-subprocess-only mode
    if not known_args.subprocess_only:
        if known_args.checkpoint:
            log(f"Run completed. Total configurations: {len(configs)}, Completed: {len(completed_configs)}")
            if len(completed_configs) == len(configs):
                log("All configurations completed successfully!")
            else:
                log(f"Remaining configurations: {len(configs) - len(completed_configs)}")
                log("You can resume using the same checkpoint file.")
        else:
            log(f"Run completed. Total configurations: {len(configs)}")


if __name__ == "__main__":
    main()

