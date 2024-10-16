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
import gc
import glob
import inspect
import multiprocessing as mp
import os
import subprocess
import sys
import traceback

mp.set_start_method('spawn', force=True)

try:
    import dgenerate
    import dgenerate.batchprocess as _batchprocess
except ImportError:
    _batchprocess = None

try:
    import torch
except ImportError:
    torch = None

cwd = os.path.dirname(os.path.abspath(__file__))

LIB_EXAMPLES_GLOB = '*main.py'
CONFIG_GLOB = '*config.dgen'

# Argument parsing
parser = argparse.ArgumentParser(prog='run')
parser.add_argument('--paths', nargs='*', help='Example paths, relative to the working directory.')
parser.add_argument('--subprocess-only', action='store_true', default=False,
                    help='Use a different subprocess for every example.')
parser.add_argument('--skip-animations', action='store_true', default=False, help='Skip rendering animations.')
parser.add_argument('--skip-library', action='store_true', default=False, help='Skip library usage examples.')
parser.add_argument('--skip-deepfloyd', action='store_true', default=False, help='Skip deep floyd examples.')
parser.add_argument('--skip-ncnn', action='store_true', default=False, help='Skip examples involving ncnn.')
parser.add_argument('--short-animations', action='store_true', default=False,
                    help='Render only 3 frames for animations.')
parser.add_argument('--torch-debug', action='store_true', default=False,
                    help='Track torch objects for extensive CUDA OOM debug output.')


def log(*args):
    print(*args, flush=True)


def add_creation_stack_trace(func):
    def wrapper(*args, **kwargs):
        obj = func(*args, **kwargs)
        frame = inspect.currentframe().f_back
        obj._DGENERATE_TORCH_OOM_DEBUG_INFO = f"{frame.f_code.co_filename}:{frame.f_lineno}"
        obj._DGENERATE_TORCH_OOM_STACK_TRACE = ''.join(traceback.format_stack(frame))
        return obj

    return wrapper


def patch_torch_functions():
    tensor_creation_funcs = [
        'empty', 'zeros', 'ones', 'rand', 'randn', 'randint', 'tensor',
        'as_tensor', 'from_numpy', 'sparse_coo_tensor'
    ]
    for func_name in tensor_creation_funcs:
        if hasattr(torch, func_name):
            setattr(torch, func_name, add_creation_stack_trace(getattr(torch, func_name)))
    if torch.cuda.is_available():
        cuda_tensor_types = [
            'FloatTensor', 'DoubleTensor', 'HalfTensor', 'ByteTensor',
            'CharTensor', 'ShortTensor', 'IntTensor', 'LongTensor'
        ]
        for tensor_type in cuda_tensor_types:
            if hasattr(torch.cuda, tensor_type):
                setattr(torch.cuda, tensor_type, add_creation_stack_trace(getattr(torch.cuda, tensor_type)))
        cuda_creation_funcs = [
            'empty', 'zeros', 'ones', 'rand', 'randn', 'randint', 'tensor',
        ]
        for func_name in cuda_creation_funcs:
            if hasattr(torch.cuda, func_name):
                setattr(torch.cuda, func_name, add_creation_stack_trace(getattr(torch.cuda, func_name)))


def patch_module_and_optimizers():
    original_module_init = torch.nn.Module.__init__

    def new_module_init(self, *args, **kwargs):
        frame = inspect.currentframe().f_back.f_back
        self._DGENERATE_TORCH_OOM_DEBUG_INFO = f"{frame.f_code.co_filename}:{frame.f_lineno}"
        self._DGENERATE_TORCH_OOM_STACK_TRACE = ''.join(traceback.format_stack(frame))
        original_module_init(self, *args, **kwargs)

    torch.nn.Module.__init__ = new_module_init

    original_optimizer_init = torch.optim.Optimizer.__init__

    def new_optimizer_init(self, *args, **kwargs):
        frame = inspect.currentframe().f_back.f_back
        self._DGENERATE_TORCH_OOM_DEBUG_INFO = f"{frame.f_code.co_filename}:{frame.f_lineno}"
        self._DGENERATE_TORCH_OOM_STACK_TRACE = ''.join(traceback.format_stack(frame))
        original_optimizer_init(self, *args, **kwargs)

    torch.optim.Optimizer.__init__ = new_optimizer_init

    original_storage_init = torch.storage._StorageBase.__init__

    def new_storage_init(self, *args, **kwargs):
        frame = inspect.currentframe().f_back.f_back
        self._DGENERATE_TORCH_OOM_DEBUG_INFO = f"{frame.f_code.co_filename}:{frame.f_lineno}"
        self._DGENERATE_TORCH_OOM_STACK_TRACE = ''.join(traceback.format_stack(frame))
        original_storage_init(self, *args, **kwargs)

    torch.storage._StorageBase.__init__ = new_storage_init


def find_gpu_tensors_in_gc():
    try:
        import graphviz
    except ImportError:
        print('pip install graphviz and graphviz native binaries to use --torch-debug')

    dot = graphviz.Digraph(comment='Torch GPU Object Reference Graph')
    dot.graph_attr.update(size="100,100!", ratio="expand", layout="circo", splines="true", nodesep="1.5", ranksep="2.0")
    dot.node_attr.update(style="filled", fillcolor="lightgrey", shape="box")

    def is_on_gpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.is_cuda
        elif isinstance(obj, torch.nn.Module):
            return any(param.is_cuda for param in obj.parameters())
        elif isinstance(obj, torch.optim.Optimizer):
            return any(param.is_cuda for group in obj.param_groups for param in group['params'])
        elif isinstance(obj, torch.storage._StorageBase):
            return obj.device.type == 'cuda'
        return False

    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    reserved_memory = torch.cuda.memory_reserved(0)

    log(f"Total CUDA memory: {total_memory} bytes")
    log(f"Allocated CUDA memory: {allocated_memory} bytes")
    log(f"Reserved CUDA memory: {reserved_memory} bytes")

    for obj in gc.get_objects():
        try:
            if isinstance(obj, (torch.Tensor, torch.nn.Module, torch.optim.Optimizer, torch.storage._StorageBase)):
                if hasattr(obj, '_DGENERATE_TORCH_OOM_DEBUG_INFO') and is_on_gpu(obj):
                    debug_info = obj._DGENERATE_TORCH_OOM_DEBUG_INFO.replace('\\', '/')
                    obj_id = id(obj)
                    obj_info = f"{obj.__class__.__name__} created at {debug_info}\n"
                    if isinstance(obj, torch.Tensor):
                        obj_info += f"Allocated: {obj.numel() * obj.element_size()} bytes\n"
                    obj_info += f"Device: {obj.device}\n"
                    dot.node(str(obj_id), obj_info)

                    log(f"Object: {obj.__class__.__name__} created at {debug_info}")
                    for ref in gc.get_referents(obj):
                        ref_id = id(ref)
                        ref_info = f"{ref.__class__.__name__}"
                        dot.node(str(ref_id), ref_info)
                        dot.edge(str(obj_id), str(ref_id))
                        log(f'Refers To: {ref}')
                    log(f"Stack Trace:\n{obj._DGENERATE_TORCH_OOM_STACK_TRACE}")
        except Exception:
            pass

    try:
        dot.save(os.path.join(os.path.dirname(__file__), 'torch_object_graph.dot'))
        dot.render('torch_object_graph', format='svg',
                   outfile=os.path.join(os.path.dirname(__file__), 'torch_object_graph.svg'))
    except subprocess.SubprocessError:
        log('Cannot render torch object graph, graphviz binary not found.')


def should_skip_config(config, known_args):
    c = os.path.relpath(config, cwd)
    if known_args.skip_animations and 'animation' in c:
        log(f'SKIPPING ANIMATION: {config}')
        return True
    if 'deepfloyd' in c and known_args.skip_deepfloyd:
        log(f'SKIPPING DEEPFLOYD: {config}')
        return True
    if 'ncnn' in c and known_args.skip_ncnn:
        log(f'SKIPPING NCNN: {config}')
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


def run_config(config, injected_args, extra_args, debug_torch, use_subprocess=False):
    log(f'RUNNING{" IN SUBPROCESS" if use_subprocess else ""}: {config}')
    dirname = os.path.dirname(config)
    _, ext = os.path.splitext(config)

    if debug_torch and not use_subprocess:
        patch_torch_functions()
        patch_module_and_optimizers()

    if use_subprocess:
        result = None
        if ext == '.dgen':
            result = subprocess.run(["dgenerate", '--file', config] + injected_args + extra_args, cwd=dirname)
        elif ext == '.py':
            result = subprocess.run([sys.executable] + [config] + injected_args, cwd=dirname)

        if result is not None:
            check_return_code([config], result.returncode)
    else:
        with open(config, mode='rt' if _batchprocess else 'rb') as f:
            if ext == '.dgen':
                try:
                    if _batchprocess is not None:
                        log('ENTERING DIRECTORY:', dirname)
                        os.chdir(dirname)
                        content = f.read()
                        try:
                            _batchprocess.ConfigRunner(injected_args + extra_args, throw=debug_torch).run_string(
                                content)
                        except dgenerate.OutOfMemoryError as e:
                            log(e)
                            find_gpu_tensors_in_gc()
                            sys.exit(1)
                        except SystemExit as e:
                            if e.code != 0:
                                raise
                        except _batchprocess.BatchProcessError as e:
                            log(e)
                            sys.exit(1)
                    else:
                        log(
                            'Cannot run example in example runner process, '
                            'dgenerate library installation not found, '
                            'running in subprocess.')
                        result = subprocess.run(
                            ["dgenerate", '--file', config] + injected_args + extra_args, cwd=dirname)
                        if result is not None:
                            check_return_code([config], result.returncode)
                except KeyboardInterrupt:
                    sys.exit(1)
            elif ext == '.py':
                try:
                    result = subprocess.run([sys.executable] + [config] + injected_args, stdin=f, cwd=dirname)
                    check_return_code([config], result.returncode)
                except KeyboardInterrupt:
                    sys.exit(1)


def run_directory_subprocess(configs, injected_args, extra_args, debug_torch, known_args):
    for config in configs:
        if should_skip_config(config, known_args):
            continue

        run_config(config, injected_args, extra_args, debug_torch)


def filter_to_directories_under_top_level(directories, top_level_directory):
    top_level_directory = os.path.abspath(top_level_directory) + os.path.sep
    return [d for d in directories if os.path.abspath(d).startswith(top_level_directory)]


def main():
    known_args, injected_args = parser.parse_known_args()
    library_installed = _batchprocess is not None and not known_args.skip_library
    debug_torch = not known_args.subprocess_only and torch is not None and known_args.torch_debug

    if known_args.paths:
        configs = []
        for path in known_args.paths:
            _, ext = os.path.splitext(path)
            if ext:
                configs.append(path)
            else:
                if library_installed:
                    configs.extend(
                        glob.glob(os.path.join(cwd, *os.path.split(path), '**', LIB_EXAMPLES_GLOB), recursive=True))
                configs.extend(
                    glob.glob(os.path.join(cwd, *os.path.split(path), '**', CONFIG_GLOB), recursive=True))
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

    top_level_dirs = sorted(set(os.path.join(cwd, os.path.relpath(config, cwd).split(os.sep)[0]) for config in configs))

    if known_args.subprocess_only:
        for config in configs:
            if should_skip_config(config, known_args):
                continue

            extra_args = []
            if known_args.short_animations and 'animation' in config:
                log(f'SHORTENING ANIMATION TO 3 FRAMES MAX: {config}')
                extra_args = ['--frame-end', '2']

            run_config(config, injected_args, extra_args, debug_torch, use_subprocess=True)
    else:
        for top_dir in top_level_dirs:
            log(f'RUNNING CONFIGURATIONS IN DIRECTORY IN SUBPROCESS: {top_dir}')
            extra_args = []
            if known_args.short_animations:
                log(f'SHORTENING ANIMATIONS IN DIRECTORY TO 3 FRAMES MAX: {top_dir}')
                extra_args = ['--frame-end', '2']

            directory_configs = filter_to_directories_under_top_level(configs, top_dir)

            p = mp.Process(target=run_directory_subprocess,
                           args=(directory_configs, injected_args, extra_args, debug_torch, known_args))
            p.start()
            p.join()
            check_return_code(directory_configs, p.exitcode)


if __name__ == "__main__":
    main()

