# Copyright (c) 2023, Teriks
#
# dgenerate is distributed under the following BSD 3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import ctypes
import queue
import subprocess
import threading
import platform

import psutil
from dgenerate.console.stdinpipe import StdinPipe
from dgenerate.files import TerminalLineReader


class ProcessMonitor:
    def __init__(self, events_per_tick=50):
        self.events_per_tick = events_per_tick

        self._closed = False

        self._process: psutil.Popen | None = None
        self._read_stderr_thread: threading.Thread | None = None
        self._read_stdout_thread: threading.Thread | None = None
        self._stderr_queue: queue.Queue | None = None
        self._stdout_queue: queue.Queue | None = None
        self.stdin_pipe: StdinPipe | None = None
        self.popen_kwargs: dict = {}
        self.popen_args: list = []

        # Callbacks for handling stdout, stderr, process exit, and restart events
        self.stderr_callback = lambda data: None
        self.stdout_callback = lambda data: None
        self.process_exit_callback = lambda return_code: None
        self.process_restarting_callback = lambda: None
        self.process_restarted_callback = lambda: None

    def cwd(self, deep=False):
        try:
            if deep:
                p = self._process
                while p.children():
                    p = p.children()[0]
                return p.cwd()
            else:
                if platform.system() == 'Windows':
                    try:
                        # pyinstaller console mode weirdness
                        return self._process.children()[0].children()[0].cwd()
                    except IndexError:
                        return self._process.cwd()
                else:
                    return self._process.cwd()
        except KeyboardInterrupt:
            pass
        except psutil.NoSuchProcess:
            pass

    def popen(self, *args, **kwargs):
        self.popen_args = args
        self.popen_kwargs = kwargs
        self.stdin_pipe = StdinPipe()

        self._process = psutil.Popen(
            *args,
            **kwargs,
            stdin=self.stdin_pipe.pipe_read,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self._stderr_queue = queue.Queue()
        self._stdout_queue = queue.Queue()

        self._read_stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._read_stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)

        self._read_stderr_thread.start()
        self._read_stdout_thread.start()

    def restart(self):
        self._process.kill()

    def close(self):
        self._closed = True
        self._process.kill()

    def process_events(self):
        return_code = None

        events = 0

        while True:
            if events > self.events_per_tick:
                break

            return_code = self._process.poll()
            if return_code is not None:
                break

            try:
                data = self._stdout_queue.get_nowait()
                self.stdout_callback(data)
                events += 1
            except queue.Empty:
                break

        while True:
            if events > self.events_per_tick:
                break

            return_code = self._process.poll()
            if return_code is not None:
                break

            try:
                data = self._stderr_queue.get_nowait()
                self.stderr_callback(data)
                events += 1
            except queue.Empty:
                break

        if return_code is None:
            return_code = self._process.poll()

        if return_code is not None and not self._closed:
            self._kill_threads()
            self.stdin_pipe.close()

            self.process_exit_callback(return_code)

            working_dir = self.process_restarting_callback()

            # Restart the process
            popen_kwargs = self.popen_kwargs.copy()
            if working_dir is not None:
                popen_kwargs['cwd'] = working_dir

            self.popen(*self.popen_args, **popen_kwargs)
            self.process_restarted_callback()

    def _kill_threads(self):
        reader_threads = [self._read_stdout_thread, self._read_stderr_thread]
        for thread in reader_threads:
            # there is no better way than to raise an exception on the thread,
            # the threads will live forever if they are blocking on read even if they are daemon threads,
            # and there is no platform independent non-blocking read IO with a subprocess
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.native_id, ctypes.py_object(SystemExit))
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.native_id, 0)
                raise SystemError('PyThreadState_SetAsyncExc failed')

    def _read_stderr(self):
        line_reader = TerminalLineReader(self._process.stderr)

        while self._process.poll() is None:
            line = line_reader.readline()
            if line is not None and line != b'':
                self._stderr_queue.put(line)

    def _read_stdout(self):
        line_reader = TerminalLineReader(self._process.stdout)

        while self._process.poll() is None:
            line = line_reader.readline()
            if line is not None and line != b'':
                self._stdout_queue.put(line)
