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
import os
import queue
import threading


class StdinPipeFullError(Exception):
    pass


class StdinPipe:
    def __init__(self, max_size=50):
        self._fifo_queue = queue.Queue(maxsize=max_size)
        self.pipe_read, self._pipe_write = os.pipe()

        # Set the pipe to non-blocking mode
        os.set_blocking(self._pipe_write, False)

        self._stop_event = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._writer, daemon=True)
        self._writer_thread.start()

    def _writer(self):
        while not self._stop_event.is_set():
            try:
                item = self._fifo_queue.get(timeout=0.1)
                while item:
                    try:
                        written = os.write(self._pipe_write, item)
                        item = item[written:]
                    except BlockingIOError:
                        if self._stop_event.is_set():
                            break
                self._fifo_queue.task_done()
            except queue.Empty:
                continue

    def write(self, data):
        try:
            self._fifo_queue.put(data, block=False)
        except queue.Full:
            raise StdinPipeFullError('stdin pipe is full.')

    def close(self):
        self._stop_event.set()
        self._writer_thread.join()
        os.close(self._pipe_write)
