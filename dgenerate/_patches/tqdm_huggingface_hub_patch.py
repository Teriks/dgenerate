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


import contextlib
import inspect
import os
import threading
import time
import typing
import huggingface_hub
import tqdm
import traceback

_original_thread_init = threading.Thread.__init__


def _patched_thread_init(self, *args, **kwargs):
    self._dgenerate_no_tqdm_thread = False
    
    # Delay imports to avoid circular import issues during module initialization.
    # Only import these modules when we actually need to check the stack frames.
    diffusers_single_file = None
    diffusers_pipeline_utils = None
    
    try:
        # Only try to import if diffusers is already available in sys.modules
        # This avoids triggering imports during the initial diffusers loading phase
        import sys
        if 'diffusers.loaders.single_file' in sys.modules:
            import diffusers.loaders.single_file
            diffusers_single_file = diffusers.loaders.single_file
        if 'diffusers.pipelines.pipeline_utils' in sys.modules:
            import diffusers.pipelines.pipeline_utils
            diffusers_pipeline_utils = diffusers.pipelines.pipeline_utils
    except ImportError:
        # If imports fail, we'll just skip the tqdm suppression for now
        pass

    # prevent these modules from creating a multithreaded tqdm progress bar
    # because the output in dgenerates case is generally not useful
    if diffusers_single_file is not None or diffusers_pipeline_utils is not None:
        for frame_info in inspect.stack():
            module = inspect.getmodule(frame_info.frame)
            if module:
                if diffusers_single_file is not None and module is diffusers_single_file:
                    self._dgenerate_no_tqdm_thread = True
                    break
                if diffusers_pipeline_utils is not None and module is diffusers_pipeline_utils:
                    self._dgenerate_no_tqdm_thread = True
                    break

    _original_thread_init(self, *args, **kwargs)

threading.Thread.__init__ = _patched_thread_init

_main_thread_id = threading.get_ident()


class ProgressAggregator:
    def __init__(self):
        self._lock = threading.RLock()  # Use RLock to allow nested locking
        self._global_progress_bar = None
        self._thread_progress = {}  # Maps thread_id -> current progress
        self._active_threads = set()
        self._last_activity_time = 0
        self._session_timeout = 1.0  # seconds of inactivity before starting new session
        self._main_thread_active_bars = set()  # Track active progress bars on main thread
        self._current_session_id = 0
        self._thread_sessions = {}  # Maps thread_id -> session_id



    def register_main_thread_bar(self, bar):
        """Register a progress bar from the main thread"""
        with self._lock:
            self._main_thread_active_bars.add(bar)

    def unregister_main_thread_bar(self, bar):
        """Unregister a progress bar from the main thread"""
        with self._lock:
            self._main_thread_active_bars.discard(bar)

    def clear_stale_bars(self):
        """Remove any bars that appear to be closed or stale"""
        active_bars = set()
        for bar in self._main_thread_active_bars:
            try:
                # Check if bar is closed
                is_closed = False
                
                # For our wrapped bars, check _is_closed
                if hasattr(bar, '_is_closed'):
                    is_closed = bar._is_closed
                
                # For all bars, check if disabled
                if bar.disable:
                    is_closed = True
                    
                # Check if the bar has been explicitly closed by checking internal state
                if hasattr(bar, 'n') and hasattr(bar, 'total'):
                    # If n >= total and bar is not being updated, it's likely finished
                    if bar.total and bar.n >= bar.total and hasattr(bar, 'last_print_n'):
                        # Check if it hasn't been updated recently
                        if bar.n == bar.last_print_n:
                            is_closed = True
                
                # Keep bar if it's not closed
                if not is_closed:
                    active_bars.add(bar)
            except:
                # Any error means the bar is probably dead
                pass
        
        self._main_thread_active_bars = active_bars

    def has_main_thread_bars(self):
        """Check if main thread has active progress bars"""
        with self._lock:
            # First clear any obviously stale bars
            self.clear_stale_bars()
            return len(self._main_thread_active_bars) > 0

    def register_thread(self, thread_id, total_bytes):
        """Register a new download thread"""
        with self._lock:
            current_time = time.time()
            
            # Check if we're starting a new session
            if (len(self._active_threads) == 0 and 
                current_time - self._last_activity_time > self._session_timeout):
                # New session - clear old progress
                self._thread_progress.clear()
                self._current_session_id += 1
                if self._global_progress_bar is not None:
                    self._global_progress_bar.close()
                    self._global_progress_bar = None
            
            self._active_threads.add(thread_id)
            self._thread_progress[thread_id] = 0
            self._thread_sessions[thread_id] = self._current_session_id
            self._last_activity_time = current_time

            # Only create global progress bar if:
            # 1. We don't already have one
            # 2. Main thread doesn't have active bars
            # 3. We have multiple active threads (concurrent downloads)
            if (self._global_progress_bar is None and 
                not self.has_main_thread_bars() and
                len(self._active_threads) >= 2):
                self._global_progress_bar = tqdm.tqdm(
                    desc="Multithreaded download",
                    unit="B",
                    unit_scale=True,
                    position=0
                )

    def update_progress(self, thread_id, new_progress):
        """Update progress for a specific thread"""
        with self._lock:
            # Only process updates from current session
            if (thread_id in self._thread_progress and 
                thread_id in self._thread_sessions and
                self._thread_sessions[thread_id] == self._current_session_id):
                
                old_progress = self._thread_progress[thread_id]
                self._thread_progress[thread_id] = new_progress

                # Check if we should create the global bar now
                if (self._global_progress_bar is None and 
                    not self.has_main_thread_bars() and
                    len(self._active_threads) >= 2):
                    self._global_progress_bar = tqdm.tqdm(
                        desc="Multithreaded download",
                        unit="B",
                        unit_scale=True,
                        position=0
                    )
                    # Update with all current progress from current session
                    total_progress = 0
                    for tid, progress in self._thread_progress.items():
                        if (tid in self._thread_sessions and 
                            self._thread_sessions[tid] == self._current_session_id):
                            total_progress += progress
                    if total_progress > 0:
                        self._global_progress_bar.update(total_progress)

                # Only update if we have a global bar and no main thread bars are active
                if self._global_progress_bar is not None and not self.has_main_thread_bars():
                    delta = new_progress - old_progress
                    if delta > 0:
                        self._global_progress_bar.update(delta)
                elif self._global_progress_bar is not None and self.has_main_thread_bars():
                    # Main thread bar appeared, close our global bar
                    self._global_progress_bar.close()
                    self._global_progress_bar = None

    def finish_thread(self, thread_id):
        """Clean up when a thread finishes"""
        with self._lock:
            if thread_id in self._active_threads:
                self._active_threads.remove(thread_id)
            
            # Clean up session tracking
            if thread_id in self._thread_sessions:
                del self._thread_sessions[thread_id]

            self._last_activity_time = time.time()

            if len(self._active_threads) == 0:
                if self._global_progress_bar is not None:
                    self._global_progress_bar.close()
                    self._global_progress_bar = None
                self._thread_progress.clear()
                self._thread_sessions.clear()


_progress_aggregator = ProgressAggregator()


class TrackedBarWrapper:
    """Wrapper to track external tqdm bars and properly unregister them when closed"""
    _instances = []  # Keep references to prevent garbage collection
    
    def __init__(self, bar):
        self._bar = bar
        self._original_close = bar.close
        bar.close = self._wrapped_close
        # Keep a reference to prevent garbage collection
        TrackedBarWrapper._instances.append(self)
    
    def _wrapped_close(self):
        _progress_aggregator.unregister_main_thread_bar(self._bar)
        self._original_close()
        # Remove self from instances
        try:
            TrackedBarWrapper._instances.remove(self)
        except ValueError:
            pass


class MainThreadTqdm(tqdm.tqdm):
    """Wrapper for main thread progress bars that registers with aggregator"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_closed = False
        _progress_aggregator.register_main_thread_bar(self)
    
    def close(self):
        if not self._is_closed:
            self._is_closed = True
            _progress_aggregator.unregister_main_thread_bar(self)
            super().close()


class ThreadSafeTqdm(tqdm.tqdm):
    def __init__(self, *args, **kwargs):
        self.thread_id = threading.get_ident()
        self._manual_n = 0

        if self.thread_id != _main_thread_id:
            kwargs['disable'] = True
            super().__init__(*args, **kwargs)

            _progress_aggregator.register_thread(self.thread_id, kwargs.get('total', 0))
        else:
            super().__init__(*args, **kwargs)

    def update(self, n=1):
        if self.thread_id != _main_thread_id:
            self._manual_n += n
            _progress_aggregator.update_progress(self.thread_id, self._manual_n)

        return super().update(n)

    def close(self):
        if self.thread_id != _main_thread_id:
            try:
                _progress_aggregator.finish_thread(self.thread_id)
            except Exception:
                pass
        super().close()


def _get_progress_bar_context(
        *,
        desc: str,
        log_level: int,
        total: typing.Optional[int] = None,
        initial: int = 0,
        unit: str = "B",
        unit_scale: bool = True,
        name: typing.Optional[str] = None,
        _tqdm_bar: typing.Optional[tqdm.tqdm] = None,
) -> typing.ContextManager[tqdm.tqdm]:
    global _main_thread_id

    if _tqdm_bar is not None:
        # If on main thread, register and track the provided bar
        if threading.get_ident() == _main_thread_id:
            _progress_aggregator.register_main_thread_bar(_tqdm_bar)
            # Wrap the bar to ensure it's unregistered when closed
            TrackedBarWrapper(_tqdm_bar)
        return contextlib.nullcontext(_tqdm_bar)

    if threading.get_ident() == _main_thread_id:
        # Main thread: create tracked progress bar
        pbar = MainThreadTqdm(
            unit=unit,
            unit_scale=unit_scale,
            total=total,
            initial=initial,
            desc=desc
        )
    else:

        if threading.current_thread()._dgenerate_no_tqdm_thread:
            return tqdm.tqdm(disable=True)

        # Worker thread: create thread-safe progress bar
        pbar = ThreadSafeTqdm(
            unit=unit,
            unit_scale=unit_scale,
            total=total,
            initial=initial,
            desc=desc
        )

    return contextlib.closing(pbar)


# Apply the patch
huggingface_hub.file_download._get_progress_bar_context = _get_progress_bar_context
