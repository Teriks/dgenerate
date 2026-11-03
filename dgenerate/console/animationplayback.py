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

"""Decode and play an animated file for the console preview.

The Vulkan viewer (``console_ui_vulkan``) and the OpenGL viewer both use this.
Short clips are decoded up front. Longer videos are decoded ahead of the
playhead so a frame is ready when its timestamp arrives. Audio, when the file
has any, is the playback clock. The clock follows samples that have reached
the speaker. Counting samples only as the device accepts a buffer makes the
picture jump by a whole period (200ms, and several periods at the start).
A loop wraps inside the playback callback. Stopping the device to flush that
period runs on the UI thread and freezes the console until the driver returns.
Volume and mute are stored in ``~/.dgenerate/console_settings.json``.
"""

import ctypes
import json
import os
import pathlib
import sys
import threading
import time

import numpy as np
import PIL.Image

_PIL_ANIMATED = {'.gif', '.webp', '.png', '.apng'}
_VIDEO = {'.mp4', '.m4v', '.mov', '.webm', '.mkv', '.avi'}
_PRELOAD_BYTES = 160 * 1024 * 1024

BAR_HEIGHT = 46
# Empty pixels between the bottom of the picture and the top of the controls.
CONTROL_GAP = 8


def is_animation_path(path: str) -> bool:
    extension = os.path.splitext(path)[1].lower()
    if extension == '.png':
        return _looks_like_apng(path)
    return extension in (_PIL_ANIMATED | _VIDEO)


def _looks_like_apng(path: str) -> bool:
    try:
        with open(path, 'rb') as handle:
            return b'acTL' in handle.read(1024 * 1024)
    except OSError:
        return False


def frame_index_at(times: np.ndarray, seconds: float) -> int:
    """Frame whose start time is the latest one that is still <= ``seconds``."""
    if len(times) == 0:
        return 0
    index = int(np.searchsorted(times, seconds, side='right') - 1)
    return max(0, min(index, len(times) - 1))


def format_clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    whole = int(seconds + 0.5)
    return f'{whole // 60}:{whole % 60:02d}'


def picture_height(widget_height: int) -> int:
    """Height left for the picture once the control bar and its gap are reserved."""
    return max(1, int(widget_height) - BAR_HEIGHT - CONTROL_GAP)


def control_layout(width: int, height: int) -> dict | None:
    """Widget-pixel rectangles for the hover timeline. Y grows downward."""
    if width < 260 or height < BAR_HEIGHT + 8:
        return None
    top = height - BAR_HEIGHT
    play = (14, top + 11, 34, top + 35)
    track_left = 108
    track_right = width - 196
    if track_right - track_left < 24:
        return None
    track = (track_left, top + 19, track_right, top + 27)
    loop = (width - 108, top + 11, width - 86, top + 35)
    mute = (width - 78, top + 11, width - 56, top + 35)
    volume = (width - 50, top + 20, width - 12, top + 26)
    return {
        'bar': (0, top, width, height),
        'play': play,
        'track': track,
        'elapsed': (46, top + 14),
        'duration': (track_right + 8, top + 14),
        'loop': loop,
        'mute': mute,
        'volume': volume,
    }


def _inside(rect, x: int, y: int) -> bool:
    left, top, right, bottom = rect
    return left <= x < right and top <= y < bottom


def control_hit(x: int, y: int, width: int, height: int) -> str | None:
    """``'play'``, ``'track'``, ``'loop'``, ``'mute'``, ``'volume'``, or ``None``."""
    layout = control_layout(width, height)
    if layout is None or not _inside(layout['bar'], x, y):
        return None
    for name in ('play', 'loop', 'mute', 'volume'):
        if _inside(layout[name], x, y):
            return name
    return 'track'


def preview_audio_settings_path() -> pathlib.Path:
    return pathlib.Path.home() / '.dgenerate' / 'console_settings.json'


def load_preview_audio(path: pathlib.Path | None = None) -> tuple[float, bool]:
    """Return the saved preview volume (0 to 1) and mute flag."""
    settings = path or preview_audio_settings_path()
    try:
        data = json.loads(settings.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return 1.0, False
    if not isinstance(data, dict):
        return 1.0, False
    try:
        volume = min(1.0, max(0.0, float(data.get('preview_volume', 1.0))))
    except (TypeError, ValueError):
        volume = 1.0
    return volume, bool(data.get('preview_muted', False))


def effective_preview_gain(volume: float, muted: bool, *, visible: bool) -> float:
    """Device gain from saved volume/mute and whether the preview widget is on screen."""
    if muted or not visible:
        return 0.0
    return min(1.0, max(0.0, float(volume)))


def save_preview_audio(volume: float, muted: bool, path: pathlib.Path | None = None) -> None:
    """Store preview volume and mute without dropping the other console settings."""
    settings = path or preview_audio_settings_path()
    try:
        data = json.loads(settings.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, UnicodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data['preview_volume'] = min(1.0, max(0.0, float(volume)))
    data['preview_muted'] = bool(muted)
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps(data), encoding='utf-8')


_preview_audio_live: tuple[float, bool] | None = None
_preview_audio_watchers: list = []


def current_preview_audio() -> tuple[float, bool]:
    """Volume and user mute shared by every preview pane and window."""
    global _preview_audio_live
    if _preview_audio_live is None:
        _preview_audio_live = load_preview_audio()
    return _preview_audio_live


def watch_preview_audio(callback) -> None:
    """``callback(volume, muted)`` runs when the shared preview mute or volume changes."""
    _preview_audio_watchers.append(callback)


def unwatch_preview_audio(callback) -> None:
    try:
        _preview_audio_watchers.remove(callback)
    except ValueError:
        return


def publish_preview_audio(volume: float, muted: bool, *, persist: bool = False) -> None:
    """Update the shared preview mute and volume, then tell every open preview."""
    global _preview_audio_live
    volume = min(1.0, max(0.0, float(volume)))
    muted = bool(muted)
    _preview_audio_live = (volume, muted)
    if persist:
        save_preview_audio(volume, muted)
    for callback in list(_preview_audio_watchers):
        callback(volume, muted)


def volume_on_slider(x: int, width: int, height: int) -> float:
    layout = control_layout(width, height)
    if layout is None:
        return 1.0
    left, _top, right, _bottom = layout['volume']
    span = max(1, right - left)
    return min(1.0, max(0.0, (x - left) / span))


def speaker_icon(rect, muted: bool, volume: float):
    """Shapes for the mute button, in widget pixels. Y grows downward.

    Each entry is ``('rect', x1, y1, x2, y2, color)``,
    ``('poly', [(x, y), ...], color)``, or
    ``('stroke', [(x, y), ...], thickness, color)``.
    The cone flares toward the sound. Waves mean audio is on. A red X means mute.
    """
    x1, _y1, _x2, _y2 = rect
    mid = (_y1 + _y2) / 2.0
    white = (1.0, 1.0, 1.0, 1.0)
    shapes = [
        ('rect', x1 + 1.0, mid - 3.5, x1 + 5.2, mid + 3.5, white),
        ('poly', [
            (x1 + 4.4, mid - 3.5),
            (x1 + 11.0, mid - 8.5),
            (x1 + 11.0, mid + 8.5),
            (x1 + 4.4, mid + 3.5),
        ], white),
    ]
    if muted or volume <= 0:
        red = (0.93, 0.27, 0.24, 1.0)
        shapes.append(('stroke', [(x1 + 13.0, mid - 6.0), (x1 + 20.0, mid + 6.0)], 2.0, red))
        shapes.append(('stroke', [(x1 + 13.0, mid + 6.0), (x1 + 20.0, mid - 6.0)], 2.0, red))
        return shapes
    shapes.append(('stroke', [
        (x1 + 13.2, mid - 4.5), (x1 + 15.8, mid), (x1 + 13.2, mid + 4.5),
    ], 1.7, white))
    if volume >= 0.45:
        shapes.append(('stroke', [
            (x1 + 16.4, mid - 8.0), (x1 + 19.8, mid), (x1 + 16.4, mid + 8.0),
        ], 1.7, white))
    return shapes


def loop_icon(rect, enabled: bool):
    """A circular arrow. A red slash means the clip stops at the end."""
    x1, y1, _x2, y2 = rect
    mid = (y1 + y2) / 2.0
    color = (1.0, 1.0, 1.0, 1.0) if enabled else (1.0, 1.0, 1.0, 0.35)
    left, right = x1 + 3.0, x1 + 16.0
    top, bottom = mid - 6.0, mid + 6.0
    shapes = [
        ('stroke', [(left + 4, top), (right, top), (right, bottom), (left, bottom), (left, top + 4)], 1.8, color),
        ('poly', [(left + 1, top + 4), (left + 7, top + 4), (left + 4, top)], color),
    ]
    if not enabled:
        shapes.append(('stroke', [(left, top), (right, bottom)], 1.8, (0.93, 0.27, 0.24, 1.0)))
    return shapes


_timer_users = 0
_LOOKAHEAD_SECONDS = 0.5


def acquire_playback_timer() -> None:
    """Ask Windows for a 1ms timer while a clip is playing.

    The default timer wakes about every 15ms, so a preview tick skips source
    frames. ``timeEndPeriod`` is paired in :func:`release_playback_timer`.
    """
    global _timer_users
    if sys.platform == 'win32' and _timer_users == 0:
        ctypes.windll.winmm.timeBeginPeriod(1)
    _timer_users += 1


def release_playback_timer() -> None:
    global _timer_users
    if _timer_users <= 0:
        return
    _timer_users -= 1
    if sys.platform == 'win32' and _timer_users == 0:
        ctypes.windll.winmm.timeEndPeriod(1)


def playback_tick_delay_ms(clip: 'AnimationClip', timeline_moving: bool) -> int:
    """Milliseconds until the next preview redraw.

    Playback waits until the next source frame so that frame is not skipped.
    A visible timeline still refreshes at least every 16ms.
    """
    if not clip.playing:
        return 16
    delay = clip.millis_until_next_frame()
    if timeline_moving:
        return min(16, delay)
    return delay


def time_on_track(x: int, width: int, height: int, duration: float) -> float:
    layout = control_layout(width, height)
    if layout is None or duration <= 0:
        return 0.0
    left, _top, right, _bottom = layout['track']
    span = max(1, right - left)
    fraction = min(1.0, max(0.0, (x - left) / span))
    return fraction * duration


class _SmoothPlaybackClock:
    """Samples that have reached the speaker.

    A playback callback is handed a whole period at once, and the device queues
    several periods before the first one is heard. Treating that queue as the
    playhead skips video frames. Between callbacks the heard position advances
    with the wall clock, and a callback may only correct it by less than one
    picture frame.
    """

    def __init__(self, sample_rate: int):
        self._rate = max(1, int(sample_rate))
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self._cursor = 0
            self._latency = 0
            self._latency_locked = False
            self._last_submit = None
            self._sync_heard = 0
            self._sync_time = None

    def submitted(self, frames: int):
        """The device just accepted ``frames`` more samples into its queue."""
        frames = int(frames)
        if frames <= 0:
            return
        now = time.perf_counter()
        with self._lock:
            if not self._latency_locked:
                # Back-to-back requests are the device filling its queue, not
                # samples that have already played.
                if self._last_submit is not None and (now - self._last_submit) >= 0.005:
                    self._latency_locked = True
                else:
                    self._latency += frames
            self._last_submit = now
            self._cursor += frames
            estimated = max(0, self._cursor - self._latency)
            if not self._latency_locked:
                # Still filling the device queue. A few microseconds between
                # those submits must not advance the playhead (int(dt * rate)
                # is often 1 at 48kHz).
                current = 0
            else:
                current = self._heard_unlocked(now)
                if estimated > current:
                    # A correction stays under one 24fps frame, so it cannot skip one.
                    current = min(estimated, current + int(self._rate * 0.020))
            self._sync_heard = current
            self._sync_time = now

    def heard(self) -> int:
        with self._lock:
            return self._heard_unlocked(time.perf_counter())

    def _heard_unlocked(self, now: float) -> int:
        if self._sync_time is None:
            return 0
        heard = self._sync_heard + int((now - self._sync_time) * self._rate)
        if heard < 0:
            return 0
        if heard > self._cursor:
            return self._cursor
        return heard


class PcmPlayer:
    """
    Plays int16 PCM. ``position`` is seconds from the start of the clip.

    miniaudio is tried first (WASAPI, CoreAudio, or Pulse). sounddevice is
    next, then waveOut on Windows. Without a backend, ``active`` stays false
    and the picture keeps its own clock. Gain is applied while audio is playing.
    """

    def __init__(self, samples: np.ndarray, sample_rate: int):
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        self._samples = np.ascontiguousarray(samples, dtype=np.int16)
        self.sample_rate = int(sample_rate)
        self.channels = int(self._samples.shape[1])
        self.active = False
        self._offset = 0
        self._paused = False
        self._gain = 1.0
        self._backend = None
        self._open_backend()

    def set_gain(self, gain: float):
        """Linear loudness from 0 to 1. Takes effect while audio is playing."""
        self._gain = min(1.0, max(0.0, float(gain)))
        if self._backend is not None:
            self._backend.set_gain(self._gain)

    def _open_backend(self):
        if self.sample_rate <= 0 or self._samples.size == 0:
            return
        # miniaudio talks to WASAPI, CoreAudio, and Pulse directly. waveOut is
        # only a fallback: a lot of Windows machines expose no legacy wave devices.
        for factory in (_MiniaudioPlayer, _SoundDevicePlayer):
            try:
                backend = factory(self._samples, self.sample_rate, self.channels)
            except Exception:
                continue
            if backend.active:
                self._backend = backend
                self.active = True
                return
        if sys.platform == 'win32':
            backend = _WaveOutPlayer(self._samples, self.sample_rate, self.channels)
            if backend.active:
                self._backend = backend
                self.active = True

    @property
    def gain(self) -> float:
        return self._gain

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def running(self) -> bool:
        """True when the backend is actively consuming samples."""
        if not self.active or self._paused or self._backend is None:
            return False
        return bool(getattr(self._backend, 'running', True))

    def set_loop(self, enabled: bool, duration: float | None = None):
        """Wrap playback at ``duration`` seconds. The device stays running.

        A soundtrack shorter than the clip is padded with silence up to
        ``duration``, so the wrap lands on the picture loop and not early.
        """
        period = len(self._samples)
        if enabled and duration and self.sample_rate > 0:
            stretched = int(round(float(duration) * self.sample_rate))
            if stretched > period:
                period = stretched
        if self._backend is not None and hasattr(self._backend, 'set_loop'):
            self._backend.set_loop(bool(enabled), period)

    @property
    def looping(self) -> bool:
        backend = self._backend
        if backend is None:
            return False
        return bool(getattr(backend, 'looping', False))

    @property
    def position(self) -> float:
        if self._backend is None or self.sample_rate <= 0:
            return self._offset / self.sample_rate if self.sample_rate else 0.0
        if hasattr(self._backend, 'timeline_frame'):
            return self._backend.timeline_frame() / self.sample_rate
        played = self._backend.played_samples()
        return (self._offset + played) / self.sample_rate

    @property
    def finished(self) -> bool:
        """The samples handed to the device have all been heard."""
        if not self.active or self._paused or self._backend is None or self.sample_rate <= 0:
            return False
        if hasattr(self._backend, 'exhausted'):
            return self._backend.exhausted()
        total = len(self._samples) - self._offset
        return total > 0 and self._backend.played_samples() >= total - 1

    def play(self, seconds: float):
        if self._backend is None or self.sample_rate <= 0:
            return
        self._offset = max(0, min(int(seconds * self.sample_rate), len(self._samples)))
        self._paused = False
        self._start_backend(self._samples, self._offset)

    def pause(self):
        if self._backend is not None and not self._paused:
            if hasattr(self._backend, 'timeline_frame'):
                self._offset = max(0, int(self._backend.timeline_frame()))
            else:
                self._offset = max(0, min(int(self.position * self.sample_rate), len(self._samples)))
            self._backend.pause()
            self._paused = True

    def resume(self):
        if self._backend is None or not self._paused:
            return
        self._paused = False
        self._start_backend(self._samples, self._offset)

    def _start_backend(self, samples: np.ndarray, cursor: int = 0):
        try:
            self._backend.start(samples, cursor)
        except Exception:
            self._drop_backend()
            return
        if self._backend is not None and not getattr(self._backend, 'active', True):
            self._drop_backend()

    def _drop_backend(self):
        """The device cannot be started. The picture keeps the wall clock."""
        backend = self._backend
        self._backend = None
        self.active = False
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass

    def close(self):
        backend = self._backend
        self._backend = None
        self.active = False
        if backend is not None:
            backend.close()


def _take_playback(samples: np.ndarray, cursor: int, count: int, loop: bool, period: int):
    """Copy ``count`` frames starting at ``cursor``.

    Returns ``(frames, new_cursor, advance)``. ``advance`` is how far the
    playhead clock moves. A looping read wraps at ``period`` and counts the
    silence after the soundtrack, so the clock reaches the clip length and
    then returns to the start. A read that is not looping stops at the end of
    ``samples`` and leaves the remainder silent, without moving the clock.
    """
    count = int(count)
    channels = int(samples.shape[1]) if getattr(samples, 'ndim', 1) == 2 else 1
    if count <= 0:
        return np.zeros((0, channels), dtype=np.int16), 0, 0
    total = len(samples)
    out = np.zeros((count, channels), dtype=np.int16)
    if total <= 0:
        return out, 0, 0
    if not loop:
        cursor = min(max(0, int(cursor)), total)
        take = min(count, total - cursor)
        if take > 0:
            out[:take] = samples[cursor:cursor + take]
        return out, cursor + take, take
    period = max(total, int(period) if period else total, 1)
    cursor = int(cursor) % period
    filled = 0
    while filled < count:
        if cursor >= period:
            cursor = 0
        if cursor < total:
            take = min(count - filled, total - cursor)
            out[filled:filled + take] = samples[cursor:cursor + take]
        else:
            take = min(count - filled, period - cursor)
        cursor += take
        filled += take
        if cursor >= period:
            cursor = 0
    return out, cursor, count


class _SampleLoop:
    """Read cursor for a device that keeps running across a loop."""

    def _init_loop(self, sample_count: int):
        self._loop = False
        self._period = max(1, int(sample_count))
        self._start_frame = 0
        self._cursor = 0

    def set_loop(self, enabled: bool, period: int | None = None):
        if period is not None and int(period) > 0:
            self._period = max(len(getattr(self, '_samples', ())), int(period), 1)
        self._loop = bool(enabled)

    @property
    def looping(self) -> bool:
        return bool(getattr(self, '_loop', False))

    def _begin_span(self, sample_count: int, cursor: int):
        if not hasattr(self, '_loop'):
            self._init_loop(sample_count)
        self._cursor = max(0, int(cursor))
        if self._loop and self._period > 0:
            self._cursor %= self._period
            self._period = max(sample_count, int(self._period), 1)
        else:
            if sample_count > 0:
                self._cursor = min(self._cursor, sample_count)
            self._period = max(sample_count, 1)
        self._start_frame = self._cursor

    def timeline_frame(self) -> int:
        heard = self.played_samples()
        raw = self._start_frame + heard
        if self._loop and self._period > 0:
            return int(raw % self._period)
        return int(raw)

    def exhausted(self) -> bool:
        if self._loop:
            return False
        total = len(self._samples)
        if total <= 0:
            return True
        return self._start_frame + self.played_samples() >= total - 1


class _WaveSlot:
    def __init__(self, frames: int, channels: int):
        self.frames = frames
        self.buf = ctypes.create_string_buffer(frames * channels * 2)
        self.header = _WAVEHDR()
        self.header.lpData = ctypes.cast(self.buf, ctypes.c_char_p)
        self.header.dwBufferLength = frames * channels * 2
        self.prepared = False
        self.queued = False


class _WaveOutPlayer(_SampleLoop):
    """winmm fallback. A feeder refills short buffers off the UI thread.

    A one-second ``waveOutWrite`` buffer makes ``waveOutReset`` block until
    that second comes back. The loop path never resets: the feeder wraps the
    read cursor, and a restart only sets a flag the feeder applies.
    """

    def __init__(self, samples: np.ndarray, sample_rate: int, channels: int):
        self.active = False
        self._sample_rate = sample_rate
        self._channels = channels
        self._samples = np.ascontiguousarray(samples, dtype=np.int16)
        self._init_loop(len(self._samples))
        self._winmm = ctypes.windll.winmm
        self._handle = None
        self._slots: list[_WaveSlot] = []
        self._state = threading.Lock()
        self._winmm_lock = threading.Lock()
        self._gain = 1.0
        self._feeding = False
        self._quit = False
        self._span = 0
        self._applied = 0
        self._pending_cursor = 0
        self._paused_samples = 0
        self._clock = _SmoothPlaybackClock(sample_rate)
        self._wake = threading.Event()
        self._thread = None
        self._open()

    def set_gain(self, gain: float):
        self._gain = min(1.0, max(0.0, float(gain)))
        if not self.active:
            return
        level = int(self._gain * 0xFFFF)
        packed = (level & 0xFFFF) | ((level & 0xFFFF) << 16)
        self._winmm.waveOutSetVolume(self._handle, ctypes.c_uint(packed))

    @property
    def running(self) -> bool:
        return bool(self.active and self._feeding)

    def _open(self):
        winmm = self._winmm

        class WAVEFORMATEX(ctypes.Structure):
            _fields_ = [
                ('wFormatTag', ctypes.c_ushort),
                ('nChannels', ctypes.c_ushort),
                ('nSamplesPerSec', ctypes.c_uint),
                ('nAvgBytesPerSec', ctypes.c_uint),
                ('nBlockAlign', ctypes.c_ushort),
                ('wBitsPerSample', ctypes.c_ushort),
                ('cbSize', ctypes.c_ushort),
            ]

        fmt = WAVEFORMATEX()
        fmt.wFormatTag = 1
        fmt.nChannels = self._channels
        fmt.nSamplesPerSec = self._sample_rate
        fmt.wBitsPerSample = 16
        fmt.nBlockAlign = self._channels * 2
        fmt.nAvgBytesPerSec = self._sample_rate * fmt.nBlockAlign
        fmt.cbSize = 0
        handle = ctypes.c_void_p()
        opened = winmm.waveOutOpen(
            ctypes.byref(handle), ctypes.c_uint(0xFFFFFFFF), ctypes.byref(fmt),
            ctypes.c_void_p(0), ctypes.c_void_p(0), ctypes.c_uint(0))
        if opened != 0:
            return
        self._handle = handle
        self.active = True
        self._thread = threading.Thread(target=self._feeder, name='preview-waveout', daemon=True)
        self._thread.start()

    def start(self, samples: np.ndarray, cursor: int = 0):
        if not self.active:
            return
        # The feeder applies this, including any waveOutReset. Doing that
        # reset here blocks the UI thread for whatever is still queued.
        with self._state:
            self._samples = np.ascontiguousarray(samples, dtype=np.int16)
            self._pending_cursor = int(cursor)
            self._span += 1
            self._feeding = True
        self._wake.set()

    def set_loop(self, enabled: bool, period: int | None = None):
        with self._state:
            super().set_loop(enabled, period)

    def _feeder(self):
        while not self._quit:
            self._wake.wait(0.02)
            self._wake.clear()
            if self._quit:
                return
            with self._state:
                if self._quit:
                    return
                restart = self._span != self._applied
                span = self._span
                cursor = self._pending_cursor
                feeding = self._feeding
            if restart:
                with self._winmm_lock:
                    if self._quit:
                        return
                    self._reset_headers()
                with self._state:
                    if self._quit or span != self._span:
                        continue
                    self._begin_span(len(self._samples), cursor)
                    self._clock.reset()
                    self._paused_samples = 0
                    self._applied = span
                    self._ensure_slots()
            if not feeding:
                continue
            self._fill()

    def _ensure_slots(self):
        if self._slots:
            return
        frames = max(1, int(self._sample_rate * 0.04))
        self._slots = [_WaveSlot(frames, self._channels) for _ in range(4)]

    def _fill(self):
        with self._state:
            if self._quit or not self._feeding or self._span != self._applied:
                return
            samples = self._samples
            loop = self._loop
            period = self._period
            ready = []
            for slot in self._slots:
                if slot.queued and (slot.header.dwFlags & 1) == 0:
                    continue
                piece, self._cursor, advance = _take_playback(
                    samples, self._cursor, slot.frames, loop, period)
                if advance <= 0 and not loop:
                    continue
                raw = np.ascontiguousarray(piece, dtype=np.int16).tobytes()
                ctypes.memmove(slot.buf, raw, min(len(raw), len(slot.buf)))
                ready.append((slot, advance))
        if not ready:
            return
        with self._winmm_lock:
            if self._quit or self._handle is None:
                return
            winmm = self._winmm
            for slot, advance in ready:
                header = slot.header
                if not slot.prepared:
                    header.dwBufferLength = len(slot.buf)
                    header.lpData = ctypes.cast(slot.buf, ctypes.c_char_p)
                    prepared = winmm.waveOutPrepareHeader(
                        self._handle, ctypes.byref(header), ctypes.sizeof(header))
                    if prepared != 0:
                        self._feeding = False
                        return
                    slot.prepared = True
                wrote = winmm.waveOutWrite(self._handle, ctypes.byref(header), ctypes.sizeof(header))
                if wrote != 0:
                    self._feeding = False
                    return
                slot.queued = True
                self._clock.submitted(advance)

    def _reset_headers(self):
        winmm = self._winmm
        if self._handle is not None and self._slots:
            winmm.waveOutReset(self._handle)
            for slot in self._slots:
                if slot.prepared:
                    winmm.waveOutUnprepareHeader(
                        self._handle, ctypes.byref(slot.header), ctypes.sizeof(slot.header))
        self._slots = []

    def pause(self):
        if not self.active:
            return
        with self._state:
            self._paused_samples = self._clock.heard()
            self._feeding = False
        with self._winmm_lock:
            if self._handle is not None:
                self._winmm.waveOutPause(self._handle)

    def played_samples(self) -> int:
        if self._feeding:
            return self._clock.heard()
        return self._paused_samples

    def close(self):
        was = self.active
        self._quit = True
        self._feeding = False
        self.active = False
        self._wake.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        if not was or self._handle is None:
            return
        with self._winmm_lock:
            self._reset_headers()
            self._winmm.waveOutClose(self._handle)
            self._handle = None


class _WAVEHDR(ctypes.Structure):
    _fields_ = [
        ('lpData', ctypes.c_char_p),
        ('dwBufferLength', ctypes.c_uint),
        ('dwBytesRecorded', ctypes.c_uint),
        ('dwUser', ctypes.c_size_t),
        ('dwFlags', ctypes.c_uint),
        ('dwLoops', ctypes.c_uint),
        ('lpNext', ctypes.c_void_p),
        ('reserved', ctypes.c_size_t),
    ]


class _SoundDevicePlayer(_SampleLoop):
    def __init__(self, samples: np.ndarray, sample_rate: int, channels: int):
        import sounddevice
        self._sounddevice = sounddevice
        self._samples = samples
        self._rate = sample_rate
        self._paused_samples = 0
        self._gain = 1.0
        self._init_loop(len(samples))
        self._clock = _SmoothPlaybackClock(sample_rate)
        self._stream = sounddevice.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype='int16',
            callback=self._callback,
        )
        self.active = True

    def _callback(self, outdata, frames, _time, status):
        del status
        piece, self._cursor, advance = _take_playback(
            self._samples, self._cursor, int(frames), self._loop, self._period)
        if self._gain < 0.999:
            piece = np.clip(piece.astype(np.float32) * self._gain, -32768, 32767).astype(np.int16)
        outdata[:] = piece
        self._clock.submitted(advance)

    def set_gain(self, gain: float):
        self._gain = min(1.0, max(0.0, float(gain)))

    @property
    def running(self) -> bool:
        return bool(self._stream.active)

    def start(self, samples: np.ndarray, cursor: int = 0):
        # An explicit seek restarts the stream. A loop does not: the callback
        # wraps, and stopping here would block the UI thread on the queued period.
        if self._stream.active:
            self._stream.stop()
        self._samples = np.ascontiguousarray(samples, dtype=np.int16)
        self._begin_span(len(self._samples), cursor)
        self._paused_samples = 0
        self._clock.reset()
        self._stream.start()

    def pause(self):
        self._paused_samples = self._clock.heard()
        self._stream.stop()

    def played_samples(self) -> int:
        return self._clock.heard() if self._stream.active else self._paused_samples

    def close(self):
        self.active = False
        self._stream.close()


class _MiniaudioPlayer(_SampleLoop):
    def __init__(self, samples: np.ndarray, sample_rate: int, channels: int):
        import miniaudio
        self._miniaudio = miniaudio
        self._samples = samples
        self._channels = channels
        self._paused_samples = 0
        self._running = False
        self._gain = 1.0
        self._init_loop(len(samples))
        self._clock = _SmoothPlaybackClock(sample_rate)
        self._device = miniaudio.PlaybackDevice(
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=channels,
            sample_rate=sample_rate,
        )
        self._started = False
        self.active = True

    def _generator(self):
        # miniaudio sends the number of frames it wants, not the number of bytes.
        required_frames = yield b''
        while self._running:
            required_frames = int(required_frames or 1024)
            piece, self._cursor, advance = _take_playback(
                self._samples, self._cursor, required_frames, self._loop, self._period)
            self._clock.submitted(advance)
            if self._gain < 0.999:
                piece = np.clip(piece.astype(np.float32) * self._gain, -32768, 32767).astype(np.int16)
            required_frames = yield np.ascontiguousarray(piece).tobytes()

    def set_gain(self, gain: float):
        self._gain = min(1.0, max(0.0, float(gain)))

    @property
    def running(self) -> bool:
        return bool(self._started and self._running)

    def start(self, samples: np.ndarray, cursor: int = 0):
        # A seek restarts the device so the queued ending is not heard at the
        # new picture time. A loop must not come through here: stop() waits
        # for the WASAPI period, and that wait freezes the UI thread. The
        # generator wraps instead. WASAPI can return MA_UNAVAILABLE from a
        # second start; _reopen replaces the device once.
        samples = np.ascontiguousarray(samples, dtype=np.int16)
        if self._started:
            self._running = False
            try:
                self._device.stop()
            except Exception:
                pass
            self._started = False
        self._samples = samples
        self._begin_span(len(self._samples), cursor)
        self._paused_samples = 0
        self._clock.reset()
        self._running = True
        generator = self._generator()
        next(generator)
        try:
            self._device.start(generator)
        except Exception:
            if not self._reopen(generator):
                self.active = False
                self._running = False
                self._started = False
                return
        self._started = True

    def _reopen(self, generator) -> bool:
        """One new device after a start failure. The old one is left unusable."""
        try:
            self._device.close()
        except Exception:
            pass
        try:
            self._device = self._miniaudio.PlaybackDevice(
                output_format=self._miniaudio.SampleFormat.SIGNED16,
                nchannels=self._channels,
                sample_rate=self._clock._rate,
            )
            self._device.start(generator)
        except Exception:
            return False
        return True

    def pause(self):
        self._paused_samples = self._clock.heard()
        self._running = False
        self._started = False
        try:
            self._device.stop()
        except Exception:
            pass

    def played_samples(self) -> int:
        return self._clock.heard() if self._running else self._paused_samples

    def close(self):
        self.active = False
        self._running = False
        self._device.close()


class AnimationClip:
    """One animated file. ``frame()`` is safe to call from the UI thread."""

    def __init__(self, path: str, frames: list[np.ndarray] | None, times: np.ndarray,
                 duration: float, audio: np.ndarray | None, sample_rate: int | None,
                 decoder=None):
        self.path = path
        self.duration = max(0.0, float(duration))
        self.times = times
        self._frames = frames
        self._decoder = decoder
        self._audio = None
        self.audio_error = None
        if audio is not None and sample_rate:
            # Keep the picture on the last frame until the soundtrack ends.
            # Looping at the video duration while audio still has a tail
            # (common with AAC) seeks mid-tail and sounds early on loop.
            audio_seconds = len(audio) / float(sample_rate)
            if audio_seconds > self.duration:
                self.duration = audio_seconds
            try:
                self._audio = PcmPlayer(audio, sample_rate)
            except Exception as error:
                self.audio_error = str(error)
        self.playing = False
        self._paused_at = 0.0
        self._loop = True
        self._audio_mark = None
        self.has_audio = audio is not None and sample_rate is not None
        self._wall_origin = None
        self._lock = threading.Lock()
        self._stop = False
        self._wanted = 0.0
        self._ready = frames[0] if frames else None
        self._ready_index = 0
        self._buffer: list[tuple[float, np.ndarray]] = []
        self._generation = 0
        self._timer_held = False
        self._thread = None
        if decoder is not None:
            self._ready = decoder.frame_at(0.0)
            self._thread = threading.Thread(target=self._decode_loop, daemon=True)
            self._thread.start()
        self._apply_audio_loop()

    @property
    def loop(self) -> bool:
        return self._loop

    @loop.setter
    def loop(self, enabled: bool):
        enabled = bool(enabled)
        if self._loop and not enabled and self.playing:
            # The wall clock takes over once the soundtrack stops wrapping.
            # Anchor it to the playhead or the picture jumps to the end.
            here = self._clock()
            self._paused_at = here
            self._wall_origin = time.perf_counter() - here
            self._audio_mark = None
        self._loop = enabled
        self._apply_audio_loop()

    def _apply_audio_loop(self) -> None:
        audio = self._audio
        if audio is None or not hasattr(audio, 'set_loop'):
            return
        audio.set_loop(self._loop, self.duration)

    def _audio_is_looping(self) -> bool:
        """The callback is wrapping, so the UI must not restart the device."""
        audio = self._audio
        if audio is None or not self._loop:
            return False
        if not audio.active or not audio.running or audio.finished:
            return False
        return bool(getattr(audio, 'looping', False))

    def _audio_should_run(self) -> bool:
        """Start the device only when gain is audible.

        Mute ducks the stream with gain 0 and leaves the device running so the
        picture clock keeps moving. Stopping the device on the UI thread freezes
        the preview. A clip that begins muted starts the device on unmute.
        """
        audio = self._audio
        return audio is not None and audio.active and audio.gain > 0.0

    def set_gain(self, gain: float):
        if self._audio is None:
            return
        gain = min(1.0, max(0.0, float(gain)))
        was_audible = self._audio.gain > 0.0
        self._audio.set_gain(gain)
        audible = gain > 0.0
        if not self.playing or not self._audio.active:
            return
        # Mute/unmute while the device is already running is gain only. play()
        # and pause() stop the backend and stall the UI paint path.
        if not was_audible and audible and not self._audio.running:
            self._audio.play(self._clock())

    def start(self):
        """Play from the start."""
        already = self.playing
        self.playing = True
        self._paused_at = 0.0
        self._wall_origin = time.perf_counter()
        self._audio_mark = None
        with self._lock:
            self._wanted = 0.0
        if not already:
            self._hold_timer()
        if self._audio_should_run():
            self._audio.play(0.0)

    def pause(self):
        if not self.playing:
            return
        self._paused_at = self._clock()
        self.playing = False
        self._wall_origin = None
        self._audio_mark = None
        self._drop_timer()
        if self._audio is not None:
            self._audio.pause()

    def toggle(self):
        if self.playing:
            self.pause()
        else:
            self.playing = True
            self._wall_origin = time.perf_counter() - self._paused_at
            self._hold_timer()
            if self._audio_should_run():
                self._audio.play(self._paused_at)

    def _hold_timer(self):
        if self._timer_held:
            return
        self._timer_held = True
        acquire_playback_timer()

    def _drop_timer(self):
        if not self._timer_held:
            return
        self._timer_held = False
        release_playback_timer()

    def seek(self, seconds: float):
        if self.duration > 0:
            seconds = min(max(0.0, seconds), self.duration)
        else:
            seconds = max(0.0, seconds)
        self._paused_at = seconds
        self._wall_origin = time.perf_counter() - seconds
        self._audio_mark = None
        self._recycle_picture(seconds)
        if self._audio is None or not self._audio.active:
            return
        if not self.playing:
            self._audio.pause()
            return
        # Keep a muted stream on the new playhead so unmute stays aligned.
        if self._audio.running or self._audio_should_run():
            self._audio.play(seconds)

    def _clock(self) -> float:
        audio = self._audio
        # Follow the soundtrack only while the device is consuming samples.
        # A muted load never calls play(), so position stays 0; using that as
        # the clock leaves the picture on the first frame. Wall clock covers
        # muted starts, short finished tracks, and missing audio.
        if (audio is not None and audio.active and self.playing
                and audio.running and not audio.finished):
            return audio.position
        if self._wall_origin is None:
            return self._paused_at
        return time.perf_counter() - self._wall_origin

    def _recycle_picture(self, seconds: float):
        """Drop decoded frames so the picture follows a new playhead.

        The audio device is left running. Restarting it flushes the queued
        period on this thread, which is the UI thread during a preview tick.
        """
        with self._lock:
            self._wanted = seconds
            self._generation += 1
            self._buffer.clear()
        if self._decoder is not None and not self._stop and (self._thread is None or not self._thread.is_alive()):
            self._thread = threading.Thread(target=self._decode_loop, daemon=True)
            self._thread.start()

    def _note_loop_wrap(self, current: float):
        mark = self._audio_mark
        self._audio_mark = current
        if mark is not None and current < mark:
            self._recycle_picture(max(0.0, current))

    def time(self) -> float:
        if not self.playing:
            return self._paused_at
        current = self._clock()
        # The callback is already wrapping. seek() would call play(), and
        # play() stops the device on this thread until the driver flushes.
        if self._audio_is_looping():
            self._note_loop_wrap(current)
            return max(0.0, current)
        if self.duration > 0 and current >= max(0.0, self.duration - 1e-3):
            if self.loop:
                self.seek(0.0)
                return 0.0
            self.pause()
            self._paused_at = self.duration
            return self.duration
        return max(0.0, current)

    @property
    def audio_active(self) -> bool:
        return self._audio is not None and self._audio.active

    def frame(self) -> np.ndarray | None:
        current = self.time()
        if self._frames is not None:
            index = frame_index_at(self.times, current)
            return self._frames[index]
        with self._lock:
            self._wanted = current
            chosen = self._ready
            covered = False
            upcoming = None
            for stamp, buffered in self._buffer:
                if stamp <= current + 1e-3:
                    chosen = buffered
                    covered = True
                else:
                    upcoming = buffered
                    break
            # A short buffer can sit entirely in the future of a 60fps stream.
            # Showing that next frame keeps the picture moving with the audio.
            if not covered and upcoming is not None:
                chosen = upcoming
            if chosen is not None:
                self._ready = chosen
            return self._ready

    def millis_until_next_frame(self) -> int:
        """Wait until the next source frame, from 1ms to 100ms."""
        if not self.playing:
            return 16
        now = self.time()
        delay = int(max(0.0, self._following_time(now) - now) * 1000)
        return min(100, max(1, delay))

    def _following_time(self, now: float) -> float:
        if self._frames is not None and len(self.times):
            index = frame_index_at(self.times, now)
            if index + 1 < len(self.times):
                return float(self.times[index + 1])
            if self.duration > now:
                return self.duration
            return now + 0.04
        step = self._decoder.step if self._decoder is not None else 1.0 / 24.0
        with self._lock:
            upcoming = [stamp for stamp, _frame in self._buffer if stamp > now + 1e-4]
        if upcoming:
            return upcoming[0]
        return now + step

    def close(self):
        self._stop = True
        self._drop_timer()
        self.playing = False
        if self._audio is not None:
            self._audio.close()
            self._audio = None
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=1.0)
        if self._decoder is not None:
            self._decoder.close()
            self._decoder = None

    def _decode_loop(self):
        """Keep the next half-second of frames decoded, without sleeping past them."""
        while not self._stop:
            with self._lock:
                generation = self._generation
                wanted = self._wanted
                last_time = self._buffer[-1][0] if self._buffer else None
            step = self._decoder.step if self._decoder is not None else 1.0 / 24.0
            if last_time is not None and last_time >= wanted + _LOOKAHEAD_SECONDS:
                time.sleep(0.002)
                continue
            request = wanted if last_time is None else last_time + step
            try:
                decoder = self._decoder
                if decoder is None:
                    return
                frame = decoder.frame_at(request)
                stamp = decoder.time
            except Exception:
                # One bad frame must not end this thread. Repeat the last
                # picture at this timestamp so the next request moves on.
                if self._stop or self._decoder is None:
                    return
                stamp = request
                with self._lock:
                    frame = self._buffer[-1][1] if self._buffer else self._ready
                if frame is None:
                    time.sleep(0.02)
                    continue
            if self._stop:
                return
            stale = False
            with self._lock:
                if generation != self._generation:
                    continue
                if self._buffer and stamp <= self._buffer[-1][0] + 1e-4:
                    stale = True
                else:
                    self._buffer.append((stamp, frame))
                    while len(self._buffer) > 2 and self._buffer[1][0] <= self._wanted:
                        self._buffer.pop(0)
                    # 16 frames is a quarter of a second at 60fps. The lookahead
                    # is half a second, so that cap kept only future frames and
                    # the picture froze on the last one the playhead had seen.
                    limit = self._buffered_frame_limit()
                    while len(self._buffer) > limit:
                        self._buffer.pop(0)
            if stale:
                time.sleep(0.002)

    def _buffered_frame_limit(self) -> int:
        step = self._decoder.step if self._decoder is not None else 1.0 / 24.0
        if step <= 0:
            step = 1.0 / 24.0
        return min(120, max(16, int(_LOOKAHEAD_SECONDS / step) + 8))


class _StreamDecoder:
    def __init__(self, container, stream):
        self._container = container
        self._stream = stream
        self._iter = None
        self._frame = None
        self._time = 0.0
        rate = stream.average_rate
        self._step = 1.0 / float(rate) if rate else 1.0 / 24.0
        self._blank = np.zeros((2, 2, 3), dtype=np.uint8)

    @property
    def step(self) -> float:
        return self._step

    @property
    def time(self) -> float:
        return self._time

    def frame_at(self, seconds: float) -> np.ndarray:
        if self._frame is not None and self._time > seconds + 1e-3:
            self._seek(seconds)
        if self._iter is None:
            self._seek(seconds)
        while True:
            if self._frame is not None and self._time <= seconds < self._time + self._step:
                return self._frame
            if self._frame is not None and self._time > seconds:
                return self._frame
            try:
                decoded = next(self._iter)
            except StopIteration:
                return self._frame if self._frame is not None else self._blank
            if decoded.pts is None:
                stamp = self._time + self._step
            else:
                stamp = float(decoded.pts * self._stream.time_base)
            # OpenGL rejects a row-padded view from the decoder. A packed copy
            # is what glTexSubImage2D can upload, including widths such as 990.
            self._frame = np.ascontiguousarray(decoded.to_ndarray(format='rgb24'))
            self._time = stamp

    def _seek(self, seconds: float):
        time_base = self._stream.time_base
        pts = int(max(0.0, seconds) / float(time_base)) if time_base else 0
        self._container.seek(pts, stream=self._stream, backward=True, any_frame=False)
        self._iter = self._container.decode(self._stream)
        self._frame = None
        self._time = 0.0

    def close(self):
        self._container.close()


def open_animation(path: str) -> AnimationClip | None:
    """Return a clip when ``path`` is an animated image or a video, else ``None``."""
    extension = os.path.splitext(path)[1].lower()
    if extension in _PIL_ANIMATED:
        clip = _open_pil_animation(path)
        if clip is not None:
            return clip
    if extension in _VIDEO:
        return _open_video(path)
    return None


def _open_pil_animation(path: str) -> AnimationClip | None:
    image = PIL.Image.open(path)
    frames = getattr(image, 'n_frames', 1)
    if frames <= 1:
        image.close()
        return None
    arrays = []
    times = []
    cursor = 0.0
    for index in range(frames):
        image.seek(index)
        delay = image.info.get('duration', 100) or 100
        arrays.append(np.asarray(image.convert('RGB')))
        times.append(cursor)
        cursor += max(0.02, float(delay) / 1000.0)
    image.close()
    return AnimationClip(path, arrays, np.asarray(times, dtype=np.float64), cursor, None, None)


def _open_video(path: str) -> AnimationClip | None:
    try:
        import av
    except ImportError:
        return None
    container = av.open(path)
    video_streams = [stream for stream in container.streams if stream.type == 'video']
    if not video_streams:
        container.close()
        return None
    stream = video_streams[0]
    stream.thread_type = 'AUTO'
    rate = float(stream.average_rate) if stream.average_rate else 24.0
    if stream.duration is not None and stream.time_base is not None:
        duration = float(stream.duration * stream.time_base)
    elif container.duration and container.time_base:
        duration = float(container.duration * container.time_base)
    else:
        duration = 0.0
    try:
        audio, sample_rate = _decode_audio(container)
    except Exception:
        audio, sample_rate = None, None
    try:
        container.seek(0)
    except Exception:
        pass
    width = int(stream.codec_context.width or 0)
    height = int(stream.codec_context.height or 0)
    frame_bytes = max(1, width * height * 3)
    estimated = int(duration * rate) if duration else 0
    if estimated and estimated * frame_bytes <= _PRELOAD_BYTES:
        frames, times, duration = _preload_video(container, stream, rate)
        container.close()
        if not frames:
            return None
        return AnimationClip(path, frames, np.asarray(times, dtype=np.float64), duration, audio, sample_rate)
    return AnimationClip(
        path, None, np.zeros(0, dtype=np.float64), duration or 1.0, audio, sample_rate,
        decoder=_StreamDecoder(container, stream))


def _preload_video(container, stream, rate: float):
    frames = []
    times = []
    step = 1.0 / rate if rate else 1.0 / 24.0
    for decoded in container.decode(stream):
        if decoded.pts is None:
            stamp = len(frames) * step
        else:
            stamp = float(decoded.pts * stream.time_base)
        frames.append(np.ascontiguousarray(decoded.to_ndarray(format='rgb24')))
        times.append(stamp)
        if sum(frame.nbytes for frame in frames) > _PRELOAD_BYTES:
            break
    duration = (times[-1] + step) if times else 0.0
    return frames, times, duration


def _pcm_matrix(array: np.ndarray, samples: int, channels: int) -> np.ndarray:
    """Frames by channels, from either planar ``(channels, samples)`` or packed audio.

    Packed s16 stereo is one row of interleaved samples, ``(1, samples * channels)``.
    Treating that row as mono frames plays the clip at half speed.
    """
    channels = max(1, int(channels))
    array = np.asarray(array)
    if samples <= 0:
        samples = int(array.size // channels)
    if array.ndim == 2 and array.shape == (channels, samples):
        array = array.T
    else:
        array = array.reshape(samples, channels)
    return np.ascontiguousarray(array, dtype=np.int16)


def _decode_audio(container) -> tuple[np.ndarray | None, int | None]:
    audio_streams = [stream for stream in container.streams if stream.type == 'audio']
    if not audio_streams:
        return None, None
    stream = audio_streams[0]
    rate = int(stream.codec_context.sample_rate or stream.rate or 0)
    if rate <= 0:
        return None, None
    try:
        import av
        resampler = av.AudioResampler(format='s16', layout='stereo', rate=rate)
    except Exception:
        return None, None
    chunks = []
    for decoded in container.decode(stream):
        converted = resampler.resample(decoded)
        if converted is None:
            continue
        if not isinstance(converted, list):
            converted = [converted]
        for frame in converted:
            channels = len(frame.layout.channels) if frame.layout is not None else 1
            chunks.append(_pcm_matrix(frame.to_ndarray(), int(frame.samples or 0), channels))
    try:
        remainder = resampler.resample(None)
    except Exception:
        remainder = None
    if remainder:
        if not isinstance(remainder, list):
            remainder = [remainder]
        for frame in remainder:
            channels = len(frame.layout.channels) if frame.layout is not None else 1
            chunks.append(_pcm_matrix(frame.to_ndarray(), int(frame.samples or 0), channels))
    if not chunks:
        return None, None
    return np.concatenate(chunks, axis=0), rate
