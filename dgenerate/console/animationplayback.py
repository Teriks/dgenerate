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

    @property
    def position(self) -> float:
        if self._backend is None:
            return self._offset / self.sample_rate if self.sample_rate else 0.0
        played = self._backend.played_samples()
        return (self._offset + played) / self.sample_rate

    @property
    def finished(self) -> bool:
        """The samples handed to the device have all been heard."""
        if not self.active or self._paused or self._backend is None or self.sample_rate <= 0:
            return False
        total = len(self._samples) - self._offset
        return total > 0 and self._backend.played_samples() >= total - 1

    def play(self, seconds: float):
        if self._backend is None or self.sample_rate <= 0:
            return
        self._offset = max(0, min(int(seconds * self.sample_rate), len(self._samples)))
        self._paused = False
        self._start_backend(self._samples[self._offset:])

    def pause(self):
        if self._backend is not None and not self._paused:
            self._offset = max(0, min(int(self.position * self.sample_rate), len(self._samples)))
            self._backend.pause()
            self._paused = True

    def resume(self):
        if self._backend is None or not self._paused:
            return
        self._paused = False
        self._start_backend(self._samples[self._offset:])

    def _start_backend(self, samples: np.ndarray):
        try:
            self._backend.start(samples)
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


class _WaveOutPlayer:
    def __init__(self, samples: np.ndarray, sample_rate: int, channels: int):
        self.active = False
        self._sample_rate = sample_rate
        self._channels = channels
        self._winmm = ctypes.windll.winmm
        self._handle = ctypes.c_void_p()
        self._headers = []
        self._buffers = []
        self._lock = threading.Lock()
        self._played_base = 0
        self._gain = 1.0
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
        return bool(self.active and self._headers)

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

    def start(self, samples: np.ndarray):
        if not self.active:
            return
        with self._lock:
            self._reset_headers()
            if len(samples) == 0:
                self._played_base = 0
                return
            data = np.ascontiguousarray(samples, dtype=np.int16).tobytes()
            chunk = self._sample_rate * self._channels * 2
            self._buffers = []
            self._headers = []
            winmm = self._winmm
            for start in range(0, len(data), chunk):
                raw = data[start:start + chunk]
                buf = ctypes.create_string_buffer(raw)
                header = _WAVEHDR()
                header.lpData = ctypes.cast(buf, ctypes.c_char_p)
                header.dwBufferLength = len(raw)
                winmm.waveOutPrepareHeader(self._handle, ctypes.byref(header), ctypes.sizeof(header))
                winmm.waveOutWrite(self._handle, ctypes.byref(header), ctypes.sizeof(header))
                self._buffers.append(buf)
                self._headers.append(header)
            self._played_base = 0

    def _reset_headers(self):
        winmm = self._winmm
        if self._headers:
            winmm.waveOutReset(self._handle)
            for header in self._headers:
                winmm.waveOutUnprepareHeader(self._handle, ctypes.byref(header), ctypes.sizeof(header))
        self._headers = []
        self._buffers = []

    def pause(self):
        if self.active:
            self._winmm.waveOutPause(self._handle)

    def played_samples(self) -> int:
        if not self.active:
            return 0
        time_info = _MMTIME()
        time_info.wType = 2
        self._winmm.waveOutGetPosition(self._handle, ctypes.byref(time_info), ctypes.sizeof(time_info))
        return int(time_info.sample) // self._channels

    def close(self):
        if not self.active:
            return
        self.active = False
        with self._lock:
            self._reset_headers()
            self._winmm.waveOutClose(self._handle)


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


class _MMTIME(ctypes.Structure):
    _fields_ = [
        ('wType', ctypes.c_uint),
        ('sample', ctypes.c_uint),
        ('pad', ctypes.c_uint),
    ]


class _SoundDevicePlayer:
    def __init__(self, samples: np.ndarray, sample_rate: int, channels: int):
        import sounddevice
        self._sounddevice = sounddevice
        self._samples = samples
        self._rate = sample_rate
        self._offset = 0
        self._paused_samples = 0
        self._gain = 1.0
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
        start = self._offset
        end = min(start + frames, len(self._samples))
        count = end - start
        if count > 0:
            chunk = self._samples[start:end]
            if self._gain < 0.999:
                chunk = np.clip(chunk.astype(np.float32) * self._gain, -32768, 32767).astype(np.int16)
            outdata[:count] = chunk
        if count < frames:
            outdata[count:] = 0
        self._offset = end
        self._clock.submitted(count)

    def set_gain(self, gain: float):
        self._gain = min(1.0, max(0.0, float(gain)))

    @property
    def running(self) -> bool:
        return bool(self._stream.active)

    def start(self, samples: np.ndarray):
        self._samples = np.ascontiguousarray(samples, dtype=np.int16)
        self._offset = 0
        self._paused_samples = 0
        self._clock.reset()
        # A seek or loop must drop queued output. Restarting the stream with
        # a new offset while the device still holds the previous ending leaves
        # the picture at the new time and the speaker on the old one.
        if self._stream.active:
            self._stream.stop()
        self._stream.start()

    def pause(self):
        self._paused_samples = self._clock.heard()
        self._stream.stop()

    def played_samples(self) -> int:
        return self._clock.heard() if self._stream.active else self._paused_samples

    def close(self):
        self.active = False
        self._stream.close()


class _MiniaudioPlayer:
    def __init__(self, samples: np.ndarray, sample_rate: int, channels: int):
        import miniaudio
        self._miniaudio = miniaudio
        self._samples = samples
        self._channels = channels
        self._cursor = 0
        self._paused_samples = 0
        self._running = False
        self._gain = 1.0
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
            required_frames = required_frames or 1024
            start = self._cursor
            end = min(start + int(required_frames), len(self._samples))
            piece = self._samples[start:end]
            self._cursor = end
            self._clock.submitted(end - start)
            if self._gain < 0.999:
                piece = np.clip(piece.astype(np.float32) * self._gain, -32768, 32767).astype(np.int16)
            payload = np.ascontiguousarray(piece).tobytes()
            missing = int(required_frames) - len(piece)
            if missing > 0:
                payload += bytes(missing * self._channels * 2)
            required_frames = yield payload

    def set_gain(self, gain: float):
        self._gain = min(1.0, max(0.0, float(gain)))

    @property
    def running(self) -> bool:
        return bool(self._started and self._running)

    def start(self, samples: np.ndarray):
        # A loop or seek must flush the device. Leaving the previous ending in
        # the hardware queue plays it over the new picture start. Stopping and
        # starting again can return MA_UNAVAILABLE on WASAPI; _reopen handles
        # that with one new device.
        self._samples = np.ascontiguousarray(samples, dtype=np.int16)
        self._cursor = 0
        self._paused_samples = 0
        self._clock.reset()
        self._running = True
        if self._started:
            try:
                self._device.stop()
            except Exception:
                pass
            self._started = False
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
        self.loop = True
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
        with self._lock:
            self._wanted = seconds
            self._generation += 1
            self._buffer.clear()
        if self._decoder is not None and not self._stop and (self._thread is None or not self._thread.is_alive()):
            self._thread = threading.Thread(target=self._decode_loop, daemon=True)
            self._thread.start()
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

    def time(self) -> float:
        if not self.playing:
            return self._paused_at
        current = self._clock()
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
                frame = self._decoder.frame_at(request)
                stamp = self._decoder.time
            except Exception:
                return
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
