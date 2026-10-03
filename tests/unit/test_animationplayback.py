import os
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import PIL.Image

from dgenerate.console.animationplayback import (
    BAR_HEIGHT,
    CONTROL_GAP,
    AnimationClip,
    PcmPlayer,
    _MiniaudioPlayer,
    _SmoothPlaybackClock,
    control_hit,
    control_layout,
    format_clock,
    frame_index_at,
    load_preview_audio,
    open_animation,
    _pcm_matrix,
    picture_height,
    playback_tick_delay_ms,
    speaker_icon,
    save_preview_audio,
    time_on_track,
    volume_on_slider,
)


class TestAnimationTimeline(unittest.TestCase):

    def test_frame_index_follows_start_times(self):
        times = np.array([0.0, 0.1, 0.25])
        self.assertEqual(frame_index_at(times, 0.0), 0)
        self.assertEqual(frame_index_at(times, 0.09), 0)
        self.assertEqual(frame_index_at(times, 0.1), 1)
        self.assertEqual(frame_index_at(times, 1.0), 2)

    def test_hover_bar_hits_play_and_track(self):
        self.assertEqual(control_hit(20, 270, 400, 300), 'play')
        self.assertEqual(control_hit(200, 276, 400, 300), 'track')
        self.assertEqual(control_hit(300, 270, 400, 300), 'loop')
        self.assertEqual(control_hit(330, 270, 400, 300), 'mute')
        self.assertEqual(control_hit(360, 276, 400, 300), 'volume')
        self.assertIsNone(control_hit(20, 20, 400, 300))
        self.assertAlmostEqual(time_on_track(108, 400, 300, 8.0), 0.0)
        self.assertGreater(time_on_track(200, 400, 300, 8.0), 0.0)
        self.assertAlmostEqual(volume_on_slider(350, 400, 300), 0.0)
        self.assertAlmostEqual(volume_on_slider(388, 400, 300), 1.0)

    def test_volume_settings_keep_other_console_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'console_settings.json')
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write('{"theme": "dgenerate"}')
            save_preview_audio(0.4, True, path=__import__('pathlib').Path(path))
            volume, muted = load_preview_audio(__import__('pathlib').Path(path))
            self.assertAlmostEqual(volume, 0.4)
            self.assertTrue(muted)
            with open(path, encoding='utf-8') as handle:
                saved = __import__('json').loads(handle.read())
            self.assertEqual(saved['theme'], 'dgenerate')

    def test_speaker_icon_is_a_flared_cone_with_waves_or_an_x(self):
        rect = control_layout(800, 600)['mute']
        audible = speaker_icon(rect, muted=False, volume=1.0)
        low = speaker_icon(rect, muted=False, volume=0.2)
        quiet = speaker_icon(rect, muted=True, volume=1.0)
        silent = speaker_icon(rect, muted=False, volume=0.0)

        def strokes(shapes):
            return [shape for shape in shapes if shape[0] == 'stroke']

        cone = next(shape for shape in audible if shape[0] == 'poly')
        left = [point[1] for point in cone[1] if point[0] < rect[0] + 8]
        right = [point[1] for point in cone[1] if point[0] >= rect[0] + 8]
        self.assertGreater(max(right) - min(right), max(left) - min(left))
        self.assertEqual(len(strokes(audible)), 2)
        self.assertEqual(len(strokes(low)), 1)
        self.assertEqual(len(strokes(quiet)), 2)
        self.assertEqual(strokes(silent), strokes(quiet))
        self.assertLess(strokes(quiet)[0][-1][1], 0.5)
        for shapes in (audible, low, quiet):
            for kind, *rest in shapes:
                if kind == 'rect':
                    points = [(rest[0], rest[1]), (rest[2], rest[3])]
                elif kind == 'poly':
                    points = rest[0]
                else:
                    points = rest[0]
                for x, y in points:
                    self.assertGreaterEqual(x, rect[0])
                    self.assertLessEqual(x, rect[2])
                    self.assertGreaterEqual(y, rect[1])
                    self.assertLessEqual(y, rect[3])

    def test_picture_sits_above_the_control_strip(self):
        self.assertEqual(picture_height(300), 300 - BAR_HEIGHT - CONTROL_GAP)
        self.assertGreaterEqual(CONTROL_GAP, 1)
        self.assertEqual(picture_height(10), 1)

    def test_clip_loops_when_the_clock_passes_the_end(self):
        frames = [np.zeros((2, 2, 3), np.uint8)]
        clip = AnimationClip('clip', frames, np.array([0.0]), 0.4, None, None)
        try:
            clip.start()
            clip._wall_origin = time.perf_counter() - 1.0
            self.assertEqual(clip.time(), 0.0)
            self.assertTrue(clip.playing)
            clip.loop = False
            clip._wall_origin = time.perf_counter() - 1.0
            self.assertAlmostEqual(clip.time(), 0.4)
            self.assertFalse(clip.playing)
        finally:
            clip.close()

    def test_short_audio_does_not_freeze_the_picture_before_the_end(self):
        frames = [np.zeros((2, 2, 3), np.uint8)]
        clip = AnimationClip('clip', frames, np.array([0.0]), 2.0, None, None)

        class _Ended:
            active = True
            finished = True
            paused = False
            running = True
            gain = 1.0
            position = 0.5

            def play(self, _seconds):
                return None

            def pause(self):
                return None

            def set_gain(self, gain):
                self.gain = gain

        clip._audio = _Ended()
        try:
            clip.start()
            clip._wall_origin = time.perf_counter() - 2.5
            self.assertEqual(clip.time(), 0.0)
            self.assertTrue(clip.playing)
        finally:
            clip._audio = None
            clip.close()

    def test_clip_duration_covers_a_longer_soundtrack(self):
        frames = [np.zeros((2, 2, 3), np.uint8)]
        # 48000 samples at 48kHz is 1.0s; video times only reach ~0.53s.
        audio = np.zeros((48000, 2), dtype=np.int16)

        class _SilentPcm:
            def __init__(self, _samples, _rate):
                self.active = False

            def close(self):
                return None

        with patch('dgenerate.console.animationplayback.PcmPlayer', _SilentPcm):
            clip = AnimationClip(
                'clip', frames, np.array([0.0, 0.5]), 0.5 + 1.0 / 30.0, audio, 48000)
            try:
                self.assertAlmostEqual(clip.duration, 1.0, places=5)
                self.assertTrue(clip.has_audio)
            finally:
                clip.close()

    def test_miniaudio_start_stops_before_a_second_start(self):
        class _Device:
            def __init__(self):
                self.stopped = 0
                self.starts = 0

            def stop(self):
                self.stopped += 1

            def start(self, _generator):
                self.starts += 1

            def close(self):
                return None

        player = _MiniaudioPlayer.__new__(_MiniaudioPlayer)
        player._miniaudio = None
        player._samples = np.zeros((8, 1), dtype=np.int16)
        player._channels = 1
        player._cursor = 0
        player._paused_samples = 0
        player._running = False
        player._gain = 1.0
        player._clock = _SmoothPlaybackClock(8000)
        player._device = _Device()
        player._started = True
        player.active = True
        player.start(np.zeros((8, 1), dtype=np.int16))
        self.assertEqual(player._device.stopped, 1)
        self.assertEqual(player._device.starts, 1)
        self.assertTrue(player._started)

    def test_mute_ducks_gain_without_stopping_the_device(self):
        frames = [np.zeros((2, 2, 3), np.uint8)]
        clip = AnimationClip('clip', frames, np.array([0.0]), 4.0, None, None)

        class _Transport:
            active = True
            finished = False
            paused = False
            running = True
            gain = 1.0
            position = 0.25
            played_at = None
            pause_calls = 0

            def play(self, seconds):
                self.paused = False
                self.running = True
                self.played_at = seconds
                self.position = seconds

            def pause(self):
                self.pause_calls += 1
                self.paused = True
                self.running = False

            def set_gain(self, gain):
                self.gain = gain

        audio = _Transport()
        clip._audio = audio
        try:
            clip.start()
            # start() seeks to 0; clear that so unmute can be checked alone.
            self.assertEqual(audio.played_at, 0.0)
            audio.played_at = None
            audio.position = 0.25
            clip.set_gain(0.0)
            self.assertEqual(audio.pause_calls, 0)
            self.assertTrue(audio.running)
            self.assertEqual(audio.gain, 0.0)
            audio.position = 0.40
            self.assertAlmostEqual(clip.time(), 0.40, places=3)
            clip.set_gain(0.8)
            self.assertIsNone(audio.played_at)
            self.assertEqual(audio.gain, 0.8)
        finally:
            clip._audio = None
            clip.close()

    def test_unmute_starts_audio_when_playback_began_muted(self):
        frames = [np.zeros((2, 2, 3), np.uint8)]
        clip = AnimationClip('clip', frames, np.array([0.0]), 4.0, None, None)

        class _Transport:
            active = True
            finished = False
            paused = False
            running = False
            gain = 0.0
            position = 0.0
            played_at = None

            def play(self, seconds):
                self.paused = False
                self.running = True
                self.played_at = seconds
                self.position = seconds

            def pause(self):
                self.paused = True
                self.running = False

            def set_gain(self, gain):
                self.gain = gain

        audio = _Transport()
        clip._audio = audio
        try:
            clip.set_gain(0.0)
            clip.start()
            self.assertIsNone(audio.played_at)
            clip._wall_origin = time.perf_counter() - 0.3
            clip.set_gain(1.0)
            self.assertAlmostEqual(audio.played_at, 0.3, delta=0.05)
            self.assertTrue(audio.running)
        finally:
            clip._audio = None
            clip.close()

    def test_muted_load_advances_on_the_wall_clock(self):
        frames = [np.zeros((2, 2, 3), np.uint8) for _ in range(3)]
        times = np.array([0.0, 0.05, 0.10])
        clip = AnimationClip('clip', frames, times, 0.15, None, None)

        class _Transport:
            active = True
            finished = False
            paused = False
            running = False
            gain = 0.0
            position = 0.0

            def play(self, _seconds):
                self.running = True

            def pause(self):
                self.running = False

            def set_gain(self, gain):
                self.gain = gain

        clip._audio = _Transport()
        try:
            clip.set_gain(0.0)
            clip.start()
            self.assertTrue(clip.playing)
            self.assertFalse(clip._audio.running)
            clip._wall_origin = time.perf_counter() - 0.08
            self.assertGreaterEqual(clip.time(), 0.07)
            frame = clip.frame()
            self.assertIs(frame, frames[1])
        finally:
            clip._audio = None
            clip.close()

    def test_packed_stereo_is_not_played_as_twice_as_many_mono_frames(self):
        packed = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int16)
        matrix = _pcm_matrix(packed, samples=3, channels=2)
        self.assertEqual(matrix.shape, (3, 2))
        self.assertEqual(matrix.tolist(), [[1, 2], [3, 4], [5, 6]])
        planar = np.array([[1, 3, 5], [2, 4, 6]], dtype=np.int16)
        self.assertEqual(_pcm_matrix(planar, samples=3, channels=2).tolist(), [[1, 2], [3, 4], [5, 6]])

    def test_audio_clock_does_not_jump_a_device_period(self):
        rate = 48000
        clock = _SmoothPlaybackClock(rate)
        # Two 200ms periods queued before anything is heard, which is what
        # the miniaudio device was doing. The playhead has to stay at zero
        # until those samples actually play.
        clock.submitted(rate // 5)
        clock.submitted(rate // 5)
        self.assertEqual(clock.heard(), 0)
        started = time.perf_counter()
        time.sleep(0.04)
        elapsed = time.perf_counter() - started
        heard = clock.heard()
        self.assertAlmostEqual(heard, elapsed * rate, delta=rate * 0.015)
        self.assertLess(heard, rate // 5)
        before = clock.heard()
        clock.submitted(rate // 5)
        self.assertLess(clock.heard() - before, rate * 0.025)

    def test_playback_tick_waits_for_the_next_source_frame(self):
        times = np.array([0.0, 0.04, 0.08])
        frames = [np.zeros((2, 2, 3), np.uint8) for _ in times]
        clip = AnimationClip('clip', frames, times, 0.12, None, None)
        try:
            clip.start()
            delay = clip.millis_until_next_frame()
            self.assertGreaterEqual(delay, 20)
            self.assertLessEqual(delay, 40)
            self.assertEqual(playback_tick_delay_ms(clip, timeline_moving=False), delay)
            self.assertLessEqual(playback_tick_delay_ms(clip, timeline_moving=True), 16)
        finally:
            clip.close()

    def test_clock_text(self):
        self.assertEqual(format_clock(0), '0:00')
        self.assertEqual(format_clock(65), '1:05')


class TestAnimationFiles(unittest.TestCase):

    def test_gif_frames_and_duration(self):
        frames = [PIL.Image.new('RGB', (8, 8), color) for color in ((255, 0, 0), (0, 255, 0), (0, 0, 255))]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'clip.gif')
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
            clip = open_animation(path)
            self.assertIsNotNone(clip)
            try:
                self.assertEqual(len(clip.times), 3)
                self.assertGreater(clip.duration, 0.2)
                self.assertEqual(tuple(clip.frame().shape), (8, 8, 3))
                self.assertFalse(clip.has_audio)
            finally:
                clip.close()

    def test_sixty_fps_stream_keeps_a_frame_at_the_playhead(self):
        class Decoder:
            def __init__(self):
                self.step = 1.0 / 60.0
                self._index = 0
                self._time = 0.0

            @property
            def time(self):
                return self._time

            def frame_at(self, _seconds):
                self._time = self._index * self.step
                frame = np.full((2, 2, 3), self._index & 255, np.uint8)
                self._index += 1
                return frame

            def close(self):
                pass

        clip = AnimationClip(
            'rate.mp4', None, np.zeros(0, np.float64), 3.0, None, None, decoder=Decoder())
        try:
            clip.start()
            deadline = time.perf_counter() + 3.0
            while clip.time() < 0.8 and time.perf_counter() < deadline:
                time.sleep(0.02)
            first = int(clip.frame()[0, 0, 0])
            while clip.time() < 1.3 and time.perf_counter() < deadline:
                time.sleep(0.02)
            second = int(clip.frame()[0, 0, 0])
            self.assertGreater(clip.time(), 1.0)
            self.assertGreater(second, first + 10)
            self.assertAlmostEqual(second / 60.0, clip.time(), delta=0.2)
        finally:
            clip.close()

    def test_mp4_frames_and_audio(self):
        av = __import__('av')
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'clip.mp4')
            container = av.open(path, 'w')
            video = container.add_stream('h264', rate=10)
            video.width = 32
            video.height = 32
            video.pix_fmt = 'yuv420p'
            audio = container.add_stream('aac', rate=8000)
            for index in range(4):
                picture = np.full((32, 32, 3), index * 40, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(picture, format='rgb24').reformat(format='yuv420p')
                frame.pts = index
                for packet in video.encode(frame):
                    container.mux(packet)
            tone = np.zeros((1, 3200), dtype=np.float32)
            sound = av.AudioFrame.from_ndarray(tone, format='fltp', layout='mono')
            sound.sample_rate = 8000
            sound.pts = 0
            for packet in audio.encode(sound):
                container.mux(packet)
            for packet in video.encode(None):
                container.mux(packet)
            for packet in audio.encode(None):
                container.mux(packet)
            container.close()

            clip = open_animation(path)
            self.assertIsNotNone(clip)
            try:
                self.assertGreaterEqual(len(clip.times), 1)
                self.assertEqual(clip.frame().shape[2], 3)
                self.assertTrue(clip.has_audio)
            finally:
                clip.close()


class TestPreviewAudioRestart(unittest.TestCase):

    def test_loop_flushes_and_restarts_the_device(self):
        """A loop must drop queued ending samples before the new start plays."""
        player = _MiniaudioPlayer.__new__(_MiniaudioPlayer)
        player._miniaudio = None
        player._channels = 1
        player._cursor = 40
        player._paused_samples = 3
        player._running = True
        player._gain = 1.0
        player._clock = _SmoothPlaybackClock(8000)
        player._started = True
        player.active = True
        player._samples = np.ones((80, 1), dtype=np.int16)

        class Device:
            def __init__(self):
                self.starts = 0
                self.stopped = 0

            def stop(self):
                self.stopped += 1

            def start(self, _generator):
                self.starts += 1

            def close(self):
                return None

        player._device = Device()
        player.start(np.zeros((80, 1), dtype=np.int16))
        self.assertEqual(player._device.stopped, 1)
        self.assertEqual(player._device.starts, 1)
        self.assertEqual(player._cursor, 0)
        self.assertTrue(player.active)
        self.assertTrue(player._running)
        self.assertTrue(player._started)

    def test_play_drops_a_backend_that_cannot_start(self):
        player = PcmPlayer.__new__(PcmPlayer)
        player._samples = np.zeros((100, 1), dtype=np.int16)
        player.sample_rate = 8000
        player._offset = 0
        player._paused = True

        class Backend:
            active = False

            def start(self, samples):
                return None

            def close(self):
                self.closed = True

        backend = Backend()
        player._backend = backend
        player.active = True
        player.play(0.0)
        self.assertFalse(player.active)
        self.assertIsNone(player._backend)
        self.assertTrue(backend.closed)
        player.play(0.0)


if __name__ == '__main__':
    unittest.main()
