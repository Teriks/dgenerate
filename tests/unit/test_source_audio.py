import math
import os
import tempfile
import unittest

import av
import numpy
import PIL.Image

try:
    import dgenerate.console.console as _console
except ImportError:
    _console = None
import dgenerate.image_process.renderloop as _image_process
import dgenerate.image_process.renderloopconfig as _image_process_config
import dgenerate.mediaoutput as _mediaoutput
import dgenerate.renderloop as _renderloop


def _write_clip(path, frames, fps, sample_rate, samples):
    container = av.open(path, 'w')
    try:
        video = container.add_stream('h264', rate=fps)
        video.width = 32
        video.height = 32
        video.pix_fmt = 'yuv420p'
        audio = container.add_stream('aac', rate=sample_rate)
        audio.layout = 'mono'
        for index in range(frames):
            image = PIL.Image.new('RGB', (32, 32), (index * 20, 0, 0))
            for packet in video.encode(av.VideoFrame.from_image(image)):
                container.mux(packet)
        frame_size = audio.codec_context.frame_size or 1024
        remainder = samples.shape[1] % frame_size
        padded = samples
        if remainder:
            padded = numpy.pad(samples, ((0, 0), (0, frame_size - remainder)))
        for start in range(0, padded.shape[1], frame_size):
            chunk = numpy.ascontiguousarray(padded[:, start:start + frame_size])
            frame = av.AudioFrame.from_ndarray(chunk, format='fltp', layout='mono')
            frame.sample_rate = sample_rate
            for packet in audio.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    finally:
        container.close()


def _tone(sample_rate, seconds):
    count = int(sample_rate * seconds)
    timeline = numpy.arange(count, dtype=numpy.float32) / sample_rate
    return numpy.sin(2 * math.pi * 440 * timeline).reshape(1, -1)


class TestSourceAudio(unittest.TestCase):
    @unittest.skipIf(_console is None, 'Tk is not available')
    def test_image_process_file_line_is_previewed(self):
        text = (
            'image-process: Wrote Frame "frame.png"\n'
            '\\image_process: Wrote File "output.mp4"\n'
            '================================================\n'
        )
        self.assertEqual(_console._preview_path_from_output(text), 'output.mp4')
    def test_read_source_audio_slices_with_the_frames(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, 'source.mp4')
        fps = 10
        frames = 8
        sample_rate = 8000
        _write_clip(path, frames, fps, sample_rate, _tone(sample_rate, frames / fps))

        full = _mediaoutput.read_source_audio(path, fps=fps, frame_start=0, frame_count=frames)
        self.assertIsNotNone(full)
        audio, rate = full
        self.assertEqual(rate, sample_rate)
        self.assertGreater(audio.shape[1], sample_rate * 0.5)

        sliced = _mediaoutput.read_source_audio(path, fps=fps, frame_start=4, frame_count=4)
        self.assertIsNotNone(sliced)
        piece, _rate = sliced
        self.assertLess(piece.shape[1], audio.shape[1])
        self.assertGreater(piece.shape[1], sample_rate * 0.2)

    def test_animation_writer_keeps_the_source_track(self):
        directory = tempfile.mkdtemp()
        source = os.path.join(directory, 'source.mp4')
        output = os.path.join(directory, 'out.mp4')
        fps = 8
        frames = 4
        sample_rate = 8000
        _write_clip(source, frames, fps, sample_rate, _tone(sample_rate, frames / fps))
        loaded = _mediaoutput.read_source_audio(source, fps=fps, frame_start=0, frame_count=frames)

        writer = _mediaoutput.MultiAnimationWriter('mp4', output, fps, allow_overwrites=True)
        writer.set_source_audio(*loaded)
        for index in range(frames):
            writer.write(PIL.Image.new('RGB', (16, 16), (index, 0, 0)))
        written = writer.filenames[0]
        writer.end()

        container = av.open(written)
        try:
            self.assertEqual(len(container.streams.audio), 1)
            self.assertGreater(container.streams.audio[0].duration or 0, 0)
        finally:
            container.close()

    def test_frame_animation_attaches_source_audio(self):
        directory = tempfile.mkdtemp()
        source = os.path.join(directory, 'source.mp4')
        fps = 8
        frames = 4
        sample_rate = 8000
        _write_clip(source, frames, fps, sample_rate, _tone(sample_rate, frames / fps))

        class Writer:
            def __init__(self):
                self.audio = None
                self.rate = None

            def set_source_audio(self, audio, rate):
                self.audio = audio
                self.rate = rate

        class Seed:
            images = [type('Frame', (), {'filename': source})()]
            control_images = None
            fps = 8
            total_frames = 4
            source_frame_start = 0

        class Config:
            animation_format = 'mp4'

        class Loop:
            _c_config = Config()

        writer = Writer()
        _renderloop.RenderLoop._attach_animation_source_audio(Loop(), writer, Seed())
        self.assertIsNotNone(writer.audio)
        self.assertEqual(writer.rate, sample_rate)
        self.assertGreater(writer.audio.shape[1], 0)

    def test_image_process_keeps_mp4_audio(self):
        directory = tempfile.mkdtemp()
        source = os.path.join(directory, 'source.mp4')
        output = os.path.join(directory, 'processed.mp4')
        fps = 8
        frames = 4
        sample_rate = 8000
        _write_clip(source, frames, fps, sample_rate, _tone(sample_rate, frames / fps))

        config = _image_process_config.ImageProcessRenderLoopConfig()
        config.input = [source]
        config.output = [output]
        config.output_overwrite = True
        config.device = 'cpu'
        loop = _image_process.ImageProcessRenderLoop(config)
        loop.run()

        container = av.open(output)
        try:
            self.assertEqual(len(container.streams.video), 1)
            self.assertEqual(len(container.streams.audio), 1)
        finally:
            container.close()


if __name__ == '__main__':
    unittest.main()
