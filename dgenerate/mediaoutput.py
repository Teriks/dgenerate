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
import collections.abc
import io
import os
import shutil
import tempfile
import typing
from fractions import Fraction

import PIL.Image
import av
import numpy
import safetensors.torch
import torch

import dgenerate.filelock as _filelock
import dgenerate.filetypes as _filetypes
import dgenerate.messages as _messages

__doc__ = """
Media output, handles writing videos, animations, and tensor files (latents). 

Provides information about supported output formats including tensor formats for latent data.
"""


class AnimationWriter:
    """
    Interface for animation writers
    """

    def __init__(self):
        pass

    def end(self, new_file: str = None):
        pass

    def write(self, pil_img_rgb: PIL.Image.Image):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def _target_is_path(target) -> bool:
    return isinstance(target, (str, os.PathLike))


def _target_is_seekable(target) -> bool:
    if _target_is_path(target):
        return True
    seekable = getattr(target, 'seekable', None)
    if not callable(seekable):
        return False
    try:
        return bool(seekable())
    except Exception:
        return False


def _rewind(target) -> None:
    """Leave a seekable stream at the start so the caller can read the container."""
    if _target_is_path(target) or not _target_is_seekable(target):
        return
    try:
        target.seek(0)
    except Exception:
        return


_PIL_ANIMATION_FORMATS = {
    'gif': 'GIF',
    'webp': 'WEBP',
    'png': 'PNG',
    'apng': 'PNG',
}


def _animation_format_of(target, animation_format: str | None) -> str:
    if animation_format:
        return animation_format.strip().lower().lstrip('.')
    if _target_is_path(target):
        extension = os.path.splitext(os.fspath(target))[1].lstrip('.').lower()
        if extension:
            return extension
    name = getattr(target, 'name', None)
    if isinstance(name, str):
        extension = os.path.splitext(name)[1].lstrip('.').lower()
        if extension:
            return extension
    return 'mp4'


class VideoWriter(AnimationWriter):
    """
    Animation writer for MP4 h264 format video
    """

    def __init__(self,
                 filename,
                 fps: float,
                 audio: numpy.ndarray | None = None,
                 audio_sample_rate: int | None = None,
                 audio_reader: 'dgenerate.mediainput.SourceAudioReader | None' = None):
        """
        :param filename: Path to write, or a binary file object. A non-seekable
            stream is written as a fragmented mp4 so the mux does not need to
            seek back to the start.
        :param fps: Frame rate, in frames per second.
        :param audio: Optional float32 audio of shape ``(channels, samples)``, 1 or 2 channels.
            Muxed as AAC when the container is mp4. The samples stay in memory
            until the container closes.
        :param audio_sample_rate: Sample rate of ``audio``.
        :param audio_reader: Source soundtrack read from disk as frames are
            written. Only the encoder's remainder is kept. ``audio`` is ignored
            when this is set.
        """

        super().__init__()
        self.filename = os.fspath(filename) if _target_is_path(filename) else None
        self._target = filename
        self.fps = fps
        self._audio = None
        self._audio_reader = audio_reader
        self._audio_sample_rate = None if audio_sample_rate is None else int(audio_sample_rate)
        self._audio_carry = None
        self._audio_pts = 0
        self._audio_layout = 'mono'
        self._video_frames_written = 0
        if audio_reader is not None and audio_reader.sample_rate:
            self._audio_sample_rate = int(audio_reader.sample_rate)
        elif audio_reader is not None:
            audio_reader.close()
            self._audio_reader = None
        if audio is not None and self._audio_reader is None:
            audio = numpy.ascontiguousarray(audio, dtype=numpy.float32)
            if audio.ndim == 1:
                audio = audio.reshape(1, -1)
            if audio.ndim == 2 and audio.shape[0] > 2 and audio.shape[1] <= 2:
                audio = numpy.transpose(audio, (1, 0))
            if audio.ndim == 2 and audio.shape[0] > 2:
                audio = audio[:2]
            self._audio = numpy.ascontiguousarray(audio, dtype=numpy.float32)
        self._container = None
        self._stream = None
        self._audio_stream = None

    def end(self, new_file=None):
        self._cleanup()

        if new_file is not None:
            self.filename = os.fspath(new_file) if _target_is_path(new_file) else None
            self._target = new_file

    def _cleanup(self):
        if self._container is not None:
            for packet in self._stream.encode():
                self._container.mux(packet)
            if self._audio_stream is not None and self._audio is not None:
                self._encode_audio()
            elif self._audio_stream is not None and self._audio_reader is not None:
                self._flush_audio_carry()
            self._container.close()
            self._container = None
            self._stream = None
            self._audio_stream = None
        if self._audio_reader is not None:
            self._audio_reader.close()
            self._audio_reader = None
            self._audio_carry = None

    def _encode_audio(self):
        audio = self._audio
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        channels = 1 if audio.shape[0] == 1 else 2
        if audio.shape[0] > 2:
            audio = audio[:2]
            channels = 2
        layout = 'mono' if channels == 1 else 'stereo'
        sample_rate = self._audio_sample_rate
        frame_size = self._audio_stream.codec_context.frame_size or 1024
        remainder = audio.shape[1] % frame_size
        if remainder:
            audio = numpy.pad(audio, ((0, 0), (0, frame_size - remainder)))

        for start in range(0, audio.shape[1], frame_size):
            chunk = numpy.ascontiguousarray(audio[:, start:start + frame_size])
            frame = av.AudioFrame.from_ndarray(chunk, format='fltp', layout=layout)
            frame.sample_rate = sample_rate
            frame.time_base = Fraction(1, sample_rate)
            frame.pts = start
            for packet in self._audio_stream.encode(frame):
                self._container.mux(packet)

        for packet in self._audio_stream.encode(None):
            self._container.mux(packet)

    def _audio_samples_through(self, frames: int) -> int:
        return int(round(frames / float(self.fps) * self._audio_sample_rate))

    def _open_audio_stream(self, channels: int):
        if self._audio_stream is not None or not self._audio_sample_rate:
            return
        self._audio_layout = 'mono' if int(channels) == 1 else 'stereo'
        self._audio_stream = self._container.add_stream('aac', rate=self._audio_sample_rate)
        self._audio_stream.layout = self._audio_layout

    def _push_audio(self, samples: numpy.ndarray):
        if self._audio_carry is not None and self._audio_carry.shape[1]:
            samples = numpy.concatenate((self._audio_carry, samples), axis=1)
        frame_size = self._audio_stream.codec_context.frame_size or 1024
        usable = samples.shape[1] - (samples.shape[1] % frame_size)
        if usable:
            self._encode_audio_blocks(samples[:, :usable], frame_size)
        self._audio_carry = samples[:, usable:] if usable < samples.shape[1] else None

    def _flush_audio_carry(self):
        frame_size = self._audio_stream.codec_context.frame_size or 1024
        if self._audio_carry is not None and self._audio_carry.shape[1]:
            pad = frame_size - self._audio_carry.shape[1]
            padded = numpy.pad(self._audio_carry, ((0, 0), (0, pad)))
            self._encode_audio_blocks(padded, frame_size)
            self._audio_carry = None
        for packet in self._audio_stream.encode(None):
            self._container.mux(packet)

    def _encode_audio_blocks(self, samples: numpy.ndarray, frame_size: int):
        rate = self._audio_sample_rate
        for start in range(0, samples.shape[1], frame_size):
            chunk = numpy.ascontiguousarray(samples[:, start:start + frame_size])
            frame = av.AudioFrame.from_ndarray(chunk, format='fltp', layout=self._audio_layout)
            frame.sample_rate = rate
            frame.time_base = Fraction(1, rate)
            frame.pts = self._audio_pts
            self._audio_pts += frame_size
            for packet in self._audio_stream.encode(frame):
                self._container.mux(packet)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()

    def write(self, img: PIL.Image.Image):
        if self._container is None:
            if self.filename is not None:
                self._container = av.open(self.filename, 'w')
            else:
                options = None
                if not _target_is_seekable(self._target):
                    options = {'movflags': 'frag_keyframe+empty_moov+default_base_moof'}
                self._container = av.open(self._target, 'w', format='mp4', options=options)
            self._stream = self._container.add_stream(
                "h264", rate=Fraction(self.fps).limit_denominator(1001))
            self._stream.codec_context.bit_rate = 8000000
            self._stream.width = img.width
            self._stream.height = img.height
            self._stream.pix_fmt = "yuv420p"
            if self._audio is not None and self._audio_sample_rate:
                self._open_audio_stream(1 if self._audio.shape[0] == 1 else 2)

        for packet in self._stream.encode(av.VideoFrame.from_image(img)):
            self._container.mux(packet)

        if self._audio_reader is not None and self._audio_sample_rate:
            need = self._audio_samples_through(self._video_frames_written + 1)
            need -= self._audio_reader.consumed
            if need > 0:
                samples = self._audio_reader.read(need)
                if samples.shape[1]:
                    self._open_audio_stream(samples.shape[0])
                    self._push_audio(samples)
            self._video_frames_written += 1


class AnimatedImageWriter(AnimationWriter):
    """
    Animation writer for animated images such as GIFs and webp
    """

    def __init__(self, filename, duration: float, pil_format: str | None = None):
        """
        :param filename: Filename to write to, or a binary file object.
        :param duration: Frame duration, (duration of a single frame) in milliseconds.
        :param pil_format: PIL format name used when ``filename`` is a stream.
        """
        super().__init__()
        self.collected_frames = []
        self.filename = filename
        self.duration = duration
        self._pil_format = pil_format

    def end(self, new_file: str = None):

        if self.collected_frames:
            save_kwargs = dict(
                save_all=True,
                append_images=self.collected_frames[1:],
                optimize=False,
                duration=self.duration,
                loop=0)
            target = self.filename
            if not _target_is_path(target):
                save_kwargs['format'] = self._pil_format or 'GIF'
            if not _target_is_path(target) and not _target_is_seekable(target):
                # GIF and WebP saves seek back into the file. Encode into a
                # buffer, then write the finished bytes to the stream.
                buffer = io.BytesIO()
                self.collected_frames[0].save(buffer, **save_kwargs)
                target.write(buffer.getvalue())
            else:
                self.collected_frames[0].save(target, **save_kwargs)
            for i in self.collected_frames:
                i.close()
            self.collected_frames.clear()

        if new_file:
            self.filename = new_file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()

    def write(self, img: PIL.Image.Image):
        self.collected_frames.append(img.copy())


def get_supported_animation_writer_formats():
    """
    Supported animation writer formats, file extensions with no period.

    :return: list of file extensions.
    """
    return _filetypes.animation_write_extensions()


def get_supported_static_image_formats():
    """
    What file extensions does PIL/Pillow support for output of at least one frame?

    File extensions are returned without a period.

    :return: list of file extensions
    """
    return _filetypes.image_write_extensions()


class UnknownAnimationFormatError(Exception):
    """
    Raised by :py:func:`.create_animation_writer` when an unknown animation format is provided.
    """


def source_has_audio(path: str) -> bool:
    """
    Does this local media file contain an audio stream?

    :param path: filesystem path
    :return: ``True`` when an audio stream can be opened
    """
    if not path or not os.path.isfile(path):
        return False
    try:
        with av.open(path) as container:
            return any(stream.type == 'audio' for stream in container.streams)
    except av.error.FFmpegError:
        return False


def read_source_audio(path: str,
                      fps: float,
                      frame_start: int = 0,
                      frame_count: int | None = None):
    """
    Decode the first audio stream and return the samples for the frames
    being written.

    This gathers the slice into one array. :py:class:`dgenerate.mediainput.SourceAudioReader`
    reads the same slice without keeping it. ``frame_start`` and ``frame_count`` are
    in video frames at ``fps``.

    :param path: filesystem path of the source media
    :param fps: video frame rate used to place the slice
    :param frame_start: first video frame, inclusive
    :param frame_count: how many video frames are written. ``None`` reads through
        the end of the audio.
    :return: ``(samples, sample_rate)`` with ``samples`` shaped ``(channels, samples)``,
        or ``None`` when the file has no usable audio
    """
    import dgenerate.mediainput as _mediainput
    reader = _mediainput.SourceAudioReader(path, fps, frame_start, frame_count)
    try:
        if reader.sample_rate is None:
            return None
        pieces = []
        while True:
            chunk = reader.read(reader.sample_rate)
            if chunk.shape[1] == 0:
                break
            pieces.append(chunk)
        if not pieces:
            return None
        audio = pieces[0] if len(pieces) == 1 else numpy.concatenate(pieces, axis=1)
        if audio.shape[1] == 0:
            return None
        return audio, reader.sample_rate
    finally:
        reader.close()


def create_animation_writer(animation_format: str,
                            out_filename: str,
                            fps: float,
                            audio: numpy.ndarray | None = None,
                            audio_sample_rate: int | None = None,
                            audio_reader: 'dgenerate.mediainput.SourceAudioReader | None' = None):
    """
    Create an animation writer of a given format.

    :raise UnknownAnimationFormatError: if the provided ``animation_format`` is unknown.

    :param animation_format: The animation format, see :py:func:`.supported_animation_writer_formats`
    :param out_filename: the output file name
    :param fps: FPS
    :param audio: Optional soundtrack for an mp4. Other formats cannot carry one.
    :param audio_sample_rate: Sample rate of ``audio``
    :param audio_reader: Source soundtrack read from disk as frames are written.
        Used instead of ``audio`` when set. Closed when the format cannot carry it.
    :return: :py:class:`.AnimationWriter`
    """
    animation_format = animation_format.strip().lower()

    if animation_format not in get_supported_animation_writer_formats():
        if audio_reader is not None:
            audio_reader.close()
        raise UnknownAnimationFormatError(f'Animation format "{animation_format}" is not a known format.')

    if animation_format == 'mp4':
        return VideoWriter(
            out_filename, fps, audio=audio, audio_sample_rate=audio_sample_rate,
            audio_reader=audio_reader)
    if audio_reader is not None:
        audio_reader.close()
    return AnimatedImageWriter(
        out_filename, 1000 / fps, pil_format=_PIL_ANIMATION_FORMATS.get(animation_format))


def _write_frame_images(file, frames) -> None:
    """Write ``frames`` as numbered PNG files for a ``.frames`` target."""
    if not _target_is_path(file):
        raise UnknownAnimationFormatError(
            'frames writes one image per frame and needs a path, not a stream.')
    path = os.fspath(file)
    root, extension = os.path.splitext(path)
    prefix = root if extension.lower() == '.frames' else path
    directory = os.path.dirname(prefix)
    if directory:
        os.makedirs(directory, exist_ok=True)
    for index, frame in enumerate(frames, start=1):
        image = frame if frame.mode == 'RGB' else frame.convert('RGB')
        image.save(f'{prefix}_frame_{index}.png')


def write_animation(file,
                    frames,
                    fps: float,
                    animation_format: str | None = None,
                    audio: numpy.ndarray | None = None,
                    audio_sample_rate: int | None = None) -> None:
    """Mux frames, and an optional soundtrack, to a path or a binary stream.

    A path's extension selects the container. A stream is mp4 unless
    ``animation_format`` is given. A seekable stream is rewound to the start
    when the container is finished. A non-seekable stream receives a
    fragmented mp4, which can be written without seeking back to the header.
    ``frames`` writes one PNG per frame next to a path and does not accept a
    stream. GIF and WebP bound for a non-seekable stream are encoded in
    memory first.

    :param file: Filesystem path or binary file object.
    :param frames: Frames as PIL images.
    :param fps: Frames per second.
    :param animation_format: Container extension without a period, such as
        ``mp4`` or ``gif``. ``None`` uses the path extension, or mp4 for a stream.
    :param audio: Optional float32 soundtrack of shape ``(channels, samples)``.
        Muxed into mp4. Other containers cannot carry it.
    :param audio_sample_rate: Sample rate of ``audio``.
    :raise UnknownAnimationFormatError: the container format is not writable.
    :raise ValueError: ``frames`` is empty.
    """
    if not frames:
        raise ValueError('write_animation needs at least one frame.')
    container_format = _animation_format_of(file, animation_format)
    if container_format == 'frames':
        _write_frame_images(file, frames)
        return
    if container_format not in get_supported_animation_writer_formats():
        raise UnknownAnimationFormatError(
            f'Animation format "{container_format}" is not a known format.')
    soundtrack = audio
    sample_rate = audio_sample_rate
    if container_format != 'mp4' and soundtrack is not None:
        _messages.warning(
            f'Audio is dropped because the container is {container_format}. '
            'Use mp4 to keep the soundtrack.')
        soundtrack = None
        sample_rate = None
    if container_format == 'mp4':
        writer = VideoWriter(
            file, fps, audio=soundtrack, audio_sample_rate=sample_rate)
    else:
        writer = AnimatedImageWriter(
            file,
            1000 / fps,
            pil_format=_PIL_ANIMATION_FORMATS.get(container_format, 'GIF'))
    try:
        for frame in frames:
            writer.write(frame if frame.mode == 'RGB' else frame.convert('RGB'))
    finally:
        writer.end()
    _rewind(file)


class MultiAnimationWriter(AnimationWriter):
    """
    Splits writes between N files with generated filename suffixes if necessary
    depending on how many images were written on the first write.
    """

    def __init__(self,
                 animation_format: str,
                 filename: str,
                 fps: float,
                 allow_overwrites=False,
                 files=None):
        """
        :param animation_format: One of :py:func:`.supported_animation_writer_formats`
        :param filename: The desired filename, if multiple images are written a 
            suffix ``_animation_N`` will be appended for each file
        :param fps: Frames per second
        :param allow_overwrites: Allow overwrites of existing files? or append ``_duplicate_N``,
            The overwrite dis-allowance is multiprocess safe between instances of this library.
        :param files: Binary streams to write instead of ``filename``. One stream
            receives one animation. A seekable stream is rewound when :py:meth:`end`
            closes it.
        """

        super().__init__()
        self.filename = filename
        self.writers = []
        self.filenames = []
        self.animation_format = animation_format
        self.fps = fps
        self.allow_overwrites = allow_overwrites
        self._files = None if files is None else list(files)
        self._audio = None
        self._audio_sample_rate = None
        self._audio_file = None

    def set_source_audio(self, audio: numpy.ndarray | None, sample_rate: int | None):
        """
        Soundtrack copied onto every mp4 this writer opens.

        Must be set before the first :py:meth:`write` of a file. The samples
        stay in memory until each file closes. :py:meth:`set_source_audio_file`
        reads the same kind of track from disk while frames are written.
        """
        self._audio = audio
        self._audio_sample_rate = sample_rate
        self._audio_file = None

    def set_source_audio_file(self, path: str, fps: float,
                              frame_start: int = 0,
                              frame_count: int | None = None):
        """
        Soundtrack read from ``path`` onto every mp4 this writer opens.

        Must be set before the first :py:meth:`write` of a file. Each output
        file opens its own reader and decodes samples as frames are written.
        """
        self._audio = None
        self._audio_sample_rate = None
        self._audio_file = (path, float(fps), int(frame_start or 0), frame_count)

    def _gen_filename(self, num_images, image_idx):
        base, ext = os.path.splitext(self.filename)
        if num_images > 1:
            return f'{base}_animation_{image_idx + 1}{ext}'
        else:
            return f'{base}{ext}'

    def write(self, img: PIL.Image.Image | collections.abc.Iterable[PIL.Image.Image]):
        if not isinstance(img, collections.abc.Iterable):
            img = [img]

        if not self.writers:
            # Lazy initialize all the writers we need
            num_images = len(img)

            if self._files is not None:
                if len(self._files) != num_images:
                    raise ValueError(
                        f'{len(self._files)} animation streams were provided for '
                        f'{num_images} images. Pass one stream per image.')
                targets = self._files
                self.filenames = []
            else:
                requested_filenames = [self._gen_filename(num_images, idx) for idx in range(0, num_images)]

                if not self.allow_overwrites:
                    # Touch all the files we will be writing to
                    # Avoid duplication
                    self.filenames = _filelock.touch_avoid_duplicate(
                        os.path.dirname(self.filename),
                        return_list=True,
                        path_maker=_filelock.suffix_path_maker(
                            requested_filenames,
                            suffix='_duplicate_'
                        ))
                else:
                    # Overwrite anything
                    self.filenames = requested_filenames
                targets = self.filenames

            for filename in targets:
                reader = None
                audio = self._audio
                rate = self._audio_sample_rate
                if self.animation_format == 'mp4' and self._audio_file is not None:
                    import dgenerate.mediainput as _mediainput
                    reader = _mediainput.SourceAudioReader(*self._audio_file)
                    if reader.sample_rate is None:
                        reader.close()
                        reader = None
                    else:
                        audio = None
                        rate = reader.sample_rate
                self.writers.append(
                    create_animation_writer(
                        self.animation_format,
                        filename,
                        self.fps,
                        audio=audio,
                        audio_sample_rate=rate,
                        audio_reader=reader))

        elif len(self.writers) != len(img):
            # Sanity check
            raise RuntimeError('To many images written, subsequent writes must '
                               'use the amount of images from the first write.')

        for writer, image in zip(self.writers, img):
            writer.write(image)

    def end(self, new_file=None):
        self.filename = new_file

        for writer in self.writers:
            writer.end(new_file=new_file)

        self.writers.clear()
        self.filenames.clear()
        if self._files:
            for stream in self._files:
                _rewind(stream)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for writer in self.writers:
            writer.__exit__(exc_type, exc_val, exc_tb)


def get_supported_tensor_formats() -> list[str]:
    """
    Get supported tensor file formats for latents output.
    
    :return: List of supported tensor formats
    """
    return _filetypes.tensor_extensions()


def save_tensor_file(tensor: torch.Tensor | numpy.ndarray,
                     path_or_file: typing.BinaryIO | str,
                     file_format: str = "pt"
                     ) -> None:
    """
    Save a tensor to disk in the specified format.
    
    :param tensor: The tensor to save (torch.Tensor or numpy.ndarray)
    :param path_or_file: Path to save to or file-like object
    :param file_format: Format to save in ("pt", "pth", or "safetensors")
    :raises ValueError: If format is not supported
    """
    file_format = file_format.lower()

    # Convert numpy array to torch tensor if needed
    if isinstance(tensor, numpy.ndarray):
        tensor = torch.from_numpy(tensor)

    if file_format in ("pt", "pth"):
        if isinstance(path_or_file, str):
            torch.save(tensor, path_or_file)
        else:
            torch.save(tensor, path_or_file)
    elif file_format == "safetensors":
        # Ensure tensor is contiguous for safetensors
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        if isinstance(path_or_file, str):
            safetensors.torch.save_file({"latents": tensor}, path_or_file)
        else:
            # safetensors doesn't support file-like objects directly
            tmp_name = None
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp_name = tmp.name

                safetensors.torch.save_file({"latents": tensor}, tmp_name)

                with open(tmp_name, 'rb') as tmp_file:
                    shutil.copyfileobj(tmp_file, path_or_file)
            finally:
                if tmp_name is not None:
                    try:
                        os.unlink(tmp_name)
                    except (OSError, PermissionError):
                        # If cleanup fails, just leave it for the OS
                        pass
    else:
        raise ValueError(f"Unsupported tensor format: {file_format}. Supported formats: pt, pth, safetensors")
