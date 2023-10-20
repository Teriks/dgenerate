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
import atexit
import contextlib
import mimetypes
import os
import pathlib
import sqlite3
import typing

import PIL.Image
import PIL.ImageOps
import PIL.ImageSequence
import av
import fake_useragent
import requests

import dgenerate.image as _image
import dgenerate.messages as _messages
import dgenerate.preprocessors as _preprocessors
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
import dgenerate.util as _util


class AnimationFrame:
    """
    A realized animation frame with attached image data.
    """
    frame_index: int
    total_frames: int
    anim_fps: typing.Union[float, int]
    anim_frame_duration: float
    image: PIL.Image.Image
    mask_image: PIL.Image.Image = None
    control_image: PIL.Image.Image = None

    def __init__(self,
                 frame_index: int,
                 total_frames: int,
                 anim_fps: typing.Union[float, int],
                 anim_frame_duration: float,
                 image: PIL.Image.Image,
                 mask_image: PIL.Image.Image = None,
                 control_image: PIL.Image.Image = None):

        self.frame_index = frame_index
        self.total_frames = total_frames
        self.fps = anim_fps
        self.duration = anim_frame_duration
        self.image = image
        self.mask_image = mask_image
        self.control_image = control_image

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    def close(self):
        self.image.close()
        if self.mask_image is not None:
            self.mask_image.close()
        if self.control_image is not None:
            self.control_image.close()


def frame_slice_count(total_frames: int, frame_start: int, frame_end: typing.Optional[int] = None) -> int:
    """
    Calculate the number of frames resulting from frame slicing.

    :param total_frames: Total frames being sliced from
    :param frame_start: The start frame
    :param frame_end: The end frame
    :return: int
    """

    return min(total_frames, (frame_end + 1 if frame_end is not None else total_frames)) - frame_start


class ImageSeedError(Exception):
    """
    Raised on image seed parsing and loading errors.
    """
    pass


class ImageSeedSizeMismatchError(ImageSeedError):
    """
    Raised when the constituents of an image seed are mismatched in dimension.
    """
    pass


class AnimationReader:
    """
    Abstract base class for animation readers.
    """

    def __init__(self,
                 width: int,
                 height: int,
                 anim_fps: typing.Union[float, int],
                 anim_frame_duration: float,
                 total_frames: int, **kwargs):
        self._width = width
        self._height = height
        self._anim_fps = anim_fps
        self._anim_frame_duration = anim_frame_duration
        self._total_frames = total_frames

    @property
    def width(self) -> int:
        return self._width

    @property
    def size(self) -> _types.Size:
        return self._width, self._height

    @property
    def height(self) -> int:
        return self._height

    @property
    def anim_fps(self) -> typing.Union[float, int]:
        return self._anim_fps

    @property
    def anim_frame_duration(self) -> float:
        return self._anim_frame_duration

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self):
        return self

    def __next__(self) -> PIL.Image.Image:
        raise StopIteration

    def frame_slice_count(self, frame_start: int = 0, frame_end: _types.OptionalInteger = None) -> int:
        return frame_slice_count(self.total_frames, frame_start, frame_end)


class VideoReader(_preprocessors.ImagePreprocessorMixin, AnimationReader):
    """
    Implementation :py:class:`.AnimationReader` that reads Video files with PyAV
    """

    def __init__(self,
                 file: typing.Union[str, typing.BinaryIO],
                 file_source: str,
                 resize_resolution: _types.OptionalSize = None,
                 preprocessor: _preprocessors.ImagePreprocessor = None):
        self._filename = file
        self._file_source = file_source
        if isinstance(file, str):
            self._container = av.open(file, 'r')
        else:
            _, ext = os.path.splitext(file_source)
            if not ext:
                raise NotImplementedError(
                    'Cannot determine the format of a video file lacking a file extension.')
            self._container = av.open(file, format=ext.lstrip('.').lower())
        self.resize_resolution = resize_resolution

        if self.resize_resolution is None:
            width = int(self._container.streams.video[0].width)
            height = int(self._container.streams.video[0].height)
            if not _image.is_aligned_by_8(width, height):
                width, height = _image.resize_image_calc(old_size=(width, height),
                                                         new_size=_image.align_by_8(width, height))
                self.resize_resolution = (width, height)
        else:
            width, height = _image.resize_image_calc(old_size=(int(self._container.streams.video[0].width),
                                                               int(self._container.streams.video[0].height)),
                                                     new_size=self.resize_resolution)

        anim_fps = int(self._container.streams.video[0].average_rate)
        anim_frame_duration = 1000 / anim_fps
        total_frames = self._container.streams.video[0].frames

        if total_frames <= 0:
            # webm decode bug?
            total_frames = sum(1 for _ in self._container.decode(video=0))
            self._container.seek(0, whence='time')
        self._iter = self._container.decode(video=0)

        super().__init__(width=width,
                         height=height,
                         anim_fps=anim_fps,
                         anim_frame_duration=anim_frame_duration,
                         total_frames=total_frames,
                         preprocessor=preprocessor)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._container.close()

    def frame_slice_count(self, frame_start=0, frame_end=None):
        return frame_slice_count(self.total_frames, frame_start, frame_end)

    def __next__(self):
        rgb_image = next(self._iter).to_image()
        rgb_image.filename = self._file_source
        return self.preprocess_image(rgb_image, self.resize_resolution)


class AnimatedImageReader(_preprocessors.ImagePreprocessorMixin, AnimationReader):
    """
    Implementation of :py:class:`.AnimationReader` that reads animated image formats using Pillow
    """

    def __init__(self,
                 file: typing.Union[str, typing.BinaryIO],
                 file_source: str,
                 resize_resolution: _types.OptionalSize = None,
                 preprocessor: _preprocessors.ImagePreprocessor = None):
        self._img = PIL.Image.open(file)
        self._file_source = file_source

        self._iter = PIL.ImageSequence.Iterator(self._img)
        self.resize_resolution = resize_resolution

        total_frames = self._img.n_frames

        anim_frame_duration = self._img.info.get('duration', 0)

        if anim_frame_duration == 0:
            # 10 frames per second for bugged gifs / webp
            anim_frame_duration = 100

        anim_fps = 1000 / anim_frame_duration

        if self.resize_resolution is None:
            width = self._img.size[0]
            height = self._img.size[1]
            if not _image.is_aligned_by_8(width, height):
                width, height = _image.resize_image_calc(old_size=(width, height),
                                                         new_size=_image.align_by_8(width, height))
                self.resize_resolution = (width, height)
        else:
            width, height = _image.resize_image_calc(old_size=self._img.size,
                                                     new_size=self.resize_resolution)

        super().__init__(width=width,
                         height=height,
                         anim_fps=anim_fps,
                         anim_frame_duration=anim_frame_duration,
                         total_frames=total_frames,
                         preprocessor=preprocessor)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._img.close()

    def frame_slice_count(self, frame_start: int = 0, frame_end: _types.OptionalInteger = None):
        return frame_slice_count(self.total_frames, frame_start, frame_end)

    def __next__(self) -> PIL.Image.Image:
        with next(self._iter) as img:
            rgb_image = _image.to_rgb(img)
            rgb_image.filename = self._file_source
            return self.preprocess_image(rgb_image, self.resize_resolution)


class MockImageAnimationReader(_preprocessors.ImagePreprocessorMixin, AnimationReader):
    """
    Implementation of :py:class:`.AnimationReader` that repeats a single PIL image
    as many times as desired in order to mock/emulate an animation.
    """

    def __init__(self,
                 img: PIL.Image.Image,
                 resize_resolution: _types.OptionalSize = None,
                 image_repetitions: int = 1,
                 preprocessor: _preprocessors.ImagePreprocessor = None):
        self._img = img
        self._idx = 0
        self.resize_resolution = resize_resolution

        total_frames = image_repetitions
        anim_fps = 30
        anim_frame_duration = 1000 / anim_fps

        if self.resize_resolution is None:
            width = self._img.size[0]
            height = self._img.size[1]
            if not _image.is_aligned_by_8(width, height):
                width, height = _image.resize_image_calc(old_size=(width, height),
                                                         new_size=_image.align_by_8(width, height))
                self.resize_resolution = (width, height)
        else:
            width, height = _image.resize_image_calc(old_size=self._img.size,
                                                     new_size=self.resize_resolution)

        super().__init__(width=width,
                         height=height,
                         anim_fps=anim_fps,
                         anim_frame_duration=anim_frame_duration,
                         total_frames=total_frames,
                         preprocessor=preprocessor)

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @total_frames.setter
    def total_frames(self, cnt):
        self._total_frames = cnt

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._img.close()

    def frame_slice_count(self, frame_start: int = 0, frame_end: _types.OptionalInteger = None) -> int:
        return frame_slice_count(self.total_frames, frame_start, frame_end)

    def __next__(self) -> PIL.Image.Image:
        if self._idx < self.total_frames:
            self._idx += 1
            return self.preprocess_image(_image.copy_img(self._img), self.resize_resolution)
        else:
            raise StopIteration


def _iterate_animation_frames_x2(seed_reader: AnimationReader,
                                 right_reader: AnimationReader,
                                 right_animation_frame_param_name: str,
                                 frame_start: int = 0,
                                 frame_end: _types.OptionalInteger = None):
    total_frames = seed_reader.frame_slice_count(frame_start, frame_end)
    right_total_frames = right_reader.frame_slice_count(frame_start, frame_end)
    out_frame_idx = 0

    # Account for videos possibly having a differing number of frames
    total_frames = min(total_frames, right_total_frames)

    have_preprocess_seed = isinstance(seed_reader, _preprocessors.ImagePreprocessorMixin)
    have_preprocess_right = isinstance(right_reader, _preprocessors.ImagePreprocessorMixin)

    preprocess_seed_old = True
    preprocess_right_old = True

    if have_preprocess_seed:
        preprocess_seed_old = seed_reader.preprocess_enabled
        seed_reader.preprocess_enabled = frame_start == 0

    if have_preprocess_right:
        preprocess_right_old = right_reader.preprocess_enabled
        right_reader.preprocess_enabled = frame_start == 0

    for in_frame_idx, frame in enumerate(zip(seed_reader, right_reader)):
        if in_frame_idx == frame_start - 1:
            # The next frame is preprocessed
            if have_preprocess_seed:
                seed_reader.preprocess_enabled = True
            if have_preprocess_right:
                right_reader.preprocess_enabled = True

        seed_image = frame[0]
        right_image = frame[1]

        if in_frame_idx >= frame_start:
            yield AnimationFrame(frame_index=out_frame_idx,
                                 total_frames=total_frames,
                                 anim_fps=seed_reader.anim_fps,
                                 anim_frame_duration=seed_reader.anim_frame_duration,
                                 image=seed_image,
                                 **{right_animation_frame_param_name: right_image})
            out_frame_idx += 1

        if frame_end is not None and in_frame_idx == frame_end:
            break

    if have_preprocess_seed:
        seed_reader.preprocess_enabled = preprocess_seed_old

    if have_preprocess_right:
        right_reader.preprocess_enabled = preprocess_right_old


def _iterate_animation_frames_x3(seed_reader: AnimationReader,
                                 mask_reader: typing.Optional[AnimationReader] = None,
                                 control_reader: typing.Optional[AnimationReader] = None,
                                 frame_start: int = 0,
                                 frame_end: _types.OptionalInteger = None):
    total_frames = seed_reader.frame_slice_count(frame_start, frame_end)
    mask_total_frames = mask_reader.frame_slice_count(frame_start, frame_end)
    control_total_frames = control_reader.frame_slice_count(frame_start, frame_end)
    out_frame_idx = 0

    # Account for videos possibly having a differing number of frames
    total_frames = min(total_frames, mask_total_frames, control_total_frames)

    have_preprocess_seed = isinstance(seed_reader, _preprocessors.ImagePreprocessorMixin)
    have_preprocess_mask = isinstance(mask_reader, _preprocessors.ImagePreprocessorMixin)
    have_preprocess_control = isinstance(control_reader, _preprocessors.ImagePreprocessorMixin)

    preprocess_seed_old = True
    preprocess_mask_old = True
    preprocess_control_old = True

    if have_preprocess_seed:
        preprocess_seed_old = seed_reader.preprocess_enabled
        seed_reader.preprocess_enabled = frame_start == 0

    if have_preprocess_mask:
        preprocess_mask_old = mask_reader.preprocess_enabled
        mask_reader.preprocess_enabled = frame_start == 0

    if have_preprocess_control:
        preprocess_control_old = control_reader.preprocess_enabled
        control_reader.preprocess_enabled = frame_start == 0

    for in_frame_idx, frame in enumerate(zip(seed_reader, mask_reader, control_reader)):

        if in_frame_idx == frame_start - 1:
            # The next frame is preprocessed
            if have_preprocess_seed:
                seed_reader.preprocess_enabled = True
            if have_preprocess_mask:
                mask_reader.preprocess_enabled = True
            if have_preprocess_control:
                control_reader.preprocess_enabled = True

        image = frame[0]
        mask = frame[1]
        control = frame[2]

        if in_frame_idx >= frame_start:
            yield AnimationFrame(frame_index=out_frame_idx,
                                 total_frames=total_frames,
                                 anim_fps=seed_reader.anim_fps,
                                 anim_frame_duration=seed_reader.anim_frame_duration,
                                 image=image,
                                 mask_image=mask,
                                 control_image=control)
            out_frame_idx += 1

        if frame_end is not None and in_frame_idx == frame_end:
            break

    if have_preprocess_seed:
        seed_reader.preprocess_enabled = preprocess_seed_old

    if have_preprocess_mask:
        mask_reader.preprocess_enabled = preprocess_mask_old

    if have_preprocess_control:
        control_reader.preprocess_enabled = preprocess_control_old


def iterate_animation_frames(seed_reader: AnimationReader,
                             mask_reader: typing.Optional[AnimationReader] = None,
                             control_reader: typing.Optional[AnimationReader] = None,
                             frame_start: int = 0,
                             frame_end: _types.OptionalInteger = None) -> typing.Generator[AnimationFrame, None, None]:
    """
    Read :py:class:`.AnimationFrame` objects from up to three :py:class:`.AnimationReader` objects simultaneously
    with an optional inclusive frame slice.

    :param seed_reader: Reads into :py:attr:`.ImageSeed.image`
    :param mask_reader: Reads into :py:attr:`.ImageSeed.mask`
    :param control_reader: Reads into :py:attr:`.ImageSeed.control`
    :param frame_start: Frame slice start, inclusive value
    :param frame_end: Frame slice end, inclusive value
    :return: Generator object yielding :py:class:`.AnimationFrame`
    """

    if mask_reader is not None and control_reader is not None:
        yield from _iterate_animation_frames_x3(seed_reader=seed_reader,
                                                mask_reader=mask_reader,
                                                control_reader=control_reader,
                                                frame_start=frame_start,
                                                frame_end=frame_end)
    elif mask_reader is not None:
        yield from _iterate_animation_frames_x2(seed_reader=seed_reader,
                                                right_reader=mask_reader,
                                                right_animation_frame_param_name='mask_image',
                                                frame_start=frame_start,
                                                frame_end=frame_end)
    elif control_reader is not None:
        yield from _iterate_animation_frames_x2(seed_reader=seed_reader,
                                                right_reader=control_reader,
                                                right_animation_frame_param_name='control_image',
                                                frame_start=frame_start,
                                                frame_end=frame_end)
    else:
        total_frames = seed_reader.frame_slice_count(frame_start, frame_end)
        out_frame_idx = 0

        have_preprocess = isinstance(seed_reader, _preprocessors.ImagePreprocessorMixin)

        preprocess_old = True
        if have_preprocess:
            preprocess_old = seed_reader.preprocess_enabled
            seed_reader.preprocess_enabled = frame_start == 0

        for in_frame_idx, frame in enumerate(seed_reader):
            if have_preprocess and in_frame_idx == frame_start - 1:
                # The next frame is preprocessed
                seed_reader.preprocess_enabled = True

            if in_frame_idx >= frame_start:
                yield AnimationFrame(frame_index=out_frame_idx,
                                     total_frames=total_frames,
                                     anim_fps=seed_reader.anim_fps,
                                     anim_frame_duration=seed_reader.anim_frame_duration,
                                     image=frame)
                out_frame_idx += 1

            if frame_end is not None and in_frame_idx == frame_end:
                break

        if have_preprocess:
            seed_reader.preprocess_enabled = preprocess_old


class ImageSeed:
    """
    An ImageSeed with attached image data
    """

    frame_index: _types.OptionalInteger = None
    total_frames: _types.OptionalInteger = None
    fps: typing.Union[int, float, None] = None
    duration: _types.OptionalFloat = None
    image: PIL.Image.Image
    mask_image: PIL.Image.Image
    control_image: PIL.Image.Image
    is_animation_frame: bool

    def __init__(self,
                 image: typing.Union[PIL.Image.Image, AnimationFrame],
                 mask_image: typing.Optional[PIL.Image.Image] = None,
                 control_image: typing.Optional[PIL.Image.Image] = None):

        self.is_animation_frame: bool = isinstance(image, AnimationFrame)

        if self.is_animation_frame:
            self.image = image.image
            self.mask_image = image.mask_image
            self.control_image = image.control_image
            if image.total_frames > 1:
                self.frame_index = image.frame_index
                self.total_frames = image.total_frames
                self.fps = image.fps
                self.duration = image.duration
            else:
                self.is_animation_frame = False
        else:
            self.image = image
            self.mask_image = mask_image
            self.control_image = control_image

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    def close(self):
        self.image.close()
        if self.mask_image is not None:
            self.mask_image.close()
        if self.control_image is not None:
            self.control_image.close()


def _exif_orient(image):
    exif = image.getexif()
    for k in exif.keys():
        if k != 0x0112:
            exif[k] = None
            del exif[k]
    new_exif = exif.tobytes()
    image.info["exif"] = new_exif
    return PIL.ImageOps.exif_transpose(image)


class ImageSeedParseResult:
    """
    The result of parsing an --image-seed path
    """
    seed_path: _types.OptionalPath = None
    seed_path_is_local: bool = False
    mask_path: _types.OptionalPath = None
    mask_path_is_local: bool = False
    control_path: _types.OptionalPath = None
    control_path_is_local: bool = False
    resize_resolution: _types.OptionalSize = None

    def is_single_image(self) -> bool:
        """
        Did this image seed path only specify a singular image/video?

        :return: bool
        """
        return self.seed_path is not None and self.mask_path is None and self.control_path is None


def _parse_image_seed_uri_legacy(uri: str) -> ImageSeedParseResult:
    parts = (x.strip() for x in uri.split(';'))
    result = ImageSeedParseResult()

    first = next(parts)
    result.seed_path = first
    if first.startswith('http://') or first.startswith('https://'):
        result.seed_path_is_local = False
    elif os.path.exists(first):
        result.seed_path_is_local = True
    else:
        raise ImageSeedError(f'Image seed file "{first}" does not exist.')

    for part in parts:
        if part == '':
            raise ImageSeedError(
                'Missing inpaint mask image or output size specification, '
                'check image seed syntax, stray semicolon?')

        if part.startswith('http://') or part.startswith('https://'):
            result.mask_path = part
            result.mask_path_is_local = False
        elif os.path.exists(part):
            result.mask_path = part
            result.mask_path_is_local = True
        else:
            try:
                dimensions = tuple(int(s.strip()) for s in part.split('x'))
                for idx, d in enumerate(dimensions):
                    if d % 8 != 0:
                        raise ImageSeedError(
                            f'Image seed resize {["width", "height"][idx]} dimension {d} is not divisible by 8.')

                result.resize_resolution = dimensions
            except ValueError:
                raise ImageSeedError(f'Inpaint mask file "{part}" does not exist.')

            if len(result.resize_resolution) == 1:
                result.resize_resolution = (result.resize_resolution[0], result.resize_resolution[0])
    return result


def parse_image_seed_uri(uri: str) -> ImageSeedParseResult:
    """
    Parse an `--image-seeds` path into its constituents

    :param uri: `--image-seeds` path
    :return: :py:class:`.ImageSeedParseResult`
    """

    parts = uri.split(';')

    non_legacy: bool = len(parts) > 3

    if not non_legacy:
        for i in parts:
            i = i.strip()
            if i.startswith('mask='):
                non_legacy = True
                break
            if i.startswith('control='):
                non_legacy = True
                break
            if i.startswith('resize='):
                non_legacy = True
                break

    if not non_legacy:
        return _parse_image_seed_uri_legacy(uri)

    result = ImageSeedParseResult()

    seed_parser = _textprocessing.ConceptUriParser('Image Seed', ['mask', 'control', 'resize'])

    try:
        parse_result = seed_parser.parse_concept_uri(uri)
    except _textprocessing.ConceptPathParseError as e:
        raise ImageSeedError(e)

    seed_path = parse_result.concept
    result.seed_path = seed_path
    if seed_path.startswith('http://') or seed_path.startswith('https://'):
        result.seed_path_is_local = False
    elif os.path.exists(seed_path):
        result.seed_path_is_local = True
    else:
        raise ImageSeedError(f'Image seed file "{seed_path}" does not exist.')

    mask_path = parse_result.args.get('mask', None)
    if mask_path is not None:
        result.mask_path = mask_path
        if mask_path.startswith('http://') or mask_path.startswith('https://'):
            result.mask_path_is_local = False
        elif os.path.exists(mask_path):
            result.mask_path_is_local = True
        else:
            raise ImageSeedError(f'Image mask file "{mask_path}" does not exist.')

    control_path = parse_result.args.get('control', None)
    if control_path is not None:
        result.control_path = control_path
        if control_path.startswith('http://') or control_path.startswith('https://'):
            result.control_path_is_local = False
        elif os.path.exists(control_path):
            result.control_path_is_local = True
        else:
            raise ImageSeedError(f'Control image file "{control_path}" does not exist.')

    resize = parse_result.args.get('resize', None)
    if resize is not None:
        dimensions = tuple(int(s.strip()) for s in resize.split('x'))
        for idx, d in enumerate(dimensions):
            if d % 8 != 0:
                raise ImageSeedError(
                    f'Image seed resize {["width", "height"][idx]} dimension {d} is not divisible by 8.')

        if len(dimensions) == 1:
            result.resize_resolution = (dimensions[0], dimensions[0])
        else:
            result.resize_resolution = dimensions

    return result


def get_web_cache_directory() -> str:
    """
    Get the default web cache directory or the value of the environmental variable DGENERATE_WEB_CACHE

    :return: string (directory path)
    """
    user_cache_path = os.environ.get('DGENERATE_WEB_CACHE')

    if user_cache_path is not None:
        path = user_cache_path
    else:
        path = os.path.expanduser(os.path.join('~', '.cache', 'dgenerate', 'web'))

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    return path


@contextlib.contextmanager
def _get_web_cache_db():
    db_file = os.path.join(get_web_cache_directory(), 'cache.db')
    lock_file = os.path.join(get_web_cache_directory(), 'cache.lock')
    db = None
    with _util.temp_file_lock(lock_file):
        try:
            db = sqlite3.connect(db_file)
            db.execute(
                'CREATE TABLE IF NOT EXISTS users (pid INTEGER, UNIQUE(pid))')
            db.execute(
                'CREATE TABLE IF NOT EXISTS files '
                '(id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT UNIQUE, mime_type TEXT)')
            db.execute(
                'INSERT OR IGNORE INTO users(pid) VALUES(?)', [os.getpid()])
            yield db
            db.commit()
        except Exception:
            if db is not None:
                db.rollback()
            raise
        finally:
            db.close()


def _wipe_web_cache_directory():
    folder = get_web_cache_directory()

    with _get_web_cache_db() as db:
        db.execute('DELETE FROM users WHERE pid = (?)', [os.getpid()])
        count = db.execute('SELECT COUNT(pid) FROM users').fetchone()[0]
        if count != 0:
            # Another instance is still using
            return
        db.execute('DROP TABLE files')
        # delete any cache files that existed
        for filename in os.listdir(get_web_cache_directory()):
            file_path = os.path.join(folder, filename)
            _, ext = os.path.splitext(filename)
            if ext:
                # Do not delete the database
                continue

            _messages.debug_log(f'Deleting File From Web Cache: "{file_path}"')
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                _messages.log(
                    f'Failed to delete cached web file "{file_path}", reason: {e}',
                    level=_messages.ERROR)


atexit.register(_wipe_web_cache_directory)


def _generate_web_cache_file(url, mime_acceptable_desc):
    cache_dir = get_web_cache_directory()

    with _get_web_cache_db() as db:
        cursor = db.cursor()

        exists = cursor.execute(
            'SELECT mime_type, id FROM files WHERE url = ?', [url]).fetchone()

        if exists is not None:
            return exists[0], os.path.join(cache_dir, f'web_{exists[1]}')

        headers = {'User-Agent': fake_useragent.UserAgent().chrome}

        req = requests.get(url, headers=headers, stream=True)
        mime_type = req.headers['content-type']

        if not mime_type_is_supported(mime_type):
            raise UnknownMimetypeError(
                f'Unknown mimetype "{mime_type}" for file "{url}". '
                f'Expected: {mime_acceptable_desc}')

        cursor.execute(
            'INSERT INTO files(mime_type, url) VALUES(?, ?)', [mime_type, url])

        path = os.path.join(cache_dir, f'web_{cursor.lastrowid}')

    with open(path, mode='wb') as new_file:
        new_file.write(req.content)
        new_file.flush()

    return mime_type, path


class UnknownMimetypeError(Exception):
    pass


def fetch_image_data_stream(uri: str) -> typing.Tuple[str, typing.BinaryIO]:
    """
    Get an open stream to a local file, or file at an HTTP or HTTPS URL, with caching for web files.

    Cacheing for downloaded files is threadsafe and multiprocess safe, multiple processes using this
    module can share the cache simultaneously, the last process alive clears the cache when it exits.

    :param uri: Local file path or URL
    :param mime_type_filter: Function accepting a string (mime-type) and returning True if that mime-type is acceptable
    :param mime_acceptable_desc: String describing acceptable mime-types to be used in exceptions or None (auto generate)

    :raises: :py:exc:`.UnknownMimetypeError`

    :rtype: (mime-type string, BinaryIO)
    """

    mime_acceptable_desc = _textprocessing.oxford_comma(get_supported_mimetypes(), conjunction='or')

    if uri.startswith('http://') or uri.startswith('https://'):
        mime_type, filename = _generate_web_cache_file(uri, mime_acceptable_desc)
        return mime_type, open(filename, mode='rb')
    else:
        mime_type = mimetypes.guess_type(uri)[0]

        if mime_type is None and uri.endswith('.webp'):
            # webp missing from mimetypes library
            mime_type = "image/webp"

        if not mime_type_is_supported(mime_type):
            raise UnknownMimetypeError(
                f'Unknown mimetype "{mime_type}" for file "{uri}". Expected: {mime_acceptable_desc}')

    return mime_type, open(uri, 'rb')


def get_supported_animated_image_mimetypes() -> typing.List[str]:
    return ['image/gif', 'image/webp', 'image/apng']


def get_supported_static_image_mimetypes() -> typing.List[str]:
    return ['image/png', 'image/jpeg', 'image/bmp', 'image/psd']


def get_supported_image_mimetypes() -> typing.List[str]:
    """
    Get all supported `--image-seeds` image mimetypes, including animated image mimetypes

    :return: list of strings
    """
    return get_supported_static_image_mimetypes() + \
           get_supported_animated_image_mimetypes()


def get_supported_video_mimetypes() -> typing.List[str]:
    """
    Get all supported `--image-seeds` video mimetypes, may contain a wildcard

    :return: list of strings
    """
    return ['video/*']


def get_supported_mimetypes() -> typing.List[str]:
    """
    Get all supported `--image-seeds` mimetypes, video mimetype may contain a wildcard.

    :return: list of strings
    """
    return get_supported_image_mimetypes() + get_supported_video_mimetypes()


def mimetype_is_animated_image(mimetype: str) -> bool:
    """
    Check if a mimetype is one that dgenerate considers an animated image

    :param mimetype: The mimetype string
    :return: bool
    """
    return mimetype in get_supported_animated_image_mimetypes()


def mimetype_is_static_image(mimetype: str) -> bool:
    """
    Check if a mimetype is one that dgenerate considers a static image

    :param mimetype: The mimetype string
    :return: bool
    """
    return mimetype in get_supported_static_image_mimetypes()


def mimetype_is_video(mimetype: str) -> bool:
    """
    Check if a mimetype is a video mimetype supported by dgenerate

    :param mime_type: The mimetype string
    :return: bool
    """
    if mimetype is None:
        return False
    return mimetype.startswith('video')


def mime_type_is_supported(mimetype: str) -> bool:
    """
    Check if dgenerate supports a given input mimetype

    :param mime_type: The mimetype string
    :return: bool
    """
    return mimetype_is_static_image(mimetype) or \
           mimetype_is_animated_image(mimetype) or \
           mimetype_is_video(mimetype)


class ImageSeedInfo:
    """Information acquired about an `--image-seeds` path"""

    fps: typing.Union[float, int]
    duration: float
    is_animation: bool
    total_frames: int

    def __init__(self,
                 is_animation: bool,
                 total_frames: int,
                 fps: typing.Union[float, int],
                 duration: float):
        self.fps = fps
        self.duration = duration
        self.is_animation = is_animation
        self.total_frames = total_frames


def get_image_seed_info(uri: typing.Union[_types.Uri, ImageSeedParseResult],
                        frame_start: int = 0,
                        frame_end: _types.OptionalInteger = None) -> ImageSeedInfo:
    """
    Get an informational object from a dgenerate `--image-seeds` path

    :param image_seed_uri: The uri string or :py:class:`.ImageSeedParseResult`
    :param frame_start: slice start
    :param frame_end: slice end
    :return: :py:class:`.ImageSeedInfo`
    """
    with next(iterate_image_seed(uri, frame_start, frame_end)) as seed:
        return ImageSeedInfo(seed.is_animation_frame, seed.total_frames, seed.fps, seed.duration)


def get_control_image_info(path: typing.Union[_types.Path, ImageSeedParseResult],
                           frame_start: int = 0,
                           frame_end: _types.OptionalInteger = None) -> ImageSeedInfo:
    """
    Get an informational object from a dgenerate `--image-seeds` path that is known to be a singular control image/video.
    More efficient in this case.

    :param path: The path string or :py:class:`.ImageSeedParseResult`
    :param frame_start: slice start
    :param frame_end: slice end
    :return: :py:class:`.ImageSeedInfo`
    """
    with next(iterate_control_image(path, frame_start, frame_end)) as seed:
        return ImageSeedInfo(seed.is_animation_frame, seed.total_frames, seed.fps, seed.duration)


def create_and_exif_orient_pil_img(
        path_or_file: typing.Union[typing.BinaryIO, str],
        file_source: str,
        resize_resolution: _types.OptionalSize = None) -> PIL.Image.Image:
    """
    Create an RGB format PIL image from a file path or binary file stream.
    The image is oriented according to any EXIF directives. Image is aligned
    to 8 pixels in every case.

    :param path_or_file: file path or binary IO object
    :param file_source: Image.filename is set to this value
    :param resize_resolution: Optional resize resolution
    :return: :py:class:`PIL.Image.Image`
    """

    if isinstance(path_or_file, str):
        file = path_or_file
    else:
        file = path_or_file

    if resize_resolution is None:
        with PIL.Image.open(file) as img, _image.to_rgb(img) as rgb_img:
            e_img = _exif_orient(rgb_img)
            e_img.filename = file_source
            if not _image.is_aligned_by_8(e_img.width, e_img.height):
                with e_img:
                    resized = _image.resize_image(e_img, _image.align_by_8(e_img.width, e_img.height))
                    return resized
            else:
                return e_img
    else:
        with PIL.Image.open(file) as img, _image.to_rgb(img) as rgb_img, _exif_orient(rgb_img) as o_img:
            o_img.filename = file_source
            resized = _image.resize_image(o_img, resize_resolution)
            return resized


class MultiContextManager:
    """
    Manages the life of multiple ContextManager implementing objects
    """

    def __init__(self, objects: typing.Iterable[typing.ContextManager]):
        self.objects = objects

    def __enter__(self):
        for obj in self.objects:
            if obj is not None:
                obj.__enter__()
        return self

    def __exit__(self, t, v, traceback):
        for obj in self.objects:
            if obj is not None:
                obj.__exit__(t, v, traceback)


def _create_image_seed_reader(manage_context: list,
                              mime_type: str,
                              file_source: str,
                              preprocessor: typing.Optional[_preprocessors.ImagePreprocessor],
                              resize_resolution: _types.OptionalSize,
                              data: typing.BinaryIO,
                              throw: bool):
    reader = None
    if mimetype_is_animated_image(mime_type):
        reader = AnimatedImageReader(file=data,
                                     file_source=file_source,
                                     resize_resolution=resize_resolution,
                                     preprocessor=preprocessor)
    elif mimetype_is_video(mime_type):
        reader = VideoReader(file=data,
                             file_source=file_source,
                             resize_resolution=resize_resolution,
                             preprocessor=preprocessor)
    elif mimetype_is_static_image(mime_type):
        reader = MockImageAnimationReader(img=create_and_exif_orient_pil_img(data, file_source, resize_resolution),
                                          resize_resolution=resize_resolution,
                                          preprocessor=preprocessor)
    else:
        if throw:
            supported = _textprocessing.oxford_comma(get_supported_mimetypes(), conjunction='or')
            raise UnknownMimetypeError(
                f'Unknown mimetype "{mime_type}" for file "{file_source}". Expected: {supported}')

    if reader is not None:
        manage_context.insert(0, reader)
    return reader


def iterate_control_image(uri: typing.Union[str, ImageSeedParseResult],
                          frame_start: int = 0,
                          frame_end: _types.OptionalInteger = None,
                          resize_resolution: _types.OptionalSize = None,
                          preprocessor: _preprocessors.ImagePreprocessor = None) -> \
        typing.Generator[ImageSeed, None, None]:
    """
    Parse and load a control image/video in an `--image-seeds` path and return a generator that 
    produces :py:class:`.ImageSeed` objects while progressively reading that file.

    One or more :py:class:`.ImageSeed` objects may be yielded depending on whether an animation is being read.

    This method is more efficient than :py:meth:`.iterate_image_seed` when it is known that 
    there is only one image/video in the path.

    :param uri: `--image-seeds` path or :py:class:`.ImageSeedParseResult`
    :param frame_start: starting frame, inclusive value
    :param frame_end: optional end frame, inclusive value
    :param resize_resolution: optional resize resolution
    :param preprocessor: optional :py:class:`dgenerate.preprocessors.ImagePreprocessor`
    :return: generator over :py:class:`.ImageSeed` objects
    """

    if isinstance(uri, ImageSeedParseResult):
        uri = uri.seed_path

    control_mime_type, control_data = fetch_image_data_stream(uri=uri)

    manage_context = [control_data]

    if control_data is None:
        raise ImageSeedError('Control image not specified or irretrievable.')

    control_reader = _create_image_seed_reader(manage_context=manage_context,
                                               mime_type=control_mime_type,
                                               file_source=uri,
                                               preprocessor=preprocessor,
                                               resize_resolution=resize_resolution,
                                               data=control_data,
                                               throw=True)

    with MultiContextManager(manage_context):
        if isinstance(control_reader, MockImageAnimationReader):
            yield ImageSeed(image=control_reader.__next__())
        else:
            yield from (ImageSeed(animation_frame) for animation_frame in
                        iterate_animation_frames(seed_reader=control_reader,
                                                 frame_start=frame_start,
                                                 frame_end=frame_end))


def _iterate_image_seed_x3(seed_reader: AnimationReader,
                           mask_reader: typing.Optional[AnimationReader] = None,
                           control_reader: typing.Optional[AnimationReader] = None,
                           frame_start: int = 0,
                           frame_end: _types.OptionalInteger = None):
    if isinstance(seed_reader, MockImageAnimationReader) and \
            isinstance(mask_reader, MockImageAnimationReader) and \
            isinstance(control_reader, MockImageAnimationReader):
        yield ImageSeed(image=seed_reader.__next__(),
                        mask_image=mask_reader.__next__(),
                        control_image=control_reader.__next__())

    else:
        readers = [seed_reader, mask_reader, control_reader]
        for i in readers:
            if isinstance(i, MockImageAnimationReader):
                others = [reader for reader in readers if not isinstance(reader, MockImageAnimationReader)]
                if len(others) > 1:
                    i.total_frames = min(others,
                                         key=lambda rd: rd.total_frames).total_frames
                else:
                    i.total_frames = others[0].total_frames

        yield from (ImageSeed(animation_frame) for animation_frame in
                    iterate_animation_frames(seed_reader=seed_reader,
                                             frame_start=frame_start,
                                             frame_end=frame_end,
                                             mask_reader=mask_reader,
                                             control_reader=control_reader))


def _iterate_image_seed_x2(seed_reader: AnimationReader,
                           right_reader: AnimationReader,
                           right_image_seed_param_name: str,
                           right_reader_iterate_param_name: str,
                           frame_start: int = 0,
                           frame_end: _types.OptionalInteger = None):
    if isinstance(seed_reader, MockImageAnimationReader) \
            and isinstance(right_reader, MockImageAnimationReader):
        yield ImageSeed(image=seed_reader.__next__(),
                        **{right_image_seed_param_name: right_reader.__next__()})
    else:
        if isinstance(seed_reader, MockImageAnimationReader) and \
                not isinstance(right_reader, MockImageAnimationReader):
            seed_reader.total_frames = right_reader.total_frames

        if not isinstance(seed_reader, MockImageAnimationReader) and \
                isinstance(right_reader, MockImageAnimationReader):
            right_reader.total_frames = seed_reader.total_frames

        yield from (ImageSeed(animation_frame) for animation_frame in
                    iterate_animation_frames(seed_reader=seed_reader,
                                             frame_start=frame_start,
                                             frame_end=frame_end,
                                             **{right_reader_iterate_param_name: right_reader}))


def iterate_image_seed(uri: typing.Union[str, ImageSeedParseResult],
                       frame_start: int = 0,
                       frame_end: _types.OptionalInteger = None,
                       resize_resolution: _types.OptionalSize = None,
                       seed_image_preprocessor: typing.Optional[_preprocessors.ImagePreprocessor] = None,
                       mask_image_preprocessor: typing.Optional[_preprocessors.ImagePreprocessor] = None,
                       control_image_preprocessor: typing.Optional[_preprocessors.ImagePreprocessor] = None) -> \
        typing.Generator[ImageSeed, None, None]:
    """
    Parse and load images/videos in an `--image-seeds` path and return a generator that
    produces :py:class:`.ImageSeed` objects while progressively reading those files.
    
    One or more :py:class:`.ImageSeed` objects may be yielded depending on whether an animation is being read.
    

    :param uri: `--image-seeds` path or :py:class:`.ImageSeedParseResult`
    :param frame_start: starting frame, inclusive value
    :param frame_end: optional end frame, inclusive value
    :param resize_resolution: optional resize resolution
    :param seed_image_preprocessor: optional :py:class:`dgenerate.preprocessors.ImagePreprocessor`
    :param mask_image_preprocessor: optional :py:class:`dgenerate.preprocessors.ImagePreprocessor`
    :param control_image_preprocessor: optional :py:class:`dgenerate.preprocessors.ImagePreprocessor`
    :return: generator over :py:class:`.ImageSeed` objects
    """

    if isinstance(uri, ImageSeedParseResult):
        parse_result = uri
    else:
        parse_result = parse_image_seed_uri(uri)

    seed_mime_type, seed_data = fetch_image_data_stream(uri=parse_result.seed_path)

    mask_mime_type, mask_data = None, None

    if parse_result.mask_path is not None:
        mask_mime_type, mask_data = fetch_image_data_stream(uri=parse_result.mask_path)

    control_mime_type, control_data = None, None
    if parse_result.control_path is not None:
        control_mime_type, control_data = fetch_image_data_stream(uri=parse_result.control_path)

    if parse_result.resize_resolution is not None:
        resize_resolution = parse_result.resize_resolution

    manage_context = [seed_data, mask_data, control_data]

    if seed_data is None:
        raise ImageSeedError(f'Image seed not specified or irretrievable.')

    seed_reader = _create_image_seed_reader(manage_context=manage_context,
                                            mime_type=seed_mime_type,
                                            file_source=parse_result.seed_path,
                                            preprocessor=seed_image_preprocessor,
                                            resize_resolution=resize_resolution,
                                            data=seed_data,
                                            throw=True)
    # Optional
    mask_reader = _create_image_seed_reader(manage_context=manage_context,
                                            mime_type=mask_mime_type,
                                            file_source=parse_result.mask_path,
                                            preprocessor=mask_image_preprocessor,
                                            resize_resolution=resize_resolution,
                                            data=mask_data,
                                            throw=False) if mask_data is not None else None

    # Optional
    control_reader = _create_image_seed_reader(manage_context=manage_context,
                                               mime_type=control_mime_type,
                                               file_source=parse_result.control_path,
                                               preprocessor=control_image_preprocessor,
                                               resize_resolution=resize_resolution,
                                               data=control_data,
                                               throw=False) if control_data is not None else None

    size_mismatch_check = [(parse_result.seed_path, 'Image seed', seed_reader),
                           (parse_result.mask_path, 'Mask image', mask_reader),
                           (parse_result.control_path, 'Control image', control_reader)]

    for left in size_mismatch_check:
        for right in size_mismatch_check:
            if left[2] is not None and right[2] is not None:
                if left[2].size != right[2].size:
                    raise ImageSeedSizeMismatchError(
                        f'{left[1]} "{left[0]}" is mismatched in dimension with {right[1].lower()} "{right[0]}"')

    with MultiContextManager(manage_context):

        if mask_reader is not None and control_reader is not None:
            yield from _iterate_image_seed_x3(
                seed_reader=seed_reader,
                mask_reader=mask_reader,
                control_reader=control_reader,
                frame_start=frame_start,
                frame_end=frame_end)

        elif mask_reader is not None:

            yield from _iterate_image_seed_x2(
                seed_reader=seed_reader,
                right_reader=mask_reader,
                right_image_seed_param_name='mask_image',
                right_reader_iterate_param_name='mask_reader',
                frame_start=frame_start,
                frame_end=frame_end)

        elif control_reader is not None:
            yield from _iterate_image_seed_x2(
                seed_reader=seed_reader,
                right_reader=control_reader,
                right_image_seed_param_name='control_image',
                right_reader_iterate_param_name='control_reader',
                frame_start=frame_start,
                frame_end=frame_end
            )
        else:
            if isinstance(seed_reader, MockImageAnimationReader):
                yield ImageSeed(image=seed_reader.__next__())
            else:
                yield from (ImageSeed(animation_frame) for animation_frame in
                            iterate_animation_frames(seed_reader=seed_reader,
                                                     frame_start=frame_start,
                                                     frame_end=frame_end))
