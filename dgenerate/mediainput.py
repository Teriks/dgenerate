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
import collections.abc
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

import dgenerate.filelock as _filelock
import dgenerate.image as _image
import dgenerate.messages as _messages
import dgenerate.preprocessors as _preprocessors
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types


def frame_slice_count(total_frames: int, frame_start: int, frame_end: _types.OptionalInteger = None) -> int:
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
    Raised when the constituent image sources of an image seed specification are mismatched in dimension.
    """
    pass


class AnimationReader:
    """
    Abstract base class for animation readers.
    """

    def __init__(self,
                 width: int,
                 height: int,
                 anim_fps: int,
                 anim_frame_duration: float,
                 total_frames: int, **kwargs):
        """
        :param width: width of the animation, X dimension
        :param height: height of the animation, Y dimension
        :param anim_fps: frames per second
        :param anim_frame_duration: frame duration in milliseconds
        :param total_frames: total frames in the animation
        :param kwargs: for mixins
        """
        self._width = width
        self._height = height
        self._anim_fps = anim_fps
        self._anim_frame_duration = anim_frame_duration
        self._total_frames = total_frames

    @property
    def width(self) -> int:
        """
        Width dimension, (X dimension).

        :return: width
        """
        return self._width

    @property
    def size(self) -> _types.Size:
        """
        returns (width, height) as a tuple.

        :return: (width, height)
        """
        return self._width, self._height

    @property
    def height(self) -> int:
        """
        Height dimension, (Y dimension).

        :return: height
        """
        return self._height

    @property
    def anim_fps(self) -> int:
        """
        Frame per second.

        :return: float or integer
        """
        return self._anim_fps

    @property
    def anim_frame_duration(self) -> float:
        """
        Duration of each frame in milliseconds.

        :return: duration
        """
        return self._anim_frame_duration

    @property
    def total_frames(self) -> int:
        """
        Total number of frames that can be read.

        :return: count
        """
        return self._total_frames

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self):
        return self

    def __next__(self) -> PIL.Image.Image:
        raise StopIteration


class VideoReader(_preprocessors.ImagePreprocessorMixin, AnimationReader):
    """
    Implementation :py:class:`.AnimationReader` that reads Video files with PyAV.

    All frame images from this animation reader will be aligned by 8 pixels.
    """

    def __init__(self,
                 file: typing.Union[str, typing.BinaryIO],
                 file_source: str,
                 resize_resolution: _types.OptionalSize = None,
                 preprocessor: _preprocessors.ImagePreprocessor = None):
        """
        :param file: a file path or binary file stream

        :param file_source: the source filename for the video data, should be the filename.
            this is for informational purpose when reading from a stream or a cached file
            and should be provided in every case even if it is a symbolic value only. It
            should possess a file extension as it is used to determine file format when
            reading from a byte stream.:py:class:`PIL.Image.Image` objects produced by
            the reader will have this value set to their *filename* attribute.

        :param resize_resolution: Progressively resize each frame to this resolution while
            reading. The provided resolution will be aligned by 8 pixels.
        :param preprocessor: optionally preprocess every frame with this image preprocessor
        """
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

    def __next__(self):
        rgb_image = next(self._iter).to_image()
        rgb_image.filename = self._file_source
        return self.preprocess_image(rgb_image, self.resize_resolution)


class AnimatedImageReader(_preprocessors.ImagePreprocessorMixin, AnimationReader):
    """
    Implementation of :py:class:`.AnimationReader` that reads animated image formats using Pillow.

    All frames from this animation reader will be aligned by 8 pixels.
    """

    def __init__(self,
                 file: typing.Union[str, typing.BinaryIO],
                 file_source: str,
                 resize_resolution: _types.OptionalSize = None,
                 preprocessor: _preprocessors.ImagePreprocessor = None):
        """
        :param file: a file path or binary file stream

        :param file_source: the source filename for the animated image, should be the filename.
            this is for informational purpose when reading from a stream or a cached file
            and should be provided in every case even if it is a symbolic value only. It
            should possess a file extension. :py:class:`PIL.Image.Image` objects produced by
            the reader will have this value set to their *filename* attribute.

        :param resize_resolution: Progressively resize each frame to this
            resolution while reading. The provided resolution will be aligned
            by 8 pixels.

        :param preprocessor: optionally preprocess every frame with this image preprocessor
        """
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

    def __next__(self) -> PIL.Image.Image:
        with next(self._iter) as img:
            rgb_image = _image.to_rgb(img)
            rgb_image.filename = self._file_source
            return self.preprocess_image(rgb_image, self.resize_resolution)


class MockImageAnimationReader(_preprocessors.ImagePreprocessorMixin, AnimationReader):
    """
    Implementation of :py:class:`.AnimationReader` that repeats a single PIL image
    as many times as desired in order to mock/emulate an animation.

    All frame images from this animation reader will be aligned by 8 pixels.
    """

    def __init__(self,
                 img: PIL.Image.Image,
                 resize_resolution: _types.OptionalSize = None,
                 image_repetitions: int = 1,
                 preprocessor: _preprocessors.ImagePreprocessor = None):
        """
        :param img: source image to copy for each frame, the image is immediately copied
            once upon construction of the mock reader, and then once per frame thereafter.
            Your copy of the image can be disposed of after the construction of this object.

        :param resize_resolution: the source image will be resized to this dimension with
            a maintained aspect ratio. This occurs once upon construction, a copy is then yielded
            for each frame that is read. The provided resolution will be aligned by 8 pixels.
        :param image_repetitions: number of frames that this mock reader provides
            using a copy of the source image.
        :param preprocessor: optionally preprocess the initial image with
            this image preprocessor, this occurs once.
        """
        self._img = _image.copy_img(img)
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

        # Only need to preprocess once

        self._img = self.preprocess_image(self._img, self.resize_resolution)

    @property
    def total_frames(self) -> int:
        """
        Settable total_frames property.

        :return: frame count
        """
        return self._total_frames

    @total_frames.setter
    def total_frames(self, cnt):
        """
        Settable total_frames property.
        """
        self._total_frames = cnt

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._img.close()

    def __next__(self) -> PIL.Image.Image:
        if self._idx < self.total_frames:
            self._idx += 1
            return _image.copy_img(self._img)
        else:
            raise StopIteration


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
    The result of parsing an --image-seeds uri
    """

    seed_path: _types.Path
    """
    The seed path, contains an image path that will be used for img2img operations
    or the base image in inpaint operations. Or a controlnet guidance path, or a comma
    seperated list of controlnet guidance paths. A path being a file path, or an HTTP/HTTPS URL.
    """

    mask_path: _types.OptionalPath = None
    """
    Optional path to an inpaint mask, may be an HTTP/HTTPS URL or file path.
    """

    control_path: _types.OptionalPath = None
    """
    Optional controlnet guidance path, or comma seperated list of controlnet guidance paths. 
    This field is only used when the secondary syntax of ``--image-seeds`` is encountered.
    
    In parses such as:
    
        * ``--image-seeds "img2img.png;control=control.png"``
        * ``--image-seeds "img2img.png;control=control1.png, control2.png"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control.png"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control1.png, control2.png"``
        
    """

    resize_resolution: _types.OptionalSize = None
    """
    Per image user specified resize resolution for all components of the ``--image-seed`` specification.
    
    This field available in parses such as:
    
        * ``--image-seeds "img2img.png;512x512"``
        * ``--image-seeds "img2img.png;mask.png;512x512"``
        * ``--image-seeds "img2img.png;control=control.png;resize=512x512"``
        * ``--image-seeds "img2img.png;control=control1.png, control2.png;resize=512x512"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control.png;resize=512x512"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control1.png, control2.png;resize=512x512"``
    """

    def is_single_spec(self) -> bool:
        """
        Is this ``--image-seeds`` uri a single resource or resource group specification existing
        within the **seed_path** attribute of this object?

        For instance could it be a single img2img image definition, or a controlnet guidance
        image or list of controlnet guidance images?

        Possible parses which trigger this condition are:

            * ``--image-seeds "img2img.png"``
            * ``--image-seeds "control-image.png"``
            * ``--image-seeds "control-image1.png, control-image2.png"``

        Since this is an ambiguous parse, it must be resolved later with the help of other specified arguments.
        Such as by the specification of ``--control-nets``, which makes the intention unambiguous.

        :return: bool
        """
        return self.mask_path is None and self.control_path is None


# noinspection HttpUrlsUsage
def _parse_image_seed_uri_legacy(uri: str) -> ImageSeedParseResult:
    parts = (x.strip() for x in uri.split(';'))
    result = ImageSeedParseResult()

    first = next(parts)
    result.seed_path = first

    first_parts = first.split(',')

    for part in first_parts:
        part = part.strip()
        if not (part.startswith('http://') or part.startswith('https://') or os.path.exists(part)):
            if len(first_parts) > 0:
                raise ImageSeedError(f'Control image file "{part}" does not exist.')
            else:
                raise ImageSeedError(f'Image sed file "{part}" does not exist.')

    for part in parts:
        if part == '':
            raise ImageSeedError(
                'Missing inpaint mask image or output size specification, '
                'check image seed syntax, stray semicolon?')

        if part.startswith('http://') or part.startswith('https://'):
            result.mask_path = part
        elif os.path.exists(part):
            result.mask_path = part
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
    Parse an ``--image-seeds`` uri into its constituents

    :param uri: ``--image-seeds`` uri
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

    # noinspection HttpUrlsUsage
    def _ensure_exists(path, title):
        if not (path.startswith('http://') or path.startswith('https://') or os.path.exists(path)):
            raise ImageSeedError(f'{title} file "{path}" does not exist.')

    seed_path = parse_result.concept
    _ensure_exists(seed_path, 'Image seed')
    result.seed_path = seed_path

    mask_path = parse_result.args.get('mask', None)
    if mask_path is not None:
        _ensure_exists(mask_path, 'Image mask')
        result.mask_path = mask_path

    control_path = parse_result.args.get('control', None)
    if control_path is not None:
        result.control_path = control_path
        for p in control_path.split(','):
            p = p.strip()
            _ensure_exists(p, 'Control image')

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
    with _filelock.temp_file_lock(lock_file):
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


# noinspection HttpUrlsUsage
def fetch_image_data_stream(uri: str) -> typing.Tuple[str, typing.BinaryIO]:
    """
    Get an open stream to a local file, or file at an HTTP or HTTPS URL, with caching for web files.

    Caching for downloaded files is multiprocess safe, multiple processes using this
    module can share the cache simultaneously, the last process alive clears the cache when it exits.

    :param uri: Local file path or URL

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
    Get all supported ``--image-seeds`` image mimetypes, including animated image mimetypes

    :return: list of strings
    """
    return get_supported_static_image_mimetypes() + get_supported_animated_image_mimetypes()


def get_supported_video_mimetypes() -> typing.List[str]:
    """
    Get all supported ``--image-seeds`` video mimetypes, may contain a wildcard

    :return: list of strings
    """
    return ['video/*']


def get_supported_mimetypes() -> typing.List[str]:
    """
    Get all supported ``--image-seeds`` mimetypes, video mimetype may contain a wildcard.

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

    :param mimetype: The mimetype string

    :return: bool
    """
    if mimetype is None:
        return False
    return mimetype.startswith('video')


# noinspection PyPep8
def mime_type_is_supported(mimetype: str) -> bool:
    """
    Check if dgenerate supports a given input mimetype

    :param mimetype: The mimetype string

    :return: bool
    """
    return mimetype_is_static_image(mimetype) or \
           mimetype_is_animated_image(mimetype) or \
           mimetype_is_video(mimetype)


def create_and_exif_orient_pil_img(
        path_or_file: typing.Union[typing.BinaryIO, str],
        file_source: str,
        resize_resolution: _types.OptionalSize = None) -> PIL.Image.Image:
    """
    Create an RGB format PIL image from a file path or binary file stream.
    The image is oriented according to any EXIF directives. Image is aligned
    to 8 pixels in every case.

    :param path_or_file: file path or binary IO object
    :param file_source: :py:attr:`PIL.Image.Image.filename` is set to this value
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


def create_animation_reader(mimetype: str,
                            file_source: str,
                            file: typing.BinaryIO,
                            resize_resolution: _types.OptionalSize = None,
                            preprocessor: typing.Optional[_preprocessors.ImagePreprocessor] = None,
                            ) -> AnimationReader:
    """
    Create an animation reader object from mimetype specification and binary file stream.

    Images will return a :py:class:`.MockImageAnimationReader` with a *total_frames* value of 1,
    which can then be adjusted by you.


    :py:class:`.VideoReader` or :py:class:`.AnimatedImageReader` will be returned for Video
    files and Animated Images respectively.

    :param mimetype: one of :py:meth:`.get_supported_mimetypes`

    :param file: the binary file stream

    :param file_source: the source filename for videos/animated images, should be the filename.
        this is for informational purpose and should be provided in every case even if it is a
        symbolic value only. It should possess a file extension. :py:class:`PIL.Image.Image`
        objects produced by the reader will have this value set to their *filename* attribute.

    :param resize_resolution: Progressively resize each frame to this
        resolution while reading. The provided resolution will be aligned
        by 8 pixels.

    :param preprocessor: optionally preprocess every frame with this image preprocessor

    :return: :py:class:`.AnimationReader`
    """

    if mimetype_is_animated_image(mimetype):
        return AnimatedImageReader(file=file,
                                   file_source=file_source,
                                   resize_resolution=resize_resolution,
                                   preprocessor=preprocessor)
    elif mimetype_is_video(mimetype):
        return VideoReader(file=file,
                           file_source=file_source,
                           resize_resolution=resize_resolution,
                           preprocessor=preprocessor)
    elif mimetype_is_static_image(mimetype):
        with create_and_exif_orient_pil_img(file, file_source,
                                            resize_resolution) as img:
            return MockImageAnimationReader(img=img,
                                            resize_resolution=resize_resolution,
                                            preprocessor=preprocessor)
    else:
        supported = _textprocessing.oxford_comma(get_supported_mimetypes(), conjunction='or')
        raise UnknownMimetypeError(
            f'Unknown mimetype "{mimetype}" for file "{file_source}". Expected: {supported}')


class AnimationReaderSpec:
    """
    Used by :py:class:`.MultiAnimationReader` to define resource paths.
    """

    path: str
    """
    File path (or HTTP/HTTPS URL with default **path_opener**)
    """

    preprocessor: typing.Optional[_preprocessors.ImagePreprocessor] = None
    """
    Optional image preprocessor associated with the file
    """

    def __init__(self, path: str,
                 preprocessor: typing.Optional[_preprocessors.ImagePreprocessor] = None):
        """
        :param path: File path or URL
        :param preprocessor: Optional image preprocessor associated with the file
        """
        self.path = path
        self.preprocessor = preprocessor


class MultiAnimationReader:
    """
    Zips together multiple automatically created :py:class:`.AnimationReader` implementations and
    allows enumeration over their reads, which are collected into a list of a defined order.

    Images when zipped together with animated files will be repeated over the total amount of frames.

    The animation with the lowest amount of frames determines the total amount of
    frames that can be read when animations are involved.

    If all paths point to images, then :py:attr:`.MultiAnimationReader.total_frames` will be 1.

    All images read by this reader are aligned by 8 pixels, there is no guarantee that
    images read from the individual readers are the same size and you must handle that condition.
    """

    readers: typing.List[AnimationReader]
    _file_streams: typing.List[typing.BinaryIO]
    _total_frames: int = 0
    _frame_start: int = 0
    _frame_end: _types.OptionalInteger = None
    _frame_index: int = -1

    @property
    def frame_index(self) -> int:
        """
        Current frame index while reading.
        """
        return self._frame_index

    @property
    def frame_end(self) -> int:
        """
        Frame slice end value (inclusive)
        """
        return self._frame_end

    @property
    def frame_start(self) -> int:
        """
        Frame slice start value (inclusive)
        """
        return self._frame_start

    @property
    def total_frames(self) -> int:
        """
        Total number of frames readable from this reader.
        """
        return self._total_frames

    @property
    def anim_fps(self) -> _types.OptionalInteger:
        """
        Frames per second, this will be None if there is only a single frame
        """
        return self._anim_fps

    @property
    def anim_frame_duration(self) -> _types.OptionalFloat:
        """
        Duration of a frame in milliseconds, this will be None if there is only a single frame
        """
        return self._anim_frame_duration

    def __init__(self,
                 specs: typing.List[AnimationReaderSpec],
                 resize_resolution: _types.OptionalSize = None,
                 frame_start: int = 0,
                 frame_end: _types.OptionalInteger = None,
                 path_opener: typing.Callable[[str], typing.BinaryIO] = fetch_image_data_stream):
        """
        :param specs: list of :py:class:`.AnimationReaderSpec`
        :param resize_resolution: optional resize resolution for frames
        :param frame_start: inclusive frame slice start frame
        :param frame_end: inclusive frame slice end frame
        :param path_opener: opens a binary file stream from paths
            mentioned by :py:class:`.AnimationReaderSpec`
        """

        if frame_end is not None:
            if frame_start > frame_end:
                raise ValueError('frame_start must be less than or equal to frame_end')

        self.readers = []
        self._file_streams = []

        self._frame_start = frame_start
        self._frame_end = frame_end
        self._anim_frame_duration = None
        self._anim_fps = None

        for spec in specs:
            mimetype, file_stream = path_opener(spec.path)

            self.readers.append(
                create_animation_reader(
                    mimetype=mimetype,
                    file_source=spec.path,
                    file=file_stream,
                    resize_resolution=resize_resolution,
                    preprocessor=spec.preprocessor)
            )
            self._file_streams.append(file_stream)

        non_images = [r for r in self.readers if not isinstance(r, MockImageAnimationReader)]

        if non_images:
            self._total_frames = min(
                non_images, key=lambda r: r.total_frames).total_frames

            self._total_frames = frame_slice_count(self.total_frames, frame_start, frame_end)

            self._anim_fps = non_images[0].anim_fps
            self._anim_frame_duration = non_images[0].anim_frame_duration

            for r in self.readers:
                if isinstance(r, MockImageAnimationReader):
                    r.total_frames = self.total_frames
        else:
            self._total_frames = 1

    def __next__(self):
        if self._frame_index == -1:
            # First call, skip up to frame start
            for idx in range(0, self._frame_start):
                for r in self.readers:
                    if isinstance(r, _preprocessors.ImagePreprocessorMixin):
                        old_val = r.preprocess_enabled
                        r.preprocess_enabled = False
                        try:
                            r.__next__().close()
                        finally:
                            r.preprocess_enabled = old_val
                    else:
                        r.__next__().close()

                if self._frame_index == self._frame_end:
                    # This should only be able to happen if frame_start > frame_end
                    # which is checked for in the constructor
                    raise AssertionError(
                        'impossible iteration termination condition '
                        'in MultiAnimationReader reader')

                self._frame_index += 1

        if self._frame_index == self._frame_end:
            raise StopIteration

        read = [r.__next__() for r in self.readers]

        self._frame_index += 1

        return read

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for r in self.readers:
            r.__exit__(exc_type, exc_val, exc_tb)
        for file_stream in self._file_streams:
            file_stream.close()


class ImageSeed:
    """
    An ImageSeed with attached image data
    """

    frame_index: _types.OptionalInteger = None
    """
    Frame index in the case that :py:attr:`.ImageSeed.is_animation_frame` is True
    """

    total_frames: _types.OptionalInteger = None
    """
    Total frame count in the case that :py:attr:`.ImageSeed.is_animation_frame` is True
    """

    anim_fps: typing.Union[int, float, None] = None
    """
    Frames per second in the case that :py:attr:`.ImageSeed.is_animation_frame` is True
    """

    anim_frame_duration: _types.OptionalFloat = None
    """
    Duration of a frame in milliseconds in the case that :py:attr:`.ImageSeed.is_animation_frame` is True
    """

    image: typing.Optional[PIL.Image.Image]
    """
    An optional image used for img2img mode, or inpainting mode in combination with :py:attr:`.ImageSeed.mask`
    """

    mask_image: typing.Optional[PIL.Image.Image]
    """
    An optional inpaint mask image, may be None
    """

    control_images: typing.Optional[typing.List[PIL.Image.Image]]
    """
    List of control guidance images, or None.
    """

    is_animation_frame: bool
    """
    Is this part of an animation?
    """

    def __init__(self,
                 image: typing.Optional[PIL.Image.Image] = None,
                 mask_image: typing.Optional[PIL.Image.Image] = None,
                 control_images: typing.Optional[typing.List[PIL.Image.Image]] = None):
        self.image = image
        self.mask_image = mask_image
        self.control_images = control_images

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.image is not None:
            self.image.close()

        if self.mask_image is not None:
            self.mask_image.close()

        if self.control_images:
            for i in self.control_images:
                i.close()


def _check_image_dimensions_match(images):
    ix: PIL.Image.Image
    for ix in images:
        iy: PIL.Image.Image
        for iy in images:
            if ix.size != iy.size:
                raise ImageSeedSizeMismatchError(
                    f'Dimension of "{_image.get_filename(ix)}" ({_textprocessing.format_size(ix.size)}) does '
                    f'not match "{_image.get_filename(iy)}" ({_textprocessing.format_size(iy.size)})')


def _flatten(xs):
    for x in xs:
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, (str, bytes)):
            yield from _flatten(x)
        else:
            yield x


ControlPreprocessorSpec = typing.Union[_preprocessors.ImagePreprocessor,
                                       typing.List[_preprocessors.ImagePreprocessor], None]


def iterate_image_seed(uri: typing.Union[str, ImageSeedParseResult],
                       frame_start: int = 0,
                       frame_end: _types.OptionalInteger = None,
                       resize_resolution: _types.OptionalSize = None,
                       seed_image_preprocessor: typing.Optional[_preprocessors.ImagePreprocessor] = None,
                       mask_image_preprocessor: typing.Optional[_preprocessors.ImagePreprocessor] = None,
                       control_image_preprocessor: ControlPreprocessorSpec = None) -> \
        typing.Generator[ImageSeed, None, None]:
    """
    Parse and load images/videos in an ``--image-seeds`` uri and return a generator that
    produces :py:class:`.ImageSeed` objects while progressively reading those files.

    This method is used to iterate over an ``--image-seeds`` uri in the case that the image source
    mentioned is to be used for img2img / inpaint operations, and handles this syntax:

        * ``--image-seeds "img2img.png;mask.png"``
        * ``--image-seeds "img2img.png;mask.png;512x512"``

    Additionally controlnet guidance resources are handled via the secondary syntax:

        * ``--image-seeds "img2img.png;control=control1.png, control2.png"``
        * ``--image-seeds "img2img.png;control=control1.png, control2.png;resize=512x512"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control1.png, control2.png"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control1.png, control2.png;resize=512x512"``

    One or more :py:class:`.ImageSeed` objects may be yielded depending on whether an animation is being read.

    :param uri: ``--image-seeds`` uri or :py:class:`.ImageSeedParseResult`
    :param frame_start: starting frame, inclusive value
    :param frame_end: optional end frame, inclusive value
    :param resize_resolution: optional resize resolution
    :param seed_image_preprocessor: optional :py:class:`dgenerate.preprocessors.ImagePreprocessor`
    :param mask_image_preprocessor: optional :py:class:`dgenerate.preprocessors.ImagePreprocessor`

    :param control_image_preprocessor: optional :py:class:`dgenerate.preprocessors.ImagePreprocessor` or list of them.
        A list is used to specify preprocessors for individual images in a multi guidance image specification
        such as uri = "seed.png;control=img1.png, img2.png".  In the case that a multi guidance image specification is used
        and only one preprocessor is given, that preprocessor will be used on every image / video in the
        specification.

    :return: generator over :py:class:`.ImageSeed` objects
    """

    if frame_end is not None:
        if frame_start > frame_end:
            raise ValueError('frame_start must be less than or equal to frame_end')

    if isinstance(uri, ImageSeedParseResult):
        parse_result = uri
    else:
        parse_result = parse_image_seed_uri(uri)

    reader_specs = [
        AnimationReaderSpec(parse_result.seed_path, seed_image_preprocessor)
    ]

    if parse_result.mask_path is not None:
        reader_specs.append(AnimationReaderSpec(parse_result.mask_path, mask_image_preprocessor))

    if parse_result.control_path is not None:
        if isinstance(control_image_preprocessor, list):
            reader_specs += [
                AnimationReaderSpec(p.strip(), control_image_preprocessor[idx])
                for idx, p in enumerate(parse_result.control_path.split(','))
            ]
        else:
            reader_specs += [
                AnimationReaderSpec(p.strip(), control_image_preprocessor)
                for p in parse_result.control_path.split(',')
            ]

    if parse_result.resize_resolution is not None:
        resize_resolution = parse_result.resize_resolution

    with MultiAnimationReader(specs=reader_specs,
                              resize_resolution=resize_resolution,
                              frame_start=frame_start,
                              frame_end=frame_end) as reader:

        is_animation = reader.total_frames > 1

        dimensions_checked = False

        for frame in reader:

            if parse_result.mask_path is not None and parse_result.control_path is not None:
                image_seed = ImageSeed(image=frame[0],
                                       mask_image=frame[1],
                                       control_images=frame[2:])
            elif parse_result.mask_path is not None:
                image_seed = ImageSeed(image=frame[0], mask_image=frame[1])
            elif parse_result.control_path is not None:
                image_seed = ImageSeed(image=frame[0], control_images=frame[1:])
            else:
                image_seed = ImageSeed(image=frame[0])

            if not dimensions_checked:
                images = list(_flatten([image_seed.image if image_seed.image else [],
                                        image_seed.mask_image if image_seed.mask_image else [],
                                        image_seed.control_images if image_seed.control_images else []]))

                _check_image_dimensions_match(images)

                dimensions_checked = True

            image_seed.is_animation_frame = is_animation
            if is_animation:
                image_seed.anim_fps = reader.anim_fps
                image_seed.anim_frame_duration = reader.anim_frame_duration
                image_seed.frame_index = reader.frame_index
                image_seed.total_frames = reader.total_frames if is_animation else None
            yield image_seed


def iterate_control_image(uri: typing.Union[str, ImageSeedParseResult],
                          frame_start: int = 0,
                          frame_end: _types.OptionalInteger = None,
                          resize_resolution: _types.OptionalSize = None,
                          preprocessor: ControlPreprocessorSpec = None) -> \
        typing.Generator[ImageSeed, None, None]:
    """
    Parse and load a control image/video in an ``--image-seeds`` uri and return a generator that
    produces :py:class:`.ImageSeed` objects while progressively reading that file.

    One or more :py:class:`.ImageSeed` objects may be yielded depending on whether an animation is being read.

    This can consist of a single resource path or a list of comma seperated image and
    video resource paths, which may be files on disk or remote files (http / https).

    This method is to be used when it is known that there is only a controlnet guidance resource
    specification in the path, and it handles this specification syntax:

        * ``--image-seeds "control1.png"``
        * ``--image-seeds "control1.png, control2.png"``

    The image or images read will be available from the :py:attr:`.ImageSeed.control_images` attribute.

    :param uri: ``--image-seeds`` uri or :py:class:`.ImageSeedParseResult`
    :param frame_start: starting frame, inclusive value
    :param frame_end: optional end frame, inclusive value
    :param resize_resolution: optional resize resolution
    :param preprocessor: optional :py:class:`dgenerate.preprocessors.ImagePreprocessor` or list of them.
        A list is used to specify preprocessors for individual images in a multi guidance image specification
        such as path = "img1.png, img2.png".  In the case that a multi guidance image specification is used
        and only one preprocessor is given, that preprocessor will be used on every image / video in the
        specification.
    :return: generator over :py:class:`.ImageSeed` objects
    """

    if frame_end is not None:
        if frame_start > frame_end:
            raise ValueError('frame_start must be less than or equal to frame_end')

    if isinstance(uri, ImageSeedParseResult):
        uri = uri.seed_path

    reader_specs = []

    if isinstance(preprocessor, list):
        reader_specs += [
            AnimationReaderSpec(p.strip(), preprocessor[idx])
            for idx, p in enumerate(uri.split(','))
        ]
    else:
        reader_specs += [
            AnimationReaderSpec(p.strip(), preprocessor)
            for p in uri.split(',')
        ]

    with MultiAnimationReader(specs=reader_specs,
                              resize_resolution=resize_resolution,
                              frame_start=frame_start,
                              frame_end=frame_end) as reader:

        is_animation = reader.total_frames > 1

        dimensions_checked = False

        for frame in reader:
            image_seed = ImageSeed(control_images=frame)

            if not dimensions_checked:
                images = list(_flatten([image_seed.control_images]))

                _check_image_dimensions_match(images)

                dimensions_checked = True

            image_seed.is_animation_frame = is_animation
            if is_animation:
                image_seed.anim_fps = reader.anim_fps
                image_seed.anim_frame_duration = reader.anim_frame_duration
                image_seed.frame_index = reader.frame_index
                image_seed.total_frames = reader.total_frames if is_animation else None
            yield image_seed


class ImageSeedInfo:
    """Information acquired about an ``--image-seeds`` uri"""

    anim_fps: _types.OptionalInteger
    """
    Animation frames per second in the case that :py:attr:`.ImageSeedInfo.is_animation` is True
    """

    anim_frame_duration: _types.OptionalFloat
    """
    Animation frame duration in milliseconds in the case that :py:attr:`.ImageSeedInfo.is_animation` is True
    """

    total_frames: _types.OptionalInteger
    """
    Animation frame count in the case that :py:attr:`.ImageSeedInfo.is_animation` is True
    """

    is_animation: bool
    """
    Does this image seed specification result in an animation?
    """

    def __init__(self,
                 is_animation: bool,
                 total_frames: _types.OptionalInteger,
                 anim_fps: _types.OptionalInteger,
                 anim_frame_duration: _types.OptionalFloat):
        self.anim_fps = anim_fps
        self.anim_frame_duration = anim_frame_duration
        self.is_animation = is_animation
        self.total_frames = total_frames


def get_image_seed_info(uri: typing.Union[_types.Uri, ImageSeedParseResult],
                        frame_start: int = 0,
                        frame_end: _types.OptionalInteger = None) -> ImageSeedInfo:
    """
    Get an informational object from a dgenerate ``--image-seeds`` uri.

    This method is used to obtain information about an ``--image-seeds`` uri in the case that the
    image source mentioned is to be used for img2img / inpaint operations, and handles this syntax:

        * ``--image-seeds "img2img.png;mask.png"``
        * ``--image-seeds "img2img.png;mask.png;512x512"``

    Additionally control net image sources are handled via the secondary syntax:

        * ``--image-seeds "img2img.png;control=control1.png, control2.png"``
        * ``--image-seeds "img2img.png;control=control1.png, control2.png;resize=512x512"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control1.png, control2.png"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control1.png, control2.png;resize=512x512"``

    :param uri: The uri string or :py:class:`.ImageSeedParseResult`
    :param frame_start: slice start
    :param frame_end: slice end
    :return: :py:class:`.ImageSeedInfo`
    """
    with next(iterate_image_seed(uri, frame_start, frame_end)) as seed:
        return ImageSeedInfo(seed.is_animation_frame, seed.total_frames, seed.anim_fps, seed.anim_frame_duration)


def get_control_image_info(uri: typing.Union[_types.Path, ImageSeedParseResult],
                           frame_start: int = 0,
                           frame_end: _types.OptionalInteger = None) -> ImageSeedInfo:
    """
    Get an informational object from a dgenerate ``--image-seeds`` uri that is known to be a
    control image/video specification.

    This can consist of a single resource path or a list of comma seperated image and
    video resource paths, which may be files on disk or remote files (http / https).

    This method is to be used when it is known that there is only a control image/video specification in the path,
    and it handles this specification syntax:

        * ``--image-seeds "control1.png"``
        * ``--image-seeds "control1.png, control2.png"``

    :param uri: The path string or :py:class:`.ImageSeedParseResult`
    :param frame_start: slice start
    :param frame_end: slice end
    :return: :py:class:`.ImageSeedInfo`
    """
    with next(iterate_control_image(uri, frame_start, frame_end)) as seed:
        return ImageSeedInfo(seed.is_animation_frame, seed.total_frames, seed.anim_fps, seed.anim_frame_duration)
