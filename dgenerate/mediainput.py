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
import mimetypes
import os
import pathlib
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


class AnimationFrame:
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


def _is_frame_in_slice(idx, frame_start, frame_end):
    return idx >= frame_start and (frame_end is None or idx <= frame_end)


def _total_frames_slice(total_frames, frame_start, frame_end):
    return min(total_frames, (frame_end + 1 if frame_end is not None else total_frames)) - frame_start


class ImageSeedSizeMismatchError(Exception):
    pass


class AnimationReader:
    # interface
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
    def size(self) -> typing.Tuple[int, int]:
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

    def frame_slice_count(self, frame_start: int = 0, frame_end: typing.Union[int, None] = None) -> int:
        return _total_frames_slice(self.total_frames, frame_start, frame_end)


class VideoReader(_preprocessors.ImagePreprocessorMixin, AnimationReader):
    def __init__(self,
                 file: typing.Union[str, typing.BinaryIO],
                 file_source: str,
                 resize_resolution: typing.Union[typing.Tuple[int, int], None] = None,
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
            total_frames = sum(1 for i in self._container.decode(video=0))
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
        return _total_frames_slice(self.total_frames, frame_start, frame_end)

    def __next__(self):
        rgb_image = next(self._iter).to_image()
        rgb_image.filename = self._file_source
        return self.preprocess_image(rgb_image, self.resize_resolution)


class GifWebpReader(_preprocessors.ImagePreprocessorMixin, AnimationReader):
    def __init__(self,
                 file: typing.Union[str, typing.BinaryIO],
                 file_source: str,
                 resize_resolution: typing.Union[typing.Tuple[int, int], None] = None,
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

    def frame_slice_count(self, frame_start: int = 0, frame_end: typing.Union[int, None] = None):
        return _total_frames_slice(self.total_frames, frame_start, frame_end)

    def __next__(self) -> PIL.Image.Image:
        with next(self._iter) as img:
            rgb_image = _image.to_rgb(img)
            rgb_image.filename = self._file_source
            return self.preprocess_image(rgb_image, self.resize_resolution)


class MockImageAnimationReader(_preprocessors.ImagePreprocessorMixin, AnimationReader):
    def __init__(self,
                 img: PIL.Image.Image,
                 resize_resolution: typing.Union[typing.Tuple[int, int], None] = None,
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

    def frame_slice_count(self, frame_start: int = 0, frame_end: typing.Union[int, None] = None) -> int:
        return _total_frames_slice(self.total_frames, frame_start, frame_end)

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
                                 frame_end: typing.Union[int, None] = None):
    total_frames = seed_reader.frame_slice_count(frame_start, frame_end)
    right_total_frames = right_reader.frame_slice_count(frame_start, frame_end)
    out_frame_idx = 0
    in_slice = None

    # Account for videos possibly having a differing number of frames
    total_frames = min(total_frames, right_total_frames)

    for in_frame_idx, frame in enumerate(zip(seed_reader, right_reader)):
        seed_image = frame[0]
        right_image = frame[1]

        if _is_frame_in_slice(in_frame_idx, frame_start, frame_end):
            if in_slice is None:
                in_slice = True
            yield AnimationFrame(frame_index=out_frame_idx,
                                 total_frames=total_frames,
                                 anim_fps=seed_reader.anim_fps,
                                 anim_frame_duration=seed_reader.anim_frame_duration,
                                 image=seed_image,
                                 **{right_animation_frame_param_name: right_image})
            out_frame_idx += 1
        elif in_slice:
            break


def _iterate_animation_frames_x3(seed_reader: AnimationReader,
                                 mask_reader: typing.Union[AnimationReader, None] = None,
                                 control_reader: typing.Union[AnimationReader, None] = None,
                                 frame_start: int = 0,
                                 frame_end: typing.Union[int, None] = None):
    total_frames = seed_reader.frame_slice_count(frame_start, frame_end)
    mask_total_frames = mask_reader.frame_slice_count(frame_start, frame_end)
    control_total_frames = control_reader.frame_slice_count(frame_start, frame_end)
    out_frame_idx = 0
    in_slice = None

    # Account for videos possibly having a differing number of frames
    total_frames = min(total_frames, mask_total_frames, control_total_frames)

    for in_frame_idx, frame in enumerate(zip(seed_reader, mask_reader, control_reader)):

        image = frame[0]
        mask = frame[1]
        control = frame[2]

        if _is_frame_in_slice(in_frame_idx, frame_start, frame_end):
            if in_slice is None:
                in_slice = True
            yield AnimationFrame(frame_index=out_frame_idx,
                                 total_frames=total_frames,
                                 anim_fps=seed_reader.anim_fps,
                                 anim_frame_duration=seed_reader.anim_frame_duration,
                                 image=image,
                                 mask_image=mask,
                                 control_image=control)
            out_frame_idx += 1
        elif in_slice:
            break


def iterate_animation_frames(seed_reader: AnimationReader,
                             mask_reader: typing.Union[AnimationReader, None] = None,
                             control_reader: typing.Union[AnimationReader, None] = None,
                             frame_start: int = 0,
                             frame_end: typing.Union[int, None] = None):
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
        in_slice = None
        for in_frame_idx, frame in enumerate(seed_reader):
            if _is_frame_in_slice(in_frame_idx, frame_start, frame_end):
                if in_slice is None:
                    in_slice = True
                yield AnimationFrame(frame_index=out_frame_idx,
                                     total_frames=total_frames,
                                     anim_fps=seed_reader.anim_fps,
                                     anim_frame_duration=seed_reader.anim_frame_duration,
                                     image=frame)
                out_frame_idx += 1
            elif in_slice:
                break


class ImageSeed:
    def __init__(self,
                 image: typing.Union[PIL.Image.Image, AnimationFrame],
                 mask_image: typing.Union[PIL.Image.Image, None] = None,
                 control_image: typing.Union[PIL.Image.Image, None] = None):

        self.is_animation_frame = isinstance(image, AnimationFrame)
        self.frame_index = None
        self.total_frames = None
        self.fps = None
        self.duration = None
        self.image = None
        self.mask_image = None
        self.control_image = None

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


def _exif_orient(image):
    exif = image.getexif()
    for k in exif.keys():
        if k != 0x0112:
            exif[k] = None
            del exif[k]
    new_exif = exif.tobytes()
    image.info["exif"] = new_exif
    return PIL.ImageOps.exif_transpose(image)


class ImageSeedParseError(Exception):
    pass


class ImageSeedParseResult:
    def __init__(self):
        self.uri: typing.Union[str, None] = None
        self.uri_is_local: bool = False
        self.mask_uri: typing.Union[str, None] = None
        self.mask_uri_is_local: bool = False
        self.control_uri: typing.Union[str, None] = None
        self.control_uri_is_local: bool = False
        self.resize_resolution: typing.Union[typing.Tuple[int, int], None] = None

    def is_single_image(self) -> bool:
        return self.uri is not None and self.mask_uri is None and self.control_uri is None


def parse_image_seed_uri_legacy(uri: str):
    parts = (x.strip() for x in uri.split(';'))
    result = ImageSeedParseResult()

    first = next(parts)
    result.uri = first
    if first.startswith('http://') or first.startswith('https://'):
        result.uri_is_local = False
    elif os.path.exists(first):
        result.uri_is_local = True
    else:
        raise ImageSeedParseError(f'Image seed file "{first}" does not exist.')

    for part in parts:
        if part == '':
            raise ImageSeedParseError(
                'Missing inpaint mask image or output size specification, check image seed syntax, stray semicolon?')

        if part.startswith('http://') or part.startswith('https://'):
            result.mask_uri = part
            result.mask_uri_is_local = False
        elif os.path.exists(part):
            result.mask_uri = part
            result.mask_uri_is_local = True
        else:
            try:
                dimensions = tuple(int(s.strip()) for s in part.split('x'))
                for idx, d in enumerate(dimensions):
                    if d % 8 != 0:
                        raise ImageSeedParseError(
                            f'Image seed resize {["width", "height"][idx]} dimension {d} is not divisible by 8.')

                result.resize_resolution = dimensions
            except ValueError:
                raise ImageSeedParseError(f'Inpaint mask file "{part}" does not exist.')

            if len(result.resize_resolution) == 1:
                result.resize_resolution = (result.resize_resolution[0], result.resize_resolution[0])
    return result


def parse_image_seed_uri(uri: str):
    parts = uri.split(';')

    non_legacy = len(parts) > 3

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
        return parse_image_seed_uri_legacy(uri)

    result = ImageSeedParseResult()

    seed_parser = _textprocessing.ConceptPathParser('Image Seed', ['mask', 'control', 'resize'])

    try:
        parse_result = seed_parser.parse_concept_path(uri)
    except _textprocessing.ConceptPathParseError as e:
        raise ImageSeedParseError(e)

    uri = parse_result.concept
    result.uri = uri
    if uri.startswith('http://') or uri.startswith('https://'):
        result.uri_is_local = False
    elif os.path.exists(uri):
        result.uri_is_local = True
    else:
        raise ImageSeedParseError(f'Image seed file "{uri}" does not exist.')

    mask_uri = parse_result.args.get('mask', None)
    if mask_uri is not None:
        result.mask_uri = mask_uri
        if mask_uri.startswith('http://') or mask_uri.startswith('https://'):
            result.mask_uri_is_local = False
        elif os.path.exists(mask_uri):
            result.mask_uri_is_local = True
        else:
            raise ImageSeedParseError(f'Image mask file "{mask_uri}" does not exist.')

    control_uri = parse_result.args.get('control', None)
    if control_uri is not None:
        result.control_uri = control_uri
        if control_uri.startswith('http://') or control_uri.startswith('https://'):
            result.control_uri_is_local = False
        elif os.path.exists(control_uri):
            result.control_uri_is_local = True
        else:
            raise ImageSeedParseError(f'Control image file "{control_uri}" does not exist.')

    resize = parse_result.args.get('resize', None)
    if resize is not None:
        dimensions = tuple(int(s.strip()) for s in resize.split('x'))
        for idx, d in enumerate(dimensions):
            if d % 8 != 0:
                raise ImageSeedParseError(
                    f'Image seed resize {["width", "height"][idx]} dimension {d} is not divisible by 8.')

        if len(dimensions) == 1:
            result.resize_resolution = (dimensions[0], dimensions[0])
        else:
            result.resize_resolution = dimensions

    return result


def image_mime_type_filter(mime_type: str) -> bool:
    return (mime_type_is_static_image(mime_type) or
            mime_type_is_video(mime_type) or
            mime_type_is_animable_image(mime_type))


WEB_FILE_CACHE = dict()


def get_web_cache_directory() -> str:
    user_cache_path = os.environ.get('DGENERATE_WEB_CACHE')

    if user_cache_path is not None:
        path = user_cache_path
    else:
        path = os.path.expanduser(os.path.join('~', '.cache', 'dgenerate', 'web'))

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    return path


def _wipe_web_cache_directory():
    folder = get_web_cache_directory()
    _messages.debug_log(f'Wiping Web Cache Directory: "{folder}"')
    for filename in os.listdir(get_web_cache_directory()):
        file_path = os.path.join(folder, filename)
        _messages.debug_log(f'Deleting File From Web Cache: "{file_path}"')
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            _messages.log(f'Failed to delete cached web file "{file_path}", reason: {e}',
                          level=_messages.ERROR)


atexit.register(_wipe_web_cache_directory)


def generate_web_cache_filename() -> str:
    name = "cached_file"
    cache_dir = get_web_cache_directory()
    filename = os.path.join(cache_dir, name)

    if not os.path.exists(filename):
        return filename

    idx = 1
    while os.path.exists(filename):
        filename = filename + f'_{idx}'
        idx += 1

    return filename


def fetch_image_data_stream(uri: str,
                            uri_desc: str,
                            mime_type_filter: typing.Callable[[str], bool] = image_mime_type_filter,
                            mime_type_reject_noun: str = 'input image',
                            mime_acceptable_desc: str = ''):
    if uri.startswith('http://') or uri.startswith('https://'):
        cache_hit = WEB_FILE_CACHE.get(uri)
        if cache_hit is not None:
            mime_type = cache_hit[0]
            file = cache_hit[1]
            return mime_type, open(file, mode='rb')

        headers = {'User-Agent': fake_useragent.UserAgent().chrome}
        req = requests.get(uri, headers=headers, stream=True)
        mime_type = req.headers['content-type']
        if mime_type_filter is not None and not mime_type_filter(mime_type):
            raise ImageSeedParseError(
                f'Unknown {mime_type_reject_noun} mimetype "{mime_type}" for situation in '
                f'parsed image seed "{uri_desc}". Expected: {mime_acceptable_desc}')

        cache_filename = generate_web_cache_filename()
        with open(cache_filename, mode='wb'):
            _write_to_file(req.content, cache_filename)

        WEB_FILE_CACHE[uri] = (mime_type, cache_filename)
        return mime_type, open(cache_filename, mode='rb')
    else:
        mime_type = mimetypes.guess_type(uri)[0]

        if mime_type is None and uri.endswith('.webp'):
            # webp missing from mimetypes library
            mime_type = "image/webp"

        if mime_type_filter is not None and not mime_type_filter(mime_type):
            raise ImageSeedParseError(
                f'Unknown {mime_type_reject_noun} mimetype "{mime_type}" for situation in '
                f'parsed image seed "{uri_desc}". Expected: {mime_acceptable_desc}')

    return mime_type, open(uri, 'rb')


def mime_type_is_animable_image(mime_type: str):
    return mime_type in {'image/gif', 'image/webp'}


def mime_type_is_static_image(mime_type: str):
    return mime_type in {'image/png', 'image/jpeg'}


def mime_type_is_video(mime_type: str):
    if mime_type is None:
        return False

    return mime_type.startswith('video')


class ImageSeedInfo:
    def __init__(self,
                 is_animation: bool,
                 fps: typing.Union[float, int],
                 duration: float):
        self.fps = fps
        self.duration = duration
        self.is_animation = is_animation


def get_image_seed_info(image_seed_path: str, frame_start: int, frame_end: int):
    with next(iterate_image_seed(image_seed_path, frame_start, frame_end)) as seed:
        return ImageSeedInfo(seed.is_animation_frame, seed.fps, seed.duration)


def get_control_image_info(path: str, frame_start: int, frame_end: int):
    with next(iterate_control_image(path, frame_start, frame_end)) as seed:
        return ImageSeedInfo(seed.is_animation_frame, seed.fps, seed.duration)


def _write_to_file(data, filepath):
    with open(filepath, 'wb') as mask_video_file:
        mask_video_file.write(data)
        mask_video_file.flush()
    return filepath


def create_and_exif_orient_pil_img(
        path_or_file: typing.Union[typing.BinaryIO, str],
        file_source: str,
        resize_resolution: typing.Union[typing.Tuple[int, int], None] = None):
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
    def __init__(self, objects: typing.Iterable[typing.ContextManager]):
        self.objects = objects

    def __enter__(self):
        for obj in self.objects:
            if obj is not None:
                obj.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        for obj in self.objects:
            if obj is not None:
                obj.__exit__(type, value, traceback)


def iterate_control_image(uri: str,
                          frame_start: int = 0,
                          frame_end: typing.Union[int, None] = None,
                          resize_resolution: typing.Union[typing.Tuple[int, int], None] = None,
                          preprocessor: _preprocessors.ImagePreprocessor = None):
    mime_acceptable_desc = 'image/png, image/jpeg, image/gif, image/webp, video/*'

    if isinstance(uri, ImageSeedParseResult):
        uri = uri.uri

    control_mime_type, control_data = fetch_image_data_stream(
        uri=uri,
        uri_desc=uri,
        mime_type_reject_noun='control image',
        mime_acceptable_desc=mime_acceptable_desc)

    manage_context = [control_data]

    if mime_type_is_animable_image(control_mime_type):
        control_reader = GifWebpReader(file=control_data,
                                       file_source=uri,
                                       resize_resolution=resize_resolution,
                                       preprocessor=preprocessor)
    elif mime_type_is_video(control_mime_type):
        control_reader = VideoReader(file=control_data,
                                     file_source=uri,
                                     resize_resolution=resize_resolution,
                                     preprocessor=preprocessor)
    elif mime_type_is_static_image(control_mime_type):
        control_image = create_and_exif_orient_pil_img(control_data, uri,
                                                       resize_resolution)
        control_reader = MockImageAnimationReader(img=control_image,
                                                  resize_resolution=resize_resolution,
                                                  preprocessor=preprocessor)

    else:
        raise ImageSeedParseError(f'Unknown control image mimetype {control_mime_type}')
    manage_context.insert(0, control_reader)

    with MultiContextManager(manage_context):
        if isinstance(control_reader, MockImageAnimationReader):
            yield ImageSeed(image=control_reader.__next__())
        else:
            yield from (ImageSeed(animation_frame) for animation_frame in
                        iterate_animation_frames(seed_reader=control_reader,
                                                 frame_start=frame_start,
                                                 frame_end=frame_end))


def _iterate_image_seed_x3(seed_reader: AnimationReader,
                           mask_reader: typing.Union[AnimationReader, None] = None,
                           control_reader: typing.Union[AnimationReader, None] = None,
                           frame_start: int = 0,
                           frame_end: typing.Union[int, None] = None):
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
                           frame_end: typing.Union[int, None] = None):
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


def iterate_image_seed(uri: str,
                       frame_start: int = 0,
                       frame_end: typing.Union[int, None] = None,
                       resize_resolution: typing.Union[typing.Tuple[int, int], None] = None,
                       seed_image_preprocessor: typing.Union[_preprocessors.ImagePreprocessor, None] = None,
                       mask_image_preprocessor: typing.Union[_preprocessors.ImagePreprocessor, None] = None,
                       control_image_preprocessor: typing.Union[_preprocessors.ImagePreprocessor, None] = None):
    if isinstance(uri, ImageSeedParseResult):
        parse_result = uri
    else:
        parse_result = parse_image_seed_uri(uri)

    mime_acceptable_desc = 'image/png, image/jpeg, image/gif, image/webp, video/*'

    seed_mime_type, seed_data = fetch_image_data_stream(
        uri=parse_result.uri,
        uri_desc=uri,
        mime_type_reject_noun='image seed',
        mime_acceptable_desc=mime_acceptable_desc)

    mask_mime_type, mask_data = None, None

    if parse_result.mask_uri is not None:
        mask_mime_type, mask_data = fetch_image_data_stream(
            uri=parse_result.mask_uri,
            uri_desc=uri,
            mime_type_reject_noun='mask image',
            mime_acceptable_desc=mime_acceptable_desc)

    control_mime_type, control_data = None, None
    if parse_result.control_uri is not None:
        control_mime_type, control_data = fetch_image_data_stream(
            uri=parse_result.control_uri,
            uri_desc=uri,
            mime_type_reject_noun='control image',
            mime_acceptable_desc=mime_acceptable_desc)

    if parse_result.resize_resolution is not None:
        resize_resolution = parse_result.resize_resolution

    manage_context = [seed_data, mask_data, control_data]

    if seed_data is not None:
        if mime_type_is_animable_image(seed_mime_type):
            seed_reader = GifWebpReader(file=seed_data,
                                        file_source=parse_result.uri,
                                        resize_resolution=resize_resolution,
                                        preprocessor=seed_image_preprocessor)
        elif mime_type_is_video(seed_mime_type):
            seed_reader = VideoReader(file=seed_data,
                                      file_source=parse_result.uri,
                                      resize_resolution=resize_resolution,
                                      preprocessor=seed_image_preprocessor)
        elif mime_type_is_static_image(seed_mime_type):
            seed_image = create_and_exif_orient_pil_img(seed_data, parse_result.uri, resize_resolution)
            seed_reader = MockImageAnimationReader(img=seed_image,
                                                   resize_resolution=resize_resolution,
                                                   preprocessor=seed_image_preprocessor)
        else:
            raise ImageSeedParseError(f'Unknown seed image mimetype {seed_mime_type}')
        manage_context.insert(0, seed_reader)
    else:
        raise ImageSeedParseError(f'Image seed not specified or irretrievable')

    mask_reader = None
    if mask_data is not None:
        if mime_type_is_animable_image(mask_mime_type):
            mask_reader = GifWebpReader(file=mask_data,
                                        file_source=parse_result.mask_uri,
                                        resize_resolution=resize_resolution,
                                        preprocessor=mask_image_preprocessor)
        elif mime_type_is_video(mask_mime_type):
            mask_reader = VideoReader(file=mask_data,
                                      file_source=parse_result.mask_uri,
                                      resize_resolution=resize_resolution,
                                      preprocessor=mask_image_preprocessor)
        elif mime_type_is_static_image(mask_mime_type):
            mask_image = create_and_exif_orient_pil_img(mask_data, parse_result.mask_uri, resize_resolution)
            mask_reader = MockImageAnimationReader(img=mask_image,
                                                   resize_resolution=resize_resolution,
                                                   preprocessor=mask_image_preprocessor)

        else:
            raise ImageSeedParseError(f'Unknown mask image mimetype {mask_mime_type}')
        manage_context.insert(0, mask_reader)

    control_reader = None
    if control_data is not None:
        if mime_type_is_animable_image(control_mime_type):
            control_reader = GifWebpReader(file=control_data,
                                           file_source=parse_result.control_uri,
                                           resize_resolution=resize_resolution,
                                           preprocessor=control_image_preprocessor)
        elif mime_type_is_video(control_mime_type):
            control_reader = VideoReader(file=control_data,
                                         file_source=parse_result.control_uri,
                                         resize_resolution=resize_resolution,
                                         preprocessor=control_image_preprocessor)
        elif mime_type_is_static_image(control_mime_type):
            control_image = create_and_exif_orient_pil_img(control_data, parse_result.control_uri,
                                                           resize_resolution)
            control_reader = MockImageAnimationReader(img=control_image,
                                                      resize_resolution=resize_resolution,
                                                      preprocessor=control_image_preprocessor)
        else:
            raise ImageSeedParseError(f'Unknown control image mimetype {control_mime_type}')
        manage_context.insert(0, control_reader)

    size_mismatch_check = [(parse_result.uri, 'Image seed', seed_reader),
                           (parse_result.mask_uri, 'Mask image', mask_reader),
                           (parse_result.control_uri, 'Control image', control_reader)]

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
