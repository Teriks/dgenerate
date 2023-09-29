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

import io
import mimetypes
import os
import tempfile

import PIL.Image
import PIL.ImageOps
import PIL.ImageSequence
import av
import requests
from fake_useragent import UserAgent

from .preprocessors import ImagePreprocessorMixin
from .textprocessing import ConceptPathParser, ConceptPathParseError
from . import messages


class AnimationFrame:
    def __init__(self, frame_index, total_frames, anim_fps, anim_frame_duration, image, mask_image=None,
                 control_image=None):
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


def _resize_image_calc(new_size, old_size):
    width = new_size[0]
    w_percent = (width / float(old_size[0]))
    hsize = int((float(old_size[1]) * float(w_percent)))

    return width - width % 8, hsize - hsize % 8


def _is_aligned_by_8(x, y):
    return x % 8 == 0 and y % 8 == 0


def _align_by_8(x, y):
    return x - x % 8, y - y % 8


def _copy_img(img):
    c = img.copy()

    if hasattr(img, 'filename'):
        c.filename = img.filename

    return c


def _resize_image(size, img):
    new_size = _resize_image_calc(size, img.size)

    if img.size == new_size:
        # probably less costly
        return _copy_img(img)

    r = img.resize(new_size, PIL.Image.LANCZOS)

    if hasattr(img, 'filename'):
        r.filename = img.filename

    return r


def _is_frame_in_slice(idx, frame_start, frame_end):
    return idx >= frame_start and (frame_end is None or idx <= frame_end)


def _total_frames_slice(total_frames, frame_start, frame_end):
    return min(total_frames, (frame_end + 1 if frame_end is not None else total_frames)) - frame_start


def _RGB(img):
    c = img.convert('RGB')
    if hasattr(img, 'filename'):
        c.filename = img.filename
    return c


class ImageSeedSizeMismatchError(Exception):
    pass


class AnimationReader:
    # interface
    def __init__(self, width, height, anim_fps, anim_frame_duration, total_frames, **kwargs):
        self._width = width
        self._height = height
        self._anim_fps = anim_fps
        self._anim_frame_duration = anim_frame_duration
        self._total_frames = total_frames

    @property
    def width(self):
        return self._width

    @property
    def size(self):
        return self._width, self._height

    @property
    def height(self):
        return self._height

    @property
    def anim_fps(self):
        return self._anim_fps

    @property
    def anim_frame_duration(self):
        return self._anim_frame_duration

    @property
    def total_frames(self):
        return self._total_frames

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def frame_slice_count(self, frame_start=0, frame_end=None):
        return _total_frames_slice(self.total_frames, frame_start, frame_end)


class VideoReader(ImagePreprocessorMixin, AnimationReader):
    def __init__(self, filename, file_source, resize_resolution=None, preprocessor=None):
        self._filename = filename
        self._file_source = file_source
        self._container = av.open(filename, 'r')
        self.resize_resolution = resize_resolution

        if self.resize_resolution is None:
            width = int(self._container.streams.video[0].width)
            height = int(self._container.streams.video[0].height)
            if not _is_aligned_by_8(width, height):
                width, height = _resize_image_calc(_align_by_8(width, height), (width, height))
                self.resize_resolution = (width, height)
        else:
            width, height = _resize_image_calc(self.resize_resolution,
                                               (int(self._container.streams.video[0].width),
                                                int(self._container.streams.video[0].height)))

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

        pre_processed = self.preprocess_pre_resize(self.resize_resolution, rgb_image)

        if pre_processed is not rgb_image:
            rgb_image.close()

        if self.resize_resolution is None:
            rgb_image = pre_processed
        else:
            rgb_image = _resize_image(self.resize_resolution, pre_processed)

        if rgb_image is not pre_processed:
            pre_processed.close()

        pre_processed = self.preprocess_post_resize(self.resize_resolution, rgb_image)
        if pre_processed is not rgb_image:
            rgb_image.close()

        return pre_processed


class GifWebpReader(ImagePreprocessorMixin, AnimationReader):
    def __init__(self, file, file_source, resize_resolution=None, preprocessor=None):
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
            if not _is_aligned_by_8(width, height):
                width, height = _resize_image_calc(_align_by_8(width, height), (width, height))
                self.resize_resolution = (width, height)
        else:
            width, height = _resize_image_calc(self.resize_resolution, self._img.size)

        super().__init__(width=width,
                         height=height,
                         anim_fps=anim_fps,
                         anim_frame_duration=anim_frame_duration,
                         total_frames=total_frames,
                         preprocessor=preprocessor)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._img.close()

    def frame_slice_count(self, frame_start=0, frame_end=None):
        return _total_frames_slice(self.total_frames, frame_start, frame_end)

    def __next__(self):
        with next(self._iter) as img:
            rgb_image = _RGB(img)
            rgb_image.filename = self._file_source

            pre_processed = self.preprocess_pre_resize(self.resize_resolution, rgb_image)

            if pre_processed is not rgb_image:
                rgb_image.close()

            if self.resize_resolution is None:
                rgb_image = pre_processed
            else:
                rgb_image = _resize_image(self.resize_resolution, pre_processed)

            if rgb_image is not pre_processed:
                pre_processed.close()

            pre_processed = self.preprocess_post_resize(self.resize_resolution, rgb_image)
            if pre_processed is not rgb_image:
                rgb_image.close()

            return pre_processed


class MockImageAnimationReader(ImagePreprocessorMixin, AnimationReader):
    def __init__(self, img, resize_resolution=None, image_repetitions=1, preprocessor=None):
        self._img = img
        self._idx = 0
        self.resize_resolution = resize_resolution

        total_frames = image_repetitions
        anim_fps = 30
        anim_frame_duration = 1000 / anim_fps

        if self.resize_resolution is None:
            width = self._img.size[0]
            height = self._img.size[1]
            if not _is_aligned_by_8(width, height):
                width, height = _resize_image_calc(_align_by_8(width, height), (width, height))
                self.resize_resolution = (width, height)
        else:
            width, height = _resize_image_calc(self.resize_resolution, self._img.size)

        super().__init__(width=width,
                         height=height,
                         anim_fps=anim_fps,
                         anim_frame_duration=anim_frame_duration,
                         total_frames=total_frames,
                         preprocessor=preprocessor)

    @property
    def total_frames(self):
        return self._total_frames

    @total_frames.setter
    def total_frames(self, cnt):
        self._total_frames = cnt

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._img.close()

    def frame_slice_count(self, frame_start=0, frame_end=None):
        return _total_frames_slice(self.total_frames, frame_start, frame_end)

    def __next__(self):
        if self._idx < self.total_frames:
            self._idx += 1

            pre_processed = self.preprocess_pre_resize(self.resize_resolution, self._img)

            if self.resize_resolution is None:
                copy_image = _copy_img(pre_processed)
            else:
                copy_image = _resize_image(self.resize_resolution, pre_processed)

            if pre_processed is not self._img:
                pre_processed.close()

            pre_processed = self.preprocess_post_resize(self.resize_resolution, copy_image)
            if pre_processed is not copy_image:
                copy_image.close()

            return pre_processed
        else:
            raise StopIteration


def create_animation_reader(file, file_source, resize_resolution=None, image_repetitions=1, preprocessor=None):
    if isinstance(file, io.IOBase):
        return GifWebpReader(file=file,
                             file_source=file_source,
                             resize_resolution=resize_resolution,
                             preprocessor=preprocessor)
    elif isinstance(file, str):
        return VideoReader(filename=file,
                           file_source=file_source,
                           resize_resolution=resize_resolution,
                           preprocessor=preprocessor)
    elif isinstance(file, PIL.Image.Image):
        return MockImageAnimationReader(img=file,
                                        resize_resolution=resize_resolution,
                                        image_repetitions=image_repetitions,
                                        preprocessor=preprocessor)
    else:
        raise ValueError(
            'File must be a filename indicating an encoded video on disk, '
            'or a file stream containing raw GIF / WebP data')


def _iterate_animation_frames_x2(seed_reader,
                                 right_reader,
                                 right_animation_frame_param_name,
                                 frame_start,
                                 frame_end):
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


def _iterate_animation_frames_x3(seed_reader,
                                 mask_reader,
                                 control_reader,
                                 frame_start,
                                 frame_end):
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


def iterate_animation_frames(seed_reader,
                             frame_start=0,
                             frame_end=None,
                             mask_reader=None,
                             control_reader=None):
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
    def __init__(self, image, mask_image=None, control_image=None):

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
        self.uri = None
        self.uri_is_local = False
        self.mask_uri = None
        self.mask_uri_is_local = False
        self.control_uri = None
        self.control_uri_is_local = False
        self.resize_resolution = None


def parse_image_seed_uri_legacy(url):
    parts = (x.strip() for x in url.split(';'))
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


def parse_image_seed_uri(url):
    parts = url.split(';')

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
        return parse_image_seed_uri_legacy(url)

    result = ImageSeedParseResult()

    seed_parser = ConceptPathParser('Image Seed', ['mask', 'control', 'resize'])

    try:
        parse_result = seed_parser.parse_concept_path(url)
    except ConceptPathParseError as e:
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


def image_seed_mime_type_filter(mime_type):
    return (mime_type_is_static_image(mime_type) or
            mime_type_is_video(mime_type) or
            mime_type_is_animable_image(mime_type))


def fetch_image_seed_data(uri,
                          uri_desc,
                          mime_type_filter=image_seed_mime_type_filter,
                          mime_type_reject_noun='input image',
                          mime_acceptable_desc=''):
    if uri.startswith('http://') or uri.startswith('https://'):
        headers = {'User-Agent': UserAgent().chrome}
        req = requests.get(uri, headers=headers)
        mime_type = req.headers['content-type']
        if mime_type_filter is not None and not mime_type_filter(mime_type):
            raise ImageSeedParseError(
                f'Unknown {mime_type_reject_noun} mimetype "{mime_type}" for situation in '
                f'parsed image seed "{uri_desc}". Expected: {mime_acceptable_desc}')
        data = req.content
    else:
        mime_type = mimetypes.guess_type(uri)[0]

        if mime_type is None and uri.endswith('.webp'):
            # webp missing from mimetypes library
            mime_type = "image/webp"

        if mime_type_filter is not None and not mime_type_filter(mime_type):
            raise ImageSeedParseError(
                f'Unknown {mime_type_reject_noun} mimetype "{mime_type}" for situation in '
                f'parsed image seed "{uri_desc}". Expected: {mime_acceptable_desc}')
        else:
            with open(uri, 'rb') as file:
                data = file.read()

    return mime_type, data


def mime_type_is_animable_image(mime_type):
    return mime_type in {'image/gif', 'image/webp'}


def mime_type_is_static_image(mime_type):
    return mime_type in {'image/png', 'image/jpeg'}


def mime_type_is_video(mime_type):
    if mime_type is None:
        return False

    return mime_type.startswith('video')


class ImageSeedInfo:
    def __init__(self, is_animation, fps, duration):
        self.fps = fps
        self.duration = duration
        self.is_animation = is_animation


def get_image_seed_info(image_seed, frame_start, frame_end):
    with next(iterate_image_seed(image_seed, frame_start, frame_end)) as seed:
        return ImageSeedInfo(seed.is_animation_frame, seed.fps, seed.duration)


def get_control_image_info(image_seed, frame_start, frame_end):
    with next(iterate_control_image(image_seed, frame_start, frame_end)) as seed:
        return ImageSeedInfo(seed.is_animation_frame, seed.fps, seed.duration)


def _write_to_file(data, filepath):
    with open(filepath, 'wb') as mask_video_file:
        mask_video_file.write(data)
        mask_video_file.flush()
    return filepath


def create_and_exif_orient_pil_img(path_or_data, file_source, resize_resolution=None):
    if isinstance(path_or_data, str):
        file = path_or_data
    else:
        file = io.BytesIO(path_or_data)

    if resize_resolution is None:
        with PIL.Image.open(file) as img, _RGB(img) as rgb_img:
            e_img = _exif_orient(rgb_img)
            e_img.filename = file_source
            if not _is_aligned_by_8(e_img.width, e_img.height):
                with e_img:
                    resized = _resize_image(_align_by_8(e_img.width, e_img.height), e_img)
                    return resized
            else:
                return e_img
    else:
        with PIL.Image.open(file) as img, _RGB(img) as rgb_img, _exif_orient(rgb_img) as o_img:
            o_img.filename = file_source
            resized = _resize_image(resize_resolution, o_img)
            return resized


class MultiContextManager:
    def __init__(self, objects):
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


def iterate_control_image(uri,
                          frame_start=0,
                          frame_end=None,
                          resize_resolution=None,
                          preprocessor=None):
    mime_acceptable_desc = 'image/png, image/jpeg, image/gif, image/webp, video/*'

    control_mime_type, control_data = fetch_image_seed_data(
        uri=uri,
        uri_desc=uri,
        mime_type_reject_noun='control image',
        mime_acceptable_desc=mime_acceptable_desc)

    manage_context = []

    if mime_type_is_animable_image(control_mime_type):
        control_reader = create_animation_reader(file=io.BytesIO(control_data),
                                                 file_source=uri,
                                                 resize_resolution=resize_resolution,
                                                 preprocessor=preprocessor)
        manage_context.append(control_reader)
    elif mime_type_is_video(control_mime_type):
        temp_dir = tempfile.TemporaryDirectory()
        video_file_path = _write_to_file(control_data, os.path.join(temp_dir.name, 'tmp_control_net'))
        control_reader = create_animation_reader(file=video_file_path,
                                                 file_source=uri,
                                                 resize_resolution=resize_resolution,
                                                 preprocessor=preprocessor)
        manage_context += [control_reader, temp_dir]
    elif mime_type_is_static_image(control_mime_type):
        control_image = create_and_exif_orient_pil_img(control_data, uri,
                                                       resize_resolution)
        control_reader = create_animation_reader(file=control_image,
                                                 file_source=uri,
                                                 resize_resolution=resize_resolution,
                                                 preprocessor=preprocessor)
        manage_context.append(control_reader)
    else:
        raise ImageSeedParseError(f'Unknown control image mimetype {control_mime_type}')

    if isinstance(control_reader, MockImageAnimationReader):
        yield ImageSeed(image=control_reader.__next__())
    else:
        yield from (ImageSeed(animation_frame) for animation_frame in
                    iterate_animation_frames(seed_reader=control_reader,
                                             frame_start=frame_start,
                                             frame_end=frame_end))


def _iterate_image_seed_x3(seed_reader,
                           mask_reader,
                           control_reader,
                           frame_start=0,
                           frame_end=None):
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


def _iterate_image_seed_x2(seed_reader,
                           right_reader,
                           right_image_seed_param_name,
                           right_reader_iterate_param_name,
                           frame_start=0,
                           frame_end=None):
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


def iterate_image_seed(uri,
                       frame_start=0,
                       frame_end=None,
                       resize_resolution=None,
                       seed_image_preprocessor=None,
                       mask_image_preprocessor=None,
                       control_image_preprocessor=None):
    parse_result = parse_image_seed_uri(uri)

    mime_acceptable_desc = 'image/png, image/jpeg, image/gif, image/webp, video/*'

    seed_mime_type, seed_data = fetch_image_seed_data(
        uri=parse_result.uri,
        uri_desc=uri,
        mime_type_reject_noun='image seed',
        mime_acceptable_desc=mime_acceptable_desc)

    mask_mime_type, mask_data = None, None

    if parse_result.mask_uri is not None:
        mask_mime_type, mask_data = fetch_image_seed_data(
            uri=parse_result.mask_uri,
            uri_desc=uri,
            mime_type_reject_noun='mask image',
            mime_acceptable_desc=mime_acceptable_desc)

    control_mime_type, control_data = None, None
    if parse_result.control_uri is not None:
        control_mime_type, control_data = fetch_image_seed_data(
            uri=parse_result.control_uri,
            uri_desc=uri,
            mime_type_reject_noun='control image',
            mime_acceptable_desc=mime_acceptable_desc)

    if parse_result.resize_resolution is not None:
        resize_resolution = parse_result.resize_resolution

    manage_context = []

    seed_reader = None
    if seed_data is not None:
        if mime_type_is_animable_image(seed_mime_type):
            seed_reader = create_animation_reader(file=io.BytesIO(seed_data),
                                                  file_source=parse_result.uri,
                                                  resize_resolution=resize_resolution,
                                                  preprocessor=seed_image_preprocessor)
            manage_context.append(seed_reader)
        elif mime_type_is_video(seed_mime_type):
            temp_dir = tempfile.TemporaryDirectory()
            video_file_path = _write_to_file(seed_data, os.path.join(temp_dir.name, 'tmp_vid'))
            seed_reader = create_animation_reader(file=video_file_path,
                                                  file_source=parse_result.uri,
                                                  resize_resolution=resize_resolution,
                                                  preprocessor=seed_image_preprocessor)
            manage_context += [seed_reader, temp_dir]
        elif mime_type_is_static_image(seed_mime_type):
            seed_image = create_and_exif_orient_pil_img(seed_data, parse_result.uri, resize_resolution)
            seed_reader = create_animation_reader(file=seed_image,
                                                  file_source=parse_result.uri,
                                                  resize_resolution=resize_resolution,
                                                  preprocessor=seed_image_preprocessor)
            manage_context.append(seed_reader)
        else:
            raise ImageSeedParseError(f'Unknown seed image mimetype {seed_mime_type}')
    else:
        raise ImageSeedParseError(f'Image seed not specified or irretrievable')

    mask_reader = None
    if mask_data is not None:
        if mime_type_is_animable_image(mask_mime_type):
            mask_reader = create_animation_reader(file=io.BytesIO(mask_data),
                                                  file_source=parse_result.mask_uri,
                                                  resize_resolution=resize_resolution,
                                                  preprocessor=mask_image_preprocessor)
            manage_context.append(mask_reader)
        elif mime_type_is_video(mask_mime_type):
            temp_dir = tempfile.TemporaryDirectory()
            video_file_path = _write_to_file(mask_data, os.path.join(temp_dir.name, 'tmp_mask'))
            mask_reader = create_animation_reader(file=video_file_path,
                                                  file_source=parse_result.mask_uri,
                                                  resize_resolution=resize_resolution,
                                                  preprocessor=mask_image_preprocessor)
            manage_context += [mask_reader, temp_dir]
        elif mime_type_is_static_image(mask_mime_type):
            mask_image = create_and_exif_orient_pil_img(mask_data, parse_result.mask_uri, resize_resolution)
            mask_reader = create_animation_reader(file=mask_image,
                                                  file_source=parse_result.mask_uri,
                                                  resize_resolution=resize_resolution,
                                                  preprocessor=mask_image_preprocessor)
            manage_context.append(mask_reader)
        else:
            raise ImageSeedParseError(f'Unknown mask image mimetype {mask_mime_type}')

    control_reader = None
    if control_data is not None:
        if mime_type_is_animable_image(control_mime_type):
            control_reader = create_animation_reader(file=io.BytesIO(control_data),
                                                     file_source=parse_result.control_uri,
                                                     resize_resolution=resize_resolution,
                                                     preprocessor=control_image_preprocessor)
            manage_context.append(control_reader)
        elif mime_type_is_video(control_mime_type):
            temp_dir = tempfile.TemporaryDirectory()
            video_file_path = _write_to_file(control_data, os.path.join(temp_dir.name, 'tmp_control_net'))
            control_reader = create_animation_reader(file=video_file_path,
                                                     file_source=parse_result.control_uri,
                                                     resize_resolution=resize_resolution,
                                                     preprocessor=control_image_preprocessor)
            manage_context += [control_reader, temp_dir]
        elif mime_type_is_static_image(control_mime_type):
            control_image = create_and_exif_orient_pil_img(control_data, parse_result.control_uri,
                                                           resize_resolution)
            control_reader = create_animation_reader(file=control_image,
                                                     file_source=parse_result.control_uri,
                                                     resize_resolution=resize_resolution,
                                                     preprocessor=control_image_preprocessor)
            manage_context.append(control_reader)
        else:
            raise ImageSeedParseError(f'Unknown control image mimetype {control_mime_type}')

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
