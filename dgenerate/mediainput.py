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


class AnimationFrame:
    def __init__(self, frame_index, total_frames, anim_fps, anim_frame_duration, image, mask_image=None):
        self.frame_index = frame_index
        self.total_frames = total_frames
        self.fps = anim_fps
        self.duration = anim_frame_duration
        self.image = image
        self.mask_image = mask_image

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    def close(self):
        self.image.close()
        if self.mask_image is not None:
            self.mask_image.close()


def _resize_image(size, img):
    width = size[0]
    w_percent = (width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(w_percent)))

    size = (width - width % 8, hsize - hsize % 8)

    if img.size == size:
        # probably less costly
        return img.copy()

    return img.resize(size, PIL.Image.LANCZOS)


def _is_frame_in_slice(idx, frame_start, frame_end):
    return idx >= frame_start and (frame_end is None or idx <= frame_end)


def _total_frames_slice(total_frames, frame_start, frame_end):
    return min(total_frames, (frame_end + 1 if frame_end is not None else total_frames)) - frame_start


def _RGB(img):
    return img.convert('RGB')


class MaskImageSizeMismatchError(Exception):
    def __init__(self, image_size, mask_size):
        super().__init__(
            f'Image seed encountered of size {image_size} using inpaint mask image of size '
            f'{mask_size}, their sizes must be equal.')


class AnimationReader:
    # interface
    def __init__(self, width, height, anim_fps, anim_frame_duration, total_frames):
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


class VideoReader(AnimationReader):
    def __init__(self, filename, resize_resolution):
        self._filename = filename
        self._container = av.open(filename, 'r')
        self.resize_resolution = resize_resolution

        width = int(self._container.streams.video[0].width)
        height = int(self._container.streams.video[0].height)
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
                         total_frames=total_frames)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._container.close()

    def frame_slice_count(self, frame_start=0, frame_end=None):
        return _total_frames_slice(self.total_frames, frame_start, frame_end)

    def __next__(self):
        if self.resize_resolution is None:
            return next(self._iter).to_image()
        else:
            with next(self._iter).to_image() as img:
                with _resize_image(self.resize_resolution, img) as r_img:
                    return _RGB(r_img)


class GifWebpReader(AnimationReader):
    def __init__(self, file, resize_resolution):
        self._img = PIL.Image.open(file)
        self._iter = PIL.ImageSequence.Iterator(self._img)
        self.resize_resolution = resize_resolution

        total_frames = self._img.n_frames
        anim_frame_duration = self._img.info['duration']
        anim_fps = 1000 / anim_frame_duration
        width = self._img.size[0]
        height = self._img.size[1]

        super().__init__(width=width,
                         height=height,
                         anim_fps=anim_fps,
                         anim_frame_duration=anim_frame_duration,
                         total_frames=total_frames)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._img.close()

    def frame_slice_count(self, frame_start=0, frame_end=None):
        return _total_frames_slice(self.total_frames, frame_start, frame_end)

    def __next__(self):
        with next(self._iter) as img:
            if self.resize_resolution is None:
                return _RGB(img)
            else:
                with _resize_image(self.resize_resolution, img) as r_img:
                    return _RGB(r_img)


class MockImageAnimationReader(AnimationReader):
    def __init__(self, img, resize_resolution, image_repetitions):
        self._img = img
        self._idx = 0
        self.resize_resolution = resize_resolution

        total_frames = image_repetitions
        anim_fps = 30
        anim_frame_duration = 1000 // anim_fps
        width = self._img.size[0]
        height = self._img.size[1]

        super().__init__(width=width,
                         height=height,
                         anim_fps=anim_fps,
                         anim_frame_duration=anim_frame_duration,
                         total_frames=total_frames)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._img.close()

    def frame_slice_count(self, frame_start=0, frame_end=None):
        return _total_frames_slice(self.total_frames, frame_start, frame_end)

    def __next__(self):
        if self._idx < self.total_frames:
            self._idx += 1
            if self.resize_resolution is None:
                return self._img.copy()
            else:
                return _resize_image(self.resize_resolution, self._img)
        else:
            raise StopIteration


def create_animation_reader(file, resize_resolution=None, image_repetitions=1):
    if isinstance(file, io.IOBase):
        return GifWebpReader(file, resize_resolution)
    elif isinstance(file, str):
        return VideoReader(file, resize_resolution)
    elif isinstance(file, PIL.Image.Image):
        return MockImageAnimationReader(file, resize_resolution, image_repetitions)
    else:
        raise ValueError(
            'File must be a filename indicating an encoded video on disk, '
            'or a file stream containing raw GIF / WebP data')


def iterate_animation_frames(animation_reader, frame_start=0, frame_end=None, inpaint_mask=None):
    total_frames = animation_reader.frame_slice_count(frame_start, frame_end)
    out_frame_idx = 0
    in_slice = None

    mask_is_image = isinstance(inpaint_mask, PIL.Image.Image)

    if inpaint_mask is None or mask_is_image:

        for in_frame_idx, frame in enumerate(animation_reader):
            if _is_frame_in_slice(in_frame_idx, frame_start, frame_end):
                if in_slice is None:
                    in_slice = True
                yield AnimationFrame(frame_index=out_frame_idx,
                                     total_frames=total_frames,
                                     anim_fps=animation_reader.anim_fps,
                                     anim_frame_duration=animation_reader.anim_frame_duration,
                                     image=frame,
                                     mask_image=inpaint_mask.copy() if mask_is_image else None)
                out_frame_idx += 1
            elif in_slice:
                break
    else:
        mask_total_frames = inpaint_mask.frame_slice_count(frame_start, frame_end)

        # Account for videos possibly having a differing number of frames
        total_frames = min(total_frames, mask_total_frames)

        for in_frame_idx, frame in enumerate(zip(animation_reader, inpaint_mask)):
            mask = frame[1]
            frame = frame[0]

            if _is_frame_in_slice(in_frame_idx, frame_start, frame_end):
                if in_slice is None:
                    in_slice = True
                yield AnimationFrame(frame_index=out_frame_idx,
                                     total_frames=total_frames,
                                     anim_fps=animation_reader.anim_fps,
                                     anim_frame_duration=animation_reader.anim_frame_duration,
                                     image=frame,
                                     mask_image=mask)
                out_frame_idx += 1
            elif in_slice:
                break


class ImageSeed:
    def __init__(self, image, mask_image=None):
        self.is_animation_frame = isinstance(image, AnimationFrame)
        self.frame_index = None
        self.total_frames = None
        self.fps = None
        self.duration = None
        self.image = None
        self.mask_image = None

        if self.is_animation_frame:
            self.image = image.image
            self.mask_image = image.mask_image
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
        self.resize_resolution = None


def parse_image_seed_uri(url):
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


def _fetch_image_seed_data(uri, uri_desc, local=False, mime_type_filter=None, mime_type_reject_noun='input image',
                           mime_acceptable_desc=''):
    if local:
        mime_type = mimetypes.guess_type(uri)[0]
        if mime_type_filter is not None and not mime_type_filter(mime_type):
            raise ImageSeedParseError(
                f'Unknown {mime_type_reject_noun} mimetype "{mime_type}" for situation in '
                f'parsed image seed "{uri_desc}". Expected: {mime_acceptable_desc}')
        else:
            with open(uri, 'rb') as file:
                data = file.read()
    else:
        headers = {'User-Agent': UserAgent().chrome}
        req = requests.get(uri, headers=headers)
        mime_type = req.headers['content-type']
        if mime_type_filter is not None and not mime_type_filter(mime_type):
            raise ImageSeedParseError(
                f'Unknown {mime_type_reject_noun} mimetype "{mime_type}" for situation in '
                f'parsed image seed "{uri_desc}". Expected: {mime_acceptable_desc}')
        data = req.content

    return mime_type, data


def _mime_type_is_animable_image(mime_type):
    return mime_type in {'image/gif', 'image/webp'}


def _mime_type_is_static_image(mime_type):
    return mime_type in {'image/png', 'image/jpeg'}


def _mime_type_is_video(mime_type):
    return mime_type.startswith('video')


class ImageSeedInfo:
    def __init__(self, is_animation, fps, duration):
        self.fps = fps
        self.duration = duration
        self.is_animation = is_animation


def get_image_seed_info(image_seed, frame_start, frame_end):
    with next(iterate_image_seed(image_seed, frame_start, frame_end)) as seed:
        return ImageSeedInfo(seed.is_animation_frame, seed.fps, seed.duration)


def _write_to_file(data, filepath):
    with open(filepath, 'wb') as mask_video_file:
        mask_video_file.write(data)
        mask_video_file.flush()
    return filepath


def iterate_image_seed(uri, frame_start=0, frame_end=None, resize_resolution=None):
    parse_result = parse_image_seed_uri(uri)

    mime_acceptable_desc = 'image/png, image/jpeg, image/gif, image/webp, video/*'

    def mime_type_filter(mime_type):
        return (_mime_type_is_static_image(mime_type) or
                _mime_type_is_video(mime_type) or
                _mime_type_is_animable_image(mime_type))

    seed_mime_type, seed_data = _fetch_image_seed_data(
        uri=parse_result.uri,
        uri_desc=uri,
        local=parse_result.uri_is_local,
        mime_type_reject_noun='image seed',
        mime_acceptable_desc=mime_acceptable_desc,
        mime_type_filter=mime_type_filter)

    mask_mime_type, mask_data = None, None

    if parse_result.mask_uri is not None:
        mask_mime_type, mask_data = _fetch_image_seed_data(
            uri=parse_result.mask_uri,
            uri_desc=uri,
            local=parse_result.mask_uri_is_local,
            mime_type_reject_noun='mask image',
            mime_acceptable_desc=mime_acceptable_desc,
            mime_type_filter=mime_type_filter)

    if parse_result.resize_resolution is not None:
        resize_resolution = parse_result.resize_resolution

    if _mime_type_is_animable_image(seed_mime_type):

        if mask_data is None:
            with create_animation_reader(io.BytesIO(seed_data), resize_resolution) as animation_reader:
                yield from (ImageSeed(animation_frame) for animation_frame in
                            iterate_animation_frames(animation_reader=animation_reader,
                                                     frame_start=frame_start,
                                                     frame_end=frame_end))

        else:
            if _mime_type_is_static_image(mask_mime_type):
                with PIL.Image.open(io.BytesIO(mask_data)) as mask_image, \
                        create_animation_reader(io.BytesIO(seed_data), resize_resolution) as animation_reader:
                    if animation_reader.size != mask_image.size:
                        raise MaskImageSizeMismatchError(animation_reader.size, mask_image.size)
                    yield from (ImageSeed(animation_frame) for animation_frame in
                                iterate_animation_frames(animation_reader=animation_reader,
                                                         frame_start=frame_start,
                                                         frame_end=frame_end,
                                                         inpaint_mask=mask_image))

            elif _mime_type_is_video(mask_mime_type):
                with tempfile.TemporaryDirectory() as temp_dir:
                    mask_video_file_path = _write_to_file(mask_data, os.path.join(temp_dir, 'tmp_mask'))

                    with create_animation_reader(io.BytesIO(seed_data), resize_resolution) as animation_reader, \
                            create_animation_reader(mask_video_file_path, resize_resolution) as mask_animation_reader:
                        if animation_reader.size != mask_animation_reader.size:
                            raise MaskImageSizeMismatchError(animation_reader.size, mask_animation_reader.size)
                        yield from (ImageSeed(animation_frame) for animation_frame in
                                    iterate_animation_frames(animation_reader=animation_reader,
                                                             frame_start=frame_start,
                                                             frame_end=frame_end,
                                                             inpaint_mask=mask_animation_reader))

            elif _mime_type_is_animable_image(mask_mime_type):
                with create_animation_reader(io.BytesIO(seed_data), resize_resolution) as animation_reader, \
                        create_animation_reader(io.BytesIO(mask_data), resize_resolution) as mask_animation_reader:
                    if animation_reader.size != mask_animation_reader.size:
                        raise MaskImageSizeMismatchError(animation_reader.size, mask_animation_reader.size)
                    yield from (ImageSeed(animation_frame) for animation_frame in
                                iterate_animation_frames(animation_reader=animation_reader,
                                                         frame_start=frame_start,
                                                         frame_end=frame_end,
                                                         inpaint_mask=mask_animation_reader))
            else:
                raise ImageSeedParseError(
                    'Unknown mimetype combination for gif/webp seed with mask.')

    elif _mime_type_is_video(seed_mime_type):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_file_path = _write_to_file(seed_data, os.path.join(temp_dir, 'tmp'))

            if mask_data is None:
                with create_animation_reader(video_file_path, resize_resolution) as animation_reader:
                    yield from (ImageSeed(animation_frame) for animation_frame in
                                iterate_animation_frames(animation_reader=animation_reader,
                                                         frame_start=frame_start,
                                                         frame_end=frame_end))
            else:
                if _mime_type_is_static_image(mask_mime_type):
                    with PIL.Image.open(io.BytesIO(mask_data)) as mask_image, \
                            create_animation_reader(video_file_path, resize_resolution) as animation_reader:
                        if animation_reader.size != mask_image.size:
                            raise MaskImageSizeMismatchError(animation_reader.size, mask_image.size)
                        yield from (ImageSeed(animation_frame) for animation_frame in
                                    iterate_animation_frames(animation_reader=animation_reader,
                                                             frame_start=frame_start,
                                                             frame_end=frame_end,
                                                             inpaint_mask=mask_image))

                elif _mime_type_is_video(mask_mime_type):
                    mask_video_file_path = _write_to_file(mask_data, os.path.join(temp_dir, 'tmp_mask'))

                    with create_animation_reader(video_file_path, resize_resolution) as animation_reader, \
                            create_animation_reader(mask_video_file_path, resize_resolution) as mask_animation_reader:
                        if animation_reader.size != mask_animation_reader.size:
                            raise MaskImageSizeMismatchError(animation_reader.size, mask_animation_reader.size)
                        yield from (ImageSeed(animation_frame) for animation_frame in
                                    iterate_animation_frames(animation_reader=animation_reader,
                                                             frame_start=frame_start,
                                                             frame_end=frame_end,
                                                             inpaint_mask=mask_animation_reader))

                elif _mime_type_is_animable_image(mask_mime_type):
                    with create_animation_reader(video_file_path, resize_resolution) as animation_reader, \
                            create_animation_reader(io.BytesIO(mask_data), resize_resolution) as mask_animation_reader:
                        if animation_reader.size != mask_animation_reader.size:
                            raise MaskImageSizeMismatchError(animation_reader.size, mask_animation_reader.size)
                        yield from (ImageSeed(animation_frame) for animation_frame in
                                    iterate_animation_frames(animation_reader=animation_reader,
                                                             frame_start=frame_start,
                                                             frame_end=frame_end,
                                                             inpaint_mask=mask_animation_reader))
                else:
                    raise ImageSeedParseError(
                        'Unknown mimetype combination for video seed with mask.')

    elif _mime_type_is_static_image(seed_mime_type):
        with PIL.Image.open(io.BytesIO(seed_data)) as img, \
                _RGB(img) as rgb_img, \
                _exif_orient(rgb_img) as o_img:

            if mask_data is None:
                if resize_resolution is not None:
                    yield ImageSeed(image=_resize_image(resize_resolution, o_img))
                else:
                    yield ImageSeed(image=o_img)

            else:
                if _mime_type_is_static_image(mask_mime_type):
                    with PIL.Image.open(io.BytesIO(mask_data)) as mask_img:
                        if o_img.size != mask_img.size:
                            raise MaskImageSizeMismatchError(o_img.size, mask_img.size)

                        with _RGB(mask_img) as mask_rgb_img, _exif_orient(mask_rgb_img) as mask_o_img:
                            if resize_resolution is not None:
                                yield ImageSeed(image=_resize_image(resize_resolution, o_img),
                                                mask_image=_resize_image(resize_resolution, mask_o_img))
                            else:
                                yield ImageSeed(image=o_img, mask_image=mask_o_img)

                elif _mime_type_is_video(mask_mime_type):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        mask_video_file_path = _write_to_file(mask_data, os.path.join(temp_dir, 'tmp_mask'))
                        print(resize_resolution)
                        with create_animation_reader(mask_video_file_path, resize_resolution) as mask_animation_reader, \
                                create_animation_reader(img, resize_resolution,
                                                        image_repetitions=mask_animation_reader.total_frames) as animation_reader:
                            if animation_reader.size != mask_animation_reader.size:
                                raise MaskImageSizeMismatchError(animation_reader.size, mask_animation_reader.size)
                            yield from (ImageSeed(animation_frame) for animation_frame in
                                        iterate_animation_frames(animation_reader=animation_reader,
                                                                 frame_start=frame_start,
                                                                 frame_end=frame_end,
                                                                 inpaint_mask=mask_animation_reader))

                elif _mime_type_is_animable_image(mask_mime_type):
                    with create_animation_reader(io.BytesIO(mask_data), resize_resolution) as mask_animation_reader, \
                            create_animation_reader(img, resize_resolution,
                                                    image_repetitions=mask_animation_reader.total_frames) as animation_reader:
                        if animation_reader.size != mask_animation_reader.size:
                            raise MaskImageSizeMismatchError(animation_reader.size, mask_animation_reader.size)
                        yield from (ImageSeed(animation_frame) for animation_frame in
                                    iterate_animation_frames(animation_reader=animation_reader,
                                                             frame_start=frame_start,
                                                             frame_end=frame_end,
                                                             inpaint_mask=mask_animation_reader))
                else:
                    raise ImageSeedParseError(
                        'Unknown mimetype combination for static image with mask.')
    else:
        raise ImageSeedParseError(f'Unknown seed image mimetype {seed_mime_type}')
