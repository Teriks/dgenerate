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
    return img.resize((width - width % 8, hsize - hsize % 8), PIL.Image.LANCZOS)


def _is_frame_in_slice(idx, frame_start, frame_end):
    return idx >= frame_start and (frame_end is None or idx <= frame_end)


def _total_frames_slice(total_frames, frame_start, frame_end):
    return min(total_frames, (frame_end + 1 if frame_end is not None else total_frames)) - frame_start


def iterate_video_frames(filename, frame_start=0, frame_end=None, resize_resolution=None, mask_filename=None):
    with av.open(filename, 'r') as container:

        anim_fps = int(container.streams.video[0].average_rate)
        anim_rate = 1000 // anim_fps
        total_frames = container.streams.video[0].frames

        if total_frames <= 0:
            # webm decode bug?
            total_frames = sum(1 for i in container.decode(video=0))
            container.seek(0, whence='time')

        total_frames = _total_frames_slice(total_frames, frame_start, frame_end)

        out_frame_idx = 0
        for in_frame_idx, frame in enumerate(container.decode(video=0)):
            if _is_frame_in_slice(in_frame_idx, frame_start, frame_end):
                if resize_resolution is not None:
                    with frame.to_image() as pframe:
                        yield AnimationFrame(out_frame_idx, total_frames, anim_fps, anim_rate,
                                             _resize_image(resize_resolution, pframe))
                else:
                    yield AnimationFrame(out_frame_idx, total_frames, anim_fps, anim_rate, frame.to_image())
                out_frame_idx += 1


def iterate_gif_webp_frames(file, frame_start=0, frame_end=None, resize_resolution=None, mask_file=None):
    with PIL.Image.open(file) as img:
        duration = img.info['duration']
        anim_fps = 1000 // duration
        anim_rate = duration

        total_frames = _total_frames_slice(img.n_frames, frame_start, frame_end)

        out_frame_idx = 0
        for in_frame_idx, frame in enumerate(PIL.ImageSequence.Iterator(img)):
            if _is_frame_in_slice(in_frame_idx, frame_start, frame_end):
                if resize_resolution is not None:
                    with _resize_image(resize_resolution, frame) as r_frame:
                        yield AnimationFrame(out_frame_idx, total_frames, anim_fps, anim_rate, r_frame.convert('RGB'))
                else:
                    yield AnimationFrame(out_frame_idx, total_frames, anim_fps, anim_rate, frame.convert('RGB'))
                out_frame_idx += 1


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


def _fetch_data(uri, local=False, mime_type_filter=None, mime_type_reject_msg='input image', mime_acceptable_desc=''):
    if local:
        mime_type = mimetypes.guess_type(uri)[0]
        if mime_type_filter is not None and not mime_type_filter(mime_type):
            raise ImageSeedParseError(
                f'Unknown {mime_type_reject_msg} mimetype "{mime_type}". Expected: {mime_acceptable_desc}')
        else:
            with open(uri, 'rb') as file:
                data = file.read()
    else:
        headers = {'User-Agent': UserAgent().chrome}
        req = requests.get(uri, headers=headers)
        mime_type = req.headers['content-type']
        if mime_type_filter is not None and not mime_type_filter(mime_type):
            raise ImageSeedParseError(
                f'Unknown {mime_type_reject_msg} mimetype "{mime_type}". Expected: {mime_acceptable_desc}')
        data = req.content

    return mime_type, data


def iterate_image_seed(uri, frame_start=0, frame_end=None, resize_resolution=None):
    parse_result = parse_image_seed_uri(uri)

    seed_mime_type, seed_data = _fetch_data(
        uri=parse_result.uri,
        local=parse_result.uri_is_local,
        mime_type_reject_msg='image seed',
        mime_acceptable_desc='image/png, image/jpeg, image/gif, image/webp, video/*',
        mime_type_filter=lambda mime_type:
        mime_type == 'image/gif' or
        mime_type == 'image/webp' or
        mime_type == 'image/png' or
        mime_type == 'image/jpeg' or
        mime_type.startswith('video'))

    mask_mime_type, mask_data = None, None

    if parse_result.mask_uri is not None:
        mask_mime_type, mask_data = _fetch_data(
            uri=parse_result.mask_uri,
            local=parse_result.mask_uri_is_local,
            mime_type_reject_msg='mask image',
            mime_acceptable_desc='image/png or image/jpeg',
            mime_type_filter=lambda mime_type:
            mime_type == 'image/png' or
            mime_type == 'image/jpeg')

    if parse_result.resize_resolution is not None:
        resize_resolution = parse_result.resize_resolution

    if seed_mime_type == 'image/gif' or seed_mime_type == 'image/webp':
        yield from (ImageSeed(animation_frame) for animation_frame in
                    iterate_gif_webp_frames(io.BytesIO(seed_data), frame_start, frame_end, resize_resolution))
    elif seed_mime_type.startswith('video'):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'tmp')
            with open(file_path, 'wb') as video_file:
                video_file.write(seed_data)
                video_file.flush()
            yield from (ImageSeed(animation_frame) for animation_frame in
                        iterate_video_frames(file_path, frame_start, frame_end, resize_resolution))
    elif seed_mime_type.startswith('image'):
        with PIL.Image.open(io.BytesIO(seed_data)) as img, \
                img.convert('RGB') as rgb_img, \
                _exif_orient(rgb_img) as o_img:

            if mask_data is not None:
                with PIL.Image.open(io.BytesIO(mask_data)) as mask_img, \
                        mask_img.convert('RGB') as mask_rgb_img, \
                        _exif_orient(mask_rgb_img) as mask_o_img:
                    if resize_resolution is not None:
                        yield ImageSeed(_resize_image(resize_resolution, o_img),
                                        mask_image=_resize_image(resize_resolution, mask_o_img))
                    else:
                        yield ImageSeed(o_img, mask_image=mask_o_img)
            else:
                if resize_resolution is not None:
                    yield ImageSeed(_resize_image(resize_resolution, o_img))
                else:
                    yield ImageSeed(o_img)
    else:
        raise Exception(f'Unknown seed image mimetype {seed_mime_type}')
