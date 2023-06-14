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

import av
import io
import os
import requests
import tempfile
import mimetypes
import PIL.Image
import PIL.ImageOps
import PIL.ImageSequence
from fake_useragent import UserAgent


class AnimationFrame:
    def __init__(self, frame_index, total_frames, anim_fps, anim_frame_duration, img):
        self.frame_index = frame_index
        self.total_frames = total_frames
        self.fps = anim_fps
        self.duration = anim_frame_duration
        self.image = img

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.image.close()

    def close(self):
        self.image.close()


def _resize_image(size, img):
    width = size[0]
    w_percent = (width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(w_percent)))
    return img.resize((width - width % 8, hsize - hsize % 8), PIL.Image.LANCZOS)


def _is_frame_in_slice(idx, frame_start, frame_end):
    return idx >= frame_start and (frame_end is None or idx <= frame_end)


def _total_frames_slice(total_frames, frame_start, frame_end):
    return min(total_frames, (frame_end + 1 if frame_end is not None else total_frames)) - frame_start


def iterate_video_frames(filename, frame_start=0, frame_end=None, resize_resolution=None):
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


def iterate_gif_webp_frames(file, frame_start=0, frame_end=None, resize_resolution=None):
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
    def __init__(self, img):
        self.is_animation_frame = isinstance(img, AnimationFrame)
        self.frame_index = None
        self.total_frames = None
        self.fps = None
        self.duration = None

        if self.is_animation_frame:
            self.image = img.image
            if img.total_frames > 1:
                self.frame_index = img.frame_index
                self.total_frames = img.total_frames
                self.fps = img.fps
                self.duration = img.duration
            else:
                self.is_animation_frame = False
        else:
            self.image = img

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.image.close()

    def close(self):
        self.image.close()


def _exif_orient(image):
    exif = image.getexif()
    for k in exif.keys():
        if k != 0x0112:
            exif[k] = None
            del exif[k]
    new_exif = exif.tobytes()
    image.info["exif"] = new_exif
    return PIL.ImageOps.exif_transpose(image)


def iterate_image_seed(url, frame_start=0, frame_end=None, resize_resolution=None):
    if url.startswith('http://') or url.startswith('https://'):
        headers = {'User-Agent': UserAgent().chrome}
        req = requests.get(url, headers=headers)
        mime_type = req.headers['content-type']
        data = req.content
    else:
        mime_type = mimetypes.guess_type(url)[0]
        if mime_type == 'image/gif' or mime_type == 'image/webp' or mime_type.startswith(
                'video') or mime_type.startswith('image'):
            with open(url, 'rb') as file:
                data = file.read()
        else:
            raise Exception(f'Unknown seed image mimetype {mime_type}')

    if mime_type == 'image/gif' or mime_type == 'image/webp':
        yield from (ImageSeed(animation_frame) for animation_frame in
                    iterate_gif_webp_frames(io.BytesIO(data), frame_start, frame_end, resize_resolution))
    elif mime_type.startswith('video'):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'tmp')
            with open(file_path, 'wb') as video_file:
                video_file.write(data)
                video_file.flush()
            yield from (ImageSeed(animation_frame) for animation_frame in
                        iterate_video_frames(file_path, frame_start, frame_end, resize_resolution))
    elif mime_type.startswith('image'):
        with PIL.Image.open(io.BytesIO(data)) as img, \
                img.convert('RGB') as rgb_img, \
                _exif_orient(rgb_img) as o_img:
            if resize_resolution is not None:
                yield ImageSeed(_resize_image(resize_resolution, o_img))
            else:
                yield ImageSeed(o_img)
    else:
        raise Exception(f'Unknown seed image mimetype {mime_type}')
