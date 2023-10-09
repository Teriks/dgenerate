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
import typing

import PIL.Image
import av


class AnimationWriter:
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


class VideoWriter(AnimationWriter):
    def __init__(self, filename, fps: typing.Union[float, int]):
        super().__init__()
        self.filename = filename
        self.fps = round(fps)
        self._container = None
        self._stream = None

    def end(self, new_file=None):
        self._cleanup()

        if new_file is not None:
            self.filename = new_file

    def _cleanup(self):
        if self._container is not None:
            for packet in self._stream.encode():
                self._container.mux(packet)
            self._container.close()
            self._container = None
            self._stream = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()

    def write(self, img: PIL.Image.Image):
        if self._container is None:
            self._container = av.open(self.filename, 'w')
            self._stream = self._container.add_stream("h264", rate=self.fps)
            self._stream.codec_context.bit_rate = 8000000
            self._stream.width = img.width
            self._stream.height = img.height
            self._stream.pix_fmt = "yuv420p"

        for packet in self._stream.encode(av.VideoFrame.from_image(img)):
            self._container.mux(packet)


class GifWebpWriter(AnimationWriter):
    def __init__(self, filename: str, duration: float):
        super().__init__()
        self.collected_frames = []
        self.filename = filename
        self.duration = duration

    def _cleanup(self):
        for i in self.collected_frames:
            i.close()
        self.collected_frames.clear()

    def end(self, new_file: str = None):

        if self.collected_frames:
            self.collected_frames[0].save(self.filename, save_all=True, append_images=self.collected_frames[1:],
                                          optimize=False, duration=self.duration, loop=0)
            self._cleanup()

        if new_file:
            self.filename = new_file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()

    def write(self, img: PIL.Image.Image):
        self.collected_frames.append(img.copy())


def supported_animation_writer_formats():
    return ['mp4', 'gif', 'webp']


def create_animation_writer(animation_format: str, out_filename: str, fps: typing.Union[float, int]):
    return VideoWriter(out_filename, fps) if animation_format == 'mp4' else GifWebpWriter(out_filename, 1000 / fps)
