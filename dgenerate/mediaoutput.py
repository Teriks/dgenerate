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
import os
import typing

import PIL.Image
import av

import dgenerate.filelock as _filelock


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


class VideoWriter(AnimationWriter):
    """
    Animation writer for MP4 h264 format video
    """

    def __init__(self, filename, fps: typing.Union[float, int]):
        """
        :param filename: Filename to write to.
        :param fps: Frame rate, in frames per second.
        """

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


class AnimatedImageWriter(AnimationWriter):
    """
    Animation writer for animated images such as GIFs and webp
    """

    def __init__(self, filename: str, duration: float):
        """
        :param filename: Filename to write to.
        :param duration: Frame duration, (duration of a single frame) in milliseconds.
        """
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
    """
    Supported animation writer formats, file extensions with no period.

    :return: list of file extensions.
    """
    return ['mp4', 'gif', 'webp']


def create_animation_writer(animation_format: str, out_filename: str, fps: typing.Union[float, int]):
    """
    Create an animation writer of a given format.

    :param animation_format: The animation format, see :py:func:`.supported_animation_writer_formats`
    :param out_filename: the output file name
    :param fps: FPS
    :return: :py:class:`.AnimationWriter`
    """
    return VideoWriter(out_filename, fps) if animation_format.strip().lower() == 'mp4' \
        else AnimatedImageWriter(out_filename, 1000 / fps)


class MultiAnimationWriter(AnimationWriter):
    """
    Splits writes between N files with generated filename suffixes if necessary
    depending on how many images were written on the first write.
    """

    def __init__(self,
                 animation_format: str,
                 filename: str,
                 fps:
                 typing.Union[float, int],
                 allow_overwrites=False):
        """
        :param animation_format: One of :py:func:`.supported_animation_writer_formats`
        :param filename: The desired filename, if multiple images are written a 
            suffix _animation_N will be appended for each file
        :param fps: Frames per second
        :param allow_overwrites: Allow overwrites of existing files? or append _duplicate_N,
            The overwrite dis-allowance is multiprocess safe between instances of this library.
        """

        super().__init__()
        self.filename = filename
        self.writers = []
        self.filenames = []
        self.animation_format = animation_format
        self.fps = fps
        self.allow_overwrites = allow_overwrites

    def _gen_filename(self, num_images, image_idx):
        base, ext = os.path.splitext(self.filename)
        if num_images > 1:
            return f'{base}_animation_{image_idx + 1}{ext}'
        else:
            return f'{base}{ext}'

    def write(self, img: typing.Union[PIL.Image.Image, typing.List[PIL.Image.Image]]):
        if not isinstance(img, list):
            img = [img]

        if not self.writers:
            # Lazy initialize all the writers we need
            num_images = len(img)

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

            for filename in self.filenames:
                self.writers.append(
                    create_animation_writer(self.animation_format, filename, self.fps))

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

    def __exit__(self, exc_type, exc_val, exc_tb):
        for writer in self.writers:
            writer.__exit__(exc_type, exc_val, exc_tb)
