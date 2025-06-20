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
import os
import shutil
import tempfile
import typing

import PIL.Image
import av
import numpy
import safetensors.torch
import torch

import dgenerate.filelock as _filelock

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


class VideoWriter(AnimationWriter):
    """
    Animation writer for MP4 h264 format video
    """

    def __init__(self, filename, fps: float):
        """
        :param filename: Filename to write to.
        :param fps: Frame rate, in frames per second.
        """

        super().__init__()
        self.filename = filename
        self.fps = fps
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
            self._stream = self._container.add_stream("h264", rate=round(self.fps))
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

    def end(self, new_file: str = None):

        if self.collected_frames:
            self.collected_frames[0].save(self.filename, save_all=True, append_images=self.collected_frames[1:],
                                          optimize=False, duration=self.duration, loop=0)
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
    PIL.Image.init()

    return ['mp4'] + [ext for ext in (ext.lstrip('.').lower() for ext, file_format
                                      in PIL.Image.EXTENSION.items() if file_format in PIL.Image.SAVE_ALL) if
                      ext in {'gif', 'webp', 'apng', 'png'}]


def get_supported_static_image_formats():
    """
    What file extensions does PIL/Pillow support for output of at least one frame?

    File extensions are returned without a period.

    :return: list of file extensions
    """
    PIL.Image.init()

    return [ext.lstrip('.').lower() for ext, fmt in PIL.Image.EXTENSION.items() if fmt in PIL.Image.SAVE]


class UnknownAnimationFormatError(Exception):
    """
    Raised by :py:func:`.create_animation_writer` when an unknown animation format is provided.
    """


def create_animation_writer(animation_format: str, out_filename: str, fps: float):
    """
    Create an animation writer of a given format.

    :raise UnknownAnimationFormatError: if the provided ``animation_format`` is unknown.

    :param animation_format: The animation format, see :py:func:`.supported_animation_writer_formats`
    :param out_filename: the output file name
    :param fps: FPS
    :return: :py:class:`.AnimationWriter`
    """
    animation_format = animation_format.strip().lower()

    if animation_format not in get_supported_animation_writer_formats():
        raise UnknownAnimationFormatError(f'Animation format "{animation_format}" is not a known format.')

    return VideoWriter(out_filename, fps) if animation_format == 'mp4' \
        else AnimatedImageWriter(out_filename, 1000 / fps)


class MultiAnimationWriter(AnimationWriter):
    """
    Splits writes between N files with generated filename suffixes if necessary
    depending on how many images were written on the first write.
    """

    def __init__(self,
                 animation_format: str,
                 filename: str,
                 fps: float,
                 allow_overwrites=False):
        """
        :param animation_format: One of :py:func:`.supported_animation_writer_formats`
        :param filename: The desired filename, if multiple images are written a 
            suffix ``_animation_N`` will be appended for each file
        :param fps: Frames per second
        :param allow_overwrites: Allow overwrites of existing files? or append ``_duplicate_N``,
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

    def write(self, img: PIL.Image.Image | collections.abc.Iterable[PIL.Image.Image]):
        if not isinstance(img, collections.abc.Iterable):
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


def get_supported_tensor_formats() -> list[str]:
    """
    Get supported tensor file formats for latents output.
    
    :return: List of supported tensor formats
    """
    return ["pt", "pth", "safetensors"]


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
