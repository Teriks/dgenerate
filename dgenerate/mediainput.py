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
import mimetypes
import os
import re
import tempfile
import typing
import urllib.parse

import PIL.Image
import PIL.ImageOps
import PIL.ImageSequence
import av
import safetensors.torch
import torch

import dgenerate.image as _image
import dgenerate.imageprocessors as _imageprocessors
import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.torchutil as _torchutil
import dgenerate.types as _types
import dgenerate.webcache as _webcache

__doc__ = """
Media input, handles reading videos/animations, static images, and tensor files (.pt, .pth, .safetensors), 
and creating readers from image seed URIs.

Also provides media download capabilities and temporary caching of web based files.

Provides information about supported input formats including tensor formats for latent data.

Note: Tensor files are loaded as-is without any preprocessing, resizing, or image processing operations.
"""


def url_aware_normpath(path):
    """
    Only ``os.path.normpath`` a file path if it is not a URL.

    :param path: the path
    :return: normalized file path or unmodified URL
    """
    if is_downloadable_url(path):
        return path
    else:
        return os.path.normpath(path)


def url_aware_basename(path):
    """
    Get the ``os.path.basename`` of a file path or URL.

    :param path: the path
    :return: basename
    """
    if is_downloadable_url(path):
        parsed = urllib.parse.urlparse(path)
        return os.path.basename(parsed.path)
    else:
        return os.path.basename(path)


def url_aware_splitext(path):
    """
    Get the ``os.path.splitext`` result for a file path or URL.

    :param path: the path
    :return: base, ext
    """
    if is_downloadable_url(path):
        parsed = urllib.parse.urlparse(path)
        return os.path.splitext(parsed.path)
    else:
        return os.path.splitext(path)


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


class ImageSeedParseError(ImageSeedError):
    """
    Raised on image seed syntactic parsing error.
    """
    pass


class ImageSeedArgumentError(ImageSeedError):
    """
    Raised when image seed URI keyword arguments receive invalid values.
    """
    pass


class ImageSeedFileNotFoundError(ImageSeedError):
    """
    Raised when a file on disk in an image seed could not be found.
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
                 fps: float,
                 frame_duration: float,
                 total_frames: int, **kwargs):
        """
        :param width: width of the animation, X dimension
        :param height: height of the animation, Y dimension
        :param fps: frames per second
        :param frame_duration: frame duration in milliseconds
        :param total_frames: total frames in the animation
        :param kwargs: for mixins
        """
        self._width = width
        self._height = height
        self._fps = fps
        self._frame_duration = frame_duration
        self._total_frames = total_frames

    @property
    def width(self) -> int:
        """
        Width dimension, (X dimension).
        """
        return self._width

    @property
    def size(self) -> _types.Size:
        """
        returns (width, height) as a tuple.
        """
        return self._width, self._height

    @property
    def height(self) -> int:
        """
        Height dimension, (Y dimension).
        """
        return self._height

    @property
    def fps(self) -> float:
        """
        Frames per second.
        """
        return self._fps

    @property
    def frame_duration(self) -> float:
        """
        Duration of each frame in milliseconds.
        """
        return self._frame_duration

    @property
    def total_frames(self) -> int:
        """
        Total number of frames that can be read.
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


class VideoReader(_imageprocessors.ImageProcessorMixin, AnimationReader):
    """
    Implementation :py:class:`.AnimationReader` that reads Video files with PyAV.
    """

    def __init__(self,
                 file: str | typing.BinaryIO,
                 file_source: str,
                 resize_resolution: _types.OptionalSize = None,
                 aspect_correct: bool = True,
                 align: int | None = None,
                 image_processor: _imageprocessors.ImageProcessor = None):
        """
        :param file: a file path or binary file stream

        :param file_source: the source filename for the video data, should be the filename.
            this is for informational purpose when reading from a stream or a cached file
            and should be provided in every case even if it is a symbolic value only. It
            should possess a file extension as it is used to determine file format when
            reading from a byte stream. :py:class:`PIL.Image.Image` objects produced by
            the reader will have this value set to their *filename* attribute.

        :param resize_resolution: Progressively resize each frame to this
            resolution while reading. The provided resolution will be aligned
            by ``align`` if it is not ``None``.

        :param aspect_correct: Should resize operations be aspect correct?

        :param align: Align by this amount of pixels, if the input file is not aligned
            to this amount of pixels, it will be aligned by resizing. Passing ``None``
            or ``1`` disables alignment.

        :param image_processor: optionally process every frame with this image processor
            
        :raises MediaIdentificationError: If the video data is an unknown format or corrupt. or if
            file_source lacks a file extension, it is needed to determine the video file format.
        """
        self._filename = file
        self._file_source = file_source
        if isinstance(file, str):
            try:
                self._container = av.open(file, 'r')
            except av.error.InvalidDataError:
                raise MediaIdentificationError(
                    f'Error loading video file, unknown format or invalid data: "{file_source}"')
        else:
            _, ext = os.path.splitext(file_source)
            if not ext:
                raise MediaIdentificationError(
                    'Cannot determine the format of a video file from a file_source lacking a file extension.')
            self._container = av.open(file, format=ext.lstrip('.').lower())

        self._aspect_correct = aspect_correct
        self._align = align
        self._resize_resolution = resize_resolution

        width = int(self._container.streams.video[0].width)
        height = int(self._container.streams.video[0].height)

        # predict size after processing
        width, height = _image.resize_image_calc(
            old_size=(width, height),
            new_size=resize_resolution,
            aspect_correct=aspect_correct,
            align=align
        )

        fps = float(self._container.streams.video[0].average_rate)
        frame_duration = 1000 / fps
        total_frames = self._container.streams.video[0].frames

        self._container.streams.video[0].thread_type = "AUTO"

        if total_frames <= 0:
            # webm decode bug?
            total_frames = sum(1 for _ in self._container.decode(video=0))
            self._container.seek(0)
        self._iter = self._container.decode(video=0)

        super().__init__(width=width,
                         height=height,
                         fps=fps,
                         frame_duration=frame_duration,
                         total_frames=total_frames,
                         image_processor=image_processor)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._container.close()

    def __next__(self):
        rgb_image = next(self._iter).to_image()
        rgb_image.filename = self._file_source
        return self.process_image(image=rgb_image,
                                  resize_resolution=self._resize_resolution,
                                  aspect_correct=self._aspect_correct,
                                  align=self._align)


class AnimatedImageReader(_imageprocessors.ImageProcessorMixin, AnimationReader):
    """
    Implementation of :py:class:`.AnimationReader` that reads animated image formats using Pillow.
    """

    def __init__(self,
                 file: str | typing.BinaryIO,
                 file_source: str,
                 resize_resolution: _types.OptionalSize = None,
                 aspect_correct: bool = True,
                 align: int | None = None,
                 image_processor: _imageprocessors.ImageProcessor = None):
        """
        :param file: a file path or binary file stream

        :param file_source: the source filename for the animated image, should be the filename.
            this is for informational purpose when reading from a stream or a cached file
            and should be provided in every case even if it is a symbolic value only. It
            should possess a file extension. :py:class:`PIL.Image.Image` objects produced by
            the reader will have this value set to their *filename* attribute.

        :param resize_resolution: Progressively resize each frame to this
            resolution while reading. The provided resolution will be aligned
            by ``align`` if it is not ``None``.

        :param aspect_correct: Should resize operations be aspect correct?

        :param align: Align by this amount of pixels, if the input file is not aligned
            to this amount of pixels, it will be aligned by resizing. Passing ``None``
            or ``1`` disables alignment.

        :param image_processor: optionally process every frame with this image processor
        
        :raise MediaIdentificationError: If the animated image data is an unknown format or corrupt.
        """

        try:
            self._img = PIL.Image.open(file)
        except PIL.UnidentifiedImageError:
            raise MediaIdentificationError(
                f'Error loading image file, unknown format or invalid data: "{file_source}"')

        self._file_source = file_source

        self._iter = PIL.ImageSequence.Iterator(self._img)
        self._aspect_correct = aspect_correct
        self._align = align
        self._resize_resolution = resize_resolution

        total_frames = self._img.n_frames

        frame_duration = self._img.info.get('duration', 0)

        if frame_duration == 0:
            # 10 frames per second for bugged gifs / webp
            frame_duration = 100

        frame_duration = float(frame_duration)

        fps = 1000 / frame_duration

        # predict size after processing
        width, height = _image.resize_image_calc(
            old_size=self._img.size,
            new_size=resize_resolution,
            aspect_correct=aspect_correct,
            align=align
        )

        super().__init__(width=width,
                         height=height,
                         fps=fps,
                         frame_duration=frame_duration,
                         total_frames=total_frames,
                         image_processor=image_processor)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._img.close()

    def __next__(self) -> PIL.Image.Image:
        with next(self._iter) as img:
            rgb_image = _image.to_rgb(img)
            rgb_image.filename = self._file_source
            return self.process_image(image=rgb_image,
                                      resize_resolution=self._resize_resolution,
                                      aspect_correct=self._aspect_correct,
                                      align=self._align)


class MockImageAnimationReader(_imageprocessors.ImageProcessorMixin, AnimationReader):
    """
    Implementation of :py:class:`.AnimationReader` that repeats a single PIL image
    as many times as desired in order to mock/emulate an animation.
    """

    def __init__(self,
                 img: PIL.Image.Image,
                 resize_resolution: _types.OptionalSize = None,
                 aspect_correct: bool = True,
                 align: int | None = None,
                 image_repetitions: int = 1,
                 image_processor: _imageprocessors.ImageProcessor = None):
        """
        :param img: source image to copy for each frame, the image is immediately copied
            once upon construction of the mock reader, and then once per frame thereafter.
            Your copy of the image can be disposed of after the construction of this object.

        :param resize_resolution: the source image will be resized to this dimension with
            a maintained aspect ratio. This occurs once upon construction, a copy is then yielded
            for each frame that is read. The provided resolution will be aligned by ``align`` if
            it is not ``None``.

        :param aspect_correct: Should resize operations be aspect correct?

        :param align: Align by this amount of pixels, if the input file is not aligned
            to this amount of pixels, it will be aligned by resizing. Passing ``None``
            or ``1`` disables alignment.

        :param image_repetitions: number of frames that this mock reader provides
            using a copy of the source image.
        :param image_processor: optionally process the initial image with
            this image processor, this occurs once.
        """
        self._img = _image.copy_img(img)
        self._idx = 0
        self._aspect_correct = aspect_correct
        self._align = align
        self._resize_resolution = resize_resolution

        total_frames = image_repetitions
        fps = 30.0
        frame_duration = 1000 / fps

        # predict size after processing
        width, height = _image.resize_image_calc(
            old_size=self._img.size,
            new_size=resize_resolution,
            aspect_correct=aspect_correct,
            align=align
        )

        super().__init__(width=width,
                         height=height,
                         fps=fps,
                         frame_duration=frame_duration,
                         total_frames=total_frames,
                         image_processor=image_processor)

        # Only need to process once
        self._processed_flag = False

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

        if not self._processed_flag:
            # All the other readers behave in a way where processing
            # happens on read, this needs to be here as to not mess up
            # the order of debugging/message output from image processors.
            # processing only needs to occur once.
            self._img = self.process_image(image=self._img,
                                           resize_resolution=self._resize_resolution,
                                           aspect_correct=self._aspect_correct,
                                           align=self._align)

            self._processed_flag = True

        if self._idx < self.total_frames:
            self._idx += 1
            return _image.copy_img(self._img)
        else:
            raise StopIteration


class MockTensorReader(AnimationReader):
    """
    Implementation of :py:class:`.AnimationReader` that yields a single tensor
    as many times as desired to mock/emulate an animation with tensor data.
    
    This reader is used for .pt, .pth, and .safetensors files containing latent tensors.
    No image processing, resizing, or alignment operations are performed on tensors.
    """

    def __init__(self,
                 tensor: torch.Tensor,
                 file_source: str,
                 tensor_repetitions: int = 1):
        """
        :param tensor: source tensor to yield for each frame
        :param file_source: source filename for the tensor data
        :param tensor_repetitions: number of frames that this mock reader provides
            using the source tensor
        """
        self._tensor = tensor
        self._file_source = file_source
        self._idx = 0

        total_frames = tensor_repetitions
        fps = 30.0
        frame_duration = 1000 / fps

        # For tensors, we don't have meaningful width/height
        # Use tensor shape if it's 4D (batch, channels, height, width)
        if len(tensor.shape) >= 2:
            height = tensor.shape[-2] if len(tensor.shape) >= 2 else 1
            width = tensor.shape[-1] if len(tensor.shape) >= 1 else 1
        else:
            width = height = 1

        super().__init__(width=width,
                         height=height,
                         fps=fps,
                         frame_duration=frame_duration,
                         total_frames=total_frames)

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
        # Tensors don't need explicit cleanup
        pass

    def __next__(self) -> torch.Tensor:
        if self._idx < self.total_frames:
            self._idx += 1
            # Return the tensor directly - cloning is only needed if the caller modifies it
            # Most use cases just read the tensor, so avoid unnecessary memory overhead
            result = self._tensor
            # Set a filename-like attribute for consistency (this doesn't modify the tensor data)
            if hasattr(result, 'filename'):
                result.filename = self._file_source
            return result
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


def is_downloadable_url(string) -> bool:
    """
    Does a string represent a URL that can be downloaded from by :py:mod:`dgenerate.mediainput`?

    :param string: the string
    :return: ``True`` or ``False``
    """
    return string.startswith('http://') or string.startswith('https://')


def get_supported_animated_image_mimetypes() -> list[str]:
    """
    Get a list of mimetypes that are considered to be supported animated image mimetypes.

    :return: list of mimetype strings.
    """
    return ['image/gif', 'image/webp', 'image/apng']


def _get_supported_read_video_codecs():
    supported_codecs = []

    for codec_name in av.codecs_available:
        try:
            codec = av.codec.Codec(codec_name, 'r')
        except:
            continue
        if codec.type == 'video':
            supported_codecs.append(codec_name)

    return supported_codecs


def _get_read_video_file_extensions() -> list[str]:
    common_extensions = {
        "mp4": ["h264", "hevc", "mpeg4"],
        "mkv": ["h264", "hevc", "mpeg4", "vp8", "vp9"],
        "avi": ["mpeg4", "msmpeg4v3"],
        "mov": ["h264", "mpeg4"],
        "flv": ["flv1"],
        "wmv": ["wmv1", "wmv2", "wmv3"],
        "webm": ["vp8", "vp9"],
        "m4v": ["h264", "mpeg4"],
        "ts": ["h264", "hevc", "mpeg2video"],
        "mpg": ["mpeg1video", "mpeg2video"],
        "3gp": ["h263"],
        "ogv": ["theora"],
        "rm": ["rv10", "rv20", "rv30", "rv40"],
        "asf": ["msmpeg4v1", "msmpeg4v2", "msmpeg4v3", "wmv1", "wmv2", "wmv3"],
        "f4v": ["h264"],
        "h264": ["h264"],
        "hevc": ["hevc"],
        "mjpeg": ["mjpeg", "mjpeg2000"],
        "vp8": ["vp8"],
        "vp9": ["vp9"],
        "vob": ["mpeg2video"],
        "divx": ["mpeg4"],
        "xvid": ["mpeg4"],
        "dv": ["dvvideo"],
        "amv": ["amv"],
        "mxf": ["mjpeg"],
        "m2ts": ["h264", "mpeg2video"],
        "mpeg": ["mpeg1video", "mpeg2video"],
        "mpv": ["mpeg1video", "mpeg2video"]
    }

    supported_codecs = _get_supported_read_video_codecs()
    supported_extensions = set()

    for ext, codecs in common_extensions.items():
        for codec in codecs:
            if codec in supported_codecs:
                supported_extensions.add(ext)

    return list(supported_extensions)


def get_supported_animation_reader_formats():
    """
    Supported animation reader formats, file extensions with no period.

    :return: list of file extensions.
    """
    PIL.Image.init()

    return _get_read_video_file_extensions() + [
        ext for ext in (ext.lstrip('.').lower() for ext, file_format
                        in PIL.Image.EXTENSION.items() if
                        file_format in PIL.Image.OPEN) if
        ext in {'gif', 'webp', 'apng'}]


def get_supported_image_formats():
    """
    What file extensions does PIL/Pillow support for reading?

    File extensions are returned without a period.

    :return: list of file extensions
    """
    PIL.Image.init()

    return [ext.lstrip('.').lower() for ext, fmt in PIL.Image.EXTENSION.items() if fmt in PIL.Image.OPEN]


def get_supported_static_image_mimetypes() -> list[str]:
    """
    Get a list of mimetypes that are considered to be supported static image mimetypes.

    :return: list of mimetype strings.
    """
    return [
        "image/psd",  # psd
        "image/palm",  # palm
        "application/photoshop",  # psd alternative
        "application/psd",  # psd alternative
        "application/octet-stream",  # bufr, pfm (generic binary stream)
        "application/x-hdf",  # h5, hdf
        "image/vnd.ms-dds",  # dds
        "application/jpg",  # jpg alternative
        "image/x-bw",  # bw
        "image/jpx",  # jp2 alternative
        "application/x-bmp",  # bmp alternative
        "image/x-icon",  # ico alternative
        "image/x-icns",  # icns
        "application/x-im",  # im
        "image/bmp",  # bmp, dib
        "image/x-tga",  # tga alternative
        "image/pipeg",  # jfif
        "image/jpeg2000-image",  # jp2 alternative
        "application/x-msmetafile",  # wmf, emf
        "application/x-grib",  # grib
        "image/png",  # png
        "image/vnd.microsoft.icon",  # ico
        "image/x-targa",  # tga, icb, vda, vst
        "application/postscript",  # ps, eps
        "image/blp",  # blp
        "application/pdf",  # pdf
        "application/x-photoshop",  # psd alternative
        "image/jpg",  # jpg alternative
        "image/j2k",  # jp2 alternative
        "application/png",  # png alternative
        "image/jpeg2000",  # jp2 alternative
        "image/x-pcx",  # pcx
        "image/tga",  # tga alternative
        "image/x-portable-anymap",  # pbm, pgm, ppm, pnm
        "image/sgi",  # sgi
        "application/bmp",  # bmp alternative
        "image/targa",  # tga alternative
        "image/vnd.adobe.photoshop",  # psd
        "application/x-jpg",  # jpg alternative
        "image/x-bitmap",  # bmp alternative
        "image/x-bmp",  # bmp alternative
        "image/x-mspaint",  # msp
        "image/x-rgb",  # rgb, rgba
        "image/jp2",  # jp2, j2k, jpc, jpf, jpx, j2c
        "image/mpo",  # mpo
        "image/x-jpeg2000-image",  # jp2 alternative
        "image/jpeg",  # jpg, jpeg, jpe
        "application/x-png",  # png alternative
        "image/tiff",  # tif, tiff
        "image/x-xbitmap"  # xbm
    ]


def get_supported_image_mimetypes() -> list[str]:
    """
    Get all supported ``--image-seeds`` image mimetypes, including animated image mimetypes

    :return: list of strings
    """
    return get_supported_static_image_mimetypes() + get_supported_animated_image_mimetypes()


def get_supported_video_mimetypes() -> list[str]:
    """
    Get all supported ``--image-seeds`` video mimetypes, may contain a wildcard

    :return: list of strings
    """
    return ['video/*']


def get_supported_mimetypes() -> list[str]:
    """
    Get all supported ``--image-seeds`` mimetypes, video mimetype may contain a wildcard.

    :return: list of strings
    """
    return list(
        set(get_supported_image_mimetypes()) |
        set(get_supported_video_mimetypes()) |
        set(get_supported_tensor_mimetypes())
    )


def get_supported_tensor_formats() -> list[str]:
    """
    Get supported tensor file formats for latent loading.
    
    :return: list of file extensions without periods
    """
    return ['pt', 'pth', 'safetensors']


def get_supported_tensor_mimetypes() -> list[str]:
    """
    Get supported tensor mimetypes for latent loading.
    
    :return: list of mimetype strings
    """
    return ['application/octet-stream']


def is_tensor_file(path: str) -> bool:
    """
    Check if a file path appears to be a tensor file based on extension.
    
    :param path: file path or URL
    :return: ``True`` if it appears to be a tensor file
    """
    _, ext = url_aware_splitext(path)
    return ext.lstrip('.').lower() in get_supported_tensor_formats()


def load_tensor_file(path_or_file: typing.BinaryIO | str, file_source: str) -> torch.Tensor:
    """
    Load a tensor from a .pt, .pth, or .safetensors file.
    
    :param path_or_file: file path or binary IO object
    :param file_source: source filename for error reporting
    :return: loaded tensor
    :raises MediaIdentificationError: if the file cannot be loaded or if the file format is not supported
    """
    _, ext = os.path.splitext(file_source)
    ext = ext.lstrip('.').lower()

    try:
        if ext in ['pt', 'pth']:
            if isinstance(path_or_file, str):
                tensor = torch.load(path_or_file, map_location='cpu', weights_only=True)
            else:
                tensor = torch.load(path_or_file, map_location='cpu', weights_only=True)
        elif ext == 'safetensors':
            if isinstance(path_or_file, str):
                tensor_dict = safetensors.torch.load_file(path_or_file, device='cpu')
            else:
                # safetensors doesn't support loading from file objects directly
                # We'd need to save to a temporary file first
                tmp_name = None
                try:
                    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as tmp:
                        tmp.write(path_or_file.read())
                        tmp.flush()
                        tmp_name = tmp.name  # Store the name before closing

                    tensor_dict = safetensors.torch.load_file(tmp_name, device='cpu')
                finally:
                    # Clean up the temporary file
                    if tmp_name is not None:
                        try:
                            os.unlink(tmp_name)
                        except (OSError, PermissionError):
                            # On Windows, sometimes the file is still locked
                            # Try again after a short delay
                            import time
                            time.sleep(0.1)
                            try:
                                os.unlink(tmp_name)
                            except (OSError, PermissionError):
                                # If it still fails, just leave it for the OS to clean up
                                pass

            # Handle both single tensor and dictionary of tensors
            if len(tensor_dict) == 1:
                tensor = next(iter(tensor_dict.values()))
            else:
                # If multiple tensors, look for common latent keys
                latent_keys = ['latent', 'latents', 'sample', 'samples']
                for key in latent_keys:
                    if key in tensor_dict:
                        tensor = tensor_dict[key]
                        break
                else:
                    # If no common key found, take the first tensor
                    tensor = next(iter(tensor_dict.values()))
        else:
            raise MediaIdentificationError(f'Unsupported tensor file format: {ext}')

        # Ensure we have a tensor
        if not isinstance(tensor, torch.Tensor):
            raise MediaIdentificationError(
                f'File "{file_source}" does not contain a valid tensor')

        return tensor

    except Exception as e:
        if isinstance(e, MediaIdentificationError):
            raise
        raise MediaIdentificationError(
            f'Error loading tensor file "{file_source}": {str(e)}')


def separate_images_and_tensors(
        items: _types.ImagesOrTensors
) -> tuple[list[PIL.Image.Image], list[torch.Tensor]]:
    """
    Separate a sequence of images or tensors into separate sequences.
    
    Note: The input should be homogeneous (all images or all tensors), but this function
    can handle mixed inputs for validation purposes.
    
    :param items: Sequence of PIL Images or torch Tensors (should be homogeneous)
    :return: Tuple of (images, tensors) where each can be empty if no items of that type exist
    """

    images = []
    tensors = []

    for item in items:
        if _image.is_image(item):
            images.append(item)
        elif _torchutil.is_tensor(item):
            tensors.append(item)

    # noinspection PyTypeChecker
    return images, tensors


def mimetype_is_tensor(mimetype: str) -> bool:
    """
    Check if a mimetype is one that dgenerate considers a tensor file

    :param mimetype: The mimetype string
    :return: bool
    """
    return mimetype in get_supported_tensor_mimetypes()


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
def mimetype_is_supported(mimetype: str) -> bool:
    """
    Check if dgenerate supports a given input mimetype

    :param mimetype: The mimetype string

    :return: bool
    """
    return mimetype_is_static_image(mimetype) or \
        mimetype_is_animated_image(mimetype) or \
        mimetype_is_video(mimetype) or \
        mimetype_is_tensor(mimetype)


class IPAdapterImageUri:
    path: str
    """
    File path or URL to an image.
    """

    scale: float
    """
    IP Adapter image scale value.
    """

    resize: str | None
    """
    Image resize dimension in the form ``WIDTHxHEIGHT`` or ``WIDTH``
    """

    aspect: bool
    """
    Aspect correct resizing?
    """

    align: int
    """
    Pixel alignment, defaults to 1.
    """

    def __init__(self, path, resize, aspect, align):
        self.path = path
        self.resize = resize
        self.aspect = aspect
        self.align = align

    def __str__(self):
        result = f'path={self.path}'
        if self.resize is not None:
            result += f', resize={self.resize}'
        result += f', aspect={self.aspect}, align={self.align}'
        return f'IPAdapterImageSpec({result})'

    def __repr__(self):
        return str(self)


class ImageSeedParseResult:
    """
    The result of parsing an ``--image-seeds`` uri
    """

    uri: str
    """
    The original URI string the image seed was parsed from.
    """

    multi_image_mode: bool
    """
    Are there multiple img2img images associated with this image seed?
    
    This indicates that ``--image-seeds "images: image1.png, image2.png"`` syntax was used, in
    order to differentiate from a control image sequence specification.
    """

    images: _types.Paths | None
    """
    Optional image paths that will be used for img2img operations
    or the base image in inpaint operations. 
    
    Or controlnet guidance paths, in the case that ``images: ...`` syntax is not
    being used (``multi_image_mode==False``) and there are multiple provided images, 
    and ``is_single_spec`` is ``True``.
    
    A path being a file path, or an HTTP/HTTPS URL.
    """

    mask_images: _types.Paths | None = None
    """
    Optional inpaint mask paths, may be HTTP/HTTPS URLs or file paths.
    
    This may be multiple masks when there are multiple img2img image paths,
    for example: ``--image-seeds "images: image1.png, image2.png; mask1.png, mask2.png"``
    
    There will always be an equal number of images and mask images.
    
    If a single mask is supplied for multiple images, the mask path is duplicated to match
    the amount of images.
    """

    control_images: _types.Paths | None = None
    """
    Optional controlnet guidance path, or a sequence of controlnet guidance paths. 
    This field is only used when the secondary syntax of ``--image-seeds`` is encountered.
    
    IE: This parameter is only filled if the keyword argument ``control`` is used.
    
    In parses such as:
    
        * ``--image-seeds "img2img.png;control=control.png"``
        * ``--image-seeds "img2img.png;control=control1.png, control2.png"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control.png"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control1.png, control2.png"``
        
    """

    adapter_images: collections.abc.Sequence[
                        collections.abc.Sequence[IPAdapterImageUri]] | None = None
    """
    IP Adapter image URIs.

    In parses such as:
    
        * ``--image-seeds "adapter: adapter-image1.png + adapter-image2.png"``
        * ``--image-seeds "adapter: adapter-image1.png + adapter-image2.png;control=control.png"``
        * ``--image-seeds "img2img.png;adapter=adapter-image1.png + adapter-image2.png"``
        * ``--image-seeds "img2img.png;adapter=adapter-image1.png + adapter-image2.png;control=control.png"``
        * ``--image-seeds "img2img.png;adapter=adapter-image1.png + adapter-image2.png;mask=inpaint-mask.png;control=control.png"``
    """

    latents: _types.Paths | None = None
    """
    Optional raw latent tensor paths (.pt, .pth, .safetensors files).
    
    In parses such as:
    
        * ``--image-seeds "latents: latents1.pt, latents2.pt"``
        * ``--image-seeds "img2img.png;latents=latents.pt"``
        * ``--image-seeds "images: img2img1.png, img2img2.png;latents=latents1.pt, latents2.pt"``
        
    Raw latents are loaded as-is without any image processing, resizing, or alignment operations.
    """

    floyd_image: _types.OptionalPath = None
    """
    Optional path to a result from a Deep Floyd IF stage, used only for img2img and inpainting mode
    with Deep Floyd. This is the only way to specify the image that was output by a stage in that case.
    
    the arguments floyd and control are mutually exclusive.
    
    In parses such as:
    
        * ``--image-seeds "img2img.png;floyd=stage1-output.png"``
        * ``--image-seeds "img2img.png;mask=mask.png;floyd=stage1-output.png"``
        
    There can only ever be one floyd stage image provided.
    """

    resize_resolution: _types.OptionalSize = None
    """
    Per image user specified resize resolution for image, mask, and control image 
    components of the ``--image-seed`` specification.
    
    This field available in parses such as:
    
        * ``--image-seeds "img2img.png;512x512"``
        * ``--image-seeds "img2img.png;mask.png;512x512"``
        * ``--image-seeds "img2img.png;control=control.png;resize=512x512"``
        * ``--image-seeds "img2img.png;control=control1.png, control2.png;resize=512x512"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control.png;resize=512x512"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control1.png, control2.png;resize=512x512"``
        
    This should override any globally defined resize value.
    """

    resize_align: _types.OptionalInteger = None
    """
    Per image user specified resize alignment for image, mask, and control image 
    components of the ``--image-seed`` specification.
    
    This field available in parses such as:
    
        * ``--image-seeds "img2img.png;control=control.png;align=64"``
        * ``--image-seeds "img2img.png;control=control1.png, control2.png;align=64"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control.png;align=64"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control1.png, control2.png;align=64"``
        
    This will overwrite any default global value (usually 1), the provided value must be divisible by 
    the global value defined by the parser, or a parse error will occur.
    """

    aspect_correct: _types.OptionalBoolean = None
    """
    Aspect correct resize setting override from the **aspect** image seed keyword argument, 
    if this is None it was not specified. This value if defined should override any globally
    defined aspect correct resize setting.
    """

    frame_start: _types.OptionalInteger = None
    """
    Optional start frame specification for per-image seed slicing.
    """

    frame_end: _types.OptionalInteger = None
    """
    Optional end frame specification for per-image seed slicing.
    """

    def get_control_image_paths(self) -> _types.Paths | None:
        """
        Return :py:attr:`.ImageSeedParseResult.seed_path` if :py:attr:`.ImageSeedParseResult.is_single_spec` is ``True``.

        If the image seed is not a single specification, return :py:attr:`.ImageSeedParseResult.control_path`.

        If :py:attr:`.ImageSeedParseResult.control_path` is not set and the image seed is not a single
        specification, return ``None``.

        :return: list of resource paths, or None
        """
        if self.is_single_spec and not self.multi_image_mode:
            return self.images
        elif self.control_images:
            return self.control_images
        else:
            return None

    @property
    def has_ip_adapter_images(self) -> bool:
        return self.adapter_images is not None

    @property
    def is_single_spec(self) -> bool:
        """
        Is this ``--image-seeds`` uri a single resource or resource group specification existing
        within the **seed_path** attribute of this object?

        For instance could it be a img2img definition / sequence of img2img images using
        the ``images: ...`` syntax, or a sequence of controlnet guidance images?

        This requires that ``mask_images``, ``control_images``, ``floyd_image``, ``adapter_images``, and ``latents`` are all undefined.

        Possible parses which trigger this condition are:

            * ``--image-seeds "img2img.png"``
            * ``--image-seeds "control-image.png"``
            * ``--image-seeds "control-image1.png, control-image2.png"``
            * ``--image-seeds "images: img2img-1.png, img2img-2.png"``

        Since this is an ambiguous parse, it must be resolved later with the help of other specified arguments.
        Such as by the specification of ``--control-nets``, which makes the intention unambiguous.

        :return: bool
        """

        return self.mask_images is None \
            and self.control_images is None \
            and self.floyd_image is None \
            and self.adapter_images is None \
            and self.latents is None


ParsedImageSeeds = collections.abc.Sequence[ImageSeedParseResult]
OptionalParsedImageSeeds = typing.Optional[ParsedImageSeeds]

_ip_adapter_image_parser = _textprocessing.ConceptUriParser(
    'Adapter Image', ['resize', 'aspect', 'align'], delimiter='|')


def _parse_ip_adapter_uri(uri: str) -> IPAdapterImageUri:
    i_strip = uri.strip()
    adapter_parts = i_strip.split('|')

    resize = None
    aspect = True
    align = 1

    if len(adapter_parts) == 1:
        path = adapter_parts[0]
    else:
        result = _ip_adapter_image_parser.parse(uri)
        path = result.concept

        try:
            resize = _textprocessing.parse_image_size(result.args.get('resize', None))
        except ValueError:
            raise ImageSeedParseError(
                f'Could not parse adapter image "resize" argument: {result.args["resize"]}')

        try:
            aspect = _types.parse_bool(result.args.get('aspect', True))
        except ValueError:
            raise ImageSeedParseError(
                f'Could not parse adapter image "aspect" argument: {result.args["aspect"]}')

        try:
            align = int(result.args.get('align', 1))
        except ValueError:
            raise ImageSeedParseError(
                f'Could not parse adapter image "align" argument: {result.args["align"]}')

    if not (is_downloadable_url(path) or os.path.exists(path)):
        raise ImageSeedFileNotFoundError(
            f'Adapter image file "{path}" does not exist.')

    if is_tensor_file(path):
        raise ImageSeedParseError(
            f'IP Adapter image path "{path}" is a tensor file. IP Adapter expect images in pixel space, '
            f'tensor files are not supported for IP Adapter images.')

    return IPAdapterImageUri(path, resize, aspect, align)


# noinspection HttpUrlsUsage
def _parse_image_seed_uri_legacy(uri: str, align: int = 8) -> ImageSeedParseResult:
    try:
        parts = _textprocessing.tokenized_split(uri, ';')
        parts_iter = iter(parts)
    except _textprocessing.TokenizedSplitSyntaxError as e:
        raise ImageSeedParseError(
            f'Parsing error in image seed URI "{uri}": {str(e).strip()}')

    result = ImageSeedParseResult()
    result.uri = uri

    try:
        first = next(parts_iter)
    except StopIteration:
        raise ImageSeedParseError(
            f'Parsing error in image seed URI "{uri}": empty specification.')

    result.multi_image_mode = first.startswith('images:')
    if result.multi_image_mode:
        result.images = []
        first = first.removeprefix('images:').strip()

    ip_adapter_mode = first.startswith('adapter:')
    if ip_adapter_mode:
        result.adapter_images = []
        first = first.removeprefix('adapter:').strip()

    latents_mode = first.startswith('latents:')
    if latents_mode:
        result.latents = []
        first = first.removeprefix('latents:').strip()

    if not ip_adapter_mode and not latents_mode:
        result.images = []

    try:
        first_parts = [t.strip() for t in _textprocessing.tokenized_split(
            first, ',', remove_quotes=True, escapes_in_quoted=True)]

    except _textprocessing.TokenizedSplitSyntaxError as e:
        raise ImageSeedParseError(
            f'Parsing error in image seed URI "{uri}": {str(e).strip()}')

    for part in first_parts:
        if ip_adapter_mode:
            result.adapter_images.append([])
            adapter_images = _textprocessing.tokenized_split(part, '+')

            for image in adapter_images:
                result.adapter_images[-1].append(_parse_ip_adapter_uri(image))
        elif latents_mode:
            if not is_tensor_file(part):
                raise ImageSeedParseError(
                    f'Latent file "{part}" must be a tensor file (.pt, .pth, or .safetensors).')
            if not (is_downloadable_url(part) or os.path.exists(part)):
                raise ImageSeedFileNotFoundError(
                    f'Latent tensor file "{part}" does not exist.')
            result.latents.append(part)
        elif result.multi_image_mode:
            if not (is_downloadable_url(part) or os.path.exists(part)):
                raise ImageSeedFileNotFoundError(
                    f'Image seed file "{part}" does not exist.')
            else:
                result.images.append(part)
        else:
            if not (is_downloadable_url(part) or os.path.exists(part)):
                if len(first_parts) > 1:
                    raise ImageSeedFileNotFoundError(
                        f'Control image file "{part}" does not exist.')
                else:
                    raise ImageSeedFileNotFoundError(
                        f'Image seed file "{part}" does not exist.')
            result.images.append(part)

    if len(first_parts) > 1 and not ip_adapter_mode and not latents_mode:
        result.images = first_parts
    elif ip_adapter_mode:
        result.images = None
    elif latents_mode:
        result.images = None

    def parse_multi_mask(mask_part):
        if not result.multi_image_mode or not result.images:
            return False, None

        try:
            masks = _textprocessing.tokenized_split(
                mask_part, ',', strict=True, remove_quotes=True, escapes_in_quoted=True)

            if len(masks) != 1 and len(masks) != len(result.images):
                raise ImageSeedParseError(
                    f'Must specify one inpaint mask for multiple seed images, '
                    f'or a mask for each seed image.')

            for m in masks:
                if not (is_downloadable_url(m) or os.path.exists(m)):
                    return False, None

            if len(masks) == 1:
                masks *= len(result.images)

            return True, masks

        except _textprocessing.TokenizedSplitSyntaxError:
            return False, None

    for part in parts_iter:
        if part == '':
            raise ImageSeedParseError(
                'Missing inpaint mask image or output size specification, '
                'check image seed syntax, stray semicolon?')

        if (multi_mask := parse_multi_mask(part))[0]:
            result.mask_images = multi_mask[1]
        else:
            # For single mask images, strip quotes if present
            if _textprocessing.is_quoted(part):
                part = _textprocessing.unquote(
                    part, escapes_in_quoted=True
                    )
            
            if is_downloadable_url(part):
                result.mask_images = [part]
            elif os.path.exists(part):
                result.mask_images = [part]
            else:
                try:
                    dimensions = _textprocessing.parse_image_size(part)
                except ValueError:
                    # This is correct and more informative
                    # though it is counter intuitive
                    raise ImageSeedFileNotFoundError(
                        f'Inpaint mask file "{part}" does not exist.')

                for d_idx, d in enumerate(dimensions):
                    if d % align != 0:
                        raise ImageSeedArgumentError(
                            f'Image seed resize {["width", "height"][d_idx]} '
                            f'dimension {d} is not divisible by {align}.')

                if result.resize_resolution is not None:
                    raise ImageSeedArgumentError(
                        'Resize resolution argument defined multiple times.')

                result.resize_resolution = dimensions

    if result.multi_image_mode:
        if not result.images:
            raise ImageSeedParseError(
                'Must specify at least one file when using the "images: ..." '
                'image seed syntax, no files were specified.'
            )

    if ip_adapter_mode:
        if not result.adapter_images:
            raise ImageSeedParseError(
                'Must specify at least one file when using the "adapter: ..." '
                'image seed syntax, no files were specified.'
            )

        if result.resize_resolution or result.mask_images:
            raise ImageSeedParseError(
                'Cannot use resize resolution or inpaint mask '
                'syntax with IP adapter only image seed input.')

    if latents_mode:
        if not result.latents:
            raise ImageSeedParseError(
                'Must specify at least one file when using the "latents: ..." '
                'image seed syntax, no files were specified.'
            )

        if result.resize_resolution or result.mask_images:
            raise ImageSeedParseError(
                'Cannot use resize resolution or inpaint mask '
                'syntax with latents only image seed input.')

    if result.images and len(result.images) > 1 and result.mask_images and not result.multi_image_mode:
        raise ImageSeedParseError(
            'Cannot use multiple image inputs with inpaint '
            'masks without using the "images:" syntax."'
        )

    if result.mask_images and len(result.mask_images) > len(result.images):
        raise ImageSeedParseError(
            'There cannot be more "mask" images than image seed image inputs.')

    return result


def parse_image_seed_uri(uri: str, align: int | None = 8) -> ImageSeedParseResult:
    """
    Parse an ``--image-seeds`` uri into its constituents

    All URI related errors raised by this function derive from :py:exc:`.ImageSeedError`.

    :raises ValueError: if ``align < 1``

    :raises ImageSeedParseError: on syntactical parsing errors
    :raises ImageSeedArgumentError: on image seed URI argument errors
    :raises ImageSeedFileNotFoundError: when a file mentioned in an image seed does not exist on disk

    :param uri: ``--image-seeds`` uri
    :param align: do not allow per image seed resize resolutions that are not aligned to this value,
        setting this value to 1 or ``None`` disables alignment checks.

    :raise ValueError: On ``align < 1``

    :return: :py:class:`.ImageSeedParseResult`
    """

    if align is None:
        align = 1
    elif align < 1:
        raise ValueError('align argument may not be less than one.')

    keyword_args = ['mask',
                    'control',
                    'adapter',
                    'latents',
                    'floyd',
                    'resize',
                    'align',
                    'aspect',
                    'frame-start',
                    'frame-end']

    try:
        parts = _textprocessing.tokenized_split(uri, ';')
    except _textprocessing.TokenizedSplitSyntaxError as e:
        raise ImageSeedParseError(f'Image seed URI parsing error: {str(e).strip()}') from e

    non_legacy: bool = len(parts) > 3

    if not non_legacy:
        for i in parts:
            for kwarg in keyword_args:
                if re.match(f'{kwarg}\\s*=', i) is not None:
                    non_legacy = True
                    break
            if non_legacy:
                break

    if not non_legacy:
        # No keyword arguments, basic old syntax
        return _parse_image_seed_uri_legacy(uri, align=align)

    result = ImageSeedParseResult()
    result.uri = uri

    seed_parser = _textprocessing.ConceptUriParser('Image Seed',
                                                   known_args=keyword_args,
                                                   args_lists=['control', 'mask', 'latents'],
                                                   args_raw=['adapter'])

    try:
        parse_result = seed_parser.parse(uri)
    except _textprocessing.ConceptUriParseError as e:
        raise ImageSeedParseError(e) from e

    # noinspection HttpUrlsUsage
    def _ensure_exists(path, title):
        if not (is_downloadable_url(path) or os.path.exists(path)):
            raise ImageSeedFileNotFoundError(f'{title} file "{path}" does not exist.')

    def parse_adapters(adapter_paths):
        if adapter_paths is not None:
            result.adapter_images = []
            adapter_paths = _textprocessing.tokenized_split(
                adapter_paths, ',', remove_quotes=True, escapes_in_quoted=True)
            for adapter_path in adapter_paths:
                a_strip = adapter_path.strip()
                if not a_strip:
                    raise ImageSeedParseError('Missing adapter image definition, stray comma?')
                result.adapter_images.append([])
                adapter_images = _textprocessing.tokenized_split(a_strip, '+')
                for image in adapter_images:
                    result.adapter_images[-1].append(_parse_ip_adapter_uri(image))

    images = parse_result.concept.strip()

    adapters_parsed = False
    result.multi_image_mode = images.startswith('images:')

    if images.startswith('adapter:'):
        adapters_parsed = True
        parse_adapters(images.removeprefix('adapter:').strip())
        result.images = None

        if not result.adapter_images:
            raise ImageSeedParseError(
                'Must specify at least one file when using the "adapter: ..." '
                'image seed syntax, no files were specified.'
            )

    elif images.startswith('latents:'):
        latent_paths = [p.strip() for p in
                        _textprocessing.tokenized_split(
                            images.removeprefix('latents:').strip(),
                            ',', strict=True, remove_quotes=True, escapes_in_quoted=True)]
        if not latent_paths:
            raise ImageSeedParseError(
                'Must specify at least one file when using the "latents: ..." '
                'image seed syntax, no files were specified.'
            )

        for latent_path in latent_paths:
            if not is_tensor_file(latent_path):
                raise ImageSeedParseError(
                    f'Latent file "{latent_path}" must be a tensor file (.pt, .pth, or .safetensors).')
            _ensure_exists(latent_path, 'Latent tensor')

        result.latents = latent_paths
        result.images = None
    elif result.multi_image_mode:
        seed_images = [p.strip() for p in
                       _textprocessing.tokenized_split(
                           images.removeprefix('images:').strip(),
                           ',', strict=True, remove_quotes=True, escapes_in_quoted=True)]

        if not seed_images:
            raise ImageSeedParseError(
                'Must specify at least one file when using the "images: ..." '
                'image seed syntax, no files were specified.'
            )

        for img in seed_images:
            _ensure_exists(img, 'Image seed')
        result.images = seed_images
    else:
        images = _textprocessing.tokenized_split(
            images, ',', remove_quotes=True, escapes_in_quoted=True)
        for i in images:
            _ensure_exists(i, 'Image seed')
        result.images = images

    mask_images = parse_result.args.get('mask', None)
    if mask_images is not None:
        if isinstance(mask_images, list):
            if not result.multi_image_mode:
                raise ImageSeedParseError(
                    'Cannot use multiple mask images without '
                    'using the syntax: --image-seeds "images: image1.png, image2.png;mask=mask1.png, mask2.png"')

            if len(mask_images) != len(result.images):
                raise ImageSeedParseError(
                    f'Must specify one inpaint mask for multiple seed images, '
                    f'or a mask for each seed image.')

            for i in mask_images:
                _ensure_exists(i, 'Inpaint mask')
        else:
            _ensure_exists(mask_images, 'Inpaint mask')
            if result.multi_image_mode:
                mask_images = [mask_images] * len(result.images)

        result.mask_images = mask_images if isinstance(mask_images, list) else [mask_images]

    if result.images is None and mask_images is not None:
        raise ImageSeedParseError(
            'Cannot use "mask" image seed argument when the only input is IP adapter images.')

    if result.mask_images is not None and len(result.mask_images) > len(result.images):
        raise ImageSeedParseError(
            'There cannot be more "mask" images than image seed image inputs.')

    control_images = parse_result.args.get('control', None)

    if control_images is not None:
        if isinstance(control_images, list):
            for f in control_images:
                if not f.strip():
                    raise ImageSeedParseError('Missing control image definition, stray comma?')

                _ensure_exists(f, 'Control image')
            result.control_images = control_images
        else:
            _ensure_exists(control_images, 'Control image')
            result.control_images = [control_images]

    latents = parse_result.args.get('latents', None)

    if latents is not None:
        if isinstance(latents, list):
            for f in latents:
                if not f.strip():
                    raise ImageSeedParseError('Missing latent tensor definition, stray comma?')
                if not is_tensor_file(f):
                    raise ImageSeedParseError(
                        f'Latent file "{f}" must be a tensor file (.pt, .pth, or .safetensors).')
                _ensure_exists(f, 'Latent tensor')
            result.latents = latents
        else:
            if not is_tensor_file(latents):
                raise ImageSeedParseError(
                    f'Latent file "{latents}" must be a tensor file (.pt, .pth, or .safetensors).')
            _ensure_exists(latents, 'Latent tensor')
            result.latents = [latents]

    adapter_arg = parse_result.args.get('adapter', None)

    if adapter_arg is not None:
        if not adapters_parsed:
            adapters_parsed = True
            parse_adapters(adapter_arg)
        else:
            raise ImageSeedParseError('IP adapter images already defined, cannot be defined '
                                      'again with the "adapter" uri argument.')

    floyd_image = parse_result.args.get('floyd', None)

    if floyd_image is not None:
        if adapters_parsed:
            raise ImageSeedParseError(
                'Image seed IP adapter images not supported with floyd stage image.')

        if result.latents is not None:
            raise ImageSeedParseError(
                'Image seed latent tensors not supported with floyd stage image.')

        _ensure_exists(floyd_image, 'Floyd image')
        if control_images is not None:
            raise ImageSeedArgumentError(
                'The image seed "control" argument cannot be used with the "floyd" argument.')
        result.floyd_image = floyd_image

    resize = parse_result.args.get('resize', None)

    if resize is not None:
        try:
            dimensions = _textprocessing.parse_image_size(resize)
        except ValueError as e:
            raise ImageSeedArgumentError(
                f'Error parsing image seed "resize" argument: {e}')
        for d_idx, d in enumerate(dimensions):
            if d % align != 0:
                raise ImageSeedArgumentError(
                    f'Image seed resize {["width", "height"][d_idx]} dimension {d} is not divisible by {align}.')

        result.resize_resolution = dimensions

    user_align = parse_result.args.get('align', None)

    if user_align is not None:
        try:
            user_align = int(user_align)
        except ValueError as e:
            raise ImageSeedArgumentError(
                f'Error parsing image seed "align" argument: {e}.')
        if user_align % align != 0:
            raise ImageSeedArgumentError(
                f'Image seed resize alignment {user_align} is not divisible by {align}.')

        result.resize_align = user_align

    aspect = parse_result.args.get('aspect', None)

    if aspect is not None:
        try:
            aspect = _types.parse_bool(aspect)
        except ValueError:
            raise ImageSeedArgumentError(
                'Image seed aspect keyword argument must be a boolean value '
                'indicating if aspect correct resizing is enabled. '
                'received an un-parseable / non boolean value.')

    result.aspect_correct = aspect

    frame_start = parse_result.args.get('frame-start', None)
    frame_end = parse_result.args.get('frame-end', None)

    if frame_start is not None:
        try:
            frame_start = int(frame_start)
        except ValueError:
            raise ImageSeedArgumentError(
                f'frame_start argument must be an integer.')

    if frame_end is not None:
        try:
            frame_end = int(frame_end)
        except ValueError:
            raise ImageSeedArgumentError(
                f'frame_end argument must be an integer.')

        if frame_start is not None and (frame_start > frame_end):
            raise ImageSeedArgumentError(
                f'frame_start argument must be less '
                f'than or equal to frame_end argument.')

    result.frame_start = frame_start
    result.frame_end = frame_end

    # Validate that images and latents have equal counts when both are provided
    if result.images is not None and result.latents is not None:
        if len(result.images) != len(result.latents):
            raise ImageSeedParseError(
                f'Number of images ({len(result.images)}) must equal number of latents ({len(result.latents)}) '
                f'when both are provided. An equal batch size is required.')

    return result


def get_web_cache_directory() -> str:
    """
    Get the default web cache directory or the value of the environmental variable ``DGENERATE_WEB_CACHE``

    :return: string (directory path)
    """

    return _webcache.get_web_cache_directory()


class UnknownMimetypeError(Exception):
    """
    Raised when an unsupported mimetype is encountered
    """
    pass


def create_web_cache_file(url,
                          mime_acceptable_desc: str | None = None,
                          mimetype_is_supported: typing.Callable[[str], bool] | None = mimetype_is_supported,
                          local_files_only: bool = False) \
        -> tuple[str, str]:
    """
    Download a file from a url and add it to dgenerate's temporary web cache that is
    available to all concurrent dgenerate processes.

    If the file exists in the cache already, return information for the existing file.

    :param url: The url

    :param mime_acceptable_desc: a string describing what mimetype values are acceptable which is used
        when :py:exc:`.UnknownMimetypeError` is raised. If ``None`` is provided, this string will be
        generated using :py:func:`.get_supported_mimetypes`

    :param mimetype_is_supported: a function that test if a mimetype string is supported, if you
        supply the value ``None`` all mimetypes are considered supported.
        
    :param local_files_only: if ``True`` no downloads will be 
        allowed, only cached files and direct paths to files on disk.

    :raise UnknownMimetypeError: if a mimetype is considered not supported

    :raise requests.RequestException: Can raise any exception
        raised by ``requests.get`` for request related errors.

    :return: tuple(mimetype_str, filepath)
    """

    if mime_acceptable_desc is None:
        mime_acceptable_desc = _textprocessing.oxford_comma(get_supported_mimetypes(), conjunction='or')

    return _webcache.create_web_cache_file(
        url=url,
        mime_acceptable_desc=mime_acceptable_desc,
        mimetype_is_supported=mimetype_is_supported,
        unknown_mimetype_exception=UnknownMimetypeError,
        local_files_only=local_files_only
    )


def request_mimetype(url, local_files_only: bool = False) -> str:
    """
    Request the mimetype of a file at a URL, if the file exists in the cache, a known mimetype
    is returned without connecting to the internet. Otherwise, connect to the internet
    to retrieve the mimetype, this action does not update the cache.

    :param url: The url
    :param local_files_only: If ``True``, do not make a request, only check the cache.

    :raise dgenerate.webcache.WebFileCacheOfflineModeException:
        If the web cache is in offline mode and the file data is not found in the cache.

    :return: mimetype string
    """

    return _webcache.request_mimetype(url, local_files_only=local_files_only)


_MIME_TYPES_GUESS_EXTRA = {
    '.webp': 'image/webp',
    '.apng': 'image/apng',
    '.tga': 'image/tga',
    '.jp2': 'image/jp2',
    '.j2k': 'image/j2k',
    '.jpx': 'image/jpx',
    '.psd': 'image/psd'
}


def guess_mimetype(filename) -> str | None:
    """
    Guess the mimetype of a filename.

    The filename does not need to exist on disk.

    :param filename: the file name
    :return: mimetype string or ``None``
    """
    mime_type = mimetypes.guess_type(filename)[0]

    if mime_type is None:
        # Check for accepted formats that the mimetypes
        # stdlib does not know about by default

        _, ext = os.path.splitext(filename)
        if ext is not None:
            mime_type = _MIME_TYPES_GUESS_EXTRA.get(ext)

    return mime_type


MediaPathOpenerFunc = typing.Callable[[str], tuple[str, typing.BinaryIO]]


# noinspection HttpUrlsUsage
def fetch_media_data_stream(uri: str, local_files_only: bool = False) -> tuple[str, typing.BinaryIO]:
    """
    Get an open stream to a local file, or file at an HTTP or HTTPS URL, with caching for web files.

    Caching for downloaded files is multiprocess safe, multiple processes using this
    module can share the cache simultaneously, the last process alive clears the cache when it exits.

    :param uri: Local file path or URL
    :param local_files_only: If ``True`` no downloads will be allowed, 
        only cached files and direct paths to files on disk.

    :raise UnknownMimetypeError: If a remote file serves an unsupported mimetype value

    :return: (mime-type string, BinaryIO)
    """

    if is_downloadable_url(uri):
        mime_type, filename = create_web_cache_file(
            uri, local_files_only=local_files_only
        )
        return mime_type, open(filename, mode='rb')
    else:
        # Check if it's a tensor file first
        if is_tensor_file(uri):
            # For tensor files, we'll use a generic binary mimetype
            mime_type = 'application/octet-stream'
        else:
            mime_acceptable_desc = _textprocessing.oxford_comma(
                get_supported_mimetypes(), conjunction='or'
            )

            mime_type = guess_mimetype(uri)

            if mime_type is None:
                raise UnknownMimetypeError(
                    f'Mimetype could not be determined for file "{uri}". '
                    f'Expected: {mime_acceptable_desc}')

            if not mimetype_is_supported(mime_type):
                raise UnknownMimetypeError(
                    f'Unknown mimetype "{mime_type}" for file "{uri}". Expected: {mime_acceptable_desc}')

    return mime_type, open(uri, 'rb')


class MediaIdentificationError(Exception):
    """
    Raised when a media file is being loaded and it fails to load
    due to containing invalid or unexpected data.
    """
    pass


def create_image(
        path_or_file: typing.BinaryIO | str,
        file_source: str,
        resize_resolution: _types.OptionalSize = None,
        aspect_correct: bool = True,
        align: int | None = None) -> PIL.Image.Image:
    """
    Create an RGB format PIL image from a file path or binary file stream.
    The image is oriented according to any EXIF directives. Image is aligned
    to ``align`` in every case, specifying ``None`` or ``1`` for ``align``
    disables alignment.
    
    :raise MediaIdentificationError: If the image data is an unknown format or corrupt.

    :param path_or_file: file path or binary IO object
    :param file_source: :py:attr:`PIL.Image.Image.filename` is set to this value
    :param resize_resolution: Optional resize resolution
    :param aspect_correct: preserve aspect ratio when resize_resolution is specified?
    :param align: Align the image by this amount of pixels, ``None`` or ``1`` indicates no alignment.
    :return: :py:class:`PIL.Image.Image`
    """

    if isinstance(path_or_file, str):
        file = path_or_file
    else:
        file = path_or_file

    try:

        if resize_resolution is None:
            with PIL.Image.open(file) as img, _image.to_rgb(img) as rgb_img:
                e_img = _exif_orient(rgb_img)
                e_img.filename = file_source
                if not _image.is_aligned(e_img.size, align=align):
                    with e_img:
                        resized = _image.resize_image(
                            e_img,
                            size=None,
                            aspect_correct=True,
                            align=align
                        )
                        return resized
                else:
                    return e_img
        else:
            with PIL.Image.open(file) as img, _image.to_rgb(img) as rgb_img, _exif_orient(rgb_img) as o_img:
                o_img.filename = file_source
                resized = _image.resize_image(img=o_img,
                                              size=resize_resolution,
                                              aspect_correct=aspect_correct,
                                              align=align)
                return resized
    except PIL.UnidentifiedImageError:
        raise MediaIdentificationError(
            f'Error loading image file, unknown format or invalid data: "{file_source}"')


def create_animation_reader(mimetype: str,
                            file_source: str,
                            file: typing.BinaryIO,
                            resize_resolution: _types.OptionalSize = None,
                            aspect_correct: bool = True,
                            align: int | None = None,
                            image_processor: _imageprocessors.ImageProcessor | None = None,
                            ) -> AnimationReader:
    """
    Create an animation reader object from mimetype specification and binary file stream.

    Images will return a :py:class:`.MockImageAnimationReader` with a *total_frames* value of 1,
    which can then be adjusted by you.

    :py:class:`.VideoReader` or :py:class:`.AnimatedImageReader` will be returned for Video
    files and Animated Images respectively.

    :raises UnknownMimetypeError: on unknown ``mimetype`` value
    :raises MediaIdentificationError: If the file data is an unknown format or corrupt.

    :param mimetype: one of :py:func:`.get_supported_mimetypes`

    :param file: the binary file stream

    :param file_source: the source filename for videos/animated images, should be the filename.
        this is for informational purpose and should be provided in every case even if it is a
        symbolic value only. It should possess a file extension. :py:class:`PIL.Image.Image`
        objects produced by the reader will have this value set to their *filename* attribute.

    :param resize_resolution: Progressively resize each frame to this
        resolution while reading. The provided resolution will be aligned
        by ``align`` pixels.

    :param align: Align by this amount of pixels, if the input file is not aligned
        to this amount of pixels, it will be aligned by resizing. Passing ``None``
        or ``1`` disables alignment.

    :param aspect_correct: Should resize operations be aspect correct?

    :param image_processor: optionally process every frame with this image processor

    :return: :py:class:`.AnimationReader`
    """

    if mimetype_is_animated_image(mimetype):
        return AnimatedImageReader(file=file,
                                   file_source=file_source,
                                   resize_resolution=resize_resolution,
                                   aspect_correct=aspect_correct,
                                   align=align,
                                   image_processor=image_processor)
    elif mimetype_is_video(mimetype):
        return VideoReader(file=file,
                           file_source=file_source,
                           resize_resolution=resize_resolution,
                           aspect_correct=aspect_correct,
                           align=align,
                           image_processor=image_processor)
    elif mimetype_is_static_image(mimetype):
        with create_image(path_or_file=file,
                          file_source=file_source,
                          align=None) as img:
            return MockImageAnimationReader(img=img,
                                            resize_resolution=resize_resolution,
                                            aspect_correct=aspect_correct,
                                            image_processor=image_processor,
                                            align=align)
    else:
        supported = _textprocessing.oxford_comma(get_supported_mimetypes(), conjunction='or')
        raise UnknownMimetypeError(
            f'Unknown mimetype "{mimetype}" for file "{file_source}". Expected: {supported}')


class MediaReaderSpec:
    """
    Used by :py:class:`.MultiMediaReader` to define resource paths.
    """

    path: str
    """
    File path (or HTTP/HTTPS URL with default ``path_opener``)
    """

    image_processor: _imageprocessors.ImageProcessor | None = None
    """
    Optional image processor associated with the file.
    
    Note: Image processors are ignored for tensor files.
    """

    aspect_correct: bool = True
    """
    Aspect correct resize enabled?
    
    Note: Resize operations are ignored for tensor files.
    """

    align: int | None = None
    """
    Images which are read are aligned to this amount of pixels, ``None`` or ``1`` will disable alignment.
    
    Note: Alignment is ignored for tensor files.
    """

    resize_resolution: _types.OptionalSize = None
    """
    Optional resize resolution.
    
    Note: Resize operations are ignored for tensor files.
    """

    def __init__(self, path: str,
                 image_processor: _imageprocessors.ImageProcessor | None = None,
                 resize_resolution: _types.OptionalSize = None,
                 aspect_correct: bool = True,
                 align: int | None = None):
        """
        :param path: File path or URL
        :param resize_resolution: Resize resolution (ignored for tensor files)
        :param aspect_correct: Aspect correct resize enabled? (ignored for tensor files)
        :param align: Images which are read are aligned to this amount of pixels,
            ``None`` or ``1`` will disable alignment. (ignored for tensor files)
        :param image_processor: Optional image processor associated with the file (ignored for tensor files)

        :raise ValueError: On ``align < 1``
        """

        if align is not None and align < 1:
            raise ValueError('align argument may not be less than 1.')

        self.aspect_correct = aspect_correct
        self.resize_resolution = resize_resolution
        self.path = path
        self.image_processor = image_processor
        self.align = align


class FrameStartOutOfBounds(ValueError):
    """
    Raised by :py:class:`.MultiMediaReader` when the provided ``frame_start``
    frame slicing value is calculated to be out of bounds.
    """
    pass


class MultiMediaReader:
    """
    Zips together multiple automatically created :py:class:`.AnimationReader` implementations and
    allows enumeration over their reads, which are collected into a list of a defined order.

    Images when zipped together with animated files will be repeated over the total amount of frames.

    The animation with the lowest amount of frames determines the total amount of
    frames that can be read when animations are involved.

    If all paths point to images, then :py:attr:`.MultiMediaReader.total_frames` will be 1.

    There is no guarantee that images read from the individual readers are the same size
    and you must handle that condition.
    """

    _readers: list[AnimationReader]
    _file_streams: list[typing.BinaryIO]
    _total_frames: int = 0
    _frame_start: int = 0
    _frame_end: _types.OptionalInteger = None
    _frame_index: int = -1

    def width(self, idx) -> int:
        """
        Width dimension, (X dimension) of a specific reader index.

        :return: width
        """
        return self._readers[idx].width

    def size(self, idx) -> _types.Size:
        """
        returns (width, height) as a tuple of a specific reader index.

        :return: (width, height)
        """
        return self._readers[idx].width, self._readers[idx].height

    def height(self, idx) -> int:
        """
        Height dimension, (Y dimension) of a specific reader index.

        :return: height
        """
        return self._readers[idx].height

    @property
    def frame_index(self) -> int:
        """
        Current frame index while reading.
        """
        return self._frame_index - self._frame_start

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
    def fps(self) -> _types.OptionalFloat:
        """
        Frames per second, this will be None if there is only a single frame
        """
        return self._fps

    @property
    def frame_duration(self) -> _types.OptionalFloat:
        """
        Duration of a frame in milliseconds, this will be None if there is only a single frame
        """
        return self._frame_duration

    def __init__(self,
                 specs: list[MediaReaderSpec],
                 frame_start: int = 0,
                 frame_end: _types.OptionalInteger = None,
                 path_opener: MediaPathOpenerFunc = fetch_media_data_stream):
        """
        :raise ValueError: if ``frame_start > frame_end``
        :raise FrameStartOutOfBounds: if ``frame_start > total_frames - 1``

        :param specs: list of :py:class:`.MediaReaderSpec`
        :param frame_start: inclusive frame slice start frame
        :param frame_end: inclusive frame slice end frame
        :param path_opener: opens a binary file stream from paths
            mentioned by :py:class:`.MediaReaderSpec`
        """

        if frame_end is not None:
            if frame_start > frame_end:
                raise ValueError('frame_start must be less than or equal to frame_end')

        self._readers = []
        self._file_streams = []

        self._frame_start = frame_start
        self._frame_end = frame_end
        self._frame_duration = None
        self._fps = None

        for spec in specs:
            if is_tensor_file(spec.path):
                # Handle tensor files directly
                mimetype, file_stream = path_opener(spec.path)
                tensor = load_tensor_file(file_stream, spec.path)
                file_stream.close()  # Close immediately after loading tensor

                # debug ignored operations on tensor files
                ignored_operations = []

                if spec.resize_resolution is not None:
                    ignored_operations.append(f"resize ({spec.resize_resolution[0]}x{spec.resize_resolution[1]})")

                if not spec.aspect_correct:  # Check if non-default value was set
                    ignored_operations.append(f"aspect ({spec.aspect_correct})")

                if spec.align is not None and spec.align != 1:  # Check if non-default value was set
                    ignored_operations.append(f"align ({spec.align})")

                if spec.image_processor is not None:
                    ignored_operations.append(f"image processor ({str(spec.image_processor)})")

                if ignored_operations:
                    operations_str = ", ".join(ignored_operations)
                    _messages.debug_log(
                        f'Tensor file "{url_aware_basename(spec.path)}" is latents input - '
                        f'mediainput module ignoring {operations_str} (tensor files are loaded as-is)'
                    )

                # Create a mock reader that yields the tensor
                self._readers.append(
                    MockTensorReader(tensor=tensor, file_source=spec.path)
                )
            else:
                # Handle regular media files
                mimetype, file_stream = path_opener(spec.path)

                self._readers.append(
                    create_animation_reader(
                        mimetype=mimetype,
                        file_source=spec.path,
                        file=file_stream,
                        resize_resolution=spec.resize_resolution,
                        aspect_correct=spec.aspect_correct,
                        align=spec.align,
                        image_processor=spec.image_processor)
                )
                self._file_streams.append(file_stream)

        non_images = [r for r in self._readers if not isinstance(r, MockImageAnimationReader)]

        if non_images:
            self._total_frames = min(
                non_images, key=lambda r: r.total_frames).total_frames

            self._total_frames = frame_slice_count(self.total_frames, frame_start, frame_end)

            self._fps = non_images[0].fps
            self._frame_duration = non_images[0].frame_duration

            for r in self._readers:
                if isinstance(r, MockImageAnimationReader):
                    r.total_frames = self.total_frames
        else:
            self._total_frames = 1

        if self.frame_start > self.total_frames - 1:
            raise FrameStartOutOfBounds(
                f'Frame slice start value {self.frame_start} is out of bounds, '
                f'total frame count is {self.total_frames}. Value must be less '
                f'than ({self.total_frames} minus 1).')

    def __next__(self):
        if self._frame_index == -1:
            # First call, skip up to frame start
            for idx in range(0, self._frame_start):
                for r in self._readers:
                    if isinstance(r, _imageprocessors.ImageProcessorMixin):
                        old_val = r.image_processor_enabled
                        r.image_processor_enabled = False
                        try:
                            r.__next__().close()
                        finally:
                            r.image_processor_enabled = old_val
                    else:
                        r.__next__().close()

                if self._frame_index == self._frame_end:
                    # This should only be able to happen if frame_start > frame_end
                    # which is checked for in the constructor
                    raise AssertionError(
                        'impossible iteration termination condition '
                        'in MultiMediaReader reader')

                self._frame_index += 1

        if self._frame_index == self._frame_end:
            raise StopIteration

        read = [r.__next__() for r in self._readers]

        self._frame_index += 1

        return read

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for r in self._readers:
            r.__exit__(exc_type, exc_val, exc_tb)
        for file_stream in self._file_streams:
            file_stream.close()


class MediaReader(AnimationReader):
    """
    Thin wrapper around :py:class:`.MultiMediaReader` which simply reads
    from a single file instead of multiple files simultaneously.

    The interface provided by this object is that of :py:class:`.AnimationReader`

    This object can read any media supported by dgenerate for input and
    supports frame slicing animated media formats and image processors.

    Static images are treated as an animation with a single frame.

    With the default path opener, URLs will be downloaded,
    dgenerate's temporary web cache will be utilized.
    """

    @property
    def frame_index(self) -> int:
        """
        Current frame index while reading.
        """
        return self._reader.frame_index

    @property
    def frame_end(self) -> int:
        """
        Frame slice end value (inclusive)
        """
        return self._reader.frame_end

    @property
    def frame_start(self) -> int:
        """
        Frame slice start value (inclusive)
        """
        return self._reader.frame_start

    def __init__(self,
                 path: str,
                 image_processor: _imageprocessors.ImageProcessor | None = None,
                 resize_resolution: _types.OptionalSize = None,
                 aspect_correct: bool = True,
                 align: int | None = None,
                 frame_start: int = 0,
                 frame_end: _types.OptionalInteger = None,
                 path_opener: MediaPathOpenerFunc = fetch_media_data_stream):
        """

        :raise ValueError: if ``frame_start > frame_end``
        :raise FrameStartOutOfBounds: if ``frame_start > total_frames - 1``

        :param path: File path or URL
        :param resize_resolution: Resize resolution
        :param aspect_correct: Aspect correct resize enabled?
        :param align: Images which are read are aligned to this amount of pixels, ``None`` or ``1`` will disable alignment.
        :param image_processor: Optional image processor associated with the file
        :param frame_start: inclusive frame slice start frame
        :param frame_end: inclusive frame slice end frame
        :param path_opener: opens a binary file stream from paths.

        """

        self._reader = MultiMediaReader(
            [MediaReaderSpec(
                path=path,
                image_processor=image_processor,
                resize_resolution=resize_resolution,
                aspect_correct=aspect_correct,
                align=align)],
            frame_start=frame_start,
            frame_end=frame_end,
            path_opener=path_opener)

        super().__init__(
            width=self._reader.width(0),
            height=self._reader.height(0),
            fps=self._reader.fps,
            frame_duration=self._reader.frame_duration,
            total_frames=self._reader.total_frames)

    def __next__(self) -> PIL.Image.Image:
        return self._reader.__next__()[0]

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._reader.__exit__(exc_type, exc_val, exc_tb)


class ImageSeed:
    """
    An ImageSeed with attached image data
    """

    uri: str
    """
    The original URI string that this image seed originates from.
    """

    frame_index: _types.OptionalInteger = None
    """
    Frame index in the case that :py:attr:`.ImageSeed.is_animation_frame` is True
    """

    total_frames: _types.OptionalInteger = None
    """
    Total frame count in the case that :py:attr:`.ImageSeed.is_animation_frame` is True
    """

    fps: _types.OptionalFloat = None
    """
    Frames per second in the case that :py:attr:`.ImageSeed.is_animation_frame` is True
    """

    frame_duration: _types.OptionalFloat = None
    """
    Duration of a frame in milliseconds in the case that :py:attr:`.ImageSeed.is_animation_frame` is True
    """

    images: _types.Images | None
    """
    An optional images used for img2img mode, or inpainting mode in combination with :py:attr:`.ImageSeed.mask_images`.
    
    May be ``None`` when using IP Adapter only images, IE: the ``adapter: ...`` uri syntax.
    
    May also be ``None`` when using latents only input, IE: the ``latents: ...`` uri syntax.
    """

    mask_images: _types.Images | None
    """
    An optional inpaint mask images, may be ``None``.
    """

    control_images: _types.Images | None
    """
    Control guidance images, or None.
    """

    adapter_images: collections.abc.Sequence[_types.Images] | None
    """
    IP Adapter images, or None.
    """

    latents: _types.Tensors | None
    """
    Raw latent tensors loaded from .pt, .pth, or .safetensors files, or None.
    
    These tensors are loaded as-is without any image processing, resizing, or alignment operations.
    """

    floyd_image: PIL.Image.Image | None
    """
    An optional image from a Deep Floyd IF stage, used for disambiguation in the case 
    of using Deep Floyd for img2img and inpainting, where the un-varied input image
    is needed as a parameter for both stages. This image is used to define the image 
    that was generated by Deep Floyd in a previous stage and to be used in the next stage, 
    where :py:attr:`ImageSeed.image` defines the img2img image that you want a variation of.
    
    This image will never be assigned a value when :py:attr:`ImageSeed.control_images` has a
    a value. As that is considered incorrect --image-seeds 
    """

    is_animation_frame: bool
    """
    Is this part of an animation?
    """

    def __init__(self,
                 images: PIL.Image.Image | None = None,
                 mask_images: PIL.Image.Image | None = None,
                 control_images: _types.Images | None = None,
                 floyd_image: PIL.Image.Image | None = None,
                 adapter_images: list[_types.Images] | None = None,
                 latents: _types.Tensors | None = None):
        self.images = images
        self.mask_images = mask_images

        if control_images is not None and floyd_image is not None:
            raise ValueError(
                'control_images and floyd_image arguments are incompatible '
                'and cannot both be specified')

        if adapter_images is not None and floyd_image is not None:
            raise ValueError(
                'adapter_images and floyd_image arguments are incompatible '
                'and cannot both be specified')

        if latents is not None and floyd_image is not None:
            raise ValueError(
                'latents and floyd_image arguments are incompatible '
                'and cannot both be specified')

        self.control_images = control_images
        self.floyd_image = floyd_image
        self.adapter_images = adapter_images
        self.latents = latents

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.images is not None:
            for i in self.images:
                if hasattr(i, 'close'):
                    i.close()

        if self.mask_images is not None:
            for i in self.mask_images:
                i.close()

        if self.control_images:
            for i in self.control_images:
                i.close()

        if self.adapter_images:
            for group in self.adapter_images:
                for image in group:
                    image.close()

        # Tensors don't need explicit cleanup, but we clear the reference
        if self.latents:
            self.latents = None


def _check_image_dimensions_match(images):
    """
    Check that all images have matching dimensions.
    
    Note: This function only checks PIL Images against other PIL Images.
    Tensor dimension validation is handled by the pipeline wrapper using VAE scale factors.
    """

    def get_dimensions(item):
        return item.size

    def get_filename(item):
        return _image.get_filename(item)

    # Only check PIL images - tensors are validated by the pipeline wrapper
    pil_images = [item for item in images if not _torchutil.is_tensor(item)]

    # Check dimensions within PIL images only
    for ix in pil_images:
        ix_dims = get_dimensions(ix)
        ix_filename = get_filename(ix)

        for iy in pil_images:
            iy_dims = get_dimensions(iy)
            iy_filename = get_filename(iy)

            if ix_dims != iy_dims:
                raise ImageSeedSizeMismatchError(
                    f'Dimension of "{ix_filename}" ({_textprocessing.format_size(ix_dims)}) does '
                    f'not match "{iy_filename}" ({_textprocessing.format_size(iy_dims)})')


def _flatten(xs):
    for x in xs:
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, (str, bytes)) and not _torchutil.is_tensor(x):
            yield from _flatten(x)
        else:
            yield x


ImageProcessorSpec = _imageprocessors.ImageProcessor | \
                     collections.abc.Sequence[_imageprocessors.ImageProcessor] | None


def _validate_image_processor_count(processors, images, error_title):
    num_processors = len(processors)
    num_images = len(images)
    if num_processors > num_images:
        raise ImageSeedError(f'Too many {error_title} image processors specified, '
                             f'there are {num_processors} processors and '
                             f'{num_images} {error_title} image sources.')


def _reshape_ip_adapter_image_seed(adapter_images, reader_output):
    reshaped_list = []
    start_index = 0
    for sublist in adapter_images:
        sublist_length = len(sublist)
        reshaped_sublist = reader_output[start_index:start_index + sublist_length]
        reshaped_list.append(reshaped_sublist)
        start_index += sublist_length

    return reshaped_list


def iterate_image_seed(uri: str | ImageSeedParseResult,
                       frame_start: int = 0,
                       frame_end: _types.OptionalInteger = None,
                       resize_resolution: _types.OptionalSize = None,
                       aspect_correct: bool = True,
                       align: int | None = None,
                       seed_image_processor: ImageProcessorSpec = None,
                       mask_image_processor: ImageProcessorSpec = None,
                       control_image_processor: ImageProcessorSpec = None,
                       check_dimensions_match: bool = True,
                       path_opener: MediaPathOpenerFunc = fetch_media_data_stream) -> \
        collections.abc.Iterator[ImageSeed]:
    """
    Parse and load images/videos/tensors in an ``--image-seeds`` uri and return an iterator that
    produces :py:class:`.ImageSeed` objects while progressively reading those files.

    This method is used to iterate over an ``--image-seeds`` uri in the case that the image source
    mentioned is to be used for img2img / inpaint operations, and handles this syntax:

        * ``--image-seeds "img2img.png"``
        * ``--image-seeds "img2img.png;mask.png"``
        * ``--image-seeds "img2img.png;mask.png;512x512"``
        * ``--image-seeds "images: img2img-1.png, img2img-2.png"``
        * ``--image-seeds "images: img2img-1.png, img2img-2.png; mask1.png, mask2.png"``
        * ``--image-seeds "images: img2img-1.png, img2img-2.png; mask1.png, mask2.png;512"``

    Additionally, controlnet guidance resources are handled with keyword arguments:

        * ``--image-seeds "img2img.png;control=control1.png, control2.png"``
        * ``--image-seeds "img2img.png;control=control1.png, control2.png;resize=512x512"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control1.png, control2.png"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control1.png, control2.png;resize=512x512"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control1.png;frame-start=2"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control1.png;frame-start=2;frame-end=5"``
        * ``--image-seeds "images: img2img-1.png, img2img-2.png;control=control1.png, control2.png"``
        * ``--image-seeds "images: img2img-1.png, img2img-2.png;mask=mask1.png, mask2.png;control=control1.png, control2.png"``

    IP Adapter Images can be specified in these ways:

        * ``--image-seeds "adapter: image.png"``
        * ``--image-seeds "adapter: adapter1-image.png, adapter2-image.png"``
        * ``--image-seeds "adapter: image1.png + image2.png"``
        * ``--image-seeds "adapter: adapter1-image1.png + adapter1-image2.png, adapter2-image1.png + adapter2-image2.png"``
        * ``--image-seeds "img2img.png;adapter=image.png"``
        * ``--image-seeds "img2img.png;adapter=adapter1-image.png, adapter2-image.png"``
        * ``--image-seeds "img2img.png;adapter=image1.png + image2.png"``
        * ``--image-seeds "img2img.png;adapter=adapter1-image1.png + adapter1-image2.png, adapter2-image1.png + adapter2-image2.png"``
        * ``--image-seeds "images: img2img-1.png, img2img-2.png;adapter=image.png"``

    Raw (noisy) latents can be specified in these ways:

        * ``--image-seeds "latents: latents1.pt, latents2.pt"``
        * ``--image-seeds "img2img.png;latents=latents.pt"``
        * ``--image-seeds "images: img2img1.png, img2img2.png;latents=latents1.pt, latents2.pt"``
        * ``--image-seeds "latents: latents.safetensors;control=control.png"``

    The ``control`` argument is supported for any IP Adapter image specification.

    The ``mask`` argument is also supported for img2img with additional IP Adapter images.

    Deep Floyd img2img and inpainting mode can be specified in these ways:

        * ``--image-seeds "img2img.png;floyd=stage1-image.png"``
        * ``--image-seeds "img2img.png;mask=mask.png;floyd=stage2-image.png"``

    Note that all keyword arguments mentioned above can be used together, except for
    ``control`` and ``floyd``, ``adapter`` and ``floyd``, or ``latents`` and ``floyd``,
    which are mutually exclusive arguments.

    For ``img2img`` sources, you may also specify a ``pt``, ``pth``, or ``safetensors`` file,
    this is for passing in latents in place of images in pixel space, image processing will not be
    applied to these inputs and will be ignored with warnings, this includes resizing, aspect
    correction, alignment, and image processors. Latents can be generated by using the
    option ``--image-format`` with the value ``pt``, ``pth``,  or ``safetensors``.

    One or more :py:class:`.ImageSeed` objects may be yielded depending on whether an animation is being read.

    :param uri: ``--image-seeds`` uri or :py:class:`.ImageSeedParseResult`
    :param frame_start: starting frame, inclusive value
    :param frame_end: optional end frame, inclusive value

    :param resize_resolution: optional global resize resolution. The URI syntax of image seeds
        allows for overriding this value.

    :param aspect_correct: should the global resize operation be aspect correct by default?
        The URI syntax for image seeds allows for overriding this value with the **aspect**
        keyword argument.

    :param align: Images which are read are aligned to this amount of pixels, ``None`` or ``1`` will disable alignment.

    :param seed_image_processor: optional :py:class:`dgenerate.imageprocessors.ImageProcessor` or list of them.
        A list is used to specify processors for individual images in a multi img2img image specification
        such as uri = "images: img2img-1.png, img2img-2.png". In the case that a multi img2img image
        specification is used and only one processor is given, that processor will be used on only the
        first image / video in the specification. Images in a multi img2img specification with no corresponding
        processor value will have their processor set to ``None``, specifying extra processors
        as compared to img2img sources will cause :py:exc:`ValueError` to be raised.

    :param mask_image_processor: optional :py:class:`dgenerate.imageprocessors.ImageProcessor` or list of them.
        A list is used to specify processors for individual mask images in a multi inpaint mask specification
        such as uri = "images: img2img-1.png, img2img-2.png;mask=mask-1.png, mask-2.png". In the case that
        a multi inpaint mask specification is used and only one processor is given, that processor will
        be used on only the first image / video in the specification. Images in an inpaint mask specification
        with no corresponding processor value will have their processor set to ``None``, specifying extra
        processors as compared to inpaint mask image sources will cause :py:exc:`ValueError` to be raised.

    :param control_image_processor: optional :py:class:`dgenerate.imageprocessors.ImageProcessor` or list of them.
        A list is used to specify processors for individual images in a multi guidance image specification
        such as uri = "img2img.png;control=img1.png, img2.png". In the case that a multi guidance image
        specification is used and only one processor is given, that processor will be used on only the
        first image / video in the specification. Images in a guidance specification with no corresponding
        processor value will have their processor set to ``None``, specifying extra processors
        as compared to control guidance image sources will cause :py:exc:`ValueError` to be raised.

    :param check_dimensions_match: Check the dimensions of input images, mask images,
        and control images to confirm that they match? For pipelines like stable cascade,
        this does not matter, input images can be any dimension as they are used as a
        style reference and not a noise base similar to IP Adapters.

    :param path_opener: a function that opens a file stream from a path,
        defaults to :py:func:`dgenerate.media.fetch_media_data_stream`.

    :raise ImageSeedError: if multiple images are passed without using
        the ``"images: ..."`` syntax for batching. Or if the ``"adapter: ..."``
        syntax is used with the ``floyd`` keyword argument for floyd stage images.
        Or if too many image processor chains are specified for the amount of
        images given.

    :return: an iterator over :py:class:`.ImageSeed` objects
    """

    if frame_end is not None:
        if frame_start > frame_end:
            raise ValueError('frame_start must be less than or equal to frame_end')

    if align is not None and align < 1:
        raise ValueError('align argument may not be less than 1.')

    if isinstance(uri, ImageSeedParseResult):
        parse_result = uri
    else:
        parse_result = parse_image_seed_uri(uri)

    if parse_result.images is not None and \
            len(parse_result.images) > 1 and \
            not parse_result.multi_image_mode:
        raise ImageSeedParseError(
            'Image seed img2img input cannot contain multiple elements '
            'without using "images: ..." batching syntax. This syntax is reserved '
            'for ControlNet input.')

    if parse_result.resize_resolution is not None:
        resize_resolution = parse_result.resize_resolution

    if parse_result.resize_align is not None:
        align = parse_result.resize_align

    if parse_result.aspect_correct is not None:
        aspect_correct = parse_result.aspect_correct

    reader_specs = []
    image_ranges = []

    def append_range(name, length):
        if image_ranges:
            start_range = image_ranges[-1][1][1]
        else:
            start_range = 0

        image_ranges.append((name, (start_range, start_range + length)))

    if parse_result.images is not None:
        if not isinstance(seed_image_processor, list):
            seed_image_processor = [seed_image_processor]

        seed_paths = parse_result.images

        _validate_image_processor_count(
            processors=seed_image_processor,
            images=seed_paths,
            error_title='seed')

        for idx, seed_path in enumerate(seed_paths):
            reader_specs.append(
                MediaReaderSpec(
                    path=seed_path,
                    image_processor=seed_image_processor[idx] if idx < len(seed_image_processor) else None,
                    resize_resolution=resize_resolution,
                    aspect_correct=aspect_correct,
                    align=align))

        append_range('images', len(seed_paths))

    if parse_result.mask_images is not None:
        if not isinstance(mask_image_processor, list):
            mask_image_processor = [mask_image_processor]

        mask_paths = parse_result.mask_images

        _validate_image_processor_count(
            processors=mask_image_processor,
            images=mask_paths,
            error_title='inpaint mask')

        for idx, mask_path in enumerate(mask_paths):
            reader_specs.append(
                MediaReaderSpec(
                    path=mask_path,
                    image_processor=mask_image_processor[idx] if idx < len(mask_image_processor) else None,
                    resize_resolution=resize_resolution,
                    aspect_correct=aspect_correct,
                    align=align))

        append_range('mask_images', len(mask_paths))

    if parse_result.control_images is not None:
        if not isinstance(control_image_processor, list):
            control_image_processor = [control_image_processor]

        control_paths = parse_result.control_images

        _validate_image_processor_count(
            processors=control_image_processor,
            images=control_paths,
            error_title='control guidance')

        for idx, control_path in enumerate(control_paths):
            reader_specs.append(
                MediaReaderSpec(
                    path=control_path,
                    image_processor=control_image_processor[idx] if idx < len(control_image_processor) else None,
                    resize_resolution=resize_resolution,
                    aspect_correct=aspect_correct,
                    align=align))

        append_range('control_images', len(control_paths))

    if parse_result.adapter_images is not None and parse_result.floyd_image is not None:
        raise ImageSeedError('IP adapter images not supported with floyd stage image.')

    if parse_result.adapter_images is not None:
        adapter_image_cnt = 0
        for adapter_images in parse_result.adapter_images:
            for image in adapter_images:
                reader_specs.append(
                    MediaReaderSpec(
                        path=image.path.strip(),
                        resize_resolution=image.resize,
                        aspect_correct=image.aspect,
                        align=image.align
                    )
                )
                adapter_image_cnt += 1

        append_range('adapter_images', adapter_image_cnt)

    elif parse_result.floyd_image is not None:
        # There should never be a reason to process floyd stage output
        # also do not resize it
        reader_specs.append(MediaReaderSpec(
            path=parse_result.floyd_image,
            resize_resolution=None,
            align=None))

        append_range('floyd_image', 1)

    if parse_result.latents is not None:
        latent_paths = parse_result.latents
        latent_cnt = 0
        for latent_path in latent_paths:
            # Latent tensors don't need image processing, resizing, or alignment
            reader_specs.append(
                MediaReaderSpec(
                    path=latent_path,
                    resize_resolution=None,
                    align=None))
            latent_cnt += 1

        append_range('latents', latent_cnt)

    if parse_result.frame_start is not None:
        frame_start = parse_result.frame_start

    if parse_result.frame_end is not None:
        frame_end = parse_result.frame_end

    with MultiMediaReader(specs=reader_specs,
                          frame_start=frame_start,
                          frame_end=frame_end,
                          path_opener=path_opener) as reader:

        is_animation = reader.total_frames > 1

        for frame in reader:

            image_seed_args = dict()

            for key, value in image_ranges:
                start = value[0]
                end = value[1]
                if key == 'adapter_images':
                    # noinspection PyTypeChecker
                    image_seed_args[key] = _reshape_ip_adapter_image_seed(
                        parse_result.adapter_images, frame[start:end])
                elif key == 'floyd_image':
                    # floyd_image should be a single image, not a list
                    image_seed_args[key] = frame[start:end][0]
                else:
                    image_seed_args[key] = frame[start:end]

            image_seed = ImageSeed(**image_seed_args)

            images = list(_flatten([image_seed.images if image_seed.images else [],
                                    image_seed.mask_images if image_seed.mask_images else [],
                                    image_seed.control_images if image_seed.control_images else []]))

            if check_dimensions_match:
                _check_image_dimensions_match(images)

            image_seed.is_animation_frame = is_animation

            if is_animation:
                image_seed.fps = reader.fps
                image_seed.frame_duration = reader.frame_duration
                image_seed.frame_index = reader.frame_index
                image_seed.total_frames = reader.total_frames if is_animation else None

            image_seed.uri = parse_result.uri
            yield image_seed


def iterate_control_image(uri: str | ImageSeedParseResult,
                          frame_start: int = 0,
                          frame_end: _types.OptionalInteger = None,
                          resize_resolution: _types.OptionalSize = None,
                          aspect_correct: bool = True,
                          align: int | None = None,
                          image_processor: ImageProcessorSpec = None,
                          path_opener: MediaPathOpenerFunc = fetch_media_data_stream) -> \
        collections.abc.Iterator[ImageSeed]:
    """
    Parse and load a control image/video in an ``--image-seeds`` uri and return an iterator that
    produces :py:class:`.ImageSeed` objects while progressively reading that file.

    One or more :py:class:`.ImageSeed` objects may be yielded depending on whether an animation is being read.

    This can consist of a single resource path or a list of comma separated image and
    video resource paths, which may be files on disk or remote files (http / https).

    This method is to be used when it is known that there is only a controlnet guidance resource
    specification in the path, and it handles this specification syntax:

        * ``--image-seeds "control1.png"``
        * ``--image-seeds "control1.png, control2.png"``
        * ``--image-seeds "control1.png, control2.png;512x512"``
        * ``--image-seeds "control1.png, control2.png;resize=512x512"``
        * ``--image-seeds "control1.png, control2.png;frame-start=2"``
        * ``--image-seeds "control1.png, control2.png;frame-start=2;frame-end=10"``
        * ``--image-seeds "control1.png, control2.png;resize=512x512;frame-start=2;frame-end=10"``

    The image or images read will be available from the :py:attr:`.ImageSeed.control_images` attribute.

    :param uri: ``--image-seeds`` uri or :py:class:`.ImageSeedParseResult`
    :param frame_start: starting frame, inclusive value
    :param frame_end: optional end frame, inclusive value

    :param resize_resolution: optional global resize resolution. The URI syntax of image seeds
        allows for overriding this value.

    :param aspect_correct: should the global resize operation be aspect correct by default?
        The URI syntax for image seeds allows for overriding this value with the **aspect**
        keyword argument.

    :param align: Images which are read are aligned to this amount of pixels, ``None`` or ``1`` will disable alignment.

    :param image_processor: optional :py:class:`dgenerate.imageprocessors.ImageProcessor` or list of them.
        A list is used to specify processors for individual images in a multi guidance image specification
        such as uri = "img1.png, img2.png". In the case that a multi guidance image specification is used and only
        one processor is given, that processor will be used on only the first image / video in the specification.
        Images in a guidance specification with no corresponding processor value will have their processor
        set to ``None``, specifying extra processors as compared to control guidance image sources will
        cause :py:exc:`ValueError` to be raised.

    :param path_opener: opens a binary file stream from paths, defaults to :py:func:`dgenerate.fetch_media_data_stream`.

    :raise ImageSeedError: If any other image inputs are specified, such as ``mask``, ``control``, or ``floyd``.
        Or if a tensor file is passed in a control guidance image specification, latents input is not supported
        for controlnet guidance images. Or if too many image processor chains are specified for the amount of
        images given.

    :raise ValueError: On ``frame_start > frame_end``, or ``align < 1``

    :return: an iterator over :py:class:`.ImageSeed` objects
    """

    if frame_end is not None:
        if frame_start > frame_end:
            raise ValueError('frame_start must be less than or equal to frame_end')

    if align is not None and align < 1:
        raise ValueError('align argument may not be less than 1.')

    if isinstance(uri, ImageSeedParseResult):
        parse_result = uri
    else:
        parse_result = parse_image_seed_uri(uri)

    if not parse_result.is_single_spec:
        raise ImageSeedError(
            f'Control guidance only image seed uri "{uri}" should not define '
            f'any other image source arguments such as "mask" or "control" or "floyd", '
            f'only a single specification is needed.')

    if parse_result.resize_resolution is not None:
        resize_resolution = parse_result.resize_resolution

    if parse_result.resize_align is not None:
        align = parse_result.resize_align

    if parse_result.aspect_correct is not None:
        aspect_correct = parse_result.aspect_correct

    reader_specs = []

    if not isinstance(image_processor, list):
        image_processor = [image_processor]

    control_guidance_image_paths = parse_result.get_control_image_paths()

    # Check for tensor files
    for path in control_guidance_image_paths:
        if is_tensor_file(path):
            raise ImageSeedError(
                f'Control image path "{path}" is a tensor file. ControlNet/T2I-Adapter expect images in pixel space, '
                f'tensor files are not supported for control images.')

    _validate_image_processor_count(
        processors=image_processor,
        images=control_guidance_image_paths,
        error_title='control guidance')

    reader_specs += [
        MediaReaderSpec(
            path=p.strip(),
            image_processor=image_processor[idx] if idx < len(image_processor) else None,
            resize_resolution=resize_resolution,
            aspect_correct=aspect_correct,
            align=align)
        for idx, p in enumerate(control_guidance_image_paths)
    ]

    if parse_result.frame_start is not None:
        frame_start = parse_result.frame_start

    if parse_result.frame_end is not None:
        frame_end = parse_result.frame_end

    with MultiMediaReader(specs=reader_specs,
                          frame_start=frame_start,
                          frame_end=frame_end,
                          path_opener=path_opener) as reader:

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
                image_seed.fps = reader.fps
                image_seed.frame_duration = reader.frame_duration
                image_seed.frame_index = reader.frame_index
                image_seed.total_frames = reader.total_frames if is_animation else None

            image_seed.uri = parse_result.uri
            yield image_seed


class ImageSeedInfo:
    """Information acquired about an ``--image-seeds`` uri"""

    fps: _types.OptionalFloat
    """
    Animation frames per second in the case that :py:attr:`.ImageSeedInfo.is_animation` is True
    """

    frame_duration: _types.OptionalFloat
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
                 fps: _types.OptionalFloat,
                 frame_duration: _types.OptionalFloat):
        self.fps = fps
        self.frame_duration = frame_duration
        self.is_animation = is_animation
        self.total_frames = total_frames


def get_image_seed_info(uri: _types.Uri | ImageSeedParseResult,
                        frame_start: int = 0,
                        frame_end: _types.OptionalInteger = None,
                        path_opener: MediaPathOpenerFunc = fetch_media_data_stream) -> ImageSeedInfo:
    """
    Get an informational object from a dgenerate ``--image-seeds`` uri.

    :param uri: The uri string or :py:class:`.ImageSeedParseResult`
    :param frame_start: slice start
    :param frame_end: slice end
    :param path_opener: a function that opens a file stream from a path, defaults to :py:func:`dgenerate.media.fetch_media_data_stream`.
    :return: :py:class:`.ImageSeedInfo`
    """

    with next(iterate_image_seed(uri, frame_start, frame_end, path_opener=path_opener, check_dimensions_match=False)) as seed:
        return ImageSeedInfo(seed.is_animation_frame, seed.total_frames, seed.fps, seed.frame_duration)


def get_control_image_info(uri: _types.Path | ImageSeedParseResult,
                           frame_start: int = 0,
                           frame_end: _types.OptionalInteger = None,
                           path_opener: MediaPathOpenerFunc = fetch_media_data_stream) -> ImageSeedInfo:
    """
    Get an informational object from a dgenerate ``--image-seeds`` uri that is known to be a
    control image/video specification.

    This can consist of a single resource path or a list of comma separated image and
    video resource paths, which may be files on disk or remote files (http / https).

    This method is to be used when it is known that there is only a control image/video specification in the path,
    and it handles this specification syntax:

        * ``--image-seeds "control1.png"``
        * ``--image-seeds "control1.png, control2.png"``

    :param uri: The path string or :py:class:`.ImageSeedParseResult`
    :param frame_start: slice start
    :param frame_end: slice end
    :param path_opener: a function that opens a file stream from a path, defaults to :py:func:`dgenerate.media.fetch_media_data_stream`.
    :return: :py:class:`.ImageSeedInfo`
    """
    with next(iterate_control_image(uri, frame_start, frame_end, path_opener=path_opener)) as seed:
        return ImageSeedInfo(seed.is_animation_frame, seed.total_frames, seed.fps, seed.frame_duration)
