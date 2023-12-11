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
import re
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
import dgenerate.imageprocessors as _imageprocessors
import dgenerate.messages as _messages
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types

__doc__ = """
Media input, handles reading videos/animations and static images, and creating readers from image seed URIs.

Also provides media download capabilities and temporary caching of web based files.

Provides information about supported input formats.
"""


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
    def fps(self) -> float:
        """
        Frame per second.

        :return: float or integer
        """
        return self._fps

    @property
    def frame_duration(self) -> float:
        """
        Duration of each frame in milliseconds.

        :return: duration
        """
        return self._frame_duration

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


class VideoReader(_imageprocessors.ImageProcessorMixin, AnimationReader):
    """
    Implementation :py:class:`.AnimationReader` that reads Video files with PyAV.

    All frame images from this animation reader will be aligned by 8 pixels by default.
    """

    def __init__(self,
                 file: typing.Union[str, typing.BinaryIO],
                 file_source: str,
                 resize_resolution: _types.OptionalSize = None,
                 aspect_correct: bool = True,
                 align: typing.Optional[int] = 8,
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

        :raises ValueError: if file_source lacks a file extension, it is needed
            to determine the video file format.
        """
        self._filename = file
        self._file_source = file_source
        if isinstance(file, str):
            self._container = av.open(file, 'r')
        else:
            _, ext = os.path.splitext(file_source)
            if not ext:
                raise ValueError(
                    'Cannot determine the format of a video file from a file_source lacking a file extension.')
            self._container = av.open(file, format=ext.lstrip('.').lower())

        self._aspect_correct = aspect_correct
        self._align = align

        width = int(self._container.streams.video[0].width)
        height = int(self._container.streams.video[0].height)

        width, height = _image.resize_image_calc(old_size=(width, height),
                                                 new_size=resize_resolution,
                                                 aspect_correct=aspect_correct,
                                                 align=align)
        self._resize_resolution = (width, height)

        fps = float(self._container.streams.video[0].average_rate)
        frame_duration = 1000 / fps
        total_frames = self._container.streams.video[0].frames

        self._container.streams.video[0].thread_type = "AUTO"

        if total_frames <= 0:
            # webm decode bug?
            total_frames = sum(1 for _ in self._container.decode(video=0))
            self._container.seek(0, whence='time')
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

    All frames from this animation reader will be aligned by 8 pixels by default.
    """

    def __init__(self,
                 file: typing.Union[str, typing.BinaryIO],
                 file_source: str,
                 resize_resolution: _types.OptionalSize = None,
                 aspect_correct: bool = True,
                 align: typing.Optional[int] = 8,
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
        """
        self._img = PIL.Image.open(file)
        self._file_source = file_source

        self._iter = PIL.ImageSequence.Iterator(self._img)
        self._aspect_correct = aspect_correct
        self._align = align

        total_frames = self._img.n_frames

        frame_duration = self._img.info.get('duration', 0)

        if frame_duration == 0:
            # 10 frames per second for bugged gifs / webp
            frame_duration = 100

        frame_duration = float(frame_duration)

        fps = 1000 / frame_duration

        width, height = _image.resize_image_calc(old_size=self._img.size,
                                                 new_size=resize_resolution,
                                                 aspect_correct=aspect_correct,
                                                 align=align)
        self._resize_resolution = (width, height)

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

    All frame images from this animation reader will be aligned by 8 pixels by default.
    """

    def __init__(self,
                 img: PIL.Image.Image,
                 resize_resolution: _types.OptionalSize = None,
                 aspect_correct: bool = True,
                 align: typing.Optional[int] = 8,
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

        total_frames = image_repetitions
        fps = 30.0
        frame_duration = 1000 / fps

        width, height = _image.resize_image_calc(old_size=self._img.size,
                                                 new_size=resize_resolution,
                                                 aspect_correct=aspect_correct,
                                                 align=align)
        self._resize_resolution = (width, height)

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


def get_supported_static_image_mimetypes() -> list[str]:
    """
    Get a list of mimetypes that are considered to be supported static image mimetypes.

    :return: list of mimetype strings.
    """
    return ['image/png',
            'application/png',
            'application/x-png',
            'image/jpeg',
            'image/jpg',
            'application/jpg',
            'application/x-jpg',
            'image/jp2',
            'image/jpx',
            'image/j2k',
            'image/jpeg2000',
            'image/jpeg2000-image',
            'image/x-jpeg2000-image',
            'image/bmp',
            'image/x-bitmap',
            'image/x-bmp',
            'application/bmp',
            'application/x-bmp',
            'image/x-targa',
            'image/x-tga',
            'image/targa',
            'image/tga',
            'image/vnd.adobe.photoshop',
            'application/x-photoshop',
            'application/photoshop',
            'application/psd',
            'image/psd']


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
def mimetype_is_supported(mimetype: str) -> bool:
    """
    Check if dgenerate supports a given input mimetype

    :param mimetype: The mimetype string

    :return: bool
    """
    return mimetype_is_static_image(mimetype) or \
           mimetype_is_animated_image(mimetype) or \
           mimetype_is_video(mimetype)


class ImageSeedParseResult:
    """
    The result of parsing an ``--image-seeds`` uri
    """

    seed_path: typing.Union[_types.Path, _types.Paths]
    """
    The seed path, contains an image path that will be used for img2img operations
    or the base image in inpaint operations. Or a controlnet guidance path, or a sequence of controlnet guidance paths. 
    A path being a file path, or an HTTP/HTTPS URL.
    """

    mask_path: _types.OptionalPath = None
    """
    Optional path to an inpaint mask, may be an HTTP/HTTPS URL or file path.
    """

    control_path: typing.Union[_types.Path, _types.Paths, None] = None
    """
    Optional controlnet guidance path, or a sequence of controlnet guidance paths. 
    This field is only used when the secondary syntax of ``--image-seeds`` is encountered.
    
    In parses such as:
    
        * ``--image-seeds "img2img.png;control=control.png"``
        * ``--image-seeds "img2img.png;control=control1.png, control2.png"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control.png"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control1.png, control2.png"``
        
    """

    floyd_path: _types.OptionalPath = None
    """
    Optional path to a result from a Deep Floyd IF stage, used only for img2img and inpainting mode
    with Deep Floyd.  This is the only way to specify the image that was output by a stage in that case.
    
    the arguments floyd and control are mutually exclusive.
    
    In parses such as:
    
        * ``--image-seeds "img2img.png;floyd=stage1-output.png"``
        * ``--image-seeds "img2img.png;mask=mask.png;floyd=stage1-output.png"``
    
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
        
    This should override any globally defined resize value.
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

    def get_control_image_paths(self) -> typing.Optional[_types.Paths]:
        """
        Return :py:attr:`.ImageSeedParseResult.seed_path` if :py:attr:`.ImageSeedParseResult.is_single_spec` is ``True``.

        If the image seed is not a single specification, return :py:attr:`.ImageSeedParseResult.control_path`.

        If :py:attr:`.ImageSeedParseResult.control_path` is not set and the image seed is not a single
        specification, return ``None``.

        :return: list of resource paths, or None
        """
        if self.is_single_spec:
            return self.seed_path if isinstance(self.seed_path, list) else [self.seed_path]
        elif self.control_path:
            return self.control_path if isinstance(self.control_path, list) else [self.control_path]
        else:
            return None

    @property
    def is_single_spec(self) -> bool:
        """
        Is this ``--image-seeds`` uri a single resource or resource group specification existing
        within the **seed_path** attribute of this object?

        For instance could it be a single img2img image definition, or a controlnet guidance
        image or sequence of controlnet guidance images?

        This requires that ``mask_path``, ``control_path``, and ``floyd_path`` are all undefined.

        Possible parses which trigger this condition are:

            * ``--image-seeds "img2img.png"``
            * ``--image-seeds "control-image.png"``
            * ``--image-seeds "control-image1.png, control-image2.png"``

        Since this is an ambiguous parse, it must be resolved later with the help of other specified arguments.
        Such as by the specification of ``--control-nets``, which makes the intention unambiguous.

        :return: bool
        """
        return self.mask_path is None and self.control_path is None and self.floyd_path is None


# noinspection HttpUrlsUsage
def _parse_image_seed_uri_legacy(uri: str, align: int = 8) -> ImageSeedParseResult:
    try:
        parts = _textprocessing.tokenized_split(uri, ';', remove_quotes=True)
        parts_iter = iter(parts)
    except _textprocessing.TokenizedSplitSyntaxError as e:
        raise ImageSeedParseError(
            f'Parsing error in image seed URI "{uri}": {str(e).strip()}')

    result = ImageSeedParseResult()

    first = next(parts_iter)
    result.seed_path = first

    if len(parts) == 1:
        try:
            first_parts = _textprocessing.tokenized_split(
                first, ',',
                strict=True,
                escapes_in_quoted=True)

        except _textprocessing.TokenizedSplitSyntaxError as e:
            raise ImageSeedParseError(
                f'Parsing error in image seed URI "{uri}": {str(e).strip()}')
    else:
        first_parts = [first]

    for part in first_parts:
        if not (is_downloadable_url(part) or os.path.exists(part)):
            if len(first_parts) > 1:
                raise ImageSeedFileNotFoundError(
                    f'Control image file "{part}" does not exist.')
            else:
                raise ImageSeedFileNotFoundError(
                    f'Image seed file "{part}" does not exist.')

    if len(first_parts) > 1:
        result.seed_path = first_parts

    for idx, part in enumerate(parts_iter):
        if part == '':
            raise ImageSeedParseError(
                'Missing inpaint mask image or output size specification, '
                'check image seed syntax, stray semicolon?')

        if is_downloadable_url(part):
            result.mask_path = part
        elif os.path.exists(part):
            result.mask_path = part
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
                        f'Image seed resize {["width", "height"][d_idx]} dimension {d} is not divisible by {align}.')

            if result.resize_resolution is not None:
                raise ImageSeedArgumentError(
                    'Resize resolution argument defined multiple times.')

            result.resize_resolution = dimensions

    return result


def parse_image_seed_uri(uri: str, align: typing.Optional[int] = 8) -> ImageSeedParseResult:
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

    :return: :py:class:`.ImageSeedParseResult`
    """

    if align is None:
        align = 1
    elif align < 1:
        raise ValueError('align argument may not be less than one.')

    keyword_args = ['mask',
                    'control',
                    'floyd',
                    'resize',
                    'aspect',
                    'frame-start',
                    'frame-end']

    try:
        parts = _textprocessing.tokenized_split(uri, ';')
    except _textprocessing.TokenizedSplitSyntaxError as e:
        raise ImageSeedParseError(f'Image seed URI parsing error: {str(e).strip()}')

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

    seed_parser = _textprocessing.ConceptUriParser('Image Seed',
                                                   known_args=keyword_args,
                                                   args_lists=['control'])

    try:
        parse_result = seed_parser.parse(uri)
    except _textprocessing.ConceptUriParseError as e:
        raise ImageSeedParseError(e)

    # noinspection HttpUrlsUsage
    def _ensure_exists(path, title):
        if not (is_downloadable_url(path) or os.path.exists(path)):
            raise ImageSeedFileNotFoundError(f'{title} file "{path}" does not exist.')

    seed_path = parse_result.concept
    _ensure_exists(seed_path, 'Image seed')
    result.seed_path = seed_path

    mask_path = parse_result.args.get('mask', None)
    if mask_path is not None:
        _ensure_exists(mask_path, 'Image mask')
        result.mask_path = mask_path

    control_path = parse_result.args.get('control', None)
    if control_path is not None:
        if isinstance(control_path, list):
            for f in control_path:
                if not f.strip():
                    raise ImageSeedParseError('Missing control image definition, stray comma?')

                _ensure_exists(f, 'Control image')
        else:
            _ensure_exists(control_path, 'Control image')

        result.control_path = control_path

    floyd_path = parse_result.args.get('floyd', None)
    if floyd_path is not None:
        _ensure_exists(floyd_path, 'Floyd image')
        if control_path is not None:
            raise ImageSeedArgumentError(
                'The image seed "control" argument cannot be used with the "floyd" argument.')
        result.floyd_path = floyd_path

    resize = parse_result.args.get('resize', None)

    if resize is not None:
        try:
            dimensions = _textprocessing.parse_image_size(resize)
        except ValueError as e:
            raise ImageSeedArgumentError(
                f'Error parsing image seed "resize" argument: {e}.')
        for d_idx, d in enumerate(dimensions):
            if d % align != 0:
                raise ImageSeedArgumentError(
                    f'Image seed resize {["width", "height"][d_idx]} dimension {d} is not divisible by {align}.')

        result.resize_resolution = dimensions

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

    return result


def get_web_cache_directory() -> str:
    """
    Get the default web cache directory or the value of the environmental variable ``DGENERATE_WEB_CACHE``

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
                '(id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT_TOKEN_STRICT UNIQUE, mime_type TEXT_TOKEN_STRICT)')
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
                # Do not delete database related files
                continue

            _messages.debug_log(f'Deleting File From Web Cache: "{file_path}"')
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                _messages.log(
                    f'Failed to delete cached web file "{file_path}", reason: {str(e).strip()}',
                    level=_messages.ERROR)


atexit.register(_wipe_web_cache_directory)


def create_web_cache_file(url,
                          mime_acceptable_desc: str = _textprocessing.oxford_comma(
                              get_supported_mimetypes(), conjunction='or'),
                          mimetype_is_supported: typing.Optional[typing.Callable[[str], bool]] = mimetype_is_supported) \
        -> tuple[str, str]:
    """
    Download a file from a url and add it to dgenerates temporary web cache that is
    available to all concurrent dgenerate processes.

    If the file exists in the cache already, return information for the existing file.

    :param url: The url

    :param mime_acceptable_desc: a string describing what mimetype values are acceptable which is used
        when :py:exc:`.UnknownMimetypeError` is raised.

    :param mimetype_is_supported: a function that test if a mimetype string is supported, if you
        supply the value ``None`` all mimetypes are considered supported.

    :raise UnknownMimetypeError: if a mimetype is considered not supported

    :return: tuple(mimetype_str, filepath)
    """

    cache_dir = get_web_cache_directory()

    def _mimetype_is_supported(mimetype):
        if mimetype_is_supported is not None:
            return mimetype_is_supported(mimetype)
        return True

    with _get_web_cache_db() as db:
        cursor = db.cursor()

        exists = cursor.execute(
            'SELECT mime_type, id FROM files WHERE url = ?', [url]).fetchone()

        # entry exists to a missing file on disk?
        missing_file_only = False

        if exists is not None:
            path = os.path.join(cache_dir, f'web_{exists[1]}')

            if os.path.exists(path):
                return exists[0], path
            else:
                # file exists in the database but is missing on disk
                missing_file_only = True

        headers = {'User-Agent': fake_useragent.UserAgent().chrome}

        with requests.get(url, headers=headers, stream=True) as req:
            mime_type = req.headers['content-type']

            if not _mimetype_is_supported(mime_type):
                raise UnknownMimetypeError(
                    f'Unknown mimetype "{mime_type}" from URL "{url}". '
                    f'Expected: {mime_acceptable_desc}')

            if not missing_file_only:
                # no record of this file existed
                cursor.execute(
                    'INSERT INTO files(mime_type, url) VALUES(?, ?)', [mime_type, url])
                path = os.path.join(cache_dir, f'web_{cursor.lastrowid}')
            else:
                # a record of this file existed but
                # the file was missing on disk
                # make sure mime_type matches
                cursor.execute(
                    'UPDATE files SET mime_type = ? WHERE id = ?', [mime_type, exists[1]])

            with open(path, mode='wb') as new_file:
                new_file.write(req.content)
                new_file.flush()

    return mime_type, path


def request_mimetype(url) -> str:
    """
    Request the mimetype of a file at a URL, if the file exists in the cache, a known mimetype
    is returned without connecting to the internet. Otherwise connect to the internet
    to retrieve the mimetype, this action does not update the cache.

    :param url: The url

    :return: mimetype string
    """

    with _get_web_cache_db() as db:
        cursor = db.cursor()

        exists = cursor.execute(
            'SELECT mime_type, id FROM files WHERE url = ?', [url]).fetchone()

        if exists is not None:
            return exists[0]

        headers = {'User-Agent': fake_useragent.UserAgent().chrome}

        with requests.get(url, headers=headers, stream=True) as req:
            mime_type = req.headers['content-type']

    return mime_type


class UnknownMimetypeError(Exception):
    """
    Raised when an unsupported mimetype is encountered
    """
    pass


_MIME_TYPES_GUESS_EXTRA = {
    '.webp': 'image/webp',
    '.apng': 'image/apng',
    '.tga': 'image/tga',
    '.jp2': 'image/jp2',
    '.j2k': 'image/j2k',
    '.jpx': 'image/jpx',
    '.psd': 'image/psd'
}


def guess_mimetype(filename) -> typing.Optional[str]:
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


# noinspection HttpUrlsUsage
def fetch_media_data_stream(uri: str) -> tuple[str, typing.BinaryIO]:
    """
    Get an open stream to a local file, or file at an HTTP or HTTPS URL, with caching for web files.

    Caching for downloaded files is multiprocess safe, multiple processes using this
    module can share the cache simultaneously, the last process alive clears the cache when it exits.

    :param uri: Local file path or URL

    :raise UnknownMimetypeError: If a remote file serves an unsupported mimetype value

    :return: (mime-type string, BinaryIO)
    """

    if is_downloadable_url(uri):
        mime_type, filename = create_web_cache_file(uri)
        return mime_type, open(filename, mode='rb')
    else:
        mime_acceptable_desc = _textprocessing.oxford_comma(
            get_supported_mimetypes(), conjunction='or')

        mime_type = guess_mimetype(uri)

        if mime_type is None:
            raise UnknownMimetypeError(
                f'Mimetype could not be determined for file "{uri}". '
                f'Expected: {mime_acceptable_desc}')

        if not mimetype_is_supported(mime_type):
            raise UnknownMimetypeError(
                f'Unknown mimetype "{mime_type}" for file "{uri}". Expected: {mime_acceptable_desc}')

    return mime_type, open(uri, 'rb')


def create_image(
        path_or_file: typing.Union[typing.BinaryIO, str],
        file_source: str,
        resize_resolution: _types.OptionalSize = None,
        aspect_correct: bool = True,
        align: typing.Optional[int] = 8) -> PIL.Image.Image:
    """
    Create an RGB format PIL image from a file path or binary file stream.
    The image is oriented according to any EXIF directives. Image is aligned
    to ``align`` in every case, specifying ``None`` or ``1`` for ``align``
    disables alignment.

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

    if resize_resolution is None:
        with PIL.Image.open(file) as img, _image.to_rgb(img) as rgb_img:
            e_img = _exif_orient(rgb_img)
            e_img.filename = file_source
            if not _image.is_aligned(e_img.size, align=align):
                with e_img:
                    resized = _image.resize_image(e_img, size=None, align=align)
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


def create_animation_reader(mimetype: str,
                            file_source: str,
                            file: typing.BinaryIO,
                            resize_resolution: _types.OptionalSize = None,
                            aspect_correct: bool = True,
                            align: typing.Optional[int] = 8,
                            image_processor: typing.Optional[_imageprocessors.ImageProcessor] = None,
                            ) -> AnimationReader:
    """
    Create an animation reader object from mimetype specification and binary file stream.

    Images will return a :py:class:`.MockImageAnimationReader` with a *total_frames* value of 1,
    which can then be adjusted by you.

    :py:class:`.VideoReader` or :py:class:`.AnimatedImageReader` will be returned for Video
    files and Animated Images respectively.

    :raise UnknownMimetypeError: on unknown ``mimetype`` value

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
                          resize_resolution=resize_resolution,
                          aspect_correct=aspect_correct,
                          align=align) as img:
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

    image_processor: typing.Optional[_imageprocessors.ImageProcessor] = None
    """
    Optional image image processor associated with the file
    """

    aspect_correct: bool = True
    """
    Aspect correct resize enabled?
    """

    align: typing.Optional[int] = 8
    """
    Images which are read are aligned to this amount of pixels, ``None`` or ``1`` will disable alignment.
    """

    resize_resolution: _types.OptionalSize = None
    """
    Optional resize resolution.
    """

    def __init__(self, path: str,
                 image_processor: typing.Optional[_imageprocessors.ImageProcessor] = None,
                 resize_resolution: _types.OptionalSize = None,
                 aspect_correct: bool = True,
                 align: typing.Optional[int] = 8):
        """
        :param path: File path or URL
        :param resize_resolution: Resize resolution
        :param aspect_correct: Aspect correct resize enabled?
        :param align: Images which are read are aligned to this amount of pixels, ``None`` or ``1`` will disable alignment.
        :param image_processor: Optional image image processor associated with the file
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
                 path_opener: typing.Callable[[str], typing.BinaryIO] = fetch_media_data_stream):
        """
        :raise ValueError: if ``frame_start`` > ``frame_end``
        :raise FrameStartOutOfBounds: if ``frame_start`` > ``total_frames - 1``

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
    dgenerates temporary web cache will be utilized.

    All images produced from this reader will be aligned to 8 pixels by default.
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
                 image_processor: typing.Optional[_imageprocessors.ImageProcessor] = None,
                 resize_resolution: _types.OptionalSize = None,
                 aspect_correct: bool = True,
                 align: typing.Optional[int] = 8,
                 frame_start: int = 0,
                 frame_end: _types.OptionalInteger = None,
                 path_opener: typing.Callable[[str], typing.BinaryIO] = fetch_media_data_stream):
        """

        :raise ValueError: if ``frame_start`` > ``frame_end``
        :raise FrameStartOutOfBounds: if ``frame_start`` > ``total_frames - 1``

        :param path: File path or URL
        :param resize_resolution: Resize resolution
        :param aspect_correct: Aspect correct resize enabled?
        :param align: Images which are read are aligned to this amount of pixels, ``None`` or ``1`` will disable alignment.
        :param image_processor: Optional image image processor associated with the file
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

    image: typing.Optional[PIL.Image.Image]
    """
    An optional image used for img2img mode, or inpainting mode in combination with :py:attr:`.ImageSeed.mask_image`
    """

    mask_image: typing.Optional[PIL.Image.Image]
    """
    An optional inpaint mask image, may be None
    """

    control_images: typing.Optional[_types.Images]
    """
    Control guidance images, or None.
    """

    floyd_image: typing.Optional[PIL.Image.Image]
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
                 image: typing.Optional[PIL.Image.Image] = None,
                 mask_image: typing.Optional[PIL.Image.Image] = None,
                 control_images: typing.Optional[_types.Images] = None,
                 floyd_image: typing.Optional[PIL.Image.Image] = None):
        self.image = image
        self.mask_image = mask_image

        if control_images is not None and floyd_image is not None:
            raise ValueError(
                'control_images and floyd_image arguments are incompatible '
                'and cannot both be specified')

        self.control_images = control_images
        self.floyd_image = floyd_image

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


ControlProcessorSpec = typing.Union[_imageprocessors.ImageProcessor,
                                    collections.abc.Sequence[_imageprocessors.ImageProcessor], None]


def _validate_control_image_processor_count(processors, guidance_images):
    num_processors = len(processors)
    num_guidance_images = len(guidance_images)
    if num_processors > num_guidance_images:
        raise ValueError('Too many control image processors specified, '
                         f'there are {num_processors} processors and '
                         f'{num_guidance_images} control guidance image sources.')


def iterate_image_seed(uri: typing.Union[str, ImageSeedParseResult],
                       frame_start: int = 0,
                       frame_end: _types.OptionalInteger = None,
                       resize_resolution: _types.OptionalSize = None,
                       aspect_correct: bool = True,
                       align: typing.Optional[int] = 8,
                       seed_image_processor: typing.Optional[_imageprocessors.ImageProcessor] = None,
                       mask_image_processor: typing.Optional[_imageprocessors.ImageProcessor] = None,
                       control_image_processor: ControlProcessorSpec = None) -> \
        collections.abc.Iterator[ImageSeed]:
    """
    Parse and load images/videos in an ``--image-seeds`` uri and return an iterator that
    produces :py:class:`.ImageSeed` objects while progressively reading those files.

    This method is used to iterate over an ``--image-seeds`` uri in the case that the image source
    mentioned is to be used for img2img / inpaint operations, and handles this syntax:

        * ``--image-seeds "img2img.png"``
        * ``--image-seeds "img2img.png;mask.png"``
        * ``--image-seeds "img2img.png;mask.png;512x512"``

    Additionally controlnet guidance resources are handled via the secondary syntax:

        * ``--image-seeds "img2img.png;control=control1.png, control2.png"``
        * ``--image-seeds "img2img.png;control=control1.png, control2.png;resize=512x512"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control1.png, control2.png"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control1.png, control2.png;resize=512x512"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control1.png;frame-start=2"``
        * ``--image-seeds "img2img.png;mask=mask.png;control=control1.png;frame-start=2;frame-end=5"``

    Deep Floyd img2img and inpainting mode are handled via a tertiary syntax:

        * ``--image-seeds "img2img.png;floyd=stage1-image.png"``
        * ``--image-seeds "img2img.png;mask=mask.png;floyd=stage2-image.png"``

    Note that all keyword arguments mentioned above can be used together, with exception
    of "control" and "floyd" which are mutually exclusive arguments.

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

    :param seed_image_processor: optional :py:class:`dgenerate.imageprocessors.ImageProcessor`
    :param mask_image_processor: optional :py:class:`dgenerate.imageprocessors.ImageProcessor`

    :param control_image_processor: optional :py:class:`dgenerate.imageprocessors.ImageProcessor` or list of them.
        A list is used to specify processors for individual images in a multi guidance image specification
        such as uri = "img2img.png;control=img1.png, img2.png".  In the case that a multi guidance image
        specification is used and only one processor is given, that processor will be used on only the
        first image / video in the specification. Images in a guidance specification with no corresponding
        processor value will have their processor set to ``None``, specifying extra processors
        as compared to control guidance image sources will cause :py:exc:`ValueError` to be raised.

    :raise ValueError: if there are more **control_image_processor** values than
        there are control guidance image sources in the URI.

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

    if isinstance(parse_result.seed_path, list):
        raise ValueError(
            'seed_path cannot contain multiple elements, use '
            f'{_types.fullname(iterate_control_image)} for that.')

    if parse_result.resize_resolution is not None:
        resize_resolution = parse_result.resize_resolution

    if parse_result.aspect_correct is not None:
        aspect_correct = parse_result.aspect_correct

    reader_specs = [
        MediaReaderSpec(path=parse_result.seed_path,
                        image_processor=seed_image_processor,
                        resize_resolution=resize_resolution,
                        aspect_correct=aspect_correct,
                        align=align)
    ]

    if parse_result.mask_path is not None:
        reader_specs.append(MediaReaderSpec(
            path=parse_result.mask_path,
            image_processor=mask_image_processor,
            resize_resolution=resize_resolution,
            aspect_correct=aspect_correct,
            align=align))

    if parse_result.control_path is not None:
        if not isinstance(control_image_processor, list):
            control_image_processor = [control_image_processor]

        control_guidance_image_paths = parse_result.get_control_image_paths()

        _validate_control_image_processor_count(
            processors=control_image_processor,
            guidance_images=control_guidance_image_paths)

        reader_specs += [
            MediaReaderSpec(
                path=p.strip(),
                image_processor=control_image_processor[idx] if idx < len(control_image_processor) else None,
                resize_resolution=resize_resolution,
                aspect_correct=aspect_correct,
                align=align)
            for idx, p in enumerate(control_guidance_image_paths)
        ]

    if parse_result.floyd_path is not None:
        # There should never be a reason to process floyd stage output
        # also do not resize it
        reader_specs.append(MediaReaderSpec(
            path=parse_result.floyd_path,
            resize_resolution=None,
            align=None))

    if parse_result.frame_start is not None:
        frame_start = parse_result.frame_start

    if parse_result.frame_end is not None:
        frame_end = parse_result.frame_end

    with MultiMediaReader(specs=reader_specs,
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

                extra_images = dict()
                if parse_result.floyd_path is not None:
                    extra_images = {'floyd_image': frame[2]}

                image_seed = ImageSeed(image=frame[0], mask_image=frame[1], **extra_images)
            elif parse_result.control_path is not None:
                image_seed = ImageSeed(image=frame[0], control_images=frame[1:])
            else:
                extra_images = dict()
                if parse_result.floyd_path is not None:
                    extra_images = {'floyd_image': frame[1]}

                image_seed = ImageSeed(image=frame[0], **extra_images)

            if not dimensions_checked:
                images = list(_flatten([image_seed.image if image_seed.image else [],
                                        image_seed.mask_image if image_seed.mask_image else [],
                                        image_seed.control_images if image_seed.control_images else []]))

                _check_image_dimensions_match(images)

                dimensions_checked = True

            image_seed.is_animation_frame = is_animation
            if is_animation:
                image_seed.fps = reader.fps
                image_seed.frame_duration = reader.frame_duration
                image_seed.frame_index = reader.frame_index
                image_seed.total_frames = reader.total_frames if is_animation else None
            yield image_seed


def iterate_control_image(uri: typing.Union[str, ImageSeedParseResult],
                          frame_start: int = 0,
                          frame_end: _types.OptionalInteger = None,
                          resize_resolution: _types.OptionalSize = None,
                          aspect_correct: bool = True,
                          align: typing.Optional[int] = 8,
                          image_processor: ControlProcessorSpec = None) -> \
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
        such as uri = "img1.png, img2.png".  In the case that a multi guidance image specification is used and only
        one processor is given, that processor will be used on only the first image / video in the specification.
        Images in a guidance specification with no corresponding processor value will have their processor
        set to ``None``, specifying extra processors as compared to control guidance image sources will
        cause :py:exc:`ValueError` to be raised.

    :raise ValueError: if there are more **image_processor** values than
        there are control guidance image sources in the URI.

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

    if parse_result.aspect_correct is not None:
        aspect_correct = parse_result.aspect_correct

    reader_specs = []

    if not isinstance(image_processor, list):
        image_processor = [image_processor]

    control_guidance_image_paths = parse_result.get_control_image_paths()

    _validate_control_image_processor_count(
        processors=image_processor,
        guidance_images=control_guidance_image_paths)

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
                image_seed.fps = reader.fps
                image_seed.frame_duration = reader.frame_duration
                image_seed.frame_index = reader.frame_index
                image_seed.total_frames = reader.total_frames if is_animation else None
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
        return ImageSeedInfo(seed.is_animation_frame, seed.total_frames, seed.fps, seed.frame_duration)


def get_control_image_info(uri: typing.Union[_types.Path, ImageSeedParseResult],
                           frame_start: int = 0,
                           frame_end: _types.OptionalInteger = None) -> ImageSeedInfo:
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
    :return: :py:class:`.ImageSeedInfo`
    """
    with next(iterate_control_image(uri, frame_start, frame_end)) as seed:
        return ImageSeedInfo(seed.is_animation_frame, seed.total_frames, seed.fps, seed.frame_duration)
