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

__doc__ = """
File extensions dgenerate can open or write.

Model, image, and video extensions are defined here. Readers, writers, and the
web cache expiry selectors use these lists instead of keeping their own.
"""

# Extensions are returned without a leading period.
# List functions return a new list. Set functions return one memoized frozenset.
# ``.cpkt`` appears in some file dialogs as a typo for ``.ckpt`` and is not loaded.

_MODEL_EXTENSIONS = (
    'safetensors',
    'gguf',
    'pt',
    'pth',
    'bin',
    'ckpt',
    'param',
    'yaml',
    'yml',
)
_TENSOR_EXTENSIONS = ('pt', 'pth', 'safetensors')
_VIDEO_WRITE_EXTENSIONS = ('mp4',)
_ANIMATED_IMAGE_READ_EXTENSIONS = frozenset({'gif', 'webp', 'apng'})
_ANIMATED_IMAGE_WRITE_EXTENSIONS = frozenset({'gif', 'webp', 'apng', 'png'})

_cache: dict[str, tuple[str, ...]] = {}
_sets: dict[str, frozenset[str]] = {}


def _remember(name: str, build) -> list[str]:
    found = _cache.get(name)
    if found is None:
        found = tuple(build())
        _cache[name] = found
    return list(found)


def _remember_set(name: str, extensions) -> frozenset[str]:
    found = _sets.get(name)
    if found is None:
        found = frozenset(extensions())
        _sets[name] = found
    return found


def model_extensions() -> list[str]:
    """
    Model extensions dgenerate can open.

    :return: extensions without a leading period
    """
    return list(_MODEL_EXTENSIONS)


def model_extension_set() -> frozenset[str]:
    """
    Memoized :py:func:`model_extensions`.

    :return: extensions without a leading period
    """
    return _remember_set('model', model_extensions)


def tensor_extensions() -> list[str]:
    """
    Latent tensor extensions. These are also model extensions.

    :return: extensions without a leading period
    """
    return list(_TENSOR_EXTENSIONS)


def tensor_extension_set() -> frozenset[str]:
    """
    Memoized :py:func:`tensor_extensions`.

    :return: extensions without a leading period
    """
    return _remember_set('tensor', tensor_extensions)


def _pillow_extensions(registered) -> list[str]:
    import PIL.Image

    PIL.Image.init()
    return [
        ext.lstrip('.').lower()
        for ext, fmt in PIL.Image.EXTENSION.items()
        if fmt in registered
    ]


def image_read_extensions() -> list[str]:
    """
    Image extensions dgenerate can read.

    :return: extensions without a leading period
    """
    import PIL.Image
    return _remember('image-read', lambda: _pillow_extensions(PIL.Image.OPEN))


def image_read_extension_set() -> frozenset[str]:
    """
    Memoized :py:func:`image_read_extensions`.

    :return: extensions without a leading period
    """
    return _remember_set('image-read', image_read_extensions)


def image_write_extensions() -> list[str]:
    """
    Image extensions dgenerate can write.

    :return: extensions without a leading period
    """
    import PIL.Image
    return _remember('image-write', lambda: _pillow_extensions(PIL.Image.SAVE))


def image_write_extension_set() -> frozenset[str]:
    """
    Memoized :py:func:`image_write_extensions`.

    :return: extensions without a leading period
    """
    return _remember_set('image-write', image_write_extensions)


def animated_image_read_extensions() -> list[str]:
    """
    Animated image extensions dgenerate can read.

    :return: extensions without a leading period
    """
    return _remember(
        'animated-image-read',
        lambda: [ext for ext in image_read_extensions() if ext in _ANIMATED_IMAGE_READ_EXTENSIONS])


def animated_image_read_extension_set() -> frozenset[str]:
    """
    Memoized :py:func:`animated_image_read_extensions`.

    :return: extensions without a leading period
    """
    return _remember_set('animated-image-read', animated_image_read_extensions)


def animated_image_write_extensions() -> list[str]:
    """
    Animated image extensions dgenerate can write.

    :return: extensions without a leading period
    """
    import PIL.Image
    return _remember(
        'animated-image-write',
        lambda: [
            ext for ext in _pillow_extensions(PIL.Image.SAVE_ALL)
            if ext in _ANIMATED_IMAGE_WRITE_EXTENSIONS
        ])


def animated_image_write_extension_set() -> frozenset[str]:
    """
    Memoized :py:func:`animated_image_write_extensions`.

    :return: extensions without a leading period
    """
    return _remember_set('animated-image-write', animated_image_write_extensions)


def _video_read_codecs() -> list[str]:
    import av

    supported_codecs = []
    for codec_name in av.codecs_available:
        try:
            codec = av.codec.Codec(codec_name, 'r')
        except Exception:
            continue
        if codec.type == 'video':
            supported_codecs.append(codec_name)
    return supported_codecs


def video_read_extensions() -> list[str]:
    """
    Video extensions dgenerate can read.

    An extension is included when this environment can decode one of its codecs.

    :return: extensions without a leading period
    """
    def build():
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
        supported_codecs = _video_read_codecs()
        supported_extensions = set()
        for ext, codecs in common_extensions.items():
            for codec in codecs:
                if codec in supported_codecs:
                    supported_extensions.add(ext)
        return list(supported_extensions)

    return _remember('video-read', build)


def video_read_extension_set() -> frozenset[str]:
    """
    Memoized :py:func:`video_read_extensions`.

    :return: extensions without a leading period
    """
    return _remember_set('video-read', video_read_extensions)


def video_write_extensions() -> list[str]:
    """
    Video extensions dgenerate can write.

    :return: extensions without a leading period
    """
    return list(_VIDEO_WRITE_EXTENSIONS)


def video_write_extension_set() -> frozenset[str]:
    """
    Memoized :py:func:`video_write_extensions`.

    :return: extensions without a leading period
    """
    return _remember_set('video-write', video_write_extensions)


def animation_read_extensions() -> list[str]:
    """
    Animation extensions dgenerate can read.

    This is every readable video extension followed by every readable animated
    image extension.

    :return: extensions without a leading period
    """
    return video_read_extensions() + animated_image_read_extensions()


def animation_read_extension_set() -> frozenset[str]:
    """
    Memoized :py:func:`animation_read_extensions`.

    :return: extensions without a leading period
    """
    return _remember_set('animation-read', animation_read_extensions)


def animation_write_extensions() -> list[str]:
    """
    Animation extensions dgenerate can write.

    This is every writable video extension followed by every writable animated
    image extension.

    :return: extensions without a leading period
    """
    return video_write_extensions() + animated_image_write_extensions()


def animation_write_extension_set() -> frozenset[str]:
    """
    Memoized :py:func:`animation_write_extensions`.

    :return: extensions without a leading period
    """
    return _remember_set('animation-write', animation_write_extensions)
