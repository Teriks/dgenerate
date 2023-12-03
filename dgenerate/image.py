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

import dgenerate.types as _types

__doc__ = """
Image operations commonly used by dgenerate.
"""


def resize_image_calc(old_size: _types.Size,
                      new_size: _types.OptionalSize,
                      aspect_correct: bool = True,
                      align: typing.Optional[int] = 8):
    """
    Calculate the new dimensions for a requested resize of an image..


    :param old_size: The old image size
    :param new_size: The new image size

    :param aspect_correct: preserve aspect ratio?

    :param align: Ensure returned size is aligned to this value.

    :return: calculated new size
    """

    if align is not None and align < 1:
        raise ValueError('align value may not be less than 1.')

    if new_size is None:
        if align is not None:
            return align_by(old_size, align=align)
        return old_size

    if align is not None:
        new_size = align_by(new_size, align=align)

    if old_size == new_size:
        return old_size

    if aspect_correct:
        width = new_size[0]
        w_percent = (width / float(old_size[0]))
        hsize = int((float(old_size[1]) * float(w_percent)))
        if align is not None:
            return width, hsize - hsize % align
        return width, hsize
    else:
        return new_size


def is_aligned(iterable: typing.Iterable[int], align: int) -> bool:
    """
    Check if all elements are aligned by a specific value.

    :param iterable: Elements to test
    :param align: The alignment value, ``None`` indicates no alignment.

    :return: bool
    """

    if align is None:
        return True

    if align < 1:
        raise ValueError('align value may not be less than 1.')

    return all(i % align == 0 for i in iterable)


def align_by(iterable: typing.Iterable[int], align: int) -> tuple:
    """
    Align all elements by a value and return a tuple

    :param iterable: Elements to align
    :param align: The alignment value, ``None`` indicates no alignment.

    :return: tuple(...)
    """

    if align is None:
        align = 1

    if align < 1:
        raise ValueError('align value may not be less than 1.')

    return tuple(i - i % align for i in iterable)


def copy_img(img: PIL.Image.Image):
    """
    Copy a :py:class:`PIL.Image.Image` while preserving its filename attribute.

    :param img: the image
    :return: a copy of the image
    """
    c = img.copy()

    if hasattr(img, 'filename'):
        c.filename = img.filename

    return c


def resize_image(img: PIL.Image.Image,
                 size: _types.OptionalSize,
                 aspect_correct: bool = True,
                 align: typing.Optional[int] = 8):
    """
    Resize a :py:class:`PIL.Image.Image` and return a copy.

    This function always returns a copy even if the images size did not change.

    The new image dimension will always be forcefully aligned by ``align``,
    specifying ``None`` or ``1`` indicates no alignment.

    The filename attribute is preserved.

    :param img: the image to resize
    :param size: requested new size for the image, may be ``None``.
    :param aspect_correct: preserve aspect ratio?
    :param align: Force alignment by this amount of pixels.
    :return: the resized image
    """
    new_size = resize_image_calc(old_size=img.size,
                                 new_size=size,
                                 aspect_correct=aspect_correct,
                                 align=align)
    if img.size == new_size:
        # probably less costly
        return copy_img(img)

    r = img.resize(new_size, PIL.Image.Resampling.LANCZOS)

    if hasattr(img, 'filename'):
        r.filename = img.filename

    return r


def to_rgb(img: PIL.Image.Image):
    """
    Convert a :py:class:`PIL.Image.Image` to RGB format while preserving its filename attribute.

    :param img: the image
    :return: a converted copy of the image
    """
    c = img.convert('RGB')
    if hasattr(img, 'filename'):
        c.filename = img.filename
    return c


def get_filename(img: PIL.Image.Image):
    """
    Get the :py:attr:`PIL.Image.Image.filename` attribute or "NO_FILENAME" if it does not exist.

    :param img: :py:class:`PIL.Image.Image`
    :return: filename string or "NO_FILENAME"
    """

    if hasattr(img, 'filename'):
        return img.filename
    return 'NO_FILENAME'
