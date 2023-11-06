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

import PIL.Image

import dgenerate.types as _types


def resize_image_calc(old_size: _types.Size,
                      new_size: _types.OptionalSize,
                      aspect_correct: bool = True):
    """
    Calculate the new dimensions for a requested resize of an image, align by 8 pixels.

    :param old_size: The old image size
    :param new_size: The new image size

    :param aspect_correct: preserve aspect ratio?

    :return: calculated new size
    """
    if new_size is None or old_size == new_size:
        return old_size

    if aspect_correct:
        width = new_size[0]
        w_percent = (width / float(old_size[0]))
        hsize = int((float(old_size[1]) * float(w_percent)))
        return width - width % 8, hsize - hsize % 8
    else:
        width = new_size[0]
        height = new_size[1]
        return width - width % 8, height - height % 8


def is_aligned_by_8(x, y) -> bool:
    """
    Check if x and y are aligned by 8

    :param x: x
    :param y: y
    :return: bool
    """
    return x % 8 == 0 and y % 8 == 0


def align_by_8(x, y) -> _types.Size:
    """
    Align x and y by 8 and return a tuple

    :param x: x
    :param y: y
    :return: tuple(x, y)
    """
    return x - x % 8, y - y % 8


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
                 aspect_correct: bool = True):
    """
    Resize a :py:class:`PIL.Image.Image` and return a copy, resize is aligned to 8 pixels.

    The filename attribute is preserved.

    :param img: the image
    :param size: the new size for the image
    :param aspect_correct: preserve aspect ratio?
    :return: the resized image
    """
    new_size = resize_image_calc(old_size=img.size,
                                 new_size=size,
                                 aspect_correct=aspect_correct)

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
