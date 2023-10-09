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


def resize_image_calc(old_size: typing.Tuple[int, int],
                      new_size: typing.Union[typing.Tuple[int, int], None]):
    if new_size is None or old_size == new_size:
        return old_size

    width = new_size[0]
    w_percent = (width / float(old_size[0]))
    hsize = int((float(old_size[1]) * float(w_percent)))

    return width - width % 8, hsize - hsize % 8


def is_aligned_by_8(x, y):
    return x % 8 == 0 and y % 8 == 0


def align_by_8(x, y):
    return x - x % 8, y - y % 8


def copy_img(img: PIL.Image.Image):
    c = img.copy()

    if hasattr(img, 'filename'):
        c.filename = img.filename

    return c


def resize_image(img: PIL.Image.Image,
                 size: typing.Union[typing.Tuple[int, int], None]):
    new_size = resize_image_calc(old_size=img.size,
                                 new_size=size)

    if img.size == new_size:
        # probably less costly
        return copy_img(img)

    r = img.resize(new_size, PIL.Image.LANCZOS)

    if hasattr(img, 'filename'):
        r.filename = img.filename

    return r


def to_rgb(img: PIL.Image.Image):
    c = img.convert('RGB')
    if hasattr(img, 'filename'):
        c.filename = img.filename
    return c
