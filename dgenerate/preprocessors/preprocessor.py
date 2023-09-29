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
from pathlib import Path

import PIL.Image

from .exceptions import ImagePreprocessorArgumentError


def names_from_class(cls):
    if not issubclass(cls, ImagePreprocessor):
        raise ValueError(
            'provided class is not a subclass of dgenerate.preprocessors.ImagePreprocessor')

    if hasattr(cls, 'NAMES'):
        if isinstance(cls.NAMES, str):
            return [cls.NAMES]
        else:
            return cls.NAMES
    else:
        return [cls.__name__]


class ImagePreprocessor:
    @staticmethod
    def get_int_arg(name, value):
        try:
            return int(value)
        except ValueError:
            raise ImagePreprocessorArgumentError(f'Argument "{name}" must be an integer value.')

    @staticmethod
    def get_float_arg(name, value):
        try:
            return float(value)
        except ValueError:
            raise ImagePreprocessorArgumentError(f'Argument "{name}" must be a floating point value.')

    @staticmethod
    def get_bool_arg(name, value):
        try:
            return bool(value)
        except ValueError:
            raise ImagePreprocessorArgumentError(f'Argument "{name}" must be a boolean value.')

    @staticmethod
    def argument_error(msg):
        raise ImagePreprocessorArgumentError(msg)

    def __init__(self, **kwargs):
        self._output_dir = None
        self._output_file = None

    def set_output_dir_or_file(self, output_dir=None, output_file=None):
        if output_dir is not None and output_file is not None:
            raise ImagePreprocessorArgumentError(
                'output_dir and output_file may not be specified simultaniously')

        self._output_dir = output_dir
        self._output_file = output_file

    def _gen_filename(self):
        def _make_path(dup_number=None):
            name = next(iter(names_from_class(self.__class__)))
            if dup_number is not None:
                name = next(iter(names_from_class(self.__class__))) + f'_{dup_number}'
            return os.path.join(self._output_dir, name) + '.png'

        path = _make_path()

        if not os.path.exists(path):
            return path

        duplicate_number = 1
        while os.path.exists(path):
            path = _make_path(duplicate_number)
            duplicate_number += 1


        return path

    def _save_image(self, image):
        if self._output_dir is not None:
            Path(self._output_dir).mkdir(parents=True, exist_ok=True)
            image.save(self._gen_filename())
        elif self._output_file is not None:
            image.save(self._output_file)

    def call_pre_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        img = self.pre_resize(image, resize_resolution)
        if img is not image:
            self._save_image(img)
            img.filename = image.filename
            return img
        return image

    def call_post_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        img = self.post_resize(image, resize_resolution)
        if img is not image:
            self._save_image(img)
            img.filename = image.filename
            return img
        return image

    def pre_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        return image

    def post_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        return image
