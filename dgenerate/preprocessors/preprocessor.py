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
import inspect
import os
import typing
from pathlib import Path

import PIL.Image

from .exceptions import ImagePreprocessorArgumentError


class ImagePreprocessor:
    @classmethod
    def get_names(cls):
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

    @classmethod
    def get_help(cls, called_by_name):
        if hasattr(cls, 'help'):
            return cls.help(called_by_name)

        if cls.__doc__:
            return ' '.join(line.strip() for line in cls.__doc__.split())
        return None

    @classmethod
    def get_accepted_args(cls, called_by_name) -> typing.List[str]:
        return [a[0] for a in
                cls.get_accepted_args_with_defaults(called_by_name)]

    @classmethod
    def get_required_args(cls, called_by_name) -> typing.List[str]:
        return [a[0] for a in
                cls.get_accepted_args_with_defaults(called_by_name) if len(a) == 1]

    @classmethod
    def get_default_args(cls, called_by_name) -> typing.List[tuple]:
        return [a for a in
                cls.get_accepted_args_with_defaults(called_by_name) if len(a) == 2]

    @classmethod
    def get_accepted_args_with_defaults(cls, called_by_name) -> typing.List[tuple]:
        if hasattr(cls, 'ARGS'):
            if isinstance(cls.ARGS, dict):
                if called_by_name not in cls.ARGS:
                    raise RuntimeError(
                        'ImagePreprocessor module implementation bug, args for '
                        f'"{called_by_name}" not specified in ARGS dictionary.')
                args_with_defaults = cls.ARGS.get(called_by_name)
            else:
                args_with_defaults = cls.ARGS

            fixed_args = []
            for arg in args_with_defaults:
                if not isinstance(arg, tuple):
                    if not isinstance(arg, str):
                        raise RuntimeError(
                            f'{cls.__name__}.ARGS["{called_by_name}"] '
                            f'contained a non tuple or str value: {arg}')
                    fixed_args.append((arg.replace('_', '-'),))
                elif len(arg) == 1:
                    fixed_args.append((arg[0].replace('_', '-'),))
                else:
                    fixed_args.append((arg[0].replace('_', '-'), arg[1]))

            return [] if fixed_args is None else fixed_args

        args_with_defaults = []
        spec = inspect.getfullargspec(cls.__init__)
        sig_args = spec.args[1:]
        defaults_cnt = len(spec.defaults) if spec.defaults else 0
        no_defaults_before = len(sig_args) - defaults_cnt
        for idx, arg in enumerate(sig_args):
            if idx < no_defaults_before:
                args_with_defaults.append((arg.replace('_', '-'),))
            else:
                args_with_defaults.append((arg.replace('_', '-'),
                                           spec.defaults[idx - defaults_cnt]))

        return args_with_defaults

    def get_int_arg(self, name, value):
        if isinstance(value, dict):
            value = value.get(name)
        try:
            return int(value)
        except ValueError:
            raise ImagePreprocessorArgumentError(f'Argument "{name}" must be an integer value.')

    def get_float_arg(self, name, value):
        if isinstance(value, dict):
            value = value.get(name)
        try:
            return float(value)
        except ValueError:
            raise ImagePreprocessorArgumentError(f'Argument "{name}" must be a floating point value.')

    def get_bool_arg(self, name, value):
        if isinstance(value, dict):
            value = value.get(name)
        try:
            return bool(value)
        except ValueError:
            raise ImagePreprocessorArgumentError(f'Argument "{name}" must be a boolean value.')

    def argument_error(self, msg):
        raise ImagePreprocessorArgumentError(msg)

    def __init__(self, **kwargs):
        output_dir = kwargs.get('output_dir')
        output_file = kwargs.get('output_file')
        device = kwargs.get('device', 'cpu')
        called_by_name = kwargs.get('called_by_name')

        if output_dir is not None and output_file is not None:
            raise ImagePreprocessorArgumentError(
                'output_dir and output_file may not be specified simultaneously')

        self.__device = device
        self.__output_dir = output_dir
        self.__output_file = output_file
        self.__called_by_name = called_by_name

    @property
    def device(self):
        return self.__device

    @property
    def called_by_name(self):
        return self.__called_by_name

    def __gen_filename(self):
        def _make_path(dup_number=None):
            name = next(iter(self.get_names()))
            if dup_number is not None:
                name = next(iter(self.get_names())) + f'_{dup_number}'
            return os.path.join(self.__output_dir, name) + '.png'

        path = _make_path()

        if not os.path.exists(path):
            return path

        duplicate_number = 1
        while os.path.exists(path):
            path = _make_path(duplicate_number)
            duplicate_number += 1

        return path

    def __save_image(self, image):
        if self.__output_dir is not None:
            Path(self.__output_dir).mkdir(parents=True, exist_ok=True)
            image.save(self.__gen_filename())
        elif self.__output_file is not None:
            image.save(self.__output_file)

    @staticmethod
    def call_pre_resize(preprocessor, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        img = preprocessor.pre_resize(image, resize_resolution)
        if img is not image:
            preprocessor.__save_image(img)
            img.filename = image.filename
            return img
        return image

    @staticmethod
    def call_post_resize(preprocessor, image: PIL.Image):
        img = preprocessor.post_resize(image)
        if img is not image:
            preprocessor.__save_image(img)
            img.filename = image.filename
            return img
        return image

    def pre_resize(self, image: PIL.Image, resize_resolution: typing.Union[None, tuple]):
        return image

    def post_resize(self, image: PIL.Image):
        return image

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)
