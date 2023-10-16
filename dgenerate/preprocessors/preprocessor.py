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
import itertools
import os
import textwrap
import typing


import PIL.Image

import dgenerate.preprocessors.exceptions as _exceptions
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
import dgenerate.util as _util
import dgenerate.messages as _messages


class ImagePreprocessor:
    @classmethod
    def get_names(cls) -> typing.List[str]:
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
    def get_help(cls, called_by_name: str) -> str:
        help_str = None
        if hasattr(cls, 'help'):
            help_str = cls.help(called_by_name)
            if help_str:
                help_str = _textprocessing.justify_left(help_str).strip()

        if cls.__doc__:
            help_str = _textprocessing.justify_left(cls.__doc__).strip()

        args_with_defaults = cls.get_accepted_args_with_defaults(called_by_name)
        arg_descriptors = []

        for arg in args_with_defaults:
            if len(arg) == 1:
                arg_descriptors.append(arg[0])
            else:
                default_value = arg[1]
                if isinstance(default_value, str):
                    default_value = _textprocessing.quote(default_value)
                arg_descriptors.append(f'{arg[0]}={default_value}')

        if arg_descriptors:
            args_part = f'\n{" " * 4}arguments:\n{" " * 8}{(chr(10) + " " * 8).join(arg_descriptors)}\n'
        else:
            args_part = ''

        if help_str:
            wrap = \
                _textprocessing.wrap_paragraphs(
                    help_str,
                    initial_indent=' ' * 4,
                    subsequent_indent=' ' * 4,
                    width=_textprocessing.long_text_wrap_width())

            return called_by_name + f':{args_part}\n' + wrap
        else:
            return called_by_name + f':{args_part}'

    @classmethod
    def get_accepted_args(cls, called_by_name: str) -> typing.List[str]:
        return [a[0] for a in
                cls.get_accepted_args_with_defaults(called_by_name)]

    @classmethod
    def get_required_args(cls, called_by_name: str) -> typing.List[str]:
        return [a[0] for a in
                cls.get_accepted_args_with_defaults(called_by_name) if len(a) == 1]

    @classmethod
    def get_default_args(cls, called_by_name: str) -> typing.List[typing.Tuple[str, typing.Any]]:
        return [a for a in
                cls.get_accepted_args_with_defaults(called_by_name) if len(a) == 2]

    @classmethod
    def get_accepted_args_with_defaults(cls, called_by_name) -> typing.List[typing.Tuple[str, typing.Any]]:
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
                    fixed_args.append((_textprocessing.dashup(arg),))
                elif len(arg) == 1:
                    fixed_args.append((_textprocessing.dashup(arg[0]),))
                else:
                    fixed_args.append((_textprocessing.dashup(arg[0]), arg[1]))

            return [] if fixed_args is None else fixed_args

        args_with_defaults = []
        spec = inspect.getfullargspec(cls.__init__)
        sig_args = spec.args[1:]
        defaults_cnt = len(spec.defaults) if spec.defaults else 0
        no_defaults_before = len(sig_args) - defaults_cnt
        for idx, arg in enumerate(sig_args):
            if idx < no_defaults_before:
                args_with_defaults.append((_textprocessing.dashup(arg),))
            else:
                args_with_defaults.append((_textprocessing.dashup(arg),
                                           spec.defaults[idx - defaults_cnt]))

        return args_with_defaults

    @staticmethod
    def get_int_arg(name: str, value: typing.Union[str, int]) -> int:
        if isinstance(value, dict):
            value = value.get(name)
        try:
            return int(value)
        except ValueError:
            raise _exceptions.ImagePreprocessorArgumentError(f'Argument "{name}" must be an integer value.')

    @staticmethod
    def get_float_arg(name: str, value: typing.Union[str, float]) -> float:
        if isinstance(value, dict):
            value = value.get(name)
        try:
            return float(value)
        except ValueError:
            raise _exceptions.ImagePreprocessorArgumentError(f'Argument "{name}" must be a floating point value.')

    @staticmethod
    def get_bool_arg(name: str, value: typing.Union[str, bool]) -> bool:
        if isinstance(value, dict):
            value = value.get(name)
        try:
            return bool(value)
        except ValueError:
            raise _exceptions.ImagePreprocessorArgumentError(f'Argument "{name}" must be a boolean value.')

    def argument_error(self, msg: str):
        raise _exceptions.ImagePreprocessorArgumentError(msg)

    def __init__(self, **kwargs):
        self.__output_file = kwargs.get('output_file')
        self.__output_overwrite = kwargs.get('output_overwrite', False)
        self.__device = kwargs.get('device', 'cpu')
        self.__called_by_name = kwargs.get('called_by_name')

    @property
    def device(self) -> str:
        return self.__device

    @property
    def called_by_name(self) -> str:
        return self.__called_by_name

    def __gen_filename(self):
        return _util.touch_avoid_duplicate(os.path.dirname(self.__output_file),
                                           _util.suffix_path_maker(self.__output_file, '_duplicate_'))

    def __save_debug_image(self, image, debug_header):
        if self.__output_file is not None:
            if not self.__output_overwrite:
                filename = self.__gen_filename()
            else:
                filename = self.__output_file

            image.save(filename)
            _messages.debug_log(f'{debug_header}: "{filename}"')


    @staticmethod
    def call_pre_resize(preprocessor, image: PIL.Image.Image,
                        resize_resolution: _types.OptionalSize) -> PIL.Image.Image:
        img_copy = image.copy()

        processed = preprocessor.pre_resize(image, resize_resolution)
        if processed is not image:
            preprocessor.__save_debug_image(
                processed,
                'Wrote Preprocessor Debug Image (because copied)')

            processed.filename = image.filename
            return processed

        # Not copied but may be modified

        identical = all(a == b for a, b in
                        itertools.zip_longest(processed.getdata(),
                                              img_copy.getdata(),
                                              fillvalue=None))

        if not identical:
            # Write the debug output if it was modified in place
            preprocessor.__save_debug_image(
                processed,
                'Wrote Preprocessor Debug Image (because modified)')

        return processed

    @staticmethod
    def call_post_resize(preprocessor, image: PIL.Image.Image) -> PIL.Image.Image:
        img_copy = image.copy()

        processed = preprocessor.post_resize(image)
        if processed is not image:
            preprocessor.__save_debug_image(
                processed,
                'Wrote Preprocessor Debug Image (because copied)')

            processed.filename = image.filename
            return processed

        # Not copied but may be modified

        identical = all(a == b for a, b in
                        itertools.zip_longest(processed.getdata(),
                                              img_copy.getdata(),
                                              fillvalue=None))

        if not identical:
            # Write the debug output if it was modified in place
            preprocessor.__save_debug_image(
                processed,
                'Wrote Preprocessor Debug Image (because modified)')

        return processed

    def pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        return image

    def post_resize(self, image: PIL.Image.Image):
        return image

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)
