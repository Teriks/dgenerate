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
import itertools
import os
import typing

import PIL.Image

import dgenerate.filelock as _filelock
import dgenerate.image as _image
import dgenerate.messages as _messages
import dgenerate.preprocessors.exceptions as _exceptions
import dgenerate.types as _types
import dgenerate.plugin as _plugin


class ImagePreprocessor(_plugin.InvokablePlugin):
    """
    Abstract base class for image preprocessor implementations.
    """

    @staticmethod
    def get_int_arg(name: str, value: typing.Union[str, int, typing.Dict]) -> int:
        """
        Convert an argument value from a string to an integer.
        Throw :py:exc:`.ImagePreprocessorArgumentError` if there
        is an error parsing the value.

        :raises ImagePreprocessorArgumentError:

        :param name: the argument name for descriptive purposes,
            and/or for specifying the dictionary key when *value*
            is a dictionary.

        :param value: an integer value as a string, or optionally a
            dictionary to get the value from using the argument *name*.

        :return: int
        """
        if isinstance(value, dict):
            value = value.get(name)
        try:
            return int(value)
        except ValueError:
            raise _exceptions.ImagePreprocessorArgumentError(f'Argument "{name}" must be an integer value.')

    @staticmethod
    def get_float_arg(name: str, value: typing.Union[str, float, typing.Dict]) -> float:
        """
        Convert an argument value from a string to a float.
        Throw :py:exc:`.ImagePreprocessorArgumentError` if there
        is an error parsing the value.

        :raises ImagePreprocessorArgumentError:

        :param name: the argument name for descriptive purposes,
            and/or for specifying the dictionary key when *value*
            is a dictionary.

        :param value: a float value as a string, or optionally a
            dictionary to get the value from using the argument *name*.

        :return: float
        """

        if isinstance(value, dict):
            value = value.get(name)
        try:
            return float(value)
        except ValueError:
            raise _exceptions.ImagePreprocessorArgumentError(f'Argument "{name}" must be a floating point value.')

    @staticmethod
    def get_bool_arg(name: str, value: typing.Union[str, bool, typing.Dict]) -> bool:
        """
        Convert an argument value from a string to a boolean value.
        Throw :py:exc:`.ImagePreprocessorArgumentError` if there
        is an error parsing the value.

        :raises ImagePreprocessorArgumentError:

        :param name: the argument name for descriptive purposes,
            and/or for specifying the dictionary key when *value*
            is a dictionary.

        :param value: a boolean value as a string, or optionally a
            dictionary to get the value from using the argument *name*.

        :return: bool
        """

        if isinstance(value, dict):
            value = value.get(name)
        try:
            return _types.parse_bool(value)
        except ValueError:
            raise _exceptions.ImagePreprocessorArgumentError(
                f'Argument "{name}" must be a boolean value.')

    def argument_error(self, msg: str):
        raise _exceptions.ImagePreprocessorArgumentError(msg)

    def __init__(self,
                 called_by_name: str,
                 device: str = 'cpu',
                 output_file: _types.OptionalPath = None,
                 output_overwrite: bool = False, **kwargs):

        super().__init__(called_by_name, **kwargs)

        self.__output_file = output_file
        self.__output_overwrite = output_overwrite
        self.__device = device

    @property
    def device(self) -> str:
        """
        The rendering device requested for this preprocessor.

        :return: device string, for example "cuda", "cuda:N", or "cpu"
        """
        return self.__device

    def __gen_filename(self):
        return _filelock.touch_avoid_duplicate(os.path.dirname(self.__output_file),
                                               _filelock.suffix_path_maker(self.__output_file, '_'))

    def __save_debug_image(self, image, debug_header):
        if self.__output_file is not None:
            if not self.__output_overwrite:
                filename = self.__gen_filename()
            else:
                filename = self.__output_file

            image.save(filename)
            _messages.debug_log(f'{debug_header}: "{filename}"')

    @staticmethod
    def call_pre_resize(preprocessor,
                        image: PIL.Image.Image,
                        resize_resolution: _types.OptionalSize) -> PIL.Image.Image:
        """
        Invoke a preprocessors :py:meth:`.ImagePreprocessor.pre_resize` method.

        Implements important behaviors depending on if the image was modified.

        This is the only appropriate way to invoke a preprocessor manually.

        :param preprocessor: :py:class:`.ImagePreprocessor` implementation instance
        :param image: the image to pass
        :param resize_resolution: the size that the image is going to be resized
            to after this step, or None if it is not being resized.

        :return: processed image, may be the same image or a copy.
        """

        img_copy = image.copy()

        processed = preprocessor.pre_resize(image, resize_resolution)
        if processed is not image:
            preprocessor.__save_debug_image(
                processed,
                'Wrote Preprocessor Debug Image (because copied)')

            processed.filename = _image.get_filename(image)
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
    def call_post_resize(preprocessor,
                         image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Invoke a preprocessors :py:meth:`.ImagePreprocessor.post_resize` method.

        Implements important behaviors depending on if the image was modified.

        This is the only appropriate way to invoke a preprocessor manually.

        :param preprocessor: :py:class:`.ImagePreprocessor` implementation instance
        :param image: the image to pass

        :return: processed image, may be the same image or a copy.
        """

        img_copy = image.copy()

        processed = preprocessor.post_resize(image)
        if processed is not image:
            preprocessor.__save_debug_image(
                processed,
                'Wrote Preprocessor Debug Image (because copied)')

            processed.filename = _image.get_filename(image)
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
        """
        Implementation of pre_resize that does nothing. Inheritor must implement.

        This method should not be invoked directly, use the class method
        :py:meth:`.ImagePreprocessor.call_pre_resize` to invoke it.

        :param image: image to process
        :param resize_resolution: image will be resized to this resolution
            after this process is complete.  If None is passed no resize is
            going to occur. It is not the duty of the inheritor to resize the
            image, in fact it should NEVER be resized.

        :return: the processed image
        """
        return image

    def post_resize(self, image: PIL.Image.Image):
        """
        Implementation of post_resize that does nothing. Inheritor must implement.

        This method should not be invoked directly, use the class method
        :py:meth:`.ImagePreprocessor.call_post_resize` to invoke it.

        :param image: image to process
        :return: the processed image
        """
        return image

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)


__all__ = _types.module_all()
