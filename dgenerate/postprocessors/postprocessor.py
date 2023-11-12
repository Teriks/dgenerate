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

import dgenerate.plugin as _plugin
import dgenerate.postprocessors.exceptions as _exceptions
import dgenerate.types as _types


class ImagePostprocessor(_plugin.InvokablePlugin):
    """
    Abstract base class for image postprocessor implementations.
    """

    def __init__(self, called_by_name, device: str = 'cpu', **kwargs):
        super().__init__(called_by_name, **kwargs)
        self.__device = device

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
            raise _exceptions.ImagePostprocessorArgumentError(f'Argument "{name}" must be an integer value.')

    @staticmethod
    def get_float_arg(name: str, value: typing.Union[str, float, typing.Dict]) -> float:
        """
        Convert an argument value from a string to a float.
        Throw :py:exc:`.ImagePostprocessorArgumentError` if there
        is an error parsing the value.

        :raises ImagePostprocessorArgumentError:

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
            raise _exceptions.ImagePostprocessorArgumentError(f'Argument "{name}" must be a floating point value.')

    @staticmethod
    def get_bool_arg(name: str, value: typing.Union[str, bool, typing.Dict]) -> bool:
        """
        Convert an argument value from a string to a boolean value.
        Throw :py:exc:`.ImagePostprocessorArgumentError` if there
        is an error parsing the value.

        :raises ImagePostprocessorArgumentError:

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
            raise _exceptions.ImagePostprocessorArgumentError(
                f'Argument "{name}" must be a boolean value.')

    def argument_error(self, msg: str):
        raise _exceptions.ImagePostprocessorArgumentError(msg)

    @property
    def device(self) -> str:
        """
        The rendering device requested for this postprocessor.

        :return: device string, for example "cuda", "cuda:N", or "cpu"
        """
        return self.__device

    def process(self, image: PIL.Image.Image) -> PIL.Image.Image:
        return image

    @staticmethod
    def call_process(postprocessor,
                     image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Invoke a postprocessors :py:meth:`.ImagePostprocessor.process` method.

        This is the only appropriate way to invoke a postprocessor manually.

        :param postprocessor: :py:class:`.ImagePostprocessor` implementation instance
        :param image: the image to pass

        :return: processed image, may be the same image or a copy.
        """

        return postprocessor.process(image)
