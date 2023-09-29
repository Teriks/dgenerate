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

from .exceptions import ImagePreprocessorArgumentError


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
        pass

    def pre_resize(self, resize_resolution: typing.Union[None, tuple], image: PIL.Image):
        return image

    def post_resize(self, resize_resolution: typing.Union[None, tuple], image: PIL.Image):
        return image
