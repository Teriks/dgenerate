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
import argparse
import typing

import dgenerate.types as _types
import dgenerate.textprocessing as _textprocessing
from dgenerate import messages as _messages
from .canny import CannyEdgeDetectPreprocess
from .exceptions import *
from .imageops import *
from .loader import Loader
from .openpose import OpenPosePreprocess
from .preprocessor import ImagePreprocessor
from .preprocessorchain import ImagePreprocessorChain
from .preprocessormixin import ImagePreprocessorMixin

_help_parser = argparse.ArgumentParser(prog='dgenerate', exit_on_error=False)
_help_parser.add_argument('--image-preprocessor-help', nargs='*', default=[], type=str)
_help_parser.add_argument('--plugin-modules', nargs='+', default=[], type=str)


class PreprocessorHelpUsageError(Exception):
    """
    Raised on argument parse errors in :py:meth:`.image_preprocessor_help`
    """
    pass


def image_preprocessor_help(args: typing.Sequence[str], throw: bool = False):
    """
    Implements ``--image-preprocessor-help`` command line option

    :param args: arguments (preprocessor names, or empty list)
    :param throw: Should we throw exceptions or handle them?

    :raises: :py:exc:`.PreprocessorHelpUsageError`
             :py:exc:`dgenerate.preprocessors.exceptions.ImagePreprocessorNotFoundError`

    :return: return-code, anything other than 0 is failure
    """
    try:
        parse_result = _help_parser.parse_args(args)
    except (argparse.ArgumentError, argparse.ArgumentTypeError) as e:
        _messages.log(f'dgenerate: error: {e}', level=_messages.ERROR)
        if throw:
            raise PreprocessorHelpUsageError(e)
        return 1

    names = parse_result.image_preprocessor_help

    module_loader = Loader()
    module_loader.load_plugin_modules(parse_result.plugin_modules)

    if len(names) == 0:
        available = ('\n' + ' ' * 4).join(_textprocessing.quote(name) for name in module_loader.get_all_names())
        _messages.log(
            f'Available image preprocessors:\n\n{" " * 4}{available}')
        return 0

    help_strs = []
    for name in names:
        try:
            help_strs.append(module_loader.get_help(name))
        except ImagePreprocessorNotFoundError:
            _messages.log(f'An image preprocessor with the name of "{name}" could not be found!',
                          level=_messages.ERROR)
            return 1

    for help_str in help_strs:
        _messages.log(help_str + '\n', underline=True)
    return 0
