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

from .openpose import *
from .canny import *
from .loader import *
from .pil_imageops import *
from .preprocessor import ImagePreprocessor
from .preprocessorchain import ImagePreprocessorChain
from .preprocessormixin import ImagePreprocessorMixin
from .. import messages
from ..textprocessing import quote, long_text_wrap_width
import textwrap


def image_preprocessor_help(args):
    if len(args) == 0:
        messages.log(
            f'Available image preprocessors:\n\n{" "*4}{(chr(10)+" "*4).join(quote(name) for name in  get_all_names())}')
        return 0

    help_strs = []
    for name in args:
        try:
            help_strs.append((name, get_help(name)))
        except ImagePreprocessorNotFoundError:
            messages.log(f'An image preprocessor with the name of "{name}" could not be found!',
                         level=messages.ERROR)
            return 1

    for name, help_str in help_strs:
        args_with_defaults = get_class_by_name(name).get_accepted_args_with_defaults(name)

        arg_descriptors = []
        for arg in args_with_defaults:
            if len(arg) == 1:
                arg_descriptors.append(arg[0])
            else:
                default_value = arg[1]
                if isinstance(default_value, str):
                    default_value = quote(default_value)
                arg_descriptors.append(f'{arg[0]}={default_value}')

        if arg_descriptors:
            args_part = f'\n{" "*4}arguments:\n{" "*8}{(chr(10)+" "*8).join(arg_descriptors)}\n'
        else:
            args_part = ''

        if help_str:
            messages.log(name + f':{args_part}\n'+textwrap.fill(help_str,
                                       initial_indent='    ',
                                       subsequent_indent='    ',
                                       width=long_text_wrap_width()), underline=True)
        else:
            messages.log(name + f':{args_part}', underline=True)
    return 0


