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

from .canny import *
from .exceptions import *
from .loader import *
from .openpose import *
from .pil_imageops import *
from .preprocessor import ImagePreprocessor
from .preprocessorchain import ImagePreprocessorChain
from .preprocessormixin import ImagePreprocessorMixin
from .. import messages
from ..textprocessing import quote


def image_preprocessor_help(args):
    if len(args) == 0:
        messages.log(
            f'Available image preprocessors:\n\n{" " * 4}{(chr(10) + " " * 4).join(quote(name) for name in get_all_names())}')
        return 0

    help_strs = []
    for name in args:
        try:
            help_strs.append(get_help(name))
        except ImagePreprocessorNotFoundError:
            messages.log(f'An image preprocessor with the name of "{name}" could not be found!',
                         level=messages.ERROR)
            return 1

    for help_str in help_strs:
        messages.log(help_str, underline=True)
    return 0
