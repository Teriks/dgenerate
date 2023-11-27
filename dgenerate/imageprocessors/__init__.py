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

import dgenerate.plugin as _plugin
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate import messages as _messages
from .canny import CannyEdgeDetectProcessor
from .exceptions import \
    ImageProcessorNotFoundError, \
    ImageProcessorArgumentError
from .imageops import \
    PosterizeProcessor, \
    SolarizeProcessor, \
    MirrorFlipProcessor, \
    SimpleColorProcessor
from .imageprocessor import ImageProcessor
from .imageprocessorchain import ImageProcessorChain
from .imageprocessorloader import ImageProcessorLoader
from .imageprocessormixin import ImageProcessorMixin
from .openpose import OpenPoseProcessor
from .upscaler import UpscalerProcessor


class ImageProcessorHelpUsageError(Exception):
    """
    Raised on argument parse errors in :py:func:`.image_processor_help`
    """
    pass


def image_processor_help(names: _types.Names,
                         plugin_module_paths: _types.Paths,
                         throw=False,
                         log_error=True):
    """
    Implements ``--image-processor-help`` command line option




    :param names: arguments (processor names, or empty list)
    :param plugin_module_paths: plugin module paths to search
    :param throw: throw on error? or simply print to stderr and return a return code.
    :param log_error: log errors to stderr?

    :raises ImageProcessorHelpUsageError:
    :raises ImageProcessorNotFoundError:

    :return: return-code, anything other than 0 is failure
    """

    module_loader = ImageProcessorLoader()

    try:
        module_loader.load_plugin_modules(plugin_module_paths)
    except _plugin.ModuleFileNotFoundError as e:
        if log_error:
            _messages.log(
                f'Plugin module could not be found: {str(e).strip()}',
                level=_messages.ERROR)
        if throw:
            raise ImageProcessorHelpUsageError(e)
        return 1

    if len(names) == 0:
        available = ('\n' + ' ' * 4).join(_textprocessing.quote(name) for name in module_loader.get_all_names())
        _messages.log(
            f'Available image processors:\n\n{" " * 4}{available}')
        return 0

    help_strs = []
    for name in names:
        try:
            help_strs.append(module_loader.get_help(name))
        except ImageProcessorNotFoundError:
            if log_error:
                _messages.log(f'An image processor with the name of "{name}" could not be found.',
                              level=_messages.ERROR)
            if throw:
                raise
            return 1

    for help_str in help_strs:
        _messages.log(help_str + '\n', underline=True)
    return 0


__all__ = _types.module_all()
