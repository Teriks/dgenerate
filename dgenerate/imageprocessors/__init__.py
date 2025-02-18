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
import dgenerate.types as _types

from .canny import CannyEdgeDetectProcessor

from .exceptions import (
    ImageProcessorNotFoundError,
    ImageProcessorArgumentError,
    ImageProcessorImageModeError,
    ImageProcessorError,
)

from .hed import HEDProcessor

from .imageops import (
    PosterizeProcessor,
    SolarizeProcessor,
    MirrorFlipProcessor,
    SimpleColorProcessor,
    LetterboxProcessor,
    ResizeProcessor
)

from .imageprocessor import ImageProcessor
from .imageprocessorchain import ImageProcessorChain
from .imageprocessorloader import ImageProcessorLoader
from .imageprocessormixin import ImageProcessorMixin

from .leres import LeresDepthProcessor
from .linart_anime import LineArtAnimeProcessor
from .lineart import LineArtProcessor
from .midas import MidasDepthProcessor
from .mlsd import MLSDProcessor
from .normal_bae import NormalBaeProcessor
from .openpose import OpenPoseProcessor
from .pidi import PidiNetProcessor
from .sam import SegmentAnythingProcessor
from .upscaler import UpscalerProcessor
from .zoe import ZoeDepthProcessor
from .teed import TEEDProcessor
from .linart_standard import LineArtStandardProcessor
from .anyline import AnylineProcessor
from .adetailer import AdetailerProcessor

from .constants import (
    IMAGE_PROCESSOR_CACHE_MEMORY_CONSTRAINTS,
    IMAGE_PROCESSOR_CPU_CACHE_GC_CONSTRAINTS,
    IMAGE_PROCESSOR_CUDA_MEMORY_CONSTRAINTS
)

try:
    import ncnn
    from .upscaler_ncnn import UpscalerNCNNProcessor
except ImportError:
    pass

__doc__ = """
Image processors implemented by dgenerate.

This includes many image processing tasks useful for creating diffusion input images, or for postprocessing.
"""


def image_processor_help(names: _types.Names,
                         plugin_module_paths: _types.OptionalPaths = None,
                         throw=False,
                         log_error=True):
    """
    Implements ``--image-processor-help`` command line option

    :param names: arguments (processor names, or empty list)
    :param plugin_module_paths: extra plugin module paths to search
    :param throw: throw on error? or simply print to stderr and return a return code.
    :param log_error: log errors to stderr?

    :raises ImageProcessorNotFoundError:
    :raises dgenerate.ModuleFileNotFoundError:

    :return: return-code, anything other than 0 is failure
    """

    try:
        return ImageProcessorLoader().loader_help(
            names=names,
            plugin_module_paths=plugin_module_paths,
            title='image processor',
            title_plural='image processors',
            throw=True,
            log_error=log_error,
            include_bases=True)
    except (ImageProcessorNotFoundError, _plugin.ModuleFileNotFoundError) as e:
        if throw:
            raise e
        return 1


__all__ = _types.module_all()
