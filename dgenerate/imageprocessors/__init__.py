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
import collections.abc

import dgenerate.imageprocessors.constants
import dgenerate.plugin as _plugin
import dgenerate.types as _types


import spandrel_extra_arches

# Enable extra architectures (only if not already installed)
spandrel_extra_arches.install(ignore_duplicates=True)

# these need to be up here to prevent circular import problems
from .imageprocessor import ImageProcessor
from .imageprocessorchain import ImageProcessorChain
from .imageprocessorloader import ImageProcessorLoader
from .imageprocessormixin import ImageProcessorMixin

from .exceptions import (
    ImageProcessorNotFoundError,
    ImageProcessorArgumentError,
    ImageProcessorImageModeError,
    ImageProcessorError,
)

# =====================


from .adetailer import AdetailerProcessor
from .anyline import AnylineProcessor
from .canny import CannyEdgeDetectProcessor
from .crop_to_mask import CropToMaskProcessor
from .outpaint_mask import OutpaintMaskProcessor
from .cv2imageops import (
    DilateProcessor,
    GaussianBlurProcessor
)
from .hed import HEDProcessor
from .imageops import (
    PosterizeProcessor,
    SolarizeProcessor,
    MirrorFlipProcessor,
    SimpleColorProcessor,
    LetterboxProcessor,
    ResizeProcessor,
    CropProcessor
)

from .leres import LeresDepthProcessor
from .linart_anime import LineArtAnimeProcessor
from .linart_standard import LineArtStandardProcessor
from .lineart import LineArtProcessor
from .midas import MidasDepthProcessor
from .mlsd import MLSDProcessor
from .normal_bae import NormalBaeProcessor
from .openpose import OpenPoseProcessor
from .inpaint import InpaintProcessor
from .paste import PasteProcessor
from .patchmatch import PatchMatchProcessor
from .pidi import PidiNetProcessor
from .sam import SegmentAnythingProcessor
from .u_sam import USAMProcessor
from .teed import TEEDProcessor
from .upscaler import UpscalerProcessor
from .yolo import YOLOProcessor
from .yolo_sam import YOLOSAMProcessor
from .zoe import ZoeDepthProcessor

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


def image_processor_names():
    """
    Implementation names for all image processors implemented by dgenerate,
    which are visible to the default :py:class:`ImageProcessorLoader` instance.

    :return: a list of latents processor implementation names.
    """

    return list(ImageProcessorLoader().get_all_names())


def image_processor_name_from_uri(uri: _types.Uri):
    """
    Extract just the implementation name from a image processor URI.

    :param uri: the URI
    :return: the implementation name.
    """

    return uri.split(';')[0].strip()


def image_processor_exists(uri: _types.Uri):
    """
    Check if a image processor implementation exists for a given URI.

    This uses the default :py:class:`ImageProcessorLoader` instance.

    :param uri: The image processor URI
    :return: ``True`` or ``False``
    """
    return image_processor_name_from_uri(uri) in image_processor_names()


def create_image_processor(uri: _types.Uri | collections.abc.Iterable[_types.Uri],
                           output_file: str | None = None,
                           output_overwrite: bool = True,
                           device: str = 'cpu',
                           model_offload: bool = False,
                           local_files_only: bool = False) -> ImageProcessor:
    """
    Create an image processor implementation using the default :py:class:`ImageProcessorLoader` instance.

    Providing a collection of URIs will create an :py:class:`ImageProcessorChain` object.

    :param output_file: Output path for the processor debug image
    :param output_overwrite: enable overwrite for the processor debug image?
    :param uri: The image processor URI
    :param device: Device to run processing on
    :param model_offload: enable cpu model offloading?
    :param local_files_only: Should the processor avoid downloading
        files from Hugging Face hub and only check the cache or local directories?
    :return: A :py:class:`ImageProcessor` implementation
    """
    return ImageProcessorLoader().load(
        uri,
        output_file=output_file,
        output_overwrite=output_overwrite,
        device=device,
        model_offload=model_offload,
        local_files_only=local_files_only
    )


__all__ = _types.module_all()
