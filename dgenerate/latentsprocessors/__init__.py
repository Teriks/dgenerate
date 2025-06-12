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

import dgenerate.latentsprocessors.exceptions as _exceptions
import dgenerate.latentsprocessors.latentsprocessor as _latentsprocessor
import dgenerate.latentsprocessors.latentsprocessorloader as _latentsprocessorloader
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.plugin as _plugin
import dgenerate.types as _types

__doc__ = """
Latents tensor processor plugin implementations and support code.

This module provides a plugin framework for processing latents tensors.
"""

from dgenerate.latentsprocessors.exceptions import (
    LatentsProcessorError,
    LatentsProcessorArgumentError,
    LatentsProcessorNotFoundError
)

from dgenerate.latentsprocessors.latentsprocessorloader import (
    LatentsProcessorLoader
)

from dgenerate.latentsprocessors.latentsprocessor import (
    LatentsProcessor
)

from dgenerate.latentsprocessors.latentsprocessorchain import (
    LatentsProcessorChain
)

from dgenerate.latentsprocessors.noise import NoiseProcessor
from dgenerate.latentsprocessors.scale import ScaleProcessor
from dgenerate.latentsprocessors.interposer import InterposerProcessor

import dgenerate.latentsprocessors.constants


def latents_processor_help(names: _types.Names,
                           plugin_module_paths: _types.OptionalPaths = None,
                           throw=False,
                           log_error=True):
    """
    Implements ``--latents-processor-help`` command line option

    :param names: arguments (processor names, or empty list)
    :param plugin_module_paths: extra plugin module paths to search
    :param throw: throw on error? or simply print to stderr and return a return code.
    :param log_error: log errors to stderr?

    :raises LatentsProcessorNotFoundError:
    :raises dgenerate.ModuleFileNotFoundError:

    :return: return-code, anything other than 0 is failure
    """

    try:
        return LatentsProcessorLoader().loader_help(
            names=names,
            plugin_module_paths=plugin_module_paths,
            title='latents processor',
            title_plural='latents processors',
            throw=True,
            log_error=log_error,
            include_bases=True)
    except (LatentsProcessorNotFoundError, _plugin.ModuleFileNotFoundError) as e:
        if throw:
            raise e
        return 1


def latents_processor_names():
    """
    Implementation names for all latents processors implemented by dgenerate,
    which are visible to the default :py:class:`LatentsProcessorLoader` instance.

    :return: a list of latents processor implementation names.
    """

    return list(LatentsProcessorLoader().get_all_names())


def latents_processor_name_from_uri(uri: _types.Uri):
    """
    Extract just the implementation name from a latents processor URI.

    :param uri: the URI
    :return: the implementation name.
    """

    return uri.split(';')[0].strip()


def latents_processor_exists(uri: _types.Uri):
    """
    Check if a latents processor implementation exists for a given URI.

    This uses the default :py:class:`LatentsProcessorLoader` instance.

    :param uri: The latents processor URI
    :return: ``True`` or ``False``
    """
    return latents_processor_name_from_uri(uri) in latents_processor_names()


def create_latents_processor(uri: _types.Uri,
                             model_type: _enums.ModelType,
                             device: str = 'cpu',
                             local_files_only: bool = False) -> LatentsProcessor:
    """
    Create a latents processor implementation using the default :py:class:`LatentsProcessorLoader` instance.

    :param uri: The latents processor URI
    :param model_type: Model type enum the processor is expected to handle
    :param device: Device to run processing on
    :param local_files_only: Should the processor avoid downloading
        files from Hugging Face hub and only check the cache or local directories?
    :return: A :py:class:`LatentsProcessor` implementation
    """
    return LatentsProcessorLoader().load(
        uri,
        model_type=model_type,
        device=device,
        local_files_only=local_files_only
    )


__all__ = _types.module_all()
