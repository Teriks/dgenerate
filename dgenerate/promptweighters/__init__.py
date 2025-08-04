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
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.types as _types
from .compelpromptweighter import CompelPromptWeighter

from .exceptions import (
    PromptWeighterError,
    PromptWeighterArgumentError,
    PromptWeighterNotFoundError,
    PromptWeightingUnsupported
)

from .promptweighter import PromptWeighter
from .promptweighterloader import PromptWeighterLoader
from .sdembedpromptweighter import SdEmbedPromptWeighter
from .llm4genpromptweighter import LLM4GENPromptWeighter

import dgenerate.promptweighters.constants


def prompt_weighter_help(names: _types.Names,
                         plugin_module_paths: _types.OptionalPaths = None,
                         throw=False,
                         log_error=True):
    """
    Implements ``--prompt-weighter-help`` command line option

    :param names: arguments (prompt weighter names, or empty list)
    :param plugin_module_paths: extra plugin module paths to search
    :param throw: throw on error? or simply print to stderr and return a return code.
    :param log_error: log errors to stderr?

    :raises PromptWeighterNotFoundError:
    :raises dgenerate.ModuleFileNotFoundError:

    :return: return-code, anything other than 0 is failure
    """

    try:
        return PromptWeighterLoader().loader_help(
            names=names,
            title='prompt weighter',
            title_plural='prompt weighters',
            plugin_module_paths=plugin_module_paths,
            throw=True,
            log_error=log_error)
    except (PromptWeighterNotFoundError, _plugin.ModuleFileNotFoundError) as e:
        if throw:
            raise e
        return 1


def prompt_weighter_names():
    """
    Implementation names for all prompt weighters implemented by dgenerate,
    which are visible to the default :py:class:`PromptWeighterLoader` instance.

    :return: a list of prompt weighter implementation names.
    """

    return list(PromptWeighterLoader().get_all_names())


def prompt_weighter_name_from_uri(uri: _types.Uri):
    """
    Extract just the implementation name from a prompt weighter URI.

    :param uri: the URI
    :return: the implementation name.
    """

    return uri.split(';')[0].strip()


def prompt_weighter_exists(uri: _types.Uri):
    """
    Check if a prompt weighter implementation exists for a given URI.

    This uses the default :py:class:`PromptWeighterLoader` instance.

    :param uri: The prompt weighter URI
    :return: ``True`` or ``False``
    """
    return prompt_weighter_name_from_uri(uri) in prompt_weighter_names()


def create_prompt_weighter(uri: _types.Uri,
                           model_type: _enums.ModelType,
                           dtype: _enums.DataType,
                           device: str | None = None,
                           local_files_only: bool = False,
                           ) -> PromptWeighter:
    """
    Create a prompt weighter implementation using the default :py:class:`PromptWeighterLoader` instance.

    :param uri: The prompt weighter URI
    :param model_type: Model type the prompt weighter is expected to handle
    :param dtype: The dtype of the pipeline
    :param device: The device the prompt weighter should operate on
    :param local_files_only: Should the prompt weighter avoid downloading
        files from Hugging Face hub and only check the cache or local directories?
    :return: A :py:class:`PromptWeighter` implementation
    """
    return PromptWeighterLoader().load(
        uri,
        model_type=model_type,
        dtype=dtype,
        device=device,
        local_files_only=local_files_only
    )


__all__ = _types.module_all()
