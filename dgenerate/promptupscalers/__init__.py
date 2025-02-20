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

import dgenerate.plugin as _plugin
import dgenerate.types as _types
import dgenerate.prompt as _prompt
from .promptupscalerloader import PromptUpscalerLoader
from .promptupscaler import PromptUpscaler
from .dynamicpromptspromptupscaler import DynamicPromptsPromptUpscaler

from .exceptions import (
    PromptUpscalerNotFoundError,
    PromptUpscalerArgumentError,
    PromptUpscalerProcessingError
)

from .constants import (
    PROMPT_UPSCALER_CUDA_MEMORY_CONSTRAINTS,
    PROMPT_UPSCALER_CACHE_GC_CONSTRAINTS,
    PROMPT_UPSCALER_CACHE_MEMORY_CONSTRAINTS
)


def prompt_upscaler_help(names: _types.Names,
                         plugin_module_paths: _types.OptionalPaths = None,
                         throw=False,
                         log_error=True):
    """
    Implements ``--prompt-upscaler-help`` command line option

    :param names: arguments (prompt upscaler names, or empty list)
    :param plugin_module_paths: extra plugin module paths to search
    :param throw: throw on error? or simply print to stderr and return a return code.
    :param log_error: log errors to stderr?

    :raises PromptUpscalerNotFoundError:
    :raises dgenerate.ModuleFileNotFoundError:

    :return: return-code, anything other than 0 is failure
    """

    try:
        return PromptUpscalerLoader().loader_help(
            names=names,
            title='prompt upscaler',
            title_plural='prompt upscalers',
            plugin_module_paths=plugin_module_paths,
            throw=True,
            log_error=log_error)
    except (PromptUpscalerNotFoundError, _plugin.ModuleFileNotFoundError) as e:
        if throw:
            raise e
        return 1


def prompt_upscaler_names():
    """
    Implementation names for all prompt upscalers implemented by dgenerate,
    which are visible to the default :py:class:`PromptUpscalerLoader` instance.

    :return: a list of prompt upscaler implementation names.
    """

    return list(PromptUpscalerLoader().get_all_names())


def prompt_upscaler_name_from_uri(uri):
    """
    Extract just the implementation name from a prompt upscaler URI.

    :param uri: the URI
    :return: the implementation name.
    """

    return uri.split(';')[0].strip()


def prompt_upscaler_exists(uri):
    """
    Check if a prompt upscaler implementation exists for a given URI.

    This uses the default :py:class:`PromptUpscalerLoader` instance.

    :param uri: The prompt upscaler URI
    :return: ``True`` or ``False``
    """
    return prompt_upscaler_name_from_uri(uri) in prompt_upscaler_names()


def create_prompt_upscaler(uri, device: str = 'cpu', local_files_only: bool = False) -> PromptUpscaler:
    """
    Create a prompt upscaler implementation using the default :py:class:`PromptUpscalerLoader` instance.

    :param uri: The prompt upscaler URI
    :param device: Device used for any text processing models.
    :param local_files_only: Should the prompt upscaler avoid downloading
        files from Hugging Face hub and only check the cache or local directories?
    :return: A :py:class:`PromptUpscaler` implementation
    """
    return PromptUpscalerLoader().load(uri, device=device, local_files_only=local_files_only)


def upscale_prompts(
        prompts: _prompt.Prompts,
        default_upscaler_uri: _types.OptionalUriOrUris = None,
        device: str = 'cpu',
        local_files_only: bool = False
):
    """
    Apply prompt upscaling to a list of prompts and return a possibly expanded list of prompts.

    :param prompts: Input prompt objects.
    :param default_upscaler_uri: The default upscaler plugin URI, or a list of URIs (chain upscalers together sequentially)
    :param device: Execution device for upscalers that can utilize hardware acceleration
    :param local_files_only: Should all prompt upscalers involved avoid downloading
        files from Hugging Face hub and only check the cache or local directories?
    :return: Altered prompts list
    """

    upscaled_prompts = []

    for prompt in prompts:

        upscaler_uris = _types.default(prompt.upscaler, default_upscaler_uri)
        if not upscaler_uris:
            upscaled_prompts.append(prompt)
            continue

        if not isinstance(upscaler_uris, list):
            upscaler_uris = [upscaler_uris]

        # start processing with the original prompt
        current_prompts = [prompt]

        for upscaler_uri in upscaler_uris:
            upscaler = create_prompt_upscaler(
                upscaler_uri, device=device, local_files_only=local_files_only
            )

            # apply upscaler exactly once per prompt
            next_prompts = []
            for p in current_prompts:
                new_results = upscaler.upscale(_prompt.Prompt.copy(p))

                if not isinstance(new_results, typing.Iterable):
                    new_results = [new_results]

                # ensure each generated prompt is only processed once per upscaler
                next_prompts.extend(new_results)

            current_prompts = next_prompts  # Move forward in the pipeline

        # collect final results after all upscalers have been applied in sequence
        upscaled_prompts.extend(current_prompts)

    return upscaled_prompts


__all__ = _types.module_all()
