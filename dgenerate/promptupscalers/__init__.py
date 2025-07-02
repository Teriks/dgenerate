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
from .dynamicpromptsupscaler import DynamicPromptsUpscaler
from .magicpromptupscaler import MagicPromptUpscaler
from .attentionpromptupscaler import AttentionUpscaler
from .translatepromptupscaler import TranslatePromptUpscaler
import collections.abc

try:
    import gpt4all
    from .gpt4allpromptupscaler import GPT4ALLPromptUpscaler
except ImportError:
    pass

from .exceptions import (
    PromptUpscalerError,
    PromptUpscalerNotFoundError,
    PromptUpscalerArgumentError,
    PromptUpscalerProcessingError
)

import dgenerate.promptupscalers.constants


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
            include_bases=True,
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


def prompt_upscaler_name_from_uri(uri: _types.Uri):
    """
    Extract just the implementation name from a prompt upscaler URI.

    :param uri: the URI
    :return: the implementation name.
    """

    return uri.split(';')[0].strip()


def prompt_upscaler_exists(uri: _types.Uri):
    """
    Check if a prompt upscaler implementation exists for a given URI.

    This uses the default :py:class:`PromptUpscalerLoader` instance.

    :param uri: The prompt upscaler URI
    :return: ``True`` or ``False``
    """
    return prompt_upscaler_name_from_uri(uri) in prompt_upscaler_names()


def create_prompt_upscaler(uri: _types.Uri, device: str = 'cpu', local_files_only: bool = False) -> PromptUpscaler:
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
        files only check the cache or locally specified files?
    :return: Altered prompts list
    """

    if not prompts:
        return []

    unique_upscaler_chains = {}

    for prompt in prompts:
        upscaler_uris = _types.default(prompt.upscaler, default_upscaler_uri)
        if not isinstance(upscaler_uris, list):
            upscaler_uris = [upscaler_uris]
        unique_upscaler_chains.setdefault(tuple(upscaler_uris), []).append(prompt)

    upscaled_prompts = []

    for upscaler_uris, batch_prompts in unique_upscaler_chains.items():

        for upscaler_uri in upscaler_uris:
            if upscaler_uri is None:
                continue

            upscaler = create_prompt_upscaler(
                upscaler_uri, device=device, local_files_only=local_files_only
            )

            next_prompts = []

            if upscaler.accepts_batch:
                batch_inputs = [_prompt.Prompt.copy(p) for p in batch_prompts]
                batch_outputs = upscaler.upscale(batch_inputs)

                if not isinstance(batch_outputs, collections.abc.Iterable):
                    batch_outputs = [batch_outputs]

                next_prompts.extend(batch_outputs)

            else:
                for p in batch_prompts:
                    new_results = upscaler.upscale(_prompt.Prompt.copy(p))

                    if not isinstance(new_results, collections.abc.Iterable):
                        new_results = [new_results]

                    next_prompts.extend(new_results)

            batch_prompts = next_prompts

        upscaled_prompts.extend(batch_prompts)

    return upscaled_prompts


__all__ = _types.module_all()
