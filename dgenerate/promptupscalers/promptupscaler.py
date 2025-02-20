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

import torch

import dgenerate.promptupscalers.exceptions as _exceptions
import dgenerate.plugin as _plugin
import dgenerate.prompt as _prompt
import dgenerate.promptupscalers.constants as _constants
import dgenerate.messages as _messages
import dgenerate.devicecache as _devicecache
import dgenerate.memoize as _memoize
import dgenerate.memory as _memory

_prompt_upscaler_cache = _memoize.create_object_cache(
    'prompt_upscaler',
    cache_type=_memory.SizedConstrainedObjectCache
)


def _cache_debug_hit(key, hit):
    _memoize.simple_cache_hit_debug("Prompt Upscaler Model", key, hit)


def _cache_debug_miss(key, new):
    _memoize.simple_cache_miss_debug("Prompt Upscaler Model", key, new)


class PromptUpscaler(_plugin.Plugin):
    """
    Abstract base class for prompt upscaler implementations.
    """

    # you cannot specify these via a URI
    HIDE_ARGS = ['local-files-only']

    def __init__(self,
                 loaded_by_name: str,
                 device: str,
                 local_files_only: bool = False,
                 **kwargs):
        """
        :param loaded_by_name: The name the prompt upscaler was loaded by
        :param local_files_only: if ``True``, the plugin should never try to download
            models from the internet automatically, and instead only look
            for them in cache / on disk.
        :param kwargs: child class forwarded arguments
        """
        super().__init__(loaded_by_name=loaded_by_name,
                         argument_error_type=_exceptions.PromptUpscalerArgumentError,
                         **kwargs)

        self.__device = device
        self.__local_files_only = local_files_only

    @staticmethod
    def memory_fence_device(device: str | torch.device, memory_required: int):
        """
        Check a specific device against an amount of memory in bytes.

        If any of the memory constraints specified by :py:attr:`dgenerate.promptupscalers.constants.PROMPT_UPSCALER_CUDA_MEMORY_CONSTRAINTS`
        are met on that device, attempt to remove cached objects off the device to free space.

        :param device: the device
        :param memory_required: the amount of memory required on the device in bytes
        :return: ``True`` if an attempt was made to free memory, ``False`` otherwise.
        """

        device = torch.device(device)

        if (device.type == 'cuda' and _memory.cuda_memory_constraints(
                _constants.PROMPT_UPSCALER_CUDA_MEMORY_CONSTRAINTS,
                extra_vars={'memory_required': memory_required},
                device=device)):
            _devicecache.clear_device_cache(device)
            return True
        return False

    def set_size_estimate(self, size_bytes: int):
        """
        Set the estimated size of this plugin in bytes for memory management
        heuristics, this is intended to be used by implementors of the
        :py:class:`PromptUpscaler` plugin class.

        For the best memory optimization, this value should be set very
        shortly before any associated model even enters CPU side ram,
        IE: before it is loaded at all.

        :raise ValueError: if ``size_bytes`` is less than zero.

        :param size_bytes: the size in bytes
        """
        if size_bytes < 0:
            raise ValueError(
                'prompt upscaler size estimate cannot be less than zero.')

        self.__size_estimate = int(size_bytes)

        _prompt_upscaler_cache.enforce_cpu_mem_constraints(
            _constants.PROMPT_UPSCALER_CACHE_MEMORY_CONSTRAINTS,
            size_var='upscaler_size',
            new_object_size=self.__size_estimate
        )

        if (_memory.memory_constraints(
                _constants.PROMPT_UPSCALER_CACHE_GC_CONSTRAINTS,
                extra_vars={'upscaler_size': self.__size_estimate})):
            # wipe out the cpu side diffusion pipelines cache
            # and do a GC pass to free up cpu side memory since
            # we are nearly out of memory anyway

            _messages.debug_log(
                f'Prompt upscaler "{self.__class__.__name__}" is clearing the CPU side object '
                f'cache due to CPU side memory constraint evaluating to to True.')

            _memoize.clear_object_caches()

    def load_object_cached(self, tag: str, estimated_size: int, method: typing.Callable):
        """
        Load a potentially large object into the CPU side ``prompt_upscaler`` object cache.

        :param tag: A unique string within the context of the image
            processor implementation constructor.
        :param estimated_size: Estimated size in bytes of the object in RAM.
        :param method: A method which loads and returns the object.
        :return: The loaded object
        """

        @_memoize.memoize(
            _prompt_upscaler_cache,
            on_hit=_cache_debug_hit,
            on_create=_cache_debug_miss)
        def load_cached(loaded_by_name=self.loaded_by_name, tag=tag):
            return method(), _memoize.CachedObjectMetadata(size=estimated_size)

        return load_cached()

    @property
    def device(self):
        """
        Device that will be used for any text processing models.
        """
        return self.__device

    @property
    def local_files_only(self) -> bool:
        """
        Is this prompt upscaler only going to look for Hugging Face hub model files in cache / on disk?
        """
        return self.__local_files_only

    def upscale(self, prompt: _prompt.Prompt) -> _prompt.Prompts:
        """
        Upscale a prompt and return it modified
        :param prompt: The incoming prompt
        :return: Modified prompts, you may return multiple prompts to indicate expansion
        """
        return [prompt]
