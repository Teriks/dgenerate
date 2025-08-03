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
import abc
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
import dgenerate.torchutil as _torchutil

_prompt_upscaler_cache = _memoize.create_object_cache(
    'prompt_upscaler',
    cache_type=_memory.SizedConstrainedObjectCache
)


def _cache_debug_hit(key, hit):
    _memoize.simple_cache_hit_debug("Prompt Upscaler Model", key, hit)


def _cache_debug_miss(key, new):
    _memoize.simple_cache_miss_debug("Prompt Upscaler Model", key, new)


class PromptUpscaler(_plugin.Plugin, abc.ABC):
    """
    Abstract base class for prompt upscaler implementations.
    """

    # you cannot specify these via a URI
    HIDE_ARGS = ['local-files-only']

    @classmethod
    def inheritable_help(cls, loaded_by_name):
        help_messages = {
            'device': (
                'The "device" argument can be used to set the device '
                'the prompt upscaler will run any models on, for example: cpu, cuda, cuda:1. '
                'this argument will default to the value of the dgenerate argument --device.'
            )
        }
        return help_messages

    def __init__(self,
                 loaded_by_name: str,
                 device: str | None = None,
                 local_files_only: bool = False,
                 **kwargs):
        """
        :param loaded_by_name: The name the prompt upscaler was loaded by
        :param device: Torch device string for running any models, passing
            ``None`` defaults the device to ``cpu``
        :param local_files_only: if ``True``, the plugin should never try to download
            models from the internet automatically, and instead only look
            for them in cache / on disk.
        :param kwargs: child class forwarded arguments
        """
        super().__init__(loaded_by_name=loaded_by_name,
                         argument_error_type=_exceptions.PromptUpscalerArgumentError,
                         **kwargs)

        if device is not None:
            if not _torchutil.is_valid_device_string(device):
                raise _exceptions.PromptUpscalerArgumentError(
                    f'Invalid device argument, {_torchutil.invalid_device_message(device, cap=False)}')

        self.__device = device if device else 'cpu'
        self.__local_files_only = local_files_only
        self.__size_estimate = 0

    def memory_guard_device(self, device: str | torch.device, memory_required: int):
        """
        Check a specific device against an amount of memory in bytes.

        If the device is a gpu device and any of the memory constraints specified by
        :py:attr:`dgenerate.promptupscalers.constants.PROMPT_UPSCALER_GPU_MEMORY_CONSTRAINTS`
        are met on that device, attempt to remove cached objects off a gpu device to free space.

        If the device is a cpu and any of the memory constraints specified by
        :py:attr:`dgenerate.promptupscalers.constants.PROMPT_UPSCALER_CACHE_GC_CONSTRAINTS`
        are met, attempt to remove cached prompt upscaler objects off the device to free space.
        Then, enforce :py:attr:`dgenerate.promptupscalers.constants.PROMPT_UPSCALER_CACHE_MEMORY_CONSTRAINTS`.

        :param device: the device
        :param memory_required: the amount of memory required on the device in bytes
        :return: ``True`` if an attempt was made to free memory, ``False`` otherwise.
        """

        device = torch.device(device)
        cleared = False

        if _memory.is_supported_gpu_device(device):
            if _memory.gpu_memory_constraints(
                    _constants.PROMPT_UPSCALER_GPU_MEMORY_CONSTRAINTS,
                    extra_vars={'memory_required': memory_required},
                    device=device):
                _messages.debug_log(
                    f'Prompt Upscaler "{self.__class__.__name__}" is clearing the GPU side object '
                    f'cache due to GPU side memory constraint evaluating to to True.')
                _devicecache.clear_device_cache(device)
                cleared = True

        elif device.type == 'cpu':
            if (_memory.memory_constraints(
                    _constants.PROMPT_UPSCALER_CACHE_GC_CONSTRAINTS,
                    extra_vars={'memory_required': memory_required})):
                _messages.debug_log(
                    f'Prompt upscaler "{self.__class__.__name__}" is clearing the CPU side object '
                    f'cache due to CPU side memory constraint evaluating to to True.')

                _memoize.clear_object_caches()
                cleared = True

            cleared = cleared or _prompt_upscaler_cache.enforce_cpu_mem_constraints(
                _constants.PROMPT_UPSCALER_CACHE_MEMORY_CONSTRAINTS,
                size_var='memory_required',
                new_object_size=memory_required
            )
        return cleared

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

    def load_object_cached(self,
                           tag: str,
                           estimated_size: int,
                           method: typing.Callable,
                           memory_guard_device: str | torch.device | None = 'cpu'
                           ):
        """
        Load a potentially large object into the CPU side ``prompt_upscaler`` object cache.

        :param tag: A unique string within the context of the image
            processor implementation constructor.
        :param estimated_size: Estimated size in bytes of the object in RAM.
        :param method: A method which loads and returns the object.
        :param memory_guard_device: call :py:meth:`PromptUpscaler.memory_guard_device` on the
            specified device before the object is loaded (on cache miss)
        :return: The loaded object
        """

        @_memoize.memoize(
            _prompt_upscaler_cache,
            on_hit=_cache_debug_hit,
            on_create=_cache_debug_miss)
        def load_cached(loaded_by_name=self.loaded_by_name, tag=tag):
            if memory_guard_device is not None:
                self.memory_guard_device(memory_guard_device, estimated_size)
            return method(), _memoize.CachedObjectMetadata(size=estimated_size)

        return load_cached()

    @property
    def device(self) -> str:
        """
        Device that will be used for any text processing models.
        """
        return self.__device

    @property
    def local_files_only(self) -> bool:
        """
        Is this prompt upscaler only going to look for resources such as models in cache / on disk?
        """
        return self.__local_files_only

    @property
    def size_estimate(self) -> int:
        """
        Estimated size of the models / objects used by this prompt upscaler.
        :return: size in bytes
        """
        return self.__size_estimate

    @property
    def accepts_batch(self):
        """
        Can this prompt upscaler accept a batch of prompts?

        The implementor must override this property, this is a
        default implementation.

        :return: Default: ``False``
        """
        return False

    @abc.abstractmethod
    def upscale(self, prompt: _prompt.PromptOrPrompts) -> _prompt.PromptOrPrompts:
        """
        Upscale a prompt / prompts and return them modified.

        :param prompt: The incoming prompt or prompts
        :return: Modified prompt / prompts, you may return multiple prompts (an iterable) to indicate expansion
        """
        return prompt
