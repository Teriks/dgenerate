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

import dgenerate.latentsprocessors.exceptions as _exceptions
import dgenerate.plugin as _plugin
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.memoize as _memoize
import dgenerate.memory as _memory
import dgenerate.latentsprocessors.constants as _constants
import dgenerate.messages as _messages
import dgenerate.devicecache as _devicecache

_latents_processor_cache = _memoize.create_object_cache(
    'latents_processor',
    cache_type=_memory.SizedConstrainedObjectCache
)


def _cache_debug_hit(key, hit):
    _memoize.simple_cache_hit_debug("Latents Processor Model", key, hit)


def _cache_debug_miss(key, new):
    _memoize.simple_cache_miss_debug("Latents Processor Model", key, new)


__doc__ = """
Base class definitions for dgenerate latents processors.
"""


class LatentsProcessor(_plugin.Plugin, abc.ABC):
    """
    Abstract base class for latents processor implementations.
    
    Latents processors perform tensor manipulation on latent space representations
    in diffusion pipelines. They can handle operations like noise injection,
    scaling, blurring, and other latent space transformations.
    """

    # you cannot specify these via a URI
    HIDE_ARGS = ['model-type', 'device', 'local-files-only']

    def __init__(self,
                 loaded_by_name: str,
                 model_type: _enums.ModelType,
                 device: str = 'cpu',
                 local_files_only: bool = False,
                 **kwargs):
        """
        Initialize the latents processor.
        
        :param loaded_by_name: The name the processor was loaded by
        :param model_type: Model type enum :py:class:`dgenerate.ModelType`
        :param device: Device to run processing on
        :param local_files_only: if ``True``, the plugin should never try to download
            models from the internet automatically, and instead only look
            for them in cache / on disk.
        :param kwargs: child class forwarded arguments
        """
        super().__init__(loaded_by_name=loaded_by_name,
                         argument_error_type=_exceptions.LatentsProcessorArgumentError,
                         **kwargs)

        self.__model_type = model_type
        self.__device = device
        self.__local_files_only = local_files_only
        self.__size_estimate = 0

    @property
    def model_type(self) -> _enums.ModelType:
        """
        The model type this processor is being used with.
        
        :return: Model type enum
        """
        return self.__model_type

    @property
    def device(self) -> str:
        """
        Device the processor runs on.
        
        :return: Device string
        """
        return self.__device

    @property
    def local_files_only(self) -> bool:
        """
        Is this processor only going to look for files in cache / on disk?
        """
        return self.__local_files_only

    @property
    def size_estimate(self) -> int:
        """
        Estimated size of the models / objects used by this latents processor.
        :return: size in bytes
        """
        return self.__size_estimate

    def memory_guard_device(self, device: str | torch.device, memory_required: int):
        """
        Check a specific device against an amount of memory in bytes.

        If the device is a cuda device and any of the memory constraints specified by
        :py:attr:`dgenerate.latentsprocessors.constants.LATENTS_PROCESSOR_CUDA_MEMORY_CONSTRAINTS`
        are met on that device, attempt to remove cached objects off the cuda device to free space.

        If the device is a cpu and any of the memory constraints specified by
        :py:attr:`dgenerate.latentsprocessors.constants.LATENTS_PROCESSOR_CACHE_GC_CONSTRAINTS`
        are met, attempt to remove cached latents processor objects off the device to free space.
        Then, enforce :py:attr:`dgenerate.latentsprocessors.constants.LATENTS_PROCESSOR_CACHE_MEMORY_CONSTRAINTS`.

        :param device: the device
        :param memory_required: the amount of memory required on the device in bytes
        :return: ``True`` if an attempt was made to free memory, ``False`` otherwise.
        """

        device = torch.device(device)
        cleared = False

        if device.type == 'cuda':
            if _memory.cuda_memory_constraints(
                    _constants.LATENTS_PROCESSOR_CUDA_MEMORY_CONSTRAINTS,
                    extra_vars={'memory_required': memory_required},
                    device=device):
                _messages.debug_log(
                    f'Latents Processor "{self.__class__.__name__}" is clearing the GPU side object '
                    f'cache due to GPU side memory constraint evaluating to to True.')

                _devicecache.clear_device_cache(device)
                cleared = True

        elif device.type == 'cpu':
            if (_memory.memory_constraints(
                    _constants.LATENTS_PROCESSOR_CACHE_GC_CONSTRAINTS,
                    extra_vars={'memory_required': memory_required})):
                _messages.debug_log(
                    f'Latents processor "{self.__class__.__name__}" is clearing the CPU side object '
                    f'cache due to CPU side memory constraint evaluating to to True.')

                _memoize.clear_object_caches()
                cleared = True

            cleared = cleared or _latents_processor_cache.enforce_cpu_mem_constraints(
                _constants.LATENTS_PROCESSOR_CACHE_MEMORY_CONSTRAINTS,
                size_var='memory_required',
                new_object_size=memory_required
            )
        return cleared

    def set_size_estimate(self, size_bytes: int):
        """
        Set the estimated size of this plugin in bytes for memory management
        heuristics, this is intended to be used by implementors of the
        :py:class:`LatentsProcessor` plugin class.

        For the best memory optimization, this value should be set very
        shortly before any associated model even enters CPU side ram,
        IE: before it is loaded at all.

        :raise ValueError: if ``size_bytes`` is less than zero.

        :param size_bytes: the size in bytes
        """
        if size_bytes < 0:
            raise ValueError(
                'latents processor size estimate cannot be less than zero.')

        self.__size_estimate = int(size_bytes)

    def load_object_cached(self,
                           tag: str,
                           estimated_size: int,
                           method: typing.Callable,
                           memory_guard_device: str | torch.device | None = 'cpu'
                           ):
        """
        Load a potentially large object into the CPU side ``latents_processor`` object cache.

        :param tag: A unique string within the context of the latents
            processor implementation constructor.
        :param estimated_size: Estimated size in bytes of the object in RAM.
        :param method: A method which loads and returns the object.
        :param memory_guard_device: call :py:meth:`LatentsProcessor.memory_guard_device` on the
            specified device before the object is loaded (on cache miss)
        :return: The loaded object
        """

        @_memoize.memoize(
            _latents_processor_cache,
            on_hit=_cache_debug_hit,
            on_create=_cache_debug_miss)
        def load_cached(loaded_by_name=self.loaded_by_name, tag=tag):
            if memory_guard_device is not None:
                self.memory_guard_device(memory_guard_device, estimated_size)
            return method(), _memoize.CachedObjectMetadata(size=estimated_size)

        return load_cached()

    @abc.abstractmethod
    def process(self,
                pipeline,
                device: str,
                latents: torch.Tensor,
                *args, **kwargs) -> torch.Tensor:
        """
        Process a latents tensor.
        
        This method should be overridden by subclasses to provide
        the actual tensor processing functionality.
        
        :param pipeline: The pipeline object
        :param device: The device the pipeline modules are on
        :param latents: Input latents tensor with shape [B, C, W, H]
        :param args: Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: Processed latents tensor with shape [B, C, W, H]
        """
        raise NotImplementedError()

