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
import gc
import typing

import diffusers
import torch

import dgenerate.devicecache as _devicecache
import dgenerate.exceptions as _d_exceptions
import dgenerate.latentsprocessors.constants as _constants
import dgenerate.latentsprocessors.exceptions as _exceptions
import dgenerate.memoize as _memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.plugin as _plugin
import dgenerate.torchutil as _torchutil
import dgenerate.types

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
    HIDE_ARGS = ['model-type', 'local-files-only']

    @classmethod
    def inheritable_help(cls, loaded_by_name):
        help_messages = {
            'device': (
                'The "device" argument can be used to set the device '
                'the processor will run on, for example: cpu, cuda, cuda:1.'
            ),
            'model-offload': (
                'The "model-offload" argument can be used to enable '
                'cpu model offloading for a processor. If this is disabled, '
                'any torch tensors or modules placed on the GPU will remain there until '
                'the processor is done being used, instead of them being moved back to the CPU '
                'after each invocation. Enabling this may help save VRAM when using a latents processor '
                'but will impact rendering speed with multiple inputs / outputs.'
            )
        }
        return help_messages

    def __init__(self,
                 loaded_by_name: str,
                 model_type: _enums.ModelType,
                 device: str = 'cpu',
                 model_offload: bool = False,
                 local_files_only: bool = False,
                 **kwargs):
        """
        Initialize the latents processor.
        
        :param loaded_by_name: The name the processor was loaded by
        :param model_type: Model type enum :py:class:`dgenerate.ModelType`
        :param device: Device to run processing on
        :param model_offload: if ``True``, any torch modules that the processor
            has registered are offloaded to the CPU immediately after processing an
            image
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
        self.__model_offload = model_offload
        self.__modules = []
        self.__modules_device = torch.device('cpu')

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
    def model_offload(self) -> bool:
        """
        Model offload status.

        :return: ``True`` or ``False``
        """
        return self.__model_offload

    @property
    def local_files_only(self) -> bool:
        """
        Is this processor only going to look for files in cache / on disk?
        """
        return self.__local_files_only

    @property
    def modules_device(self) -> torch.device:
        """
        The rendering device that this processors modules currently exist on.

        This will change with calls to :py:meth:`.LatentsProcessor.to` and
        possibly when the processor is used.

        :return: :py:class:`torch.device`, using ``str()`` on this object
            will yield a device string such as "cuda", "cuda:N", or "cpu"
        """
        return self.__modules_device

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

        If the device is a gpu device and any of the memory constraints specified by
        :py:attr:`dgenerate.latentsprocessors.constants.LATENTS_PROCESSOR_GPU_MEMORY_CONSTRAINTS`
        are met on that device, attempt to remove cached objects off a gpu device to free space.

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

        if _memory.is_supported_gpu_device(device):
            if _memory.gpu_memory_constraints(
                    _constants.LATENTS_PROCESSOR_GPU_MEMORY_CONSTRAINTS,
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

    def register_module(self, module):
        """
        Register :py:class:`torch.nn.Module` objects.

        These will be brought on to the cpu during finalization.

        All of these modules can be cast to a specific device with :py:attr:`.LatentsProcessor.to`

        :param module: the module
        """
        self.__modules.append(module)

    def __flush_diffusion_pipeline_after_oom(self):
        _messages.debug_log(
            f'Latents processor "{self.__class__.__name__}" is clearing the GPU side object '
            f'cache for device {self.device} due to VRAM out of memory condition.')

        _devicecache.clear_device_cache(self.device)

    def __to_cpu_ignore_error(self):
        try:
            self.to('cpu')
        except:
            pass

    @staticmethod
    def __flush_mem_ignore_error():
        try:
            _memory.torch_gc()
        except:
            pass

    def __with_memory_safety(self, func, args: dict, oom_attempt=0):
        raise_exc = None

        try:
            try_again = False
            try:
                return func(**args)
            except _d_exceptions.TORCH_CUDA_OOM_EXCEPTIONS as e:
                _d_exceptions.raise_if_not_cuda_oom(e)

                if oom_attempt == 0:
                    self.__flush_diffusion_pipeline_after_oom()
                    try_again = True
                else:
                    _messages.debug_log(
                        f'LatentsProcessor "{self.__class__.__name__}" failed attempt at '
                        f'OOM recovery in {dgenerate.types.fullname(func)}()')

                    self.__to_cpu_ignore_error()
                    self.__flush_mem_ignore_error()
                    raise_exc = _d_exceptions.OutOfMemoryError(e)

            if try_again:
                return self.__with_memory_safety(func, args, oom_attempt=1)
        except MemoryError as e:
            gc.collect()
            raise _d_exceptions.OutOfMemoryError('cpu (system memory)') from e
        except Exception as e:
            if not isinstance(e, _d_exceptions.OutOfMemoryError):
                self.__to_cpu_ignore_error()
                self.__flush_mem_ignore_error()
            raise

        if raise_exc is not None:
            raise raise_exc

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

    def process(self,
                pipeline: diffusers.DiffusionPipeline,
                latents: torch.Tensor) -> torch.Tensor:
        """
        Process a latents tensor.

        :param pipeline: The pipeline object
        :param latents: Input latents tensor with shape [B, C, H, W]
        :return: Processed latents tensor with shape [B, C, H, W]
        """

        self.to(self.device)
        try:
            return self.__with_memory_safety(
                self.impl_process,
                {'pipeline': pipeline,
                 'latents': latents})
        finally:
            if self.__model_offload:
                self.to('cpu')
                self.__flush_mem_ignore_error()

    @abc.abstractmethod
    def impl_process(self,
                     pipeline: diffusers.DiffusionPipeline,
                     latents: torch.Tensor) -> torch.Tensor:
        """
        Process a latents tensor.
        
        This method should be overridden by subclasses to provide
        the actual tensor processing functionality.
        
        :param pipeline: The pipeline object
        :param latents: Input latents tensor with shape [B, C, H, W]
        :return: Processed latents tensor with shape [B, C, H, W]
        """
        raise NotImplementedError()

    def __to(self, device: torch.device | str, attempt=0):
        device = torch.device(device)

        if device.type != 'cpu':
            _latents_processor_cache.size -= self.__size_estimate
        else:
            _latents_processor_cache.size += self.__size_estimate

        self.__modules_device = device

        try_again = False

        for m in self.__modules:
            if not hasattr(m, '_DGENERATE_LATENTS_PROCESSOR_DEVICE') or \
                    not _torchutil.devices_equal(m._DGENERATE_LATENTS_PROCESSOR_DEVICE, device):

                self.memory_guard_device(device, self.size_estimate)

                m._DGENERATE_LATENTS_PROCESSOR_DEVICE = device
                _messages.debug_log(
                    f'Moving LatentsProcessor registered module: {dgenerate.types.fullname(m)}.to("{device}")')

                try:
                    m.to(device)
                except _d_exceptions.TORCH_CUDA_OOM_EXCEPTIONS as e:
                    _d_exceptions.raise_if_not_cuda_oom(e)

                    if attempt == 0:
                        # hail marry
                        self.__flush_diffusion_pipeline_after_oom()
                        try_again = True
                        break
                    else:
                        _messages.debug_log(
                            f'LatentsProcessor "{self.__class__.__name__}" failed attempt '
                            f'at OOM recovery in to({device})')

                        m._DGENERATE_LATENTS_PROCESSOR_DEVICE = torch.device('cpu')

                        self.__to_cpu_ignore_error()
                        self.__flush_mem_ignore_error()
                        raise _d_exceptions.OutOfMemoryError(e) from e
                except MemoryError as e:
                    # out of cpu side memory
                    self.__flush_mem_ignore_error()
                    raise _d_exceptions.OutOfMemoryError('cpu (system memory)') from e

        if try_again:
            self.__to(device, attempt=1)

        return self

    def to(self, device: torch.device | str) -> "LatentsProcessor":
        """
        Move all :py:class:`torch.nn.Module` modules registered
        to this latents processor to a specific device.

        :raise dgenerate.OutOfMemoryError: if there is not enough memory on the specified device

        :param device: The device string, or torch device object
        :return: the latents processor itself
        """
        return self.__to(device)
