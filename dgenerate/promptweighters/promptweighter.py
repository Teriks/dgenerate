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

import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.promptweighters.exceptions as _exceptions
import dgenerate.plugin as _plugin
import dgenerate.memoize as _memoize
import dgenerate.memory as _memory
import dgenerate.promptweighters.constants as _constants
import dgenerate.messages as _messages
import dgenerate.devicecache as _devicecache

_prompt_weighter_cache = _memoize.create_object_cache(
    'prompt_weighter',
    cache_type=_memory.SizedConstrainedObjectCache
)


def _cache_debug_hit(key, hit):
    _memoize.simple_cache_hit_debug("Prompt Weighter Model", key, hit)


def _cache_debug_miss(key, new):
    _memoize.simple_cache_miss_debug("Prompt Weighter Model", key, new)


class PromptWeighter(_plugin.Plugin):
    """
    Abstract base class for prompt weighter implementations.
    """

    # you cannot specify these via a URI
    HIDE_ARGS = ['model-type', 'dtype', 'local-files-only']

    def __init__(self,
                 loaded_by_name: str,
                 model_type: _enums.ModelType,
                 dtype: _enums.DataType,
                 local_files_only: bool = False,
                 **kwargs):
        """
        :param loaded_by_name: The name the prompt weighter was loaded by
        :param model_type: Model type enum :py:class:`dgenerate.ModelType`
        :param dtype: Data type enum :py:class:`dgenerate.DataType`
        :param local_files_only: if ``True``, the plugin should never try to download
            models from the internet automatically, and instead only look
            for them in cache / on disk.
        :param kwargs: child class forwarded arguments
        """
        super().__init__(loaded_by_name=loaded_by_name,
                         argument_error_type=_exceptions.PromptWeighterArgumentError,
                         **kwargs)

        self.__model_type = model_type
        self.__dtype = dtype
        self.__local_files_only = local_files_only
        self.__size_estimate = 0

    @property
    def dtype(self) -> _enums.DataType:
        """
        Embeddings data type.
        """
        return self.__dtype

    @property
    def model_type(self) -> _enums.ModelType:
        """
        Model type that will use the embeddings.
        """
        return self.__model_type

    @property
    def local_files_only(self) -> bool:
        """
        Is this prompt weighter only going to look for Hugging Face hub model files in cache / on disk?
        """
        return self.__local_files_only

    @property
    def size_estimate(self) -> int:
        """
        Estimated size of the models / objects used by this prompt weighter.
        :return: size in bytes
        """
        return self.__size_estimate

    def memory_fence_device(self, device: str | torch.device, memory_required: int):
        """
        Check a specific device against an amount of memory in bytes.

        If the device is a cuda device and any of the memory constraints specified by
        :py:attr:`dgenerate.promptweighters.constants.PROMPT_WEIGHTER_CUDA_MEMORY_CONSTRAINTS`
        are met on that device, attempt to remove cached objects off the cuda device to free space.

        If the device is a cpu and any of the memory constraints specified by
        :py:attr:`dgenerate.promptweighters.constants.PROMPT_WEIGHTER_CACHE_GC_CONSTRAINTS`
        are met, attempt to remove cached prompt weighter objects off the device to free space.
        Then, enforce :py:attr:`dgenerate.promptweighters.constants.PROMPT_WEIGHTER_CACHE_MEMORY_CONSTRAINTS`.

        :param device: the device
        :param memory_required: the amount of memory required on the device in bytes
        :return: ``True`` if an attempt was made to free memory, ``False`` otherwise.
        """

        device = torch.device(device)
        cleared = False

        if device.type == 'cuda':
            if _memory.cuda_memory_constraints(
                    _constants.PROMPT_WEIGHTER_CUDA_MEMORY_CONSTRAINTS,
                    extra_vars={'memory_required': memory_required},
                    device=device):
                _devicecache.clear_device_cache(device)
                cleared = True

        elif device.type == 'cpu':
            if (_memory.memory_constraints(
                    _constants.PROMPT_WEIGHTER_CACHE_GC_CONSTRAINTS,
                    extra_vars={'weighter_size': memory_required})):

                _messages.debug_log(
                    f'Prompt weighter "{self.__class__.__name__}" is clearing the CPU side object '
                    f'cache due to CPU side memory constraint evaluating to to True.')

                _memoize.clear_object_caches()
                cleared = True

            cleared = cleared or _prompt_weighter_cache.enforce_cpu_mem_constraints(
                _constants.PROMPT_WEIGHTER_CACHE_MEMORY_CONSTRAINTS,
                size_var='weighter_size',
                new_object_size=memory_required
            )
        return cleared

    def set_size_estimate(self, size_bytes: int):
        """
        Set the estimated size of this plugin in bytes for memory management
        heuristics, this is intended to be used by implementors of the
        :py:class:`PromptWeighter` plugin class.

        For the best memory optimization, this value should be set very
        shortly before any associated model even enters CPU side ram,
        IE: before it is loaded at all.

        :raise ValueError: if ``size_bytes`` is less than zero.

        :param size_bytes: the size in bytes
        """
        if size_bytes < 0:
            raise ValueError(
                'prompt weighter size estimate cannot be less than zero.')

        self.__size_estimate = int(size_bytes)

    def load_object_cached(self,
                           tag: str,
                           estimated_size: int,
                           method: typing.Callable,
                           memory_fence_device: str | torch.device | None = 'cpu'
                           ):
        """
        Load a potentially large object into the CPU side ``prompt_weighter`` object cache.

        :param tag: A unique string within the context of the image
            processor implementation constructor.
        :param estimated_size: Estimated size in bytes of the object in RAM.
        :param method: A method which loads and returns the object.
        :param memory_fence_device: call :py:meth:`PromptWeighter.memory_fence_device` on the
            specified device before the object is loaded (on cache miss)
        :return: The loaded object
        """

        @_memoize.memoize(
            _prompt_weighter_cache,
            on_hit=_cache_debug_hit,
            on_create=_cache_debug_miss)
        def load_cached(loaded_by_name=self.loaded_by_name, tag=tag):
            if memory_fence_device is not None:
                self.memory_fence_device(memory_fence_device, estimated_size)
            return method(), _memoize.CachedObjectMetadata(size=estimated_size)

        return load_cached()

    def translate_to_embeds(self,
                            pipeline,
                            device: str,
                            args: dict[str, any]):
        """
        Translate the pipeline prompt arguments to ``prompt_embeds`` and ``pooled_prompt_embeds`` as needed.

        :param pipeline: The pipeline object
        :param device: The device the pipeline modules are on
        :param args: Call arguments to the pipeline
        :return: ``args``, supplemented with prompt embedding arguments
        """
        pass

    def cleanup(self):
        """
        Preform any cleanup required after translating the pipeline arguments to embeds
        """
        pass
