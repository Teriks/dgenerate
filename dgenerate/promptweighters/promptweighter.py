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

import diffusers
import torch

import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.promptweighters.exceptions as _exceptions
import dgenerate.plugin as _plugin
import dgenerate.memoize as _memoize
import dgenerate.memory as _memory
import dgenerate.promptweighters.constants as _constants
import dgenerate.messages as _messages
import dgenerate.devicecache as _devicecache
import dgenerate.pipelinewrapper as _pipelinewrapper
import dgenerate.torchutil as _torchutil

_prompt_weighter_cache = _memoize.create_object_cache(
    'prompt_weighter',
    cache_type=_memory.SizedConstrainedObjectCache
)


def _cache_debug_hit(key, hit):
    _memoize.simple_cache_hit_debug("Prompt Weighter Model", key, hit)


def _cache_debug_miss(key, new):
    _memoize.simple_cache_miss_debug("Prompt Weighter Model", key, new)


class PromptWeighter(_plugin.Plugin, abc.ABC):
    """
    Abstract base class for prompt weighter implementations.
    """

    # you cannot specify these via a URI
    HIDE_ARGS = ['model-type', 'dtype', 'local-files-only', 'device']

    @classmethod
    def inheritable_help(cls, loaded_by_name):
        return None

    def __init__(self,
                 loaded_by_name: str,
                 model_type: _enums.ModelType,
                 dtype: _enums.DataType,
                 device: str | None = None,
                 local_files_only: bool = False,
                 **kwargs):
        """
        :param loaded_by_name: The name the prompt weighter was loaded by
        :param model_type: Model type enum :py:class:`dgenerate.ModelType`
        :param dtype: Data type enum :py:class:`dgenerate.DataType`
        :param device: The device the prompt weighter should operate on
        :param local_files_only: if ``True``, the plugin should never try to download
            models from the internet automatically, and instead only look
            for them in cache / on disk.
        :param kwargs: child class forwarded arguments
        """
        super().__init__(loaded_by_name=loaded_by_name,
                         argument_error_type=_exceptions.PromptWeighterArgumentError,
                         **kwargs)

        if device is not None:
            if not _torchutil.is_valid_device_string(device):
                raise _exceptions.PromptWeighterArgumentError(
                    f'Invalid device argument, {_torchutil.invalid_device_message(device, cap=False)}')

        self.__model_type = model_type
        self.__dtype = dtype
        self.__device = device if device else 'cpu'
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
    def device(self) -> str:
        """
        The device the prompt weighter operates on.
        """
        return self.__device

    @property
    def local_files_only(self) -> bool:
        """
        Is this prompt weighter only going to look for resources such as model files in cache / on disk?
        """
        return self.__local_files_only

    @property
    def size_estimate(self) -> int:
        """
        Estimated size of the models / objects used by this prompt weighter.
        :return: size in bytes
        """
        return self.__size_estimate

    def get_extra_supported_args(self) -> list[str]:
        """
        Overridable method.

        Return a list of extra supported prompt arguments that are not
        typically supported by the given model that embed generation
        is occurring for.

        This can be used to make use of the ``--second-prompts``, ``--third-prompts``,
        ``--second-model-second-prompts``, or ``--second-model-third-prompts`` arguments
        for additional textual inputs to the prompt weighter plugin.

        This works even if the model you are generating embeds for would
        normally reject these inputs.

        Particularly for arguments that :py:class:`dgenerate.pipelinewrapper.DiffusionPipelineWrapper`
        will automatically reject based on the underlying pipeline argument signature.

        You should return a list of pipeline argument names, the values can be any of the following:

        * ``prompt_2``
        * ``negative_prompt_2``
        * ``prompt_3``
        * ``negative_prompt_3``
        * ``clip_skip``

        If you return a value outside the set of values listed here,
        a :py:exc:`RuntimeError` will be raised by :py:class:`dgenerate.pipelinewrapper.DiffusionPipelineWrapper`
        upon attempting to consume the values from this method when calling a pipeline.

        :return: List of pipeline argument names.
        """
        return []

    def memory_guard_device(self, device: str | torch.device, memory_required: int):
        """
        Check a specific device against an amount of memory in bytes.

        If the device is a gpu device and any of the memory constraints specified by
        :py:attr:`dgenerate.promptweighters.constants.PROMPT_WEIGHTER_GPU_MEMORY_CONSTRAINTS`
        are met on that device, attempt to remove cached objects off a gpu device to free space.

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

        if _memory.is_supported_gpu_device(device):
            if _memory.gpu_memory_constraints(
                    _constants.PROMPT_WEIGHTER_GPU_MEMORY_CONSTRAINTS,
                    extra_vars={'memory_required': memory_required},
                    device=device):
                _messages.debug_log(
                    f'Prompt Weighter "{self.__class__.__name__}" is clearing the GPU side object '
                    f'cache due to GPU side memory constraint evaluating to to True.')

                _devicecache.clear_device_cache(device)
                cleared = True

        elif device.type == 'cpu':
            if (_memory.memory_constraints(
                    _constants.PROMPT_WEIGHTER_CACHE_GC_CONSTRAINTS,
                    extra_vars={'memory_required': memory_required})):
                _messages.debug_log(
                    f'Prompt weighter "{self.__class__.__name__}" is clearing the CPU side object '
                    f'cache due to CPU side memory constraint evaluating to to True.')

                _memoize.clear_object_caches()
                cleared = True

            cleared = cleared or _prompt_weighter_cache.enforce_cpu_mem_constraints(
                _constants.PROMPT_WEIGHTER_CACHE_MEMORY_CONSTRAINTS,
                size_var='memory_required',
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
                           memory_guard_device: str | torch.device | None = 'cpu'
                           ):
        """
        Load a potentially large object into the CPU side ``prompt_weighter`` object cache.

        :param tag: A unique string within the context of the image
            processor implementation constructor.
        :param estimated_size: Estimated size in bytes of the object in RAM.
        :param method: A method which loads and returns the object.
        :param memory_guard_device: call :py:meth:`PromptWeighter.memory_guard_device` on the
            specified device before the object is loaded (on cache miss)
        :return: The loaded object
        """

        @_memoize.memoize(
            _prompt_weighter_cache,
            on_hit=_cache_debug_hit,
            on_create=_cache_debug_miss)
        def load_cached(loaded_by_name=self.loaded_by_name, tag=tag):
            if memory_guard_device is not None:
                self.memory_guard_device(memory_guard_device, estimated_size)
            return method(), _memoize.CachedObjectMetadata(size=estimated_size)

        return load_cached()

    @staticmethod
    def move_text_encoders(pipeline, device: str):
        """
        Utility for moving all of a pipelines text encoders to a device.

        :param pipeline: The diffusion pipeline.
        :param device: The desired device.
        """
        for name, module in _pipelinewrapper.get_pipeline_modules(pipeline).items():
            # module will never be None, _pipelinewrapper.get_torch_pipeline_modules
            # filters that out for you
            if name.startswith('text_encoder'):
                module.to(device)

    @abc.abstractmethod
    def translate_to_embeds(self,
                            pipeline: diffusers.DiffusionPipeline,
                            args: dict[str, typing.Any]):
        """
        Override me to implement.

        Translate the pipeline prompt arguments to ``prompt_embeds`` and ``pooled_prompt_embeds`` as needed.

        :param pipeline: The pipeline object
        :param args: Call arguments to the pipeline
        :return: ``args``, supplemented with prompt embedding arguments
        """
        pass

    def cleanup(self):
        """
        Perform any cleanup required after translating the pipeline arguments to embeds
        """
        pass
