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
import itertools
import os
import typing

import PIL.Image
import torch

import dgenerate.devicecache as _devicecache
import dgenerate.exceptions as _d_exceptions
import dgenerate.filelock as _filelock
import dgenerate.image as _image
import dgenerate.imageprocessors.constants as _constants
import dgenerate.imageprocessors.exceptions as _exceptions
import dgenerate.memoize as _memoize
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.plugin as _plugin
import dgenerate.types
import dgenerate.torchutil as _torchutil

_image_processor_cache = _memoize.create_object_cache(
    'image_processor',
    cache_type=_memory.SizedConstrainedObjectCache
)


def _cache_debug_hit(key, hit):
    _memoize.simple_cache_hit_debug("Image Processor Model", key, hit)


def _cache_debug_miss(key, new):
    _memoize.simple_cache_miss_debug("Image Processor Model", key, new)


_in_filetypes = None
_out_filetypes = None

class ImageProcessor(_plugin.Plugin, abc.ABC):
    """
    Abstract base class for image processor implementations.
    """

    @staticmethod
    def image_out_filetypes():
        """
        Utility for derived classes to get a list of supported image output file types for use with ``FILE_ARGS``.
        :return: List of supported image output file types, for example ``['*.png', '*.jpg']``.
        """
        import dgenerate.mediaoutput as _mediaoutput
        global _out_filetypes
        if _out_filetypes is None:
            _out_filetypes = ['*.' + i for i in _mediaoutput.get_supported_static_image_formats()]
            return list(_out_filetypes)
        else:
            return list(_out_filetypes)

    @staticmethod
    def image_in_filetypes():
        """
        Utility for derived classes to get a list of supported image input file types for use with ``FILE_ARGS``.
        :return: List of supported image input file types, for example ``['*.png', '*.jpg']``.
        """
        import dgenerate.mediainput as _mediainput
        global _in_filetypes
        if _in_filetypes is None:
            _in_filetypes = ['*.' + i for i in _mediainput.get_supported_image_formats()]
            return list(_in_filetypes)
        else:
            return list(_in_filetypes)

    # you cannot specify these via a URI
    HIDE_ARGS = ['local-files-only']

    FILE_ARGS = {'output-file': {'mode': 'out', 'filetypes': [('Images', image_out_filetypes())]}}

    @classmethod
    def inheritable_help(cls, loaded_by_name):
        help_messages = {
            'device': (
                'The "device" argument can be used to set the device '
                'the processor will run on, for example: cpu, cuda, cuda:1. '
                'If you are using this image processor as a preprocess or '
                'postprocess step for dgenerate, or with the image-process '
                'subcommand, or \\image_process directive, this argument will '
                'default to the value of --device.'
            ),
            'output-file': (
                'The "output-file" argument can be used to set the output '
                'path for a processor debug image, this will save the '
                'processed image to a path of your choosing.'
            ),
            'output-overwrite': (
                'The "output-overwrite" argument can be used to enable '
                'overwrite for a processor debug image. If this is not enabled, '
                'new images written by the processor while it is being used '
                'will be written with a numbered suffix instead of being overwritten.'
            ),
            'model-offload': (
                'The "model-offload" argument can be used to enable '
                'cpu model offloading for a processor. If this is disabled, '
                'any torch tensors or modules placed on the GPU will remain there until '
                'the processor is done being used, instead of them being moved back to the CPU '
                'after each image. Enabling this may help save VRAM when using an image processor '
                'as a preprocessor or postprocessor for diffusion with dgenerate but will impact '
                'rendering speed when generating many images.'
            )
        }
        return help_messages

    def __init__(self,
                 loaded_by_name: str,
                 device: str | None = None,
                 output_file: dgenerate.types.OptionalPath = None,
                 output_overwrite: bool = False,
                 model_offload: bool = False,
                 local_files_only: bool = False,
                 **kwargs):
        """
        :param loaded_by_name: The name the processor was loaded by
        :param device: the device the processor will run on, for example: cpu, cuda, cuda:1.
            Specifying ``None`` causes the device to default to cpu.
        :param output_file: output a debug image to this path
        :param output_overwrite: can the debug image output path be overwritten?
        :param model_offload: if ``True``, any torch modules that the processor
            has registered are offloaded to the CPU immediately after processing an
            image
        :param local_files_only: if ``True``, the plugin should never try to download
            models from the internet automatically, and instead only look
            for them in cache / on disk.
        :param kwargs: child class forwarded arguments
        """

        super().__init__(loaded_by_name=loaded_by_name,
                         argument_error_type=_exceptions.ImageProcessorArgumentError,
                         **kwargs)

        if device is not None:
            if not _torchutil.is_valid_device_string(device):
                raise _exceptions.ImageProcessorArgumentError(
                    f'Invalid device argument, {_torchutil.invalid_device_message(device, cap=False)}')

        self.__output_file = output_file
        self.__output_overwrite = output_overwrite
        self.__device = device if device else 'cpu'
        self.__modules = []
        self.__modules_device = torch.device('cpu')
        self.__model_offload = model_offload
        self.__size_estimate = 0
        self.__local_files_only = local_files_only

    # noinspection PyMethodMayBeStatic
    @property
    def image_modes(self) -> list[str]:
        """
        Returns a list of PIL image modes that this processor can handle.

        This may be overridden by implementers

        :return: ``['RGB']``
        """
        return ['RGB']

    def set_size_estimate(self, size_bytes: int):
        """
        Set the estimated size of this plugin in bytes for memory
        management heuristics, this is intended to be used by implementors
        of the :py:class:`ImageProcessor` plugin class.

        For the best memory optimization, this value should be set very
        shortly before any associated model even enters CPU side ram, IE:
        before it is loaded at all.

        :raise ValueError: if ``size_bytes`` is less than zero.

        :param size_bytes: the size in bytes
        """
        if size_bytes < 0:
            raise ValueError(
                'image processor size estimate cannot be less than zero.')

        self.__size_estimate = int(size_bytes)

    def memory_guard_device(self, device: str | torch.device, memory_required: int):
        """
        Check a specific device against an amount of memory in bytes.

        If the device is a gpu device and any of the memory constraints specified by
        :py:attr:`dgenerate.imageprocessors.constants.IMAGE_PROCESSOR_GPU_MEMORY_CONSTRAINTS`
        are met on that device, attempt to remove cached objects off a gpu device to free space.

        If the device is a cpu and any of the memory constraints specified by
        :py:attr:`dgenerate.imageprocessors.constants.IMAGE_PROCESSOR_CACHE_GC_CONSTRAINTS`
        are met, attempt to remove cached image processor objects off the device to free space.
        Then, enforce :py:attr:`dgenerate.imageprocessors.constants.IMAGE_PROCESSOR_CACHE_MEMORY_CONSTRAINTS`.

        :param device: the device
        :param memory_required: the amount of memory required on the device in bytes
        :return: ``True`` if an attempt was made to free memory, ``False`` otherwise.
        """

        device = torch.device(device)
        cleared = False

        if _memory.is_supported_gpu_device(device):
            if _memory.gpu_memory_constraints(
                    _constants.IMAGE_PROCESSOR_GPU_MEMORY_CONSTRAINTS,
                    extra_vars={'memory_required': memory_required},
                    device=device):
                _messages.debug_log(
                    f'Image Processor "{self.__class__.__name__}" is clearing the GPU side object '
                    f'cache due to GPU side memory constraint evaluating to to True.')

                _devicecache.clear_device_cache(device)
                cleared = True

        elif device.type == 'cpu':

            if (_memory.memory_constraints(
                    _constants.IMAGE_PROCESSOR_CACHE_GC_CONSTRAINTS,
                    extra_vars={'memory_required': memory_required})):
                _messages.debug_log(
                    f'Image Processor "{self.__class__.__name__}" is clearing the CPU side object '
                    f'cache due to CPU side memory constraint evaluating to to True.')

                _memoize.clear_object_caches()
                cleared = True

            cleared = cleared or _image_processor_cache.enforce_cpu_mem_constraints(
                _constants.IMAGE_PROCESSOR_CACHE_MEMORY_CONSTRAINTS,
                size_var='memory_required',
                new_object_size=memory_required
            )
        return cleared

    def load_object_cached(self,
                           tag: str,
                           estimated_size: int,
                           method: typing.Callable,
                           memory_guard_device: str | torch.device | None = 'cpu'
                           ):
        """
        Load a potentially large object into the CPU side ``image_processor`` object cache.

        :param tag: A unique string within the context of the image
            processor implementation constructor.
        :param estimated_size: Estimated size in bytes of the object in RAM.
        :param method: A method which loads and returns the object.
        :param memory_guard_device: call :py:meth:`ImageProcessor.memory_guard_device` on the
            specified device before the object is loaded (on cache miss)
        :return: The loaded object
        """

        @_memoize.memoize(
            _image_processor_cache,
            on_hit=_cache_debug_hit,
            on_create=_cache_debug_miss)
        def load_cached(loaded_by_name=self.loaded_by_name, tag=tag):
            if memory_guard_device is not None:
                self.memory_guard_device(memory_guard_device, estimated_size)
            return method(), _memoize.CachedObjectMetadata(size=estimated_size)

        return load_cached()

    @property
    def size_estimate(self) -> int:
        """
        Estimated size of the models / objects used by this image processor.
        :return: size in bytes
        """
        return self.__size_estimate

    @property
    def device(self) -> str:
        """
        The rendering device requested for this processor.

        Torch modules associated with the processor will not be
        on this device until the processor is used.

        :return: device string, for example "cuda", "cuda:N", or "cpu"
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
        Is this image processor only going to look for resources such as models in cache / on disk?
        """
        return self.__local_files_only

    @property
    def modules_device(self) -> torch.device:
        """
        The rendering device that this processors modules currently exist on.

        This will change with calls to :py:meth:`.ImageProcessor.to` and
        possibly when the processor is used.

        :return: :py:class:`torch.device`, using ``str()`` on this object
            will yield a device string such as "cuda", "cuda:N", or "cpu"
        """
        return self.__modules_device

    def __gen_filename(self):
        return _filelock.touch_avoid_duplicate(os.path.dirname(self.__output_file),
                                               _filelock.suffix_path_maker(self.__output_file, '_'))

    def __save_debug_image(self, image, debug_header):
        if self.__output_file is not None:
            if not self.__output_overwrite:
                filename = self.__gen_filename()
            else:
                filename = self.__output_file
            image.save(filename)
            _messages.debug_log(f'{debug_header}: "{filename}"')

    def __to_cpu_ignore_error(self):
        try:
            self.to('cpu')
        except:
            pass

    @staticmethod
    def __flush_mem_ignore_error():
        try:
            _memory.torch_gc()
            gc.collect()
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
                        f'ImageProcessor "{self.__class__.__name__}" failed attempt at '
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

    def pre_resize(self,
                   image: PIL.Image.Image,
                   resize_resolution: dgenerate.types.OptionalSize = None) -> PIL.Image.Image:
        """
        Invoke a processors :py:meth:`.ImageProcessor.impl_pre_resize` method.

        Implements important behaviors depending on if the image was modified.

        This is the only appropriate way to invoke a processor manually.

        The original image will be closed if the implementation returns a new image
        instead of modifying it in place, you should not count on the original image
        being open and usable once this function completes though it is safe to
        use the input image in a ``with`` context, if you need to retain a
        copy, pass a copy.

        :raise dgenerate.OutOfMemoryError: if the execution device runs out of memory
        :raise dgenerate.ImageProcessorImageModeError: if a passed image has an invalid format

        :param self: :py:class:`.ImageProcessor` implementation instance
        :param image: the image to pass
        :param resize_resolution: the size that the image is going to be resized
            to after this step, or None if it is not being resized.

        :return: processed image, may be the same image or a copy.
        """
        if image.mode not in self.image_modes:
            raise _exceptions.ImageProcessorImageModeError(
                f'Invalid image mode: {image.mode}')

        self.to(self.device)

        img_copy = image.copy()

        processed = self.__with_memory_safety(
            self.impl_pre_resize,
            {'image': image,
             'resize_resolution': resize_resolution})

        if processed is not image:
            image.close()

            self.__save_debug_image(
                processed,
                'Wrote Processor Debug Image (because copied)')

            processed.filename = _image.get_filename(image)
            return processed

        # Not copied but may be modified

        identical = all(a == b for a, b in
                        itertools.zip_longest(processed.getdata(),
                                              img_copy.getdata(),
                                              fillvalue=None))

        if not identical:
            # Write the debug output if it was modified in place
            self.__save_debug_image(
                processed,
                'Wrote Processor Debug Image (because modified)')

        return processed

    def post_resize(self,
                    image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Invoke a processors :py:meth:`.ImageProcessor.impl_post_resize` method.

        Implements important behaviors depending on if the image was modified.

        This is the only appropriate way to invoke a processor manually.

        The original image will be closed if the implementation returns a new image
        instead of modifying it in place, you should not count on the original image
        being open and usable once this function completes though it is safe to
        use the input image in a ``with`` context, if you need to retain a
        copy, pass a copy.

        :raise dgenerate.OutOfMemoryError: if the execution device runs out of memory
        :raise dgenerate.ImageProcessorImageModeError: if a passed image has an invalid format

        :param self: :py:class:`.ImageProcessor` implementation instance
        :param image: the image to pass

        :return: processed image, may be the same image or a copy.
        """
        if image.mode not in self.image_modes:
            raise _exceptions.ImageProcessorImageModeError(
                f'Invalid image mode: {image.mode}')

        img_copy = image.copy()

        processed = self.__with_memory_safety(
            self.impl_post_resize, {'image': image})

        if self.__model_offload:
            self.to('cpu')
            self.__flush_mem_ignore_error()

        if processed is not image:
            image.close()

            self.__save_debug_image(
                processed,
                'Wrote Processor Debug Image (because copied)')

            processed.filename = _image.get_filename(image)
            return processed

        # Not copied but may be modified

        identical = all(a == b for a, b in
                        itertools.zip_longest(processed.getdata(),
                                              img_copy.getdata(),
                                              fillvalue=None))

        if not identical:
            # Write the debug output if it was modified in place
            self.__save_debug_image(
                processed,
                'Wrote Processor Debug Image (because modified)')

        return processed

    def _process_pre_resize(self, image: PIL.Image.Image, resize_resolution: dgenerate.types.OptionalSize):
        filename = _image.get_filename(image)

        _messages.debug_log('Starting Image Process - '
                            f'{self}.pre_resize('
                            f'image="{filename}", resize_resolution={resize_resolution})')

        processed = self.pre_resize(image, resize_resolution)

        _messages.debug_log(f'Finished Image Process - {self}.pre_resize')
        return processed

    def _process_post_resize(self, image: PIL.Image.Image):
        filename = _image.get_filename(image)

        _messages.debug_log('Starting Image Process - '
                            f'{self}.post_resize('
                            f'image="{filename}")')

        processed = self.post_resize(image)

        _messages.debug_log(f'Finished Image Process - {self}.post_resize')
        return processed

    def get_alignment(self) -> int | None:
        """
        Overridable method.

        Get required input image alignment, which will be forcefully applied.

        If this function returns ``None``, specific alignment is not required and will never be forced.

        :return: integer or ``None``
        """
        return None

    def process(self,
                image: PIL.Image.Image,
                resize_resolution: dgenerate.types.OptionalSize = None,
                aspect_correct: bool = True,
                align: int | None = None):
        """
        Perform image processing on an image, including the requested resizing step.

        Invokes the image processor pre and post resizing with
        appropriate arguments and correct resource management.

        The original image will be closed if the implementation returns a new image
        instead of modifying it in place, you should not count on the original image
        being open and usable once this function completes though it is safe to
        use the input image in a ``with`` context, if you need to retain a
        copy, pass a copy.

        :raise dgenerate.OutOfMemoryError: if the execution device runs out of memory
        :raise dgenerate.ImageProcessorImageModeError: if a passed image has an invalid format

        :param image: image to process
        :param resize_resolution: image will be resized to this dimension by this method.
        :param aspect_correct: Should the resize operation be aspect correct?
        :param align: Align by this amount of pixels, if the input image is not aligned
            to this amount of pixels, it will be aligned by resizing. Passing ``None``
            or ``1`` disables alignment.

        :return: the processed image
        """

        forced_alignment = self.get_alignment()
        if forced_alignment is not None:
            if (not _image.is_aligned(image.size, align=forced_alignment)) and align != forced_alignment:
                align = forced_alignment
                _messages.warning(
                    f'"{self.loaded_by_name}" image processor requires an image alignment of {align}, '
                    f'this alignment has been forced to prevent an error.'
                )

        # This is the actual size it will end
        # up being resized to by resize_image
        calculate_new_size = _image.resize_image_calc(old_size=image.size,
                                                      new_size=resize_resolution,
                                                      aspect_correct=aspect_correct,
                                                      align=align)

        pre_processed = self._process_pre_resize(image,
                                                 calculate_new_size)

        if resize_resolution is None:
            image = pre_processed
        else:
            image = _image.resize_image(img=pre_processed,
                                        size=resize_resolution,
                                        aspect_correct=aspect_correct,
                                        align=align)

        if image is not pre_processed:
            pre_processed.close()

        return self._process_post_resize(image)

    @abc.abstractmethod
    def impl_pre_resize(self, image: PIL.Image.Image,
                        resize_resolution: dgenerate.types.OptionalSize) -> PIL.Image.Image:
        """
        Inheritor must implement.

        This method should not be invoked directly, use the class method
        :py:meth:`.ImageProcessor.call_pre_resize` to invoke it.

        :param image: image to process
        :param resize_resolution: image will be resized to this resolution
            after this process is complete. If None is passed no resize is
            going to occur. It is not the duty of the inheritor to resize the
            image, in fact it should NEVER be resized.

        :return: the processed image
        """
        return image

    @abc.abstractmethod
    def impl_post_resize(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Inheritor must implement.

        This method should not be invoked directly, use the class method
        :py:meth:`.ImageProcessor.call_post_resize` to invoke it.

        :param image: image to process
        :return: the processed image
        """
        return image

    def __repr__(self):
        return str(self)

    def register_module(self, module):
        """
        Register :py:class:`torch.nn.Module` objects.

        These will be brought on to the cpu during finalization.

        All of these modules can be cast to a specific device with :py:attr:`.ImageProcessor.to`

        :param module: the module
        """
        self.__modules.append(module)

    def __flush_diffusion_pipeline_after_oom(self):

        _messages.debug_log(
            f'Image processor "{self.__class__.__name__}" is clearing the GPU side object '
            f'cache for device {self.device} due to VRAM out of memory condition.')

        _devicecache.clear_device_cache(self.device)

    def __to(self, device: torch.device | str, attempt=0):

        device = torch.device(device)

        if device.type != 'cpu':
            _image_processor_cache.size -= self.__size_estimate
        else:
            _image_processor_cache.size += self.__size_estimate

        self.__modules_device = device

        try_again = False

        for m in self.__modules:
            if not hasattr(m, '_DGENERATE_IMAGE_PROCESSOR_DEVICE') or \
                    not _torchutil.devices_equal(m._DGENERATE_IMAGE_PROCESSOR_DEVICE, device):

                self.memory_guard_device(device, self.size_estimate)

                m._DGENERATE_IMAGE_PROCESSOR_DEVICE = device
                _messages.debug_log(
                    f'Moving ImageProcessor registered module: {dgenerate.types.fullname(m)}.to("{device}")')

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
                            f'ImageProcessor "{self.__class__.__name__}" failed attempt '
                            f'at OOM recovery in to({device})')

                        m._DGENERATE_IMAGE_PROCESSOR_DEVICE = torch.device('cpu')

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

    def to(self, device: torch.device | str) -> "ImageProcessor":
        """
        Move all :py:class:`torch.nn.Module` modules registered
        to this image processor to a specific device.

        :raise dgenerate.OutOfMemoryError: if there is not enough memory on the specified device

        :param device: The device string, or torch device object
        :return: the image processor itself
        """
        return self.__to(device)


__all__ = dgenerate.types.module_all()
