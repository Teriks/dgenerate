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
import gc
import itertools
import os
import typing

import PIL.Image
import torch

import dgenerate.exceptions as _d_exceptions
import dgenerate.filelock as _filelock
import dgenerate.image as _image
import dgenerate.imageprocessors.constants as _constants
import dgenerate.imageprocessors.exceptions as _exceptions
import dgenerate.memory as _memory
import dgenerate.messages as _messages
import dgenerate.pipelinewrapper.cache as _m_cache
import dgenerate.pipelinewrapper.pipelines as _pipelines
import dgenerate.pipelinewrapper.util as _util
import dgenerate.plugin as _plugin
import dgenerate.types


class ImageProcessor(_plugin.Plugin):
    """
    Abstract base class for image processor implementations.
    """

    @classmethod
    def inheritable_help(cls, subclass, loaded_by_name):
        hidden_args = subclass.get_hidden_args(loaded_by_name)
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

        help_str = '\n\n'.join(
            message for arg, message in help_messages.items() if arg not in hidden_args)
        return help_str

    def __init__(self,
                 loaded_by_name: str,
                 device: typing.Optional[str] = None,
                 output_file: dgenerate.types.OptionalPath = None,
                 output_overwrite: bool = False,
                 model_offload: bool = False,
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
        :param kwargs: child class forwarded arguments
        """

        super().__init__(loaded_by_name=loaded_by_name,
                         argument_error_type=_exceptions.ImageProcessorArgumentError,
                         **kwargs)

        if device is not None:
            try:
                if not _util.is_valid_device_string(device):
                    raise _exceptions.ImageProcessorArgumentError(
                        f'Invalid device argument: "{device}" is not a valid device string.')
            except _util.InvalidDeviceOrdinalException as e:
                raise _exceptions.ImageProcessorArgumentError(
                    f'Invalid device argument: {e}')

        self.__output_file = output_file
        self.__output_overwrite = output_overwrite
        self.__device = device if device else 'cpu'
        self.__modules = []
        self.__modules_device = torch.device('cpu')
        self.__model_offload = model_offload
        self.__size_estimate = 0

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
        Set the estimated size of this model in bytes for memory management
        heuristics, this is intended to be used by implementors of the
        :py:class:`ImageProcessor` plugin class.

        For the best memory optimization, this value should be set very
        shortly before the model even enters CPU side ram, IE: before it
        is loaded at all.

        :raise ValueError: if ``size_bytes`` is less than zero.

        :param size_bytes: the size in bytes
        """
        if size_bytes < 0:
            raise ValueError(
                'image processor size estimate cannot be less than zero.')

        self.__size_estimate = int(size_bytes)

        if (_memory.memory_constraints(
                _constants.IMAGE_PROCESSOR_MEMORY_CONSTRAINTS,
                extra_vars={'processor_size': self.__size_estimate})):
            # wipe out the cpu side diffusion pipelines cache
            # and do a GC pass to free up cpu side memory since
            # we are nearly out of memory anyway

            _messages.debug_log(
                f'Image processor "{self.__class__.__name__}" is clearing the entire CPU side diffusion '
                f'model cache due to CPU side memory constraint evaluating to to True.')

            _m_cache.clear_model_cache()

    @property
    def size_estimate(self) -> int:
        """
        Get the estimated size of this model in bytes.

        :return: size bytes
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
            torch.cuda.empty_cache()
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
        except MemoryError:
            gc.collect()
            raise _d_exceptions.OutOfMemoryError('cpu (system memory)')
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
        Get required input image alignment, which will be forcefully applied.

        If this function returns ``None``, specific alignment is not required and will never be forced.

        :return: integer or ``None``
        """
        return None

    def process(self,
                image: PIL.Image.Image,
                resize_resolution: dgenerate.types.OptionalSize = None,
                aspect_correct: bool = True,
                align: int | None = 8):
        """
        Preform image processing on an image, including the requested resizing step.

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
                _messages.log(
                    f'"{self.loaded_by_name}" image processor requires an image alignment of {align}, '
                    f'this alignment has been forced to prevent an error.', level=_messages.WARNING)

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

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: dgenerate.types.OptionalSize):
        """
        Implementation of pre_resize that does nothing. Inheritor must implement.

        This method should not be invoked directly, use the class method
        :py:meth:`.ImageProcessor.call_pre_resize` to invoke it.

        :param image: image to process
        :param resize_resolution: image will be resized to this resolution
            after this process is complete.  If None is passed no resize is
            going to occur. It is not the duty of the inheritor to resize the
            image, in fact it should NEVER be resized.

        :return: the processed image
        """
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Implementation of post_resize that does nothing. Inheritor must implement.

        This method should not be invoked directly, use the class method
        :py:meth:`.ImageProcessor.call_post_resize` to invoke it.

        :param image: image to process
        :return: the processed image
        """
        return image

    def __str__(self):
        return f'{self.__class__.__name__}(loaded_by_name="{self.loaded_by_name}")'

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

    def __gpu_memory_fence_to(self, device: torch.device | str):
        if (device.type == 'cuda' and _memory.cuda_memory_constraints(
                _constants.IMAGE_PROCESSOR_CUDA_MEMORY_CONSTRAINTS,
                extra_vars={'processor_size': self.size_estimate},
                device=device)):

            # if there is a diffusion pipeline cached in
            # VRAM on the device we are moving to, it is guaranteed
            # to be a huge chunk of VRAM.
            #
            # Cacheing the last called diffusion pipeline on the GPU is
            # only to enhance the speed of execution, and it is
            # not required to be on the GPU if we are running
            # low on VRAM while trying to have it there next
            # to an image processor
            #
            # So we can fence the GPU when it is low on memory
            # IE: remove the pipeline and put the processor on
            #
            # when executing image processing after calling a large
            # diffusion model, this can prevent OOM at the cost of
            # execution speed for the next invocation of the diffusion
            # model

            active_pipe = _pipelines.get_last_called_pipeline()

            if active_pipe is not None \
                    and _pipelines.get_torch_device(active_pipe).index == device.index:
                # get rid of this reference immediately
                # noinspection PyUnusedLocal
                active_pipe = None

                _messages.debug_log(
                    f'Image processor "{self.__class__.__name__}" is attempting to evacuate any previously '
                    f'called diffusion pipeline in VRAM due to cuda memory constraint evaluating '
                    f'to True.')

                # potentially free up VRAM on the GPU we are
                # about to move to
                _pipelines.destroy_last_called_pipeline()

    def __flush_diffusion_pipeline_after_oom(self):
        _messages.debug_log(
            f'Image processor "{self.__class__.__name__}" is attempting to evacuate any previously '
            f'called diffusion pipeline in VRAM due to initial cuda out of memory condition.')
        _pipelines.destroy_last_called_pipeline()

    def __to(self, device: torch.device | str, attempt=0):
        device = torch.device(device)

        self.__modules_device = device

        try_again = False

        for m in self.__modules:
            if not hasattr(m, '_DGENERATE_IMAGE_PROCESSOR_DEVICE') or m._DGENERATE_IMAGE_PROCESSOR_DEVICE != device:

                self.__gpu_memory_fence_to(device)

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
                        raise _d_exceptions.OutOfMemoryError(e)
                except MemoryError as e:
                    # out of cpu side memory
                    self.__flush_mem_ignore_error()
                    raise _d_exceptions.OutOfMemoryError('cpu (system memory)')

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
