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
import itertools
import os
import typing

import PIL.Image

import dgenerate.filelock as _filelock
import dgenerate.image as _image
import dgenerate.imageprocessors.exceptions as _exceptions
import dgenerate.messages as _messages
import dgenerate.plugin as _plugin
import dgenerate.types


class ImageProcessor(_plugin.Plugin):
    """
    Abstract base class for image processor implementations.
    """

    def __init__(self,
                 loaded_by_name: str,
                 device: str = 'cpu',
                 output_file: dgenerate.types.OptionalPath = None,
                 output_overwrite: bool = False, **kwargs):

        super().__init__(loaded_by_name=loaded_by_name,
                         argument_error_type=_exceptions.ImageProcessorArgumentError,
                         **kwargs)

        self.__output_file = output_file
        self.__output_overwrite = output_overwrite
        self.__device = device

    @property
    def device(self) -> str:
        """
        The rendering device requested for this processor.

        :return: device string, for example "cuda", "cuda:N", or "cpu"
        """
        return self.__device

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

        :param self: :py:class:`.ImageProcessor` implementation instance
        :param image: the image to pass
        :param resize_resolution: the size that the image is going to be resized
            to after this step, or None if it is not being resized.

        :return: processed image, may be the same image or a copy.
        """

        img_copy = image.copy()

        processed = self.impl_pre_resize(image, resize_resolution)
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

        :param self: :py:class:`.ImageProcessor` implementation instance
        :param image: the image to pass

        :return: processed image, may be the same image or a copy.
        """

        img_copy = image.copy()

        processed = self.impl_post_resize(image)
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

    def process(self,
                image: PIL.Image.Image,
                resize_resolution: dgenerate.types.OptionalSize = None,
                aspect_correct: bool = True,
                align: typing.Optional[int] = 8):
        """
        Preform image processing on an image, including the requested resizing step.

        Invokes the image processor pre and post resizing with
        appropriate arguments and correct resource management.

        The original image will be closed if the implementation returns a new image
        instead of modifying it in place, you should not count on the original image
        being open and usable once this function completes though it is safe to
        use the input image in a ``with`` context, if you need to retain a
        copy, pass a copy.

        :param image: image to process
        :param resize_resolution: image will be resized to this dimension by this method.
        :param aspect_correct: Should the resize operation be aspect correct?
        :param align: Align by this amount of pixels, if the input image is not aligned
            to this amount of pixels, it will be aligned by resizing. Passing ``None``
            or ``1`` disables alignment.

        :return: the processed image
        """

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


__all__ = dgenerate.types.module_all()