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

import PIL.Image

import dgenerate.image as _image
import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types


class OutpaintMaskProcessor(_imageprocessor.ImageProcessor):
    """
    Create a mask that can be used to extend the incoming image.

    The "box" argument is the size of the outer mask as it extends from the inner image,
    This may be specified as a uniform pixel padding "WIDTHxHEIGHT", or as a
    four part padding "LEFTxTOPxRIGHTxBOTTOM", you may also specify padding width and
    height simultaneously with a single integer.

    The "pre-resize" argument determines if the processing occurs before or after dgenerate resizes the image.
    This defaults to False, meaning the image is processed after dgenerate is done resizing it.
    """

    NAMES = ['outpaint-mask']

    # hide inherited arguments
    # that are device related
    HIDE_ARGS = ['device', 'model-offload']

    def __init__(self,
                 box: str,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param box: Size of the outer mask
        """
        super().__init__(**kwargs)

        try:
            dim = _textprocessing.parse_dimensions(box)

            if len(dim) not in {1, 2, 4}:
                raise self.argument_error('Argument "box" must be a 1, 2, or 4 dimensional padding value.')

            if len(dim) == 1:
                self._box_size = dim[0]
            else:
                self._box_size = dim
        except ValueError as e:
            raise self.argument_error(f'Could not parse the "box" argument as a dimension: {e}') from e

        self._pre_resize = pre_resize

    def _process(self, image: PIL.Image.Image):
        return _image.letterbox_image(
            PIL.Image.new('RGB', image.size, (0, 0, 0)),
            box_size=self._box_size,
            box_is_padding=True,
            box_color='#FFFFFF',
            inner_size=image.size)

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Operation is performed by this method if ``pre_resize`` constructor argument was ``True``.

        :param image: image to process
        :param resize_resolution: purely informational, is unused by this imageprocessor
        :return: the letterboxed image
        """
        if self._pre_resize:
            return self._process(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Operation is performed by this method if ``pre_resize`` constructor argument was ``False``.

        :param image: image to process
        :return: the letterboxed image
        """
        if not self._pre_resize:
            return self._process(image)
        return image

    def to(self, device) -> "OutpaintMaskProcessor":
        """
        Does nothing for this processor.
        :param device: the device
        :return: this processor
        """
        return self
