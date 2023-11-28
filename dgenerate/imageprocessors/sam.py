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

import PIL.Image
import controlnet_aux as _cna
import controlnet_aux.util as _cna_util
import cv2
import numpy as np

import dgenerate.image as _image
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.imageprocessors import imageprocessor as _imageprocessor


class SegmentAnythingProcessor(_imageprocessor.ImageProcessor):
    """
    Segment Anything Model.

    The argument "detect_resolution" is the resolution the image is resized to before segment detection
    is run on it. It should be a single dimension for example: "detect_resolution=512" or the X/Y dimensions
    seperated by an "x" character, like so: "detect_resolution=1024x512". If you do not specify this argument,
    the detector runs on the input image at its full resolution. After processing the image will be resized
    to whatever you have requested dgenerate itself to resize it to.

    The argument "detect_aspect" determines if the image resize requested by "detect_resolution" before
    detection runs is aspect correct, this defaults to true.

    The argument "detect_align" determines the pixel alignment of the image resize requested by
    "detect_resolution", it defaults to 1 indicating no requested alignment.

    The argument "mobile_sam" indicates that a variant of the sam detector with a smaller memory footprint
    should be used: https://huggingface.co/dhkim2810/MobileSAM, by default the model used is the original
    from here: https://huggingface.co/ybelkada/segment-anything
    """

    NAMES = ['sam']

    def __init__(self,
                 detect_resolution: typing.Optional[str] = None,
                 detect_aspect: bool = True,
                 detect_align: int = 1,
                 mobile_sam=False, **kwargs):
        super().__init__(**kwargs)

        if mobile_sam:
            self._sam = _cna.SamDetector.from_pretrained("dhkim2810/MobileSAM", model_type="vit_t", filename="mobile_sam.pt")
        else:
            self._sam = _cna.SamDetector.from_pretrained('ybelkada/segment-anything', subfolder='checkpoints')

        self._detect_aspect = detect_aspect
        self._detect_align = detect_align

        if detect_resolution is not None:
            try:
                self._detect_resolution = _textprocessing.parse_image_size(detect_resolution)
            except ValueError:
                raise self.argument_error('Could not parse the "detect_resolution" argument as an image dimension.')
        else:
            self._detect_resolution = None

    def __str__(self):
        args = [
            ('detect_resolution', self._detect_resolution),
            ('detect_aspect', self._detect_aspect),
            ('detect_align', self._detect_align),
        ]
        return f'{self.__class__.__name__}({", ".join(f"{k}={v}" for k, v in args)})'

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Pre resize, segment-anything detection occurs here.


        :param image: image to process
        :param resize_resolution: purely informational, is unused by this processor
        :return: possibly a segment-anything image, or the input image
        """

        original_size = image.size

        if self._detect_resolution is not None:
            with image:
                resized = _image.resize_image(
                    image,
                    self._detect_resolution,
                    aspect_correct=self._detect_aspect,
                    align=self._detect_align
                )
            image = resized

        input_image = np.array(image, dtype=np.uint8)

        input_image = _cna_util.HWC3(input_image)

        detected_map = _cna_util.HWC3(self._sam.show_anns(self._sam.mask_generator.generate(input_image)))

        if resize_resolution is not None:
            # resize it to what dgenerate requested, this happens automatically,
            # but not with linear interpolation, so we will do it here, dgenerate will
            # see that it is already at the requested size and not resize it any further
            detected_map = cv2.resize(detected_map, resize_resolution, interpolation=cv2.INTER_LINEAR)
        elif self._detect_resolution is not None:
            # resize it to its original size since we changed its size before detection
            # and dgenerate did not request a resize
            detected_map = cv2.resize(detected_map, original_size, interpolation=cv2.INTER_LINEAR)

        return PIL.Image.fromarray(detected_map)

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Post resize, nothing happens here.

        :param image: image
        :return: the input image
        """
        return image
