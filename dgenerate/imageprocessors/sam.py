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
import os
import typing
import torch
import PIL.Image
import cv2
import huggingface_hub
import numpy

import controlnet_aux as _cna
import controlnet_aux.util as _cna_util
import dgenerate.image as _image
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.imageprocessors import imageprocessor as _imageprocessor


class SegmentAnythingProcessor(_imageprocessor.ImageProcessor):
    """
    Segment Anything Model, this processor attempts to creates cutouts for every distinct objects in an image.

    The "detect-resolution" argument is the resolution the image is resized to internal to the processor before
    detection is run on it. It should be a single dimension for example: "detect-resolution=512" or the X/Y dimensions
    seperated by an "x" character, like so: "detect-resolution=1024x512". If you do not specify this argument,
    the detector runs on the input image at its full resolution. After processing the image will be resized to
    whatever you have requested dgenerate resize it to via --output-size or --resize/--align in the case of the
    image-process sub-command, if you have not requested any resizing the output will be resized back to the original
    size of the input image.

    The "detect-aspect" argument determines if the image resize requested by "detect-resolution" before
    detection runs is aspect correct, this defaults to true.

    The "detect-align" argument determines the pixel alignment of the image resize requested by
    "detect-resolution", it defaults to 1 indicating no requested alignment.

    The "pre-resize" argument determines if the processing occurs before or after dgenerate resizes the image.
    This defaults to False, meaning the image is processed after dgenerate is done resizing it.
    """

    NAMES = ['sam']

    def __init__(self,
                 detect_resolution: typing.Optional[str] = None,
                 detect_aspect: bool = True,
                 detect_align: int = 1,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param detect_resolution: the input image is resized to this dimension before being processed,
            providing ``None`` indicates it is not to be resized.  If there is no resize requested during
            the processing action via ``resize_resolution`` it will be resized back to its original size.
        :param detect_aspect: if the input image is resized by ``detect_resolution`` or ``detect_align``
            before processing, will it be an aspect correct resize?
        :param detect_align: the input image is forcefully aligned to this amount of pixels
            before being processed.
        :param pre_resize: process the image before it is resized, or after? default is ``False`` (after).
        :param kwargs: forwarded to base class
        """

        super().__init__(**kwargs)

        if detect_align < 1:
            raise self.argument_error('Argument "detect-align" may not be less than 1.')

        self._detect_aspect = detect_aspect
        self._detect_align = detect_align
        self._pre_resize = pre_resize

        if detect_resolution is not None:
            try:
                self._detect_resolution = _textprocessing.parse_image_size(detect_resolution)
            except ValueError:
                raise self.argument_error('Could not parse the "detect-resolution" argument as an image dimension.')
        else:
            self._detect_resolution = None

        self.set_size_estimate(2564363480)  # 2.56436348 GB
        self._sam = self._from_pretrained('ybelkada/segment-anything', subfolder='checkpoints')

    def _from_pretrained(self,
                         pretrained_model_or_path,
                         model_type="vit_h",
                         filename="sam_vit_h_4b8939.pth",
                         subfolder=None,
                         cache_dir=None):

        if os.path.isdir(pretrained_model_or_path):
            model_path = os.path.join(pretrained_model_or_path, filename)
        else:
            model_path = huggingface_hub.hf_hub_download(
                pretrained_model_or_path, filename, subfolder=subfolder,
                cache_dir=cache_dir)

        sam = _cna.segment_anything.build_sam.sam_model_registry[model_type](checkpoint=model_path)
        self.register_module(sam)

        mask_generator = _cna.segment_anything.SamAutomaticMaskGenerator(sam)

        return _cna.SamDetector(mask_generator)

    def __str__(self):
        args = [
            ('detect_resolution', self._detect_resolution),
            ('detect_aspect', self._detect_aspect),
            ('detect_align', self._detect_align),
            ('pre_resize', self._pre_resize)
        ]
        return f'{self.__class__.__name__}({", ".join(f"{k}={v}" for k, v in args)})'

    @torch.inference_mode()
    def _process(self, image, resize_resolution):
        original_size = image.size

        with image:
            resized = _image.resize_image(
                image,
                self._detect_resolution,
                aspect_correct=self._detect_aspect,
                align=self._detect_align
            )

        image = resized

        input_image = numpy.array(image, dtype=numpy.uint8)

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

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Pre resize.

        :param image: image to process
        :param resize_resolution: resize resolution
        :return: possibly a segment-anything image, or the input image
        """

        if self._pre_resize:
            return self._process(image, resize_resolution)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Post resize.

        :param image: image
        :return: possibly a segment-anything image, or the input image
        """

        if not self._pre_resize:
            return self._process(image, None)
        return image


__all__ = _types.module_all()
