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
import einops
import numpy
import torch

import dgenerate.image as _image
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.imageprocessors import imageprocessor as _imageprocessor


class ZoeDepthProcessor(_imageprocessor.ImageProcessor):
    """
    zoe depth detector, a SOTA depth estimation model which produces high-quality depth maps.

    The "gamma-corrected" argument determines if gamma correction is preformed on the produced depth math.

    The "detect-resolution" argument is the resolution the image is resized to internal to the processor before
    detection is run on it. It should be a single dimension for example: "detect-resolution=512" or the X/Y dimensions
    seperated by an "x" character, like so: "detect-resolution=1024x512". If you do not specify this argument,
    the detector runs on the input image at its full resolution. After processing the image will be resized to
    whatever you have requested dgenerate resize it to via --output-size or --resize/--align in the case of the
    image-process sub-command, if you have not requested any resizing the output will be resized back to the original
    size of the input image.

    The "detect-aspect" argument determines if the image resize requested by "detect_resolution" before
    detection runs is aspect correct, this defaults to true.

    The "pre-resize" argument determines if the processing occurs before or after dgenerate resizes the image.
    This defaults to False, meaning the image is processed after dgenerate is done resizing it.
    """

    NAMES = ['zoe']

    def __init__(self,
                 gamma_corrected: bool = False,
                 detect_resolution: typing.Optional[str] = None,
                 detect_aspect: bool = True,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param gamma_corrected: perform gamma correction on the depth map?
        :param detect_resolution: the input image is resized to this dimension before being processed,
            providing ``None`` indicates it is not to be resized.  If there is no resize requested during
            the processing action via ``resize_resolution`` it will be resized back to its original size.
        :param detect_aspect: if the input image is resized by ``detect_resolution`` or ``detect_align``
            before processing, will it be an aspect correct resize?
        :param pre_resize: process the image before it is resized, or after? default is ``False`` (after).
        :param kwargs: forwarded to base class
        """

        super().__init__(**kwargs)

        self._detect_aspect = detect_aspect
        self._pre_resize = pre_resize
        self._gamma_corrected = gamma_corrected

        if detect_resolution is not None:
            try:
                self._detect_resolution = _textprocessing.parse_image_size(detect_resolution)
            except ValueError:
                raise self.argument_error('Could not parse the "detect-resolution" argument as an image dimension.')
        else:
            self._detect_resolution = None

        self.set_size_estimate(1443288172)  # 1.443288172 GB ZoeD_M12_N.pt['model']
        self._zoe = _cna.ZoeDetector.from_pretrained("lllyasviel/Annotators")
        self.register_module(self._zoe)

    def __str__(self):
        args = [
            ('gamma_corrected', self._gamma_corrected),
            ('detect_resolution', self._detect_resolution),
            ('detect_aspect', self._detect_aspect),
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
                align=8
            )

        image_depth = _cna_util.HWC3(numpy.array(resized, dtype=numpy.uint8))

        image_depth = torch.from_numpy(image_depth).float().to(self.modules_device)
        image_depth = image_depth / 255.0
        image_depth = einops.rearrange(image_depth, 'h w c -> 1 c h w')
        output = self._zoe.model.infer(image_depth).cpu()

        image_depth.cpu()
        del image_depth

        depth = output[0, 0].numpy()
        del output

        vmin = numpy.percentile(depth, 2)
        vmax = numpy.percentile(depth, 85)

        depth -= vmin
        depth /= vmax - vmin
        depth = 1.0 - depth

        if self._gamma_corrected:
            depth = numpy.power(depth, 2.2)
        depth_image = (depth * 255.0).clip(0, 255).astype(numpy.uint8)

        detected_map = _cna_util.HWC3(depth_image)

        if resize_resolution is not None:
            detected_map = cv2.resize(detected_map, resize_resolution, interpolation=cv2.INTER_LINEAR)
        elif self._detect_resolution is not None:
            detected_map = cv2.resize(detected_map, original_size, interpolation=cv2.INTER_LINEAR)

        return PIL.Image.fromarray(detected_map)

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Pre resize.

        :param image: image to process
        :param resize_resolution: resize resolution
        :return: possibly a zoe depth detected image, or the input image
        """

        if self._pre_resize:
            return self._process(image, resize_resolution)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Post resize.

        :param image: image
        :return: possibly a zoe depth detected image, or the input image
        """

        if not self._pre_resize:
            return self._process(image, None)
        return image


__all__ = _types.module_all()
