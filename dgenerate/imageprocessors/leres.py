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
import cv2
import numpy
import torch

import controlnet_aux as _cna
import controlnet_aux.util as _cna_util
import dgenerate.image as _image
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.imageprocessors import imageprocessor as _imageprocessor


class LeresDepthProcessor(_imageprocessor.ImageProcessor):
    """
    LeReS depth detector.

    The "threshold-near" argument is the near threshold, think the low threshold of canny.

    The "threshold-far" argument is the far threshold, think the high threshold of canny.

    The "boost" argument determines if monocular depth boost is used.

    The argument "detect-resolution" is the resolution the image is resized to internal to the processor before
    detection is run on it. It should be a single dimension for example: "detect-resolution=512" or the X/Y dimensions
    seperated by an "x" character, like so: "detect-resolution=1024x512". If you do not specify this argument,
    the detector runs on the input image at its full resolution. After processing the image will be resized to
    whatever you have requested dgenerate resize it to via --output-size or --resize/--align in the case of the
    image-process sub-command, if you have not requested any resizing the output will be resized back to the original
    size of the input image.

    The argument "detect-aspect" determines if the image resize requested by "detect-resolution" before
    detection runs is aspect correct, this defaults to true.

    The argument "pre-resize" determines if the processing occurs before or after dgenerate resizes the image.
    This defaults to False, meaning the image is processed after dgenerate is done resizing it.
    """

    NAMES = ['leres']

    # Force incoming image alignment to 64 pixels, required
    def get_alignment(self):
        return 64

    def __init__(self,
                 threshold_near: int = 0,
                 threshold_far: int = 0,
                 boost: bool = False,
                 detect_resolution: typing.Optional[str] = None,
                 detect_aspect: bool = True,
                 pre_resize: bool = False,
                 **kwargs):
        """
        :param threshold_near: argument is the near threshold, think the low threshold of canny
        :param threshold_far: argument is the far threshold, think the high threshold of canny
        :param boost: argument determines if monocular depth boost is used
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

        self._detect_aspect = detect_aspect
        self._pre_resize = pre_resize
        self._boost = boost
        self._threshold_near = threshold_near
        self._threshold_far = threshold_far

        if detect_resolution is not None:
            try:
                self._detect_resolution = _textprocessing.parse_image_size(detect_resolution)
            except ValueError:
                raise self.argument_error('Could not parse the "detect-resolution" argument as an image dimension.')
        else:
            self._detect_resolution = None

        self._leres = _cna.LeresDetector.from_pretrained("lllyasviel/Annotators")
        self.register_module(self._leres)

    def __str__(self):
        args = [
            ('threshold_near', self._threshold_near),
            ('threshold_far', self._threshold_far),
            ('boost', self._boost),
            ('detect_resolution', self._detect_resolution),
            ('detect_aspect', self._detect_aspect),
            ('pre_resize', self._pre_resize)
        ]
        return f'{self.__class__.__name__}({", ".join(f"{k}={v}" for k, v in args)})'

    def _process(self, image, resize_resolution):
        original_size = image.size

        with image:
            # must be aligned to 8 pixels, forcefully align
            resized = _image.resize_image(
                image,
                self._detect_resolution,
                aspect_correct=self._detect_aspect,
                align=8
            )

        image = resized

        input_image = numpy.array(image, dtype=numpy.uint8)
        input_image = _cna_util.HWC3(input_image)
        height, width, dim = input_image.shape

        with torch.no_grad():

            if self._boost:
                depth = _cna.leres.estimateboost(input_image, self._leres.model, 0, self._leres.pix2pixmodel,
                                                 max(width, height))
            else:
                depth = _cna.leres.estimateleres(input_image, self._leres.model, width, height)

            numbytes = 2
            depth_min = depth.min()
            depth_max = depth.max()
            max_val = (2 ** (8 * numbytes)) - 1

            # check output before normalizing and mapping to 16 bit
            if depth_max - depth_min > numpy.finfo("float").eps:
                out = max_val * (depth - depth_min) / (depth_max - depth_min)
            else:
                out = numpy.zeros(depth.shape)

            # single channel, 16 bit image
            depth_image = out.astype("uint16")

            # convert to uint8
            depth_image = cv2.convertScaleAbs(depth_image, alpha=(255.0 / 65535.0))

            # remove near
            if self._threshold_near != 0:
                threshold_near = ((self._threshold_near / 100) * 255)
                depth_image = cv2.threshold(depth_image, threshold_near, 255, cv2.THRESH_TOZERO)[1]

            # invert image
            depth_image = cv2.bitwise_not(depth_image)

            # remove bg
            if self._threshold_far != 0:
                threshold_far = ((self._threshold_far / 100) * 255)
                depth_image = cv2.threshold(depth_image, threshold_far, 255, cv2.THRESH_TOZERO)[1]

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
        :return: possibly a LeReS depth detected image, or the input image
        """

        if self._pre_resize:
            return self._process(image, resize_resolution)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Post resize.

        :param image: image
        :return: possibly a LeReS depth detected image, or the input image
        """

        if not self._pre_resize:
            return self._process(image, None)
        return image


__all__ = _types.module_all()
