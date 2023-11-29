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
import numpy as np
import torch

import dgenerate.image as _image
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types
from dgenerate.imageprocessors import imageprocessor as _imageprocessor


class HEDProcessor(_imageprocessor.ImageProcessor):
    """
    HED detection (holistically-nested edge detection), this is an edge detection algorithm
    that can produced something akin to thick lineart.

    The argument "detect-resolution" is the resolution the image is resized to internal to the processor before
    detection is run on it. It should be a single dimension for example: "detect-resolution=512" or the X/Y dimensions
    seperated by an "x" character, like so: "detect-resolution=1024x512". If you do not specify this argument,
    the detector runs on the input image at its full resolution. After processing the image will be resized to
    whatever you have requested dgenerate resize it to via --output-size or --resize/--align in the case of the
    image-process sub-command, if you have not requested any resizing the output will be resized back to the original
    size of the input image.

    The argument "detect-aspect" determines if the image resize requested by "detect_resolution" before
    detection runs is aspect correct, this defaults to true.

    The argument "detect-align" determines the pixel alignment of the image resize requested by
    "detect_resolution", it defaults to 1 indicating no requested alignment.

    The "scribble" argument determines whether or not scribble mode is enabled, this produces thicker lines.

    The "safe" argument enables numerically safe / more precise stepping

    The "pre-resize" argument determines if the processing occurs before or after dgenerate resizes the image.
    This defaults to False, meaning the image is processed after dgenerate is done resizing it.
    """

    NAMES = ['hed']

    def __init__(self,
                 detect_resolution: typing.Optional[str] = None,
                 detect_aspect: bool = True,
                 detect_align: int = 1,
                 scribble: bool = False,
                 safe: bool = False,
                 pre_resize: bool = False,
                 **kwargs):

        super().__init__(**kwargs)

        self._hed = _cna.HEDdetector.from_pretrained("lllyasviel/Annotators")

        self._hed.to(self.device)

        self._detect_aspect = detect_aspect
        self._detect_align = detect_align
        self._safe = safe
        self._scribble = scribble
        self._pre_resize = pre_resize

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
            ('safe', self._safe),
            ('scribble', self._scribble),
            ('pre_resize', self._pre_resize)
        ]
        return f'{self.__class__.__name__}({", ".join(f"{k}={v}" for k, v in args)})'

    def _process(self, image, resize_resolution, return_to_original_size=False):
        original_size = image.size

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
        H, W, C = input_image.shape
        with torch.no_grad():
            image_hed = torch.from_numpy(input_image.copy()).float().to(self.device)
            image_hed = einops.rearrange(image_hed, 'h w c -> 1 c h w')
            edges = self._hed.netNetwork(image_hed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            if self._safe:
                edge = _cna_util.safe_step(edge)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = edge
        detected_map = _cna_util.HWC3(detected_map)

        if resize_resolution is not None:
            detected_map = cv2.resize(detected_map, resize_resolution, interpolation=cv2.INTER_LINEAR)
        elif self._detect_resolution is not None and return_to_original_size:
            detected_map = cv2.resize(detected_map, original_size, interpolation=cv2.INTER_LINEAR)

        if self._scribble:
            detected_map = _cna_util.nms(detected_map, 127, 3.0)
            detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
            detected_map[detected_map > 4] = 255
            detected_map[detected_map < 255] = 0

        return PIL.Image.fromarray(detected_map)

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Pre resize.

        :param image: image to process
        :param resize_resolution: resize resolution
        :return: possibly an HED detected image, or the input image
        """

        if not self._pre_resize:
            return image

        return self._process(image, resize_resolution, return_to_original_size=True)

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Post resize.

        :param image: image
        :return: possibly an HED detected image, or the input image
        """

        if self._pre_resize:
            return image

        return self._process(image, None)
