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
import controlnet_aux as _cna
import controlnet_aux.open_pose as _cna_open_pose
import controlnet_aux.util as _cna_util
import cv2
import numpy

import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.types as _types


class OpenPoseProcessor(_imageprocessor.ImageProcessor):
    """
    Generate an OpenPose rigging from the input image (of a human/humanoid) for use with a ControlNet.
    "include-body" is a boolean value indicating if a body rigging should be generated.
    "include-hand" is a boolean value indicating if a detailed hand/finger rigging should be generated.
    "include-face" is a boolean value indicating if a detailed face rigging should be generated.
    "pre-resize" is a boolean value determining if the processing should take place before or after the image
    is resized by dgenerate.
    """

    NAMES = ['openpose']

    def __init__(self,
                 include_body: bool = True,
                 include_hand: bool = False,
                 include_face: bool = False,
                 pre_resize: bool = False, **kwargs):
        """

        :param include_body: generate a body rig?
        :param include_hand: include detailed hand rigging?
        :param include_face: include detailed face rigging?
        :param pre_resize: process the image before it is resized, or after? default is after (False)
        :param kwargs: forwarded to base class
        """

        super().__init__(**kwargs)

        self._openpose = _cna.OpenposeDetector.from_pretrained('lllyasviel/Annotators')
        self._openpose.to(self.device)
        self._include_body = include_body
        self._include_hand = include_hand
        self._include_face = include_face
        self._pre_resize = pre_resize

    def __str__(self):
        args = [
            ('include_body', self._include_body),
            ('include_hand', self._include_hand),
            ('include_face', self._include_face),
            ('pre_resize', self._pre_resize)
        ]
        return f'{self.__class__.__name__}({", ".join(f"{k}={v}" for k, v in args)})'

    def _process(self, image: PIL.Image.Image):

        input_image = _cna_util.HWC3(numpy.array(image, dtype=numpy.uint8))

        height, width = input_image.shape[:2]

        poses = self._openpose.detect_poses(input_image,
                                            self._include_hand,
                                            self._include_face)

        canvas = _cna_open_pose.draw_poses(poses, height, width,
                                           draw_body=self._include_body,
                                           draw_hand=self._include_hand,
                                           draw_face=self._include_face)

        detected_map = canvas
        detected_map = _cna_util.HWC3(detected_map)

        detected_map = cv2.resize(detected_map, (width, height), interpolation=cv2.INTER_LINEAR)

        return PIL.Image.fromarray(detected_map)

    def impl_pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        """
        Pre resize, OpenPose rig generation may or may not occur here depending
        on the boolean value of the processor argument "pre-resize"


        :param image: image to process
        :param resize_resolution: purely informational, is unused by this processor
        :return: possibly an OpenPose rig image, or the input image
        """
        if self._pre_resize:
            return self._process(image)
        return image

    def impl_post_resize(self, image: PIL.Image.Image):
        """
        Post resize, OpenPose rig generation may or may not occur here depending
        on the boolean value of the processor argument "pre-resize"

        :param image: image to process
        :return: possibly an OpenPose rig image, or the input image
        """
        if not self._pre_resize:
            return self._process(image)
        return image


__all__ = _types.module_all()
