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

import dgenerate.types as _types
from dgenerate.preprocessors.preprocessor import ImagePreprocessor


class OpenPosePreprocess(ImagePreprocessor):
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
                 include_body=True,
                 include_hand=False,
                 include_face=False,
                 pre_resize=False, **kwargs):

        super().__init__(**kwargs)

        self._openpose = _cna.OpenposeDetector.from_pretrained('lllyasviel/Annotators')
        self._openpose.to(self.device)
        self._include_body = self.get_bool_arg("include-body", include_body)
        self._include_hand = self.get_bool_arg("include-hand", include_hand)
        self._include_face = self.get_bool_arg("include-face", include_face)
        self._pre_resize = self.get_bool_arg("pre-resize", pre_resize)

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

    def pre_resize(self, image: PIL.Image.Image, resize_resolution: _types.OptionalSize):
        if self._pre_resize:
            return self._process(image)
        return image

    def post_resize(self, image: PIL.Image.Image):
        if not self._pre_resize:
            return self._process(image)
        return image
