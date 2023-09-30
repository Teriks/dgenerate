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
from controlnet_aux import OpenposeDetector
from .preprocessor import ImagePreprocessor


class OpenPosePreprocess(ImagePreprocessor):
    """
    Generate an OpenPose rigging from the input image (of a human/humanoid) for use with a ControlNet.
    "include-body" is a boolean value indicating if a body rigging should be generated.
    "include-hand" is a boolean value indicating if a detailed hand/finger rigging should be generated.
    "include-face" is a boolean value indicating if a detailed face rigging should be generated.
    """

    NAMES = ['openpose']

    def __init__(self,
                 include_body=True,
                 include_hand=False,
                 include_face=False, **kwargs):

        super().__init__(**kwargs)

        self._openpose = OpenposeDetector.from_pretrained('lllyasviel/Annotators')
        self._openpose.to(self.device)
        self._include_body = include_body
        self._include_hand = include_hand
        self._include_face = include_face

    def pre_resize(self, image: PIL.Image, resize_resolution: tuple):
        return image

    def post_resize(self, image: PIL.Image):
        return self._openpose(image,
                              include_body=self._include_body,
                              include_hand=self._include_hand,
                              include_face=self._include_face)

