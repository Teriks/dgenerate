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
import dgenerate.extras.controlnet_aux as _cna
import dgenerate.extras.controlnet_aux.open_pose as _cna_open_pose
import dgenerate.extras.controlnet_aux.util as _cna_util
import numpy
import torch

import dgenerate.hfhub as _hfhub
import dgenerate.image as _image
import dgenerate.imageprocessors.imageprocessor as _imageprocessor
import dgenerate.textprocessing as _textprocessing
import dgenerate.types as _types


class OpenPoseProcessor(_imageprocessor.ImageProcessor):
    """
    Generate an OpenPose rigging from the input image (of a human/humanoid) for use with a ControlNet.

    "include-body" is a boolean value indicating if a body rigging should be generated.

    "include-hand" is a boolean value indicating if a detailed hand/finger rigging should be generated.

    "include-face" is a boolean value indicating if a detailed face rigging should be generated.

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

    NAMES = ['openpose']

    def __init__(self,
                 include_body: bool = True,
                 include_hand: bool = False,
                 include_face: bool = False,
                 detect_resolution: str | None = None,
                 detect_aspect: bool = True,
                 detect_align: int = 1,
                 pre_resize: bool = False,
                 **kwargs):
        """

        :param include_body: generate a body rig?
        :param include_hand: include detailed hand rigging?
        :param include_face: include detailed face rigging?
        :param pre_resize: process the image before it is resized, or after? default is after (False)
        :param detect_resolution: the input image is resized to this dimension before being processed,
            providing ``None`` indicates it is not to be resized. If there is no resize requested during
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

        self._include_body = include_body
        self._include_hand = include_hand
        self._include_face = include_face
        self._pre_resize = pre_resize
        self._detect_align = detect_align
        self._detect_aspect = detect_aspect

        if detect_resolution is not None:
            try:
                self._detect_resolution = _textprocessing.parse_image_size(detect_resolution)
            except ValueError as e:
                raise self.argument_error(
                    f'Could not parse the "detect-resolution" argument as an image dimension: {e}') from e
        else:
            self._detect_resolution = None

        self.set_size_estimate(510 * (1000 ** 2))  # 510 MB body_pose_model.pth + hand_pose_model.pth + facenet.pth
        with (_hfhub.with_hf_errors_as_model_not_found(),
              _hfhub.offline_mode_context(self.local_files_only)):
            self._openpose = self.load_object_cached(
                tag="lllyasviel/Annotators",
                estimated_size=self.size_estimate,
                method=lambda: _cna.OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            )
        self.register_module(self._openpose)

    @torch.inference_mode()
    def _process(self, image):
        original_size = image.size

        with image:
            resized = _image.resize_image(
                image,
                self._detect_resolution,
                aspect_correct=self._detect_aspect,
                align=self._detect_align
            )

        image = resized

        input_image = _cna_util.HWC3(numpy.array(image, dtype=numpy.uint8))

        height, width = input_image.shape[:2]

        poses = self._openpose.detect_poses(input_image,
                                            self._include_hand,
                                            self._include_face)

        canvas = _cna_open_pose.draw_poses(poses, height, width,
                                           draw_body=self._include_body,
                                           draw_hand=self._include_hand,
                                           draw_face=self._include_face)

        detected_map = _cna_util.HWC3(canvas)

        detected_map = _image.cv2_resize_image(detected_map, original_size)

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
