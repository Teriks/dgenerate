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
import torch
import transformers
import os
import diffusers
import dgenerate.extras.DistillT5.models.T5_encoder as _distill_t5_encoder

DistillT5EncoderModel = _distill_t5_encoder.T5EncoderWithProjection


class SiglipImageEncoder:
    """
    Siglip Vision Model and Siglib Image Processor
    """


    def __init__(self,
                 image_encoder: transformers.SiglipVisionModel,
                 feature_extractor: transformers.SiglipImageProcessor
                 ):
        self.image_encoder: transformers.SiglipVisionModel = image_encoder
        self.feature_extractor: transformers.SiglipImageProcessor = feature_extractor

    @staticmethod
    def from_pretrained(
            pretrained_model_name_or_path: str | os.PathLike,
            revision: str = "main",
            variant: str | None = None,
            subfolder: str | None = None,
            torch_dtype: torch.dtype | None = None,
            token: str | None = None,
            local_files_only: bool = False
    ):
        """
        Load from pretrained weights.

        :param pretrained_model_name_or_path: HuggingFace hub slug or path on disk.
        :param revision: repo revision (branch)
        :param variant: model variant
        :param subfolder: model subfolder
        :param torch_dtype: load as dtype
        :param token: HF auth token
        :param local_files_only: only look in cache?
        :return:
        """

        image_encoder = transformers.SiglipVisionModel.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            variant=variant,
            subfolder=subfolder if subfolder else '',
            torch_dtype=torch_dtype,
            token=token,
            local_files_only=local_files_only)

        feature_extractor = transformers.SiglipImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            variant=variant,
            subfolder=subfolder if subfolder else '',
            token=token,
            local_files_only=local_files_only)

        return SiglipImageEncoder(image_encoder, feature_extractor)

    def register(self, pipeline: diffusers.DiffusionPipeline):
        """
        Register to an existing diffusion pipeline.

        :param pipeline: the pipeline
        """
        pipeline.register_modules(
            image_encoder=self.image_encoder,
            feature_extractor=self.feature_extractor
        )

    @property
    def dtype(self) -> torch.dtype:
        """Return the model dtype."""
        return self.image_encoder.dtype

    def to(self, *args, **kwargs):
        """Cast to device or dtype."""
        self.image_encoder.to(*args, **kwargs)
        return self
