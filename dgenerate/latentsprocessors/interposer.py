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
import torch.nn as nn
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

import dgenerate.latentsprocessors.exceptions as _exceptions
import dgenerate.latentsprocessors.latentsprocessor as _latentsprocessor
import dgenerate.pipelinewrapper.enums as _enums
import dgenerate.messages as _messages

from dgenerate.extras.sd_latent_interposer.interposer import InterposerModel as _InterposerModel
from dgenerate.extras.sd_latent_interposer.interposer import config as _config as _INTERPOSER_CONFIG

__doc__ = """
Latent space interposer processor implementation.

This processor converts latents between different diffusion model latent spaces
(e.g., SD1.x, SDXL, SD3, Flux) using trained neural network models.
"""


class InterposerProcessor(_latentsprocessor.LatentsProcessor):
    """
    Converts latents between different diffusion model latent spaces.

    This processor uses pre-trained neural network models to convert latents from
    one diffusion model's latent space to another (e.g., SD1.x to SDXL, SDXL to SD3).
    The conversion models are downloaded from the Hugging Face Hub or loaded from
    local cache when available.

    Supported conversions:
    - v1 (SD 1.x) ↔ xl (SDXL) ↔ v3 (SD3)
    - fx (Flux) → v1/xl/v3
    - ca (Stable Cascade) → v1/xl/v3

    The processor automatically handles batched tensors and preserves the batch
    dimension while converting the latent representations.
    """

    NAMES = ['interposer']

    def __init__(self,
                 source: str,
                 target: str,
                 version: float = 4.0,
                 **kwargs):
        """
        :param source: Source latent space (v1, xl, v3, fx, ca)
        :param target: Target latent space (v1, xl, v3)
        :param version: Model version to use (default: 4.0)
        :param kwargs: Additional arguments passed to base class
        """
        super().__init__(**kwargs)

        # Validate source and target
        valid_sources = ['v1', 'xl', 'v3', 'fx', 'ca']
        valid_targets = ['v1', 'xl', 'v3']

        if source not in valid_sources:
            raise self.argument_error(
                f"Invalid source '{source}'. Must be one of: {valid_sources}")

        if target not in valid_targets:
            raise self.argument_error(
                f"Invalid target '{target}'. Must be one of: {valid_targets}")

        self._source = source
        self._target = target
        self._version = version
        self._model_name = f"{source}-to-{target}"

        # Check if conversion is supported
        if self._model_name not in _INTERPOSER_CONFIG:
            raise self.argument_error(
                f"Conversion from '{source}' to '{target}' is not supported")

        # No-op if source equals target
        self._is_noop = (source == target)

        if not self._is_noop:
            # Load the interposer model
            self._model = self._load_interposer_model()
            self.register_module(self._model)

    def _get_model_path(self, model_name: str) -> str:
        """Get the path to an interposer model file."""
        fname = f"{model_name}_interposer-v{self._version}.safetensors"

        # Try to find model in the extras directory first
        extras_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'extras', 'sd_latent_interposer', 'models'
        )

        # Check local path: [models/xl-to-v1_interposer-v4.2.safetensors]
        local_path = os.path.join(extras_path, fname)
        if os.path.isfile(local_path):
            _messages.debug_log("InterposerProcessor: Using local model from extras")
            return local_path

        # Check versioned local path: [models/v4.2/xl-to-v1_interposer-v4.2.safetensors]
        versioned_path = os.path.join(extras_path, f"v{self._version}", fname)
        if os.path.isfile(versioned_path):
            _messages.debug_log("InterposerProcessor: Using versioned local model from extras")
            return versioned_path

        # Download from Hugging Face hub if not found locally and not local_files_only
        if not self.local_files_only:
            _messages.debug_log("InterposerProcessor: Using HF Hub model")
            return str(hf_hub_download(
                repo_id="city96/SD-Latent-Interposer",
                subfolder=f"v{self._version}",
                filename=fname,
            ))
        else:
            raise self.argument_error(
                f"Local interposer model file not found: {fname} "
                f"and local_files_only=True prevents downloading from Hugging Face Hub")

    def _load_interposer_model(self) -> InterposerModel:
        """Load and initialize the interposer model."""

        def _load():
            config_dict = _INTERPOSER_CONFIG[self._model_name]
            model = InterposerModel(**config_dict)
            model.eval()

            # Load model weights
            path = self._get_model_path(self._model_name)
            state_dict = load_file(path)
            model.load_state_dict(state_dict)

            return model

        # Estimate model size (approximate based on configuration)
        config_dict = _INTERPOSER_CONFIG[self._model_name]
        estimated_size = (
                config_dict["ch_mid"] * config_dict["blocks"] * 1024 +  # Core blocks
                config_dict["ch_in"] * config_dict["ch_mid"] * 256 +  # Head
                config_dict["ch_mid"] * config_dict["ch_out"] * 256  # Tail
        )

        return self.load_object_cached(
            tag=f"interposer_model_{self._model_name}_v{self._version}",
            estimated_size=estimated_size,
            method=_load,
            memory_guard_device=self.device
        )

    def process(self,
                pipeline,
                latents: torch.Tensor) -> torch.Tensor:
        """
        Convert latents between diffusion model latent spaces.
        
        :param pipeline: The pipeline object (unused but required by interface)
        :param latents: Input latents tensor with shape [B, C, W, H]
        :return: Converted latents tensor with shape [B, C', W, H] where C' depends on target space
        """
        if self._is_noop:
            return latents

        original_device = latents.device
        original_dtype = latents.dtype

        # Move model to same device as latents if needed
        if self.modules_device != latents.device:
            self.to(latents.device)

        with torch.no_grad():
            # Convert to float32 for processing (models trained on fp32)
            latents_fp32 = latents.float()

            # Process the latents
            converted_latents = self._model(latents_fp32)

            # Convert back to original dtype and ensure on original device
            converted_latents = converted_latents.to(dtype=original_dtype, device=original_device)

        return converted_latents
