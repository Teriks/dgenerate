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

import dgenerate.latentsprocessors.exceptions as _exceptions
import dgenerate.latentsprocessors.latentsprocessor as _latentsprocessor
import dgenerate.pipelinewrapper.enums as _enums

__doc__ = """
Latents scaling processor implementation.
"""


class LatentsScalerProcessor(_latentsprocessor.LatentsProcessor):
    """
    Scales and normalizes latents tensors with various operations.
    
    This processor provides scaling, clamping, and normalization operations
    for latents tensors, useful for controlling the magnitude and range
    of latent representations. The scaling is applied first, followed by
    optional normalization to unit variance, and finally value clamping.

    The "scale" argument multiplies all tensor values by the specified factor.
    Values greater than 1.0 increase the magnitude while values less than 1.0
    decrease it. If "normalize" is enabled, the tensor will be normalized to
    have unit variance after scaling. The "clamp_min" and "clamp_max" arguments
    can be used to constrain the final values to a specific range.

    This processor is useful for adjusting the dynamic range of latents before
    or after other processing steps in the diffusion pipeline.
    """
    
    NAMES = ['scaler']

    def __init__(self,
                 loaded_by_name: str,
                 model_type: _enums.ModelType,
                 device: str = 'cpu',
                 local_files_only: bool = False,
                 scale: float = 1.0,
                 clamp_min: float = None,
                 clamp_max: float = None,
                 normalize: bool = False,
                 **kwargs):
        """
        :param loaded_by_name: The name the processor was loaded by
        :param model_type: Model type enum
        :param device: Device to run processing on
        :param local_files_only: Whether to avoid downloading files from the internet
        :param scale: Scaling factor to multiply tensor values. Must be positive
        :param clamp_min: Minimum value to clamp tensor values to (optional)
        :param clamp_max: Maximum value to clamp tensor values to (optional)
        :param normalize: Whether to normalize tensor to unit variance after scaling
        :param kwargs: Additional arguments passed to base class
        """
        super().__init__(loaded_by_name=loaded_by_name,
                         model_type=model_type,
                         device=device,
                         local_files_only=local_files_only,
                         **kwargs)
        
        if scale <= 0:
            raise _exceptions.LatentsProcessorArgumentError(
                f"scale must be positive, got {scale}")
        
        if clamp_min is not None and clamp_max is not None and clamp_min >= clamp_max:
            raise _exceptions.LatentsProcessorArgumentError(
                f"clamp_min ({clamp_min}) must be less than clamp_max ({clamp_max})")
        
        self._scale = scale
        self._clamp_min = clamp_min
        self._clamp_max = clamp_max
        self._normalize = normalize

    @property
    def scale(self) -> float:
        """Get the scaling factor."""
        return self._scale

    @property
    def clamp_min(self) -> float:
        """Get the minimum clamp value."""
        return self._clamp_min

    @property
    def clamp_max(self) -> float:
        """Get the maximum clamp value."""
        return self._clamp_max

    @property
    def normalize(self) -> bool:
        """Get whether normalization is enabled."""
        return self._normalize

    def process(self,
                pipeline,
                latents: torch.Tensor) -> torch.Tensor:
        """
        Scale and process the latents tensor.
        
        :param pipeline: The pipeline object
        :param latents: Input latents tensor with shape [B, C, W, H]
        :return: Scaled and processed latents tensor, shape [B, C, W, H]
        """
        try:
            import torch
            
            result = latents.clone()
            
            # Apply scaling
            if self._scale != 1.0:
                result = result * self._scale
            
            # Apply normalization
            if self._normalize:
                # Normalize to have unit variance
                std = result.std()
                if std > 1e-8:  # Avoid division by zero
                    result = result / std
            
            # Apply clamping
            if self._clamp_min is not None or self._clamp_max is not None:
                result = result.clamp(min=self._clamp_min, max=self._clamp_max)
            
            return result
            
        except Exception as e:
            raise _exceptions.LatentsProcessorError(f"Failed to scale latents: {e}") from e 