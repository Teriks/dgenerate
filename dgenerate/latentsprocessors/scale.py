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

import dgenerate.latentsprocessors.latentsprocessor as _latentsprocessor

__doc__ = """
Latents scaling processor implementation.
"""


class ScaleProcessor(_latentsprocessor.LatentsProcessor):
    """
    Scales and normalizes latents tensors.
    
    This processor provides scaling, clamping, and normalization operations
    for latents tensors, useful for controlling the magnitude and range
    of latent representations. The scaling is applied first, followed by
    optional normalization to unit variance, and finally value clamping.

    The "factor" argument multiplies all tensor values by the specified factor.
    Values greater than 1.0 increase the magnitude while values less than 1.0
    decrease it. If "normalize" is enabled, the tensor will be normalized to
    have unit variance after scaling. The "clamp_min" and "clamp_max" arguments
    can be used to constrain the final values to a specific range.

    This processor is useful for adjusting the dynamic range of latents before
    or after other processing steps in the diffusion pipeline.
    """

    NAMES = ['scale']

    # hide inherited arguments
    # that are device related
    HIDE_ARGS = ['device', 'model-offload']

    def __init__(self,
                 factor: float = 1.0,
                 clamp_min: float | None = None,
                 clamp_max: float | None = None,
                 normalize: bool = False,
                 **kwargs):
        """
        :param factor: Scaling factor to multiply tensor values. Must be positive
        :param clamp_min: Minimum value to clamp tensor values to (optional)
        :param clamp_max: Maximum value to clamp tensor values to (optional)
        :param normalize: Whether to normalize tensor to unit variance after scaling
        :param kwargs: Additional arguments passed to base class
        """
        super().__init__(**kwargs)

        if factor <= 0:
            raise self.argument_error(
                f"scale must be positive, got {factor}")

        if clamp_min is not None and clamp_max is not None and clamp_min >= clamp_max:
            raise self.argument_error(
                f"clamp_min ({clamp_min}) must be less than clamp_max ({clamp_max})")

        self._factor = factor
        self._clamp_min = clamp_min
        self._clamp_max = clamp_max
        self._normalize = normalize

    def impl_process(self,
                     pipeline,
                     latents: torch.Tensor) -> torch.Tensor:
        """
        Scale and process the latents tensor.
        
        :param pipeline: The pipeline object
        :param latents: Input latents tensor with shape [B, C, H, W]
        :return: Scaled and processed latents tensor, shape [B, C, H, W]
        """

        result = latents.clone()

        # Apply scaling
        if self._factor != 1.0:
            result = result * self._factor

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
