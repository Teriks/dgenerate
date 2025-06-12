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
Noise injection latents processor implementation.
"""


class NoiseProcessor(_latentsprocessor.LatentsProcessor):
    """
    Injects noise into latents tensors using the pipeline's scheduler.

    This processor uses the scheduler's add_noise() method to properly inject noise
    according to the diffusion model's noise schedule, making it suitable for
    denoising experiments and creative latent space manipulation. The noise injection
    respects the training schedule of the diffusion models, ensuring that the added
    noise follows the same distribution and scaling as during training.

    The "timestep" argument controls the amount of noise added to the latents.
    Higher timestep values result in more noise, lower values result in less noise,
    following the diffusion model's training schedule. This provides direct control
    over the noise level without additional scaling factors.

    If "seed" is provided, the random number generator will be seeded for
    reproducible noise generation across multiple runs.
    """

    NAMES = ['noise']

    # hide inherited arguments
    # that are device related
    HIDE_ARGS = ['device', 'model-offload']

    def __init__(self,
                 timestep: int = 100,
                 seed: int | None = None,
                 **kwargs):
        """
        :param timestep: Timestep for noise injection. Higher values add more noise
        :param seed: Random seed for reproducible noise generation (optional)
        :param kwargs: Additional arguments passed to base class
        """
        super().__init__(**kwargs)

        if timestep < 0:
            raise self.argument_error(
                f"timestep must be non-negative, got {timestep}")

        self._timestep = timestep
        self._seed = seed

    def impl_process(self,
                     pipeline,
                     latents: torch.Tensor) -> torch.Tensor:
        """
        Inject noise into the latents tensor using the pipeline's scheduler.
        
        :param pipeline: The pipeline object
        :param latents: Input latents tensor with shape [B, C, H, W]
        :return: Latents tensor with scheduler-based noise injection, shape [B, C, H, W]
        """

        # Create generator for reproducible noise if seed is provided
        generator = None
        if self._seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self._seed)

        # Validate timestep against scheduler limits
        if hasattr(pipeline.scheduler, 'num_train_timesteps'):
            max_timestep = pipeline.scheduler.num_train_timesteps - 1
            if self._timestep > max_timestep:
                raise self.argument_error(
                    f"Timestep {self._timestep} exceeds scheduler's maximum timestep {max_timestep}")

        # Create timestep tensor
        timesteps = torch.tensor([self._timestep], dtype=torch.long)

        # Generate noise using the same distribution as the scheduler
        noise = torch.randn(latents.shape, generator=generator, dtype=latents.dtype)

        # Use scheduler to add noise properly
        if hasattr(pipeline.scheduler, 'add_noise'):
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
        else:
            # Fallback for schedulers without add_noise method
            # This shouldn't happen with modern schedulers, but just in case
            # Use a simple linear interpolation based on timestep
            if hasattr(pipeline.scheduler, 'num_train_timesteps'):
                max_timestep = pipeline.scheduler.num_train_timesteps - 1
                alpha_t = 1.0 - (self._timestep / max_timestep)
            else:
                # Conservative fallback
                alpha_t = 1.0 - (self._timestep / 1000.0)
            noisy_latents = alpha_t * latents + (1.0 - alpha_t) * noise

        return noisy_latents
