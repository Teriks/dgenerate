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

DEFAULT_INFERENCE_STEPS: int = 30
"""
Default value for inference steps.
"""

DEFAULT_GUIDANCE_SCALE: float = 5.0
"""
Default value for guidance scale.
"""

DEFAULT_IMAGE_SEED_STRENGTH: float = 0.8
"""
Default image seed strength for img2img.
"""

DEFAULT_IMAGE_GUIDANCE_SCALE: float = 1.5
"""
Default image guidance scale for pix2pix.
"""

DEFAULT_SDXL_HIGH_NOISE_FRACTION: float = 0.8
"""
Default SDXL high noise fraction.
"""

DEFAULT_X4_UPSCALER_NOISE_LEVEL: int = 20
"""
Default x4 upscaler noise level.
"""

DEFAULT_FLOYD_SUPERRESOLUTION_NOISE_LEVEL: int = 250
"""
Default noise level for floyd super resolution upscalers.
"""

DEFAULT_FLOYD_SUPERRESOLUTION_IMG2IMG_NOISE_LEVEL: int = 250
"""
Default noise level for floyd super resolution upscalers when preforming img2img.
"""

DEFAULT_FLOYD_SUPERRESOLUTION_INPAINT_NOISE_LEVEL: int = 0
"""
Default noise level for floyd super resolution upscalers when inpainting.
"""

DEFAULT_OUTPUT_WIDTH: int = 512
"""
Default output width for txt2img.
"""

DEFAULT_OUTPUT_HEIGHT: int = 512
"""
Default output height for txt2img.
"""

DEFAULT_SDXL_OUTPUT_WIDTH: int = 1024
"""
Default output width for SDXL txt2img.
"""

DEFAULT_SDXL_OUTPUT_HEIGHT: int = 1024
"""
Default output height for SDXL txt2img.
"""

DEFAULT_KOLORS_OUTPUT_WIDTH: int = 1024
"""
Default output width for Kolors txt2img.
"""

DEFAULT_KOLORS_OUTPUT_HEIGHT: int = 1024
"""
Default output height for Kolors txt2img.
"""

DEFAULT_FLOYD_IF_OUTPUT_WIDTH: int = 64
"""
Default output width for Deep Floyd IF txt2img first stage.
"""

DEFAULT_FLOYD_IF_OUTPUT_HEIGHT: int = 64
"""
Default output height for Deep Floyd IF txt2img first stage.
"""

DEFAULT_SEED: int = 0
"""
Default RNG seed.
"""

DEFAULT_S_CASCADE_DECODER_GUIDANCE_SCALE: float = 0
"""
Default guidance scale for the Stable Cascade decoder.
"""

DEFAULT_S_CASCADE_DECODER_INFERENCE_STEPS: int = 10
"""
Default inference steps for the Stable Cascade decoder.
"""

DEFAULT_S_CASCADE_OUTPUT_HEIGHT: int = 1024
"""
Default output height for Stable Cascade.
"""

DEFAULT_S_CASCADE_OUTPUT_WIDTH: int = 1024
"""
Default output width for Stable Cascade.
"""

DEFAULT_SD3_OUTPUT_HEIGHT: int = 1024
"""
Default output height for Stable Diffusion 3.
"""

DEFAULT_SD3_OUTPUT_WIDTH: int = 1024
"""
Default output width for Stable Diffusion 3.
"""

DEFAULT_FLUX_OUTPUT_HEIGHT: int = 1024
"""
Default output height for Flux.
"""

DEFAULT_FLUX_OUTPUT_WIDTH: int = 1024
"""
Default output width for Flux.
"""

DEFAULT_ADETAILER_MASK_SHAPE: str = 'rectangle'
"""
Default detector mask shape
"""

DEFAULT_ADETAILER_MASK_PADDING: int = 32
"""
Default adetailer mask padding
"""

DEFAULT_ADETAILER_DETECTOR_PADDING: int = 0
"""
Default detector padding
"""

DEFAULT_ADETAILER_MASK_DILATION: int = 4
"""
Default adetailer mask dilation
"""

DEFAULT_ADETAILER_MASK_BLUR: int = 4
"""
Default adetailer mask blur.
"""

DEFAULT_PAG_SCALE: float = 3.0
"""
Default pag scale
"""

DEFAULT_PAG_ADAPTIVE_SCALE: float = 0.0
"""
Default pag adaptive scale
"""

DEFAULT_SDXL_REFINER_PAG_SCALE: float = 3.0
"""
Default sdxl refiner pag scale
"""

DEFAULT_SDXL_REFINER_PAG_ADAPTIVE_SCALE: float = 0.0
"""
Default sdxl refiner pag adaptive scale
"""

PIPELINE_WRAPPER_CACHE_GC_CONSTRAINTS: list[str] = ['used_percent > 70']
"""
Cache constraint expressions for when to clear all object caches, 
before calling the diffusion pipeline wrapper.

Syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, all cached objects will be garbage collected
before a generation starts.
"""

PIPELINE_CACHE_MEMORY_CONSTRAINTS: list[str] = ['pipeline_size > (available * 0.75)']
"""
Cache constraint expressions for when to clear the CPU side 
diffusion pipeline cache upon a new diffusion pipeline being created, 
syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, all cached diffusion pipeline objects will be garbage collected.

Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
and ``pipeline_size`` (the estimated size of the new pipeline before it is brought into memory, in bytes)
"""

UNET_CACHE_MEMORY_CONSTRAINTS: list[str] = ['unet_size > (available * 0.75)']
"""
Cache constraint expressions for when to clear the CPU side 
unet model cache upon a new unet model being created, 
syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, all cached unet objects will be garbage collected.

Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
and ``unet_size`` (the estimated size of the new UNet before it is brought into memory, in bytes)
"""

VAE_CACHE_MEMORY_CONSTRAINTS: list[str] = ['vae_size > (available * 0.75)']
"""
Cache constraint expressions for when to clear the CPU side 
vae model cache upon a new vae model being created, 
syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, all cached vae objects will be garbage collected.

Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
and ``vae_size`` (the estimated size of the new VAE before it is brought into memory, in bytes)
"""

CONTROLNET_CACHE_MEMORY_CONSTRAINTS: list[str] = ['controlnet_size > (available * 0.75)']
"""
Cache constraint expressions for when to clear the CPU side 
controlnet model cache upon a new controlnet model being created, 
syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, all cached controlnet objects will be garbage collected.

Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
and ``controlnet_size`` (the estimated size of the new ControlNet before it is brought into memory, in bytes)
"""

ADAPTER_CACHE_MEMORY_CONSTRAINTS: list[str] = ['adapter_size > (available * 0.75)']
"""
Cache constraint expressions for when to clear the CPU side 
adapter model cache upon a new adapter model being created, 
syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, all cached adapter objects will be garbage collected.

Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
and ``adapter_size`` (the estimated size of the new T2IAdapter before it is brought into memory, in bytes)
"""

TEXT_ENCODER_CACHE_MEMORY_CONSTRAINTS: list[str] = ['text_encoder_size > (available * 0.75)']
"""
Cache constraint expressions for when to clear the CPU side 
text encoder model cache upon a new text encoder model being created, 
syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, all cached text encoder objects will be garbage collected.

Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
and ``text_encoder_size`` (the estimated size of the new Text Encoder before it is brought into memory, in bytes)
"""

IMAGE_ENCODER_CACHE_MEMORY_CONSTRAINTS: list[str] = ['image_encoder_size > (available * 0.75)']
"""
Cache constraint expressions for when to clear the CPU side 
image encoder model cache upon a new image encoder model being created, 
syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, all cached image encoder objects will be garbage collected.

Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
and ``image_encoder_size`` (the estimated size of the new Image Encoder before it is brought into memory, in bytes)
"""

TRANSFORMER_CACHE_MEMORY_CONSTRAINTS: list[str] = ['transformer_size > (available * 0.75)']
"""
Cache constraint expressions for when to clear the CPU side 
transformer model cache upon a new transformer model being created, 
syntax provided via :py:func:`dgenerate.memory.memory_constraints`

If any of these constraints are met, all cached transformer objects will be garbage collected.

Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
and ``transformer_size`` (the estimated size of the new transformer model before it is brought into memory, in bytes)
"""

