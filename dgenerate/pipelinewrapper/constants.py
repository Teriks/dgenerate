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

import dgenerate.globalconfig


LATENTS_PROCESSOR_SEP: str = '+'
"""
Used to separate unique latents processor chains 
when running processors on batched latents input.
"""

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
Default noise level for floyd super resolution upscalers when performing img2img.
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

DEFAULT_INPAINT_CROP_PADDING: int = 32
"""
Default padding in pixels for inpaint crop operations.
Applied on all sides around the mask bounds.
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

DEFAULT_ADETAILER_DETECTOR_CONFIDENCE: float = 0.3
"""
Default detector confidence
"""

DEFAULT_ADETAILER_MASK_DILATION: int = 4
"""
Default adetailer mask dilation
"""

DEFAULT_ADETAILER_MODEL_MASKS: bool = False
"""
Default adetailer model masks setting.
"""

DEFAULT_ADETAILER_MASK_BLUR: int = 4
"""
Default adetailer mask blur.
"""

DEFAULT_YOLO_DETECTOR_PADDING: int = 0
"""
Default YOLO detector padding.
"""

DEFAULT_YOLO_MASK_SHAPE: str = 'rectangle'
"""
Default YOLO mask shape.
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


DEFAULT_TEA_CACHE_REL_L1_THRESHOLD: float = 0.6
"""
Default relative L1 threshold for TeaCache (Timestep Embedding Aware Cache) for Flux.
"""

DEFAULT_RAS_SAMPLE_RATIO: float = 0.5
"""
Default sample ratio for RAS (Reinforcement Attention System) for Stable Diffusion 3.
Controls the average sample ratio for each RAS step, must be between 0 and 1.
"""

DEFAULT_RAS_HIGH_RATIO: float = 1.0
"""
Default high ratio for RAS (Reinforcement Attention System) for Stable Diffusion 3.
Controls the ratio of high-value tokens to be cached in RAS, must be between 0 and 1.
"""

DEFAULT_RAS_STARVATION_SCALE: float = 0.1
"""
Default starvation scale for RAS (Reinforcement Attention System) for Stable Diffusion 3.
Controls the starvation scale in RAS patch selection, typically between 0 and 1.
"""

DEFAULT_RAS_ERROR_RESET_STEPS: list = [12,22]
"""
Default error reset steps for RAS (Reinforcement Attention System) for Stable Diffusion 3.
Comma-separated string of step numbers for dense sampling steps to reset accumulated error in RAS.
"""

DEFAULT_RAS_METRIC: str = "std"
"""
Default RAS metric for RAS (Reinforcement Attention System) for Stable Diffusion 3.
"""

DEFAULT_RAS_START_STEP: int = 4
"""
Default starting step for RAS (Reinforcement Attention System) for Stable Diffusion 3.
Controls when RAS begins applying its sampling strategy.
"""

DEFAULT_RAS_SKIP_NUM_STEP: int = 0
"""
Default skip num step for RAS (Reinforcement Attention System) for Stable Diffusion 3.
Controls the number of steps to skip between RAS steps. The actual number of tokens skipped
will be rounded down to the nearest multiple of 64 to ensure efficient memory access patterns
for attention computation. When used with skip_num_step_length greater than 0, this value
determines how the number of skipped tokens changes over time.
"""

DEFAULT_RAS_SKIP_NUM_STEP_LENGTH: int = 0
"""
Default skip num step length for RAS (Reinforcement Attention System) for Stable Diffusion 3.
Controls the length of steps to skip between RAS steps. When set to 0, static dropping is used
where the number of skipped tokens remains constant. When greater than 0, dynamic dropping is
enabled where the number of skipped tokens varies over time based on skip_num_step. The pattern
of skipping will repeat every skip_num_step_length steps.
"""

DEFAULT_DEEP_CACHE_INTERVAL: int = 5
"""
Default cache interval for DeepCache.
Controls how frequently the attention layers are cached during the diffusion process.
"""

DEFAULT_DEEP_CACHE_BRANCH_ID: int = 1
"""
Default branch ID for DeepCache.
Controls which branches to apply DeepCache to in the UNet.
"""

DEFAULT_SDXL_REFINER_DEEP_CACHE_INTERVAL: int = 5
"""
Default cache interval for DeepCache on SDXL Refiner.
Controls how frequently the attention layers are cached during the diffusion process.
"""

DEFAULT_SDXL_REFINER_DEEP_CACHE_BRANCH_ID: int = 1
"""
Default branch ID for DeepCache on SDXL Refiner.
Controls which branches to apply DeepCache to in the UNet.
"""

DEFAULT_SADA_SD_MAX_DOWNSAMPLE: int = 1
"""
Maximum downsample factor for SD models.
Controls the maximum downsample factor in the SADA algorithm.
Lower values can improve quality but may reduce speedup.
"""

DEFAULT_SADA_SD_SX: int = 3
"""
Spatial downsample factor X for SD models.
Controls the spatial downsample factor in the X dimension.
Higher values can increase speedup but may affect quality.
"""

DEFAULT_SADA_SD_SY: int = 3
"""
Spatial downsample factor Y for SD models.
Controls the spatial downsample factor in the Y dimension.
Higher values can increase speedup but may affect quality.
"""

DEFAULT_SADA_SD_LAGRANGE_TERM: int = 4
"""
Lagrangian interpolation terms for SD models.
Number of terms to use in Lagrangian interpolation.
Set to 0 to disable Lagrangian interpolation.
"""

DEFAULT_SADA_SD_LAGRANGE_INT: int = 4
"""
Lagrangian interpolation interval for SD models.
Interval for Lagrangian interpolation. Must be compatible with lagrange_step.
"""

DEFAULT_SADA_SD_LAGRANGE_STEP: int = 24
"""
Lagrangian interpolation step for SD models.
Step value for Lagrangian interpolation. Must be compatible with lagrange_int.
"""

DEFAULT_SADA_SD_MAX_FIX: int = 5 * 1024
"""
Maximum fixed memory for SD models.
Maximum amount of fixed memory to use in SADA optimization.
"""

DEFAULT_SADA_SDXL_MAX_DOWNSAMPLE: int = 2
"""
Maximum downsample factor for SDXL/Kolors models.
Controls the maximum downsample factor in the SADA algorithm.
Higher than SD defaults for better speedup on larger models.
"""

DEFAULT_SADA_SDXL_SX: int = 3
"""
Spatial downsample factor X for SDXL/Kolors models.
Controls the spatial downsample factor in the X dimension.
"""

DEFAULT_SADA_SDXL_SY: int = 3
"""
Spatial downsample factor Y for SDXL/Kolors models.
Controls the spatial downsample factor in the Y dimension.
"""

DEFAULT_SADA_SDXL_LAGRANGE_TERM: int = 4
"""
Lagrangian interpolation terms for SDXL/Kolors models.
Number of terms to use in Lagrangian interpolation.
"""

DEFAULT_SADA_SDXL_LAGRANGE_INT: int = 4
"""
Lagrangian interpolation interval for SDXL/Kolors models.
Interval for Lagrangian interpolation. Must be compatible with lagrange_step.
"""

DEFAULT_SADA_SDXL_LAGRANGE_STEP: int = 24
"""
Lagrangian interpolation step for SDXL/Kolors models.
Step value for Lagrangian interpolation. Must be compatible with lagrange_int.
"""

DEFAULT_SADA_SDXL_MAX_FIX: int = 10 * 1024
"""
Maximum fixed memory for SDXL/Kolors models.
Maximum amount of fixed memory to use in SADA optimization.
Higher than SD defaults due to larger model size.
"""

# Flux defaults (from flux_demo.py)
DEFAULT_SADA_FLUX_MAX_DOWNSAMPLE: int = 0
"""
Maximum downsample factor for Flux models.
Set to 0 as Flux uses a different architecture that doesn't use spatial downsampling.
"""

DEFAULT_SADA_FLUX_SX: int = 0
"""
Spatial downsample factor X for Flux models.
Not used for Flux architecture, set to 0.
"""

DEFAULT_SADA_FLUX_SY: int = 0
"""
Spatial downsample factor Y for Flux models.
Not used for Flux architecture, set to 0.
"""

DEFAULT_SADA_FLUX_LAGRANGE_TERM: int = 3
"""
Lagrangian interpolation terms for Flux models.
Number of terms to use in Lagrangian interpolation.
Lower than SD/SDXL defaults for Flux architecture.
"""

DEFAULT_SADA_FLUX_LAGRANGE_INT: int = 4
"""
Lagrangian interpolation interval for Flux models.
Interval for Lagrangian interpolation. Must be compatible with lagrange_step.
"""

DEFAULT_SADA_FLUX_LAGRANGE_STEP: int = 20
"""
Lagrangian interpolation step for Flux models.
Step value for Lagrangian interpolation. Must be compatible with lagrange_int.
Lower than SD/SDXL defaults for Flux architecture.
"""

DEFAULT_SADA_FLUX_MAX_FIX: int = 0
"""
Maximum fixed memory for Flux models.
Set to 0 as Flux uses a different optimization approach.
"""

DEFAULT_SADA_ACC_RANGE: tuple = (10, 47)
"""
Acceleration range start / end step for all models.
Defines the starting step for SADA acceleration. 

Start step must be at least 3 as SADA leverages third-order
dynamics.
"""

DEFAULT_SADA_MAX_INTERVAL: int = 4
"""
Maximum interval for optimization for all models.
Maximum interval between optimizations in the SADA algorithm.
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

dgenerate.globalconfig.register_all()
