.. py:currentmodule:: dgenerate.pipelinewrapper.constants

.. data:: LATENTS_PROCESSOR_SEP
    :annotation: = '+'

    Used to separate unique latents processor chains 
    when running processors on batched latents input.

.. data:: DEFAULT_FALLBACK_INFERENCE_STEPS
    :annotation: = 30

    Fallback step count when ``--inference-steps`` is omitted and no model rule
    applies, or when a pipeline ``__call__`` has no ``num_inference_steps`` default.

    Also used in LTX distilled step warnings for message wording only.

.. data:: DEFAULT_FALLBACK_GUIDANCE_SCALE
    :annotation: = 5.0

    Fallback guidance when ``--guidance-scales`` is omitted and no model rule
    applies, or when a pipeline has no ``guidance_scale`` default (Wan, legacy
    LTX-Video, image wrapper).

.. data:: DEFAULT_IMAGE_SEED_STRENGTH
    :annotation: = 0.8

    Default image seed strength for img2img.

.. data:: DEFAULT_IMAGE_GUIDANCE_SCALE
    :annotation: = 1.5

    Default image guidance scale for pix2pix.

.. data:: DEFAULT_SDXL_HIGH_NOISE_FRACTION
    :annotation: = 0.8

    Default SDXL high noise fraction.

.. data:: DEFAULT_X4_UPSCALER_NOISE_LEVEL
    :annotation: = 20

    Default x4 upscaler noise level.

.. data:: DEFAULT_FLOYD_SUPERRESOLUTION_NOISE_LEVEL
    :annotation: = 250

    Default noise level for floyd super resolution upscalers.

.. data:: DEFAULT_FLOYD_SUPERRESOLUTION_IMG2IMG_NOISE_LEVEL
    :annotation: = 250

    Default noise level for floyd super resolution upscalers when performing img2img.

.. data:: DEFAULT_FLOYD_SUPERRESOLUTION_INPAINT_NOISE_LEVEL
    :annotation: = 0

    Default noise level for floyd super resolution upscalers when inpainting.

.. data:: DEFAULT_OUTPUT_WIDTH
    :annotation: = 512

    Default output width for txt2img.

.. data:: DEFAULT_OUTPUT_HEIGHT
    :annotation: = 512

    Default output height for txt2img.

.. data:: DEFAULT_SDXL_OUTPUT_WIDTH
    :annotation: = 1024

    Default output width for SDXL txt2img.

.. data:: DEFAULT_SDXL_OUTPUT_HEIGHT
    :annotation: = 1024

    Default output height for SDXL txt2img.

.. data:: DEFAULT_KOLORS_OUTPUT_WIDTH
    :annotation: = 1024

    Default output width for Kolors txt2img.

.. data:: DEFAULT_KOLORS_OUTPUT_HEIGHT
    :annotation: = 1024

    Default output height for Kolors txt2img.

.. data:: DEFAULT_HYDIT_OUTPUT_WIDTH
    :annotation: = 1024

    Default output width for Hunyuan-DiT txt2img.

.. data:: DEFAULT_HYDIT_OUTPUT_HEIGHT
    :annotation: = 1024

    Default output height for Hunyuan-DiT txt2img.

.. data:: DEFAULT_HYIMAGE21_OUTPUT_WIDTH
    :annotation: = 2048

    Default output width for HunyuanImage 2.1 txt2img.

.. data:: DEFAULT_HYIMAGE21_OUTPUT_HEIGHT
    :annotation: = 2048

    Default output height for HunyuanImage 2.1 txt2img.

.. data:: DEFAULT_FLOYD_IF_OUTPUT_WIDTH
    :annotation: = 64

    Default output width for Deep Floyd IF txt2img first stage.

.. data:: DEFAULT_FLOYD_IF_OUTPUT_HEIGHT
    :annotation: = 64

    Default output height for Deep Floyd IF txt2img first stage.

.. data:: DEFAULT_SEED
    :annotation: = 0

    Default RNG seed.

.. data:: DEFAULT_INPAINT_CROP_PADDING
    :annotation: = 32

    Default padding in pixels for inpaint crop operations.
    Applied on all sides around the mask bounds.

.. data:: DEFAULT_S_CASCADE_DECODER_GUIDANCE_SCALE
    :annotation: = 0

    Default guidance scale for the Stable Cascade decoder.

.. data:: DEFAULT_S_CASCADE_DECODER_INFERENCE_STEPS
    :annotation: = 10

    Default inference steps for the Stable Cascade decoder.

.. data:: DEFAULT_S_CASCADE_OUTPUT_HEIGHT
    :annotation: = 1024

    Default output height for Stable Cascade.

.. data:: DEFAULT_S_CASCADE_OUTPUT_WIDTH
    :annotation: = 1024

    Default output width for Stable Cascade.

.. data:: DEFAULT_SD3_OUTPUT_HEIGHT
    :annotation: = 1024

    Default output height for Stable Diffusion 3.

.. data:: DEFAULT_SD3_OUTPUT_WIDTH
    :annotation: = 1024

    Default output width for Stable Diffusion 3.

.. data:: DEFAULT_FLUX_OUTPUT_HEIGHT
    :annotation: = 1024

    Default output height for Flux.

.. data:: DEFAULT_FLUX_OUTPUT_WIDTH
    :annotation: = 1024

    Default output width for Flux.

.. data:: DEFAULT_ADETAILER_MASK_SHAPE
    :annotation: = 'rectangle'

    Default detector mask shape

.. data:: DEFAULT_ADETAILER_MASK_PADDING
    :annotation: = 32

    Default adetailer mask padding

.. data:: DEFAULT_ADETAILER_DETECTOR_PADDING
    :annotation: = 0

    Default detector padding

.. data:: DEFAULT_ADETAILER_DETECTOR_CONFIDENCE
    :annotation: = 0.3

    Default detector confidence

.. data:: DEFAULT_ADETAILER_MASK_DILATION
    :annotation: = 4

    Default adetailer mask dilation

.. data:: DEFAULT_ADETAILER_MODEL_MASKS
    :annotation: = False

    Default adetailer model masks setting.

.. data:: DEFAULT_ADETAILER_MASK_BLUR
    :annotation: = 4

    Default adetailer mask blur.

.. data:: DEFAULT_YOLO_DETECTOR_PADDING
    :annotation: = 0

    Default YOLO detector padding.

.. data:: DEFAULT_YOLO_MASK_SHAPE
    :annotation: = 'rectangle'

    Default YOLO mask shape.

.. data:: DEFAULT_PAG_SCALE
    :annotation: = 3.0

    Default pag scale

.. data:: DEFAULT_PAG_ADAPTIVE_SCALE
    :annotation: = 0.0

    Default pag adaptive scale

.. data:: DEFAULT_SDXL_REFINER_PAG_SCALE
    :annotation: = 3.0

    Default sdxl refiner pag scale

.. data:: DEFAULT_SDXL_REFINER_PAG_ADAPTIVE_SCALE
    :annotation: = 0.0

    Default sdxl refiner pag adaptive scale

.. data:: DEFAULT_TEA_CACHE_REL_L1_THRESHOLD
    :annotation: = 0.6

    Default relative L1 threshold for TeaCache (Timestep Embedding Aware Cache) for Flux.

.. data:: DEFAULT_REGIONE_WARMUP_STEP
    :annotation: = 6

    Default RegionE warmup steps (Flux Kontext preset).

.. data:: DEFAULT_REGIONE_POST_STEP
    :annotation: = 2

    Default RegionE post steps (Flux Kontext preset).

.. data:: DEFAULT_REGIONE_REFRESH_STEP
    :annotation: = 16

    Default RegionE mask refresh step (Flux Kontext preset).

.. data:: DEFAULT_REGIONE_THRESHOLD
    :annotation: = 0.93

    Default RegionE region partition threshold (Flux Kontext preset).

.. data:: DEFAULT_REGIONE_CACHE_THRESHOLD
    :annotation: = 0.04

    Default RegionE velocity-decay cache threshold (Flux Kontext preset).

.. data:: DEFAULT_REGIONE_EROSION_DILATION
    :annotation: = True

    Default RegionE mask erosion/dilation flag.

.. data:: DEFAULT_SPECTRUM_WEIGHT
    :annotation: = 0.5

    Default Spectrum blend weight. ``1`` is the Chebyshev forecast and ``0`` is a
    one-step Taylor step. ``0.5`` is the robust setting from the project README.

.. data:: DEFAULT_SPECTRUM_ORDER
    :annotation: = 4

    Default number of Chebyshev polynomials used by Spectrum, not counting ``T_0``.

.. data:: DEFAULT_SPECTRUM_LAMBDA
    :annotation: = 0.1

    Default ridge penalty for the Spectrum Chebyshev fit.

.. data:: DEFAULT_SPECTRUM_WARMUP_STEPS
    :annotation: = 2

    Default number of leading denoiser passes Spectrum always runs.

    The paper's 50-step setup uses 5. Two passes is enough to start a forecast on
    the shorter schedules used here. Raise it when a 50-step sample drifts.

.. data:: DEFAULT_SPECTRUM_WINDOW_SIZE
    :annotation: = 2.0

    Default initial Spectrum skip interval. ``2`` predicts every other step after warmup.

.. data:: DEFAULT_SPECTRUM_FLEX_WINDOW
    :annotation: = 0.75

    Default amount added to the Spectrum skip interval after each full pass.

    ``0.75`` is the paper's moderate schedule. ``0`` holds the interval fixed.
    ``3`` is the paper's aggressive schedule.

.. data:: DEFAULT_DYPE_METHOD
    :annotation: = 'yarn'

    Default position encoding method used by the DyPE adapter.

.. data:: DYPE_METHODS
    :annotation: = ('yarn', 'ntk', 'base')

    Position encoding methods exposed by the DyPE adapter.

.. data:: DEFAULT_DYPE_MSCALE
    :annotation: = 'constant'

    Default YARN attention magnitude scale mode used by the DyPE adapter.

.. data:: DYPE_MSCALES
    :annotation: = ('constant', 'decay', 'off')

    YARN attention magnitude scale modes exposed by the DyPE adapter.

.. data:: DEFAULT_DYPE_AXIS_SCALING
    :annotation: = 'shared'

    Default spatial axis scaling mode used by the DyPE adapter.

.. data:: DYPE_AXIS_SCALINGS
    :annotation: = ('independent', 'shared')

    Spatial axis scaling modes exposed by the DyPE adapter.

.. data:: DEFAULT_RAS_SAMPLE_RATIO
    :annotation: = 0.5

    Default sample ratio for RAS (Reinforcement Attention System) for Stable Diffusion 3.
    Controls the average sample ratio for each RAS step, must be between 0 and 1.

.. data:: DEFAULT_RAS_HIGH_RATIO
    :annotation: = 1.0

    Default high ratio for RAS (Reinforcement Attention System) for Stable Diffusion 3.
    Controls the ratio of high-value tokens to be cached in RAS, must be between 0 and 1.

.. data:: DEFAULT_RAS_STARVATION_SCALE
    :annotation: = 0.1

    Default starvation scale for RAS (Reinforcement Attention System) for Stable Diffusion 3.
    Controls the starvation scale in RAS patch selection, typically between 0 and 1.

.. data:: DEFAULT_RAS_ERROR_RESET_STEPS
    :annotation: = [12, 22]

    Default error reset steps for RAS (Reinforcement Attention System) for Stable Diffusion 3.
    Comma-separated string of step numbers for dense sampling steps to reset accumulated error in RAS.

.. data:: DEFAULT_RAS_METRIC
    :annotation: = 'std'

    Default RAS metric for RAS (Reinforcement Attention System) for Stable Diffusion 3.

.. data:: DEFAULT_RAS_START_STEP
    :annotation: = 4

    Default starting step for RAS (Reinforcement Attention System) for Stable Diffusion 3.
    Controls when RAS begins applying its sampling strategy.

.. data:: DEFAULT_RAS_SKIP_NUM_STEP
    :annotation: = 0

    Default skip num step for RAS (Reinforcement Attention System) for Stable Diffusion 3.
    Controls the number of steps to skip between RAS steps. The actual number of tokens skipped
    will be rounded down to the nearest multiple of 64 to ensure efficient memory access patterns
    for attention computation. When used with skip_num_step_length greater than 0, this value
    determines how the number of skipped tokens changes over time.

.. data:: DEFAULT_RAS_SKIP_NUM_STEP_LENGTH
    :annotation: = 0

    Default skip num step length for RAS (Reinforcement Attention System) for Stable Diffusion 3.
    Controls the length of steps to skip between RAS steps. When set to 0, static dropping is used
    where the number of skipped tokens remains constant. When greater than 0, dynamic dropping is
    enabled where the number of skipped tokens varies over time based on skip_num_step. The pattern
    of skipping will repeat every skip_num_step_length steps.

.. data:: DEFAULT_DEEP_CACHE_INTERVAL
    :annotation: = 5

    Default cache interval for DeepCache.
    Controls how frequently the attention layers are cached during the diffusion process.

.. data:: DEFAULT_DEEP_CACHE_BRANCH_ID
    :annotation: = 1

    Default branch ID for DeepCache.
    Controls which branches to apply DeepCache to in the UNet.

.. data:: DEFAULT_SDXL_REFINER_DEEP_CACHE_INTERVAL
    :annotation: = 5

    Default cache interval for DeepCache on SDXL Refiner.
    Controls how frequently the attention layers are cached during the diffusion process.

.. data:: DEFAULT_SDXL_REFINER_DEEP_CACHE_BRANCH_ID
    :annotation: = 1

    Default branch ID for DeepCache on SDXL Refiner.
    Controls which branches to apply DeepCache to in the UNet.

.. data:: DEFAULT_SADA_SD_MAX_DOWNSAMPLE
    :annotation: = 1

    Maximum downsample factor for SD models.
    Controls the maximum downsample factor in the SADA algorithm.
    Lower values can improve quality but may reduce speedup.

.. data:: DEFAULT_SADA_SD_SX
    :annotation: = 3

    Spatial downsample factor X for SD models.
    Controls the spatial downsample factor in the X dimension.
    Higher values can increase speedup but may affect quality.

.. data:: DEFAULT_SADA_SD_SY
    :annotation: = 3

    Spatial downsample factor Y for SD models.
    Controls the spatial downsample factor in the Y dimension.
    Higher values can increase speedup but may affect quality.

.. data:: DEFAULT_SADA_SD_LAGRANGE_TERM
    :annotation: = 4

    Lagrangian interpolation terms for SD models.
    Number of terms to use in Lagrangian interpolation.
    Set to 0 to disable Lagrangian interpolation.

.. data:: DEFAULT_SADA_SD_LAGRANGE_INT
    :annotation: = 4

    Lagrangian interpolation interval for SD models.
    Interval for Lagrangian interpolation. Must be compatible with lagrange_step.

.. data:: DEFAULT_SADA_SD_LAGRANGE_STEP
    :annotation: = 24

    Lagrangian interpolation step for SD models.
    Step value for Lagrangian interpolation. Must be compatible with lagrange_int.

.. data:: DEFAULT_SADA_SD_MAX_FIX
    :annotation: = 5 * 1024

    Maximum fixed memory for SD models.
    Maximum amount of fixed memory to use in SADA optimization.

.. data:: DEFAULT_SADA_SDXL_MAX_DOWNSAMPLE
    :annotation: = 2

    Maximum downsample factor for SDXL/Kolors models.
    Controls the maximum downsample factor in the SADA algorithm.
    Higher than SD defaults for better speedup on larger models.

.. data:: DEFAULT_SADA_SDXL_SX
    :annotation: = 3

    Spatial downsample factor X for SDXL/Kolors models.
    Controls the spatial downsample factor in the X dimension.

.. data:: DEFAULT_SADA_SDXL_SY
    :annotation: = 3

    Spatial downsample factor Y for SDXL/Kolors models.
    Controls the spatial downsample factor in the Y dimension.

.. data:: DEFAULT_SADA_SDXL_LAGRANGE_TERM
    :annotation: = 4

    Lagrangian interpolation terms for SDXL/Kolors models.
    Number of terms to use in Lagrangian interpolation.

.. data:: DEFAULT_SADA_SDXL_LAGRANGE_INT
    :annotation: = 4

    Lagrangian interpolation interval for SDXL/Kolors models.
    Interval for Lagrangian interpolation. Must be compatible with lagrange_step.

.. data:: DEFAULT_SADA_SDXL_LAGRANGE_STEP
    :annotation: = 24

    Lagrangian interpolation step for SDXL/Kolors models.
    Step value for Lagrangian interpolation. Must be compatible with lagrange_int.

.. data:: DEFAULT_SADA_SDXL_MAX_FIX
    :annotation: = 10 * 1024

    Maximum fixed memory for SDXL/Kolors models.
    Maximum amount of fixed memory to use in SADA optimization.
    Higher than SD defaults due to larger model size.

.. data:: DEFAULT_SADA_FLUX_MAX_DOWNSAMPLE
    :annotation: = 0

    Maximum downsample factor for Flux models.
    Set to 0 as Flux uses a different architecture that doesn't use spatial downsampling.

.. data:: DEFAULT_SADA_FLUX_SX
    :annotation: = 0

    Spatial downsample factor X for Flux models.
    Not used for Flux architecture, set to 0.

.. data:: DEFAULT_SADA_FLUX_SY
    :annotation: = 0

    Spatial downsample factor Y for Flux models.
    Not used for Flux architecture, set to 0.

.. data:: DEFAULT_SADA_FLUX_LAGRANGE_TERM
    :annotation: = 3

    Lagrangian interpolation terms for Flux models.
    Number of terms to use in Lagrangian interpolation.
    Lower than SD/SDXL defaults for Flux architecture.

.. data:: DEFAULT_SADA_FLUX_LAGRANGE_INT
    :annotation: = 4

    Lagrangian interpolation interval for Flux models.
    Interval for Lagrangian interpolation. Must be compatible with lagrange_step.

.. data:: DEFAULT_SADA_FLUX_LAGRANGE_STEP
    :annotation: = 20

    Lagrangian interpolation step for Flux models.
    Step value for Lagrangian interpolation. Must be compatible with lagrange_int.
    Lower than SD/SDXL defaults for Flux architecture.

.. data:: DEFAULT_SADA_FLUX_MAX_FIX
    :annotation: = 0

    Maximum fixed memory for Flux models.
    Set to 0 as Flux uses a different optimization approach.

.. data:: DEFAULT_SADA_ACC_RANGE
    :annotation: = (10, 47)

    Acceleration range start / end step for all models.
    Defines the starting step for SADA acceleration. 

    Start step must be at least 3 as SADA leverages third-order
    dynamics.

.. data:: DEFAULT_SADA_MAX_INTERVAL
    :annotation: = 4

    Maximum interval for optimization for all models.
    Maximum interval between optimizations in the SADA algorithm.

.. data:: DEFAULT_WAN_FPS
    :annotation: = 16.0

    Default frames per second for Wan text-to-video and image-to-video.

.. data:: DEFAULT_WAN_ANIMATE_FPS
    :annotation: = 30.0

    Default frames per second for Wan-Animate.

.. data:: DEFAULT_WAN_ANIMATE_2_FPS
    :annotation: = 24.0

    Default frames per second for Wan-Animate-2.

.. data:: DEFAULT_WAN_VAE_DTYPE
    :annotation: = 'float32'

    Default Wan VAE dtype. AutoencoderKLWan is fragile in bfloat16.

.. data:: DEFAULT_WAN_ANIMATE_SEGMENT_FRAME_LENGTH
    :annotation: = 77

    Default Wan-Animate segment length in frames.

.. data:: DEFAULT_WAN_ANIMATE_PREV_SEGMENT_FRAMES
    :annotation: = 1

    Default number of frames Wan-Animate carries over from the previous segment.

.. data:: DEFAULT_WAN_ANIMATE_2_INFERENCE_STEPS
    :annotation: = 40

    Fallback when ``--inference-steps`` is omitted on Wan-Animate-2 (modular pipeline has no step default).

.. data:: DEFAULT_WAN_ANIMATE_2_DISTILLED_INFERENCE_STEPS
    :annotation: = 10

    Fallback when ``--inference-steps`` is omitted on distilled Wan-Animate-2.

.. data:: DEFAULT_WAN_ANIMATE_2_SEGMENT_FRAME_LENGTH
    :annotation: = 81

    Default Wan-Animate-2 segment length in frames.

.. data:: DEFAULT_WAN_ANIMATE_2_PREV_SEGMENT_FRAMES
    :annotation: = 1

    Default number of frames Wan-Animate-2 carries over from the previous segment.

.. data:: DEFAULT_LTX_FPS
    :annotation: = 24.0

    Default frames per second for LTX video.

.. data:: DEFAULT_LTX_PROMPT_ENHANCER
    :annotation: = 'google/gemma-4-E2B-it'

    Default ``--ltx-prompt-enhancer`` model when ``--pipeline-prompt-enhance`` is on
    and no enhancer URI was given.

.. data:: PIPELINE_CACHE_MEMORY_CONSTRAINTS
    :annotation: = ['pipeline_size > (available * 0.75)']

    Cache constraint expressions for when to clear the CPU side 
    diffusion pipeline cache upon a new diffusion pipeline being created, 
    syntax provided via :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, all cached diffusion pipeline objects will be garbage collected.

    Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
    and ``pipeline_size`` (the estimated size of the new pipeline before it is brought into memory, in bytes)

.. data:: UNET_CACHE_MEMORY_CONSTRAINTS
    :annotation: = ['unet_size > (available * 0.75)']

    Cache constraint expressions for when to clear the CPU side 
    unet model cache upon a new unet model being created, 
    syntax provided via :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, all cached unet objects will be garbage collected.

    Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
    and ``unet_size`` (the estimated size of the new UNet before it is brought into memory, in bytes)

.. data:: VAE_CACHE_MEMORY_CONSTRAINTS
    :annotation: = ['vae_size > (available * 0.75)']

    Cache constraint expressions for when to clear the CPU side 
    vae model cache upon a new vae model being created, 
    syntax provided via :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, all cached vae objects will be garbage collected.

    Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
    and ``vae_size`` (the estimated size of the new VAE before it is brought into memory, in bytes)

.. data:: CONTROLNET_CACHE_MEMORY_CONSTRAINTS
    :annotation: = ['controlnet_size > (available * 0.75)']

    Cache constraint expressions for when to clear the CPU side 
    controlnet model cache upon a new controlnet model being created, 
    syntax provided via :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, all cached controlnet objects will be garbage collected.

    Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
    and ``controlnet_size`` (the estimated size of the new ControlNet before it is brought into memory, in bytes)

.. data:: ADAPTER_CACHE_MEMORY_CONSTRAINTS
    :annotation: = ['adapter_size > (available * 0.75)']

    Cache constraint expressions for when to clear the CPU side 
    adapter model cache upon a new adapter model being created, 
    syntax provided via :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, all cached adapter objects will be garbage collected.

    Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
    and ``adapter_size`` (the estimated size of the new T2IAdapter before it is brought into memory, in bytes)

.. data:: TEXT_ENCODER_CACHE_MEMORY_CONSTRAINTS
    :annotation: = ['text_encoder_size > (available * 0.75)']

    Cache constraint expressions for when to clear the CPU side 
    text encoder model cache upon a new text encoder model being created, 
    syntax provided via :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, all cached text encoder objects will be garbage collected.

    Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
    and ``text_encoder_size`` (the estimated size of the new Text Encoder before it is brought into memory, in bytes)

.. data:: IMAGE_ENCODER_CACHE_MEMORY_CONSTRAINTS
    :annotation: = ['image_encoder_size > (available * 0.75)']

    Cache constraint expressions for when to clear the CPU side 
    image encoder model cache upon a new image encoder model being created, 
    syntax provided via :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, all cached image encoder objects will be garbage collected.

    Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
    and ``image_encoder_size`` (the estimated size of the new Image Encoder before it is brought into memory, in bytes)

.. data:: TRANSFORMER_CACHE_MEMORY_CONSTRAINTS
    :annotation: = ['transformer_size > (available * 0.75)']

    Cache constraint expressions for when to clear the CPU side 
    transformer model cache upon a new transformer model being created, 
    syntax provided via :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, all cached transformer objects will be garbage collected.

    Extra variables include: ``cache_size`` (the current estimated cache size in bytes), 
    and ``transformer_size`` (the estimated size of the new transformer model before it is brought into memory, in bytes)