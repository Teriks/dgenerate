.. py:currentmodule:: dgenerate.pipelinewrapper

.. data:: DEFAULT_INFERENCE_STEPS
    :annotation: = 30

    Default value for inference steps.

.. data:: DEFAULT_GUIDANCE_SCALE
    :annotation: = 5.0

    Default value for guidance scale.

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

    Default noise level for floyd super resolution upscalers when preforming img2img.

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

.. data:: DEFAULT_FLOYD_IF_OUTPUT_WIDTH
    :annotation: = 64

    Default output width for Deep Floyd IF txt2img first stage.

.. data:: DEFAULT_FLOYD_IF_OUTPUT_HEIGHT
    :annotation: = 64

    Default output height for Deep Floyd IF txt2img first stage.

.. data:: DEFAULT_SEED
    :annotation: = 0

    Default RNG seed.

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

.. data:: DEFAULT_ADETAILER_MASK_DILATION
    :annotation: = 4

    Default adetailer mask dilation

.. data:: DEFAULT_ADETAILER_MASK_BLUR
    :annotation: = 4

    Default adetailer mask blur.

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

.. data:: PIPELINE_WRAPPER_CACHE_GC_CONSTRAINTS
    :annotation: = ['used_percent > 70']

    Cache constraint expressions for when to clear all object caches, 
    before calling the diffusion pipeline wrapper.

    Syntax provided via :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, all cached objects will be garbage collected
    before a generation starts.

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