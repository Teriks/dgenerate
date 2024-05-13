.. py:currentmodule:: dgenerate.pipelinewrapper


.. data:: CACHE_MEMORY_CONSTRAINTS
    :annotation: = ['used_percent > 70']

    Cache constraint expressions for when to clear all model caches (DiffusionPipeline, VAE, and ControlNet),
    syntax provided via :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, a call to :py:func:`.enforce_cache_constraints` will call
    :py:func:`.clear_model_cache` and force a garbage collection.


.. data:: PIPELINE_CACHE_MEMORY_CONSTRAINTS
    :annotation: = ['pipeline_size > (available * 0.75)']

    Cache constraint expressions for when to clear the DiffusionPipeline cache,
    syntax provided via :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, a call to :py:func:`.enforce_pipeline_cache_constraints` will call
    :py:func:`.clear_pipeline_cache` and force a garbage collection.

    Extra variables include: ``cache_size`` (the current estimated cache size in bytes),
    and ``pipeline_size`` (the estimated size of the new pipeline before it is brought into memory, in bytes)


.. data:: UNET_CACHE_MEMORY_CONSTRAINTS
    :annotation: = ['unet_size > (available * 0.75)']

    Cache constraint expressions for when to clear UNet cache,
    syntax provided via :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, a call to :py:func:`.enforce_unet_cache_constraints` will call
    :py:func:`.clear_unet_cache` and force a garbage collection.

    Extra variables include: ``cache_size`` (the current estimated cache size in bytes),
    and ``unet_size`` (the estimated size of the new UNet before it is brought into memory, in bytes)


.. data:: VAE_CACHE_MEMORY_CONSTRAINTS
    :annotation: = ['vae_size > (available * 0.75)']

    Cache constraint expressions for when to clear VAE cache,
    syntax provided via :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, a call to :py:func:`.enforce_vae_cache_constraints` will call
    :py:func:`.clear_vae_cache` and force a garbage collection.

    Extra variables include: ``cache_size`` (the current estimated cache size in bytes),
    and ``vae_size`` (the estimated size of the new VAE before it is brought into memory, in bytes)


.. data:: CONTROL_NET_CACHE_MEMORY_CONSTRAINTS
    :annotation: = ['control_net_size > (available * 0.75)']

    Cache constraint expressions for when to clear the ControlNet cache,
    syntax provided via :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, a call to :py:func:`.enforce_control_net_cache_constraints` will call
    :py:func:`.clear_control_net_cache` and force a garbage collection.

    Extra variables include: ``cache_size`` (the current estimated cache size in bytes),
    and ``control_net_size`` (the estimated size of the new ControlNet before it is brought into memory, in bytes)


.. data:: DEFAULT_INFERENCE_STEPS
    :annotation: = 30

    Default value for inference steps.

.. data:: DEFAULT_GUIDANCE_SCALE
    :annotation: = 5

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
