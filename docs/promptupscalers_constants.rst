.. py:currentmodule:: dgenerate.promptupscalers

.. data:: PROMPT_UPSCALER_CUDA_MEMORY_CONSTRAINTS
    :annotation: = ['memory_required > (available * 0.70)']

    Cache constraint expressions for when to attempt to fully clear cuda VRAM 
    upon a prompt upscaler plugin requesting a device memory fence, syntax 
    provided via :py:func:`dgenerate.memory.cuda_memory_constraints`

    If any of these constraints are met, an effort is made to clear modules off a GPU 
    which are cached for fast repeat usage but are okay to flush.

    The only available extra variable is: ``memory_required``, which is the
    amount of memory the prompt upscaler plugin requested to fence the device
    for.

.. data:: PROMPT_UPSCALER_CACHE_GC_CONSTRAINTS
    :annotation: = ['upscaler_size > (available * 0.70)']

    Cache constraint expressions for when to attempt to fully clear CPU side ram before 
    the initial loading of a prompt upscaler module into ram, syntax provided via
    :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, an effort is made to clear objects out of 
    cpu side ram which are cached for fast repeat usage but are okay to flush,
    prior to loading a prompt upscaler model.

    The only available extra variable is: ``upscaler_size`` (the estimated size 
    of the prompt upscaler module that needs to enter ram, in bytes)

.. data:: PROMPT_UPSCALER_CACHE_MEMORY_CONSTRAINTS
    :annotation: = ['upscaler_size > (available * 0.70)']

    Cache constraint expressions for when to attempt to clear the prompt upscaler
    cache before bringing a new prompt upscaler online, this cache caches prompt 
    upscaler objects for reuse. :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, all prompt upscaler
    objects are cleared from the CPU cache.

    The only available extra variable is: ``upscaler_size`` (the estimated size 
    of the prompt upscaler module that needs to enter ram, in bytes)