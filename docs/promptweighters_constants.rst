.. py:currentmodule:: dgenerate.promptweighters

.. data:: PROMPT_WEIGHTER_CUDA_MEMORY_CONSTRAINTS
    :annotation: = ['memory_required > (available * 0.70)']

    Cache constraint expressions for when to attempt to fully clear cuda VRAM 
    upon a prompt weighter plugin requesting a device memory fence, syntax 
    provided via :py:func:`dgenerate.memory.cuda_memory_constraints`

    If any of these constraints are met, an effort is made to clear modules off a GPU 
    which are cached for fast repeat usage but are okay to flush.

    The only available extra variable is: ``memory_required``, which is the
    amount of memory the prompt weighter plugin requested to fence the device
    for.

.. data:: PROMPT_WEIGHTER_CACHE_GC_CONSTRAINTS
    :annotation: = ['weighter_size > (available * 0.70)']

    Cache constraint expressions for when to attempt to fully clear CPU side ram before 
    the initial loading of a prompt weighter module into ram, syntax provided via
    :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, an effort is made to clear objects out of 
    cpu side ram which are cached for fast repeat usage but are okay to flush,
    prior to loading a prompt weighter model.

    The only available extra variable is: ``weighter_size`` (the estimated size 
    of the prompt weighter module that needs to enter ram, in bytes)

.. data:: PROMPT_WEIGHTER_CACHE_MEMORY_CONSTRAINTS
    :annotation: = ['weighter_size > (available * 0.70)']

    Cache constraint expressions for when to attempt to clear the prompt weighter
    cache before bringing a new prompt weighter online, this cache caches prompt 
    weighter objects for reuse. :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, all prompt weighter
    objects are cleared from the CPU cache.

    The only available extra variable is: ``weighter_size`` (the estimated size 
    of the prompt weighter module that needs to enter ram, in bytes)