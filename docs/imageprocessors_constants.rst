.. py:currentmodule:: dgenerate.imageprocessors.constants

.. data:: IMAGE_PROCESSOR_CUDA_MEMORY_CONSTRAINTS
    :annotation: = ['memory_required > (available * 0.70)']

    Cache constraint expressions for when to attempt to clear cuda VRAM
    upon a image processor plugin calling :py:meth:`dgenerate.imageprocessors.ImageProcessor.memory_guard_device`
    on a cuda device, syntax provided via :py:func:`dgenerate.memory.cuda_memory_constraints`

    If any of these constraints are met, an effort is made to clear modules off a GPU
    which are cached for fast repeat usage but are okay to flush.

    The only available extra variable is: ``memory_required``, which is the
    amount of memory the image processor plugin requested to be available.

.. data:: IMAGE_PROCESSOR_CACHE_GC_CONSTRAINTS
    :annotation: = ['memory_required > (available * 0.70)']

    Cache constraint expressions for when to attempt to clear objects out of any CPU side cache
    upon a image processor plugin calling :py:meth:`dgenerate.imageprocessors.ImageProcessor.memory_guard_device`
    on the cpu, syntax provided via :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, an effort is made to clear 
    objects out of any named CPU side cache.

    The only available extra variable is: ``memory_required``, which is the
    amount of memory the image processor plugin requested to be available.

.. data:: IMAGE_PROCESSOR_CACHE_MEMORY_CONSTRAINTS
    :annotation: = ['memory_required > (available * 0.70)']

    Cache constraint expressions for when to attempt to clear specifically the image processor 
    object cache upon a image processor plugin calling :py:meth:`dgenerate.imageprocessors.ImageProcessor.memory_guard_device`
    on the cpu, syntax provided via :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, an effort is made to clear objects 
    out of the image processor object cache.

    Available extra variables are: ``memory_required``, which is the
    amount of memory the image processor plugin requested to be available,
    and ``cache_size`` which is the current size of the image processor object cache.