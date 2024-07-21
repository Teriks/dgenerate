.. py:currentmodule:: dgenerate.imageprocessors


.. data:: IMAGE_PROCESSOR_CUDA_MEMORY_CONSTRAINTS
    :annotation: = ['processor_size > (available * 0.70)']

    Cache constraint expressions for when to attempt to fully clear cuda VRAM before
    moving an image processor on to a cuda device, syntax provided via
    :py:func:`dgenerate.memory.cuda_memory_constraints`

    If any of these constraints are met, an effort is made to clear modules off the GPU
    which are cached for fast repeat usage but are okay to flush, prior to moving
    an image processor to the GPU.

    The only available extra variable is: ``pipeline_size`` (the estimated size
    of the image processor module that needs to enter VRAM, in bytes)


.. data:: IMAGE_PROCESSOR_MEMORY_CONSTRAINTS
    :annotation: = ['processor_size > (available * 0.70)']

    Cache constraint expressions for when to attempt to fully clear CPU side ram before
    the initial loading of an image processor module into ram, syntax provided via
    :py:func:`dgenerate.memory.memory_constraints`

    If any of these constraints are met, an effort is made to clear modules out of
    cpu side ram which are cached for fast repeat usage but are okay to flush,
    prior to loading an image processor model.

    The only available extra variable is: ``pipeline_size`` (the estimated size
    of the image processor module that needs to enter ram, in bytes)
