Animation Slicing
=================

Animated inputs can be sliced by a frame range either globally using
``--frame-start`` and ``--frame-end`` or locally using the named argument
syntax for ``--image-seeds``, for example:

    * ``--image-seeds "animated.gif;frame-start=3;frame-end=10"``.

When using animation slicing at the ``--image-seed`` level, all image input definitions
other than the main image must be specified using keyword arguments.

For example here are some possible definitions:

    * ``--image-seeds "seed.gif;frame-start=3;frame-end=10"``
    * ``--image-seeds "seed.gif;mask=mask.gif;frame-start=3;frame-end=10``
    * ``--image-seeds "seed.gif;control=control-guidance.gif;frame-start=3;frame-end=10``
    * ``--image-seeds "seed.gif;mask=mask.gif;control=control-guidance.gif;frame-start=3;frame-end=10``
    * ``--image-seeds "seed.gif;floyd=stage1.gif;frame-start=3;frame-end=10"``
    * ``--image-seeds "seed.gif;mask=mask.gif;floyd=stage1.gif;frame-start=3;frame-end=10"``

Specifying a frame slice locally in an image seed overrides the global frame
slice setting defined by ``--frame-start`` or ``--frame-end``, and is specific only
to that image seed, other image seed definitions will not be affected.

Perhaps you only want to run diffusion on the first frame of an animated input in
order to save time in finding good parameters for generating every frame. You could
slice to only the first frame using ``--frame-start 0 --frame-end 0``, which will be much
faster than rendering the entire video/gif outright.

The slice range zero indexed and also inclusive, inclusive means that the starting and ending frames
specified by ``--frame-start`` and ``--frame-end`` will be included in the slice.  Both slice points
do not have to be specified at the same time. You can exclude the tail end of a video with
just ``--frame-end`` alone, or seek to a certain start frame in the video with ``--frame-start`` alone
and render from there onward, this applies for keyword arguments in the ``--image-seeds`` definition as well.

If your slice only results in the processing of a single frame, an animated file format will
not be generated, only a single image output will be generated for that image seed during the
generation step.


.. code-block:: bash

    #!/usr/bin/env bash

    # Generate using only the first frame

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse" \
    --image-seeds https://upload.wikimedia.org/wikipedia/commons/7/7b/Muybridge_race_horse_~_big_transp.gif \
    --image-seed-strengths 0.5 \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512 \
    --animation-format mp4 \
    --frame-start 0 \
    --frame-end 0
