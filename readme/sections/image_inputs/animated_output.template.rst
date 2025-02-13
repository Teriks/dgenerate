Animated Output
===============

``dgenerate`` supports many video formats through the use of PyAV (ffmpeg), as well as GIF & WebP.

See ``--help`` for information about all formats supported for the ``--animation-format`` option.

When an animated image seed is given, animated output will be produced in the format of your choosing.

In addition, every frame will be written to the output folder as a uniquely named image.

By specifying ``--animation-format frames`` you can tell dgenerate that you just need
the frame images and not to produce any coalesced animation file for you. You may also
specify ``--no-frames`` to indicate that you only want an animation file to be produced
and no intermediate frames, though using this option with ``--animation-format frames``
is considered an error.

If the animation is not 1:1 aspect ratio, the width will be fixed to the width of the
requested output size, and the height calculated to match the aspect ratio of the animation.
Unless ``--no-aspect`` or the ``--image-seeds`` keyword argument ``aspect=false`` are specified,
in which case the video will be resized to the requested dimension exactly.

If you do not set an output size, the size of the input animation will be used.

.. code-block:: bash

    #!/usr/bin/env bash

    # Use a GIF of a man riding a horse to create an animation of an astronaut riding a horse.

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse" \
    --image-seeds https://upload.wikimedia.org/wikipedia/commons/7/7b/Muybridge_race_horse_~_big_transp.gif \
    --image-seed-strengths 0.5 \
    --output-path astronaut \
    --inference-steps 50 \
    --guidance-scales 10 \
    --output-size 512x512 \
    --animation-format mp4


The above syntax is the same syntax used for generating an animation with a control
image when ``--control-nets`` or ``--t2i-adapters`` is used.

Animations can also be generated using an alternate syntax for ``--image-seeds``
that allows the specification of a control image source when it is desired to use
``--control-nets`` with img2img or inpainting.

For more information about this see: `Specifying ControlNets`_

And also: `Specifying T2I Adapters`_

As well as the information about ``--image-seeds`` from dgenerates ``--help``
output.

IP Adapter images can also be animated inputs see: `Specifying IP Adapters`_

In general, every image component of an ``--image-seeds`` specification may be an
animated file, animated files may be mixed with static images. The animated input with the
shortest length determines the number of output frames, and any static image components
are duplicated over that amount of frames.
