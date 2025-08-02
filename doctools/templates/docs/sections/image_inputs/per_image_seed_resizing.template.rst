Per Image Seed Resizing
=======================

If you want to specify multiple image seeds that will have different output sizes irrespective
of their input size or a globally defined output size defined with ``--output-size``,
You can specify their output size individually at the end of each provided image seed.

This will work when using a mask image for inpainting as well, including when using animated inputs.

This also works when `Specifying ControlNets`_ and guidance images for controlnets.

Resizing in this fashion will resize any img2img image, inpaint mask, or control image to the specified
size, generally all of these images need to be the same size. In combination with the URI argument
``aspect=False`` this can be used to force multiple images of different sizes to the same dimension.

This does not resize IP Adapter images as they have their own special per image resizing
syntax discussed in: `Specifying IP Adapters`_

Here are some possible definitions:

    * ``--image-seeds "my-image-seed.png;512x512"`` (img2img)
    * ``--image-seeds "my-image-seed.png;my-mask-image.png;512x512"`` (inpainting)
    * ``--image-seeds "my-image-seed.png;resize=512x512"`` (img2img)
    * ``--image-seeds "my-image-seed.png;mask=my-mask-image.png;resize=512x512"`` (inpainting)

The alternate syntax with named arguments is for disambiguation when `Specifying ControlNets`_, or
performing per image seed `Animation Slicing`_, or specifying the previous Deep Floyd stage output
with the ``floyd`` keyword argument.

When one dimension is specified, that dimension is the width, and the height.

The height of an image is calculated to be aspect correct by default for all resizing
methods unless ``--no-aspect`` has been given as an argument on the command line or the
``aspect`` keyword argument is used in the ``--image-seeds`` definition.

The the aspect correct resize behavior can be controlled on a per image seed definition basis
using the ``aspect`` keyword argument.  Any value given to this argument overrides the presence
or absense of the ``--no-aspect`` command line argument.

the ``aspect`` keyword argument can only be used when all other components of the image seed
definition are defined using keyword arguments. ``aspect=false`` disables aspect correct resizing,
and ``aspect=true`` enables it.

Some possible definitions:

    * ``--image-seeds "my-image-seed.png;resize=512x512;aspect=false"`` (img2img)
    * ``--image-seeds "my-image-seed.png;mask=my-mask-image.png;resize=512x512;aspect=false"`` (inpainting)


You may also specify the ``align`` keyword argument in order to force a specific image alignment for
all incoming images, this alignment value must be divisible by 8, and can be used with or without
the specification of ``resize``.

Some possible definitions with ``align``:

    * ``--image-seeds "my-image-seed.png;resize=1000;align=64"`` (equates to ``960x960``)
    * ``--image-seeds "my-image-seed.png;align=64"`` (force the original size to 64 pixel alignment)

The following example performs img2img generation, followed by inpainting generation using 2 image seed definitions.
The involved images are resized using the basic syntax with no keyword arguments present in the image seeds.

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --image-seeds "my-image-seed.png;1024" "my-image-seed.png;my-mask-image.png;512x512" \
    --prompts "Face of a yellow cat, high resolution, sitting on a park bench" \
    --image-seed-strengths 0.8 \
    --guidance-scales 10 \
    --inference-steps 100
