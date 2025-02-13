Inpainting
==========

Inpainting on an image can be preformed by providing a mask image with your image seed. This mask should be a black and white image
of identical size to your image seed.  White areas of the mask image will be used to tell the AI what areas of the seed image should be filled
in with generated content.

For using inpainting on animated image seeds, jump to: `Inpainting Animations`_

Some possible definitions for inpainting are:

    * ``--image-seeds "my-image-seed.png;my-mask-image.png"``
    * ``--image-seeds "my-image-seed.png;mask=my-mask-image.png"``

The format is your image seed and mask image separated by ``;``, optionally ``mask`` can be named argument.
The alternate syntax is for disambiguation when preforming img2img or inpainting operations while `Specifying ControlNets`_
or other operations where keyword arguments might be necessary for disambiguation such as per image seed `Animation Slicing`_,
and the specification of the image from a previous Deep Floyd stage using the ``floyd`` argument.

Mask images can be downloaded from URL's just like any other resource mentioned in an ``--image-seeds`` definition,
however for this example files on disk are used for brevity.

You can download them here:

 * `my-image-seed.png <https://raw.githubusercontent.com/Teriks/dgenerate/v@VERSION/examples/media/dog-on-bench.png>`_
 * `my-mask-image.png <https://raw.githubusercontent.com/Teriks/dgenerate/v@VERSION/examples/media/dog-on-bench-mask.png>`_

The command below generates a cat sitting on a bench with the images from the links above, the mask image masks out
areas over the dog in the original image, causing the dog to be replaced with an AI generated cat.

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-2-inpainting \
    --image-seeds "my-image-seed.png;my-mask-image.png" \
    --prompts "Face of a yellow cat, high resolution, sitting on a park bench" \
    --image-seed-strengths 0.8 \
    --guidance-scales 10 \
    --inference-steps 100