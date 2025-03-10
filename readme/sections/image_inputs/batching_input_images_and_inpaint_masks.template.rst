Batching Input Images and Inpaint Masks
=======================================

For most model types excluding Stable Cascade, you can process multiple input images for ``img2img`` and
``inpaint`` mode on the GPU simultaneously.

This is done using the ``images: ...`` syntax of ``--image-seeds``

Here is an example of ``img2img`` usage:

.. code-block:: bash

    #! /usr/bin/env bash

    # Standard img2img, this results in two outputs
    # each of the images are resized to 1024 so they match
    # in dimension, which is a requirement for batching

    dgenerate stabilityai/stable-diffusion-2 \
    --inference-steps 30 \
    --guidance-scales 8 \
    --image-seeds "images: examples/media/earth.jpg, examples/media/mountain.png;1024" \
    --image-seed-strengths 0.9 \
    --vae-tiling  \
    --vae-slicing \
    --seeds 70466855166895  \
    --output-path batching \
    --prompts "A detailed view of the planet mars"

    # The --batch-size must be divisible by the number of provided images
    # this results in 4 images being produced, 2 variations of each input image

    dgenerate stabilityai/stable-diffusion-2 \
    --inference-steps 30 \
    --guidance-scales 8 \
    --image-seeds "images: examples/media/earth.jpg, examples/media/mountain.png;1024" \
    --batch-size 4
    --image-seed-strengths 0.9 \
    --vae-tiling  \
    --vae-slicing \
    --seeds 70466855166895  \
    --output-path batching \
    --prompts "A detailed view of the planet mars"

And an ``inpainting`` example:

.. code-block:: bash

    #! /usr/bin/env bash

    # With inpainting, we can either provide just one mask
    # for every input image, or a separate mask for each input image
    # if we wish to provide separate masks we could simply separate
    # them with commas as we do with the images in the images:
    # specification

    # These images have different aspect ratios and dimensions
    # so we are using the extended syntax of --image-seeds to
    # force them to all be the same shape

    # The same logic for --batch-size still applies as mentioned
    # in the img2img example

    dgenerate stabilityai/stable-diffusion-2-inpainting \
    --inference-steps 30 \
    --guidance-scales 8 \
    --image-seeds "images: ../../media/dog-on-bench.png, ../../media/beach.jpg;mask=../../media/dog-on-bench-mask.png;resize=1024;aspect=False" \
    --image-seed-strengths 1 \
    --vae-tiling \
    --vae-slicing \
    --seeds 39877139643371 \
    --output-path batching \
    --prompts "A fluffy orange cat, realistic, high quality; deformed, scary"


In the case of Stable Cascade, this syntax results in multiple images being passed to Stable Cascade
as an image/style prompt, and does not result in multiple outputs or batching behavior.

This Stable Cascade functionality is demonstrated in the example config: `examples/stablecascade/img2img/multiple-inputs-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/stablecascade/img2img/multiple-inputs-config.dgen>`_


