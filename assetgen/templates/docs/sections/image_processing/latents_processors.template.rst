Latents Processors
==================

Latents processor operate on the latent space representation of images in diffusion models.
Unlike image processors which work with pixel data, latents processors manipulate the compressed, abstract representation
that diffusion models use internally. These processors are particularly useful when working with dgenerate's latents
interchange functionality.

dgenerate supports three main ways to use latents processors:

The argument ``--latents-processors`` which processes latents when using raw latent input through ``--image-seeds "latents: ..."`` or ``img2img.png;latents ...`` syntax.

The argument  ``--img2img-latents-processors`` which processes latents during img2img generation when using latents input as img2img data, e.g. ``--image-seeds latents.pt``

And the argument ``--latents-post-processors`` which processes latents after generation when outputting to latent formats (pt, pth, safetensors)

Using the option ``--latents-processor-help`` with no arguments will yield a list of available latents processor names:

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate --latents-processor-help

Output:

@COMMAND_OUTPUT[dgenerate --no-stdin --latents-processor-help]

Each processor has its own set of arguments that can be specified using URI syntax,
which can be viewed with: ``dgenerate --latents-processor-help PROCESSOR_NAMES``
in the same fashion as other dgenerate plugins.

Processing Raw Latent Input
---------------------------

When using the ``--image-seeds "latents: ..."`` or ``--image-seeds "img2img.png;latents= ..."`` syntax to pass in
raw / noisy latents, you can use ``--latents-processors`` to run a process on this type of latent input.

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2 \
    --image-seeds "latents: partially_denoised.pt" \
    --latents-processors "scale;factor=1.5" \
    --denoising-start 0.8

Processing Img2Img Latents
--------------------------

When using latents as ``img2img`` input, they will be decoded by the receiving VAE, for
this usage you should use ``--img2img-latents-processors``.

There is a separate option for this use, as latents can be used for ``img2img`` input and
as raw latents input simultaneously if desired.

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2 \
    --image-seeds "fully_denoised_img2img.pt" \
    --img2img-latents-processors "noise;timestep=50;seed=42"

For instance, ``--img2img-latents-processors`` acts on ``fully_denoised_img2img.pt``, which
will end up being decoded and used as an ``img2img`` source.  And ``--latents-processors``
acts on ``partially_denoised.pt``, which will be passed straight into the model without
decoding as a starting point for inference.

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2 \
    --image-seeds "fully_denoised_img2img.pt;latents=partially_denoised.pt" \
    --img2img-latents-processors "noise;timestep=50;seed=42"
    --latents-processors "scale;factor=1.5" \
    --denoising-start 0.8

Multiple Processors and Chaining
--------------------------------

Like image processors, multiple latents processors can be chained together:

.. code-block:: bash

    dgenerate stabilityai/stable-diffusion-2 \
    --image-seeds "latents: noisy_input.pt" \
    --latents-processors "scale;factor=1.2" "noise;timestep=20"

When using multiple latent inputs (batching), you can specify different processor chains for each input using
the + delimiter, just like image processors:

.. code-block:: bash

     dgenerate stabilityai/stable-diffusion-2 \
    --image-seeds "latents: latents1.pt, latents: latents2.pt" \
    --latents-processors "scale;factor=1.5" + "noise;timestep=30"


With ``img2img`` input batching:

.. code-block:: bash

     dgenerate stabilityai/stable-diffusion-2 \
    --image-seeds "images: img2img.png, img2img.png;latents=latents1.pt, latents2.pt" \
    --latents-processors "scale;factor=1.5" + "noise;timestep=30"


Latents Interposer
------------------

The ``interposer`` latents processor can be used to convert fully denoised latents into
a space / distribution that a VAE designed for another model type can understand.

This can allow you to convert between the latent space of one model type to another.

This only works for fully denoised latents, and not for partially denoised latents.

@EXAMPLE[@PROJECT_DIR/examples/latents_interposer/sd1-to-sdxl-config.dgen]

The ``interposer`` supports several conversions, described in its help output:

@COMMAND_OUTPUT[dgenerate --no-stdin --latents-processor-help interposer]
