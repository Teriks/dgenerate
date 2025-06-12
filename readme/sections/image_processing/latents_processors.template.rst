Latents Processors
=================

Latents processor operate on the latent space representation of images in diffusion models.
Unlike image processors which work with pixel data, latents processors manipulate the compressed, abstract representation
that diffusion models use internally. These processors are particularly useful when working with dgenerate's latents
interchange functionality.

dgenerate supports three main ways to use latents processors:

* ``--latents-processors`` - Process latents when using raw latent input through ``--image-seeds "latents: ..."`` or ``img2img.png;latents ...`` syntax
* ``--img2img-latents-processors`` - Process latents during img2img generation when using latent input through ``--image-seeds "latents.pt`` syntax
* ``--latents-post-processors`` - Process latents when outputting to latent formats (pt, pth, safetensors)

Using the option ``--latents-processor-help`` with no arguments will yield a list of available latents processor names:

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate --latents-processor-help

Available Processors
------------------

dgenerate includes several built-in latents processors:

* ``noise`` - Injects noise into latents using the pipeline's scheduler
* ``scale`` - Scales and normalizes latents tensors
* ``interposer`` - Converts fully denoised latents between different model spaces (e.g., SD1.x ↔ SDXL ↔ SD3)

Each processor has its own set of arguments that can be specified using URI syntax:

.. code-block:: bash

    processor_name;arg1=value1;arg2=value2

Processing Raw Latent Input
--------------------------

When using cooperative denoising with SD1.5/2.x, SD3, or Flux models:

.. code-block:: bash

    dgenerate model_name \
    --image-seeds "latents: partial_denoised.pt" \
    --latents-processors "scale;factor=1.5" \
    --denoising-start 0.8

Converting Between Model Spaces
-----------------------------

The interposer processor can convert fully denoised latents between different model architectures:

.. code-block:: bash

    # Generate initial latents with SD1.x model
    dgenerate runwayml/stable-diffusion-v1-5 \
    --prompts "a beautiful landscape" \
    --image-format pt \
    --output-path stage1

    # Convert and process with SDXL
    dgenerate stabilityai/stable-diffusion-xl-base-1.0 \
    --image-seeds stage1/*.pt \
    --img2img-latents-processors "interposer;source=v1;target=xl" \
    --prompts "a beautiful landscape"

Processing Img2Img Latents
-------------------------

When using latents alongside img2img input:

.. code-block:: bash

    dgenerate model_name \
    --image-seeds "input.png;latents=noisy_start.pt" \
    --img2img-latents-processors "noise;timestep=50;seed=42"

Multiple Processors and Chaining
------------------------------

Like image processors, multiple latents processors can be chained together:

.. code-block:: bash

    dgenerate model_name \
    --image-seeds "latents: input.pt" \
    --latents-processors "scale;factor=1.2" "noise;timestep=20"

When using multiple latent inputs, you can specify different processor chains for each input using
the + delimiter:

.. code-block:: bash

    dgenerate model_name \
    --image-seeds "latents: latents1.pt, latents: latents2.pt" \
    --latents-processors "scale;factor=1.5" + "noise;timestep=30"

Processor Arguments
-----------------

All latents processors support these base arguments:

* ``device`` - Override which device any hardware accelerated processing occurs on
* ``model-offload`` - Force torch modules/tensors to evacuate GPU memory after processing

The ``device`` argument defaults to the value of ``--device`` and follows the same syntax for specifying
device ordinals (e.g., ``device=cuda:1`` for the second GPU).

Memory Management
---------------

Latents processors operate in the model's latent space, which is typically much smaller than pixel space.
However, some processors (like the interposer) may load additional models that require significant memory.
The ``model-offload`` argument can help manage GPU memory by moving processor-related data to CPU after
each processing step.

Model Compatibility
-----------------

Different model families handle latents differently:

* SD1.5/2.x, SD3, and Flux models use the ``latents: ...`` or ``latents=...`` syntax
* SDXL/Kolors models take latents directly as img2img input without special syntax
* When using ``--denoising-start`` with SDXL/Kolors, input latents are not decoded by the VAE

For more details on working with latents, see the `Latents Interchange`_ section. 