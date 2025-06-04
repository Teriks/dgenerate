Latents Interchange
===================

The ``--image-seeds`` argument supports a special latents syntax that allows you to pass raw latent tensors
between diffusion pipeline stages, enabling advanced techniques like cooperative denoising where multiple
models work together on the same image generation process.

This feature is particularly useful for splitting the denoising process across different models or stages,
allowing for creative combinations and enhanced control over the generation process.

In addition to this, latents can also be passed into the ``--image-seeds`` arguments ``img2img``
position, and they will be decoded by the receiving VAE (converted back into an image) prior
to use.

Supported Model Types:

- Stable Diffusion 1.5/2.x models (use ``latents: ...`` or ``latents= ...`` syntax)
- Stable Diffusion 3 models (use ``latents: ...`` or ``latents= ...`` syntax)
- Flux models (use ``latents: ...`` or ``latents= ...`` syntax)
- SDXL / Kolors models (use direct img2img latents input, no special syntax)

Latents Input Syntax
--------------------

There are two primary syntax forms for working with latents in ``--image-seeds``:

Direct Latents Input:

.. code-block:: bash

    --image-seeds "latents: path/to/latents.pt"

This syntax passes raw latent tensors directly to the pipeline without any img2img source image.
The latents file can be in ``pt``, ``pth``, or ``safetensors`` format.  It will not be decoded
by the receiving VAE.

Note for SDXL: SDXL models do not use the ``latents: ...`` syntax (or ``latents= ...`` syntax) for
cooperative denoising. Instead, SDXL takes latent tensors directly through the standard img2img slot
without special syntax.

Both SD3 and Flux models use the ``latents:`` syntax similar to SD 1.5/2.x models.

Combined Image and Latents Input:

.. code-block:: bash

    --image-seeds "img2img.png;latents=path/to/latents.pt"

Inpainting:

.. code-block:: bash

    --image-seeds "img2img.png;mask=mask.png;latents=path/to/latents.pt"

This syntax allows you to specify both an img2img source image and noisy starting latents simultaneously.

Generating Latents
------------------

To generate latent tensors that can be used with the latents syntax, use the ``--image-format``
argument with one of the supported tensor formats:

.. code-block:: bash

    # Generate latents in PyTorch format

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "a beautiful landscape" \
    --image-format pt \
    --output-path latents_output

    # Generate latents in SafeTensors format

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "a beautiful landscape" \
    --image-format safetensors \
    --output-path latents_output

The generated tensor files contain the raw latent representation that can be passed to subsequent
pipeline invocations using the latents syntax. Note that in the case above, the latents are fully
denoised, so they are only good for use as img2img input where they will be decoded by the
receiving VAE prior to use, and not for cooperative denoising.

To generate latents for cooperative denoising, you must use the ``--denoising-end`` argument to
specify that denoising is to be stopped at a certain percentage of the total denoising steps.

Cooperative Denoising
---------------------

Cooperative denoising is a technique where the diffusion process is split between multiple models,
with each model handling a specific portion of the denoising steps. This is accomplished using
the ``--denoising-start`` and ``--denoising-end`` arguments in combination with latents interchange.

This is supported for SD1.5/2.x (with certain schedulers), SDXL, Kolors, SD3, and Flux models.

The process works as follows:

1. A model denoises from pure noise up to a specified percentage (e.g., 80%)
2. The intermediate latents are saved and passed to the next stage
3. A different model continues denoising from where the first stage left off

This technique allows you to:

- Combine the strengths of different models
- Create unique artistic effects by switching models mid-generation
- Optimize generation speed by using faster models for early denoising stages
- Experiment with different model combinations


This is compatible with the following stateless schedulers when using an SD1.5/2.x model:

* ``EulerDiscreteScheduler``
* ``LMSDiscreteScheduler``
* ``EDMEulerScheduler``
* ``DPMSolverMultistepScheduler``
* ``DDIMScheduler``
* ``DDPMScheduler``
* ``PNDMScheduler``

Stable Diffusion 1.5/2.x Cooperative Denoising:

@EXAMPLE[../../../examples/latents_interchange/sd/cooperative-denoising-config.dgen]

Stable Diffusion 3 Cooperative Denoising:

@EXAMPLE[../../../examples/latents_interchange/sd3/cooperative-denoising-config.dgen]

Flux Cooperative Denoising:

@EXAMPLE[../../../examples/latents_interchange/flux/cooperative-denoising-config.dgen]

SDXL and Kolors models handle latents interchange differently than other model families.
For SDXL or Kolors cooperative denoising, latents are passed directly as img2img input without
the special ``latents:`` syntax:

@EXAMPLE[../../../examples/latents_interchange/sdxl/two-stage-refining-config.dgen]

Advanced Usage with Image Input:

You can also combine cooperative denoising with img2img input by using the combined syntax,
this works for SD1.5/2.x, SD3, and Flux models using the ``latents= ...`` syntax:

.. code-block:: bash

    #!/usr/bin/env bash
    
    # First stage: Process an input image with partial denoising
    dgenerate stabilityai/stable-diffusion-2-1 \
    --image-seeds "input.png" \
    --prompts "enhanced version of the input" \
    --image-seed-strengths 0.7 \
    --denoising-end 0.6 \
    --image-format pt \
    --output-path cooperative
    
    # Second stage: Continue with a different model using the latents
    dgenerate Lykon/DreamShaper \
    --image-seeds "input.png;latents={{ quote(last_images) }}" \
    --prompts "artistic interpretation with enhanced details" \
    --denoising-start 0.6 \
    --output-path cooperative

The same sort of technique mentioned above will also work with inpainting.

Note that when inpainting with SDXL and refining an image, you simply run the
refiner in ``img2img`` mode with the generated image you wish to refine, passed
as an actual image and not latents.

Img2Img with VAE Decode Example:

For cases where you want to generate latents and then decode them through a different model's VAE:

@EXAMPLE[../../../examples/latents_interchange/sd/img2img-with-vae-decode-config.dgen]