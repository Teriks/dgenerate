Specifying T2I Adapters
=======================

One or more T2I Adapters models may be specified with ``--t2i-adapters``, and multiple
T2I Adapter guidance images can be specified via ``--image-seeds`` in the case that you specify
multiple T2I Adapter models.

T2I Adapters are similar to ControlNet models and are mutually exclusive with ControlNet models,
IE: they cannot be used together.

T2I Adapters are more lightweight than ControlNet models, but only support txt2img generation
with control images for guidance, img2img and inpainting is not supported with T2I Adapters.

T2I Adapter models are supported for these model types:

    * ``--model-type torch``
    * ``--model-type torch-sdxl``

You can provide a huggingface repository slug / blob link, .pt, .pth, .bin, .ckpt, or .safetensors files.

Control images for the T2I Adapters can be provided using ``--image-seeds``

When using ``--t2i-adapters`` specifying control images via ``--image-seeds`` can be accomplished like this:

    * ``--image-seeds "control-image.png"`` (txt2img)

Multiple control image sources can be specified like this when using multiple T2I Adapters:

    * ``--image-seeds "control-1.png, control-2.png"`` (txt2img)


It is considered a syntax error if you specify a non-equal amount of control guidance
images and ``--t2i-adapters`` URIs and you will receive an error message if you do so.

Available URI arguments are: ``scale``, ``revision``, ``variant``, ``subfolder``, ``dtype``

The ``scale`` argument indicates the affect scale of the T2I Adapter model.

When using SDXL, the dgenerate argument ``--sdxl-t2i-adapter-factors`` can be used to specify
multiple adapter factors to try generating images with, the adapter factor is value between ``0.0`` and ``1.0``
indicating the fraction of time-steps over which the T2I adapter guidance is applied.

For example, a ``--sdxl-t2i-adapter-factors`` value of ``0.5`` would mean to only apply guidance
over the first half of the time-steps needed to generate the image.

When using multiple T2I Adapters, this value applies to all T2I Adapter models mentioned.

These examples use: `vermeer_canny_edged.png <https://raw.githubusercontent.com/Teriks/dgenerate/@REVISION/examples/media/vermeer_canny_edged.png>`_

.. code-block:: bash

    #!/usr/bin/env bash

    # SD1.5 example, use "vermeer_canny_edged.png" as a control guidance image

    dgenerate Lykon/dreamshaper-8 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --prompts "Painting, Girl with a pearl earring by Leonardo Da Vinci, masterpiece; low quality, low resolution, blank eyeballs" \
    --t2i-adapters "TencentARC/t2iadapter_canny_sd15v2;scale=0.5" \
    --image-seeds "vermeer_canny_edged.png"

    # SDXL example

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --vae "AutoencoderKL;model=madebyollin/sdxl-vae-fp16-fix" \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --inference-steps 30 \
    --guidance-scales 8 \
    --prompts "Taylor Swift, high quality, masterpiece, high resolution; low quality, bad quality, sketches" \
    --t2i-adapters "TencentARC/t2i-adapter-canny-sdxl-1.0;scale=0.5" \
    --image-seeds "vermeer_canny_edged.png" \
    --output-size 1024


If you want to select the repository revision, such as ``main`` etc, use the named argument ``revision``

.. code-block:: bash

    #!/usr/bin/env bash

    # This is only a syntax example

    dgenerate huggingface/model \
    --prompts "Syntax example" \
    --t2i-adapters "huggingface/t2i_repo;revision=main"


If your weights file exists in a subfolder of the repository, use the named argument ``subfolder``

.. code-block:: bash

    #!/usr/bin/env bash

    # This is only a syntax example

    dgenerate huggingface/model \
    --prompts "Syntax example" \
    --t2i-adapters "huggingface/t2i_repo;subfolder=repo_subfolder"


If you are loading a .safetensors or other file from a path on disk, simply do:

.. code-block:: bash

    #!/usr/bin/env bash

    # This is only a syntax example

    dgenerate Lykon/dreamshaper-8 \
    --prompts "Syntax example" \
    --t2i-adapters "my_t2i_model.safetensors"