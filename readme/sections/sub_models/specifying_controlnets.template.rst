Specifying ControlNets
=======================

One or more ControlNet models may be specified with ``--control-nets``, and multiple control
net guidance images can be specified via ``--image-seeds`` in the case that you specify
multiple controlnet models.

ControlNet models are supported for these model types:

    * ``--model-type torch``
    * ``--model-type torch-sdxl``
    * ``--model-type kolors``
    * ``--model-type torch-sd3`` (img2img and inpainting not supported)
    * ``--model-type torch-flux``

You can provide a huggingface repository slug / blob link, .pt, .pth, .bin, .ckpt, or .safetensors files.

Control images for the ControlNets can be provided using ``--image-seeds``

When using ``--control-nets`` specifying control images via ``--image-seeds`` can be accomplished in these ways:

    * ``--image-seeds "control-image.png"`` (txt2img)
    * ``--image-seeds "img2img-seed.png;control=control-image.png"`` (img2img)
    * ``--image-seeds "img2img-seed.png;mask=mask.png;control=control-image.png"`` (inpainting)

Multiple control image sources can be specified in these ways when using multiple controlnets:

    * ``--image-seeds "control-1.png, control-2.png"`` (txt2img)
    * ``--image-seeds "control-1.png, control-2.png;align=64"`` (resize arguments work here)
    * ``--image-seeds "img2img-seed.png;control=control-1.png, control-2.png"`` (img2img)
    * ``--image-seeds "img2img-seed.png;mask=mask.png;control=control-1.png, control-2.png"`` (inpainting)


It is considered a syntax error if you specify a non-equal amount of control guidance
images and ``--control-nets`` URIs and you will receive an error message if you do so.

``resize=WIDTHxHEIGHT`` can be used to select a per ``--image-seeds`` resize dimension for all image
sources involved in that particular specification, as well as ``align``, ``aspect=true/false``, and
the frame slicing arguments ``frame-start`` and ``frame-end``.

ControlNet guidance images may actually be animations such as MP4s, GIFs etc. Frames can be
taken from multiple videos simultaneously. Any possible combination of image/video parameters can be used.
The animation with least amount of frames in the entire specification determines the frame count, and
any static images present are duplicated across the entire animation. The first animation present
in an image seed specification always determines the output FPS of the animation.

Arguments pertaining to the loading of each ControlNet model specified with ``--control-nets`` may be
declared in the same way as when using ``--vae`` with the addition of a ``scale`` argument.

Available arguments are: ``--model-type`` values are: ``scale``, ``start``, ``end``, ``revision``, ``variant``, ``subfolder``, ``dtype``

Most named arguments apply to loading from a huggingface repository or folder
that may or may not be a local git repository on disk, when loading directly from a .safetensors file
or other file from a path on disk the available arguments are ``scale``, ``start``, and ``end``.

The ``scale`` argument indicates the affect scale of the controlnet model.

For torch, the ``start`` argument indicates at what fraction of the total inference steps
at which the controlnet model starts to apply guidance. If you have multiple
controlnet models specified, they can apply guidance over different segments
of the inference steps using this option, it defaults to 0.0, meaning start at the
first inference step.

for torch, the ``end`` argument indicates at what fraction of the total inference steps
at which the controlnet model stops applying guidance. It defaults to 1.0, meaning
stop at the last inference step.


These examples use: `vermeer_canny_edged.png <vermeer_canny_edged.png_1_>`_


.. code-block:: bash

    #!/usr/bin/env bash

    # SD1.5 example, use "vermeer_canny_edged.png" as a control guidance image

    dgenerate Lykon/dreamshaper-8 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --prompts "Painting, Girl with a pearl earring by Leonardo Da Vinci, masterpiece; low quality, low resolution, blank eyeballs" \
    --control-nets "lllyasviel/sd-controlnet-canny;scale=0.5" \
    --image-seeds "vermeer_canny_edged.png"


    # If you have an img2img image seed, use this syntax

    dgenerate Lykon/dreamshaper-8 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --prompts "Painting, Girl with a pearl earring by Leonardo Da Vinci, masterpiece; low quality, low resolution, blank eyeballs" \
    --control-nets "lllyasviel/sd-controlnet-canny;scale=0.5" \
    --image-seeds "my-image-seed.png;control=vermeer_canny_edged.png"


    # If you have an img2img image seed and an inpainting mask, use this syntax

    dgenerate Lykon/dreamshaper-8 \
    --inference-steps 40 \
    --guidance-scales 8 \
    --prompts "Painting, Girl with a pearl earring by Leonardo Da Vinci, masterpiece; low quality, low resolution, blank eyeballs" \
    --control-nets "lllyasviel/sd-controlnet-canny;scale=0.5" \
    --image-seeds "my-image-seed.png;mask=my-inpaint-mask.png;control=vermeer_canny_edged.png"

    # SDXL example

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --vae "AutoencoderKL;model=madebyollin/sdxl-vae-fp16-fix" \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --inference-steps 30 \
    --guidance-scales 8 \
    --prompts "Taylor Swift, high quality, masterpiece, high resolution; low quality, bad quality, sketches" \
    --control-nets "diffusers/controlnet-canny-sdxl-1.0;scale=0.5" \
    --image-seeds "vermeer_canny_edged.png" \
    --output-size 1024


If you want to select the repository revision, such as ``main`` etc, use the named argument ``revision``

.. code-block:: bash

    #!/usr/bin/env bash

    # This is only a syntax example

    dgenerate huggingface/model \
    --prompts "Syntax example" \
    --control-nets "huggingface/cn_repo;revision=main"


If your weights file exists in a subfolder of the repository, use the named argument ``subfolder``

.. code-block:: bash

    #!/usr/bin/env bash

    # This is only a syntax example

    dgenerate huggingface/model \
    --prompts "Syntax example" \
    --control-nets "huggingface/cn_repo;subfolder=repo_subfolder"


If you are loading a .safetensors or other file from a path on disk, simply do:

.. code-block:: bash

    #!/usr/bin/env bash

    # This is only a syntax example

    dgenerate Lykon/dreamshaper-8 \
    --prompts "Syntax example" \
    --control-nets "my_cn_model.safetensors"