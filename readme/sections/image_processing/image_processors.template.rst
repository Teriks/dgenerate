Image Processors
================

Images provided through ``--image-seeds`` can be processed before being used for image generation
through the use of the arguments ``--seed-image-processors``, ``--mask-image-processors``, and
``--control-image-processors``. In addition, dgenerates output can be post processed with the
used of the ``--post-processors`` argument, which is useful for using the ``upscaler`` processor.
An important note about ``--post-processors`` is that post processing occurs before any image grid
rendering is preformed when ``--batch-grid-size`` is specified with a ``--batch-size`` greater than one,
meaning that the output images are processed with your processor before being put into a grid.

Each of these options can receive one or more specifications for image processing actions,
multiple processing actions will be chained together one after another.

Using the option ``--image-processor-help`` with no arguments will yield a list of available image processor names.

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate --image-processor-help

Output:

.. code-block:: text

    Available image processors:

        "adetailer"
        "anyline"
        "canny"
        "flip"
        "grayscale"
        "hed"
        "invert"
        "leres"
        "letterbox"
        "lineart"
        "lineart-anime"
        "lineart-standard"
        "midas"
        "mirror"
        "mlsd"
        "normal-bae"
        "openpose"
        "pidi"
        "posterize"
        "resize"
        "sam"
        "solarize"
        "teed"
        "upscaler"
        "zoe"


Specifying one or more specific processors for example: ``--image-processor-help canny openpose`` will yield
documentation pertaining to those processor modules. This includes accepted arguments and their types for the
processor module and a description of what the module does.

Custom image processor modules can also be loaded through the ``--plugin-modules`` option as discussed
in the `Writing Plugins`_ section.


Image processor arguments
-------------------------

All processors posses the arguments: ``output-file`` and  ``output-overwrite``.

The ``output-file`` argument can be used to write the processed image to a specific file, if multiple
processing steps occur such as when rendering an animation or multiple generation steps, a numbered suffix
will be appended to this filename. Note that an output file will only be produced in the case that the
processor actually modifies an input image in some way. This can be useful for debugging an image that
is being fed into diffusion or a ControlNet.

The ``output-overwrite`` is a boolean argument can be used to tell the processor that you do not want numbered
suffixes to be generated for ``output-file`` and to simply overwrite it.

Some processors inherit the arguments: ``device``, and ``model-offload``.

The ``device`` argument can be used to override what device any hardware accelerated image processing
occurs on if any. It defaults to the value of ``--device`` and has the same syntax for specifying device
ordinals, for instance if you have multiple GPUs you may specify ``device=cuda:1`` to run image processing
on your second GPU, etc. Not all image processors respect this argument as some image processing is only
ever CPU based.

The ``model-offload`` argument is a boolean argument that can be used to force any torch modules / tensors
associated with an image processor to immediately evacuate the GPU or other non CPU processing device
as soon as the processor finishes processing an image.  Usually, any modules / tensors will be
brought on to the desired device right before processing an image, and left on the device until
the image processor object leaves scope and is garbage collected.

``model-offload`` can be useful for achieving certain GPU or processing device memory constraints, however
it is slower when processing multiple images in a row, as the modules / tensors must be brought on to the
desired device repeatedly for each image. In the context of dgenerate invocations where processors can
be used as preprocessors or postprocessors, the image processor object is garbage collected when the
invocation completes, this is also true for the ``\image_process`` directive.  Using this argument
with a preprocess specification, such as ``--control-image-processors`` may yield a noticeable memory
overhead reduction when using a single GPU, as any models from the image processor will be moved to the
CPU immediately when it is done with an image, clearing up VRAM space before the diffusion models enter GPU VRAM.

For an example, images can be processed with the canny edge detection algorithm or OpenPose (rigging generation)
before being used for generation with a model + a ControlNet.

This image of a `horse <https://raw.githubusercontent.com/Teriks/dgenerate/v@VERSION/examples/media/horse2.jpeg>`_
is used in the example below with a ControlNet that is trained to generate images from canny edge detected input.

.. code-block:: bash

    #!/usr/bin/env bash

    # --control-image-processors is only used for control images
    # in this case the single image seed is considered a control image
    # because --control-nets is being used

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --vae "AutoencoderKL;model=madebyollin/sdxl-vae-fp16-fix" \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --inference-steps 30 \
    --guidance-scales 8 \
    --prompts "Majestic unicorn, high quality, masterpiece, high resolution; low quality, bad quality, sketches" \
    --control-nets "diffusers/controlnet-canny-sdxl-1.0;scale=0.5" \
    --image-seeds "horse.jpeg" \
    --control-image-processors "canny;lower=50;upper=100" \
    --gen-seeds 2 \
    --output-size 1024 \
    --output-path unicorn


Multiple controlnet images, and input image batching
-----------------------------------------------------


Each ``--*-image-processors`` option has a special additional syntax, which is used to
describe which processor or processor chain is affecting which input image in an
``--image-seeds`` specification.

For instance if you have multiple control guidance images, and multiple controlnets which are going
to use those images, or frames etc. and you want to process each guidance image with a separate
processor OR processor chain. You can specify how each image is processed by delimiting the
processor specification groups with + (the plus symbol)

Like this:

    * ``--control-nets "huggingface/controlnet1" "huggingface/controlnet2"``
    * ``--image-seeds "image1.png, image2.png"``
    * ``--control-image-processors "affect-image1" + "affect-image2"``


Specifying a non-equal amount of control guidance images and ``--control-nets`` URIs is
considered a syntax error and you will receive an error message if you do so.

You can use processor chaining as well:

    * ``--control-nets "huggingface/controlnet1" "huggingface/controlnet2"``
    * ``--image-seeds "image1.png, image2.png"``
    * ``--control-image-processors "affect-image1" "affect-image1-again" + "affect-image2"``

In the case that you would only like the second image affected:

    * ``--control-nets "huggingface/controlnet1" "huggingface/controlnet2"``
    * ``--image-seeds "image1.png, image2.png"``
    * ``--control-image-processors + "affect-image2"``


The plus symbol effectively creates a NULL processor as the first entry in the example above.

When multiple guidance images are present, it is a syntax error to specify more processor chains
than control guidance images.  Specifying less processor chains simply means that the trailing
guidance images will not be processed, you can avoid processing leading guidance images
with the mechanism described above.

This can be used with an arbitrary amount of control image sources and controlnets, take
for example the specification:

    * ``--control-nets "huggingface/controlnet1" "huggingface/controlnet2" "huggingface/controlnet3"``
    * ``--image-seeds "image1.png, image2.png, image3.png"``
    * ``--control-image-processors + + "affect-image3"``


The two + (plus symbol) arguments indicate that the first two images mentioned in the control image
specification in ``--image-seeds`` are not to be processed by any processor.

This same syntax applies to ``img2img`` and ``mask`` images when using the ``images: ...`` batching
syntax described in: `Batching Input Images and Inpaint Masks`_

.. code-block:: bash

    #! /usr/bin/env bash

    # process these two images as img2img inputs in one go on the GPU
    # mirror the second image horizontally, the + indicates that
    # we are skipping processing the first image

    dgenerate stabilityai/stable-diffusion-2 \
    --inference-steps 30 \
    --guidance-scales 8 \
    --image-seeds "images: examples/media/horse2.jpeg, examples/media/horse2.jpeg" \
    --seed-image-processors + mirror \
    --image-seed-strengths 0.9 \
    --vae-tiling  \
    --vae-slicing \
    --output-path unicorn \
    --prompts "A fancy unicorn"

    # Now with inpainting

    dgenerate stabilityai/stable-diffusion-2 \
    --inference-steps 30 \
    --guidance-scales 8 \
    --image-seeds "images: examples/media/horse1.jpg, examples/media/horse1.jpg;mask=examples/media/horse1-mask.jpg, examples/media/horse1-mask.jpg" \
    --seed-image-processors + mirror \
    --mask-image-processors + mirror \
    --image-seed-strengths 0.9 \
    --vae-tiling  \
    --vae-slicing \
    --output-path mars_horse \
    --prompts "A photo of a horse standing on mars"


