Specifying IP Adapters
======================

One or more IP Adapter models can be specified with the ``--ip-adapters`` argument.

The URI syntax for this argument is identical to ``--loras``, which is discussed in: `Specifying LoRAs`_

IP Adapters are supported for these model types:

    * ``--model-type torch``
    * ``--model-type torch-pix2pix``
    * ``--model-type torch-sdxl``
    * ``--model-type kolors``
    * ``--model-type torch-flux`` (basic adapter image specification only)

Here is a brief example of loading an IP Adapter in the most basic way and passing it an image via ``--image-seeds``.

This example nearly duplicates an image created with a code snippet in the diffusers documentation page
`found here <https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter#general-tasks>`_.

.. code-block:: bash

    #!/usr/bin/env bash

    # this uses one IP Adapter input image with the IP Adapter h94/IP-Adapter

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 \
    --model-type torch-sdxl \
    --dtype float16 \
    --variant fp16 \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --inference-steps 30 \
    --guidance-scales 5 \
    --sdxl-high-noise-fractions 0.8 \
    --seeds 0 \
    --output-path basic \
    --model-cpu-offload \
    --image-seeds "adapter: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_diner.png" \
    --ip-adapters h94/IP-Adapter;subfolder=sdxl_models;weight-name=ip-adapter_sdxl.bin \
    --output-size 1024x1024 \
    --prompts "a polar bear sitting in a chair drinking a milkshake; \
               deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality"


The main complexity of working with IP Adapters comes when specifying the ``--image-seeds`` URI for tasks other than the most basic usage
shown above.

Each IP Adapter model can accept multiple IP Adapter input images, and they do not need to all be the same dimension or aligned in any
particular way for the model to work.

In addition, IP Adapter models can be used with ControlNet and T2I Adapter models introducing additional complexities in specifying
image input.

If you specify multiple IP Adapters, they must all have the same ``variant`` URI argument value or you will receive a usage error.

----

basic --image-seeds specification
---------------------------------

The first syntax we can use with ``--image-seeds`` is designed to allow using IP Adapter images alone or with ControlNet images.

    * ``--image-seeds "adapter: adapter-image.png"`` (txt2img)
    * ``--image-seeds "adapter: adapter-image.png;control=control-image.png"`` (txt2img + ControlNet or T2I Adapter)

You may specify multiple IP Adapter images with the ``+`` image syntax, and multiple control images as you normally would with control images.

    * ``--image-seeds "adapter: adapter-image1.png + adapter-image2.png"``
    * ``--image-seeds "adapter: adapter-image1.png + adapter-image2.png;control=control-image1.png, control-image2.png"``


If you have multiple IP Adapter models loaded via ``--ip-adapters``, a comma delimits the images passed to each IP Adapter model.

    * ``--image-seeds "adapter: model1-adapter-image1.png + model1-adapter-image2.png, model2-adapter-image1.png + model2-adapter-image2.png"``


If you specify the ``resize``, ``aspect``, or ``align`` arguments for resizing the ``--image-seeds`` components, these arguments do
not affect the IP Adapter images.  Only the control images in the cases being discussed here.

In order to resize IP adapter images from the ``--image-seeds`` URI, you must use a sub-uri syntax for each adapter image.

This is always true for all adapter image specification syntaxes.

This sub-uri syntax uses the pipe ``|`` symbol to delimit its URI arguments for the specific IP Adapter image.

    * ``--image-seeds "adapter: adapter-image.png|resize=256|align=8|aspect=True"``
    * ``--image-seeds "adapter: adapter-image1.png|resize=256|align=8|aspect=True + adapter-image2.png|resize=256|align=8|aspect=True"``


This sub-uri syntax allows resizing each IP Adapter input image individually.

This syntax supports the arguments ``resize``, ``align``, and ``aspect``, which refer to the resize
dimension, image alignment, and whether or not the image resize that occurs is aspect correct.

These arguments mirror the behavior of the top level ``--image-seeds`` arguments with the same names.

However, alignment for IP Adapter images defaults to 1, meaning that there is no forced alignment
unless you force it manually.

----


img2img --image-seeds specification
-----------------------------------

You may use a traditional img2img input image along with IP Adapter input images.

The adapter images are then specified with the URI argument ``adapter``.

The exact same syntax is used when specifying the IP Adapter images this way as when using the ``adapter:`` prefix mentioned in the section above.

Including the ``+`` syntax and sub-uri resizing syntax.


    * ``--image-seeds "img2img-input.png;adapter=adapter-image.png"`` (img2img)
    * ``--image-seeds "img2img-input.png;adapter=adapter-image.png;control=control-image.png"`` (img2img + ControlNet or T2I Adapter)


----

inpainting --image-seeds specification
--------------------------------------

You may use inpainting with IP Adapter images by specifying an img2img input image and the ``mask`` argument of the ``--image-seeds`` URI.

The ``mask`` argument in this case does not refer to IP Adapter mask images, but simply inpainting mask images.


    * ``--image-seeds "img2img-input.png;mask=inpaint-mask.png;adapter=adapter-image.png"`` (inpaint)
    * ``--image-seeds "img2img-input.png;mask=inpaint-mask.png;adapter=adapter-image.png;control=control-image.png"`` (inpaint + ControlNet or T2I Adapter)


----

quoting IP Adapter image URLs with plus symbols
-----------------------------------------------

If you happen to need to download an IP Adapter image from a URL containing a plus symbol, the URL can be quoted
using single or double quotes depending on context.

There are quite a few different ways to quote the URI itself that will work, especially in config scripts where ``;`` is not
considered to be any kind of significant operator, and ``|`` is only used as an operator with the ``\exec`` directive.


    * ``--image-seeds "adapter: 'https://url.com?arg=hello+world' + image2.png"``
    * ``--image-seeds 'adapter:"https://url.com?arg=hello+world" + image2.png'``
    * ``--image-seeds "img2img.png;adapter='https://url.com?arg=hello+world' + image2.png"``
    * ``--image-seeds 'img2img.png;adapter="https://url.com?arg=hello+world" + image2.png'``

----

animated inputs & combinatorics
-------------------------------

Animated inputs work for IP Adapter images, when you specify an image seed with animated components such as videos or gifs,
the shortest animation dictates the amount of frames which will be processed in total, and any static images specified in
the image seed are duplicated across those frames.

The IP Adapter syntax introduces a lot of possible combinations for ``--image-seeds`` input images, and
not all possible combinations are covered in this documentation as it would be hard to do so.

If you find a combination that behaves strangely or incorrectly, or that should work but doesn't, please submit an issue :)