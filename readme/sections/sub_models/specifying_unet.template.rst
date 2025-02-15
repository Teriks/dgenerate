Specifying a UNet
=================

An alternate UNet model can be specified via a URI with the ``--unet`` option, in a
similar fashion to ``--vae`` and other model arguments that accept URIs.

UNets are supported for these model types:

    * ``--model-type torch``
    * ``--model-type torch-if``
    * ``--model-type torch-ifs``
    * ``--model-type torch-ifs-img2img``
    * ``--model-type torch-pix2pix``
    * ``--model-type torch-upscaler-x2``
    * ``--model-type torch-upscaler-x4``
    * ``--model-type torch-sdxl``
    * ``--model-type torch-sdxl-pix2pix``
    * ``--model-type torch-s-cascade``

This is useful in particular for using the latent consistency scheduler as well as the
``lite`` variants of the unet models used with Stable Cascade.

The first component of the ``--unet`` URI is the model path itself.

You can provide a path to a huggingface repo, or a folder on disk (downloaded huggingface repository).

The latent consistency UNet for SDXL can be specified with the ``--unet`` argument.

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --variant fp16 --dtype float16 \
    --unet latent-consistency/lcm-sdxl \
    --scheduler LCMScheduler \
    --inference-steps 4 \
    --guidance-scales 8 \
    --gen-seeds 2 \
    --output-size 1024 \
    --prompts "a close-up picture of an old man standing in the rain"

Loading arguments available when specifying a UNet are: ``revision``, ``variant``, ``subfolder``, and ``dtype``

In the case of ``--unet`` the ``variant`` loading argument defaults to the value
of ``--variant`` if you do not specify it in the URI.

The ``--unet2`` option can be used to specify a UNet for the
`SDXL Refiner <#specifying-an-sdxl-refiner>`_ or `Stable Cascade Decoder <#specifying-a-stable-cascade-decoder>`_,
and uses the same syntax as ``--unet``.

Here is an example of using the ``lite`` variants of Stable Cascade's
UNet models which have a smaller memory footprint using ``--unet`` and ``--unet2``.

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-cascade-prior \
    --model-type torch-s-cascade \
    --variant bf16 \
    --dtype bfloat16 \
    --unet "stabilityai/stable-cascade-prior;subfolder=prior_lite" \
    --unet2 "stabilityai/stable-cascade;subfolder=decoder_lite" \
    --model-cpu-offload \
    --model-cpu-offload2 \
    --s-cascade-decoder "stabilityai/stable-cascade;dtype=float16" \
    --inference-steps 20 \
    --guidance-scales 4 \
    --s-cascade-decoder-inference-steps 10 \
    --s-cascade-decoder-guidance-scales 0 \
    --gen-seeds 2 \
    --prompts "an image of a shiba inu, donning a spacesuit and helmet"