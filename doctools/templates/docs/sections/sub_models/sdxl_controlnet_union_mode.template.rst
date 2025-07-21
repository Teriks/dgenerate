SDXL ControlNet Union Mode
---------------------------

SDXL can utilize a combined control-net model called ControlNet Union, i.e ``xinsir/controlnet-union-sdxl-1.0``.

This model is a union (combined weights) of different several different controlnet models for SDXL in one
file under one HuggingFace repository.

Contained within the safetensors file are ControlNet weights which cover 12 different types of control image input.

When using this controlnet repository, you must specify which image input guidance mode you want to use.

You can do this by specifying the mode name to the ``mode`` URI argument of ``--control-nets``.

The controlnet "mode" option may be set to one of:

    * ``openpose``
    * ``depth``
    * ``hed``
    * ``pidi``
    * ``scribble``
    * ``ted``
    * ``canny``
    * ``lineart``
    * ``anime_lineart``
    * ``mlsd``
    * ``normal``
    * ``segment``


Here is an example making use of ``depth`` and ``openpose``:

.. code-block:: bash

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    # You can utilize multiple SDXL ControlNet union models with different modes

    # The models used must be exactly the same model specification,
    # or dgenerate will throw a relevant error, the only URI arguments
    # that can vary are the "mode" and "scale" argument.

    # The "start" and "end" argument can be used with the first specification.
    # Only the "start" and "end" values from the first specification are used
    # and any further specifications are ignored.  These values apply to
    # the ControlNet model globally, (technically it is one model being used)

    # The "scale" argument can be applied per "mode", to indicate the amount
    # that particular mode contributes to guidance

    # Use depth + pose below, two images are used (the same images),
    # for each "mode" that is mentioned.

    # even 50/50 split on mode contribution

    stabilityai/stable-diffusion-xl-base-1.0 --model-type sdxl
    --variant fp16 --dtype float16
    --vae AutoencoderKL;model=madebyollin/sdxl-vae-fp16-fix
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0
    --gen-seeds 2
    --inference-steps 30
    --guidance-scales 8
    --output-path multiple
    --output-size 1024
    --model-cpu-offload
    --vae-tiling
    --vae-slicing
    --prompts "A boxer throwing a punch in the ring"
    --control-nets xinsir/controlnet-union-sdxl-1.0;scale=0.5;mode=depth xinsir/controlnet-union-sdxl-1.0;scale=0.5;mode=openpose
    --image-seeds "examples/media/man-fighting-pose.jpg, examples/media/man-fighting-pose.jpg"
    --control-image-processors midas + "openpose;include-hand=true;include-face=true;output-file=boxer/boxer-openpose.png"