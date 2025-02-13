Flux ControlNet Union Mode
---------------------------

Flux can also utilize a ControlNet Union model, more specifically: ``InstantX/FLUX.1-dev-Controlnet-Union``.

This model is a union (combined weights) of seven different trained controlnet models for Flux in one file under
one HuggingFace repository.

When using this controlnet repository, you must specify which image input guidance mode you want to use.

You can do this by specifying the mode name to the ``mode`` URI argument of ``--control-nets``.

The controlnet "mode" option may be set to one of:

 * ``canny``
 * ``tile``
 * ``depth``
 * ``blur``
 * ``pose``
 * ``gray``
 * ``lq`` (enhance low quality image)


.. code-block:: bash

    #!/usr/bin/env bash

    # Use a character from the examples media folder
    # of this repository to generate an openpose rigging,
    # and then feed that image to Flux using the ControlNet
    # union repository, with the mode specified as "pose"

    dgenerate black-forest-labs/FLUX.1-schnell \
    --model-type torch-flux \
    --dtype bfloat16 \
    --model-sequential-offload \
    --control-nets InstantX/FLUX.1-dev-Controlnet-Union;scale=0.8;mode=pose \
    --image-seeds examples/media/man-fighting-pose.jpg \
    --control-image-processors openpose \
    --inference-steps 4 \
    --guidance-scales 0 \
    --gen-seeds 1 \
    --output-path output \
    --output-size 1024x1024 \
    --prompts "a boxer throwing a punch in the ring"


You can specify multiple instances of this controlnet URI with different modes if desired.

Everything else about controlnet URI usage, such as URI arguments, is unchanged from
what is described in the main `Specifying ControlNets`_ section.