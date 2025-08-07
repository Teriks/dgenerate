Quantization
============

Quantization via ``bitsandbytes`` and ``sdnq`` is supported for certain
diffusion submodels, for instance, the unet/transformer, all text encoders,
and controlnet models.

It is also supported for certain plugins which utilize LLMs, such as the
``magicprompt`` upscaler, and ``llm4gen`` prompt weighter.

Quantization in dgenerate is implemented through layer substitution, and
can run as a pre-process on models as they are loaded into memory with
very little processing time / overhead.

Quantization can be used to effectively cut the VRAM overhead for inference
in half or even by a fourth at the cost of slightly reduced output quality
due to precision loss.

There are a few ways to utilize quantization with dgenerate, the easiest
way being the ``--quantizer`` and ``--quantizer-map`` arguments.

The ``--quantizer`` argument takes a dgenerate quantizer URI to define
the quantization backend and settings, and applies the quantization
pre-process to the unet/transformer, and all text encoders of the
diffusion pipeline as it loads.

You can control which sub modules of the diffusion pipeline get quantized
by using the ``--quantizer-map`` argument, which accepts a list
of ``diffusers`` module names, e.g. ``unet``, ``text_encoder``, ``text_encoder_2``, 
``transformer``, ``controlnet``, etc.

.. code-block:: bash

    #!/usr/bin/env bash

    # only quantize the listed sub models

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 \
    --model-type sdxl \
    --dtype float16 \
    --variant fp16 \
    --quantizer "bnb;bits=8" \
    --quantizer-map unet text_encoder text_encoder_2 \
    --inference-steps 30 \
    --guidance-scales 5 \
    --prompts "a cute cat"


Quantization URI can also be supplied via sub-model URIs, the arguments
``--unet``, ``--transformer``, ``--text-encoders``, and ``--control-nets`` all support a ``quantizer``
sub URI argument for specifying the quantization backend for that particular sub-model.

This allows you to set specific quantization settings for sub-models individually.

When specifying from the command line, this may require some sub-quoting depending
on the shell, ``;`` is generally a special shell character, it is also used by
dgenerate as a URI argument seperator.

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 \
    --model-type sdxl \
    --dtype float16 \
    --variant fp16 \
    --unet 'stabilityai/stable-diffusion-xl-base-1.0;subfolder=unet;quantizer="bnb;bits=8"' \
    --inference-steps 30 \
    --guidance-scales 5 \
    --prompts "a cute cat"


ControlNet Quantization
-----------------------
ControlNet models are **NOT** quantized by default when using the global ``--quantizer`` 
argument. To quantize ControlNets, you must either:

1. Add ``controlnet`` to the ``--quantizer-map`` list to apply global quantization
2. Specify individual quantization settings per ControlNet using the ``quantizer`` URI argument

.. code-block:: bash

    #!/usr/bin/env bash

    # Method 1: Global quantization with controlnet in quantizer-map

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 \
    --model-type sdxl \
    --dtype float16 \
    --variant fp16 \
    --quantizer "bnb;bits=8" \
    --quantizer-map unet text_encoder text_encoder_2 controlnet \
    --control-nets "diffusers/controlnet-canny-sdxl-1.0" \
    --inference-steps 30 \
    --guidance-scales 5 \
    --prompts "a cute cat"

.. code-block:: bash

    #!/usr/bin/env bash

    # Method 2: Individual ControlNet quantization

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 \
    --model-type sdxl \
    --dtype float16 \
    --variant fp16 \
    --control-nets 'diffusers/controlnet-canny-sdxl-1.0;quantizer="bnb;bits=4"' \
    --inference-steps 30 \
    --guidance-scales 5 \
    --prompts "a cute cat"

.. code-block:: bash

    #!/usr/bin/env bash

    # ControlNet NOT quantized, only unet and text encoders

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 \
    --model-type sdxl \
    --dtype float16 \
    --variant fp16 \
    --quantizer "bnb;bits=8" \
    --control-nets "diffusers/controlnet-canny-sdxl-1.0" \
    --inference-steps 30 \
    --guidance-scales 5 \
    --prompts "a cute cat"

ControlNet quantization is only supported for Hugging Face repository loads 
and local directory paths. Single file ControlNet loads do not support quantization.

Quantizer usage documentation can be obtained with ``--quantizer-help`` or the
equivalent ``\quantizer_help`` config directive, you can use this argument or
directive to list quantization backend names, when you supply backend names as
arguments to this option, documentation will he listed for that backend. This
covers the URI arguments and how they affect the quantization pre-process.

The ``bitsandbytes`` backend documentation is as follows:


@COMMAND_OUTPUT[dgenerate --no-stdin --quantizer-help bnb]

And for ``sdnq``:

@COMMAND_OUTPUT[dgenerate --no-stdin --quantizer-help sdnq]





