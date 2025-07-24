Quantization
============

Quantization via ``bitsandbytes`` and ``sdnq`` is supported for certain
diffusion submodels, for instance, the unet/transformer, and all text encoders.

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
of ``diffusers`` module names, e.g.

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
``--unet``, ``--transformer``, and ``--text-encoders`` all support a ``quantizer``
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


Quantizer usage documentation can be obtained with ``--quantizer-help``, you can use this
argument alone to list quantization backend names, when you supply backend names
as arguments to this option, documentation will he listed for that backend. This
covers the URI arguments and how they affect the quantization pre-process.

The ``bitsandbytes`` backend documentation is as follows:


@COMMAND_OUTPUT[dgenerate --no-stdin --quantizer-help bnb]

And for ``sdnq``:

@COMMAND_OUTPUT[dgenerate --no-stdin --quantizer-help sdnq]





