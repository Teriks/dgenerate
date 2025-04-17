Specifying Text Encoders
========================

Diffusion pipelines supported by dgenerate may use a varying number of
text encoder sub models, currently up to 3. ``--model-type torch-sd3``
for instance uses 3 text encoder sub models, all of which can be
individually specified from the command line if desired.

To specify a Text Encoder models directly use ``--text-encoders`` for
the primary model and ``--second-model-text-encoders`` for the SDXL Refiner or
Stable Cascade decoder.

Text Encoder URIs do not support loading from blob links or a single file,
text encoders must be loaded from a huggingface slug or a folder on disk
containing the models and configuration.

The syntax for specifying text encoders is similar to that of ``--vae``

The URI syntax for ``--text-encoders`` is ``TextEncoderClass;model=(huggingface repository slug or folder path)``

Loading arguments available when specifying a Text Encoder are: ``model``, ``revision``, ``variant``, ``subfolder``, ``dtype``, and ``quantizer``

The ``variant`` argument defaults to the value of ``--variant``

The ``dtype`` argument defaults to the value of ``--dtype``

The ``quantizer`` URI argument can be used to specify a quantization backend
for the text encoder using the same URI syntax as ``--quantizer``

The other named arguments are available when loading from a huggingface repository or folder
that may or may not be a local git repository on disk.

Available encoder classes are:

* CLIPTextModel
* CLIPTextModelWithProjection
* T5EncoderModel
* DistillT5EncoderModel (see: [LifuWang/DistillT5](https://huggingface.co/LifuWang/DistillT5))

You can query the text encoder types and position for a model by passing ``help``
as an argument to ``--text-encoders`` or ``--second-model-text-encoders``. This feature
may not be used for both arguments simultaneously, and also may not be used
when passing ``help`` or ``helpargs`` to any ``--scheduler`` type argument.

.. code-block:: bash

    #!/usr/bin/env bash

    # ask for text encoder help on the main model that is mentioned

    dgenerate https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/sd3_medium_incl_clips.safetensors \
    --model-type torch-sd3 \
    --variant fp16 \
    --dtype float16 \
    --text-encoders help

    # outputs:

    # Text encoder type help:
    #
    #     0 = CLIPTextModelWithProjection
    #     1 = CLIPTextModelWithProjection
    #     2 = T5EncoderModel

    # this means that there are 3 text encoders that we
    # could potentially specify manually in the order
    # displayed for this model

When specifying multiple text encoders, a special syntax is allowed to indicate that
a text encoder should be loaded from defaults, this syntax involves the plus
symbol. When a plus symbol is encountered it is regarded as "use default".

For instance in the example below, only the last of the three text encoders
involved in the Stable Diffusion 3 pipeline is specified, as it is the only
one not included with the main model file.

This text encoder is loaded from a subfolder of the Stable Diffusion 3
repository on huggingface.

.. code-block:: bash

    #!/usr/bin/env bash

    # This is an example of individually specifying text encoders
    # specifically for stable diffusion 3, this model from the blob
    # link includes the clip encoders, so we only need to specify
    # the T5 encoder, which is encoder number 3, the + symbols indicate
    # the first 2 encoders are assigned their default value, they are
    # loaded from the checkpoint file for the main model

    dgenerate https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/sd3_medium_incl_clips.safetensors \
    --model-type torch-sd3 \
    --variant fp16 \
    --dtype float16 \
    --inference-steps 30 \
    --guidance-scales 5.00 \
    --text-encoders + + \
        "T5EncoderModel;model=stabilityai/stable-diffusion-3-medium-diffusers;subfolder=text_encoder_3" \
    --clip-skips 0 \
    --gen-seeds 2 \
    --output-path output \
    --model-sequential-offload \
    --prompts "a horse outside a barn"


You may also use the URI value ``null``, to indicate that you do not want to ever load a specific text encoder at all.

For instance, you can prevent Stable Diffusion 3 from loading and using the T5 encoder all together.

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-3-medium-diffusers \
    --model-type torch-sd3 \
    --variant fp16 \
    --dtype float16 \
    --inference-steps 30 \
    --guidance-scales 5.00 \
    --text-encoders + + null \
    --clip-skips 0 \
    --gen-seeds 2 \
    --output-path output \
    --model-sequential-offload \
    --prompts "a horse outside a barn"


Any text encoder shared via the ``\use_modules`` directive in a config files is considered a default
value for the text encoder in the next pipeline that runs, using ``+`` will maintain this value
and using ``null`` will override it.

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    # this model will load all three text encoders,
    # they are not cached individually as we did not explicitly
    # specify any of them, they are cached with the pipeline
    # as a whole

    stabilityai/stable-diffusion-3-medium-diffusers
    --model-type torch-sd3
    --variant fp16
    --dtype float16
    --inference-steps 30
    --guidance-scales 5.00
    --clip-skips 0
    --gen-seeds 2
    --output-path output
    --model-sequential-offload
    --prompts "a horse outside a barn"

    # store all the text encoders from the last pipeline
    # into the variable "encoders"

    \save_modules encoders text_encoder text_encoder_2 text_encoder_3

    # share them with the next pipeline

    \use_modules encoders

    # use all the encoders except the T5 encoder (third encoder)
    # sharing modules this way saves a significant amount
    # of memory

    stabilityai/stable-diffusion-3-medium-diffusers
    --model-type torch-sd3
    --variant fp16
    --dtype float16
    --inference-steps 30
    --guidance-scales 5.00
    --clip-skips 0
    --text-encoders + + null
    --gen-seeds 2
    --output-path output
    --model-sequential-offload
    --prompts "a horse outside a barn"