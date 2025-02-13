Basic Usage
===========

The example below attempts to generate an astronaut riding a horse using 5 different
random seeds, 3 different inference steps values, and 3 different guidance scale values.

It utilizes the ``stabilityai/stable-diffusion-2-1`` model repo on `Hugging Face <https://huggingface.co/stabilityai/stable-diffusion-2-1>`_.

45 uniquely named images will be generated ``(5 x 3 x 3)``

Also Adjust output size to ``512x512`` and output generated images to the ``astronaut`` folder in the current working directory.

When ``--output-path`` is not specified, the default output location is the ``output`` folder
in the current working directory, if the path that is specified does not exist then it will be created.

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-2-1 \
    --prompts "an astronaut riding a horse" \
    --gen-seeds 5 \
    --output-path astronaut \
    --inference-steps 30 40 50 \
    --guidance-scales 5 7 10 \
    --output-size 512x512


Loading models from huggingface blob links is also supported:

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.safetensors \
    --prompts "an astronaut riding a horse" \
    --gen-seeds 5 \
    --output-path astronaut \
    --inference-steps 30 40 50 \
    --guidance-scales 5 7 10 \
    --output-size 512x512


SDXL is supported and can be used to generate highly realistic images.

Prompt only generation, img2img, and inpainting is supported for SDXL.

Refiner models can be specified, ``fp16`` model variant and a datatype of ``float16`` is
recommended to prevent out of memory conditions on the average GPU :)

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-diffusion-xl-base-1.0 --model-type torch-sdxl \
    --sdxl-high-noise-fractions 0.6 0.7 0.8 \
    --gen-seeds 5 \
    --inference-steps 50 \
    --guidance-scales 12 \
    --sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0 \
    --prompts "real photo of an astronaut riding a horse on the moon" \
    --variant fp16 --dtype float16 \
    --output-size 1024

