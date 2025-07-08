Specifying a Stable Cascade Decoder
===================================

When the main model is a Stable Cascade prior model and ``--model-type torch-s-cascade`` is specified,
you may specify a decoder model with ``--s-cascade-decoder``.

The syntax (and URI arguments) for specifying the decoder model is identical to specifying an SDXL refiner
model as mentioned above.

.. code-block:: bash

    #!/usr/bin/env bash

    dgenerate stabilityai/stable-cascade-prior \
    --model-type torch-s-cascade \
    --variant bf16 \
    --dtype bfloat16 \
    --model-cpu-offload \
    --second-model-cpu-offload \
    --s-cascade-decoder "stabilityai/stable-cascade;dtype=float16" \
    --inference-steps 20 \
    --guidance-scales 4 \
    --second-model-inference-steps 10 \
    --second-model-guidance-scales 0 \
    --gen-seeds 2 \
    --prompts "an image of a shiba inu, donning a spacesuit and helmet"