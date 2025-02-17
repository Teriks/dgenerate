Specifying a Transformer (SD3 and Flux)
=======================================

Stable Diffusion 3 and Flux do not use a UNet architecture, and instead use a
Transformer model in place of a UNet.

A specific transformer model can be specified using the ``--transformer`` argument.

This argument is nearly identical to ``--unet``, however it can support single file loads
from safetensors files or huggingface blob links if desired.

In addition to the arguments that ``--unet`` supports, ``--transformer`` supports the ``quantizer``
URI argument for enabling a quantization backend using the same URI syntax as ``--quantizer``.

SD3 Example:

.. code-block:: bash

    #!/usr/bin/env bash

    # This just loads the default transformer out of the repo on huggingface

    dgenerate stabilityai/stable-diffusion-3-medium-diffusers \
    --model-type torch-sd3 \
    --transformer "stabilityai/stable-diffusion-3-medium-diffusers;subfolder=transformer" \
    --variant fp16 \
    --dtype float16 \
    --inference-steps 30 \
    --guidance-scales 5.00 \
    --clip-skips 0 \
    --gen-seeds 2 \
    --output-path output \
    --model-sequential-offload \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution; artwork"

Flux Example:

.. code-block:: bash

    #!/usr/bin/env bash

    # use Flux with quantized transformer and T5 text encoder (bitsandbytes, 4 bits)

    dgenerate black-forest-labs/FLUX.1-dev \
    --model-type torch-flux \
    --dtype bfloat16 \
    --transformer "black-forest-labs/FLUX.1-dev;subfolder=transformer;quantizer='bnb;bits=4;bits4_compute_dtype=bfloat16'" \
    --text-encoders + "T5EncoderModel;model=black-forest-labs/FLUX.1-dev;subfolder=text_encoder_2;quantizer='bnb;bits=4;bits4_compute_dtype=bfloat16'" \
    --model-cpu-offload \
    --inference-steps 20 \
    --guidance-scales 3.5 \
    --gen-seeds 1 \
    --output-path output \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution"