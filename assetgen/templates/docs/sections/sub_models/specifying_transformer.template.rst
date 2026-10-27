.. _specifying-a-transformer:

Specifying a Transformer (SD3, Flux, and later flow models)
===========================================================

Stable Diffusion 3, Flux, Flux.2, Z-Image, and Qwen-Image do not use a UNet.
They use a Transformer in its place. LTX does too; see `Video Generation`_.

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
    --model-type sd3 \
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
    --model-type flux \
    --dtype bfloat16 \
    --transformer "black-forest-labs/FLUX.1-dev;subfolder=transformer;quantizer='bnb;bits=4'" \
    --text-encoders + "T5EncoderModel;model=black-forest-labs/FLUX.1-dev;subfolder=text_encoder_2;quantizer='bnb;bits=4'" \
    --model-cpu-offload \
    --inference-steps 20 \
    --guidance-scales 3.5 \
    --gen-seeds 1 \
    --output-path output \
    --prompts "Photo of a horse standing near the open door of a red barn, high resolution"

An SD3, Flux, Flux.2, Z-Image, Qwen-Image, or LTX transformer can also be a
pre-quantized ``.gguf`` file. The repository still supplies the VAE and text
encoders. Do not set ``quantizer=`` on that URI and do not use ``--quantizer gguf``.
Flux.2 Klein, Qwen-Image, and LTX 2.5 are recognized from the file, including
ComfyUI layouts. Quantize the text encoders with ``bnb`` or ``sdnq`` and
``--quantizer-map`` if needed. See
`examples/stablediffusion3/gguf <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/stablediffusion3/gguf>`_,
`examples/flux/gguf <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/flux/gguf>`_,
`examples/flux2/gguf <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/flux2/gguf>`_,
`examples/z-image/gguf <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/z-image/gguf>`_,
`examples/qwen-image/gguf <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/qwen-image/gguf>`_,
and `examples/ltx2/gguf <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/ltx2/gguf>`_.

A transformer-only file can also be the model itself. dgenerate reads the
checkpoint, picks the matching Diffusers repository, and loads the text
encoders, tokenizers, and VAE from there. The repository plus ``--transformer``
form above is unchanged. ``--text-encoders`` and ``--vae`` still replace any
of those modules. ``+`` keeps a detected encoder and ``null`` skips one, so
SD3 can run without T5:

.. code-block:: bash

    dgenerate https://huggingface.co/city96/stable-diffusion-3-medium-gguf/blob/main/sd3_medium-Q4_K_S.gguf \
    --model-type sd3 \
    --variant fp16 \
    --text-encoders + + null \
    --prompts "a horse outside a barn"

A file that already contains text encoders keeps the single-file loader and
still honors ``null``. See
`examples/flux/gguf/flux-dev-as-model-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/flux/gguf/flux-dev-as-model-config.dgen>`_
and
`examples/stablediffusion3/text_encoders/without-t5-as-model-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/stablediffusion3/text_encoders/without-t5-as-model-config.dgen>`_.

A transformer can also be a pre-quantized ComfyUI ``.safetensors`` file in FP8,
INT8 (ConvRot), or NVFP4, such as the ComfyUI, Black Forest Labs, and Lightricks
releases. The same rules apply: point ``--transformer`` at the file and do not
set ``quantizer=``. See :ref:`comfyui-quantized-checkpoints`.

``--model-type ltx2`` and ``ltx-video`` also accept ``--transformer``.
That argument replaces the diffusion transformer only. See :ref:`video-generation`.