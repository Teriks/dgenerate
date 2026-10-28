Specifying Control LoRAs
========================

``--control-loras`` is a lightweight alternative to ``--control-nets`` for
``--model-type sd`` (SD1.5 and SD2), ``sdxl``, and ``flux``. It cannot be
combined with ``--control-nets`` or ``--t2i-adapters``.

One control image is required. A plain ``--image-seeds`` path is that image,
the same way it is for ``--control-nets``. Several Control LoRAs share the
one image. Use ``--control-image-processors`` to build the signal the LoRA
expects, for example ``canny``. Style LoRAs still use ``--loras`` and are
applied after the Control LoRA.

Flux.1 loads the LoRA onto the transformer and runs ``FluxControlPipeline``.
``--quantizer`` on a full-precision Flux checkpoint fuses that LoRA before
quantization and caches the transformer. A transformer that is already packed
(GGUF, Comfy, bitsandbytes, or SDNQ) keeps the Control LoRA active as an adapter.

SD and SDXL clone the UNet into a ControlNet and fuse a Stability-style control
LoRA into that clone. The clone is built from full-precision UNet weights, so a
checkpoint that is already packed cannot be cloned. ``--quantizer`` quantizes
the UNet and the text encoders. The clone stays full precision unless
``controlnet`` is included in ``--quantizer-map``. That clone is a second UNet,
so leaving it out of the map keeps a full-precision copy in memory.

The URI arguments are ``scale``, ``revision``, ``subfolder``, and ``weight-name``.
``scale`` is the LoRA weight, defaulting to 1.0, and it is combined with
``--control-lora-fuse-scale`` when the weights are fused.

.. code-block:: jinja

    #! /usr/bin/env dgenerate --file
    #! dgenerate @VERSION

    # SDXL canny Control LoRA. The ControlNet clone is quantized because
    # controlnet is listed in --quantizer-map.

    stabilityai/stable-diffusion-xl-base-1.0
    --model-type sdxl
    --variant fp16
    --dtype float16
    --quantizer "bnb;bits=8"
    --quantizer-map unet text_encoder text_encoder_2 controlnet
    --control-loras "stabilityai/control-lora;subfolder=control-LoRAs-rank128;weight-name=control-lora-canny-rank128.safetensors"
    --control-image-processors "canny;lower=100;upper=200"
    --image-seeds "control.png"
    --inference-steps 30
    --guidance-scales 5
    --prompts "aerial view, a futuristic research complex"
