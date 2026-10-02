.. _flow-image-models:

Flux.2, Z-Image, and Qwen-Image
===============================

``--model-type flux2``, ``z-image``, and ``qwen-image`` are flow-matching
image models. Each one has a single text encoder. LoRAs load with ``--loras``
and are fused the same way as Flux.1. Pixel width and height snap down to a
multiple of 16 before the pipeline call, because latents are packed into 2x2
patches. 1024 is already aligned. The default sample size is 1024.

``--t2i-adapters``, ``--ip-adapters``, ``--pag``, ``--clip-skips``,
``--sdxl-refiner``, and ``--prompt-weighter`` are rejected. ``--control-nets``
is accepted on ``z-image`` and ``qwen-image``.

``--model-cpu-offload``, ``--model-sequential-offload``, and
``--model-group-offload`` are mutually exclusive. Group offload keeps weights
in CPU memory, so the pipeline cache still counts them. BitsAndBytes, SDNQ,
and other quantized modules stay where they were loaded. On a Z-Image
ControlNet the transformer is offloaded before the ControlNet, because those
two objects share modules.

Flux.2
------

``--model-type flux2`` covers both full Flux.2 and Flux.2 Klein. The checkpoint
``model_index.json`` class name selects which one. Klein is not a separate
``--model-type``.

* Text to image, full Flux.2: ``black-forest-labs/FLUX.2-dev``. Guidance 4, 50 steps.
  Gated. ``--image-seeds`` with no mask is a reference image (or a reference video),
  passed as ``image``. There is no strength. ``--image-seed-strengths`` is an error.
* Text to image, Klein: ``black-forest-labs/FLUX.2-klein-base-9B``. Guidance 4, 50 steps.
  Gated. There is no img2img class and no strength. References work the same way.
* Inpaint, Klein only: the same Klein repo, with ``--image-seeds "image.png;mask.png"``.
  Strength defaults to 0.8. ``reference=`` in that seed is ``image_reference``.
  A reference video is zipped with the inpaint clip, one frame at a time.

Full Flux.2 encodes prompts with Mistral. Klein encodes them with Qwen3.
A distilled Klein checkpoint ignores ``--guidance-scales`` above 1. Neither
pipeline takes a negative prompt.

Examples: `examples/flux2 <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/flux2>`_.

Z-Image
-------

``--model-type z-image`` is text to image, img2img, and inpaint.

* Turbo: ``Tongyi-MAI/Z-Image-Turbo``. 8 steps, ``--guidance-scales 0``. Public.
* Base: ``Tongyi-MAI/Z-Image``. Public. The 2-step LoRA is a normal ``--loras``
  load, not a new model type:

  .. code-block::

      --loras "alibaba-pai/Z-Image-Fun-Lora-Distill;weight-name=Z-Image-Fun-Lora-Distill-2-Steps-2603.safetensors;scale=1.0"
      --inference-steps 2
      --guidance-scales 1

* Set ``--image-seed-strengths 0.6`` for img2img. That is the pipeline's own
  default. If the option is omitted, dgenerate uses 0.8.
* Set ``--image-seed-strengths 1`` for inpaint. The mask blends the encoded
  image. It is not concatenated into the transformer.

Classifier-free guidance is on when guidance is greater than 0. A negative
prompt is optional.

Prequantized SDNQ Turbo is ``Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32``
(uint4, SVD rank 32). Do not pass ``--quantizer sdnq`` to it. That quantizes
again. A full-precision repo with ``--quantizer sdnq`` still quantizes on load.
``--transformer`` may point at an SDNQ transformer directory; detection uses
that directory, not only the pipeline repo. The installed ``sdnq`` package
reads this checkpoint's ``quantization_config.json``.

ControlNet unions are ``--control-nets`` on ``z-image`` only, one model.
Text to image uses
``Z-Image-Turbo-Fun-Controlnet-Union.safetensors`` from
``alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union``. Inpaint uses
``Z-Image-Turbo-Fun-Controlnet-Union-2.0.safetensors`` from
``alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0``. The inpaint pipeline
rejects the 1.0 union. There is no img2img ControlNet class. The pipeline
rebuilds the ControlNet with ``from_transformer`` against the loaded
transformer, so both are loaded together. ``scale=0.75`` matches the pipeline
default.

Examples: `examples/z-image <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/z-image>`_.

Qwen-Image
----------

``--model-type qwen-image`` is text to image, img2img, and inpaint on
``Qwen/Qwen-Image``. Public.

``--guidance-scales`` maps to ``true_cfg_scale`` only. ``--qwen-guidance-scale``
is the distilled embedded guidance; omitting it leaves that guidance unset.
True CFG runs when that scale is above 1.
If you do not write a negative prompt and guidance is above 1, dgenerate passes
a single space, which is what the pipeline expects. Img2img and inpaint
strength default to 0.6. The inpaint mask is packed and concatenated with the
latents.

Examples: `examples/qwen-image <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/qwen-image>`_.

Related model types
-------------------

These are separate ``--model-type`` values. An image seed with no mask is a
condition image. There is no strength.

* ``flux2-klein-kv`` is ``black-forest-labs/FLUX.2-klein-9b-kv``. Reference
  images are ``--image-seeds`` with no mask. The pipeline caches their
  attention state after the first step. It has no guidance scale. Gated.
* ``z-image-omni`` is Z-Image with a SigLIP condition image. ``--image-seeds``
  is optional. The diffusers example uses ``Z-a-o/Z-Image-Turbo``.
* ``qwen-image-edit`` is an instruction edit. ``Qwen/Qwen-Image-Edit`` is one
  image. ``Qwen/Qwen-Image-Edit-2509`` is edit-plus, and
  ``images: a.png, b.png`` passes both images in one call. A mask selects
  edit-inpaint on the edit checkpoint. Edit-plus has no inpaint.
* ``qwen-image-layered`` is ``Qwen/Qwen-Image-Layered``. One image becomes a
  stack of layers, and each layer is written. ``--qwen-layered-layers``
  defaults to 4. ``--qwen-layered-resolution`` is 640 or 1024, and defaults
  to 640.
  ``--qwen-layered-cfg-normalize`` and ``--qwen-layered-use-en-prompt`` are off
  unless you set them.

Qwen-Image ControlNet stays ``--model-type qwen-image`` with ``--control-nets``.
Text to image uses the seed image as the control image. Inpaint uses
``--image-seeds "image.png;mask.png"`` with one ControlNet: that image and mask
are ``control_image`` and ``control_mask``. ``start`` and ``end`` are passed
through. The union repo is ``InstantX/Qwen-Image-ControlNet-Union``.

This Diffusers version has no Flux.2 ControlNet class. Modular pipelines are a
different call shape and stay unsupported.

Examples: `examples/flux2 <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/flux2>`_,
`examples/z-image <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/z-image>`_,
`examples/qwen-image <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/qwen-image>`_.

Overrides
---------

These are one value. Omitting one leaves the pipeline default, and it does not
multiply the number of images.

* ``--flux2-caption-upsample-temperature`` is full Flux.2 caption upsampling.
  Klein rejects it.
* ``--flux2-text-encoder-out-layers 10,20,30`` selects the text-encoder layers.
  Klein's default is ``9,18,27``.
* ``--z-image-cfg-normalization`` turns on Z-Image CFG normalization.
  ``--z-image-cfg-truncation`` replaces the default of ``1``.
* ``--qwen-guidance-scale`` is the distilled guidance embedded in the Qwen-Image
  transformer. ``--guidance-scales`` stays the true CFG scale.
* ``--inpaint-crop`` with one padding integer, and without feathering or masked
  paste, is passed as ``padding_mask_crop`` on Klein and Qwen inpaint. The
  pipeline then crops and pastes. A two-sided or four-sided padding, a feather,
  or masked paste stays on dgenerate's crop.
* Z-Image ControlNet accepts ``scale=`` only. ``start`` and ``end`` other than
  ``0`` and ``1`` are an error.

Console recipes
---------------

The Console UI recipes are ``Flux.2 (Dev)``, ``Flux.2 Klein``, ``Flux.2 Klein KV``,
``Z-Image (Turbo)``, ``Z-Image Turbo (SDNQ)``, ``Z-Image (Base, 2-step LoRA)``,
``Z-Image (ControlNet)``, ``Z-Image Omni``, ``Qwen-Image``, ``Qwen-Image Edit``,
and ``Qwen-Image Layered``.
