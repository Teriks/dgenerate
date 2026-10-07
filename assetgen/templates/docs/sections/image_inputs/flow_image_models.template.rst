.. _flow-image-models:

Flux.2, Z-Image, and Qwen-Image
===============================

``--model-type flux2``, ``z-image``, and ``qwen-image``, and the edit, layered,
Omni, and Klein KV types below, are flow-matching image models. Each has one
text encoder. LoRAs use ``--loras`` and fuse the same way as Flux.1.
:ref:`specifying-a-transformer` replaces the
diffusion transformer, including a ``.gguf`` file. Pixel width and height snap
down to a multiple of 16, because latents are packed into 2x2 patches. 1024 is
already aligned. The default sample size is 1024.

``--t2i-adapters``, ``--ip-adapters``, ``--pag``, ``--clip-skips``,
``--sdxl-refiner``, and ``--prompt-weighter`` are rejected. ``--control-nets``
is accepted on ``z-image`` and ``qwen-image`` only. See
`Specifying ControlNets`_.

Adetailer works where an inpaint pipeline exists: Flux.2 Klein, Z-Image,
Qwen-Image, and Qwen-Image Edit. ``--adetailer-detectors`` takes an
``--image-seeds`` image and no mask, and inpaints each detection.
``--post-processors adetailer`` does the same on the image just generated.
Full Flux.2, Klein KV, Z-Image Omni, Qwen-Image Layered, and Qwen edit-plus
have no inpaint pipeline, so adetailer is rejected. Qwen still uses true CFG.
Flux.2 Klein still has no negative prompt. A detection crop is aligned to
a multiple of 16, which is the size the inpaint call uses. Strength near 1
redraws that crop. Generated images are under
`examples/adetailer/post_processor <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/adetailer/post_processor>`_.
An image you already have is under
`examples/adetailer/arbitrary_image <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/adetailer/arbitrary_image>`_.
Flux.2 Klein, Z-Image, and Qwen-Image each have both.

``--model-cpu-offload``, ``--model-sequential-offload``, and
``--model-group-offload`` are mutually exclusive. Group offload keeps weights
in CPU memory, so the pipeline cache still counts them. BitsAndBytes, SDNQ,
and other quantized modules stay where they were loaded.
``--torch-compile`` compiles the repeated transformer blocks. Quantized
weights stay eager, and dgenerate warns.

Flux.2
------

``--model-type flux2`` is both full Flux.2 and Flux.2 Klein. ``model_index.json``
selects the pipeline. Klein is not its own ``--model-type``. Neither pipeline
takes a negative prompt. There is no img2img strength and no ControlNet.
``--dtype bfloat16`` and ``FlowMatchEulerDiscreteScheduler`` are the usual
settings. Output size 1024.

``--image-seeds`` with no mask are reference images, passed as ``image``.
There is no strength, and ``--image-seed-strengths`` is an error on those
seeds. A mask is Klein inpaint only. Full Flux.2 rejects a mask.

Full Flux.2
~~~~~~~~~~~

``black-forest-labs/FLUX.2-dev`` is gated. Guidance 4 and 50 steps.
The text encoder is Mistral. ``--flux2-text-encoder-out-layers`` defaults to
``10,20,30``. ``--flux2-caption-upsample-temperature`` turns on caption
upsampling. Klein rejects that argument.

See `examples/flux2/basic/dev-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/flux2/basic/dev-config.dgen>`_.

The Turbo LoRA is a normal ``--loras`` load, 8 steps, guidance 2.5, with an
explicit sigma list. ``--sigmas`` takes one CSV string (or ``expr:``), not
space-separated floats:

.. code-block::

    --loras "fal/FLUX.2-dev-Turbo;weight-name=flux.2-turbo-lora.safetensors;scale=1.0"
    --inference-steps 8
    --sigmas "1.0,0.6509,0.4374,0.2932,0.1893,0.1108,0.0495,0.00031"
    --guidance-scales 2.5

See `examples/flux2/lora/dev-turbo-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/flux2/lora/dev-turbo-config.dgen>`_.

A GGUF transformer is ``--transformer`` pointing at a ``.gguf`` file. The
parent repository still supplies the VAE and text encoder. Hidden width 6144
selects the dev config. Do not set ``quantizer=`` on that URI.

See `examples/flux2/gguf/dev-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/flux2/gguf/dev-config.dgen>`_.

Flux.2 Klein
~~~~~~~~~~~~

Klein encodes prompts with Qwen3. ``--flux2-text-encoder-out-layers`` defaults
to ``9,18,27``.

* Base 9B: ``black-forest-labs/FLUX.2-klein-base-9B``. Gated. Guidance 4, 50 steps.
* Distilled 4B: ``black-forest-labs/FLUX.2-klein-4B``. Public. Four steps.
  A distilled Klein checkpoint ignores ``--guidance-scales`` above 1.

Text to image has no image seed. References, when you use them, are
``--image-seeds`` with no mask, the same as full Flux.2.

Inpaint is the same Klein repository with ``--image-seeds "image.png;mask.png"``.
White mask pixels are repainted. The init image also stays packed as a
condition every step, so weak edits are normal at low strength. Use
``--image-seed-strengths 1`` and guidance nearer 8. ``reference=`` in that
seed is an extra reference image, passed as ``image_reference``. A reference
video is zipped with the inpaint clip, one frame at a time. Full Flux.2 has
no inpaint class.

See `examples/flux2/klein/config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/flux2/klein/config.dgen>`_
and `examples/flux2/klein/inpaint-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/flux2/klein/inpaint-config.dgen>`_.

GGUF width 3072 selects Klein 4B. Width 4096 selects Klein 9B and uses the
base-9B transformer config.

See `examples/flux2/gguf/klein-4b-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/flux2/gguf/klein-4b-config.dgen>`_.

Flux.2 Klein KV
~~~~~~~~~~~~~~~

``--model-type flux2-klein-kv`` is ``black-forest-labs/FLUX.2-klein-9b-kv``.
Gated. The published ``model_index.json`` class is ``Flux2KleinPipeline``;
dgenerate still loads ``Flux2KleinKVPipeline`` when the model type is
``flux2-klein-kv``.

Reference images are ``--image-seeds`` with no mask and no strength. The
pipeline caches their attention after the first step. There is no guidance
scale, no inpaint, and no ControlNet. Four steps is the short schedule.
A GGUF of this transformer is also width 4096, so it uses the Klein 9B
module config.

See `examples/flux2/klein-kv/config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/flux2/klein-kv/config.dgen>`_
and `examples/flux2/gguf/klein-kv-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/flux2/gguf/klein-kv-config.dgen>`_.

Z-Image
-------

``--model-type z-image`` is text to image, img2img, and inpaint. A negative
prompt is optional. Classifier-free guidance is on when guidance is greater
than 0.

* Turbo: ``Tongyi-MAI/Z-Image-Turbo``. 8 steps, ``--guidance-scales 0``. Public.
* Base: ``Tongyi-MAI/Z-Image``. Public. Guidance around 4, and 28 to 50 steps.
  The 2-step LoRA is a normal ``--loras`` load, not a new model type:

  .. code-block::

      --loras "alibaba-pai/Z-Image-Fun-Lora-Distill;weight-name=Z-Image-Fun-Lora-Distill-2-Steps-2603.safetensors;scale=1.0"
      --inference-steps 2
      --guidance-scales 1

* Set ``--image-seed-strengths 0.6`` for img2img. That is the pipeline's own
  default. If the option is omitted, dgenerate uses 0.8.
* Set ``--image-seed-strengths 1`` for inpaint. The mask blends the encoded
  image. It is not concatenated into the transformer.

Prequantized SDNQ Turbo is ``Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32``
(uint4, SVD rank 32). Do not pass ``--quantizer sdnq`` to it. That quantizes
again. A full-precision repo with ``--quantizer sdnq`` still quantizes on load.
``--transformer`` may point at an SDNQ transformer directory; detection uses
that directory, not only the pipeline repo.

A GGUF transformer stays on ``--model-type z-image``. The parent repository
supplies the text encoder and VAE. Base and Turbo share one transformer shape,
so a base GGUF still uses the Turbo module config. Sampling stays with the
checkpoint: Turbo is 8 steps and guidance 0, base is the longer schedule.

See `examples/z-image <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/z-image>`_.

ControlNet unions are ``--control-nets`` on ``z-image`` only, one model.
Text to image uses
``Z-Image-Turbo-Fun-Controlnet-Union.safetensors`` from
``alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union``. Inpaint uses
``Z-Image-Turbo-Fun-Controlnet-Union-2.0.safetensors`` from
``alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0``. The inpaint pipeline
rejects the 1.0 union. There is no img2img ControlNet class. The pipeline
rebuilds the ControlNet with ``from_transformer`` against the loaded
transformer, so both are loaded together. ``scale=0.75`` matches the pipeline
default. ``start`` and ``end`` other than ``0`` and ``1`` are an error.
On a Z-Image ControlNet the transformer is offloaded before the ControlNet,
because those two objects share modules.

Z-Image Omni
~~~~~~~~~~~~

``--model-type z-image-omni`` is Z-Image with an optional SigLIP condition
image. ``--image-seeds`` with no mask is that image. There is no strength,
no mask, and no ControlNet. The diffusers example uses ``Z-a-o/Z-Image-Turbo``.
CFG flags are the same ``--z-image-`` options.

Qwen-Image
----------

``--model-type qwen-image`` is text to image, img2img, and inpaint on
``Qwen/Qwen-Image``. Public. 50 steps and guidance 4 are the usual settings.
``--dtype bfloat16`` and ``FlowMatchEulerDiscreteScheduler``.

``--guidance-scales`` is true CFG (``true_cfg_scale``). ``--qwen-guidance-scale``
is the distilled guidance embedded in the transformer. Omit it to leave that
guidance unset. True CFG runs when ``--guidance-scales`` is above 1.
A negative prompt is optional. If you do not write one and guidance is above 1,
dgenerate passes a single space, which is what the pipeline expects.

Img2img and inpaint strength default to 0.6. The inpaint mask is packed and
concatenated with the latents.

See `examples/qwen-image/basic/config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/qwen-image/basic/config.dgen>`_,
`examples/qwen-image/img2img/config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/qwen-image/img2img/config.dgen>`_,
and `examples/qwen-image/inpaint/config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/qwen-image/inpaint/config.dgen>`_.

The Lightning LoRA is a normal ``--loras`` load. It is trained at guidance 1
and 8 steps:

.. code-block::

    --loras "lightx2v/Qwen-Image-Lightning;weight-name=Qwen-Image-Lightning-8steps-V1.0.safetensors;scale=1.0"
    --inference-steps 8
    --guidance-scales 1

See `examples/qwen-image/lora/lightning-8step-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/qwen-image/lora/lightning-8step-config.dgen>`_.

A GGUF transformer uses the same ``--model-type``. QuantStack files already
use Diffusers tensor names. A Comfy file prefixed with ``model.diffusion_model.``
is stripped on load. Edit uses this same module, so an edit GGUF keeps the
Qwen-Image transformer config. Layered is detected separately, below.

See `examples/qwen-image/gguf <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/qwen-image/gguf>`_.

ControlNet stays ``--model-type qwen-image`` with ``--control-nets``.
More than one ControlNet is allowed. Text to image uses the seed image as
the control image with ``InstantX/Qwen-Image-ControlNet-Union``. Inpaint
uses ``--image-seeds "image.png;mask.png"`` with one inpainting ControlNet
(``InstantX/Qwen-Image-ControlNet-Inpainting``): that image and mask are
``control_image`` and ``control_mask``. White mask pixels are repainted;
black are kept. Prefer a small white region — wiping most of the frame
often yields a black fill. The shipped example inverts ``horse1-mask.jpg``
so the horse is painted. Union ControlNets have
``extra_condition_channels=0`` and cannot be used with a mask; the
inpainting ControlNet has ``extra_condition_channels=4``. ``start`` and
``end`` are passed through. ``scale=1`` matches the pipeline default.

See `examples/qwen-image/controlnet/config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/qwen-image/controlnet/config.dgen>`_
and `examples/qwen-image/controlnet/inpaint-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/qwen-image/controlnet/inpaint-config.dgen>`_.

Qwen-Image Edit
~~~~~~~~~~~~~~~

``--model-type qwen-image-edit`` is an instruction edit. The image seed is
required. There is no strength.

* ``Qwen/Qwen-Image-Edit`` takes one image.
* ``Qwen/Qwen-Image-Edit-2509`` is edit-plus. ``images: a.png, b.png`` passes
  both images in one call. Edit-plus has no inpaint.
* A mask on the original edit checkpoint selects edit-inpaint. Strength
  applies only to that inpaint.

``--guidance-scales`` is still true CFG.

See `examples/qwen-image/edit/config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/qwen-image/edit/config.dgen>`_,
`examples/qwen-image/edit-plus/config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/qwen-image/edit-plus/config.dgen>`_,
and `examples/qwen-image/edit-inpaint/config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/qwen-image/edit-inpaint/config.dgen>`_.

Qwen-Image Layered
~~~~~~~~~~~~~~~~~~

``--model-type qwen-image-layered`` is ``Qwen/Qwen-Image-Layered``. One image
in, one file per layer out. ``--qwen-layered-layers`` defaults to 4.
``--qwen-layered-resolution`` is 640 or 1024, and defaults to 640.
``--qwen-layered-cfg-normalize`` and ``--qwen-layered-use-en-prompt`` are off
unless you set them.

The layered transformer has an extra ``addition_t_embedding`` and 3D position
embeddings. A layered GGUF is recognized from that embedding and loaded with
the layered config. A Qwen-Image GGUF is not a substitute.

See `examples/qwen-image/layered/config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/qwen-image/layered/config.dgen>`_
and `examples/qwen-image/gguf/layered-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/qwen-image/gguf/layered-config.dgen>`_.

Other call arguments
--------------------

These are one value. Omitting one leaves the pipeline default, and it does not
multiply the number of images.

* ``--flux2-caption-upsample-temperature`` is full Flux.2 caption upsampling.
  Klein rejects it.
* ``--flux2-text-encoder-out-layers`` selects the text-encoder layers.
  Full Flux.2 defaults to ``10,20,30``. Klein defaults to ``9,18,27``.
* ``--z-image-cfg-normalization`` turns on Z-Image CFG normalization.
  ``--z-image-cfg-truncation`` replaces the default of ``1``.
* ``--qwen-guidance-scale`` is the distilled guidance embedded in the Qwen-Image
  transformer. ``--guidance-scales`` stays the true CFG scale.
* ``--inpaint-crop`` with one padding integer, and without feathering or masked
  paste, is passed as ``padding_mask_crop`` on Klein and Qwen inpaint. The
  pipeline then crops and pastes. A two-sided or four-sided padding, a feather,
  or masked paste stays on dgenerate's crop.

Console recipes
---------------

The Console UI recipes are ``Flux.2 (Dev)``, ``Flux.2 Klein``, ``Flux.2 Klein KV``,
``Z-Image (Turbo)``, ``Z-Image Turbo (SDNQ)``, ``Z-Image (Base, 2-step LoRA)``,
``Z-Image (ControlNet)``, ``Z-Image Omni``, ``Qwen-Image``, ``Qwen-Image Edit``,
and ``Qwen-Image Layered``. GGUF recipes for those models set ``--transformer``
to a quantized file and leave the parent repository as the model line.
