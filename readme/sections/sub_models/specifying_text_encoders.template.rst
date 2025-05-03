Specifying Text Encoders
========================

Diffusion pipelines supported by dgenerate may use a varying number of
text encoder sub models, currently up to 3. ``--model-type torch-sd3``
for instance uses 3 text encoder sub models, all of which can be
individually specified from the command line if desired.

To specify a Text Encoder models directly use ``--text-encoders`` for
the primary model and ``--second-model-text-encoders`` for the SDXL Refiner or
Stable Cascade decoder.

The syntax for specifying text encoders is similar to that of ``--vae``

The URI syntax for ``--text-encoders`` is ``TextEncoderClass;model=(huggingface repository slug or folder path)``

Loading arguments available when specifying a Text Encoder are: ``model``, ``revision``, ``variant``, ``subfolder``, ``dtype``, ``quantizer``, and ``mode``

The ``model`` argument is the path to the the model, this may be a Hugging Face slug, a folder on disk,
a checkpoint file on disk, a URL pointing to a single file model, or a Hugging Face blob link.

The ``revision`` argument is used to specify the repo revision when loading out
of a Hugging Face repo, or a checked out repo on disk.

The ``variant`` argument defaults to the value of ``--variant``, specifying ``null`` explicitly
indicates to not use any variant, even if ``--variant`` is specified.

The ``subfolder`` argument specifies the subfolder when loading from a Hugging Face repository,
when loading from a single file checkpoint that has text encoders packaged with it, this can
be used to specify the sub-model inside the checkpoint, for instance ``text_encoder`` will
work identically on a single file checkpoint containing said text encoder as it does
with a Hugging Face repository or folder on disk.  This is useful for monolithic
checkpoints from places like CivitAI which contain a UNet + Text Encoders.

The ``dtype`` argument defaults to the value of ``--dtype`` and specifies the dtype
for the weights to be loaded in, for example: ``float32``, ``float16``, or ``bfloat16``.

The ``quantizer`` URI argument can be used to specify a quantization backend
for the text encoder using the same URI syntax as ``--quantizer``, this is supported
when loading from Hugging Face repo slugs / folders on disk, and when using the ``mode``
argument with monolithic (non-sharded) checkpoints.  This is not supported when
loading a submodule out of a combined checkpoint file with ``subfolder``.

The ``mode`` URI argument can be used to provide an additional hint about the loading
method for a single file checkpoint. 

Flux & T5 universal modes:

* ``clip-l`` for monolithic Flux CLIP-L checkpoints
* ``t5-xxl`` for monolithic T5 checkpoints (SD3 and Flux)

SD3 and SD3.5 specific modes:

* ``clip-l-sd3`` for SD3 CLIP-L checkpoints
* ``clip-l-sd35`` for SD3.5 CLIP-L checkpoints
* ``clip-l-sd35-large`` for SD3.5 large variant CLIP-L checkpoints
* ``clip-g-sd3`` for SD3 CLIP-G checkpoints
* ``clip-g-sd35`` for SD3.5 CLIP-G checkpoints
* ``clip-g-sd35-large`` for SD3.5 large variant CLIP-G checkpoints

These SD3/SD3.5 specific modes are designed with the correct architecture parameters for
each model variant.

diffusers usually shards T5 / large weights for performance, though monolithic checkpoints
are often available for use with ComfyUI or distributed on CivitAI. This is for compatibility 
with other software. The ``mode`` option is mutually exclusive with ``subfolder``.

Available encoder classes are:

* ``CLIPTextModel``
* ``CLIPTextModelWithProjection``
* ``T5EncoderModel``
* ``DistillT5EncoderModel`` (see: [LifuWang/DistillT5](https://huggingface.co/LifuWang/DistillT5))

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

@EXAMPLE[../../../examples/stablediffusion3/text_encoders/specify-encoders-config.dgen]

You may also use the URI value ``null``, to indicate that you do not want to ever load a specific text encoder at all.

For instance, you can prevent Stable Diffusion 3 from loading and using the T5 encoder all together.

@EXAMPLE[../../../examples/stablediffusion3/text_encoders/without-t5-config.dgen]


Any text encoder shared via the ``\use_modules`` directive in a config files is considered a default
value for the text encoder in the next pipeline that runs, using ``+`` will maintain this value
and using ``null`` will override it.

@EXAMPLE[../../../examples/stablediffusion3/text_encoders/share-encoders-config.dgen]


Monolithic CLIP-L, CLIP-G, and T5-XXL checkpoints (Used with Flux and SD3) can be loaded by utilizing the ``mode`` argument.

For instance, this can be used to load the Flux text encoders from ComfyUI style checkpoints,
which are also sometimes distributed alongside Flux transformer only checkpoints on CivitAI
with additional fine-tuning.

@EXAMPLE[../../../examples/flux/civitai/clip-L-T5-XXL-monolithic-config.dgen]

This can also be utilized with SD3.

@EXAMPLE[../../../examples/stablediffusion_3/civitai/clip-L-G-T5-XXL-monolithic-config.dgen]