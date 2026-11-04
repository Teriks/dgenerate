.. _specifying-text-encoders:

Specifying Text Encoders
========================

Diffusion pipelines supported by dgenerate may use a varying number of
text encoder sub models, currently up to 3. ``--model-type sd3``
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
* ``t5-xxl`` for monolithic Flux T5 checkpoints

SD3 and SD3.5 specific modes:

* ``clip-l-sd3`` for SD3/SD3.5 medium CLIP-L checkpoints
* ``clip-g-sd3`` for SD3/SD3.5 medium CLIP-G checkpoints
* ``t5-xxl-sd3`` for SD3/SD3.5 T5-XXL checkpoints
* ``clip-l-sd35-large`` for SD3.5 large variant CLIP-L checkpoints
* ``clip-g-sd35-large`` for SD3.5 large variant CLIP-G checkpoints

These SD3/SD3.5 specific modes are designed with the correct architecture parameters for
each model variant.

A pre-quantized text encoder loads with no ``mode`` and no ``quantizer``. Point ``model``
at a ``.gguf`` file or a ComfyUI quantized ``.safetensors`` file and name the encoder class
that matches it. dgenerate matches a configuration shipped with it from the file's
vocabulary, hidden size, and layer count. Quantized linears stay packed. This is how a
large encoder is loaded without the full-precision weights: a city96 T5-XXL or UMT5-XXL
GGUF, a llama.cpp Qwen, Mistral, or Gemma GGUF, or a ComfyUI FP8, INT8, or NVFP4
safetensors file of the same encoders. ``UMT5EncoderModel`` needs no ``mode``. CLIP and
the Flux / SD3 T5 files can still use ``mode``.

diffusers usually shards T5 / large weights for performance, though monolithic checkpoints
are often available for use with ComfyUI or distributed on CivitAI. This is for compatibility 
with other software. The ``mode`` option is mutually exclusive with ``subfolder``.

Available encoder classes are:

* ``CLIPTextModel``
* ``CLIPTextModelWithProjection``
* ``T5EncoderModel``
* ``UMT5EncoderModel`` (Wan)
* ``DistillT5EncoderModel`` (see: [LifuWang/DistillT5](https://huggingface.co/LifuWang/DistillT5))
* ``ChatGLMModel`` (Kolors)
* ``Gemma3ForConditionalGeneration``
* ``Gemma4UnifiedForConditionalGeneration`` (LTX-2.5)
* ``Mistral3ForConditionalGeneration`` (Flux.2)
* ``Qwen2_5_VLForConditionalGeneration`` (Qwen-Image, Kandinsky 5)
* ``Qwen2_5_VLTextModel`` (HunyuanVideo 1.5)
* ``Qwen3ForCausalLM`` (Flux.2 Klein)
* ``Qwen3Model`` (Z-Image)
* ``Qwen3VLForConditionalGeneration`` (Qwen-Image 2.1)
* ``Qwen3VLModel`` (Krea 2)

You can query the text encoder types and position for a model by passing ``help``
as an argument to ``--text-encoders`` or ``--second-model-text-encoders``. This feature
may not be used for both arguments simultaneously, and also may not be used
when passing ``help`` or ``helpargs`` to any ``--scheduler`` type argument.

.. code-block:: bash

    #!/usr/bin/env bash

    # ask for text encoder help on the main model that is mentioned

    dgenerate https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/sd3_medium_incl_clips.safetensors \
    --model-type sd3 \
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

@EXAMPLE[@PROJECT_DIR/examples/stablediffusion3/text_encoders/specify-encoders-config.dgen]

You may also use the URI value ``null``, to indicate that you do not want to ever load a specific text encoder at all.

For instance, you can prevent Stable Diffusion 3 from loading and using the T5 encoder all together.

@EXAMPLE[@PROJECT_DIR/examples/stablediffusion3/text_encoders/without-t5-config.dgen]


Any text encoder shared via the ``\use_modules`` directive in a config files is considered a default
value for the text encoder in the next pipeline that runs, using ``+`` will maintain this value
and using ``null`` will override it.

@EXAMPLE[@PROJECT_DIR/examples/stablediffusion3/text_encoders/share-encoders-config.dgen]


Monolithic CLIP-L, CLIP-G, and T5-XXL checkpoints (Used with Flux and SD3) can be loaded by utilizing the ``mode`` argument.

For instance, this can be used to load the Flux text encoders from ComfyUI style checkpoints,
which are also sometimes distributed alongside Flux transformer only checkpoints on CivitAI
with additional fine-tuning.

@EXAMPLE[@PROJECT_DIR/examples/flux/civitai/clip-L-T5-XXL-monolithic-config.dgen]

This can also be utilized with SD3.

Language model text encoders, such as ``Qwen3ForCausalLM`` (Flux.2 Klein),
``Qwen3Model`` (Z-Image), ``Mistral3ForConditionalGeneration`` (Flux.2),
``Qwen2_5_VLForConditionalGeneration`` (Qwen-Image, Kandinsky 5),
``Qwen2_5_VLTextModel`` (HunyuanVideo 1.5) and
``Gemma4UnifiedForConditionalGeneration`` (LTX-2.5), can be loaded from a
single ``.safetensors`` or ``.gguf`` file, with no ``mode`` or ``subfolder``. The model
configuration is matched from the file. A ComfyUI file trimmed to fewer decoder layers
than the full model, such as the Flux.2 Mistral encoder, builds that many layers.
LTX-2.5's pre-quantized Gemma 4 encoder is that class pointed at
``gemma4-12b-with-proj-ltx-2.5-Q4_K_M.gguf``,
``gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors``, or an NVFP4
file of the same layout. The file is already quantized. Do not set
``quantizer``. Connector tensors packed into the file are ignored; the
pipeline repository still supplies the connectors. On the dev transformer,
keep dynamic shifting, 30 steps, guidance 3, audio guidance 7, STG 1 on
block 28, and modality guidance 3. LTX-2's pre-quantized
Gemma 3 encoder is ``Gemma3ForConditionalGeneration`` pointed at
``gemma_3_12B_it_fp8_scaled.safetensors`` or
``gemma_3_12B_it_fp4_mixed.safetensors`` from ``Comfy-Org/ltx-2``. The file
is already quantized. Do not set ``quantizer``. The parent is
``Lightricks/LTX-2``. On the 19B dev transformer, keep 30 steps, guidance 4,
and audio guidance 4; that parent scheduler already uses dynamic shifting.
The original all-in-one LTX-2 checkpoints, including
``ltx-2-19b-dev-fp8``, are not supported.

``T5EncoderModel`` and ``UMT5EncoderModel`` load the same way from a ``.gguf`` file or
from a monolithic ``.safetensors`` file. A GGUF file uses llama.cpp tensor names; those
are renamed to the Hugging Face names on load. Quantized linears stay packed. T5 keeps
its feed-forward output projection dense, because T5 runs that projection in float32.

ComfyUI quantized text encoder files (FP8, INT8, NVFP4, FP4, W4A8) work with both
of these, and with the ``mode`` argument for CLIP and T5. Their linears stay
quantized. Where ComfyUI publishes a 4-bit file, the examples include it next
to the FP8 or INT8 file. T5-XXL and UMT5 have no 4-bit ComfyUI file; those
examples use the Q4_K_S GGUF. See :ref:`gguf-checkpoints` and
:ref:`comfyui-quantized-checkpoints`.

@EXAMPLE[@PROJECT_DIR/examples/flux/gguf/text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/flux/comfy_quant/text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/stablediffusion3/comfy_quant/text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/ltx_video/gguf/text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/ltx2/comfy_quant/ltx25-distilled-text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/ltx2/comfy_quant/ltx25-distilled-text-encoder-nvfp4-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/ltx2/gguf/ltx25-distilled-text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/ltx2/gguf/ltx25-dev-text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/ltx2/comfy_quant/ltx25-dev-text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/ltx2/comfy_quant/ltx25-dev-text-encoder-nvfp4-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/ltx2/gguf/ltx2-distilled-text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/ltx2/gguf/ltx2-distilled-text-encoder-fp4-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/ltx2/gguf/ltx2-dev-text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/ltx2/gguf/ltx2-dev-text-encoder-fp4-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/wan/gguf/text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/wan/gguf/text-encoder-wan22-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/wan/comfy_quant/text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/wan/comfy_quant/text-encoder-wan22-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/qwen_image/comfy_quant/text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/qwen_image/comfy_quant/text-encoder-nvfp4-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/flux2/comfy_quant/dev-text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/flux2/comfy_quant/dev-text-encoder-fp4-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/krea2/gguf/text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/krea2/comfy_quant/text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/qwen_image_21/gguf/text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/qwen_image_21/comfy_quant/text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/qwen_image_21/comfy_quant/text-encoder-w4a8-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/qwen_image/comfy_quant/edit-text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/qwen_image/comfy_quant/edit-text-encoder-nvfp4-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/z_image/comfy_quant/text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/z_image/comfy_quant/text-encoder-fp4-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/flux2/gguf/klein-9b-text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/flux2/comfy_quant/klein-9b-text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/flux2/comfy_quant/klein-9b-text-encoder-fp4-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/kandinsky5/comfy_quant/text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/kandinsky5/comfy_quant/text-encoder-nvfp4-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/hyvideo_15/comfy_quant/text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/hyvideo_15/comfy_quant/text-encoder-nvfp4-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/hyvideo_15/gguf/text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/hyimage_21/gguf/text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/hyimage_21/comfy_quant/text-encoder-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/hyimage_21/comfy_quant/text-encoder-nvfp4-config.dgen]

@EXAMPLE[@PROJECT_DIR/examples/stablediffusion3/singlefile/clip-L-G-T5-XXL-monolithic-config.dgen]