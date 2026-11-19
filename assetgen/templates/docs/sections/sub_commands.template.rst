Sub Commands
============

dgenerate implements additional functionality through the option ``--sub-command``.

For a list of available sub-commands use ``--sub-command-help``, which by default
will list available sub-command names.

For additional information on a specific sub-command use ``--sub-command-help NAME``

Multiple sub-command names can be specified to ``--sub-command-help`` if desired.

All sub-commands respect the ``--plugin-modules`` and ``--verbose`` arguments
even if their help output does not specify them, these arguments are handled
by dgenerate and not the sub-command.

Sub Command: image-process
--------------------------

The ``image-process`` sub-command can be used to run image processors implemented
by dgenerate on any file of your choosing including animated images and videos.

It has a similar but slightly different design/usage to the main dgenerate
command itself.

It can be used to run canny edge detection, openpose, etc. on any image or
video/animated file that you want.

The help output of ``image-process`` is as follows:


@COMMAND_OUTPUT[dgenerate --sub-command image-process --help]


Overview of specifying ``image-process`` inputs and outputs

.. code-block:: bash

    #!/usr/bin/env bash

    # Overview of specifying outputs, image-process can do simple operations
    # like resizing images and forcing image alignment with --align, without the
    # need to specify any other processing operations with --processors. Running
    # image-process on an image with no other arguments simply aligns it to 8 pixels,
    # given the defaults for its command line arguments

    # More file formats than .png are supported for static image output, all
    # extensions mentioned in the image-process --help documentation for --frame-format
    # are supported, the supported formats are identical to that mentioned in the --image-format
    # option help section of dgenerate's --help output

    # my_file.png -> my_file_processed_1.png

    dgenerate --sub-command image-process my_file.png --resize 512x512

    # my_file.png -> my_file.png (overwrite)

    dgenerate --sub-command image-process my_file.png --resize 512x512 --output-overwrite

    # my_file.png -> my_file.png (overwrite)

    dgenerate --sub-command image-process my_file.png -o my_file.png --resize 512x512 --output-overwrite

    # my_file.png -> my_dir/my_file_processed_1.png

    dgenerate --sub-command image-process my_file.png -o my_dir/ --resize 512x512 --no-aspect

    # my_file_1.png -> my_dir/my_file_1_processed_1.png
    # my_file_2.png -> my_dir/my_file_2_processed_2.png

    dgenerate --sub-command image-process my_file_1.png my_file_2.png -o my_dir/ --resize 512x512

    # my_file_1.png -> my_dir_1/my_file_1_processed_1.png
    # my_file_2.png -> my_dir_2/my_file_2_processed_2.png

    dgenerate --sub-command image-process my_file_1.png my_file_2.png \
    -o my_dir_1/ my_dir_2/ --resize 512x512

    # my_file_1.png -> my_dir_1/renamed.png
    # my_file_2.png -> my_dir_2/my_file_2_processed_2.png

    dgenerate --sub-command image-process my_file_1.png my_file_2.png \
    -o my_dir_1/renamed.png my_dir_2/ --resize 512x512


A few usage examples with processors:

.. code-block:: bash

    #!/usr/bin/env bash

    # image-process can support any input format that dgenerate itself supports
    # including videos and animated files. It also supports all output formats
    # supported by dgenerate for writing videos/animated files, and images.

    # create a video rigged with OpenPose, frames will be rendered to the directory "output" as well.

    dgenerate --sub-command image-process my-video.mp4 \
    -o output/rigged-video.mp4 --processors "openpose;include-hand=true;include-face=true"

    # Canny edge detected video, also using processor chaining to mirror the frames
    # before they are edge detected

    dgenerate --sub-command image-process my-video.mp4 \
    -o output/canny-video.mp4 --processors mirror "canny;blur=true;threshold-algo=otsu"


Sub Command: civitai-links
--------------------------

The ``civitai-links`` sub-command can be used to list the hard links for models available on a CivitAI model page.

These links can be used directly with dgenerate, it will automatically download the model for you.

You only need to select which models you wish to use from the links listed by this command.

See: `Utilizing CivitAI links and Other Hosted Models`_ for more information about how to use these links.

To get direct links to CivitAI models you can use the ``civitai-links`` sub-command
or the ``\civitai_links`` directive inside of a config to list all available models
on a CivitAI model page.

For example:

.. code-block:: bash

    #!/usr/bin/env bash

    # get links for the Crystal Clear XL model on CivitAI

    dgenerate --sub-command civitai-links "https://civitai.com/models/122822?modelVersionId=133832"

    # you can also automatically append your API token to the end of the URLs with --token
    # some models will require that you authenticate to download, this will add your token
    # to the URL for you

    dgenerate --sub-command civitai-links "https://civitai.com/models/122822?modelVersionId=133832" --token $MY_API_TOKEN


This will list every model link on the page, with title, there may be many model links
depending on what the page has available for download.

Output from the above example:

.. code-block:: text

    Models at: https://civitai.com/models/122822?modelVersionId=133832
    ==================================================================

    CCXL (Model): https://civitai.com/api/download/models/133832?format=SafeTensor&size=full&fp=fp16


Sub Command: auto1111-metadata
------------------------------

The ``auto1111-metadata`` sub-command can be used to add Automatic1111 style metadata to an image
generated by dgenerate.

This same metadata can be added automatically when generating  images using the dgenerate
option ``--output-auto1111-metadata``.

You must use the dgenerate options ``--output-configs`` or ``--output-metadata`` for this to work.

The dgenerate option ``--output-configs`` will write the generation config to a file in the output directory,
which can be read by this sub-command.

Alternatively, you can use the dgenerate option ``--output-metadata`` to write the metadata directly to the image file,
which can then be read out of the PNG metadata (DgenerateConfig), or EXIF UserComment if you are using jpeg.

This sub-command supports PNG and JPEG files only.

This sub-command also exists as the config directive: ``\auto1111_metadata``.

The help output of ``image-process`` is as follows:

@COMMAND_OUTPUT[dgenerate --sub-command auto1111-metadata --help]


Example of using the ``auto1111-metadata`` sub-command with ``--output-metadata``


.. code-block:: bash

    #!/usr/bin/env bash

    # in this example, we are using --output-metadata to write the dgenerate config to the image file itself.

    dgenerate stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --inference-steps 30 \
    --guidance-scales 5 \
    --clip-skips 0 \
    --seeds 0 \
    --output-metadata \
    --output-path output \
    --output-size 512x512 \
    --prompts "hello world!"

    # Make a copy of the image, with Automatic1111 metadata added to it

    dgenerate --sub-command auto1111-metadata output\s_0_g_5-0_i_30_cs_0_step_1.png --output image_with_auto1111_metadata.png

    # Overwrite the image, dgenerates original metadata (the config) will be lost

    dgenerate --sub-command auto1111-metadata output\s_0_g_5-0_i_30_cs_0_step_1.png


You can also use the output from ``--output-configs`` for this task


.. code-block:: bash

    #!/usr/bin/env bash

    # in this example, we are using --output-configs to write the dgenerate config to a file next to the image.

    dgenerate stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --inference-steps 30 \
    --guidance-scales 5 \
    --clip-skips 0 \
    --seeds 0 \
    --output-configs \
    --output-path output \
    --output-size 512x512 \
    --prompts "hello world!"

    # Make a copy of the image, with Automatic1111 metadata added to it

    dgenerate --sub-command auto1111-metadata output\s_0_g_5-0_i_30_cs_0_step_1.png --config output\s_0_g_5-0_i_30_cs_0_step_1.dgen --output image_with_auto1111_metadata.png

    # Overwrite the image, dgenerates original metadata (the config) will be lost

    dgenerate --sub-command auto1111-metadata output\s_0_g_5-0_i_30_cs_0_step_1.png --config output\s_0_g_5-0_i_30_cs_0_step_1.dgen



Sub Command: to-diffusers
--------------------------

The ``to-diffusers`` sub-command can be used to convert single file diffusion model checkpoints from CivitAI
and elsewhere into diffusers format (a folder on disk with configuration).

This can be useful if you want to load a single file checkpoint with quantization.

dgenerate does this conversion for you automatically when necessary and caches the result on disk,
see: `Checkpoint Conversion Cache`_ for more information about where these files are cached.

You may also save models loaded from Hugging Face repos.

This sub-command also exists as the config directive: ``\to_diffusers``

In memory caching / memoization is disabled for this command to prevent unnecessary resource usage,
the models involved with the loaded pipeline are garbage collected immediately after the conversion happens.

.. code-block:: text

    #!/usr/bin/env bash

    # convert a CivitAI checkpoint (https://civitai.com/models/2711/21-sd-modern-buildings-style-md)
    # into a diffusers compatible model folder, containing separate checkpoint files for each
    # model component and related configuration

    dgenerate --sub-command to-diffusers \
    "https://civitai.com/api/download/models/3002?type=Model&format=PickleTensor&size=full&fp=fp16" \
    --model-type sd \
    --dtypes float16 float32 \
    --output modern_buildings


The help output of ``to-diffusers`` is as follows:

@COMMAND_OUTPUT[dgenerate --sub-command to-diffusers --help]


Sub Command: quantize
---------------------

The ``quantize`` sub-command reads a diffusers model and writes quantized
weights. This sub-command also exists as the config directive: ``\quantize``

The model is a folder on disk, a Hugging Face repository, one weight
file, or a blob link to one weight file.
``--subfolders`` selects component folders such as ``transformer`` and
``text_encoder``. Without it, every component that has weights is quantized.
Tokenizers and schedulers have no weights, so they are left out.
``--weight-name`` selects a file inside that model, such as
``model.fp16.safetensors``. A repo-relative path such as
``text_encoder/model.fp16.safetensors`` selects that one file. The same name
is used in every ``--subfolders`` component.

``--formats`` is the list of quants to write, and every name in it is written
in one run. A name can be a GGUF type (``Q4_K``, ``Q8_0``, ``Q6_K``, and the
rest that dgenerate loads), a ComfyUI safetensors quant (``float8_e4m3fn``,
``int8_tensorwise``, ``nvfp4``, ``mxfp8``, ``convrot_w4a4``, ``asym_w4a8_int8``,
``w6a8_int8``, ``float8_e5m2``), a short name, or a preset.

Short names: ``q4`` is ``Q4_K``, ``q8`` is ``Q8_0``, ``fp8`` is
``float8_e4m3fn``, ``int8`` is ``int8_tensorwise``, ``int4`` is ConvRot INT4,
``w4a8`` and ``w6a8`` are the grouped Comfy quants. ``Q4_0`` is requested by
that name. ``--list-formats`` prints the full set.

``f16``, ``fp16``, and ``F16`` are one format. ``f32``, ``fp32``, and ``F32``
are another. ``bf16`` and ``BF16`` are a third. Each writes a GGUF in which
every tensor is that dtype: linears, biases, norms, and embeddings. Nothing
is packed. The file is a conversion of the component, and it is about the
size of the weights in that dtype. ``float16`` and ``float32`` are the
safetensors form of the same idea. Every floating tensor is cast to that
dtype, integer tensors are left as they are, and the file has no quant
metadata. ``fp8`` is not one of these. ``fp8`` is the ``float8_e4m3fn`` quant.

The container is part of the name:

* ``f16``, ``fp16``, or ``F16`` writes ``transformer-F16.gguf``. Every tensor is F16.
* ``f32``, ``fp32``, or ``F32`` writes ``transformer-F32.gguf``. Every tensor is F32.
* ``bf16`` or ``BF16`` writes ``transformer-BF16.gguf``. Every tensor is BF16.
* ``float16`` writes ``transformer-float16.safetensors``. Floating tensors are
  float16.
* ``float32`` writes ``transformer-float32.safetensors``. Floating tensors are
  float32.

``--llama`` accepts the GGUF floats and refuses ``float16`` and ``float32``.
A text encoder becomes ``text_encoder-F16-llama.gguf``. A Gemma 4 or Qwen
vision tower is written beside it as ``mmproj-text_encoder-F16-llama.gguf``,
in that same dtype, and the language file still stores the llama.cpp
hyperparameters and tokenizer. dgenerate loads that GGUF the same way it loads
a quantized one. Point ``--text-encoders`` at the file and leave ``quantizer``
unset. The weights are already F16 or F32, so nothing is unpacked.

``--comfy`` accepts both containers. ``--formats f16 --comfy`` writes
``transformer-F16-comfy.gguf`` for an architecture ComfyUI-GGUF loads: FLUX.1,
SD1, SDXL, SD3, Wan, Qwen-Image, MiniMax-H3 video and image, LTX-Video, and
the GGUF text encoders. ``--formats float16 --comfy`` writes
``transformer-float16-comfy.safetensors``, which is what ComfyUI's usual
loader reads. FLUX.2, FLUX.3, Z-Image, LTX-2, Krea 2, Hunyuan-DiT,
HunyuanVideo 1.5, HunyuanImage 2.1, Kandinsky 5, Stable Cascade, CLIP, and
Gemma 4 have no ComfyUI-GGUF architecture, so the safetensors file is the
conversion for those.
dgenerate loads the GGUF as it loads any other ComfyUI GGUF. It loads the
safetensors file as an ordinary full-precision checkpoint, not as a quantized
one. A component with no key map is still an error.

Without ``--llama`` and without ``--comfy``, the file keeps the tensor names
from the component folder. Load a diffusers-named GGUF with that component's
option and leave the original repository as the model, so the config still
comes from there.

Presets: ``dgenerate`` writes ``Q4_K`` and ``Q8_0``. ``comfy`` writes FP8 and
INT8. ``gguf`` writes every quantized GGUF type. It leaves out ``F16``,
``F32``, and ``BF16``. ``comfy-all`` writes every Comfy safetensors quant. It
leaves out ``float16`` and ``float32``. Ask for the plain formats by name.

Files keep the tensor names of the component they came from, unless ``--comfy``
is set. ``--comfy`` writes the names ComfyUI and ComfyUI-GGUF load. That covers
FLUX.1, SD 1.5, SDXL, SD3, Wan, Qwen-Image, MiniMax-H3 video and image, and
LTX-Video, plus safetensors quants of FLUX.2, FLUX.3, Z-Image, LTX-2, Krea 2,
Hunyuan-DiT, HunyuanVideo 1.5, HunyuanImage 2.1, Kandinsky 5, and Stable
Cascade. MiniMax-H3 video uses ``--subfolders transformer`` for text and
keyframes, or ``transformer_ref`` for references. The MiniMax-H3-Image
``transformer`` folder is already Comfy names; ``transformer_diffusers`` is
the Diffusers copy. GGUF text encoders cover T5, UMT5, Llama 3 / Mistral,
Qwen2, Qwen3, and Gemma 3. A Qwen2-VL vision tower is written beside the text
encoder as ``mmproj-text_encoder-Q8_0-comfy.gguf``. ComfyUI-GGUF pairs that
file by name, and it picks one mmproj when several quants share a folder, so
write one GGUF format per folder when the model has a vision tower. CLIP is
written as safetensors, which is what Comfy's CLIP loader reads.
ComfyUI-GGUF has no architecture for FLUX.2, FLUX.3, Z-Image, LTX-2, Krea 2,
Hunyuan-DiT, HunyuanVideo 1.5, HunyuanImage 2.1, Kandinsky 5, Stable Cascade,
CLIP, or Gemma 4, so those GGUF files are refused. Those files are named like
``transformer-Q4_K-comfy.gguf``. A component with no key map is an error, so
the command does not write a file Comfy cannot load.

``--llama`` writes llama.cpp tensor names instead, the layout Unsloth
text-encoder GGUFs use and dgenerate loads. That covers T5, UMT5, Llama,
Qwen, and Gemma, including Gemma 4. A full Gemma 4 model, such as the
LTX-2.5 prompt enhancer, writes its vision and audio towers as a CLIP
``mmproj`` (``gemma4v`` and ``gemma4a``) beside the language file. Qwen3.5,
including the FLUX.3 prompt upsampler, is written as ``qwen35``. Its vision
tower stays a Qwen3-VL mmproj. A Llama whose head counts are not the
32 and 8 ComfyUI-GGUF hardcodes is included. Files are named like
``text_encoder-Q4_K-llama.gguf``. ``--llama`` and ``--comfy`` cannot be used
together, and ``--llama`` applies to GGUF formats. When ``config.json`` is
beside the weights, the file stores the llama.cpp hyperparameters. A BPE
``tokenizer.json`` (or ``vocab.json`` and ``merges.txt``) and
``chat_template.json`` are stored too, from that folder or from the sibling
``tokenizer`` folder. A tied embedding is also written as ``output.weight``.
dgenerate loads either layout: a Llama or Mistral file written without
``--llama`` keeps Hugging Face query and key order, and a ``--llama`` file
is restored on load.

``--quantizer`` writes an SDNQ or bitsandbytes diffusers repository from the
same URI used when generating images. Quote it so the shell keeps the
semicolons. ``sdnq;type=uint4;group-size=32;quant-conv=true;use-svd=true`` and
``bnb;bits=4;bits4-quant-type=nf4;bits4-use-double-quant=true`` are written as
separate folders named from the options that differ from the URI default.
Several URIs write several repositories. Bitsandbytes quantization runs on
CUDA. Point dgenerate at the folder and leave its ``--quantizer`` unset.
Components you did not select are copied at full precision, including
``model_index.json``. ``--comfy`` applies to the GGUF and Comfy safetensors
files; these repositories keep diffusers tensor names.

``--hf-upload user/name`` uploads the written files to that Hugging Face model
repository after the quant finishes. A new repository is public unless
``--hf-private`` is set or the account default is private. An existing
repository keeps its visibility.

One ``--quantizer`` repository and no ``--formats`` is uploaded at the repo
root, so dgenerate can load it as a model. Several files keep the names
written under ``--output``.

The token is ``--auth-token``, ``HF_TOKEN``, or a saved ``huggingface-cli``
login.

``--hf-upload-card`` writes a ``README.md`` model card. With no value, the
card is built from the source model, the formats, every quantizer setting,
and the command that produced the files. Quote a value when it contains
spaces. Without this option, the upload has no model card.

* ``extra:TEXT`` and ``extra-file:PATH`` insert that text into the generated
  card, after the introduction.
* ``file:PATH`` uses that UTF-8 file as the whole card.
* ``text:TEXT`` uses that text as the whole card.

``--hf-upload-card-preview`` prints that card and does not quantize or upload.
Pass ``--output`` to also write ``README.md`` there. With no
``--hf-upload-card``, the preview is the generated card. ``extra:`` and
``extra-file:`` are inserted into that card. ``file:`` and ``text:`` replace
it. Use the preview to read the card before a long quant or a publish.

A single file is named from its parent folder when the filename is a generic
component weight, so ``text_encoder/model.fp16.safetensors`` is written as
``text_encoder-Q4_K.gguf``. ``config.json`` beside that file is read when it
is there. A blob link to an index also downloads the shards that index names.
``--quantizer`` still needs the whole diffusers model, and it honors
``--variant`` rather than ``--weight-name``.

Point a diffusers-named file at the matching component option and leave the
original repository as the model, so the config still comes from there. A GGUF
text encoder is read for T5, UMT5, Qwen, Mistral, Gemma, and CLIP. CLIP GGUF
and a Comfy CLIP safetensors quant keep Hugging Face names, so either file
loads with ``--text-encoders``. ``--comfy`` leaves those safetensors names in
place. It does not write a CLIP GGUF, because ComfyUI-GGUF has no CLIP
architecture.

A quantized GGUF keeps tensors that cannot take the block size. A matrix whose
width is not a multiple of the quant block is stored as BF16, and biases and
norms are stored as F32. ``F16``, ``F32``, and ``BF16`` do not do that. Every
tensor in that file is the dtype you named. A Comfy quant that fits no linear
in a component is skipped. ``float16`` and ``float32`` are written whenever
the component has tensors.

.. code-block:: bash

    # Q4_K, Q8_0, and FP8 of the transformer
    dgenerate --sub-command quantize black-forest-labs/FLUX.1-dev \
        --subfolders transformer --formats q4 q8 fp8 -o quants

    # The short dgenerate pair and the short Comfy pair, every component
    dgenerate --sub-command quantize ./model --formats dgenerate comfy -o quants

    # SDNQ uint4 with a group size and a low-rank residual, and 4-bit NF4
    dgenerate --sub-command quantize ./model -o quants \
        --quantizer "sdnq;type=uint4;group-size=32;use-svd=true" \
        "bnb;bits=4;bits4-quant-type=nf4"

    # Q4_K and FP8 with ComfyUI tensor names
    dgenerate --sub-command quantize ./model --subfolders transformer \
        --formats q4 fp8 --comfy -o quants

    # llama.cpp names for a text encoder, the Unsloth GGUF layout
    dgenerate --sub-command quantize ./model --subfolders text_encoder \
        --formats q4 --llama -o quants

    # llama.cpp F16, no quant: text_encoder-F16-llama.gguf
    dgenerate --sub-command quantize ./model --subfolders text_encoder \
        --formats fp16 --llama -o quants

    # ComfyUI-GGUF F16, no quant: transformer-F16-comfy.gguf
    dgenerate --sub-command quantize ./model --subfolders transformer \
        --formats f16 --comfy -o quants

    # ComfyUI safetensors FP16, no quant: transformer-float16-comfy.safetensors
    # FLUX.2, Z-Image, CLIP, and Gemma 4 use this file
    dgenerate --sub-command quantize ./model --subfolders transformer \
        --formats float16 --comfy -o quants

    # One file inside a component, or a blob link to that file
    dgenerate --sub-command quantize ./model --subfolders text_encoder \
        --weight-name model.fp16.safetensors --formats q4 -o quants
    dgenerate --sub-command quantize \
        "https://huggingface.co/org/repo/blob/main/text_encoder/model.fp16.safetensors" \
        --formats q4 -o quants

    # Names, short names, and presets
    dgenerate --sub-command quantize --list-formats

    # Publish Q4_K with a generated model card. --hf-upload-card-preview
    # prints the card and does not quantize or upload.
    dgenerate --sub-command quantize ./model --subfolders transformer \
        --formats q4 -o quants --hf-upload user/model-q4 --hf-upload-card
    dgenerate --sub-command quantize ./model --hf-upload-card-preview \
        --quantizer "sdnq;type=uint4;group-size=32;use-svd=true"

    # Use a file, or the text after text:, as the model card
    dgenerate --sub-command quantize ./model --formats q4 -o quants \
        --hf-upload user/model-q4 --hf-upload-card file:README.md
    dgenerate --sub-command quantize ./model --formats q4 -o quants \
        --hf-upload user/model-q4 --hf-upload-card "text:# My quant"

    # Add a note to the generated card
    dgenerate --sub-command quantize ./model --formats q4 -o quants \
        --hf-upload user/model-q4 --hf-upload-card "extra:Use guidance 3.5."
    dgenerate --sub-command quantize ./model --formats q4 -o quants \
        --hf-upload user/model-q4 --hf-upload-card extra-file:notes.md

The help output of ``quantize`` is as follows:

@COMMAND_OUTPUT[dgenerate --sub-command quantize --help]


Sub Command: prompt-upscale
---------------------------

The ``prompt-upscale`` sub-command can be use to run on prompt texts without invoking image generation.

See: `Prompt Upscaling`_ for more information about prompt upscaling plugins.

This sub-command is designed in the same vein as ``dgenerate --sub-command image-process`` and the ``\image_process`` directive.

This sub-command also exists as the config directive: ``\prompt_upscale``

It allows you to output the prompts in various formats such as plain text, or structured json, toml, and yaml.

Prompts can be written to a file or printed to stdout, and in the case of the config directive ``\prompt_upscale``
they can also be written to a config template variable as a python list.

A comprehensive example of the ``\prompt_upscale`` config directive which might be helpful for understanding
this sub-commands functionality is available in the `examples folder <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/config_directives/prompt_upscale/prompt-upscale-directive-config.dgen>`_.

.. code-block:: text

    #!/usr/bin/env bash

    # upscale two prompts with magic prompt
    # using the default accelerator for your system
    # and print them as structured yaml to stdout

    dgenerate --sub-command prompt-upscale \
    --prompts \
    "a cat sitting on a bench in a park" \
    "a dog sitting on a bench in a park" \
    --upscaler magicprompt;variations=10 -of yaml

The help output of ``prompt-upscale`` is as follows:

@COMMAND_OUTPUT[dgenerate --sub-command prompt-upscale --help]


Sub Command: assistant
----------------------

The ``assistant`` sub-command writes a dgenerate config from a plain language request.
This sub-command also exists as the config directive: ``\assistant``

A local Qwen model, running through the ``xllamacpp`` extra, writes the config. It is
guided by the closest example configs and documentation, retrieved from an index of
the dgenerate examples and docs that ships with dgenerate. The config is then checked
with dgenerate's config runner and argument parser, without loading any diffusion
models, and the model is asked to fix anything dgenerate rejects.

Install the extra with ``pip install dgenerate[xllamacpp]``. Without it the command
exits with status 1.

The request is the positional text after the options. When it is omitted, the request
is read from stdin. On a terminal you are prompted to type it, then press Ctrl+Z and
Enter on Windows, or Ctrl+D on Linux or macOS. An empty request exits with status 2.

The finished config is printed to stdout. ``-o`` / ``--output`` writes it to a file
instead. Paths named in the request are input files. Relative paths are relative to
the current directory, and when ``--output`` is set they are rewritten relative to
that file. A path that does not exist is reported on stderr and left in the config
so you can see it.

The chat and embedding models are downloaded from Hugging Face the first time they
are used. ``--model`` and ``--embed-model`` accept only the supported Qwen GGUFs
below. A path to some other file is rejected. Each embedding model has an index
packaged with dgenerate, so an install does not need the examples directory to
retrieve documentation.

Supported ``--model`` values are every non-BF16 quant of Unsloth's Qwen 3.8 27B
chat GGUF, smallest download first. The default is
``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf``. GPU labels are a rough
fit (file size plus room for context).

* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ1_S.gguf`` (6.2 GB, Qwen 3.8, 8 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ1_M.gguf`` (6.7 GB, Qwen 3.8, 8 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_XXS.gguf`` (7.3 GB, Qwen 3.8, 8 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_S.gguf`` (8.4 GB, Qwen 3.8, 12 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf`` (9.8 GB, Qwen 3.8, 12 GB GPUs, default)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf`` (10.9 GB, Qwen 3.8, 12 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf`` (12.0 GB, Qwen 3.8, 16 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf`` (13.1 GB, Qwen 3.8, 16 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf`` (14.3 GB, Qwen 3.8, 16 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_S.gguf`` (15.4 GB, Qwen 3.8, 16 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf`` (16.1 GB, Qwen 3.8, 24 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf`` (16.5 GB, Qwen 3.8, 24 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_1.gguf`` (17.5 GB, Qwen 3.8, 24 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf`` (17.6 GB, Qwen 3.8, 24 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q5_K_S.gguf`` (18.7 GB, Qwen 3.8, 24 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q5_K_M.gguf`` (19.8 GB, Qwen 3.8, 24 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q5_K_XL.gguf`` (20.9 GB, Qwen 3.8, 24 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf`` (22.0 GB, Qwen 3.8, 24 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K_M.gguf`` (23.1 GB, Qwen 3.8, 24 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K_L.gguf`` (24.2 GB, Qwen 3.8, 32 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K_XL.gguf`` (25.3 GB, Qwen 3.8, 32 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_L.gguf`` (28.0 GB, Qwen 3.8, 32 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf`` (29.0 GB, Qwen 3.8, 32 GB GPUs)
* ``unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf`` (31.5 GB, Qwen 3.8, 32 GB GPUs)

Supported ``--embed-model`` values. The default is
``Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf``.

* ``Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf`` (0.6 GB, default)
* ``Qwen/Qwen3-Embedding-4B-GGUF/Qwen3-Embedding-4B-Q8_0.gguf`` (4.3 GB, stronger retrieval)
* ``Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf`` (8.1 GB, strongest retrieval)

The Console UI Assistant window offers the same lists and does not accept a
model file of your own.

``--ctx`` is the chat context size in tokens, 49152 by default. ``--gpu-layers``
is how many layers go on the GPU: ``-1`` lets the loader decide, and ``0`` is CPU
only. Sampling follows the Qwen 3.8 recommendations on the Unsloth model card, and
the values that differ follow ``--think``. Without it, temperature is 0.7, top-p
is 0.8, and the presence penalty is 1.5. With ``--think``, temperature is 1.0,
top-p is 0.95, and the presence penalty is 0. Top-k is 20, min-p is 0, and the
repeat penalty is 1 either way. ``--temperature``, ``--top-p``, ``--top-k``,
``--min-p``, ``--presence-penalty``, and ``--repeat-penalty`` each override that
one default when you pass them. ``--max-tokens`` limits the reply. ``0``,
the default, uses whatever room is left in the context window and does not allocate
a larger cache. If the instructions cannot fit in ``--ctx``, with room left for
the reply, the command stops before loading a model. If the retrieved examples
make the prompt longer than the window, it stops before loading the chat model.

``--think`` lets the model reason before it writes the config, which is slower.
``--reasoning-effort`` is ``low``, ``medium``, or ``xhigh``, and only applies with
``--think``. The default is ``medium``. Qwen 3.8 uses extra high when the effort
is left unset in its own template; this command always sends the effort you choose.

``--no-check`` skips the dgenerate validation. Otherwise the config is checked, and
``--max-repairs`` (default 2) is how many times the model is asked to fix a config
that dgenerate rejects. Problems and notes are printed to stderr. A config that
still has problems is written anyway, with a warning to review it before running.
Status 0 means a config was written. A missing model or a failure while loading one
exits with status 1.

``--show-context`` prints the retrieved examples and documentation to stderr.
``--progress-plain``, ``--progress-thought``, ``--progress-talk``, and ``--progress-tool``
are terminal color sequences written before status text, reasoning, the model's reply,
and tool calls. They are off by default. ``\033[0m`` resets the color, ``\033[32m``
is green, ``\033[36m`` is cyan, and ``\033[33m`` is yellow.

``-ofm`` / ``--offline-mode`` uses only models already on disk or in the Hugging Face cache.
In this mode, the ``fetch_link`` tool lists Hugging Face files already in that cache and does not call the Hub.
A quant that is not cached is not listed, so the example file is kept.
It does not fetch a CivitAI link or any other URL.
A CivitAI page is not in the local cache, so the config is written from the request
and the retrieved examples.
Passing ``-ofm`` / ``--offline-mode`` before ``--sub-command`` does the same thing.

``-v`` / ``--verbose`` shows llama.cpp output on stderr.

The Console UI Assistant window runs this sub-command. Help in that window
describes the request, including pasted CivitAI and Hugging Face links.
Its model list is the same set of Qwen GGUFs. The Context tokens field is
``--ctx``, saved with the other Assistant options, and 49152 by default.
It can be raised up to the
model's native 262144 tokens. Extra VRAM, beside that field, is the additional
video memory the context cache needs at the current token count. It updates as
the field changes. 49152 tokens is 3 GiB, and 262144 tokens is 16 GiB.

.. code-block:: bash

    #!/usr/bin/env bash

    # print a config to the terminal

    dgenerate --sub-command assistant "use ltx 2 with the canny ic lora to turn clip.gif into a dancing fox"

    # write it to a file; clip.gif is looked up in the current directory
    # and the path in the config is relative to fox.dgen

    dgenerate --sub-command assistant -o fox.dgen "use ltx 2 with the canny ic lora to turn clip.gif into a dancing fox"

    # read the request from stdin

    echo "generate a cute cat, 30 steps" | dgenerate --sub-command assistant -o cat.dgen

    # stay offline, and let the model think before answering

    dgenerate --offline-mode --sub-command assistant --think --reasoning-effort medium \
    -o cat.dgen "generate a cute cat, 30 steps"

The help output of ``assistant`` is as follows:

@COMMAND_OUTPUT[dgenerate --sub-command assistant --help]


Sub Command: kernel-warmup
--------------------------

The ``kernel-warmup`` sub-command downloads the quant kernels for this computer.
This sub-command also exists as the config directive: ``\kernel_warmup``

ComfyUI quantized checkpoints fetch a platform-specific kernel from the
Hugging Face Hub the first time they run. This command fetches the kernels
for the current accelerator, and no others, so a later run can stay offline.

CUDA and Intel XPU ComfyUI matmuls use kernels that ship with PyTorch, so there
is nothing to download for those. GGUF linears unpack with Triton on CUDA, ROCm,
and XPU. Apple Metal downloads the MLX quant kernel. AMD ROCm downloads the
AITER kernel.

The network installer runs this after a new install. An older dgenerate that
does not provide the subcommand skips the step.

If the Hub cannot be reached, the command prints ``Cannot download platform
specific quant kernels, you are not online.`` and exits with status 1.

.. code-block:: bash

    dgenerate --sub-command kernel-warmup

The help output of ``kernel-warmup`` is as follows:

@COMMAND_OUTPUT[dgenerate --sub-command kernel-warmup --help]


Sub Command: transpile
----------------------

The ``transpile`` sub-command reads a dgenerate config and writes a Python script
that runs the same actions through dgenerate's Python API. This sub-command also
exists as the config directive: ``\transpile``

A generation line
becomes a ``DgenerateArguments`` object. ``{% if %}``, ``{% for %}``,
``{% set %}``, ``{% filter %}``, and ``{% with %}`` become Python. Shell-parsed directives such as ``\image_process`` and
``\download`` become calls that pass already split arguments. ``\set``,
``\setp``, ``\print``, and ``\echo`` stay on their own handlers.

The default script keeps the config interpreter's line numbers and error text.
``--relaxed`` writes the same actions without that bookkeeping, so the script
is easier to read and edit. ``--format`` wraps long lists and calls and leaves
a blank line between invocations. Config comments are left out. ``--comments``
keeps comments on their own line and comments at the end of a line. A syntax
error in the config stops translation. Both modes run the config. Neither mode
loads a diffusion model while translating it.

The config is a path to a ``.dgen`` file. ``--stdin`` reads the config from
stdin instead. The script is printed to stdout. ``-o`` / ``--output`` writes
it to a file instead. ``--tree`` prints the parse tree and does not write a
script.

.. code-block:: bash

    # Print the script
    dgenerate --sub-command transpile examples/config_syntax/basic-syntax-config.dgen

    # Write a readable script
    dgenerate --sub-command transpile config.dgen --relaxed -o config.py

    # Translate a config piped on stdin
    dgenerate --sub-command transpile --stdin --relaxed < config.dgen

The help output of ``transpile`` is as follows:

@COMMAND_OUTPUT[dgenerate --sub-command transpile --help]

