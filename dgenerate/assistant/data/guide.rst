Assistant usage guide
=====================

This guide is for writing dgenerate configs. Follow these recipes when a
request chains steps, names a model family, or uses inpainting, outpainting,
upscaling, ControlNet, LoRAs, or video.

Template continuations and delayed expansion
--------------------------------------------

A line that starts with ``{`` is a template continuation (a heredoc).
The whole ``{% if %}...{% endif %}`` or ``{% for %}...{% endfor %}``
is rendered first, then the result is run as config. Nested ``if`` /
``for`` stay in that same render. Directives inside the block
(``\set``, ``\setp``, ``\sete``, ``\download``, ``\gen_seeds``) have
not run yet, so ``{{ name }}`` in the same block cannot see them.
``{{ civit_ai_token }}`` is fine if that variable was set *before*
the block.

Wrong: wrapping the download and generation in ``{% if token %}``.
The download-directive example does this; do not copy it. ``{{ model }}``
inside that block is empty because the block is rendered before
``\set`` runs::

    {% if civit_ai_token.strip() %}
        \set model https://civitai.com/api/download/models/1?token={{civit_ai_token}}
        {{ model }}
        --model-type sdxl
        --prompts "a fox"
    {% endif %}

Right: one early ``\exit`` at the top when the token is missing, then
``\set`` and the invocation at the top level. Do not copy the
``{% if token.strip() %}`` wrap from the docs. Official ``.dgen`` files
say "to run this example" because they are examples; a generated
config is a script the user will run, so ``\print`` is a real
instruction, never an example. ``{{ model }}`` may start an invocation
only when the next line starts with ``--`` and ``model`` was set
*before* any ``{% %}`` that contains that line.

CivitAI::

Only when the config downloads from a ``civitai.com`` link. A Hugging Face
repo or a local file does not use ``CIVIT_AI_TOKEN``.

    \set civit_ai_token %CIVIT_AI_TOKEN%
    {% if not civit_ai_token.strip() %}
        \print Set CIVIT_AI_TOKEN environmental variable.
        \exit
    {% endif %}
    \set model https://civitai.com/api/download/models/1?token={{civit_ai_token}}
    {{ model }}
    --model-type sdxl
    --prompts "a fox"

Gated Hugging Face checkpoints
------------------------------

``HF_TOKEN`` is only for gated checkpoints. Public repos get no token
block. Do not copy a token block from an example of a different repo.

Public: SD 1.5, SD 2.1, SDXL (base, refiner, inpainting),
``black-forest-labs/FLUX.1-schnell``, Kolors, Stable Cascade, the
diffusion upscalers, pix2pix.

Gated: ``black-forest-labs/FLUX.1-dev``, ``FLUX.1-Fill-dev``,
``FLUX.1-Kontext-dev``, SD3 and SD3.5, ``Lightricks/LTX-2.5-Diffusers``,
``Lightricks/LTX-Video``, ``DeepFloyd/IF-I-M-v1.0``.

For a gated repo::

    \set token %HF_TOKEN%
    {% if not token.strip() and not '--auth-token' in injected_args %}
        \print Set HF_TOKEN environmental variable or pass --auth-token.
        \exit
    {% endif %}

A ``{{ name }}`` in the middle of an option line at the top level is
rendered when that invocation runs, so ``\set`` *above* it is visible.
``\set optimization`` inside ``{% if have_cuda() %}...{% endif %}`` is
fine when ``{{ optimization }}`` is *after* ``{% endif %}``.

first draft
-----------

A generated config is a script the user will run, not an example.
The ``.dgen`` examples and the manual print "to run this example".
Do not copy that sentence. Only a gated checkpoint gets a token
``\print``, and then it is a real instruction::

    \print Set HF_TOKEN environmental variable or pass --auth-token.

``CIVIT_AI_TOKEN`` only when a ``civitai.com`` link is in the config::

    \print Set CIVIT_AI_TOKEN environmental variable.

SD 1.5, SDXL, FLUX.1-schnell, Kolors, and Cascade do not get an
``HF_TOKEN`` block. A Hugging Face repo does not get a ``CIVIT_AI_TOKEN``
block.

When the request asks for prompt weighting, the first draft must use
``--prompt-weighter sd-embed`` and marks on the subjects that matter::

    --prompt-weighter sd-embed
    --prompts "((requested style)) of a (main subject:1.3), (second subject:1.2)"

``compel`` uses ``word+`` / ``word++`` instead. A weighter with a plain
sentence does nothing. Flux has no ``;`` negative prompt.

nested for loops and last_images
--------------------------------

``last_images`` inside a ``{% %}`` continuation is the list from before
the block. Invocations in the block, including earlier ``{% for %}``
iterations, have not run yet, so they cannot update ``last_images``.

Wrong::

    {% for image in last_images %}
        stabilityai/stable-diffusion-xl-base-1.0
        --image-seeds {{ quote(image) }}
        --prompts "refine this"

        other-model
        --image-seeds {{ quote(last_images) }}
    {% endfor %}

Right, expand the list on the option line (the DeepFloyd examples)::

    --image-seeds {% for image in last_images %}"{{ image }};floyd={{ image }}" {% endfor %}

Right, one invocation per item using the loop variable, then the next
step *after* ``{% endfor %}``::

    {% for image in last_images %}
        stabilityai/stable-diffusion-xl-base-1.0
        --image-seeds {{ quote(image) }}
        --prompts "refine this"
    {% endfor %}

    other-model
    --image-seeds {{ quote(last_images) }}

Do not ``\set seed {{ image }}`` and then ``{{ seed }}`` in the same
loop. Use ``{{ image }}`` directly.

last_images and chaining invocations
------------------------------------

``last_images`` and ``last_animations`` are lists of files the last
invocation or ``\image_process`` wrote. They are replaced by the next
step, so save a result you need twice::

    stabilityai/stable-diffusion-xl-base-1.0
    --model-type sdxl
    --output-path base
    --prompts "a mountain landscape"

    \set input_image {{ quote(first(last_images)) }}

    later-step
    --image-seeds {{ input_image }}

Write ``--image-seeds {{ quote(last_images) }}`` or
``{{ quote(first(last_images)) }}`` for one file. Never guess names
dgenerate writes, such as ``output/s_1_g_5-0_i_30_step_1.png``.
``path/to/`` is only for a file the user must supply. If the request
says generate an image and then edit it, the first step generates it
and later steps use ``last_images``, not ``path/to/``.

generate then outpaint expand patchmatch
----------------------------------------

Generate first with a normal (not inpainting) model. Save
``last_images``. Build a mask with ``outpaint-mask``, letterbox the
saved image by the same padding, run ``patchmatch`` on that letterboxed
image with the mask, then inpaint. ``\image_process`` uses ``--output``
and ``-ox``, never ``--output-path``. If two ``\image_process`` lines
both read the generated image, ``last_images`` is the first process
output after the first line, so save the generated file first::

    stabilityai/stable-diffusion-xl-base-1.0
    --model-type sdxl --dtype float16 --variant fp16
    --output-path base --prompts "a mountain landscape"

    \set input_image {{ quote(first(last_images)) }}
    \set outpaint_box 128

    \image_process {{ input_image }}
    --processors outpaint-mask;box={{ outpaint_box }}
    --output mask.png -ox

    \image_process {{ input_image }}
    --processors letterbox;box-size={{ outpaint_box }};box-is-padding=True \
        patchmatch;mask=mask.png;seed=42
    --output filled.png -ox

    diffusers/stable-diffusion-xl-1.0-inpainting-0.1
    --model-type sdxl --dtype float16 --variant fp16
    --image-seeds filled.png;mask.png
    --image-seed-strengths 0.85
    --output-path outpaint
    --prompts "a mountain landscape"

``letterbox`` must run before ``patchmatch`` so the image is the same
size as the mask. Do not pass the mask file as the ``\image_process``
input of the patchmatch step.

inpainting models need a mask
-----------------------------

Repos with ``inpainting`` in the name, and Flux Fill
(``--model-type flux-fill``), only run with ``--image-seeds``
``image.png;mask.png`` (white = change, black = keep). A step that
generates from a prompt alone must use a regular model of that family,
for example ``stabilityai/stable-diffusion-xl-base-1.0`` for SDXL, not
the inpainting repo. After an image exists, switch to the inpainting
repo for the inpaint or outpaint step.

Hugging Face repo names
-----------------------

A repo id is exactly ``organization/name``. Copy it from the examples
or the models table. Do not add extra path parts
(``stabilityai/stable-diffusion-v1-5/stable-diffusion-v1-5`` is wrong;
the SD 1.5 repo is ``stable-diffusion-v1-5/stable-diffusion-v1-5``).
Do not move a name to another org
(``diffusers/stable-diffusion-xl-base-1.0`` does not exist;
``stabilityai/stable-diffusion-xl-base-1.0`` does). The SDXL inpainting
repo is ``diffusers/stable-diffusion-xl-1.0-inpainting-0.1``.

variants
--------

``--variant`` must be a variant that repo actually ships, usually
``fp16`` or ``bf16`` when the examples use it. Do not invent names such
as ``v1-5-pruned-emaonly``. If the examples for that repo omit
``--variant``, omit it. Cascade examples use ``--variant bf16`` on the
prior.

image_process directive options
-------------------------------

``\image_process`` is not a dgenerate generation invocation. Its
options are ``--processors``, ``--output`` / ``-o``,
``--output-overwrite`` / ``-ox``, ``--resize``, ``--device``. It has
no ``--output-path``. Processor URIs look like
``canny;lower=100;upper=200``. Unknown processor names fail. See
``dgenerate --image-processor-help``.

sdxl refiner
------------

The usual way is one invocation with
``--sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0``.
A two-stage config writes latents with ``--image-format pt`` and
``--denoising-end``, then runs the refiner repo with
``--image-seeds {{ quote(last_images) }}`` and ``--denoising-start``.
Do not run the refiner as a second invocation on a PNG without those
denoising options; that is ordinary img2img, not refining.

hires fix img2img upscale
-------------------------

Hires fix is generate small, then img2img larger with the same family.
Do not use ``--image-format pt`` unless the next step consumes latents.
``--image-seed-strengths`` only works when ``--image-seeds`` is an
image (img2img), not a text-to-image step::

    stable-diffusion-v1-5/stable-diffusion-v1-5
    --dtype float16 --output-size 512 --output-path base
    --prompts "a tiger in a bamboo forest"

    stable-diffusion-v1-5/stable-diffusion-v1-5
    --dtype float16 --output-size 1024 --output-path hires
    --image-seeds {{ quote(last_images) }}
    --image-seed-strengths 0.4
    --prompts "a tiger in a bamboo forest"

diffusion upscalers
-------------------

``--model-type upscaler-x4`` is
``stabilityai/stable-diffusion-x4-upscaler``.
``--model-type upscaler-x2`` is
``stabilityai/sd-x2-latent-upscaler``.
``--output-size`` is the size fed into the upscaler, not the final
size. A 512 image with ``--output-size 512`` and ``upscaler-x4``
writes 2048. Do not shrink a 512 image to 256 unless you want a 1024
result. Feed ``--image-seeds {{ quote(first(last_images)) }}``.

controlnet depth canny processors
---------------------------------

Use a ControlNet repo with ``--control-nets`` and the user's image as
``--image-seeds``. Produce the control image with
``--control-image-processors``, for example ``midas`` or ``canny``.
``output-file=`` on a processor writes a debug image; it is not an
input you must name later. Do not invent a second invocation that
reads that debug path. SD 1.5 depth:
``lllyasviel/sd-controlnet-depth`` with ``midas``.

loras
-----

``--loras org/repo`` or
``--loras org/repo;weight-name=file.safetensors;scale=0.8``.
Only use LoRA repos and ``weight-name`` files that appear in the
examples. Known SDXL files in ``goofyai/SDXL-Lora-Collection`` include
``leonardo_illustration.safetensors``. If the request names a style
with no example LoRA, use ``path/to/style.safetensors`` and say in a
comment that the user must supply it. Do not invent a ``weight-name``.

schedulers
----------

``--schedulers`` takes class names such as
``EulerAncestralDiscreteScheduler``,
``DPMSolverMultistepScheduler``, ``DDIMScheduler``.
Do not invent names like ``DPMPlus2KAncestralDiscreteScheduler``.
The Karras form is
``DPMSolverMultistepScheduler;use-karras-sigmas=true``.

prompts and negative prompts
----------------------------

One ``;`` splits positive from negative. Quality words
(``highly detailed``, ``sharp focus``, ``8k``, ``masterpiece``) belong
before the ``;``. After it, only defects:
``blurry, low quality, deformed, watermark``. Flux, Flux Fill, Flux
Kontext, and many video models take no negative prompt. Text that
should appear in the image goes in single quotes: ``a sign that says
'DGEN'``.

prompt weighting sd-embed compel
--------------------------------

Prompt weighting changes how strongly the text encoder attends to
parts of the prompt. Without ``--prompt-weighter``, characters such as
``(horse:1.3)``, ``((red barn))``, and ``horse+`` are ordinary text.
The weighter parses them and scales those token embeddings. On SD 1.5
and SDXL it also concatenates embeddings so the 77-token CLIP cutoff
does not silently drop the end of a long prompt.

Use a weighter when the request emphasizes a subject, or when the
user pastes Automatic1111 / CivitAI / InvokeAI syntax. Do not sprinkle
weights on every quality word. ``--prompt-weighter`` with a plain
sentence and no ``(phrase:1.3)`` / ``((phrase))`` / ``word+`` does
nothing; either write those marks on the subjects that matter, or
omit the weighter.

``sd-embed`` is the usual choice. It is Automatic1111 / CivitAI /
ComfyUI syntax, and it is the one that works on SD3::

    --prompt-weighter sd-embed
    --prompts "(Photo) of a (horse:1.3) by a (red barn), (high resolution:1.2); artwork"

- ``(phrase)`` or ``((phrase))`` raises attention (each pair of
  parentheses is another step up).
- ``(phrase:1.3)`` sets the multiplier. ``1.0`` is normal, above
  ``1.0`` stronger, below ``1.0`` weaker. Stay near ``0.8`` to
  ``1.4``.
- ``[phrase]`` lowers attention.

``compel`` is InvokeAI syntax, for SD 1.5, SDXL, Cascade, and Flux.
``+`` / ``-`` after a word or group, or a number after a group::

    --prompt-weighter compel
    --prompts "Photo+ of a horse+ by a (red barn)+, (high resolution)++; artwork"

``--prompt-weighter compel;syntax=sdwui`` accepts Automatic1111
syntax and translates it. The weights will not match Automatic1111
exactly; prefer ``sd-embed`` when the user pasted a CivitAI prompt.

SD3 needs ``sd-embed``, not ``compel``. Flux works with either;
``sd-embed`` if the prompt uses parentheses. Kolors and LTX have no
prompt weighter. Flux still gets no ``;`` negative prompt when using
``sd-embed``.

One prompt only: put ``<weighter: sd-embed>`` in that ``--prompts``
value. The refiner or Cascade decoder can take
``--second-model-prompt-weighter`` if it should differ.

seeds formats and batch
-----------------------

``--gen-seeds 4`` writes four images with four random seeds.
Use ``1`` unless the user asked for several images. ``--seeds 42`` is
the fixed seed 42, not forty-two images. ``--image-format jpg`` writes
JPEG; ``png`` is the default. ``--output-path`` is a directory name
for results, never the user's input file.

adetailer face fix
------------------

``--adetailer-detectors Bingsu/adetailer;weight-name=face_yolov8n.pt``
on an img2img or inpaint invocation with ``--image-seeds`` of the
user's photo. It is not a ControlNet and not a background-removal
mask. For "replace the background" use inpainting with a real mask
or ``yolo`` / ``sam`` mask processors, not a face detector on the
whole animal.

flux kontext video ltx
----------------------

Flux Schnell: ``black-forest-labs/FLUX.1-schnell``, 4 steps, guidance
0. Flux Dev: ``black-forest-labs/FLUX.1-dev``. Flux Kontext edits an
image: ``--model-type flux-kontext`` and ``--image-seeds`` the photo,
prompt is an instruction. LTX-2.5
(``Lightricks/LTX-2.5-Diffusers``, ``--model-type ltx``) makes video
and audio. Animate a still with that repo, ``--guidance-scales 1``,
``--model-sequential-offload``, ``--animation-format mp4``, and
``--image-seeds {{ quote(first(last_images)) }}``.
Clip length and frame rate are ``--video-lengths`` (seconds) and
``--video-fps``. Every other LTX-only option is prefixed ``--ltx-``. Use the duration the
user asked for, including a long one. When they did not name a duration,
use 4, or omit ``--video-lengths`` so the duration head chooses it.
``--ltx-video-min-seconds`` and ``--ltx-video-max-seconds`` clamp that
head. ``2 4`` with ``6 8`` is two clips, 2 to 6 and 4 to 8. Other
``--ltx-`` value lists are tried in turn. Describe sound in the prompt.
``--ltx-latent-upscale`` is the two-stage pass in that same generation.
``--output-size`` is the finished clip and must be divisible by 64.
The distilled checkpoint needs no stage LoRA. The full transformer uses
``--transformer`` with ``subfolder=transformer_full``,
``shift-terminal=0.1``, ``--ltx-stg-scales 1``, ``--ltx-modality-scales 3``,
``--ltx-stg-blocks 28``, video guidance 3, and ``--ltx-audio-guidance-scales 7``.
Its refine LoRA is ``--ltx-stage-loras``, not a second config.
``--ltx-prompt-enhancer google/gemma-4-E2B-it`` rewrites the prompt.
``--ltx-video-decoder diffusion`` uses the diffusion decoder. NATTEN comes from the kernels package, which is installed with dgenerate.
``--ltx-image-crfs`` recompresses a conditioning still.
``--ltx-ic-lora`` is the IC-LoRA. The last frame is ``last-frame=``, not ``end=``.
``ltx-index`` is the latent frame and ``strength`` is from 0 to 1.
``--image-seed-strengths`` fills LTX groups that omit ``strength``.
Extra conditions in one clip are separated by `` ++ ``.
Use ``Lightricks/LTX-Video`` only when the user names that
older video-only model.
Wan 2.1 / 2.2 is ``--model-type wan`` with a ``Wan-AI`` repo.
``--video-lengths`` is seconds and ``--video-fps`` defaults to 16.
``last-frame=`` is first-last-frame and needs a FLF2V checkpoint such as
``Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers``. A Wan 2.1 I2V checkpoint only
embeds the first frame. A video seed is video-to-video.
``control=``, ``mask=``, and ``reference=`` are VACE.
The Wan VAE defaults to ``float32``; override with ``--vae`` and
``AutoencoderKLWan`` plus ``dtype=``.
Wan-Animate is ``--model-type wan-animate`` with a character still plus
``wan-pose=`` and ``wan-face=``, or ``wan-driving=`` with ``--wan-animate-preprocess``
(openpose and yolo ``crops=True`` face crop) or ``--wan-pose-image-processors`` and
``--wan-face-image-processors``. ``--video-lengths`` is rejected for animate.
Wan-Animate-2 is ``--model-type wan-animate-2`` with a character still plus
``wan-driving=``. The base repo samples in 40 steps and the distilled repo in 10.
``--video-fps`` defaults to 24. ``--video-lengths`` is rejected.
Do not put ``last_animations`` on ``--image-seeds`` for Kontext, Fill,
or image-to-video.

gguf files
----------

A ``.gguf`` file replaces the diffusion transformer or UNet, not the
whole pipeline. The first line is still the Hugging Face repo, which
supplies the VAE and text encoders. Pass the file with
``--transformer``. Never put the ``.gguf`` path on the model line.
Do not set ``quantizer=`` on that URI and do not use
``--quantizer gguf``. Quantize the text encoder separately with
``bnb`` or ``sdnq`` and ``--quantizer-map text_encoder``.
``--model-sequential-offload`` works with these GGUF transformers.

Flux, SD3, Flux.2, Z-Image, Qwen-Image, LTX-2.5, and Wan all take a GGUF
transformer. Flux.2 Klein, Qwen-Image, Z-Image, and LTX-2.5 are
recognized from the file, including ComfyUI layouts, so
``--transformer`` needs no ``config=``. Flux.2 hidden width picks the
config: 3072 is Klein 4B (``black-forest-labs/FLUX.2-klein-4B``,
public), 4096 is Klein 9B (``black-forest-labs/FLUX.2-klein-base-9B``),
and 6144 is Flux.2 dev. Z-Image base and Turbo share one transformer
shape, so a base GGUF still uses the Turbo module config. The parent
repo is the checkpoint you want: ``Tongyi-MAI/Z-Image`` for base
(guidance about 4, 28 to 50 steps) and ``Tongyi-MAI/Z-Image-Turbo``
for Turbo (8 steps, guidance 0).

See ``examples/flux/gguf``, ``examples/stablediffusion3/gguf``,
``examples/flux2/gguf``, ``examples/z-image/gguf``,
``examples/qwen-image/gguf``, ``examples/ltx2/gguf``,
and ``examples/wan/gguf``.
The distilled LTX-2.5 GGUF keeps guidance at 1 and omits
``--inference-steps``. The dev GGUF is the full transformer: guidance 3,
audio guidance 7, and
``FlowMatchEulerDiscreteScheduler;use-dynamic-shifting=true;shift-terminal=0.1``
so ``--inference-steps`` applies. Qwen-Image Edit uses the same transformer
module as Qwen-Image. Qwen-Image Layered is selected by its extra
``addition_t_embedding``. Klein KV is width 4096, the same module as Klein 9B.

do not loop in comments
-----------------------

Write one short comment per choice, then the invocation. Never repeat
the same sentence. If a later step is unclear, write the invocation
anyway; a comment loop means the config is unfinished.

controlnet output-file
----------------------

``output-file=`` on ``--control-image-processors`` writes a debug image.
It is not an input for a second invocation. Keep the processors on the
ControlNet invocation. Do not invent a follow-up step that loads that
path.

schedulers help
---------------

``--schedulers help`` and ``--schedulers helpargs`` only print names.
They are not a generation. Use a real class such as
``DPMSolverMultistepScheduler``.
