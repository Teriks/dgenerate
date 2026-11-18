Assistant usage guide
=====================

Recipes for dgenerate configs: chained steps, a model family, inpainting,
outpainting, upscaling, ControlNet, LoRAs, or video.

Template continuations and delayed expansion
--------------------------------------------

A line starting with ``{`` is one template continuation (a heredoc). The whole
``{% if %}...{% endif %}`` or ``{% for %}...{% endfor %}`` renders first, then
runs. Nested ``if`` / ``for`` stay in that render. ``\set``, ``\setp``,
``\sete``, ``\download``, and ``\gen_seeds`` inside the block have not run, so
``{{ name }}`` there is empty. ``{{ civit_ai_token }}`` works only if it was
set *before* the block.

Wrong. The download example wraps ``\set`` and the generation in
``{% if token %}``. Do not copy it. ``{{ model }}`` is empty because the block
renders before ``\set``::

    {% if civit_ai_token.strip() %}
        \set model https://civitai.com/api/download/models/1
        {{ model }}
        --model-type sdxl
        --prompts "a fox"
    {% endif %}

Right: one early ``\exit`` at the top, then ``\set`` and the invocation at
top level. Do not copy ``{% if token.strip() %}`` from the docs. Official
``.dgen`` files say "to run this example" because they are examples. A
generated config is a script the user will run, so ``\print`` is a real
instruction, never an example. ``{{ model }}`` may start an invocation only
when the next line starts with ``--`` and ``model`` was set *before* any
``{% %}`` that contains that line.

CivitAI, only for a ``civitai.com`` or ``civitai.red`` download. A Hugging
Face repo or a local file does not use ``CIVIT_AI_TOKEN``. Do not put
``token=`` on the URL. dgenerate appends ``CIVIT_AI_TOKEN`` from the
environment when that query is absent::

    \set civit_ai_token %CIVIT_AI_TOKEN%
    {% if not civit_ai_token.strip() %}
        \print Set CIVIT_AI_TOKEN environmental variable.
        \exit
    {% endif %}
    \set model https://civitai.com/api/download/models/1
    {{ model }}
    --model-type sdxl
    --prompts "a fox"

A ``{{ name }}`` in the middle of a top-level option is rendered when that
invocation runs, so ``\set`` above it is visible. ``\set optimization`` inside
``{% if have_cuda() %}...{% endif %}`` is fine when ``{{ optimization }}`` is
*after* ``{% endif %}``.

Gated Hugging Face checkpoints
------------------------------

``HF_TOKEN`` is only for a gated checkpoint. A public repo gets no token
block. Do not copy a token block from an example of a different repo.

Public: SD 1.5, SD 2.1, SDXL (base, refiner, inpainting),
``black-forest-labs/FLUX.1-schnell``, Kolors, Stable Cascade, the diffusion
upscalers, pix2pix.

Gated: ``black-forest-labs/FLUX.1-dev``, ``FLUX.1-Fill-dev``,
``FLUX.1-Kontext-dev``, SD3, SD3.5, ``Lightricks/LTX-2.5-Diffusers``,
``Lightricks/LTX-Video``, ``DeepFloyd/IF-I-M-v1.0``.

For a gated repo, once at the top::

    \set token %HF_TOKEN%
    {% if not token.strip() and not '--auth-token' in injected_args %}
        \print Set HF_TOKEN environmental variable or pass --auth-token.
        \exit
    {% endif %}

first draft
-----------

A generated config is a script the user will run. Examples and the manual
say "to run this example". Do not copy that sentence. Only a gated
checkpoint gets a token ``\print``, and it is a real instruction::

    \print Set HF_TOKEN environmental variable or pass --auth-token.

``CIVIT_AI_TOKEN`` only when a ``civitai.com`` or ``civitai.red`` link is in
the config::

    \print Set CIVIT_AI_TOKEN environmental variable.

SD 1.5, SDXL, FLUX.1-schnell, Kolors, and Cascade get no ``HF_TOKEN`` block.
A Hugging Face repo gets no ``CIVIT_AI_TOKEN`` block.

Prompt weighting uses ``--prompt-weighter sd-embed`` and marks on the
subjects that matter::

    --prompt-weighter sd-embed
    --prompts "((requested style)) of a (main subject:1.3), (second subject:1.2)"

``compel`` uses ``word+`` / ``word++``. A weighter with a plain sentence does
nothing. Flux has no ``;`` negative prompt.

nested for loops and last_images
--------------------------------

``last_images`` inside a ``{% %}`` block is the list from before the block.
Invocations inside, including earlier loop iterations, have not run, so they
cannot update ``last_images``.

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

Right, one invocation per item, then the next step *after* ``{% endfor %}``::

    {% for image in last_images %}
        stabilityai/stable-diffusion-xl-base-1.0
        --image-seeds {{ quote(image) }}
        --prompts "refine this"
    {% endfor %}

    other-model
    --image-seeds {{ quote(last_images) }}

Do not ``\set seed {{ image }}`` and then ``{{ seed }}`` in the same loop.
Use ``{{ image }}``.

last_images and chaining invocations
------------------------------------

``last_images`` and ``last_animations`` are the files the last invocation or
``\image_process`` wrote. The next step replaces them, so save a result you
need twice::

    stabilityai/stable-diffusion-xl-base-1.0
    --model-type sdxl
    --output-path base
    --prompts "a mountain landscape"

    \set input_image {{ quote(first(last_images)) }}

    later-step
    --image-seeds {{ input_image }}

Write ``--image-seeds {{ quote(last_images) }}`` or
``{{ quote(first(last_images)) }}`` for one file. Never guess names dgenerate
writes, such as ``output/s_1_g_5-0_i_30_step_1.png``. ``path/to/`` is only a
file the user must supply. Generate, then edit: the first step generates,
later steps use ``last_images``, not ``path/to/``.

generate then outpaint expand patchmatch
----------------------------------------

Generate with a normal (not inpainting) model. Save ``last_images``. Build a
mask with ``outpaint-mask``, letterbox the saved image by the same padding,
``patchmatch`` that letterboxed image with the mask, then inpaint.
``\image_process`` uses ``--output`` and ``-ox``, never ``--output-path``.
Two ``\image_process`` lines that both need the generated image must save it
first, because ``last_images`` becomes the first process output::

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

``letterbox`` runs before ``patchmatch`` so the image matches the mask. Do
not pass the mask file as the ``\image_process`` input of the patchmatch step.

inpainting models need a mask
-----------------------------

Repos with ``inpainting`` in the name, and Flux Fill
(``--model-type flux-fill``), need ``--image-seeds``
``image.png;mask.png`` (white = change, black = keep). A prompt-only step
uses a regular model of that family, for example
``stabilityai/stable-diffusion-xl-base-1.0`` for SDXL, not the inpainting
repo. After an image exists, switch to the inpainting repo.

Hugging Face repo names
-----------------------

A repo id is exactly ``organization/name``. Copy it from the examples or the
models table. Do not add path parts
(``stabilityai/stable-diffusion-v1-5/stable-diffusion-v1-5`` is wrong; SD 1.5
is ``stable-diffusion-v1-5/stable-diffusion-v1-5``). Do not move a name to
another org (``diffusers/stable-diffusion-xl-base-1.0`` does not exist;
``stabilityai/stable-diffusion-xl-base-1.0`` does). SDXL inpainting is
``diffusers/stable-diffusion-xl-1.0-inpainting-0.1``.

variants
--------

``--variant`` must be one that repo ships, usually ``fp16`` or ``bf16`` when
the examples use it. Do not invent names such as ``v1-5-pruned-emaonly``. If
the examples omit ``--variant``, omit it. Cascade uses ``--variant bf16`` on
the prior.

image_process directive options
-------------------------------

``\image_process`` is not a generation. Options are ``--processors``,
``--output`` / ``-o``, ``--output-overwrite`` / ``-ox``, ``--resize``,
``--device``. No ``--output-path``. Processor URIs look like
``canny;lower=100;upper=200``. Unknown names fail. See
``dgenerate --image-processor-help``.

sdxl refiner
------------

Prefer one invocation with
``--sdxl-refiner stabilityai/stable-diffusion-xl-refiner-1.0``.
A two-stage config writes latents with ``--image-format pt`` and
``--denoising-end``, then the refiner repo with
``--image-seeds {{ quote(last_images) }}`` and ``--denoising-start``.
A second invocation on a PNG without those denoising options is img2img, not
a refine.

hires fix img2img upscale
-------------------------

Hires fix is generate small, then img2img larger with the same family. Do
not use ``--image-format pt`` unless the next step consumes latents.
``--image-seed-strengths`` is img2img on an image, not a text-to-image step::

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

``upscaler-x4`` is ``stabilityai/stable-diffusion-x4-upscaler``.
``upscaler-x2`` is ``stabilityai/sd-x2-latent-upscaler``.
``--output-size`` is the size fed in, not the final size. A 512 image with
``--output-size 512`` and ``upscaler-x4`` writes 2048. Do not shrink a 512
image to 256 unless you want 1024. Feed
``--image-seeds {{ quote(first(last_images)) }}``.

controlnet depth canny processors
---------------------------------

Use a ControlNet repo with ``--control-nets`` and the user's image as
``--image-seeds``. Make the control image with
``--control-image-processors``, for example ``midas`` or ``canny``.
``output-file=`` writes a debug image. Do not load that path in a later
invocation. SD 1.5 depth: ``lllyasviel/sd-controlnet-depth`` with ``midas``.

loras
-----

``--loras org/repo`` or
``--loras org/repo;weight-name=file.safetensors;scale=0.8``.
URI arguments are only ``scale``, ``revision``, ``subfolder``,
``weight-name``, and ``component``. A trigger word goes in ``--prompts``.
``token=`` is not a LoRA argument. It is valid only on
``--textual-inversions``. Use only LoRA repos and ``weight-name`` files from
the examples. Known SDXL files in ``goofyai/SDXL-Lora-Collection`` include
``leonardo_illustration.safetensors``. If the request names a style with no
example LoRA, use ``path/to/style.safetensors`` and say in a comment that the
user must supply it. Do not invent a ``weight-name``.

schedulers
----------

``--schedulers`` takes class names such as
``EulerAncestralDiscreteScheduler``, ``DPMSolverMultistepScheduler``,
``DDIMScheduler``. Do not invent names like
``DPMPlus2KAncestralDiscreteScheduler``. The Karras form is
``DPMSolverMultistepScheduler;use-karras-sigmas=true``.

prompts and negative prompts
----------------------------

One ``;`` splits positive from negative. Quality words (``highly detailed``,
``sharp focus``, ``8k``, ``masterpiece``) go before the ``;``. After it, only
defects: ``blurry, low quality, deformed, watermark``. Flux, Flux Fill, Flux
Kontext, and many video models take no negative prompt. Text in the image
goes in single quotes: ``a sign that says 'DGEN'``.

prompt weighting sd-embed compel
--------------------------------

Without ``--prompt-weighter``, ``(horse:1.3)``, ``((red barn))``, and
``horse+`` are ordinary text. The weighter scales those embeddings. On SD
1.5 and SDXL it also lifts the 77-token CLIP cutoff.

Use a weighter when the request emphasizes a subject or pastes
Automatic1111 / CivitAI / InvokeAI syntax. Do not weight every quality word.
A weighter with a plain sentence and no ``(phrase:1.3)`` / ``((phrase))`` /
``word+`` does nothing.

``sd-embed`` is the usual choice, including SD3. It is Automatic1111 /
CivitAI / ComfyUI syntax::

    --prompt-weighter sd-embed
    --prompts "(Photo) of a (horse:1.3) by a (red barn), (high resolution:1.2); artwork"

- ``(phrase)`` or ``((phrase))`` raises attention, one step per pair.
- ``(phrase:1.3)`` sets the multiplier. ``1.0`` is normal. Stay near ``0.8``
  to ``1.4``.
- ``[phrase]`` lowers attention.

``compel`` is InvokeAI syntax for SD 1.5, SDXL, Cascade, and Flux.
``+`` / ``-`` after a word or group, or a number after a group::

    --prompt-weighter compel
    --prompts "Photo+ of a horse+ by a (red barn)+, (high resolution)++; artwork"

``--prompt-weighter compel;syntax=sdwui`` accepts Automatic1111 syntax and
translates it. Weights will not match Automatic1111. Prefer ``sd-embed`` for
a pasted CivitAI prompt.

SD3 needs ``sd-embed``, not ``compel``. Flux works with either; use
``sd-embed`` when the prompt has parentheses. Kolors and LTX have no
weighter. Flux still gets no ``;`` negative with ``sd-embed``.

One prompt only: put ``<weighter: sd-embed>`` in that ``--prompts`` value.
The refiner or Cascade decoder can take ``--second-model-prompt-weighter``.

seeds formats and batch
-----------------------

``--gen-seeds 4`` writes four images with four random seeds. Use ``1`` unless
asked for several. ``--seeds 42`` is the seed 42, not forty-two images.
``--image-format jpg`` writes JPEG. ``png`` is the default. ``--output-path``
is a directory for results, never the user's input file.

adetailer face fix
------------------

``--adetailer-detectors Bingsu/adetailer;weight-name=face_yolov8n.pt`` on an
img2img or inpaint invocation with ``--image-seeds`` of the user's photo. It
is not a ControlNet and not a background-removal mask. To replace a
background, inpaint with a real mask or ``yolo`` / ``sam``, not a face
detector on the whole animal.

flux kontext
------------

Flux Schnell: ``black-forest-labs/FLUX.1-schnell``, 4 steps, guidance 0.
Flux Dev: ``black-forest-labs/FLUX.1-dev``. Flux Kontext edits an image:
``--model-type flux-kontext``, ``--image-seeds`` the photo, and the prompt
is an instruction. Do not put ``last_animations`` on ``--image-seeds`` for
Kontext, Fill, or image-to-video.

ltx2 ltx-video
--------------

``ltx2``, LTX-2, or LTX means LTX-2.5:
``Lightricks/LTX-2.5-Diffusers``, ``--model-type ltx2``,
``--guidance-scales 1``, ``--model-sequential-offload``,
``--animation-format mp4``, and
``--image-seeds {{ quote(first(last_images)) }}`` to animate a still.
It makes video and audio. Describe the sound. Spoken words go in single
quotes. ``--video-lengths`` is seconds and snaps to ``8k+1`` frames.
``--video-fps`` defaults to 24. Width and height must be divisible by 32.
Other LTX options use the ``--ltx-`` prefix. Use the duration the user
named, including a long one. If they named none, omit ``--video-lengths``
so the duration head chooses it. ``--ltx-video-min-seconds`` and
``--ltx-video-max-seconds`` clamp that head. ``2 4`` with ``6 8`` is two
clips, 2 to 6 and 4 to 8. Other ``--ltx-`` value lists are tried in turn.
``--ltx-latent-upscale`` is the two-stage pass in that generation.
``--output-size`` must be divisible by 64. Distilled needs no stage LoRA.
The full transformer uses ``subfolder=transformer_full``, ``shift-terminal=0.1``,
``--ltx-stg-scales 1``, ``--ltx-modality-scales 3``, ``--ltx-stg-blocks 28``,
video guidance 3, and ``--ltx-audio-guidance-scales 7``. Its refine LoRA is
``--ltx-stage-loras``, not a second config.
``--pipeline-prompt-enhance`` turns on the Gemma enhancer. Any other
``--pipeline-prompt-enhance-*`` flag turns it on with the built-in prompt.
``--prompt-upsampler`` is a ``--transformer`` URI for another upsampler and
turns enhancement on. Kandinsky 6 rejects it.
``--ltx-video-decoder diffusion`` uses the diffusion decoder. Linux loads
NATTEN through kernels. Windows uses Triton, or PyTorch when Triton is absent.
``--ltx-diffusion-decode-tile`` sets that tile, for example ``384x384x24``.
``--ltx-image-crfs`` recompresses a conditioning still. ``--ltx-ic-lora`` is
the IC-LoRA. The last frame is ``last-frame=``, not ``end=``. ``ltx-index``
is the latent frame and ``strength`` is 0 to 1. ``--image-seed-strengths``
fills groups that omit ``strength``. Extra conditions are separated by a
space, ``++``, and a space.
Use ``Lightricks/LTX-Video`` (``--model-type ltx-video``) only when the user
names that older video-only model. Never put ``--model-type ltx2`` on
``Lightricks/LTX-Video``, or ``--model-type ltx-video`` on
``Lightricks/LTX-2``, ``Lightricks/LTX-2.3``, or
``Lightricks/LTX-2.5-Diffusers``.

wan video animate
-----------------

Wan 2.1 / 2.2 is ``--model-type wan`` on a public ``Wan-AI`` repo, no
``HF_TOKEN``. Text to video is ``Wan-AI/Wan2.1-T2V-1.3B-Diffusers``,
``--guidance-scales 5``, ``--video-fps 16``, ``--output-size 832x480``.
``--video-lengths`` is seconds and snaps to ``4k+1`` frames. Five seconds at
16 fps is 81 frames. Width and height must be divisible by 16. Put the
negative after ``;`` in ``--prompts``.
A still is image-to-video on ``Wan-AI/Wan2.1-I2V-14B-480P-Diffusers``.
Its GGUF is ``wan2.1-i2v-14b-480p-Q4_K_S.gguf`` from
``city96/Wan2.1-I2V-14B-480P-gguf`` on that parent.
``last-frame=`` needs ``Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers`` at
``1280x720`` and ``wan2.1-flf2v-14b-720p-Q4_K_S.gguf``. A Wan 2.1 I2V
checkpoint only embeds the first frame. A video seed is video-to-video and
rejects ``--video-lengths``. Trim with ``frame-end=``.
``control=``, ``mask=``, and ``reference=`` are VACE on
``Wan-AI/Wan2.1-VACE-1.3B-diffusers``.
Wan 2.2 text and video to video are ``Wan-AI/Wan2.2-T2V-A14B-Diffusers``,
guidance 4 and ``--wan-low-noise-guidance-scales 3``. GGUF experts:
``Wan2.2-T2V-A14B-HighNoise-Q4_K_S.gguf`` on ``--transformer`` and
``Wan2.2-T2V-A14B-LowNoise-Q4_K_S.gguf`` on ``--wan-second-transformer``,
both from ``QuantStack/Wan2.2-T2V-A14B-GGUF``. A Wan 2.2 still is
``Wan-AI/Wan2.2-I2V-A14B-Diffusers`` with that guidance. Its experts are
``Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf`` and
``Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf`` from
``QuantStack/Wan2.2-I2V-A14B-GGUF``.
The Wan VAE defaults to ``float32``. Override with ``--vae`` and
``AutoencoderKLWan`` plus ``dtype=``. The VAE is not a ``--quantizer-map``
target.
Wan-Animate is ``--model-type wan-animate``: a character still plus
``wan-pose=`` and ``wan-face=``, or ``wan-driving=`` with
``--wan-animate-preprocess`` (openpose and yolo ``crops=True`` face crop) or
``--wan-pose-image-processors`` and ``--wan-face-image-processors``.
``--video-fps`` defaults to 30. Omitting steps uses 20 and omitting guidance
uses 1. ``--video-lengths`` is rejected. ``replace`` mode needs
``wan-background=`` and ``mask=``.
Wan-Animate-2 is ``--model-type wan-animate-2``: a character still plus
``wan-driving=`` only. Omitting steps uses 40 on the base repo and 10 on the
distilled repo. ``--video-fps`` defaults to 24. ``--video-lengths`` is
rejected. No ``wan-pose=``, ``wan-face=``, or ``wan-background=``.

minimax kandinsky 6 flux3
-------------------------

FLUX.3 is ``--model-type flux3``, not Flux.2. Distilled is
``black-forest-labs/flux-3-image-klein``, 16 steps, guidance 1. Base is
``black-forest-labs/flux-3-image-klein-base``, 50 steps, guidance 3.5.
Both, and ``black-forest-labs/flux-3-image-klein-comfy``, are gated.
``--image-seeds`` are references, no mask and no strength. Omit
``--pipeline-prompt-enhance`` unless asked. It loads the Qwen3.5 upsampler.
MiniMax-H3-Image is ``--model-type minimax-h3-image``,
``MiniMaxAI/MiniMax-H3-Image``. One still. No negative prompt, no guidance
scale, no ``--inference-steps``. Default size 2048, sides multiples of 32.
``--image-seeds`` are references, no mask and no strength, named Picture 1,
Picture 2, and so on.
MiniMax-H3 is ``--model-type minimax-h3-video``, ``MiniMaxAI/MiniMax-H3``.
Video and audio at 24 fps, 30 steps. ``--video-lengths`` is 5 to 15 seconds.
No negative prompt and no guidance scale. A still and ``last-frame=`` are
keyframes. ``reference=`` is images, videos, and audio, and cannot combine
with a keyframe. ``--control-nets`` is one Fun branch,
``alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0`` with
``weight-name=MiniMax-H3-Fun-Controlnet-Union-2.0.safetensors``. A bare
``--image-seeds`` path is the control video. ``mask=`` inpaints. It cannot
combine with a keyframe or ``reference=``. Quants use ``--transformer`` on that parent.
fl2va ``MiniMax-H3-fl2va-curve-Q4_0.gguf`` and ``minimax_h3_fl2va_pruned_int8_convrot``
are text and keyframes. ref2va ``MiniMax-H3-ref2va-curve-Q4_0.gguf`` and
``minimax_h3_ref2va_pruned_int8_convrot`` are references. Image quants are
``minimax_h3_image_pruned-Q4_K_M.gguf`` and ``minimax_h3_image_pruned_int8_convrot``.
Kandinsky 6 video is ``--model-type kandinsky6-video``, not Kandinsky 5.
Lite distilled is ``Kandinsky-6.0-Lite-distill-5s-Diffusers``, 10 steps,
guidance 1, audio, 864x480, 24 fps. Lite, full Lite, and Pro distilled are
public. Pro ``Kandinsky-6.0-Pro-5s-Diffusers`` is gated and needs HF_TOKEN.
50 steps, guidance 5.
One still is the first frame. No ``--loras``.
``--pipeline-prompt-enhance`` turns on ``expand_prompts``.
HunyuanVideo 1.5 is ``--model-type hyvideo-15``. Its GGUF stays on the
480p text-to-video parent. A still keeps the i2v transformer.
Kandinsky 5 video is ``--model-type kandinsky5-video``. The Pro GGUF
``kandinsky5Pro_t2v_sft_5s-Q4_K_S.gguf`` stays on the Pro text-to-video parent.

torch compile
-------------

Wan-Animate-2 compiles its transformer and VAE blocks on every load. Do not
add ``--torch-compile`` for that model. It ignores module names on
``--torch-compile`` and a URI ``compile=true``. Add ``--torch-compile`` for
other models only when the user asks, including Stable Diffusion 3 and
Stable Cascade. The option also compiles ControlNet and VAE decoder blocks.
Module names limit which components are compiled. With no names, the default
set is used. ``prompt_upsampler`` and ``prompt_enhancer`` are aliases for
the FLUX.3 upsampler or the LTX Gemma enhancer. ``compile=true`` on
``--unet``, ``--vae``, ``--text-encoders``, ``--control-nets``,
``--transformer``, or ``--prompt-upsampler`` compiles that module even when
the option is omitted. ``--image-encoder`` has no ``compile`` argument. Name
``image_encoder`` on ``--torch-compile``. Do not compile GGUF, SDNQ, or
bitsandbytes. Those stay eager and dgenerate warns. Weights streamed from
disk skip ``--torch-compile``, including a URI ``compile=true``.
``DGENERATE_TORCH_COMPILE=0`` disables every compile, including
Wan-Animate-2. TorchDynamo, Inductor, and Triton stay quiet unless the user
passed ``-v``.

gguf files
----------

A ``.gguf`` replaces the transformer or UNet, not the pipeline. The model
line stays the Hugging Face repo that supplies the VAE and text encoders.
``--transformer`` gets the file. Exception: the model line can be the GGUF
when the tensors select the repo, as with an LTX-2 or LTX-2.5 file.
Do not set ``quantizer=`` and do not use ``--quantizer gguf``. Quantize a
text encoder with ``bnb`` or ``sdnq`` and ``--quantizer-map text_encoder``.
``--model-sequential-offload`` works with these files. Copy the parent from
the models table. Do not move a file onto a sibling repo. Wan 2.2 low noise
uses ``--wan-second-transformer``. High noise uses ``--transformer``.
Kandinsky 6 Pro is
``Diffusion Model/Full/Kandinsky-6.0-Pro-5s-Q4_K_S.gguf`` on the Pro parent.
The published folder has a space before ``Full``. dgenerate redirects it.
Flux.2 Klein, Qwen-Image, Z-Image, LTX-2, LTX-2.5, and Wan need no
``config=``, including Comfy layouts. Flux.2 width 3072 is Klein 4B
(public), 4096 is Klein 9B, and 6144 is Flux.2 dev. Z-Image base and Turbo
share a shape, so a base GGUF still uses the Turbo module config. The parent
is the checkpoint: ``Tongyi-MAI/Z-Image`` (guidance about 4, 28 to 50 steps)
or ``Tongyi-MAI/Z-Image-Turbo`` (8 steps, guidance 0). Qwen-Image Edit uses
the Qwen-Image module. Layered is selected by ``addition_t_embedding``.
Klein KV is width 4096, the Klein 9B module.

ltx2 gemma gguf
---------------

ltx2, LTX-2, or LTX means an LTX-2.5 recipe. Gemma 3 and
``Lightricks/LTX-2`` only when the user names 19B, LTX-2.0, or Gemma 3.
Distilled LTX-2.5 GGUF keeps guidance 1 and omits ``--inference-steps``.
Dev GGUF is guidance 3, audio guidance 7, and
``FlowMatchEulerDiscreteScheduler;use-dynamic-shifting=true;shift-terminal=0.1``.
Gemma 4 is ``Gemma4UnifiedForConditionalGeneration`` at
``gemma4-12b-with-proj-ltx-2.5-Q4_K_M.gguf`` from
``elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF``, which also has
``gemma4-12b-with-proj-ltx-2.5-Q2_K.gguf`` and
``gemma4-12b-with-proj-ltx-2.5-Q5_K_M.gguf``. There is no Q8_0. Copy one of
those three. INT8 ConvRot and NVFP4 are the other layouts. Do not set
``quantizer``. Parent stays ``Lightricks/LTX-2.5-Diffusers``. With the dev
transformer keep 30 steps, guidance 3, audio guidance 7, STG 1 on block 28,
and modality 3.
``keyframes_abs_pos_embedding`` or ``prompt_adaln_single`` selects LTX-2.5.
``audio_patchify_proj`` without those selects ``Lightricks/LTX-2``. The 19B
dev file keeps that parent, guidance 4. Distilled 19B sets
``use-dynamic-shifting=false`` and guidance 1. Its encoder is
``Gemma3ForConditionalGeneration`` at ``gemma_3_12B_it_fp8_scaled.safetensors``
or ``gemma_3_12B_it_fp4_mixed.safetensors`` from ``Comfy-Org/ltx-2``, with no
``quantizer``. With the 19B dev GGUF keep 30 steps, guidance 4, audio
guidance 4, and the parent scheduler.
Examples: ``examples/flux/gguf``, ``examples/stablediffusion3/gguf``,
``examples/flux2/gguf``, ``examples/z_image/gguf``,
``examples/qwen_image/gguf``, ``examples/ltx2/gguf``, ``examples/wan/gguf``.

do not loop in comments
-----------------------

One short comment per choice, then the invocation. Never repeat the same
sentence. If a later step is unclear, write the invocation anyway.

controlnet output-file
----------------------

``output-file=`` on ``--control-image-processors`` writes a debug image. It
is not an input for a second invocation. Keep the processors on the
ControlNet invocation.

schedulers help
---------------

``--schedulers help`` and ``--schedulers helpargs`` only print names. They
are not a generation. Use a real class such as
``DPMSolverMultistepScheduler``.
