.. _video-generation:

Video Generation
================

``--model-type ltx`` generates a clip in one pipeline call.
One combination of prompt, seed, guidance, steps, image seed, ``--ltx-video-lengths``,
``--ltx-video-fps``, ``--ltx-audio-guidance-scales``, and ``--ltx-audio-guidance-rescales``
writes one animation file. LTX does not run once per input frame.

``--ltx-video-lengths`` is a length in seconds. ``--ltx-video-fps`` is the frame rate.
``--ltx-audio-guidance-scales`` and ``--ltx-audio-guidance-rescales`` are the audio CFG
and audio rescale. All of those are combinatorial arguments, the same way
``--prompts`` and ``--seeds`` are. The frame count inside a clip is not a
separate product factor.

Audio from LTX-2.5 is muxed into an mp4. GIF, WebP, and ``--animation-format frames`` drop that audio.

Conditioning stays in ``--image-seeds``.

LTX-2.5 (``ltx``)
-----------------

Repository: ``Lightricks/LTX-2.5-Diffusers``.

* No image seed is text to video. Omitting ``--ltx-video-lengths`` lets the model's duration head choose the length.
* One image is the first frame.
* ``last-frame=`` is the last frame. A first frame and ``last-frame=`` can be used together.
* ``ltx-index`` and ``strength`` place a condition on a chosen latent frame.
  See `Condition placement`_.
* Either slot can be a video or animated image instead of a still. See `Video conditioning`_.
* With ``--ltx-ic-lora``, a plain path is instead the reference clip for an IC-LoRA, such as canny, depth, or pose control. See `IC-LoRA control`_.
* ``images:`` is rejected.

Width and height must be divisible by 32.

``--model-sequential-offload``, ``--model-cpu-offload``, and ``--model-group-offload``
work the same way they do for image models. The examples under `examples/ltx/ltx2 <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/ltx/ltx2>`_
use the published
repository as-is. ``--ltx-latent-upscale`` runs the two-stage sampler in that
same generation: a half-resolution pass, the checkpoint latent upsampler, then
a short refine at ``--output-size``. ``--transformer`` is
described under `Submodels`_. ``model_index.json`` selects the pipeline: ``LTX2Pipeline``
is LTX-2.5, and ``LTXPipeline`` is :ref:`ltx-video-legacy`.

.. _ltx-video-legacy:

LTX-Video (legacy)
------------------

``Lightricks/LTX-Video`` is the earlier 2B checkpoint. ``model_index.json`` selects
``LTXPipeline``. It writes a picture only. There is no soundtrack, no duration
head, and no IC-LoRA.

``--inference-steps`` and ``--guidance-scales`` are sent as written. The distilled
sigma table and the guidance rewrites under `Guidance, steps, and sigmas`_ apply
to LTX-2.5 only. The published configs use 50 steps, guidance ``3``,
``--ltx-video-fps 25``, and ``--ltx-video-lengths 4.84``, which is 121 frames,
at ``768x512``. Omitting ``--ltx-video-lengths`` uses the pipeline default of
161 frames. dgenerate warns when the clip is shorter than 121 frames or smaller
than about 704 by 480, because the model follows the prompt at the published size.

A prompt can name what to avoid after one ``;``. The examples use
``worst quality, inconsistent motion, blurry, jittery, distorted``.

The noise schedule shifts with the number of latent tokens. A size or length
past what that shift can represent is rejected. Use a smaller ``--output-size``
or a shorter clip.

A first frame, ``last-frame=``, an opening or closing clip, ``ltx-index``, and
``strength`` work as described under `Video conditioning`_ and `Condition placement`_.
``--loras`` loads a diffusers LoRA onto the transformer. ``--transformer`` replaces
that transformer, including a city96 ``.gguf`` file. The ``Lightricks/LTX-Video``
repository still supplies the VAE and text encoders. ``--scheduler`` accepts
``FlowMatchEulerDiscreteScheduler`` and its URI arguments.

These options are rejected: ``--ltx-audio-guidance-scales``,
``--ltx-audio-guidance-rescales``, ``--sigmas``, ``--ltx-ic-lora``,
``--ltx-latent-upscale``, ``--ltx-stg-scales``, ``--ltx-modality-scales``,
``--ltx-video-decoder diffusion``, ``--ltx-prompt-enhancer``,
``--ltx-image-crfs``, ``--ltx-no-cross-timestep``,
``--ltx-video-min-seconds``, and ``--ltx-video-max-seconds``.

The Console UI recipes are ``LTX-Video`` and ``LTX-Video (GGUF)``.

* `examples/ltx/ltx_video/text-to-video-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx_video/text-to-video-config.dgen>`_
* `examples/ltx/ltx_video/image-to-video-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx_video/image-to-video-config.dgen>`_
* `examples/ltx/ltx_video/video-extension-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx_video/video-extension-config.dgen>`_
* `examples/ltx/ltx_video/lora-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx_video/lora-config.dgen>`_
* `examples/ltx/ltx_video/gguf-text-to-video-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx_video/gguf-text-to-video-config.dgen>`_
* `examples/ltx/ltx_video/gguf-image-to-video-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx_video/gguf-image-to-video-config.dgen>`_

Video conditioning
------------------

The main ``--image-seeds`` path and ``last-frame=`` each accept a video or an animated image
as well as a still. Both LTX-2.5 and the earlier LTX-Video accept this. A file with a
single frame is treated as a still.

* A video in the main path is the opening clip. The generated clip continues from it.
* A video in ``last-frame=`` is the closing clip. The generated clip leads into it.
* A still and a video can be mixed, for example a video first and a still ``last-frame=``.

.. code-block:: bash

    # continue an existing clip
    dgenerate Lightricks/LTX-2.5-Diffusers --model-type ltx \
    --ltx-video-fps 25 --ltx-video-lengths 4 --output-size 512x512 \
    --image-seeds "input.gif" \
    --prompts "The singer keeps swaying, then points at the camera."

    # lead into an existing clip
    dgenerate Lightricks/LTX-2.5-Diffusers --model-type ltx \
    --ltx-video-fps 25 --ltx-video-lengths 4 --output-size 512x512 \
    --image-seeds ";last-frame=input.gif" \
    --prompts "A singer walks in and starts to sway at the microphone."

    # frames 16 through 40 of a clip, then a still last frame
    dgenerate Lightricks/LTX-2.5-Diffusers --model-type ltx \
    --ltx-video-fps 25 --ltx-video-lengths 4 --output-size 512x512 \
    --image-seeds "input.gif;frame-start=16;frame-end=40;last-frame=last.png" \
    --prompts "The scene fades into a pencil sketch of mountains."

``--frame-start`` and ``--frame-end``, or ``frame-start=`` and ``frame-end=`` in the seed,
choose which frames of the video are used. The slice applies to both the main path and
``last-frame=``. A slice in the seed overrides the global options, as described under
`Animation Slicing`_.

Frame counts:

* Each conditioning clip is cut to a frame count of ``8k+1``, the same rule as the
  output length. A 54 frame gif gives 49 conditioning frames.
* A clip longer than the output is cut to fit. With ``--ltx-video-lengths`` set, only the
  frames that can be used are decoded. Without it the whole slice is decoded, so use
  ``--frame-end`` on long files.
* When there is an opening and a closing condition, the opening clip is shortened so
  the two do not overlap.
* A closing video needs a fixed output length. LTX-2.5 picks its own length when
  ``--ltx-video-lengths`` is omitted, so set ``--ltx-video-lengths`` when ``last-frame=`` is a video.
  An opening video works either way.

Frames are used as they are and are not resampled. When the file's frame rate differs
from ``--ltx-video-fps`` dgenerate prints a warning, because motion will play faster or
slower. Set ``--ltx-video-fps`` to the file's rate to keep the speed.

``resize=``, ``aspect=``, and ``align=`` in the seed apply to every conditioning frame.

``--seed-image-processors`` runs on every frame of the main path and of ``last-frame=``. Give it two
chains separated by ``+`` to process them differently. The first chain runs on the main path and
the second on ``last-frame=``. A leading or trailing ``+`` leaves one side unprocessed.

.. code-block:: bash

    # grayscale first frame, original last frame
    dgenerate Lightricks/LTX-2.5-Diffusers --model-type ltx \
    --ltx-video-lengths 3 --output-size 512x512 \
    --image-seeds "painting.png;last-frame=painting.png" \
    --seed-image-processors grayscale + \
    --prompts "A black and white painting slowly fills with warm color."

    # process only last-frame=
    --seed-image-processors + "canny;lower=50;upper=100"

Every condition is applied at full strength. The model keeps the conditioning frames
and generates around them. It does not restyle the whole input video.

See `examples/ltx/ltx2/video_conditioning <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/ltx/ltx2/video_conditioning>`_,
`examples/ltx/ltx2/image_conditioning <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/ltx/ltx2/image_conditioning>`_,
and `examples/ltx/ltx_video/video-extension-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx_video/video-extension-config.dgen>`_.

IC-LoRA control
---------------

An IC-LoRA (in-context LoRA) guides LTX-2.5 with a reference video, for example canny
edges, a depth map, or a pose skeleton. Load it with ``--ltx-ic-lora``. The reference clip
comes from ``--image-seeds``, and ``--control-image-processors`` turns it into the signal
the IC-LoRA expects. The reference frames guide the output and are not placed in it.

With ``--ltx-ic-lora``, a plain seed path is the reference clip, the same way a plain path is the
control image when ``--control-nets`` is given to an image model. To add a first or last
frame, put the reference in ``control=``:

.. code-block:: bash

    # reference clip only
    dgenerate Lightricks/LTX-2.5-Diffusers --model-type ltx \
    --ltx-ic-lora "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control;weight-name=ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" \
    --ltx-video-fps 25 --ltx-video-lengths 4 --output-size 512x512 \
    --image-seeds "input.gif" \
    --control-image-processors "canny;lower=50;upper=100" \
    --prompts "A man in a shiny silver suit sings at a vintage microphone."

    # first frame plus reference clip
    --image-seeds "first.png;control=input.gif"

* ``--ltx-ic-lora`` takes the same URI as ``--loras``, plus ``attention`` and ``downscale``.
  ``scale`` is the LoRA weight. ``attention``, from 0 to 1, is how strongly the generated
  video attends to the reference, 1 by default.
* ``--loras`` can be used at the same time, for example a style LoRA. All of them are fused
  into the transformer together, and ``--lora-fuse-scale`` applies to all of them.
* Lightricks publishes IC-LoRAs for the distilled checkpoint, which is what
  ``Lightricks/LTX-2.5-Diffusers`` loads by default.
* The reference is one video, animated image, or still. It uses the same frame slicing,
  ``resize=``, and frame count rules as `Video conditioning`_.
* Without ``--ltx-video-lengths`` the output is as long as the reference clip, cut to ``8k+1``
  frames. With ``--ltx-video-lengths`` a longer reference is cut to the output length.
* Some IC-LoRAs read the reference at a reduced size. dgenerate reads
  ``reference_downscale_factor`` from the IC-LoRA's safetensors metadata, and ``downscale``
  in the URI overrides it. The union control LoRA uses 2, so the output width and height
  must be divisible by 64.
* ``--ltx-ic-lora`` needs an LTX-2 checkpoint. The earlier LTX-Video pipeline rejects it.

See the examples in `examples/ltx/ltx2/ic_lora <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/ltx/ltx2/ic_lora>`_.
`canny-anime-lora-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx2/ic_lora/canny-anime-lora-config.dgen>`_
and `depth-realism-lora-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx2/ic_lora/depth-realism-lora-config.dgen>`_
add a style LoRA from ``--loras`` to the IC-LoRA.

In the Console UI, the LTX-2.5 recipes under ``Edit -> Insert Code -> Recipe`` have an IC-LoRA
field, an IC-LoRA control clip, and a control clip processor. ``Edit -> Insert URI -> Sub Model URI``
builds an ``--ltx-ic-lora`` URI, and ``Edit -> Insert URI -> Image Seed URI`` accepts a last frame or
closing clip for ``last-frame=``. The LTX recipes also take a separate processor for the first and last frame.

Guidance, steps, and sigmas
---------------------------

The loaded scheduler decides which path runs. dgenerate does not pick a path from
the repository name or a ``--transformer`` subfolder. ``--transformer`` replaces
weights only. ``--scheduler`` accepts ``FlowMatchEulerDiscreteScheduler`` and URI
arguments on that class, such as ``shift`` and ``use-dynamic-shifting``. Those
arguments overlay the checkpoint scheduler config. Other scheduler names are
rejected. Omitting ``--scheduler`` keeps the checkpoint scheduler.
``--scheduler help`` and ``--scheduler helpargs`` still print help.

The image-model defaults still apply if you omit the options: ``--guidance-scales``
is ``5`` and ``--inference-steps`` is ``30``. LTX then rewrites those defaults when
they are still unused. A value you set yourself is kept, except as noted below.

**Scheduler without dynamic shifting (the published LTX-2.5 scheduler)**

* The clip uses Diffusers' distilled 8-value sigma table. ``--inference-steps`` is
  not sent to the pipeline. dgenerate warns with the ignored value, including the
  default 30, and still uses the table.
* If ``--guidance-scales`` is still ``5``, it is replaced with ``1`` (unguided).
  Any other guidance value is used as written.
* Video guidance and audio guidance receive that same number unless
  ``--ltx-audio-guidance-scales`` is set.
* After the clip runs, the written config records 8 steps and the guidance that
  was actually used.

**Scheduler with dynamic shifting**

* ``--inference-steps`` is sent as ``num_inference_steps``. The default ``30`` is
  used if you omit it.
* If ``--guidance-scales`` is still ``5``, video guidance becomes ``3`` and audio
  guidance becomes ``7``, with a warning, unless ``--ltx-audio-guidance-scales``
  is set. Any other guidance value is used for video, and for audio unless
  ``--ltx-audio-guidance-scales`` is set.

``--sigmas`` (CSV list or ``expr:``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This path wins over both of the above.

* A CSV list is the schedule. An ``expr:`` expression is evaluated against the
  distilled table when the scheduler has no dynamic shifting, or against
  ``scheduler.set_timesteps(--inference-steps)`` when it does.
* The step count becomes the length of the resulting list. ``--inference-steps``
  is overwritten to match, including in the written config.
* ``--guidance-scales`` is used as written, including leftover ``5``. Nothing is
  rewritten to 1, 3, or 7 on this path. Video and audio guidance get the same
  value unless ``--ltx-audio-guidance-scales`` is set.

``--sigmas`` is combinatorial with ``--guidance-scales``, ``--inference-steps``,
``--guidance-rescales``, ``--ltx-audio-guidance-scales``,
``--ltx-audio-guidance-rescales``, ``--ltx-video-lengths``, and ``--ltx-video-fps``. See
:ref:`specifying-sigmas` and `examples/ltx/ltx2/sigmas/sigmas-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx2/sigmas/sigmas-config.dgen>`_.

``--guidance-rescales``
~~~~~~~~~~~~~~~~~~~~~~~

LTX accepts this. A value you set is sent as video guidance rescale, and as
audio rescale unless ``--ltx-audio-guidance-rescales`` is set. If you omit it,
the pipeline keeps its own default (``0.7``). The rescale only applies while
classifier-free guidance is on (guidance greater than 1).

``--max-sequence-length``
~~~~~~~~~~~~~~~~~~~~~~~~~

LTX accepts this as Gemma's prompt token budget, from 1 to 1024. If you omit
it, the pipeline keeps 1024.

``--vae-slicing``
~~~~~~~~~~~~~~~~~

LTX accepts this on the video VAE and the audio VAE. ``--vae-tiling`` is
rejected: the video VAE is always tiled.

Audio
-----

One prompt drives both the picture and the soundtrack. Gemma encodes that
string once; the text connectors split the packed hidden states into video
tokens and audio tokens. There is no audio-only prompt argument, so
``--second-prompts`` cannot be a soundtrack prompt.

Audio CFG is a separate pipeline scale. ``--ltx-audio-guidance-scales`` sets it
and is combinatorial. Omit that option and audio copies the video guidance
that will be sent, except the unused default ``5`` on a full scheduler, which
uses audio ``7``. A distilled unused ``5`` becomes video and audio ``1``.

``--ltx-audio-guidance-rescales`` is the matching combinatorial audio rescale.
Omit it and audio copies ``--guidance-rescales`` when that is set, otherwise
the pipeline default ``0.7`` is left in place.

Diffusers suggests keeping audio guidance higher than video guidance when
you set them yourself. See `examples/ltx/ltx2/audio/audio-guidance-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx2/audio/audio-guidance-config.dgen>`_.

Chaining
--------

``last_images`` and ``last_animations`` work the same way they do for image models. A still written by an
earlier invocation can be the first frame of an LTX clip, and an animation written by an earlier
invocation can be its opening clip.
The pipeline cache counts the video checkpoint and moves the previous pipeline back to CPU before the clip runs.

Submodels
---------

``--vae``, ``--unet``, and ``--text-encoders`` are rejected. Those slots do not match this pipeline.
``--transformer`` and ``--loras`` do.

``--quantizer`` quantizes the diffusion transformer, the text encoder, and the text
connectors. ``--quantizer-map`` can limit that to ``transformer``, ``text_encoder``, or
``connectors``. The unused prompt-enhancer Gemma is not loaded. ``--transformer`` replaces
the diffusion transformer and accepts the same URI as it does for Flux, including
``subfolder``, ``dtype``, and ``quantizer``. A quantizer on that URI wins over ``--quantizer``.
Replacing the transformer does not change the scheduler.

The repository's default transformer is distilled. The full (non-distilled) transformer is
in its ``transformer_full`` subfolder. Load it with
``--transformer "Lightricks/LTX-2.5-Diffusers;subfolder=transformer_full"`` and give it a
normal schedule with
``--scheduler "FlowMatchEulerDiscreteScheduler;use-dynamic-shifting=true;shift-terminal=0.1"``,
otherwise it runs the distilled 8-value sigma table. Set ``--inference-steps`` and the guidance
stack for a guided model, for example 30 steps, ``--guidance-scales 3``,
``--ltx-audio-guidance-scales 7``, ``--ltx-stg-scales 1``, ``--ltx-modality-scales 3``,
and ``--ltx-stg-blocks 28``.
Lightricks IC-LoRAs are trained on the distilled transformer.
See `examples/ltx/ltx2/full_transformer/text-to-video-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx2/full_transformer/text-to-video-config.dgen>`_.

``--loras`` loads diffusers-format adapters onto that transformer and fuses them, including
``--lora-fuse-scale`` and each URI ``scale``. IC-LoRAs load with ``--ltx-ic-lora`` instead, see `IC-LoRA control`_.
See `examples/ltx/ltx2/lora/cinemagraph-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx2/lora/cinemagraph-config.dgen>`_
and `examples/ltx/ltx_video/lora-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx_video/lora-config.dgen>`_.

Two-stage generation
~~~~~~~~~~~~~~~~~~~~

``--ltx-latent-upscale`` runs both stages in the same generation, the way
``--sdxl-refiner`` follows a base pass. ``--output-size`` is the finished clip,
and both sides must be divisible by 64. Stage 1 denoises at half of that size
with the main guidance, steps, and ``--sigmas``. The checkpoint ``latent_upsampler``
doubles the video latents. Stage 2 refines at the full size with the published
3-value sigma table. ``--ltx-noise-scales`` defaults to the first of those sigmas.
Stage 2 guidance defaults to 1. ``--ltx-stage-sigmas``, ``--ltx-stage-guidance-scales``,
and ``--ltx-stage-audio-guidance-scales`` override the refine pass.

The distilled checkpoint needs no stage LoRA. See
`examples/ltx/ltx2/two_stage/image-to-video-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx2/two_stage/image-to-video-config.dgen>`_.

The full transformer does. Load ``transformer_full``, give stage 1 dynamic shifting
and ``shift-terminal=0.1``, and put the distilled LoRA on the refine pass only:

``--ltx-stage-loras "Lightricks/LTX-2.5-Diffusers;weight-name=ltx-2.5-22b-distilled-lora-450-bf16.safetensors"``

See `examples/ltx/ltx2/two_stage/full-transformer-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx2/two_stage/full-transformer-config.dgen>`_.

Guidance
~~~~~~~~

``--ltx-stg-scales`` is spatio-temporal guidance. ``0`` turns it off. When it is on
and ``--ltx-stg-blocks`` is omitted, block ``28`` is used.
``--ltx-modality-scales`` is modality-isolation guidance. ``1`` turns it off.
``--ltx-audio-stg-scales`` and ``--ltx-audio-modality-scales`` copy the video values
when omitted. ``--ltx-no-cross-timestep`` selects the LTX-2.0 cross-modality timestep.
LTX-2.3 and LTX-2.5 leave that timestep on.

A full-transformer one-pass call uses video guidance ``3``, audio guidance ``7``,
STG ``1``, and modality guidance ``3``. The distilled checkpoint stays at guidance ``1``
and does not use those extra terms.

Decode, prompts, and duration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--ltx-video-decoder diffusion`` decodes with the checkpoint diffusion decoder
instead of the convolutional VAE, in the same generation. ``conv`` is the default.
``--ltx-decode-timesteps`` and ``--ltx-decode-noise-scales`` are the decode arguments.
See `examples/ltx/ltx2/decode/diffusion-decoder-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx2/decode/diffusion-decoder-config.dgen>`_.

``--ltx-prompt-enhancer`` loads a model such as ``google/gemma-4-E2B-it`` and rewrites
the prompt before denoising. ``--ltx-system-prompt`` overrides the built-in text or
image system prompt. See `examples/ltx/ltx2/basic/prompt-enhancer-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx2/basic/prompt-enhancer-config.dgen>`_.

``--ltx-image-crfs`` recompresses a conditioning still before the VAE encode. Omit it
and the pipeline default is used (``18`` on LTX-2.5). ``0`` skips recompression.
See `examples/ltx/ltx2/image_conditioning/image-crf-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx2/image_conditioning/image-crf-config.dgen>`_.

``--ltx-video-min-seconds`` and ``--ltx-video-max-seconds`` clamp the duration head.
They apply only when ``--ltx-video-lengths`` is omitted. Give the same number of
values to each. The value in each position is used together:
``--ltx-video-min-seconds 2 4 --ltx-video-max-seconds 6 8`` writes two clips, one
from 2 to 6 seconds and one from 4 to 8. A single value with the other option
omitted uses that option's pipeline default (``1`` and ``20``). Each bound pair
is then tried in turn with the other arguments. See
`examples/ltx/ltx2/basic/duration-bounds-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx2/basic/duration-bounds-config.dgen>`_
and `examples/ltx/ltx2/basic/duration-head-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx2/basic/duration-head-config.dgen>`_.

Condition placement
~~~~~~~~~~~~~~~~~~~

``ltx-index`` is an LTX image-seed keyword. ``strength`` is the weight of that
condition, from 0 to 1, and the same keyword sets img2img strength on image
models that accept it. A path with neither keyword is still the first frame at
full strength, which is the same as ``ltx-index=0`` and ``strength=1``.
``--image-seed-strengths`` fills any LTX group that omits ``strength``. Several
values are tried in turn. ``last-frame=`` stays at strength 1.

``ltx-index``
^^^^^^^^^^^^^

``ltx-index`` is which latent frame the file is written into. It is not an output
frame number and not ``frame-start``.

LTX's video VAE compresses time by 8. A clip of ``8k+1`` output frames is
``k+1`` latent frames. The first latent frame is output frame 0. Each latent
frame after that covers the next 8 output frames. At 24 fps that chunk is about
a third of a second.

A still placed on a latent frame is held across the output frames that latent
frame covers, so the picture stays still there instead of moving through it.
``ltx-index=0`` holds the opening. ``ltx-index=-1`` holds the last latent frame,
the same slot ``last-frame=`` uses. ``ltx-index=4`` on a long clip is the latent frame
that begins at output frame 33 (``1 + 4*8``).

A video or animated image placed at an index starts at that latent frame. Its
frame count is still cut to ``8k+1`` before it is placed, using the same rule as
an opening clip. The clip is trimmed so it fits in the frames that remain after
that index.

``strength``
^^^^^^^^^^^^

``strength`` is how strongly that latent frame must match the file, from 0
to 1. ``1`` keeps the conditioning frames. A lower value lets the generated
frames leave them. Omitting it is ``1``, unless ``--image-seed-strengths``
is set. On an img2img seed the same keyword overrides ``--image-seed-strengths``
for that seed.

``last-frame=``
^^^^^^^^^^^^^^^

``last-frame=`` is the last frame at strength 1. It does not take ``strength``.
To hold the end more loosely, drop ``last-frame=`` and place that file with
``ltx-index=-1``:

.. code-block:: bash

    # last frame, full strength
    --image-seeds "start.jpg;last-frame=end.jpg"

    # last frame, partial strength
    --image-seeds "start.jpg ++ end.jpg;ltx-index=-1;strength=0.4"

Several conditions
^^^^^^^^^^^^^^^^^^

One ``--image-seeds`` value can carry several conditions. Separate them with
`` ++ `` (a space, two plus signs, and a space). The first group is the primary
path and may omit ``ltx-index``. Each later group is one file and must include
``ltx-index``. ``strength`` is optional on every group. ``last-frame=``, ``control=``,
masks, and latents belong on the primary group only.

.. code-block:: bash

    --image-seeds "open.mp4;ltx-index=0 ++ mid.jpg;ltx-index=4;strength=0.6 ++ close.jpg;ltx-index=-1"

``open.mp4`` starts at the first latent frame. ``mid.jpg`` is held at latent
frame 4, loosely. ``close.jpg`` is the last latent frame at full strength.

``control=`` is not placed in the clip. With ``--ltx-ic-lora`` it is the
reference the IC-LoRA reads. ``ltx-index`` does not move that reference.

See `examples/ltx/ltx2/image_conditioning/indexed-config.dgen <https://github.com/Teriks/dgenerate/blob/@REVISION/examples/ltx/ltx2/image_conditioning/indexed-config.dgen>`_.

What LTX rejects
----------------

ControlNets, T2I adapters, IP adapters, textual inversions, a replacement UNet, VAE, or text encoder,
an image encoder, the SDXL refiner, Stable Cascade, Adetailer, PAG and PAG scales,
any scheduler other than ``FlowMatchEulerDiscreteScheduler``, prompt weighters, second or third prompts,
clip skip, inpaint crop, HiDiffusion, TeaCache, DeepCache, SADA, RAS,
mask processors, raw latents and latents processors, ``--denoising-start`` /
``--denoising-end``, ``--batch-size`` greater than 1, ``--batch-grid-size``, latent output
formats, the safety checker, ``--vae-tiling``, and ``--original-config``.
Seed processors are limited to two chains, and
control processors to one.
``--quantizer-map`` may only name ``transformer``, ``text_encoder``, or ``connectors``.

LTX-2.5 configs are in `examples/ltx/ltx2 <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/ltx/ltx2>`_,
and LTX-Video configs are in `examples/ltx/ltx_video <https://github.com/Teriks/dgenerate/tree/@REVISION/examples/ltx/ltx_video>`_.
