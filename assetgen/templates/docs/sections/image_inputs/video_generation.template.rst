.. _video-generation:

Video Generation
================

``--model-type ltx`` generates a clip in one pipeline call.
One combination of prompt, seed, guidance, steps, image seed, ``--video-lengths``,
``--video-fps``, ``--audio-guidance-scales``, and ``--audio-guidance-rescales``
writes one animation file. LTX does not run once per input frame.

``--video-lengths`` is a length in seconds. ``--video-fps`` is the frame rate.
``--audio-guidance-scales`` and ``--audio-guidance-rescales`` are the audio CFG
and audio rescale. All of those are combinatorial arguments, the same way
``--prompts`` and ``--seeds`` are. The frame count inside a clip is not a
separate product factor.

Audio from LTX-2.5 is muxed into an mp4. GIF, WebP, and ``--animation-format frames`` drop that audio.

Conditioning stays in ``--image-seeds``.

LTX-2.5 (``ltx``)
-----------------

Repository: ``Lightricks/LTX-2.5-Diffusers``.

* No image seed is text to video. Omitting ``--video-lengths`` lets the model's duration head choose the length.
* One image is the first frame.
* ``end=`` is the last frame. A first frame and ``end=`` can be used together.
* Either slot can be a video or animated image instead of a still. See `Video conditioning`_.
* With ``--ic-lora``, a plain path is instead the reference clip for an IC-LoRA, such as canny, depth, or pose control. See `IC-LoRA control`_.
* ``images:`` is rejected.

Width and height must be divisible by 32.

``--model-sequential-offload`` and ``--model-cpu-offload`` work the same way they
do for image models. The examples under ``examples/ltx/ltx2`` use the published
repository as-is. The two-stage sampler is not wired up. ``--transformer`` is
described under `Submodels`_. ``model_index.json`` selects the pipeline: ``LTX2Pipeline``
is LTX-2.5, and ``LTXPipeline`` is the earlier LTX-Video model. The earlier model
has no audio. Its transformer accepts a city96 ``.gguf`` file. See ``examples/ltx/ltx_video``.

Video conditioning
------------------

The main ``--image-seeds`` path and ``end=`` each accept a video or an animated image
as well as a still. Both LTX-2.5 and the earlier LTX-Video accept this. A file with a
single frame is treated as a still.

* A video in the main path is the opening clip. The generated clip continues from it.
* A video in ``end=`` is the closing clip. The generated clip leads into it.
* A still and a video can be mixed, for example a video first and a still ``end=``.

.. code-block:: bash

    # continue an existing clip
    dgenerate Lightricks/LTX-2.5-Diffusers --model-type ltx \
    --video-fps 25 --video-lengths 4 --output-size 512x512 \
    --image-seeds "input.gif" \
    --prompts "The singer keeps swaying, then points at the camera."

    # lead into an existing clip
    dgenerate Lightricks/LTX-2.5-Diffusers --model-type ltx \
    --video-fps 25 --video-lengths 4 --output-size 512x512 \
    --image-seeds ";end=input.gif" \
    --prompts "A singer walks in and starts to sway at the microphone."

    # frames 16 through 40 of a clip, then a still last frame
    dgenerate Lightricks/LTX-2.5-Diffusers --model-type ltx \
    --video-fps 25 --video-lengths 4 --output-size 512x512 \
    --image-seeds "input.gif;frame-start=16;frame-end=40;end=last.png" \
    --prompts "The scene fades into a pencil sketch of mountains."

``--frame-start`` and ``--frame-end``, or ``frame-start=`` and ``frame-end=`` in the seed,
choose which frames of the video are used. The slice applies to both the main path and
``end=``. A slice in the seed overrides the global options, as described under
`Animation Slicing`_.

Frame counts:

* Each conditioning clip is cut to a frame count of ``8k+1``, the same rule as the
  output length. A 54 frame gif gives 49 conditioning frames.
* A clip longer than the output is cut to fit. With ``--video-lengths`` set, only the
  frames that can be used are decoded. Without it the whole slice is decoded, so use
  ``--frame-end`` on long files.
* When there is an opening and a closing condition, the opening clip is shortened so
  the two do not overlap.
* A closing video needs a fixed output length. LTX-2.5 picks its own length when
  ``--video-lengths`` is omitted, so set ``--video-lengths`` when ``end=`` is a video.
  An opening video works either way.

Frames are used as they are and are not resampled. When the file's frame rate differs
from ``--video-fps`` dgenerate prints a warning, because motion will play faster or
slower. Set ``--video-fps`` to the file's rate to keep the speed.

``resize=``, ``aspect=``, and ``align=`` in the seed apply to every conditioning frame.

``--seed-image-processors`` runs on every frame of the main path and of ``end=``. Give it two
chains separated by ``+`` to process them differently. The first chain runs on the main path and
the second on ``end=``. A leading or trailing ``+`` leaves one side unprocessed.

.. code-block:: bash

    # grayscale first frame, original last frame
    dgenerate Lightricks/LTX-2.5-Diffusers --model-type ltx \
    --video-lengths 3 --output-size 512x512 \
    --image-seeds "painting.png;end=painting.png" \
    --seed-image-processors grayscale + \
    --prompts "A black and white painting slowly fills with warm color."

    # process only end=
    --seed-image-processors + "canny;lower=50;upper=100"

Every condition is applied at full strength. The model keeps the conditioning frames
and generates around them. It does not restyle the whole input video.

See ``examples/ltx/ltx2/video_conditioning``, ``examples/ltx/ltx2/image_conditioning``, and
``video-extension-config.dgen`` in ``examples/ltx/ltx_video``.

IC-LoRA control
---------------

An IC-LoRA (in-context LoRA) guides LTX-2.5 with a reference video, for example canny
edges, a depth map, or a pose skeleton. Load it with ``--ic-lora``. The reference clip
comes from ``--image-seeds``, and ``--control-image-processors`` turns it into the signal
the IC-LoRA expects. The reference frames guide the output and are not placed in it.

With ``--ic-lora``, a plain seed path is the reference clip, the same way a plain path is the
control image when ``--control-nets`` is given to an image model. To add a first or last
frame, put the reference in ``control=``:

.. code-block:: bash

    # reference clip only
    dgenerate Lightricks/LTX-2.5-Diffusers --model-type ltx \
    --ic-lora "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control;weight-name=ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors" \
    --video-fps 25 --video-lengths 4 --output-size 512x512 \
    --image-seeds "input.gif" \
    --control-image-processors "canny;lower=50;upper=100" \
    --prompts "A man in a shiny silver suit sings at a vintage microphone."

    # first frame plus reference clip
    --image-seeds "first.png;control=input.gif"

* ``--ic-lora`` takes the same URI as ``--loras``, plus ``attention`` and ``downscale``.
  ``scale`` is the LoRA weight. ``attention``, from 0 to 1, is how strongly the generated
  video attends to the reference, 1 by default.
* ``--loras`` can be used at the same time, for example a style LoRA. All of them are fused
  into the transformer together, and ``--lora-fuse-scale`` applies to all of them.
* Lightricks publishes IC-LoRAs for the distilled checkpoint, which is what
  ``Lightricks/LTX-2.5-Diffusers`` loads by default.
* The reference is one video, animated image, or still. It uses the same frame slicing,
  ``resize=``, and frame count rules as `Video conditioning`_.
* Without ``--video-lengths`` the output is as long as the reference clip, cut to ``8k+1``
  frames. With ``--video-lengths`` a longer reference is cut to the output length.
* Some IC-LoRAs read the reference at a reduced size. dgenerate reads
  ``reference_downscale_factor`` from the IC-LoRA's safetensors metadata, and ``downscale``
  in the URI overrides it. The union control LoRA uses 2, so the output width and height
  must be divisible by 64.
* ``--ic-lora`` needs an LTX-2 checkpoint. The earlier LTX-Video pipeline rejects it.

See the examples in ``examples/ltx/ltx2/ic_lora``. ``canny-anime-lora-config.dgen`` and
``depth-realism-lora-config.dgen`` add a style LoRA from ``--loras`` to the IC-LoRA.

In the Console UI, the LTX-2.5 recipes under ``Edit -> Insert Code -> Recipe`` have an IC-LoRA
field, an IC-LoRA control clip, and a control clip processor. ``Edit -> Insert URI -> Sub Model URI``
builds an ``--ic-lora`` URI, and ``Edit -> Insert URI -> Image Seed URI`` accepts a last frame or
closing clip for ``end=``. The LTX recipes also take a separate processor for the first and last frame.

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
  ``--audio-guidance-scales`` is set.
* After the clip runs, the written config records 8 steps and the guidance that
  was actually used.

**Scheduler with dynamic shifting**

* ``--inference-steps`` is sent as ``num_inference_steps``. The default ``30`` is
  used if you omit it.
* If ``--guidance-scales`` is still ``5``, video guidance becomes ``3`` and audio
  guidance becomes ``7``, with a warning, unless ``--audio-guidance-scales``
  is set. Any other guidance value is used for video, and for audio unless
  ``--audio-guidance-scales`` is set.

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
  value unless ``--audio-guidance-scales`` is set.

``--sigmas`` is combinatorial with ``--guidance-scales``, ``--inference-steps``,
``--guidance-rescales``, ``--audio-guidance-scales``,
``--audio-guidance-rescales``, ``--video-lengths``, and ``--video-fps``. See
:ref:`specifying-sigmas` and ``examples/ltx/ltx2/sigmas/sigmas-config.dgen``.

``--guidance-rescales``
~~~~~~~~~~~~~~~~~~~~~~~

LTX accepts this. A value you set is sent as video guidance rescale, and as
audio rescale unless ``--audio-guidance-rescales`` is set. If you omit it,
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

Audio CFG is a separate pipeline scale. ``--audio-guidance-scales`` sets it
and is combinatorial. Omit that option and audio copies the video guidance
that will be sent, except the unused default ``5`` on a full scheduler, which
uses audio ``7``. A distilled unused ``5`` becomes video and audio ``1``.

``--audio-guidance-rescales`` is the matching combinatorial audio rescale.
Omit it and audio copies ``--guidance-rescales`` when that is set, otherwise
the pipeline default ``0.7`` is left in place.

Diffusers suggests keeping audio guidance higher than video guidance when
you set them yourself. See ``examples/ltx/ltx2/audio/audio-guidance-config.dgen``.

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

``--loras`` loads diffusers-format adapters onto that transformer and fuses them, including
``--lora-fuse-scale`` and each URI ``scale``. The LTX stage-2 distilled LoRA does not turn on
two-stage sampling. IC-LoRAs load with ``--ic-lora`` instead, see `IC-LoRA control`_.
See ``examples/ltx/ltx2/lora/cinemagraph-config.dgen`` and ``examples/ltx/ltx_video/lora-config.dgen``.

What LTX rejects
----------------

ControlNets, T2I adapters, IP adapters, textual inversions, a replacement UNet, VAE, or text encoder,
an image encoder, the SDXL refiner, Stable Cascade, Adetailer, PAG and PAG scales,
any scheduler other than ``FlowMatchEulerDiscreteScheduler``, prompt weighters, second or third prompts,
clip skip, inpaint crop, HiDiffusion, TeaCache, DeepCache, SADA, RAS,
mask processors, raw latents and latents processors, ``--denoising-start`` /
``--denoising-end``, ``--batch-size`` greater than 1, ``--batch-grid-size``, latent output
formats, the safety checker, ``--vae-tiling``, and ``--original-config``.
``--image-seed-strengths`` is not used. Seed processors are limited to two chains, and
control processors to one.
``--quantizer-map`` may only name ``transformer``, ``text_encoder``, or ``connectors``.

LTX-2.5 examples live under ``examples/ltx/ltx2``, and LTX-Video examples under ``examples/ltx/ltx_video``.
