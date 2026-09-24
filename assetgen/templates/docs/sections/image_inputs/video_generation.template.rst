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
* ``control=`` and ``images:`` are rejected.

Width and height must be divisible by 32.

``--model-sequential-offload`` and ``--model-cpu-offload`` work the same way they
do for image models. The examples under ``examples/video/ltx`` use the published
repository as-is. The two-stage sampler is not wired up. ``--transformer`` is
described under `Submodels`_. ``model_index.json`` selects the pipeline: ``LTX2Pipeline``
is LTX-2.5, and ``LTXPipeline`` is the earlier LTX-Video model. The earlier model
has no audio. Its transformer accepts a city96 ``.gguf`` file. See ``examples/ltx/basic_ltx``.

Guidance, steps, and sigmas
---------------------------

The loaded scheduler decides which path runs. dgenerate does not pick a path from
the repository name or a ``--transformer`` subfolder. ``--transformer`` replaces
weights only.

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

**``--sigmas`` (CSV list or ``expr:``)**

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
:ref:`specifying-sigmas` and ``examples/video/ltx/sigmas-config.dgen``.

**``--guidance-rescales``**

LTX accepts this. A value you set is sent as video guidance rescale, and as
audio rescale unless ``--audio-guidance-rescales`` is set. If you omit it,
the pipeline keeps its own default (``0.7``). The rescale only applies while
classifier-free guidance is on (guidance greater than 1).

**``--max-sequence-length``**

LTX accepts this as Gemma's prompt token budget, from 1 to 1024. If you omit
it, the pipeline keeps 1024.

**``--vae-slicing``**

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
you set them yourself. See ``examples/video/ltx/audio-guidance-config.dgen``.

Chaining
--------

``last_images`` and ``last_animations`` work the same way they do for image models. A still written by an
earlier invocation can be the first frame of an LTX clip.
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
two-stage sampling. IC-LoRAs expect their own pipeline and are not a separate mode here.

What LTX rejects
----------------

ControlNets, T2I adapters, IP adapters, textual inversions, a replacement UNet, VAE, or text encoder,
an image encoder, the SDXL refiner, Stable Cascade, Adetailer, PAG and PAG scales,
a custom scheduler, prompt weighters, second or third prompts,
clip skip, inpaint crop, HiDiffusion, TeaCache, DeepCache, SADA, RAS,
mask or control processors, raw latents and latents processors, ``--denoising-start`` /
``--denoising-end``, ``--batch-size`` greater than 1, ``--batch-grid-size``, latent output
formats, the safety checker, ``--vae-tiling``, ``--frame-start`` / ``--frame-end``, and ``--original-config``.
``--image-seed-strengths`` is not used. Seed processors are limited to one chain.
``--quantizer-map`` may only name ``transformer``, ``text_encoder``, or ``connectors``.

Examples live under ``examples/ltx/basic_ltx2`` and ``examples/ltx/basic_ltx``.
